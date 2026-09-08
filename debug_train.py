import argparse
import math
import random
from collections import defaultdict

import torch

from lln.data import (
    SECTION_ANSWER,
    SECTION_PROMPT,
    SECTION_THINK,
    _compact_record,
    decode_ids,
    encode_prompt,
    encode_record,
    load_dictionary,
    load_records,
    make_batch,
)
from lln.model import LLN, parameter_count, parameter_size_mb


def pick_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def section_weights(sections, think_weight=0.25, answer_weight=1.0):
    return torch.where(
        sections.eq(SECTION_THINK),
        torch.full_like(sections, think_weight),
        torch.where(
            sections.eq(SECTION_ANSWER),
            torch.full_like(sections, answer_weight),
            torch.zeros_like(sections),
        ),
    )


def grad_stats(model):
    total_sq = 0.0
    nonzero = 0
    finite = True
    max_abs = 0.0
    groups = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            finite = False
        norm = float(g.norm().item())
        total_sq += norm * norm
        if norm > 0.0:
            nonzero += 1
        max_abs = max(max_abs, float(g.abs().max().item()))
        root = name.split(".", 1)[0]
        groups[root] += norm * norm
    grouped = {k: math.sqrt(v) for k, v in groups.items()}
    return math.sqrt(total_sq), nonzero, finite, max_abs, grouped


def parameter_delta(before, model):
    total_sq = 0.0
    max_abs = 0.0
    count = 0
    with torch.no_grad():
        for old, p in zip(before, model.parameters()):
            d = p.float() - old
            total_sq += float(d.norm().item()) ** 2
            max_abs = max(max_abs, float(d.abs().max().item()))
            count += int(torch.count_nonzero(d).item())
    return math.sqrt(total_sq), max_abs, count


def prepare_examples(records, word_to_id, seq_len):
    encoded = []
    for record in records:
        ids, sections = encode_record(record, word_to_id)
        ids, sections = _compact_record(ids, sections, seq_len + 1)
        encoded.append((ids, sections))
    return encoded


def select_batch(encoded, batch_size, device, seq_len):
    if not encoded:
        raise RuntimeError("No encoded examples available")
    selected = encoded[:min(batch_size, len(encoded))]
    if len(selected) < batch_size:
        selected = (selected * ((batch_size + len(selected) - 1) // len(selected)))[:batch_size]
    return make_batch(selected, batch_size, seq_len, device, indices=list(range(batch_size)))


def print_alignment(x, y, sections, id_to_word, limit=64):
    print("\n=== TOKEN ALIGNMENT (example 0) ===")
    row_x = x[0].detach().cpu().tolist()
    row_y = y[0].detach().cpu().tolist()
    row_s = sections[0].detach().cpu().tolist()
    shown = 0
    for i, (xi, yi, sec) in enumerate(zip(row_x, row_y, row_s)):
        if sec < 0:
            continue
        sx = id_to_word.get(xi, "<UNK>")
        sy = id_to_word.get(yi, "<UNK>")
        print(f"pos={i:4d} input={xi:5d} {sx!r:24s} -> target={yi:5d} {sy!r:24s} section={int(sec)}")
        shown += 1
        if shown >= limit:
            break


def weighted_metrics(model, x, y, sections):
    weights = section_weights(sections)
    model.eval()
    with torch.no_grad():
        logits, weighted_loss = model(x, y, loss_weights=weights)
        token_loss = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, model.vocab_size),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)

    valid = sections >= 0
    metrics = {"weighted_loss": float(weighted_loss.item())}
    for name, sec_id in (("prompt", SECTION_PROMPT), ("think", SECTION_THINK), ("answer", SECTION_ANSWER)):
        mask = valid & sections.eq(sec_id)
        count = int(mask.sum().item())
        if count == 0:
            metrics[f"{name}_loss"] = float("nan")
            metrics[f"{name}_acc"] = float("nan")
            metrics[f"{name}_count"] = 0
            continue
        section_loss = float(token_loss[mask].mean().item())
        predictions = logits.argmax(dim=-1)
        acc = float((predictions[mask] == y[mask]).float().mean().item())
        metrics[f"{name}_loss"] = section_loss
        metrics[f"{name}_acc"] = acc
        metrics[f"{name}_count"] = count

    target_probs = torch.softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    target_rank = (logits.float() > logits.float().gather(-1, y.unsqueeze(-1))).sum(-1) + 1
    answer_mask = valid & sections.eq(SECTION_ANSWER)
    if answer_mask.any():
        metrics["answer_target_p"] = float(target_probs[answer_mask].mean().item())
        metrics["answer_target_rank"] = float(target_rank[answer_mask].float().mean().item())
    else:
        metrics["answer_target_p"] = float("nan")
        metrics["answer_target_rank"] = float("nan")
    return logits, metrics


def print_metrics(prefix, metrics):
    print(
        f"{prefix} weighted={metrics['weighted_loss']:.6f} "
        f"prompt_loss={metrics['prompt_loss']:.6f} prompt_acc={metrics['prompt_acc']:.3f} "
        f"think_loss={metrics['think_loss']:.6f} think_acc={metrics['think_acc']:.3f} "
        f"answer_loss={metrics['answer_loss']:.6f} answer_acc={metrics['answer_acc']:.3f} "
        f"answer_p={metrics['answer_target_p']:.4f} "
        f"answer_rank={metrics['answer_target_rank']:.2f}"
    )


def print_top_predictions(logits, y, sections, id_to_word, topk=5, limit=32):
    print("\n=== TEACHER-FORCED TOP PREDICTIONS (example 0) ===")
    probs = torch.softmax(logits[0].float(), dim=-1)
    shown = 0
    for i in range(logits.size(1)):
        if sections[0, i].item() < 0:
            continue
        values, indices = torch.topk(probs[i], k=topk)
        target = int(y[0, i])
        target_rank = int((probs[i] > probs[i, target]).sum().item()) + 1
        choices = ", ".join(
            f"{id_to_word.get(int(t), '<UNK>')}:{float(v):.3f}" for v, t in zip(values, indices)
        )
        print(
            f"pos={i:4d} target={id_to_word.get(target, '<UNK>')!r:18s} "
            f"rank={target_rank:4d} top={choices}"
        )
        shown += 1
        if shown >= limit:
            break


def causality_check(model, x):
    if x.size(1) < 4:
        return 0.0
    base = x.clone()
    probe_pos = max(1, x.size(1) // 3)
    changed = base.clone()
    changed[:, probe_pos + 1:] = torch.roll(changed[:, probe_pos + 1:], shifts=1, dims=1)
    model.eval()
    with torch.no_grad():
        base_logits, _ = model(base)
        changed_logits, _ = model(changed)
    prefix = base_logits[:, :probe_pos + 1].float()
    changed_prefix = changed_logits[:, :probe_pos + 1].float()
    return float((prefix - changed_prefix).abs().max().item())


def generation_test(model, record, word_to_id, id_to_word, device, max_new_tokens):
    prompt = record[0]
    prompt_ids = encode_prompt(prompt, word_to_id)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    before = model.generate(
        input_ids.clone(),
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        repetition_window=64,
        frequency_penalty=0.08,
        presence_penalty=0.20,
        hard_repeat_threshold=6,
        stop_ids={word_to_id.get("<EOS>", 2)},
    )
    return decode_ids(before[0].detach().cpu().tolist(), id_to_word)


def main():
    parser = argparse.ArgumentParser(description="LLN diagnostic suite: alignment, gradients, overfit, held-out and autoregressive tests")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recurrent-steps", type=int, default=1)
    parser.add_argument("--output-clusters", type=int, default=120)
    parser.add_argument("--memory-slots", type=int, default=4)
    parser.add_argument("--examples", type=int, default=8, help="Number of examples used for overfit")
    parser.add_argument("--eval-examples", type=int, default=8, help="Held-out examples after the overfit set")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--generation-tokens", type=int, default=80)
    parser.add_argument("--no-think-weight", action="store_true", help="Train think tokens at the same weight as answer tokens")
    args = parser.parse_args()

    if args.examples < 1 or args.eval_examples < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("examples, eval-examples, batch-size and steps must be >= 1")
    if args.heads < 1 or args.dim % args.heads:
        raise ValueError("dim must be divisible by heads")
    if args.seq_len < 8:
        raise ValueError("seq-len must be at least 8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = pick_dtype(args.dtype)

    word_to_id, id_to_word, token_types, meta = load_dictionary(args.dictionary, with_metadata=True)
    records = load_records(args.dataset)
    required = args.examples + args.eval_examples
    if len(records) < required:
        raise ValueError(f"dataset has {len(records)} records, but debug needs at least {required}")

    train_records = records[:args.examples]
    eval_records = records[args.examples:args.examples + args.eval_examples]
    train_encoded = prepare_examples(train_records, word_to_id, args.seq_len)
    eval_encoded = prepare_examples(eval_records, word_to_id, args.seq_len)

    x, y, sections = select_batch(train_encoded, args.batch_size, device, args.seq_len)
    eval_x, eval_y, eval_sections = select_batch(eval_encoded, min(args.batch_size, len(eval_encoded)), device, args.seq_len)
    train_weight_sections = section_weights(sections, think_weight=1.0 if args.no_think_weight else 0.25)
    eval_weight_sections = section_weights(eval_sections, think_weight=1.0 if args.no_think_weight else 0.25)

    model_cfg = {
        "vocab_size": len(word_to_id),
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "max_seq_len": args.seq_len,
        "recurrent_steps": args.recurrent_steps,
        "output_clusters": args.output_clusters,
        "memory_slots": args.memory_slots,
        "type_count": int(meta.get("type_count", 6)),
    }
    model = LLN(**model_cfg).to(device=device, dtype=dtype)
    model.set_token_types(token_types)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print("=" * 72)
    print("LLN TRAINING DEBUG / LEARNING + GENERALIZATION")
    print("=" * 72)
    print(f"device={device} dtype={dtype}")
    print(f"parameters={parameter_count(model):,}")
    print(f"model_weight_size={parameter_size_mb(model, torch.tensor([], dtype=dtype).element_size()):.1f} MB")
    print(f"vocab={len(word_to_id):,} dataset_records={len(records):,}")
    print(f"overfit_examples={len(train_encoded)} heldout_examples={len(eval_encoded)}")
    print(f"batch_size={args.batch_size} seq_len={args.seq_len} steps={args.steps} lr={args.lr:g}")
    print(f"recurrent_steps={args.recurrent_steps} output_clusters={args.output_clusters} memory_slots={args.memory_slots}")
    print(f"loss_weights=think:{1.0 if args.no_think_weight else 0.25:g} answer:1 prompt:0")
    print("shuffle=disabled fixed_train_batch=yes")

    print_alignment(x, y, sections, id_to_word)

    initial_train_logits, initial_train_metrics = weighted_metrics(model, x, y, sections)
    _, initial_eval_metrics = weighted_metrics(model, eval_x, eval_y, eval_sections)
    print("\n=== INITIAL METRICS ===")
    print_metrics("TRAIN", initial_train_metrics)
    print_metrics("HELDOUT", initial_eval_metrics)
    print(f"baseline_ln_vocab={math.log(len(word_to_id)):.6f}")
    print_top_predictions(initial_train_logits, y, sections, id_to_word)
    print(f"causality_max_abs_diff={causality_check(model, x):.6g}")

    print("\n=== BACKWARD DIAGNOSTIC ===")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    logits, loss = model(x, y, loss_weights=train_weight_sections)
    loss.backward()
    grad_norm, nonzero, finite, grad_max, grad_groups = grad_stats(model)
    print(f"loss={float(loss.detach()):.6f}")
    print(f"grad_global_norm={grad_norm:.6g} grad_nonzero_tensors={nonzero} grad_finite={finite} grad_max_abs={grad_max:.6g}")
    for name, value in sorted(grad_groups.items(), key=lambda item: item[1], reverse=True):
        print(f"  grad[{name}]={value:.6g}")

    initial_params = [p.detach().float().clone() for p in model.parameters()]
    initial_generation = generation_test(model, train_records[0], word_to_id, id_to_word, device, args.generation_tokens)
    print("\n=== INITIAL AUTOREGRESSIVE GENERATION ===")
    print(initial_generation)

    print("\n=== OVERFIT LOOP ===")
    best_loss = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y, loss_weights=train_weight_sections)
        if not torch.isfinite(loss):
            print(f"step={step:4d} loss=NONFINITE -- abort")
            break
        loss.backward()
        grad_norm, _, finite, grad_max, _ = grad_stats(model)
        if not finite:
            print(f"step={step:4d} loss={float(loss):.6f} nonfinite_grad -- abort")
            break
        optimizer.step()
        value = float(loss.detach())
        best_loss = min(best_loss, value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            delta, delta_max, changed = parameter_delta(initial_params, model)
            print(
                f"step={step:4d} loss={value:.6f} delta_norm={delta:.6g} "
                f"delta_max={delta_max:.6g} changed_values={changed:,} "
                f"grad_norm={grad_norm:.6g} grad_max={grad_max:.6g}"
            )

    final_train_logits, final_train_metrics = weighted_metrics(model, x, y, sections)
    _, final_eval_metrics = weighted_metrics(model, eval_x, eval_y, eval_sections)
    delta, delta_max, changed = parameter_delta(initial_params, model)

    print("\n=== FINAL METRICS ===")
    print_metrics("TRAIN", final_train_metrics)
    print_metrics("HELDOUT", final_eval_metrics)
    print(f"parameter_delta_norm={delta:.6g} parameter_delta_max={delta_max:.6g} changed_values={changed:,}")
    print_top_predictions(final_train_logits, y, sections, id_to_word)
    print(f"causality_max_abs_diff_after={causality_check(model, x):.6g}")

    final_generation = generation_test(model, train_records[0], word_to_id, id_to_word, device, args.generation_tokens)
    heldout_generation = generation_test(model, eval_records[0], word_to_id, id_to_word, device, args.generation_tokens)
    print("\n=== FINAL AUTOREGRESSIVE GENERATION / SEEN EXAMPLE ===")
    print(final_generation)
    print("\n=== FINAL AUTOREGRESSIVE GENERATION / HELDOUT EXAMPLE ===")
    print(heldout_generation)

    print("\n=== VERDICT ===")
    loss_drop = initial_train_metrics["weighted_loss"] - final_train_metrics["weighted_loss"]
    heldout_drop = initial_eval_metrics["weighted_loss"] - final_eval_metrics["weighted_loss"]
    print(f"train_loss_drop={loss_drop:.6f}")
    print(f"heldout_loss_drop={heldout_drop:.6f}")
    print(f"train_answer_acc={final_train_metrics['answer_acc']:.4f}")
    print(f"heldout_answer_acc={final_eval_metrics['answer_acc']:.4f}")
    print(f"train_think_acc={final_train_metrics['think_acc']:.4f}")
    print(f"heldout_think_acc={final_eval_metrics['think_acc']:.4f}")

    if final_train_metrics["weighted_loss"] < initial_train_metrics["weighted_loss"] * 0.5 and final_train_metrics["answer_acc"] > 0.90:
        print("LEARNABILITY=CONFIRMED")
    else:
        print("LEARNABILITY=WEAK_OR_BROKEN")

    if final_eval_metrics["weighted_loss"] < initial_eval_metrics["weighted_loss"] * 0.8 or final_eval_metrics["answer_acc"] > 0.20:
        print("GENERALIZATION=SIGNAL_PRESENT")
    else:
        print("GENERALIZATION=NOT_DETECTED")

    if final_train_metrics["answer_acc"] > 0.90 and final_eval_metrics["answer_acc"] < 0.10:
        print("DIAGNOSIS=MEMORIZATION_WITHOUT_GENERALIZATION")
    elif final_train_metrics["answer_acc"] < 0.50:
        print("DIAGNOSIS=TRAINING_OBJECTIVE_OR_CAPACITY_PROBLEM")
    else:
        print("DIAGNOSIS=NEEDS_AUTOREGRESSIVE_ERROR_ANALYSIS")

    print("\nInitial generation was captured before any optimizer step:")
    print(initial_generation)
    print("\nExample heldout prompt:")
    print(eval_records[0][0])


if __name__ == "__main__":
    main()
