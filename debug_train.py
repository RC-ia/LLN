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
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


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
    return math.sqrt(total_sq), nonzero, finite, max_abs, {k: math.sqrt(v) for k, v in groups.items()}


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


def batch_from_indices(encoded, indices, batch_size, device, seq_len):
    if not indices:
        raise RuntimeError("No indices supplied")
    chosen = [encoded[i] for i in indices]
    if len(chosen) < batch_size:
        chosen = (chosen * ((batch_size + len(chosen) - 1) // len(chosen)))[:batch_size]
    return make_batch(chosen, batch_size, seq_len, device, indices=list(range(batch_size)))


def evaluate_dataset(model, encoded, device, seq_len, batch_size, return_example=None):
    sums = defaultdict(float)
    counts = defaultdict(int)
    total_weighted_sum = 0.0
    total_weight = 0.0
    total_correct = defaultdict(int)
    total_tokens = defaultdict(int)
    total_target_prob = 0.0
    total_target_rank = 0.0
    total_answer_tokens = 0
    captured = None

    model.eval()
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(encoded))))
            actual_bs = len(batch_indices)
            x, y, sections = batch_from_indices(encoded, batch_indices, actual_bs, device, seq_len)
            weights = section_weights(sections)
            logits, _ = model(x, y, loss_weights=weights)
            token_loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, model.vocab_size), y.reshape(-1), reduction="none"
            ).reshape_as(y)
            valid = sections >= 0
            predictions = logits.argmax(dim=-1)
            target_probs = torch.softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
            target_rank = (logits.float() > logits.float().gather(-1, y.unsqueeze(-1))).sum(-1) + 1

            valid_weight = weights[valid]
            valid_loss = token_loss[valid]
            total_weighted_sum += float((valid_loss * valid_weight).sum().item())
            total_weight += float(valid_weight.sum().item())

            for name, sec_id in (("prompt", SECTION_PROMPT), ("think", SECTION_THINK), ("answer", SECTION_ANSWER)):
                mask = valid & sections.eq(sec_id)
                if mask.any():
                    n = int(mask.sum().item())
                    sums[f"{name}_loss"] += float(token_loss[mask].sum().item())
                    total_correct[name] += int((predictions[mask] == y[mask]).sum().item())
                    total_tokens[name] += n
                    counts[name] += n

            answer_mask = valid & sections.eq(SECTION_ANSWER)
            if answer_mask.any():
                total_target_prob += float(target_probs[answer_mask].sum().item())
                total_target_rank += float(target_rank[answer_mask].float().sum().item())
                total_answer_tokens += int(answer_mask.sum().item())

            if return_example is not None and captured is None and return_example in batch_indices:
                local = batch_indices.index(return_example)
                captured = (x[local:local + 1].clone(), y[local:local + 1].clone(), sections[local:local + 1].clone(), logits[local:local + 1].clone())

    metrics = {"weighted_loss": total_weighted_sum / max(total_weight, 1e-12)}
    for name in ("prompt", "think", "answer"):
        n = total_tokens[name]
        metrics[f"{name}_loss"] = sums[f"{name}_loss"] / max(n, 1)
        metrics[f"{name}_acc"] = total_correct[name] / max(n, 1)
        metrics[f"{name}_count"] = n
    metrics["answer_target_p"] = total_target_prob / max(total_answer_tokens, 1)
    metrics["answer_target_rank"] = total_target_rank / max(total_answer_tokens, 1)
    return metrics, captured


def print_metrics(prefix, metrics):
    print(
        f"{prefix} weighted={metrics['weighted_loss']:.6f} "
        f"prompt_loss={metrics['prompt_loss']:.6f} prompt_acc={metrics['prompt_acc']:.3f} "
        f"think_loss={metrics['think_loss']:.6f} think_acc={metrics['think_acc']:.3f} "
        f"answer_loss={metrics['answer_loss']:.6f} answer_acc={metrics['answer_acc']:.3f} "
        f"answer_p={metrics['answer_target_p']:.4f} answer_rank={metrics['answer_target_rank']:.2f}"
    )


def print_alignment(x, y, sections, id_to_word, limit=64):
    print("\n=== TOKEN ALIGNMENT (example 0) ===")
    row_x = x[0].detach().cpu().tolist()
    row_y = y[0].detach().cpu().tolist()
    row_s = sections[0].detach().cpu().tolist()
    shown = 0
    for i, (xi, yi, sec) in enumerate(zip(row_x, row_y, row_s)):
        if sec < 0:
            continue
        print(f"pos={i:4d} input={xi:5d} {id_to_word.get(xi, '<UNK>')!r:24s} -> target={yi:5d} {id_to_word.get(yi, '<UNK>')!r:24s} section={int(sec)}")
        shown += 1
        if shown >= limit:
            break


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
        choices = ", ".join(f"{id_to_word.get(int(t), '<UNK>')}:{float(v):.3f}" for v, t in zip(values, indices))
        print(f"pos={i:4d} target={id_to_word.get(target, '<UNK>')!r:18s} rank={target_rank:4d} top={choices}")
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
    return float((base_logits[:, :probe_pos + 1].float() - changed_logits[:, :probe_pos + 1].float()).abs().max().item())


def generation_test(model, record, word_to_id, id_to_word, device, max_new_tokens):
    input_ids = torch.tensor([encode_prompt(record[0], word_to_id)], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(
        input_ids,
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
    return decode_ids(out[0].detach().cpu().tolist(), id_to_word)


def main():
    parser = argparse.ArgumentParser(description="LLN diagnostic suite: full-dataset learning, held-out generalization and autoregressive behavior")
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
    parser.add_argument("--examples", type=int, default=8, help="Training examples; all are used over repeated batches")
    parser.add_argument("--eval-examples", type=int, default=8, help="Held-out examples immediately after the training split")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--generation-tokens", type=int, default=80)
    parser.add_argument("--no-think-weight", action="store_true")
    args = parser.parse_args()

    if min(args.examples, args.eval_examples, args.batch_size, args.steps) < 1:
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

    # One fixed example is kept only for detailed alignment/prediction display.
    x0, y0, sections0 = batch_from_indices(train_encoded, [0], 1, device, args.seq_len)

    model_cfg = {
        "vocab_size": len(word_to_id), "dim": args.dim, "layers": args.layers, "heads": args.heads,
        "max_seq_len": args.seq_len, "recurrent_steps": args.recurrent_steps,
        "output_clusters": args.output_clusters, "memory_slots": args.memory_slots,
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
    print(f"train_examples={len(train_encoded)} heldout_examples={len(eval_encoded)}")
    print(f"batch_size={args.batch_size} seq_len={args.seq_len} steps={args.steps} lr={args.lr:g}")
    print(f"recurrent_steps={args.recurrent_steps} output_clusters={args.output_clusters} memory_slots={args.memory_slots}")
    print(f"loss_weights=think:{1.0 if args.no_think_weight else 0.25:g} answer:1 prompt:0")
    print("shuffle=train_each_epoch_without_replacement fixed_batch=no")

    print_alignment(x0, y0, sections0, id_to_word)
    _, initial_train_example = evaluate_dataset(model, train_encoded, device, args.seq_len, args.batch_size, return_example=0)
    initial_train_metrics, _ = evaluate_dataset(model, train_encoded, device, args.seq_len, args.batch_size)
    initial_eval_metrics, _ = evaluate_dataset(model, eval_encoded, device, args.seq_len, args.batch_size)
    print("\n=== INITIAL METRICS ===")
    print_metrics("TRAIN", initial_train_metrics)
    print_metrics("HELDOUT", initial_eval_metrics)
    print(f"baseline_ln_vocab={math.log(len(word_to_id)):.6f}")
    if initial_train_example is not None:
        print_top_predictions(initial_train_example[3], initial_train_example[1], initial_train_example[2], id_to_word)
    print(f"causality_max_abs_diff={causality_check(model, x0):.6g}")

    print("\n=== BACKWARD DIAGNOSTIC ===")
    weights0 = section_weights(sections0, think_weight=1.0 if args.no_think_weight else 0.25)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    _, loss = model(x0, y0, loss_weights=weights0)
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

    # Full training split, with every example visited each epoch. The last partial
    # batch is kept at its actual size; no example is silently excluded.
    order = list(range(len(train_encoded)))
    rng = random.Random(args.seed)
    best_train = float("inf")
    batches_per_epoch = math.ceil(len(train_encoded) / args.batch_size)
    epoch = 0
    seen = set()

    print("\n=== TRAINING LOOP ===")
    for step in range(1, args.steps + 1):
        if step == 1 or (step - 1) % batches_per_epoch == 0:
            rng.shuffle(order)
            epoch += 1
            seen.clear()
        batch_indices = order[((step - 1) % batches_per_epoch) * args.batch_size: ((step - 1) % batches_per_epoch + 1) * args.batch_size]
        seen.update(batch_indices)
        actual_bs = len(batch_indices)
        x, y, sections = batch_from_indices(train_encoded, batch_indices, actual_bs, device, args.seq_len)
        weights = section_weights(sections, think_weight=1.0 if args.no_think_weight else 0.25)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y, loss_weights=weights)
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
        best_train = min(best_train, value)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            delta, delta_max, changed = parameter_delta(initial_params, model)
            print(f"step={step:4d} epoch={epoch:3d} train_examples_seen_in_epoch={len(seen):4d}/{len(train_encoded)} loss={value:.6f} delta_norm={delta:.6g} delta_max={delta_max:.6g} changed_values={changed:,} grad_norm={grad_norm:.6g} grad_max={grad_max:.6g}")

    final_train_metrics, final_train_example = evaluate_dataset(model, train_encoded, device, args.seq_len, args.batch_size, return_example=0)
    final_eval_metrics, final_eval_example = evaluate_dataset(model, eval_encoded, device, args.seq_len, args.batch_size, return_example=0)
    delta, delta_max, changed = parameter_delta(initial_params, model)
    print("\n=== FINAL METRICS ===")
    print_metrics("TRAIN_ALL", final_train_metrics)
    print_metrics("HELDOUT_ALL", final_eval_metrics)
    print(f"parameter_delta_norm={delta:.6g} parameter_delta_max={delta_max:.6g} changed_values={changed:,}")
    if final_train_example is not None:
        print_top_predictions(final_train_example[3], final_train_example[1], final_train_example[2], id_to_word)
    print(f"causality_max_abs_diff_after={causality_check(model, x0):.6g}")

    final_generation = generation_test(model, train_records[0], word_to_id, id_to_word, device, args.generation_tokens)
    heldout_generation = generation_test(model, eval_records[0], word_to_id, id_to_word, device, args.generation_tokens)
    print("\n=== FINAL AUTOREGRESSIVE GENERATION / SEEN EXAMPLE ===")
    print(final_generation)
    print("\n=== FINAL AUTOREGRESSIVE GENERATION / HELDOUT EXAMPLE ===")
    print(heldout_generation)

    print("\n=== VERDICT ===")
    train_loss_drop = initial_train_metrics["weighted_loss"] - final_train_metrics["weighted_loss"]
    heldout_loss_drop = initial_eval_metrics["weighted_loss"] - final_eval_metrics["weighted_loss"]
    print(f"train_loss_drop={train_loss_drop:.6f}")
    print(f"heldout_loss_drop={heldout_loss_drop:.6f}")
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

    print("\nExample heldout prompt:")
    print(eval_records[0][0])


if __name__ == "__main__":
    main()
