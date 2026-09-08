import argparse
import math
import random

import torch

from lln.data import (
    SECTION_ANSWER,
    SECTION_THINK,
    decode_ids,
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
    rows = []
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
        rows.append((name, norm))
    rows.sort(key=lambda item: item[1], reverse=True)
    return math.sqrt(total_sq), nonzero, finite, max_abs, rows[:12]


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


def print_alignment(x, y, sections, id_to_word, limit=64):
    print("\n=== TOKEN ALIGNMENT (example 0) ===")
    row_x = x[0].detach().cpu().tolist()
    row_y = y[0].detach().cpu().tolist()
    row_s = sections[0].detach().cpu().tolist()
    for i, (xi, yi, sec) in enumerate(zip(row_x, row_y, row_s)):
        if i >= limit:
            print(f"... ({len(row_x) - limit} more positions)")
            break
        if sec < 0:
            continue
        sx = id_to_word.get(int(xi), "<UNK>")
        sy = id_to_word.get(int(yi), "<UNK>")
        print(f"pos={i:4d} input={xi:5d} {sx!r:24s} -> target={yi:5d} {sy!r:24s} section={int(sec)}")


def print_prediction(logits, y, sections, id_to_word, topk=5, limit=32):
    print("\n=== PREDICTIONS (example 0) ===")
    probs = torch.softmax(logits[0].float(), dim=-1)
    shown = 0
    for i in range(logits.size(1)):
        if sections[0, i].item() < 0:
            continue
        values, indices = torch.topk(probs[i], k=topk)
        target = int(y[0, i])
        best = int(indices[0])
        target_rank = int((probs[i] > probs[i, target]).sum().item()) + 1
        choices = ", ".join(
            f"{id_to_word.get(int(t), '<UNK>')}:{float(v):.3f}" for v, t in zip(values, indices)
        )
        print(
            f"pos={i:4d} target={id_to_word.get(target, '<UNK>')!r:20s} "
            f"pred={id_to_word.get(best, '<UNK>')!r:20s} rank={target_rank:4d} top={choices}"
        )
        shown += 1
        if shown >= limit:
            break


def main():
    parser = argparse.ArgumentParser(description="LLN isolated training debugger / tiny-batch overfit test")
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
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-think-weight", action="store_true")
    args = parser.parse_args()

    if args.examples < 1 or args.batch_size < 1 or args.steps < 1:
        raise ValueError("examples, batch-size and steps must be >= 1")
    if args.heads < 1 or args.dim % args.heads:
        raise ValueError("dim must be divisible by heads")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = pick_dtype(args.dtype)

    word_to_id, id_to_word, token_types, meta = load_dictionary(args.dictionary, with_metadata=True)
    records = load_records(args.dataset)
    encoded = [encode_record(record, word_to_id) for record in records[: args.examples]]
    if not encoded:
        raise RuntimeError("No examples available")

    selected = encoded[: min(args.batch_size, len(encoded))]
    if len(selected) < args.batch_size:
        selected = (selected * ((args.batch_size + len(selected) - 1) // len(selected)))[: args.batch_size]

    x, y, sections = make_batch(
        selected,
        args.batch_size,
        args.seq_len,
        device,
        indices=list(range(args.batch_size)),
    )
    weights = section_weights(sections, think_weight=1.0 if args.no_think_weight else 0.25)

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
    print("LLN TRAINING DEBUG / TINY-BATCH OVERFIT")
    print("=" * 72)
    print(f"device={device} dtype={dtype}")
    print(f"parameters={parameter_count(model):,}")
    print(f"model_weight_size={parameter_size_mb(model, torch.tensor([], dtype=dtype).element_size()):.1f} MB")
    print(f"vocab={len(word_to_id):,} dataset_records={len(records):,} debug_examples={len(encoded)}")
    print(f"batch_size={args.batch_size} seq_len={args.seq_len} steps={args.steps} lr={args.lr:g}")
    print(f"recurrent_steps={args.recurrent_steps} output_clusters={args.output_clusters} memory_slots={args.memory_slots}")
    print(f"loss_weights=think:{1.0 if args.no_think_weight else 0.25:g} answer:1 prompt:0")
    print("shuffle=disabled fixed_batch=yes")

    print_alignment(x, y, sections, id_to_word)

    model.eval()
    with torch.no_grad():
        logits0, loss0 = model(x, y, loss_weights=weights)
    print(f"\ninitial_weighted_loss={float(loss0):.6f}")
    print(f"initial_baseline_ln_vocab={math.log(len(word_to_id)):.6f}")
    print_prediction(logits0, y, sections, id_to_word)

    optimizer.zero_grad(set_to_none=True)
    model.train()
    logits, loss = model(x, y, loss_weights=weights)
    loss.backward()
    grad_norm, nonzero, finite, grad_max, grad_rows = grad_stats(model)
    print("\n=== BACKWARD DIAGNOSTIC ===")
    print(f"loss={float(loss):.6f}")
    print(f"grad_global_norm={grad_norm:.6g} grad_nonzero_tensors={nonzero} grad_finite={finite} grad_max_abs={grad_max:.6g}")
    print("largest-gradient-tensors:")
    for name, norm in grad_rows:
        print(f"  {name}: {norm:.6g}")

    initial_params = [p.detach().float().clone() for p in model.parameters()]
    best_loss = float(loss)
    print("\n=== OVERFIT LOOP ===")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        model.train()
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
        best_loss = min(best_loss, value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            delta, delta_max, changed = parameter_delta(initial_params, model)
            print(
                f"step={step:4d} loss={value:.6f} delta_norm={delta:.6g} "
                f"delta_max={delta_max:.6g} changed_values={changed:,} "
                f"grad_norm={grad_norm:.6g} grad_max={grad_max:.6g}"
            )

    model.eval()
    with torch.no_grad():
        logits_final, final_loss = model(x, y, loss_weights=weights)
    delta, delta_max, changed = parameter_delta(initial_params, model)
    print("\n=== VERDICT ===")
    print(f"initial_loss={float(loss0):.6f}")
    print(f"final_loss={float(final_loss):.6f}")
    print(f"best_loss={best_loss:.6f}")
    print(f"loss_drop={float(loss0) - float(final_loss):.6f}")
    print(f"parameter_delta_norm={delta:.6g} parameter_delta_max={delta_max:.6g} changed_values={changed:,}")
    if float(final_loss) < float(loss0) * 0.5:
        print("RESULT=LEARNING_SIGNAL_CONFIRMED")
    elif delta > 0.0 and float(final_loss) < float(loss0):
        print("RESULT=WEAK_LEARNING_OR_OPTIMIZATION_ISSUE")
    elif delta == 0.0:
        print("RESULT=PARAMETERS_DID_NOT_CHANGE")
    else:
        print("RESULT=NO_MEANINGFUL_OVERFIT")

    print_prediction(logits_final, y, sections, id_to_word)
    print("\nExample decoded input:")
    print(decode_ids(x[0].detach().cpu().tolist(), id_to_word))
    print("Example decoded target:")
    print(decode_ids(y[0].detach().cpu().tolist(), id_to_word))


if __name__ == "__main__":
    main()
