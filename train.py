import argparse
import time
from pathlib import Path

import torch

from lln.data import build_dataset, load_dictionary, make_batch
from lln.model import LLN, parameter_count, parameter_size_mb


def pick_dtype(name: str, device: torch.device):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if device.type == "cpu" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 requested but this CPU/PyTorch build does not support it")
        return torch.bfloat16
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description="Train LLN on integer token IDs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save", default="lln_model.pt")
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")

    dtype = pick_dtype(args.dtype, device)
    word_to_id, id_to_word = load_dictionary("data/dictionary.json")
    data = build_dataset(word_to_id, repeats=args.repeats)

    model = LLN(
        vocab_size=len(word_to_id),
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        max_seq_len=max(args.seq_len, 256),
    ).to(device=device, dtype=dtype)

    n_params = parameter_count(model)
    mb = parameter_size_mb(model, torch.tensor([], dtype=dtype).element_size())
    print(f"device={device} dtype={dtype}")
    print(f"parameters={n_params:,} model_weight_size={mb:.1f} MB")
    print(f"vocab={len(word_to_id)} dataset_ids={len(data):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    model.train()
    total_tokens = 0
    start = time.perf_counter()
    last_log = start

    for step in range(1, args.steps + 1):
        x, y = make_batch(data, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_tokens += x.numel()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            elapsed = max(now - last_log, 1e-9)
            tokens_s = (args.batch_size * args.seq_len * args.log_every) / elapsed if step > args.log_every else total_tokens / max(now - start, 1e-9)
            print(f"step={step:6d} loss={loss.item():.6f} tok/s={tokens_s:,.0f}")
            last_log = now

    save_path = Path(args.save)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": len(word_to_id),
            "dim": args.dim,
            "layers": args.layers,
            "heads": args.heads,
            "max_seq_len": max(args.seq_len, 256),
            "dictionary": "data/dictionary.json",
        },
    }, save_path)
    print(f"saved={save_path}")
    print(f"elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
