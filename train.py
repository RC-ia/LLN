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
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000, help="Additional steps to run")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save", default="lln_model.pt")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing checkpoint")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")

    dtype = pick_dtype(args.dtype, device)

    data = build_dataset(args.dataset, args.dictionary, repeats=args.repeats)
    word_to_id, _ = load_dictionary(args.dictionary)

    model_cfg = {
        "vocab_size": len(word_to_id),
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "max_seq_len": max(args.seq_len, 256),
    }

    save_path = Path(args.save)
    checkpoint = None
    start_step = 0
    resumed = False

    if save_path.exists() and not args.no_resume:
        print(f"checkpoint={save_path} found; attempting to resume")
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        saved_cfg = checkpoint.get("config", {})
        comparable = {k: saved_cfg.get(k) for k in model_cfg}

        if comparable != model_cfg:
            print("checkpoint architecture/vocabulary differs from current dataset/config; starting a new model")
            checkpoint = None
        else:
            resumed = True
            start_step = int(checkpoint.get("step", 0))

    model = LLN(**model_cfg).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        saved_optimizer = checkpoint.get("optimizer")
        if saved_optimizer is not None:
            optimizer.load_state_dict(saved_optimizer)
        else:
            print("checkpoint has no optimizer state; resuming model weights with a fresh optimizer")

    n_params = parameter_count(model)
    mb = parameter_size_mb(model, torch.tensor([], dtype=dtype).element_size())
    print(f"device={device} dtype={dtype}")
    print(f"parameters={n_params:,} model_weight_size={mb:.1f} MB")
    print(f"vocab={len(word_to_id)} dataset_ids={len(data):,}")
    print(f"resume={resumed} starting_step={start_step}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))

    model.train()
    start = time.perf_counter()
    last_log = start

    for local_step in range(1, args.steps + 1):
        global_step = start_step + local_step
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

        if local_step == 1 or local_step % args.log_every == 0 or local_step == args.steps:
            now = time.perf_counter()
            elapsed = max(now - last_log, 1e-9)
            window_steps = args.log_every if local_step > args.log_every else local_step
            tokens_s = (args.batch_size * args.seq_len * window_steps) / elapsed
            print(f"step={global_step:6d} loss={loss.item():.6f} tok/s={tokens_s:,.0f}")
            last_log = now

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": start_step + args.steps,
        "config": {
            **model_cfg,
            "dictionary": str(args.dictionary),
            "dataset": str(args.dataset),
        },
    }, save_path)
    print(f"saved={save_path}")
    print(f"elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
