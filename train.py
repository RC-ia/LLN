import argparse
import time
from pathlib import Path

import torch

from lln.data import build_dataset, load_dictionary, make_batch
from lln.model import LLN, parameter_count, parameter_size_mb


LOSS_SCHEME_VERSION = 4
ARCHITECTURE_VERSION = LLN.ARCHITECTURE_VERSION


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


def sections_to_weights(sections: torch.Tensor, think_weight: float, answer_weight: float) -> torch.Tensor:
    if think_weight < 0.0 or answer_weight <= 0.0:
        raise ValueError("think_weight must be >= 0 and answer_weight must be > 0")
    return torch.where(
        sections.eq(1),
        torch.full_like(sections, think_weight, dtype=torch.float32),
        torch.where(
            sections.eq(2),
            torch.full_like(sections, answer_weight, dtype=torch.float32),
            torch.zeros_like(sections, dtype=torch.float32),
        ),
    )


def make_master_parameters(model: torch.nn.Module):
    return [torch.nn.Parameter(p.detach().float().clone(), requires_grad=True) for p in model.parameters()]


def copy_model_to_master(model: torch.nn.Module, master_params) -> None:
    with torch.no_grad():
        for model_param, master_param in zip(model.parameters(), master_params):
            master_param.copy_(model_param.float())


def copy_master_to_model(model: torch.nn.Module, master_params) -> None:
    with torch.no_grad():
        for model_param, master_param in zip(model.parameters(), master_params):
            model_param.copy_(master_param.to(dtype=model_param.dtype))


def copy_grads_to_master(model: torch.nn.Module, master_params) -> None:
    for model_param, master_param in zip(model.parameters(), master_params):
        if model_param.grad is None:
            master_param.grad = None
        else:
            grad = model_param.grad.detach().float()
            if master_param.grad is None:
                master_param.grad = grad.clone()
            else:
                master_param.grad.copy_(grad)


def main():
    parser = argparse.ArgumentParser(description="Train LLN on bounded complete examples")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256, help="Maximum context per training example")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000, help="Additional steps to run")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--think-weight", type=float, default=0.25)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save", default="lln_model.pt")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.seq_len < 8:
        raise ValueError("--seq-len must be at least 8")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")

    dtype = pick_dtype(args.dtype, device)

    data = build_dataset(
        args.dataset,
        args.dictionary,
        repeats=args.repeats,
        seed=1234,
        max_len=args.seq_len,
        return_sections=True,
    )
    word_to_id, _ = load_dictionary(args.dictionary)

    max_record_len = max(len(ids) for ids, _ in data)
    model_cfg = {
        "vocab_size": len(word_to_id),
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "max_seq_len": args.seq_len,
    }

    save_path = Path(args.save)
    checkpoint = None
    start_step = 0
    resumed = False
    checkpoint_grad_scale = None

    if save_path.exists() and not args.no_resume:
        print(f"checkpoint={save_path} found; attempting to resume")
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        saved_cfg = checkpoint.get("config", {})
        comparable = {k: saved_cfg.get(k) for k in model_cfg}
        if comparable != model_cfg:
            print("checkpoint architecture/vocabulary or sequence length differs; starting a new model")
            checkpoint = None
        elif saved_cfg.get("architecture_version") != ARCHITECTURE_VERSION:
            print("checkpoint uses an older architecture; starting a new model")
            checkpoint = None
        elif saved_cfg.get("loss_scheme_version") != LOSS_SCHEME_VERSION:
            print("checkpoint uses an older training objective; starting a new model")
            checkpoint = None
        elif saved_cfg.get("think_weight") != args.think_weight or saved_cfg.get("answer_weight") != args.answer_weight:
            print("checkpoint loss weights differ from current config; starting a new model")
            checkpoint = None
        else:
            resumed = True
            start_step = int(checkpoint.get("step", 0))
            checkpoint_grad_scale = checkpoint.get("grad_scale")

    model = LLN(**model_cfg).to(device=device, dtype=dtype)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])

    master_params = make_master_parameters(model)
    optimizer = torch.optim.AdamW(master_params, lr=args.lr, weight_decay=0.01)
    if checkpoint is not None:
        saved_optimizer = checkpoint.get("optimizer")
        saved_master = checkpoint.get("master_params")
        if saved_master is not None and len(saved_master) == len(master_params):
            for master_param, saved_value in zip(master_params, saved_master):
                master_param.data.copy_(saved_value.to(device=device, dtype=torch.float32))
        else:
            copy_model_to_master(model, master_params)
        if saved_optimizer is not None:
            try:
                optimizer.load_state_dict(saved_optimizer)
            except (ValueError, RuntimeError):
                print("checkpoint optimizer state incompatible; reinitializing optimizer")
        copy_master_to_model(model, master_params)

    n_params = parameter_count(model)
    mb = parameter_size_mb(model, torch.tensor([], dtype=dtype).element_size())
    master_mb = parameter_size_mb(model, 4)
    print(f"device={device} dtype={dtype}")
    print(f"parameters={n_params:,} model_weight_size={mb:.1f} MB")
    print(f"master_weight_size={master_mb:.1f} MB")
    print(f"vocab={len(word_to_id)} records={len(data):,} max_training_record_ids={max_record_len:,}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size}")
    print(f"loss_weights=prompt:0 think:{args.think_weight:g} answer:{args.answer_weight:g}")
    print("long_record_policy=preserve_prompt_and_answer_truncate_think")
    print("optimizer=AdamW fp32_master_params")
    print(f"architecture_version={ARCHITECTURE_VERSION}")
    print(f"resume={resumed} starting_step={start_step}")

    grad_scale = float(checkpoint_grad_scale) if checkpoint_grad_scale is not None else (
        1024.0 if (device.type == "cuda" and dtype == torch.float16) else 1.0
    )
    if grad_scale <= 0.0:
        grad_scale = 1024.0 if (device.type == "cuda" and dtype == torch.float16) else 1.0
    print(f"grad_scale_start={grad_scale:g}")

    model.train()
    start = time.perf_counter()
    last_log = start

    for local_step in range(1, args.steps + 1):
        global_step = start_step + local_step
        x, y, batch_sections = make_batch(data, args.batch_size, args.seq_len, device)
        loss_weights = sections_to_weights(batch_sections, args.think_weight, args.answer_weight)
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            _, loss = model(x, y, loss_weights=loss_weights)

        if grad_scale != 1.0:
            (loss * grad_scale).backward()
            found_inf = False
            inv_scale = 1.0 / grad_scale
            for param in model.parameters():
                if param.grad is None:
                    continue
                param.grad.data.mul_(inv_scale)
                if not torch.isfinite(param.grad).all():
                    found_inf = True
                    break
            if found_inf:
                model.zero_grad(set_to_none=True)
                grad_scale = max(1.0, grad_scale / 2.0)
                if local_step == 1 or local_step % args.log_every == 0:
                    print(f"step={global_step:6d} skipped=nonfinite_grad grad_scale={grad_scale:g}")
                continue
        else:
            loss.backward()

        copy_grads_to_master(model, master_params)
        torch.nn.utils.clip_grad_norm_(master_params, 1.0)

        if not all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in master_params
        ):
            model.zero_grad(set_to_none=True)
            for p in master_params:
                p.grad = None
            grad_scale = max(1.0, grad_scale / 2.0)
            continue

        optimizer.step()
        copy_master_to_model(model, master_params)
        model.zero_grad(set_to_none=True)
        for p in master_params:
            p.grad = None

        if dtype == torch.float16 and device.type == "cuda" and grad_scale < 65536.0:
            grad_scale = min(65536.0, grad_scale * 1.001)

        if local_step == 1 or local_step % args.log_every == 0 or local_step == args.steps:
            now = time.perf_counter()
            elapsed = max(now - last_log, 1e-9)
            window_steps = args.log_every if local_step > args.log_every else local_step
            tokens_s = (args.batch_size * args.seq_len * window_steps) / elapsed
            print(f"step={global_step:6d} loss={loss.item():.6f} tok/s={tokens_s:,.0f} grad_scale={grad_scale:g}")
            last_log = now

    torch.save({
        "model": model.state_dict(),
        "master_params": [p.detach().cpu() for p in master_params],
        "optimizer": optimizer.state_dict(),
        "step": start_step + args.steps,
        "grad_scale": grad_scale,
        "config": {
            **model_cfg,
            "dictionary": str(args.dictionary),
            "dataset": str(args.dataset),
            "loss_scheme_version": LOSS_SCHEME_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "think_weight": args.think_weight,
            "answer_weight": args.answer_weight,
            "dtype": str(dtype),
            "optimizer": "AdamW_fp32_master",
        },
    }, save_path)
    print(f"saved={save_path}")
    print(f"elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
