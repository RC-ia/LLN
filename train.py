import argparse
import math
import random
import time
from pathlib import Path

import torch

from lln.data import build_dataset, load_dictionary, make_batch
from lln.model import LLN, parameter_count, parameter_size_mb


LOSS_SCHEME_VERSION = 5
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


def learning_rate_at(step: int, base_lr: float, min_lr: float, warmup_steps: int, total_schedule_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        frac = step / float(warmup_steps)
        return base_lr * max(frac, 1e-3)
    decay_span = max(1, total_schedule_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(decay_span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


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
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--think-weight", type=float, default=0.25)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--recurrent-steps", type=int, default=2, help="Times the same transformer blocks are reused")
    parser.add_argument("--output-clusters", type=int, default=120, help="Factorized softmax cluster count")
    parser.add_argument("--memory-slots", type=int, default=4, help="Latent scratchpad slots")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save", default="lln_model.pt")
    parser.add_argument("--repeats", type=int, default=2000, help="Examples per prepared epoch; defaults to one full dataset pass")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.seq_len < 8:
        raise ValueError("--seq-len must be at least 8")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.min_lr <= 0.0 or args.min_lr > args.lr:
        raise ValueError("--min-lr must be > 0 and <= --lr")
    if args.recurrent_steps < 1:
        raise ValueError("--recurrent-steps must be >= 1")
    if args.output_clusters < 1:
        raise ValueError("--output-clusters must be >= 1")
    if args.memory_slots < 1:
        raise ValueError("--memory-slots must be >= 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is not available")

    dtype = pick_dtype(args.dtype, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = build_dataset(
        args.dataset,
        args.dictionary,
        repeats=args.repeats,
        seed=args.seed,
        max_len=args.seq_len,
        return_sections=True,
    )
    word_to_id, _, token_types, dictionary_meta = load_dictionary(args.dictionary, with_metadata=True)

    model_cfg = {
        "vocab_size": len(word_to_id),
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "max_seq_len": args.seq_len,
        "recurrent_steps": args.recurrent_steps,
        "output_clusters": args.output_clusters,
        "memory_slots": args.memory_slots,
        "type_count": int(dictionary_meta.get("type_count", 6)),
    }

    save_path = Path(args.save)
    checkpoint = None
    start_step = 0
    resumed = False
    checkpoint_grad_scale = None
    checkpoint_epoch = 0
    checkpoint_cursor = 0

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
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            checkpoint_cursor = int(checkpoint.get("epoch_cursor", 0))

    model = LLN(**model_cfg).to(device=device, dtype=dtype)
    model.set_token_types(token_types)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    use_multi_gpu = device.type == "cuda" and gpu_count > 1
    train_model = model
    if use_multi_gpu:
        if args.batch_size < gpu_count:
            raise ValueError(
                f"--batch-size={args.batch_size} is too small for {gpu_count} GPUs; "
                f"use at least --batch-size {gpu_count} to activate all GPUs"
            )
        train_model = torch.nn.DataParallel(
            model,
            device_ids=list(range(gpu_count)),
            output_device=0,
        )

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
    if use_multi_gpu:
        print(f"parallel_training=DataParallel gpus={gpu_count} device_ids={list(range(gpu_count))}")
        print("parallel_batch_policy=each GPU receives approximately batch_size/gpu_count examples")
    else:
        print("parallel_training=disabled")
    print(f"parameters={n_params:,} model_weight_size={mb:.1f} MB")
    print(f"master_weight_size={master_mb:.1f} MB")
    print(f"vocab={len(word_to_id):,} records={len(data):,}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size}")
    print(f"recurrent_steps={args.recurrent_steps} output_clusters={args.output_clusters} cluster_size={model.lm_head.cluster_size} memory_slots={args.memory_slots}")
    print(f"token_types={model_cfg['type_count']} dictionary_metadata={dictionary_meta.get('version', 1)}")
    print(f"loss_weights=prompt:0 think:{args.think_weight:g} answer:{args.answer_weight:g}")
    print("long_record_policy=preserve_prompt_and_answer_truncate_think")
    print("optimizer=AdamW fp32_master_params")
    print(f"architecture_version={ARCHITECTURE_VERSION}")
    print(f"training_schedule=warmup:{args.warmup_steps} max_lr:{args.lr:g} min_lr:{args.min_lr:g}")
    print("sampling_policy=shuffled_epoch_without_replacement")
    print(f"resume={resumed} starting_step={start_step} epoch={checkpoint_epoch} cursor={checkpoint_cursor}")

    grad_scale = float(checkpoint_grad_scale) if checkpoint_grad_scale is not None else (
        1024.0 if (device.type == "cuda" and dtype == torch.float16) else 1.0
    )
    if grad_scale <= 0.0:
        grad_scale = 1024.0 if (device.type == "cuda" and dtype == torch.float16) else 1.0
    print(f"grad_scale_start={grad_scale:g}")

    epoch = checkpoint_epoch
    cursor = checkpoint_cursor
    order = list(range(len(data)))
    random.Random(args.seed + epoch).shuffle(order)
    if cursor >= len(order):
        epoch += 1
        cursor = 0
        order = list(range(len(data)))
        random.Random(args.seed + epoch).shuffle(order)

    total_schedule_steps = max(1, start_step + args.steps)
    train_model.train()
    start = time.perf_counter()
    last_log = start

    for local_step in range(1, args.steps + 1):
        global_step = start_step + local_step
        if cursor + args.batch_size > len(order):
            epoch += 1
            cursor = 0
            order = list(range(len(data)))
            random.Random(args.seed + epoch).shuffle(order)
        batch_indices = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size

        x, y, batch_sections = make_batch(data, args.batch_size, args.seq_len, device, indices=batch_indices)
        loss_weights = sections_to_weights(batch_sections, args.think_weight, args.answer_weight)
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        current_lr = learning_rate_at(global_step, args.lr, args.min_lr, args.warmup_steps, total_schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            _, loss = train_model(x, y, loss_weights=loss_weights)
            if use_multi_gpu:
                loss = loss.mean()

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
                cursor -= args.batch_size
                if local_step == 1 or local_step % args.log_every == 0:
                    print(f"step={global_step:6d} skipped=nonfinite_grad grad_scale={grad_scale:g}")
                continue
        else:
            loss.backward()

        copy_grads_to_master(model, master_params)
        torch.nn.utils.clip_grad_norm_(master_params, 1.0)

        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in master_params):
            model.zero_grad(set_to_none=True)
            for p in master_params:
                p.grad = None
            grad_scale = max(1.0, grad_scale / 2.0)
            cursor -= args.batch_size
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
            print(f"step={global_step:6d} loss={loss.item():.6f} tok/s={tokens_s:,.0f} lr={current_lr:.6g} grad_scale={grad_scale:g}")
            last_log = now

    torch.save({
        "model": model.state_dict(),
        "master_params": [p.detach().cpu() for p in master_params],
        "optimizer": optimizer.state_dict(),
        "step": start_step + args.steps,
        "grad_scale": grad_scale,
        "epoch": epoch,
        "epoch_cursor": cursor,
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
            "schedule": "warmup_cosine",
            "base_lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "sampling_policy": "shuffled_epoch_without_replacement",
            "parallel_training": "DataParallel" if use_multi_gpu else "single_gpu",
            "gpu_count": gpu_count,
        },
    }, save_path)
    print(f"saved={save_path}")
    print(f"elapsed={time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
