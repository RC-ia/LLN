import argparse
import torch

from lln.data import decode_ids, encode_prompt, load_tokenizer, token_id, tokenizer_fingerprint
from lln.model import LLN


def checkpoint_dtype(checkpoint) -> torch.dtype:
    dtype_name = checkpoint.get("config", {}).get("dtype", "torch.float32")
    if dtype_name in {"torch.float16", "float16"}:
        return torch.float16
    if dtype_name in {"torch.bfloat16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(description="Generate text from an LLN checkpoint")
    parser.add_argument("--model", default="lln_model.pt")
    parser.add_argument("--tokenizer", "--dictionary", dest="tokenizer", default="data/tokenizer.json",
                        help="Path to the exact BPE tokenizer used for training")
    parser.add_argument("--prompt", default="eu gosto de")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("--temperature must be greater than 0")
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0")
    if args.no_repeat_ngram < 0:
        raise ValueError("--no-repeat-ngram must be >= 0")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available")

    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    cfg = checkpoint["config"].copy()
    tokenizer, id_to_token = load_tokenizer(args.tokenizer)
    saved_architecture_version = cfg.get("architecture_version")
    if saved_architecture_version is not None and saved_architecture_version != LLN.ARCHITECTURE_VERSION:
        raise ValueError(
            f"checkpoint architecture version {saved_architecture_version} does not match "
            f"runtime version {LLN.ARCHITECTURE_VERSION}"
        )

    saved_tokenizer_hash = cfg.get("tokenizer_fingerprint")
    if not saved_tokenizer_hash:
        raise ValueError("checkpoint does not contain a tokenizer fingerprint; retrain with LLN v6")
    if saved_tokenizer_hash != tokenizer_fingerprint(tokenizer):
        raise ValueError(
            "tokenizer does not match checkpoint; use the exact tokenizer JSON from training"
        )

    model_keys = {
        "vocab_size", "dim", "layers", "heads", "kv_heads", "max_seq_len",
        "dropout", "rope_theta",
    }
    cfg = {key: value for key, value in cfg.items() if key in model_keys}

    model_dtype = checkpoint_dtype(checkpoint)
    model = LLN(**cfg).to(device=device, dtype=model_dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = torch.tensor(
        [encode_prompt(args.prompt, tokenizer)],
        dtype=torch.long,
        device=device,
    )
    out = model.generate(
        ids,
        max_new_tokens=args.new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        stop_ids={token_id(tokenizer, "</ANSWER>"), token_id(tokenizer, "<EOS>")},
    )[0].tolist()

    print("architecture_version:", LLN.ARCHITECTURE_VERSION)
    print("dtype:", model_dtype)
    print("attention_heads:", model.heads)
    print("kv_heads:", model.kv_heads)
    print("rope_theta:", model.rope_theta)
    print("tokenizer:", args.tokenizer)
    print("tokenizer_vocab_size:", tokenizer.get_vocab_size())
    print("tokenizer_fingerprint:", tokenizer_fingerprint(tokenizer)[:12])
    print("temperature:", args.temperature)
    print("repetition_penalty:", args.repetition_penalty)
    print("no_repeat_ngram:", args.no_repeat_ngram)
    print("input_ids:", ids[0].tolist())
    print("output_ids:", out)
    print("text:", decode_ids(out, tokenizer))


if __name__ == "__main__":
    main()
