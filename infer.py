import argparse
import torch

from lln.data import load_dictionary, encode_prompt, decode_ids
from lln.model import LLN


def main():
    parser = argparse.ArgumentParser(description="Generate text from an LLN checkpoint")
    parser.add_argument("--model", default="lln_model.pt")
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

    word_to_id, id_to_word = load_dictionary("data/dictionary.json")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    cfg = checkpoint["config"].copy()
    for key in ("dictionary", "dataset", "loss_scheme_version", "think_weight", "answer_weight"):
        cfg.pop(key, None)

    model = LLN(**cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = torch.tensor(
        [encode_prompt(args.prompt, word_to_id)],
        dtype=torch.long,
        device=device,
    )
    out = model.generate(
        ids,
        max_new_tokens=args.new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        stop_ids={word_to_id["</ANSWER>"], word_to_id["<EOS>"]},
    )[0].tolist()

    print("temperature:", args.temperature)
    print("repetition_penalty:", args.repetition_penalty)
    print("no_repeat_ngram:", args.no_repeat_ngram)
    print("input_ids:", ids[0].tolist())
    print("output_ids:", out)
    print("text:", decode_ids(out, id_to_word))


if __name__ == "__main__":
    main()
