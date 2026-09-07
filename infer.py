import argparse
import torch

from lln.data import load_dictionary, encode_sentence, decode_ids
from lln.model import LLN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lln_model.pt")
    parser.add_argument("--prompt", default="eu gosto de")
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    word_to_id, id_to_word = load_dictionary("data/dictionary.json")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = LLN(**cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = torch.tensor([encode_sentence(args.prompt, word_to_id)[:-1]], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=args.new_tokens)[0].tolist()
    print("input_ids:", ids[0].tolist())
    print("output_ids:", out)
    print("text:", decode_ids(out, id_to_word))


if __name__ == "__main__":
    main()
