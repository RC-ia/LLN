import argparse
from pathlib import Path

from lln.data import DEFAULT_TOKENIZER, SPECIAL_TOKENS, train_tokenizer_from_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train a ByteLevel BPE tokenizer for LLN"
    )
    parser.add_argument("dataset", help="Path to a UTF-8 .txt or structured JSON dataset")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TOKENIZER),
        help="Output tokenizer JSON (keep this exact file for training and inference)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Target vocabulary size, including special tokens and byte alphabet",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum occurrence count for a learned BPE merge",
    )
    args = parser.parse_args()

    tokenizer = train_tokenizer_from_dataset(
        args.dataset,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"dataset={Path(args.dataset)}")
    print(f"tokenizer={Path(args.output)}")
    print(f"vocab={tokenizer.get_vocab_size():,}")
    print(f"target_vocab={args.vocab_size:,}")
    print(f"special_tokens={len(SPECIAL_TOKENS)}")
    for special in SPECIAL_TOKENS:
        print(f"special_id[{special}]={tokenizer.token_to_id(special)}")


if __name__ == "__main__":
    main()
