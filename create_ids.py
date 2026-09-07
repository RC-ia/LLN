import argparse
from pathlib import Path

from lln.data import create_dictionary_from_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create an integer ID dictionary from a text or structured JSON dataset"
    )
    parser.add_argument("dataset", help="Path to a UTF-8 .txt or structured .json dataset")
    parser.add_argument("--output", default="data/dictionary.json", help="Output dictionary JSON")
    args = parser.parse_args()

    word_to_id = create_dictionary_from_dataset(args.dataset, args.output)
    print(f"dataset={Path(args.dataset)}")
    print(f"dictionary={Path(args.output)}")
    print(f"vocab={len(word_to_id)}")
    print(f"special_tokens=7 (<PAD>, <BOS>, <EOS>, <UNK>, <USER>, <THINK>, <ANSWER>)")


if __name__ == "__main__":
    main()
