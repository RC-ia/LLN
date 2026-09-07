import argparse
from pathlib import Path

from lln.data import create_dictionary_from_dataset


def main():
    parser = argparse.ArgumentParser(description="Create an integer ID dictionary from a text dataset")
    parser.add_argument("dataset", help="Path to a UTF-8 text dataset")
    parser.add_argument("--output", default="data/dictionary.json", help="Output dictionary JSON")
    args = parser.parse_args()

    word_to_id = create_dictionary_from_dataset(args.dataset, args.output)
    print(f"dataset={Path(args.dataset)}")
    print(f"dictionary={Path(args.output)}")
    print(f"vocab={len(word_to_id)}")
    print(f"special_tokens=4")


if __name__ == "__main__":
    main()
