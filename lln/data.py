import json
import random
import re
from pathlib import Path

import torch

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
DEFAULT_DATASET = Path("data/dataset.txt")
DEFAULT_DICTIONARY = Path("data/dictionary.json")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def create_dictionary_from_dataset(
    dataset_path: str | Path,
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
):
    """Create a deterministic word -> integer ID dictionary from a text dataset."""
    dataset_path = Path(dataset_path)
    dictionary_path = Path(dictionary_path)

    text = dataset_path.read_text(encoding="utf-8")
    words = normalize_text(text).split()
    unique_words = sorted(set(words))

    word_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for word in unique_words:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.write_text(
        json.dumps(word_to_id, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return word_to_id


def load_dictionary(path: str | Path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    word_to_id = {str(k): int(v) for k, v in raw.items()}
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def encode_sentence(text: str, word_to_id: dict[str, int]) -> list[int]:
    text = normalize_text(text)
    ids = [word_to_id["<BOS>"]]
    for word in text.split():
        ids.append(word_to_id.get(word, word_to_id["<UNK>"]))
    ids.append(word_to_id["<EOS>"])
    return ids


def build_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
    repeats: int = 1,
    seed: int = 1234,
):
    """Read a text corpus, build its dictionary, and return a flat integer dataset."""
    dataset_path = Path(dataset_path)
    word_to_id = create_dictionary_from_dataset(dataset_path, dictionary_path)

    lines = [
        normalize_text(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if normalize_text(line)
    ]
    if not lines:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    rng = random.Random(seed)
    sequences = [encode_sentence(line, word_to_id) for line in lines]
    data = []
    for _ in range(max(1, repeats)):
        seq = list(rng.choice(sequences))
        data.extend(seq)
    return data


def make_batch(data: list[int], batch_size: int, seq_len: int, device):
    if len(data) <= seq_len + 1:
        raise ValueError("Dataset is too small for the requested sequence length")
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + seq_len], dtype=torch.long) for i in starts])
    y = torch.stack([torch.tensor(data[i + 1:i + seq_len + 1], dtype=torch.long) for i in starts])
    return x.to(device), y.to(device)


def decode_ids(ids: list[int], id_to_word: dict[int, str]) -> str:
    words = []
    for idx in ids:
        word = id_to_word.get(int(idx), "<UNK>")
        if word in {"<BOS>", "<PAD>"}:
            continue
        if word == "<EOS>":
            break
        words.append(word)
    return " ".join(words)
