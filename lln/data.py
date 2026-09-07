import json
import random
import re
from pathlib import Path

import torch

SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>",
    "<USER>", "<THINK>", "<ANSWER>",
]
DEFAULT_DATASET = Path("data/dataset.json")
DEFAULT_DICTIONARY = Path("data/dictionary.json")


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_records(dataset_path: str | Path) -> list[tuple[str, str | None, str]]:
    """Load plain-text lines or chat records with optional reasoning.

    JSON format:
      [{"messages": [{"role": "user", "content": "..."},
                     {"role": "assistant", "reasoning_content": "...",
                      "content": "..."}]}]

    Each record becomes: <BOS> <USER> user <THINK> reasoning <ANSWER> answer <EOS>.
    Missing reasoning is allowed.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() != ".json":
        lines = [normalize_text(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if normalize_text(line)]
        return [(line, None, "") for line in lines]

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON dataset must contain a top-level list")

    records = []
    for item in raw:
        messages = item.get("messages", []) if isinstance(item, dict) else []
        user = next((m for m in messages if m.get("role") == "user"), None)
        assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
        if not user or not assistant:
            continue
        user_text = normalize_text(user.get("content", ""))
        answer_text = normalize_text(assistant.get("content", ""))
        reasoning = normalize_text(assistant.get("reasoning_content", "")) or None
        if user_text and answer_text:
            records.append((user_text, reasoning, answer_text))

    if not records:
        raise ValueError(f"No valid user/assistant records found in {dataset_path}")
    return records


def record_texts(records: list[tuple[str, str | None, str]]) -> list[str]:
    texts = []
    for user_text, reasoning, answer in records:
        parts = ["<USER>", user_text]
        if reasoning:
            parts.extend(["<THINK>", reasoning])
        parts.extend(["<ANSWER>", answer])
        texts.append(" ".join(parts))
    return texts


def create_dictionary_from_dataset(
    dataset_path: str | Path,
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
):
    """Create a deterministic word -> integer ID dictionary from the dataset."""
    records = load_records(dataset_path)
    texts = record_texts(records)
    words = " ".join(texts).split()

    word_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for word in sorted(set(words)):
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    dictionary_path = Path(dictionary_path)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.write_text(
        json.dumps(word_to_id, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return word_to_id


def load_dictionary(path: str | Path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    word_to_id = {str(k): int(v) for k, v in raw.items()}
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def encode_text(text: str, word_to_id: dict[str, int]) -> list[int]:
    ids = []
    for word in normalize_text(text).split():
        ids.append(word_to_id.get(word, word_to_id["<UNK>"]))
    return ids


def encode_record(record: tuple[str, str | None, str], word_to_id: dict[str, int]) -> list[int]:
    user_text, reasoning, answer = record
    ids = [word_to_id["<BOS>"], word_to_id["<USER>"]]
    ids += encode_text(user_text, word_to_id)
    if reasoning:
        ids.append(word_to_id["<THINK>"])
        ids += encode_text(reasoning, word_to_id)
    ids.append(word_to_id["<ANSWER>"])
    ids += encode_text(answer, word_to_id)
    ids.append(word_to_id["<EOS>"])
    return ids


def encode_prompt(text: str, word_to_id: dict[str, int]) -> list[int]:
    return [word_to_id["<BOS>"], word_to_id["<USER>"]] + encode_text(text, word_to_id) + [word_to_id["<THINK>"]]


def build_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
    repeats: int = 1,
    seed: int = 1234,
):
    records = load_records(dataset_path)
    word_to_id = create_dictionary_from_dataset(dataset_path, dictionary_path)
    rng = random.Random(seed)
    sequences = [encode_record(record, word_to_id) for record in records]
    data = []
    for _ in range(max(1, repeats)):
        data.extend(list(rng.choice(sequences)))
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
