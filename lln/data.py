import json
import random
import re
from pathlib import Path

import torch

SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>",
    "<USER>", "</USER>", "<THINK>", "</THINK>",
    "<ANSWER>", "</ANSWER>",
]
DEFAULT_DATASET = Path("data/dataset.json")
DEFAULT_DICTIONARY = Path("data/dictionary.json")

SECTION_PROMPT = 0
SECTION_THINK = 1
SECTION_ANSWER = 2


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_records(dataset_path: str | Path) -> list[tuple[str, str | None, str]]:
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
        parts = ["<USER>", user_text, "</USER>"]
        if reasoning:
            parts.extend(["<THINK>", reasoning, "</THINK>"])
        parts.extend(["<ANSWER>", answer, "</ANSWER>"])
        texts.append(" ".join(parts))
    return texts


def create_dictionary_from_dataset(dataset_path: str | Path, dictionary_path: str | Path = DEFAULT_DICTIONARY):
    records = load_records(dataset_path)
    words = " ".join(record_texts(records)).split()
    word_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for word in sorted(set(words)):
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    dictionary_path = Path(dictionary_path)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.write_text(json.dumps(word_to_id, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return word_to_id


def load_dictionary(path: str | Path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    word_to_id = {str(k): int(v) for k, v in raw.items()}
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


def encode_text(text: str, word_to_id: dict[str, int]) -> list[int]:
    return [word_to_id.get(word, word_to_id["<UNK>"]) for word in normalize_text(text).split()]


def encode_record(record: tuple[str, str | None, str], word_to_id: dict[str, int]) -> tuple[list[int], list[int]]:
    user_text, reasoning, answer = record
    ids = [word_to_id["<BOS>"], word_to_id["<USER>"]]
    sections = [SECTION_PROMPT, SECTION_PROMPT]

    user_ids = encode_text(user_text, word_to_id)
    ids += user_ids
    sections += [SECTION_PROMPT] * len(user_ids)
    ids.append(word_to_id["</USER>"])
    sections.append(SECTION_PROMPT)

    if reasoning:
        ids.append(word_to_id["<THINK>"])
        sections.append(SECTION_THINK)
        think_ids = encode_text(reasoning, word_to_id)
        ids += think_ids
        sections += [SECTION_THINK] * len(think_ids)
        ids.append(word_to_id["</THINK>"])
        sections.append(SECTION_THINK)

    ids.append(word_to_id["<ANSWER>"])
    sections.append(SECTION_ANSWER)
    answer_ids = encode_text(answer, word_to_id)
    ids += answer_ids
    sections += [SECTION_ANSWER] * len(answer_ids)
    ids.append(word_to_id["</ANSWER>"])
    sections.append(SECTION_ANSWER)
    ids.append(word_to_id["<EOS>"])
    sections.append(SECTION_ANSWER)
    return ids, sections


def encode_prompt(text: str, word_to_id: dict[str, int]) -> list[int]:
    return [
        word_to_id["<BOS>"],
        word_to_id["<USER>"],
        *encode_text(text, word_to_id),
        word_to_id["</USER>"],
        word_to_id["<THINK>"],
    ]


def _compact_record(
    ids: list[int],
    sections: list[int],
    max_len: int,
) -> tuple[list[int], list[int]]:
    """Keep prompt and final answer together when a reasoning trace is too long."""
    if max_len < 8:
        raise ValueError("max_len must be at least 8")
    if len(ids) <= max_len:
        return ids, sections

    answer_start = next(i for i, s in enumerate(sections) if s == SECTION_ANSWER)
    prompt_ids = ids[:answer_start]
    prompt_sections = sections[:answer_start]
    answer_ids = ids[answer_start:]
    answer_sections = sections[answer_start:]

    remaining = max_len - len(prompt_ids) - len(answer_ids)
    if remaining < 0:
        prompt_ids = prompt_ids[max(0, -remaining):]
        prompt_sections = prompt_sections[-len(prompt_ids):] if prompt_ids else []
        remaining = max_len - len(prompt_ids) - len(answer_ids)
        if remaining < 0:
            answer_ids = answer_ids[:max_len - len(prompt_ids)]
            answer_sections = answer_sections[:len(answer_ids)]
            return prompt_ids + answer_ids, prompt_sections + answer_sections

    think_open = next((i for i, s in enumerate(sections) if s == SECTION_THINK), None)
    if think_open is not None and remaining > 0:
        think_content_start = think_open
        think_content_end = answer_start
        keep_think = ids[think_content_start:think_content_end][:remaining]
        keep_sec = sections[think_content_start:think_content_start + len(keep_think)]
        return prompt_ids + keep_think + answer_ids, prompt_sections + keep_sec + answer_sections

    return prompt_ids + answer_ids, prompt_sections + answer_sections


def build_dataset(
    dataset_path: str | Path = DEFAULT_DATASET,
    dictionary_path: str | Path = DEFAULT_DICTIONARY,
    repeats: int = 1,
    seed: int = 1234,
    return_sections: bool = False,
    max_len: int | None = None,
):
    records = load_records(dataset_path)
    word_to_id = create_dictionary_from_dataset(dataset_path, dictionary_path)
    sequences = [encode_record(record, word_to_id) for record in records]

    if max_len is not None:
        sequences = [_compact_record(ids, sec, max_len + 1) for ids, sec in sequences]

    if repeats <= 0:
        repeats = len(sequences)
    repeated = []
    for _ in range(max(1, repeats) // max(1, len(sequences))):
        repeated.extend(sequences)
    remainder = max(1, repeats) % max(1, len(sequences))
    if remainder:
        order = list(range(len(sequences)))
        random.Random(seed + 1).shuffle(order)
        repeated.extend(sequences[i] for i in order[:remainder])
    return repeated


def make_batch(data, batch_size: int, seq_len: int, device, sections=None, indices=None):
    if not data:
        raise ValueError("Dataset contains no training examples")
    if sections is not None:
        raise ValueError("sections argument is no longer used; build_dataset returns complete records")

    if indices is None:
        chosen = [random.choice(data) for _ in range(batch_size)]
    else:
        if len(indices) != batch_size:
            raise ValueError("indices length must equal batch_size")
        chosen = [data[i] for i in indices]

    x_rows, y_rows, w_rows = [], [], []
    pad_id = 0

    for ids, sec in chosen:
        if len(ids) > seq_len + 1:
            raise ValueError("Training record exceeds seq_len; build_dataset must be called with max_len=seq_len")
        if len(ids) < 2:
            continue

        x_ids = ids[:-1]
        y_ids = ids[1:]
        y_sec = sec[1:]
        pad_count = seq_len - len(x_ids)
        if pad_count > 0:
            x_ids += [pad_id] * pad_count
            y_ids += [pad_id] * pad_count
            y_sec += [-1] * pad_count

        x_rows.append(x_ids)
        y_rows.append(y_ids)
        w_rows.append(y_sec)

    if not x_rows:
        raise ValueError("No valid training examples")

    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    batch_sections = torch.tensor(w_rows, dtype=torch.float32, device=device)
    return x, y, batch_sections


def decode_ids(ids: list[int], id_to_word: dict[int, str]) -> str:
    words = []
    for idx in ids:
        word = id_to_word.get(int(idx), "<UNK>")
        if word in {"<BOS>", "<PAD>"}:
            continue
        words.append(word)
        if word == "<EOS>":
            break
    return " ".join(words)
