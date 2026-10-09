import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<PAD>", "<BOS>", "<EOS>", "<UNK>",
    "<USER>", "</USER>", "<THINK>", "</THINK>",
    "<ANSWER>", "</ANSWER>",
]
DEFAULT_DATASET = Path("data/dataset.json")
DEFAULT_TOKENIZER = Path("data/tokenizer.json")
# Backward-compatible constant name; the artifact is now a tokenizer JSON, not a word dictionary.
DEFAULT_DICTIONARY = DEFAULT_TOKENIZER

SECTION_PROMPT = 0
SECTION_THINK = 1
SECTION_ANSWER = 2

TYPE_SPECIAL = 0
TYPE_WORD = 1
TYPE_NUMBER = 2
TYPE_PUNCT = 3
TYPE_OPERATOR = 4
TYPE_CODE = 5
TYPE_NAMES = {
    TYPE_SPECIAL: "special",
    TYPE_WORD: "word",
    TYPE_NUMBER: "number",
    TYPE_PUNCT: "punct",
    TYPE_OPERATOR: "operator",
    TYPE_CODE: "code",
}


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def classify_token(token: str) -> int:
    """Assign a compact semantic/syntactic type without changing the fixed token ID."""
    if token in SPECIAL_TOKENS or (token.startswith("<") and token.endswith(">")):
        return TYPE_SPECIAL
    if re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", token):
        return TYPE_NUMBER
    if re.fullmatch(r"[+\-*/%=<>^~|&]+", token):
        return TYPE_OPERATOR
    if re.fullmatch(r"[^\w\s]+", token):
        return TYPE_PUNCT
    if any(ch in token for ch in "{}[]();:=\\/\"'`#$") or token.endswith((".py", ".js", ".ts", ".json")):
        return TYPE_CODE
    return TYPE_WORD


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    """Fingerprint vocabulary, merges, pre-tokenization, decoder and special tokens."""
    state = json.loads(tokenizer.to_str())
    payload = json.dumps(
        state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dictionary_fingerprint(value) -> str:
    """Compatibility helper; fingerprints tokenizer JSON or a legacy ID mapping."""
    if hasattr(value, "to_str"):
        return tokenizer_fingerprint(value)
    payload = json.dumps(
        sorted((str(token), int(idx)) for token, idx in value.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer is missing required special token {token!r}")
    return int(value)


def load_tokenizer(path: str | Path = DEFAULT_TOKENIZER) -> tuple[Tokenizer, dict[int, str]]:
    """Load a saved ByteLevel BPE tokenizer and validate stable special-token IDs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {path}. Train it first with "
            f"'python create_ids.py data/dataset.json --output {path}'."
        )
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:
        raise ValueError(
            f"{path} is not a valid LLN tokenizer JSON. It may be a legacy word dictionary; "
            "train a BPE tokenizer with create_ids.py."
        ) from exc

    for expected_id, special in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(special)
        if actual_id != expected_id:
            raise ValueError(
                f"Incompatible tokenizer: {special} must have ID {expected_id}, got {actual_id}. "
                "Retrain it using create_ids.py and use the same tokenizer for training and inference."
            )
    id_to_token = {int(idx): str(token) for token, idx in tokenizer.get_vocab().items()}
    return tokenizer, id_to_token


def train_tokenizer_from_dataset(
    dataset_path: str | Path,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train deterministic ByteLevel BPE so unseen words can be composed from subwords/bytes."""
    min_vocab_size = len(SPECIAL_TOKENS) + len(pre_tokenizers.ByteLevel.alphabet())
    if vocab_size < min_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {min_vocab_size} to reserve special tokens and byte alphabet"
        )
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")

    records = load_records(dataset_path)
    corpus = record_texts(records)
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    for expected_id, special in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(special) != expected_id:
            raise RuntimeError(f"BPE trainer assigned an unexpected ID to special token {special}")

    path = Path(tokenizer_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path), pretty=True)
    return tokenizer


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


def encode_text(text: str, tokenizer: Tokenizer) -> list[int]:
    return tokenizer.encode(normalize_text(text), add_special_tokens=False).ids


def encode_record(
    record: tuple[str, str | None, str], tokenizer: Tokenizer
) -> tuple[list[int], list[int]]:
    user_text, reasoning, answer = record
    ids = [token_id(tokenizer, "<BOS>"), token_id(tokenizer, "<USER>")]
    sections = [SECTION_PROMPT, SECTION_PROMPT]

    user_ids = encode_text(user_text, tokenizer)
    ids += user_ids
    sections += [SECTION_PROMPT] * len(user_ids)
    ids.append(token_id(tokenizer, "</USER>"))
    sections.append(SECTION_PROMPT)

    if reasoning:
        ids.append(token_id(tokenizer, "<THINK>"))
        sections.append(SECTION_THINK)
        think_ids = encode_text(reasoning, tokenizer)
        ids += think_ids
        sections += [SECTION_THINK] * len(think_ids)
        ids.append(token_id(tokenizer, "</THINK>"))
        sections.append(SECTION_THINK)

    ids.append(token_id(tokenizer, "<ANSWER>"))
    sections.append(SECTION_ANSWER)
    answer_ids = encode_text(answer, tokenizer)
    ids += answer_ids
    sections += [SECTION_ANSWER] * len(answer_ids)
    ids.append(token_id(tokenizer, "</ANSWER>"))
    sections.append(SECTION_ANSWER)
    ids.append(token_id(tokenizer, "<EOS>"))
    sections.append(SECTION_ANSWER)
    return ids, sections


def encode_prompt(text: str, tokenizer: Tokenizer) -> list[int]:
    return [
        token_id(tokenizer, "<BOS>"),
        token_id(tokenizer, "<USER>"),
        *encode_text(text, tokenizer),
        token_id(tokenizer, "</USER>"),
        token_id(tokenizer, "<THINK>"),
    ]


def _compact_record(ids: list[int], sections: list[int], max_len: int) -> tuple[list[int], list[int]]:
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
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    repeats: int = 1,
    seed: int = 1234,
    return_sections: bool = False,
    max_len: int | None = None,
    vocab_size: int = 8000,
    min_frequency: int = 2,
):
    records = load_records(dataset_path)
    tokenizer_path = Path(tokenizer_path)
    # Bootstrap once for a fresh checkout. Existing tokenizers are never silently retrained.
    if not tokenizer_path.exists():
        print(f"Tokenizer not found; training ByteLevel BPE at {tokenizer_path}")
        train_tokenizer_from_dataset(
            dataset_path, tokenizer_path, vocab_size=vocab_size, min_frequency=min_frequency
        )
    tokenizer, _ = load_tokenizer(tokenizer_path)
    sequences = [encode_record(record, tokenizer) for record in records]

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
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if sections is not None:
        raise ValueError("sections argument is no longer used; build_dataset returns complete records")

    if indices is None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        chosen = [random.choice(data) for _ in range(batch_size)]
    else:
        chosen = [data[i] for i in indices]
        if not chosen:
            raise ValueError("indices must contain at least one example")

    max_tokens = 0
    for ids, sec in chosen:
        if len(ids) > seq_len + 1:
            raise ValueError(
                "Training record exceeds seq_len; build_dataset must be called with max_len=seq_len"
            )
        if len(ids) < 2:
            raise ValueError("Training record must contain at least two tokens")
        max_tokens = max(max_tokens, len(ids) - 1)

    target_len = min(seq_len, max_tokens)
    x_rows, y_rows, w_rows = [], [], []
    pad_id = 0

    for ids, sec in chosen:
        x_ids = ids[:-1]
        y_ids = ids[1:]
        y_sec = sec[1:]
        pad_count = target_len - len(x_ids)
        if pad_count < 0:
            raise ValueError("record is longer than the dynamic batch length")
        if pad_count > 0:
            x_ids += [pad_id] * pad_count
            y_ids += [pad_id] * pad_count
            y_sec += [-1] * pad_count

        x_rows.append(x_ids)
        y_rows.append(y_ids)
        w_rows.append(y_sec)

    x = torch.tensor(x_rows, dtype=torch.long, device=device)
    y = torch.tensor(y_rows, dtype=torch.long, device=device)
    batch_sections = torch.tensor(w_rows, dtype=torch.float32, device=device)
    return x, y, batch_sections

def decode_ids(ids: list[int], tokenizer: Tokenizer) -> str:
    """Decode BPE IDs, treating control tokens as boundaries between text segments."""
    chunks: list[str] = []
    segment: list[int] = []

    def flush_segment() -> None:
        if segment:
            chunks.append(tokenizer.decode(segment, skip_special_tokens=True))
            segment.clear()

    try:
        for raw_id in ids:
            idx = int(raw_id)
            token = tokenizer.id_to_token(idx)
            if token in SPECIAL_TOKENS:
                flush_segment()
                if token == "<EOS>":
                    break
                if token not in {"<PAD>", "<BOS>"}:
                    # Control markers separate fields which were encoded individually.
                    chunks.append(" ")
            else:
                segment.append(idx)
        flush_segment()
        return re.sub(r"\s+", " ", "".join(chunks)).strip()
    except Exception as exc:
        raise ValueError("Could not decode IDs with the supplied LLN tokenizer") from exc
