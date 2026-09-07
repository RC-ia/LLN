import json
import random
from pathlib import Path


def load_dictionary(path: str | Path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    word_to_id = {str(k): int(v) for k, v in raw.items()}
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


BASE_SENTENCES = [
    "eu gosto de gato",
    "eu gosto de cachorro",
    "eu quero comer comida",
    "eu quero beber agua",
    "ele gosta de gato",
    "ela gosta de peixe",
    "ele quer comer",
    "ela quer beber agua",
    "o gato corre rapido",
    "o cachorro corre rapido",
    "o gato corre devagar",
    "o cachorro bebe agua",
    "eu vejo o gato",
    "eu vejo o cachorro",
    "nos gostamos de casa",
    "eu estou na casa",
    "o carro e grande",
    "o gato e pequeno",
]


def encode_sentence(text: str, word_to_id: dict[str, int]) -> list[int]:
    ids = [word_to_id["<BOS>"]]
    for word in text.lower().split():
        if word not in word_to_id:
            raise KeyError(f"Word not in dictionary: {word!r}")
        ids.append(word_to_id[word])
    ids.append(word_to_id["<EOS>"])
    return ids


def build_dataset(word_to_id: dict[str, int], repeats: int = 2000, seed: int = 1234):
    rng = random.Random(seed)
    sequences = [encode_sentence(s, word_to_id) for s in BASE_SENTENCES]
    data = []
    for _ in range(repeats):
        seq = list(rng.choice(sequences))
        data.extend(seq)
    return data


def make_batch(data: list[int], batch_size: int, seq_len: int, device):
    # Inputs and targets are integer IDs. No text enters the model.
    import torch

    if len(data) <= seq_len + 1:
        raise ValueError("Dataset is too small for the requested sequence length")
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + seq_len], dtype=torch.long) for i in starts])
    y = torch.stack([torch.tensor(data[i + 1:i + seq_len + 1], dtype=torch.long) for i in starts])
    return x.to(device), y.to(device)


def decode_ids(ids: list[int], id_to_word: dict[int, str]) -> str:
    words = []
    for idx in ids:
        word = id_to_word.get(int(idx), f"<UNK:{idx}>")
        if word in {"<BOS>", "<PAD>"}:
            continue
        if word == "<EOS>":
            break
        words.append(word)
    return " ".join(words)
