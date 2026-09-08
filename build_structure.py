import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from lln.data import SPECIAL_TOKENS, classify_token, load_records, normalize_text


def tokenized_records(dataset_path):
    records = load_records(dataset_path)
    texts = []
    for user, reasoning, answer in records:
        parts = [user]
        if reasoning:
            parts.append(reasoning)
        parts.append(answer)
        texts.append(normalize_text(" ".join(parts)).split())
    return texts


def build_context_counts(token_sequences, vocab):
    # Sparse co-occurrence graph. Window 4 keeps this inexpensive enough for a small dataset.
    neighbors = defaultdict(Counter)
    frequency = Counter()
    window = 4
    for seq in token_sequences:
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in seq]
        for i, a in enumerate(ids):
            frequency[a] += 1
            lo = max(0, i - window)
            hi = min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                b = ids[j]
                neighbors[a][b] += 1
    return neighbors, frequency


def cosine_sparse(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def build_groups(neighbors, frequency, vocab_size, min_similarity, top_neighbors):
    parent = list(range(vocab_size))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ranked = []
    for token_id, row in neighbors.items():
        candidates = [n for n, _ in row.most_common(top_neighbors)]
        candidates = sorted(candidates, key=lambda n: cosine_sparse(row, neighbors.get(n, {})), reverse=True)
        for n in candidates[:top_neighbors]:
            sim = cosine_sparse(row, neighbors.get(n, {}))
            if sim >= min_similarity:
                ranked.append((sim, token_id, n))

    # Connect only the strongest local similarities, preventing one common token
    # from merging the entire vocabulary into a single component.
    ranked.sort(reverse=True)
    for _, a, b in ranked:
        union(a, b)

    groups = defaultdict(list)
    for token_id in range(vocab_size):
        groups[find(token_id)].append(token_id)

    return list(groups.values())


def assign_positions(groups, id_to_token, token_frequency):
    group_id = [-1] * len(id_to_token)
    position = [0] * len(id_to_token)

    next_group = 0
    for members in sorted(groups, key=lambda g: (-len(g), min(g))):
        # Numeric families get true numeric ordering. Everything else uses a
        # deterministic frequency/lexical ordering, which is only a hypothesis
        # and is intentionally recorded as such in the metadata.
        tokens = [id_to_token[i] for i in members]
        numeric = []
        for idx, token in zip(members, tokens):
            raw = token.replace(",", ".")
            try:
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
                    numeric.append((float(raw), idx))
            except ValueError:
                pass

        if len(numeric) == len(members) and numeric:
            ordered = [idx for _, idx in sorted(numeric)]
            position_rule = "numeric_ascending"
        else:
            ordered = sorted(
                members,
                key=lambda idx: (-token_frequency.get(idx, 0), id_to_token[idx]),
            )
            position_rule = "frequency_desc_then_lexical"

        for pos, idx in enumerate(ordered):
            group_id[idx] = next_group
            position[idx] = pos
        next_group += 1

    return group_id, position, next_group


def main():
    parser = argparse.ArgumentParser(description="Build experimental token family/group/position metadata")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--dictionary", default="data/dictionary.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-similarity", type=float, default=0.30)
    parser.add_argument("--top-neighbors", type=int, default=3)
    args = parser.parse_args()

    dictionary_path = Path(args.dictionary)
    vocab = json.loads(dictionary_path.read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in vocab.items()}
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token

    sequences = tokenized_records(args.dataset)
    neighbors, frequency = build_context_counts(sequences, vocab)
    groups = build_groups(
        neighbors,
        frequency,
        len(vocab),
        min_similarity=args.min_similarity,
        top_neighbors=args.top_neighbors,
    )
    group_ids, positions, group_count = assign_positions(groups, id_to_token, frequency)

    output = Path(args.output) if args.output else dictionary_path.with_suffix(".structure.json")
    metadata = {
        "version": 1,
        "method": "sparse_context_similarity_components",
        "min_similarity": args.min_similarity,
        "top_neighbors": args.top_neighbors,
        "group_count": group_count,
        "position_rules": {
            "numeric": "numeric_ascending",
            "other": "frequency_desc_then_lexical",
        },
        "token_groups": group_ids,
        "token_positions": positions,
        "token_frequencies": [frequency.get(i, 0) for i in range(len(vocab))],
        "type_ids": [classify_token(t) for t in id_to_token],
    }
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"dictionary={dictionary_path}")
    print(f"dataset={args.dataset}")
    print(f"vocab={len(vocab):,}")
    print(f"groups={group_count:,}")
    print(f"output={output}")
    print("method=sparse_context_similarity_components")
    print("numeric_position=numeric_ascending")
    print("other_position=frequency_desc_then_lexical")

    # Show a few useful numeric families when they exist.
    number_type = {idx for idx, tok in enumerate(id_to_token) if classify_token(tok) == 2}
    numeric_groups = defaultdict(list)
    for idx in number_type:
        numeric_groups[group_ids[idx]].append(idx)
    examples = 0
    for gid, members in sorted(numeric_groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) >= 2:
            members = sorted(members, key=lambda idx: positions[idx])
            preview = ", ".join(f"{id_to_token[i]}:{positions[i]}" for i in members[:12])
            print(f"numeric_group={gid} size={len(members)} {preview}")
            examples += 1
            if examples >= 5:
                break


if __name__ == "__main__":
    main()
