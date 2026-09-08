import argparse
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from lln.data import SPECIAL_TOKENS, classify_token, load_records, normalize_text


# Only tokens that are entirely numeric are treated as members of the numeric
# structural family. Punctuation-attached forms such as "2.", "3:", "1,"
# remain normal lexical tokens and must not contaminate numeric ordering.
PURE_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def parse_pure_number(token):
    if not PURE_NUMBER_RE.fullmatch(token):
        return None
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


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


def build_context_counts(token_sequences, vocab, blocked_ids=None):
    # Sparse co-occurrence graph. Window 4 keeps this inexpensive enough for a small dataset.
    # Numeric tokens are excluded from this graph so semantic grouping cannot
    # destroy the dedicated numeric structural family.
    blocked_ids = blocked_ids or set()
    neighbors = defaultdict(Counter)
    frequency = Counter()
    window = 4
    for seq in token_sequences:
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in seq]
        for i, a in enumerate(ids):
            frequency[a] += 1
            if a in blocked_ids:
                continue
            lo = max(0, i - window)
            hi = min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                b = ids[j]
                if b in blocked_ids:
                    continue
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


def build_groups(neighbors, vocab_size, min_similarity, top_neighbors, excluded_ids=None):
    excluded_ids = excluded_ids or set()
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
        if token_id in excluded_ids:
            continue
        candidates = [n for n, _ in row.most_common(top_neighbors) if n not in excluded_ids]
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
        if token_id not in excluded_ids:
            groups[find(token_id)].append(token_id)

    return list(groups.values())


def assign_positions(groups, id_to_token, token_frequency, numeric_ids):
    group_id = [-1] * len(id_to_token)
    position = [0] * len(id_to_token)

    next_group = 0

    # One dedicated structural numeric family. Position is the rank among the
    # numeric values observed in the dictionary, so sparse values still receive
    # compact positions: 1 -> 0, 2 -> 1, 5 -> 2, ...
    numeric_values = []
    for idx in numeric_ids:
        value = parse_pure_number(id_to_token[idx])
        if value is not None:
            numeric_values.append((value, idx))
    numeric_values.sort(key=lambda item: (item[0], id_to_token[item[1]]))

    if numeric_values:
        for pos, (_, idx) in enumerate(numeric_values):
            group_id[idx] = next_group
            position[idx] = pos
        next_group += 1

    for members in sorted(groups, key=lambda g: (-len(g), min(g))):
        # Non-numeric families use deterministic frequency/lexical ordering.
        # This is a structural hypothesis, not a claim of semantic ordering.
        ordered = sorted(
            members,
            key=lambda idx: (-token_frequency.get(idx, 0), id_to_token[idx]),
        )

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

    numeric_ids = {
        idx for idx, token in enumerate(id_to_token)
        if parse_pure_number(token) is not None
    }

    sequences = tokenized_records(args.dataset)
    neighbors, frequency = build_context_counts(
        sequences,
        vocab,
        blocked_ids=numeric_ids,
    )
    groups = build_groups(
        neighbors,
        len(vocab),
        min_similarity=args.min_similarity,
        top_neighbors=args.top_neighbors,
        excluded_ids=numeric_ids,
    )
    group_ids, positions, group_count = assign_positions(
        groups,
        id_to_token,
        frequency,
        numeric_ids,
    )

    output = Path(args.output) if args.output else dictionary_path.with_suffix(".structure.json")
    numeric_values = [
        idx for idx in numeric_ids if parse_pure_number(id_to_token[idx]) is not None
    ]
    metadata = {
        "version": 2,
        "method": "context_components_plus_dedicated_numeric_family",
        "min_similarity": args.min_similarity,
        "top_neighbors": args.top_neighbors,
        "group_count": group_count,
        "numeric_family": {
            "enabled": bool(numeric_values),
            "classification": "pure_numeric_only",
            "ordering": "numeric_ascending_rank",
            "hybrid_tokens_are_excluded": True,
            "member_count": len(numeric_values),
        },
        "position_rules": {
            "numeric": "numeric_ascending_rank",
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
    print("method=context_components_plus_dedicated_numeric_family")
    print(f"numeric_tokens={len(numeric_ids):,}")
    print("numeric_classification=pure_numeric_only")
    print("numeric_position=numeric_ascending_rank")
    print("hybrid_tokens_excluded=true")
    print("other_position=frequency_desc_then_lexical")

    # Show a few useful numeric entries. The family is global by design.
    if numeric_ids:
        examples = sorted(
            numeric_ids,
            key=lambda idx: (parse_pure_number(id_to_token[idx]), id_to_token[idx]),
        )[:20]
        preview = ", ".join(
            f"{id_to_token[i]}:{positions[i]}" for i in examples
        )
        numeric_group_id = group_ids[examples[0]]
        print(f"numeric_group={numeric_group_id} size={len(numeric_ids)} {preview}")

    for token in ("1", "2", "3", "5.0", "10.0", "50.0", "2.", "3:", "1,"):
        idx = vocab.get(token)
        if idx is None:
            continue
        print(
            f"token={token!r} id={idx} group={group_ids[idx]} "
            f"position={positions[idx]} numeric={parse_pure_number(token) is not None}"
        )


if __name__ == "__main__":
    main()
