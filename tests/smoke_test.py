import json
import tempfile
from pathlib import Path

import torch

from lln.data import (
    SPECIAL_TOKENS,
    build_dataset,
    decode_ids,
    dictionary_fingerprint,
    encode_record,
    encode_text,
    load_tokenizer,
    make_batch,
    normalize_text,
    token_id,
    tokenizer_fingerprint,
    train_tokenizer_from_dataset,
)
from lln.model import LLN, parameter_count


def main():
    torch.manual_seed(1234)

    vocab_size = 24
    model = LLN(
        vocab_size=vocab_size,
        dim=32,
        layers=2,
        heads=4,
        kv_heads=2,
        max_seq_len=16,
        dropout=0.0,
        rope_theta=10000.0,
    )

    ids = torch.tensor(
        [[1, 4, 8, 6, 10, 7], [1, 5, 9, 6, 11, 7]],
        dtype=torch.long,
    )

    # The architecture must use fewer KV heads than query heads for this GQA case.
    assert model.blocks[0].self_attn.heads == 4
    assert model.blocks[0].self_attn.kv_heads == 2
    assert parameter_count(model) > 0

    model.eval()
    with torch.no_grad():
        full = model._generate_full(
            ids.clone(),
            max_new_tokens=4,
            temperature=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            repetition_window=0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            hard_repeat_threshold=100,
            stop_ids={23},
        )
        cached = model._generate_cached(
            ids.clone(),
            max_new_tokens=4,
            temperature=1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            repetition_window=0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            hard_repeat_threshold=100,
            stop_ids={23},
        )
        assert torch.equal(full, cached), (
            f"cached generation diverged: {full.tolist()} != {cached.tolist()}"
        )

        _, caches = model._prefill_cache(ids)
        assert len(caches) == 2
        assert caches[0][0].shape == (2, 2, 16, 8)
        assert caches[0][1].shape == (2, 2, 16, 8)

        logits, loss = model(ids, ids)
        assert logits.shape == (ids.size(0), ids.size(1), vocab_size)
        assert loss is not None and torch.isfinite(loss)

        # Future tokens must not influence logits at earlier causal positions.
        changed = ids.clone()
        changed[:, 4:] = torch.flip(changed[:, 4:], dims=[1])
        changed_logits, _ = model(changed)
        assert torch.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-5, rtol=1e-5)

        weights = torch.ones_like(ids, dtype=torch.float32)
        weights[:, :2] = 0.0
        _, weighted_loss = model(ids, ids, loss_weights=weights)
        assert weighted_loss is not None and torch.isfinite(weighted_loss)

    # Dynamic batching should avoid padding every batch to the global seq_len.
    examples = [
        ([1, 4, 8, 2], [0, 0, 2, 2]),
        ([1, 4, 9, 10, 11, 2], [0, 0, 2, 2, 2, 2]),
    ]
    x, y, sections = make_batch(examples, 2, 16, torch.device("cpu"), indices=[0, 1])
    assert x.shape == (2, 5)
    assert y.shape == (2, 5)
    assert sections.shape == (2, 5)

    fp1 = dictionary_fingerprint({"a": 0, "b": 1})
    fp2 = dictionary_fingerprint({"a": 0, "b": 2})
    assert fp1 != fp2

    # BPE must preserve the special-token ID contract and encode unseen words
    # from byte/subword pieces instead of mapping the whole word to <UNK>.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dataset_path = tmp / "dataset.json"
        tokenizer_path = tmp / "tokenizer.json"
        dataset_path.write_text(
            json.dumps(
                [
                    {"messages": [
                        {"role": "user", "content": "Olá! Como você está?"},
                        {"role": "assistant", "content": "Estou bem; tokenização funciona."},
                    ]},
                    {"messages": [
                        {"role": "user", "content": "Explique modelos pequenos."},
                        {"role": "assistant", "content": "Modelos podem aprender padrões."},
                    ]},
                    {"messages": [
                        {"role": "user", "content": "2 + 2"},
                        {"role": "assistant", "content": "4."},
                    ]},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        trained = train_tokenizer_from_dataset(
            dataset_path, tokenizer_path, vocab_size=300, min_frequency=1
        )
        tokenizer, id_to_token = load_tokenizer(tokenizer_path)
        assert tokenizer.get_vocab_size() >= 266
        for expected_id, special in enumerate(SPECIAL_TOKENS):
            assert token_id(tokenizer, special) == expected_id
        assert tokenizer_fingerprint(trained) == tokenizer_fingerprint(tokenizer)

        unseen = "Xylophone-unseen tokenização-improvável 🧠"
        unseen_ids = encode_text(unseen, tokenizer)
        assert token_id(tokenizer, "<UNK>") not in unseen_ids
        assert decode_ids(unseen_ids, tokenizer) == normalize_text(unseen)
        assert all(token in id_to_token for token in unseen_ids)

        record = (normalize_text("Pergunta inédita"), None, normalize_text("Resposta inédita!"))
        record_ids, record_sections = encode_record(record, tokenizer)
        assert record_ids[0] == token_id(tokenizer, "<BOS>")
        assert record_ids[1] == token_id(tokenizer, "<USER>")
        assert record_ids[-1] == token_id(tokenizer, "<EOS>")
        assert len(record_ids) == len(record_sections)
        assert decode_ids(record_ids, tokenizer) == f"{record[0]} {record[2]}"

        # Fresh training should bootstrap a persisted tokenizer only when absent.
        bootstrap_path = tmp / "bootstrap-tokenizer.json"
        bootstrap_data = build_dataset(
            dataset_path, bootstrap_path, repeats=3, max_len=128
        )
        assert bootstrap_path.exists()
        assert len(bootstrap_data) == 3

    # Feed-forward weights must participate in backpropagation.
    model.train()
    _, loss = model(ids, ids)
    assert loss is not None
    loss.backward()
    assert model.blocks[0].mlp.gate_proj.weight.grad is not None
    assert torch.isfinite(model.blocks[0].mlp.gate_proj.weight.grad).all()

    # The output layer is tied: no independent LM-head weight is allocated.
    named_parameters = dict(model.named_parameters())
    assert "token_embedding.weight" in named_parameters
    assert not any("lm_head" in name for name in named_parameters)

    print("LLN v6 dense Transformer smoke tests: PASS")


if __name__ == "__main__":
    main()
