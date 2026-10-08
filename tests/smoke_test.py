import torch

from lln.data import make_batch, dictionary_fingerprint
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
