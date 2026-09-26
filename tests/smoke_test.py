import torch

from lln.data import make_batch, dictionary_fingerprint
from lln.model import LLN


def main():
    torch.manual_seed(1234)

    vocab_size = 24
    model = LLN(
        vocab_size=vocab_size,
        dim=32,
        layers=2,
        heads=4,
        max_seq_len=16,
        recurrent_steps=2,
        output_clusters=4,
        memory_slots=2,
        type_count=6,
    )

    token_types = [0, 1, 1, 1, 1, 1] + [2] * (vocab_size - 6)
    model.set_token_types(token_types)

    ids = torch.tensor(
        [[1, 4, 8, 6, 10, 7], [1, 5, 9, 6, 11, 7]],
        dtype=torch.long,
    )

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
        assert torch.equal(full, cached), f"cached generation diverged: {full} != {cached}"

        logits, loss = model(ids, ids)
        assert logits.shape == (ids.size(0), ids.size(1), vocab_size)
        assert torch.isfinite(loss)

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

    # Specialist branches must participate in the first backward pass.
    model.train()
    _, loss = model(ids, ids)
    loss.backward()
    assert model.blocks[0].specialists.experts[0].down.weight.grad is not None

    print("LLN smoke tests: PASS")


if __name__ == "__main__":
    main()
