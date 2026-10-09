# LLN architecture experiment: v6 baseline vs v7 center test

This experiment adds a tagged architecture without changing the default v6 architecture.

## Architecture tags

- `v6-standard`: current dense Transformer preset: `dim=1024`, `layers=8`, `heads=8`, `kv_heads=4`.
- `v7-center-test`: `dim=768`, `layers=12`, `heads=12`, `kv_heads=4`. Layers 4–9 (zero-based indices 3–8) use a SwiGLU hidden width multiplied by 1.5; the other six layers use the standard width for dim=768.

The v7 tag is stored in the checkpoint config and uses architecture version 7. The default v6 tag retains architecture version 6 and remains compatible with existing v6 checkpoints when all other config fields match.

## Run both experiments

Use the same dataset, tokenizer, seed, batch size, sequence length, learning rate, loss weights, and number of steps. Separate output filenames prevent one experiment from overwriting the other. `--no-resume` guarantees each run starts from fresh random weights.

Example for a Kaggle GPU, matching the current 4k-context experiment and an LR of 1e-4:

```bash
python train.py --architecture v6-standard --dim 1024 --layers 8 --heads 8 --kv-heads 4 --seq-len 4096 --batch-size 2 --steps 2000 --lr 1e-4 --lr-schedule constant --think-weight 0.25 --answer-weight 1.0 --seed 1234 --save lln_v6_baseline.pt --no-resume
```

Then run the experimental preset:

```bash
python train.py --architecture v7-center-test --seq-len 4096 --batch-size 2 --steps 2000 --lr 1e-4 --lr-schedule constant --think-weight 0.25 --answer-weight 1.0 --seed 1234 --save lln_v7_center_test.pt --no-resume
```

Both commands use the existing default paths `data/dataset.json` and `data/tokenizer.json`. Add `--dataset` and `--tokenizer` if your files live elsewhere.

## Compare

Compare more than training loss:

1. Validation loss on the same held-out examples.
2. A fixed prompt set that is not included in training.
3. Coherence, language consistency, answer correctness, and repetition.
4. Actual training time and peak VRAM.

The two runs are a controlled initial screening, not proof that v7 is better. Do not resume one architecture from the other's checkpoint.
