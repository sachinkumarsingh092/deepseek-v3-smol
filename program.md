# DeepSeek-V3 Autoresearch Program

You are an AI research agent optimizing a toy DeepSeek-V3 model (MLA + MoE) trained on TinyShakespeare.

## Goal

Lower **val_bpb** (validation bits per byte). Lower is better.

## Rules

1. You may ONLY edit `train.py`. Do NOT touch `prepare.py`.
2. Each experiment runs for a fixed 5-minute wall-clock budget.
3. After each run, compare the final `val_bpb` to the previous best.
4. If improved, keep the change. If not, revert `train.py`.
5. Commit each successful improvement with a short message.

## Experiment Setup

Before your first experiment:

```bash
uv run prepare.py         # one-time data download
cp train.py train_best.py # save baseline
uv run train.py           # run baseline, note val_bpb
```

## Experiment Loop

For each experiment:

1. Read `train.py` and decide what to change.
2. Make ONE focused change (not multiple changes at once).
3. Run `uv run train.py` and wait for the `=== FINAL ===` line.
4. Compare the new `val_bpb` to the previous best.
5. If improved: `cp train.py train_best.py` and log the result.
6. If not improved: `cp train_best.py train.py` (revert).
7. Pick the next experiment and repeat.

## What You Can Change

Everything in `train.py` is fair game:

### Hyperparameters (ModelConfig)
- `d_model`, `n_layers`, `n_dense_layers`
- `n_heads`, `d_head`
- `d_kv_latent`, `d_q_latent`, `d_head_rope`
- `d_ff_dense`, `d_ff_expert`
- `n_routed_experts`, `n_shared_experts`, `top_k`
- `max_seq_len`, `batch_size`
- `lr`, `weight_decay`
- `gamma`, `balance_alpha`

### Architecture
- Number of dense vs MoE layers
- Expert FFN structure
- Attention mechanism details
- Normalization strategy
- Activation functions

### Optimizer
- AdamW betas, epsilon
- Learning rate schedule (warmup, cosine decay, etc.)
- Gradient clipping value
- Try other optimizers

### Training
- Batch size
- Gradient accumulation
- Any regularization (dropout, etc.)

## Tips

- Start with quick wins: learning rate, batch size, layer count.
- The model is ~47M params. You can make it larger or smaller.
- MoE balance is handled by the bias trick -- tuning `gamma` matters.
- `max_seq_len=128` is small. Try increasing it if memory allows.
- Watch for overfitting: if train loss drops but val doesn't, add regularization.

## Logging

After each experiment, report:
```
Experiment N: <one-line description>
  val_bpb: X.XXX (previous: Y.YYY, delta: Z.ZZZ)
  Status: KEPT / REVERTED
```
