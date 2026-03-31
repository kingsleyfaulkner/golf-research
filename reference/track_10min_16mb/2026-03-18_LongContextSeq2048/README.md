# Long Context Seq2048 v2

**Date**: 2026-03-19
**val_bpb**: 1.2058 | **Artifact Size**: 15,867,270 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

Doubles the training sequence length from 1024 to 2048 tokens with tuned learning rates, achieving a large BPB improvement (-0.019) over the baseline. Longer sequences allow the model to learn longer-range dependencies and see more coherent document context per training example, at the cost of ~19% slower step time (51.89ms vs 43.54ms) resulting in fewer total training steps.

## Key Changes from Baseline

### Architecture Changes
- None. Same 9-layer, 512-dim, 8-head (4 KV) architecture with 2x MLP expansion.

### Training Changes
- `TRAIN_SEQ_LEN=2048` (default 1024): Doubled sequence length. With `TRAIN_BATCH_TOKENS=524288` unchanged, this halves the number of sequences per batch (256 vs 512).
- `TIED_EMBED_LR=0.04` (default 0.05): Slightly reduced tied embedding learning rate.
- `MATRIX_LR=0.032` (default 0.04): Reduced Muon learning rate for matrix parameters (20% lower).
- `SCALAR_LR=0.032` (default 0.04): Reduced Adam learning rate for scalar/vector parameters (20% lower).
- All LR changes are baked into the code defaults (not just env var overrides).
- Default `run_id` changed to `"long_context_seq2048_v2"`.
- Total steps reduced from ~13,780 to ~11,564 due to slower step time at seq_len=2048.

### Quantization/Compression Changes
- None. Standard int8+zlib quantization.

### Evaluation Changes
- None. Standard continuous-stream BPB evaluation.

## Detailed Code Diff Analysis

The diff is minimal -- only hyperparameter changes, no architectural or algorithmic modifications:

**Sequence length** (`train_seq_len`): Changed from 1024 to 2048 in the `Hyperparameters` class default. With the same total batch tokens (524,288), this means 256 sequences per step instead of 512. Each sequence sees twice as much contiguous text, which is particularly valuable for a model with 1024-token RoPE context -- at seq_len=2048, the model trains on positions beyond its original design, potentially learning to extrapolate.

**Learning rate tuning**: All three learning rates are reduced:
- `tied_embed_lr`: 0.05 -> 0.04 (-20%)
- `matrix_lr`: 0.04 -> 0.032 (-20%)
- `scalar_lr`: 0.04 -> 0.032 (-20%)

The LR reduction likely compensates for the longer sequences producing higher-variance gradients (each gradient step covers fewer independent sequences) and prevents instability at the longer context.

**Comment updates**: The config header is updated to reflect "Long Context Seq2048 v2" with the tuned LRs.

## Results

| Seed | Steps | Pre-Quant val_bpb | Post-Quant val_bpb | Artifact Size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 11,564 | 1.2005 | 1.2058 | 15,867,270 B |
| 1338 | - | - | 1.2062 | - |
| 1339 | - | - | 1.2072 | - |

- Step average: 51.89ms
- Train time: 600,038ms
- Quant degradation: ~0.005 BPB
- Peak memory: 10,247 MiB allocated

Multi-seed statistics:
- Mean val_bpb: 1.2064
- Std: 0.00072
- One-sided t-test vs baseline (1.2244): t=31.42, p=0.00051

Improvement over baseline: -0.019 BPB. This was the first record-breaking submission, beating the baseline by well over the 0.005 nat threshold required for a new record.

## Based On

Direct modification of baseline ([NaiveBaseline](../2026-03-17_NaiveBaseline/)). The "v2" in the name suggests an earlier seq2048 attempt with different LR settings that performed worse.
