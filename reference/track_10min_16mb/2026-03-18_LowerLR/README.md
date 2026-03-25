# Lower LR

**Date**: 2026-03-18
**val_bpb**: 1.2230 | **Artifact Size**: 15,854,246 bytes | **Hardware**: 8xH200 SXM, 600s

## Summary

A pure hyperparameter tuning attempt that halves the Muon/Adam learning rates from the baseline defaults. A systematic sweep over 6 LR settings showed that the default `MATRIX_LR=0.04` was too high; the optimum is around 0.02. This yields a modest -0.0014 BPB improvement with zero code changes.

## Key Changes from Baseline

### Architecture Changes
- None. Identical 9-layer, 512-dim, 8-head (4 KV) architecture.

### Training Changes
- `MATRIX_LR=0.02` (default 0.04): Halved Muon learning rate for matrix parameters.
- `SCALAR_LR=0.02` (default 0.04): Halved Adam learning rate for scalar/vector parameters.
- `TIED_EMBED_LR=0.03` (default 0.05): Reduced tied embedding learning rate by 40%.
- All changes applied via environment variables at runtime; no code defaults were modified.

### Quantization/Compression Changes
- None. Standard int8+zlib quantization.

### Evaluation Changes
- None. Standard continuous-stream BPB evaluation.

## Detailed Code Diff Analysis

The diff contains only a trivial comment rewording in the module docstring -- no functional code changes at all:

```
-Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
+Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
```

All improvements come from runtime environment variable overrides of the learning rate hyperparameters.

## Results

LR sweep results (all 8xH200, 600s):

| MATRIX_LR | val_bpb (post-quant) | Delta vs baseline |
|-----------|----------------------|-------------------|
| 0.06 | 1.2445 | +0.0159 (much worse) |
| 0.04 (default) | 1.2286 | -- |
| 0.03 | 1.2279 | -0.0007 |
| 0.025 | 1.2250 | -0.0036 |
| **0.02** | **1.2230** | **-0.0056** |
| 0.015 | 1.2234 | -0.0052 |

Final submission run:

| Seed | Steps | Pre-Quant val_bpb | Post-Quant val_bpb | Artifact Size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 14,421 | 1.2183 | 1.2230 | 15,854,246 B |

- Step average: 41.60ms (H200 is ~5% faster than H100)
- Train time: 599,847ms
- Total tokens: 7,560,609,792
- Quant degradation: ~0.005 BPB
- Peak memory: 10,246 MiB allocated

Note: This was run on 8xH200 (141GB HBM3e), not H100. The ~5% faster step time yields ~400 extra steps compared to H100, which may account for a small portion of the improvement. The improvement over baseline is modest enough (-0.0014 BPB) that it does not meet the 0.005 nat threshold for a new record.

## Based On

Direct modification of baseline ([NaiveBaseline](../2026-03-17_NaiveBaseline/)). No code changes; purely a hyperparameter sweep.
