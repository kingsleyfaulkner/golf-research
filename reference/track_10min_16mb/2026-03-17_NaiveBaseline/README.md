# Naive Baseline

**Date**: 2026-03-18
**val_bpb**: 1.2244 | **Artifact Size**: 15,863,489 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

This is the unmodified baseline `train_gpt.py` run as a reference point. It establishes the starting BPB for all subsequent attempts using the standard 9-layer, 512-dim GPT architecture with int8+zlib quantization. The only code difference from the canonical baseline is a trivial comment rewording.

## Key Changes from Baseline

### Architecture Changes
- None. Standard 9-layer, 512-dim, 8-head (4 KV) architecture with 2x MLP expansion.

### Training Changes
- None. Default hyperparameters: `MATRIX_LR=0.04`, `SCALAR_LR=0.04`, `TIED_EMBED_LR=0.05`, `WARMDOWN_ITERS=1200`, `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=524288`.

### Quantization/Compression Changes
- None. Standard int8 per-row quantization with zlib compression.

### Evaluation Changes
- None. Standard continuous-stream BPB evaluation on full fineweb_val split.

## Detailed Code Diff Analysis

The only diff is a trivial rewording in the module docstring:
```
-Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
+Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
```

No functional changes whatsoever.

## Results

| Seed | Steps | Pre-Quant val_loss | Pre-Quant val_bpb | Post-Quant val_loss | Post-Quant val_bpb | Artifact Size |
|------|-------|--------------------|--------------------|---------------------|--------------------|---------------|
| 1337 | 13,780 | 2.0606 | 1.2172 | 2.0727 | 1.2244 | 15,863,489 B |

- Step average: 43.54ms
- Train time: 600,038ms
- Total tokens: 7,224,688,640
- Quant degradation: ~0.007 BPB
- Peak memory: 10,184 MiB allocated

## Based On

This **is** the baseline. All other attempts build on or modify this configuration.
