# 10L Mixed Precision

**Date**: 2026-03-19
**val_bpb**: 1.2147 | **Artifact Size**: 15,928,974 bytes | **Hardware**: 8×H100 SXM, 600s

## Summary

Adds a 10th transformer layer (from 9) and uses lower learning rates, with a mixed-precision int8/int6 post-training quantization scheme to fit the larger model under the 16MB cap. Middle layers (3-6) are rounded to int6 precision (64 levels) for better zlib compression, while early/late layers retain full int8 precision where sensitivity is highest.

## Key Changes from Baseline

### Architecture Changes
- 10 transformer layers instead of 9 (18.9M params vs baseline's ~17M)
- No other architectural changes

### Training Changes
- Lower learning rates: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 (baseline: 0.04/0.04/0.05)
- All other training hyperparameters unchanged

### Quantization/Compression Changes
- Added `prune_ratio` parameter: zeroes out small int8 values (|val| <= threshold) for better compression
- Added `int4_layers` parameter: specifies which layers get reduced precision (default: none)
- Added `int4_step` parameter: rounding step for reduced-precision layers (2=int7, 4=int6, 8=int5, 16=int4)
- Used INT4_LAYERS=3,4,5,6 with INT4_STEP=4 (effectively int6 for middle layers): int8 values are rounded to nearest multiple of 4, reducing effective precision to 64 levels while stored in int8 container
- Compression remains zlib-9

### Evaluation Changes
- No changes to evaluation

## Detailed Code Diff Analysis

The core addition is a post-quantization step that runs after standard int8 quantization. Two new code paths are added:

**Pruning** (`prune_ratio > 0`): Iterates over quantized int8 tensors and zeroes out values whose absolute value is below `127 * prune_ratio`. This creates more zeros for better compression. Not used in the submitted configuration.

**Mixed precision** (`int4_layers`): For layers whose index appears in the comma-separated list, the int8 quantized values are further rounded: `((t / step).round() * step).clamp(-127, 127)`. With `INT4_STEP=4`, this rounds every value to the nearest multiple of 4 (i.e., only values -124, -120, ..., -4, 0, 4, ..., 120, 124 survive), giving 63 effective levels. The low entropy of these rounded values compresses much better under zlib.

The layer selection (3,4,5,6 = the 4 middle layers out of 10) is based on the observation that early layers (input processing) and late layers (output prediction) are more sensitive to quantization noise than middle layers.

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_loss | 2.0480 |
| Pre-quant val_bpb | 1.2129 |
| Post-quant val_loss | 2.0510 |
| Post-quant val_bpb | 1.2147 |
| Quant gap (bpb) | 0.0018 |
| Training steps | 13,100 / 20,000 |
| Step avg | 45.78ms |
| Train time | 599,732ms |
| Peak memory | 11,389 MiB |
| Artifact size | 15,928,974 bytes (code: 48,917 + model: 15,880,057) |

Note: Run performed on 8xH200 (slightly faster than H100). On H100, expected ~12,500-12,700 steps.

## Based On

Direct modification of baseline. LR sweep data showed 0.02 optimal for MATRIX_LR (1.2230 bpb vs 1.2286 at default 0.04 on 9L baseline).
