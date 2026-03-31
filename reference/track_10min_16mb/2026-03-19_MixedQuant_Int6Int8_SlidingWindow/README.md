# Mixed Quant (Int6 Blocks + Int8 Embeddings) + Sliding Window Eval

**Date**: 2026-03-19
**val_bpb**: 1.1630 | **Artifact Size**: 15,353,490 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

Combines four orthogonal improvements over the baseline: 3x MLP expansion for greater model capacity, mixed-precision post-training quantization (int6 for STE-protected block weights, int8 for unprotected embeddings), optimized training hyperparameters, and sliding window evaluation with stride=64. The key insight is that fake int6 quantization via STE during training makes block weights robust to aggressive quantization, but the embedding (which never sees fake quantization) needs gentler int8 treatment -- this mixed scheme reduces the quantization penalty from +0.048 to +0.0015 BPB.

## Key Changes from Baseline

### Architecture Changes
- MLP expansion factor increased from 2x to 3x (hidden=1536), growing parameters from ~17.1M to ~21.8M
- Added `forward_logits()` method to GPT class for sliding window evaluation (returns logits without computing loss)

### Training Changes
- Warmdown iterations increased from 1200 to 3000 (longer learning rate decay phase)
- Validation disabled during training (`val_loss_every` 1000 -> 0)
- Tied embedding LR reduced from 0.05 to 0.030
- Matrix LR reduced from 0.04 to 0.020
- Scalar LR reduced from 0.04 to 0.020
- Muon momentum increased from 0.95 to 0.99
- Muon momentum warmup start raised from 0.85 to 0.92
- Muon momentum warmup steps increased from 500 to 1500

### Quantization/Compression Changes
- Int6 quantization-aware training (QAT) via Straight-Through Estimator (STE) in `CastedLinear.forward()`: forward pass uses fake-quantized weights ([-31, 31] range), backward passes gradients through as-is
- Mixed-precision post-training quantization: int6 (31 levels) for 2D block weights with STE protection, int8 (127 levels) for `tok_emb` embedding without STE protection
- Configurable `INT8_FULL_RANGE_PATTERNS` env var to select which tensors get int8 treatment
- Clip quantile changed from 99.99984% (applied as int8 range) to the same percentile but applied as int6 range
- zstd-22 compression replaces zlib-9 (with zlib-9 fallback if zstandard not installed)
- Quantized artifact saved as `.int6.ptz` instead of `.int8.ptz`

### Evaluation Changes
- Sliding window evaluation with configurable stride (default 64): each scored token gets (seq_len - stride) = 960 tokens of context
- `eval_val_sliding_window()` function distributes windows across ranks, processes in batches of 32 (configurable via `SW_EVAL_BATCH`)
- Sliding window improves BPB by ~0.034 with zero artifact cost

## Detailed Code Diff Analysis

The most important change is the STE fake-quantization injected into `CastedLinear.forward()`. During training, 2D weight matrices are quantized to int6 range ([-31, 31]) with per-row scaling, then the STE trick `w = w + (w_q - w).detach()` ensures the forward pass sees quantized values while gradients flow through the original weights. This trains the model to be robust to int6 quantization.

The mixed quantization scheme in `quantize_state_dict_int8()` checks each tensor name against `INT8_FULL_RANGE_PATTERNS` (default: "tok_emb"). Matching tensors get int8 (127 levels), others get int6 (31 levels). This is critical because the embedding never sees fake quantization during training.

The sliding window eval creates overlapping windows of size 1024 advanced by stride=64. Only the last 64 tokens per window are scored, so every scored token has 960+ tokens of context. The BPB computation correctly handles the byte-counting per the challenge's tokenizer-agnostic metric.

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb (last step) | 1.1950 |
| Int6/int8 mixed roundtrip val_bpb (standard) | 1.1965 |
| Int6/int8 mixed roundtrip val_bpb (sliding, stride=64) | **1.1630** |
| Quantization penalty (standard eval) | +0.0015 BPB |
| Sliding window eval time | 72.6s |
| Compressed artifact (int6+zlib-9) | 15,296,720 bytes |
| Code size | 56,770 bytes |
| Total submission size | 15,353,490 bytes |
| Training steps | 12,395 / 20,000 |
| Step time | 48.41ms average |
| Total train tokens | ~6.50B |

### Improvement Breakdown

| Component | val_bpb | vs Baseline |
|-----------|---------|-------------|
| Naive baseline (int8, standard eval) | 1.2244 | -- |
| + Wider MLP 3x + tuned hyperparams | 1.1950 | -0.0294 |
| + Mixed quant (int6 blocks, int8 embed) | 1.1965 | +0.0015 penalty |
| + Sliding window stride=64 | **1.1630** | -0.0335 additional |
| **Total improvement** | | **-0.0614** |

## Based On

Direct modification of baseline.
