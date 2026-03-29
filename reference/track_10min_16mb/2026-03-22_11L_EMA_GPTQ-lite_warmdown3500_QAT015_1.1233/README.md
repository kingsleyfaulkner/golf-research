# 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15

**Date**: 2026-03-22
**val_bpb**: 1.1233 (3-seed mean) | **Artifact Size**: ~15,555,017 bytes (15.55 MB) | **Hardware**: 8xH100 SXM, 600s

## Summary

Builds on the PR #374 stack (11L, XSA4, Partial RoPE, LN Scale, Shared Value Embedding, Tight SWA) with two novel post-training optimizations: GPTQ-lite (per-row optimal clip percentile search across 5 candidates during int6 quantization) and EMA weight averaging (decay=0.997), plus tuned warmdown (3500 iters) and earlier Late QAT threshold (0.15). GPTQ-lite is a zero-training-cost improvement that reduces quantization error by trying multiple clipping percentiles per weight row and selecting the one minimizing reconstruction MSE.

## Key Changes from Baseline

### Architecture Changes
- **11 layers** (up from 9), U-Net skip connections (5 encoder + 6 decoder)
- **3x MLP expansion** (hidden=1536), relu-squared activation
- **Exclusive Self Attention (XSA)** on last 4 layers via GQA-aware efficient implementation
- **Partial RoPE**: 16 of 64 head dimensions with NTK-aware scaling for seq_len=2048
- **LN Scale**: 1/sqrt(layer_idx+1) damping on RMSNorm outputs
- **Shared Value Embedding (VE128)**: single 128-dim embedding table shared across layers 9 and 10, with per-layer learned scales, injected into attention value projections
- **SmearGate**: learned token blending gate
- **BigramHash Embedding**: 2048 buckets, 128-dim, projected to 512-dim
- **FlashAttention 3** with BSTHD layout
- **Orthogonal init + muP-scaled output projections**
- **Sequence length 2048**
- **DTG (Dynamic Token Gating)**: optional gated residual (sigmoid gate on block input), present but disabled by default

### Training Changes
- **Batch size**: 786,432 tokens/step
- **Warmdown**: 3500 iters (up from 3000 in prior attempt, 1200 in baseline)
- **EMA**: decay=0.997, every step, applied before quantization
- **Tight SWA**: every 50 steps when LR scale < 0.2 (tighter collection than baseline's 200-step interval)
- **Late QAT threshold**: 0.15 (earlier activation than prior 0.1), enabling STE int6 fake-quantization sooner during warmdown
- **Muon**: lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- **AdamW**: lr=0.035 (tied embed), WD=0.04
- **Gradient clipping**: 0.3
- **QAT in CastedLinear**: STE int6 forward pass with per-row scaling

### Quantization/Compression Changes
- **GPTQ-lite**: per-row optimal clip percentile search for int6 quantization. Tries 5 percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and selects the clipping that minimizes reconstruction MSE. Zero training cost, purely post-training
- **Int6 per-row** for MLP + attention weights (range [-31, 31])
- **Int8 per-row** for embeddings
- **zstd level 22** compression
- **Control tensors** (smear, dtg_gate, ve_layer_scales, ve_shared.scale) in fp32/fp16

### Evaluation Changes
- **Sliding window evaluation**: stride=64, compiled forward_logits
- **Separate eval_seq_len**: 2048

## Detailed Code Diff Analysis

The most significant new contribution is the GPTQ-lite quantization in `quantize_int6_per_row`. Instead of using the row maximum as the clipping boundary (standard approach), it evaluates 5 clip percentiles per row:

```python
for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
    if pct < 1.0:
        row_clip = torch.quantile(t32.abs(), pct, dim=1)
    else:
        row_clip = t32.abs().amax(dim=1)
    # quantize with this clip, compute MSE, keep best
```

This finds the optimal trade-off between clipping outliers and minimizing overall reconstruction error. The name "GPTQ-lite" reflects the spirit of GPTQ's layer-wise optimization but with a much simpler grid search over clip thresholds.

The ValueEmbedding module is a notable architectural addition from PR #374: a shared embedding table that maps input tokens to low-dimensional vectors (128-dim), projected to KV dimension, and added to the value projection in specific layers (9, 10). This allows the model to reinject token identity information deep in the network. Each target layer has an independently learned scale parameter.

The Block now supports an optional DTG (Dynamic Token Gating) mechanism: a learned sigmoid gate that blends the block's input with its output, allowing the model to dynamically skip computation for certain tokens. The gate operates on detached inputs to avoid gradient flow through the gating decision.

## Results

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact Size |
|------|-------|----------|-------------------|---------------|
| **1337** | 7,101 | 1.8958 | **1.1228** | 15.56 MB |
| 42 | ~7,100 | 1.8972 | 1.1236 | 15.54 MB |
| 2024 | ~7,100 | 1.8971 | 1.1236 | 15.59 MB |

Mean: 1.1233, std: 0.0005. Submitted: seed 1337 (best).

Incremental impact from PR #374:
| Change | Impact |
|--------|--------|
| GPTQ-lite clip search | -0.0006 BPB |
| EMA (decay=0.997) | -0.0006 BPB |
| Warmdown 3000->3500 | -0.0002 BPB |
| Late QAT 0.1->0.15 | -0.0001 BPB |
| **Total** | **-0.0013 BPB** |

## Based On

Builds on [PR #374](https://github.com/openai/parameter-golf/pull/374) (11L, XSA4, Partial RoPE, LN Scale, VE128, Tight SWA, 1.1246 BPB). Author: Tianhao Wu (@signalrush).
