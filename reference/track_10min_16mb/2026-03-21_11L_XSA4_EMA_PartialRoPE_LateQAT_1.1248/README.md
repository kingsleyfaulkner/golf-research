# 11L Partial RoPE + LN Scale + EMA + XSA4

**Date**: 2026-03-21
**val_bpb**: 1.1248 | **Artifact Size**: 15,612,308 bytes (15.6 MB) | **Hardware**: 8xH100 SXM, 600s

## Summary

Extends the PR #287 stack (11L U-Net, XSA4, EMA, orthogonal init, SmearGate, BigramHash, 3x MLP, int6+zstd) with two zero-parameter improvements: Partial RoPE (applying rotary embeddings to only 16 of 64 head dimensions) and LN Scale (damping RMSNorm outputs by 1/sqrt(layer_idx+1) in deeper layers). Together these yield -0.0023 BPB over the prior record. Note: the Late QAT flag is present but inactive due to torch.compile constant-folding the class attribute.

## Key Changes from Baseline

### Architecture Changes
- **11 layers** (up from 9), with U-Net encoder-decoder skip connections (5 encoder + 6 decoder)
- **3x MLP expansion** (hidden=1536, up from 2x/1024), `mlp_mult` changed from int to float
- **Exclusive Self Attention (XSA)** on last 4 layers: subtracts self-value projection from attention output using a GQA-aware reshape (no repeat_interleave), enabled via `use_xsa` flag
- **Partial RoPE**: rotary embeddings applied to only the first 16 of 64 head dimensions; remaining 48 dims attend without positional bias. Implemented via split in `apply_rotary_emb` and NTK-aware base frequency scaling for longer sequences
- **LN Scale**: RMSNorm outputs scaled by `1.0 / math.sqrt(layer_idx + 1)` per block, stabilizing deeper layers
- **SmearGate**: learned gate blending current token with previous token embedding (~512 params)
- **BigramHash Embedding**: 2048-bucket hash table mapping token bigrams to 128-dim embeddings, projected to 512-dim with learned scale
- **Sequence length 2048** (up from 1024) with NTK-aware RoPE scaling
- **FlashAttention 3**: direct `flash_attn_func` calls replacing `F.scaled_dot_product_attention`, with BSTHD tensor layout (no transpose)
- **MTP heads** (multi-token prediction) supported but disabled by default (mtp_num_heads=0)
- **Orthogonal init + muP-scaled output projections**: all large linear layers get orthogonal init, output projections scaled by 1/sqrt(2*num_layers)

### Training Changes
- **Batch size**: 786,432 tokens/step (up from 524,288)
- **Muon weight decay**: 0.02 default (configurable via MUON_WD), applied as decoupled WD in optimizer step
- **AdamW** replaces Adam for token and scalar params, with weight_decay=0.01
- **Muon momentum**: 0.95 default (warmup from 0.85 to target over 500 steps in baseline; this submission uses 0.99 with warmup from 0.92 over 1500 steps via env vars)
- **Gradient clipping**: 0.3 (baseline had 0.0 / disabled)
- **Matrix/Scalar LR**: 0.025 (down from 0.04), tied_embed_lr: 0.035 (down from 0.05)
- **Warmdown**: 3000 iters (up from 1200)
- **EMA weight averaging**: decay=0.997, updated every step; applied after training, before quantization
- **SWA**: supported (collects every 200 steps when scale < 0.5), but disabled when EMA is active
- **Late QAT**: STE int6 fake-quantization flag present (activates when LR scale < 0.1), but torch.compile constant-folds the `_qat_enabled` class attribute so it never activates
- **QAT in CastedLinear**: STE forward pass with int6 fake-quantization (round to [-32,31] per-row), enabled via class-level `_qat_enabled` flag

### Quantization/Compression Changes
- **Int6 mixed quantization**: int6 per-row for MLP and attention weights (range [-32, 31]), int8 per-row for embeddings
- **zstd level 22** compression (with zlib-9 fallback), replacing baseline's zlib-9
- **Control tensors** (smear gate, scales, gains, skip weights) kept in fp32/fp16

### Evaluation Changes
- **Sliding window evaluation**: stride=64, each token scored with maximum context via `eval_val_sliding`
- **Compiled forward_logits**: separate method returning logits without loss, compiled for eval
- **Separate eval_seq_len**: configurable evaluation sequence length independent of training

## Detailed Code Diff Analysis

The most impactful changes are the Partial RoPE and LN Scale additions. In `Rotary.__init__`, a new `rope_dims` parameter controls how many dimensions receive rotary encoding. When `rope_dims < head_dim`, `apply_rotary_emb` splits the input into a rotated portion and a pass-through portion, allowing 75% of head dimensions to learn position-invariant attention patterns.

The LN Scale is a one-liner in `Block.__init__`: `self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0`, applied as a multiplicative factor to the RMSNorm output before both attention and MLP. This progressively dampens deeper layers, improving gradient flow in the 11-layer model.

The XSA implementation is notably efficient: instead of using `repeat_interleave` to expand KV heads, it reshapes the output into `[B, T, Hkv, group, D]` and performs the self-value subtraction via GQA-aware broadcasting, avoiding memory allocation.

The quantization pipeline was overhauled from simple int8 to a mixed int6/int8 scheme with separate `_classify_param` routing, enabling more aggressive compression for MLP and attention weights while preserving embedding quality with int8.

## Results

| Seed | Steps | Sliding BPB (s64) | Artifact Size |
|------|-------|--------------------|---------------|
| **2025** | 7,051 | **1.1248** | 15,612,308 |
| 42 | 7,061 | 1.1250 | 15,528,666 |
| 1337 | 7,063 | 1.1253 | 15,639,340 |

Mean val_bpb: 1.1250, std: 0.0005. Pre-quant val_bpb: 1.1418. Int6 roundtrip val_bpb: 1.1485.

## Based On

Builds on [PR #287](https://github.com/openai/parameter-golf/pull/287) (11L, XSA4, EMA, OrthoInit, SmearGate, BigramHash, int6+zstd, 1.1271 BPB). Previous chain: PR #70 (9L, 1.1659) -> PR #164 (9L, 1.1524) -> PR #198 (11L, 1.1318) -> PR #287 (11L, 1.1271) -> this.
