# 73.7M Ternary U-Net Transformer (BitNet b1.58)

**Date**: 2026-03-24
**val_bpb**: 1.1570 (3-seed mean, sliding stride=16) | **Artifact Size**: 15,994,104 bytes (15.99 MB) | **Hardware**: 8xH100 SXM, 599s

## Summary

A radically different approach from the standard int6/int8 quantization submissions: trains a 73.7M parameter model using BitNet b1.58 ternary quantization (weights constrained to {-1, 0, +1} via STE), enabling a much wider and more parameter-rich architecture (768-dim, 10L, 4x MLP) that fits in 16 MB thanks to ~1.6 bits/param via base-3 packing + LZMA compression. Uses a custom `train_gpt_cuda_ternary.py` script with 8192-vocab BPE tokenizer, YaRN positional encoding, polynomial softcap, FP8 QAT for non-ternary params, and NeoMuon optimizer with 3 Newton-Schulz steps. Despite the unconventional approach, achieves competitive 1.1570 BPB.

## Key Changes from Baseline

### Architecture Changes
- **Completely different model**: 10 layers, **768-dim** (up from 512), 8 heads, 4 KV heads, head_dim=96
- **4x MLP expansion** (hidden=3072, up from 2x/1024) with relu-squared, fused gate+up projection
- **BitNet b1.58 ternary weights**: `TernaryLinear` layer with per-group (128) absmean scaling and STE. Weights quantized to {-1, 0, +1} during forward pass, gradients flow through via straight-through estimator
- **8192-vocab BPE tokenizer** (up from 1024), with custom `fineweb_8192_bpe.model`
- **Factored tied embedding**: 8192x254 bottleneck with learned 254-to-768 and 768-to-254 projections. EMBED_DIM=254 (=256-2 to fit code within byte budget) saves ~4 MB for wider MLP
- **YaRN positional encoding** (max_len=2048, ROPE_BASE=5000): frequency-interpolated RoPE variant that applies ramp-based interpolation across frequency bands
- **Polynomial softcap** (degree 5, cap=10) with Z-loss regularization (1e-4), replacing simple tanh softcap
- **U-Net skip connections** with learned skip weights (ones-init) and per-block residual mix from input embedding
- **Fused QKV projection**: single TernaryLinear for combined Q/K/V
- **NormedTernaryLinear**: ternary linear with built-in RMSNorm for output projections
- **GroupedTernaryLinear**: block-diagonal ternary linear for grouped computation
- **TverskyProjection**: similarity-based projection module using learnable prototypes and membership functions (configurable sigmoid/poly/tanh)
- **QATLinear/QATEmbedding**: FP8 or FP4 STE-based fake-quantization for non-ternary parameters
- **Refiner**: optional convolutional refinement module (disabled by default)
- **Differential Attention**: optional (disabled by default)
- **No SmearGate or BigramHash** (both disabled)
- **FlashAttention 3** via `flash_attn_func`

### Training Changes
- **NeoMuon** with **3 Newton-Schulz steps** (down from 5): compensates for ternary STE gradient attenuation; the author found 3 steps equivalent to 5 at convergence, saving ~190 steps worth of compute
- **RMS normalization before NS orthogonalization**: `F.rms_norm(g.float(), (g.size(-1),)).bfloat16()` applied to gradients before Newton-Schulz, unique to this submission
- **Muon WD=0.0** (ternary weights incompatible with standard weight decay)
- **Adam LR=0.05, WD=0.05** for non-matrix params
- **Matrix LR=0.04, Scalar LR=0.02**
- **Muon momentum=0.95** (warmup 0.85->0.95 over 500 steps) -- lower than other submissions' 0.99
- **524,288 batch tokens** (same as baseline, optimal for ternary STE noise)
- **Sequence length 1024** (not 2048 like other submissions)
- **Warmdown fraction 0.2** (wallclock-proportional, not fixed iters)
- **No EMA or SWA** (incompatible with ternary weights)
- **Batch and sequence length scheduling**: configurable ramp-up over first 33% of training
- **91.8 ms/step**, ~6,530 steps in 600s
- **Ternary diagnostics**: zero-fraction and churn tracking during training

### Quantization/Compression Changes
- **Base-3 ternary packing**: 5 trits per byte (values {-1,0,+1} mapped to {0,1,2}), yielding ~1.6 bits/param for the 64.9M ternary parameters
- **Alternative bitmask packing**: non-zero bitmask + sign bitmask (used when it compresses smaller than base-3)
- **FP8 QAT (e4m3)**: non-ternary 2D weights stored as float8_e4m3fn (~2.5 MB for ~5M fp params), only 0.002 BPB roundtrip penalty
- **FP4 quantization**: optional int4 per-row with absmax scaling
- **FP16 storage**: for remaining small tensors
- **LZMA preset=9** compression (replaces zstd/zlib), achieving 39% better compression than int8+zlib for ternary weights
- **Shrinkage fix**: corrects ternary zero-fraction scale mismatch in dequantization by normalizing `q_absmean`, eliminating roundtrip gaps
- **Temperature scaling (T=0.90)**: 5-point grid search on training tokens, compensating for relu-squared logit underconfidence
- **Complete removal** of baseline's int8 quantization pipeline (CONTROL_TENSOR_NAME_PATTERNS, quantize_state_dict_int8, etc.)

### Evaluation Changes
- **Sliding window evaluation** with **stride=16** (much denser than other submissions' stride=64), yielding ~0.025 BPB improvement over chunked eval
- **Temperature scaling** applied during eval
- **Sliding batch size**: 256 sequences per batch

## Detailed Code Diff Analysis

This is fundamentally a different training script (`train_gpt_cuda_ternary.py`), not an incremental modification. The baseline's entire quantization pipeline, evaluation functions, and much of the model code is replaced.

The core innovation is `TernaryLinear.forward`:
```python
w_g = w.reshape(-1, g)
scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
q = (w_g / scale).round().clamp(-1, 1)
w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
```
This implements BitNet b1.58's per-group absmean STE: during forward, weights are quantized to {-1,0,+1} with group-wise scaling, but gradients flow through to the full-precision weights via the `w + (w_q - w).detach()` trick.

The `q_sd` serialization function intelligently routes each parameter: large 2D matrices get ternary quantization, smaller tensors get FP8/FP4/FP16 based on configuration. The `deq_sd` dequantizer applies the shrinkage correction during reconstruction:
```python
q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
t = (q * (scale / q_absmean))
```
This corrects for the fact that ternary quantized weights have a different mean absolute value than the original, which would otherwise introduce a systematic scale error.

The YaRN RoPE implementation uses frequency-dependent interpolation:
```python
ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
```
Low-frequency components are interpolated more aggressively while high-frequency components retain their original scale, enabling better extrapolation to longer sequences.

The NeoMuon modification adds RMS normalization of gradients before Newton-Schulz orthogonalization, which the author found stabilizes ternary training where STE introduces gradient noise.

## Results

| Seed | Steps | ms/step | Sliding BPB (s16) | val_bpb | Roundtrip BPB | Artifact |
|------|-------|---------|--------------------|---------|---------------|----------|
| 42 | 6,530 | 91.7 | **1.1565** | 1.1816 | 1.1837 | 15,993,853 |
| 1337 | 6,520 | 91.9 | 1.1568 | 1.1825 | 1.1839 | 15,995,705 |
| 7 | 6,530 | 91.8 | 1.1578 | 1.1823 | 1.1850 | 15,992,753 |
| **Mean** | 6,527 | 91.8 | **1.1570** (std 0.0007) | 1.1821 | 1.1842 | 15,994,104 |

Key architecture decisions (from 250+ run ablation log):
- Width over depth: 768d/10L outperforms 512d/25L (faster steps: 91ms vs 127ms, yielding 6,530 vs 4,720 steps)
- 4x relu-squared MLP: -0.024 BPB over relu at zero cost; 4x width adds -0.008 BPB over 3x
- EMBED_DIM=254: frees ~4 MB for wider MLP

## Based On

Direct modification of baseline with completely novel ternary training approach. Author: Ciprian-Florin Ifrim (@CiprianFlorin-Ifrim). The author documents 250+ experimental runs in a separate RESULTS.md covering binary and ternary BitNet scaling, with comprehensive ablations of applicable/rejected techniques.
