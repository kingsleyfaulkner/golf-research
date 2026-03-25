# 11L MLP3x + WD=0.04 + Int6 QAT + zstd-22 + Sliding Window Eval

**Date**: 2026-03-20
**val_bpb**: 1.1502 (mean, 3 seeds) | **Artifact Size**: ~15,427,455 bytes (mean) | **Hardware**: 8xH100 SXM, 600s

## Summary

An 11-layer transformer with 3x MLP expansion, int6 quantization-aware training, aggressive weight decay (0.04 on both Muon and AdamW), FP16 embedding export, zstd-22 compression, and sliding window evaluation. The deeper architecture (11 vs 9 layers) is funded by the byte savings from int6+zstd-22 compression. This is a highly configurable variant with env-var-driven int6 layer ranges, optional QAT, optional SWA, Block LARS, NTK-aware RoPE scaling, and zstd compression -- essentially a kitchen-sink experimentation framework built on the baseline.

## Key Changes from Baseline

### Architecture Changes
- No architecture changes to the model itself (layers, dims, attention remain the same)
- All architecture parameters (layer count, MLP mult, etc.) configured via env vars at launch time
- NTK-aware RoPE scaling added to `Rotary` and `CausalSelfAttention`: supports dynamic base scaling for eval at longer sequence lengths via `ntk_alpha` parameter
- `forward_logits()` standalone function added for sliding window eval (operates on uncompiled model)

### Training Changes
- Optional QAT: `CastedLinear` gains `_qat` and `_qat_int6` flags; when enabled, forward pass applies fake quantization with STE
- QAT int6 implementation: quantizes to int8 then rounds to multiples of 4 (`round(q_raw/4)*4`), clamped to [-128, 124] -- equivalent to 64 distinct levels
- Optional Muon weight decay: decoupled `p.mul_(1 - lr * wd)` applied after optimizer step
- Optional Adam weight decay: uses `AdamW` instead of `Adam` when `adam_weight_decay > 0`
- Stochastic Weight Averaging (SWA): accumulates running average of model weights during warmdown phase (when `scale < swa_start_frac`), applied before serialization
- Block LARS: scales gradients per block (attn/mlp/other) by `trust * ||W|| / ||grad||` with configurable trust, min/max scale
- All these features are opt-in via environment variables (disabled by default)

### Quantization/Compression Changes
- New `quantize_float_tensor_int6()` function: quantizes to int8 containers but rounds to multiples of 4 (64 levels), different from the other attempts' [-31, 31] approach
- Mixed quantization: `int6_layer_start` / `int6_layer_end` env vars control which layers get int6 vs int8
- FP16 embedding export: `tok_emb.weight` stored as fp16 passthrough when `fp16_embed_export=1`
- zstd-22 compression via `zstandard` module (optional, falls back to zlib-9)
- SWA-averaged weights are used for quantization and export

### Evaluation Changes
- Sliding window evaluation via `eval_val_sliding()` with configurable stride, seq_len, and batch size
- NTK-aware RoPE scaling during eval: `set_ntk_alpha()` adjusts all attention modules for longer eval sequences
- Interleaved window distribution across ranks (round-robin) for balanced load

## Detailed Code Diff Analysis

The QAT implementation in `CastedLinear` uses a different int6 approach than the other attempts: instead of quantizing to a [-31, 31] range, it quantizes to full int8 then rounds to the nearest multiple of 4, giving 64 distinct levels in [-128, 124]. This is functionally similar but uses a different numerical range.

The SWA implementation is straightforward: during warmdown (when the LR scale drops below `swa_start_frac`), it accumulates model state dicts and divides by count before serialization. This averages over the last portion of training, smoothing out noise in the final weights.

Block LARS is an interesting addition that scales gradients per parameter group (attention, MLP, other) based on the ratio of weight norm to gradient norm, providing adaptive learning rate scaling per block type.

The NTK-aware RoPE scaling modifies the rotary base frequency when eval sequence length exceeds train sequence length: `base_scaled = base * ntk_alpha` (or for the `Rotary.forward` path, automatic scaling based on `seq_len / train_seq_len`). This allows evaluation at longer contexts without the positional encoding breaking down.

## Results

| Seed | slide_loss (nats) | slide_bpb | rt_bpb | Artifact |
|------|-------------------|-----------|--------|----------|
| 1337 | 1.94265607 | 1.15055135 | 1.18484075 | 15,360,260 |
| 42 | 1.94207795 | 1.15020896 | 1.18456681 | 15,556,813 |
| 123 | 1.94121940 | 1.14970047 | 1.18421993 | 15,365,293 |
| **Mean** | **1.94198447** | **1.15015359** | **1.18454250** | **15,427,455** |
| **Std** | **0.00072288** | **0.00043** | | |

- Mean improvement: 0.1307 nats over baseline
- t-statistic: 313.20 (df=2, p << 0.001)
- All 3 artifacts under 16MB
- ~10,070 training steps at ~59.6ms/step

### Run Configuration (best result)

```
NUM_LAYERS=11, MLP_MULT=3, MATRIX_LR=0.025, SCALAR_LR=0.025
TIED_EMBED_LR=0.035, FP16_EMBED_EXPORT=1, INT6_LAYER_START=0
INT6_LAYER_END=10, QAT_ENABLED=1, QAT_INT6=1
MUON_WEIGHT_DECAY=0.04, ADAM_WEIGHT_DECAY=0.04
MUON_MOMENTUM=0.99, WARMDOWN_ITERS=3000, USE_ZSTD=1
EVAL_STRIDE=64
```

## Based On

Direct modification of baseline, but shares many techniques with [MixedQuant_Int6Int8_SlidingWindow](../2026-03-19_MixedQuant_Int6Int8_SlidingWindow/) (QAT, sliding window, tuned Muon hyperparams). Adds 11 layers, SWA, Block LARS, NTK RoPE, and a more configurable env-var-driven design.
