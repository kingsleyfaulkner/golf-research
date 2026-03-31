# Seq2048 FP16Emb TunedLR

**Date**: 2026-03-19
**val_bpb**: 1.1586-1.1610 (sliding window) / 1.1810-1.1821 (standard) | **Artifact Size**: 15,558,319 bytes | **Hardware**: 8×H100 SXM, 600s

## Summary

A comprehensive stacked improvement combining 10 layers, 3x MLP expansion (hidden=1536), STE int6 quantization-aware training (QAT), zstd-22 compression, FP16 tied embedding passthrough, Muon momentum 0.99, gradient clipping, sequence length 2048, and sliding window evaluation at stride=64. The int6 QAT completely eliminates the quantization gap (pre-quant = post-quant loss), and the byte savings from int6+zstd fund the wider MLP. This is one of the strongest submissions in batch 2.

## Key Changes from Baseline

### Architecture Changes
- 10 transformer layers (baseline: 9)
- MLP hidden dimension: 1536 (3x model_dim; baseline: 1024 = 2x)
- MLP constructor takes explicit `hidden` dimension instead of `mlp_mult * dim`
- TransformerBlock and GPT model updated to pass `mlp_hidden` through
- Added `forward_per_token_loss` method on GPT model for sliding window evaluation
- Sequence length: 2048 (baseline: 1024)

### Training Changes
- MATRIX_LR: 0.02 (baseline: 0.04)
- SCALAR_LR: 0.02 (baseline: 0.04)
- TIED_EMBED_LR: 0.04 (baseline: 0.05)
- Muon momentum: 0.99 (baseline: 0.95)
- Muon momentum warmup start: 0.92 (baseline: 0.85)
- Muon momentum warmup steps: 1500 (baseline: 500)
- Warmdown iterations: 3600 (baseline: 1200)
- Gradient clipping: GRAD_CLIP_NORM=0.3 (baseline: 0.0 = disabled)
- STE int6 QAT enabled during training via global `_QAT_ENABLED` flag
- QAT applies `_fake_quantize_int6` in every CastedLinear forward pass during training: quantize to [-31,31], dequantize, with gradients flowing through via straight-through estimator

### Quantization/Compression Changes
- Full int6 quantization: all 2D block weights quantized to [-31,31] (63 levels) stored in int8 container
- `quantize_float_tensor` parameterized with `use_int6` flag; when enabled, `qmax=31`
- FP16 passthrough for tied embedding (tok_emb.weight): configurable via `FP16_PASSTHROUGH_PATTERNS` env var
- zstd-22 compression replaces zlib-9: `zstandard` library used for both compression and decompression
- `USE_INT6`, `USE_ZSTD`, `ZSTD_LEVEL` flags control quantization/compression scheme
- Decompression path updated to detect and use zstd
- Requires `pip install zstandard`

### Evaluation Changes
- Sliding window evaluation at stride=64 (default via `EVAL_STRIDE=64`)
- Added `eval_val_sliding` function: batched sliding window eval with configurable stride and batch_size
- Both standard and sliding window results reported
- `eval_seq_len` parameter (default 0 = use train_seq_len)
- QAT explicitly disabled (`_QAT_ENABLED = False`) before eval/serialization

## Detailed Code Diff Analysis

**STE Int6 QAT** (`_fake_quantize_int6`): During training, every `CastedLinear` forward pass applies fake int6 quantization when `_QAT_ENABLED` is true. The function computes per-row amax, scales to [-31,31], rounds, and dequantizes. The STE trick `w + (dequant - w).detach()` ensures the forward pass uses quantized values while gradients flow through as if no quantization happened. This teaches the model to be robust to int6 quantization, completely eliminating the quant gap. The global flag approach (vs per-module flag) is simple but requires careful toggling around eval.

**Int6 + zstd compression pipeline**: The `quantize_float_tensor` function uses `qmax=31` when `use_int6=True`. Since values are in [-31,31] but stored in int8 containers, the data has very low entropy (only 63 possible values out of 256). zstd-22 exploits this much better than zlib-9, achieving smaller compressed sizes. The net byte savings fund the 3x MLP expansion.

**FP16 embedding passthrough**: Configured via `FP16_PASSTHROUGH_PATTERNS` env var (default "tok_emb"). Before checking if a tensor should be int6-quantized, the code checks if any pattern matches the tensor name. Matching tensors are kept as fp16, which is crucial for the tied embedding that serves dual input/output roles.

**3x MLP expansion**: With `mlp_hidden=1536` (3x the 512 model_dim), each MLP has 50% more capacity than baseline's 2x expansion. The byte budget for this comes from int6+zstd compression producing significantly smaller artifacts than int8+zlib.

**Sliding window eval**: Similar to other submissions in this batch. Windows advance by `stride=64`, only scoring the last `stride` tokens per window. Uses `forward_per_token_loss` which returns per-token cross-entropy losses shaped `(batch, seq_len)` for efficient batched evaluation.

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact Size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 8,319 | 1.1821 | 1.1610 | 15,558,319 |
| 42 | ~8,300 | ~1.1815 | 1.1598 | ~15,558,000 |
| 3 | ~8,300 | ~1.1810 | 1.1586 | ~15,558,000 |

**Mean val_bpb (sliding)**: 1.1598 (std: 0.00120)

| Metric | Baseline | This |
|--------|----------|------|
| Post-quant val_bpb | 1.2244 | 1.1598 (sliding) |
| Quant gap | ~0.007 | 0.0000 (QAT eliminates it) |
| Step avg | ~43ms | ~72ms |
| QAT overhead | -- | ~28% (72ms vs ~56ms without QAT at seq2048) |
| Sliding window eval time | -- | ~370s |
| Artifact size | ~15.8MB | 15,558,319 bytes |

Statistical significance vs baseline SOTA (1.2244 BPB):
- Improvement: 0.1144 nats
- t-statistic: -93.6, df=2, p << 0.01

## Based On

Stacked combination of techniques from multiple prior attempts:
- Sequence length 2048 and Muon tuning from [TrainingOptSeq4096](../2026-03-19_TrainingOptSeq4096/) (adapted to 2048)
- Sliding window evaluation from [SlidingWindowEval](../2026-03-19_SlidingWindowEval/)
- FP16 embedding passthrough and int6 quantization concepts from [WarmdownQuantization](../2026-03-19_WarmdownQuantization/)
- 10 layers from [10L_MixedPrecision](../2026-03-19_10L_MixedPrecision/)
- STE QAT and zstd-22 compression are novel to this submission
