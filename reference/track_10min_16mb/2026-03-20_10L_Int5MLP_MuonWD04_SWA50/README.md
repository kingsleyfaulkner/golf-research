# 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

**Date**: 2026-03-20
**val_bpb**: 1.14276 | **Artifact Size**: ~15.9 MB | **Hardware**: 8xH100 SXM, 600s

## Summary

Builds on the SmearGate/BigramHash/OrthoInit foundation (from PR #162) by introducing mixed int5/int6 quantization to free bytes for a 10th transformer layer. Uses int5 for MLP weights (most compressible) and int6 for attention weights, with an enlarged BigramHash (10240 buckets vs 4096) and a more selective SWA that only averages the most-converged checkpoints (last 40% of warmdown). Weight magnitude pruning (3%) before quantization further improves zstd compression.

## Key Changes from Baseline

### Architecture Changes
- 10 layers (up from 9), giving 5 encoder + 5 decoder with U-Net skip connections
- MLP expansion increased from 2x to 3x (hidden 1024 -> 1536)
- SmearGate: learned per-dimension sigmoid gate blending each token with the previous token's embedding
- BigramHashEmbedding with 10240 buckets (dim=128, projected to 512 via CastedLinear), providing token-pair context
- Sequence length increased from 1024 to 2048

### Training Changes
- Muon weight decay added: 0.04 (decoupled, applied as `p *= 1 - lr * wd`)
- AdamW replaces Adam for embeddings/scalars with weight_decay=0.04
- Learning rates halved: matrix_lr 0.04->0.02, scalar_lr 0.04->0.02, tied_embed_lr 0.05->0.03
- Muon momentum increased 0.95->0.99, warmup start 0.85->0.92, warmup steps 500->1500
- Warmdown iters extended 1200->3000
- Grad clip norm added: 0.3 (was 0.0/disabled)
- Batch size increased 524288->786432 tokens/step
- Seed changed from 1337 to 42
- Val/train logging more frequent: val_loss_every 1000->500, train_log_every 200->100
- Orthogonal init with muP-scaled output projections (`1/sqrt(2*num_layers)`)
- Stochastic Weight Averaging: swa_start_frac=0.4 (only last 40% of warmdown), every 50 steps, averaging ~24 checkpoints
- 3% magnitude pruning of large weight matrices before quantization

### Quantization/Compression Changes
- Mixed int5/int6 quantization replacing uniform int8:
  - Int5 (clip_range=15, [-16,15]) for MLP weights -- saves ~1.86MB vs uniform int6
  - Int6 (clip_range=31, [-32,31]) for attention weights
  - FP16 passthrough for tied embeddings (tok_emb) and last-layer key projection (blocks.8.attn.c_k)
  - FP32 passthrough for control tensors (scales, gates, skip_weights)
- zstd level 22 compression replaces zlib-9 (~5% better on quantized data)
- Per-row fp16 scales for int5/int6 quantization

### Evaluation Changes
- Sliding window evaluation with stride=64 for improved BPB (~0.034 free improvement)
- Dedicated `forward_logits()` method for inference without loss computation
- `eval_val_sliding()` function implementing batched sliding window evaluation

## Detailed Code Diff Analysis

The most impactful change is the mixed-precision quantization scheme controlled by `_classify_param()`, which routes tensors to different bit widths by category. MLP weights get int5 quantization (clip=15), while attention weights get int6 (clip=31). This asymmetry is justified because MLP activations use ReLU-squared (naturally sparse/compressible), while attention weights are more precision-sensitive. The byte savings from int5 MLP quantization directly fund the additional 10th transformer layer.

The `mixed_quantize_int6()` function (despite the name, handles both int5 and int6) also classifies "bigram" as an int6-eligible category. The `FP16_KEEP_NAME_PATTERNS` mechanism preserves the token embedding and the last block's key projection in fp16, recognizing these as quantization-sensitive components.

The SWA implementation accumulates checkpoint weights only when `scale < swa_start_frac` (0.4), meaning it averages only from the most-converged portion of the warmdown phase. This quality-over-quantity approach produces smoother weight distributions that compress better.

The magnitude pruning pass (`torch.quantile(param.abs(), 0.03)`) zeros out the smallest 3% of weights in large matrices before quantization, creating more zero values that zstd-22 can exploit for better compression ratios.

## Results

| Seed | val_bpb | Artifact Size | Valid |
|------|---------|---------------|-------|
| 42 | 1.14271 | 15,965,978 | Yes |
| 1337 | 1.14298 | 15,830,186 | Yes |
| 2024 | 1.14260 | ~15.8M | Yes |
| **Mean** | **1.14276** | | |
| **Std** | **0.00016** | | |

### Ablation (from original README)

| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=8192 | 1.1434 | -0.0012 |
| + bigram=10240 | **1.1426** | **-0.0008** |

## Based On

Builds on [2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA](../2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/) (PR #162 by @unnir), adding the 10th layer via int5 MLP savings, enlarged BigramHash, and refined SWA.
