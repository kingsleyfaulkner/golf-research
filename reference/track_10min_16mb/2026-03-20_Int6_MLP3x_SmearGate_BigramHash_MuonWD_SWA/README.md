# Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA

**Date**: 2026-03-20
**val_bpb**: 1.1458 | **Artifact Size**: 15,862,650 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

A comprehensive improvement over the baseline that stacks seven orthogonal techniques: int6 quantization frees bytes to fund 3x MLP expansion, SmearGate and BigramHash add lightweight bigram-level context at the embedding layer, orthogonal initialization with muP scaling accelerates convergence, Muon weight decay improves quantization quality, and SWA smooths the final checkpoint. Achieves 1.1458 BPB, a 0.029 improvement over the prior SOTA of 1.1748.

## Key Changes from Baseline

### Architecture Changes
- MLP expansion increased from 2x to 3x (hidden 1024 -> 1536), the single largest quality contributor
- SmearGate: learned per-dimension sigmoid gate blending each token's embedding with the previous token's via `(1-g)*x + g*x_prev`, adding ~512 parameters
- BigramHashEmbedding: 4096-bucket hash table (dim=128, projected to 512) mapping adjacent token pairs to learned embeddings via XOR hash (`36313 * curr ^ 27191 * prev % 4095`), adding ~524K parameters
- Sequence length doubled from 1024 to 2048
- Orthogonal weight initialization for all large matrices with muP-scaled output projections (`1/sqrt(2*num_layers)`)

### Training Changes
- Muon weight decay 0.01 (baseline had none), applied as decoupled `p *= 1 - lr * wd`
- AdamW replaces Adam for embeddings/scalars with weight_decay=0.01
- Learning rates halved: matrix_lr 0.04->0.02, scalar_lr 0.04->0.02, tied_embed_lr 0.05->0.03
- Muon momentum 0.95->0.99, warmup start 0.85->0.92, warmup steps 500->1500
- Warmdown iters 1200->3000
- Grad clip norm 0.0->0.3
- Batch size 524288->786432 tokens/step
- Val logging every 500 steps (was 1000), train logging every 100 (was 200)
- Stochastic Weight Averaging: every 50 steps during last 50% of warmdown (scale < 0.5), averaging ~30 checkpoints

### Quantization/Compression Changes
- Int6 per-row quantization ([-32, 31] with fp16 scales) for MLP and attention weight matrices
- Int8 kept for remaining float tensors (fallback path uses original quantize_float_tensor)
- FP16 passthrough for tied embeddings (tok_emb) and last-layer key projection (blocks.8.attn.c_k)
- FP32 passthrough for control tensors (scales, gates, skip_weights, smear, bigram.scale)
- Small tensors (numel <= 65536) kept as fp16 passthrough
- zstd level 22 compression replaces zlib-9

### Evaluation Changes
- Sliding window evaluation with stride=64 for improved BPB
- `forward_logits()` method for inference without loss computation
- `eval_val_sliding()` with batched window processing

## Detailed Code Diff Analysis

The core insight is that int6 quantization (6 bits per weight instead of 8) saves enough bytes to fund a 50% wider MLP (3x expansion vs 2x). The `quantize_int6_per_row()` function uses per-row fp16 scales with clip range [-32, 31], while `_classify_param()` routes tensors: "mlp" and "attn" categories go to int6, everything else falls back to int8.

The SmearGate operates after token embedding and RMS normalization but before the transformer blocks. It shifts a fraction of each token's representation toward the previous token using a learned per-dimension gate. This provides cheap bigram context without attention. The BigramHashEmbedding complements it with an additive signal from a hash table lookup of consecutive token pairs.

The Muon optimizer is extended with `weight_decay` parameter, implemented as decoupled weight decay (`p.data.mul_(1.0 - lr * wd)`) applied before the gradient update. This differs from L2 regularization and helps control weight magnitudes, directly benefiting int6 quantization quality.

The SWA implementation collects model snapshots every 50 steps when the warmdown scale drops below 0.5, then uniformly averages all collected checkpoints at the end of training. This produces smoother weight distributions that both generalize better and quantize with less error.

## Results

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.93492 | 1.14597 |
| 42 | 1.93591 | 1.14656 |
| 7 | 1.93314 | 1.14492 |
| **Mean** | **1.93466** | **1.14582** |
| **Std** | **0.00139** | **0.00082** |

- Pre-quant val_bpb: 1.1616
- Quantization penalty: ~0.016 bpb (int6 vs full precision)
- Training: 7,379 steps in 600s (81.3 ms/step)
- Artifact: 15,862,650 bytes (15,810,407 model + 52,243 code)
- Improvement over prior SOTA (1.1748): -0.0290 bpb / -0.0503 nats

## Based On

Direct modification of baseline. This submission is the foundation that later attempts (10L Int5MLP, 11L XSA variants) build upon.
