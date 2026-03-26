# LeakyReLU-squared + Legal Score-First TTT + Parallel Muon

**Date**: 2026-03-23
**val_bpb**: 1.1194 (3-seed mean, post-TTT) | **Artifact Size**: ~15,990,006 bytes (15.99 MB) | **Hardware**: 8xH100 SXM, 600s

## Summary

Combines three independent improvements on top of the PR #414 stack: LeakyReLU(0.5)-squared activation (-0.003 BPB vs relu-squared, a one-line change), legal score-first test-time training (TTT) that adapts the model on already-scored validation chunks (-0.0025 BPB), and Parameter Banking with Parallel Muon optimizer (no BPB change but ~2ms/step faster due to overlapped communication). The TTT protocol uses backward-looking adaptation with `torch.inference_mode()` during scoring, providing a hard guarantee that scoring is stateless.

## Key Changes from Baseline

### Architecture Changes
- **11 layers**, 512-dim, 8 heads (4 KV heads, GQA), U-Net skip connections
- **3x MLP expansion** (hidden=1536) with **LeakyReLU(0.5)-squared** activation: `F.leaky_relu(x, negative_slope=0.5).square()` replaces `torch.relu(x).square()`, preserving negative gradient flow while maintaining the squaring inductive bias
- **Exclusive Self Attention (XSA)** on last 4 layers
- **Partial RoPE**: 16/64 dims with NTK-aware scaling
- **LN Scale**: 1/sqrt(layer+1) damping
- **Shared Value Embedding (VE128)**: layers 9-10
- **SmearGate** + **BigramHash** (1536 buckets, down from 2048 in prior submissions)
- **FlashAttention 3** with BSTHD layout
- **Sequence length 2048**
- **Orthogonal init + muP-scaled output projections**

### Training Changes
- **Parameter Banking + Parallel Muon**: replaces standard DDP Muon with a fundamentally different communication pattern:
  - 4 contiguous 3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights
  - Batched Newton-Schulz orthogonalization via `torch.bmm` (operates on 3D tensors with `was_2d` handling)
  - Post-backward async reduce-scatter -> local NS5 on shard -> async all-gather, overlapping communication with computation
  - No DDP wrapping for bank parameters; launches biggest reduce-scatters first for optimal overlap
  - 83.3ms/step vs ~85ms baseline
- **Batch size**: 786,432 tokens/step
- **Warmdown**: 3500 iters
- **EMA**: decay=0.997, every step + **Tight SWA**: every 50 steps
- **Late QAT**: threshold=0.15
- **Muon**: lr=0.025, momentum=0.99, WD=0.04
- **AdamW**: lr=0.035 (tied embed), WD=0.04
- **Gradient clipping**: 0.3
- **LZMA compression** support added (import lzma)

### Quantization/Compression Changes
- **GPTQ-lite int6** with optimal clip percentile search
- **lzma** compression added alongside zstd/zlib options
- Int6 per-row for MLP + attention, int8 for embeddings

### Evaluation Changes
- **Sliding window evaluation**: stride=64
- **Legal Test-Time Training (TTT)**: backward-looking, score-first adaptation protocol:
  1. Validation tokens split into 1,893 non-overlapping 32K-token chunks
  2. For each chunk: **SCORE** under `torch.inference_mode()` (no gradients, no weight mutation possible), then **TRAIN** with SGD(lr=0.002, momentum=0.9) on the already-scored chunk for 3 epochs
  3. All blocks unfrozen during adaptation (freeze_blocks=0)
  4. Cosine LR decay across chunks, gradient clipping at 1.0
  5. Last chunk scored but never trained on
  6. Total eval time: ~530s (120s standard eval + 410s TTT)

## Detailed Code Diff Analysis

The Parallel Muon optimizer is the largest structural change. The `Muon.__init__` method now includes a `_build` phase that analyzes parameter banks: computing padded sizes for even distribution across ranks, pre-allocating gradient/shard/momentum/update buffers, and sorting banks by size descending so the largest reduce-scatters launch first.

The optimizer operates in three phases:
1. `launch_reduce_scatters()`: called right after backward, launches async reduce-scatter for all banks
2. Adam steps on small params while RS is in-flight (overlap)
3. `step()`: waits for each RS, runs local NS5 on the shard, launches async all-gather, each all-gather overlaps with next bank's NS5

The Newton-Schulz function was modified to support batched 3D inputs (B, M, N) using `.mT` transposition and per-matrix normalization via `X.norm(dim=(-2, -1), keepdim=True)`.

The TTT implementation adds extensive hyperparameters (ttt_enabled, ttt_lr, ttt_epochs, ttt_chunk_tokens, ttt_freeze_blocks, ttt_momentum, ttt_batch_seqs, ttt_grad_clip) and uses SGD with momentum rather than Adam, adapting the model chunk-by-chunk in a strictly backward-looking manner.

The LeakyReLU change is literally one line in `MLP.forward`, but the ablation shows it as the single largest contributor (-0.0021 BPB post-TTT).

## Results

| Seed | ms/step | Steps | Pre-TTT BPB | **Post-TTT BPB** | TTT Gain | TTT Time | Artifact |
|------|---------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 83.3 | 7,179 | 1.1217 | **1.1192** | -0.0025 | 410s | 15,977,386 |
| 42 | 83.4 | 7,182 | 1.1227 | 1.1200 | -0.0027 | 408s | 15,876,510 |
| 2025 | 83.4 | 7,193 | 1.1212 | **1.1189** | -0.0023 | 408s | 15,990,006 |
| **Mean** | 83.4 | 7,185 | 1.1218 | **1.1194** (std 0.0006) | -0.0025 | ~409s | |

Ablation (all seed 1337):
| Change | Pre-TTT BPB | Post-TTT BPB | Delta |
|--------|-------------|-------------|-------|
| PR #414 base (relu-squared, BIGRAM=2048) | 1.1234 | -- | -- |
| + Parameter Banking + Parallel Muon | 1.1234 | -- | +/-0.0000 |
| + Legal TTT (3ep, freeze=2) | -- | 1.1217 | -0.0017 |
| + TTT freeze=0 (all blocks) | -- | 1.1213 | -0.0004 |
| + BigramHash 2048->3072 | -- | 1.1204 | -0.0009 |
| + **LeakyReLU(0.5)-squared** | 1.1213 | **1.1183** | **-0.0021** |

## Based On

Builds on [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush. Techniques sourced from:
- LeakyReLU-squared: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- Parameter Banking + Parallel Muon: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- TTT recipe: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon (adapted: freeze=0 vs original freeze=2)
