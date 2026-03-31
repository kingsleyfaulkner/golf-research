# 11L XSA4 + EMA + Int6 MLP3x + WD=0.04

**Date**: 2026-03-20
**val_bpb**: 1.1271 | **Artifact Size**: 15,534,645 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

The current record holder at time of submission, extending the 11L Efficient Partial XSA approach with two key changes: XSA applied to the last 4 layers (up from 3), and Exponential Moving Average (EMA) replacing Stochastic Weight Averaging (SWA). EMA with decay=0.997 updates a shadow model every step, producing smoother weights than periodic SWA snapshots, which both generalize better and compress more efficiently under int6+zstd. Achieves 1.1271 BPB with a 15.5MB artifact.

## Key Changes from Baseline

### Architecture Changes
- 11 layers (up from 9), 5 encoder + 6 decoder with U-Net skip connections
- MLP expansion 2x -> 3x (hidden 1024 -> 1536)
- SmearGate: learned per-dimension gate blending token with previous token's embedding
- BigramHashEmbedding: 4096 buckets, dim=128, projected to 512
- Exclusive Self Attention (XSA) on last 4 layers (up from 3 in the prior XSA attempt), configurable via `xsa_last_n`
- Efficient GQA-aware XSA: reshapes into KV head groups for zero-allocation self-value subtraction
- NTK-aware RoPE: auto-scales base frequency for seq_len > train_seq_len
- FlashAttention 3 (direct `flash_attn_func` from `flash_attn_interface`)
- Attention tensor layout `[B, T, H, D]` for FA3 compatibility
- Sequence length 2048
- MTP infrastructure present (disabled by default)
- QAT infrastructure present (disabled by default)

### Training Changes
- EMA weight averaging (decay=0.997) replaces SWA -- updates every step instead of periodic snapshots
  - `ema_state[name].mul_(d).add_(t.float(), alpha=1-d)` in float32 for accumulation precision
  - Applied at end of training: EMA weights loaded into base_model before quantization
  - SWA fallback still present but disabled when EMA is active
- Muon weight decay 0.02, AdamW weight decay 0.01
- Learning rates halved: matrix_lr 0.04->0.02, scalar_lr 0.04->0.02, tied_embed_lr 0.05->0.03
- Muon momentum 0.95->0.99, warmup 0.85->0.92, warmup steps 500->1500
- Warmdown iters 1200->3000 (via env var override)
- Grad clip 0.0->0.3
- Batch 524288->786432 tokens/step
- Orthogonal init with muP-scaled output projections

### Quantization/Compression Changes
- Mixed int6/int8: int6 per-row for MLP + attention, int8 for embeddings
- zstd level 22 compression
- FP16 passthrough for tok_emb and late key projection
- MTP head parameters excluded from export
- Control tensors (smear, bigram.scale, skip_weights) in fp32

### Evaluation Changes
- Sliding window evaluation with stride=64
- Separate eval model without MTP heads for round-trip validation
- Compiled eval model for performance
- Both standard and sliding window evaluations on int6-roundtripped weights

## Detailed Code Diff Analysis

The critical change from the prior XSA attempt is replacing SWA with EMA. The EMA state is maintained as a dictionary of float32 shadow tensors initialized from the model's initial state. Every training step updates the shadow via `ema_state[name].mul_(d).add_(t.float(), alpha=1-d)` with d=0.997. At the end of training, the EMA weights replace the model weights before quantization. The float32 accumulation prevents precision loss from compounding over thousands of steps.

The SWA code path remains but is gated: `if args.swa_enabled and not args.ema_enabled`. When EMA is active (`ema_enabled=True`), SWA is bypassed. The original README explains this provides smoother averaging than periodic checkpoints, yielding better generalization and artifact compression.

XSA coverage is extended from 3 to 4 layers (`xsa_last_n=4`), now applying to layers 7-10 (the last 4 of 11). The XSA paper shows self-attention bias increases monotonically across depth, so adding layer 7 captures more of the high-bias region. The cost is an additional ~0.5ms/step for the fourth XSA layer.

The codebase is otherwise structurally identical to the 11L_EfficientPartialXSA_FA3_SWA120 attempt, sharing the same FlashAttention 3 integration, NTK RoPE, BigramHash, SmearGate, orthogonal init, and mixed quantization pipeline.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (sliding s64) | **1.1271** |
| Pre-quant val_bpb | 1.1427 |
| Int6 roundtrip val_bpb | 1.1494 |
| Training steps | 7,103 in 600s (84ms/step) |
| Train tokens | ~5.6B (7,103 x 786,432) |
| Compressed artifact | 15,468,512 bytes |
| Code size | 66,133 bytes |
| **Total submission** | **15,534,645 bytes** |

### Multi-seed Reproducibility

| Seed | Steps | val_bpb (sliding s64) | Artifact Size |
|------|-------|----------------------|---------------|
| **1337** | 7,103 | **1.1271** | 15,534,645 |
| 42 | 7,094 | 1.1286 | 15,745,973 |
| 2025 | 7,107 | 1.1284 | 15,649,516 |
| **Mean** | | **1.1280** | |

### Progression

| PR | Config | val_bpb |
|----|--------|---------|
| #70 | 9L baseline | 1.1659 |
| #164 | 9L + int6/SmearGate/etc | 1.1524 |
| #198 | 11L + partial XSA + SWA | 1.1318 |
| **This** | **11L + XSA4 + EMA** | **1.1271** |

## Based On

Builds on [2026-03-20_11L_EfficientPartialXSA_FA3_SWA120](../2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/), extending XSA from 3 to 4 layers and replacing SWA with EMA (decay=0.997).
