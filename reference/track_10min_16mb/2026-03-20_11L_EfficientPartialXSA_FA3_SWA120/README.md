# 11L + Efficient Partial XSA + FA3 + SWA/120

**Date**: 2026-03-20
**val_bpb**: 1.1307 | **Artifact Size**: 15,892,986 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

Extends the SmearGate/BigramHash/Int6 foundation to 11 layers and introduces two novel contributions: Efficient Partial Exclusive Self Attention (XSA) applied selectively to the deepest 3 layers, and FlashAttention 3 for Hopper-optimized attention. XSA subtracts each token's self-value projection from the attention output, forcing deeper layers to learn from context rather than self-reference. A GQA-aware reshape eliminates the memory overhead of standard XSA, keeping the per-step cost under 2ms.

## Key Changes from Baseline

### Architecture Changes
- 11 layers (up from 9), giving 5 encoder + 6 decoder with U-Net skip connections
- MLP expansion 2x -> 3x (hidden 1024 -> 1536)
- SmearGate: learned per-dimension gate blending token with previous token's embedding
- BigramHashEmbedding: 4096 buckets (reduced from some other attempts), dim=128, projected to 512
- Exclusive Self Attention (XSA) on last 3 layers (layers 8, 9, 10), configurable via `xsa_last_n`
- Efficient GQA-aware XSA implementation using free reshape instead of `repeat_interleave`:
  - Reshapes attention output into KV head groups `[B, T, Hkv, group, D]` (zero-copy view)
  - Broadcasts normalized value vector `[B, T, Hkv, 1, D]` across groups
  - Subtracts self-value projection: `y = y - dot(y, v_norm) * v_norm`
- NTK-aware RoPE: auto-scales base frequency when seq_len exceeds train_seq_len (1024)
- FlashAttention 3 replaces PyTorch's `scaled_dot_product_attention` (direct `flash_attn_func` calls)
- Attention tensor layout changed: `[B, T, H, D]` instead of `[B, H, T, D]` (FA3 convention)
- Sequence length 1024 -> 2048 with NTK RoPE scaling
- Multi-Token Prediction (MTP) support added (configurable, disabled by default with `mtp_num_heads=0`)
- QAT support added via `CastedLinear._qat_enabled` flag (STE int6 quantization during training, disabled by default)

### Training Changes
- Muon weight decay 0.02 (`muon_wd`), AdamW weight decay 0.01 (`adam_wd`)
- Learning rates halved: matrix_lr 0.04->0.02, scalar_lr 0.04->0.02, tied_embed_lr 0.05->0.03
- Muon momentum 0.95->0.99, warmup 0.85->0.92, warmup steps 500->1500
- Warmdown iters 1200->3000 (used at env-var level; code default stays at 1200 but overridden by run command)
- Grad clip 0.0->0.3
- Batch 524288->786432 tokens/step
- Orthogonal init with muP-scaled output projections
- SWA every 200 steps when scale < 0.5 (run command overrides to every 120 steps, yielding ~13 checkpoint average)
- MTP heads params excluded from export (only used during training)

### Quantization/Compression Changes
- Mixed int6/int8 quantization: int6 per-row for MLP + attention, int8 for embeddings
- FP16 passthrough for tok_emb and late key projection
- zstd level 22 compression
- Control tensors (smear gate, bigram scale, skip weights) kept in fp32

### Evaluation Changes
- Sliding window evaluation with stride=64
- Separate eval model instantiated without MTP heads for round-trip validation
- NTK RoPE auto-scaling during eval at seq_len=2048
- Compiled eval model (`torch.compile` on `forward_logits`)
- Both standard and sliding window evaluations performed on int6-roundtripped weights

## Detailed Code Diff Analysis

The headline contribution is the Efficient Partial XSA. Standard XSA with GQA requires `repeat_interleave` to expand value vectors from `num_kv_heads` to `num_heads`, doubling memory per layer. This implementation exploits the group structure of GQA: by reshaping the attention output into `[B, T, Hkv, group_size, D]` (a free view), the normalized value vector `[B, T, Hkv, 1, D]` can broadcast across groups without any memory allocation. This reduces XSA overhead from ~7ms to ~2ms per step.

Partial application is equally important: the `xsa_last_n` parameter controls how many of the deepest layers use XSA. Setting it to 3 (out of 11) targets the layers with the highest self-attention bias while minimizing compute overhead. The `use_xsa` flag is set per-block in `GPT.__init__()` so that `torch.compile` can optimize the non-XSA layers without the branch.

The NTK-aware RoPE modification in `Rotary` auto-scales the base frequency when the sequence length exceeds `train_seq_len`: `new_base = base * (scale ** (dim / (dim - 2)))`. This allows training at 2048 with RoPE calibrated for 1024, extending the effective context without retraining the position encoding.

FlashAttention 3 is used via direct `flash_attn_func` import from `flash_attn_interface`, which expects `[B, T, H, D]` tensor layout (not `[B, H, T, D]`). The cos/sin cache layout is correspondingly changed from `[1, 1, T, D]` to `[1, T, 1, D]`, and the q_gain broadcast is adjusted from `[None, :, None, None]` to `[None, None, :, None]`.

The codebase also adds infrastructure for QAT (Quantization-Aware Training via STE in CastedLinear) and MTP (Multi-Token Prediction auxiliary heads), though both are disabled by default in this submission.

## Results

| Metric | Value |
|--------|-------|
| val_bpb (sliding s64) | 1.1307 |
| Pre-quant val_bpb | 1.1437 |
| Model parameters | 26,829,913 |
| Training steps | 6,976 in 600s (~86ms/step) |
| SWA checkpoints | 13 (every 120 steps) |
| Artifact size | 15,892,986 bytes |
| Model (int6+zstd) | 15,827,986 bytes |
| Code | 65,000 bytes |

## Based On

Builds on [2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA](../2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/), adding 11th layer, Efficient Partial XSA, FlashAttention 3, NTK RoPE, and MTP/QAT infrastructure.
