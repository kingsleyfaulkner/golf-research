# Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init

**Date**: 2026-03-19
**val_bpb**: 1.1748 (mean, 3 seeds) | **Artifact Size**: 15,374,243 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

A 10-layer transformer with int8 quantization (no QAT), using FP16 embedding passthrough, decoupled Muon weight decay, a novel "overtone" spectral embedding initialization, phase-transition residual mixing, NTK-aware RoPE for extended eval contexts, and sliding window evaluation. Unlike the other attempts in this batch, this one does not use int6 quantization or QAT -- instead it improves the standard int8 pipeline through better initialization, weight decay, and evaluation techniques.

## Key Changes from Baseline

### Architecture Changes
- 10 transformer layers (up from 9 baseline): extra layer funded by weight decay making weights more compressible
- NTK-aware dynamic RoPE scaling in `Rotary`: automatically adjusts rotary base when eval `seq_len > train_seq_len` using `base * (scale ** (dim / (dim - 2)))` formula
- `train_seq_len` parameter threaded through `CausalSelfAttention` -> `Rotary` for NTK scaling reference
- `forward_logits()` method added to GPT class for sliding window evaluation
- Configurable `eval_seq_len` (separate from `train_seq_len`) for evaluation at different context lengths

### Training Changes
- Warmdown iterations increased from 1200 to 2500
- Tied embedding LR doubled from 0.05 to 0.10
- AdamW replaces Adam for both token embedding and scalar optimizers (weight_decay=0.01)
- Decoupled Muon weight decay: `p.mul_(1 - 0.02 * lr)` applied to all matrix params after each optimizer step (hardcoded 0.02 factor)
- **Overtone spectral embedding init**: SVD-based power-law spectrum shaping on the tied embedding. After normal init, computes SVD and reshapes singular values to follow `S_k ~ S_0 * k^{-0.5}`, creating a power-law decay analogous to guitar harmonics. This gives the embedding a structured spectrum rather than random noise
- **Phase-transition residual mixing init**: `resid_mix` parameters initialized with sigmoid schedule `sigmoid(3.0 * (i/N - 0.5))` so early layers trust `x0` more (near 0) and late layers trust the residual more (near 1)

### Quantization/Compression Changes
- FP16 embedding passthrough: `tok_emb.weight` stored as fp16 in the quantized artifact instead of being int8 quantized. The int8 quantization hurts the embedding because errors compound through both the input embedding and output projection paths (tied embeddings)
- Still uses int8 quantization with zlib-9 for all other weights (no int6, no zstd)
- `load_state_dict` uses `strict=False` during roundtrip validation

### Evaluation Changes
- Sliding window evaluation via `eval_val_sliding()`: uses overlapping windows with configurable stride
- Compiled `forward_logits` for efficient batch inference with fixed batch size padding (256 sequences) to avoid recompilation
- First window scores all positions; subsequent windows score only the last `stride` positions
- Supports NTK-scaled eval at longer sequence lengths
- Eval batch padded to fixed size to avoid torch.compile recompilation on variable batch sizes

## Detailed Code Diff Analysis

The overtone init is one of the more creative changes. After the standard normal initialization of the embedding, it performs SVD, then replaces the singular values with a power-law decay: `target_S = S[0] * (1/k)^0.5`. The result `(U * target_S) @ V` preserves the random directions but gives the embedding a structured spectral profile. The intuition is that natural language embeddings should have a few dominant dimensions (common features) and many diminishing ones (rare features), matching a power-law distribution.

The phase-transition residual mixing initializes the `resid_mix` learned blend parameter per block with a sigmoid curve. Block 0 gets `sigmoid(3.0 * (0 - 0.5)) = sigmoid(-1.5) ~ 0.18` (mostly trust x0), while the last block gets `sigmoid(3.0 * (1 - 0.5)) = sigmoid(1.5) ~ 0.82` (mostly trust residual). This creates a smooth transition from "input-heavy" early layers to "residual-heavy" late layers.

The sliding window implementation pads each batch to a fixed 256 sequences to avoid triggering torch.compile recompilation on the last (smaller) batch. It also tracks `score_offset` per window: the first window scores all positions (offset=0), while subsequent windows only score the last `stride` positions (offset = seq_len - stride).

The `eval_val()` function is generalized to accept `seq_len_override` for evaluation at different sequence lengths than training, supporting the NTK-aware RoPE extension.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.98492632 | 1.17558517 | 10,424 | 57.55 |
| 42 | 1.98265459 | 1.17423973 | 10,710 | 56.06 |
| 7 | 1.98298356 | 1.17443456 | 10,498 | 57.18 |
| **Mean** | **1.98352149** | **1.17475315** | | |

- p-value: 0.0001
- Artifact: 15,374,243 bytes (model) + 50,651 bytes (code)
- Sliding window eval time: ~162s

## Based On

Direct modification of baseline. Shares the sliding window evaluation technique with the other attempts in this batch but takes a fundamentally different approach to compression (int8 + fp16 embed rather than int6 QAT).
