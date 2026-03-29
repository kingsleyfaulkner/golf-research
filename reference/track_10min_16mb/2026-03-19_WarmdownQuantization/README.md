# Warmdown Quantization

**Date**: 2026-03-19
**val_bpb**: 1.2154 (standard eval) / 1.1574 (sliding window, per submission.json) | **Artifact Size**: 15,977,717 bytes | **Hardware**: 8×H100 SXM, 600s

## Summary

Attacks the post-training quantization bottleneck from multiple angles: int6 quantization (63 levels) for all block weights, FP16 passthrough for the tied embedding and last 2 layers' key weights, NTK-aware RoPE scaling for longer eval sequences, configurable MLP hidden dimension, and optional sliding window evaluation. The key insight is that on well-trained models, the quantization penalty dominates over most hyperparameter improvements.

## Key Changes from Baseline

### Architecture Changes
- Added `mlp_hidden` parameter: explicit MLP hidden dimension (default 0 = use mlp_mult * dim as before)
- MLP constructor takes explicit hidden dim instead of computing from multiplier
- TransformerBlock and GPT model pass through `mlp_hidden`
- NTK-aware RoPE scaling: Rotary module now takes `train_seq_len` parameter; when eval sequence length exceeds training length, it recomputes inv_freq using NTK-scaled base frequency (`base * (scale ** (dim / (dim - 2)))`)
- Added `get_logits` method on GPT model for inference without loss computation

### Training Changes
- No changes to training loop or optimizer hyperparameters in the code defaults
- README specifies: WARMDOWN_ITERS=20000, MATRIX_LR=0.06, TIED_EMBED_LR=0.07, SCALAR_LR=0.06, GRAD_CLIP_NORM=1.0, MUON_BACKEND_STEPS=5
- The extreme warmdown (20000 >> actual ~12,200 steps) means the entire training run is in the LR decay phase, producing tighter weight distributions with fewer outliers

### Quantization/Compression Changes
- Int6 quantization: `quantize_float_tensor` takes a `bits` parameter; default call uses `bits=6`, giving max_val=31 and 63 quantization levels instead of 255
- FP16 tied embedding passthrough: tok_emb.weight is explicitly kept as fp16 instead of being int8-quantized (dual role as input lookup and output projection makes it highly sensitive to quantization)
- Late-K passthrough: last 2 layers' c_k.weight (key projection) kept as fp16 instead of int8 quantization
- Scales and clamp values adjusted for the configurable bit width
- MLP_HIDDEN=992 used to offset the fp16 embedding overhead

### Evaluation Changes
- Added `eval_seq_len` parameter (default 2048): separate sequence length for evaluation
- Added `eval_stride` parameter (default 0): when > 0, enables sliding window evaluation
- Standard eval function updated to accept `eval_seq_len` parameter, using it instead of `train_seq_len` for validation
- Added `eval_val_sliding` function: sliding window eval distributed across ranks, scoring only the last `eval_stride` tokens per window (first window scores all)
- Validation tokens loaded at `max(train_seq_len, eval_seq_len)` to support longer eval sequences
- Both standard and sliding window evals run at the end; results reported separately

## Detailed Code Diff Analysis

**Int6 quantization**: The `quantize_float_tensor` function is parameterized by `bits`. For `bits=6`, `max_val = 2^(6-1) - 1 = 31`, so weights are quantized to [-31, 31] (63 levels) with per-row scaling. The int8 container stores these reduced-range values, which have much lower entropy and compress significantly better under zlib. The call site passes `bits=6` for all float tensors not handled by passthrough.

**FP16 passthrough strategy**: Two categories of tensors bypass quantization entirely:
1. `tok_emb.weight`: The tied embedding serves as both input lookup and output projection head. Int8 quantization causes disproportionate damage because errors affect both input representations and output logit accuracy.
2. Last 2 layers' `c_k.weight`: Key projection weights in the final layers are kept in fp16, following the "Late-K" trick from PR #99. The code dynamically computes `num_layers_total` from the state dict to identify the last 2 layers.

**NTK-aware RoPE**: When `seq_len > train_seq_len`, the Rotary module recomputes frequencies using a scaled base: `new_base = base * (scale ** (dim / (dim - 2)))`. This allows the model trained at seq_len=1024 to extrapolate to longer sequences during evaluation without catastrophic position encoding breakdown. The README notes eval@1408 (1.375x) is optimal for well-trained models, while eval@2048 is better for undertrained ones.

**Aggressive warmdown**: Setting WARMDOWN_ITERS=20000 with ~12,200 actual steps means the LR is already at 61% of peak at step 0 and decays linearly to near-zero. This produces much tighter weight distributions, reducing post-quant penalty from 0.014 BPB (default WD=1200) to 0.005 BPB (WD=20000).

## Results

From the README (standard eval at eval_seq_len=1408):

| Metric | Baseline | This |
|--------|----------|------|
| Post-quant val_bpb | 1.2244 | 1.2154 |
| Improvement | -- | -0.009 BPB / -0.017 nats |

From submission.json (likely with sliding window eval and additional optimizations):

| Metric | Value |
|--------|-------|
| val_loss | 1.9543 |
| val_bpb | 1.1574 |
| Artifact size | 15,977,717 bytes (code: 51,200) |

Note: The submission.json reports results from a combined configuration (int6 + MLP3x + sliding window + FP16 embeddings + Late-K passthrough) that achieves significantly better BPB than the README's standalone warmdown analysis.

## Based On

Direct modification of baseline. Combines insights from multiple angles: aggressive warmdown schedule, quantization-aware compression choices (int6, FP16 passthrough), and NTK-RoPE extrapolation for evaluation.
