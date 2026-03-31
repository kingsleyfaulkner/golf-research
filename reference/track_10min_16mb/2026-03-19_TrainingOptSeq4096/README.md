# Training Opt Seq4096

**Date**: 2026-03-19
**val_bpb**: 1.2014 | **Artifact Size**: 15,868,326 bytes | **Hardware**: 8×H100 SXM, 600s

## Summary

Combines 4x longer training context (seq_len=4096) with aggressively tuned Muon optimizer hyperparameters: higher momentum (0.99), lower learning rates (0.02), smaller batch (3/4), and longer warmdown. The longer sequences give each token much better context during training, and the optimizer tuning dramatically reduces the int8 quantization penalty. Net improvement of 0.023 BPB over baseline.

## Key Changes from Baseline

### Architecture Changes
- No architectural changes (9 layers, 512 dim, same model)

### Training Changes
- Sequence length: 4096 (baseline: 1024) -- 4x longer context per training sequence
- Batch size: 393,216 tokens/step (baseline: 524,288) -- 3/4 batch for more optimizer updates per second
- Warmdown iterations: 3000 (baseline: 1200) -- proportionally longer for the ~8400-step run
- Muon momentum: 0.99 (baseline: 0.95) -- stronger gradient smoothing
- Muon momentum warmup start: 0.92 (baseline: 0.85)
- Muon momentum warmup steps: 1500 (baseline: 500) -- prevents early instability with higher momentum
- MATRIX_LR: 0.020 (baseline: 0.04)
- SCALAR_LR: 0.020 (baseline: 0.04)
- TIED_EMBED_LR: 0.030 (baseline: 0.05)
- Run ID hardcoded to "training_opt_seq4096_v1"

### Quantization/Compression Changes
- No changes (standard int8 + zlib-9)
- Lower LR produces tighter weight distributions, reducing quant penalty to 0.0034 BPB (vs 0.007+ at default LR)

### Evaluation Changes
- No changes to evaluation methodology

## Detailed Code Diff Analysis

The diff is purely hyperparameter changes -- no new code paths, functions, or logic. All changes are in the `Hyperparameters` class defaults:

The 4x sequence length increase costs ~71ms/step (vs ~43ms at 1024), yielding only ~8,400 steps in the 600s budget instead of ~13,800. Despite seeing fewer total steps, each step processes tokens with 4x more context, resulting in substantially better model quality.

The optimizer changes work synergistically: higher momentum (0.99) with lower LR (0.02) produces smoother weight distributions that quantize better. The extended momentum warmup (1500 steps from 0.92 to 0.99) prevents the higher momentum from causing instability in early training. The 3/4 batch size partially compensates for the slower step time by fitting more updates into the time budget.

## Results

| Seed | Steps | val_loss | val_bpb | Artifact Size |
|------|-------|----------|---------|---------------|
| 1337 | 8,394 | 2.0286 | 1.2014 | 15,868,326 |
| 1338 | ~8,400 | ~2.028 | 1.1995 | ~15,868,000 |
| 1339 | ~8,400 | ~2.030 | 1.2032 | ~15,868,000 |

**Mean val_bpb**: 1.2014 (std: 0.00187)

| Metric | Baseline | This |
|--------|----------|------|
| Pre-quant val_bpb | 1.2172 | 1.1980 |
| Post-quant val_bpb | 1.2244 | 1.2014 |
| Quant gap | ~0.007 | 0.0034 |
| Steps completed | ~13,800 | ~8,400 |
| Step avg | ~43ms | 71.47ms |
| Train time | 600s | 599,921ms |
| Peak memory | ~10,000 MiB | 7,748 MiB |

Statistical significance vs SOTA (1.2161 BPB):
- One-sided t-test against 1.2111 (SOTA - 0.005): t=9.06, df=2, p=0.006

## Based On

Direct modification of baseline. Builds on the insight from "Long Context Seq2048 v2" (1.2161 BPB) that longer sequence lengths improve quality, pushing to 4096 with complementary optimizer tuning.
