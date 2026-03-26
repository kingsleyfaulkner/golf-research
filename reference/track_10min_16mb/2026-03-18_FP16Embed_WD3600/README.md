# FP16 Tied Embedding + LR/Warmdown Tuning

**Date**: 2026-03-18
**val_bpb**: 1.2197 | **Artifact Size**: 15,896,222 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

Keeps the tied embedding weight (`tok_emb.weight`) in fp16 during int8 quantization instead of quantizing it, dramatically reducing the post-quantization BPB degradation from ~0.007 to ~0.0005. To fit within the 16MB budget after the larger embedding, MLP hidden size is reduced from 1024 to 992. Additionally tunes warmdown iterations (3600 vs 1200) and matrix learning rate (0.06 vs 0.04) for better convergence within the 10-minute wallclock cap.

## Key Changes from Baseline

### Architecture Changes
- New `mlp_hidden` hyperparameter allows overriding the MLP hidden dimension directly (instead of `mlp_mult * dim`).
- MLP hidden size reduced from 1024 (2x512) to 992 to compensate for the larger fp16 embedding in the artifact.
- `mlp_hidden` parameter threaded through `MLP`, `TransformerBlock`, and `GPT` constructors.

### Training Changes
- `WARMDOWN_ITERS=3600` (default 1200): Longer warmdown schedule better suited to the ~13,700 steps actually completed in 10 minutes.
- `MATRIX_LR=0.06` (default 0.04): Higher Muon learning rate for matrix parameters.
- Both changes applied via environment variables at runtime (code defaults unchanged).

### Quantization/Compression Changes
- `tok_emb.weight` kept as fp16 passthrough instead of int8 quantization. One-line change: the quantization function now checks `or name == "tok_emb.weight"` alongside the small-tensor check.
- This costs ~500KB extra in the artifact (1024 vocab x 512 dim x 2 bytes = 1MB fp16 vs ~512KB int8) but virtually eliminates the quantization gap on the output head.

### Evaluation Changes
- None. Standard continuous-stream BPB evaluation.

## Detailed Code Diff Analysis

**FP16 embedding passthrough** (the key change): In the int8 quantization function, the condition for keeping a tensor in float format is expanded from just checking `numel <= INT8_KEEP_FLOAT_MAX_NUMEL` to also matching `name == "tok_emb.weight"`. Since the embedding is tied to the output head, it's the most sensitive tensor -- every token prediction passes through it, so quantization noise is amplified across the entire vocabulary distribution.

```python
-        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
+        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or name == "tok_emb.weight":
```

**MLP hidden override**: A new `mlp_hidden` parameter (default 0, meaning use `mlp_mult * dim`) is added to the `Hyperparameters`, `MLP`, `TransformerBlock`, and `GPT` classes. When set to 992, each MLP layer has 992 hidden units instead of 1024, saving ~9 x 2 x 512 x 32 = 294,912 parameters (~300KB at int8), which partially offsets the fp16 embedding cost.

**Runtime LR/warmdown changes**: The author sets `WARMDOWN_ITERS=3600` and `MATRIX_LR=0.06` via environment variables. The default warmdown of 1200 iterations is designed for runs that complete many more steps; with only ~13,700 steps in 600s, a longer warmdown (26% of total steps vs 9%) allows more gradual convergence. The higher matrix LR compensates for the reduced MLP capacity.

## Results

| Seed | Steps | val_loss | val_bpb | Artifact Size |
|------|-------|----------|---------|---------------|
| 1337 | 13,692 | 2.0595 | 1.2197 | 15.90 MB |
| 42 | 13,722 | 2.0600 | 1.2201 | 15.90 MB |

Pre-quant vs post-quant gap: ~0.0005 BPB (baseline gap is ~0.007).

Improvement over baseline: ~0.005 BPB (~0.013 nats).

Also validated on 8xH200 SXM (3 seeds, 1.2163-1.2179 BPB range -- consistent improvement).

Failed experiments noted by the author:
- SwiGLU: better per-step but 45% slower, net negative
- Depth recurrence: needs more steps than 10 min allows
- QAT: overhead not worth the small quant gap reduction
- LZMA compression: worse than zlib for int8 data
- Higher embed LR (0.08): hurt convergence

## Based On

Direct modification of baseline ([NaiveBaseline](../2026-03-17_NaiveBaseline/)).
