# 010 Int6 QAT (Quantization-Aware Training)

Applies fake int6 per-row quantization with straight-through estimator (STE) during training to make the model robust to post-training int6 quantization.

## Change from baseline

- All 2D weight matrices apply fake int6 quantization during forward pass
- Quantization range: [-31, 31] (6-bit signed integer)
- Per-row scales computed from the 99.99998th percentile of absolute values
- STE: forward uses quantized weights, backward passes gradients through unquantized weights
- Only active during training; inference uses full-precision or post-training quantized weights
- Control tensors (scales, gains, skip weights) and embeddings are excluded from fake quantization

## Source

- `reference/track_10min_16mb/2026-03-19_int6_STE QAT_ MLP_bigram _U_Net/` (embedded source, lines 637-654)
- Also used in:
  - `reference/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/` (1.1598 BPB)
  - `reference/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` (1.1502 BPB)
  - `reference/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/` (1.1556 BPB)
  - All later SOTA submissions use QAT as standard practice

## How it works

During each forward pass of a linear layer with 2D weights:

```python
# Compute per-row clipping threshold (99.99998th percentile)
clip_abs = torch.quantile(w.abs(), 0.9999984, dim=1).clamp_min(1e-8)
scale = clip_abs / 31  # per-row scale for int6 range

# Quantize: clip, round to int6 grid, rescale
w_clipped = torch.clamp(w, -clip_abs[:, None], clip_abs[:, None])
w_q = (torch.round(w_clipped / scale[:, None]) * scale[:, None])

# STE: use quantized value in forward, but gradient flows to original w
w = w + (w_q - w).detach()
```

The key insight is `w + (w_q - w).detach()`: this evaluates to `w_q` in the forward pass but has gradient `dL/dw` in the backward pass (since `.detach()` blocks the gradient through `w_q - w`). The model learns weight distributions that quantize cleanly.

## Expected impact

- Reduces post-training int6 quantization gap from ~0.007 BPB to ~0.001 BPB
- This is critical for int6 deployment: without QAT, int6 quantization costs ~0.007 BPB; with QAT, it costs ~0.001 BPB, a net saving of ~0.006 BPB
- Slight training slowdown (~5-10%) from the extra quantization arithmetic per forward pass
- Enables using int6 quantization (which saves ~25% model size vs int8), freeing bytes for larger models (more layers, wider MLP)

## Status

**Not yet runnable.** Requires Composer changes to add QAT support.

### Required Composer changes

1. **New model trainer feature: `FakeQuantize`** in `composer/training/model_trainers/features/`

   A training feature that wraps linear layer forward passes with fake quantization. Implementation approach:

   **Option A — Forward hook based** (non-invasive):
   - Register `forward_pre_hook` on all `nn.Linear` modules matching a pattern
   - In the hook, replace `module.weight.data` with the STE-quantized version for the forward pass
   - Restore original weights after the forward pass
   - Pro: no changes to model code. Con: hook overhead, tricky with `torch.compile`

   **Option B — Custom Linear wrapper** (like the reference `CastedLinear`):
   - New `QATLinear` module that applies fake quantization in `forward()`
   - Replace `nn.Linear` instances at model construction time based on config
   - Pro: clean, compile-friendly. Con: requires modifying model construction

   **Option C — Compile-compatible parametrize** (recommended):
   - Use `torch.nn.utils.parametrize.register_parametrization` to add a `FakeQuantize` parametrization to weight tensors
   - The parametrization applies the STE quantization in the forward pass
   - Pro: cleanest integration, works with `torch.compile(fullgraph=True)`, no hooks
   - Con: needs careful handling of the STE `.detach()` within the parametrization

   Config schema:
   ```yaml
   features:
     - FakeQuantize:
         scheme: int6              # quantization scheme
         quant_range: 31           # signed integer range [-31, 31]
         clip_percentile: 0.9999984  # per-row clipping quantile
         patterns: ["blocks.*"]    # which modules to apply to
         exclude_patterns: ["tok_emb", "bigram_hash"]  # skip these
         min_ndim: 2               # only quantize 2D+ tensors
   ```

2. **Integration considerations**:
   - Must be compatible with `torch.compile(fullgraph=True)` — the STE `w + (w_q - w).detach()` pattern compiles cleanly, but hooks may break fullgraph
   - Should only apply during training (not inference)
   - The `clip_percentile` (0.9999984 ≈ 1 - 1.6e-6) clips extreme outliers; this value is tuned and should be configurable
   - Later records introduce "late QAT" where fake quantization is only enabled after a warmdown threshold (e.g., when LR scale < 0.15), so the feature should support an optional `enable_after_scale` parameter

3. **Testing**: The round-trip test is straightforward — train with QAT, then apply real int6 post-training quantization and measure the BPB gap. With QAT, the gap should be < 0.002 BPB; without QAT, it's typically ~0.007 BPB.
