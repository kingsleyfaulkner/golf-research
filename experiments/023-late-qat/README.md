# 023 Late QAT

Delays QAT activation until the warmdown phase, only enabling STE int6 fake-quantization when the LR scale drops below a threshold (0.15).

## Change from baseline

- QAT starts disabled — no fake quantization during the initial high-LR training phase
- When the WallclockWarmdown scale drops below `late_qat_threshold` (0.15), QAT activates for all remaining steps
- Once enabled, all 2D weight matrices in CastedLinear apply STE int6 fake-quantization (same as 010-qat-int6)
- Threshold 0.15 means QAT activates during the last ~15% of the LR schedule

## Source

- `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (lines 83, 1225-1227)
- Prior attempt used threshold 0.1; this record found 0.15 gives -0.0001 BPB improvement (earlier activation, smaller quant gap)
- Also used in `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

## How it works

```python
# In training loop, after computing warmdown scale:
if late_qat_threshold > 0 and scale < late_qat_threshold and not CastedLinear._qat_enabled:
    CastedLinear._qat_enabled = True
    log(f"late_qat:enabled step:{step} scale:{scale:.4f}")
```

**Why late rather than always-on**:
- Always-on QAT (010-qat-int6) injects quantization noise from step 0, when the LR is high and weights are changing rapidly. The model must learn to be robust to quantization while simultaneously learning the task.
- Late QAT only activates when the LR is low and weights are nearly converged. The model first learns the task at full precision, then fine-tunes for quantization robustness during warmdown.
- The warmdown phase is naturally suited for this: the small LR means weights change slowly, so the fake-quantization noise is a small perturbation that the model can adapt to.

**Tradeoff**: Late QAT at threshold 0.15 means ~85% of training is without quantization noise (higher quality gradients) and ~15% is with it (learning quantization robustness). The original record notes this is slightly better than threshold 0.1 (10%) — earlier activation gives more time to adapt.

**Note**: The PR #374 record found a torch.compile bug where `CastedLinear._qat_enabled` was constant-folded at compile time, making the late activation ineffective. This was fixed in later records by using a module-level flag or recompiling.

## Expected impact

- Estimated ~-0.0001 BPB over always-on QAT (marginal)
- The main benefit is allowing full-precision training during the high-LR phase, potentially reaching a better pre-quant solution
- Combined with EMA (019), the late-QAT weights are smoothed over both the pre-QAT and post-QAT training phases

## Builds on

- [010 Int6 QAT](../010-qat-int6/) — Late QAT is a scheduling extension of the same fake-quantization mechanism

## Status

**Not yet runnable.** Requires the same Composer `FakeQuantize` feature as 010-qat-int6, plus an `enable_after_scale` parameter.

### Required Composer changes

Same as 010-qat-int6, with an additional scheduling parameter:

```yaml
features:
  - FakeQuantize:
      scheme: int6
      quant_range: 31
      clip_percentile: 0.9999984
      patterns: ["blocks.*"]
      exclude_patterns: ["tok_emb", "bigram_hash"]
      min_ndim: 2
      enable_after_scale: 0.15  # only activate when LR scale < this
```

The `enable_after_scale` parameter hooks into the `WallclockWarmdown` scheduler's scale value. When `scale >= threshold`, fake quantization is bypassed (full precision forward). When `scale < threshold`, fake quantization activates and remains on for all subsequent steps.

**torch.compile consideration**: The QAT flag must not be constant-folded by `torch.compile`. Options:
- Use a `torch.compiler.is_compiling()` guard
- Store the flag as a buffer rather than a class attribute
- Recompile the model when QAT activates (one-time cost)
