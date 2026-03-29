# 012 Stochastic Weight Averaging (SWA)

Accumulates a uniform average of model weights during the warmdown phase, producing a smoother final checkpoint.

## Change from baseline

- After each training step during warmdown (when LR scale < `start_fraction`), add current model weights to a running sum
- At end of training, divide the accumulated sum by checkpoint count and load as the final model
- `start_fraction`: 0.5 (begin averaging when LR scale drops below 50%)
- `interval`: every step (some records use every 50 steps)

## Source

- `reference/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` (SWA with configurable `swa_start_frac`, lines 1279-1323)
- Also used in:
  - `reference/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` (SWA every 50 steps when scale < 0.4, ~30 checkpoints averaged)
  - `reference/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` (SWA every 50 steps when scale < 0.5)
  - `reference/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` (SWA every 120 steps)
- Later records switch to EMA (exponential moving average) which supersedes SWA

## How it works

The implementation is straightforward:

```python
# During training, when LR warmdown scale drops below threshold:
if scale < swa_start_frac:
    if swa_state is None:
        swa_state = {k: v.clone() for k, v in model.state_dict().items()}
        swa_count = 1
    else:
        for k, v in model.state_dict().items():
            swa_state[k].add_(v)
        swa_count += 1

# After training ends:
for k in swa_state:
    swa_state[k].div_(swa_count)
model.load_state_dict(swa_state)
```

By averaging over many late-training checkpoints, SWA smooths out SGD noise in the weight landscape, producing weights that:
1. Generalize better (flatter loss basin)
2. Quantize with less error (smoother weight distributions compress better)

## Expected impact

- Estimated ~-0.001 to -0.002 BPB
- Works synergistically with longer warmdown (002) — more warmdown steps = more checkpoints to average
- Also improves post-training quantization quality by smoothing weight distributions
- Later records found EMA (decay=0.997, every step) slightly outperforms SWA

## Status

**Not yet runnable.** Requires Composer changes to add SWA support.

### Required Composer changes

1. **New training feature: `StochasticWeightAveraging`** in `composer/training/features/`

   A training feature that hooks into the training loop to accumulate weight averages. Implementation:

   ```python
   class StochasticWeightAveraging(TrainingFeature):
       def __init__(self, start_fraction: float = 0.5, interval: int = 1):
           self.start_fraction = start_fraction
           self.interval = interval
           self.swa_state: dict[str, Tensor] | None = None
           self.swa_count = 0

       def on_step_end(self, trainer, step, scale, **kwargs):
           """Called after each optimizer step."""
           if scale >= self.start_fraction:
               return
           if step % self.interval != 0:
               return
           model = trainer.base_model
           if self.swa_state is None:
               self.swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
               self.swa_count = 1
           else:
               for k, v in model.state_dict().items():
                   self.swa_state[k].add_(v.detach())
               self.swa_count += 1

       def on_training_end(self, trainer, **kwargs):
           """Apply averaged weights before serialization."""
           if self.swa_state is not None and self.swa_count > 1:
               for k in self.swa_state:
                   self.swa_state[k].div_(self.swa_count)
               trainer.base_model.load_state_dict(self.swa_state, strict=True)
   ```

   Config schema:
   ```yaml
   features:
     - StochasticWeightAveraging:
         start_fraction: 0.5    # begin averaging when LR scale < this
         interval: 50            # average every N steps (1 = every step)
   ```

2. **Integration requirements**:
   - Needs access to the current LR warmdown `scale` value from the `WallclockWarmdown` scheduler
   - Must hook into `on_step_end` (after optimizer step, before next forward)
   - Must hook into `on_training_end` (before checkpoint serialization)
   - The accumulated state dict consumes 1x model size in memory (~80MB for 20M params in fp32) — negligible on H100

3. **EMA variant** (future extension):
   The same feature could support EMA as an alternative mode:
   ```yaml
   features:
     - ExponentialMovingAverage:
         decay: 0.997            # EMA decay factor
         interval: 1             # update every step
   ```
   EMA computes `ema_state = decay * ema_state + (1 - decay) * current_state` instead of uniform averaging, giving more weight to recent checkpoints. Later records found EMA slightly outperforms SWA.
