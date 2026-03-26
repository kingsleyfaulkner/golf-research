# 019 Exponential Moving Average (EMA)

Maintains a continuously-updated shadow copy of model weights using exponential moving average, producing smoother final weights than SWA's periodic snapshots.

## Change from baseline

- At training start, initialize EMA state as a float32 clone of all model parameters
- After every training step, update: `ema[name] = decay * ema[name] + (1-decay) * param`
- At end of training, replace model weights with EMA weights before quantization
- `decay`: 0.997 (gives ~333-step half-life)
- Updated every step (not periodically like SWA)
- Accumulation in float32 to prevent precision loss over thousands of steps

## Source

- `reference/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` (lines 1313-1315, 1389-1393, 1429-1434)
- Supersedes SWA (012) in all records from 2026-03-20 onward
- Also used in:
  - `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (EMA 0.997)
  - `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (EMA 0.997)
  - `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (EMA 0.997)

## How it works

```python
# Init: clone all model params to float32 shadow
ema_state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}

# Every training step:
d = 0.997
with torch.no_grad():
    for name, t in model.state_dict().items():
        ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

# End of training: replace model with EMA weights
avg_state = {name: t.to(dtype=model.state_dict()[name].dtype) for name, t in ema_state.items()}
model.load_state_dict(avg_state, strict=True)
```

**Why EMA over SWA (012)**:
- SWA averages discrete snapshots uniformly — all checkpoints weighted equally regardless of when they were taken
- EMA weights recent steps exponentially more than old ones, with a smooth decay
- With decay=0.997, the effective window is ~333 steps (half-life), naturally focusing on the well-converged late-training region
- EMA updates every step (vs SWA every 50-120 steps), capturing finer-grained weight trajectories
- The progression table from the source record shows -0.0047 BPB improvement when moving from SWA to EMA (1.1318 → 1.1271, though XSA 3→4 also contributed)

**Float32 accumulation**: The `.float()` conversion before accumulation is critical. At bf16/fp16 precision, the `alpha=0.003` additive term would be lost to rounding for most parameters, causing the EMA to stagnate. Float32 ensures every step's contribution is faithfully recorded.

## Expected impact

- Estimated ~-0.002 to -0.004 BPB over SWA (based on source record's progression)
- Same memory cost as SWA: 1x model size in float32 (~80MB for 20M params)
- Negligible compute: one multiply-add per parameter per step
- Works synergistically with longer warmdown — the EMA naturally smooths over the entire warmdown trajectory

## Builds on

- [012 Stochastic Weight Averaging](../012-swa/) — EMA is the successor technique, sharing the same Composer integration point

## Status

**Not yet runnable.** Requires Composer changes — same integration point as 012-swa but with a different averaging mode.

### Required Composer changes

1. **Extend the weight averaging feature** proposed in 012-swa to support both SWA and EMA modes:

   ```python
   class WeightAveraging(TrainingFeature):
       def __init__(self, mode: str = "ema", decay: float = 0.997,
                    swa_start_fraction: float = 0.5, swa_interval: int = 50):
           self.mode = mode  # "ema" or "swa"
           self.decay = decay
           self.swa_start_fraction = swa_start_fraction
           self.swa_interval = swa_interval
           self.state: dict[str, Tensor] | None = None
           self.swa_count = 0

       def on_training_start(self, trainer, **kwargs):
           if self.mode == "ema":
               # EMA: initialize shadow from model at step 0
               self.state = {
                   name: t.detach().float().clone()
                   for name, t in trainer.base_model.state_dict().items()
               }

       def on_step_end(self, trainer, step, scale, **kwargs):
           model = trainer.base_model
           if self.mode == "ema":
               d = self.decay
               with torch.no_grad():
                   for name, t in model.state_dict().items():
                       self.state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
           elif self.mode == "swa":
               if scale >= self.swa_start_fraction or step % self.swa_interval != 0:
                   return
               if self.state is None:
                   self.state = {name: t.detach().float().clone() for name, t in model.state_dict().items()}
                   self.swa_count = 1
               else:
                   for name, t in model.state_dict().items():
                       self.state[name].add_(t.detach().float())
                   self.swa_count += 1

       def on_training_end(self, trainer, **kwargs):
           if self.state is None:
               return
           if self.mode == "swa" and self.swa_count > 1:
               for name in self.state:
                   self.state[name].div_(self.swa_count)
           avg = {name: t.to(dtype=trainer.base_model.state_dict()[name].dtype)
                  for name, t in self.state.items()}
           trainer.base_model.load_state_dict(avg, strict=True)
   ```

   Config schema:
   ```yaml
   features:
     - WeightAveraging:
         mode: ema
         decay: 0.997
   ```

   Or for SWA mode:
   ```yaml
   features:
     - WeightAveraging:
         mode: swa
         start_fraction: 0.5
         interval: 50
   ```

2. **Integration requirements**: Same as 012-swa — needs `on_training_start`, `on_step_end`, and `on_training_end` hooks, plus access to the warmdown `scale` value for the SWA mode.
