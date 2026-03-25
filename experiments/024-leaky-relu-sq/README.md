# 024 LeakyReLU(0.5)-Squared Activation

Replaces `relu(x).square()` with `leaky_relu(x, 0.5).square()`, preserving negative gradient flow while maintaining the squaring inductive bias.

## Change from baseline

- MLP activation: `ReLU^2` → `LeakyReLU(0.5)^2`
- `F.leaky_relu(x, negative_slope=0.5).square()` replaces `F.relu(x).square()`
- Negative inputs produce `(0.5 * x)^2 = 0.25 * x^2` instead of 0
- Zero additional parameters

## Source

- `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (line 729)
- Single largest contributor in the ablation: **-0.0021 BPB post-TTT, -0.003 BPB pre-TTT**
- Originated by @parinzee (PR #493) and @sofiabod (PR #518)
- The current SOTA activation — every top submission after this record uses it

## How it works

Standard `relu(x)^2` kills all negative inputs (gradient = 0 for x < 0). With `leaky_relu(x, 0.5)^2`:

```python
# Before (baseline):
x = F.relu(x).square()        # x<0 → 0, x>0 → x²

# After:
x = F.leaky_relu(x, 0.5).square()  # x<0 → 0.25x², x>0 → x²
```

The negative slope of 0.5 means:
- Positive inputs: unchanged (`x^2`)
- Negative inputs: attenuated but not killed (`(0.5x)^2 = 0.25x^2`)
- Gradient for negative inputs: `0.5 * x` (vs 0 with ReLU)

This prevents the "dead neuron" problem where neurons with persistent negative pre-activations stop learning entirely. In a small model (17-22M params) training for only 10 minutes, every parameter must contribute — dead neurons are a significant capacity waste.

The choice of 0.5 (not the standard 0.01 default) is aggressive: negative activations contribute 25% of the squared magnitude rather than 0.01%. This was found empirically to be the best value.

## Expected impact

- **-0.003 BPB** (the ablation clearly isolates this as a one-line change)
- This is one of the highest-impact single changes in the entire competition
- Zero compute cost (LeakyReLU is the same speed as ReLU on modern GPUs)
- Zero parameter cost

## Status

**Not yet runnable.** Requires Composer changes to support compound activations with custom parameters.

### Required Composer changes

Composer's activation registry (`composer/activations.py`) maps string names to `nn.Module` classes:

```python
_ACTIVATION_MODULES = {
    "ReLU": nn.ReLU,
    "ReLU^2": ReLU2,
    "LeakyReLU": nn.LeakyReLU,  # exists but defaults to negative_slope=0.01
    ...
}
```

The issues are:
1. `LeakyReLU` is registered but with default `negative_slope=0.01` — the record needs 0.5
2. There's no `LeakyReLU^2` compound activation (leaky relu followed by squaring)

**Option A — New compound activation class** (minimal change):

Add to `composer/activations.py`:
```python
class LeakyReLU2(nn.Module):
    def __init__(self, negative_slope: float = 0.5):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope).square()

_ACTIVATION_MODULES["LeakyReLU^2"] = LeakyReLU2
```

Config:
```yaml
mlp:
  FFN:
    activation: LeakyReLU^2
```

**Option B — Parameterized activation config** (more general):

Support activation parameters in YAML:
```yaml
mlp:
  FFN:
    activation:
      LeakyReLU^2:
        negative_slope: 0.5
```

This requires changing `TorchActivation` validation to accept dict configs with parameters, not just string names.

**Option C — Custom module registration** (most flexible):

Allow users to register custom activations via config:
```yaml
mlp:
  FFN:
    activation:
      Compose:
        - LeakyReLU:
            negative_slope: 0.5
        - Square
```

This is the most general but overkill for this specific case.

**Recommendation**: Option A is simplest and sufficient. Add `LeakyReLU^2` with configurable `negative_slope` (default 0.5) to the activation registry.
