# 028 Polynomial Softcap

Replaces tanh-based logit softcapping with a degree-5 polynomial approximation, enabling faster computation and a different saturation profile.

## Change from baseline

- SoftCap activation: `Tanh` → polynomial degree 5
- SoftCap limit: 30.0 → 10.0 (tighter cap with polynomial)
- Optional Z-loss regularization (weight 1e-4) to prevent logit collapse

## Source

- `reference/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`
- Part of a ternary quantization strategy — the polynomial softcap was found to work better with ternary weights and the lower cap of 10.0

## Context: ternary quantization strategy

This experiment originates from a ternary (BitNet b1.58) approach. The polynomial softcap uses `cap * poly5(logits / cap)` where `poly5` is a degree-5 polynomial approximation of tanh. This is faster than tanh (no transcendental function) and provides a different gradient profile near saturation. The tighter cap (10.0 vs 30.0) may be specific to the ternary approach where logit magnitudes are naturally smaller.

The Z-loss regularization (`z_loss_weight * log(sum(exp(logits)))^2`) penalizes large logit norms, preventing the model from producing overconfident predictions that saturate the softcap. This is standard practice in large language models (PaLM, Gemini) but not used in the baseline.

## How it works

Standard tanh softcap:
```python
logits = limit * tanh(logits / limit)  # limit=30.0
```

Polynomial softcap (degree 5):
```python
x = logits / cap  # cap=10.0
# Polynomial approximation of tanh, clamped to [-1, 1]
p = x * (1 + x^2 * (coeff_2 + x^2 * coeff_4))  # Padé-style
logits = cap * clamp(p, -1, 1)
```

Z-loss:
```python
log_z = torch.logsumexp(logits, dim=-1)
z_loss = z_loss_weight * (log_z ** 2).mean()
total_loss = ce_loss + z_loss
```

## Expected impact

- Unknown in isolation — the record combined this with many other changes
- The polynomial is slightly faster than tanh on GPU (avoids exp/log)
- The tighter cap (10.0 vs 30.0) may interact with learning rate and gradient scaling
- Z-loss could improve training stability independently of the softcap choice

## Status

**Not yet runnable.** Requires Composer changes to support polynomial softcap activation.

### Required Composer changes

1. **New activation for SoftCap**: The existing `SoftCapConfig` accepts any `TorchActivation` (Tanh, Sigmoid, etc.). A polynomial softcap needs a new activation module:

   ```python
   class PolynomialTanh(nn.Module):
       def __init__(self, degree: int = 5):
           super().__init__()
           self.degree = degree

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Degree-5 polynomial approximation of tanh
           x2 = x * x
           if self.degree >= 5:
               return x * (1 + x2 * (-1/3 + x2 * 2/15)).clamp(-1, 1)
           elif self.degree >= 3:
               return x * (1 - x2 / 3).clamp(-1, 1)
           return x.clamp(-1, 1)
   ```

   Register in `composer/activations.py`:
   ```python
   _ACTIVATION_MODULES["PolynomialTanh"] = PolynomialTanh
   ```

   Config:
   ```yaml
   features:
     - SoftCap:
         limit: 10.0
         activation: PolynomialTanh
   ```

   Note: The exact polynomial coefficients used in the record may differ from the standard Taylor expansion. The author may have used optimized coefficients for the [-10, 10] range.

2. **Z-loss as a head feature** (optional, separate from polynomial softcap):

   ```yaml
   features:
     - SoftCap:
         limit: 10.0
         activation: PolynomialTanh
     - ZLoss:
         weight: 1.0e-4
   ```
