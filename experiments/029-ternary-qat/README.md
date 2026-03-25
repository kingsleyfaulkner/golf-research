# 029 Ternary QAT (BitNet b1.58)

Constrains weights to {-1, 0, +1} during training via STE, enabling extreme compression (~1.6 bits/param) that funds much larger models within the 16 MB budget.

## Change from baseline

- All large 2D weight matrices use `TernaryLinear` instead of standard `nn.Linear`
- During forward pass: weights quantized to {-1, 0, +1} with per-group (128) absmean scaling
- STE: gradients flow through to full-precision master weights via `w + (q * scale - w).detach()`
- Post-training: ternary weights packed as base-3 (5 trits per byte) or bitmask pairs
- Compression: LZMA preset=9 (39% better than zstd for ternary data)
- Non-ternary params (embeddings, norms, scales) stored as FP8 e4m3

## Source

- `reference/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`
- Author: Ciprian-Florin Ifrim (@CiprianFlorin-Ifrim), 250+ experimental runs
- Achieves 73.7M trainable parameters in 15.99 MB (vs ~20M params at int6 in 15.5 MB)
- Competitive BPB (1.1570) despite the extreme quantization, demonstrating that scale can compensate for precision

## Context: ternary quantization strategy

This is the core innovation of the ternary approach. Instead of training a normal model and post-training quantizing to int6/int8, this approach trains with ternary constraints from step 0. The key insight is that at ~1.6 bits/param (vs 6 bits for int6), you can fit 3.75× more parameters in the same byte budget, enabling a fundamentally wider/deeper model.

## How it works

### Forward pass (TernaryLinear)
```python
def forward(self, x):
    w = self.weight  # full-precision master weights
    g = self.group_size  # 128

    # Per-group absmean scaling
    w_g = w.reshape(-1, g)
    scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)

    # Quantize to {-1, 0, +1}
    q = (w_g / scale).round().clamp(-1, 1)

    # STE: forward sees ternary, backward sees full-precision
    w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()

    return F.linear(x, w_ternary, self.bias)
```

### Serialization (base-3 packing)
Ternary values {-1, 0, +1} map to {0, 1, 2}. Five trits pack into one byte (3^5 = 243 < 256):
```python
packed = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4  # 5 trits → 1 byte
```
This yields ~1.6 bits/param. With LZMA compression, the actual rate is even lower for sparse ternary matrices.

### Shrinkage correction
During dequantization, the ternary weights have a different absmean than the original full-precision weights (because rounding to {-1,0,+1} changes the distribution). The fix normalizes by the quantized absmean:
```python
q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
reconstructed = q * (scale / q_absmean)
```

### Key differences from int6 QAT (010)
| | Int6 QAT (010) | Ternary QAT (029) |
|---|---|---|
| Quantization levels | 63 ([-31, 31]) | 3 ({-1, 0, +1}) |
| Bits per param | ~6 | ~1.6 |
| Scaling | per-row | per-group (128) |
| Model capacity | ~20M params in 16 MB | ~74M params in 16 MB |
| Training impact | Minor noise | Significant constraint |
| Requires width increase | No | Yes (compensates precision loss) |

## Expected impact

- Enables 3-4× more parameters within the 16 MB budget
- The extreme quantization degrades per-parameter quality, but the scale advantage compensates
- The record achieves 1.1570 BPB with 73.7M ternary params vs 1.1233 BPB with ~25M int6 params — competitive but not SOTA
- Requires a co-designed architecture (wider model, factored embeddings, adjusted hyperparameters)

## Status

**Not yet runnable.** Requires significant Composer changes for ternary training support.

### Required Composer changes

1. **New layer: `TernaryLinear`** in `composer/nn/layers/`

   A new linear layer that applies per-group ternary STE during forward:
   ```python
   class TernaryLinear(nn.Linear):
       def __init__(self, in_features, out_features, bias=False, group_size=128):
           super().__init__(in_features, out_features, bias=bias)
           self.group_size = group_size

       def forward(self, x):
           w = self.weight
           g = self.group_size
           w_g = w.reshape(-1, g)
           scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
           q = (w_g / scale).round().clamp(-1, 1)
           w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
           return F.linear(x, w_ternary.to(x.dtype), self.bias)
   ```

   Config integration — either as a model-level feature that replaces `nn.Linear` instances, or as an explicit layer type in model.yaml.

2. **Ternary serialization** in `scripts/quant.py`

   New quantization scheme: `ternary` with base-3 packing and LZMA compression. The `quant.py` script needs:
   - Ternary weight detection (all values in {-1, 0, +1} after rounding)
   - Base-3 packing (5 trits per byte)
   - Bitmask alternative (for sparse ternary matrices)
   - LZMA compression option
   - Shrinkage-corrected dequantization

3. **Architecture co-design**: Ternary training works best with wider models and factored embeddings. The experiment should ideally test ternary QAT together with the wider-768 architecture (026), not independently at 512-dim where the precision loss cannot be compensated by scale.

4. **Optimizer considerations**: Standard weight decay is incompatible with ternary weights (decaying toward zero conflicts with the ternary constraint). Muon weight decay should be disabled for ternary layers.
