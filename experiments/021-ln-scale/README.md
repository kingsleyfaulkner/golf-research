# 021 LN Scale (Layer-Depth RMSNorm Scaling)

Scales RMSNorm outputs by `1/sqrt(layer_idx + 1)` per layer, progressively dampening deeper layers to stabilize gradient flow.

## Change from baseline

- After each RMSNorm (before attention and before MLP), multiply output by `1/sqrt(layer_idx + 1)`
- Layer 0: scale = 1.000 (unchanged)
- Layer 5: scale = 0.408
- Layer 10: scale = 0.302
- Zero additional parameters — the scale factor is a fixed constant per layer
- Applied to both `attn_norm` and `mlp_norm` outputs in each block

## Source

- `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (lines 739, 744-747)
- Contributed -0.0023 BPB together with Partial RoPE (combined effect)
- Also used in:
  - `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

## How it works

A one-liner in each block's `__init__`:
```python
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
```

Applied in `forward`:
```python
s = self.ln_scale_factor
attn_out = self.attn(self.attn_norm(x) * s)
x = x + self.attn_scale * attn_out
x = x + self.mlp_scale * self.mlp(self.mlp_norm(x) * s)
```

Scale values for an 11-layer model:

| Layer | 1/sqrt(i+1) |
|-------|-------------|
| 0 | 1.000 |
| 1 | 0.707 |
| 2 | 0.577 |
| 3 | 0.500 |
| 4 | 0.447 |
| 5 | 0.408 |
| 6 | 0.378 |
| 7 | 0.354 |
| 8 | 0.333 |
| 9 | 0.316 |
| 10 | 0.302 |

**Intuition**: In deep transformers, the residual stream accumulates signal from every layer. Without dampening, the norm of hidden states grows with depth, causing deeper layers to operate at different scales than early layers. LN Scale counteracts this by progressively reducing the input magnitude to attention and MLP, keeping all layers operating at similar scales. This is related to the depth scaling in muP and nGPT approaches.

## Expected impact

- Estimated ~-0.001 BPB (combined with Partial RoPE the pair contributes -0.0023 BPB)
- Zero parameter cost, negligible compute (one scalar multiply per norm per layer)
- Particularly beneficial for deeper models (11+ layers) where signal accumulation is larger
- May interact well with orthogonal init (006) since both address gradient flow

## Status

**Not yet runnable.** Requires Composer changes to support per-layer RMSNorm scaling.

### Required Composer changes

1. **Per-layer scale factor on `RMSNorm`** (or on the block)

   **Option A — RMSNorm parameter**: Add an optional `depth_scale` factor to `RMSNorm`:
   ```python
   class RMSNorm(nn.Module):
       def __init__(self, eps=None, depth_scale: float = 1.0):
           self.depth_scale = depth_scale
       def forward(self, x):
           return F.rms_norm(x, (x.size(-1),)) * self.depth_scale
   ```

   The `RepeatingBlock` would need to pass `layer_idx` and `num_layers` so the scale can be computed at build time: `depth_scale = 1.0 / sqrt(layer_idx + 1)`.

   Config:
   ```yaml
   ln_1:
     RMSNorm:
       depth_scale: !expr "1.0 / (self.layer_idx + 1) ** 0.5"
   ```

   **Option B — Block-level feature**: Add a `DepthScale` feature to the architecture config that applies `1/sqrt(layer_idx+1)` to all norm outputs within a block. This avoids modifying `RMSNorm` itself:
   ```yaml
   block:
     Residual:
       features:
         - DepthScale:
             formula: inv_sqrt  # 1/sqrt(layer_idx + 1)
   ```

2. **Layer index propagation**: Same requirement as 014-phase-transition-resid — the `RepeatingBlock` must pass `layer_idx` to child modules at build time. This is a shared dependency across multiple experiments (014, 017-xsa, 021).

3. **Alternative — Post-build hook**: A model-level feature that iterates over blocks after construction and sets the scale factor:
   ```python
   for i, block in enumerate(model.blocks):
       for norm in [block.attn_norm, block.mlp_norm]:
           norm.depth_scale = 1.0 / math.sqrt(i + 1)
   ```
   This is simpler but less general.
