# 017 Exclusive Self Attention (XSA)

Subtracts each token's self-value projection from the attention output on the deepest N layers, forcing those layers to learn from context rather than self-reference.

## Change from baseline

- New `ExclusiveSelfAttention` feature on `CausalGroupedQueryAttention`
- Applied to the last N layers only (default N=3-4, configurable via `xsa_last_n`)
- After the standard attention computation, subtracts the self-value component: `y = y - dot(y, v_norm) * v_norm`
- Uses GQA-aware reshape to avoid `repeat_interleave` memory overhead
- No additional parameters — purely a modification of the attention output

## Source

- `reference/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` (lines 625-636, 649-652)
- Also used in:
  - `reference/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` (XSA on last 4 layers, 1.1271 BPB)
  - `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (XSA 4, 1.1248 BPB)
  - `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (XSA 4, 1.1233 BPB)
- Every SOTA record from 2026-03-20 onward uses XSA — it became standard alongside 11 layers

## How it works

Standard self-attention allows each token to attend to itself, which in deep layers becomes a "shortcut" — the model copies its own representation rather than learning from context. XSA removes this shortcut by projecting out the self-value component from the attention output.

The efficient GQA-aware implementation:

```python
def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
    """y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    # Reshape y into KV head groups — free view, no memory alloc
    y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
    vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D]
    # Project out self-value component per KV head group
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
```

Key design choices:
- **Partial application**: Only the deepest N layers use XSA. Early layers benefit from self-attention (building initial representations), while deep layers suffer from self-attention bias (copying rather than reasoning). The sweet spot is N=3-4 for an 11-layer model.
- **GQA-aware reshape**: With 8 query heads and 4 KV heads (group_size=2), standard XSA would need `repeat_interleave` to expand v from 4 to 8 heads. Instead, reshaping y into `[B, T, 4, 2, D]` groups allows broadcasting v as `[B, T, 4, 1, D]` — a free view with zero memory allocation.
- **Per KV-head projection**: The subtraction operates within each KV head group, so query heads sharing the same KV head have the same self-value component removed.

## Expected impact

- Estimated ~-0.003 to -0.005 BPB based on progression from non-XSA to XSA records
- Overhead: ~2ms/step for 3-4 XSA layers (the efficient implementation minimizes this)
- Every SOTA submission from 2026-03-20 uses XSA, suggesting it's a consistent win

## Status

**Not yet runnable.** Requires Composer changes to add XSA as an attention feature.

### Required Composer changes

1. **New attention feature: `ExclusiveSelfAttention`** in `composer/nn/layers/attention/features/`

   An `AttentionFeature` that post-processes the attention output to remove the self-value component.

   ```python
   @register_module_config(ModulePath.ATTENTION_FEATURE)
   class ExclusiveSelfAttentionConfig(AttentionFeatureConfig):
       type: Literal["ExclusiveSelfAttention"] = "ExclusiveSelfAttention"

       def build(self, device="cpu", **kwargs) -> "ExclusiveSelfAttention":
           return ExclusiveSelfAttention()

   class ExclusiveSelfAttention(AttentionFeature):
       _is_post_attention: bool = True  # flag to apply after attention, not before

       def post_attention(
           self, y: torch.Tensor, v: torch.Tensor, **kwargs
       ) -> torch.Tensor:
           B, T, H, D = y.shape
           Hkv = v.size(-2)
           group = H // Hkv
           y_g = y.reshape(B, T, Hkv, group, D)
           vn = F.normalize(v, dim=-1).unsqueeze(-2)
           proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
           return (y_g - proj).reshape(B, T, H, D)
   ```

2. **Post-attention hook in attention module**

   The current `AttentionFeature` interface operates on Q/K/V *before* attention. XSA needs a post-attention hook that receives the attention output `y` and the value tensor `v`. The `CausalGroupedQueryAttention` module needs to:
   - Retain `v` after the attention computation
   - Call `post_attention(y, v)` on any features with `_is_post_attention = True`
   - This is a new extension point in the attention feature protocol

3. **Partial application across layers**

   XSA should only apply to specific layers. Two approaches:

   **Option A — Per-layer feature override**: The `RepeatingBlock` could support per-layer feature overrides via a `layer_range` parameter:
   ```yaml
   attn:
     CausalGroupedQueryAttention:
       features:
         - QKNorm
         - RoPE:
             rope_theta: 10000
         - AttentionScale:
             q: 1.5
         - ExclusiveSelfAttention:
             last_n_layers: 4  # only active on last 4 layers
   ```
   The feature would need `layer_idx` and `num_layers` at build time (same pattern as 014-phase-transition-resid).

   **Option B — Separate decoder block definition**: Define the decoder's last N layers with a different block config that includes XSA. This uses existing Composer capabilities but requires splitting the decoder into two `RepeatingBlock` sections.

4. **torch.compile compatibility**: All ops (`reshape`, `normalize`, `sum`, `mul`, `sub`) are standard and compile-friendly with `fullgraph=True`. The conditional `if self.use_xsa` per-block is resolved at compile time since it's set once during `__init__`.
