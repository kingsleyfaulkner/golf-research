# 020 Partial RoPE

Applies rotary position embeddings to only a fraction of the head dimensions (16/64), leaving the remaining dimensions free to learn position-invariant attention patterns.

## Change from baseline

- RoPE applied to first 16 of 64 head dimensions (25%)
- Remaining 48 dimensions attend without any positional bias
- `apply_rotary_emb` splits input: `x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]`
- Only the `x_rope` portion gets cos/sin rotation; `x_pass` is concatenated back unchanged
- Inverse frequency table sized for `rope_dims` (16) not `head_dim` (64)
- NTK-aware scaling also adjusted to use `rope_dims` for the dimension ratio
- Zero additional parameters

## Source

- `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (lines 561-606, ROPE_DIMS=16)
- Contributed -0.0023 BPB together with LN Scale (combined effect, hard to separate)
- Also used in:
  - `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (rope_dims=16)
  - `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (rope_dims=16)

## How it works

Standard RoPE rotates all head dimensions, encoding absolute position into every dimension of Q and K. This forces all attention patterns to be position-dependent. With Partial RoPE, only a small fraction of dimensions carry positional information:

```python
def apply_rotary_emb(x, cos, sin):
    rd = cos.size(-1) * 2  # = rope_dims (e.g., 16)
    if rd < x.size(-1):
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    # Full RoPE fallback (rope_dims == head_dim)
    ...
```

The Rotary module computes inv_freq for only `rope_dims` dimensions:
```python
rd = self.rope_dims  # 16
inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2) / rd))
```

**Why this works**: In a 64-dim head, only a few dimensions are needed to encode position. The remaining 48 dimensions can learn content-based (position-invariant) attention patterns — "what matches" rather than "what's nearby". This is especially valuable for:
- Long-range dependencies (no position decay)
- Pattern matching across different positions
- Reducing the effective position frequency space (less interference between position and content signals)

## Expected impact

- Estimated ~-0.001 to -0.002 BPB (combined with LN Scale, the pair contributes -0.0023 BPB)
- Zero parameter cost, negligible compute change
- Risk: with only 25% of dims carrying position, the model has less positional resolution — but 16 dims provide 8 frequency bands, which is more than enough for 2048-token contexts

## Status

**Not yet runnable.** Requires Composer changes to support partial RoPE dimensions.

### Required Composer changes

1. **New parameter on `RoPEConfig`**: `rope_dims` (or `partial_dims`)

   Add to `RoPEConfig` in `composer/nn/layers/attention/features/rotary_embedding.py`:
   ```python
   class RoPEConfig(RotaryEmbeddingConfig, ABC):
       rope_dims: int | None = None  # None = full head_dim (default behavior)
   ```

   The `RoPE` module constructor would use `rope_dims or head_dim` when computing `inv_freq`, and store `rope_dims` for the split in `forward()`.

2. **Modified `apply_rotary_emb`** (or equivalent in `RoPE.forward`):

   When `rope_dims < head_dim`, split Q and K into rotated and pass-through portions:
   ```python
   if self.rope_dims < self.head_dim:
       q_rope, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
       k_rope, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
       q_rope = apply_rotary(q_rope, cos, sin)
       k_rope = apply_rotary(k_rope, cos, sin)
       q = torch.cat([q_rope, q_pass], dim=-1)
       k = torch.cat([k_rope, k_pass], dim=-1)
   ```

3. **Config example**:
   ```yaml
   features:
     - RoPE:
         rope_theta: 10000
         rope_dims: 16  # apply RoPE to first 16 of 64 head dims
   ```

4. **NTK scaling adjustment**: When using NTK-aware scaling (or YARN), the dimension ratio should use `rope_dims` not `head_dim` for computing the scaled base frequency.
