# 022 Shared Value Embedding

Reinjects token identity into attention values at specific deep layers via a shared auxiliary embedding table.

## Change from baseline

- New `ValueEmbedding` attention feature on decoder layers 9 and 10 (the deepest layers)
- Shared 128-dim embedding table maps input tokens to low-dim vectors, projected to KV dimension (256)
- Per-layer learned scale parameter (init 0.1) controls injection strength
- Added to value tensor: `v = v + ve_scale * ve_proj(ve_embed(input_ids))`
- Requires `input_ids` forwarded via `ForwardInputIds` embedding feature
- Adds ~131K parameters (1024 vocab × 128 dim table + 128 × 256 projection)

## Source

- `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (lines 562-577, 689-698, 728-732)
- From PR #374 architecture stack
- Also used in `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

## How it works

In deep transformer layers, the original token identity can be diluted by many layers of residual mixing and attention. The Value Embedding reinjects this signal directly into the attention value projection:

```python
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        self.embed = nn.Embedding(vocab_size, ve_dim)  # 1024 x 128
        self.proj = CastedLinear(ve_dim, model_dim)     # 128 x 256 (kv_dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, token_ids):
        h = self.embed(token_ids)
        return self.proj(h) * self.scale
```

The embedding is computed once per forward pass and cached, then the scaled result is added to `v` in layers 9 and 10. Each layer has its own scale parameter but shares the embedding table and projection.

## Composer's ValueEmbedding

Composer already has a `ValueEmbedding` attention feature (`composer/nn/layers/attention/features/value_embedding.py`), but it uses a different design:
- **Gated injection**: uses a learned per-head gate (`ve_gate`) with activation (e.g., Sigmoid) rather than a simple scalar scale
- **Full KV-dim embedding**: embeds directly to `num_kv_heads × head_dim` rather than a bottleneck dim with projection
- **Per-head gating**: gate output is per KV head, not a single scalar

The Composer version is more expressive but has more parameters. For this experiment, we can use Composer's `ValueEmbedding` as a reasonable approximation, though the exact config won't match the record's simpler scalar-scaled version.

## Expected impact

- Part of the PR #374 stack that improved from 1.1271 to 1.1246 BPB
- Hard to isolate from other concurrent changes (Tight SWA, DTG infrastructure)
- The concept of reinjecting token identity in deep layers is well-motivated, especially for the 11-layer U-Net where deep decoder layers are far from the original embedding

## Status

**Runnable** (with Composer's existing `ValueEmbedding` attention feature, which differs slightly from the record's implementation).

Requires `ForwardInputIds` embedding feature to make `input_ids` available in the forward context.

## Notes on Composer config

The Composer `ValueEmbedding` differs from the record's implementation:
- Composer uses gated injection (learned per-head gate with activation); record uses simple scalar scale
- Composer embeds directly to KV dim; record uses a 128-dim bottleneck with projection
- To closely match the record, Composer would need a simpler `ValueEmbedding` variant with scalar scaling and bottleneck projection. However, the Composer version is a reasonable test of the same concept.

The feature should only be applied to specific deep layers (9, 10). This requires either:
- Per-layer feature overrides in `RepeatingBlock` (same pattern as 017-xsa)
- Or splitting the decoder into two sections: layers without VE and layers with VE
