# 013 Overtone Spectral Embedding Initialization

SVD-based power-law spectrum shaping on the tied embedding, giving it a structured spectral profile instead of random noise.

## Change from baseline

- After normal `N(0, 0.005)` embedding initialization, apply SVD and reshape singular values to power-law decay
- Formula: `U, S, V = svd(tok_emb.weight)` then `S_target[k] = S[0] * k^{-0.5}` then `tok_emb.weight = (U * S_target) @ V`
- Preserves the random directions (U, V) but imposes a structured magnitude spectrum
- The result has a few dominant dimensions and many diminishing ones, matching natural language statistics

## Source

- `reference/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/` (diff lines 182-186)
- Unique to this record — not adopted by later submissions

## How it works

The standard normal init produces an embedding with a flat singular value spectrum (all dimensions equally important). The overtone init reshapes this to a power-law `S_k ~ k^{-0.5}`:

```python
with torch.no_grad():
    U, S, V = torch.linalg.svd(tok_emb.weight.data, full_matrices=False)
    target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
    tok_emb.weight.data = (U * target_S[None, :]) @ V
```

The analogy is guitar harmonics: the fundamental (first singular value) is loudest, and overtones decay as a power law. For embeddings, this means:
- The first few dimensions capture broad semantic features (common patterns)
- Higher dimensions capture increasingly specific features (rare patterns)
- This matches the power-law distribution of word/token frequencies in natural language

## Expected impact

- Unknown in isolation — always combined with other changes in the source record
- Theoretically provides a better starting point for training, potentially accelerating early convergence
- Risk: the power-law shape may not be optimal; the model would learn its own spectrum eventually anyway
- The 0.5 exponent is a reasonable default (Zipf's law has exponent ~1.0, but singular values scale differently)

## Status

**Not yet runnable.** Requires Composer changes to add overtone initialization.

### Required Composer changes

1. **New `InitMethod` value: `OVERTONE`** in `composer/nn/init.py`

   Add to the `InitMethod` enum:
   ```python
   OVERTONE = "overtone"
   """SVD power-law spectrum shaping. After normal init, reshapes singular values to S_k ~ S_0 * k^{-exponent}."""
   ```

2. **Update `init_embedding()`** in `composer/nn/init.py`

   Add a branch for `InitMethod.OVERTONE`:
   ```python
   elif init_method == InitMethod.OVERTONE:
       # First do normal init
       apply_init(nn.init.trunc_normal_, module.weight, mean=0.0, std=std, a=-3*std, b=3*std, generator=generator)
       # Then reshape spectrum
       with torch.no_grad():
           U, S, V = torch.linalg.svd(module.weight.data, full_matrices=False)
           k = torch.arange(1, S.shape[0] + 1, dtype=S.dtype, device=S.device)
           target_S = S[0] * k.pow(-0.5)
           module.weight.data = (U * target_S.unsqueeze(0)) @ V
   ```

3. **Optional: configurable exponent**

   The power-law exponent (0.5) could be made configurable. The `TokenEmbedding` config already supports `init_std`; a new `init_exponent` field could control the decay rate:
   ```yaml
   embedding:
     TokenEmbedding:
       init_method: overtone
       init_std: 0.005
       init_exponent: 0.5  # power-law decay exponent
   ```

4. **SVD considerations**: `torch.linalg.svd` on a 1024×512 matrix is fast (<1ms) and runs once at init, so no performance concern. However, for very large vocabularies this could be slower.
