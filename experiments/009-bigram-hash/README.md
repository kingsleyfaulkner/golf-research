# 009 BigramHash Embedding

Adds a hash-table embedding that injects token-pair (bigram) context before the transformer blocks.

## Change from baseline

- New `BigramHash` module inserted after `TokenEmbedding`, before RMSNorm
- `num_buckets`: 4096
- `hash_dim`: 128 (internal embedding dimension)
- Projected from 128 → 512 (model_dim) via a zero-initialized linear layer
- Hash function: `(prev_id * 92821 + cur_id) % 4096`
- Table initialized with `N(0, 0.01)`
- Adds ~524K parameters (4096×128 table + 128×512 projection)

## Source

- `reference/track_10min_16mb/2026-03-19_int6_STE QAT_ MLP_bigram _U_Net/` (embedded train.log source, lines 789-806)
- Also used in:
  - `reference/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` (4096 buckets)
  - `reference/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` (10240 buckets)

## How it works

The module maps each consecutive token pair `(prev_token, cur_token)` to a learned embedding via a simple hash:

```python
h = ((prev_ids * 92821 + cur_ids) % num_buckets)
output = proj(table(h))  # [batch, seq, model_dim]
```

The output is added to the token embedding: `x = tok_emb(ids) + bigram_hash(ids)`. The projection starts at zero so the module initially has no effect and learns to contribute during training.

This gives the model cheap character-pair / bigram context without attention — useful because the 1024-token vocabulary means many tokens are sub-word, and bigram patterns (like common letter pairs) carry significant signal.

## Expected impact

- Estimated ~-0.001 to -0.003 BPB
- Always used alongside other changes in records, so isolated contribution is uncertain
- The 10L Int5MLP record scaled buckets to 10240 for additional gain
- Negligible compute cost; the hash + lookup + projection is tiny vs attention

## Status

**Not yet runnable.** Requires Composer changes to add a `BigramHash` layer module.

### Required Composer changes

1. **New module: `BigramHash`** in `composer/nn/layers/` (or `composer/nn/embedding/`)

   A new configurable layer that:
   - Takes `input_ids` (not hidden states) as input — it needs the raw token IDs to compute the hash
   - Contains an `nn.Embedding(num_buckets, hash_dim)` hash table
   - Contains a linear projection `hash_dim → hidden_size` (zero-initialized)
   - Computes the hash: `(prev_ids * hash_prime + cur_ids) % num_buckets`
   - Returns a tensor of shape `[batch, seq, hidden_size]` to be added to the token embedding

   Config schema:
   ```yaml
   BigramHash:
     num_buckets: 4096      # number of hash buckets
     hash_dim: 128           # internal embedding dimension
     hash_prime: 92821       # prime multiplier for hash function
     init_std: 0.01          # table initialization std
     zero_init_proj: true    # zero-initialize the projection
   ```

2. **Integration point**: The `BigramHash` output needs to be added to the `TokenEmbedding` output. This could be implemented as:
   - An embedding feature (like `PositionalEmbedding`) that hooks into the embedding pipeline
   - Or a standalone layer in the block sequence that receives `input_ids` via `ForwardContext`

   The cleanest approach is likely an **embedding feature** since it needs `input_ids` (available in the embedding stage) and produces an additive signal to the token embedding. The existing `ForwardInputIds` feature already pushes `input_ids` to the forward context, so a `BigramHash` feature could consume that.

3. **Forward context approach** (alternative): Push `input_ids` via `ForwardInputIds`, then add a `BigramHash` layer in the block sequence that reads from context. This avoids modifying the embedding module but requires the layer to pull `input_ids` from the forward context rather than receiving them as a direct input.
