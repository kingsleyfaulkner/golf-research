# 007 Sequence Length 2048

Doubles the training sequence length from 1024 to 2048 tokens, keeping total batch tokens constant.

## Change from baseline

- `context_length`: 1024 → 2048 in model.yaml
- `sequence_length`: follows model (via `!expr "model.context_length"`)
- Sequences per batch halved: 512 → 256 (total_batch_tokens stays at 524,288)
- Step time increases ~19% (51.89ms vs 43.54ms), reducing total steps from ~13,780 to ~11,564

## Source

- `reference/track_10min_16mb/2026-03-18_LongContextSeq2048/` (1.2058 BPB vs 1.2244 baseline)
- This was the first record-breaking submission and the single strongest training-side improvement in the early records
- Note: the source record also reduced LRs; this experiment isolates the sequence length change only

## Expected impact

- Estimated ~-0.015 to -0.019 BPB
- Longer sequences give the model more coherent context per training example
- Risk: fewer total training steps may partially offset the per-step improvement
- The baseline RoPE theta=10000 with head_dim=64 should support 2048 positions without modification

## Status

**Runnable.**
