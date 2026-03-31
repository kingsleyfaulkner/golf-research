# 008 Lower Learning Rates

Halves the Muon and Adam learning rates based on a systematic 6-point sweep.

## Change from baseline

- Muon `lr`: 0.04 → 0.02
- Adam (block_scalars) `lr`: 0.04 → 0.02
- Adam (embedding) `lr`: 0.05 → 0.03

## Source

- `reference/track_10min_16mb/2026-03-18_LowerLR/`
- 6-point LR sweep showed baseline matrix_lr=0.04 was ~2x too high; optimum at 0.02
- Sweep results: 0.06 (+0.016), 0.04 (baseline), 0.03 (-0.001), 0.025 (-0.004), **0.02 (-0.006)**, 0.015 (-0.005)

## Expected impact

- Estimated ~-0.001 to -0.002 BPB
- Modest improvement on its own but establishes the correct LR baseline for combining with other techniques
- Note: source record ran on H200, not H100; results may differ slightly

## Status

**Runnable.**
