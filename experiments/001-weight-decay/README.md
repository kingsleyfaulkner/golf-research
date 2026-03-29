# 001 Weight Decay

Adds weight decay of 0.04 to all optimizer groups (Muon and Adam).

## Change from baseline

- `weight_decay`: 0.0 → 0.04 on Muon (matrix params)
- `weight_decay`: 0.0 → 0.04 on Adam (embeddings, block scalars)

## Source

Both top Parameter Golf submissions use WD=0.04 across all optimizer groups:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50` (1st place, 1.1428 BPB)
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (2nd place, 1.1458 BPB)

## Expected impact

- Regularizes weight magnitudes, directly improving post-training quantization quality
- Ablation from 1st place submission shows WD=0.04 contributes ~0.0001 BPB directly, but the indirect benefit to quantization is larger
