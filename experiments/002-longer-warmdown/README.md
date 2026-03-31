# 002 Longer Warmdown

Extends the learning rate warmdown from 1200 to 3000 steps.

## Change from baseline

- `warmdown_steps`: 1200 → 3000

## Source

Both top submissions use warmdown of 3000 steps:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

## Expected impact

- More gradual LR decay allows better weight stabilization before final checkpoint
- Works synergistically with SWA (not yet implemented) — the longer tail provides more well-converged checkpoints to average
- Estimated ~0.0005 BPB improvement
