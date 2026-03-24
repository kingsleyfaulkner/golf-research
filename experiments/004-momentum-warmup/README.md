# 004 Momentum Warmup

Widens the Muon momentum warmup ramp: 0.92→0.99 over 1500 steps (vs baseline 0.85→0.95 over 500).

## Change from baseline

- `momentum`: 0.95 → 0.99 (final value)
- Momentum schedule `initial`: 0.85 → 0.92
- Momentum schedule `final`: 0.95 → 0.99
- Momentum schedule `warmup_steps`: 500 → 1500

## Source

Both top submissions use momentum 0.92→0.99 over 1500 steps:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

## Expected impact

- Higher final momentum (0.99) provides stronger smoothing in later training
- Longer warmup (1500 steps) prevents instability from high momentum early on
- Extends the effective convergence window within the 10-minute budget
