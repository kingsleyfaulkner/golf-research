# 005 MLP 3x Expansion

Increases the FFN intermediate size from 2x to 3x hidden dimension (1024 → 1536).

## Change from baseline

- `mlp_mult`: 2 → 3 in model.yaml
- `intermediate_size`: 1024 → 1536 (computed via `mlp_mult * hidden_size`)
- Adds ~2.4M parameters (from ~17M to ~19.4M)

## Source

Both top submissions use 3x MLP expansion, funded by aggressive quantization:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50` (3x MLP + 10 layers)
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (3x MLP + 9 layers)

## Expected impact

- Single largest architectural contributor: ~0.003 BPB improvement
- The wider FFN increases model capacity where it matters most (feed-forward layers dominate parameter count)
- Requires int6 quantization to fit under 16 MB — at int6 the baseline uses 12.3 MB, leaving headroom for ~3.7 MB of extra parameters

## Notes

- The larger model will train slightly slower per step but should converge to a lower loss within the 10-minute budget
- Must be evaluated with int6 quantization to assess contest viability
