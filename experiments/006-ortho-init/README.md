# 006 Orthogonal Initialization

Orthogonal initialization for large weight matrices with muP-scaled output projections.

## Change from baseline

- Orthogonal init for weight matrices with both dimensions >= 64
- Output projections (attn.proj, mlp.proj) scaled by `1/sqrt(2 * num_layers)`
- All other initialization unchanged (embeddings still N(0, 0.005))

## Source

Both top submissions use orthogonal init:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

## Expected impact

- Accelerates early convergence, which is critical in the 10-minute training budget
- Estimated ~0.001-0.002 BPB improvement

## Status

**Not yet runnable.** Requires Composer changes to support `init_method: orthogonal` in the architecture config. The baseline uses `init_method: normal` for embeddings and default PyTorch init for other layers. Orthogonal init needs to be added as an option in Composer's `NNModule.reset_parameters()` or as a new feature.
