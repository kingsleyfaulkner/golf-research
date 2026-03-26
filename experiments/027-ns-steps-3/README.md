# 027 Muon NS Steps 3

Reduces Newton-Schulz orthogonalization steps from 5 to 3 in the Muon optimizer, trading marginally less precise orthogonalization for faster per-step compute.

## Change from baseline

- Muon `ns_steps`: 5 → 3

## Source

- `reference/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`
- Part of a ternary quantization strategy — the author found 3 NS steps equivalent to 5 at convergence, saving ~190 steps worth of compute over 600s
- NeoMuon also adds RMS gradient normalization before NS (not included in this experiment — see notes)

## Context: ternary quantization strategy

This experiment originates from a ternary (BitNet b1.58) approach where STE gradient noise makes precise orthogonalization less critical. The author found that 3 Newton-Schulz iterations produce sufficiently orthogonal updates even with noisy ternary gradients. Whether this transfers to standard training (cleaner gradients) is an open question — with clean gradients, the difference between 3 and 5 NS steps may be more significant.

## How it works

The Newton-Schulz iteration approximates the polar decomposition (orthogonalization) of the gradient matrix. More steps = more precise orthogonalization. The iteration converges geometrically:

| NS Steps | Approx orthogonality error |
|----------|---------------------------|
| 3 | ~1e-3 |
| 5 | ~1e-5 |
| 10 | ~1e-10 |

At 3 steps, the update is "good enough" orthogonal for training — the per-step quality loss is marginal, but the time savings allow more total steps within the wallclock budget.

The NS computation is a meaningful fraction of Muon's step time (~10-15%), so reducing from 5 to 3 steps saves ~1-2ms/step, yielding ~15-25 extra steps over 600s.

## Expected impact

- Estimated 0 to +0.001 BPB change (marginal quality change either direction)
- ~1-2ms/step faster, yielding ~15-25 extra training steps
- Net effect depends on whether the extra steps offset the slightly less precise updates
- Lower risk in combination with aggressive momentum (0.99) which already smooths updates

## Status

**Runnable.**
