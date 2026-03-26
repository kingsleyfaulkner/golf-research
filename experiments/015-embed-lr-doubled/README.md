# 015 Doubled Tied Embedding Learning Rate

Doubles the tied embedding learning rate from 0.05 to 0.10.

## Change from baseline

- Adam (embedding) `lr`: 0.05 → 0.10

## Source

- `reference/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/` (tied_embed_lr=0.10)
- This is the only record to use a value this high; later top records settled on 0.03-0.035

## Expected impact

- Uncertain — later records moved in the opposite direction (lower LR)
- Rationale: with tied embeddings, the weight serves double duty (input embedding + output head), receiving conflicting gradients from both roles; a higher LR may help it find a compromise faster
- Risk: the source record combined this with overtone init, 10 layers, weight decay, and other changes, so the higher LR may only work in that specific context
- Experiment 008-lower-lr found the opposite direction was beneficial at baseline, so this tests the hypothesis that embed LR should diverge from matrix LR

## Status

**Runnable.**
