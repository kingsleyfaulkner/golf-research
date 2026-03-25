# 011 11 Transformer Layers

Increases model depth from 9 to 11 layers (5 encoder + 6 decoder).

## Change from baseline

- `num_layers`: 9 → 11
- `num_encoder_layers`: 4 → 5 (computed: `11 // 2`)
- `num_decoder_layers`: 5 → 6 (computed: `11 - 5`)
- Adds ~2.6M parameters (from ~17.1M to ~19.7M at mlp_mult=2)
- Step time increases ~25% (~48ms → ~60ms), reducing total steps from ~12,400 to ~10,000

## Source

- `reference/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` (11L + MLP 3x, 1.1502 BPB)
- Also used in:
  - `reference/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` (11L, 1.1307 BPB)
  - `reference/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` (11L, 1.1271 BPB)
  - `reference/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (11L, 1.1248 BPB)
  - `reference/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` (11L, 1.1233 BPB)
- Every record from 2026-03-20 onward uses 11 layers — it became the standard depth

## Expected impact

- Estimated ~-0.002 to -0.005 BPB (always combined with other changes in records, hard to isolate)
- More depth adds representational capacity, especially for the encoder-decoder skip connection architecture
- Risk: with int8 quantization (baseline), 11 layers may not fit under 16 MB — int6 quantization is needed to fund the extra parameters
- At mlp_mult=2 (baseline), the model should still fit under 16 MB with int8 (~17.1M + 2.6M = 19.7M params × 1 byte ≈ 18.8 MB compressed)

## Status

**Runnable.**
