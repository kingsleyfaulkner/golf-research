# 026 Wider Model (768-dim)

Tests the width-over-depth tradeoff by increasing hidden dimension from 512 to 768 while keeping layer count moderate.

## Change from baseline

- `hidden_size`: 512 → 768
- `head_dim`: 64 → 96 (or keep 64 with 12 heads — see notes)
- `num_attention_heads`: 8 → 8 (768 / 96 = 8 heads with head_dim=96)
- `num_key_value_heads`: 4 (unchanged)
- `intermediate_size`: 1024 → 1536 (at mlp_mult=2) or 2304 (at mlp_mult=3)
- Parameters increase significantly: ~17M → ~38M at mlp_mult=2

## Source

- `reference/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`
- Part of a ternary quantization strategy where extreme compression (~1.6 bits/param) funds a much wider model
- The record's ablation over 250+ runs found 768d/10L outperforms 512d/25L: faster steps (91ms vs 127ms) yielding 6,530 vs 4,720 steps in 600s
- 4x MLP expansion at 768-dim gives hidden=3072, the record's sweet spot

## Context: ternary quantization strategy

This experiment originates from a radically different approach to the Parameter Golf challenge: instead of int6/int8 quantization on a standard-width model, the source record uses BitNet b1.58 ternary quantization ({-1, 0, +1} weights at ~1.6 bits/param) to fund a much larger model (73.7M params) that fits in 16 MB. The width increase is one component of that strategy. Testing it independently at 512→768 with standard int8 quantization explores whether width benefits transfer, though the model will likely exceed 16 MB without aggressive quantization.

## Expected impact

- Unknown at standard int8 quantization — the wider model will not fit under 16 MB without int6 or more aggressive compression
- At int6: ~38M × 0.75 bytes ≈ 28.5 MB compressed — still too large
- This experiment is primarily valuable as a data point for the width-vs-depth tradeoff, not as a contest-viable configuration
- The ternary record found width matters more than depth for per-step quality

## Notes

- `head_dim=96` with 8 heads gives 768-dim but is non-standard for RoPE (typically power-of-2)
- Alternative: 12 heads × 64 head_dim = 768-dim (more standard but different attention compute)
- Step time will increase substantially (~50%) due to wider matmuls

## Status

**Runnable** (but will exceed 16 MB at int8 quantization).
