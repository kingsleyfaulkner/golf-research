# 018 FlashAttention 3

Replaces PyTorch's `scaled_dot_product_attention` with direct FlashAttention 3 calls optimized for Hopper (H100/H200) GPUs.

## Change from baseline

- Import `flash_attn_func` from `flash_attn_interface` (FA3 package)
- Replace `torch.nn.functional.scaled_dot_product_attention` with `flash_attn_func(q, k, v, causal=True)`
- Attention tensor layout changes from `[B, H, T, D]` to `[B, T, H, D]` (FA3 convention)
- Corresponding adjustments to RoPE cos/sin cache layout and q_gain broadcast dimensions

## Source

- `reference/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` (line 35, 649)
- Also used by all subsequent SOTA records (2026-03-20 onward)
- FA3 is Hopper-optimized (warp-specialized, FP8-capable, async TMA) vs FA2's generic CUDA approach

## How it works

FlashAttention 3 uses Hopper-specific hardware features:
- **Warp specialization**: Splits producer (TMA loads) and consumer (tensor core matmuls) across warp groups
- **Asynchronous TMA**: Overlaps global memory loads with computation
- **Native FP8 support**: Can run attention in FP8 precision (not used here, but available)
- **Ping-pong scheduling**: Alternates between softmax and matmul stages for better utilization

The API is simpler than PyTorch SDP — just `flash_attn_func(q, k, v, causal=True)` — but requires the `[B, T, H, D]` tensor layout (heads-last) rather than PyTorch's `[B, H, T, D]` (heads-second).

The baseline uses PyTorch's `scaled_dot_product_attention` with `flash_sdp=True` in the CUDABackend, which dispatches to FA2 via cuDNN. FA3 bypasses this entirely with a direct call.

## Expected impact

- Estimated ~5-10% faster attention computation on H100/H200
- At 86ms/step for 11L model, attention is ~40% of step time, so ~2-4ms/step improvement
- Over 600s this yields ~50-150 additional training steps
- Indirect BPB benefit from more training steps within the wallclock budget

## Status

**Not yet runnable.** Requires Composer changes to support FlashAttention 3.

### Required Composer changes

1. **New attention backend option** in `composer/nn/layers/attention/backends.py`

   Add FA3 as a backend choice alongside the existing PyTorch SDP backends:

   ```python
   class AttentionBackend(StrEnum):
       PYTORCH_SDP = "pytorch_sdp"   # existing: torch.nn.functional.scaled_dot_product_attention
       FLASH_ATTN_3 = "flash_attn_3" # new: flash_attn_interface.flash_attn_func
   ```

2. **FA3 attention function** in `CausalGroupedQueryAttention`

   When the backend is `flash_attn_3`:
   - Skip PyTorch SDP entirely
   - Call `flash_attn_func(q, k, v, causal=True)` directly
   - Use `[B, T, H, D]` layout (the module must handle the layout difference)
   - The `flash_attn_interface` package must be installed (it's separate from the `flash_attn` FA2 package)

3. **CUDABackend config extension**

   ```yaml
   features:
     - CUDABackend:
         attention_backend: flash_attn_3  # new option
         allow_tf32: true
         cudnn_allow_tf32: true
   ```

   Or as a model trainer feature:
   ```yaml
   features:
     - FlashAttention3:
         enabled: true
   ```

4. **Tensor layout considerations**:
   - FA3 expects `[B, T, H, D]`; Composer's attention uses `[B, H, T, D]` internally
   - The cleanest approach is to transpose before/after the FA3 call, or to change the internal layout when FA3 is selected
   - The RoPE cos/sin cache and AttentionScale q_gain broadcast dimensions must match the chosen layout

5. **Dependency**: Requires `flash-attn-interface` package (FA3), which is separate from `flash-attn` (FA2). Only works on Hopper+ GPUs (H100, H200). Should gracefully fall back to FA2/SDP on non-Hopper hardware.
