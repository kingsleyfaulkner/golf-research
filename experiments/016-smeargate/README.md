# 016 SmearGate

Learned per-dimension gate that blends each token's embedding with the previous token's, injecting bigram context before the transformer blocks.

## Change from baseline

- New `SmearGate` module applied after token embedding, before RMS norm
- Gate: single `dim`-sized parameter vector (512 scalars)
- Forward: `g = sigmoid(gate); output = g * x + (1-g) * x_prev`
- `x_prev` is the input shifted right by 1 position (zero-padded at position 0)
- Gate initialized at `3.0` so `sigmoid(3.0) ≈ 0.95` — starts near-identity (mostly pass-through)
- The gate is optimized by the scalar Adam group (same as other 1D parameters)

## Source

- `reference/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/` (train_gpt_v5.py, lines 824-844)
- Technique originated by @unnir in parameter-golf PR #102/#135
- Also used in:
  - `reference/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` (1.1458 BPB)
  - `reference/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` (1.1428 BPB)
  - All later top submissions that use BigramHash also use SmearGate

## How it works

```python
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # sigmoid(3.0) ≈ 0.95 → mostly pass-through at init
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev
```

The module provides cheap bigram context without attention. Where a transformer must discover token-pair relationships through self-attention (costing O(n²) compute), SmearGate injects this signal for ~512 parameters and negligible compute.

The per-dimension gate learns *which* embedding dimensions benefit from previous-token context. Dimensions encoding position-independent features (e.g., word meaning) may stay near 1.0 (no blending), while dimensions encoding context-sensitive features (e.g., sub-word continuation) may drop toward 0.0 (heavy blending).

The forward path in the model is:
1. `x = tok_emb(input_ids)`
2. `x = x + bigram_hash(input_ids)` (if BigramHash is enabled)
3. `x = smeargate(x)` ← SmearGate applied here
4. `x = rms_norm(x)`
5. `x0 = x` (captured for skip connections)
6. Transformer blocks...

## Expected impact

- Estimated ~-0.001 to -0.002 BPB
- Always used together with BigramHash in the records, so isolated contribution is uncertain
- Negligible compute and parameter cost (~512 params, one shifted concat + elementwise ops)
- Pairs naturally with BigramHash (009): SmearGate provides continuous blending while BigramHash provides discrete token-pair lookup

## Status

**Not yet runnable.** Requires Composer changes to add a SmearGate layer module.

### Required Composer changes

1. **New module: `SmearGate`** in `composer/nn/layers/` (e.g., `composer/nn/layers/smear_gate.py`)

   ```python
   @register_module_config(ModulePath.LAYER)
   class SmearGateConfig(NNModuleConfig):
       type: Literal["SmearGate"] = "SmearGate"
       init_value: float | None = None  # sigmoid^{-1}(desired_init), default 3.0

       def build(self, device="cpu", **kwargs) -> "SmearGate":
           config = self.with_defaults(**kwargs)
           return SmearGate(
               hidden_size=config.hidden_size,
               init_value=config.init_value or 3.0,
               device=device,
           )

   class SmearGate(NNModule):
       def __init__(self, hidden_size: int, init_value: float = 3.0, device="cpu"):
           super().__init__()
           self.gate = nn.Parameter(
               torch.full((hidden_size,), init_value, dtype=torch.float32, device=device)
           )

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           g = torch.sigmoid(self.gate).to(dtype=x.dtype)
           x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
           return g * x + (1.0 - g) * x_prev
   ```

2. **Integration in model.yaml**: SmearGate should be placed in the block sequence after embedding and before the first ForwardContext. The natural location is as a layer in the `SequentialBlock`:

   ```yaml
   block:
     SequentialBlock:
       smeargate:
         SmearGate:
           init_value: 3.0

       forward_x0:
         ForwardContext:
           key: x0

       encoder:
         RepeatingBlock:
           ...
   ```

   This works because `SequentialBlock` processes its children in order, so SmearGate runs on the post-embedding hidden states before they're captured as x0.

3. **Optimizer routing**: The gate parameter is 1D, so it naturally falls into the `block_scalars` group (matched by `max_ndim: 1` in the LayerWise config). No optimizer config changes needed.

4. **torch.compile compatibility**: The forward pass uses only standard ops (`sigmoid`, `cat`, `zeros_like`, elementwise multiply/add) — all compile-friendly with `fullgraph=True`.
