# 014 Phase-Transition Residual Mixing Initialization

Initializes AddContext residual mixing weights with a sigmoid schedule so early layers trust x0 more and late layers trust the residual more.

## Change from baseline

- `resid_mix` parameters initialized with sigmoid curve across layers instead of uniform `[1.0, 0.0]`
- Formula: `phase = sigmoid(3.0 * (i / (N-1) - 0.5))` for layer `i` of `N` total layers
- Layer 0 (first): `hidden_weight ≈ 0.18, context_weight ≈ 0.82` (trust x0 embedding)
- Layer N-1 (last): `hidden_weight ≈ 0.82, context_weight ≈ 0.18` (trust residual stream)
- Middle layers: approximately 50/50 blend

## Source

- `reference/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/` (diff lines 190-196)
- Unique to this record — not adopted by later submissions (they keep the default ones/zeros init)

## How it works

The baseline initializes `resid_mix` as `[ones, zeros]` for every layer — 100% hidden state, 0% x0 context. This means every layer starts by ignoring the original embedding and only uses the accumulated residual. The model must learn to use x0 from scratch.

The phase-transition init provides a smooth gradient:

```python
num_layers = len(blocks)
for i, block in enumerate(blocks):
    phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
    block.resid_mix.data[0] = phase * torch.ones(dim)      # hidden_weight
    block.resid_mix.data[1] = (1 - phase) * torch.ones(dim) # context_weight
```

For 9 layers, the initialization values are:

| Layer | i/(N-1) | sigmoid | hidden_weight | context_weight |
|-------|---------|---------|---------------|----------------|
| 0 | 0.000 | 0.182 | 0.182 | 0.818 |
| 1 | 0.125 | 0.234 | 0.234 | 0.766 |
| 2 | 0.250 | 0.295 | 0.295 | 0.705 |
| 3 | 0.375 | 0.365 | 0.365 | 0.635 |
| 4 | 0.500 | 0.500 | 0.500 | 0.500 |
| 5 | 0.625 | 0.635 | 0.635 | 0.365 |
| 6 | 0.750 | 0.705 | 0.705 | 0.295 |
| 7 | 0.875 | 0.766 | 0.766 | 0.234 |
| 8 | 1.000 | 0.818 | 0.818 | 0.182 |

The intuition is that early layers should blend heavily with the raw embedding (x0 carries the most direct token information) while late layers should rely on the transformed residual stream (which has accumulated contextual information through attention).

## Expected impact

- Unknown in isolation — always combined with other changes in the source record
- Should accelerate early convergence by providing a better starting point for the mixing weights
- Risk: the sigmoid schedule may not match the optimal mixing profile; the model learns its own schedule anyway
- The `3.0` steepness parameter controls how sharp the transition is; a value of 0 would give uniform 0.5 everywhere

## Status

**Not yet runnable.** Requires Composer changes to support custom initialization of AddContext weights.

### Required Composer changes

1. **New `init_method` option for `AddContext`** in `composer/nn/layers/add_context.py`

   The `AddContextConfig` already has `init_method` and `init_std` fields but `AddContext.reset_parameters()` only supports `ones_` for hidden_weight and `zeros_` for context_weight. A new init method is needed:

   ```python
   class AddContext(NNModule):
       def reset_parameters(self):
           if self._init_method == "phase_transition":
               # Requires layer_idx and num_layers to be known at init time
               phase = torch.sigmoid(torch.tensor(
                   self._init_steepness * (self._layer_idx / max(self._num_layers - 1, 1) - 0.5)
               ))
               if self.hidden_weight is not None:
                   nn.init.constant_(self.hidden_weight, phase.item())
               if self.context_weight is not None:
                   nn.init.constant_(self.context_weight, (1 - phase).item())
           else:
               # existing default init
               if self.hidden_weight is not None:
                   nn.init.ones_(self.hidden_weight)
               if self.context_weight is not None:
                   nn.init.zeros_(self.context_weight)
   ```

2. **Layer index propagation**

   The main challenge is that `AddContext` needs to know its position in the layer stack (`layer_idx`, `num_layers`) at initialization time. Composer's `RepeatingBlock` already tracks this for `layer_idx` and `num_layers` parameters passed to child modules. The `AddContext` config needs to accept these:

   ```yaml
   resid_mix:
     AddContext:
       context_key: x0
       hidden_weight: vector
       context_weight: vector
       init_method: phase_transition
       init_steepness: 3.0  # sigmoid steepness parameter
   ```

   The `RepeatingBlock` would need to pass `layer_idx` and `num_layers` to the `AddContext.build()` method so the sigmoid value can be computed per layer.

3. **Alternative: post-init hook**

   A simpler approach that avoids modifying AddContext: add a model-level `features` entry that runs after model construction and overwrites the resid_mix parameters:
   ```yaml
   features:
     - PhaseTransitionInit:
         pattern: "*.resid_mix.*"
         steepness: 3.0
   ```
   This is more invasive but doesn't require changing AddContext's interface.
