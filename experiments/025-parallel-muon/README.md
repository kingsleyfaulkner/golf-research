# 025 Parallel Muon (Async Communication Overlap)

Replaces standard DDP-based Muon with an async reduce-scatter/all-gather pipeline that overlaps communication with Newton-Schulz computation.

## Change from baseline

- Parameter Banking: 66 separate `nn.Linear` weights consolidated into 4 contiguous 3D `nn.Parameter` banks
- Newton-Schulz orthogonalization operates on 3D batched tensors via `torch.bmm`
- Three-phase optimizer step:
  1. `launch_reduce_scatters()`: async reduce-scatter for all banks (biggest first) — called right after backward
  2. Adam steps on small params while reduce-scatter is in-flight (overlap)
  3. `step()`: wait for each RS, local NS5 on shard, launch async all-gather; each all-gather overlaps with next bank's NS5
- No DDP wrapping for bank parameters — communication managed entirely by the optimizer
- Banks sorted by size descending to launch biggest reduce-scatters first for optimal overlap

## Source

- `reference/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` (lines 127-256)
- From PR #399 by @abaybektursun
- Ablation shows +/-0.0000 BPB change (same model quality) but **83.3ms/step vs ~85ms/step** (2ms faster)
- Over 600s this yields ~17 additional training steps (~0.2% more training)

## How it works

Standard Muon with DDP:
1. Forward + backward (DDP all-reduces gradients synchronously)
2. NS5 on full gradient (all ranks compute same thing)
3. Update weights

Parallel Muon:
1. Forward + backward (no DDP for bank params)
2. `launch_reduce_scatters()` — each rank gets 1/N of the gradient
3. **While RS is in-flight**: Adam steps on non-bank params (scalar, embedding)
4. For each bank (biggest first):
   a. Wait for reduce-scatter completion
   b. Local NS5 on 1/N shard only (N× less compute per rank)
   c. Launch async all-gather to broadcast update to all ranks
   d. **While AG is in-flight**: process next bank's NS5
5. Wait for final all-gather, apply updates

The key insight is that NS5 (Newton-Schulz iteration, 5 steps) is computationally expensive. By sharding the gradient and running NS5 on only the local shard, each rank does 1/N of the work. The all-gather then broadcasts the combined update. Communication (RS + AG) overlaps with computation (NS5 + Adam).

```python
def launch_reduce_scatters(self):
    for m in self._bank_meta:  # sorted by size descending
        pg = m['padded_grad']
        pg[:m['B']].copy_(p.grad.bfloat16())
        fut = dist.reduce_scatter_tensor(m['shard'], pg, op=ReduceOp.AVG, async_op=True)
        self._rs_futures.append(fut)

def step(self):
    for i, m in enumerate(self._bank_meta):
        if prev_ag_handle is not None:
            prev_ag_handle.wait()
            apply_update(prev_m)  # apply previous bank's gathered update
        self._rs_futures[i].wait()  # wait for this bank's reduce-scatter
        update = zeropower_via_newtonschulz5(m['shard'])  # local NS5 on shard
        prev_ag_handle = dist.all_gather_into_tensor(m['full_update'], update, async_op=True)
```

## Expected impact

- **0 BPB change** — identical model quality to standard Muon
- **~2ms/step faster** (83.3 vs 85ms) from communication/computation overlap
- Over 600s: ~17 extra training steps, worth ~0.0001-0.0002 BPB indirectly
- Most valuable at scale (8 GPUs) where communication is a larger fraction of step time

## Status

**Not yet runnable.** Requires significant Composer Muon optimizer changes.

### Required Composer changes

1. **Parameter Banking in model construction**

   The model must consolidate linear weights into contiguous 3D parameter banks. This is a fundamental change to how the model stores parameters:
   ```python
   # Instead of 66 separate nn.Linear weights:
   # self.c_q = nn.Linear(512, 512)  # [512, 512]
   # self.c_k = nn.Linear(512, 256)  # [256, 512]
   # ...

   # Consolidate into 4 banks:
   # bank_0 = nn.Parameter(torch.zeros(N_0, max_rows, max_cols))
   # Linear layers become views into the bank
   ```

   This requires either:
   - A model-level feature that consolidates parameters post-construction
   - Or a new `BankedLinear` layer type

2. **Async Muon optimizer** in `composer/training/optim/muon.py`

   Replace the current synchronous DDP-based Muon with the three-phase async protocol:
   - `launch_reduce_scatters()` method callable from the training loop right after backward
   - `step()` modified to wait/compute/gather with async overlap
   - Pre-allocated buffers for padded gradients, shards, momentum, and updates
   - Bank sorting by size for optimal overlap

3. **Training loop integration**

   The training loop must call `launch_reduce_scatters()` immediately after backward, before Adam steps:
   ```python
   loss.backward()
   muon_optimizer.launch_reduce_scatters()  # async RS starts
   adam_optimizer.step()                     # overlaps with RS
   muon_optimizer.step()                    # wait RS, NS5, AG
   ```

   This requires a new hook point in the Composer training loop between backward and optimizer step.

4. **Batched Newton-Schulz**: The NS5 function must support 3D inputs `[B, M, N]` with per-matrix normalization:
   ```python
   norms = X.norm(dim=(-2, -1), keepdim=True)
   X = X / (norms + eps)
   for _ in range(steps):
       A = X @ X.mT
       B = b * A + c * A @ A
       X = a * X + B @ X
   ```
