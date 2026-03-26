# SmearGate + OrthoInit + Muon WD + Int6 STE QAT + MLP 3x + Sliding Window

**Date**: 2026-03-19
**val_bpb**: 1.1556 | **Artifact Size**: 15,878,809 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

A 9-layer, 22.4M parameter transformer combining SmearGate (learned per-dimension previous-token blending), bigram hash embeddings, orthogonal weight initialization, Muon weight decay integrated directly into the optimizer, int6 QAT via STE, 3x MLP expansion, and sliding window evaluation. The SmearGate and bigram hash provide cheap bigram context before the transformer even begins attention, while orthogonal init gives the model a better starting point for the constrained 10-minute training budget. Note: this attempt uses `train_gpt_v5.py` rather than modifying the baseline `train_gpt.py`.

## Key Changes from Baseline

### Architecture Changes
- MLP expansion factor increased from 2x to 3x (hidden=1536)
- **SmearGate**: learned per-dimension gate blending each token's embedding with the previous token's. Gate initialized at `sigmoid(3.0) ~ 0.95` (mostly pass-through). Applied after embedding + bigram hash, before RMS norm. Only ~512 scalar parameters
- **BigramHash embedding**: 4096-bucket hash table mapping (prev_token, cur_token) pairs via `(prev * 92821 + cur) % 4096` to learned 128-dim embeddings, projected to model dim (512) via a CastedLinear. Gives the model direct token-pair features at minimal cost
- `forward_logits()` method added to GPT class for sliding window eval, includes SmearGate and BigramHash in the forward path
- Both BigramHash and SmearGate are optional (configured via `bigram_hash_buckets` and `use_smeargate`)

### Training Changes
- **Orthogonal weight initialization**: all non-zero-init CastedLinear weights initialized with `nn.init.orthogonal_()` instead of the default PyTorch init. Singular values all equal 1, providing uniform gradient flow. Muon's Newton-Schulz step orthogonalizes updates, so starting orthogonal means early steps are immediately useful
- **Muon weight decay integrated into optimizer**: `weight_decay` parameter added to Muon class, applied as decoupled `p.mul_(1 - wd * lr)` before gradient update (not after, unlike the other attempts)
- Warmdown iterations increased from 1200 to 3000
- Validation disabled during training (`val_loss_every` 1000 -> 0)
- Tied embedding LR reduced from 0.05 to 0.030
- Matrix LR reduced from 0.04 to 0.020
- Scalar LR reduced from 0.04 to 0.020
- Muon momentum increased from 0.95 to 0.99
- Muon momentum warmup start raised from 0.85 to 0.92
- Muon momentum warmup steps increased from 500 to 1500
- Muon weight decay set to 0.01
- BigramHash table weight added to the embedding optimizer (Adam)
- BigramHash projection weight added to matrix params (Muon)
- SmearGate gate parameter added to scalar optimizer

### Quantization/Compression Changes
- Int6 QAT via STE in `CastedLinear.forward()`: fake-quantizes 2D weights to [-31, 31] range per row during training forward pass, with gradients flowing through via `w + (w_q - w).detach()`
- Post-training int6 quantization: all 2D tensors quantized to [-31, 31] range in int8 containers
- FP16 passthrough for tensors matching `FP16_PASSTHROUGH_PATTERNS` (default: "tok_emb,bigram_hash") -- these tensors skip quantization entirely and are stored as fp16
- zstd-22 compression replaces zlib-9 (with fallback)
- Artifact saved as `.int6.ptz`

### Evaluation Changes
- Sliding window evaluation with stride=64, identical implementation to [MixedQuant_Int6Int8_SlidingWindow](../2026-03-19_MixedQuant_Int6Int8_SlidingWindow/)
- Eval batch size configurable via `SW_EVAL_BATCH` env var (default 32)

## Detailed Code Diff Analysis

The SmearGate is elegantly simple: a single `dim`-sized parameter vector passed through sigmoid to produce per-dimension blend weights. The forward pass is `g * x + (1 - g) * x_prev` where `x_prev` is the shifted input (zero-padded at position 0). Initialized at `sigmoid(3.0) ~ 0.95`, it starts near-identity and learns per-dimension how much previous-token context to inject.

The BigramHash maps token pairs to embeddings via a hash function. The prime multiplier 92821 is chosen to distribute bigram hashes uniformly across 4096 buckets. The 128-dim hash embeddings are projected to model dim through a CastedLinear (which gets int6 QAT during training). The hash table itself is stored as fp16 passthrough since it's an embedding without STE protection.

The Muon weight decay integration is notably different from the other attempts: it's built directly into the `Muon.step()` method and applied *before* the gradient update (the others apply it *after* the optimizer step in the training loop). The code comment explains the dual benefit: tighter weight distributions improve both generalization and quantization compression.

The orthogonal init applies to all CastedLinear modules that don't have `_zero_init=True`. This replaces the default PyTorch Kaiming uniform init, providing all-ones singular values. The init comment notes this is especially valuable with Muon since the optimizer already orthogonalizes gradient updates -- starting from orthogonal means early updates don't waste time correcting random initialization.

## Results

| Metric | Value |
|--------|-------|
| Post-quant sliding window val_bpb | **1.1556** |
| Post-quant sliding window val_loss | 1.9511 |
| Post-quant standard val_bpb | 1.1891 |
| Post-quant standard val_loss | 2.0077 |
| Quantization gap (standard eval) | ~0.0001 BPB |
| Model parameters | 22,368,840 |
| Artifact size (int6+zstd-22) | 15,878,809 bytes |
| Training steps | 12,047 |
| Training time | 600s |
| Sliding window eval time | 75s |
| Peak GPU memory | 11,340 MiB |

## Based On

Direct modification of baseline (uses `train_gpt_v5.py`). Shares the int6 QAT + STE technique and Muon hyperparameter tuning with [MixedQuant_Int6Int8_SlidingWindow](../2026-03-19_MixedQuant_Int6Int8_SlidingWindow/), but adds SmearGate, BigramHash, and orthogonal init as novel architectural contributions.
