# Sliding Window Eval

**Date**: 2026-03-19
**val_bpb**: 1.1925 | **Artifact Size**: 15,874,829 bytes | **Hardware**: 8×H100 SXM, 600s

## Summary

Implements sliding window evaluation with stride=64, providing a 0.032 BPB improvement over baseline with zero training changes. The baseline evaluates by chopping the validation set into non-overlapping 1024-token chunks where the first token has zero context; sliding window evaluation ensures every token is scored with 960+ tokens of context. The script also includes unused experimental code for QAT, weight looping with LoRA, and layer reuse, all disabled by default.

## Key Changes from Baseline

### Architecture Changes
- Added `num_loops` parameter for block reuse (default 1 = no looping, unused in this submission)
- Added `AttentionLoRA` module for per-loop LoRA adapters on Q/K/V/proj (default rank=0 = disabled, unused)
- Added `lora_rank` parameter (default 0, unused)
- Attention forward signature extended with optional `lora` parameter
- Block forward signature extended with optional `lora` parameter
- GPT model forward loop rewritten to support looped block reuse with LoRA adapters
- Added `forward_logits` method to GPT model (returns logits without computing loss, used by sliding eval)

### Training Changes
- Added `qat` flag (default 1, but set to 0 in the submission command)
- Added `fake_quantize_int8_per_row` function implementing STE fake quantization for int8
- CastedLinear extended with `_qat` flag to optionally apply fake quantization during training
- Added `lora_lr` hyperparameter (default 0.01)
- Added optimizer group for LoRA parameters when adapters are present
- Logging extended with unique_layers, loops, effective_depth, lora_rank, lora_params

### Quantization/Compression Changes
- No changes to quantization (standard int8 + zlib-9)

### Evaluation Changes
- Added `eval_stride` parameter (default 64): stride for sliding window evaluation
- Added `eval_batch_seqs` parameter (default 32): number of windows per forward pass
- Added `eval_val_sliding` function: implements sliding window evaluation where windows of `train_seq_len` advance by `stride`, only scoring the last `stride` tokens per window (first window scores all)
- Final evaluation dispatches to sliding window when `eval_stride < train_seq_len`, otherwise uses standard eval
- Windows are batched (configurable via `eval_batch_seqs`) and distributed across ranks

## Detailed Code Diff Analysis

The most impactful change is the `eval_val_sliding` function (~90 lines). It builds a list of window start positions spaced `stride` apart across the validation set, distributes them across ranks, and processes them in batches. For each window:

1. Input tokens `x` and target tokens `y` are extracted from the validation set
2. The model's `forward_logits` (non-compiled, no loss computation) produces logits
3. Cross-entropy loss is computed per-position
4. Only positions from `max(wlen - stride, 0)` to `wlen` are scored (for the first window starting at position 0, all positions are scored)
5. Byte counts are accumulated using the same BPB methodology as the standard eval

With stride=64 and seq_len=1024, each token gets ~960 tokens of context (vs an average of ~512 in the baseline's non-overlapping chunking). The cost is ~16x more forward passes (1024/64), increasing eval time from ~16s to ~70s.

The LoRA and looping code, while present in the script, forms an interesting architecture for future work: blocks can be reused N times (effective depth = unique_layers * num_loops) with per-loop LoRA adapters providing specialization. All of this is disabled via `NUM_LOOPS=1 LORA_RANK=0 QAT=0`.

## Results

| Metric | Baseline | This |
|--------|----------|------|
| Pre-quant val_bpb | 1.2172 | 1.2196 |
| Post-quant val_bpb (standard) | 1.2244 | ~1.224 |
| **Post-quant val_bpb (sliding)** | N/A | **1.1925** |
| Improvement | -- | -0.0319 |
| Training steps | ~13,800 | 13,450 |
| Step avg | ~43ms | 44.61ms |
| Train time | 600s | 600,028ms |
| Eval time | ~16s | 69,881ms (sliding, stride=64, batch_seqs=1024) |
| Peak memory | ~10,000 MiB | 10,119 MiB |
| Artifact size | 15,863,489 | 15,874,829 (code: 58,340 + model: 15,816,489) |

The pre-quant BPB is nearly identical (training unchanged). The 0.032 BPB improvement comes entirely from scoring tokens with richer context during evaluation.

## Based On

Direct modification of baseline. Training is identical; the improvement is purely an evaluation strategy change. The script also carries forward experimental code (QAT, LoRA, looping) that was disabled for this submission.
