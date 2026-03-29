# LoRA TTT

**Date**: 2026-03-19
**val_bpb**: 1.1929 | **Artifact Size**: 15,882,446 bytes | **Hardware**: 8xH100 SXM, 600s

## Summary

Adds per-document LoRA test-time training (TTT) at evaluation time on top of the unmodified baseline model. The key insight is that most of the BPB improvement (~0.034 of the total ~0.035 gain) comes from document-isolated evaluation with strided windows rather than the LoRA adaptation itself. This approach reduces BPB by ~0.031 over the naive baseline without changing the trained model at all.

## Key Changes from Baseline

### Architecture Changes
- Attention forward pass modified to accept optional `q_delta` and `v_delta` LoRA residuals on Q and V projections.
- `TransformerBlock.forward` accepts optional `q_delta_fn` and `v_delta_fn` callables.
- `GPT.forward` accepts an optional `lora` argument (`BatchedTTTLoRA`) that injects low-rank deltas at every layer.
- LM head gets an additive LoRA bypass: `logits = logits + lora.lm_head_lora(x)`.
- Forward returns per-token losses (shape `[bsz, seq_len]`) when LoRA is active, vs. scalar mean loss for training.

### Training Changes
- Training is identical to baseline. No changes to optimizer, schedule, or hyperparameters.
- Rotary `inv_freq` explicitly cast to float32 to avoid precision issues after compile.

### Quantization/Compression Changes
- Same int8+zlib quantization as baseline.
- Code size increases from 47,642 to 58,509 bytes due to TTT evaluation code (~10KB overhead).

### Evaluation Changes
- New `eval_val_ttt_lora` function replaces standard continuous-stream evaluation.
- Documents identified by BOS token boundaries; each document evaluated independently (no cross-document context).
- Documents split into 256-token chunks within 1024-token sliding context windows.
- For each chunk: score first (accumulate loss/bytes), then train LoRA on that chunk's loss (no data leakage).
- LoRA adapters reset between documents.
- Rank-8 LoRA on `lm_head`, `c_q`, and `c_v` in all 9 transformer blocks.
- Adam optimizer with `lr=0.01`, `betas=(0.9, 0.95)`, one gradient step per chunk.
- Batch size of 64 documents, sorted by length for efficiency.
- New hyperparameters: `TTT_LORA_RANK=8`, `TTT_LORA_LR=0.01`, `TTT_CHUNK_SIZE=256`, `TTT_EVAL_SEQ_LEN=1024`, `TTT_BATCH_SIZE=64`.

## Detailed Code Diff Analysis

The diff adds ~230 lines, almost entirely for evaluation-time LoRA infrastructure:

**LoRA modules** (`BatchedLinearLoRA`, `BatchedTTTLoRA`): Per-batch-element low-rank adapters. `A` is Kaiming-uniform initialized, `B` is zero-initialized (so LoRA starts as identity). Each batch element has independent weights, enabling parallel per-document adaptation.

**Document discovery** (`_find_docs`): Scans validation tokens for BOS boundaries to isolate individual documents. Each document includes the next document's BOS token to match continuous-stream token counts.

**Chunk windowing** (`_compute_chunk_window`): Implements the sliding window logic -- each 256-token prediction chunk is evaluated within a 1024-token context window that extends backward for maximum context.

**BPB accumulation** (`_accumulate_bpb`): Token-level loss accumulation with byte counting that accounts for leading-space tokens at document boundaries (matching the official BPB metric).

**Main eval loop** (`eval_val_ttt_lora`): Processes documents in batches of 64. For each chunk in a batch: (1) forward pass through frozen base model + LoRA, (2) accumulate BPB scores, (3) if not the last chunk, backprop through LoRA params and take one Adam step. Documents within a batch are masked when they run out of chunks.

**Ablation results from the README show the contribution breakdown**:
- Document isolation alone: -0.011 BPB
- Adding stride (chunk=256): -0.034 BPB cumulative
- Adding LoRA TTT: -0.037 BPB cumulative

This means ~92% of the gain comes from eval methodology (doc isolation + strided windows), not from the actual test-time training.

## Results

| Seed | val_loss | val_bpb |
|------|----------|---------|
| Run 1 | - | 1.1927 |
| Run 2 | - | 1.1935 |
| Run 3 | - | 1.1921 |
| Run 4 | - | 1.1929 |
| **Mean** | **2.0142** | **1.1928** |
| Std | - | 0.0005 |

Submitted at bpb=1.195. p-value < 1.195: 0.00234.

Improvement over baseline: -0.0316 BPB.

## Based On

Direct modification of baseline. Training is identical to [NaiveBaseline](../2026-03-17_NaiveBaseline/); all changes are evaluation-side only.
