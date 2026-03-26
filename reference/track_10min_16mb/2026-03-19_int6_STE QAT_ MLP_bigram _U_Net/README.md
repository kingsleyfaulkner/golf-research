# Int6 STE QAT + MLP + Bigram + U-Net

**Date**: 2026-03-19
**val_bpb**: 1.1598 (sliding window stride=64) | **Artifact Size**: 16.19 MB (int6+zstd-22) | **Hardware**: 8×H100 SXM, 600s

## Summary

This attempt combines multiple architectural innovations: int6 quantization-aware training (QAT) with straight-through estimator (STE), MLP expansion, bigram hash embeddings, and U-Net skip connections. The train script is embedded in the train.log rather than saved separately, so no standalone `train_gpt.py` was preserved.

## Status

**Incomplete record** — only `train.log` is available. No standalone train script, README, or submission.json was preserved. The analysis below is derived entirely from the training log.

## Key Details from Training Log

### Architecture
- Model params: 22,368,328
- GQA: 8 heads, 4 KV heads
- Tied embeddings with embed_lr=0.03
- Sequence length: 1024
- Batch: 524,288 tokens/step
- Suggests MLP 3x expansion, bigram hash, and U-Net skip connections based on the folder name

### Training
- Muon optimizer: matrix_lr=0.02, scalar_lr=0.02
- Warmdown iterations: wallclock-based (600s cap)
- Reached step 12,123/20,000 before wallclock stop
- Step average: 49.49ms

### Quantization
- Int6 quantization with zstd-22 compression
- Post-quant int6 roundtrip: val_bpb=1.1931
- Sliding window eval (stride=64): val_bpb=1.1598

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_loss (step 12123) | 2.0158 |
| Pre-quant val_bpb | 1.1939 |
| Int6 roundtrip val_bpb | 1.1931 |
| Sliding window val_bpb (stride=64) | 1.1598 |
| Artifact size (int6+zstd-22) | 16.19 MB |
| Steps completed | 12,123 / 20,000 |
| Training time | ~600s |
| Peak memory | 11,273 MiB allocated |

## Notes

- The artifact size of 16.19 MB exceeds the 16 MB contest limit
- The folder name suggests this is a combined approach using int6 STE QAT, bigram hash embeddings, and U-Net architecture
- No diff analysis possible without the standalone train script
- The train.log contains the full modified train_gpt.py source (first ~1400 lines), followed by training output

## Based On

Cannot be determined from available materials. The combination of techniques (int6 QAT, bigram hash, U-Net, MLP expansion) suggests it builds on multiple concurrent exploration threads from 2026-03-19.
