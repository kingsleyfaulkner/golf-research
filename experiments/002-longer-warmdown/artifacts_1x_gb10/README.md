# 002 Longer Warmdown

Extends the learning rate warmdown from 1200 to 3000 steps.

## Change from baseline

- `warmdown_steps`: 1200 → 3000

## Source

Both top submissions use warmdown of 3000 steps:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`

## Expected impact

- More gradual LR decay allows better weight stabilization before final checkpoint
- Works synergistically with SWA (not yet implemented) — the longer tail provides more well-converged checkpoints to average
- Estimated ~0.0005 BPB improvement

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
tokenizers.default.SentencePiece.model_path: /home/kingsley/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Results

- **Steps:** 677
- **Tokens:** 88.7M
- **Train loss:** 2.5939
- **Val loss:** 2.5872
- **Val BPB:** 1.5323

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10](../../baseline/artifacts_1x_gb10))

- **Val BPB:** 1.5323 vs 1.5297 (+0.0026)

| | train loss | full | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Experiment** | 2.5939 | 1.5323 | 1.5486 | 1.5328 | 1.6451 | 1.6147 |
| **Baseline** | 2.6172 | 1.5297 | 1.5433 | 1.5305 | 1.6281 | 1.6081 |
| **Delta** | -0.0232 | +0.0026 | +0.0052 | +0.0023 | +0.0170 | +0.0066 |

## Quantization

| | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: |
| **BPB** | 1.5486 | 1.5328 | 1.6451 | 1.6147 |
| **Size** | 10.3 MB | 14.0 MB | 8.6 MB | 9.2 MB |

## Config Changes vs Baseline

**train.yaml:**

```diff
@@ -62,7 +62,7 @@
               #       eps: 1.0e-8
         scheduler:
           WallclockWarmdown:
-            warmdown_steps: 1200
+            warmdown_steps: 3000
             decay_type: linear
         compile:
           fullgraph: true
```

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
