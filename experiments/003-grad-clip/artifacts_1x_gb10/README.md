# 003 Gradient Clipping

Adds gradient norm clipping at 0.3 to the Muon optimizer.

## Change from baseline

- Added `GradClip` feature with `max_grad_norm: 0.3` to Muon

## Source

Top submissions use gradient clipping at 0.3:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`

## Expected impact

- Stabilizes training with the aggressive Muon learning rate (0.04)
- Prevents rare gradient spikes from destabilizing later training
- Estimated ~0.0003 BPB improvement, primarily by enabling other techniques to work better

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
tokenizers.default.SentencePiece.model_path: /home/kingsley/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Results

- **Steps:** 675
- **Tokens:** 88.5M
- **Train loss:** 2.5856
- **Val loss:** 2.5843
- **Val BPB:** 1.5306

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 1.5306 vs 1.5347 (-0.0041)

| | train loss | full | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Experiment** | 2.5856 | 1.5306 | 1.5466 | 1.5319 | 1.6248 | 1.6050 |
| **Baseline** | 2.4895 | 1.5347 | 1.5494 | 1.5522 | 1.6563 | 1.6697 |
| **Delta** | +0.0961 | -0.0041 | -0.0028 | -0.0204 | -0.0315 | -0.0647 |

## Quantization

| | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: |
| **BPB** | 1.5466 | 1.5319 | 1.6248 | 1.6050 |
| **Size** | 9.7 MB | 13.9 MB | 8.6 MB | 9.2 MB |

## Config Changes vs Baseline

**train.yaml:**

```diff
@@ -20,6 +20,8 @@
                 ns_steps: 5
                 weight_decay: 0.0
                 features:
+                  - GradClip:
+                      max_grad_norm: 0.3
                   - HyperparameterSchedule:
                       parameter: momentum
                       initial: 0.85
@@ -63,7 +65,7 @@
     data:
       TokenizedDataset:
         path: /workspace/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
-        shuffle: false
+        shuffle: true
         bin_header_bytes: 1024
     features:
       - SystemDiagnostics:
```

**model.yaml:**

```diff
@@ -6,7 +6,6 @@
       TokenEmbedding:
         init_method: normal
         init_std: 0.005
-        dtype: bfloat16
         norm: RMSNorm
     block:
       SequentialBlock:
@@ -93,7 +92,6 @@
     features:
       - TiedLayers:
           heads.clm.head.weight: embedding.tok_emb.weight
-      - CachedRoPE
 models:
   baseline:
     DecoderTransformer:
```

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
