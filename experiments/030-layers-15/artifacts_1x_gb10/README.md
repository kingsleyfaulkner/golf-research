# 030-layers-15

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
```

## Results

- **Steps:** 505
- **Tokens:** 66.2M
- **Train loss:** 2.7746
- **Val loss:** 2.7014
- **Val BPB:** 1.5999

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 1.5999 vs 1.5347 (+0.0652)

| | train loss | full | int8 | turboquip4c |
| :--- | ---: | ---: | ---: | ---: |
| **Experiment** | 2.7746 | 1.5999 | 1.6009 | 1.6887 |
| **Baseline** | 2.4895 | 1.5347 | 1.5522 | 1.5765 |
| **Delta** | +0.2851 | +0.0652 | +0.0486 | +0.1122 |

## Quantization

| | int8 | turboquip4c |
| :--- | ---: | ---: |
| **BPB** | 1.6009 | 1.6887 |
| **Size** | 21.0 MB | 7.2 MB |

## Config Changes vs Baseline

**train.yaml:**

```diff
@@ -1,5 +1,5 @@
 manifest: !include model.yaml
-model_name: baseline
+model_name: 030-layers-15
 training:
   pre_training:
     gpus: !env WORLD_SIZE:1
```

**model.yaml:**

```diff
@@ -95,11 +95,11 @@
           heads.clm.head.weight: embedding.tok_emb.weight
       - CachedRoPE
 models:
-  baseline:
+  030-layers-15:
     DecoderTransformer:
       context_length: 1024
       vocab_size: 1024
-      num_layers: 9
+      num_layers: 15
       hidden_size: !expr "self.num_attention_heads * self.head_dim"
       num_attention_heads: 8
       num_key_value_heads: 4
```

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
