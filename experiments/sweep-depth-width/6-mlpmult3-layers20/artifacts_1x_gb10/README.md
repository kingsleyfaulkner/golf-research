# 6-mlpmult3-layers20

## Sweep Overrides

```yaml
model.num_layers: 20
model.mlp_mult: 3
training.pre_training.batch_size: 32
```

## Results

- **Steps:** 17
- **Tokens:** 2.2M
- **Train loss:** 5.6322
- **Val loss:** 5.5179
- **Val BPB:** 3.2680

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 3.2680 vs 1.5347 (+1.7333)

| | train loss | full | int8 |
| :--- | ---: | ---: | ---: |
| **Experiment** | 5.6322 | 3.2680 | 3.2679 |
| **Baseline** | 2.4895 | 1.5347 | 1.5522 |
| **Delta** | +3.1427 | +1.7333 | +1.7157 |

## Quantization

| | int8 |
| :--- | ---: |
| **BPB** | 3.2679 |
| **Size** | 23.0 MB |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
