# 4-mlpmult5-layers15

## Sweep Overrides

```yaml
model.num_layers: 15
model.mlp_mult: 5
```

## Results

- **Steps:** 19
- **Tokens:** 2.5M
- **Train loss:** 5.3615
- **Val loss:** 5.2019
- **Val BPB:** 3.0809

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 3.0809 vs 1.5347 (+1.5461)

| | train loss | full | int8 |
| :--- | ---: | ---: | ---: |
| **Experiment** | 5.3615 | 3.0809 | 3.0809 |
| **Baseline** | 2.4895 | 1.5347 | 1.5522 |
| **Delta** | +2.8720 | +1.5461 | +1.5287 |

## Quantization

| | int8 |
| :--- | ---: |
| **BPB** | 3.0809 |
| **Size** | 27.5 MB |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
