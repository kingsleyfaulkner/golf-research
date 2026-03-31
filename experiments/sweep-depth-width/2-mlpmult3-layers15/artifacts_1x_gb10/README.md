# 2-mlpmult3-layers15

## Sweep Overrides

```yaml
model.num_layers: 15
model.mlp_mult: 3
```

## Results

- **Steps:** 23
- **Tokens:** 3.0M
- **Train loss:** 5.0146
- **Val loss:** 4.9807
- **Val BPB:** 2.9499

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 2.9499 vs 1.5347 (+1.4151)

| | train loss | full | int8 |
| :--- | ---: | ---: | ---: |
| **Experiment** | 5.0146 | 2.9499 | 2.9499 |
| **Baseline** | 2.4895 | 1.5347 | 1.5522 |
| **Delta** | +2.5252 | +1.4151 | +1.3977 |

## Quantization

| | int8 |
| :--- | ---: |
| **BPB** | 2.9499 |
| **Size** | 17.9 MB |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
