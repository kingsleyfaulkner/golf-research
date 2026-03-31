# 5-mlpmult2-layers20

## Sweep Overrides

```yaml
model.num_layers: 20
model.mlp_mult: 2
training.pre_training.batch_size: 32
```

## Results

- **Steps:** 19
- **Tokens:** 2.5M
- **Train loss:** 5.4927
- **Val loss:** 5.3506
- **Val BPB:** 3.1689

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 3.1689 vs 1.5347 (+1.6342)

| | train loss | full | int8 |
| :--- | ---: | ---: | ---: |
| **Experiment** | 5.4927 | 3.1689 | 3.1692 |
| **Baseline** | 2.4895 | 1.5347 | 1.5522 |
| **Delta** | +3.0032 | +1.6342 | +1.6169 |

## Quantization

| | int8 |
| :--- | ---: |
| **BPB** | 3.1692 |
| **Size** | 19.1 MB |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
