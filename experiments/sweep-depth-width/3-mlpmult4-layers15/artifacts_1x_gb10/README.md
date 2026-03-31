# 3-mlpmult4-layers15

## Sweep Overrides

```yaml
model.num_layers: 15
model.mlp_mult: 4
```

## Results

- **Steps:** 7
- **Tokens:** 0.9M
- **Train loss:** 7.2579
- **Val loss:** 6.9954
- **Val BPB:** 4.1431

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 4.1431 vs 1.5347 (+2.6083)

| | train loss | full |
| :--- | ---: | ---: |
| **Experiment** | 7.2579 | 4.1431 |
| **Baseline** | 2.4895 | 1.5347 |
| **Delta** | +4.7684 | +2.6083 |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
