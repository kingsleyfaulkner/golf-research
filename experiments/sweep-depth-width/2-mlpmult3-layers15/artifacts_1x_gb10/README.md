# 2-mlpmult3-layers15

## Sweep Overrides

```yaml
model.num_layers: 15
model.mlp_mult: 3
```

## Results

- **Steps:** 8
- **Tokens:** 1.0M
- **Train loss:** 7.0872
- **Val loss:** 6.7140
- **Val BPB:** 3.9764

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 3.9764 vs 1.5347 (+2.4417)

| | train loss | full |
| :--- | ---: | ---: |
| **Experiment** | 7.0872 | 3.9764 |
| **Baseline** | 2.4895 | 1.5347 |
| **Delta** | +4.5978 | +2.4417 |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
