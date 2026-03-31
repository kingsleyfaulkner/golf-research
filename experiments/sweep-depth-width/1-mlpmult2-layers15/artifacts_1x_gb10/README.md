# 1-mlpmult2-layers15

## Sweep Overrides

```yaml
model.num_layers: 15
model.mlp_mult: 2
```

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.max_wallclock_seconds: 10
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
manifest.tokenizers.default.SentencePiece.model_path: /home/kingsley/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Results

- **Steps:** 9
- **Tokens:** 1.2M
- **Train loss:** 6.7045
- **Val loss:** 6.5022
- **Val BPB:** 3.8510

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10_2](../../baseline/artifacts_1x_gb10_2))

- **Val BPB:** 3.8510 vs 1.5347 (+2.3163)

| | train loss | full |
| :--- | ---: | ---: |
| **Experiment** | 6.7045 | 3.8510 |
| **Baseline** | 2.4895 | 1.5347 |
| **Delta** | +4.2150 | +2.3163 |

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
