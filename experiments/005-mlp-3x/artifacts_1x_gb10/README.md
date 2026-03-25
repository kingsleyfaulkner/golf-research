# 005 MLP 3x Expansion

Increases the FFN intermediate size from 2x to 3x hidden dimension (1024 → 1536).

## Change from baseline

- `mlp_mult`: 2 → 3 in model.yaml
- `intermediate_size`: 1024 → 1536 (computed via `mlp_mult * hidden_size`)
- Adds ~2.4M parameters (from ~17M to ~19.4M)

## Source

Both top submissions use 3x MLP expansion, funded by aggressive quantization:
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50` (3x MLP + 10 layers)
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (3x MLP + 9 layers)

## Expected impact

- Single largest architectural contributor: ~0.003 BPB improvement
- The wider FFN increases model capacity where it matters most (feed-forward layers dominate parameter count)
- Requires int6 quantization to fit under 16 MB — at int6 the baseline uses 12.3 MB, leaving headroom for ~3.7 MB of extra parameters

## Notes

- The larger model will train slightly slower per step but should converge to a lower loss within the 10-minute budget
- Must be evaluated with int6 quantization to assess contest viability

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
tokenizers.default.SentencePiece.model_path: /home/kingsley/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Results

- **Steps:** 618
- **Tokens:** 81.0M
- **Train loss:** 2.5710
- **Val loss:** 2.5913
- **Val BPB:** 1.5347

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10](../../baseline/artifacts_1x_gb10))

- **Val BPB:** 1.5347 vs 1.5297 (+0.0050)

| | train loss | full | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Experiment** | 2.5710 | 1.5347 | 1.5469 | 1.5355 | 1.6020 | 1.5915 |
| **Baseline** | 2.6172 | 1.5297 | 1.5433 | 1.5305 | 1.6281 | 1.6081 |
| **Delta** | -0.0462 | +0.0050 | +0.0036 | +0.0050 | -0.0261 | -0.0166 |

## Quantization

| | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: |
| **BPB** | 1.5469 | 1.5355 | 1.6020 | 1.5915 |
| **Size** | 11.9 MB | 17.1 MB | 10.9 MB | 11.7 MB |

## Config Changes vs Baseline

**model.yaml:**

```diff
@@ -105,7 +105,7 @@
       num_attention_heads: 8
       num_key_value_heads: 4
       head_dim: 64
-      mlp_mult: 2
+      mlp_mult: 3
       intermediate_size: !expr "self.mlp_mult * self.hidden_size"
       # Encoder-decoder layer split
       num_encoder_layers: !expr "self.num_layers // 2"
```

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
