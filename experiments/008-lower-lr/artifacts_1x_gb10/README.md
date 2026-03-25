# 008 Lower Learning Rates

Halves the Muon and Adam learning rates based on a systematic 6-point sweep.

## Change from baseline

- Muon `lr`: 0.04 → 0.02
- Adam (block_scalars) `lr`: 0.04 → 0.02
- Adam (embedding) `lr`: 0.05 → 0.03

## Source

- `reference/track_10min_16mb/2026-03-18_LowerLR/`
- 6-point LR sweep showed baseline matrix_lr=0.04 was ~2x too high; optimum at 0.02
- Sweep results: 0.06 (+0.016), 0.04 (baseline), 0.03 (-0.001), 0.025 (-0.004), **0.02 (-0.006)**, 0.015 (-0.005)

## Expected impact

- Estimated ~-0.001 to -0.002 BPB
- Modest improvement on its own but establishes the correct LR baseline for combining with other techniques
- Note: source record ran on H200, not H100; results may differ slightly

## Status

**Runnable.**

## Runtime Overrides

```yaml
training.pre_training.batch_size: 16
training.pre_training.data.TokenizedDataset.path: /home/kingsley/github/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
tokenizers.default.SentencePiece.model_path: /home/kingsley/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
```

## Results

- **Steps:** 678
- **Tokens:** 88.9M
- **Train loss:** 2.5614
- **Val loss:** 2.5768
- **Val BPB:** 1.5261

## Train Loss Curve

![Train loss vs runtime (log scale)](loss_chart.svg)

## vs Baseline ([artifacts_1x_gb10](../../baseline/artifacts_1x_gb10))

- **Val BPB:** 1.5261 vs 1.5297 (-0.0036)

| | train loss | full | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Experiment** | 2.5614 | 1.5261 | 1.5448 | 1.5278 | 1.5707 | 1.5623 |
| **Baseline** | 2.6172 | 1.5297 | 1.5433 | 1.5305 | 1.6281 | 1.6081 |
| **Delta** | -0.0558 | -0.0036 | +0.0015 | -0.0027 | -0.0574 | -0.0458 |

## Quantization

| | int6 | int8 | mxfp4 | nvfp4 |
| :--- | ---: | ---: | ---: | ---: |
| **BPB** | 1.5448 | 1.5278 | 1.5707 | 1.5623 |
| **Size** | 8.3 MB | 12.1 MB | 8.6 MB | 9.2 MB |

## Config Changes vs Baseline

**train.yaml:**

```diff
@@ -2,19 +2,10 @@
 model_name: baseline
 training:
   pre_training:
-    # Global batch = max(gpus, 8) * batch_size * sequence_length tokens.
-    # At the reference 8 GPUs this is 524,288 tokens; with more GPUs the
-    # global batch scales up rather than accumulating.
-    # GradientAccumulation auto-computes micro-batching from world_size:
-    #   1 GPU  -> grad_accum = 8 (8 micro-batches of 64 seqs)
-    #   8 GPUs -> grad_accum = 1 (no accumulation needed)
     gpus: !env WORLD_SIZE:1
     batch_size: 64
     sequence_length: !expr "model.context_length"
     total_batch_tokens: !expr "max(self.gpus, 8) * self.batch_size * self.sequence_length"
-    # 10-minute wallclock cap matching the challenge constraint.
-    # Combined with max_steps as a safety limit - training stops when
-    # either condition is met first.
     max_wallclock_seconds: 600
     warmup_steps: 10
     model_trainer:
@@ -23,7 +14,7 @@
           LayerWise:
             default_optimizer:
               Muon:
-                lr: 0.04
+                lr: 0.02
                 momentum: 0.95
                 nesterov: true
                 ns_steps: 5
@@ -41,25 +32,16 @@
                 patterns: ["embedding.*"]
                 optimizer:
                   Adam:
-                    lr: 0.05
+                    lr: 0.03
                     betas: [0.9, 0.95]
                     eps: 1.0e-8
               - name: block_scalars
                 max_ndim: 1
                 optimizer:
                   Adam:
-                    lr: 0.04
+                    lr: 0.02
                     betas: [0.9, 0.95]
                     eps: 1.0e-8
-              # With tied embeddings, the head group shares weights with
-              # embedding via TiedLayers.  If untied, add a head group:
-              # - name: head
-              #   patterns: ["heads.*"]
-              #   optimizer:
-              #     Adam:
-              #       lr: 0.008
-              #       betas: [0.9, 0.95]
-              #       eps: 1.0e-8
         scheduler:
           WallclockWarmdown:
             warmdown_steps: 1200
```

## Platform

- **GPU:** NVIDIA GB10 (119.7 GB)
- **GPUs:** 1
- **CPU:** aarch64 (20 cores)
- **RAM:** 120 GB
- **Software:** PyTorch 2.10.0+cu130, CUDA 13.0
