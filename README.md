# Golf Research

Research experiments for [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge - train the best language model that fits in 16 MB and trains in under 10 minutes on 8xH100 GPUs. Models are evaluated by compression on the FineWeb validation set using a tokeniser-agnostic bits-per-byte (BPB) metric.

Built on [Composer](https://github.com/daystrom-ai/composer), a configuration-driven library for building transformer language models.

## Results

See [RESULTS.md](RESULTS.md) for the full results table.

## Repository Structure

```
golf-research/
├── experiments/          # Experiment configurations and results
│   ├── baseline/         # OpenAI's reference baseline architecture
│   ├── 001-weight-decay/ # ...
│   └── each has model.yaml, train.yaml, README.md, artifacts_*/
├── scripts/              # See scripts/README.md for full docs
│   ├── run_experiment.sh # Full pipeline: train + quant + eval + archive
│   ├── run_quant.sh      # Standalone quantization
│   ├── run_eval.sh       # Standalone evaluation
│   ├── quant.py          # Quantization (int4-8, mxfp4, nvfp4)
│   ├── eval.py           # BPB evaluation (multi-GPU support)
│   └── update_results.py # Regenerate RESULTS.md from artifacts
├── runpod/
│   ├── launch.sh         # Launch experiment on RunPod via SkyPilot
│   ├── experiment.yaml   # SkyPilot task definition for RunPod
│   └── volume.yaml       # RunPod network volume config
├── RESULTS.md            # Auto-generated results table
└── pyproject.toml        # Python dependencies
```

## Quick Start

### Local (with GPUs)

```bash
# Install dependencies
uv sync

# Run an experiment
./scripts/run_experiment.sh baseline

# Run evaluation only
./scripts/run_eval.sh baseline

# Update results table from existing artifacts
python3 scripts/update_results.py
```

### RunPod (via SkyPilot)

```bash
# One-time setup
uv sync
runpod config              # enter API key
uv run sky check           # verify RunPod enabled
uv run sky volumes apply runpod/volume.yaml

# Launch with network volume (US-NE-1 only)
./runpod/launch.sh RTXPRO6000:2 baseline

# Launch ephemeral (any data center, downloads data fresh)
./runpod/launch.sh H100-SXM:8 baseline --ephemeral --retry

# Interactive cluster (SSH in, tear down manually)
./runpod/launch.sh RTXPRO6000:2 --interactive

# Monitor
uv run sky logs golf-research -f
uv run sky queue golf-research

# SSH into running cluster
uv run ssh golf-research

# Tear down
uv run sky down golf-research -y
```

### Experiment Script Options

See [scripts/README.md](scripts/README.md) for full documentation.

```bash
# Full run (train + quant + eval + archive + push)
./scripts/run_experiment.sh baseline

# Skip training (re-quant/eval existing checkpoint)
./scripts/run_experiment.sh baseline --no-train --artifacts-dir artifacts_8x_h100

# Skip eval or quant
./scripts/run_experiment.sh baseline --no-eval
./scripts/run_experiment.sh baseline --no-quant

# Override training config
./scripts/run_experiment.sh baseline --set training.pre_training.max_wallclock_seconds=30

# Limit eval batches
./scripts/run_experiment.sh baseline --max-eval-batches 50

# Archive existing artifacts only (no training)
./scripts/run_experiment.sh baseline --archive-only

# Regenerate README for existing artifacts
./scripts/run_experiment.sh baseline --readme-only artifacts_8x_rtx_pro_6000
```

## Creating a New Experiment

1. Create a new directory under `experiments/` with a numeric prefix (e.g. `007-my-change`)
2. Copy `model.yaml` and `train.yaml` from baseline, make your change
3. Add a `README.md` describing the change, source, and expected impact
4. Run: `./runpod/launch.sh RTXPRO6000:8 007-my-change`

Each experiment should change exactly one thing from baseline to isolate its effect.

## License

[Apache 2.0](LICENSE)
