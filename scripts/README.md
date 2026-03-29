# Scripts

## run_experiment.sh

Full experiment pipeline: train → quant → eval → archive → readme → commit.

```bash
./scripts/run_experiment.sh <experiment> [options]
```

| Option | Description |
|:---|:---|
| `--no-train` | Skip training (use existing checkpoint) |
| `--no-eval` | Skip evaluation |
| `--no-quant` | Skip quantization |
| `--no-push` | Skip git commit and push |
| `--quant-schemes S,S` | Quant schemes (default: int6,int8,mxfp4,nvfp4) |
| `--set key=value` | Override training config (repeatable) |
| `--max-eval-batches N` | Limit eval to N batches |
| `--artifacts-dir DIR` | Work on a specific artifacts dir (e.g. artifacts_8x_h100) |
| `--archive-only` | Archive existing artifacts (skip training) |
| `--readme-only DIR` | Regenerate README for an existing artifacts dir |

Examples:

```bash
./scripts/run_experiment.sh baseline
./scripts/run_experiment.sh baseline --no-eval --no-push
./scripts/run_experiment.sh baseline --no-train --artifacts-dir artifacts_8x_h100
./scripts/run_experiment.sh baseline --set training.pre_training.max_wallclock_seconds=30
./scripts/run_experiment.sh baseline --readme-only artifacts_8x_rtx_pro_6000
```

GPU batch size is auto-detected (H100→256, RTX PRO 6000→80). On failure, artifacts are archived with a `_failed` suffix and pushed for diagnosis.

## run_quant.sh

Quantize a trained checkpoint. Works with both uncompressed checkpoint dirs and archived `.tar.gz` files.

```bash
./scripts/run_quant.sh <experiment> [artifacts_dir] [--schemes S,S]
```

```bash
./scripts/run_quant.sh baseline                                            # int8
./scripts/run_quant.sh baseline artifacts_8x_h100 --schemes int6,int8,mxfp4,nvfp4
```

Supported schemes: `int4`, `int5`, `int6`, `int7`, `int8`, `mxfp4`, `nvfp4`.

| Scheme | Description | Typical Size |
|:---|:---|---:|
| int8 | Per-row symmetric, 8-bit integer | ~12 MB |
| int6 | Per-row symmetric, 6-bit with bit-packing | ~12 MB |
| mxfp4 | OCP Microscaling FP4 (E2M1 + E8M0 block scales, block=32) | ~8.5 MB |
| nvfp4 | NVIDIA FP4 (E2M1 + E4M3 block scales + FP32 tensor scale, block=16) | ~9 MB |

Compressed with zstd-22 (zlib-9 fallback).

## run_eval.sh

Evaluate checkpoints. Auto-discovers all quantized files and local parameter-golf data/tokenizer paths.

```bash
./scripts/run_eval.sh <experiment> [artifacts_dir] [--schemes S,S] [-- eval.py options]
```

```bash
./scripts/run_eval.sh baseline artifacts_8x_h100                          # all schemes
./scripts/run_eval.sh baseline artifacts_8x_h100 --schemes int6,nvfp4     # specific schemes
./scripts/run_eval.sh baseline artifacts_8x_h100 -- --max-batches 50      # limit batches
```

When `--schemes` is specified, the full-precision eval is skipped.

## update_results.py

Regenerate RESULTS.md from all experiment artifacts. Reads `eval_report.json` and `quant_report.json` from each artifacts folder.

```bash
python3 scripts/update_results.py
```

## gen_artifact_readme.py

Generate a README for an artifact folder with results summary, baseline comparison, config diffs, and hardware info.

```bash
python3 scripts/gen_artifact_readme.py <artifact_dir> <experiment_dir> <root_dir>
```

Also available via `run_experiment.sh --readme-only`:

```bash
./scripts/run_experiment.sh baseline --readme-only artifacts_8x_rtx_pro_6000
```

## quant.py

Low-level quantization script. Typically called via `run_quant.sh` or `run_experiment.sh`.

```bash
python3 scripts/quant.py --schemes int6,int8,mxfp4,nvfp4
python3 scripts/quant.py --checkpoint path/to/checkpoint.tar.gz --schemes int8
```

## eval.py

Low-level evaluation script. Typically called via `run_eval.sh` or `run_experiment.sh`. Supports multi-GPU via torchrun.

```bash
python3 scripts/eval.py
python3 scripts/eval.py --schemes int6,int8 --max-batches 50
torchrun --nproc_per_node=8 scripts/eval.py
```
