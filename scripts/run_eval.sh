#!/usr/bin/env bash
set -euo pipefail

# Run evaluation on experiment artifacts.
#
# Evaluates both full-precision and quantized checkpoints if present.
# Supports intN, mxfp4, and nvfp4 quantization schemes.
#
# Usage:
#   ./scripts/run_eval.sh <experiment> [artifacts_dir] [--schemes S,S,...] [-- eval.py options...]
#
# Examples:
#   ./scripts/run_eval.sh baseline                                                  # all schemes
#   ./scripts/run_eval.sh baseline artifacts_8x_h100                                # archived
#   ./scripts/run_eval.sh baseline artifacts_tmp --schemes int6,nvfp4               # specific schemes
#   ./scripts/run_eval.sh baseline artifacts_tmp --schemes int8 -- --max-batches 50 # with eval opts

EXPERIMENT=""
ARTIFACTS_NAME=""
SCHEMES=""
EVAL_ARGS=()
PARSING_EVAL_ARGS=false

while [[ $# -gt 0 ]]; do
    if [[ "$PARSING_EVAL_ARGS" == true ]]; then
        EVAL_ARGS+=("$1")
        shift
        continue
    fi
    case "$1" in
        --schemes) SCHEMES="$2"; shift 2 ;;
        --bits) SCHEMES="$(echo "$2" | sed 's/\([0-9]\+\)/int\1/g')"; shift 2 ;; # backwards compat
        --) PARSING_EVAL_ARGS=true; shift ;;
        *)
            if [[ -z "$EXPERIMENT" ]]; then
                EXPERIMENT="$1"
            else
                ARTIFACTS_NAME="$1"
            fi
            shift ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Usage: $0 <experiment> [artifacts_dir] [--schemes S,S,...] [-- eval.py options...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_DIR="$ROOT_DIR/experiments/$EXPERIMENT"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory '$EXPERIMENT_DIR' does not exist"
    exit 1
fi

# --- Detect GPUs ---
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    NUM_GPUS=1
fi

# --- NCCL workarounds for Blackwell GPUs ---
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
unset NCCL_ASYNC_ERROR_HANDLING
unset NCCL_AVOID_RECORD_STREAMS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$EXPERIMENT_DIR"

# Tee output to console.log in the artifacts directory
if [[ -n "$ARTIFACTS_NAME" ]]; then
    LOG_DIR="$EXPERIMENT_DIR/$ARTIFACTS_NAME"
else
    LOG_DIR="$EXPERIMENT_DIR/artifacts"
fi
mkdir -p "$LOG_DIR"
export FORCE_COLOR=1
exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_DIR/console.log")) 2>&1

# Pass --schemes filter if specified
if [[ -n "$SCHEMES" ]]; then
    EVAL_ARGS=("--schemes" "$SCHEMES" "${EVAL_ARGS[@]}")
fi

# Auto-detect local parameter-golf data paths (train.yaml points to /workspace on RunPod)
PARAMETER_GOLF_DIR="$HOME/github/parameter-golf/data"
if [[ -d "$PARAMETER_GOLF_DIR" ]]; then
    VAL_PATTERN="$PARAMETER_GOLF_DIR/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
    TOKENIZER="$PARAMETER_GOLF_DIR/tokenizers/fineweb_1024_bpe.model"
    if compgen -G "$VAL_PATTERN" > /dev/null 2>&1; then
        EVAL_ARGS=("--val-data" "$VAL_PATTERN" "${EVAL_ARGS[@]}")
    fi
    if [[ -f "$TOKENIZER" ]]; then
        EVAL_ARGS=("--tokenizer" "$TOKENIZER" "${EVAL_ARGS[@]}")
    fi
fi

# Build checkpoint and report paths
if [[ -n "$ARTIFACTS_NAME" ]]; then
    ARTIFACTS_DIR="$EXPERIMENT_DIR/$ARTIFACTS_NAME"
    if [[ ! -d "$ARTIFACTS_DIR" ]]; then
        echo "Error: artifacts directory '$ARTIFACTS_DIR' does not exist"
        exit 1
    fi
    EVAL_ARGS=("--checkpoint" "$ARTIFACTS_DIR/checkpoint" "--report" "$ARTIFACTS_DIR/eval_report.json" "${EVAL_ARGS[@]}")
    echo "Running evaluation: $EXPERIMENT ($ARTIFACTS_NAME, ${NUM_GPUS} GPU(s))"
else
    echo "Running evaluation: $EXPERIMENT (${NUM_GPUS} GPU(s))"
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
    torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/eval.py" "${EVAL_ARGS[@]}"
else
    python3 "$SCRIPT_DIR/eval.py" "${EVAL_ARGS[@]}"
fi
