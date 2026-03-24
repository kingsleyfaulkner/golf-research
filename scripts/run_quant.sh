#!/usr/bin/env bash
set -euo pipefail

# Quantize a trained model checkpoint.
#
# Supports intN (int4-int8), mxfp4, and nvfp4 quantization schemes.
# Works with both fresh artifacts/ (uncompressed checkpoint) and
# archived artifacts_*/ (checkpoint.tar.gz) directories.
#
# Usage:
#   ./scripts/run_quant.sh <experiment> [artifacts_dir] [--schemes S,S,...]
#
# Examples:
#   ./scripts/run_quant.sh baseline                                            # int8
#   ./scripts/run_quant.sh baseline artifacts_8x_h100                          # int8, archived
#   ./scripts/run_quant.sh baseline artifacts_8x_h100 --schemes int6,int8,nvfp4

SCHEMES=int8
EXPERIMENT=""
ARTIFACTS_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --schemes) SCHEMES="$2"; shift 2 ;;
        --bits) SCHEMES="$(echo "$2" | sed 's/\([0-9]\+\)/int\1/g')"; shift 2 ;; # backwards compat
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
    echo "Usage: $0 <experiment> [artifacts_dir] [--schemes S,S,...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_DIR="$ROOT_DIR/experiments/$EXPERIMENT"

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory '$EXPERIMENT_DIR' does not exist"
    exit 1
fi

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

if [[ -n "$ARTIFACTS_NAME" ]]; then
    ARTIFACTS_DIR="$EXPERIMENT_DIR/$ARTIFACTS_NAME"
    if [[ ! -d "$ARTIFACTS_DIR" ]]; then
        echo "Error: artifacts directory '$ARTIFACTS_DIR' does not exist"
        exit 1
    fi
    CHECKPOINT="$ARTIFACTS_DIR/checkpoint"
    CHECKPOINT_TAR="$ARTIFACTS_DIR/checkpoint.tar.gz"
    if [[ -d "$CHECKPOINT" ]]; then
        echo "Quantizing: $EXPERIMENT ($ARTIFACTS_NAME, schemes=$SCHEMES, uncompressed)"
        python3 "$SCRIPT_DIR/quant.py" --checkpoint "$CHECKPOINT" --schemes "$SCHEMES"
    elif [[ -f "$CHECKPOINT_TAR" ]]; then
        echo "Quantizing: $EXPERIMENT ($ARTIFACTS_NAME, schemes=$SCHEMES, compressed)"
        python3 "$SCRIPT_DIR/quant.py" --checkpoint "$CHECKPOINT_TAR" --schemes "$SCHEMES" --output-dir "$ARTIFACTS_DIR/checkpoint"
    else
        echo "Error: no checkpoint found in '$ARTIFACTS_DIR' (looked for checkpoint/ and checkpoint.tar.gz)"
        exit 1
    fi
else
    if [[ ! -d "artifacts/checkpoint" ]]; then
        echo "Error: no artifacts/checkpoint directory found - run training first"
        exit 1
    fi
    echo "Quantizing: $EXPERIMENT (schemes=$SCHEMES)"
    python3 "$SCRIPT_DIR/quant.py" --schemes "$SCHEMES"
fi
