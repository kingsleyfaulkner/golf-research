#!/usr/bin/env bash
# Run an experiment locally using ~/github/parameter-golf data paths.
# Wraps run_experiment.sh with the correct data path override.
#
# Usage:
#   ./scripts/run_local_experiment.sh <experiment> [options]
#
# All options are passed through to run_experiment.sh.
#
# Examples:
#   ./scripts/run_local_experiment.sh baseline
#   ./scripts/run_local_experiment.sh baseline --no-quant --no-eval
#   ./scripts/run_local_experiment.sh 008-lower-lr --no-push

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$HOME/github/parameter-golf/data/datasets/fineweb10B_sp1024"

# Activate the golf-research venv
# (torch must be pip-installed separately: see pyproject.toml comment)
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

# Source secrets for HF token etc.
[[ -f "$HOME/.secrets" ]] && source "$HOME/.secrets"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: local data not found at $DATA_DIR"
    echo "Clone and prepare: https://github.com/openai/parameter-golf"
    exit 1
fi

TOKENIZER="$HOME/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
VAL_DATA="${DATA_DIR}/fineweb_val_*.bin"

exec "$SCRIPT_DIR/run_experiment.sh" "$@" \
    --set "training.pre_training.data.TokenizedDataset.path=${DATA_DIR}/fineweb_train_*.bin" \
    --set "manifest.tokenizers.default.SentencePiece.model_path=${TOKENIZER}"
