#!/usr/bin/env bash

# Run a training experiment and archive artifacts with hardware-tagged naming.
#
# Pipeline: train -> quant -> eval -> archive -> readme -> commit
#
# On failure at any stage, the script archives whatever artifacts exist
# with a _failed suffix and pushes to git for remote diagnosis.
#
# Usage:
#   ./scripts/run_experiment.sh <experiment> [options]
#
# Options:
#   --no-train             Skip training (use existing checkpoint)
#   --no-eval              Skip evaluation
#   --no-quant             Skip quantization
#   --no-push              Skip git commit and push
#   --quant-schemes S,S    Quant schemes (default: int6,int8,mxfp4,nvfp4)
#   --set key=value        Override training config (can be repeated)
#   --max-eval-batches N   Limit eval to N batches
#   --artifacts-dir DIR    Work on a specific artifacts dir (e.g. artifacts_8x_h100)
#   --archive-only         Skip training/quant/eval, just archive existing artifacts
#   --readme-only DIR      Regenerate README for an existing artifacts dir

TRAIN=true
EVAL=true
QUANT=true
PUSH=true
ARCHIVE_ONLY=false
README_ONLY=false
README_TARGET=""
ARTIFACTS_NAME=""
QUANT_SCHEMES=int6,int8,mxfp4,nvfp4
EXPERIMENT=""
EXTRA_ARGS=()
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-train) TRAIN=false; shift ;;
        --no-eval) EVAL=false; shift ;;
        --no-quant) QUANT=false; shift ;;
        --no-push) PUSH=false; shift ;;
        --push-only) TRAIN=false; QUANT=false; EVAL=false; ARCHIVE_ONLY=true; shift ;;
        --archive-only) ARCHIVE_ONLY=true; shift ;;
        --readme-only) README_ONLY=true; README_TARGET="$2"; shift 2 ;;
        --artifacts-dir) ARTIFACTS_NAME="$2"; shift 2 ;;
        --quant-schemes) QUANT_SCHEMES="$2"; shift 2 ;;
        --set) EXTRA_ARGS+=("--set" "$2"); shift 2 ;;
        --max-eval-batches) EVAL_ARGS+=("--max-batches" "$2"); shift 2 ;;
        *) EXPERIMENT="$1"; shift ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Usage: $0 <experiment> [options]"
    echo ""
    echo "Options:"
    echo "  --no-train             Skip training (use existing checkpoint)"
    echo "  --no-eval              Skip evaluation"
    echo "  --no-quant             Skip quantization"
    echo "  --no-push              Skip git commit and push"
    echo "  --quant-schemes S,S    Quant schemes (default: int6,int8,mxfp4,nvfp4)"
    echo "  --set key=value        Override training config"
    echo "  --max-eval-batches N   Limit eval to N batches"
    echo "  --artifacts-dir DIR    Work on a specific artifacts dir (e.g. artifacts_8x_h100)"
    echo "  --archive-only         Archive existing artifacts (skip training)"
    echo "  --readme-only DIR      Regenerate README for an existing artifacts dir"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_DIR="$ROOT_DIR/experiments/$EXPERIMENT"

# Ensure we start from a clean state on main
cd "$ROOT_DIR"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Warning: not on main (on $CURRENT_BRANCH), switching to main"
    git checkout main
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Warning: working tree is dirty, stashing changes"
    git stash
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Error: experiment directory '$EXPERIMENT_DIR' does not exist"
    exit 1
fi

# --- Handle --readme-only ---
if [[ "$README_ONLY" == true ]]; then
    TARGET_DIR="$EXPERIMENT_DIR/$README_TARGET"
    if [[ ! -d "$TARGET_DIR" ]]; then
        echo "Error: artifacts directory '$TARGET_DIR' does not exist"
        exit 1
    fi
    python3 "$SCRIPT_DIR/gen_artifact_readme.py" "$TARGET_DIR" "$EXPERIMENT_DIR" "$ROOT_DIR"
    echo "README generated: $TARGET_DIR/README.md"
    exit 0
fi

if [[ "$TRAIN" == true ]] && [[ ! -f "$EXPERIMENT_DIR/train.yaml" ]]; then
    echo "Error: no train.yaml found in '$EXPERIMENT_DIR'"
    exit 1
fi

export FORCE_COLOR=1

# --- Detect GPUs ---
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    NUM_GPUS=1
fi

# NCCL workarounds for Blackwell GPUs (RTX PRO 6000 etc.)
# Only apply on Blackwell - these cripple NVLink on H100/A100.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
    *GB10*|*RTX*PRO*6000*|*RTX*6000*|*B200*|*B100*)
        export NCCL_P2P_DISABLE=1
        export NCCL_SHM_DISABLE=1
        echo "Blackwell GPU detected ($GPU_NAME) — applying NCCL workarounds"
        ;;
    *)
        echo "Non-Blackwell GPU ($GPU_NAME) — NCCL defaults"
        ;;
esac
export TORCHINDUCTOR_COMPILE_THREADS=1
unset NCCL_ASYNC_ERROR_HANDLING
unset NCCL_AVOID_RECORD_STREAMS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$EXPERIMENT_DIR"

# --- Auto-detect GPU type and set batch_size ---
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$GPU_NAME" in
        # *H100*|*H200*)
        #     AUTO_ARGS=("--set" "training.pre_training.batch_size=256")
        #     echo "Detected $GPU_NAME — batch_size=256"
        #     ;;
        *GB10*)
            AUTO_ARGS=("--set" "training.pre_training.batch_size=16")
            echo "Detected $GPU_NAME — batch_size=16"
            ;;
        *RTX*PRO*6000*|*RTX*6000*)
            AUTO_ARGS=("--set" "training.pre_training.batch_size=80")
            echo "Detected $GPU_NAME — batch_size=80"
            ;;
        *)
            AUTO_ARGS=()
            echo "GPU: $GPU_NAME — using default batch_size"
            ;;
    esac
else
    AUTO_ARGS=()
fi

# Resolve working artifacts directory
if [[ -n "$ARTIFACTS_NAME" ]]; then
    ARTIFACTS_DIR="$EXPERIMENT_DIR/$ARTIFACTS_NAME"
    if [[ ! -d "$ARTIFACTS_DIR" ]]; then
        echo "Error: artifacts directory '$ARTIFACTS_DIR' does not exist"
        exit 1
    fi
    USE_NAMED_ARTIFACTS=true
else
    ARTIFACTS_DIR="$EXPERIMENT_DIR/artifacts"
    USE_NAMED_ARTIFACTS=false
fi

archive_and_push() {
    local SUFFIX="${1:-}"

    if [[ ! -d "$ARTIFACTS_DIR" ]]; then
        echo "No artifacts to archive"
        return
    fi

    # Parse GPU info for folder naming (best-effort)
    local GPU_INFO=""
    if [[ -f "$ARTIFACTS_DIR/system.json" ]]; then
        GPU_INFO=$(python3 -c "
import json, re, sys
with open('$ARTIFACTS_DIR/system.json') as f:
    info = json.load(f)
gpus = info.get('gpus', [])
world_size = info.get('distributed', {}).get('world_size', 1)
if not gpus:
    sys.exit(0)
name = gpus[0]['name'].upper().replace('NVIDIA', '').strip()
name = re.sub(r'\d+\s*GB.*', '', name).strip()
name = re.sub(r'BLACKWELL.*', '', name).strip()
name = re.sub(r'SERVER.*', '', name).strip()
name = re.sub(r'EDITION.*', '', name).strip()
name = re.sub(r'GEFORCE\s*', '', name).strip()
name = re.sub(r'\s+', '_', name).lower().strip('_')
print(f'{world_size}x_{name}')
" 2>/dev/null || true)
    fi
    if [[ -z "$GPU_INFO" ]]; then
        GPU_INFO="${NUM_GPUS}x_unknown"
    fi

    echo "Hardware: $GPU_INFO"

    local TARGET_DIR="$ARTIFACTS_DIR"

    if [[ "$USE_NAMED_ARTIFACTS" == false ]]; then
        # Determine target folder name
        local TARGET_BASE="$EXPERIMENT_DIR/artifacts_${GPU_INFO}${SUFFIX}"
        TARGET_DIR="$TARGET_BASE"
        if [[ -d "$TARGET_DIR" ]]; then
            local INDEX=2
            while [[ -d "${TARGET_BASE}_${INDEX}" ]]; do
                ((INDEX++))
            done
            TARGET_DIR="${TARGET_BASE}_${INDEX}"
        fi

        echo "Archiving artifacts to: $(basename "$TARGET_DIR")"

        # Compress only the full-precision checkpoint (quant files are ephemeral)
        local CHECKPOINT_DIR="$ARTIFACTS_DIR/checkpoint"
        if [[ -d "$CHECKPOINT_DIR" ]]; then
            find "$CHECKPOINT_DIR" -type f | grep -E '_(int[0-9]+|mxfp[0-9]+|nvfp[0-9]+|turboquip[0-9]+[a-z]*)$' | xargs rm -f 2>/dev/null || true
            echo "Compressing checkpoint..."
            tar -czf "$ARTIFACTS_DIR/checkpoint.tar.gz" -C "$ARTIFACTS_DIR" checkpoint
            rm -rf "$CHECKPOINT_DIR"
            echo "Checkpoint compressed to checkpoint.tar.gz"
        fi

        # Rename artifacts folder
        mv "$ARTIFACTS_DIR" "$TARGET_DIR"
        echo "Artifacts moved to $(basename "$TARGET_DIR")"

        # Upload checkpoint to HuggingFace
        local HF_REPO="k14r/golf-research-artifacts"
        local HF_PATH="experiments/${EXPERIMENT}/$(basename "$TARGET_DIR")/checkpoint.tar.gz"
        if [[ -f "$TARGET_DIR/checkpoint.tar.gz" ]] && python3 -c "import huggingface_hub" &>/dev/null 2>&1; then
            echo "Uploading checkpoint to HuggingFace ($HF_REPO)..."
            python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
api.upload_file(path_or_fileobj='$TARGET_DIR/checkpoint.tar.gz', path_in_repo='$HF_PATH', repo_id='$HF_REPO')
print('Upload complete')
" || true
        fi
    fi

    # Generate artifact README
    python3 "$SCRIPT_DIR/gen_artifact_readme.py" "$TARGET_DIR" "$EXPERIMENT_DIR" "$ROOT_DIR" || true

    if [[ "$PUSH" == true ]]; then
        cd "$ROOT_DIR"
        python3 "$SCRIPT_DIR/update_results.py" || true

        local RELATIVE_TARGET="experiments/${EXPERIMENT}/$(basename "$TARGET_DIR")"
        local RAND_SUFFIX=$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')
        local BRANCH_NAME="results/${EXPERIMENT}-${GPU_INFO}${SUFFIX}-${RAND_SUFFIX}"
        local COMMIT_MSG="Add $EXPERIMENT results ($GPU_INFO${SUFFIX})"
        local IS_FAILURE=false
        [[ "$SUFFIX" == *"_failed"* ]] && IS_FAILURE=true

        git fetch origin main 2>/dev/null || true
        git checkout -b "$BRANCH_NAME" origin/main 2>/dev/null || git checkout -b "$BRANCH_NAME"
        git add "$RELATIVE_TARGET" RESULTS.md 2>/dev/null || true
        # Exclude checkpoint binaries (uploaded to HuggingFace instead)
        git reset HEAD "$RELATIVE_TARGET/checkpoint.tar.gz" 2>/dev/null || true
        git commit -m "$COMMIT_MSG" || true
        git push -u origin "$BRANCH_NAME" || true

        if [[ "$IS_FAILURE" == false ]] && command -v gh &>/dev/null; then
            local README_FILE="$TARGET_DIR/README.md"
            local PR_BODY=""
            if [[ -f "$README_FILE" ]]; then
                PR_BODY=$(cat "$README_FILE")
            else
                PR_BODY="Results for $EXPERIMENT on $GPU_INFO"
            fi
            gh pr create \
                --base main \
                --head "$BRANCH_NAME" \
                --title "$COMMIT_MSG" \
                --body "$PR_BODY" || true
        elif [[ "$IS_FAILURE" == true ]]; then
            echo "Run failed — branch pushed but no PR created"
        fi

        git checkout -f main 2>/dev/null || git checkout -f -
    else
        echo "Skipping git push (--no-push)"
    fi
}

# Trap errors: archive whatever exists with _failed suffix
trap 'echo "ERROR: Experiment failed at line $LINENO"; archive_and_push "_failed"; cd "$ROOT_DIR"; git checkout -f main 2>/dev/null || true' ERR

# Enable exit-on-error now that the trap is set
set -euo pipefail

if [[ "$ARCHIVE_ONLY" == false ]]; then
    if [[ "$USE_NAMED_ARTIFACTS" == false ]]; then
        # --- Clean previous artifacts ---
        if [[ "$TRAIN" == true ]] && [[ -d "$ARTIFACTS_DIR" ]]; then
            echo "Removing existing artifacts/"
            rm -rf "$ARTIFACTS_DIR"
        fi

        # --- Tee all output to artifacts/console.log ---
        mkdir -p "$ARTIFACTS_DIR"
        exec > >(tee >(sed 's/\x1b\[[0-9;]*m//g' >> "$ARTIFACTS_DIR/console.log")) 2>&1

        # --- Save overrides ---
        ALL_SET_ARGS=("${AUTO_ARGS[@]}" "${EXTRA_ARGS[@]}")
        if [[ ${#ALL_SET_ARGS[@]} -gt 0 ]]; then
            {
                echo "# Runtime overrides applied via --set"
                for arg in "${ALL_SET_ARGS[@]}"; do
                    if [[ "$arg" != "--set" ]]; then
                        key="${arg%%=*}"
                        val="${arg#*=}"
                        echo "${key}: ${val}"
                    fi
                done
            } > "$ARTIFACTS_DIR/overrides.yaml"
            echo "Overrides saved to artifacts/overrides.yaml"
        fi
    fi

    # --- Detect sweep mode ---
    IS_SWEEP=false
    if [[ -f "$EXPERIMENT_DIR/sweep.yaml" ]]; then
        IS_SWEEP=true
    fi

    # --- Run training ---
    if [[ "$TRAIN" == true ]]; then
        echo "Running experiment: $EXPERIMENT (${NUM_GPUS} GPU(s))${IS_SWEEP:+ [sweep]}"
        if [[ "$NUM_GPUS" -gt 1 ]]; then
            torchrun --nproc_per_node="$NUM_GPUS" -m composer.train --config train.yaml "${ALL_SET_ARGS[@]}" || true
        else
            python3 -m composer.train --config train.yaml "${ALL_SET_ARGS[@]}"
        fi

        if [[ "$IS_SWEEP" == true ]]; then
            # Verify at least one variant produced a checkpoint
            VARIANT_COUNT=0
            for VARIANT_DIR in "$EXPERIMENT_DIR"/[0-9]-*/; do
                [[ -d "$VARIANT_DIR/artifacts/checkpoint" ]] && ((VARIANT_COUNT++)) || true
            done
            if [[ "$VARIANT_COUNT" -eq 0 ]]; then
                echo "Error: no sweep variants produced checkpoints"
                exit 1
            fi
            echo "Sweep complete - $VARIANT_COUNT variant(s) with checkpoints"
        else
            # Verify checkpoint exists
            if [[ ! -d "$ARTIFACTS_DIR/checkpoint" ]]; then
                echo "Error: no checkpoint directory found in artifacts/ - training failed"
                exit 1
            fi
            CHECKPOINT_FILES=$(find "$ARTIFACTS_DIR/checkpoint" -type f | head -1)
            if [[ -z "$CHECKPOINT_FILES" ]]; then
                echo "Error: checkpoint directory is empty - training failed"
                exit 1
            fi
            echo "Checkpoint found - training succeeded"
        fi
    else
        echo "Skipping training (--no-train)"
    fi

    if [[ "$IS_SWEEP" == true ]]; then
        # --- Process each sweep variant ---
        # Defer push until all variants are archived
        SAVED_PUSH="$PUSH"
        PUSH=false

        for VARIANT_DIR in "$EXPERIMENT_DIR"/[0-9]-*/; do
            [[ -d "$VARIANT_DIR/artifacts" ]] || continue
            VARIANT_NAME=$(basename "$VARIANT_DIR")
            echo ""
            echo "===== Processing sweep variant: $VARIANT_NAME ====="

            VARIANT_ARTIFACTS="$VARIANT_DIR/artifacts"

            # Copy runtime overrides into variant artifacts for README generation
            if [[ -f "$ARTIFACTS_DIR/overrides.yaml" ]] && [[ ! -f "$VARIANT_ARTIFACTS/overrides.yaml" ]]; then
                cp "$ARTIFACTS_DIR/overrides.yaml" "$VARIANT_ARTIFACTS/overrides.yaml"
            fi

            # Quantize
            if [[ "$QUANT" == true ]] && [[ -d "$VARIANT_ARTIFACTS/checkpoint" ]]; then
                echo "Running quantization (schemes=$QUANT_SCHEMES)..."
                python3 "$SCRIPT_DIR/quant.py" --checkpoint "$VARIANT_ARTIFACTS/checkpoint" --schemes "$QUANT_SCHEMES" || true
            fi

            # Eval
            if [[ "$EVAL" == true ]]; then
                echo "Running evaluation: $VARIANT_NAME (${NUM_GPUS} GPU(s))"
                EVAL_CMD_ARGS=("--checkpoint" "$VARIANT_ARTIFACTS/checkpoint" "--report" "$VARIANT_ARTIFACTS/eval_report.json")
                _PG_DIR="$HOME/github/parameter-golf/data"
                if [[ -d "$_PG_DIR" ]]; then
                    _VAL="$_PG_DIR/datasets/fineweb10B_sp1024/fineweb_val_*.bin"
                    _TOK="$_PG_DIR/tokenizers/fineweb_1024_bpe.model"
                    compgen -G "$_VAL" > /dev/null 2>&1 && EVAL_CMD_ARGS+=("--val-data" "$_VAL")
                    [[ -f "$_TOK" ]] && EVAL_CMD_ARGS+=("--tokenizer" "$_TOK")
                fi
                EVAL_CMD_ARGS+=("${EVAL_ARGS[@]}")
                if [[ "$NUM_GPUS" -gt 1 ]]; then
                    torchrun --nproc_per_node="$NUM_GPUS" "$SCRIPT_DIR/eval.py" "${EVAL_CMD_ARGS[@]}" || true
                else
                    python3 "$SCRIPT_DIR/eval.py" "${EVAL_CMD_ARGS[@]}" || true
                fi
            fi

            # Archive variant (no push yet)
            SAVED_ARTIFACTS_DIR="$ARTIFACTS_DIR"
            SAVED_USE_NAMED="$USE_NAMED_ARTIFACTS"
            SAVED_EXPERIMENT_DIR="$EXPERIMENT_DIR"
            ARTIFACTS_DIR="$VARIANT_ARTIFACTS"
            USE_NAMED_ARTIFACTS=false
            EXPERIMENT_DIR="$VARIANT_DIR"
            EXPERIMENT="$EXPERIMENT/$VARIANT_NAME"
            archive_and_push ""
            EXPERIMENT="${EXPERIMENT%/*}"
            EXPERIMENT_DIR="$SAVED_EXPERIMENT_DIR"
            ARTIFACTS_DIR="$SAVED_ARTIFACTS_DIR"
            USE_NAMED_ARTIFACTS="$SAVED_USE_NAMED"
        done

        PUSH="$SAVED_PUSH"

        # Rename root artifacts/ (console.log, overrides) to artifacts_<gpu>
        if [[ -d "$ARTIFACTS_DIR" ]] && [[ "$USE_NAMED_ARTIFACTS" == false ]]; then
            # Reuse GPU info from first variant
            FIRST_SYSTEM=$(find "$EXPERIMENT_DIR"/[0-9]-*/artifacts_*/system.json 2>/dev/null | head -1)
            if [[ -n "$FIRST_SYSTEM" ]]; then
                ROOT_GPU_TAG=$(python3 -c "
import json, re, sys
with open('$FIRST_SYSTEM') as f: info = json.load(f)
gpus = info.get('gpus', [])
ws = info.get('distributed', {}).get('world_size', 1)
if not gpus: sys.exit(0)
name = gpus[0]['name'].upper().replace('NVIDIA','').strip()
for pat in [r'\d+\s*GB.*', r'BLACKWELL.*', r'SERVER.*', r'EDITION.*', r'GEFORCE\s*']:
    name = re.sub(pat, '', name).strip()
name = re.sub(r'\s+', '_', name).lower().strip('_')
print(f'{ws}x_{name}')
" 2>/dev/null || echo "${NUM_GPUS}x_unknown")
            else
                ROOT_GPU_TAG="${NUM_GPUS}x_unknown"
            fi
            ROOT_ARTIFACTS="$EXPERIMENT_DIR/artifacts_${ROOT_GPU_TAG}"
            if [[ ! -d "$ROOT_ARTIFACTS" ]]; then
                mv "$ARTIFACTS_DIR" "$ROOT_ARTIFACTS"
            fi
        fi

        # Single combined push for all sweep variants
        if [[ "$PUSH" == true ]]; then
            cd "$ROOT_DIR"
            python3 "$SCRIPT_DIR/update_results.py" || true

            RAND_SUFFIX=$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')
            BRANCH_NAME="results/${EXPERIMENT}-${ROOT_GPU_TAG:-${NUM_GPUS}x}-${RAND_SUFFIX}"
            COMMIT_MSG="Add $EXPERIMENT sweep results (${ROOT_GPU_TAG:-${NUM_GPUS}x})"

            git fetch origin main 2>/dev/null || true
            git checkout -b "$BRANCH_NAME" origin/main 2>/dev/null || git checkout -b "$BRANCH_NAME"
            git add "experiments/${EXPERIMENT}"/[0-9]-*/artifacts_*/ "experiments/${EXPERIMENT}"/[0-9]-*/overrides.yaml RESULTS.md 2>/dev/null || true
            git add "experiments/${EXPERIMENT}"/artifacts_*/ 2>/dev/null || true
            # Exclude checkpoint binaries
            git reset HEAD "experiments/${EXPERIMENT}"/[0-9]-*/artifacts_*/checkpoint.tar.gz 2>/dev/null || true
            git commit -m "$COMMIT_MSG" || true
            git push -u origin "$BRANCH_NAME" || true

            if command -v gh &>/dev/null; then
                gh pr create --base main --head "$BRANCH_NAME" \
                    --title "$COMMIT_MSG" \
                    --body "Sweep results for $EXPERIMENT" || true
            fi

            git checkout -f main 2>/dev/null || git checkout -f -
        else
            echo "Skipping git push (--no-push)"
        fi

        echo "Done - sweep results committed and pushed"
    else
        # --- Single experiment flow ---

        # Quantize
        if [[ "$QUANT" == true ]]; then
            echo "Running quantization (schemes=$QUANT_SCHEMES)..."
            if [[ "$USE_NAMED_ARTIFACTS" == true ]]; then
                python3 "$SCRIPT_DIR/quant.py" --checkpoint "$ARTIFACTS_DIR/checkpoint" --schemes "$QUANT_SCHEMES"
            else
                python3 "$SCRIPT_DIR/quant.py" --schemes "$QUANT_SCHEMES"
            fi
        fi

        # Eval
        if [[ "$EVAL" == true ]]; then
            if [[ "$USE_NAMED_ARTIFACTS" == true ]]; then
                "$SCRIPT_DIR/run_eval.sh" "$EXPERIMENT" "$(basename "$ARTIFACTS_DIR")" "${EVAL_ARGS[@]}"
            else
                "$SCRIPT_DIR/run_eval.sh" "$EXPERIMENT" "${EVAL_ARGS[@]}"
            fi
        fi

        # Archive and push
        trap - ERR
        archive_and_push ""
        echo "Done - results committed and pushed"
    fi
else
    echo "Archive-only mode: skipping training"
    IS_SWEEP=false
    [[ -f "$EXPERIMENT_DIR/sweep.yaml" ]] && IS_SWEEP=true

    if [[ "$IS_SWEEP" == true ]]; then
        SAVED_PUSH="$PUSH"
        PUSH=false
        for VARIANT_DIR in "$EXPERIMENT_DIR"/[0-9]-*/; do
            VARIANT_ARTIFACTS=""
            if [[ -d "$VARIANT_DIR/artifacts" ]]; then
                VARIANT_ARTIFACTS="$VARIANT_DIR/artifacts"
            else
                for ad in "$VARIANT_DIR"/artifacts_*/; do
                    [[ -d "$ad" ]] && VARIANT_ARTIFACTS="$ad" && break
                done
            fi
            [[ -z "$VARIANT_ARTIFACTS" ]] && continue
            VARIANT_NAME=$(basename "$VARIANT_DIR")
            echo ""
            echo "===== Archiving sweep variant: $VARIANT_NAME ====="
            SAVED_ARTIFACTS_DIR="$ARTIFACTS_DIR"
            SAVED_USE_NAMED="$USE_NAMED_ARTIFACTS"
            SAVED_EXPERIMENT_DIR="$EXPERIMENT_DIR"
            ARTIFACTS_DIR="$VARIANT_ARTIFACTS"
            USE_NAMED_ARTIFACTS=$([[ "$VARIANT_ARTIFACTS" == */artifacts ]] && echo false || echo true)
            EXPERIMENT_DIR="$VARIANT_DIR"
            EXPERIMENT="$EXPERIMENT/$VARIANT_NAME"
            archive_and_push ""
            EXPERIMENT="${EXPERIMENT%/*}"
            EXPERIMENT_DIR="$SAVED_EXPERIMENT_DIR"
            ARTIFACTS_DIR="$SAVED_ARTIFACTS_DIR"
            USE_NAMED_ARTIFACTS="$SAVED_USE_NAMED"
        done
        PUSH="$SAVED_PUSH"
        # Reuse the sweep push block from the training path
        if [[ "$PUSH" == true ]]; then
            cd "$ROOT_DIR"
            python3 "$SCRIPT_DIR/update_results.py" || true
            RAND_SUFFIX=$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')
            BRANCH_NAME="results/${EXPERIMENT}-${NUM_GPUS}x-${RAND_SUFFIX}"
            COMMIT_MSG="Add $EXPERIMENT sweep results"
            git fetch origin main 2>/dev/null || true
            git checkout -b "$BRANCH_NAME" origin/main 2>/dev/null || git checkout -b "$BRANCH_NAME"
            git add "experiments/${EXPERIMENT}"/[0-9]-*/artifacts_*/ "experiments/${EXPERIMENT}"/[0-9]-*/overrides.yaml RESULTS.md 2>/dev/null || true
            git reset HEAD "experiments/${EXPERIMENT}"/[0-9]-*/artifacts_*/checkpoint.tar.gz 2>/dev/null || true
            git commit -m "$COMMIT_MSG" || true
            git push -u origin "$BRANCH_NAME" || true
            if command -v gh &>/dev/null; then
                gh pr create --base main --head "$BRANCH_NAME" --title "$COMMIT_MSG" --body "Sweep results for $EXPERIMENT" || true
            fi
            git checkout -f main 2>/dev/null || git checkout -f -
        fi
    else
        if [[ ! -d "$ARTIFACTS_DIR" ]]; then
            echo "Error: no artifacts directory found at '$ARTIFACTS_DIR'"
            exit 1
        fi
        archive_and_push ""
    fi
    echo "Done - results committed and pushed"
fi
