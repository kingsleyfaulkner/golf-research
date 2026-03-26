#!/usr/bin/env bash
set -euo pipefail

# Launch a training experiment on RunPod via SkyPilot.
#
# Usage:
#   ./scripts/runpod_launch.sh <accelerators> [experiment] [--retry]
#
# Accelerator names (from `uv run sky gpus list --cloud runpod`):
#   RTXPRO6000, H100-SXM, H100, H100-NVL, L40S, A100-80GB, etc.
#
# Examples:
#   ./scripts/runpod_launch.sh RTXPRO6000:2 baseline
#   ./scripts/runpod_launch.sh H100-SXM:8 baseline --retry
#   ./scripts/runpod_launch.sh H100-SXM:2 baseline


ACCELERATORS=""
EXPERIMENT="baseline"
EPHEMERAL=false
INTERACTIVE=false
EXTRA_FLAGS=()
SET_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retry-until-up|--retry) EXTRA_FLAGS+=("--retry-until-up"); shift ;;
        --ephemeral) EPHEMERAL=true; shift ;;
        --interactive|-i) INTERACTIVE=true; shift ;;
        --set) SET_ARGS+=("--set" "$2"); shift 2 ;;
        *) if [[ -z "$ACCELERATORS" ]]; then ACCELERATORS="$1"; else EXPERIMENT="$1"; fi; shift ;;
    esac
done

if [[ -z "$ACCELERATORS" ]]; then
    echo "Usage: $0 <accelerators> [experiment] [--retry] [--ephemeral] [--interactive]"
    echo ""
    echo "Examples:"
    echo "  $0 RTXPRO6000:2 baseline"
    echo "  $0 H100-SXM:8 baseline --retry"
    echo "  $0 H100-SXM:8 baseline --ephemeral --retry"
    echo "  $0 RTXPRO6000:2 --interactive    # setup only, SSH in manually"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# e.g. "RTXPRO6000:2" -> "2x-rtxpro6000"
GPU_TAG=$(echo "$ACCELERATORS" | sed 's/\(.*\):\(.*\)/\2x-\1/' | tr '[:upper:]' '[:lower:]')
JOB_NAME="${EXPERIMENT}-${GPU_TAG}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    if command -v gh &>/dev/null; then
        GITHUB_TOKEN=$(gh auth token 2>/dev/null)
    fi
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        echo "Error: GITHUB_TOKEN not set and gh CLI not available"
        exit 1
    fi
fi

if [[ "$EPHEMERAL" == true ]]; then
    TASK_YAML="${ROOT_DIR}/runpod/experiment-ephemeral.yaml"
else
    TASK_YAML="${ROOT_DIR}/runpod/experiment.yaml"
fi

ENV_FLAGS=(
    --env EXPERIMENT="${EXPERIMENT}"
    --env EXTRA_SET_ARGS="${SET_ARGS[*]}"
    --env GITHUB_TOKEN="${GITHUB_TOKEN}"
    --env HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
    --env GIT_USER_EMAIL="$(git config user.email)"
    --env GIT_USER_NAME="$(git config user.name)"
)

if [[ "$INTERACTIVE" == true ]]; then
    echo "Launching interactive cluster on ${ACCELERATORS}"
    uv run sky launch "${TASK_YAML}" \
        --cluster golf-research \
        --gpus "${ACCELERATORS}" \
        "${ENV_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}" \
        --yes
    echo ""
    echo "Cluster ready. SSH in with:  uv run ssh golf-research"
    echo "Tear down with:              uv run sky down golf-research -y"
else
    echo "Launching experiment: ${EXPERIMENT} on ${ACCELERATORS} (job: ${JOB_NAME})"
    uv run sky launch "${TASK_YAML}" \
        --cluster golf-research \
        --name "${JOB_NAME}" \
        --down \
        --idle-minutes-to-autostop 0 \
        --detach-run \
        --gpus "${ACCELERATORS}" \
        "${ENV_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}" \
        --yes
fi
