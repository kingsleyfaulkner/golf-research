#!/usr/bin/env bash
set -euo pipefail

# Launch a training experiment on RunPod via SkyPilot.
#
# Jobs targeting the same hardware are queued onto the same cluster,
# reusing the pod and skipping provisioning/setup overhead.
#
# Usage:
#   ./runpod/launch.sh <accelerators> [experiment] [options]
#
# Accelerator names (from `uv run sky gpus list --cloud runpod`):
#   RTXPRO6000, H100-SXM, H100, H100-NVL, L40S, A100-80GB, etc.
#
# Examples:
#   ./runpod/launch.sh RTXPRO6000:2 baseline
#   ./runpod/launch.sh H100-SXM:8 baseline --retry
#   ./runpod/launch.sh H100-SXM:8 baseline --ephemeral
#   ./runpod/launch.sh --status                  # show all clusters
#   ./runpod/launch.sh --down 8x-h100-sxm        # tear down a cluster

ACCELERATORS=""
EXPERIMENT="baseline"
EPHEMERAL=false
INTERACTIVE=false
SHOW_STATUS=false
TEARDOWN_CLUSTER=""
IDLE_MINUTES=0
EXTRA_FLAGS=()
EXPERIMENT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --retry-until-up|--retry) EXTRA_FLAGS+=("--retry-until-up"); shift ;;
        --ephemeral) EPHEMERAL=true; shift ;;
        --interactive|-i) INTERACTIVE=true; shift ;;
        --status) SHOW_STATUS=true; shift ;;
        --down) TEARDOWN_CLUSTER="$2"; shift 2 ;;
        --idle-minutes) IDLE_MINUTES="$2"; shift 2 ;;
        --set) EXPERIMENT_ARGS+=("--set" "$2"); shift 2 ;;
        --quant-schemes) EXPERIMENT_ARGS+=("--quant-schemes" "$2"); shift 2 ;;
        --max-eval-batches) EXPERIMENT_ARGS+=("--max-eval-batches" "$2"); shift 2 ;;
        --no-train) EXPERIMENT_ARGS+=("--no-train"); shift ;;
        --no-eval) EXPERIMENT_ARGS+=("--no-eval"); shift ;;
        --no-quant) EXPERIMENT_ARGS+=("--no-quant"); shift ;;
        --no-push) EXPERIMENT_ARGS+=("--no-push"); shift ;;
        *) if [[ -z "$ACCELERATORS" ]]; then ACCELERATORS="$1"; else EXPERIMENT="$1"; fi; shift ;;
    esac
done

if [[ "$SHOW_STATUS" == true ]]; then
    echo "=== Clusters ==="
    uv run sky status
    echo ""
    for cluster in $(uv run sky status --output json 2>/dev/null | python3 -c "
import json, sys
clusters = json.load(sys.stdin)
for c in clusters:
    if c.get('name', '').startswith('golf-'):
        print(c['name'])
" 2>/dev/null); do
        echo "=== Job queue: ${cluster} ==="
        uv run sky queue "$cluster" 2>/dev/null || echo "  (unavailable)"
        echo ""
    done
    exit 0
fi

if [[ -n "$TEARDOWN_CLUSTER" ]]; then
    CLUSTER_NAME="golf-${TEARDOWN_CLUSTER}"
    echo "Tearing down cluster: ${CLUSTER_NAME}"
    uv run sky down "${CLUSTER_NAME}" -y
    exit 0
fi

if [[ -z "$ACCELERATORS" ]]; then
    echo "Usage: $0 <accelerators> [experiment] [options]"
    echo ""
    echo "Options:"
    echo "  --retry              Retry until resources available"
    echo "  --ephemeral          No network volume (fresh disk)"
    echo "  --interactive, -i    SSH access, manual teardown"
    echo "  --idle-minutes N     Minutes idle before auto-teardown (default: 0)"
    echo "  --set key=value      Override training config"
    echo "  --quant-schemes S,S  Quant schemes (default: int6,int8,mxfp4,nvfp4)"
    echo "  --max-eval-batches N Limit eval to N batches"
    echo "  --no-train           Skip training"
    echo "  --no-eval            Skip evaluation"
    echo "  --no-quant           Skip quantization"
    echo "  --no-push            Skip git commit and push"
    echo ""
    echo "Cluster management:"
    echo "  --status             Show all clusters and job queues"
    echo "  --down <gpu-tag>     Tear down cluster (e.g. --down 8x-h100-sxm)"
    echo ""
    echo "Examples:"
    echo "  $0 RTXPRO6000:2 baseline"
    echo "  $0 H100-SXM:8 baseline --retry"
    echo "  $0 H100-SXM:8 baseline --ephemeral --retry"
    echo "  $0 RTXPRO6000:2 baseline --no-quant --no-eval"
    echo "  $0 RTXPRO6000:2 --interactive"
    echo "  $0 --status"
    echo "  $0 --down 8x-h100-sxm"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# e.g. "RTXPRO6000:2" -> "2x-rtxpro6000"
GPU_TAG=$(echo "$ACCELERATORS" | sed 's/\(.*\):\(.*\)/\2x-\1/' | tr '[:upper:]' '[:lower:]')
JOB_NAME="${EXPERIMENT}-${GPU_TAG}"

# Cluster name encodes hardware type (and ephemeral suffix)
BASE_CLUSTER="golf-${GPU_TAG}"
CLUSTER_NAME="${BASE_CLUSTER}"
if [[ "$EPHEMERAL" == true ]]; then
    CLUSTER_NAME="${BASE_CLUSTER}-eph"
fi

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
    --env EXPERIMENT_ARGS="${EXPERIMENT_ARGS[*]}"
    --env GITHUB_TOKEN="${GITHUB_TOKEN}"
    --env HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
    --env GIT_USER_EMAIL="$(git config user.email)"
    --env GIT_USER_NAME="$(git config user.name)"
)

# Check if the cluster already exists and is UP
cluster_is_up() {
    local status
    status=$(uv run sky status "$1" --output json 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['status'])" 2>/dev/null) || return 1
    [[ "$status" == "UP" ]]
}

if [[ "$INTERACTIVE" == true ]]; then
    echo "Launching interactive cluster: ${CLUSTER_NAME} on ${ACCELERATORS}"
    uv run sky launch "${TASK_YAML}" \
        --cluster "${CLUSTER_NAME}" \
        --gpus "${ACCELERATORS}" \
        "${ENV_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}" \
        --yes
    echo ""
    echo "Cluster ready. SSH in with:  uv run sky ssh ${CLUSTER_NAME}"
    echo "Tear down with:              uv run sky down ${CLUSTER_NAME} -y"
    echo "Job queue:                   uv run sky queue ${CLUSTER_NAME}"

elif cluster_is_up "$CLUSTER_NAME"; then
    echo "Queuing job onto existing cluster: ${CLUSTER_NAME} (job: ${JOB_NAME})"
    uv run sky exec "${CLUSTER_NAME}" "${TASK_YAML}" \
        --name "${JOB_NAME}" \
        --detach-run \
        "${ENV_FLAGS[@]}"
    echo ""
    echo "Job queued. Monitor with:    uv run sky queue ${CLUSTER_NAME}"
    echo "Stream logs with:            uv run sky logs ${CLUSTER_NAME}"

elif [[ "$CLUSTER_NAME" == "$BASE_CLUSTER" ]] && cluster_is_up "${BASE_CLUSTER}-eph"; then
    # No --ephemeral flag but ephemeral cluster is running — queue onto it
    CLUSTER_NAME="${BASE_CLUSTER}-eph"
    echo "Queuing job onto existing ephemeral cluster: ${CLUSTER_NAME} (job: ${JOB_NAME})"
    uv run sky exec "${CLUSTER_NAME}" "${TASK_YAML}" \
        --name "${JOB_NAME}" \
        --detach-run \
        "${ENV_FLAGS[@]}"
    echo ""
    echo "Job queued. Monitor with:    uv run sky queue ${CLUSTER_NAME}"
    echo "Stream logs with:            uv run sky logs ${CLUSTER_NAME}"

elif [[ "$CLUSTER_NAME" == "${BASE_CLUSTER}-eph" ]] && cluster_is_up "${BASE_CLUSTER}"; then
    # --ephemeral flag but non-ephemeral cluster is running — queue onto it
    CLUSTER_NAME="${BASE_CLUSTER}"
    echo "Queuing job onto existing cluster: ${CLUSTER_NAME} (job: ${JOB_NAME})"
    uv run sky exec "${CLUSTER_NAME}" "${TASK_YAML}" \
        --name "${JOB_NAME}" \
        --detach-run \
        "${ENV_FLAGS[@]}"
    echo ""
    echo "Job queued. Monitor with:    uv run sky queue ${CLUSTER_NAME}"
    echo "Stream logs with:            uv run sky logs ${CLUSTER_NAME}"

else
    echo "Launching new cluster: ${CLUSTER_NAME} on ${ACCELERATORS} (job: ${JOB_NAME})"
    uv run sky launch "${TASK_YAML}" \
        --cluster "${CLUSTER_NAME}" \
        --name "${JOB_NAME}" \
        --down \
        --idle-minutes-to-autostop "${IDLE_MINUTES}" \
        --detach-run \
        --gpus "${ACCELERATORS}" \
        "${ENV_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}" \
        --yes
    echo ""
    echo "Queue more jobs with:        $0 ${ACCELERATORS} <experiment>"
    echo "Monitor with:                uv run sky queue ${CLUSTER_NAME}"
    echo "Stream logs with:            uv run sky logs ${CLUSTER_NAME}"
    echo "Tear down with:              $0 --down ${GPU_TAG}"
fi
