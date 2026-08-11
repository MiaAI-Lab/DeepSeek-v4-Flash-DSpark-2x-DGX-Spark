#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
EXPECT="running"
SEMANTIC=0
SEMANTIC_DEADLINE_SECONDS=120

usage() {
  echo "Usage: $(basename "$0") [--expect running|stopped] [--semantic]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --expect) shift; EXPECT="${1:-}" ;;
    --semantic) SEMANTIC=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "$EXPECT" in running|stopped) ;; *) echo "--expect must be running or stopped" >&2; exit 2 ;; esac

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
VLLM_PROXY_HOST="${VLLM_PROXY_HOST:-172.30.0.1}"
VLLM_PROXY_PORT="${VLLM_PROXY_PORT:-8888}"
BASE_URL="${BASE_URL:-http://$VLLM_PROXY_HOST:$VLLM_PROXY_PORT/v1}"

head_ids() {
  docker ps -aq --filter "label=com.docker.compose.project=$PROJECT_NAME"
}

worker_ids() {
  ssh "$WORKER_HOST" "docker ps -aq --filter 'label=com.docker.compose.project=$PROJECT_NAME'"
}

if [ "$EXPECT" = "stopped" ]; then
  failed=0
  head_ids | grep -q . && { echo "Head resources remain." >&2; failed=1; }
  if ! worker_resources="$(worker_ids)"; then
    echo "Could not verify worker state." >&2
    failed=1
  elif [ -n "$worker_resources" ]; then
    echo "Worker resources remain." >&2
    failed=1
  fi
  [ "$failed" -eq 0 ] || exit 1
  echo "DSpark is stopped on both nodes."
  exit 0
fi

: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"
: "${VLLM_ORIGIN_KEY_FILE:?VLLM_ORIGIN_KEY_FILE must be set in $ENV_FILE}"

failed=0
for service in vllm-dspark origin-auth-proxy; do
  count="$(docker ps -q \
    --filter "label=com.docker.compose.project=$PROJECT_NAME" \
    --filter "label=com.docker.compose.service=$service" | wc -l | tr -d ' ')"
  [ "$count" = "1" ] || { echo "Head service $service is not running exactly once." >&2; failed=1; }
done
worker_count="$(ssh "$WORKER_HOST" "docker ps -q --filter 'label=com.docker.compose.project=$PROJECT_NAME' --filter 'label=com.docker.compose.service=vllm-dspark' | wc -l" | tr -d ' ')"
[ "$worker_count" = "1" ] || { echo "Worker rank is not running exactly once." >&2; failed=1; }

head_image="$(docker inspect -f '{{.Config.Image}}' "${PROJECT_NAME}-vllm-dspark-1" 2>/dev/null || true)"
worker_image="$(ssh "$WORKER_HOST" "docker inspect -f '{{.Config.Image}}' '${PROJECT_NAME}-vllm-dspark-1' 2>/dev/null" || true)"
[ "$head_image" = "$DSPARK_VLLM_IMAGE" ] || { echo "Head image mismatch." >&2; failed=1; }
[ "$worker_image" = "$DSPARK_VLLM_IMAGE" ] || { echo "Worker image mismatch." >&2; failed=1; }

runtime_marker='if isinstance(raw, dict):'
runtime_encoder='/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py'
kernel_marker='self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")'
kernel_backend='/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py'
if ! docker exec "${PROJECT_NAME}-vllm-dspark-1" grep -Fq "$runtime_marker" "$runtime_encoder"; then
  echo "Head Issue #21 runtime marker is missing." >&2
  failed=1
fi
if ! docker exec "${PROJECT_NAME}-vllm-dspark-1" grep -Fq "$kernel_marker" "$kernel_backend"; then
  echo "Head Issue #22 runtime marker is missing." >&2
  failed=1
fi
if ! ssh "$WORKER_HOST" "docker exec '${PROJECT_NAME}-vllm-dspark-1' grep -Fq '$kernel_marker' '$kernel_backend'"; then
  echo "Worker Issue #22 runtime marker is missing." >&2
  failed=1
fi
if ! ssh "$WORKER_HOST" "docker exec '${PROJECT_NAME}-vllm-dspark-1' grep -Fq '$runtime_marker' '$runtime_encoder'"; then
  echo "Worker Issue #21 runtime marker is missing." >&2
  failed=1
fi

api_ready=0
if python3 "$SCRIPT_DIR/scripts/smoke-openai-compat.py" \
    --api-liveness \
    --profile direct-origin \
    --base-url "$BASE_URL" \
    --key-file "$VLLM_ORIGIN_KEY_FILE" \
    --model "$SERVED_MODEL_NAME"; then
  api_ready=1
else
  echo "Origin API liveness failed; see classified state above." >&2
  failed=1
fi

if [ "$SEMANTIC" -eq 1 ] && [ "$api_ready" -eq 1 ]; then
  if ! python3 "$SCRIPT_DIR/scripts/smoke-openai-compat.py" \
      --semantic-canary \
      --profile direct-origin \
      --wall-timeout "$SEMANTIC_DEADLINE_SECONDS" \
      --base-url "$BASE_URL" \
      --key-file "$VLLM_ORIGIN_KEY_FILE" \
      --model "$SERVED_MODEL_NAME"; then
    echo "Semantic readiness failed; the lane was not stopped or restarted." >&2
    failed=1
  fi
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi
if [ "$SEMANTIC" -eq 1 ]; then
  echo "DSpark process/API liveness and bounded tensor-parallel generation are semantic-ready."
else
  echo "DSpark head/worker process and authenticated API liveness are healthy (generation not requested)."
fi
