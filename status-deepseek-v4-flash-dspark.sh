#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
EXPECT="running"

usage() {
  echo "Usage: $(basename "$0") [--expect running|stopped]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --expect) shift; EXPECT="${1:-}" ;;
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
API_URL="${API_URL:-http://$VLLM_PROXY_HOST:$VLLM_PROXY_PORT/v1/models}"

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

if ! python3 - "$API_URL" "$VLLM_ORIGIN_KEY_FILE" "$SERVED_MODEL_NAME" <<'PY'
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

url, key_file, expected_model = sys.argv[1:]
try:
    urllib.request.urlopen(url, timeout=5)
except urllib.error.HTTPError as error:
    if error.code != 401:
        raise
    error.close()
else:
    raise SystemExit("unauthenticated origin request unexpectedly succeeded")

key = Path(key_file).read_text().strip()
request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
models = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
if expected_model not in models:
    raise SystemExit(f"served model mismatch: expected {expected_model}")
PY
then
  echo "Authenticated origin/model check failed." >&2
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi
echo "DSpark head, worker, authenticated proxy, pinned image, and model are healthy."
