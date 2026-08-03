#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
RUNS="${RUNS:-2}"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${VLLM_ORIGIN_KEY_FILE:?VLLM_ORIGIN_KEY_FILE must be set in $ENV_FILE}"
VLLM_PROXY_HOST="${VLLM_PROXY_HOST:-172.30.0.1}"
VLLM_PROXY_PORT="${VLLM_PROXY_PORT:-8888}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"

if ! python3 "$SCRIPT_DIR/scripts/smoke-openai-compat.py" \
    --profile direct \
    --base-url "http://$VLLM_PROXY_HOST:$VLLM_PROXY_PORT/v1" \
    --key-file "$VLLM_ORIGIN_KEY_FILE" \
    --model "$SERVED_MODEL_NAME" \
    --runs "$RUNS"; then
  echo "Semantic smoke failed; stopping both ranks to avoid a partial cluster." >&2
  "$SCRIPT_DIR/stop-deepseek-v4-flash-dspark.sh" || true
  "$SCRIPT_DIR/status-deepseek-v4-flash-dspark.sh" --expect stopped
  exit 1
fi
