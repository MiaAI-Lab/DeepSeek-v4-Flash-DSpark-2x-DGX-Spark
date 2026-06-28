#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
API_URL="${API_URL:-http://127.0.0.1:8888/v1/models}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8888/v1/chat/completions}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-100}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.dspark.example to .env.dspark and edit node-specific values." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"

cd "$SCRIPT_DIR"

echo "Syncing DSpark deployment files to ${WORKER_HOST}:${SCRIPT_DIR}"
ssh "$WORKER_HOST" "mkdir -p '$SCRIPT_DIR'"
scp "$COMPOSE_FILE" "${WORKER_HOST}:${SCRIPT_DIR}/docker-compose.dspark.yml"
scp "$ENV_FILE" "${WORKER_HOST}:${SCRIPT_DIR}/.env.dspark"

echo "Starting DSpark worker on ${WORKER_HOST}..."
ssh "$WORKER_HOST" "cd '$SCRIPT_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d"

echo "Starting DSpark head..."
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

echo "Waiting for DSpark vLLM API..."
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
    echo "DeepSeek V4 Flash DSpark is running: $API_URL"
    COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps
    ssh "$WORKER_HOST" "cd '$SCRIPT_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml ps"
    echo "Running minimal OpenAI-compatible chat request..."
    curl -fsS --max-time 60 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":8,"temperature":0.0}' >/dev/null
    echo "Minimal chat request succeeded."
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for DSpark API. Recent head logs:" >&2
COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs --tail=120 vllm-dspark >&2 || true
echo "Recent worker logs:" >&2
ssh "$WORKER_HOST" "cd '$SCRIPT_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark" >&2 || true
exit 1
