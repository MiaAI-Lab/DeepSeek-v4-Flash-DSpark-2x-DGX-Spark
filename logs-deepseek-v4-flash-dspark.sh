#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"
TAIL="${TAIL:-160}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"

cd "$SCRIPT_DIR"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER2_HOST="${WORKER2_HOST:-}"
WORKER2_DIR="${WORKER2_SCRIPT_DIR:-${WORKER2_DIR:-$WORKER_DIR}}"

show_logs() {
  local project="$1"
  echo "== head logs: $project =="
  COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs --tail="$TAIL" vllm-dspark || true
  echo
  echo "== worker logs: $project =="
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && COMPOSE_DISABLE_ENV_FILE=1 docker compose -p '$project' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail='$TAIL' vllm-dspark" || true
  echo
  if [ -n "${WORKER2_HOST:-}" ]; then
    echo "== worker2 logs: $project =="
    ssh "$WORKER2_HOST" "cd '${WORKER2_DIR:-$WORKER_DIR}' && COMPOSE_DISABLE_ENV_FILE=1 docker compose -p '$project' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail='$TAIL' vllm-dspark" || true
    echo
  fi
}

show_logs "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  show_logs "$LEGACY_PROJECT_NAME"
fi
