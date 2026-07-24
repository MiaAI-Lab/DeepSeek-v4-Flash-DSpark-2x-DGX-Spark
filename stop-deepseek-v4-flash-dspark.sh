#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"
RECOVERY_RANK=""
DRY_RUN="${DRY_RUN:-0}"

case "${1:-}" in
  --recovery-rank)
    RECOVERY_RANK="${2:-}"
    shift 2
    ;;
  "") ;;
  *)
    echo "usage: $0 [--recovery-rank head|worker]" >&2
    exit 2
    ;;
esac
if [ "$#" -ne 0 ] || { [ -n "$RECOVERY_RANK" ] && [ "$RECOVERY_RANK" != "head" ] && [ "$RECOVERY_RANK" != "worker" ]; }; then
  echo "usage: $0 [--recovery-rank head|worker]" >&2
  exit 2
fi

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"

cd "$SCRIPT_DIR"

WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-}"

if [ -n "$RECOVERY_RANK" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: recovery stop rank=$RECOVERY_RANK project=$PROJECT_NAME"
    exit 0
  fi
  if [ "$RECOVERY_RANK" = "head" ]; then
    echo "Stopping DSpark head project ${PROJECT_NAME} for recovery..."
    COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down
  else
    echo "Stopping DSpark worker project ${PROJECT_NAME} on ${WORKER_HOST} for recovery..."
    ssh "$WORKER_HOST" "cd '$WORKER_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml down"
  fi
  exit 0
fi

stop_project() {
  local project="$1"

  echo "Stopping DSpark head project ${project}..."
  COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down || true

  echo "Stopping DSpark worker project ${project} on ${WORKER_HOST}..."
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' docker compose -p '$project' --env-file .env.dspark -f docker-compose.dspark.yml down" || true
}

stop_project "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  stop_project "$LEGACY_PROJECT_NAME"
fi

echo "DeepSeek V4 Flash DSpark stopped."
