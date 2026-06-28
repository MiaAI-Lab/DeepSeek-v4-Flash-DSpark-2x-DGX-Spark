#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"

cd "$SCRIPT_DIR"

echo "Stopping DSpark head..."
COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down || true

echo "Stopping DSpark worker on ${WORKER_HOST}..."
ssh "$WORKER_HOST" "cd '$SCRIPT_DIR' && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file .env.dspark -f docker-compose.dspark.yml down" || true

echo "DeepSeek V4 Flash DSpark stopped."
