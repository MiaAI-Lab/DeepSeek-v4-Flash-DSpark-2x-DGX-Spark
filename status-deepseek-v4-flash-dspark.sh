#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"
API_URL="${API_URL:-}"
PORT="${PORT:-8888}"
AUTH_HEADER_ARGS=()

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

# Default the endpoint from the configured bind address. vLLM binds exactly
# VLLM_HOST (README API note: HEAD_NODE_IP), so 127.0.0.1 is wrong for a
# LAN-IP bind. A wildcard bind is probed on loopback. An explicit API_URL
# from the environment still wins.
_dspark_host="${VLLM_HOST:-127.0.0.1}"
case "$_dspark_host" in 0.0.0.0|::|"") _dspark_host=127.0.0.1 ;; esac
API_URL="${API_URL:-http://${_dspark_host}:${VLLM_PORT:-8888}/v1/models}"

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
: "${DSPARK_VLLM_IMAGE:=vllm-dspark-runtime:dspark-nvfp4-stage-c}"

cd "$SCRIPT_DIR"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"

show_compose() {
  local project="$1"
  echo "== head compose: $project =="
  COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps || true
  echo
  echo "== worker compose: $project =="
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && COMPOSE_DISABLE_ENV_FILE=1 docker compose -p '$project' --env-file .env.dspark -f docker-compose.dspark.yml ps" || true
  echo
}

show_compose "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  show_compose "$LEGACY_PROJECT_NAME"
fi

echo "== head matching containers =="
docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}' | grep -E 'deepseek|dspark|vllm' || true
echo
echo "== worker matching containers =="
ssh "$WORKER_HOST" "docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}' | grep -E 'deepseek|dspark|vllm' || true" || true
echo
echo "== images =="
docker image inspect "$DSPARK_VLLM_IMAGE" --format "head $DSPARK_VLLM_IMAGE {{.Id}}" || true
ssh "$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' --format 'worker $DSPARK_VLLM_IMAGE {{.Id}}'" || true
echo
echo "== port/API =="
if command -v ss >/dev/null 2>&1; then
  ss -ltn "( sport = :$PORT )" || true
fi
curl -fsS --max-time 5 "${AUTH_HEADER_ARGS[@]}" "$API_URL" || true
echo
