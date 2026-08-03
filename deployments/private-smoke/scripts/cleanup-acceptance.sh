#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.dspark}"
LITELLM_ENV_FILE="${LITELLM_ENV_FILE:-$ROOT_DIR/deployments/private-smoke/litellm/.env}"
STOP_DSPARK_BIN="${STOP_DSPARK_BIN:-$ROOT_DIR/stop-deepseek-v4-flash-dspark.sh}"
STATUS_DSPARK_BIN="${STATUS_DSPARK_BIN:-$ROOT_DIR/status-deepseek-v4-flash-dspark.sh}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
EGRESS_POLICY_BIN="${EGRESS_POLICY_BIN:-$ROOT_DIR/deployments/private-smoke/litellm/egress-policy.sh}"
COMPOSE_FILE="${LITELLM_COMPOSE_FILE:-$ROOT_DIR/deployments/private-smoke/litellm/docker-compose.yml}"
PRISMA_CACHE_VOLUME="${PRISMA_CACHE_VOLUME:-dspark-private-litellm-prisma-cache}"
failed=0

if ! ENV_FILE="$ENV_FILE" "$STOP_DSPARK_BIN"; then
  failed=1
fi
if ! ENV_FILE="$ENV_FILE" "$STATUS_DSPARK_BIN" --expect stopped; then
  failed=1
fi
if ! "$DOCKER_BIN" compose -p dspark-private-litellm --env-file "$LITELLM_ENV_FILE" \
  -f "$COMPOSE_FILE" down --remove-orphans; then
  failed=1
fi
if "$DOCKER_BIN" volume inspect "$PRISMA_CACHE_VOLUME" >/dev/null 2>&1 &&
   ! "$DOCKER_BIN" volume rm "$PRISMA_CACHE_VOLUME" >/dev/null; then
  failed=1
fi
if ! DSPARK_EGRESS_NONINTERACTIVE_REMOVE=1 "$EGRESS_POLICY_BIN" --remove; then
  failed=1
fi

if [ "$failed" -ne 0 ]; then
  echo "Acceptance cleanup was incomplete; inspect both ranks and the private gateway." >&2
  exit 1
fi
echo "Acceptance cleanup verified both ranks stopped and the private gateway removed."
