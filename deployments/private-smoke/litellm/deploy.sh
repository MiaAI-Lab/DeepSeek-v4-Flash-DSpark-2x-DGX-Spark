#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${LITELLM_ENV_FILE:-$SCRIPT_DIR/.env}"
COMPOSE=(docker compose -p dspark-private-litellm --env-file "$ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yml")
POLICY_INSTALLED=0

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

cleanup_failed_gateway() {
  local status=$?
  trap - ERR
  "${COMPOSE[@]}" down --remove-orphans || true
  if [ "$POLICY_INSTALLED" -eq 1 ]; then
    "$SCRIPT_DIR/egress-policy.sh" --remove || true
  fi
  echo "Private gateway deployment failed; the active gateway was not changed." >&2
  exit "$status"
}
trap cleanup_failed_gateway ERR

ENV_FILE="${DSPARK_ENV_FILE:-$ROOT_DIR/.env.dspark}" \
  "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
[ "$(docker network inspect -f '{{.Internal}}' dspark-smoke)" = "true" ]
[ ! -e "$LITELLM_VIRTUAL_KEY_FILE" ] || { echo "Virtual key output already exists." >&2; exit 1; }

"$SCRIPT_DIR/smoke.sh" --snapshot-before
"$SCRIPT_DIR/egress-policy.sh" --install
POLICY_INSTALLED=1
"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" up -d
for _ in $(seq 1 90); do
  health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' dspark-private-litellm-litellm-1 2>/dev/null || true)"
  [ "$health" = "healthy" ] && break
  sleep 2
done
[ "${health:-}" = "healthy" ] || { echo "Private LiteLLM did not become healthy." >&2; false; }
"$SCRIPT_DIR/bootstrap-virtual-key.sh"
"$SCRIPT_DIR/smoke.sh" --all-interfaces
trap - ERR
echo "Private LiteLLM smoke gateway is running only on ${HEAD_TAILSCALE_IP}:4001."
