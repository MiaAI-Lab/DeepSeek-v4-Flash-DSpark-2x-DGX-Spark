#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${LITELLM_ENV_FILE:-$SCRIPT_DIR/.env}"
COMPOSE=(docker compose -p dspark-private-litellm --env-file "$ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yml")
POLICY_INSTALLED=0
PRISMA_CACHE_VOLUME="dspark-private-litellm-prisma-cache"
LITELLM_IMAGE="ghcr.io/berriai/litellm-database@sha256:5fa5f99cd5576e359a0e50395ad14edbe922ef41c152f67c534e4f8b6238c5ec"
PRISMA_CACHE_SENTINEL="binaries/5.4.2/ac9d7041ed77bcc8a8dbd2ab6616b39013829574/node_modules/prisma/build/index.js"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

cleanup_failed_gateway() {
  local status=$?
  trap - ERR
  "${COMPOSE[@]}" down --remove-orphans || true
  docker volume rm "$PRISMA_CACHE_VOLUME" >/dev/null 2>&1 || true
  if [ "$POLICY_INSTALLED" -eq 1 ]; then
    "$SCRIPT_DIR/egress-policy.sh" --remove || true
  fi
  echo "Private gateway deployment failed; the active gateway was not changed." >&2
  exit "$status"
}
trap cleanup_failed_gateway ERR

prepare_prisma_cache() {
  docker volume inspect "$PRISMA_CACHE_VOLUME" >/dev/null 2>&1 ||
    docker volume create --label dspark-smoke-cache=prisma "$PRISMA_CACHE_VOLUME" >/dev/null
  if docker run --rm --network none --read-only --cap-drop ALL \
      --security-opt no-new-privileges:true --user 1000:1000 \
      -v "$PRISMA_CACHE_VOLUME:/cache:ro" --entrypoint test "$LITELLM_IMAGE" \
      -r "/cache/$PRISMA_CACHE_SENTINEL"; then
    return
  fi
  docker run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true \
    -v "$PRISMA_CACHE_VOLUME:/cache" --entrypoint sh "$LITELLM_IMAGE" -eu -c \
    'find /cache -mindepth 1 -delete; cp -a /root/.cache/prisma-python/. /cache/'
  docker run --rm --network none --read-only --cap-drop ALL \
    --security-opt no-new-privileges:true --user 1000:1000 \
    -v "$PRISMA_CACHE_VOLUME:/cache:ro" --entrypoint test "$LITELLM_IMAGE" \
    -r "/cache/$PRISMA_CACHE_SENTINEL"
}

ENV_FILE="${DSPARK_ENV_FILE:-$ROOT_DIR/.env.dspark}" \
  "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
[ "$(docker network inspect -f '{{.Internal}}' dspark-smoke)" = "true" ]
[ ! -e "$LITELLM_VIRTUAL_KEY_FILE" ] || { echo "Virtual key output already exists." >&2; exit 1; }

"$SCRIPT_DIR/smoke.sh" --snapshot-before
"$SCRIPT_DIR/egress-policy.sh" --install
POLICY_INSTALLED=1
prepare_prisma_cache
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
