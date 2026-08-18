#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
CHAT_URL="${CHAT_URL:-}"
CONCURRENCY="${CONCURRENCY:-6}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Default the endpoint from the configured bind address. vLLM binds exactly
# VLLM_HOST (README API note: HEAD_NODE_IP), so 127.0.0.1 is wrong for a
# LAN-IP bind. A wildcard bind is probed on loopback. An explicit CHAT_URL
# from the environment still wins.
_dspark_host="${VLLM_HOST:-127.0.0.1}"
case "$_dspark_host" in 0.0.0.0|::|"") _dspark_host=127.0.0.1 ;; esac
CHAT_URL="${CHAT_URL:-http://${_dspark_host}:${VLLM_PORT:-8888}/v1/chat/completions}"

MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
AUTH_HEADER_ARGS=()
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
fi
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Running ${CONCURRENCY}-way smoke test against ${CHAT_URL}"

for i in $(seq 1 "$CONCURRENCY"); do
  (
    curl -fsS --max-time 180 "${AUTH_HEADER_ARGS[@]}" "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK and the number '"$i"'."}],"temperature":0.0}' \
      >"$tmpdir/$i.json"
  ) &
done

fail=0
for job in $(jobs -p); do
  if ! wait "$job"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "Smoke test failed. Responses are in $tmpdir until this script exits." >&2
  exit 1
fi

for i in $(seq 1 "$CONCURRENCY"); do
  if ! grep -q '"choices"' "$tmpdir/$i.json"; then
    echo "Smoke response $i did not contain choices." >&2
    cat "$tmpdir/$i.json" >&2
    exit 1
  fi
done

echo "Smoke test passed: ${CONCURRENCY}/${CONCURRENCY} requests succeeded."
