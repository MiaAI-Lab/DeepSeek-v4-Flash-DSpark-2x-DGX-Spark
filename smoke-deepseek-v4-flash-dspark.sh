#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8888/v1/chat/completions}"
CONCURRENCY="${CONCURRENCY:-6}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SINGLE_STREAM_RECOVERY=0
DRY_RUN="${DRY_RUN:-0}"

case "${1:-}" in
  --single-stream-recovery)
    SINGLE_STREAM_RECOVERY=1
    shift
    ;;
  "") ;;
  *)
    echo "usage: $0 [--single-stream-recovery]" >&2
    exit 2
    ;;
esac
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--single-stream-recovery]" >&2
  exit 2
fi

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
if [ "$SINGLE_STREAM_RECOVERY" -eq 1 ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN: one bounded streaming request against $CHAT_URL"
    exit 0
  fi
  response="$(mktemp)"
  trap 'rm -f "$response"' EXIT
  curl -N -fsS --max-time 180 "$CHAT_URL" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK."}],"stream":true,"max_tokens":8,"temperature":0.0}' \
    >"$response"
  if ! grep -q '^data: ' "$response" || ! grep -q '^data: \\[DONE\\]' "$response"; then
    echo "Recovery streaming verification did not produce a first event and terminal completion." >&2
    exit 1
  fi
  echo "Recovery streaming verification passed: 1/1 request completed."
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Running ${CONCURRENCY}-way smoke test against ${CHAT_URL}"

for i in $(seq 1 "$CONCURRENCY"); do
  (
    curl -fsS --max-time 180 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Reply with OK and the number '"$i"'."}],"max_tokens":'"$MAX_TOKENS"',"temperature":0.0}' \
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
