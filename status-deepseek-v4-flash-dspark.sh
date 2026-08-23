#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"
API_URL="${API_URL:-}"
PORT="${PORT:-8888}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# DSPARK_API_KEYS auth (begin)
AUTH_HEADER_ARGS=()
case "${DSPARK_API_KEYS:-}" in
  *[$'\r\n\v\f']*)
    echo "error: DSPARK_API_KEYS must be a single-line space-separated list" >&2
    exit 2
    ;;
  *\\*)
    echo "error: DSPARK_API_KEYS must not contain backslashes" >&2
    exit 2
    ;;
esac
_dspark_keys_set=0
case "${DSPARK_API_KEYS:-}" in
  *[!$' \t']*) _dspark_keys_set=1 ;;
esac
if [ -n "${VLLM_API_KEY:-}" ] && [ "$_dspark_keys_set" = "1" ]; then
  # The server entrypoint refuses this combination too (exit 2); fail the same
  # way here so a probe never guesses which variable the server honoured.
  echo "error: VLLM_API_KEY and DSPARK_API_KEYS are both set; set exactly one of them" >&2
  exit 2
fi
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
elif [ "$_dspark_keys_set" = "1" ]; then
  _dspark_keys=()
  read -r -a _dspark_keys <<< "${DSPARK_API_KEYS}"
  for _dspark_key in "${_dspark_keys[@]}"; do
    case "$_dspark_key" in
      -*) echo "error: DSPARK_API_KEYS contains a token beginning with '-'" >&2; exit 2 ;;
    esac
  done
  # Multi-key auth via --api-key: probe with the first parsed key. Without this
  # the health poll never sees a 200 against a keyed server and waits out its
  # full timeout on a cluster that is actually serving.
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${_dspark_keys[0]}")
fi
# DSPARK_API_KEYS auth (end)

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
# Issue #32 GB10 memory/NVRM observer (begin)
# Opportunistic report of the report-only observer; never changes the exit
# status and not gated on DSPARK_GB10_OBSERVER, so a manually attached
# observer is visible too. Absent script/state dir prints a quiet note.
OBSERVER_SCRIPT="$SCRIPT_DIR/scripts/gb10-memory-observer.py"
observer_run() {
  # Bounded best-effort: never let a wedged observer stall status/logs.
  if command -v timeout >/dev/null 2>&1; then
    timeout 15 "$@"
  else
    "$@"
  fi
}
observer_state_dir() {
  printf '%s' "${DSPARK_GB10_OBSERVER_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/dspark-observer}"
}
show_observer_records() {
  # $1 = state dir; tails the observer's record file, tolerating absence.
  local state_dir="$1" records="$1/records.ndjson"
  if [ -f "$records" ]; then
    echo "-- latest records (tail ${TAIL:-160}): $records --"
    tail -n "${TAIL:-160}" -- "$records" 2>/dev/null || true
  else
    echo "no records in $state_dir yet"
  fi
}
echo "== GB10 observer: head =="
if [ -f "$OBSERVER_SCRIPT" ]; then
  observer_run python3 "$OBSERVER_SCRIPT" status || true
else
  echo "not installed ($OBSERVER_SCRIPT missing)"
fi
show_observer_records "$(observer_state_dir)"
echo
echo "== GB10 observer: worker ${WORKER_HOST} =="
# Remote body ships verbatim via a quoted heredoc; config travels as %q'd
# environment assignments ahead of bash -s, so no nested quoting survives.
# A custom DSPARK_GB10_OBSERVER_STATE_DIR is forwarded only when actually set;
# otherwise DS_OBS_STATE_DIR stays empty and the WORKER resolves its own
# ${XDG_STATE_HOME:-$HOME/.local/state}/dspark-observer below (never the head's).
DS_OBS_STATE_DIR_FWD=""
if [ -n "${DSPARK_GB10_OBSERVER_STATE_DIR:-}" ]; then
  DS_OBS_STATE_DIR_FWD="DS_OBS_STATE_DIR=$(printf '%q' "$DSPARK_GB10_OBSERVER_STATE_DIR")"
fi
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
    "DS_OBS_SCRIPT=$(printf '%q' "$WORKER_DIR/scripts/gb10-memory-observer.py") DS_TAIL=$(printf '%q' "${TAIL:-160}") $DS_OBS_STATE_DIR_FWD bash -s" <<'DS_REMOTE_OBSERVER'
obs_script="$DS_OBS_SCRIPT"
state_dir="$DS_OBS_STATE_DIR"
: "${state_dir:=${XDG_STATE_HOME:-$HOME/.local/state}/dspark-observer}"
if [ -f "$obs_script" ]; then
  if command -v timeout >/dev/null 2>&1; then
    timeout 15 python3 "$obs_script" status || true
  else
    python3 "$obs_script" status || true
  fi
else
  echo "not installed ($obs_script missing)"
fi
records="$state_dir/records.ndjson"
if [ -f "$records" ]; then
  echo "-- latest records (tail $DS_TAIL): $records --"
  tail -n "$DS_TAIL" -- "$records" 2>/dev/null || true
else
  echo "no records in $state_dir yet"
fi
DS_REMOTE_OBSERVER
then
  :
else
  echo "worker unreachable; skipping worker GB10 observer report."
fi
echo
# Issue #32 GB10 memory/NVRM observer (end)
