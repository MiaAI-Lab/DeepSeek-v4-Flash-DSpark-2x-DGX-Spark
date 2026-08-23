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

show_logs() {
  local project="$1"
  echo "== head logs: $project =="
  COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" logs --tail="$TAIL" vllm-dspark || true
  echo
  echo "== worker logs: $project =="
  ssh "$WORKER_HOST" "cd '$WORKER_DIR' && COMPOSE_DISABLE_ENV_FILE=1 docker compose -p '$project' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail='$TAIL' vllm-dspark" || true
  echo
}

show_logs "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  show_logs "$LEGACY_PROJECT_NAME"
fi

# Issue #32 GB10 memory/NVRM observer (begin)
# Opportunistic tail of the report-only observer's latest records; never
# changes the exit status and not gated on DSPARK_GB10_OBSERVER.
OBSERVER_SCRIPT="$SCRIPT_DIR/scripts/gb10-memory-observer.py"
observer_run() {
  # Bounded best-effort: never let a wedged observer stall logs.
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
  local state_dir="$1" records="$1/records.ndjson"
  if [ -f "$records" ]; then
    echo "-- GB10 observer records (tail $TAIL): $records --"
    tail -n "$TAIL" -- "$records" 2>/dev/null || true
    echo
  else
    echo "-- GB10 observer: no records in $state_dir yet --"
    echo
  fi
}
echo "== GB10 observer records: head =="
if [ -f "$OBSERVER_SCRIPT" ]; then
  observer_run python3 "$OBSERVER_SCRIPT" status || true
fi
show_observer_records "$(observer_state_dir)"
echo "== GB10 observer records: worker ${WORKER_HOST} =="
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
    "DS_OBS_SCRIPT=$(printf '%q' "$WORKER_DIR/scripts/gb10-memory-observer.py") DS_TAIL=$(printf '%q' "$TAIL") $DS_OBS_STATE_DIR_FWD bash -s" <<'DS_REMOTE_OBSERVER'
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
  echo "-- GB10 observer: not installed ($obs_script missing) --"
fi
records="$state_dir/records.ndjson"
if [ -f "$records" ]; then
  echo "-- GB10 observer records (tail $DS_TAIL): $records --"
  tail -n "$DS_TAIL" -- "$records" 2>/dev/null || true
else
  echo "-- GB10 observer: no records in $state_dir yet --"
fi
DS_REMOTE_OBSERVER
then
  :
else
  echo "worker unreachable; skipping worker GB10 observer records."
fi
echo
# Issue #32 GB10 memory/NVRM observer (end)
