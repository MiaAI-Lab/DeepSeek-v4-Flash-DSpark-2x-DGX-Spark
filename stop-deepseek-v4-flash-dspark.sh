#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
SIDECAR_COMPOSE_FILE="${SIDECAR_COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.vl-sidecar.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"

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

# A stop that cannot reach the worker must not report success: a powered-down
# worker resurrects its stale rank (compose restart: unless-stopped) the next
# time it boots, and nothing here will have stopped it. Mirror start's ssh
# hardening (BatchMode/ConnectTimeout) so stop never hangs on a prompt either.
STOP_FAILURES=0
stop_warn() {
  echo "WARN: $*" >&2
  STOP_FAILURES=$((STOP_FAILURES + 1))
}

# After docker rm -f, compose down often has nothing left and prints
# "No resource found to remove for project …" — noise, not a failure.
filter_compose_empty_project() {
  grep -v 'No resource found to remove for project' || true
}
if ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "true" >/dev/null 2>&1; then
  WORKER_REACHABLE=1
else
  WORKER_REACHABLE=0
  echo "WARN: cannot reach worker ${WORKER_HOST}; its ranks will NOT be stopped." >&2
fi

local_project_has_resources() {
  local project="$1"
  {
    docker ps -aq --filter "label=com.docker.compose.project=$project"
    docker network ls -q --filter "label=com.docker.compose.project=$project"
    docker volume ls -q --filter "label=com.docker.compose.project=$project"
  } | grep -q .
}

# Force-remove VL / 0731 containers by compose project label + known names.
# Used when compose down misses a service (e.g. worker missing vl-sidecar.yml).
force_rm_project_containers() {
  local project="$1"
  local where="$2" # local | remote
  local cmd
  cmd=$(cat <<EOF
ids=\$(docker ps -aq --filter "label=com.docker.compose.project=$project" 2>/dev/null || true)
names=\$(docker ps -aq --filter "name=${project}-vl-sidecar" --filter "name=${project}-vllm-dspark" 2>/dev/null || true)
all=\$(printf '%s\n%s\n' "\$ids" "\$names" | awk 'NF' | sort -u)
if [ -n "\$all" ]; then
  echo "Force-removing containers for project $project..."
  # shellcheck disable=SC2086
  docker rm -f \$all >/dev/null 2>&1 || true
fi
EOF
)
  if [ "$where" = "local" ]; then
    bash -c "$cmd" || true
  elif [ "${WORKER_REACHABLE:-1}" = "1" ]; then
    ssh "$WORKER_HOST" "$cmd" || stop_warn "force-remove on ${WORKER_HOST} failed"
  fi
}

stop_vl_sidecar_head() {
  local project="$1"
  # Text-only ship: skip noisy VL down when nothing is running.
  if [ "${ENABLE_VL_SIDECAR:-0}" != "1" ] \
    && ! docker ps -aq --filter "name=${project}-vl-sidecar" | grep -q .; then
    return 0
  fi
  if [ ! -f "$SIDECAR_COMPOSE_FILE" ]; then
    echo "No $SIDECAR_COMPOSE_FILE on head; force-removing any VL containers..."
    force_rm_project_containers "$project" local
    return 0
  fi
  echo "Stopping VL vision sidecar on head (project ${project})..."
  # NODE_RANK required for compose file interpolation; value unused for down.
  COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
    docker compose -p "$project" --env-file "$ENV_FILE" -f "$SIDECAR_COMPOSE_FILE" down 2>&1 \
    | filter_compose_empty_project || true
  # Belt-and-suspenders: name filter in case compose project label drifted.
  docker ps -aq --filter "name=${project}-vl-sidecar" | xargs -r docker rm -f >/dev/null 2>&1 || true
}

stop_vl_sidecar_worker() {
  local project="$1"
  if [ "${WORKER_REACHABLE:-1}" != "1" ]; then
    stop_warn "worker ${WORKER_HOST} unreachable: VL sidecar worker rank not stopped"
    return 0
  fi
  # Text-only: still sweep if a zombie VL exists; otherwise skip SSH noise.
  local worker_has_vl=0
  if ssh "$WORKER_HOST" "docker ps -aq --filter 'name=${project}-vl-sidecar' 2>/dev/null | grep -q ."; then
    worker_has_vl=1
  fi
  if [ "${ENABLE_VL_SIDECAR:-0}" != "1" ] && [ "$worker_has_vl" != "1" ]; then
    return 0
  fi
  # Ensure worker has the compose file so `down` can target the VL service.
  if [ -f "$SIDECAR_COMPOSE_FILE" ]; then
    ssh "$WORKER_HOST" "mkdir -p '$WORKER_DIR'" || true
    scp -q "$SIDECAR_COMPOSE_FILE" "${WORKER_HOST}:${WORKER_DIR}/docker-compose.vl-sidecar.yml" || true
    scp -q "$ENV_FILE" "${WORKER_HOST}:${WORKER_DIR}/.env.dspark" || true
  fi
  echo "Stopping VL vision sidecar on worker ${WORKER_HOST} (project ${project})..."
  ssh "$WORKER_HOST" "
    cd '$WORKER_DIR' || exit 0
    if [ -f docker-compose.vl-sidecar.yml ]; then
      env -u MASTER_ADDR -u MASTER_PORT -u HEADLESS \
        COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 \
        HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' \
        docker compose -p '$project' --env-file .env.dspark \
          -f docker-compose.vl-sidecar.yml down 2>&1 \
          | grep -v 'No resource found to remove for project' || true
    fi
    ids=\$(docker ps -aq --filter 'name=${project}-vl-sidecar' 2>/dev/null || true)
    if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null 2>&1 || true; fi
  " || stop_warn "VL sidecar worker stop failed on ${WORKER_HOST}"
}

stop_main_head() {
  local project="$1"
  if local_project_has_resources "$project" || docker ps -aq --filter "name=${project}-vllm-dspark" | grep -q .; then
    echo "Stopping DSpark 0731 on head (project ${project})..."
    # rm -f first: compose down can still wait on stop_grace_period.
    docker ps -aq --filter "name=${project}-vllm-dspark" | xargs -r docker rm -f >/dev/null 2>&1 || true
    COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
      docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down --remove-orphans -t 1 2>&1 \
      | filter_compose_empty_project || true
  else
    echo "No DSpark 0731 head resources for project ${project}; skipping."
  fi
}

stop_main_worker() {
  local project="$1"
  if [ "${WORKER_REACHABLE:-1}" != "1" ]; then
    stop_warn "worker ${WORKER_HOST} unreachable: main DSpark worker rank not stopped"
    return 0
  fi
  ssh "$WORKER_HOST" "
    cd '$WORKER_DIR' || exit 1
    if {
      docker ps -aq --filter 'label=com.docker.compose.project=$project'
      docker network ls -q --filter 'label=com.docker.compose.project=$project'
      docker volume ls -q --filter 'label=com.docker.compose.project=$project'
      docker ps -aq --filter 'name=${project}-vllm-dspark'
    } | grep -q .; then
      echo 'Stopping DSpark 0731 on worker $WORKER_HOST (project $project)...'
      docker ps -aq --filter 'name=${project}-vllm-dspark' | xargs -r docker rm -f >/dev/null 2>&1 || true
      env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS \
        COMPOSE_DISABLE_ENV_FILE=1 HF_CACHE='$WORKER_HF_CACHE' \
        VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' NODE_RANK=1 HEADLESS=1 \
        docker compose -p '$project' --env-file .env.dspark \
          -f docker-compose.dspark.yml down --remove-orphans -t 1 2>&1 \
          | grep -v 'No resource found to remove for project' || true
    else
      echo 'No DSpark 0731 worker resources for project $project on $WORKER_HOST; skipping.'
    fi
  " || stop_warn "main DSpark worker stop failed on ${WORKER_HOST}"
}

stop_project() {
  local project="$1"

  # Vision first so a failed main down cannot leave Qwen occupying VRAM.
  stop_vl_sidecar_head "$project"
  stop_vl_sidecar_worker "$project"

  stop_main_head "$project"
  stop_main_worker "$project"

  # Sweep anything left under this compose project on both nodes.
  force_rm_project_containers "$project" local
  force_rm_project_containers "$project" remote
}

stop_project "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  stop_project "$LEGACY_PROJECT_NAME"
fi

# Issue #32 GB10 memory/NVRM observer (begin)
# Report-only host observer, opt-in via DSPARK_GB10_OBSERVER (unset/0 = no-op).
# Stopped AFTER rank teardown, only when autostop is enabled, and never added
# to STOP_FAILURES: an observer stop failure must not flip the stale-rank
# verdict that a worker could still be serving.
OBSERVER_SCRIPT="$SCRIPT_DIR/scripts/gb10-memory-observer.py"
# Match where start ships the worker copy (never the head path remotely).
REMOTE_OBSERVER_SCRIPT="$WORKER_DIR/scripts/gb10-memory-observer.py"
observer_enabled() {
  # Strict 0/1: any other value is rejected loudly instead of being silently
  # treated as on/off; the observer CLI enforces the same contract.
  local v="${DSPARK_GB10_OBSERVER:-}"
  case "$v" in
    ''|0|1) [ "$v" = "1" ] ;;
    *)
      echo "error: DSPARK_GB10_OBSERVER must be 0 or 1 (got: $v)" >&2
      exit 2
      ;;
  esac
}
observer_autostop() {
  # Strict 0/1 with default-on; mirrors the observer CLI's own validation so a
  # typo like AUTOSTOP=true cannot wedge the stop hook on its ConfigError.
  local v="${DSPARK_GB10_OBSERVER_AUTOSTOP:-1}"
  case "$v" in
    ''|0|1) [ "$v" = "1" ] ;;
    *)
      echo "error: DSPARK_GB10_OBSERVER_AUTOSTOP must be 0 or 1 (got: $v)" >&2
      exit 2
      ;;
  esac
}
observer_remote_env() {
  # Exported into the worker shell so per-node config travels intact; values
  # are %q-quoted for the remote shell.
  local name
  for name in DSPARK_GB10_OBSERVER DSPARK_GB10_OBSERVER_INTERVAL DSPARK_GB10_OBSERVER_STATE_DIR DSPARK_GB10_OBSERVER_AUTOSTOP; do
    if [ -n "${!name+x}" ]; then
      printf '%s=%q ' "$name" "${!name}"
    fi
  done
}
observer_stop_head() {
  command -v timeout >/dev/null 2>&1 || { echo "WARN: GB10 observer not stopped on head: timeout(1) unavailable." >&2; return 0; }
  echo "Stopping GB10 memory observer on head..."
  if ! timeout 20 python3 "$OBSERVER_SCRIPT" stop </dev/null >/dev/null 2>&1; then
    echo "WARN: GB10 observer did not stop cleanly on head (continuing)." >&2
  fi
}
observer_stop_worker() {
  command -v timeout >/dev/null 2>&1 || { echo "WARN: GB10 observer not stopped on ${WORKER_HOST}: timeout(1) unavailable." >&2; return 0; }
  echo "Stopping GB10 memory observer on worker ${WORKER_HOST}..."
  if ! timeout 30 ssh -o BatchMode=yes "$WORKER_HOST" "$(observer_remote_env)timeout 20 python3 $(printf '%q' "$REMOTE_OBSERVER_SCRIPT") stop </dev/null >/dev/null 2>&1"; then
    echo "WARN: GB10 observer did not stop cleanly on worker ${WORKER_HOST} (continuing)." >&2
  fi
}
if observer_enabled && observer_autostop; then
  observer_stop_head
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "true" >/dev/null 2>&1; then
    observer_stop_worker
  else
    echo "WARN: cannot reach worker ${WORKER_HOST}; its GB10 observer keeps running until reboot or manual stop." >&2
  fi
fi
# Issue #32 GB10 memory/NVRM observer (end)

if [ "$STOP_FAILURES" -gt 0 ]; then
  echo "WARN: $STOP_FAILURES remote stop step(s) failed on ${WORKER_HOST}; the worker may still be serving a stale rank (restart: unless-stopped restores it on reboot). Re-run ./stop-deepseek-v4-flash-dspark.sh once the worker is reachable." >&2
  exit 1
fi

if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ]; then
  echo "DeepSeek V4 Flash DSpark stopped (0731 + VL vision sidecar)."
else
  echo "DeepSeek V4 Flash DSpark stopped (0731 text-only; any leftover VL sidecar swept)."
fi
