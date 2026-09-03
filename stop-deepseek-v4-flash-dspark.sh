#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
LEGACY_PROJECT_NAME="${LEGACY_PROJECT_NAME:-$(basename "$SCRIPT_DIR" | tr '[:upper:]' '[:lower:]')}"

STOP_NFS=0
usage() {
  cat <<'EOF'
Usage: ./stop-deepseek-v4-flash-dspark.sh [--nfs]

  (default)  Stop vLLM on worker then head. Leaves the NFS share up.
  --nfs      Also stop the DSpark-owned NFS exporter (dspark-nfs) and remove
             the worker dspark-hf volume. Does not touch vllm-fn-nfs (Qwen).
EOF
}
for arg in "$@"; do
  case "$arg" in
    --nfs) STOP_NFS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"

cd "$SCRIPT_DIR"

WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER2_HOST="${WORKER2_HOST:-}"
WORKER2_DIR="${WORKER2_SCRIPT_DIR:-${WORKER2_DIR:-$WORKER_DIR}}"
WORKER2_HF_CACHE="${WORKER2_HF_CACHE:-${WORKER_HF_CACHE:-${HF_CACHE:-}}}"
WORKER2_VLLM_HOST_IP="${WORKER2_VLLM_HOST_IP:-}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-}"
DSPARK_WORKER_HF_NFS="${DSPARK_WORKER_HF_NFS:-0}"
NFS_VOLUME="${NFS_VOLUME:-dspark-hf}"
NFS_CONTAINER="${NFS_CONTAINER:-dspark-nfs}"
WORKER_COMPOSE_FILES="-f docker-compose.dspark.yml"
WORKER_HF_COMPOSE_ENV="HF_CACHE='$WORKER_HF_CACHE'"
WORKER2_COMPOSE_FILES="-f docker-compose.dspark.yml"
WORKER2_HF_COMPOSE_ENV="HF_CACHE='$WORKER2_HF_CACHE'"
if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
  WORKER_COMPOSE_FILES="-f docker-compose.dspark.yml -f docker-compose.dspark-nfs.override.yml"
  WORKER_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER_HF_CACHE'"
  WORKER2_COMPOSE_FILES="-f docker-compose.dspark.yml -f docker-compose.dspark-nfs.override.yml"
  WORKER2_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER2_HF_CACHE'"
fi

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
WORKER2_REACHABLE=0
if [ -n "$WORKER2_HOST" ]; then
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER2_HOST" "true" >/dev/null 2>&1; then
    WORKER2_REACHABLE=1
  else
    echo "WARN: cannot reach worker2 ${WORKER2_HOST}; its ranks will NOT be stopped." >&2
  fi
fi

local_project_has_resources() {
  local project="$1"
  {
    docker ps -aq --filter "label=com.docker.compose.project=$project"
    docker network ls -q --filter "label=com.docker.compose.project=$project"
    docker volume ls -q --filter "label=com.docker.compose.project=$project"
  } | grep -q .
}

# Force-remove leftover containers by compose project label + known names.
force_rm_project_containers() {
  local project="$1"
  local where="$2" # local | remote | remote2
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
  elif [ "$where" = "remote2" ]; then
    if [ -n "$WORKER2_HOST" ] && [ "${WORKER2_REACHABLE:-0}" = "1" ]; then
      ssh "$WORKER2_HOST" "$cmd" || stop_warn "force-remove on ${WORKER2_HOST} failed"
    fi
  elif [ "${WORKER_REACHABLE:-1}" = "1" ]; then
    ssh "$WORKER_HOST" "$cmd" || stop_warn "force-remove on ${WORKER_HOST} failed"
  fi
}

stop_vl_sidecar_head() {
  local project="$1"
  # Sweep leftover Qwen sidecar containers from older checkouts.
  docker ps -aq --filter "name=${project}-vl-sidecar" | xargs -r docker rm -f >/dev/null 2>&1 || true
}

stop_vl_sidecar_worker() {
  local project="$1"
  if [ "${WORKER_REACHABLE:-1}" != "1" ]; then
    return 0
  fi
  ssh "$WORKER_HOST" "
    ids=\$(docker ps -aq --filter 'name=${project}-vl-sidecar' 2>/dev/null || true)
    if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null 2>&1 || true; fi
    rm -f '$WORKER_DIR/docker-compose.vl-sidecar.yml' 2>/dev/null || true
  " || true
}

stop_main_head() {
  local project="$1"
  if local_project_has_resources "$project" || docker ps -aq --filter "name=${project}-vllm-dspark" | grep -q .; then
    echo "Stopping DSpark on head (project ${project})..."
    # rm -f first: compose down can still wait on stop_grace_period.
    docker ps -aq --filter "name=${project}-vllm-dspark" | xargs -r docker rm -f >/dev/null 2>&1 || true
    COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
      docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down --remove-orphans -t 1 2>&1 \
      | filter_compose_empty_project || true
  else
    echo "No DSpark head resources for project ${project}; skipping."
  fi
}

stop_main_worker() {
  local project="$1"
  local host="${2:-$WORKER_HOST}"
  local wdir="${3:-$WORKER_DIR}"
  local hf_env="${4:-$WORKER_HF_COMPOSE_ENV}"
  local vllm_ip="${5:-$WORKER_VLLM_HOST_IP}"
  local rank="${6:-1}"
  local compose_files="${7:-$WORKER_COMPOSE_FILES}"
  local reachable=1
  if [ "$host" = "$WORKER_HOST" ]; then
    reachable="${WORKER_REACHABLE:-1}"
  elif [ "$host" = "$WORKER2_HOST" ]; then
    reachable="${WORKER2_REACHABLE:-1}"
  fi
  if [ "$reachable" != "1" ]; then
    stop_warn "worker ${host} unreachable: DSpark rank not stopped"
    return 0
  fi
  ssh "$host" "
    cd '$wdir' || exit 1
    if {
      docker ps -aq --filter 'label=com.docker.compose.project=$project'
      docker network ls -q --filter 'label=com.docker.compose.project=$project'
      docker volume ls -q --filter 'label=com.docker.compose.project=$project'
      docker ps -aq --filter 'name=${project}-vllm-dspark'
    } | grep -q .; then
      echo 'Stopping DSpark on worker $host (project $project)...'
      docker ps -aq --filter 'name=${project}-vllm-dspark' | xargs -r docker rm -f >/dev/null 2>&1 || true
      env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS \
        COMPOSE_DISABLE_ENV_FILE=1 $hf_env \
        VLLM_HOST_IP='$vllm_ip' NODE_RANK=$rank HEADLESS=1 \
        docker compose -p '$project' --env-file .env.dspark \
          $compose_files down --remove-orphans -t 1 2>&1 \
          | grep -v 'No resource found to remove for project' || true
    else
      echo 'No DSpark worker resources for project $project on $host; skipping.'
    fi
  " || stop_warn "main DSpark worker stop failed on ${host}"
}

stop_project() {
  local project="$1"

  # Leftover Qwen sidecar containers from older checkouts, then main ranks.
  stop_vl_sidecar_head "$project"
  stop_vl_sidecar_worker "$project"

  stop_main_head "$project"
  stop_main_worker "$project"
  if [ -n "$WORKER2_HOST" ]; then
    stop_main_worker "$project" "$WORKER2_HOST" "$WORKER2_DIR" "$WORKER2_HF_COMPOSE_ENV" "$WORKER2_VLLM_HOST_IP" 2 "$WORKER2_COMPOSE_FILES"
  fi

  # Sweep anything left under this compose project on both nodes.
  force_rm_project_containers "$project" local
  force_rm_project_containers "$project" remote
  force_rm_project_containers "$project" remote2
}

stop_project "$PROJECT_NAME"
if [ "$LEGACY_PROJECT_NAME" != "$PROJECT_NAME" ]; then
  stop_project "$LEGACY_PROJECT_NAME"
fi

if [ "$STOP_NFS" = "1" ]; then
  # shellcheck source=files/nfs-share.sh
  source "$SCRIPT_DIR/files/nfs-share.sh"
  nfs_stop_owned_server || STOP_FAILURES=$((STOP_FAILURES + 1))
  if [ "${WORKER_REACHABLE:-0}" = "1" ]; then
    echo "Removing worker NFS volume ($NFS_VOLUME)..."
    ssh "$WORKER_HOST" "docker volume rm $NFS_VOLUME 2>/dev/null && echo '  Worker volume: removed.' || echo '  Worker volume: not present.'" \
      || stop_warn "worker volume rm failed on ${WORKER_HOST}"
  else
    stop_warn "worker ${WORKER_HOST} unreachable: NFS volume $NFS_VOLUME not removed"
  fi
  if [ -n "$WORKER2_HOST" ]; then
    if [ "${WORKER2_REACHABLE:-0}" = "1" ]; then
      echo "Removing worker2 NFS volume ($NFS_VOLUME)..."
      ssh "$WORKER2_HOST" "docker volume rm $NFS_VOLUME 2>/dev/null && echo '  Worker2 volume: removed.' || echo '  Worker2 volume: not present.'" \
        || stop_warn "worker2 volume rm failed on ${WORKER2_HOST}"
    else
      stop_warn "worker2 ${WORKER2_HOST} unreachable: NFS volume $NFS_VOLUME not removed"
    fi
  fi
fi

if [ "$STOP_FAILURES" -gt 0 ]; then
  echo "WARN: $STOP_FAILURES remote stop step(s) failed on ${WORKER_HOST}; the worker may still be serving a stale rank (restart: unless-stopped restores it on reboot). Re-run ./stop-deepseek-v4-flash-dspark.sh once the worker is reachable." >&2
  exit 1
fi

echo "DeepSeek V4 Flash DSpark stopped."
