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
  else
    ssh "$WORKER_HOST" "$cmd" || true
  fi
}

stop_vl_sidecar_head() {
  local project="$1"
  if [ ! -f "$SIDECAR_COMPOSE_FILE" ]; then
    echo "No $SIDECAR_COMPOSE_FILE on head; force-removing any VL containers..."
    force_rm_project_containers "$project" local
    return 0
  fi
  echo "Stopping VL vision sidecar on head (project ${project})..."
  # NODE_RANK required for compose file interpolation; value unused for down.
  COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
    docker compose -p "$project" --env-file "$ENV_FILE" -f "$SIDECAR_COMPOSE_FILE" down || true
  # Belt-and-suspenders: name filter in case compose project label drifted.
  docker ps -aq --filter "name=${project}-vl-sidecar" | xargs -r docker rm -f >/dev/null 2>&1 || true
}

stop_vl_sidecar_worker() {
  local project="$1"
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
          -f docker-compose.vl-sidecar.yml down || true
    fi
    ids=\$(docker ps -aq --filter 'name=${project}-vl-sidecar' 2>/dev/null || true)
    if [ -n \"\$ids\" ]; then docker rm -f \$ids >/dev/null 2>&1 || true; fi
  " || true
}

stop_main_head() {
  local project="$1"
  if local_project_has_resources "$project" || docker ps -aq --filter "name=${project}-vllm-dspark" | grep -q .; then
    echo "Stopping DSpark 0731 on head (project ${project})..."
    COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
      docker compose -p "$project" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" down || true
  else
    echo "No DSpark 0731 head resources for project ${project}; skipping."
  fi
}

stop_main_worker() {
  local project="$1"
  ssh "$WORKER_HOST" "
    cd '$WORKER_DIR' || exit 1
    if {
      docker ps -aq --filter 'label=com.docker.compose.project=$project'
      docker network ls -q --filter 'label=com.docker.compose.project=$project'
      docker volume ls -q --filter 'label=com.docker.compose.project=$project'
      docker ps -aq --filter 'name=${project}-vllm-dspark'
    } | grep -q .; then
      echo 'Stopping DSpark 0731 on worker $WORKER_HOST (project $project)...'
      env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS \
        COMPOSE_DISABLE_ENV_FILE=1 HF_CACHE='$WORKER_HF_CACHE' \
        VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' NODE_RANK=1 HEADLESS=1 \
        docker compose -p '$project' --env-file .env.dspark \
          -f docker-compose.dspark.yml down || true
    else
      echo 'No DSpark 0731 worker resources for project $project on $WORKER_HOST; skipping.'
    fi
  " || true
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

echo "DeepSeek V4 Flash DSpark stopped (0731 + VL vision sidecar)."
