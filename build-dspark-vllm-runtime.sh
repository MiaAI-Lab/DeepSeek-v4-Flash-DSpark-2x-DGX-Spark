#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/rafaelcaricio/vllm.git}"
VLLM_BRANCH="${VLLM_BRANCH:-codex/dspark-harness-integration}"
VLLM_CHECKOUT="${VLLM_CHECKOUT:-$HOME/models/spark/vllm-dspark}"
DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:clean}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready}"

clone_or_update() {
  local dir="$1"
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch origin "$VLLM_BRANCH"
    git -C "$dir" checkout "$VLLM_BRANCH"
    git -C "$dir" pull --ff-only origin "$VLLM_BRANCH"
  else
    mkdir -p "$(dirname "$dir")"
    git clone "$VLLM_REPO_URL" "$dir"
    git -C "$dir" checkout "$VLLM_BRANCH"
  fi
}

build_local() {
  clone_or_update "$VLLM_CHECKOUT"
  docker pull "$BASE_IMAGE"
  docker build \
    -f "$VLLM_CHECKOUT/docker/Dockerfile.dspark-runtime-overlay" \
    -t "$DSPARK_VLLM_IMAGE" \
    "$VLLM_CHECKOUT"
  docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_VLLM_IMAGE" -c \
    "import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; print('dspark overlay ok', d.__name__, p.__name__)"
}

build_local

if [ "${BUILD_WORKER:-1}" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  worker_checkout="${WORKER_VLLM_CHECKOUT:-$VLLM_CHECKOUT}"
  echo "Building DSpark vLLM runtime on ${WORKER_HOST}:${worker_checkout}"
  ssh "$WORKER_HOST" "mkdir -p '$(dirname "$worker_checkout")'"
  ssh "$WORKER_HOST" "if [ ! -d '$worker_checkout/.git' ]; then git clone '$VLLM_REPO_URL' '$worker_checkout'; fi"
  ssh "$WORKER_HOST" "cd '$worker_checkout' && git fetch origin '$VLLM_BRANCH' && git checkout '$VLLM_BRANCH' && git pull --ff-only origin '$VLLM_BRANCH'"
  ssh "$WORKER_HOST" "docker pull '$BASE_IMAGE'"
  ssh "$WORKER_HOST" "docker build -f '$worker_checkout/docker/Dockerfile.dspark-runtime-overlay' -t '$DSPARK_VLLM_IMAGE' '$worker_checkout'"
  ssh "$WORKER_HOST" "docker run --rm --entrypoint /opt/env/bin/python '$DSPARK_VLLM_IMAGE' -c \"import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; print('dspark overlay ok', d.__name__, p.__name__)\""
fi
