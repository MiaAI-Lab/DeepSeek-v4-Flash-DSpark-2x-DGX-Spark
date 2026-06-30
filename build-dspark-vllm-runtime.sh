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

DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
DSPARK_BASE_IMAGE="${DSPARK_BASE_IMAGE:-vllm-dspark-runtime:dspark-r0b0tlab-overlay}"
WORKER_BUILD="${WORKER_BUILD:-1}"

"$SCRIPT_DIR/scripts/verify-overlay-sources.sh"

build_one() {
  local host="$1"
  local checkout="$2"
  if [ "$host" = "local" ]; then
    docker build \
      -f "$SCRIPT_DIR/recipe/Dockerfile.dspark-runtime-overlay" \
      -t "$DSPARK_BASE_IMAGE" \
      "$SCRIPT_DIR/recipe/overlay"
    docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_BASE_IMAGE" -c \
      "import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; print('dspark overlay ok', d.__name__, p.__name__)"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-a" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-a" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-a" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-b" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-b" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-b" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-c" \
      -t "$DSPARK_VLLM_IMAGE" \
      "$SCRIPT_DIR"
    docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_VLLM_IMAGE" -c \
      "import vllm; print('dspark nvfp4 stage-c image ok', vllm.__version__)"
  else
    ssh "$host" "mkdir -p '$checkout'"
    rsync -az --delete "$SCRIPT_DIR/" "$host:$checkout/"
    ssh "$host" "cd '$checkout' && DSPARK_BASE_IMAGE='$DSPARK_BASE_IMAGE' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' WORKER_BUILD=0 ./build-dspark-vllm-runtime.sh"
  fi
}

build_one local "$SCRIPT_DIR"

if [ "$WORKER_BUILD" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  build_one "$WORKER_HOST" "${WORKER_CHECKOUT:-$SCRIPT_DIR}"
fi
