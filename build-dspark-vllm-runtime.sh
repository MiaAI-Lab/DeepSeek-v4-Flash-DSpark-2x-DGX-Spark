#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"

DEFAULT_BASE_IMAGE="vllm-dspark-runtime:mia-raf-pr1"
DEFAULT_STAGE_C_IMAGE="vllm-dspark-runtime:dspark-nvfp4-stage-c"

# docker build -t cannot include @digest (BuildKit: "build tag cannot contain a
# digest", issue #173). Do not strip the digest and keep name:tag — that would
# retag the digest-pinned Anemll serve image as Stage-C.
dspark_docker_build_tag() {
  local ref="${1:?}"
  local fallback="${2:?}"
  case "$ref" in
    *@*) printf '%s' "$fallback" ;;
    *) printf '%s' "$ref" ;;
  esac
}

if [ "${1:-}" = "--tag-selftest" ]; then
  _fail=0
  _got="$(dspark_docker_build_tag \
    'ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:deadbeef' \
    "$DEFAULT_STAGE_C_IMAGE")"
  [ "$_got" = "$DEFAULT_STAGE_C_IMAGE" ] || _fail=1
  _got="$(dspark_docker_build_tag "$DEFAULT_STAGE_C_IMAGE" "$DEFAULT_STAGE_C_IMAGE")"
  [ "$_got" = "$DEFAULT_STAGE_C_IMAGE" ] || _fail=1
  _got="$(dspark_docker_build_tag "$DEFAULT_BASE_IMAGE" "$DEFAULT_BASE_IMAGE")"
  [ "$_got" = "$DEFAULT_BASE_IMAGE" ] || _fail=1
  exit "$_fail"
fi

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-$DEFAULT_STAGE_C_IMAGE}"
DSPARK_BASE_IMAGE="${DSPARK_BASE_IMAGE:-$DEFAULT_BASE_IMAGE}"
WORKER_BUILD="${WORKER_BUILD:-1}"

overlay_tag="$(dspark_docker_build_tag "$DSPARK_BASE_IMAGE" "$DEFAULT_BASE_IMAGE")"
stage_c_tag="$(dspark_docker_build_tag "$DSPARK_VLLM_IMAGE" "$DEFAULT_STAGE_C_IMAGE")"
if [ "$overlay_tag" != "$DSPARK_BASE_IMAGE" ]; then
  echo "DSPARK_BASE_IMAGE has a digest; docker build -t cannot use it (issue #173)." >&2
  echo "Tagging overlay as $overlay_tag (will not retag the digest-pinned ref)." >&2
fi
if [ "$stage_c_tag" != "$DSPARK_VLLM_IMAGE" ]; then
  echo "DSPARK_VLLM_IMAGE has a digest; docker build -t cannot use it (issue #173)." >&2
  echo "Tagging Stage-C as $stage_c_tag. Point DSPARK_VLLM_IMAGE at that tag to serve it; the digest pin is unchanged." >&2
fi

"$SCRIPT_DIR/scripts/verify-overlay-sources.sh"

build_one() {
  local host="$1"
  local checkout="$2"
  if [ "$host" = "local" ]; then
    docker build \
      -f "$SCRIPT_DIR/recipe/Dockerfile.dspark-runtime-overlay" \
      -t "$overlay_tag" \
      "$SCRIPT_DIR/recipe/overlay"
    docker run --rm --entrypoint /opt/env/bin/python "$overlay_tag" -c \
      "import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; assert d.map_dspark_stacked_param_name('model.layers.0.ffn.shared_experts.w1.weight') == ('model.layers.0.ffn.shared_experts.gate_up_proj.weight', 0); print('dspark overlay ok', d.__name__, p.__name__)"
    docker build \
      --build-arg BASE_IMAGE="$overlay_tag" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-a" \
      -t "$overlay_tag-nvfp4-a" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$overlay_tag-nvfp4-a" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-b" \
      -t "$overlay_tag-nvfp4-b" \
      "$SCRIPT_DIR"
    docker build \
      --build-arg BASE_IMAGE="$overlay_tag-nvfp4-b" \
      -f "$SCRIPT_DIR/recipe/nvfp4/Dockerfile.stage-c" \
      -t "$stage_c_tag" \
      "$SCRIPT_DIR"
    docker run --rm --entrypoint /opt/env/bin/python "$stage_c_tag" -c \
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
  build_one "$WORKER_HOST" "${WORKER_CHECKOUT:-${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}}"
fi
