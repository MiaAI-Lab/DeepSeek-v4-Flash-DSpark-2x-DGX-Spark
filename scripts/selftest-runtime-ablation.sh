#!/usr/bin/env bash
# Manual no-GPU probe against the configured/pinned runtime image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.dspark}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
IMAGE="${DSPARK_VLLM_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.1}"
DIRECTION="${DSPARK_ABLATE_SOURCE_FILE:-$ROOT/files/direction_r1.pt}"
if [[ "$DIRECTION" != /* ]]; then DIRECTION="$ROOT/$DIRECTION"; fi

[ -f "$DIRECTION" ] || { echo "missing direction: $DIRECTION" >&2; exit 1; }
[ -f "$ROOT/patches/hotfix-dsv4-runtime-ablation.py" ] || exit 1

echo "Testing runtime ablation against $IMAGE"
docker run --rm --entrypoint bash \
  -e ABLATE=1 \
  -e DSV4_ABLATE_FILE=/tmp/direction_r1.pt \
  -e DSV4_ABLATE_LAMBDA=3.5 \
  -e DSV4_ABLATE_LAYERS=10-42 \
  -e VLLM_CACHE_ROOT=/tmp/vllm-cache \
  -v "$DIRECTION:/tmp/direction_r1.pt:ro" \
  -v "$ROOT/patches/hotfix-dsv4-runtime-ablation.py:/tmp/hotfix.py:ro" \
  -v "$ROOT/scripts/selftest-runtime-ablation.py:/tmp/selftest.py:ro" \
  "$IMAGE" -lc 'python3 /tmp/hotfix.py && python3 /tmp/selftest.py'
