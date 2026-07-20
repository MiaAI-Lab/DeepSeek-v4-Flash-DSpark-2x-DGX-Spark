#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${DSPARK_VLLM_IMAGE:-dspark-r0b0tlab:production-candidate}"
PINNED_BASE_IMAGE="ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8"
BASE_IMAGE="${DSPARK_BASE_IMAGE:-${PINNED_BASE_IMAGE}}"
REVISION="${IMAGE_REVISION:-$(git -C "$ROOT" rev-parse HEAD)}"

if [[ -n "$(git -C "$ROOT" status --porcelain)" && "${ALLOW_DIRTY_BUILD:-0}" != "1" ]]; then
  echo "Refusing to label a dirty tree as revision ${REVISION}; commit first or set ALLOW_DIRTY_BUILD=1 for a non-release experiment." >&2
  exit 1
fi
if [[ "${BASE_IMAGE}" != "${PINNED_BASE_IMAGE}" && "${ALLOW_UNPINNED_BASE:-0}" != "1" ]]; then
  echo "Refusing unpinned production base: ${BASE_IMAGE}" >&2
  exit 1
fi

docker build --progress=plain \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "IMAGE_REVISION=${REVISION}" \
  -f "$ROOT/recipe/Dockerfile.production" \
  -t "$IMAGE" \
  "$ROOT"

docker run --rm --gpus all "$IMAGE" audit
printf 'DSPARK_PRODUCTION_IMAGE_BUILD_PASS image=%s\n' "$IMAGE"
