#!/usr/bin/env bash
# SHA-256 gate for WebBrain MoonViT tower + PatchMerger.
set -euo pipefail

TOWER_SHA_EXPECTED="1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15"
PROJ_SHA_EXPECTED="7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a"

TOWER="${DSV4_MOONVIT_TOWER:-${1:-}}"
PROJ="${DSV4_MOONVIT_PROJECTOR:-${2:-}}"

if [[ -z "${TOWER}" || -z "${PROJ}" ]]; then
  DEFAULT_SRC="${HF_HOME:-${HOME}/.cache/huggingface}/webbrain-0731-moonvit-src"
  TOWER="${TOWER:-${DEFAULT_SRC}/vision_tower.safetensors}"
  PROJ="${PROJ:-${DEFAULT_SRC}/mm_projector.safetensors}"
fi

echo "tower: ${TOWER}"
echo "projector: ${PROJ}"
[[ -f "${TOWER}" ]] || { echo "missing tower" >&2; exit 1; }
[[ -f "${PROJ}" ]] || { echo "missing projector" >&2; exit 1; }

TSHA=$(sha256sum "${TOWER}" | awk '{print $1}')
PSHA=$(sha256sum "${PROJ}" | awk '{print $1}')
echo "tower_sha256=${TSHA}"
echo "projector_sha256=${PSHA}"

ok=0
if [[ "${TSHA}" == "${TOWER_SHA_EXPECTED}" ]]; then
  echo "tower: MATCH"
else
  echo "tower: FAIL (expected ${TOWER_SHA_EXPECTED})" >&2
  ok=1
fi
if [[ "${PSHA}" == "${PROJ_SHA_EXPECTED}" ]]; then
  echo "projector: MATCH"
else
  echo "projector: FAIL (expected ${PROJ_SHA_EXPECTED})" >&2
  ok=1
fi
exit "${ok}"
