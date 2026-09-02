#!/usr/bin/env bash
# Hotfix: DeepGEMM SM121 indexer-logits header alias (default OFF, opt-in).
#
# The vendored DeepGEMM in ghcr.io/anemll/dspark-vllm-gx10:0.1.1 names the
# Lightning-indexer logits kernels by the *exact* compute capability
# (GB10 = 12.1 -> "sm121_fp8_mqa_logits<...>" and
# "impls/sm121_fp8_mqa_logits.cuh") but ships only the sm120_* headers.
# Production works today only because the persisted JIT cache
# (VLLM_CACHE_ROOT/deep_gemm) holds cubins compiled from sm120_* by an earlier
# build; on any cache MISS (fresh volume, new cache root, a new head count, the
# fp4 indexer cache path) the compile fails:
#   "Failed to open .../impls/sm121_fp8_mqa_logits.cuh"
#   "identifier sm121_fp8_mqa_logits is undefined"
# and the engine dies on the first request that needs the kernel.
#
# This script writes four one-line alias headers that include the sm120_*
# implementation and #define the sm121_* symbol to it. SM120 and SM121 share
# the Blackwell consumer ISA; the code path is byte-identical to what the
# working cache entries were compiled from. Idempotent; --status reports.
set -euo pipefail

IMPLS="${DEEP_GEMM_IMPLS_DIR:-/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls}"
KERNELS=(fp8_mqa_logits fp8_paged_mqa_logits fp4_mqa_logits fp4_paged_mqa_logits)
MARK="// [dspark-deepgemm-sm121-alias]"

status() {
  local rc=0
  for k in "${KERNELS[@]}"; do
    if [ -f "$IMPLS/sm121_$k.cuh" ] && grep -qF "$MARK" "$IMPLS/sm121_$k.cuh"; then
      printf 'deepgemm sm121 alias %-24s: APPLIED\n' "$k"
    elif [ -f "$IMPLS/sm121_$k.cuh" ]; then
      printf 'deepgemm sm121 alias %-24s: PRESENT (not ours; left alone)\n' "$k"
    else
      printf 'deepgemm sm121 alias %-24s: NOT APPLIED\n' "$k"; rc=1
    fi
  done
  return $rc
}

if [ "${1:-}" = "--status" ]; then status; exit $?; fi

[ -d "$IMPLS" ] || { echo "FATAL: DeepGEMM impls dir missing: $IMPLS" >&2; exit 1; }
for k in "${KERNELS[@]}"; do
  src="$IMPLS/sm120_$k.cuh"; dst="$IMPLS/sm121_$k.cuh"
  [ -f "$src" ] || { echo "FATAL: $src missing; refusing to alias" >&2; exit 1; }
  if [ -f "$dst" ]; then
    grep -qF "$MARK" "$dst" && { echo "  [skip] $dst already applied"; continue; }
    echo "  [keep] $dst exists and is not ours; leaving it"; continue
  fi
  tmp="$(mktemp "$IMPLS/.sm121_$k.XXXXXX")"
  # DeepGEMM's include parser accepts only its own <deep_gemm/...> spelling.
  printf '%s\n#pragma once\n#include <deep_gemm/impls/sm120_%s.cuh>\n#define sm121_%s sm120_%s\n' "$MARK" "$k" "$k" "$k" > "$tmp"
  chmod 0644 "$tmp"; mv -f "$tmp" "$dst"
  echo "  [OK]   $dst -> sm120_$k.cuh"
done
echo "=== Verification ==="; status
