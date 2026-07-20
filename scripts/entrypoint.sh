#!/usr/bin/env bash
set -euo pipefail

AUDIT_BIN=/usr/local/bin/audit_runtime.py

reject_non_nvfp4_kv() {
  local previous=""
  for arg in "$@"; do
    if [[ "$previous" == "--kv-cache-dtype" && "$arg" != "nvfp4_ds_mla" ]]; then
      echo "production DSpark requires --kv-cache-dtype nvfp4_ds_mla" >&2
      exit 64
    fi
    if [[ "$arg" == --kv-cache-dtype=* && "${arg#*=}" != "nvfp4_ds_mla" ]]; then
      echo "production DSpark requires --kv-cache-dtype nvfp4_ds_mla" >&2
      exit 64
    fi
    previous="$arg"
  done
  if [[ -n "${KV_CACHE_DTYPE:-}" && "${KV_CACHE_DTYPE}" != "nvfp4_ds_mla" ]]; then
    echo "production DSpark requires KV_CACHE_DTYPE=nvfp4_ds_mla" >&2
    exit 64
  fi
}

if [[ "${1:-}" == "audit" ]]; then
  exec "$AUDIT_BIN"
fi
if (( $# == 0 )); then
  echo "explicit dual-node vLLM command required" >&2
  exit 64
fi
reject_non_nvfp4_kv "$@"
"$AUDIT_BIN"
exec "$@"
