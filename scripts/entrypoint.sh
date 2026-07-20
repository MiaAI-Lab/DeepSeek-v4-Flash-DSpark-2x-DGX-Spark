#!/usr/bin/env bash
set -euo pipefail

AUDIT_BIN=/usr/local/bin/audit_runtime.py

reject_non_nvfp4_kv() {
  local joined=" $* "
  local command_name="${1##*/}"
  local kv_flag_count

  if [[ "${KV_CACHE_DTYPE:-}" != "nvfp4_ds_mla" ]]; then
    echo "production DSpark requires KV_CACHE_DTYPE=nvfp4_ds_mla" >&2
    exit 64
  fi
  if [[ "${command_name}" == "vllm" ]]; then
    [[ " ${*:2} " == *" serve "* ]] || {
      echo "production image requires the vLLM serve subcommand" >&2
      exit 64
    }
  elif [[ "${command_name}" == "bash" && "${2:-}" == "-lc" ]]; then
    [[ "${3:-}" == *"VLLM_BIN"*" serve "* ]] || {
      echo "production image rejects a shell command that is not the audited vLLM launcher" >&2
      exit 64
    }
  else
    echo "production image accepts only an explicit vLLM serve command or the audit subcommand" >&2
    exit 64
  fi
  kv_flag_count="$(grep -o -- '--kv-cache-dtype' <<<"${joined}" | wc -l || true)"
  if [[ "${kv_flag_count}" != "1" ]]; then
    echo "production DSpark requires exactly one explicit --kv-cache-dtype argument" >&2
    exit 64
  fi
  if [[ ! "${joined}" =~ --kv-cache-dtype(=|[[:space:]])[\"\']?nvfp4_ds_mla[\"\']?([^[:alnum:]_]|$) ]]; then
    echo "production DSpark requires an explicit --kv-cache-dtype nvfp4_ds_mla argument" >&2
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
