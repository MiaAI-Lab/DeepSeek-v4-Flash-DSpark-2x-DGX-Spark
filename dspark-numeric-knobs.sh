#!/usr/bin/env bash
# Shared numeric-knob validation for the DSpark launcher and validator.
#
# MAX_NUM_SEQS / MTP_NUM_TOKENS / MAX_NUM_BATCHED_TOKENS are interpolated into
# bash arithmetic (and forwarded unevaluated into the container's own $(( )) via
# compose). Validate them the way VLLM_PORT is: reject non-integers, reject
# out-of-range/oversized values BEFORE any arithmetic (so a huge decimal cannot
# wrap 64-bit and slip through as a negative), and normalise with 10# (so a
# leading zero is decimal, never octal).
#
# Usage:  dspark_validate_numeric_knobs [ENV_SNAPSHOT]
#   ENV_SNAPSHOT (optional): a file whose matching KEY= lines are rewritten to the
#   normalised value. The launcher passes its private snapshot here so the worker,
#   which consumes it via `docker compose --env-file` over ssh (no exported vars),
#   sees the same decimal value as the head. Omit it (the validator) to skip.
# Returns 2 on any invalid value; callers under `set -e` should use `|| exit $?`.

dspark_validate_numeric_knobs() {
  local snapshot="${1:-}"
  local knob val
  # generous upper bounds — well above any real config, purely to bound arithmetic
  local -A _max=([MAX_NUM_SEQS]=4096 [MTP_NUM_TOKENS]=64 [MAX_NUM_BATCHED_TOKENS]=8388608)
  local -A _min=([MAX_NUM_SEQS]=1 [MTP_NUM_TOKENS]=0 [MAX_NUM_BATCHED_TOKENS]=1)
  for knob in MAX_NUM_SEQS MTP_NUM_TOKENS MAX_NUM_BATCHED_TOKENS; do
    val="${!knob:-}"
    [ -z "$val" ] && continue
    if ! [[ "$val" =~ ^[0-9]+$ ]]; then
      echo "$knob must be a non-negative integer: $val" >&2
      return 2
    fi
    # length guard: >10 digits could overflow 64-bit arithmetic before the bound
    # check below, so reject as out-of-range without evaluating it.
    if (( ${#val} > 10 )); then
      echo "$knob must be between ${_min[$knob]} and ${_max[$knob]}: $val" >&2
      return 2
    fi
    printf -v "$knob" '%d' "$((10#$val))"   # normalise: strip leading zeros, decimal
    if (( ${!knob} < _min[$knob] || ${!knob} > _max[$knob] )); then
      echo "$knob must be between ${_min[$knob]} and ${_max[$knob]}: $val" >&2
      return 2
    fi
    export "$knob"
    if [ -n "$snapshot" ] && [ -f "$snapshot" ] && grep -q "^$knob=" "$snapshot"; then
      sed -i "s|^$knob=.*|$knob=${!knob}|" "$snapshot"
    fi
  done
}
