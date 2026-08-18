#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-100}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"
ENABLE_VLLM_GB10_PATCH="${ENABLE_VLLM_GB10_PATCH:-0}"
VLLM_GB10_PATCH_DIR="${VLLM_GB10_PATCH_DIR:-$SCRIPT_DIR/vllm_patch_gb10}"
DSPARK_PROPOSER_FILE="${DSPARK_PROPOSER_FILE:-$SCRIPT_DIR/recipe/vllm/v1/spec_decode/dspark_proposer.py}"
CLI_VLLM_HOST=""
CLI_VLLM_PORT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--host HOST] [--port PORT]

Options:
  --host HOST  vLLM API bind address (default: VLLM_HOST or 127.0.0.1)
  --port PORT  vLLM API listen port (default: VLLM_PORT or 8888)
  -h, --help   Show this help message

Command-line options override values from $ENV_FILE.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "--host requires a value." >&2; exit 2; }
      CLI_VLLM_HOST="$2"
      shift 2
      ;;
    --host=*)
      CLI_VLLM_HOST="${1#*=}"
      [ -n "$CLI_VLLM_HOST" ] || { echo "--host requires a value." >&2; exit 2; }
      shift
      ;;
    --port)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "--port requires a value." >&2; exit 2; }
      CLI_VLLM_PORT="$2"
      shift 2
      ;;
    --port=*)
      CLI_VLLM_PORT="${1#*=}"
      [ -n "$CLI_VLLM_PORT" ] || { echo "--port requires a value." >&2; exit 2; }
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      [ "$#" -eq 0 ] || { echo "Unexpected positional argument: $1" >&2; usage >&2; exit 2; }
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.dspark.example to .env.dspark and edit node-specific values." >&2
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "Missing $COMPOSE_FILE." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Vision mode flag selects 0731 GPU util (and whether the VL sidecar starts).
#   ENABLE_VL_SIDECAR=1 → vision coexist → GPU_MEMORY_UTILIZATION_VISION (default 0.80)
#   ENABLE_VL_SIDECAR=0 → text-only     → GPU_MEMORY_UTILIZATION_TEXT   (default 0.835)
# Explicit GPU_MEMORY_UTILIZATION in the env file is overridden by this profile
# so one flag is enough to switch modes safely.
if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ]; then
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_VISION:-0.80}"
  DSPARK_SERVE_MODE="vision"
else
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_TEXT:-0.835}"
  DSPARK_SERVE_MODE="text"
fi
export GPU_MEMORY_UTILIZATION ENABLE_VL_SIDECAR DSPARK_SERVE_MODE

# Checkpoint flag: official 0731 vs Keys abliterated weights.
#   ABLITERATED=0 → DSPARK_MODEL_OFFICIAL
#   ABLITERATED=1 → DSPARK_MODEL_ABLITERATED
DSPARK_MODEL_OFFICIAL="${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-0731}"
DSPARK_MODEL_ABLITERATED="${DSPARK_MODEL_ABLITERATED:-drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32}"
DEFAULT_OFFICIAL_REVISION="9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
if [ "${ABLITERATED:-0}" = "1" ]; then
  DSPARK_MODEL="$DSPARK_MODEL_ABLITERATED"
  DSPARK_REVISION="${DSPARK_REVISION_ABLITERATED:-}"
else
  DSPARK_MODEL="$DSPARK_MODEL_OFFICIAL"
  if [ -z "${DSPARK_REVISION+x}" ]; then
    DSPARK_REVISION="$DEFAULT_OFFICIAL_REVISION"
  fi
fi
export ABLITERATED DSPARK_MODEL DSPARK_MODEL_OFFICIAL DSPARK_MODEL_ABLITERATED DSPARK_REVISION

# CLI values have highest precedence; the env file remains the persistent
# configuration source when no command-line override is provided.
VLLM_HOST="${CLI_VLLM_HOST:-${VLLM_HOST:-127.0.0.1}}"
VLLM_PORT="${CLI_VLLM_PORT:-${VLLM_PORT:-${PORT:-8888}}}"
if [ -z "$VLLM_HOST" ]; then
  echo "VLLM host must not be empty." >&2
  exit 2
fi
if ! [[ "$VLLM_PORT" =~ ^[0-9]+$ ]]; then
  echo "VLLM port must be an integer between 1 and 65535: $VLLM_PORT" >&2
  exit 2
fi
if (( 10#$VLLM_PORT < 1 || 10#$VLLM_PORT > 65535 )); then
  echo "VLLM port must be between 1 and 65535: $VLLM_PORT" >&2
  exit 2
fi
VLLM_PORT="$((10#$VLLM_PORT))"
# Keep PORT as a backwards-compatible alias, but use VLLM_PORT internally.
PORT="$VLLM_PORT"
DEFAULT_THINKING="${DEFAULT_THINKING:-low}"
case "$DEFAULT_THINKING" in
  off|low|high|max) ;;
  *)
    echo "DEFAULT_THINKING must be one of: off, low, high, max (got: $DEFAULT_THINKING)" >&2
    exit 2
    ;;
esac
export VLLM_HOST VLLM_PORT PORT DEFAULT_THINKING

# A wildcard is valid for binding but not a useful health-check destination.
API_HOST="${API_HOST:-$VLLM_HOST}"
case "$API_HOST" in
  0.0.0.0|::|\[::\]) API_HOST="127.0.0.1" ;;
esac
URL_HOST="$API_HOST"
if [[ "$URL_HOST" == *:* && "$URL_HOST" != \[*\] ]]; then
  URL_HOST="[$URL_HOST]"
fi
API_URL="${API_URL:-http://$URL_HOST:$VLLM_PORT/v1/models}"
CHAT_URL="${CHAT_URL:-http://$URL_HOST:$VLLM_PORT/v1/chat/completions}"
AUTH_HEADER_ARGS=()
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${MASTER_PORT:?MASTER_PORT must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"
: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"

VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-$WORKER_HOST}"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
# Per-node CX7/RoCE pins (3-node ring: facing ports often differ by hostname).
# Set WORKER_NCCL_* in the head .env; start script injects them on remote compose.
# Do not put WORKER_* first in docker-compose substitution — that is not rank-aware.
WORKER_NCCL_IB_HCA="${WORKER_NCCL_IB_HCA:-$NCCL_IB_HCA}"
WORKER_NCCL_SOCKET_IFNAME="${WORKER_NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
WORKER_TP_SOCKET_IFNAME="${WORKER_TP_SOCKET_IFNAME:-${TP_SOCKET_IFNAME:-$WORKER_NCCL_SOCKET_IFNAME}}"
WORKER_GLOO_SOCKET_IFNAME="${WORKER_GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME:-$WORKER_NCCL_SOCKET_IFNAME}}"
# RoCEv2 GID index differs per node and drifts after reboot/link events.
# Default: resolve from sysfs at launch (NCCL_IB_GID_AUTO=1). Do not reuse one
# literal for both ranks — that wedges NCCL with "unhandled system error".
# Set NCCL_IB_GID_AUTO=0 and pin NCCL_IB_GID_INDEX / WORKER_NCCL_IB_GID_INDEX
# only if you need a manual override.
NCCL_IB_GID_AUTO="${NCCL_IB_GID_AUTO:-1}"
# Optional match IPs if the RoCE address is not on NCCL_SOCKET_IFNAME /
# WORKER_NCCL_SOCKET_IFNAME (rare). Prefer interface IPv4 when unset.
NCCL_IB_GID_MATCH_IP="${NCCL_IB_GID_MATCH_IP:-}"
WORKER_NCCL_IB_GID_MATCH_IP="${WORKER_NCCL_IB_GID_MATCH_IP:-}"
# Preserve env pins for AUTO=0; do NOT default worker to head index before resolve.
ENV_NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
ENV_WORKER_NCCL_IB_GID_INDEX="${WORKER_NCCL_IB_GID_INDEX:-}"
WORKER_NCCL_IB_GID_INDEX="${ENV_WORKER_NCCL_IB_GID_INDEX}"
REMOTE_WORKER_DIR="$(printf '%q' "$WORKER_DIR")"
REMOTE_COMPOSE_FILE="$REMOTE_WORKER_DIR/docker-compose.dspark.yml"
REMOTE_ENV_FILE="$REMOTE_WORKER_DIR/.env.dspark"
REMOTE_VLLM_GB10_PATCH_DIR="$REMOTE_WORKER_DIR/vllm_patch_gb10"
REMOTE_COMPOSE="cd $REMOTE_WORKER_DIR && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"
STARTUP_LOG_SINCE=""

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

# Strip user@ from ssh targets / host strings → bare host or IPv4.
host_without_user() {
  local h="$1"
  if [[ "$h" == *@* ]]; then
    printf '%s' "${h##*@}"
  else
    printf '%s' "$h"
  fi
}

ipv4_to_gid_suffix() {
  # IPv4-mapped RoCEv2 GID ends with ffff:aabb:ccdd for a.b.c.d
  local ip="$1" a b c d
  IFS=. read -r a b c d <<<"$ip" || return 1
  printf '%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
}

# First IPv4 on an interface: empty host = local, else ssh target.
iface_ipv4() {
  local ssh_target="$1" ifname="$2"
  local cmd
  cmd="ip -4 -o addr show dev $(printf '%q' "$ifname") 2>/dev/null | awk '{print \$4}' | head -1 | cut -d/ -f1"
  if [ -z "$ssh_target" ]; then
    bash -c "$cmd"
  else
    # shellcheck disable=SC2029
    ssh "$ssh_target" "$cmd"
  fi
}

# Resolve RoCEv2 GID index for HCA whose GID embeds match_ip.
# $1=ssh target (empty=local)  $2=HCA  $3=IPv4 to match
resolve_rocev2_gid_index() {
  local ssh_target="$1" hca="$2" match_ip="$3"
  local hex remote
  hex="$(ipv4_to_gid_suffix "$match_ip")" || return 1
  remote=$(
    cat <<EOF
hca=$(printf '%q' "$hca")
hex=$(printf '%q' "$hex")
for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
  [ -e "\$g" ] || continue
  i=\${g##*/}
  t=\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null || true)
  [ "\$t" = "RoCE v2" ] || continue
  case \$(cat "\$g" 2>/dev/null) in
    *ffff:\${hex}) echo "\$i"; exit 0 ;;
  esac
done
exit 1
EOF
  )
  if [ -z "$ssh_target" ]; then
    bash -c "$remote"
  else
    # shellcheck disable=SC2029
    ssh "$ssh_target" "bash -s" <<<"$remote"
  fi
}

pick_gid_match_ip() {
  # $1=ssh  $2=ifname  $3=explicit match  $4=fallback vllm ip  $5=fallback host/ip
  local ssh_target="$1" ifname="$2" explicit="$3" vllm_ip="$4" fallback="$5"
  local ip
  if [ -n "$explicit" ]; then
    printf '%s' "$explicit"
    return 0
  fi
  ip="$(iface_ipv4 "$ssh_target" "$ifname" || true)"
  if [ -n "$ip" ]; then
    printf '%s' "$ip"
    return 0
  fi
  if [ -n "$vllm_ip" ] && [[ "$vllm_ip" != *@* ]] && [[ "$vllm_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s' "$vllm_ip"
    return 0
  fi
  fallback="$(host_without_user "$fallback")"
  if [[ "$fallback" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf '%s' "$fallback"
    return 0
  fi
  return 1
}

resolve_nccl_gid_indexes() {
  local head_match worker_match resolved_head resolved_worker

  if [ "$NCCL_IB_GID_AUTO" = "0" ]; then
    NCCL_IB_GID_INDEX="${ENV_NCCL_IB_GID_INDEX:-}"
    WORKER_NCCL_IB_GID_INDEX="${ENV_WORKER_NCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}"
    if [ -z "$NCCL_IB_GID_INDEX" ] || [ -z "$WORKER_NCCL_IB_GID_INDEX" ]; then
      echo "NCCL_IB_GID_AUTO=0 requires NCCL_IB_GID_INDEX and preferably WORKER_NCCL_IB_GID_INDEX in $ENV_FILE." >&2
      exit 1
    fi
    echo "Using pinned NCCL GID indexes (auto off): head=$NCCL_IB_GID_INDEX worker=$WORKER_NCCL_IB_GID_INDEX"
    return 0
  fi

  head_match="$(pick_gid_match_ip "" "$NCCL_SOCKET_IFNAME" "$NCCL_IB_GID_MATCH_IP" "$VLLM_HOST_IP" "$MASTER_ADDR")" || {
    echo "FATAL: could not determine head RoCE IPv4 for GID match (if=$NCCL_SOCKET_IFNAME)." >&2
    exit 1
  }
  worker_match="$(pick_gid_match_ip "$WORKER_HOST" "$WORKER_NCCL_SOCKET_IFNAME" "$WORKER_NCCL_IB_GID_MATCH_IP" "$WORKER_VLLM_HOST_IP" "$WORKER_HOST")" || {
    echo "FATAL: could not determine worker RoCE IPv4 for GID match (if=$WORKER_NCCL_SOCKET_IFNAME)." >&2
    exit 1
  }

  echo "Resolving RoCEv2 GID indexes from sysfs (head if=$NCCL_SOCKET_IFNAME ip=$head_match hca=$NCCL_IB_HCA; worker if=$WORKER_NCCL_SOCKET_IFNAME ip=$worker_match hca=$WORKER_NCCL_IB_HCA)..."
  resolved_head="$(resolve_rocev2_gid_index "" "$NCCL_IB_HCA" "$head_match")" || {
    echo "FATAL: could not resolve head RoCEv2 GID index for $NCCL_IB_HCA / $head_match." >&2
    echo "Check: ibstat | grep -A3 $NCCL_IB_HCA ; show_gids | grep $NCCL_IB_HCA" >&2
    exit 1
  }
  resolved_worker="$(resolve_rocev2_gid_index "$WORKER_HOST" "$WORKER_NCCL_IB_HCA" "$worker_match")" || {
    echo "FATAL: could not resolve worker RoCEv2 GID index for $WORKER_NCCL_IB_HCA / $worker_match." >&2
    echo "Check on worker: show_gids | grep $WORKER_NCCL_IB_HCA" >&2
    exit 1
  }

  if [ -n "$ENV_NCCL_IB_GID_INDEX" ] && [ "$ENV_NCCL_IB_GID_INDEX" != "$resolved_head" ]; then
    echo "Note: $ENV_FILE has NCCL_IB_GID_INDEX=$ENV_NCCL_IB_GID_INDEX but sysfs resolved head=$resolved_head (using resolved)."
  fi
  if [ -n "$ENV_WORKER_NCCL_IB_GID_INDEX" ] && [ "$ENV_WORKER_NCCL_IB_GID_INDEX" != "$resolved_worker" ]; then
    echo "Note: $ENV_FILE has WORKER_NCCL_IB_GID_INDEX=$ENV_WORKER_NCCL_IB_GID_INDEX but sysfs resolved worker=$resolved_worker (using resolved)."
  fi

  NCCL_IB_GID_INDEX="$resolved_head"
  WORKER_NCCL_IB_GID_INDEX="$resolved_worker"
  echo "RoCEv2 GID index: head=$NCCL_IB_GID_INDEX (match $head_match) worker=$WORKER_NCCL_IB_GID_INDEX (match $worker_match)"
}

remote_nccl_env() {
  # Rebuild each call so GID resolve after early init is visible on the worker.
  printf "NCCL_IB_HCA='%s' NCCL_SOCKET_IFNAME='%s' TP_SOCKET_IFNAME='%s' GLOO_SOCKET_IFNAME='%s' NCCL_IB_GID_INDEX='%s' VLLM_HOST='%s' VLLM_PORT='%s'" \
    "$WORKER_NCCL_IB_HCA" \
    "$WORKER_NCCL_SOCKET_IFNAME" \
    "$WORKER_TP_SOCKET_IFNAME" \
    "$WORKER_GLOO_SOCKET_IFNAME" \
    "$WORKER_NCCL_IB_GID_INDEX" \
    "$VLLM_HOST" \
    "$VLLM_PORT"
}

compose_base() {
  env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
    WORKER_HOST="$WORKER_HOST" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    NCCL_IB_HCA="$NCCL_IB_HCA" \
    NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}" \
    VLLM_HOST="$VLLM_HOST" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_HOST_IP="$VLLM_HOST_IP" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    DSPARK_MODEL="$DSPARK_MODEL" \
    DSPARK_REVISION="${DSPARK_REVISION:-}" \
    ENABLE_VLLM_GB10_PATCH="$ENABLE_VLLM_GB10_PATCH" \
    VLLM_GB10_PATCH_DIR="$VLLM_GB10_PATCH_DIR" \
    GB10_HYBRID_NVFP4_M_THRESHOLD="${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}" \
    NODE_RANK="$1" \
    HEADLESS="$2" \
    docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "${@:3}"
}

remote_compose() {
  ssh "$WORKER_HOST" "$REMOTE_COMPOSE $(remote_nccl_env) $*"
}

log_since() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

print_startup_logs() {
  local since="$1"

  compose_base 0 "" logs --since "$since" vllm-dspark || true
  remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" || true
}

wait_with_startup_logs() {
  local since
  since="$(log_since)"

  sleep "$WAIT_SECONDS"
  print_startup_logs "$since"
}

print_initial_startup_logs() {
  compose_base 0 "" logs --tail=100 vllm-dspark || true
  remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=100 vllm-dspark" || true
}

print_failure_logs() {
  local since="${STARTUP_LOG_SINCE:-$(log_since)}"

  echo "Startup failed. Recent head logs:" >&2
  compose_base 0 "" logs --since "$since" vllm-dspark >&2 || true
  echo "Recent worker logs:" >&2
  remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" >&2 || true
}

on_error() {
  local status=$?
  trap - ERR
  print_failure_logs
  exit "$status"
}

print_resolved_profile() {
  echo "Resolved DSpark profile:"
  echo "  project: $PROJECT_NAME"
  echo "  serve mode: $DSPARK_SERVE_MODE (ENABLE_VL_SIDECAR=${ENABLE_VL_SIDECAR:-0})"
  echo "  checkpoint: $DSPARK_MODEL (ABLITERATED=${ABLITERATED:-0})"
  if [ -n "${DSPARK_REVISION:-}" ]; then
    echo "  revision: $DSPARK_REVISION"
  else
    echo "  revision: (default branch tip / unpinned)"
  fi
  echo "  image: $DSPARK_VLLM_IMAGE"
  echo "  model: ${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
  echo "  served model: ${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
  echo "  max model len: ${MAX_MODEL_LEN:-1000000}"
  echo "  max num seqs: ${MAX_NUM_SEQS:-12}"
  echo "  max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-8192}"
  echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION:-0.80} (text default ${GPU_MEMORY_UTILIZATION_TEXT:-0.835} / vision default ${GPU_MEMORY_UTILIZATION_VISION:-0.80})"
  echo "  mtp speculative tokens: ${MTP_NUM_TOKENS:-5} (dspark_block_size min is 5)"
  echo "  default thinking: $DEFAULT_THINKING (off/low/high/max)"
  echo "  cudagraph capture size: $(( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-5} + 1) ))"
  echo "  API bind: $VLLM_HOST:$VLLM_PORT"
  echo "  API probe: $API_URL"
  echo "  head fabric IP: $VLLM_HOST_IP"
  echo "  worker host/ip: $WORKER_HOST / $WORKER_VLLM_HOST_IP"
  echo "  head NCCL HCA/if: $NCCL_IB_HCA / $NCCL_SOCKET_IFNAME"
  echo "  worker NCCL HCA/if: $WORKER_NCCL_IB_HCA / $WORKER_NCCL_SOCKET_IFNAME"
  echo "  NCCL_IB_GID_AUTO: $NCCL_IB_GID_AUTO"
  echo "  head NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-}"
  echo "  worker NCCL_IB_GID_INDEX: ${WORKER_NCCL_IB_GID_INDEX:-}"
  echo "  worker dir: $WORKER_DIR"
  echo "  worker cache: ${WORKER_HF_CACHE:-${HF_CACHE:-}}"
  echo "  GB10 vLLM patch: $ENABLE_VLLM_GB10_PATCH"
  if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ]; then
    echo "  VL sidecar: ${VL_SIDECAR_MODEL:-cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit} TP=${VL_SIDECAR_TP_SIZE:-2} nnodes=${VL_SIDECAR_NNODES:-2} on 127.0.0.1:${VL_SIDECAR_PORT:-8889} (util ${VL_SIDECAR_GPU_UTIL:-0.04}/GPU, kv ${VL_SIDECAR_KV_CACHE_DTYPE:-int4_per_token_head}, master-port ${VL_SIDECAR_MASTER_PORT:-25100})"
    echo "  vision MCP install: ${INSTALL_VISION_MCP:-1} (only when ENABLE_VL_SIDECAR=1; harnesses: ${VISION_MCP_HARNESSES:-auto})"
  else
    echo "  VL sidecar: disabled (text-only 0731)"
  fi
  if [ -f "$SCRIPT_DIR/patches/hotfix-nvfp4-ds-mla-issue22.sh" ]; then
    if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" = "1" ]; then
      echo "  Issue #22 hotfix: SKIPPED (DSPARK_SKIP_ISSUE22_HOTFIX=1)"
    else
      echo "  Issue #22 hotfix: will apply on start (not gated by DSPARK_SKIP_HOTFIX)"
    fi
  else
    echo "  Issue #22 hotfix: not found"
  fi
  if [ "${DSPARK_SKIP_HOTFIX:-0}" = "1" ]; then
    echo "  DSV4 perf hotfixes (#50312/#50004/#49486/#48407/#48957/#50298/#44993-grammar): SKIPPED (DSPARK_SKIP_HOTFIX=1)"
  else
    echo "  DSV4 perf hotfixes (#50312/#50004/#49486/#48407/#48957/#50298/#44993-grammar): will apply on start"
  fi
  if [ "${DSPARK_SKIP_SPIN_WAIT_HOTFIX:-0}" = "1" ]; then
    echo "  GB10 shm spin-wait hotfix (#79): SKIPPED (DSPARK_SKIP_SPIN_WAIT_HOTFIX=1)"
  else
    echo "  GB10 shm spin-wait hotfix (#79): will apply on start (busy_loop_s 1s -> 2ms)"
  fi
  if [ "${DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX:-0}" = "1" ]; then
    echo "  Suppress stops in <think>: SKIPPED (DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=1)"
  elif [ "${DSPARK_SUPPRESS_STOPS_IN_REASONING:-${VLLM_SUPPRESS_STOPS_IN_REASONING:-1}}" = "0" ]; then
    echo "  Suppress stops in <think>: hotfix applies but guard off (DSPARK_SUPPRESS_STOPS_IN_REASONING=0)"
  else
    echo "  Suppress stops in <think>: will apply (client stop dormant until </think>)"
  fi
  if [ "$ENABLE_VLLM_GB10_PATCH" = "1" ]; then
    echo "  GB10 vLLM patch dir: $VLLM_GB10_PATCH_DIR"
    echo "  GB10 hybrid NVFP4 M threshold: ${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}"
  fi
}

validate_compose() {
  echo "Validating head compose config..."
  compose_base 0 "" config --quiet
  echo "Validating worker compose config..."
  remote_compose "NODE_RANK=1 HEADLESS=1 HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml config --quiet"
}

need_cmd docker
need_cmd ssh
need_cmd scp
need_cmd curl

if [ "$ENABLE_VLLM_GB10_PATCH" != "0" ] && [ "$ENABLE_VLLM_GB10_PATCH" != "1" ]; then
  echo "ENABLE_VLLM_GB10_PATCH must be 0 or 1." >&2
  exit 1
fi

if [ "$ENABLE_VLLM_GB10_PATCH" = "1" ] && [ ! -d "$VLLM_GB10_PATCH_DIR" ]; then
  echo "Missing GB10 vLLM patch directory: $VLLM_GB10_PATCH_DIR" >&2
  exit 1
fi

if [ ! -f "$DSPARK_PROPOSER_FILE" ]; then
  echo "Missing DSpark proposer bind-mount source: $DSPARK_PROPOSER_FILE" >&2
  exit 1
fi

docker compose version >/dev/null
docker image inspect "$DSPARK_VLLM_IMAGE" >/dev/null || {
  echo "Missing local Docker image $DSPARK_VLLM_IMAGE." >&2
  echo "Pull it (e.g. docker pull $DSPARK_VLLM_IMAGE) or run ./build-dspark-vllm-runtime.sh for a local Stage-C build." >&2
  exit 1
}

ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "true" >/dev/null || {
  echo "Cannot reach worker with passwordless SSH: $WORKER_HOST" >&2
  exit 1
}

ssh "$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' >/dev/null" || {
  echo "Missing worker Docker image $DSPARK_VLLM_IMAGE." >&2
  echo "Pull it on the worker (e.g. docker pull $DSPARK_VLLM_IMAGE) or run ./build-dspark-vllm-runtime.sh." >&2
  exit 1
}

already_running_hint() {
  echo "This is not a failed start: dockerd likely restored ranks after a reboot (compose restart: unless-stopped). The cluster may already be serving. Run ./stop-deepseek-v4-flash-dspark.sh only if you want a cold start. Supervisors: treat exit 3 as already-up (systemd SuccessExitStatus=3)." >&2
}

if docker ps --format '{{.Names}}' | grep -qx "${PROJECT_NAME}-vllm-dspark-1"; then
  echo "DSpark head container already exists for project $PROJECT_NAME. Stop it first or use PROJECT_NAME=..." >&2
  already_running_hint
  exit 3
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "( sport = :$VLLM_PORT )" | tail -n +2 | grep -q .; then
  echo "Port $VLLM_PORT is already listening on the head node. Stop the conflicting service first." >&2
  exit 1
fi

if ssh "$WORKER_HOST" "if docker ps --format '{{.Names}}' | grep -qx '${PROJECT_NAME}-vllm-dspark-1'; then echo 'DSpark worker container already exists for project $PROJECT_NAME (head is not up — likely a stale rank after a head-only reboot). Stop it first.' >&2; exit 1; fi"; then
  :
else
  worker_rc=$?
  echo "Cannot start: worker check on $WORKER_HOST failed (ssh exit $worker_rc)." >&2
  exit "$worker_rc"
fi

cd "$SCRIPT_DIR"
resolve_nccl_gid_indexes
STARTUP_LOG_SINCE="$(log_since)"
trap on_error ERR
print_resolved_profile

echo "Syncing DSpark deployment files to ${WORKER_HOST}:${WORKER_DIR}"
ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR"
scp "$COMPOSE_FILE" "${WORKER_HOST}:${REMOTE_COMPOSE_FILE}"
scp "$ENV_FILE" "${WORKER_HOST}:${REMOTE_ENV_FILE}"
SIDECAR_COMPOSE_FILE="${SIDECAR_COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.vl-sidecar.yml}"
if [ -f "$SIDECAR_COMPOSE_FILE" ]; then
  scp "$SIDECAR_COMPOSE_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/docker-compose.vl-sidecar.yml"
fi
ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR/recipe/vllm/v1/spec_decode"
scp "$DSPARK_PROPOSER_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/recipe/vllm/v1/spec_decode/dspark_proposer.py"
DSPARK_HOTFIX_FILE="$SCRIPT_DIR/patches/hotfix-nvfp4-ds-mla-issue22.sh"
if [ -f "$DSPARK_HOTFIX_FILE" ]; then
  echo "Syncing Issue #22 hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_HOTFIX_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-nvfp4-ds-mla-issue22.sh"
fi
DSPARK_SPIN_WAIT_HOTFIX="${DSPARK_SPIN_WAIT_HOTFIX:-$SCRIPT_DIR/patches/hotfix-gb10-spin-wait.sh}"
if [ -f "$DSPARK_SPIN_WAIT_HOTFIX" ]; then
  echo "Syncing GB10 shm spin-wait hotfix (#79) to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_SPIN_WAIT_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-gb10-spin-wait.sh"
fi
# DSV4 v0.27 .sh hotfixes — entrypoint applies them before exec vllm (issue #38).
for _hf_sync in hotfix-dsv4-mtp-buffer-50312.sh hotfix-dsv4-adaptive-topk-50004.sh hotfix-dsv4-skip-topk-49486.sh hotfix-dsv4-dense-prefill-indexer-48407.sh hotfix-dsv4-skip-empty-c128-48957.sh hotfix-dsv4-flashmla-workspace-50298.sh hotfix-dsv4-grammar-advance.sh; do
  if [ -f "$SCRIPT_DIR/patches/$_hf_sync" ]; then
    echo "Syncing $_hf_sync to ${WORKER_HOST}:${WORKER_DIR}/patches/"
    ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
    scp "$SCRIPT_DIR/patches/$_hf_sync" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/$_hf_sync"
  fi
done
DSPARK_ENCODING_ISSUE21_HOTFIX="${DSPARK_ENCODING_ISSUE21_HOTFIX:-$SCRIPT_DIR/patches/hotfix-encoding-dsv4-issue21.py}"
if [ -f "$DSPARK_ENCODING_ISSUE21_HOTFIX" ]; then
  echo "Syncing Issue #21 encoding hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ENCODING_ISSUE21_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-encoding-dsv4-issue21.py"
fi
DSPARK_ISSUE31_GPU_HOTFIX="${DSPARK_ISSUE31_GPU_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py}"
if [ -f "$DSPARK_ISSUE31_GPU_HOTFIX" ]; then
  echo "Syncing GPU-resident V2 thinking-budget hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE31_GPU_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py"
fi
DSPARK_ISSUE55_HOTFIX="${DSPARK_ISSUE55_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue55-tool-truncation.py}"
if [ -f "$DSPARK_ISSUE55_HOTFIX" ]; then
  echo "Syncing Issue #55 tool-call truncation hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE55_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue55-tool-truncation.py"
fi
DSPARK_ISSUE27_HOTFIX="${DSPARK_ISSUE27_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py}"
if [ -f "$DSPARK_ISSUE27_HOTFIX" ]; then
  echo "Syncing Issue #27 partial-prefill hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE27_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
fi
DSPARK_ISSUE43_HOTFIX="${DSPARK_ISSUE43_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py}"
if [ -f "$DSPARK_ISSUE43_HOTFIX" ]; then
  echo "Syncing Issue #43 decode-fairness hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE43_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
fi
DSPARK_ISSUE26_HOTFIX="${DSPARK_ISSUE26_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue26-hybrid-swa-min.py}"
if [ -f "$DSPARK_ISSUE26_HOTFIX" ]; then
  echo "Syncing Issue #26 hybrid-SWA-min hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE26_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue26-hybrid-swa-min.py"
fi
DSPARK_SUPPRESS_STOPS_HOTFIX="${DSPARK_SUPPRESS_STOPS_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-suppress-stops-in-reasoning.py}"
if [ -f "$DSPARK_SUPPRESS_STOPS_HOTFIX" ]; then
  echo "Syncing suppress-stops-in-reasoning hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  # A leftover directory with this name (root-owned) would make scp fail.
  ssh "$WORKER_HOST" "if [ -d '${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-suppress-stops-in-reasoning.py' ]; then docker run --rm -v '${REMOTE_WORKER_DIR}/patches:/p' alpine:3.20 rm -rf /p/hotfix-dsv4-suppress-stops-in-reasoning.py; fi"
  scp "$DSPARK_SUPPRESS_STOPS_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-suppress-stops-in-reasoning.py"
fi
DSPARK_ASSISTANT_FINAL_HOTFIX="${DSPARK_ASSISTANT_FINAL_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-assistant-final-continuation.py}"
if [ -f "$DSPARK_ASSISTANT_FINAL_HOTFIX" ]; then
  echo "Syncing assistant-final continuation hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ASSISTANT_FINAL_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-assistant-final-continuation.py"
fi
if [ "$ENABLE_VLLM_GB10_PATCH" = "1" ]; then
  echo "Syncing GB10 vLLM patch to ${WORKER_HOST}:${WORKER_DIR}/vllm_patch_gb10"
  tar -C "$VLLM_GB10_PATCH_DIR" \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - . | ssh "$WORKER_HOST" "mkdir -p $REMOTE_VLLM_GB10_PATCH_DIR && tar -C $REMOTE_VLLM_GB10_PATCH_DIR --no-overwrite-dir -xf -"
fi
validate_compose

echo "Starting DSpark worker on ${WORKER_HOST}..."
remote_compose "NODE_RANK=1 HEADLESS=1 HF_CACHE='$WORKER_HF_CACHE' VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml up -d"

echo "Starting DSpark head..."
compose_base 0 "" up -d

# VL TP=2 sidecar is launched AFTER the main API is healthy (see wait loop):
# DeepSeek and VL must not GPU-profile concurrently. VL uses a separate
# NCCL master port (VL_SIDECAR_MASTER_PORT, default 25100).
SIDECAR_COMPOSE_FILE="${SIDECAR_COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.vl-sidecar.yml}"

if [ "${DSPARK_SKIP_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip DSV4 v0.27 perf hotfixes (DSPARK_SKIP_HOTFIX=1)."
fi
if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip Issue #22 hotfix (DSPARK_SKIP_ISSUE22_HOTFIX=1)."
fi
if [ "${DSPARK_SKIP_SPIN_WAIT_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip GB10 shm spin-wait hotfix (DSPARK_SKIP_SPIN_WAIT_HOTFIX=1)."
fi
echo "Issue #22 / v0.27 .sh hotfixes run in the compose entrypoint before vllm (no mid-boot stop)."

echo "Waiting for DSpark vLLM API..."
print_initial_startup_logs
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "${AUTH_HEADER_ARGS[@]}" "$API_URL" >/dev/null 2>&1; then
    echo "DeepSeek V4 Flash DSpark is running: $API_URL"
    compose_base 0 "" ps
    remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml ps"
    # VL sidecar TP=2 (Qwen3-VL): worker-first, then head API rank. 0731 stays
    # text-only; agents use ds4f-vision MCP. Same compose project as DeepSeek
    # so stop tears it down. Separate NCCL master port from DeepSeek.
    if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ] && [ -f "$SIDECAR_COMPOSE_FILE" ]; then
      VL_MASTER_PORT="${VL_SIDECAR_MASTER_PORT:-25100}"
      echo "Starting VL sidecar TP=${VL_SIDECAR_TP_SIZE:-2} (${VL_SIDECAR_MODEL:-cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit}, port ${VL_SIDECAR_PORT:-8889}, master-port ${VL_MASTER_PORT})..."
      echo "  VL worker first on ${WORKER_HOST}..."
      remote_compose "MASTER_ADDR='$MASTER_ADDR' NODE_RANK=1 HEADLESS=1 HF_CACHE='$WORKER_HF_CACHE' docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.vl-sidecar.yml up -d"
      echo "  VL head (API rank)..."
      env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
        NODE_RANK=0 \
        docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$SIDECAR_COMPOSE_FILE" up -d
      SIDECAR_MODELS_URL="http://127.0.0.1:${VL_SIDECAR_PORT:-8889}/v1/models"
      SIDECAR_READY=0
      for _sidecar_i in $(seq 1 "${VL_SIDECAR_WAIT_ATTEMPTS:-90}"); do
        if curl -fsS --max-time 5 "$SIDECAR_MODELS_URL" 2>/dev/null | grep -q "qwen3-vl"; then
          SIDECAR_READY=1
          break
        fi
        sleep "${VL_SIDECAR_WAIT_SECONDS:-2}"
      done
      if [ "$SIDECAR_READY" = "1" ]; then
        echo "VL sidecar is ready: $SIDECAR_MODELS_URL"
        # Only register MCP when vision mode is on (this block) and install is
        # not explicitly disabled. INSTALL_VISION_MCP defaults to follow the flag.
        if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ] && [ "${INSTALL_VISION_MCP:-1}" = "1" ]; then
          echo "Registering ds4f-vision MCP into detected harnesses (pi/omp/hermes/opencode/goose/grok/openclaw/zcode/prime/factory/commandcode)..."
          if ! "$SCRIPT_DIR/scripts/install-ds4f-vision-mcp.sh"; then
            echo "WARN: vision MCP harness install failed (non-fatal)." >&2
          fi
        elif [ "${INSTALL_VISION_MCP:-1}" = "0" ]; then
          echo "Skipping vision MCP install (INSTALL_VISION_MCP=0)."
        fi
      else
        echo "WARN: VL sidecar not ready at $SIDECAR_MODELS_URL — skipping vision MCP install." >&2
        echo "  Recent VL head logs:" >&2
        COMPOSE_DISABLE_ENV_FILE=1 docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$SIDECAR_COMPOSE_FILE" logs --tail=80 >&2 || true
        echo "  Recent VL worker logs:" >&2
        remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.vl-sidecar.yml logs --tail=80" >&2 || true
      fi
    fi
    echo "Running minimal OpenAI-compatible thinking-budget chat request..."
    curl -fsS --max-time 60 "${AUTH_HEADER_ARGS[@]}" "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":32,"temperature":0.6,"top_p":0.95,"thinking_token_budget":1,"chat_template_kwargs":{"thinking":true,"reasoning_effort":"low"}}' >/dev/null
    echo "Minimal thinking-budget chat request succeeded."
    exit 0
  fi
  wait_with_startup_logs
done

echo "Timed out waiting for DSpark API. Recent head logs:" >&2
compose_base 0 "" logs --tail=120 vllm-dspark >&2 || true
echo "Recent worker logs:" >&2
remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark" >&2 || true
exit 1
