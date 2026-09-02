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

Two-node TP=2 is the default. Three nodes: ./start-tp3.sh (see docs/TP3.md).
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

# Source one private, normalized snapshot and reuse it for every Compose/worker
# consumer. The operator's file remains byte-identical.
_dspark_env_clean=
_cleanup_dspark_env() {
  [ -z "$_dspark_env_clean" ] || rm -f -- "$_dspark_env_clean"
}
trap _cleanup_dspark_env EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
_dspark_env_clean="$(mktemp)"
chmod 600 "$_dspark_env_clean"
# DSPARK_API_KEYS ambient guard (begin)
_dspark_ambient_has=0
_dspark_ambient_keys=""
if [ -n "${DSPARK_API_KEYS+x}" ]; then
  _dspark_ambient_has=1
  _dspark_ambient_keys="$DSPARK_API_KEYS"
fi
unset DSPARK_API_KEYS
sed $'1s/^\xEF\xBB\xBF//; s/\r$//' "$ENV_FILE" > "$_dspark_env_clean"
set -a
# shellcheck disable=SC1090
source "$_dspark_env_clean"
set +a
if [ "$_dspark_ambient_has" = "1" ] && [ "$_dspark_ambient_keys" != "${DSPARK_API_KEYS:-}" ]; then
  echo "error: DSPARK_API_KEYS is set in the environment but does not match .env.dspark; set it only in .env.dspark" >&2
  exit 2
fi
# DSPARK_API_KEYS ambient guard (end)
COMPOSE_ENV_FILE="$_dspark_env_clean"

# GPU util comes from GPU_MEMORY_UTILIZATION_TEXT (default 0.835).
# Explicit GPU_MEMORY_UTILIZATION in the env file is overridden by this so
# the profile stays in one place.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_TEXT:-0.835}"
export GPU_MEMORY_UTILIZATION

# TP=3 concurrent slots: CLI (./start-tp3.sh --max-num-seqs) beats
# TP3_MAX_NUM_SEQS in .env.dspark; both leave the 2-node MAX_NUM_SEQS alone.
if [ "${DSPARK_TP3:-0}" = "1" ]; then
  if [ -n "${_START_TP3_MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS="$_START_TP3_MAX_NUM_SEQS"
  elif [ -n "${TP3_MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS="$TP3_MAX_NUM_SEQS"
  fi
  export MAX_NUM_SEQS
fi

# Checkpoint flag: official Vision-Exp vs Keys abliterated weights.
#   ABLITERATED=0 → DSPARK_MODEL_OFFICIAL
#   ABLITERATED=1 → DSPARK_MODEL_ABLITERATED
DSPARK_MODEL_OFFICIAL="${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
DSPARK_MODEL_ABLITERATED="${DSPARK_MODEL_ABLITERATED:-drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit}"
DEFAULT_OFFICIAL_REVISION="86f746b36186f0e567729a5c06a8c918caba82a9"
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

# Vision-Exp: Anemll SpeculativeConfig requires
# num_speculative_tokens % num_nextn_predict_layers == 0 when k > n_predict.
# Checkpoint n_predict=3 and dspark_block_size=5 → k in {6,9,…}. Official and
# Keys ablit share that layout (0731 was n_predict=1, which is why k=5 booted).
_mtp_k="${MTP_NUM_TOKENS:-6}"
case "$_mtp_k" in
  ''|*[!0-9]*)
    echo "error: MTP_NUM_TOKENS must be a non-negative integer (got ${_mtp_k})" >&2
    exit 2
    ;;
esac
if [ "$_mtp_k" -lt 5 ] || [ $((_mtp_k % 3)) -ne 0 ]; then
  echo "error: Vision-Exp requires MTP_NUM_TOKENS >= 5 and divisible by 3 (num_nextn_predict_layers=3); got ${_mtp_k}" >&2
  exit 2
fi
unset _mtp_k

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

source "$SCRIPT_DIR/dspark-numeric-knobs.sh"
dspark_validate_numeric_knobs "$_dspark_env_clean" || exit $?
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
# DSPARK_API_KEYS auth (begin)
AUTH_HEADER_ARGS=()
case "${DSPARK_API_KEYS:-}" in
  *[$'\r\n\v\f']*)
    echo "error: DSPARK_API_KEYS must be a single-line space-separated list" >&2
    exit 2
    ;;
  *\\*)
    echo "error: DSPARK_API_KEYS must not contain backslashes" >&2
    exit 2
    ;;
esac
_dspark_keys_set=0
case "${DSPARK_API_KEYS:-}" in
  *[!$' \t']*) _dspark_keys_set=1 ;;
esac
if [ -n "${VLLM_API_KEY:-}" ] && [ "$_dspark_keys_set" = "1" ]; then
  # The server entrypoint refuses this combination too (exit 2); fail the same
  # way here so a probe never guesses which variable the server honoured.
  echo "error: VLLM_API_KEY and DSPARK_API_KEYS are both set; set exactly one of them" >&2
  exit 2
fi
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer $VLLM_API_KEY")
elif [ "$_dspark_keys_set" = "1" ]; then
  _dspark_keys=()
  read -r -a _dspark_keys <<< "${DSPARK_API_KEYS}"
  for _dspark_key in "${_dspark_keys[@]}"; do
    case "$_dspark_key" in
      -*) echo "error: DSPARK_API_KEYS contains a token beginning with '-'" >&2; exit 2 ;;
    esac
  done
  # Multi-key auth via --api-key: probe with the first parsed key. Without this
  # the health poll never sees a 200 against a keyed server and waits out its
  # full timeout on a cluster that is actually serving.
  AUTH_HEADER_ARGS=(-H "Authorization: Bearer ${_dspark_keys[0]}")
fi
# DSPARK_API_KEYS auth (end)

# DSPARK redaction pre-flight (begin)
if { [ "$_dspark_keys_set" = "1" ] || [ -n "${VLLM_API_KEY:-}" ]; } && [ ! -f "$SCRIPT_DIR/patches/hotfix-vllm-redact-api-key-log.sh" ]; then
  echo "error: API keys are configured but patches/hotfix-vllm-redact-api-key-log.sh is missing; keyed starts require the startup-log redaction hotfix" >&2
  exit 1
fi
# DSPARK redaction pre-flight (end)

# Issue #138 Responses history compatibility pre-flight (begin).
# Only the literal 1 enables it; normalize every other spelling to 0 before
# either Compose rank sees the flag. Relative path overrides are rooted at
# this checkout.
case "${DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT:-0}" in
  1) DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT=1 ;;
  *) DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT=0 ;;
esac
DSPARK_ISSUE138_HOTFIX="${DSPARK_ISSUE138_HOTFIX:-patches/hotfix-vllm-issue138-responses-history.py}"
case "$DSPARK_ISSUE138_HOTFIX" in
  /*) ;;
  *) DSPARK_ISSUE138_HOTFIX="$SCRIPT_DIR/$DSPARK_ISSUE138_HOTFIX" ;;
esac
export DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT DSPARK_ISSUE138_HOTFIX
if [ "$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT" = "1" ] && [ ! -f "$DSPARK_ISSUE138_HOTFIX" ]; then
  echo "error: issue #138 Responses history compatibility is enabled but patcher is missing: $DSPARK_ISSUE138_HOTFIX" >&2
  exit 1
fi
# Issue #138 Responses history compatibility pre-flight (end).
# Issue #141 is exact-1 and default-off. Normalize the effective switch once so
# head and worker receive the same 0/1 value, and fail before remote side effects
# when an enabled start cannot mount the selected local patch source.
DSPARK_ISSUE141_HOTFIX="${DSPARK_ISSUE141_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py}"
case "$DSPARK_ISSUE141_HOTFIX" in
  /*) ;;
  *) DSPARK_ISSUE141_HOTFIX="$SCRIPT_DIR/${DSPARK_ISSUE141_HOTFIX#./}" ;;
esac
DSPARK_ISSUE141_EFFECTIVE=0
if [ "${DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK:-0}" = "1" ]; then
  DSPARK_ISSUE141_EFFECTIVE=1
  if [ ! -f "$DSPARK_ISSUE141_HOTFIX" ]; then
    echo "error: DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK=1 but patch source is missing: $DSPARK_ISSUE141_HOTFIX" >&2
    exit 1
  fi
fi
DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK="$DSPARK_ISSUE141_EFFECTIVE"
export DSPARK_ISSUE141_HOTFIX DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK
DSPARK_ISSUE117_HOTFIX="$SCRIPT_DIR/patches/hotfix-vllm-issue117-shm-ring-buffer.py"
if [ "${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}" != "1" ] && { [ ! -f "$DSPARK_ISSUE117_HOTFIX" ] || [ -L "$DSPARK_ISSUE117_HOTFIX" ]; }; then
  echo "Issue #117 SHM ring hotfix is enabled but its local patcher is missing or not a regular file: $DSPARK_ISSUE117_HOTFIX" >&2
  exit 1
fi
# Report item 6: sequence-parallel Lightning indexer for long prefills.
# Exact 1 applies the patcher at boot on both ranks (fail-closed); anything
# else is normalized to 0 (stock bytes).
DSPARK_SP_INDEXER_HOTFIX="${DSPARK_SP_INDEXER_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-sp-indexer-prefill.py}"
case "$DSPARK_SP_INDEXER_HOTFIX" in
  /*) ;;
  *) DSPARK_SP_INDEXER_HOTFIX="$SCRIPT_DIR/${DSPARK_SP_INDEXER_HOTFIX#./}" ;;
esac
DSPARK_SP_INDEXER_EFFECTIVE=0
if [ "${DSPARK_ENABLE_SP_INDEXER:-0}" = "1" ]; then
  DSPARK_SP_INDEXER_EFFECTIVE=1
  if [ ! -f "$DSPARK_SP_INDEXER_HOTFIX" ]; then
    echo "error: DSPARK_ENABLE_SP_INDEXER=1 but patch source is missing: $DSPARK_SP_INDEXER_HOTFIX" >&2
    exit 1
  fi
fi
DSPARK_ENABLE_SP_INDEXER="$DSPARK_SP_INDEXER_EFFECTIVE"
export DSPARK_SP_INDEXER_HOTFIX DSPARK_ENABLE_SP_INDEXER
# DeepGEMM SM121 indexer-logits header alias (opt-in, see item8 design §5).
DSPARK_DEEPGEMM_ALIAS_HOTFIX="${DSPARK_DEEPGEMM_ALIAS_HOTFIX:-$SCRIPT_DIR/patches/hotfix-deepgemm-sm121-mqa-header-alias.sh}"
case "$DSPARK_DEEPGEMM_ALIAS_HOTFIX" in
  /*) ;;
  *) DSPARK_DEEPGEMM_ALIAS_HOTFIX="$SCRIPT_DIR/${DSPARK_DEEPGEMM_ALIAS_HOTFIX#./}" ;;
esac
DSPARK_DEEPGEMM_ALIAS_EFFECTIVE=0
if [ "${DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS:-0}" = "1" ]; then
  DSPARK_DEEPGEMM_ALIAS_EFFECTIVE=1
  if [ ! -f "$DSPARK_DEEPGEMM_ALIAS_HOTFIX" ]; then
    echo "error: DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1 but patch source is missing: $DSPARK_DEEPGEMM_ALIAS_HOTFIX" >&2
    exit 1
  fi
fi
DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS="$DSPARK_DEEPGEMM_ALIAS_EFFECTIVE"
export DSPARK_DEEPGEMM_ALIAS_HOTFIX DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS
DSPARK_ISSUE136_XGRAMMAR_HOTFIX="${DSPARK_ISSUE136_XGRAMMAR_HOTFIX:-$SCRIPT_DIR/patches/hotfix-vllm-issue136-xgrammar-termination.py}"
if [ "${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}" = "1" ] && { [ ! -f "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ] || [ -L "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ]; }; then
  echo "Issue #136 XGrammar hotfix is enabled but its local patcher is missing or not a regular file: $DSPARK_ISSUE136_XGRAMMAR_HOTFIX" >&2
  exit 1
fi
export DSPARK_ISSUE136_XGRAMMAR_HOTFIX DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${MASTER_PORT:?MASTER_PORT must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"
: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"

# ./start-tp3.sh sets DSPARK_TP3=1. Ignore TP_SIZE/NNODES from .env on the
# two-node path so a copied 3-node snippet cannot boot nnodes=3 with one worker.
DSPARK_TP3="${DSPARK_TP3:-0}"
if [ "$DSPARK_TP3" = "1" ]; then
  TP_SIZE=3
  NNODES=3
else
  TP_SIZE=2
  NNODES=2
fi
export TP_SIZE NNODES DSPARK_TP3

VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-$WORKER_HOST}"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$SCRIPT_DIR}}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-${HF_CACHE:-}}"
# Worker hub weights: default is a local copy (prepare downloads on both nodes).
# Set DSPARK_WORKER_HF_NFS=1 to mount the head HF cache over NFS (ConnectX);
# JIT caches stay on the worker host path.
DSPARK_WORKER_HF_NFS="${DSPARK_WORKER_HF_NFS:-0}"
NFS_VOLUME="${NFS_VOLUME:-dspark-hf}"
NFS_CONTAINER="${NFS_CONTAINER:-dspark-nfs}"
NFS_OVERRIDE_FILE="${NFS_OVERRIDE_FILE:-$SCRIPT_DIR/docker-compose.dspark-nfs.override.yml}"
# Per-node CX7/RoCE pins (3-node ring: facing ports often differ by hostname).
# Set WORKER_NCCL_* in the head .env; start script injects them on remote compose.
# Do not put WORKER_* first in docker-compose substitution — that is not rank-aware.
WORKER_NCCL_IB_HCA="${WORKER_NCCL_IB_HCA:-$NCCL_IB_HCA}"
WORKER_NCCL_SOCKET_IFNAME="${WORKER_NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"
WORKER_TP_SOCKET_IFNAME="${WORKER_TP_SOCKET_IFNAME:-${TP_SOCKET_IFNAME:-$WORKER_NCCL_SOCKET_IFNAME}}"
WORKER_GLOO_SOCKET_IFNAME="${WORKER_GLOO_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME:-$WORKER_NCCL_SOCKET_IFNAME}}"
WORKER2_HOST="${WORKER2_HOST:-}"
WORKER2_VLLM_HOST_IP="${WORKER2_VLLM_HOST_IP:-$WORKER2_HOST}"
WORKER2_DIR="${WORKER2_SCRIPT_DIR:-${WORKER2_DIR:-$WORKER_DIR}}"
WORKER2_HF_CACHE="${WORKER2_HF_CACHE:-${WORKER_HF_CACHE:-}}"
WORKER2_NCCL_IB_HCA="${WORKER2_NCCL_IB_HCA:-$WORKER_NCCL_IB_HCA}"
WORKER2_NCCL_SOCKET_IFNAME="${WORKER2_NCCL_SOCKET_IFNAME:-$WORKER_NCCL_SOCKET_IFNAME}"
WORKER2_TP_SOCKET_IFNAME="${WORKER2_TP_SOCKET_IFNAME:-${WORKER_TP_SOCKET_IFNAME}}"
WORKER2_GLOO_SOCKET_IFNAME="${WORKER2_GLOO_SOCKET_IFNAME:-${WORKER_GLOO_SOCKET_IFNAME}}"
WORKER2_NCCL_IB_GID_MATCH_IP="${WORKER2_NCCL_IB_GID_MATCH_IP:-}"
ENV_WORKER2_NCCL_IB_GID_INDEX="${WORKER2_NCCL_IB_GID_INDEX:-}"
WORKER2_NCCL_IB_GID_INDEX="${ENV_WORKER2_NCCL_IB_GID_INDEX}"
if [ "$DSPARK_TP3" = "1" ]; then
  : "${WORKER2_HOST:?WORKER2_HOST must be set in $ENV_FILE for ./start-tp3.sh}"
  if [ ! -f "$SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" ]; then
    echo "Missing TP=3 patcher: $SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" >&2
    exit 1
  fi
  REMOTE_WORKER2_DIR="$(printf '%q' "$WORKER2_DIR")"
  REMOTE_COMPOSE2="cd $REMOTE_WORKER2_DIR && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"
  REMOTE_COMPOSE_FILE2="$REMOTE_WORKER2_DIR/docker-compose.dspark.yml"
  REMOTE_ENV_FILE2="$REMOTE_WORKER2_DIR/.env.dspark"
  REMOTE_VLLM_GB10_PATCH_DIR2="$REMOTE_WORKER2_DIR/vllm_patch_gb10"
fi
# RoCEv2 GID index differs per node/HCA and drifts after reboot/link events.
# Default (NCCL_IB_GID_AUTO=1): validate every selected HCA/port from sysfs,
# then leave NCCL_IB_GID_INDEX unset on both ranks — a pin is one global value
# per rank, and NCCL selects the RoCEv2/IPv4 GID per HCA when it is absent.
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
REMOTE_ISSUE136_ENABLE="$(printf '%q' "${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}")"
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

WORKER_COMPOSE_FILES="-f docker-compose.dspark.yml"
WORKER_HF_COMPOSE_ENV="HF_CACHE='$WORKER_HF_CACHE'"
WORKER2_COMPOSE_FILES="-f docker-compose.dspark.yml"
WORKER2_HF_COMPOSE_ENV="HF_CACHE='$WORKER2_HF_CACHE'"
if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
  IFACE="${NFS_IFACE:-$NCCL_SOCKET_IFNAME}"
  HF_CACHE_DIR="${HF_CACHE:-$HOME/.cache/huggingface}"
  WORKER_IP="$(host_without_user "$WORKER_HOST")"
  # shellcheck source=files/nfs-share.sh
  source "$SCRIPT_DIR/files/nfs-share.sh"
  nfs_detect_server_ip
  WORKER_COMPOSE_FILES="-f docker-compose.dspark.yml -f docker-compose.dspark-nfs.override.yml"
  WORKER_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER_HF_CACHE'"
  REMOTE_NFS_OVERRIDE_FILE="$REMOTE_WORKER_DIR/docker-compose.dspark-nfs.override.yml"
  if [ "$DSPARK_TP3" = "1" ]; then
    WORKER2_COMPOSE_FILES="-f docker-compose.dspark.yml -f docker-compose.dspark-nfs.override.yml"
    WORKER2_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER2_HF_CACHE'"
    REMOTE_NFS_OVERRIDE_FILE2="$REMOTE_WORKER2_DIR/docker-compose.dspark-nfs.override.yml"
  fi
fi

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

# NCCL_IB_HCA is not a bare sysfs device name. NCCL (parseStringList in
# src/misc/utils.cc) accepts an optional leading "^" (exclude), then an optional
# "=" (exact name match instead of prefix match), then a comma-separated list of
# name[:port[:rail[:plane]]] tokens. Empty names are dropped; only the first
# MAX_IB_DEVS=32 non-empty entries are stored, and each stored name is truncated
# to netIf::prefix's 63-byte payload. An empty token list matches every
# device/port. A port field that is absent *or empty* means -1, i.e. any port -
# "devA" and "devA:" select the same thing. A non-empty port field is atoi():
# optional whitespace and sign, then leading decimal digits, stopping at the
# first non-digit. So ":08" is port 8 (atoi is base 10, never
# octal) and ":abc" is 0, which matches no real port. A port outside the resolver's conservative
# nine-digit arithmetic bound is clamped instead of evaluated, because $(( )) wraps modulo 2^64
# and one such value (18446744073709551615) wraps to -1, the "any port"
# wildcard. Only the port field takes part in matching here; rail/plane are
# parsed off and ignored.
#
# The selector is applied to the same candidate universe ncclIbInit builds:
# only ACTIVE ports whose link layer is Ethernet or InfiniBand, capped at
# MAX_IB_DEVS=32 entries. Both filters run before NCCL_IB_HCA, so a DOWN
# sibling port (common on these dual-port cards) neither fails the resolve nor
# constrains the index - NCCL never opens it either.
#
# The resolver below mirrors those semantics on the node that owns the sysfs
# tree, validates every selected member against its own local address (one
# shared match IP must not silently drop a member that uses another link
# address), and fails closed - exit 1 when a selected member has no usable
# RoCEv2 GID. It stops there: usable index sets are reported per member and
# never reconciled, because auto mode pins nothing. NCCL selects the
# RoCEv2/IPv4 GID per HCA when NCCL_IB_GID_INDEX is absent, so members whose
# usable sets are disjoint are still fine; only a member with no usable GID
# at all is fatal.
#
# Body is a quoted heredoc (nothing expands here); resolve_rocev2_gid_index
# prepends the inputs as printf %q assignments, so selector tokens are
# transported literally and never glob-expanded (set -f).
NCCL_HCA_RESOLVER_BODY="$(cat <<'RESOLVER'
set -f
sysroot="${NCCL_GID_RESOLVE_SYSROOT:-/sys/class/infiniband}"
orig_spec=$spec

search_not=0
search_exact=0
case "$spec" in "^"*) search_not=1; spec="${spec#^}" ;; esac
case "$spec" in "="*) search_exact=1; spec="${spec#=}" ;; esac

max_ib_devs=32
ntok=0
selector_truncated=0
OLDIFS=$IFS
IFS=,
set -- $spec
IFS=$OLDIFS
for tok in "$@"; do
  name=${tok%%:*}
  [ -n "$name" ] || continue
  if [ "$ntok" -ge "$max_ib_devs" ]; then
    selector_truncated=1
    continue
  fi
  # NCCL stores the name in netIf::prefix[64]. C locale makes printf's string
  # precision byte-oriented, matching snprintf's 63-byte payload limit.
  LC_ALL=C printf -v name '%.63s' "$name"
  port=-1
  case "$tok" in *:*)
    p=${tok#*:}
    p=${p%%:*}
    # Absent or empty port field means "any port"; only a non-empty field is
    # atoi()'d. Match once against the whole field so conversion cannot restart
    # after an embedded newline. Force base 10 so "08"/"010" parse the way
    # atoi() reads them instead of becoming a bad (or wrong) octal literal.
    if [ -n "$p" ]; then
      if [[ $p =~ ^[[:space:]]*([+-]?[0-9]+) ]]; then
        digits=${BASH_REMATCH[1]}
      else
        digits=
      fi
      sign=
      mag=$digits
      case "$mag" in -*) sign=-; mag=${mag#-} ;; +*) mag=${mag#+} ;; esac
      # Strip leading zeros so the width test below measures the magnitude and
      # not the padding ("0000008" is one digit wide).
      while :; do
        case "$mag" in 0?*) mag=${mag#0} ;; *) break ;; esac
      done
      if [ -z "$mag" ]; then
        port=0
      elif [ ${#mag} -gt 9 ]; then
        # Outside the conservative nine-digit bound. Evaluating arbitrary-width
        # text with $(( )) can wrap the value
        # modulo 2^64 - and 18446744073709551615 wraps to exactly -1, which is
        # the "any port" wildcard - so an unrepresentable port would silently
        # *widen* the selection. Clamp to a value no sysfs port can have: the
        # token then matches nothing and the resolve fails closed.
        port=${sign}999999999
      else
        port=$(( 10#$mag ))
        [ -z "$sign" ] || port=$(( 0 - port ))
      fi
    fi
  ;; esac
  ntok=$((ntok + 1))
  eval "tok_name_$ntok=\$name"
  eval "tok_port_$ntok=\$port"
done
[ "$selector_truncated" = 0 ] || echo "  note: selector list truncated to first $max_ib_devs non-empty entries; NCCL ignores later entries" >&2

pair_matches() { # $1=dev $2=port -> 0 when the token list matches
  [ "$ntok" -gt 0 ] || return 0
  i=1
  while [ "$i" -le "$ntok" ]; do
    eval "n=\$tok_name_$i"
    eval "p=\$tok_port_$i"
    match=0
    if [ "$search_exact" = "1" ]; then
      [ "$1" = "$n" ] && match=1
    else
      case "$1" in "$n"*) match=1 ;; esac
    fi
    if [ "$match" = "1" ]; then
      if [ "$p" -eq -1 ] || [ "$p" -eq "$2" ]; then return 0; fi
    fi
    i=$((i + 1))
  done
  return 1
}

ipv4_hex() { # a.b.c.d -> aabb:ccdd
  _oldifs=$IFS
  IFS=.
  set -- $1
  IFS=$_oldifs
  [ $# -eq 4 ] || return 1
  case "$1$2$3$4" in *[!0-9]*) return 1 ;; esac
  printf '%02x%02x:%02x%02x' "$1" "$2" "$3" "$4"
}

# Candidate universe, mirroring ncclIbInit: a port is a candidate only when it
# is ACTIVE and its link layer is Ethernet or InfiniBand, and both tests happen
# *before* NCCL_IB_HCA is applied. A DOWN sibling port therefore cannot be
# selected into a fail-closed error or drag the index intersection, exactly as
# NCCL never opens it. An attribute that cannot be read is not evidence of
# inactivity, so the port stays a candidate. NCCL then keeps at most
# MAX_IB_DEVS entries and ignores the rest; the cap is mirrored here so the
# resolved index describes the devices NCCL will actually use. The same
# MAX_IB_DEVS value separately caps the selector entries stored above.
selected=""
nsel=0
skipped_state=""
skipped_link=""
capped=""
for dev in $(ls "$sysroot" 2>/dev/null); do
  [ -d "$sysroot/$dev/ports" ] || continue
  for port in $(ls "$sysroot/$dev/ports" 2>/dev/null); do
    st=$(cat "$sysroot/$dev/ports/$port/state" 2>/dev/null || true)
    st=${st#*: }
    case "$st" in
      ''|ACTIVE) : ;;
      *) skipped_state="$skipped_state $dev:$port($st)"; continue ;;
    esac
    ll=$(cat "$sysroot/$dev/ports/$port/link_layer" 2>/dev/null || true)
    case "$ll" in
      ''|Ethernet|InfiniBand) : ;;
      *) skipped_link="$skipped_link $dev:$port($ll)"; continue ;;
    esac
    if pair_matches "$dev" "$port"; then m=1; else m=0; fi
    [ "$m" -ne "$search_not" ] || continue
    if [ "$nsel" -ge "$max_ib_devs" ]; then capped="$capped $dev:$port"; continue; fi
    selected="$selected $dev:$port"
    nsel=$((nsel + 1))
  done
done
[ -z "$capped" ] || echo "  note: selection truncated at MAX_IB_DEVS=$max_ib_devs; NCCL ignores:$capped" >&2
if [ -z "$selected" ]; then
  why=""
  [ -z "$skipped_state" ] || why="$why; not ACTIVE:$skipped_state"
  [ -z "$skipped_link" ] || why="$why; unsupported link layer:$skipped_link"
  echo "FATAL: NCCL_IB_HCA selector matched no candidate HCA/port under $sysroot (selector: $orig_spec)$why" >&2
  exit 1
fi

fail_members=""
mem_n=0
for pair in $selected; do
  dev=${pair%%:*}
  port=${pair##*:}
  pdir="$sysroot/$dev/ports/$port"
  mem_n=$((mem_n + 1))
  eval "mem_pair_$mem_n=\$pair"
  # Collect every usable index for this member, not just the first one.
  usable=""
  for g in $(ls "$pdir/gids" 2>/dev/null); do
    t=$(cat "$pdir/gid_attrs/types/$g" 2>/dev/null || true)
    [ "$t" = "RoCE v2" ] || continue
    gid=$(cat "$pdir/gids/$g" 2>/dev/null || true)
    src=""
    case "$gid" in *ffff:"$hex") src="match-ip $match_ip" ;; esac
    if [ -z "$src" ]; then
      nd=$(cat "$pdir/gid_attrs/ndevs/$g" 2>/dev/null || true)
      if [ -n "$nd" ]; then
        for oip in $(ip -4 -o addr show dev "$nd" 2>/dev/null | awk '{print $4}' | cut -d/ -f1); do
          oh=$(ipv4_hex "$oip") || continue
          case "$gid" in *ffff:"$oh") src="own-addr $oip on $nd"; break ;; esac
        done
      fi
    fi
    [ -n "$src" ] || continue
    usable="$usable $g"
    eval "src_${mem_n}_$g=\$src"
  done
  eval "mem_usable_$mem_n=\$usable"
  if [ -z "$usable" ]; then
    fail_members="$fail_members $dev:$port"
    continue
  fi
done
if [ -n "$fail_members" ]; then
  echo "FATAL: no usable RoCEv2 GID on selected member(s):$fail_members (no GID matches $match_ip or an IPv4 on the member's own netdev)" >&2
  exit 1
fi
# Validation-only outcome: audit each member's whole usable set. No index is
# chosen and nothing is written to stdout - the caller leaves
# NCCL_IB_GID_INDEX unset so NCCL selects the RoCEv2/IPv4 GID per HCA.
i=1
while [ "$i" -le "$mem_n" ]; do
  eval "pair=\$mem_pair_$i"
  eval "u=\$mem_usable_$i"
  for g in $u; do
    eval "s=\${src_${i}_$g:-}"
    echo "  member $pair -> RoCEv2 gid index $g (via $s)" >&2
  done
  i=$((i + 1))
done
exit 0
RESOLVER
)"

# Validate that every member an NCCL_IB_HCA selector picks on the target node
# exposes a usable RoCEv2 GID (RoCE v2 type whose address matches the preferred
# IPv4 or an IPv4 on the member's own netdev). Per-member usable indexes are
# audited on stderr; nothing is written to stdout. Exit 1 = a selected member
# is missing/unresolvable (fail closed).
# $1=ssh target (empty=local)  $2=NCCL_IB_HCA selector  $3=preferred IPv4
resolve_rocev2_gid_index() {
  local ssh_target="$1" hca_spec="$2" match_ip="$3"
  local hex remote
  hex="$(ipv4_to_gid_suffix "$match_ip")" || return 1
  remote="spec=$(printf '%q' "$hca_spec")
hex=$(printf '%q' "$hex")
match_ip=$(printf '%q' "$match_ip")
$NCCL_HCA_RESOLVER_BODY"
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
  local head_match worker_match

  if [ "$NCCL_IB_GID_AUTO" = "0" ]; then
    NCCL_IB_GID_INDEX="${ENV_NCCL_IB_GID_INDEX:-}"
    WORKER_NCCL_IB_GID_INDEX="${ENV_WORKER_NCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}"
    if [ -z "$NCCL_IB_GID_INDEX" ] || [ -z "$WORKER_NCCL_IB_GID_INDEX" ]; then
      echo "NCCL_IB_GID_AUTO=0 requires NCCL_IB_GID_INDEX and preferably WORKER_NCCL_IB_GID_INDEX in $ENV_FILE." >&2
      exit 1
    fi
    if [ "${DSPARK_TP3:-0}" = "1" ]; then
      WORKER2_NCCL_IB_GID_INDEX="${ENV_WORKER2_NCCL_IB_GID_INDEX:-$WORKER_NCCL_IB_GID_INDEX}"
      if [ -z "$WORKER2_NCCL_IB_GID_INDEX" ]; then
        echo "NCCL_IB_GID_AUTO=0 requires WORKER2_NCCL_IB_GID_INDEX (or WORKER_NCCL_IB_GID_INDEX) in $ENV_FILE." >&2
        exit 1
      fi
    fi
    if [ "${DSPARK_TP3:-0}" = "1" ]; then
      echo "Using pinned NCCL GID indexes (auto off): head=$NCCL_IB_GID_INDEX worker=$WORKER_NCCL_IB_GID_INDEX worker2=$WORKER2_NCCL_IB_GID_INDEX"
    else
      echo "Using pinned NCCL GID indexes (auto off): head=$NCCL_IB_GID_INDEX worker=$WORKER_NCCL_IB_GID_INDEX"
    fi
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

  echo "Validating RoCEv2 GIDs from sysfs (head if=$NCCL_SOCKET_IFNAME ip=$head_match selector=$NCCL_IB_HCA; worker if=$WORKER_NCCL_SOCKET_IFNAME ip=$worker_match selector=$WORKER_NCCL_IB_HCA)..."
  resolve_rocev2_gid_index "" "$NCCL_IB_HCA" "$head_match" || {
    echo "FATAL: could not validate head RoCEv2 GIDs (NCCL_IB_HCA=$NCCL_IB_HCA, match $head_match)." >&2
    echo "Check: ibstat ; show_gids   # every selected member must exist under /sys/class/infiniband with a usable RoCE v2 GID" >&2
    exit 1
  }
  resolve_rocev2_gid_index "$WORKER_HOST" "$WORKER_NCCL_IB_HCA" "$worker_match" || {
    echo "FATAL: could not validate worker RoCEv2 GIDs (WORKER_NCCL_IB_HCA=$WORKER_NCCL_IB_HCA, match $worker_match)." >&2
    echo "Check on worker: ibstat ; show_gids" >&2
    exit 1
  }

  # AUTO=1 pins nothing: report and drop any stale pins from the env file.
  if [ -n "$ENV_NCCL_IB_GID_INDEX" ]; then
    echo "Note: ignoring NCCL_IB_GID_INDEX=$ENV_NCCL_IB_GID_INDEX from $ENV_FILE (NCCL_IB_GID_AUTO=1 leaves GID selection to NCCL per HCA)."
  fi
  if [ -n "$ENV_WORKER_NCCL_IB_GID_INDEX" ]; then
    echo "Note: ignoring WORKER_NCCL_IB_GID_INDEX=$ENV_WORKER_NCCL_IB_GID_INDEX from $ENV_FILE (NCCL_IB_GID_AUTO=1 leaves GID selection to NCCL per HCA)."
  fi
  NCCL_IB_GID_INDEX=""
  WORKER_NCCL_IB_GID_INDEX=""
  echo "RoCEv2 GIDs validated on both ranks; NCCL_IB_GID_INDEX left unset so NCCL selects the RoCEv2/IPv4 GID per HCA."

  if [ "${DSPARK_TP3:-0}" = "1" ]; then
    local worker2_match
    worker2_match="$(pick_gid_match_ip "$WORKER2_HOST" "$WORKER2_NCCL_SOCKET_IFNAME" "$WORKER2_NCCL_IB_GID_MATCH_IP" "$WORKER2_VLLM_HOST_IP" "$WORKER2_HOST")" || {
      echo "FATAL: could not determine worker2 RoCE IPv4 for GID match (if=$WORKER2_NCCL_SOCKET_IFNAME)." >&2
      exit 1
    }
    echo "Validating RoCEv2 GIDs for worker2 (if=$WORKER2_NCCL_SOCKET_IFNAME ip=$worker2_match selector=$WORKER2_NCCL_IB_HCA)..."
    resolve_rocev2_gid_index "$WORKER2_HOST" "$WORKER2_NCCL_IB_HCA" "$worker2_match" || {
      echo "FATAL: could not validate worker2 RoCEv2 GIDs (WORKER2_NCCL_IB_HCA=$WORKER2_NCCL_IB_HCA, match $worker2_match)." >&2
      echo "Check on worker2: ibstat ; show_gids" >&2
      exit 1
    }
    if [ -n "${ENV_WORKER2_NCCL_IB_GID_INDEX:-}" ]; then
      echo "Note: ignoring WORKER2_NCCL_IB_GID_INDEX=$ENV_WORKER2_NCCL_IB_GID_INDEX from $ENV_FILE (NCCL_IB_GID_AUTO=1 leaves GID selection to NCCL per HCA)."
    fi
    WORKER2_NCCL_IB_GID_INDEX=""
    echo "RoCEv2 GIDs validated on worker2; NCCL_IB_GID_INDEX left unset."
  fi
}

remote_nccl_env() {
  # Rebuild each call so GID resolve after early init is visible on the worker.
  # NCCL_IB_GID_INDEX is always emitted, even empty under NCCL_IB_GID_AUTO=1:
  # the empty process-env value overrides a stale worker .env.dspark entry at
  # compose interpolation, and the shared entrypoint normalization makes the
  # defined-empty variable truly absent in the container (NCCL would parse a
  # defined-empty value as GID index 0).
  printf "NCCL_IB_HCA='%s' NCCL_SOCKET_IFNAME='%s' TP_SOCKET_IFNAME='%s' GLOO_SOCKET_IFNAME='%s' NCCL_IB_GID_INDEX='%s' NCCL_IB_MERGE_NICS='%s' NCCL_IB_SUBNET_AWARE_ROUTING='%s' NCCL_IB_SUBNET_PREFIX_LEN='%s' VLLM_HOST='%s' VLLM_PORT='%s'" \
    "$WORKER_NCCL_IB_HCA" \
    "$WORKER_NCCL_SOCKET_IFNAME" \
    "$WORKER_TP_SOCKET_IFNAME" \
    "$WORKER_GLOO_SOCKET_IFNAME" \
    "$WORKER_NCCL_IB_GID_INDEX" \
    "${NCCL_IB_MERGE_NICS:-}" \
    "${NCCL_IB_SUBNET_AWARE_ROUTING:-}" \
    "${NCCL_IB_SUBNET_PREFIX_LEN:-}" \
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
    TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}" \
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}" \
    NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}" \
    NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-}" \
    NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-}" \
    NCCL_IB_SUBNET_PREFIX_LEN="${NCCL_IB_SUBNET_PREFIX_LEN:-}" \
    VLLM_HOST="$VLLM_HOST" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_HOST_IP="$VLLM_HOST_IP" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    DSPARK_MODEL="$DSPARK_MODEL" \
    DSPARK_REVISION="${DSPARK_REVISION:-}" \
    ENABLE_VLLM_GB10_PATCH="$ENABLE_VLLM_GB10_PATCH" \
    VLLM_GB10_PATCH_DIR="$VLLM_GB10_PATCH_DIR" \
    GB10_HYBRID_NVFP4_M_THRESHOLD="${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}" \
    TP_SIZE="$TP_SIZE" \
    NNODES="$NNODES" \
    TP3_PATCH_DIR="${TP3_PATCH_DIR:-$SCRIPT_DIR/patches/tp3}" \
    NODE_RANK="$1" \
    HEADLESS="$2" \
    docker compose -p "$PROJECT_NAME" --env-file "$COMPOSE_ENV_FILE" -f "$COMPOSE_FILE" "${@:3}"
}

remote_nccl_env2() {
  printf "NCCL_IB_HCA='%s' NCCL_SOCKET_IFNAME='%s' TP_SOCKET_IFNAME='%s' GLOO_SOCKET_IFNAME='%s' NCCL_IB_GID_INDEX='%s' NCCL_IB_MERGE_NICS='%s' NCCL_IB_SUBNET_AWARE_ROUTING='%s' NCCL_IB_SUBNET_PREFIX_LEN='%s' VLLM_HOST='%s' VLLM_PORT='%s'" \
    "$WORKER2_NCCL_IB_HCA" \
    "$WORKER2_NCCL_SOCKET_IFNAME" \
    "$WORKER2_TP_SOCKET_IFNAME" \
    "$WORKER2_GLOO_SOCKET_IFNAME" \
    "$WORKER2_NCCL_IB_GID_INDEX" \
    "${NCCL_IB_MERGE_NICS:-}" \
    "${NCCL_IB_SUBNET_AWARE_ROUTING:-}" \
    "${NCCL_IB_SUBNET_PREFIX_LEN:-}" \
    "$VLLM_HOST" \
    "$VLLM_PORT"
}

remote_compose() {
  # The head may use an absolute local mount override; the worker always uses
  # the canonical synced relative path.
  ssh "$WORKER_HOST" "$REMOTE_COMPOSE DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX=$REMOTE_ISSUE136_ENABLE DSPARK_ISSUE136_XGRAMMAR_HOTFIX='./patches/hotfix-vllm-issue136-xgrammar-termination.py' TP_SIZE='$TP_SIZE' NNODES='$NNODES' TP3_PATCH_DIR='./patches/tp3' $(remote_nccl_env) $*"
}

remote_compose2() {
  ssh "$WORKER2_HOST" "$REMOTE_COMPOSE2 DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX=$REMOTE_ISSUE136_ENABLE DSPARK_ISSUE136_XGRAMMAR_HOTFIX='./patches/hotfix-vllm-issue136-xgrammar-termination.py' TP_SIZE='$TP_SIZE' NNODES='$NNODES' TP3_PATCH_DIR='./patches/tp3' $(remote_nccl_env2) $*"
}

log_since() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

print_startup_logs() {
  local since="$1"

  compose_base 0 "" logs --since "$since" vllm-dspark || true
  remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" || true
  if [ "$DSPARK_TP3" = "1" ]; then
    remote_compose2 "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" || true
  fi
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
  if [ "$DSPARK_TP3" = "1" ]; then
    remote_compose2 "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=100 vllm-dspark" || true
  fi
}

print_failure_logs() {
  local since="${STARTUP_LOG_SINCE:-$(log_since)}"

  echo "Startup failed. Recent head logs:" >&2
  compose_base 0 "" logs --since "$since" vllm-dspark >&2 || true
  echo "Recent worker logs:" >&2
  remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" >&2 || true
  if [ "$DSPARK_TP3" = "1" ]; then
    echo "Recent worker2 logs:" >&2
    remote_compose2 "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --since '$since' vllm-dspark" >&2 || true
  fi
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
  if [ "${DSPARK_TP3:-0}" = "1" ]; then
    echo "  max num seqs: ${MAX_NUM_SEQS:-6} (TP=3; CUDA-graph capture $(( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-6} + 1) )))"
  else
    echo "  max num seqs: ${MAX_NUM_SEQS:-6}"
  fi
  echo "  max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-8192}"
  echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION:-0.835} (from GPU_MEMORY_UTILIZATION_TEXT=${GPU_MEMORY_UTILIZATION_TEXT:-0.835})"
  echo "  mtp speculative tokens: ${MTP_NUM_TOKENS:-6} (Vision-Exp: >=5 and divisible by 3)"
  echo "  default thinking: $DEFAULT_THINKING (off/low/high/max)"
  echo "  issue31 GPU thinking_token_budget hotfix: ${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0} (0=stock V2 / 1=apply)"
  if [ "$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT" = "1" ]; then
    echo "  issue138 Responses history compatibility: 1 (apply)"
  else
    echo "  issue138 Responses history compatibility: 0 (stock)"
  fi
  echo "  issue136 XGrammar termination hotfix: ${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0} (0=stock / 1=preflight+apply)"
  if [ "${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}" = "1" ]; then
    echo "  issue117 SHM ring hotfix: skipped (issue79 remains independent)"
  else
    echo "  issue117 SHM ring hotfix: preflight+apply (5000ms reader recheck)"
  fi
  echo "  issue133 Triton specialization hotfix: will apply on start"
  echo "  issue141 sparse-MLA fixed-64 workaround: $DSPARK_ISSUE141_EFFECTIVE (0=stock / 1=apply)"
  echo "  SP indexer prefill (item 6): $DSPARK_SP_INDEXER_EFFECTIVE (0=stock / 1=apply; min keys ${DSPARK_SP_INDEXER_MIN_KEYS:-8192})"
  echo "  DeepGEMM sm121 header alias: $DSPARK_DEEPGEMM_ALIAS_EFFECTIVE (0=stock / 1=apply)"
  echo "  cudagraph capture size: $(( ( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-6} + 1) + 7 ) / 8 * 8 )) (rounded up to a multiple of 8 so spec-decode keeps the full-concurrency shape)"
  echo "  API bind: $VLLM_HOST:$VLLM_PORT"
  echo "  API probe: $API_URL"
  echo "  head fabric IP: $VLLM_HOST_IP"
  echo "  worker host/ip: $WORKER_HOST / $WORKER_VLLM_HOST_IP"
  if [ "$DSPARK_TP3" = "1" ]; then
    echo "  tensor parallel / nnodes: $TP_SIZE / $NNODES (start-tp3)"
    echo "  worker2 host/ip: $WORKER2_HOST / $WORKER2_VLLM_HOST_IP"
    echo "  worker2 NCCL HCA/if: $WORKER2_NCCL_IB_HCA / $WORKER2_NCCL_SOCKET_IFNAME"
    echo "  worker2 NCCL_IB_GID_INDEX: ${WORKER2_NCCL_IB_GID_INDEX:-}"
    echo "  worker2 dir: $WORKER2_DIR"
  else
    echo "  tensor parallel / nnodes: $TP_SIZE / $NNODES"
  fi
  echo "  head NCCL HCA/if: $NCCL_IB_HCA / $NCCL_SOCKET_IFNAME"
  echo "  worker NCCL HCA/if: $WORKER_NCCL_IB_HCA / $WORKER_NCCL_SOCKET_IFNAME"
  echo "  NCCL_IB_GID_AUTO: $NCCL_IB_GID_AUTO"
  echo "  head NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-<unset>}"
  echo "  worker NCCL_IB_GID_INDEX: ${WORKER_NCCL_IB_GID_INDEX:-<unset>}"
  echo "  worker dir: $WORKER_DIR"
  if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
    echo "  worker weights: NFS $NFS_VOLUME from ${NFS_SERVER_IP:-$IFACE} (head $HF_CACHE_DIR, no worker copy)"
    echo "  worker JIT cache: $WORKER_HF_CACHE (local overlays on the NFS mount)"
  else
    echo "  worker cache: ${WORKER_HF_CACHE:-${HF_CACHE:-}} (local bind, DSPARK_WORKER_HF_NFS=0)"
  fi
  echo "  GB10 vLLM patch: $ENABLE_VLLM_GB10_PATCH"
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
    echo "  DSV4 perf hotfixes (#50312/#49486+52492/#48407/#48957/#50298/#44993-grammar): SKIPPED (DSPARK_SKIP_HOTFIX=1)"
  else
    echo "  DSV4 perf hotfixes (#50312/#49486+52492/#48407/#48957/#50298/#44993-grammar): will apply on start"
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
  remote_compose "NODE_RANK=1 HEADLESS=1 $WORKER_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='$DSPARK_ISSUE141_EFFECTIVE' DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT='$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT' DSPARK_ISSUE138_HOTFIX='./patches/hotfix-vllm-issue138-responses-history.py' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER_COMPOSE_FILES config --quiet"
  if [ "$DSPARK_TP3" = "1" ]; then
    echo "Validating worker2 compose config..."
    remote_compose2 "NODE_RANK=2 HEADLESS=1 $WORKER2_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER2_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='$DSPARK_ISSUE141_EFFECTIVE' DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT='$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT' DSPARK_ISSUE138_HOTFIX='./patches/hotfix-vllm-issue138-responses-history.py' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER2_COMPOSE_FILES config --quiet"
  fi
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

if [ "$DSPARK_TP3" = "1" ]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER2_HOST" "true" >/dev/null || {
    echo "Cannot reach worker2 with passwordless SSH: $WORKER2_HOST" >&2
    exit 1
  }
  ssh "$WORKER2_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' >/dev/null" || {
    echo "Missing worker2 Docker image $DSPARK_VLLM_IMAGE." >&2
    echo "Pull it on worker2 (e.g. docker pull $DSPARK_VLLM_IMAGE)." >&2
    exit 1
  }
fi

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

if [ "$DSPARK_TP3" = "1" ]; then
  if ssh "$WORKER2_HOST" "if docker ps --format '{{.Names}}' | grep -qx '${PROJECT_NAME}-vllm-dspark-1'; then echo 'DSpark worker2 container already exists for project $PROJECT_NAME. Stop it first.' >&2; exit 1; fi"; then
    :
  else
    worker_rc=$?
    echo "Cannot start: worker2 check on $WORKER2_HOST failed (ssh exit $worker_rc)." >&2
    exit "$worker_rc"
  fi
fi

# Pairwise CX /24s are fine for RoCE but not for Gloo/NCCL TCP bootstrap.
# Rank 2 will try the head's CX IP (e.g. 10.0.22.1) and time out. After GID
# resolve (which must still match the RoCE iface), move bootstrap to the
# 10.0.0.x loopback aliases already used as MASTER_ADDR / VLLM_HOST_IP.
apply_tp3_bootstrap_ifaces() {
  [ "$DSPARK_TP3" = "1" ] || return 0
  # lo has 127.0.0.1 plus 10.0.0.x; Gloo binds 127.0.0.1 and the mesh dies.
  # On this DGX Spark ring the RJ45 NIC is named enP7s7 on every node and
  # shares 192.168.1.0/24 — the only L3 that reaches all three ranks.
  local bif="${TP3_BOOTSTRAP_IFNAME:-enP7s7}"
  echo "TP=3 bootstrap: Gloo/NCCL-socket/TP TCP on ${bif} (LAN all-to-all). NCCL_IB_HCA stays on ConnectX."
  GLOO_SOCKET_IFNAME="$bif"
  TP_SOCKET_IFNAME="$bif"
  NCCL_SOCKET_IFNAME="$bif"
  WORKER_GLOO_SOCKET_IFNAME="$bif"
  WORKER_TP_SOCKET_IFNAME="$bif"
  WORKER_NCCL_SOCKET_IFNAME="$bif"
  WORKER2_GLOO_SOCKET_IFNAME="$bif"
  WORKER2_TP_SOCKET_IFNAME="$bif"
  WORKER2_NCCL_SOCKET_IFNAME="$bif"
  # Pairwise CX /24s: a single HCA makes spark1 try 10.0.22.1 → 10.0.23.3 (RTR timeout).
  local hcas="${TP3_NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1}"
  echo "TP=3 RoCE: NCCL_IB_HCA=${hcas} (both CX ports; GID index already resolved)."
  NCCL_IB_HCA="$hcas"
  WORKER_NCCL_IB_HCA="$hcas"
  WORKER2_NCCL_IB_HCA="$hcas"
  # Dual-port CX: each port is a different /24 to a different neighbor.
  # MERGE_NICS=1 bonds them and QPs the wrong peer; default prefix /16 treats
  # 10.0.22 and 10.0.23 as one subnet. Spark-ring recipe: MERGE=0 + subnet /24.
  NCCL_IB_MERGE_NICS=0
  NCCL_IB_SUBNET_AWARE_ROUTING=1
  NCCL_IB_SUBNET_PREFIX_LEN=24
  echo "TP=3 RoCE: NCCL_IB_MERGE_NICS=0 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24"
}

cd "$SCRIPT_DIR"
resolve_nccl_gid_indexes
apply_tp3_bootstrap_ifaces
STARTUP_LOG_SINCE="$(log_since)"
trap on_error ERR
print_resolved_profile

echo "Syncing DSpark deployment files to ${WORKER_HOST}:${WORKER_DIR}"
ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR"
scp "$COMPOSE_FILE" "${WORKER_HOST}:${REMOTE_COMPOSE_FILE}"
if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
  [ -f "$NFS_OVERRIDE_FILE" ] || { echo "Missing NFS compose override: $NFS_OVERRIDE_FILE" >&2; exit 1; }
  scp "$NFS_OVERRIDE_FILE" "${WORKER_HOST}:${REMOTE_NFS_OVERRIDE_FILE}"
fi
# Stream into a private sibling, then atomically replace the worker env file.
ssh "$WORKER_HOST" "
  set -euo pipefail
  _env_final=$REMOTE_ENV_FILE
  _env_tmp=\"\${_env_final}.tmp.\$\$\"
  _cleanup_remote_env() { [ -z \"\$_env_tmp\" ] || rm -f -- \"\$_env_tmp\"; }
  trap _cleanup_remote_env EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  umask 077
  cat > \"\$_env_tmp\"
  chmod 600 \"\$_env_tmp\"
  mv -f -- \"\$_env_tmp\" \"\$_env_final\"
  _env_tmp=
  trap - EXIT HUP INT TERM
" < "$COMPOSE_ENV_FILE"
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
if [ -f "$DSPARK_ISSUE117_HOTFIX" ] && [ ! -L "$DSPARK_ISSUE117_HOTFIX" ]; then
  echo "Syncing Issue #117 SHM ring hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE117_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-issue117-shm-ring-buffer.py"
fi
# DSV4 v0.27 .sh hotfixes — entrypoint applies them before exec vllm (issue #38).
for _hf_sync in hotfix-dsv4-mtp-buffer-50312.sh hotfix-dsv4-skip-topk-49486.sh hotfix-dsv4-dense-prefill-indexer-48407.sh hotfix-dsv4-skip-empty-c128-48957.sh hotfix-dsv4-flashmla-workspace-50298.sh hotfix-dsv4-grammar-advance.sh hotfix-vllm-redact-api-key-log.sh; do
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
DSPARK_EMPTY_ENCODER_OUTPUT_HOTFIX="${DSPARK_EMPTY_ENCODER_OUTPUT_HOTFIX:-$SCRIPT_DIR/patches/hotfix-vllm-empty-encoder-output.py}"
if [ -f "$DSPARK_EMPTY_ENCODER_OUTPUT_HOTFIX" ]; then
  echo "Syncing Issue #109 empty-encoder-output hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_EMPTY_ENCODER_OUTPUT_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-empty-encoder-output.py"
fi
DSPARK_ISSUE27_HOTFIX="${DSPARK_ISSUE27_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py}"
if [ -f "$DSPARK_ISSUE27_HOTFIX" ]; then
  echo "Syncing Issue #27 partial-prefill hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE27_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
fi
DSPARK_ADAPTIVE_CHUNK_HOTFIX="${DSPARK_ADAPTIVE_CHUNK_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-adaptive-prefill-chunk.py}"
if [ -f "$DSPARK_ADAPTIVE_CHUNK_HOTFIX" ]; then
  echo "Syncing adaptive prefill-chunk hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ADAPTIVE_CHUNK_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-adaptive-prefill-chunk.py"
fi
DSPARK_REPLICATE_MARKOV_HOTFIX="${DSPARK_REPLICATE_MARKOV_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-replicate-markov-head.py}"
if [ -f "$DSPARK_REPLICATE_MARKOV_HOTFIX" ]; then
  echo "Syncing replicate-Markov-head hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_REPLICATE_MARKOV_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-replicate-markov-head.py"
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
DSPARK_ISSUE133_HOTFIX="${DSPARK_ISSUE133_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-issue133-triton-specialization.py}"
if [ -f "$DSPARK_ISSUE133_HOTFIX" ]; then
  echo "Syncing Issue #133 Triton specialization hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE133_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue133-triton-specialization.py"
fi
if [ -f "$DSPARK_ISSUE141_HOTFIX" ]; then
  echo "Syncing Issue #141 sparse-MLA decode workaround to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE141_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py"
fi
if [ -f "$DSPARK_SP_INDEXER_HOTFIX" ]; then
  echo "Syncing SP indexer prefill hotfix (item 6) to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_SP_INDEXER_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-sp-indexer-prefill.py"
fi
if [ -f "$DSPARK_DEEPGEMM_ALIAS_HOTFIX" ]; then
  echo "Syncing DeepGEMM sm121 header alias hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_DEEPGEMM_ALIAS_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-deepgemm-sm121-mqa-header-alias.sh"
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
DSPARK_VISION_EXP_HOTFIX="${DSPARK_VISION_EXP_HOTFIX:-$SCRIPT_DIR/patches/hotfix-dsv4-vision-exp.py}"
if [ -f "$DSPARK_VISION_EXP_HOTFIX" ]; then
  echo "Syncing Vision-Exp native image hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_VISION_EXP_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-vision-exp.py"
fi
if [ -d "$SCRIPT_DIR/patches/vision_exp" ]; then
  echo "Syncing patches/vision_exp/ to ${WORKER_HOST}:${WORKER_DIR}/patches/vision_exp/"
  # Replace the dest dir. `scp -r vision_exp patches/` nests into
  # patches/vision_exp/vision_exp when the dest already exists.
  ssh "$WORKER_HOST" "rm -rf '${REMOTE_WORKER_DIR}/patches/vision_exp' && mkdir -p '${REMOTE_WORKER_DIR}/patches/vision_exp'"
  scp -r "$SCRIPT_DIR/patches/vision_exp/." "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/vision_exp/"
fi
if [ -f "$DSPARK_ISSUE138_HOTFIX" ]; then
  echo "Syncing issue #138 Responses history compatibility patcher to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE138_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-issue138-responses-history.py"
fi
if [ -f "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ] && [ ! -L "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ]; then
  echo "Syncing Issue #136 XGrammar termination hotfix to ${WORKER_HOST}:${WORKER_DIR}/patches/"
  ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches'"
  scp "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-issue136-xgrammar-termination.py"
fi
# Compose always bind-mounts patches/tp3. Create it on the worker even for
# TP=2 so Docker does not invent a root-owned empty dir (or fail the mount).
ssh "$WORKER_HOST" "mkdir -p '${REMOTE_WORKER_DIR}/patches/tp3'"
if [ -f "$SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" ]; then
  scp "$SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/tp3/apply_tp3_patch.py"
fi
if [ "$ENABLE_VLLM_GB10_PATCH" = "1" ]; then
  echo "Syncing GB10 vLLM patch to ${WORKER_HOST}:${WORKER_DIR}/vllm_patch_gb10"
  tar -C "$VLLM_GB10_PATCH_DIR" \
    --exclude='*.egg-info' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - . | ssh "$WORKER_HOST" "mkdir -p $REMOTE_VLLM_GB10_PATCH_DIR && tar -C $REMOTE_VLLM_GB10_PATCH_DIR --no-overwrite-dir -xf -"
fi

sync_tp3_patch_dir() {
  local host="$1"
  local dest_q="$2"
  echo "Syncing patches/tp3 to ${host} (refuse boot if this fails)"
  ssh "$host" "mkdir -p ${dest_q}/patches/tp3" || {
    echo "FATAL: failed to mkdir patches/tp3 on ${host}" >&2
    exit 1
  }
  scp "$SCRIPT_DIR/patches/tp3/apply_tp3_patch.py" "${host}:${dest_q}/patches/tp3/apply_tp3_patch.py" || {
    echo "FATAL: failed to sync apply_tp3_patch.py to ${host} -- refusing to boot TP=3." >&2
    exit 1
  }
  if [ -f "$SCRIPT_DIR/patches/dsv4_tp_pad.py" ]; then
    scp "$SCRIPT_DIR/patches/dsv4_tp_pad.py" "${host}:${dest_q}/patches/dsv4_tp_pad.py" || {
      echo "FATAL: failed to sync dsv4_tp_pad.py to ${host}" >&2
      exit 1
    }
  fi
}

push_compose_env_file() {
  local host="$1"
  local remote_env_q="$2"
  ssh "$host" "
    set -euo pipefail
    _env_final=$remote_env_q
    _env_tmp=\"\${_env_final}.tmp.\$\$\"
    _cleanup_remote_env() { [ -z \"\$_env_tmp\" ] || rm -f -- \"\$_env_tmp\"; }
    trap _cleanup_remote_env EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    umask 077
    cat > \"\$_env_tmp\"
    chmod 600 \"\$_env_tmp\"
    mv -f -- \"\$_env_tmp\" \"\$_env_final\"
    _env_tmp=
    trap - EXIT HUP INT TERM
  " < "$COMPOSE_ENV_FILE"
}

if [ "$DSPARK_TP3" = "1" ]; then
  sync_tp3_patch_dir "$WORKER_HOST" "$REMOTE_WORKER_DIR"
  echo "Syncing DSpark deployment files to ${WORKER2_HOST}:${WORKER2_DIR}"
  ssh "$WORKER2_HOST" "mkdir -p $REMOTE_WORKER2_DIR $REMOTE_WORKER2_DIR/recipe/vllm/v1/spec_decode $REMOTE_WORKER2_DIR/patches"
  scp "$COMPOSE_FILE" "${WORKER2_HOST}:${REMOTE_COMPOSE_FILE2}"
  if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
    [ -f "$NFS_OVERRIDE_FILE" ] || { echo "Missing NFS compose override: $NFS_OVERRIDE_FILE" >&2; exit 1; }
    scp "$NFS_OVERRIDE_FILE" "${WORKER2_HOST}:${REMOTE_NFS_OVERRIDE_FILE2}"
  fi
  push_compose_env_file "$WORKER2_HOST" "$REMOTE_ENV_FILE2"
  scp "$DSPARK_PROPOSER_FILE" "${WORKER2_HOST}:${REMOTE_WORKER2_DIR}/recipe/vllm/v1/spec_decode/dspark_proposer.py"
  tar -C "$SCRIPT_DIR" \
    --exclude='patches/__pycache__' \
    --exclude='patches/*/__pycache__' \
    -cf - patches | ssh "$WORKER2_HOST" "tar -C $REMOTE_WORKER2_DIR --no-overwrite-dir -xf -" || {
    echo "FATAL: failed to sync patches/ to ${WORKER2_HOST} -- refusing to boot TP=3." >&2
    exit 1
  }
  if [ "$ENABLE_VLLM_GB10_PATCH" = "1" ]; then
    tar -C "$VLLM_GB10_PATCH_DIR" \
      --exclude='*.egg-info' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -cf - . | ssh "$WORKER2_HOST" "mkdir -p $REMOTE_VLLM_GB10_PATCH_DIR2 && tar -C $REMOTE_VLLM_GB10_PATCH_DIR2 --no-overwrite-dir -xf -"
  fi
  sync_tp3_patch_dir "$WORKER2_HOST" "$REMOTE_WORKER2_DIR"
fi

if [ "$DSPARK_WORKER_HF_NFS" = "1" ]; then
  echo "Sharing head HF cache over NFS for the worker (no local checkpoint copy)..."
  nfs_ensure_server
  if [ -z "$WORKER_HF_CACHE" ] || [ "$WORKER_HF_CACHE" = "${HF_CACHE:-}" ]; then
    WORKER_HF_CACHE="$(ssh "$WORKER_HOST" 'printf %s "$HOME/.cache/huggingface"')"
    [ -n "$WORKER_HF_CACHE" ] || { echo "Could not resolve worker HOME for JIT cache overlays." >&2; exit 1; }
    WORKER_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER_HF_CACHE'"
    echo "Worker JIT cache defaulted to $WORKER_HF_CACHE"
  fi
  nfs_ensure_worker_jit_dirs "$WORKER_HF_CACHE"
  nfs_ensure_worker_volume recreate
  _nfs_model_rel="hub/models--$(printf '%s' "$DSPARK_MODEL" | sed 's|/|--|g')"
  if nfs_worker_has_model "$_nfs_model_rel"; then
    echo "Worker sees $_nfs_model_rel over NFS ($NFS_VOLUME)."
  else
    echo "WORKER cannot see $_nfs_model_rel over NFS. Check: docker logs ${NFS_CONTAINER} (or the existing NFSv4 exporter on ${NFS_SERVER_IP:-$IFACE})." >&2
    echo "Download weights on the head first: ./prepare-dspark-model-cache.sh --yes" >&2
    exit 1
  fi
  if [ "$DSPARK_TP3" = "1" ]; then
    if [ -z "$WORKER2_HF_CACHE" ] || [ "$WORKER2_HF_CACHE" = "${HF_CACHE:-}" ]; then
      WORKER2_HF_CACHE="$(ssh "$WORKER2_HOST" 'printf %s "$HOME/.cache/huggingface"')"
      [ -n "$WORKER2_HF_CACHE" ] || { echo "Could not resolve worker2 HOME for JIT cache overlays." >&2; exit 1; }
      WORKER2_HF_COMPOSE_ENV="HF_CACHE='$NFS_VOLUME' DSPARK_JIT_CACHE='$WORKER2_HF_CACHE'"
      echo "Worker2 JIT cache defaulted to $WORKER2_HF_CACHE"
    fi
    ssh "$WORKER2_HOST" "mkdir -p $(for d in $(nfs_jit_subdirs); do printf '%s/%s ' "$WORKER2_HF_CACHE" "$d"; done)"
    # 3-node ring: worker2 is on a different CX /24 than worker1. Do not reuse
    # NFS_SERVER_IP from NCCL_SOCKET_IFNAME (spark1↔spark2). Pick a head CX
    # IPv4 worker2 can ping, or WORKER2_NFS_SERVER_IP.
    WORKER2_NFS_SERVER_IP="$(nfs_pick_server_ip_for_host "$WORKER2_HOST" "${WORKER2_NFS_SERVER_IP:-}")" || {
      echo "FATAL: no head ConnectX IPv4 is pingable from worker2 ${WORKER2_HOST}." >&2
      echo "Set WORKER2_NFS_SERVER_IP to the spark1 address on the spark1↔spark3 link (not 10.0.22.1)." >&2
      exit 1
    }
    if [ "$WORKER2_NFS_SERVER_IP" = "$NFS_SERVER_IP" ]; then
      echo "Worker2 NFS via same head IP as worker1: $WORKER2_NFS_SERVER_IP"
    else
      echo "Worker2 NFS via ring peer link: $WORKER2_NFS_SERVER_IP (worker1 uses $NFS_SERVER_IP)"
    fi
    nfs_grant_subnet "$(nfs_subnet24 "$WORKER2_NFS_SERVER_IP")"
    nfs_ensure_host_volume "$WORKER2_HOST" "$WORKER2_NFS_SERVER_IP"
    if timeout 45 ssh "$WORKER2_HOST" "docker run --rm -v '${NFS_VOLUME}:/hf:ro' alpine:latest test -d '/hf/${_nfs_model_rel}'" >/dev/null 2>&1; then
      echo "Worker2 sees $_nfs_model_rel over NFS ($NFS_VOLUME @ $WORKER2_NFS_SERVER_IP)."
    else
      echo "WORKER2 cannot see $_nfs_model_rel over NFS at ${WORKER2_NFS_SERVER_IP}." >&2
      echo "Export must allow $(nfs_subnet24 "$WORKER2_NFS_SERVER_IP") (live vllm-fn-nfs clients are often only worker1's /24)." >&2
      exit 1
    fi
  fi
fi
validate_compose

if [ "${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}" != "1" ]; then
  echo "Checking Issue #117 SHM ring compatibility on the worker before either rank starts..."
  remote_compose "NODE_RANK=1 HEADLESS=1 $WORKER_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER_COMPOSE_FILES run --rm --no-deps --entrypoint python3 vllm-dspark /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py --check"
  echo "Checking Issue #117 SHM ring compatibility on the head before either rank starts..."
  compose_base 0 "" run --rm --no-deps --entrypoint python3 vllm-dspark /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py --check
fi

if [ "${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}" = "1" ]; then
  echo "Checking Issue #136 XGrammar compatibility on the worker before either rank starts..."
  remote_compose "NODE_RANK=1 HEADLESS=1 $WORKER_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER_COMPOSE_FILES run --rm --no-deps --entrypoint python3 vllm-dspark /opt/hotfix-vllm-issue136-xgrammar-termination.py --check"
  if [ "$DSPARK_TP3" = "1" ]; then
    echo "Checking Issue #136 XGrammar compatibility on worker2 before ranks start..."
    remote_compose2 "NODE_RANK=2 HEADLESS=1 $WORKER2_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER2_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER2_COMPOSE_FILES run --rm --no-deps --entrypoint python3 vllm-dspark /opt/hotfix-vllm-issue136-xgrammar-termination.py --check"
  fi
  echo "Checking Issue #136 XGrammar compatibility on the head before either rank starts..."
  compose_base 0 "" run --rm --no-deps --entrypoint python3 vllm-dspark /opt/hotfix-vllm-issue136-xgrammar-termination.py --check
fi

echo "Starting DSpark worker on ${WORKER_HOST}..."
remote_compose "NODE_RANK=1 HEADLESS=1 $WORKER_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='$DSPARK_ISSUE141_EFFECTIVE' DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py' DSPARK_ENABLE_SP_INDEXER='$DSPARK_SP_INDEXER_EFFECTIVE' DSPARK_SP_INDEXER_HOTFIX='./patches/hotfix-dsv4-sp-indexer-prefill.py' DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS='$DSPARK_DEEPGEMM_ALIAS_EFFECTIVE' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT='$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT' DSPARK_ISSUE138_HOTFIX='./patches/hotfix-vllm-issue138-responses-history.py' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER_COMPOSE_FILES up -d"

if [ "$DSPARK_TP3" = "1" ]; then
  echo "Starting DSpark worker2 on ${WORKER2_HOST}..."
  remote_compose2 "NODE_RANK=2 HEADLESS=1 $WORKER2_HF_COMPOSE_ENV VLLM_HOST_IP='$WORKER2_VLLM_HOST_IP' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' DSPARK_MODEL='$DSPARK_MODEL' DSPARK_REVISION='${DSPARK_REVISION:-}' DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='$DSPARK_ISSUE141_EFFECTIVE' DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py' DSPARK_ENABLE_SP_INDEXER='$DSPARK_SP_INDEXER_EFFECTIVE' DSPARK_SP_INDEXER_HOTFIX='./patches/hotfix-dsv4-sp-indexer-prefill.py' DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS='$DSPARK_DEEPGEMM_ALIAS_EFFECTIVE' ENABLE_VLLM_GB10_PATCH='$ENABLE_VLLM_GB10_PATCH' VLLM_GB10_PATCH_DIR='./vllm_patch_gb10' GB10_HYBRID_NVFP4_M_THRESHOLD='${GB10_HYBRID_NVFP4_M_THRESHOLD:-128}' DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT='$DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT' DSPARK_ISSUE138_HOTFIX='./patches/hotfix-vllm-issue138-responses-history.py' docker compose -p '$PROJECT_NAME' --env-file .env.dspark $WORKER2_COMPOSE_FILES up -d"
fi

echo "Starting DSpark head..."
compose_base 0 "" up -d

if [ "${DSPARK_SKIP_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip DSV4 v0.27 perf hotfixes (DSPARK_SKIP_HOTFIX=1)."
fi
if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip Issue #22 hotfix (DSPARK_SKIP_ISSUE22_HOTFIX=1)."
fi
if [ "${DSPARK_SKIP_SPIN_WAIT_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip GB10 shm spin-wait hotfix (DSPARK_SKIP_SPIN_WAIT_HOTFIX=1)."
fi
if [ "${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}" = "1" ]; then
  echo "Entrypoint will skip Issue #117 SHM ring hotfix; the Issue #79 setting is unchanged."
fi
echo "Issue #22 / v0.27 .sh hotfixes run in the compose entrypoint before vllm (no mid-boot stop)."

echo "Waiting for DSpark vLLM API..."
print_initial_startup_logs
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "${AUTH_HEADER_ARGS[@]}" "$API_URL" >/dev/null 2>&1; then
    echo "DeepSeek V4 Flash DSpark is running: $API_URL"
    compose_base 0 "" ps
    remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml ps"
    if [ "$DSPARK_TP3" = "1" ]; then
      remote_compose2 "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml ps"
    fi
    if [ "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then
      echo "Running minimal OpenAI-compatible thinking-budget chat request..."
      curl -fsS --max-time 60 "${AUTH_HEADER_ARGS[@]}" "$CHAT_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":32,"temperature":0.6,"top_p":0.95,"thinking_token_budget":1,"chat_template_kwargs":{"thinking":true,"reasoning_effort":"low"}}' >/dev/null
      echo "Minimal thinking-budget chat request succeeded."
    else
      echo "Running minimal OpenAI-compatible chat request (stock V2; no thinking_token_budget)..."
      curl -fsS --max-time 60 "${AUTH_HEADER_ARGS[@]}" "$CHAT_URL" \
        -H "Content-Type: application/json" \
        -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":32,"temperature":0.6,"top_p":0.95,"chat_template_kwargs":{"thinking":true,"reasoning_effort":"low"}}' >/dev/null
      echo "Minimal chat request succeeded."
    fi
    # Issue #117: burn the spec-decode/prefill Triton shape buckets before real
    # traffic can JIT them mid-serve (a compiling rank can stall its peer past
    # torch's 600 s NCCL watchdog). Non-fatal: warmup gaps degrade back to the
    # mid-serve-JIT status quo, never to a failed boot.
    if [ "${DSPARK_BOOT_SHAPE_WARMUP:-1}" = "1" ]; then
      # Authenticated clusters need a valid bearer or every sweep request 401s
      # and warms nothing. Hand the child the same credential this script's
      # smoke probe uses: the first already-parsed DSPARK_API_KEYS key, else
      # VLLM_API_KEY (they are mutually exclusive upstream). The launcher-to-
      # warmup handoff uses the environment, not a script argument or log line.
      _warmup_bearer="${VLLM_API_KEY:-}"
      if [ "$_dspark_keys_set" = "1" ]; then
        _warmup_bearer="${_dspark_keys[0]}"
      fi
      # Sampler-cache postcondition (see the sweep script): hand the child the
      # HOST path of this node's persistent Triton cache. Only the compose
      # default layout (container path under /cache/huggingface, backed by the
      # HF_CACHE bind mount) is mappable from here; any custom TRITON_CACHE_DIR
      # gets an empty path and the sweep skips that check with a note.
      _warmup_tcache_container="${TRITON_CACHE_DIR:-/cache/huggingface/triton-cache}"
      _warmup_tcache_host=""
      case "$_warmup_tcache_container" in
        /cache/huggingface/*)
          _warmup_tcache_host="${HF_CACHE:-${HOME}/.cache/huggingface}${_warmup_tcache_container#/cache/huggingface}"
          ;;
      esac
      DSPARK_WARMUP_MAX_CONCURRENCY="${MAX_NUM_SEQS:-6}" \
        DSPARK_WARMUP_BEARER="$_warmup_bearer" \
        DSPARK_WARMUP_TRITON_CACHE_DIR="$_warmup_tcache_host" \
        bash "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" \
        "${CHAT_URL%/v1/chat/completions}" "${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}" || \
        echo "WARN: boot shape warmup incomplete — uncovered shapes may JIT mid-serve (issue #117)" >&2
    else
      echo "Boot shape warmup: SKIPPED (DSPARK_BOOT_SHAPE_WARMUP=0)"
    fi
    exit 0
  fi
  wait_with_startup_logs
done

echo "Timed out waiting for DSpark API. Recent head logs:" >&2
compose_base 0 "" logs --tail=120 vllm-dspark >&2 || true
echo "Recent worker logs:" >&2
remote_compose "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark" >&2 || true
if [ "$DSPARK_TP3" = "1" ]; then
  echo "Recent worker2 logs:" >&2
  remote_compose2 "docker compose -p '$PROJECT_NAME' --env-file .env.dspark -f docker-compose.dspark.yml logs --tail=120 vllm-dspark" >&2 || true
fi
exit 1
