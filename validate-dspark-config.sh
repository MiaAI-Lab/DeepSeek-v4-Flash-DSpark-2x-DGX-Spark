#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark.yml}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.dspark.example to .env.dspark and edit it." >&2
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

# Match start-deepseek-v4-flash-dspark.sh: flag selects main util profile.
if [ "${ENABLE_VL_SIDECAR:-0}" = "1" ]; then
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_VISION:-0.80}"
  DSPARK_SERVE_MODE="vision"
else
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_TEXT:-0.835}"
  DSPARK_SERVE_MODE="text"
fi
export GPU_MEMORY_UTILIZATION

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
export DSPARK_MODEL DSPARK_REVISION

# Mirror start-deepseek-v4-flash-dspark.sh exactly: same whitelist, same 0/1
# gates, so this validator and the start script can never disagree about one
# env file (otherwise: "config OK" here, `exit 2` there).
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"
case "$KV_CACHE_DTYPE" in
  nvfp4_ds_mla|fp8|fp8_ds_mla) ;;
  *)
    echo "KV_CACHE_DTYPE must be nvfp4_ds_mla, fp8, or fp8_ds_mla (got: $KV_CACHE_DTYPE)" >&2
    exit 2
    ;;
esac
ENABLE_DSPARK_SPECULATIVE="${ENABLE_DSPARK_SPECULATIVE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
case "$ENABLE_DSPARK_SPECULATIVE:$ENFORCE_EAGER" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "ENABLE_DSPARK_SPECULATIVE and ENFORCE_EAGER must be 0 or 1." >&2; exit 2 ;;
esac
export KV_CACHE_DTYPE ENABLE_DSPARK_SPECULATIVE ENFORCE_EAGER

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${MASTER_PORT:?MASTER_PORT must be set in $ENV_FILE}"
: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"

echo "DSpark config:"
echo "  worker: ${WORKER_HOST}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  image: ${DSPARK_VLLM_IMAGE}"
echo "  serve mode: $DSPARK_SERVE_MODE (ENABLE_VL_SIDECAR=${ENABLE_VL_SIDECAR:-0})"
echo "  checkpoint: $DSPARK_MODEL (ABLITERATED=${ABLITERATED:-0})"
if [ -n "${DSPARK_REVISION:-}" ]; then
  echo "  revision: $DSPARK_REVISION"
else
  echo "  revision: (default branch tip / unpinned)"
fi

# Warn when the pinned revision is not in the local HF cache: starting the
# stack would silently begin a very large download (~155 GB for 0731). This is
# easy to hit when upgrading a deployment that predates the issue #19 pin, as
# its cached snapshot is whatever `main` was at install time.
check_revision_cached() {
  [ -n "${DSPARK_REVISION:-}" ] || return 0

  hf_home="${HF_HOME:-${DSPARK_HF_CACHE:-$HOME/.cache/huggingface}}"
  case "$hf_home" in
    */huggingface) hub_dir="$hf_home/hub" ;;
    *) hub_dir="$hf_home/hub" ;;
  esac
  model_dir="$hub_dir/models--$(printf '%s' "$DSPARK_MODEL" | sed 's|/|--|g')"
  snapshots_dir="$model_dir/snapshots"

  # No local cache at all: a first-time install is expected to download.
  [ -d "$snapshots_dir" ] || return 0
  [ -d "$snapshots_dir/$DSPARK_REVISION" ] && return 0

  cached="$(ls -1 "$snapshots_dir" 2>/dev/null | tr '\n' ' ' | sed 's/ $//')"
  [ -n "$cached" ] || return 0

  echo ""
  echo "  [WARN] Pinned revision is NOT in the local HF cache:"
  echo "           pinned: $DSPARK_REVISION"
  echo "           cached: $cached"
  echo "         Starting will download the full checkpoint (~155 GB for 0731)."
  echo "         To keep using the cached weights, set in $ENV_FILE on BOTH nodes:"
  echo "           DSPARK_REVISION=${cached%% *}"
  echo "         To fetch the pinned revision deliberately, run"
  echo "         ./prepare-dspark-model-cache.sh first."
  echo ""
}
check_revision_cached
echo "  model: ${DSPARK_MODEL}"
echo "  served model: ${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
echo "  max model len: ${MAX_MODEL_LEN:-1048576}"
echo "  max num seqs: ${MAX_NUM_SEQS:-6}"
echo "  max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-8192}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION} (text ${GPU_MEMORY_UTILIZATION_TEXT:-0.835} / vision ${GPU_MEMORY_UTILIZATION_VISION:-0.80})"
echo "  KV cache dtype: ${KV_CACHE_DTYPE} (both layouts share the same 584B/token DSV4 record, so the KV pool is dtype-independent)"
echo "  speculative decoding: ${ENABLE_DSPARK_SPECULATIVE} / enforce eager: ${ENFORCE_EAGER}"
if [ "$ENABLE_DSPARK_SPECULATIVE" = "1" ]; then
  echo "  spec tokens (MTP_NUM_TOKENS): ${MTP_NUM_TOKENS:-5} with draft_sample_method=probabilistic (min 5 = dspark_block_size)"
else
  echo "  spec tokens (MTP_NUM_TOKENS): inactive (ENABLE_DSPARK_SPECULATIVE=0)"
fi
if [ "$ENFORCE_EAGER" = "1" ]; then
  echo "  cudagraph capture size: none (--enforce-eager; no graphs are captured)"
elif [ "$ENABLE_DSPARK_SPECULATIVE" = "1" ]; then
  echo "  cudagraph capture size: $(( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-5} + 1) )) (max_num_seqs * (mtp + 1))"
else
  echo "  cudagraph capture size: ${MAX_NUM_SEQS:-6} (max_num_seqs * 1; speculation off)"
fi
echo "  breakable cudagraph: ${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
echo "  dspark slot clamp: ${DSPARK_SLOT_CLAMP:-1}"
echo "  sampling override: none (no --override-generation-config; --generation-config vllm only)"
echo "  WO projection: ${VLLM_USE_B12X_WO_PROJECTION:-1}"
echo "  host bind: ${VLLM_HOST:-127.0.0.1}"
echo
echo "Rendered vLLM command:"
env -u MASTER_PORT -u NODE_RANK -u HEADLESS -u WORKER_HOST -u MASTER_ADDR \
  COMPOSE_DISABLE_ENV_FILE=1 \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  DSPARK_MODEL="$DSPARK_MODEL" \
  DSPARK_REVISION="${DSPARK_REVISION:-}" \
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" config \
  | grep -E -- '--max-model-len|--max-num-seqs|--max-num-batched-tokens|--max-cudagraph-capture-size|--gpu-memory-utilization|--master-port|--kv-cache-dtype|--speculative-config|--enforce-eager|--async-scheduling|--enable-chunked-prefill|--generation-config|--revision|image:|VLLM_USE_B12X_WO_PROJECTION|VLLM_USE_BREAKABLE_CUDAGRAPH|VLLM_USE_FLASHINFER_SAMPLER|MTP_NUM_TOKENS|DSPARK_REVISION'
