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

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${MASTER_PORT:?MASTER_PORT must be set in $ENV_FILE}"
: "${DSPARK_VLLM_IMAGE:?DSPARK_VLLM_IMAGE must be set in $ENV_FILE}"
: "${DSPARK_MODEL_REVISION:?DSPARK_MODEL_REVISION must be set in $ENV_FILE}"
: "${VLLM_ORIGIN_KEY_FILE:?VLLM_ORIGIN_KEY_FILE must be set in $ENV_FILE}"

if ! [[ "$DSPARK_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "DSPARK_MODEL_REVISION must be a 40-character commit SHA." >&2
  exit 1
fi
if ! [[ "$DSPARK_VLLM_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "DSPARK_VLLM_IMAGE must use an immutable @sha256 digest." >&2
  exit 1
fi
if [ ! -f "$VLLM_ORIGIN_KEY_FILE" ] || [ -L "$VLLM_ORIGIN_KEY_FILE" ] || [ ! -s "$VLLM_ORIGIN_KEY_FILE" ]; then
  echo "VLLM_ORIGIN_KEY_FILE must be a non-empty regular file, not a symlink." >&2
  exit 1
fi
key_mode="$(stat -f '%Lp' "$VLLM_ORIGIN_KEY_FILE" 2>/dev/null || stat -c '%a' "$VLLM_ORIGIN_KEY_FILE")"
if [ "$key_mode" != "600" ]; then
  echo "VLLM_ORIGIN_KEY_FILE must have mode 0600." >&2
  exit 1
fi

if [ "${VALIDATE_RENDER:-1}" = "0" ]; then
  exit 0
fi

echo "DSpark config:"
echo "  worker: ${WORKER_HOST}"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  image: ${DSPARK_VLLM_IMAGE}"
echo "  model: ${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}"
echo "  served model: ${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
echo "  max model len: ${MAX_MODEL_LEN:-1048576}"
echo "  max num seqs: ${MAX_NUM_SEQS:-6}"
echo "  max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-8192}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION:-0.80}"
echo "  spec tokens (MTP_NUM_TOKENS): ${MTP_NUM_TOKENS:-5} with draft_sample_method=probabilistic (min 5 = dspark_block_size)"
echo "  cudagraph capture size: $(( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-5} + 1) )) (max_num_seqs * (mtp + 1))"
echo "  breakable cudagraph: ${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
echo "  dspark slot clamp: ${DSPARK_SLOT_CLAMP:-1}"
echo "  sampling override: none (no --override-generation-config; --generation-config vllm only)"
echo "  WO projection: ${VLLM_USE_B12X_WO_PROJECTION:-1}"
echo "  host bind: ${VLLM_HOST:-127.0.0.1}"
echo
echo "Rendered vLLM command (secret values redacted):"
env -u MASTER_PORT -u NODE_RANK -u HEADLESS -u WORKER_HOST -u MASTER_ADDR \
  COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" config \
  | python3 -c 'import pathlib,sys; secret=pathlib.Path(sys.argv[1]).read_text().strip(); data=sys.stdin.read(); sys.stdout.write(data.replace(secret, "[REDACTED]") if secret else data)' "$VLLM_ORIGIN_KEY_FILE" \
  | grep -E -- '--revision|--max-model-len|--max-num-seqs|--max-num-batched-tokens|--max-cudagraph-capture-size|--gpu-memory-utilization|--master-port|--kv-cache-dtype|--speculative-config|--async-scheduling|--enable-chunked-prefill|--generation-config|image:|VLLM_USE_B12X_WO_PROJECTION|VLLM_USE_BREAKABLE_CUDAGRAPH|VLLM_USE_FLASHINFER_SAMPLER|MTP_NUM_TOKENS'
