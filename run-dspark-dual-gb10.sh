#!/usr/bin/env bash
set -euo pipefail

IMAGE="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
NAME="${NAME:-dspark_vllm}"
MODEL_DIR="${DSPARK_MODEL_DIR:-${HOME}/models/llm/dspark/deepseek-ai/DeepSeek-V4-Flash-DSpark/hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark}"
HEAD_IP="${MASTER_ADDR:-head-roce-ip}"
WORKER_IP="${WORKER_HOST:-worker-host-or-roce-ip}"
MASTER_PORT="${MASTER_PORT:-25000}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
SPEC_TOKENS="${MTP_NUM_TOKENS:-5}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4_ds_mla}"

HEAD_ETH_IF="${HEAD_ETH_IF:-enp1s0f0np0}"
WORKER_ETH_IF="${WORKER_ETH_IF:-enp1s0f1np1}"
HEAD_IB_HCA="${HEAD_IB_HCA:-rocep1s0f0}"
WORKER_IB_HCA="${WORKER_IB_HCA:-rocep1s0f1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

common_args=(
  --gpus all --ipc=host --network host
  --shm-size=64g
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:-1
  --cap-add=IPC_LOCK
  --device=/dev/infiniband
  -v "${MODEL_DIR}:/model:ro"
  -v "${HF_CACHE:-${HOME}/.cache/huggingface}:/cache/huggingface"
  -e HF_HOME=/cache/huggingface
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
  -e VLLM_TRITON_MLA_SPARSE="${VLLM_TRITON_MLA_SPARSE:-1}"
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
  -e VLLM_SKIP_INIT_MEMORY_CHECK="${VLLM_SKIP_INIT_MEMORY_CHECK:-1}"
  -e VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
  -e VLLM_USE_B12X_WO_PROJECTION="${VLLM_USE_B12X_WO_PROJECTION:-1}"
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM="${VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM:-0}"
  -e VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M="${VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M:-16}"
  -e VLLM_B12X_W4A16_FORCE_TILE_CONFIG="${VLLM_B12X_W4A16_FORCE_TILE_CONFIG:-}"
  -e B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-0}"
  -e VLLM_DSPARK_CONFIDENCE_THRESHOLD="${VLLM_DSPARK_CONFIDENCE_THRESHOLD:-0.0}"
  -e VLLM_DSPARK_CONFIDENCE_SCHEDULER="${VLLM_DSPARK_CONFIDENCE_SCHEDULER:-off}"
  -e VLLM_DSPARK_LOCAL_ARGMAX="${VLLM_DSPARK_LOCAL_ARGMAX:-1}"
  -e VLLM_DSPARK_REPLICATE_MARKOV_W1="${VLLM_DSPARK_REPLICATE_MARKOV_W1:-1}"
  -e VLLM_DSPARK_FUSED_MARKOV_ARGMAX="${VLLM_DSPARK_FUSED_MARKOV_ARGMAX:-0}"
  -e VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK="${VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK:-1}"
  -e VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT="${VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT:-0}"
  -e VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP="${VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP:-1}"
  -e VLLM_DSV4_B12X_COMPRESSED_MLA="${VLLM_DSV4_B12X_COMPRESSED_MLA:-0}"
  -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE="${VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE:-0}"
  -e VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT="${VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT:-0}"
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
  -e FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
  -e FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
  -e TILELANG_CLEANUP_TEMP_FILES="${TILELANG_CLEANUP_TEMP_FILES:-1}"
  -e DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-0}"
  -e DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-/opt/env/bin/nvcc}"
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  -e NCCL_NET="${NCCL_NET:-IB}"
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}"
  -e NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
  -e NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
  -e NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  -e NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  -e MASTER_ADDR="${HEAD_IP}"
  -e MASTER_PORT="${MASTER_PORT}"
  -e MTP_NUM_TOKENS="${SPEC_TOKENS}"
)

make_cmd() {
  local rank="$1" headless="$2"
  cat <<EOF
set -euo pipefail
export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:\${PATH:-}"
export CUDA_HOME="\${CUDA_HOME:-/opt/env/targets/sbsa-linux}"
export CUDA_PATH="\${CUDA_PATH:-\${CUDA_HOME}}"
export CUDAToolkit_ROOT="\${CUDAToolkit_ROOT:-\${CUDA_HOME}}"
export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\${LD_LIBRARY_PATH:-}"
exec /opt/env/bin/vllm serve /model \\
  --served-model-name "${SERVED_MODEL_NAME}" \\
  --host 0.0.0.0 --port "${PORT}" \\
  --trust-remote-code \\
  --tensor-parallel-size 2 \\
  --pipeline-parallel-size 1 \\
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \\
  --block-size 256 \\
  --max-model-len "${MAX_MODEL_LEN}" \\
  --max-num-seqs "${MAX_NUM_SEQS}" \\
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \\
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \\
  --enable-prefix-caching \\
  --speculative-config '{"method":"dspark","num_speculative_tokens":${SPEC_TOKENS}}' \\
  --tokenizer-mode deepseek_v4 \\
  --distributed-executor-backend mp \\
  --tool-call-parser deepseek_v4 \\
  --enable-auto-tool-choice \\
  --reasoning-parser deepseek_v4 \\
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \\
  --default-chat-template-kwargs '{"thinking":true}' \\
  --enable-flashinfer-autotune \\
  --nnodes 2 --node-rank "${rank}" --master-addr "${HEAD_IP}" --master-port "${MASTER_PORT}" ${headless}
EOF
}

echo "== stopping previous ${NAME} containers =="
docker rm -f "${NAME}" 2>/dev/null || true
ssh "${WORKER_IP}" "docker rm -f '${NAME}' 2>/dev/null || true"

echo "== starting worker rank 1 (${WORKER_IP}) =="
WORKER_CMD="$(make_cmd 1 --headless)"
ssh "${WORKER_IP}" docker run -d --name "${NAME}" \
  "${common_args[@]}" \
  -e NODE_RANK=1 -e HEADLESS=1 \
  -e NCCL_IB_HCA="${WORKER_IB_HCA}" \
  -e NCCL_SOCKET_IFNAME="${WORKER_ETH_IF}" \
  --entrypoint bash "${IMAGE}" -lc "$(printf '%q' "${WORKER_CMD}")"

sleep 10

echo "== starting head rank 0 (${HEAD_IP}) =="
HEAD_CMD="$(make_cmd 0 '')"
docker run -d --name "${NAME}" \
  "${common_args[@]}" \
  -e NODE_RANK=0 -e HEADLESS= \
  -e NCCL_IB_HCA="${HEAD_IB_HCA}" \
  -e NCCL_SOCKET_IFNAME="${HEAD_ETH_IF}" \
  --entrypoint bash "${IMAGE}" -lc "${HEAD_CMD}"

echo "API: http://${HEAD_IP}:${PORT}/v1/models"
echo "Logs: docker logs -f ${NAME}; ssh ${WORKER_IP} docker logs -f ${NAME}"
