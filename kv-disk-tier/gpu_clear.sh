#!/usr/bin/env bash
# Clear THIS node's GPU before a launch: remove a previous dsv4 container and
# wait until nvidia-smi reports no compute apps. An NCCL-stuck worker from a
# failed handshake holds ~77 GB and the next launch will fail on memory.
#
# Deliberately does NOT kill arbitrary GPU processes. An earlier private version
# `kill -9`d every pid from `nvidia-smi --query-compute-apps`, which is fine on a
# dedicated box and hostile anywhere else. If something else is using the GPU,
# this tells you and exits non-zero so you can decide.
set -u
NAME="${CONTAINER_NAME:-dsv4-0731}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# The offload staging region is a /dev/shm file that is NOT unlinked on an
# unclean shutdown. It is charged against vLLM's host-RAM gate at startup and
# surfaces as "Free memory on device ... less than desired GPU memory
# utilization" -- an error that sends you to --gpu-memory-utilization, which
# cannot fix it. Remove it here instead.
rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true

for _ in $(seq 1 30); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  if [ -z "$apps" ]; then
    echo "GPU_CLEAR ($(hostname))"
    exit 0
  fi
  sleep 2
done

echo "GPU_STILL_BUSY ($(hostname)): $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')" >&2
echo "Something else is using this GPU. Stop it, then re-run." >&2
exit 1
