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
#
# SINGLE-TENANT ASSUMPTION. The glob is deliberately every vLLM instance on the
# node, not just this deployment's: the file name embeds the instance id of the
# process that created it, and that process (with its id) lived in the
# container just removed, so the id cannot be recovered here. On a node that
# serves anything else this would unlink a live sibling instance's staging
# region. The launcher owns the whole node -- stop-deepseek-v4-flash-dspark.sh
# removes the same glob -- so this is safe only there.
rm -f /dev/shm/vllm_offload_*.mmap 2>/dev/null || true

for _ in $(seq 1 30); do
  # A FAILED nvidia-smi is not "no compute apps": `apps` would be empty, the
  # script would report a clean GPU on a broken driver/toolkit, and the launch
  # would proceed into the OOM this check exists to prevent. Unknown state is
  # reported as such and refuses to certify the GPU. stderr is left attached so
  # nvidia-smi's own diagnostic reaches the operator.
  if apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
    if [ -z "$apps" ]; then
      echo "GPU_CLEAR ($(hostname))"
      exit 0
    fi
    sleep 2
  else
    rc=$?
    echo "GPU_UNKNOWN ($(hostname)): nvidia-smi failed (exit $rc); the GPU state" >&2
    echo "cannot be determined. Refusing to report a clean GPU -- fix the driver" >&2
    echo "or NVIDIA tooling, then re-run." >&2
    exit 2
  fi
done

echo "GPU_STILL_BUSY ($(hostname)): $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')" >&2
echo "Something else is using this GPU. Stop it, then re-run." >&2
exit 1
