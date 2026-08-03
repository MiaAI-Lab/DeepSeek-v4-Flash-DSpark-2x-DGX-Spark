#!/usr/bin/env bash
set -euo pipefail

: "${HEAD_HOST:?HEAD_HOST is required}"
: "${WORKER_HOST:?WORKER_HOST is required}"
: "${HEAD_IP:?HEAD_IP is required}"
: "${WORKER_IP:?WORKER_IP is required}"
: "${HCA:?HCA is required}"
: "${IFACE:?IFACE is required}"
: "${NCCL_TESTS_IMAGE:?NCCL_TESTS_IMAGE is required}"

NCCL_TEST_TIMEOUT_SECONDS="${NCCL_TEST_TIMEOUT_SECONDS:-120}"
[[ "$NCCL_TEST_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "NCCL_TEST_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
}

test_binary="${1:-all_reduce_perf}"
shift || true
[ "$test_binary" = "all_reduce_perf" ] || { echo "Only all_reduce_perf is allowed." >&2; exit 2; }

run_host() {
  local host="$1" command="$2"
  case "$host" in
    localhost|127.0.0.1) bash -lc "$command" ;;
    *) ssh "$host" "$command" ;;
  esac
}

for host in "$HEAD_HOST" "$WORKER_HOST"; do
  run_host "$host" "docker image inspect '$NCCL_TESTS_IMAGE' >/dev/null"
done

# Extract the pinned diagnostic runtime to an identical path on both hosts.
# Running the rank directly avoids crossing a Docker/PMIx security boundary:
# the host daemons and rank processes retain the same UID and PMIx namespace.
mpi_runtime="/tmp/dspark-openmpi-runtime-$$-$RANDOM"
cleanup() {
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    run_host "$host" "pkill -TERM -f '^$mpi_runtime/' 2>/dev/null || true; sleep 1; pkill -KILL -f '^$mpi_runtime/' 2>/dev/null || true" 2>/dev/null || true
    run_host "$host" "find '$mpi_runtime' -depth -delete" 2>/dev/null || true
  done
}
trap cleanup EXIT

# Extract one identical OpenMPI runtime, nccl-tests binary, and NCCL library
# from the pinned image on both hosts. CUDA and the NVIDIA driver remain the
# host-provided DGX stack.
for host in "$HEAD_HOST" "$WORKER_HOST"; do
  run_host "$host" "set -e; mkdir -p '$mpi_runtime'; cid=\$(docker create '$NCCL_TESTS_IMAGE'); trap 'docker rm -f \"\$cid\" >/dev/null 2>&1 || true' EXIT; docker cp \"\$cid:/opt/openmpi/.\" '$mpi_runtime'; docker cp \"\$cid:/opt/nccl-tests/build/all_reduce_perf\" '$mpi_runtime/bin/all_reduce_perf'; docker cp \"\$cid:/usr/lib/aarch64-linux-gnu/libnccl.so.2.28.9\" '$mpi_runtime/lib/libnccl.so.2'; docker rm \"\$cid\" >/dev/null; trap - EXIT; chmod 0755 '$mpi_runtime/bin/all_reduce_perf'; OPAL_PREFIX='$mpi_runtime' LD_LIBRARY_PATH='$mpi_runtime/lib' '$mpi_runtime/bin/mpirun' -V | grep -F '4.1.6' >/dev/null; LD_LIBRARY_PATH='$mpi_runtime/lib' ldd '$mpi_runtime/bin/all_reduce_perf' | grep -F 'not found' && exit 1 || true"
done

mpi_head_host="${HEAD_HOST#*@}"
mpi_worker_host="${WORKER_HOST#*@}"
export NCCL_IB_HCA="$HCA" NCCL_SOCKET_IFNAME="$IFACE"
mpi_export_args=(-x HCA -x IFACE)
while IFS='=' read -r key _; do
  case "$key" in NCCL_*) mpi_export_args+=(-x "$key") ;; esac
done < <(env)
timeout --signal=TERM --kill-after=10s "${NCCL_TEST_TIMEOUT_SECONDS}s" \
  env OPAL_PREFIX="$mpi_runtime" LD_LIBRARY_PATH="$mpi_runtime/lib" \
  "$mpi_runtime/bin/mpirun" --prefix "$mpi_runtime" \
  --mca btl_tcp_if_include "$IFACE" --host "$mpi_head_host:1,$mpi_worker_host:1" -np 2 \
  "${mpi_export_args[@]}" \
  "$mpi_runtime/bin/$test_binary" "$@"
