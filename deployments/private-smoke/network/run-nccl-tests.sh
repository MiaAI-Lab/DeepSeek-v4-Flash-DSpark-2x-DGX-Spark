#!/usr/bin/env bash
set -euo pipefail

: "${HEAD_HOST:?HEAD_HOST is required}"
: "${WORKER_HOST:?WORKER_HOST is required}"
: "${HEAD_IP:?HEAD_IP is required}"
: "${WORKER_IP:?WORKER_IP is required}"
: "${HCA:?HCA is required}"
: "${IFACE:?IFACE is required}"
: "${NCCL_TESTS_IMAGE:?NCCL_TESTS_IMAGE is required}"

test_binary="${1:-all_reduce_perf}"
shift || true
[ "$test_binary" = "all_reduce_perf" ] || { echo "Only all_reduce_perf is allowed." >&2; exit 2; }

command -v mpirun >/dev/null 2>&1 || {
  echo "mpirun is required on the control host for the two-node NCCL gate." >&2
  exit 1
}

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

# OpenMPI starts this wrapper once per host. The wrapper passes only the MPI
# coordination variables into an isolated host-networked test container.
wrapper="$(mktemp "${TMPDIR:-/tmp}/dspark-nccl-wrapper.XXXXXX")"
remote_wrapper="/tmp/dspark-nccl-wrapper-$$-$RANDOM"
cleanup() {
  find "$wrapper" -maxdepth 0 -type f -delete 2>/dev/null || true
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    run_host "$host" "find '$remote_wrapper' -maxdepth 0 -type f -delete" 2>/dev/null || true
  done
}
trap cleanup EXIT
cat >"$wrapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
env_args=()
while IFS='=' read -r key _; do
  case "$key" in OMPI_*|PMIX_*|NCCL_*|CUDA_VISIBLE_DEVICES) env_args+=(--env "$key") ;; esac
done < <(env)
exec docker run --rm --network host --ipc host --gpus all \
  --volume /dev/infiniband:/dev/infiniband \
  --volume /tmp:/tmp --volume /run:/run \
  --ulimit memlock=-1:-1 \
  "${env_args[@]}" \
  --env NCCL_IB_HCA="$HCA" --env NCCL_SOCKET_IFNAME="$IFACE" \
  "$NCCL_TESTS_IMAGE" "$@"
EOF
chmod 0700 "$wrapper"

for host in "$HEAD_HOST" "$WORKER_HOST"; do
  case "$host" in
    localhost|127.0.0.1) install -m 0700 "$wrapper" "$remote_wrapper" ;;
    *) scp -q "$wrapper" "$host:$remote_wrapper"; ssh "$host" "chmod 0700 '$remote_wrapper'" ;;
  esac
done

mpirun --mca btl_tcp_if_include "$IFACE" --host "$HEAD_HOST:1,$WORKER_HOST:1" -np 2 \
  -x HCA -x IFACE -x NCCL_TESTS_IMAGE \
  "$remote_wrapper" "$@"
