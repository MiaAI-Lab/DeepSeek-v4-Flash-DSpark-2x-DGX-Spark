# files/nfs-share.sh — share the head HuggingFace cache over NFS (ConnectX).
# Sourced by start-deepseek-v4-flash-dspark.sh. Requires: WORKER_HOST, IFACE,
# HF_CACHE_DIR, and host_without_user (or WORKER_IP).
#
# Same pattern as Qwen3.8-Flash-vLLM: the worker does not keep a local copy of
# the checkpoint. If NFSv4 is already listening on the ConnectX address (for
# example vllm-fn-nfs), this reuses it and does not start a second nfsd.

NFS_IMAGE="${NFS_IMAGE:-dspark-nfs:local}"
NFS_CONTAINER="${NFS_CONTAINER:-dspark-nfs}"
NFS_VOLUME="${NFS_VOLUME:-dspark-hf}"
NFS_DOCKERFILE_DIR="${NFS_DOCKERFILE_DIR:-$SCRIPT_DIR/files/nfs-server}"

if ! declare -F ssh_worker >/dev/null 2>&1; then
  ssh_worker() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "$@"
  }
fi
if ! declare -F nfs_info >/dev/null 2>&1; then
  nfs_info() { echo "$*"; }
  nfs_ok() { echo "$*"; }
  nfs_err() { echo "$*" >&2; exit 1; }
fi

nfs_detect_server_ip() {
  if [ -n "${NFS_SERVER_IP:-}" ]; then
    return 0
  fi
  NFS_SERVER_IP=$(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
  [ -n "$NFS_SERVER_IP" ] || nfs_err "Could not detect NFS_SERVER_IP from IFACE=$IFACE. Set NFS_SERVER_IP in .env.dspark (head ConnectX address, e.g. 10.0.22.1 — not the 10.0.0.1 loopback alias)."
}

nfs_clients() {
  local cidr net mask addr worker_ip
  worker_ip="${WORKER_IP:-}"
  if [ -z "$worker_ip" ] && declare -F host_without_user >/dev/null 2>&1; then
    worker_ip="$(host_without_user "$WORKER_HOST")"
  fi
  cidr=$(ip -4 -o addr show dev "$IFACE" 2>/dev/null | awk '{print $4}' | head -1)
  if [[ "$cidr" == */* ]]; then
    addr="${cidr%/*}"
    mask="${cidr#*/}"
    net="${addr%.*}.0/${mask}"
    if [ -n "$worker_ip" ]; then
      echo "${worker_ip},${net}"
    else
      echo "${net}"
    fi
  else
    echo "${worker_ip:-*}"
  fi
}

nfs_jit_subdirs() {
  echo "triton-cache tilelang-cache vllm-cache flashinfer b12x-cute-cache nccl-fr"
}

nfs_ensure_jit_dirs() {
  local root="$1"
  # Overlay mount points must already exist on the (read-only) NFS export.
  # shellcheck disable=SC2086
  mkdir -p "$root" $(for d in $(nfs_jit_subdirs); do printf '%s/%s ' "$root" "$d"; done)
}

nfs_ensure_server() {
  export PATH="/usr/sbin:/sbin:${PATH}"
  nfs_detect_server_ip
  local clients
  clients="$(nfs_clients)"

  [ -d "$NFS_DOCKERFILE_DIR" ] || nfs_err "Missing $NFS_DOCKERFILE_DIR"
  [ -d "$HF_CACHE_DIR" ] || nfs_err "HF cache not found at $HF_CACHE_DIR"
  nfs_ensure_jit_dirs "$HF_CACHE_DIR"

  # A live NFSv4 server is enough — do not docker rm a working share (Qwen's
  # vllm-fn-nfs or a previous dspark-nfs). Kernel nfsd in a privileged
  # container can leave rpcbind in D-state, which makes `docker rm -f` hang.
  if timeout 3 rpcinfo -t "$NFS_SERVER_IP" nfs 4 >/dev/null 2>&1; then
    nfs_ok "NFS share already up on ${NFS_SERVER_IP}:2049 → $HF_CACHE_DIR"
    return 0
  fi

  nfs_info "Building NFS image $NFS_IMAGE ..."
  docker build -q -t "$NFS_IMAGE" "$NFS_DOCKERFILE_DIR" >/dev/null

  docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1 || true

  nfs_info "Exporting $HF_CACHE_DIR via NFS on $NFS_SERVER_IP (clients: $clients)"
  docker run -d --name "$NFS_CONTAINER" --restart unless-stopped \
    --privileged --network host \
    -v "$HF_CACHE_DIR:/export:ro" \
    -e "NFS_CLIENTS=$clients" \
    "$NFS_IMAGE" >/dev/null

  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    if timeout 3 rpcinfo -t "$NFS_SERVER_IP" nfs 4 >/dev/null 2>&1; then
      nfs_ok "NFS server ready on ${NFS_SERVER_IP}:2049 (NFSv4)"
      return 0
    fi
    sleep 0.5
  done
  docker logs "$NFS_CONTAINER" >&2 || true
  nfs_err "NFS server did not become ready on $NFS_SERVER_IP. Check: docker logs $NFS_CONTAINER"
}

nfs_ensure_worker_volume() {
  local recreate="${1:-}"
  nfs_detect_server_ip
  if [ "$recreate" != "recreate" ] && ssh_worker "docker volume inspect '$NFS_VOLUME' >/dev/null 2>&1"; then
    nfs_ok "Worker volume $NFS_VOLUME already exists"
    return 0
  fi
  nfs_info "Creating worker NFS volume $NFS_VOLUME → ${NFS_SERVER_IP}:/"
  ssh_worker "docker volume rm '$NFS_VOLUME' >/dev/null 2>&1 || true"
  ssh_worker "docker volume create --driver local \
    --opt type=nfs \
    --opt o=addr=${NFS_SERVER_IP},nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600 \
    --opt device=:/ \
    '$NFS_VOLUME' >/dev/null"
  nfs_ok "Worker volume $NFS_VOLUME → nfs://${NFS_SERVER_IP}/"
}

nfs_ensure_worker_jit_dirs() {
  local root="${1:-}"
  [ -n "$root" ] || nfs_err "nfs_ensure_worker_jit_dirs: missing host path"
  # shellcheck disable=SC2086
  ssh_worker "mkdir -p $(for d in $(nfs_jit_subdirs); do printf '%s/%s ' "$root" "$d"; done)"
}

nfs_worker_has_model() {
  local rel="$1"
  ssh_worker "docker run --rm -v '${NFS_VOLUME}:/hf:ro' alpine:latest test -d '/hf/${rel}'" >/dev/null 2>&1
}

# Head CX7 IPv4s on the onboard dual-port (enp1s*), not the 10.0.0.x lo alias
# and not docker bridges. A 3-node QSFP ring puts each peer on a different /24.
nfs_head_cx_ipv4s() {
  ip -4 -o addr show | awk '$2 ~ /^enp1s/ { split($4, a, "/"); print a[1] }'
}

nfs_pick_server_ip_for_host() {
  local host="$1"
  local pinned="${2:-}"
  local ip
  if [ -n "$pinned" ]; then
    printf '%s' "$pinned"
    return 0
  fi
  for ip in $(nfs_head_cx_ipv4s); do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
      "ping -c1 -W1 $(printf '%q' "$ip") >/dev/null 2>&1"; then
      printf '%s' "$ip"
      return 0
    fi
  done
  return 1
}

nfs_subnet24() {
  local ip="$1"
  printf '%s.0/24' "${ip%.*}"
}

# Additive: allow a new /24 on an already-running exporter (vllm-fn-nfs or
# dspark-nfs). Does not recreate the container (Qwen may be using the share).
nfs_grant_subnet() {
  local subnet="$1"
  local c opts line
  opts="${NFS_OPTS:-ro,sync,no_subtree_check,no_root_squash,insecure,fsid=0}"
  for c in vllm-fn-nfs "${NFS_CONTAINER:-dspark-nfs}"; do
    [ -n "$(docker ps -q --filter "name=^/${c}$")" ] || continue
    line="/export ${subnet}(${opts})"
    docker exec "$c" sh -c "
      set -e
      if grep -Fq '${subnet}(' /etc/exports 2>/dev/null; then
        echo 'exports already allow ${subnet}'
        exit 0
      fi
      echo '${line}' >> /etc/exports
      exportfs -rav
    " || nfs_info "WARN: could not add ${subnet} to ${c} exports"
  done
}

nfs_ensure_host_volume() {
  local host="$1"
  local server_ip="$2"
  ssh "$host" "docker volume rm '$NFS_VOLUME' >/dev/null 2>&1 || true"
  ssh "$host" "docker volume create --driver local \
    --opt type=nfs \
    --opt o=addr=${server_ip},nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600 \
    --opt device=:/ \
    '$NFS_VOLUME' >/dev/null"
  nfs_ok "Worker volume $NFS_VOLUME on ${host} → nfs://${server_ip}/"
}

nfs_stop_owned_server() {
  # Only the DSpark-owned container. Never docker rm vllm-fn-nfs — Qwen may
  # still be using that export.
  if docker ps -aq --filter "name=^/${NFS_CONTAINER}$" | grep -q .; then
    echo "Stopping NFS share ($NFS_CONTAINER) on head..."
    if timeout 15 docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1; then
      echo "  $NFS_CONTAINER: removed."
    else
      echo "  $NFS_CONTAINER: docker rm timed out or failed (kernel nfsd/rpcbind)." >&2
      return 1
    fi
  else
    echo "NFS share $NFS_CONTAINER is not running."
  fi
}
