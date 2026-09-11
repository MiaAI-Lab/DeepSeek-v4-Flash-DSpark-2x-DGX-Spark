#!/usr/bin/env bash
# check-nfs-export-compat.sh — LIVE evidence run for the hardened NFS export.
#
# The export hardening (root_squash, dropped `insecure`, fail-closed client
# list) raised two review questions that CPU tests cannot answer:
#
#   1. Compatibility — can a worker read an EXISTING, previously populated HF
#      cache through a root_squash export?
#   2. Security — is the 0600 `huggingface-cli` token file NOT readable
#      through it?
#
# This script answers both, and also exercises the fail-closed empty-client
# path, on the real two-node lane. Run it on the HEAD node, with the worker
# reachable, in a maintenance window. It starts or reuses the DSpark-owned
# exporter (dspark-nfs) and creates/recreates the worker NFS volume
# `dspark-hf`; it never stops a share it did not start.
#
# Usage:  bash scripts/check-nfs-export-compat.sh
# Exit:   0 = every expectation held; 1 = an expectation failed (details above)
#
# Output is meant to be attached to the PR as the compatibility evidence.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark}"
[ -f "$ENV_FILE" ] && { set -a; # shellcheck disable=SC1090
                       source "$ENV_FILE"; set +a; }

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
NFS_IMAGE="${NFS_IMAGE:-dspark-nfs:local}"
NFS_CONTAINER="${NFS_CONTAINER:-dspark-nfs}"
NFS_VOLUME="${NFS_VOLUME:-dspark-hf}"
NFS_PROBE_IMAGE="${NFS_PROBE_IMAGE:-alpine:3.20}"
HF_CACHE_DIR="${HF_CACHE:-$HOME/.cache/huggingface}"
IFACE="${NFS_IFACE:-${NCCL_SOCKET_IFNAME:-}}"
DSPARK_MODEL="${DSPARK_MODEL:-${DSPARK_MODEL_OFFICIAL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}}"
MODEL_REL="hub/models--$(printf '%s' "$DSPARK_MODEL" | sed 's|/|--|g')"
WORKER_API_KEY_TOKEN_REL="token"

FAILURES=0
ok()  { echo "  PASS: $*"; }
bad() { echo "  FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }
info() { echo "  .. $*"; }

echo "== NFS export compatibility evidence =="
echo "head cache: $HF_CACHE_DIR"
echo "worker:     $WORKER_HOST (volume $NFS_VOLUME)"
echo "model dir:  $MODEL_REL"
echo

# ── 1. Fail-closed: an empty client list must refuse to start ───────────────
echo "== 1. empty NFS_CLIENTS must fail closed =="
if ! docker image inspect "$NFS_IMAGE" >/dev/null 2>&1; then
  info "building $NFS_IMAGE"
  docker build -q -t "$NFS_IMAGE" "$SCRIPT_DIR/files/nfs-server" >/dev/null
fi
set +e
_out="$(docker run --rm --entrypoint /entrypoint.sh -e NFS_CLIENTS= "$NFS_IMAGE" 2>&1)"
_rc=$?
set -e
if [ "$_rc" -ne 0 ] && printf '%s' "$_out" | grep -q "refusing to export to"; then
  ok "empty client list refused (exit $_rc, message present)"
else
  bad "empty client list did NOT fail closed (exit $_rc): $_out"
fi

if [ -z "${NFS_SERVER_IP:-}" ] && [ -z "$IFACE" ]; then
  bad "set NFS_SERVER_IP or NFS_IFACE/NCCL_SOCKET_IFNAME to run the live checks"
  echo; echo "evidence run incomplete: $FAILURES failure(s)"; exit 1
fi

# ── 2. Start/reuse the exporter (repo path) ────────────────────────────────
echo
echo "== 2. exporter up with the hardened options =="
export HF_CACHE_DIR
# The share functions expect these names, as in the launcher.
host_without_user() { local h="$1"; printf '%s' "${h##*@}"; }
# shellcheck source=files/nfs-share.sh
source "$SCRIPT_DIR/files/nfs-share.sh"
if nfs_ensure_server >/dev/null 2>&1; then
  ok "exporter up on ${NFS_SERVER_IP:-$IFACE}:2049"
else
  bad "nfs_ensure_server failed; see: docker logs $NFS_CONTAINER"
fi
_opts="$(docker exec "$NFS_CONTAINER" sh -c 'grep -o "([^)]*)" /etc/exports | head -1' 2>/dev/null || true)"
case "$_opts" in
  *root_squash*) ok "live exports carry root_squash: $_opts" ;;
  *) bad "live exports do not show root_squash: ${_opts:-<unreadable>}" ;;
esac

# ── 3. Worker mounts the export ────────────────────────────────────────────
echo
echo "== 3. worker NFS volume =="
if nfs_ensure_worker_volume recreate >/dev/null 2>&1; then
  ok "worker volume $NFS_VOLUME (re)created"
else
  bad "could not create the worker volume"
fi

probe() { # probe <shell test> -> rc of the worker-side container
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
    "docker run --rm -v '$NFS_VOLUME:/hf:ro' '$NFS_PROBE_IMAGE' sh -c '$1'" >/dev/null 2>&1
}

# ── 4. Compatibility: the existing populated cache is readable ─────────────
echo
echo "== 4. compatibility: existing cache readable through the squash =="
if probe "test -d /hf/$MODEL_REL"; then
  ok "model directory visible: /$MODEL_REL"
else
  bad "model directory NOT visible: /$MODEL_REL — worker loads would fail"
fi
if probe "test -r /hf/$MODEL_REL/config.json"; then
  ok "model file readable: config.json (world-readable blobs pass the squash)"
else
  bad "model file NOT readable: config.json — cache files are not world-readable; chmod a+r the cache (or pin NFS_OPTS without root_squash) before serving"
fi

# ── 5. Security: the token file is NOT readable ────────────────────────────
echo
echo "== 5. security: HF token not readable through the squash =="
_mode="$(stat -c %a "$HF_CACHE_DIR/$WORKER_API_KEY_TOKEN_REL" 2>/dev/null || echo '')"
if [ -z "$_mode" ]; then
  info "no $HF_CACHE_DIR/$WORKER_API_KEY_TOKEN_REL present — nothing to prove"
elif probe "test -r /hf/$WORKER_API_KEY_TOKEN_REL"; then
  if [ "$_mode" = "600" ]; then
    bad "token is mode $_mode but READABLE from the worker — root_squash not in effect"
  else
    bad "token is mode $_mode (world-readable); root_squash cannot protect a public file — chmod 600 it"
  fi
else
  ok "token (mode $_mode) is NOT readable from the worker (root_squash maps worker root to nobody)"
fi

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "evidence run: $FAILURES expectation(s) failed"
  exit 1
fi
echo "evidence run: all expectations held (attach this output to the PR)"
