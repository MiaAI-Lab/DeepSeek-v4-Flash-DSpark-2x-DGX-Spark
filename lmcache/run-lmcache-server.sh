#!/usr/bin/env bash
# Launch the per-node LMCache MP server. Run once per node with that node's
# fabric IP. Requires the derived image (see README.md).
#
# Boot order is load-bearing: BOTH servers must be up and verified here BEFORE
# the engine starts (see README, "Operational risk"). This script fails loudly
# rather than leaving a half-up server behind a green exit code.
set -euo pipefail
FABRIC_IP="${1:?usage: run-lmcache-server.sh <this-node-fabric-ip> [image]}"
IMAGE="${2:-dspark-vllm-gx10:lmcache054}"
DISK="${LMCACHE_DISK_DIR:-$HOME/lmcache-disk}"
PORT="${LMCACHE_PORT:-6667}"
# 0 = kernel default (Docker's default too), so this is a no-op unless set.
# A negative value biases the OOM killer away from the cache server on 128 GB
# unified-memory boards; see README, "Operational risk" — it makes the engine
# the likelier victim instead, which is the recoverable failure of the two.
OOM_SCORE_ADJ="${LMCACHE_OOM_SCORE_ADJ:-0}"

die() { echo "error: $*" >&2; exit 1; }

# --- preflight (fail before we touch a running cache) -----------------------
command -v docker >/dev/null 2>&1 || die "docker not found on PATH"
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "image '$IMAGE' not present locally; build the derived image first (README)"

# The server must bind THIS node's fabric IP; a typo binds nothing and the
# engine's lookups then go to a socket that never answers.
if command -v ip >/dev/null 2>&1; then
  ip -o -4 addr show 2>/dev/null | grep -qw "$FABRIC_IP" \
    || die "$FABRIC_IP is not bound on this host; pass THIS node's fabric IP"
fi

# cupy is load-bearing: without it the server silently fails GPU-context
# creation and every engine registration kills the vLLM head (LMCache #4759).
docker run --rm --entrypoint python3 "$IMAGE" -c 'import cupy, lmcache' >/dev/null 2>&1 \
  || die "image '$IMAGE' cannot import cupy and lmcache; see README requirements"

mkdir -p "$DISK" || die "cannot create L2 dir $DISK"
[ -w "$DISK" ] || die "L2 dir $DISK is not writable"

# Replacing a cache server under a LIVE engine is the documented wedge
# condition, so two guards stand between here and the docker rm below:
#
# Guard 1 — the model container. The real-world failure mode is a DEAD server
# under a LIVE (wedged) engine (README 'Operational risk' 2+3); proceeding
# there would rm + recreate an EMPTY server against that engine — the wedge
# itself. So this refusal has NO override: LMCACHE_FORCE_REPLACE is documented
# to mean "the pair is already down", and a live model container falsifies
# that claim. Stop the pair first; the recovery is a full-pair restart.
# Fail CLOSED: a docker-ps failure here must not be read as "nothing is
# running" — that would take the guard down exactly when the host is sick.
if ! model_ps_a="$(docker ps -q -f 'name=vllm-dspark' 2>&1)"; then
  die "docker ps (name filter) failed — refusing to recreate the cache server without trustworthy state: $model_ps_a"
fi
if ! model_ps_b="$(docker ps -q -f 'label=com.docker.compose.service=vllm-dspark' 2>&1)"; then
  die "docker ps (compose service label filter) failed — refusing to recreate the cache server without trustworthy state: $model_ps_b"
fi
# NOTE: the name filter is a substring match on purpose. It is strictly
# over-eager (any container whose name contains vllm-dspark) and it also
# catches hand-run engine containers that carry no compose labels. Refusing
# too hard is the safe side of the wedge; the label filter covers the
# compose-managed case precisely.
if [ -n "$model_ps_a" ] || [ -n "$model_ps_b" ]; then
  die "a model (vllm-dspark) container is RUNNING. Recreating the lmcache server
       under a live engine is the documented wedge (README 'Operational risk'):
       the fresh server has no GPU contexts and the engine parks every lookup
       hit forever. Stop the whole pair first
       (./stop-deepseek-v4-flash-dspark.sh), then re-run this script.
       LMCACHE_FORCE_REPLACE does NOT bypass this guard."
fi

# Guard 2 — a still-running (possibly poisoned) server. Overridable only for
# the legitimate pair-is-down case: stale/empty server, no engine running.
if ! server_ps="$(docker ps -q -f 'name=^lmcache-server$' 2>&1)"; then
  die "docker ps (server filter) failed — refusing to recreate the cache server without trustworthy state: $server_ps"
fi
if [ -n "$server_ps" ]; then
  [ "${LMCACHE_FORCE_REPLACE:-0}" = "1" ] \
    || die "lmcache-server is already RUNNING and no model container is up.
       Only re-create it with LMCACHE_FORCE_REPLACE=1 if the pair is already
       down (e.g. stale server from an aborted boot)."
fi
docker rm -f lmcache-server >/dev/null 2>&1 || true

# --restart no is deliberate: a dead server must be VISIBLE (see README —
# an auto-restarted empty server currently wedges the engine silently).
docker run -d --name lmcache-server --network host --ipc host --gpus all \
  --restart no \
  --oom-score-adj "$OOM_SCORE_ADJ" \
  -e PYTHONHASHSEED=0 \
  -v "$DISK:/lmcache-disk" \
  --entrypoint lmcache "$IMAGE" server \
  --host "$FABRIC_IP" --port "$PORT" --chunk-size 256 \
  --l1-size-gb "${LMCACHE_L1_GB:-12}" --l1-use-lazy --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"/lmcache-disk"}' \
  --disable-observability >/dev/null

# --- verify the resulting state, not the exit code of docker run -----------
for _ in $(seq 1 30); do
  [ -n "$(docker ps -q -f name="^lmcache-server$")" ] || break
  if (exec 3<>"/dev/tcp/${FABRIC_IP}/${PORT}") 2>/dev/null; then
    exec 3<&- 2>/dev/null || true
    echo "lmcache server up on ${FABRIC_IP}:${PORT} (L1 ${LMCACHE_L1_GB:-12}G, fs L2 at $DISK)"
    echo "NOTE: start the engine only after EVERY node reports this line."
    exit 0
  fi
  sleep 1
done
echo "--- lmcache-server logs ---" >&2
docker logs --tail=50 lmcache-server >&2 2>&1 || true
die "lmcache-server did not come up listening on ${FABRIC_IP}:${PORT}"
