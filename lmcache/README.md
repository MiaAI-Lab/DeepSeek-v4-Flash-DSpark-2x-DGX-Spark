# EXPERIMENTAL: LMCache KV offload for the 2× Spark pair

Persistent KV cache across engine restarts: a ~107K-token context that costs
~65 s to re-prefill reloads in **~1.8–1.9 s** (measured n=2 on this recipe,
GB10 pair, `nvfp4_ds_mla`). KV is held by per-node `lmcache server` processes
(L1 CPU + filesystem L2 on the local NVMe) that survive engine restarts.

**Status: experimental.** Serving-path verified (boot, store, warm hits,
reload) across repeated trials; the *failure paths* upstream are still being
hardened — see Operational risk and Known issues. Do not enable on a pair you
cannot restart.

## Operational risk — owner decision required

Enabling this is an explicit operational trade-off, not a free win. The
failure modes below are known and **not** fully mitigated in this option
directory; an owner has to accept them before turning the flag on.

**1. The cache servers live outside Compose.** `run-lmcache-server.sh` starts
one plain `docker run` container per node. Compose does not know about it, so
it is not started, stopped, restarted, health-checked or supervised with the
model containers, and `./stop-deepseek-v4-flash-dspark.sh` leaves it running.
Boot order is therefore manual and load-bearing: **every node's server must be
up and verified before the engine starts.** The engine's connector is
constructed at boot; a missing server at engine-boot time was not exercised in
this PR's testing, so treat it as unverified rather than as a soft failure.
The launcher prints its verification line only after the container is running
*and* the ZMQ port answers; anything else exits non-zero.

**2. Confirmed unified-memory cold-boot OOM.** On 128 GB unified-memory nodes
the kernel global OOM killer has killed the `lmcache server` process during an
engine **cold boot** — the weight-load spike lands on top of the server's
resident L1. Confirmed from `dmesg` on the head node (reported upstream on
LMCache #4759). *Symptom:* the server container is gone (or `docker ps` shows
it exited) while the engine boots or shortly after; from the engine's side
this then presents as case 3 below, not as an obvious error. *Recovery:*
restart the whole pair — servers first, then the engine. *Reductions:* size L1
so the server is not the fattest process on the board (`LMCACHE_L1_GB`), and
optionally set `LMCACHE_OOM_SCORE_ADJ` to a negative value (e.g. `-500`) so
the killer prefers the engine, which Compose restarts, over the server, which
nothing restarts. That knob defaults to `0` (kernel default, no change)
because it is a node-wide policy choice, not ours to make.

**3. A cache server that dies mid-serve takes the engine with it — silently.**
This is a **hang, not graceful degradation**. A server that is restarted or
killed loses its GPU contexts; the current upstream code then *parks every
lookup-hit request forever* — the scheduler heartbeat never starts, so dead
servers stay "healthy", and lookup errors do not propagate to the request.
Requests that would have hit the cache stop returning; the engine does not
crash and does not fall back to re-prefill, so an operator watching only for
container exits sees a healthy pair serving nothing. The only reliable signal
is the server's own exit plus `No GPU context found` in its log — which is why
the launcher uses `--restart no` (a dead server must stay dead and visible)
and will not recreate the server at all while a `vllm-dspark` model container
is running: the fresh-server-under-live-engine case IS this wedge, so the only
supported recovery is a full-pair restart (servers first, then the engine).
Once LMCache PR #4764 lands the parks become graceful degradation; until then,
**the blast radius of a dead cache server is the model service.**

**Owner decision:** turn this on only where a full pair restart is acceptable
at any moment, server exits are alerted on, and someone is on the hook to do
the restart. Otherwise leave `DSPARK_ENABLE_LMCACHE` unset. Moving the servers
into Compose would not fix 2 or 3 — Compose restarting an emptied server is
exactly the wedge in case 3 — so it is deliberately left out of scope until
the upstream heartbeat fix lands.

## How it fits this recipe

- One `lmcache server` container per node (see `run-lmcache-server.sh`),
  ZMQ on the fabric IPs. Outside Compose — see Operational risk.
- `patch-compose-lmcache.py` generates `docker-compose.lmcache.yml` from
  `docker-compose.dspark.yml`. **Every** LMCache-specific change sits inside
  one entrypoint branch taken only when `DSPARK_ENABLE_LMCACHE` is exactly
  `1`: `--kv-transfer-config` for `LMCacheMPConnector`, `PYTHONHASHSEED=0`,
  and `unset PYTORCH_CUDA_ALLOC_CONF` (vLLM rejects KV connectors alongside
  `expandable_segments:True`). The stock `PYTORCH_CUDA_ALLOC_CONF` service
  env entry is left untouched and no `PYTHONHASHSEED` env entry is added, so
  with the flag off the engine keeps stock allocator/hash behaviour and stock
  argv; the config still differs from stock by the two inert deltas below.
- Launch with
  `COMPOSE_FILE=$PWD/docker-compose.lmcache.yml ./start-deepseek-v4-flash-dspark.sh`.

## Requirements baked into a derived image (the pinned Anemll image lacks them)

```
pip install --no-deps lmcache==0.5.4
pip install sortedcontainers aiofile aiofiles cupy-cuda13x
```
`cupy` matters: without it the server *silently* fails GPU-context creation
and every engine registration kills the vLLM head (LMCache #4759 covers the
fail-fast ask). The lmcache wheel's bundled `cuda_ops` is ABI-mismatched
against this image's torch — it soft-falls back to torch ops (works; slower
stores). Building it from source against the image's torch works
(`TORCH_CUDA_ARCH_LIST=12.1a`; the image's CUDA toolkit is header-trimmed —
fill cusparse/cusolver/cufft headers from the `nvidia-*-cu13` pip wheels).

## Non-negotiable configuration

- **`PYTHONHASHSEED=0` on every process** (servers AND engine): chunk keys use
  Python's randomized `hash()`; without pinning, every restart invalidates the
  entire cache (LMCache #1788). `run-lmcache-server.sh` passes it to the
  server; the generated compose exports it into the engine, but only inside
  the `DSPARK_ENABLE_LMCACHE=1` branch — never on the stock path.
- **L1 sized to hold your largest context** (`--l1-size-gb 12` for ~150K-token
  docs) — mid-store L1 eviction has stalled stores in our testing.
- **Treat the pair + servers as one lifecycle unit.** Restarting a server
  loses its GPU contexts; the current upstream code then parks every
  lookup-hit request forever (scheduler heartbeat bug — fix upstream as
  LMCache PR #4764 — plus non-propagating lookup errors). Until those land:
  run servers with `--restart no`, alert on server exit, and on ANY
  `No GPU context found` in a server log, restart the whole pair.
- **Servers up before the engine, on every node.** The launcher exits non-zero
  unless its container is running and its port answers; do not proceed on a
  non-zero exit.

## Known issues (upstream)

- LMCache #4759 — the full GB10 field report (hang modes, evidence, stacks)
- LMCache PR #4764 — scheduler heartbeat never starts (dead servers stay
  "healthy" forever); merged = parks become graceful degradation
- LMCache PR #4754 — timeline-semaphore event IPC selector (defensive on
  driver 580/CUDA 13 platforms)
- Kernel global OOM kill of the server during engine **cold boot** on 128 GB
  unified-memory nodes (engine weight-load spike + resident L1) — confirmed
  from `dmesg`, reported on LMCache #4759. See Operational risk (2).

## Usage

On each node, from the recipe checkout (derived image already built):

```
# 1. one server per node, bound to THAT node's fabric IP. Both must print
#    their "lmcache server up ..." line BEFORE step 3 — the script preflights
#    the image (cupy/lmcache importable), the IP, and the L2 dir, then verifies
#    the container is running and the port answers. Non-zero exit = do not boot.
./lmcache/run-lmcache-server.sh 192.168.104.10        # head
./lmcache/run-lmcache-server.sh 192.168.104.11        # worker

# 2. generate the compose overlay (once; start-deepseek-v4-flash-dspark.sh
#    scp's $COMPOSE_FILE to the worker for you)
python3 lmcache/patch-compose-lmcache.py \
  docker-compose.dspark.yml docker-compose.lmcache.yml \
  tcp://192.168.104.10:6667,tcp://192.168.104.11:6667

# 3. enable + launch through the normal recipe entry point
export DSPARK_ENABLE_LMCACHE=1
COMPOSE_FILE=$PWD/docker-compose.lmcache.yml ./start-deepseek-v4-flash-dspark.sh
```

Knobs on `run-lmcache-server.sh`: `LMCACHE_DISK_DIR`, `LMCACHE_L1_GB`,
`LMCACHE_PORT`, `LMCACHE_OOM_SCORE_ADJ` (default `0`), and
`LMCACHE_FORCE_REPLACE=1` to override the guard against re-creating a
currently-running server. It applies only when the pair is actually down:
while any model container is live, server re-creation is refused
unconditionally (see Operational risk 3). "Model container" means either a
compose container labelled `com.docker.compose.service=vllm-dspark` or any
container whose name contains `vllm-dspark` (deliberately over-eager —
refusing too hard is the safe side of the wedge).

With `DSPARK_ENABLE_LMCACHE` unset, `0`, or anything other than exactly `1`,
the generated compose boots the stock recipe unchanged — every LMCache change
is gated at runtime, so one compose file serves both modes. Verify for
yourself:

```
diff <(docker compose -f docker-compose.dspark.yml config) \
     <(docker compose -f docker-compose.lmcache.yml config)
```

with the flag unset, the only deltas are inert: an added
`DSPARK_ENABLE_LMCACHE: "0"` pass-through, the `KVT_ARGS=""` gate, and an
empty `$${KVT_ARGS}` expansion in the serve argv (an empty unquoted expansion
adds no argument, so the exec'd argv is unchanged). The env is therefore not
literally byte-identical to stock — the inert `DSPARK_ENABLE_LMCACHE=0`
variable is added — but no stock behaviour depends on it.
`PYTORCH_CUDA_ALLOC_CONF` still renders as `expandable_segments:True` and
`PYTHONHASHSEED` is absent from the service env, exactly as on stock.

To verify it's working: the head engine log shows `LMCacheMPConnector`
at startup, and a repeated long-context request logs a lookup hit with TTFT
dropping from full-prefill cost to ~2 s.
