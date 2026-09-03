# Optional: 3× DGX Spark (TP=3)

Default serve is still **two nodes, TP=2**:

```bash
./start-deepseek-v4-flash-dspark.sh
```

Three Sparks use a separate launcher so a `TP_SIZE=3` line in `.env.dspark`
cannot silently change the 2-node path.

## Prerequisites on the third Spark (the launcher checks, but does not do, these)

- **Passwordless SSH** from the head to `WORKER2_HOST` (`ssh -o BatchMode=yes <host> true`).
- **The pinned vLLM image already pulled** there. The launcher refuses to boot
  and prints the exact command; run it yourself first (≈ 19 GB on disk):
  `ssh <worker2> docker pull "$DSPARK_VLLM_IMAGE"` with the value from `.env.dspark`.
- **A ConnectX link from the head to spark3** with its own `/24` (e.g. head
  `10.0.23.1` ↔ spark3 `10.0.23.3`), and a LAN interface reachable from all
  three nodes for the bootstrap (default `enP7s7`).
- Nothing else: the launcher creates `WORKER2_DIR`, syncs compose, env,
  `patches/` and `patches/tp3/`, and (with `DSPARK_WORKER_HF_NFS=1`) mounts
  the head's HF cache over NFS, so spark3 needs no local checkpoint.

```bash
# in .env.dspark, in addition to the usual WORKER_HOST / fabric knobs:
WORKER2_HOST=10.0.0.3
WORKER2_VLLM_HOST_IP=10.0.0.3
WORKER2_NFS_SERVER_IP=10.0.23.1        # head IP on the spark1<->spark3 link (not 10.0.22.1)
# spark3's port facing the head (do NOT copy WORKER_NCCL_*):
WORKER2_NCCL_IB_HCA=rocep1s0f1
WORKER2_NCCL_SOCKET_IFNAME=enp1s0f1np1
WORKER2_TP_SOCKET_IFNAME=enp1s0f1np1
WORKER2_GLOO_SOCKET_IFNAME=enp1s0f1np1
# optional (default to WORKER_*):
# WORKER2_HF_CACHE=
# WORKER2_DIR=
# TP3_MAX_NUM_SEQS=16

./prepare-dspark-model-cache.sh --yes   # also copies weights to WORKER2 when NFS=0
./start-tp3.sh                         # or: ./start-tp3.sh --max-num-seqs 16
scripts/validate_tp3.sh 127.0.0.1:8888
```

On a **QSFP ring**, spark3 is not on the spark1↔spark2 NFS subnet. Worker 1
mounts `NFS_SERVER_IP` (often `10.0.22.1`). Worker 2 must use the head address
on the spark1↔spark3 link (`WORKER2_NFS_SERVER_IP`, e.g. `10.0.23.1`) and that
`/24` must be in the live exporter's clients (start adds it when it can).
Pin `WORKER2_NCCL_*` to spark3's facing port toward spark1, not a copy of
`WORKER_NCCL_*`. Start then moves **Gloo / NCCL socket / TP TCP** onto
`TP3_BOOTSTRAP_IFNAME` (default `enP7s7` / 192.168.1.0/24). Do not use `lo`:
Gloo binds `127.0.0.1` and the mesh fails. `NCCL_IB_HCA` is then set to
**both** CX ports (`rocep1s0f0,rocep1s0f1`) so spark1 can reach spark2 on
`10.0.22.0/24` and spark3 on `10.0.23.0/24` (a single facing HCA times out
in QP RTR). Override with `TP3_NCCL_IB_HCA` if the roce names differ.
Start also sets `NCCL_IB_MERGE_NICS=0`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, and
`NCCL_IB_SUBNET_PREFIX_LEN=24` so NCCL pairs GIDs per `/24` (default `/16`
would treat `10.0.22` and `10.0.23` as one subnet).

**First boot is slow.** Spark3 starts with an empty JIT cache and the TP=3
shard shapes are new for all three nodes, so the first `start-tp3.sh` spends
several minutes compiling (≈ 8 min to healthy on the reference pair, engine
init alone 193 s) and the first few requests still hit Triton JIT spikes. Do
not restart a boot that is merely compiling; wait for `HEALTH_OK` /
`boot-shape-warmup`. Later boots reuse the persisted caches.

Stop is the same script; if `WORKER2_HOST` is set it tears down rank 2 as well:

```bash
./stop-deepseek-v4-flash-dspark.sh
```

## Why a patch is required

V4-Flash has **8 attention output groups**. TP must divide that count. 2 does;
3 does not. Stock vLLM either refuses to start (`64 % 3 != 0`) or **silently**
keeps `8 // 3 == 2` local groups and drops the rest.

The fix (from [localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark))
pads **groups 8→9**, keeps **heads-per-group = 8**, and zero-fills / `-inf`-fills
the pad lanes. The entrypoint runs `patches/tp3/apply_tp3_patch.py` inside the
container only when `TP_SIZE=3`.

Do not pad heads inside a group: that changes the o_proj BMM `r=4096` contract
and DeepGEMM rejects it.

## Recreate after patch edits

The patcher writes into the container layer. `docker compose` restart reuses
that layer; a stale marker then fails closed (`STALE`). After changing
`patches/tp3/`:

```bash
./stop-deepseek-v4-flash-dspark.sh
./start-tp3.sh
```

`./stop` already `docker rm -f` the vLLM containers.

## Concurrency

The 2-node recipe stays at `MAX_NUM_SEQS` (default `6`). TP=3 slots:

```bash
./start-tp3.sh --max-num-seqs 16
# or in .env.dspark (ignored by ./start-deepseek-v4-flash-dspark.sh):
# TP3_MAX_NUM_SEQS=16
```

CLI wins over `TP3_MAX_NUM_SEQS`. CUDA-graph capture size is
`MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)` rounded up to a multiple of 8 (48 at
6 slots). If capture OOMs, lower
`GPU_MEMORY_UTILIZATION_TEXT` toward `0.78` rather than cutting
`MAX_MODEL_LEN` first. Needs a recreate (`./stop-… && ./start-tp3.sh`).

## What to expect vs the 2-node lane

Warm measurements on the reference cluster (2026-09-02, same image, prose
prompts, 512 output tokens; TP=3 at 16 slots, TP=2 at 6):

| Load | TP=2 | TP=3 |
|---|---|---|
| decode, 1 stream | 38.6 tok/s | 41.9 tok/s (+9 %) |
| decode, 2 streams | 63.9 agg | 72.0 agg (+13 %) |
| decode, 4 streams | 86.0 agg | 89.7 agg (+4 %) |
| decode, 8 / 12 / 16 streams | n/a (6 slots) | 142.8 / 164.3 / 198.0 agg |
| TTFT 8k / 16k / 32k / 64k | 4.4 / 8.7 / 18.0 / 36.0 s | 5.0 / 9.4 / 18.6 / 38.2 s (+4–13 %) |
| TTFT 128k / 256k | 74.9 / 164.8 s | 91.5 / 202.2 s (+22–23 %) |
| KV cache per rank | ≈ 16.8 / 15 GiB | ≈ 35 / 34 / 35 GiB (5.0 M tokens, 4.8× at 1M) |

So TP=3 buys capacity (16 slots, ≈ 200 tok/s aggregate, three times the KV)
and slightly faster decode per stream, and costs prefill latency, most at
≥ 128k. In DeepSeek's MLA every rank holds the full latent KV and the indexer
is replicated, so the attention share of prefill does not shrink with a third
GPU; the pad (heads 64 → 72) and the three-node all-reduce add on top.
Single-user long-context work is better served by the 2-node lane.
