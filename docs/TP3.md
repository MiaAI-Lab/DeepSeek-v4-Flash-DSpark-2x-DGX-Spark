# Optional: 3× DGX Spark (TP=3)

Default serve is still **two nodes, TP=2**:

```bash
./start-deepseek-v4-flash-dspark.sh
```

Three Sparks use a separate launcher so a `TP_SIZE=3` line in `.env.dspark`
cannot silently change the 2-node path:

```bash
# in .env.dspark, in addition to the usual WORKER_HOST / fabric knobs:
WORKER2_HOST=10.0.0.3
WORKER2_VLLM_HOST_IP=10.0.0.3
# optional port/HCA overrides (default to WORKER_*):
# WORKER2_NCCL_IB_HCA=
# WORKER2_NCCL_SOCKET_IFNAME=
# WORKER2_HF_CACHE=
# WORKER2_DIR=

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
