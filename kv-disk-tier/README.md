# `kv-disk-tier` — disk backed KV cache

Backs the prefix cache with NVMe so a large context survives GPU KV eviction and
restores from disk instead of paying a cold prefill. Multi-node TP safe (per-node
sharded tier) and needs no vLLM source change — modules are bind-mounted.

## Build (once per node)

```bash
KV_DISK_CACHE_SRC=/opt/dsv4-kv ./build.sh   # builds libdsv4_batch_copy.so + libdsv4_host_kv.so (sm_121a)
```

`libdsv4_batch_copy.so` is the scatter-gather copy kernel that replaces
`cuMemcpyBatchAsync` for large copies, which segfaults in the driver above ~23k
descriptors. `libdsv4_host_kv.so` backs the experimental `KV_DISK_CACHE_DIRECT_IO=1` path.
`build.sh` needs an image with `nvcc` — the serving image
(`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) has none, so a separate build image is
used (`IMG`).

## Stage on both nodes

The tier modules are bind-mounted from `KV_DISK_CACHE_SRC` (default
`/opt/dsv4-kv`), which must exist on **both** nodes with byte-identical Python
and `.so` artifacts. From a clean checkout:

```bash
KV_SRC=/opt/dsv4-kv
for h in spark1 spark2; do
  ssh "$h" mkdir -p "$KV_SRC"
  scp kv-disk-tier/{dsv4_kv_disk_tier,dsv4_shard_tier,dsv4_vllm_patches,dsv4_sitecustomize}.py \
      kv-disk-tier/{build.sh,gpu_clear.sh} "$h:$KV_SRC/"
done

# Build the CUDA kernels (sm_121a) on each node, or build once and scp the .so.
for h in spark1 spark2; do
  ssh "$h" "IMG=vllm-dspark-runtime:dspark-nvfp4-stage-c KV_SRC=$KV_SRC bash $KV_SRC/build.sh"
done

# Confirm both nodes received matching artifacts before booting.
for h in spark1 spark2; do ssh "$h" "md5sum $KV_SRC/dsv4_*.py $KV_SRC/libdsv4_*.so"; done
```

Pinned identities: serving image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
(vLLM `0.25.2.dev0+g752a3a504.d20260714`), build image
`vllm-dspark-runtime:dspark-nvfp4-stage-c`, model
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380`.
The patches target that exact vLLM build; a drifted source tree fails the
startup preflight rather than silently skipping a required patch.

## Enable

Set in `.env.dspark` (start syncs it to the worker):

```bash
KV_DISK_CACHE_ENABLE=1               # the on/off switch
KV_DISK_CACHE_SRC=/opt/dsv4-kv         # host dir with the modules + built .so (both nodes)
KV_DISK_CACHE_DIR=${HOME}/kvdisk       # per-node NVMe cache dir
KV_DISK_CACHE_CPU_BYTES=4294967296     # pinned CPU staging tier; caps largest restorable prompt
KV_DISK_CACHE_BYTES=150000000000       # per-node NVMe quota
```

`--kv-transfer-config` and the tier JSON are assembled from those knobs.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required and kept: the tier
exempts its own `OffloadingConnector` from vLLM's generic rejection, because
without expandable segments large prefills exhaust the driver's memdesc pool
(see the NVRM note in [Design notes](#design-notes)). `PYTHONHASHSEED=0` is
required (block hashes are salted from it). If a previous run wedged a GPU
(NCCL-stuck worker holding memory), `gpu_clear.sh` is a manual helper: it
removes the stale container, clears leftover `/dev/shm` staging, and waits for
`nvidia-smi` to report no compute apps before the next launch.

`KV_DISK_CACHE_CPU_BYTES` sizes the connector's **primary CPU tier** — the fast
restore tier. It is mandatory (vLLM's `OffloadingConnector` requires a primary
tier), and it determines how many evicted blocks stay in CPU RAM and restore
without touching NVMe versus spilling to disk. Blocks beyond this budget cascade
to the NVMe tier. Under `KV_DISK_CACHE_DIRECT_IO=1` the primary tier is still
allocated (the connector requires it) but its staging copy is bypassed.

## Disable

Remove `KV_DISK_CACHE_ENABLE` (or set it to `0`). Everything else is inert.

## Capacity

The offloaded block is one concatenated KV cell of 4,263,168 B (~4.06 MB) per
worker. With the shipped `dsv4_block_size_factor=4` geometry the
main MLA group covers **1024 tokens per offloaded block**. The CPU staging tier
is 503 blocks (4 GiB ≈ 515K tokens of the main group), and the per-node disk
quota (`KV_DISK_CACHE_BYTES=150000000000`) holds **~35,185 blocks ≈ 36M tokens**
of the main group — comfortably past a full 1M-token context. Raise
`KV_DISK_CACHE_BYTES` (with matching NVMe free space) to retain more.

## Direct I/O (experimental)

`KV_DISK_CACHE_DIRECT_IO=1` allocates the KV cache from `cudaHostAlloc` so the
disk tier can DMA straight into it, skipping the GPU↔CPU staging copy on store
and load. **It trades prefill throughput for a faster large-restore path.**
Measured (2026-09-05, 2× DGX Spark): **~8–20% slower prefill** (KV writes land
in host-pinned memory; the penalty grows with prompt length — ~8% at 2k tokens,
~20% at 64k), decode unchanged. In exchange, large restores skip one copy
(disk → host-KV instead of disk → staging → GPU). The staging tier it is meant
to replace is 4 GiB/node (`KV_DISK_CACHE_CPU_BYTES`); as shipped that tier is
still allocated (vLLM requires a primary tier) and its capacity bounds how many
disk blocks can be concurrently promoted, so the win is the I/O shortcut on
large restores — not a RAM saving and not a free win. **Off by default**;
enable it only for a workload dominated by very large restores. Byte-verification
of the direct round-trip: `KV_DISK_CACHE_DIRECT_VERIFY=1`.

## Optional tuning

- `KV_DISK_CACHE_SG_THRESHOLD=20000` — copies with ≥ this many descriptors use
  the scatter-gather kernel; smaller copies use the faster driver batch path
  (the driver segfaults above ~23k descriptors, so 20000 stays clear of the wall).
- `KV_DISK_CACHE_MAX_COPIES_PER_BATCH=8192` — bounds copies per launch with a
  per-slice stream sync.
- `KV_DISK_CACHE_MAX_OFFLOAD_BLOCKS_PER_REQUEST=0` — cap on offload keys a single
  request may store (`0` = unlimited). A prompt that fits in GPU KV doesn't need
  to spill to disk; set this to stop one huge request from flushing its whole
  prefix.

## Design notes

- **Expandable segments are required.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  collapses the per-allocation memdesc pressure that otherwise exhausts the
  NVIDIA driver's fixed-size pool (NVRM `_memdescAllocInternal`
  `NV_ERR_NO_MEMORY`) during large single-request prefills. vLLM rejects
  `expandable_segments:True` for every KV connector, but the
  `OffloadingConnector` used here copies GPU↔CPU staging with
  `cuMemcpyBatchAsync` and persists to NVMe — it never pins KV memory — so a
  `sitecustomize` hook exempts it from that check. Verified: ~750K-token fill
  OK (TTFT ~604 s, restore ~2.3 s), ~293K OK (TTFT ~134 s, restore ~1.0 s),
  ~149K OK (TTFT ~42 s), ~88K OK (TTFT ~20 s).

- **Disk writes happen in the copy handler.** The store path writes the cell
  file inside `transfer_async`, while the connector still pins the GPU blocks;
  the shard agent acks only after verifying each cell file exists with the right
  size, so a store can never race the block-free.

## Performance evidence

Reproduced on the two-node TP=2 lane (pinned image/model above, all other knobs
default). Cold prefill and decode via
`scripts/benchmark-0731.py --base-url http://127.0.0.1:8000/v1 --model default
--prompt-lengths 2048,8192,32768,65536 --concurrency 1 --max-tokens 1024`:

| config | prefill tok/s @ 2K / 8K / 32K / 64K | decode tok/s |
| --- | --- | --- |
| `KV_DISK_CACHE_DIRECT_IO=0` | 1612 / 1714 / 1777 / 1752 | 63 / 66 / 46 / 55 |
| `KV_DISK_CACHE_DIRECT_IO=1` | 1485 / 1586 / 1599 / 1394 | 63 / 58 / 58 / 58 |

Restore vs cold fill (same prompt, evicted between): a 12k-token prefix restores
from NVMe in ~1.9 s TTFT versus ~19 s cold fill; a buried needle is recalled
byte-identically on restore (`/tmp/recall_test.py`), and
`KV_DISK_CACHE_DIRECT_VERIFY=1` byte-checks each direct-I/O cell. These
prompts fit the CPU tier; the disk tier itself is exercised by the large-context
fills above (~750K-token fill → ~2.3 s restore), consistent with the
~36M-token disk capacity in [Capacity](#capacity).

