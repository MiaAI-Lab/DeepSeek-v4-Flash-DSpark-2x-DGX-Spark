# `kv-disk-tier` — disk backed KV cache

Backs the prefix cache with NVMe so a large context survives GPU KV eviction and
restores from disk instead of paying a cold prefill. Multi-node TP safe (per-node
sharded tier) and needs no vLLM source change — modules are bind-mounted.

## Build (once per node)

```bash
KV_SRC=/opt/dsv4-kv ./build.sh   # builds libdsv4_batch_copy.so + libdsv4_host_kv.so (sm_121a)
```

`libdsv4_batch_copy.so` is the scatter-gather copy kernel that replaces
`cuMemcpyBatchAsync` for large copies, which segfaults in the driver above ~23k
descriptors. `libdsv4_host_kv.so` backs the experimental `DSV4_HOST_KV=1` path.
`build.sh` needs an image with `nvcc` — the serving image
(`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`) has none, so a separate build image is
used (`IMG`).

## Enable

Set in `.env.dspark` (start syncs it to the worker):

```bash
DSPARK_ENABLE_DISK_TIER=1      # the on/off switch
KV_SRC=/opt/dsv4-kv            # host dir with the modules + built .so (both nodes)
KVDISK_DIR=${HOME}/kvdisk      # per-node NVMe cache dir
KV_CPU_BYTES=4294967296        # pinned CPU staging tier; caps largest restorable prompt
KV_DISK_BYTES=150000000000     # per-node NVMe quota
```

`--kv-transfer-config` and the tier JSON are assembled from those knobs, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is kept by default (the tier
exempts its own `OffloadingConnector` from vLLM's generic rejection; set
`DSV4_ALLOW_EXPANDABLE_SEGMENTS=0` to unset it — see below).
`PYTHONHASHSEED=0` is required (block hashes are salted from it). `gpu_clear.sh`
clears a stale container and leftover `/dev/shm` staging before a relaunch.

## Disable

Remove `DSPARK_ENABLE_DISK_TIER` (or set it to `0`). Everything else is inert.

## Capacity

The offloaded block is one concatenated KV cell. With the shipped
`dsv4_block_size_factor=4` geometry the cell is 272,842,752 B (~260 MB) and the
main MLA group covers **1024 tokens per offloaded block**. The CPU staging tier
is 7 blocks (4 GiB), and the per-node disk quota (`KV_DISK_BYTES=150000000000`)
holds **~549 blocks ≈ 562K tokens** of the main group — not a full 1M. A full
1M-token prompt cannot be disk-cached at the default geometry; raise
`KV_DISK_BYTES` (with matching NVMe free space) to retain more.

## Optional tuning

- `DSV4_SG_THRESHOLD=20000` — copies with ≥ this many descriptors use the
  scatter-gather kernel; smaller copies use the faster driver batch path (the
  driver segfaults above ~23k descriptors, so 20000 stays clear of the wall).
- `DSV4_MAX_COPIES_PER_BATCH=8192` — bounds copies per launch with a per-slice
  stream sync.
- `DSV4_MAX_OFFLOAD_BLOCKS_PER_REQUEST=0` — cap on offload keys a single request
  may store (`0` = unlimited). A prompt that fits in GPU KV doesn't need to
  spill to disk; set this to stop one huge request from flushing its whole
  prefix.(default) — keep `expandable_segments:True`
  by exempting the `OffloadingConnector` from vLLM's generic KV-connector check
  (the connector does not pin KV memory, so expandable segments are safe). This
  collapses GPU memdesc pressure during prefill and lifts the large-context
  ceiling — see the NVRM OOM note below. Set `0` to unset
  `PYTORCH_CUDA_ALLOC_CONF` (the old behaviour) prefill and lifts the large-context
  ceiling — see the NVRM OOM note below.

## Recent fixes

- **Store/load race (crash).** `_sliding_window_lookup_patched` now uses the
  vLLM 0.25.x `LookupResult` enum (`HIT` / `HIT_PENDING` / `RETRY` / `MISS`)
  and mirrors the stock function exactly. The previous code used the old
  bool/`None` contract, so *every* lookup was counted as a hit — including
  mid-store `HIT_PENDING` blocks — and `update_state_after_alloc` handed them
  to `prepare_load()`, whose stock `assert block.is_ready` killed the
  EngineCore on any two requests sharing a prefix. `HIT_PENDING`/`RETRY` now
  defer the lookup instead; a deferred streak returns `None` (no load issued),
  so no preemptive extra lookup is needed.

- **Large-copy chunking + SG threshold.** Batches ≥ `DSV4_SG_THRESHOLD`
  (default `20000`) descriptors go through the scatter-gather kernel instead of
  `cuMemcpyBatchAsync` (which segfaults above ~23k); batches larger than
  `DSV4_MAX_COPIES_PER_BATCH` (default `8192`) are sliced with a per-slice
  stream sync. Smaller copies use the faster driver path directly.

- **Eager-offload budget.** `DSV4_MAX_OFFLOAD_BLOCKS_PER_REQUEST` (default `0` =
  unlimited) caps how many offload keys one request may store, so an in-GPU
  prompt doesn't spill its whole prefix to disk.
by default

Large single-request contexts trigger a GPU-driver memory-descriptor exhaustion
(`NVRM: nvCheckOkFailedNoLog … _memdescAllocInternal … NV_ERR_NO_MEMORY`), which
is *not* host RAM and *not* fixed by the chunking above. Root cause: the disk
tier (via vLLM's generic KV-connector check) forced `PYTORCH_CUDA_ALLOC_CONF`
empty, so the whole engine ran with `expandable_segments` off — each prefill
allocation then created a burst of memdescs that exhausted the driver pool.

The tier now exempts the `OffloadingConnector` from that rejection by default
(it does not pin KV memory, so expandable segments are safe), keeping
`expandable_segments:True` across the engine. Verified on this fleet — all
single-request fills, no engine. Verified on this fleet with the knob on — all single-request fills, no
driver OOM, engine hang, or crash:

- ~750K-token fill → OK (TTFT ~604 s), restore ~2.3 s.
- ~293K-token fill → OK (TTFT ~134 s), restore ~1.0 s.
- ~149K-token fill → OK (TTFT ~42 s).
- ~88K-token fill → OK (TTFT ~20 s).

With the knob off: ~293K hung the engine and ~750K crashed the container. The
fix collapses the memdesc pressure, so the driver-level ceiling is no longer the
binding limit. The disk *quota* still caps cacheable tokens at ~562K; a larger
prompt prefills fine but its blocks beyond the quota are simply not stored
(`cannot store blocks` warnings).
