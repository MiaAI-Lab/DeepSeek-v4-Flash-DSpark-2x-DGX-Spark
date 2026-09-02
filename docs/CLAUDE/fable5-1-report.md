# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark — decode / prefill / concurrency / stability audit

**Author:** Claude (Fable 5.1) · **Date:** 2026-09-02 · **Repo head:** `8494a49` (main)
**Method:** static read of the repo (launcher, compose, patches, docs, prior audits), the in-container vLLM /
b12x / FlashInfer sources on the live image, host and fabric inspection on **both** nodes (read-only), the
live boot log, `/metrics`, and one short decode probe with the repo's own `scripts/bench-miaai.py`
(c=1 and c=6, 256-token prompts, 3 trials each). Nothing was changed, restarted or committed.

> **Important live caveat.** At audit time the cluster was running the **TP=3** experiment
> (`./start-tp3.sh`: `tensor_parallel_size=3`, `max_num_seqs=16`, heads padded 64→72). This report
> targets the **default TP=2 lane** (`./start-deepseek-v4-flash-dspark.sh`), using the Aug-9 and
> Aug-19 TP=2 boot logs in `results/` for TP=2-specific facts, and uses the live TP=3 measurements
> where they illuminate the general mechanics. Every finding says which lane it applies to.

---

## 0. Executive summary

The stack is healthy and unusually well engineered (fail-closed idempotent hotfixes, persisted JIT
caches, per-node GID resolution, honest results tables). Decode at c=1 is fundamentally
**weight-streaming-bound** (≈15–17 GB of expert/attention bytes per rank per step at TP=2), so no knob
doubles c=1 tok/s. But there is a real, addressable **fixed per-step cost** (≈108 NCCL collectives per
decode step, an uncaptured c=6 CUDA-graph shape, a 12-collective serialized draft loop), a set of
**host/fabric misconfigurations** that cost KV capacity and stability today, and two **kernel-level
gaps** (no FP4 tensor-core MoE path for this checkpoint; a replicated Lightning indexer) that bound
prefill. The live TP=3 result (c=1 ≈ 64 tok/s, no faster than TP=2) is direct evidence that the fixed
per-step costs — not bandwidth — decide c=1 once bytes/rank shrink.

### The ten things that matter (ranked by expected value ÷ effort)

| # | Finding | Lane | Class | Expected effect | Effort / restart |
|---|---|---|---|---|---|
| 1 | **Unified memory is exhausted at boot on both nodes** (`NVRM … NV_ERR_NO_MEMORY` 12:59–13:00 today; worker 10.4 GB swap used, `polkitd` 3.1 GB RSS leak, GNOME+Xorg holding GPU contexts, `MemAvailable` 5.4 GB). Worker KV pool is already the pair's minimum (33.82 vs 35.01 GiB). | both | stability + KV capacity | reclaim ~5–8 GB host RAM → then raise `GPU_MEMORY_UTILIZATION_TEXT` 0.83→0.86–0.88 = **+20–35 % KV tokens**, fewer stalls | host-side, no engine change (then one restart for util) |
| 2 | **c=6 decode shape is not CUDA-graph captured on TP=2**: argv `--max-cudagraph-capture-size 42` is truncated to 40, then rounded to multiples of 7 → FULL graphs `[7,14,21,28,35]`; the 42-token c=6 step runs eager. The Aug-19 A/B (k=5, max 32) never captured its c=6 shape either (36 > 32), so its "no lift" verdict does not settle this. | TP=2 | config | c=6 aggregate **+3–10 %** (est.) | one line in compose; restart |
| 3 | **NCCL is pinned to one ConnectX-7 PCIe function, and each PF is Gen5 ×4 (~15.7 GB/s ≈ 126 Gb/s)**. The 200 Gb/s port is only reachable at full rate through both PFs (socket-direct: `rocep1s0f1`+`roceP2p1s0f1`). RoCE MTU is 1024 (Ethernet MTU 1500), RX/TX rings 1024, and the second PF carries a duplicate `10.0.22.0/24` IP on the same wire (ARP-flux candidate). Zero RoCE retransmits, but pause frames and `rx_out_of_buffer` are non-zero. | both | fabric | prefill comm bandwidth up to ~2×; large all-gathers (logits, 10 MB/step at c=6) faster; decode latency ≈ unchanged | host + `.env` HCA lists; restart |
| 4 | **DSpark Markov head is TP-sharded** (`VocabParallelEmbedding` + `ParallelLMHead`) → the 6-step sequential draft loop does **12 serialized collectives per decode step** (all-reduce embed + all-gather bias each step). Stage-C's `VLLM_DSPARK_REPLICATE_MARKOV_W1` existed for exactly this; Anemll 0.1.1 does not have it. Both matrices are 66 MB — replication is free. | both | vLLM patch | **−12 collectives/step** (~0.5–1.5 ms/step; 1–3 %, more at TP=3) | ~60-line patch in `nvidia/dspark.py`/`qwen3_dspark.py`; restart |
| 5 | **B12X runs W4A16 (dequant→bf16 MMA) for everything** because the checkpoint's `fp4_e8m0_k32` (MXFP4/E8M0) source is not accepted by b12x's NVFP4 tensor-core path (modelopt-only). Prefill MoE is therefore compute-inefficient on Blackwell tensor cores. Also `B12X_W4A16_TC_DECODE` (small-M packed decode, m ≤ 8 = exactly c=1) is available and untested. | both | kernel | TC-decode: 0–10 % c=1 (A/B). MXFP4×MXFP8 MoE: prefill MoE 2–3× → **prefill +30–60 %** short/medium ctx (engineering) | env A/B now; kernel work later |
| 6 | **Lightning indexer is replicated across TP ranks** (`wq_b`, `weights_proj` are `ReplicatedLinear`; each rank scores all 64 index heads against the full compressed context). At ≥128K context the indexer dominates prefill (900K prefill = 875 tok/s). | both | engineering | sequence-parallel indexer + distributed top-k merge → **long-ctx prefill up to ~1.8×**, frees ~1.5 GB/rank replicated indexer cache | multi-day kernel/plumbing work |
| 7 | **Prefill chunk cap 1024 is static.** With no decode lanes active a c=1 prefill still pays per-chunk fixed cost (86 all-reduces, indexer/compressor setup, ~1.5k launches) 8× per 8192 tokens. | both | scheduler patch | c=1 TTFT **−10–25 %** on ≤32K prompts; #27/#43 fairness unchanged when decodes exist | ~30-line patch next to `hotfix-dsv4-issue27` |
| 8 | **`nvfp4_ds_mla` is byte-identical to `fp8_ds_mla`** (584 B/row: 448 B fp8 NoPE + 128 B bf16 RoPE + 8 B scale). There is no 4-bit KV today. A true NVFP4 row (~390 B) would add ~50 % KV tokens. | both | upstream/kernel | +50 % KV (future) | FlashInfer SM120 kernel work |
| 9 | **Live `.env.dspark` still has `DSPARK_MAX_INFLIGHT_PREFILLS=2`** — the setting `#154` reverted to `1` on 2026-08-28 after measuring 3.7–5.1× fairness spread vs 1.7–2.0× at `1`. | TP=2/3 | config | restores mixed-request fairness (or keep 2 knowingly for 32K×c4 decode floor 8→25 tok/s) | `.env`; restart |
| 10 | **Boot warmup gaps**: FlashInfer sparse-MLA autotune warms only `tokens=16`; CuTeDSL warmup is skipped ("no compile units were requested"); first requests of this boot still JIT'd `_prepare_dflash_inputs_kernel` and `_topk_topp_kernel`. | both | launcher/config | removes first-request latency cliffs; closes the mid-serve-JIT → TP-watchdog hazard (#117) | script changes; restart |

What is **not** worth chasing (evidence in §3): raising `MAX_NUM_BATCHED_TOKENS` for the 8150/8168
warning (tested, no gain), `MTP_NUM_TOKENS` beyond 6 (per-position acceptance 0.28/0.21 at positions
5–6 already), `GPU_MEMORY_UTILIZATION` before fixing host memory (item 1), TP=3 for c=1 speed (§2.3).

---

## 1. Live snapshot (2026-09-02, both nodes)

| Item | Head `spark1` | Worker `spark2` |
|---|---|---|
| Image / vLLM | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a8394849…`, vLLM `0.25.2.dev0+g752a3a504.d20260714`, torch 2.11.0+cu130, CUDA 13.0, driver 580.159.03 | same |
| Kernel libs | flashinfer-python 0.6.15 (cubin/jit-cache 0.6.13), b12x 0.15.3, triton 3.6.0 (+tokenspeed-triton 3.8.10), tilelang 0.1.9, nvidia-nccl-cu13 2.30.7 (pynccl) / torch-bundled NCCL 2.28.9, xgrammar 0.2.3 | same |
| Live lane | **TP=3**, `max_num_seqs 16`, capture 112, FULL graphs 15, PIECEWISE 17, weights 60.69 GiB/rank, KV **35.01 GiB** | KV **33.82 GiB** (pair minimum) → 5,021,871 tokens, 4.79× at 1M |
| TP=2 reference (Aug-9/README) | weights 79.17 GiB/rank; KV 17.04 GiB / 2,331,430 tokens / 2.22× at util 0.83; FULL graphs 5 `[7,14,21,28,35]`, PIECEWISE `[1,2,4,8,16,24,32,40]`, **"Truncating max_cudagraph_capture_size to 40"** | |
| Host memory | 121.7 GiB total, MemAvailable 8.6 GB, **swap 6.0/16 GB used**, `multi-user.target` | MemAvailable **5.4 GB**, **swap 10.4/16 GB used**, `polkitd` **3.1 GB RSS** (4 d 17 h), `graphical.target`, Xorg + gnome-shell hold GPU contexts |
| Kernel log (this boot) | `NVRM: … NV_ERR_NO_MEMORY … kgrctxAllocMainCtxBuffer` ×12 at 13:00:38–39 | `NVRM: … NV_ERR_NO_MEMORY … _memdescAllocInternal` ×12 at 12:59:18–13:00:40 |
| Fabric | one CX7 ASIC (`sys_image_guid 30c5:9903:00be:5b55`), 2 ports × 2 PFs (PCI 0000:01:00.x and 0002:01:00.x), **each PF PCIe Gen5 ×4**; RoCE `active_mtu 1024` (max 4096); Ethernet MTU 1500 (max 9978); rings 1024/1024 (max 8192); adaptive coalescing rx/tx 8 µs | same |
| Fabric health (since boot, 5 d) | RoCE `hw_counters` all zero (no retransmit/OOS); PHY FEC-corrected lane errors 13 M / 24 M; pause frames rx 12,569 / tx 22,084 (1.7 s cumulative tx pause) | PHY lane errors 8 M / **48 M**; **`rx_out_of_buffer` 13,071**; pause rx 22,084 / tx 12,569 |
| Control-plane path | `10.0.0.1/32` on `lo`, routed via CX7 `enp1s0f1np1` (correct). RTT to worker 0.33 ms avg (0.12–0.54); via second PF 0.70 ms | |
| GPU during the decode probe | SM **2190 MHz** steady (max 3003, application clock 2418), ~26 W rail power, no power-cap event during decode; cumulative SW-power-cap time 29,297 s since boot | 2172 MHz, ~26 W |
| CPU | 20× performance governor, THP madvise, earlyoom inactive (good), irqbalance inactive | same |

### Live decode probe (TP=3 lane, `scripts/bench-miaai.py`, thinking=false, 256-token prompt, 128 output)

| Case | Per-stream decode (median of 3) | Aggregate | TTFT |
|---|---|---|---|
| c=1 | **64.3 tok/s** (57.3 / 64.3 / 66.5) | — | 0.36–0.94 s |
| c=6 | **33.7 tok/s** (32.4 / 33.7 / 33.9) | **156 tok/s** (135 / 156.5 / 156.2) | 0.84–1.45 s |

Spec-decode over the probe window: 2004 accepted / 4140 drafted = **48.4 %**, ≈ **3.9 tokens per step**
(k=6); engine 10-s windows showed per-position acceptance 0.92 / 0.74 / 0.60 / 0.41 / 0.28 / 0.21.
Published TP=2 on this cluster: c=1 62–83 tok/s (75.5 official), c=6 aggregate 162–191.
**TP=3 did not raise decode.**

---

## 2. Where the time goes (first-principles model, checked against measurements)

### 2.1 Bytes per decode step (TP=2, c=1, k=6 → 7 verified target tokens + 6 draft query tokens)

| Component | Bytes / rank / step | Note |
|---|---|---|
| Routed experts: ≤42 distinct of 256 per layer (7 tok × top-6; expected distinct ≈ 39) × 6.7 MB/rank (w13 8.4 MB + w2 4.2 MB + E8M0 scales 0.8 MB = 13.4 MB full, halved by TP) + 1 shared, × 43 layers | **≈ 12 GB** | dominant; hash layers 0–2 route by token id |
| Attention weights (fp8; `fused_wqa_wkv` and indexer `wq_b`/`weights_proj` **replicated**, `wq_b`/`wo_a`/`wo_b` sharded) | ≈ 2.4 GB | ~57 MB/layer/rank |
| Draft (3 DSV4 layers, 6 query tokens, 36 expert slots) | ≈ 0.9 GB | |
| `lm_head` (bf16 129,280×4,096 = 1.06 GB, sharded) read for target logits **and** draft base logits | ≈ 1.06 GB | |
| Markov `w2` 6× (33 MB/rank each) | ≈ 0.2 GB | |
| **Total** | **≈ 16.5–17 GB** | at 230–250 GB/s effective → **68–74 ms** |

At ~4.15 accepted tokens/step this streaming bound alone gives 56–61 tok/s; the measured 62–83 tok/s
means effective bandwidth/overlap is slightly better than assumed, but the conclusion stands:
**c=1 decode on TP=2 is ~80 % weight streaming.** The remaining ~10–15 ms is fixed cost (below).

At TP=3 bytes/rank drop to ≈ 11.5 GB (≈ 48 ms) yet the measured step is ≈ 61 ms (64 tok/s ÷ 3.9
tok/step) — i.e. **~13+ ms of fixed cost at TP=3**, which is what erased the bandwidth gain
(3-rank ring collectives ≈ 2× the latency of a 2-rank exchange; attention padded 64→72 heads;
two different links in the ring).

### 2.2 Collectives per decode step (TP=2)

| Source | Count | Type |
|---|---|---|
| `wo_b` RowParallel per layer | 43 | all-reduce (T×4096 bf16) |
| FusedMoE per layer (shared expert folded in) | 43 | all-reduce |
| Draft layers (3 × 2) | 6 | all-reduce |
| Target / draft input embedding (`VocabParallelEmbedding`) | 2 | all-reduce |
| Target logits + draft base logits (`use_all_gather()` is True on CUDA → **every** rank gathers full logits) | 2 | all-gather (T×129,280 bf16; 10.9 MB at c=6) |
| **Markov loop**: 6 × (`markov_w1` embed all-reduce + `markov_w2` bias all-gather) — **serialized** | **12** | all-reduce + all-gather |
| **Total** | **≈ 108 / step** | at 30–50 µs each on RoCE (no GDR, host-bounce) ≈ **3–5 ms/step** |

Plus the ZMQ scheduler broadcast to the remote worker every step (RTT ≈ 0.33 ms, overlapped by
async scheduling). On a 58–65 ms c=1 step, collectives are ~5–8 %; on a ~150 ms c=6 step ~2–3 %.

### 2.3 What this means for the levers

* **Bandwidth is fixed** (LPDDR5X). Only fewer bytes per accepted token helps: better acceptance
  (draft quality — model-fixed), or **more tokens per step via concurrency** (why c=6 gives 2.2× the
  aggregate). TP=3 cuts bytes/rank by a third but adds fixed cost; it is a **capacity** win
  (5.0 M KV tokens, 16 slots), not a c=1 win — unless items 2 and 4 are fixed first.
* **Fixed cost is software**: uncaptured graphs (#2), serialized draft collectives (#4), eager
  Python launch overhead on mixed steps, the 8 logits gathers.
* **Prefill is compute/kernel-bound** (not DRAM-bound): W4A16 MoE and a replicated, quadratic
  indexer (#5, #6) — the FreeToken write-up in `docs/FREETOKEN_2608.16157.md` reaches the same
  diagnosis from the other direction.

---

## 3. Findings in detail

### 3.1 [P0 · stability + capacity] Unified memory pressure on both hosts

**Evidence.**
* `journalctl -k` on **both** nodes during this boot: repeated `NVRM: … Out of memory
  [NV_ERR_NO_MEMORY] … kgrctxAllocMainCtxBuffer / _memdescAllocInternal` (12:59–13:00 local,
  exactly the profile/autotune/capture window). The boot survived because the failing allocations
  were retried elsewhere, but this is the driver telling you the SoC memory is at the ceiling.
* Worker: `MemAvailable 5.4 GB`, swap **10.4 GB used**, `polkitd` **3.1 GB RSS** (a known polkit
  leak under GNOME sessions), `graphical.target` with `Xorg` (18 MiB GPU) and `gnome-shell` (6 MiB
  GPU) holding GPU contexts, four active `zurih` logind sessions. Head: swap 6.0 GB used, page
  cache 14 GB (it is also the NFS exporter for the worker's weights).
* Worker's `Available KV cache memory` is **1.19 GiB lower** than the head's (33.82 vs 35.01 GiB);
  vLLM sizes the pool from the minimum rank, so the worker's host bloat directly costs KV on both.
* `lmcache/README.md` independently documents the kernel OOM-killing `lmcache server` during a
  cold boot on these nodes.
* `expandable_segments:True` is set (good), `earlyoom` inactive (good, per README).

**Why it matters for tok/s.** On a unified-memory SoC every host page competes with the GPU pool.
Swap-in of vLLM's own 4.6 GB RSS Python process during decode shows up as step-time jitter
(the 57 → 66 tok/s spread across three otherwise identical c=1 trials is consistent with that), and
the pressure is what forces `GPU_MEMORY_UTILIZATION_TEXT` to stay at 0.83.

**Actions (host-side, no engine change).**
```bash
# worker (spark2)
sudo systemctl restart polkit                      # reclaims ~3 GB immediately
sudo systemctl set-default multi-user.target && sudo systemctl isolate multi-user.target
#  → Xorg/gnome-shell off the GPU; ~1–2 GB host RAM back; GDM seat no longer holds a GPU context
sudo swapoff -a && sudo swapon -a                  # only after MemAvailable > used swap; forces pages back
# both nodes: consider vm.swappiness=10 and a 2 GB earlyoom-free headroom check in start-*.sh
```
Then, in a maintenance restart, step `GPU_MEMORY_UTILIZATION_TEXT` 0.83 → 0.85 → 0.87 while
watching `MemAvailable` on both nodes (keep ≥ 6 GB) and the `NVRM` log. Each +0.01 ≈ +1.2 GiB KV
≈ +165 K tokens at TP=2. Suggest a launcher pre-flight: refuse to start if `MemAvailable <
(util × MemTotal − weights − 8 GiB)` or if swap-used > 2 GB on either node.

### 3.2 [P1 · decode c=6] The c=6 decode shape has no CUDA graph on the TP=2 lane

**Evidence (code + Aug-9 TP=2 boot log).**
* compose: `--max-cudagraph-capture-size $(( MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1) ))` = **42**.
* `vllm/config/vllm.py` builds the default list `[1,2,4,8,16,24,32,40]`; 42 is not on the stride, and
  `max_num_tokens` (8192) is not ≤ 42, so nothing appends 42 → **"Truncating
  max_cudagraph_capture_size to 40"** (present in `results/serve-vision-start.log`, k=6, seqs=6).
* `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode` then rounds each size **up** to a
  multiple of `uniform_decode_query_len` = 7 and drops anything > 40: `[7,14,21,28,35]` → the log
  shows **"Capturing CUDA graphs (FULL): 5/5"**. A six-request decode step is 42 tokens →
  `cudagraph_utils.dispatch()` returns `CUDAGraphMode.NONE` → **fully eager**.
* The stashed Aug-19 A/B (`results/rank15-20260819-130541/RESULTS.md`) ran k=5 with max 32 and buckets
  `[1,2,4,6,8,12,16,24,32]` → after rounding to multiples of 6: `[6,12,18,24,30]`; the c=6 step (36
  tokens) was **still eager in both phases**, so "no c=6 lift" was measured with the shape uncaptured
  on both sides. The TP=3 boot today is fine (112 is a multiple of both 8 and 7; FULL 15/15).

**Fix (one expression in `docker-compose.dspark.yml`).** Round the capture size up to a multiple of 8
so the rounding-to-7 pass keeps the full-concurrency shape:
```yaml
--max-cudagraph-capture-size $$(( ( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-6} + 1) + 7 ) / 8 * 8 ))
```
For 6×7 = 42 → 48 → default list `[1,2,4,8,16,24,32,40,48]` → rounded `[7,14,21,28,35,42]`
(48 → 49 > 48 is dropped; 40 → 42 is kept). Verify in the boot log: no "Truncating", and
`Capturing CUDA graphs (FULL): 6/6`. Cost: one more graph (~0.1 GiB).

**Expected effect.** Eager c=6 steps issue ~1,500–1,800 kernel launches from Python per step. With
async scheduling and a ~150 ms GPU step the CPU mostly stays ahead, so the win is bounded:
estimate **+3–10 % c=6 aggregate**, larger at higher `MAX_NUM_SEQS`. Cheap enough to just do.

### 3.3 [P1 · fabric] One PCIe ×4 function is feeding a 200 Gb/s port; MTU 1024; duplicate subnet

**Evidence.**
* `lspci`/sysfs: four `Mellanox MT2910 [ConnectX-7]` functions `0000:01:00.0/.1` and `0002:01:00.0/.1`;
  **all four RoCE devices share `sys_image_guid 30c5:9903:00be:5b55`** and `phys_switch_id
  555bbe000399c530`, with `phys_port_name` p0/p1 repeated → one ASIC, two physical ports, **two PCIe
  functions per port** (socket-direct layout). Each function is **PCIe Gen5 ×4** (`32.0 GT/s`,
  width 4 → ~15.7 GB/s raw ≈ 126 Gb/s). A single PF cannot carry 200 Gb/s.
* TP=2 lane pins `NCCL_IB_HCA=rocep1s0f1` (head) / `rocep1s0f0` (worker) — **one PF each**. The
  second PFs (`roceP2p1s0f1`, `roceP2p1s0f0`) carry `10.0.22.101/.102` **on the same wire** as
  `10.0.22.1/.2` (same port, second PF) — a textbook ARP-flux setup (Linux answers ARP for any
  local address on any interface); RTT via the second PF is 0.70 ms vs 0.33 ms via the first.
* `ibv_devinfo`: `active_mtu 1024` on both nodes because Ethernet MTU is 1500 (`maxmtu 9978`).
  RoCE MTU 4096 needs Ethernet MTU ≥ 4200 (use 9000).
* `ethtool -S`: head tx pause 22,084 / rx 12,569 (1.7 s cumulative pause), worker `rx_out_of_buffer`
  13,071 with rings at 1024/8192 max — buffer-limited bursts (weight loading over NFS on this same
  link is the likely source; pauses stall NCCL traffic too because they are global, not PFC).
  PHY FEC-corrected lane errors are in the tens of millions (worker lane 1: 48 M) — RS-FEC is
  absorbing them (RoCE `hw_counters` are all zero), but worth a `mlxlink -d rocep1s0f1 -m -e` look
  at the cable/EEPROM.
* No GPUDirect: `NCCL_DMABUF_ENABLE` unset and (per `docs/ENVS.md`) the GB10 driver reports
  `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0` → every collective bounces through a host buffer
  (on unified memory that is an extra GPU-side memcpy per op, not a PCIe hop).

**Where it bites.** Decode collectives are small and latency-bound (344 KB at c=6) — MTU/PF width
barely matter. **Prefill** does 86 all-reduces of `[1024×4096]` bf16 = 8 MB per 1024-token chunk
(≈ 690 MB/chunk/rank) plus a 10.9 MB logits all-gather per step at c=6; at ~11 GB/s effective on one
PF that is ~60 ms per chunk (~15 % of a 400 ms short-context chunk, ~5 % of a 1.17 s 900K-context
chunk). The NFS weight load also runs through this PF.

**Actions.**
1. Ethernet MTU 9000 on the CX7 ports (all PFs, both ends, both nodes) → `active_mtu 4096`.
2. `ethtool -G <port> rx 8192 tx 8192` on both nodes (persist via netplan/udev).
3. Either remove the IPs from the second PFs (they duplicate the primary subnet) or move them to a
   distinct subnet — do not leave two addresses of one /24 on two PFs of the same port.
4. A/B `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` (head) / `rocep1s0f0,roceP2p1s0f0` (worker) so NCCL can
   use both PCIe paths into the port (the launcher's GID resolver already supports multi-member
   selectors). Measure with a 2-node `all_reduce` microbench at 256 KB, 8 MB and 64 MB **before**
   trusting it; also try `NCCL_IB_QPS_PER_CONNECTION=2..4`, `NCCL_PROTO=LL` vs `Simple` for the
   small-message regime, and `NCCL_BUFFSIZE`. TP=3 already selects both *ports*; the second *PF*
   per port is a separate question there too.
5. Keep `NCCL_IB_GID_INDEX` auto-resolution as is (it is correct today: index 3 both ranks).

### 3.4 [P2 · decode] DSpark's Markov head is TP-sharded → 12 serialized collectives per step

**Evidence.** `vllm/model_executor/models/qwen3_dspark.py::DSparkMarkovHead`:
`markov_w1 = VocabParallelEmbedding(vocab, 256)` (`# TODO(ben): profile … replicate or TP-shard`),
`markov_w2 = ParallelLMHead(draft_vocab, 256)`; `bias()` goes through `LogitsProcessor`, whose
`_gather_logits` **all-gathers** on CUDA (`Platform.use_all_gather()` → True). The V2 speculator
`dspark/speculator.py::_sample_sequential` runs `for i in range(n_spec): embed(prev) → bias() →
gumbel_sample(...)` — six dependent iterations, each with one all-reduce (embedding, line 493 of
`vocab_parallel_embedding.py`) and one all-gather of `[num_reqs × 129,280]`. All inside the draft's
FULL graph, so no launch overhead, but the network round-trips serialize.

**Fix.** Replicate both matrices (each 129,280 × 256 × 2 B = **66 MB**, trivially affordable) and
bypass the gather: `markov_w1 → nn.Embedding`/`ReplicatedLinear`-style full copy,
`markov_w2 → ReplicatedLinear(256, vocab)`; `bias()` returns `F.linear(embed, w2)` directly (apply
`logits_processor.scale` if non-1). Weight loading for the draft goes through
`DSparkDeepseekV4ForCausalLM.load_weights` — the `mtp.*.markov_head.*` names map to model-level
params; replicated loaders take the unsharded tensor unchanged. This is precisely what Stage-C's
`VLLM_DSPARK_REPLICATE_MARKOV_W1` did. Ship it as another idempotent `patches/hotfix-*.py`.

**Optional second step (greedy drafting only, `DRAFT_SAMPLE_METHOD=greedy`)**: use
`LogitsProcessor`'s vocab-parallel argmax (`_gather_logits` sibling at line 112: O(batch × 2 × tp)
communication) for `compute_draft_logits` instead of a full all-gather — removes one more 9 MB
gather per step. Probabilistic drafting needs full draft logits for the rejection sampler, so it
cannot use this shortcut without also sharding the rejection sampler.

**Expected.** −12 collectives/step ≈ 0.5–1.5 ms → **1–3 % at TP=2, more at TP=3** (where every
collective is a 3-rank ring). Small, clean, safe; stacks with 3.2.

### 3.5 [P1/P3 · MoE kernels] W4A16 everywhere; TC-decode untested; no MXFP4 tensor-core path

**Evidence.**
* `vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py`: `_source_format() =
  "fp4_e8m0_k32"`, `quant_mode="w4a16"` hard-coded at every call site.
  `b12x/integration/tp_moe.py::_validate_fp4_source_format_for_quant_mode`: *"the NVFP4 kernels
  currently support only source_format='modelopt_nvfp4'"* → any non-modelopt FP4 source must use
  W4A16. Boot log confirms: `Using 'B12X_MXFP4' Mxfp4 MoE backend`, `W4A16FusedMoeKernel` JIT.
* Consequence for **prefill**: each 1024-token chunk runs ~6,144 expert-token pairs per layer through
  a dequant-to-bf16 MMA path (~125 dense bf16 TFLOPS class on GB10) instead of FP4×FP8 tensor cores
  (~4× the dense rate). MoE is the largest prefill FLOP consumer at short/medium context.
* Consequence for **decode**: none (bandwidth-bound), *except* kernel efficiency at tiny M.
  `b12x/moe/fused/w4a16/kernel.py`: `_W4A16_SMALL_M_DIRECT_MAX_M = 8` (c=1 decode has M = 7 → the
  small-M direct kernel), and an alternative **`B12X_W4A16_TC_DECODE`** ("small-M packed decode that
  folds the top-k sum into the FC2 store epilogue … never regresses vs the packed GEMM within this
  range", default **off**). It is a plain env var but is **not** in the compose `environment:` block,
  so it cannot be enabled from `.env.dspark` today.
* `cutedsl_warmup.py: Skipping CuTeDSL warmup because no compile units were requested` — the
  W4A16 kernels rely entirely on the persisted disk cache (`B12X_CUTE_COMPILE_CACHE_DIR`), so a new
  M-bucket first seen under load still JITs mid-serve (the #117 hazard class).

**Actions.**
1. Add `B12X_W4A16_TC_DECODE: "${B12X_W4A16_TC_DECODE:-0}"` to compose; A/B `=1` at c=1 (M=7) and
   c=2 (M=14 > 8 → falls back, so also check no regression). Expected 0–10 % c=1.
2. Add a boot-time MoE M-sweep (M = 1,2,4,7,8,14,21,28,35,42 and 1024+) so every W4A16 specialization
   is compiled before "ready" — extend `scripts/boot-shape-warmup.sh` or register compile units.
3. Engineering options for prefill: (a) ask/contribute b12x support for `fp4_e8m0_k32` ×
   MXFP8-activation tensor-core MoE (SM120 `mma … kind::mxf8f6f4` supports E2M1 × E4M3 with E8M0
   block scales natively — the checkpoint's format is the hardware-native one); (b) re-quantize the
   experts offline to modelopt NVFP4 (E4M3 K16 scales) to unlock the existing NVFP4 path — requires a
   quality gate (RULER-lite, tool battery) because it is a re-quantization. Expected prefill MoE
   2–3×, i.e. **prefill +30–60 %** on ≤32K prompts where MoE dominates.
4. The `VLLM_B12X_W4A16_FORCE_BLOCKS_*` knobs are already at their defaults (`0`/`16`/empty); leave
   them unless a profile shows occupancy trouble.

### 3.6 [P3 · prefill at long context] The Lightning indexer is replicated across TP ranks

**Evidence.** `models/deepseek_v4/attention.py::DeepseekV4Indexer`: `# no tensor parallel, just
replicated` — `wq_b` (1024→8192) and `weights_proj` are `ReplicatedLinear`; the indexer computes all
64 index heads × 128 dims against the **entire** compressed key range on **every** rank, and the
indexer K cache (132 B per compressed token per C4 layer, 21 layers) is held on every rank
(~1.5 GB/rank at the 2.3 M-token pool). Score FLOPs per 1024-token chunk at 900K context ≈
1024 × 64 × 128 × 225K × 2 ≈ 3.8 TFLOP per layer × 21 layers ≈ 80 TFLOP — ~0.8 s of tensor-core time
per chunk, **duplicated on both GPUs**. That is the quadratic term behind 875 tok/s at 900K vs
2,563 tok/s at 2K.

**Proposal.** Sequence-parallel indexer: rank r scores only compressed keys in its half of the range
(shard the indexer K cache by position → also halves its memory), computes a local top-512, then the
ranks exchange `(score, index)` pairs for 2×512 candidates per query token (an all-gather of
1024 × 1024 × 8 B ≈ 8 MB per layer per chunk — cheap) and merge. Attention itself is unchanged
(top-k indices are global). Touches `sparse_attn_indexer.py` (`SparseAttnIndexer` op), the indexer
cache spec (per-rank position sharding), and the C128A candidate path. Expected **≤ 1.8× long-context
prefill** (the merge and unchanged attention/MoE keep it below 2×), and it composes with 3.5.
This is the only lever that moves the 128K-TTFT-80 s / 900K-TTFT-1028 s numbers materially.

### 3.7 [P2 · prefill/TTFT] Static 1024-token chunk cap

**Evidence.** `--long-prefill-token-threshold 1024` caps every prefill chunk (issue #27/#43 fix for
decode starvation). With **no** decode lane active (single agent turn, cold prompt), the cap still
applies, so an 8K prompt pays 8 chunks × (86 all-reduces + indexer/compressor setup + ~1.5k
launches + MoE route-pack at M=1024) instead of one 8192-token chunk. `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`
is 1024, so mixed chunks (1024 prefill + decode tokens) also lose the multi-stream input-GEMM
overlap by a few tokens.

**Proposal.** In the scheduler hotfix chain, make the chunk cap adaptive: `threshold = 1024 if any
running request is decode-active else max_num_batched_tokens`. Fairness under mixed load is
unchanged (the #43 decode floor still applies); c=1 cold TTFT improves. Also consider 2048 as the
static value with `MAX_NUM_BATCHED_TOKENS=16384` if adaptivity is not wanted (GLM's suggestion;
DS4F's counter-argument was only that the cap isn't binding at c=6 — true, and irrelevant at c=1).
Expected c=1 TTFT **−10–25 %** on ≤32K prompts; at ≥128K the indexer (3.6) dominates instead.

### 3.8 [capacity · upstream] `nvfp4_ds_mla` is not a 4-bit KV cache

`vllm/v1/kv_cache_interface.py`: *"DeepseekV4 uses the padded 584-byte sparse-MLA envelope for both
fp8_ds_mla and nvfp4_ds_mla … 448 B NoPE + 128 B RoPE + 8 B fp8 scale = 584 B per token"*;
`attention.py::get_kv_cache_spec` sets alignment 584 for nvfp4 vs 576 for fp8. Issue #22's fix
routed nvfp4 to the fp8 kernel *because the bytes are the same*. So the recipe's KV is effectively
FP8; there is no accuracy risk from "FP4 KV" and no memory win either. A real NVFP4 row (448 dims →
224 B + 28 B scales + 128 B RoPE + 8 B ≈ 390 B) would give **+50 % KV tokens** — a FlashInfer
SM120 kernel feature to track upstream. Related: DCP (`decode_context_parallel_size`) is not wired
for the DSV4 sparse backends in this image (`grep -r dcp models/deepseek_v4/` is empty) — with TP
the whole KV is replicated on both ranks; DCP=2 would double capacity and halve per-rank
attention bytes at long context at the price of ~43 more collectives per step.

### 3.9 [P1 · fairness] Live `DSPARK_MAX_INFLIGHT_PREFILLS=2`

`.env.dspark` still carries the Rank-2 experiment value (`2`). CHANGELOG 2026-08-28 (#154): same-code
A/B showed four-lane 8K fairness spreads of 3.72×/5.14×/4.94× at `2` vs 2.04×/1.68×/1.69× at `1`;
the shipped default went back to `1`. The upside of `2` is the 32K×c4 decode floor (8.2 → 24.6 tok/s,
PR #90). Decide explicitly; if you keep `2`, note it in the env comment as a deliberate trade.

### 3.10 [P1 · latency cliffs] Warmup / autotune coverage

* `flashinfer_sparse_mla_warmup.py`: *"Warming up DeepSeek V4 sparse MLA attention for mixed
  tokens=16"* — only one shape; the Aug-19 log still showed `tactic=-1` fallbacks at T=6 and T=32 with
  last-dim 256. Extend the warmup shape list to the real uniform-decode row counts
  (7·c for c=1..MAX_NUM_SEQS) and the draft's 6·c. The autotune cache is persisted under
  `vllm-cache/flashinfer_autotune_cache/<hash>/`, so this is a one-time cost per config hash.
* CuTeDSL warmup is skipped (3.5). Add an M-sweep.
* This boot's first requests still JIT'd `_prepare_dflash_inputs_kernel` and `_topk_topp_kernel` on
  both ranks (new TP=3 config key → cold Triton cache; expected once per config). The
  `boot-shape-warmup.sh` ladder assumes k=5 (`+6` in its BLOCK key comment) — re-derive for k=6
  (`next_pow2(scheduled_tokens + 7)`).
* `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` is unset but the vLLM-side cache path is used; fine.

### 3.11 [stability] Other observations

| Item | Evidence | Action |
|---|---|---|
| Stochastic sparse-MLA stall (#141) | Documented; default-off 64-row chunk workaround; prefill (1024-row calls) already runs on FlashInfer's >64-row "paged fallback" path where the stall was sampled | Keep watching missing `finish_reason`; if you enable the workaround, re-measure **prefill** (it turns one 1024-row call into 16 sequential 64-row calls per layer) |
| Two NCCLs in one process | pynccl 2.30.7 (TP collectives) vs torch-bundled 2.28.9 (ProcessGroup/watchdog) | harmless; know it when reading FR dumps |
| Docker json-file logs unbounded | `LogConfig {json-file map[]}` on both nodes | add `logging: options: max-size: 200m, max-file: 5` |
| Grace CPU at `performance`, `OMP_NUM_THREADS→1` by vLLM, spin-wait hotfix (#79) applied | good | keep; CPU spin steals SoC power budget from the GPU |
| GPU clocks | SM held at 2190 MHz during decode (max 3003, app clock 2418); no power-cap event during the probe; 8.1 h cumulative SW power cap since boot (prefill-heavy periods) | `nvidia-smi -q -d SUPPORTED_CLOCKS` is N/A on GB10 — no user lever; decode is bandwidth-bound anyway |
| `torch.compile` "not supported" warning; `thinking_token_budget` V2 warning; four unknown `VLLM_BUILD_*` env warnings | cosmetic | ignore or silence |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` (stock 512) | halves indexer logits workspace → more indexer launches per long-context chunk | restore 512 once 3.1 frees memory |
| Head is the NFS exporter for worker weights on the same CX7 PF | 156 GB load at ~1.1 GB/s (disk-bound) but bursts trip global pause frames | fine after MTU/ring fix; keep `DSPARK_WORKER_HF_NFS=1` |

---

## 4. Action plan

**Phase 0 — host-side, no engine restart (do first).**
1. Worker: `systemctl restart polkit`; switch to `multi-user.target`; verify `nvidia-smi` shows only
   `VLLM::Worker_TP1`. Head: nothing required beyond checking swap.
2. Both: MTU 9000 on CX7 ports, RX/TX rings 8192, fix the duplicate-subnet IPs on the second PFs.
3. Docker log rotation.

**Phase 1 — one maintenance restart, config only (measure each with §5).**
4. Compose: capture size rounded to a multiple of 8 (§3.2). Verify `FULL: 6/6`.
5. `.env`: decide `DSPARK_MAX_INFLIGHT_PREFILLS` (1 per #154, or 2 knowingly).
6. `GPU_MEMORY_UTILIZATION_TEXT` 0.83 → 0.85 (→ 0.87 next window) after Phase 0.
7. Compose: pass `B12X_W4A16_TC_DECODE` through; A/B `=1`.
8. Launcher: extend FlashInfer warmup shapes and add the MoE M-sweep (§3.10).
9. Fabric A/B in a separate window: both PFs in `NCCL_IB_HCA`, `NCCL_IB_QPS_PER_CONNECTION`,
   `NCCL_PROTO` — with a 2-node all-reduce microbench first, then `bench-ttft.py`.

**Phase 2 — small vLLM patches (idempotent hotfix style, CPU tests like the existing ones).**
10. Replicated Markov head (§3.4).
11. Adaptive prefill chunk cap (§3.7).
12. Optional: vocab-parallel argmax for greedy drafting; A/B `DRAFT_SAMPLE_METHOD=greedy` (official
    cards) vs `probabilistic` for acceptance and tok/s at temperature 0.6.

**Phase 3 — engineering (choose by workload).**
13. Long-context agents: sequence-parallel indexer (§3.6).
14. Prefill-heavy / RAG: MXFP4 tensor-core MoE (§3.5.3).
15. Capacity: true FP4 KV row / DCP upstream tracking (§3.8); TP=3 as a capacity lane after 10–11
    land (re-measure c=1; today it is 64 vs ~70–75 on TP=2).

---

## 5. Measurement protocol (use the repo's tools; one variable per restart)

```bash
# decode matrix (median-of-trials; thinking=false; unique cold prefixes)
python3 scripts/bench-miaai.py --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-vision-exp --prompt 256   --concurrency 1 --repeat 5
python3 scripts/bench-miaai.py ... --prompt 256   --concurrency 6 --repeat 5
python3 scripts/bench-miaai.py ... --prompt 2048  --concurrency 4 --repeat 3
python3 scripts/bench-miaai.py ... --prompt 32768 --concurrency 1 --repeat 3
python3 scripts/spec-acceptance.py --trials 5 --prompt 256          # acceptance + per-position curve
python3 scripts/bench-ttft.py ...                                   # prefill sweep (for 3.3/3.5/3.7)

# boot-log gates after each restart (both ranks)
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | grep -E "Truncating|Capturing CUDA graphs \(FULL\).*100%|No tuned config|tactic=-1|jit_monitor|Available KV cache memory"

# where the step time goes (one-off): enable the torch profiler at boot, then
#   VLLM_TORCH_PROFILER_DIR=/cache/huggingface/profiles  (compose env) → curl -X POST :8888/start_profile ;
#   run c=1 × 64 tokens ; curl -X POST :8888/stop_profile ; open the trace: split NCCL vs GEMM vs
#   attention vs idle. This single trace will confirm or refute the §2 split before Phase 2 work.

# fabric microbench (inside the container, both nodes; before/after 3.3)
#   torchrun --nnodes 2 --nproc_per_node 1 --master_addr 10.0.0.1 --master_port 25100 allreduce_bench.py
#   sizes 256 KB / 8 MB / 64 MB; report latency and GB/s; repeat with NCCL_IB_HCA=<one PF> vs <both PFs>
```
Always compare against the same-day baseline (this box swings 57–66 tok/s at c=1 between
otherwise identical trials while the worker is swapping), and read `/metrics`
`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total` over the same window.

---

## Appendix A — Model / lane facts used above

* `config.json`: 43 layers (0–2 hash-routed MoE, no router GEMM), hidden 4096, 64 heads × head_dim 512
  (`q_lora_rank` 1024, `o_lora_rank` 1024, `o_groups` 8, rope 64), 256 routed experts top-6 + 1 shared,
  `moe_intermediate_size` 2048, `expert_dtype fp4` (E8M0 block-32 scales), other linears fp8 block
  128×128; `index_n_heads` 64 × `index_head_dim` 128, `index_topk` 512, `sliding_window` 128;
  `compress_ratios`: layers 0–1 SWA-only, layers 2–42 alternate C4A (21) / C128A (20), MTP layers
  43–45 SWA-only; `num_nextn_predict_layers` 3, `dspark_block_size` 5, `dspark_markov_rank` 256;
  YaRN ×16 from 64K to 1M.
* Compose argv (TP=2): `--tensor-parallel-size 2 --kv-cache-dtype nvfp4_ds_mla --block-size 256
  --max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192
  --long-prefill-token-threshold 1024 --max-cudagraph-capture-size 42 --gpu-memory-utilization 0.83
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill --speculative-config
  {"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}
  --moe-backend flashinfer_b12x --enable-flashinfer-autotune`; `VLLM_USE_BREAKABLE_CUDAGRAPH=0`,
  `VLLM_USE_B12X_MOE=1`, `CUTE_DSL_ARCH=sm_121a`, persisted Triton/TileLang/CuTe/autotune caches.
* Hotfix chain applied at boot (all verified in the live log): #21 encoder, #55 tool truncation,
  #22 nvfp4→fp8 kernel route, #79 spin-wait, #50312 MTP buffer, #49486+#52492 short-context top-k,
  #48407 (dormant), #48957 empty-C128 skip, #50298 FlashMLA workspace, #44993 grammar advance,
  Vision-Exp ViT/aligner, empty-encoder-output, #27 partial-prefill cap, #43 decode floor, #26 hybrid
  SWA min, #133 Triton specialization, suppress-stops-in-reasoning; TP=3 pad patch when `TP_SIZE=3`.
* KV groups: 1× `MLAAttentionSpec` (block 256) + 3× `SlidingWindowMLASpec` (blocks 64/4/8) — the
  hybrid coordinator issue (#26) is documented in `ISSUE26_HANDOFF.md`.

## Appendix B — Commands that produced the evidence (all read-only)

```bash
docker exec deepseek-v4-flash-vllm-dspark-1 env | grep -E '^(NCCL|VLLM|B12X|TRITON|TILELANG|CUTE|TORCH)'
docker logs deepseek-v4-flash-vllm-dspark-1 | grep -E 'Truncating|Capturing CUDA graphs|Available KV|autotun|jit_monitor'
ibv_devinfo -d rocep1s0f1 | grep -E 'active_mtu|max_mtu'; ip -d link show enp1s0f1np1 | grep -o 'mtu [0-9]*'
ethtool -g enp1s0f1np1; ethtool -S enp1s0f1np1 | grep -E 'pause|out_of_buffer|err_lane'
for d in /sys/class/infiniband/roce*; do cat $d/sys_image_guid; done   # one ASIC
cat /sys/bus/pci/devices/0000:01:00.1/current_link_{speed,width}      # 32 GT/s ×4
ip route get 10.0.0.2; ping -c5 -i0.2 10.0.0.2
ssh zurih@10.0.0.2 'free -g; swapon --show; ps -eo rss,comm --sort=-rss | head; systemctl get-default; nvidia-smi'
journalctl -k --since '-14 days' | grep -i 'NV_ERR_NO_MEMORY'
curl -s :8888/metrics | grep -E 'spec_decode_num_(accepted|draft)_tokens_total'
python3 scripts/bench-miaai.py --model deepseek-v4-flash-vision-exp --prompt 256 --concurrency {1,6} --repeat 3
```


---

## Addendum (2026-09-02, later) — implementation status

### Verification of items 1–5 and 7 (done by another agent; checked here)

| Item | Verified state | Notes |
|---|---|---|
| 1 host memory | ✅ worker: `multi-user.target`, swap 0, `polkitd` gone; head swap 1.8 GB (was 6.0). **New:** worker `wireplumber` at 3.8 GB RSS (PipeWire session manager, 5 d) — stop/mask it. `GPU_MEMORY_UTILIZATION_TEXT` raised to 0.85 in `.env.dspark`. | server was down at check time, so KV effect not yet observed |
| 2 cudagraph 42→48 | ✅ compose renders `--max-cudagraph-capture-size $((…+7)/8*8))` = 48; launcher profile print updated | expect `FULL: 6/6` in the next TP=2 boot log |
| 3 fabric | ✅ MTU 9000 → RoCE `active_mtu 4096`; rings 8192/8192; second PFs moved to `10.0.122.0/24`; `.env` uses exact selectors `=rocep1s0f1,roceP2p1s0f1` / `=rocep1s0f0,roceP2p1s0f0`; the launcher's own resolver returns GID index 3 for both members on both nodes | NCCL's use of both PFs is unverified until a boot (`NCCL_DEBUG=INFO` shows the NET/IB device list) |
| 4 Markov replicate | ✅ `patches/hotfix-dsv4-replicate-markov-head.py` applies in compose order, idempotent, imports; `nn.Embedding` + `ReplicatedLinear` take the unsharded `mtp.*.markov_head.*` tensors through the existing loader; `tests/test_replicate_markov_head.py` passes | runtime TP=2 correctness still needs the first live boot + `spec-acceptance.py` (acceptance must be unchanged) |
| 5 TC-decode | ✅ `B12X_W4A16_TC_DECODE` passed through compose; live `.env` sets `1` | that is the A/B "on" arm — measure c=1 vs `0` |
| 7 adaptive chunk | ✅ patch applies (with #27 before, #43 after), idempotent; predicate `any(not r.is_prefill_chunk …)` matches the stock scheduler's own decode-active test (line 438) and `_update_after_schedule` maintains the flag; 5 CPU tests pass | first-step false positive for freshly admitted requests is conservative (keeps 1024), not a regression |
| — | ⚠️ `DSPARK_MAX_INFLIGHT_PREFILLS=2` still in the live `.env` (item 9 not assigned) | decide per #154 |
| — | ⚠️ none of this is committed; AGENTS.md wants PR → `main` | `git status` lists the files |

### Item 6 — delivered (opt-in, default off)

`patches/hotfix-dsv4-sp-indexer-prefill.py` + compose/launcher/CI/docs wiring (`DSPARK_ENABLE_SP_INDEXER`,
`DSPARK_SP_INDEXER_MIN_KEYS`). Validation: `tests/test_sp_indexer_prefill.py` (apply on the real image
file; split/bounds math vs brute force, TP 2/3/4) and `scripts/test-sp-indexer-gpu.py` — **PASS on GB10
with the real kernels** for emulated TP=2 and TP=3. Remaining: live A/B (`bench-ttft.py` 32K → 900K,
`ruler-lite.py`) in a maintenance window with the flag on; expected long-context prefill speed-up
approaches 2× on the indexer term (≈1.5–1.8× end-to-end at ≥128K), no change below 32K.

### Item 8 — design delivered, plus one new hazard

`docs/CLAUDE/item8-fp4-kv-design.md`: why `nvfp4_ds_mla` is 584 B/row, the 292-byte layouts that fit the
hybrid allocator's page constraint (recommend option A: e2m1 NoPE with per-128 UE8M0 scales + fp8 RoPE),
FlashInfer/vLLM touchpoints (all JIT-built from sources present in the image), the cheap MXFP4
*indexer*-cache experiment, and the validation gate. Found on the way: **the vendored DeepGEMM emits
`sm121_*` indexer-logits kernels but ships only `sm120_*` headers** — production runs on persisted JIT
cache entries built from `sm120_*`; a cache miss fails to compile. `patches/hotfix-deepgemm-sm121-mqa-header-alias.sh`
(opt-in `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS`) fixes it and was exercised by the item-6 GPU test.

### A/B sequence 2026-09-02 (afternoon) — stopped at step A; details in `ab-results-2026-09-02.md`

1. Baseline boot (all four opt-in knobs 0, util 0.83, capture 48, dual-PF, alias on): gates clean on both ranks, KV 16.55 / 14.73 GiB, MemAvailable ≥ 4.8 GB both nodes, swap flat during runs.
2. Decode c=1 52.4 tok/s (49.8–56.9), c=2 43.4, c=6 aggregate 148; TTFT 4.5 / 18 / 79 s; numbered-word acceptance 47 %; quality probe OK.
3. Natural-prose acceptance 25 % / pos0 0.68 **with the Markov replicate OFF** — so that patch is not the cause of the low natural acceptance seen earlier today.
4. Recovery gate (c=1 ≥ 60, pos0 ≥ 0.85) failed → per plan, ran the `baseline-singlepf` diagnostic (one PF per node) instead of B–F.
5. Single-PF: c=1 65.7 (51.5–76.3) on the first 3 trials, 54.8 (47.7–57.9) on a 5-trial repeat; c=2 / c=6 / TTFT / acceptance unchanged within noise → tie, not a fabric problem.
6. c=1 on this lane swings ~48–76 tok/s with the prompt drawn, so the noise band is ≈ ±10 tok/s; 3 trials cannot resolve a 1–3 % knob effect.
7. The pos0 ≥ 0.85 bar traces to AUDIT.md's "good prompts" curve; live cumulative pos0 was already ≈ 0.78 (MERGED.md / GROK-4.6.md), so 0.68–0.71 is a pre-existing live-vs-audit gap, not a regression.
8. Sequence continued (goal check): **markov** → acceptance unchanged (26.0 % / pos0 0.69 — the patch is correct, not a bug) but no c=1 gain and c=6 aggregate 122 vs 148 → OFF; **tcdecode** → c=1 46 vs 55 median over 8 trials (7/8 below baseline min) → OFF.
9. **adaptive-chunk** booted clean twice but head MemAvailable was 2.36 / 2.28 GB after boot vs 3.8–5.2 GB on every other boot → "< 3 GB" rule fired both times: OFF, unmeasured (larger chunk workspace lands in unified/host memory outside the util budget; not a leak across restarts). **sp-indexer** → ON: ruler-lite 8/8 at 32k+131k, cold 128K TTFT 76.0 / 74.4 s vs 79.0 / 81.2 s baseline (≈ −6 %, far below the 1.5–1.8× design estimate), 8K/32K/decode/acceptance unchanged.
10. **mtp3** not bootable (launcher: k ≥ 5 and k % 3 == 0 for Vision-Exp). Final: `.env.dspark` = baseline + `DSPARK_ENABLE_SP_INDEXER=1` (others 0, dual-PF, util 0.83, k=6), server restarted on it; nothing committed. Raw: `results/ab-{baseline,baseline-singlepf,markov,tcdecode,sp-indexer}-20260902T*.md`, `results/ruler-lite-sp-indexer-20260902.json`.

### A/B sequence 2 (2026-09-02, 15:00–16:30 UTC) — every `.env.dspark` knob now measured; details in `ab-results-2026-09-03.md`

1. Same-day baseline `base2` (= sequence-1 final config): c=1 56.6 tok/s pooled over 8 trials (50.8–67.0), c=2 42.7, c=6 aggregate 139, TTFT 4.5 / 18.2 / 78.5 s, natural 24.7 % / pos0 0.66, numbered 47.3 %; c=4 8K fairness case fully serialized (TTFT 4.7 / 9.4 / 14.3 / 19.3 s, spread 14.6 s).
2. Head MemAvailable on the untouched baseline was 2.5 GB (≈ 1 GB more user-space load than yesterday), so the hard 3 GB gate was replaced by a 2.0 GB gate (`AB_MEM_GATE_KB` in `ab-boot.sh`) judged relative to base2 plus the swap-I/O check; every later non-chunk boot came back at 2.9–4.1 GB.
3. **chunk2048** (§3.7 static fallback) → OFF, unmeasured: head 1.0 GB and swap +1 GB, the adaptive-chunk cost signature; chunk4096 skipped. Prefill chunks above 1024 wait for the host-memory owner (item 1).
4. **greedy** drafting (Phase-2 #12) → OFF, tie: acceptance and decode identical to base2 at temp 0 and 0.6.
5. **idxlogits512** (§3.11) → OFF, tie: 32K −1.5 %, 128K −1.8 %, inside the ±2 % noise; first knob to retry if a ≥256K sweep is run.
6. **inflight2** (§3.9, PR #90 vs #154) → **ON**: c=4 8K TTFT spread 9.1 vs 14.6 s (all trials below base2's best), aggregate +12 %, worst stream 16.5 vs 19.3 s; c=1/c=2/c=6, 32K TTFT, acceptance and memory unchanged.
7. **capture42** (item 2 isolated for the first time via a one-line compose edit, reverted): c=6 aggregate 122 vs 139 with capture 48, all trials below base2's min; c=1/c=2/TTFT unchanged → capture 48 confirmed.
8. Final: `.env.dspark` = sequence-1 config + `DSPARK_MAX_INFLIGHT_PREFILLS=2` (+ explicit `DRAFT_SAMPLE_METHOD=probabilistic`); final boot clean on both ranks, KV 16.76 / 15.47 GiB, MemAvailable 3.3 / 4.0 GB settled; nothing committed.
9. **Not tried, honestly:** everything reachable from `.env.dspark` without root has now been measured on this lane. Remaining items are engineering: MXFP4 tensor-core MoE / NVFP4 re-quant (§3.5), FP4 KV (§3.8), warmup coverage (§3.10), the NCCL tuning compose does not pass through, the one-off torch profiler trace, and the root-only host-memory owner hunt that gates any larger prefill chunk.
