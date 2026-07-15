# DeepSeek V4 Flash DSpark C12 NVFP4 KV on 2x DGX Spark

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-DSpark`
with vLLM TP=2, DSpark speculative decoding, and a **1M-token** default max
model length using the experimental `nvfp4_ds_mla` KV-cache path.

## Current runtime (this checkout)

The default Docker image is the prebuilt Anemll GX10/DGX Spark port of vLLM
0.25 with native DSpark / NVFP4 DS-MLA / b12x MoE support:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

Source: [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10).
Pull on **both** nodes before first start:

```bash
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

`docker-compose.dspark.yml` is aligned with that image layout:

- entrypoint cleared; command uses `/usr/local/bin/vllm serve`
- CUDA under `/usr/local/cuda` (not Stage-C `/opt/env`)
- `--moe-backend flashinfer_b12x`
- DSpark is **built into the image** (no Stage-C bind-mount of
  `dspark_proposer.py` over `/opt/env/...`)
- optional `vllm_patch_gb10/` mount remains for experimental hybrid NVFP4
- HF cache at `/cache/huggingface`; prefer `HF_HUB_OFFLINE=1` once both nodes
  have a full local hub cache (online re-download can fill worker disks)

Alternative: set `DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c`
and run `./build-dspark-vllm-runtime.sh` for the historical multi-stage Stage-C
overlay build. Stage-C recipes and overlay sources remain under `recipe/`.

This repo still vendors Keys' DSpark concurrency patch and Stage-C overlay
sources for local image builds and documentation. With the Anemll image, that
logic ships inside the image rather than as a host bind-mount.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

**Default agent-serving profile** (`.env.dspark.example` and README defaults):

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- model: `deepseek-ai/DeepSeek-V4-Flash-DSpark` (HF hub id; resolved offline from cache when `HF_HUB_OFFLINE=1`)
- `max_model_len=1048576` (**1M** — keep this as the documented default)
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `kv_cache_dtype=nvfp4_ds_mla`
- `gpu_memory_utilization=0.85`
- `MTP_NUM_TOKENS=3`
- API bind address `0.0.0.0:8888`

Local `.env.dspark` may lower `MAX_MODEL_LEN` (for example `512000`) for a
specific cluster without changing the recipe default.

> [!IMPORTANT]
> This profile is meant for real deep-context agent serving: up to **1M tokens
> per separate session** with `MAX_NUM_SEQS=6`. The KV cache is a shared pool,
> so six sessions do not each reserve 1M tokens up front. Normal agent
> sessions can run concurrently while retaining the 1M ceiling for unusually
> long requests.

> [!IMPORTANT]
> For long coding tasks and big prompts, use:
>
> ```env
> MAX_MODEL_LEN=1048576
> MAX_NUM_SEQS=4
> MAX_NUM_BATCHED_TOKENS=16384
> GPU_MEMORY_UTILIZATION=0.87
> ```

This repo documents the validated 1M NVFP4 agent profile, historical Stage-C
checkpoints, and the current Anemll prebuilt runtime:

- default `max_model_len=1048576` (1M), `max_num_seqs=6`, `kv_cache_dtype=nvfp4_ds_mla`
- default image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (~2.8M-token KV pool on this cluster)
- historical Stage-C C12 pool: `3,225,280 tokens`
- single-stream decode stayed above `50 tok/s` on validated C12 gates
- deterministic direct prompts completed with no Chinese drift or repeated junk
- 2/4/6 concurrent code-gate prompts completed cleanly (Stage-C C12)
- DSpark concurrency patch validated at `max_model_len=200000`, `max_num_seqs=16`
  (static C16 `315.1` / staggered C16 `205.0` tok/s aggregate)

If you already deployed an older copy and saw agent garble, loops, Chinese
drift, or prompt/tool XML leaking into replies, keep the C12 NVFP4 profile and
validate direct API behavior before changing agent harness settings. The fix
path does not switch production to fp8 or a smaller fallback model.

> [!WARNING]
> If direct vLLM prompts are clean but an agent harness still garbles, check the
> harness session replay, fallback model list, and prompt/tool XML handling
> before changing DSpark weights or falling back to fp8.

## Result

### Live Anemll image lane (this checkout)

Validated worker-first launch with the prebuilt Anemll image and this repo's
compose/start scripts (TP=2, two nodes).

Runtime:

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- model id: `deepseek-ai/DeepSeek-V4-Flash-DSpark` (HF cache under `HF_CACHE`)
- served model name: configurable via `SERVED_MODEL_NAME` (example: `deepseek-v4-flash`)
- `kv_cache_dtype=nvfp4_ds_mla`
- default recipe: `max_model_len=1048576`, `max_num_seqs=6`,
  `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.85`, `MTP_NUM_TOKENS=3`
- `--moe-backend flashinfer_b12x`
- `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_USE_B12X_WO_PROJECTION=1`
- `HF_HUB_OFFLINE=1` recommended after both nodes have a complete hub cache
- fabric: explicit `VLLM_HOST_IP` / `WORKER_VLLM_HOST_IP`, plus matching
  `NCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`

Boot evidence on this cluster (Anemll image, 1M max-model-len profile):

```text
Available KV cache memory: 19.03 GiB
GPU KV cache size: 2,826,378 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.70x
Application startup complete.
```

Direct API smoke: `/v1/models` HTTP 200 and OpenAI-compatible chat completions
returned non-empty assistant content on both head and worker ranks.

Concurrency code-prompt bench (`max_tokens=256`, temperature 0; aggregate =
total completion tokens / batch wall):

| Concurrency | Success | Batch wall (s) | Completion tokens | Aggregate tok/s | Mean per-stream tok/s |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 4.62 | 256 | 55.4 | 55.5 |
| 2 | 2/2 | 6.10 | 512 | 84.0 | 43.1 |
| 3 | 3/3 | 15.55 | 768 | 49.4 | 16.6 |
| 4 | 4/4 | 14.55 | 1024 | 70.4 | 17.8 |
| 5 | 5/5 | 19.93 | 1280 | 64.2 | 13.0 |
| 6 | 6/6 | 8.84 | 1536 | 173.7 | 29.9 |

### 2026-07-02 Keys C12 NVFP4 Checkpoint (historical Stage C)

Earlier high-concurrency lane on Tony's Stage C NVFP4 image with Keys' C12
serving profile (kept for comparison; not the current default image).

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image: `vllm-dspark-runtime:dspark-nvfp4-stage-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1048576`
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.85`
- `MTP_NUM_TOKENS=3`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `thinking=false`
- `--generation-config vllm`
- no `--override-generation-config`

Boot evidence:

```text
GPU KV cache size: 3,225,280 tokens
Maximum concurrency for 1,000,000 tokens per request: ~3.2x
Application startup complete.
```

Code-gate validation:

| concurrency | success | server generation tok/s | acceptance | bad outputs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 52.79 | 0.585 | 0 |
| 2 | 2/2 | 79.76 | 0.600 | 0 |
| 4 | 4/4 | 134.70 | 0.602 | 0 |
| 6 | 6/6 | 127.78 | 0.615 | 0 |
| 12 | 12/12 | 230.10 | 0.602 | 0 |

The upstream checkpoint note for this run was not imported into this checkout;
this repo keeps the runtime changes and validation summary without the upstream
benchmark artifact folder.

Do not enable `VLLM_USE_B12X_FP8_GEMM=1` on this Stage C image. That flag hit a
DeepGEMM layout assertion during DSpark drafter warmup in testing.

### 2026-06-30 Clean Agent-Serving Checkpoint

The prior conservative clean endpoint was reproduced on Asusi/Spark4 before
sending the model back through Hermes/OpenClaw-style harnesses.

Runtime:

- endpoint tested: `http://100.90.25.78:8888/v1`
- served model: `deepseek-v4-flash-dspark`
- image used on that lane: `vllm-dspark-runtime:mia-raf-pr1-nvfp4-keys-c`
- model path: `/cache/huggingface/fraserprice/DeepSeek-V4-Flash-DSpark`
- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1048576`
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.80`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `--generation-config vllm`
- `--override-generation-config '{"temperature":0.0,"top_p":1.0}'`
- explicit per-node `VLLM_HOST_IP` values

Boot evidence:

```text
GPU KV cache size: 1,990,142 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.90x
Application startup complete.
```

Direct validation:

- `/v1/models` reported `"max_model_len": 1048576`
- deterministic sanity prompt returned `NVFP4 DSPARK OK`
- five longer English prompts completed with no CJK drift and no repeated junk
- code-gate server decode mean: `54.22 tok/s`
- 2/4/6 concurrent direct prompts all succeeded cleanly

Concurrency:

| concurrency | success | aggregate tok/s | stability |
| ---: | ---: | ---: | --- |
| 2 | 2/2 | 60.95 | no CJK/repeat junk |
| 4 | 4/4 | 83.21 | no CJK/repeat junk |
| 6 | 6/6 | 104.11 | no CJK/repeat junk |

The upstream checkpoint note for this run was not imported into this checkout.

### 1M NVFP4 Profile

Validated on 2x DGX Spark, one GPU per node, TP=2, single stream.

| Case | server tok/s | TTFC | acceptance | accepted/draft |
| --- | ---: | ---: | ---: | ---: |
| p256/g64 | 54.46 | 0.506s | 0.667 | 3.33 |
| p256/g256 | 65.38 | 0.324s | 0.718 | 3.59 |
| p512/g64 | 56.26 | 2.738s | 0.625 | 3.13 |
| p512/g256 | 54.41 | 0.422s | 0.550 | 2.75 |
| p512/g256 warmup1 | 56.73 | 0.417s | 0.585 | 2.92 |

Boot logs reported:

```text
GPU KV cache size: 2,044,166 tokens
Maximum concurrency for 1,048,576 tokens per request: 1.95x
```

The API reported:

```json
{"max_model_len":1048576}
```

The upstream checkpoint note for this run was not imported into this checkout.

### DSpark Concurrency Profile

Validated on the same 2x DGX Spark TP=2 deployment using Keys' DSpark
concurrency patch, `kv_cache_dtype=nvfp4_ds_mla`, `max_model_len=200000`,
`max_num_seqs=16`, `MTP_NUM_TOKENS=5`, and
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.

Patch source:

- [drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- Tested patch commit: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

The live fix documented here keeps `kv_cache_dtype=nvfp4_ds_mla` and refreshes
the repo's already-vendored Keys overlay with the path-adjusted Patch 2b update
from that commit. In Patch 2b, ragged `query_start_loc` detection no longer
depends on `num_rejected_tokens_gpu`. Treat the service as validated only after
the built-in OpenAI-compatible chat smoke request plus agent-client validation
pass on the live service.

Static simultaneous batch, one TP=2 replica:

| concurrency | best aggregate tok/s | per-stream tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 1 | 57.6 | 57.6 | 0.635 |
| 4 | 140.8 | 35.2 | 0.619 |
| 8 | 252.6 | 31.6 | 0.635 |
| 16 | 315.1 | 19.7 | 0.609 |

Staggered independent arrivals, one TP=2 replica:

| concurrency | success | aggregate tok/s | acceptance |
| ---: | ---: | ---: | ---: |
| 4 | 4/4 | 109.2 | 0.544 |
| 8 | 8/8 | 147.3 | 0.534 |
| 16 | 16/16 | 205.0 | 0.567 |

Correctness sanity check: deterministic victim output remained byte-identical
under churn. A medium-churn condense test measured `0.529` acceptance and
`99.7 tok/s` across the churn window.

The upstream checkpoint note for this run was not imported into this checkout.

### Historical 60 tok/s DSpark Baseline

The older ~60 tok/s number was reproduced, but it is a separate diagnostic
profile, not this repo's default 1M NVFP4 deployment:

- image rebuilt from `rafaelcaricio/vllm#1` commit `3519c3b88`
- `max_model_len=262144`
- `max_num_seqs=1`
- `kv_cache_dtype=fp8`
- `MTP_NUM_TOKENS=5`
- `thinking=false`
- `temperature=0.0`, `top_p=1.0`
- measured `63.97 tok/s` on the `code_completion` gate with `67.9%`
  DSpark acceptance

Use this to diagnose image/runtime drift. Do not confuse it with the production
1M NVFP4 path. The upstream checkpoint note for this run was not imported into
this checkout.

### 2026-06-29 Full-1M Concurrency Microbench

The 200K/16 profile above maximizes raw concurrency. For agent fleets that want
the **full 1M context ceiling AND concurrency**, run `max_model_len=1048576`
with `max_num_seqs=6`. Every request can still grow to 1M while up to 6 sessions
run at once, because the shared KV pool — not a per-slot reservation — is the
real limit (see [How the KV cache works](#how-the-kv-cache-works-why-1m--concurrency-is-safe)).

Validated on the 2026-06-29 code-completion microbench deployment (NVFP4,
`max_model_len=1048576`, `max_num_seqs=6`,
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`, `VLLM_USE_B12X_WO_PROJECTION=1`):

- Boot: `GPU KV cache size: 1,901,239 tokens`, `Maximum concurrency for 1,048,576 tokens per request: 1.81x`
- 6 concurrent requests: **6/6 success**, **~182 tok/s aggregate** (~30 tok/s per stream), no OOM / no preemption failures
- Single-stream decode on this same profile: ~67 tok/s (code)

This is the right shape when most sessions sit far below 1M (typical agent
turns) but you still want the 1M ceiling available. The newer 2026-06-30
agent-stability checkpoint above is the safer number to cite for Hermes/OpenClaw
harness validation.

> Higher concurrency is not free: under sustained pressure you can see added
> scheduler churn, prefill contention, and KV fragmentation. 1M/6 is validated
> for normal-length agent traffic; for guaranteed deep-context work under load,
> 1M/2 is conservative and 500K/4 is a balanced middle.

## How the KV cache works (why 1M + concurrency is safe)

> [!NOTE]
> `max_model_len` and `max_num_seqs` are ceilings, not reservations. The real
> limit is the sum of live tokens across active requests fitting inside the
> shared KV pool.

Three independent knobs, often confused:

| knob | what it is | this build |
| --- | --- | --- |
| **KV cache pool** | total shared KV memory in tokens, sized from `gpu_memory_utilization` after weights load | ~2.8M tokens on Anemll image (this checkout); ~3.2M on historical Stage-C C12 |
| `max_model_len` | per-request **ceiling** — how long any one request may grow | **1,048,576 (1M)** default |
| `max_num_seqs` | **concurrency cap** — max active sequences the scheduler runs at once | 6 |

The pool is **shared and allocated on demand**: PagedAttention hands KV blocks
to each request as it generates tokens and frees them when it finishes.
`max_model_len` and `max_num_seqs` are **ceilings, not reservations** — vLLM does
NOT pre-allocate `max_num_seqs × max_model_len` of KV. So the real constraint is:

```
sum(live tokens across all active requests) <= KV pool
```

Worked examples at 1M ceiling / 6 slots:

```
6 requests x  50k tokens =  300k   fits easily
6 requests x 200k tokens =  1.2M   fits in the Anemll / C12 pools
6 requests x 500k tokens =  3.0M   near pool capacity depending on image
3 requests x 1M   tokens =  3.0M   near pool capacity depending on image
6 requests x 1M   tokens =  6.0M   impossible — excess requests queue/preempt
```

The boot log's `Maximum concurrency for 1,048,576 tokens per request: ~2.7x`
(Anemll image on this cluster) only means a few *simultaneous full-1M*
requests fit. Agent turns are almost never near 1M, so six normal-length
sessions share the pool while the 1M ceiling stays available for the rare long
one. That is exactly why `1M + max_num_seqs=6` is useful: you are not
reserving 6×1M, you are sharing one pool across short requests under a high
ceiling.

## Gotcha: gibberish, loops, Chinese drift, or prompt/XML leakage

> [!WARNING]
> This failure mode is often caused by stale runtime images, inherited sampling
> defaults, or agent orchestration state. Validate the direct OpenAI-compatible
> API path first, then test the agent harness.

If the model boots and basic prompts like `hi` work, but real agent traffic
randomly turns into repeated characters, Chinese drift, leaked tool/schema XML,
or Telegram-visible junk, do not assume the weights are bad.

On this deployment there are three checks to make before blaming the weights:

1. **Runtime image + DSpark path:** with the Anemll image, confirm both nodes
   run the same tag (`docker image inspect $DSPARK_VLLM_IMAGE`) and that compose
   uses `/usr/local/bin/vllm` (not a Stage-C `/opt/env` path). For historical
   Stage-C builds, also ensure the Keys proposer path under
   `recipe/vllm/v1/spec_decode/dspark_proposer.py` and overlay sources are
   consistent with the image you built.
2. **Model cache on both nodes:** a full offline HF hub cache for
   `deepseek-ai/DeepSeek-V4-Flash-DSpark` must exist on head **and** worker
   (`HF_HUB_OFFLINE=1` once complete). Incomplete caches or online re-downloads
   have filled worker disks and broken TP=2 start.
3. **Decode/fallback safety:** for long OpenAI-compatible agent prompts, avoid
   unstable sampling and hidden fallback transitions. The server keeps
   `--generation-config vllm` and does not install a server-side
   `--override-generation-config`; explicit client request parameters still
   win.

The compose launcher includes `--generation-config vllm`, sets `thinking=false`,
uses DSpark speculative decoding with `MTP_NUM_TOKENS=3` and
`draft_sample_method=probabilistic`, and enables the FlashInfer sampler. For
exact deterministic curl checks, send `temperature: 0` in the request body.

Also clear agent fallback lists during validation. A model that looks fixed in
direct vLLM tests can still appear poisoned if the orchestration layer silently
falls back, reboots a session, or replays a stale prompt/tool transcript into
the visible message stream. Keep OpenClaw/Hermes changes separate from model
runtime validation unless you are deliberately testing that harness.

Validation gates to run after a live fix:

```text
direct vLLM prompts: clean
direct concurrent vLLM prompts: clean
agent harness prompts: clean, DeepSeek, no fallback
MTP3 probabilistic draft sampling active
```

This keeps NVFP4 KV and MTP3. Do not switch to fp8 or drop to a smaller fallback
model just to hide the symptom unless you intentionally accept the context and
quality tradeoff.

## Important Caveat

> [!CAUTION]
> This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good
> 584-byte sparse-MLA cache envelope while routing the runtime through
> `nvfp4_ds_mla`. It is **not** the unresolved true-layout 416-byte NVFP4 kernel
> fix. The true-layout experiments were useful for diagnosis but failed past
> roughly 411 real prompt tokens, so they are intentionally not presented here
> as the reproducible recipe.

## Credits

See [`CREDITS.md`](CREDITS.md) for the full attribution and license notes.

### Special thanks

**[drowzeys ("Keys")](https://github.com/drowzeys/)** — this repo would not run
correctly under real concurrency without Keys' public work. Keys published the
DSpark in-server concurrency patch, the request-stable main-KV slot mapping, the
ragged `query_start_loc` path for mixed prefill/decode batches, and the early
`nvfp4_ds_mla` KV-cache wiring on DGX Spark. Our overlay, bind-mounted
proposer, and measured concurrency numbers all build directly on that
foundation.

### Other contributors

- **[drowzeys](https://github.com/drowzeys/) / Keys concurrency patch:**
  [Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- **[tonyd2wild](https://github.com/tonyd2wild/)** — NVFP4 1M recipe lineage,
  garble-fix launcher defaults, and the non-uniform batch guard we merged into
  the runtime proposer bind-mount
- **Rafael Caricio** — DSpark vLLM integration and deployment work:
  [vllm#1](https://github.com/rafaelcaricio/vllm/pull/1),
  [spark_vllm_docker#1](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
- **Fraser Price** — DeepSeek V4 Flash DSpark model/runtime work:
  [DeepSeek-V4-Flash-DSpark](https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark),
  [dspark-vllm](https://github.com/fraserprice/dspark-vllm)
- **MiaAI-Lab** — two-node DGX Spark packaging and worker-first launch runbook:
  [DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
- **[Anemll](https://github.com/Anemll/dspark-vllm-gx10)** — prebuilt
  `ghcr.io/anemll/dspark-vllm-gx10` vLLM 0.25 image for two-node GB10 / DGX Spark
  with NVFP4 DS-MLA and b12x MoE
- **Upstream foundations** — vLLM, FlashInfer, NVIDIA Blackwell/CUDA/NCCL
  tooling, DeepSeek V4 Flash, and DeepSeek-AI DeepSpec / DSpark research

### MiaAI-Lab contribution

MiaAI-Lab maintains this fork's validated 1M NVFP4-KV recipe, Stage A/B/C
runtime packaging, sanitized two-node launch flow, Keys patch integration, and
compose/start tooling. This checkout defaults to the Anemll prebuilt image while
keeping Stage-C build scripts for optional local rebuilds.

## License Notes

Repo scripts and docs are published under this repo's `LICENSE`. The vLLM
overlay/runtime files are vLLM-derived and retain their Apache-2.0 lineage and
SPDX headers where present. Base images, FlashInfer/TileLang/Triton/CUDA/NCCL,
and model weights are separate upstream artifacts with their own licenses and
usage terms.

## Files

| path | purpose |
| --- | --- |
| `docker-compose.dspark.yml` | two-node vLLM/DSpark service (Anemll image layout by default) |
| `.env.dspark.example` | sanitized cluster template; default image Anemll `0.1.1`, **1M** context |
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test; image must exist on both nodes |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `status-deepseek-v4-flash-dspark.sh` | shows head/worker container state |
| `logs-deepseek-v4-flash-dspark.sh` | tails head/worker DSpark logs |
| `smoke-deepseek-v4-flash-dspark.sh` | direct concurrent OpenAI-compatible smoke test |
| `validate-dspark-config.sh` | renders and checks the local DSpark compose/env config |
| `prepare-dspark-model-cache.sh` | downloads/verifies the model cache |
| `build-dspark-vllm-runtime.sh` | optional Stage-C local image build (not required for Anemll) |
| `recipe/overlay/` | Stage-C DSpark vLLM overlay sources for local image builds |
| `recipe/vllm/v1/spec_decode/dspark_proposer.py` | Stage-C/proposer reference; start script may sync to worker |
| `recipe/nvfp4/Dockerfile.stage-*` | Stage A/B/C NVFP4 image layers for local builds |
| `patches/keys-concurrency.patch` | full path-adjusted Keys concurrency patch reference |
| `vllm_patch_gb10/` | optional experimental GB10 hybrid NVFP4 vLLM plugin |
| `docs/PATCHES.md` | plain-English Patch 1 / Patch 2 / Patch 2b concurrency explanation |
| `scripts/verify-overlay-sources.sh` | checks overlay sources before Stage-C image build |

## Quick Start

Run from the head node.

```bash
cp .env.dspark.example .env.dspark
```

Edit these values for your cluster:

- `WORKER_HOST`
- `WORKER_SCRIPT_DIR` if the worker checkout/deployment path differs from the head
- `MASTER_ADDR`
- `NCCL_IB_HCA`
- `NCCL_SOCKET_IFNAME`
- `NCCL_IB_GID_INDEX`
- `HF_CACHE`
- `WORKER_HF_CACHE` if the worker cache path differs from the head
- `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP` for each node's fabric IP

Example cluster fabric values (edit for your nodes):

```env
WORKER_HOST=10.0.0.2
MASTER_ADDR=10.0.0.1
VLLM_HOST_IP=10.0.0.1
WORKER_VLLM_HOST_IP=10.0.0.2
MASTER_PORT=25000
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
TP_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

Keep these **default** agent-serving knobs unless you are deliberately
experimenting (do not treat a temporary local `MAX_MODEL_LEN` override as the
recipe default):

- `VLLM_HOST=0.0.0.0` if Hermes/OpenClaw or another machine must reach the API
- `MAX_MODEL_LEN=1048576` (**1M**)
- `MAX_NUM_SEQS=6`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MTP_NUM_TOKENS=3`
- `HF_HUB_OFFLINE=1` after both nodes have a full model cache
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`

Pull the default runtime image on **head and worker**:

```bash
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

Optional: build the historical Stage-C image instead:

```bash
./build-dspark-vllm-runtime.sh
# then set DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
```

Prepare the model cache on both nodes (or rsync a verified hub snapshot):

```bash
./prepare-dspark-model-cache.sh
```

Start the service:

```bash
./start-deepseek-v4-flash-dspark.sh
```

Optional experimental GB10 hybrid NVFP4 plugin:

```bash
ENABLE_VLLM_GB10_PATCH=1 ./start-deepseek-v4-flash-dspark.sh
```

When enabled, the launcher syncs `vllm_patch_gb10/` to the worker, mounts it in
both containers, installs it with `pip install -e --no-deps`, sets
`VLLM_PLUGINS=gb10_hybrid_nvfp4`, and starts vLLM with
`--quantization modelopt_gb10_hybrid`. The default is disabled. Tune the
dispatcher threshold with `GB10_HYBRID_NVFP4_M_THRESHOLD`; the default is `128`.

The start script prints the resolved non-secret runtime profile, syncs
compose/env (and related files) to the worker path, validates rendered Docker
Compose on both nodes, starts the worker first, then starts the head and
follows startup logs while waiting for the API. If startup fails, it prints
recent head and worker logs before exiting.

The API serves at:

```text
http://HEAD_NODE_IP:8888/v1
```

For head-node-only tests, set `VLLM_HOST=127.0.0.1`. For Hermes/OpenClaw or
another machine to use the endpoint, keep `VLLM_HOST=0.0.0.0` and control
access at the network/firewall layer.

## Runtime Profile

### C12 Agent-Serving Profile (default: 1M context)

Core vLLM flags (from `docker-compose.dspark.yml`):

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (override with `DSPARK_VLLM_IMAGE`)
- `/usr/local/bin/vllm serve …`
- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla`
- `--block-size 256`
- `--max-model-len 1048576` (**default 1M**)
- `--max-num-seqs 6`
- `--max-num-batched-tokens 8192`
- `--max-cudagraph-capture-size 24` (`max_num_seqs * (MTP_NUM_TOKENS + 1)` → `6 * 4`)
- `--gpu-memory-utilization 0.85`
- `--moe-backend flashinfer_b12x`
- `--async-scheduling`
- `--enable-chunked-prefill`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":${MTP_NUM_TOKENS:-3},"draft_sample_method":"probabilistic"}'`
- `--generation-config vllm`

Key runtime env:

- `DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- `HF_HUB_OFFLINE=1` when hub caches are complete on both nodes
- `ENABLE_VLLM_GB10_PATCH=0` by default; set to `1` to load the optional
  `vllm_patch_gb10/` plugin and add `--quantization modelopt_gb10_hybrid`
- `GB10_HYBRID_NVFP4_M_THRESHOLD=128`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0`
- `B12X_W4A16_TC_DECODE=0`
- `DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc`

### 200k Concurrency Profile

For DSpark concurrency, use the included overlay files with Keys'
concurrency patch and set:

- `MAX_MODEL_LEN=200000`
- `MAX_NUM_SEQS=16`
- `VLLM_USE_B12X_WO_PROJECTION=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`

### 1M Single-Stream Legacy Profile

For conservative single-stream testing, set `MAX_NUM_SEQS=1` and
`VLLM_USE_B12X_WO_PROJECTION=0`. Keep `MTP_NUM_TOKENS=3` unless you are
deliberately running an experiment; the current local runtime uses
probabilistic DSpark draft sampling at MTP3.

## Verify

After launch:

```bash
curl -fsS http://127.0.0.1:8888/v1/models
```

Confirm the returned model entry reports:

```json
"max_model_len": 1048576
```

Then check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency"
```

On the Anemll image at 1M max-model-len / 0.85 GPU util, expect roughly:

```text
GPU KV cache size: approximately 2.8M tokens
Maximum concurrency for 1,048,576 tokens per request: approximately 2.7x
```

Historical Stage-C C12 boots reported ~2–3.2M tokens and ~1.9–3.2x depending on
image and util; always trust the live boot log for your node.

Before pointing an agent harness at the endpoint, run the included smoke test:

```bash
./smoke-deepseek-v4-flash-dspark.sh
```

If direct OpenAI-compatible prompts are clean but an agent still garbles,
investigate the agent session, fallback list, or harness prompt replay before
blaming the DSpark weights.

## Notes

- The old speed checkpoint is single stream, not aggregate throughput.
- The high-concurrency benchmark is aggregate throughput and was validated at
  `max_model_len=200000`, not full 1M context.
- Full context and high concurrency compete for the same KV pool. The C12
  1M profile is intended for normal agent traffic where most sessions sit far
  below the 1M ceiling; it is not twelve simultaneous full-1M requests.
- To combine DSpark concurrency with longer context, pick a lower context
  target first, then raise concurrency slowly while watching boot logs, KV
  allocation, acceptance, and request errors.
- 1M was validated as booted/advertised `max_model_len` with KV headroom and
  short-prompt speed probes. This repo does not claim a full 1M-token retrieval
  or correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The **default** agent-serving profile is `MAX_MODEL_LEN=1048576` (1M),
  `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `GPU_MEMORY_UTILIZATION=0.85`, `MTP_NUM_TOKENS=3`,
  `DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
  `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`,
  `VLLM_USE_B12X_WO_PROJECTION=1`, no generation override, and
  `VLLM_DSV4_B12X_COMPRESSED_MLA=0`. Local `.env.dspark` may temporarily lower
  context (for example 512k) without changing that recipe default.
- Worker-first startup avoids a race during multi-node `mp` initialization and
  validates rendered compose on both nodes before starting containers.
- Requires matching images on both nodes, correct NCCL/RoCE settings, and a
  two-node Blackwell-class/DGX Spark setup.
- It is recommended to **disable earlyoom** on the DGX Spark hosts (`sudo systemctl stop earlyoom && sudo systemctl disable earlyoom`).
  The earlyoom daemon can OOM-kill vLLM worker or head processes under high GPU
  memory pressure (e.g., during concurrent deep-context workloads), even when the
  system has available swap or the OOM is transient. Disabling it avoids spurious
  process termination and service disruption.
- The example template binds to `0.0.0.0:8888` for multi-host agents; set
  `VLLM_HOST=127.0.0.1` for head-only testing and control exposure at the
  firewall.
- The next max-sequence ladder to try is approximately 1.25M, 1.5M, then
  1.75M, with the same boot/log/speed gates. Raw KV math alone is not enough
  because DeepSeek V4 sparse MLA also allocates max-length-dependent workspaces.
