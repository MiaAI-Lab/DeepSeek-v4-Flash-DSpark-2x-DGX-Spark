# DeepSeek V4 Flash 0731 DSpark on 2x DGX Spark

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

> [!IMPORTANT]
> **This is the updated recipe for DeepSeek v4 Flash GA (0731).**

Self-contained two-node DGX Spark recipe for serving `DeepSeek-V4-Flash-0731`
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
When using Stage-C, also merge `docker-compose.stage-c.override.yml` and enable
the Stage-C env block in `.env.dspark` (see [`docs/ENVS.md`](docs/ENVS.md)).

This repo still vendors Keys' DSpark concurrency patch and Stage-C overlay
sources for local image builds and documentation. With the Anemll image, that
logic ships inside the image rather than as a host bind-mount.

> [!NOTE]
> **Environment variables differ by image.** The default Anemll `0.1.1` image does
> **not** register every `VLLM_DSPARK_*` / `VLLM_USE_B12X_WO_PROJECTION` kill-switch
> from the Stage-C overlay. Setting those on Anemll only yields
> `Unknown vLLM environment variable` warnings (no-ops). See
> [`docs/ENVS.md`](docs/ENVS.md). Stage-C users should merge
> `docker-compose.stage-c.override.yml`.


**Default agent-serving profile** (`.env.dspark.example` and README defaults):

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- `ABLITERATED=0` → official [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ `DSPARK_REVISION=9e165c30…` (`ABLITERATED=1` → [Keys abliterated](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32))
- text-only 0731 on `:8888`
- `max_model_len=1048576` (**1M** — keep this as the documented default)
- `max_num_seqs=6`
- `max_num_batched_tokens=8192`
- `kv_cache_dtype=nvfp4_ds_mla`
- `GPU_MEMORY_UTILIZATION_TEXT=0.835` (start exports this as `GPU_MEMORY_UTILIZATION`; do not set `GPU_MEMORY_UTILIZATION` by hand)
- **Available KV (text-only, this cluster @ util 0.835):** ~**18.08 GiB** → **GPU KV cache size ~2,493,464 tokens** (~2.38× concurrency at 1M; trust the live boot log)
- `MTP_NUM_TOKENS=5` (checkpoint `dspark_block_size` is 5; k must be ≥ 5)
- `DEFAULT_THINKING=max` (`off`, `low`, `high`, or `max`; request-level overrides still win)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (keep regular CUDA graphs; Anemll auto-enables the slower breakable path when unset)
- API bind address `0.0.0.0:8888`

Local `.env.dspark` may lower `MAX_MODEL_LEN` (for example `512000`) or raise
`MTP_NUM_TOKENS` / util for a specific cluster without changing the recipe
default.

> [!IMPORTANT]
> This profile is meant for real deep-context agent serving: up to **1M tokens
> per separate session** with `MAX_NUM_SEQS=6`. The KV cache is a shared pool,
> so six sessions do not each reserve 1M tokens up front. Normal agent
> sessions can run concurrently while retaining the 1M ceiling for unusually
> long requests.

> [!IMPORTANT]
> For long coding tasks and big prompts, raise text util (do not set
> `GPU_MEMORY_UTILIZATION` by hand — start maps it from
> `GPU_MEMORY_UTILIZATION_TEXT`):
>
> ```env
> MAX_MODEL_LEN=1048576
> MAX_NUM_SEQS=4
> MAX_NUM_BATCHED_TOKENS=16384
> GPU_MEMORY_UTILIZATION_TEXT=0.87
> ```

This repo documents the validated 0731 1M NVFP4 agent profile, historical
preview / Stage-C checkpoints, and the current Anemll prebuilt runtime:

- default checkpoint `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- default `max_model_len=1048576` (1M), `max_num_seqs=6`, `kv_cache_dtype=nvfp4_ds_mla`, `MTP_NUM_TOKENS=5`
- default image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- text-only Available KV @ util **0.835**: ~**18.08 GiB** / ~**2.49M** tokens (see live boot log below)
- 0731 is **text-only** on `:8888`
- 900K acceptance + concurrency/prefill sweep published under `results/`
- historical Stage-C C12 pool: `3,225,280 tokens`
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

### DeepSeek V4 Flash 0731 lane (current default)

Current default from [#14](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/pull/14).
The 0731 checkpoint keeps the same DSpark block-size-5 structure and 1M
context ceiling as the preview checkpoint; message encoding is not identical.
See [`docs/DEEPSEEK_V4_FLASH_0731.md`](docs/DEEPSEEK_V4_FLASH_0731.md) for the
pinned revision, encoder install / reasoning-effort compatibility layer,
validation requirements, and full sweep.

Runtime:

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- model id: `deepseek-ai/DeepSeek-V4-Flash-0731` (revision `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`)
- served model name: `deepseek-v4-flash-0731`
- `kv_cache_dtype=nvfp4_ds_mla`
- recipe defaults: `max_model_len=1048576`, `max_num_seqs=6`,
  `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.835`, `MTP_NUM_TOKENS=5`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- compose installs checkpoint `encoding/encoding_dsv4.py` into vLLM on both ranks
  and applies the Issue #21 `encode_arguments_to_dsml` dict-args hotfix
  (override with `DSPARK_ENCODING_FILE` when needed)
- `--moe-backend flashinfer_b12x`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `HF_HUB_OFFLINE=1` recommended after both nodes have a complete hub cache
- fabric: explicit `VLLM_HOST_IP` / `WORKER_VLLM_HOST_IP`, plus matching
  `NCCL_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`

Live boot evidence on this cluster (**text-only**, 0731, Anemll `0.1.1`,
`GPU_MEMORY_UTILIZATION_TEXT=0.835`; local knobs then were `MAX_NUM_SEQS=4`,
`MTP_NUM_TOKENS=6`):

```text
Available KV cache memory: 18.08 GiB
GPU KV cache size: 2,493,464 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.38x
Application startup complete.
```

## Benchmarks

### Official decode benchmark (`deepseek-v4-flash-0731`)

Live decode bench against this cluster's OpenAI-compatible endpoint
(`port 8888`, served model `deepseek-v4-flash-0731`): **2048** completion
tokens, concurrency **1–6**, wall time **5m 48s**, status **COMPLETED**.

![Decode benchmark for deepseek-v4-flash-0731](docs/benchmarks.png)

Official numbers from that report (rows shown in the capture):

| Load | TTFT | Streams | Aggregate tok/s | Stream tok/s |
| ---: | ---: | :---: | ---: | ---: |
| x1 | 168 ms | 1/1 | 82.4 | 82.4 |
| x2 | 295 ms | 2/2 | 98.0 | 53.0 |
| x3 | 338 ms | 3/3 | 134.6 | 45.8 |
| x4 | 5.36 s | 4/4 | 120.4 | 33.6 |

- **Aggregate** — total tok/s across all concurrent streams
- **Stream** — per-stream average tok/s

Peak aggregate in this capture is **134.6 tok/s** at x3. At x4, TTFT jumps to
**5.36 s** while aggregate falls to **120.4 tok/s** and per-stream average to
**33.6 tok/s**. The run was configured through concurrency 6; the published
screenshot includes the completed x1–x4 rows above.

### Prefill / concurrency sweep (PR #14)

PR #14 live validation (recipe defaults, MTP-5 / seqs=6 / util=0.80):

- advertised context: 1,048,576 tokens
- 900K request: 899,994 prompt tokens, 1,028.85 s TTFT, ~874.8 prefill tok/s,
  requested sentinel returned
- clean system/user role boundary; reasoning emitted separately from final content
- `deepseek_v4` tool parser returned valid OpenAI function arguments
- multi-turn role handling passed

Throughput highlights (medians; full table + raw JSON in
[`docs/DEEPSEEK_V4_FLASH_0731.md`](docs/DEEPSEEK_V4_FLASH_0731.md) and
`results/deepseek-v4-flash-0731-2x-dgx-spark.json`):

| Prompt | Concurrency | Prefill tok/s | Decode tok/s | Aggregate tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 447 | 75.4 | 69.1 |
| 256 | 6 | 197 | 36.9 | 191.2 |
| 2,048 | 1 | 2,563 | 68.8 | 62.0 |
| 2,048 | 6 | 342 | 34.7 | 143.7 |
| 8,192 | 1 | 1,713 | 73.9 | 43.7 |
| 8,192 | 6 | 454 | 23.6 | 73.1 |
| 32,768 | 1 | 1,428 | 64.0 | 16.6 |
| 32,768 | 6 | 550 | 10.8 | 27.9 |
| 131,072 | 1 | 1,665 | 65.2 | 5.9 |
| 131,072 | 2 | 1,306 | 30.9 | 6.6 |

Regular CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`) vs Anemll auto breakable
graphs, matched natural-completion probe at full 1M context:

| Mode | Breakable graphs | Regular graphs | Change |
| --- | ---: | ---: | ---: |
| C1 decode, warm median | 74.55 tok/s | 95.9 tok/s | +28.6% |
| C2 aggregate decode, median | 134.2 tok/s | 151.8 tok/s | +13.1% |
| C4 aggregate decode | not measured | 263.7 tok/s | - |
| C6 aggregate decode | not measured | 340.5 tok/s | - |

Reproduce the sweep:

```bash
python3 scripts/benchmark-0731.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --output results/deepseek-v4-flash-0731.json
```

### Historical preview Anemll lane (`DeepSeek-V4-Flash-DSpark`)

Earlier preview-checkpoint validation with the same Anemll image and this
repo's compose/start scripts (TP=2, two nodes). Kept for comparison; not the
current default.

Runtime:

- image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
- model id: `deepseek-ai/DeepSeek-V4-Flash-DSpark` (HF cache under `HF_CACHE`)
- served model name: configurable via `SERVED_MODEL_NAME` (example: `deepseek-v4-flash`)
- `kv_cache_dtype=nvfp4_ds_mla`
- profile used then: `max_model_len=1048576`, `max_num_seqs=6`,
  `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.85`, `MTP_NUM_TOKENS=3`
- `--moe-backend flashinfer_b12x`
- `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_USE_B12X_WO_PROJECTION=1`

Boot evidence on this cluster (preview checkpoint, 1M max-model-len profile):

```text
Available KV cache memory: 19.03 GiB
GPU KV cache size: 2,826,378 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.70x
Application startup complete.
```

Direct API smoke: `/v1/models` HTTP 200 and OpenAI-compatible chat completions
returned non-empty assistant content on both head and worker ranks.

### Historical real-life decode speed (preview Anemll lane)

Streaming decode-only bench on the preview Anemll lane with **agent /
file-writing** prompts (`max_tokens=512`, temperature 0, unique nonce per
request, 3 trials, median by aggregate). Prefill and the first token are
**excluded**.

| Metric | Formula |
| --- | --- |
| Per-stream decode tok/s | `(completion_tokens − 1) / (t_last − t_first)` |
| Aggregate decode tok/s | `sum(completion_tokens − 1) / (max t_last − min t_first)` |

| Concurrency | Success | Agg decode tok/s | Mean stream decode tok/s | Decode window (s) | Decode tokens |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 66.6 | 66.6 | 7.67 | 511 |
| 2 | 2/2 | 93.3 | 47.2 | 10.95 | 1022 |
| 3 | 3/3 | 92.8 | 31.9 | 16.52 | 1533 |
| 4 | 4/4 | 123.8 | 32.8 | 16.51 | 2044 |
| 5 | 5/5 | 121.1 | 25.7 | 21.11 | 2555 |
| 6 | 6/6 | 153.7 | 26.8 | 19.94 | 3066 |

Trial aggregates (decode tok/s): C1 `[66.5, 69.4, 66.6]`, C2 `[92.1, 95.6, 93.3]`,
C3 `[89.4, 92.8, 93.5]`, C4 `[129.1, 123.8, 121.7]`, C5 `[125.0, 121.1, 111.6]`,
C6 `[153.7, 148.8, 157.0]`.

**Agg decode** is fleet generation after first tokens; **mean stream** is what
one concurrent chat feels like once tokens start (~67 tok/s alone, ~27 at C=6).
C3 ≈ C2 and C5 ≈ C4 on aggregate under multi-stream contention while per-stream
decode falls.

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
| **KV cache pool** | total shared KV memory in tokens, sized from `gpu_memory_utilization` after weights load | ~2.49M tokens on 0731 / Anemll (this cluster @ util 0.835); ~2.8M on preview Anemll @ util 0.85; ~3.2M on historical Stage-C C12 |
| `max_model_len` | per-request **ceiling** — how long any one request may grow | **1,048,576 (1M)** default |
| `max_num_seqs` | **concurrency cap** — max active sequences the scheduler runs at once | 6 (recipe default; this cluster currently runs 4) |

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

The boot log's `Maximum concurrency for 1,048,576 tokens per request: ~2.4x`
(0731 on this cluster) only means a few *simultaneous full-1M*
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
   `deepseek-ai/DeepSeek-V4-Flash-0731` must exist on head **and** worker
   (`HF_HUB_OFFLINE=1` once complete). Incomplete caches or online re-downloads
   have filled worker disks and broken TP=2 start. Confirm the snapshot also
   contains `encoding/encoding_dsv4.py` so compose can install the 0731 encoder.
3. **Decode/fallback safety:** for long OpenAI-compatible agent prompts, avoid
   unstable sampling and hidden fallback transitions. The server keeps
   `--generation-config vllm` and does not install a server-side
   `--override-generation-config`; explicit client request parameters still
   win.

The compose launcher includes `--generation-config vllm` and defaults to
`DEFAULT_THINKING=max` (from `.env.dspark.example`). It validates `off`, `low`,
`high`, or `max` and translates the selected mode into vLLM chat-template
kwargs; explicit request-level overrides still win. It uses DSpark speculative
decoding with `MTP_NUM_TOKENS=5` and
`draft_sample_method=probabilistic`, keeps regular CUDA graphs via
`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, and enables the FlashInfer sampler. For
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
MTP5 probabilistic draft sampling active
reasoning / tool-call encoding semantics intact on 0731
```

This keeps NVFP4 KV and MTP5. Do not switch to fp8 or drop to a smaller fallback
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

**[@u1tra_instinct](https://x.com/u1tra_instinct)** — special thanks for the
optional abliterated weights path used when `ABLITERATED=1`:
[drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32).

### Other contributors

- **[@u1tra_instinct](https://x.com/u1tra_instinct)** — optional abliterated
  weights:
  [keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32)
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
| `docker-compose.dspark.yml` | two-node vLLM/DSpark service (Anemll image layout by default; installs 0731 encoder) |
| `.env.dspark.example` | sanitized cluster template; default image Anemll `0.1.1`, **0731** / **1M** context |
| [`docs/DEEPSEEK_V4_FLASH_0731.md`](docs/DEEPSEEK_V4_FLASH_0731.md) | 0731 checkpoint, encoder notes, sweep method, and measured results |
| [`docs/benchmarks.png`](docs/benchmarks.png) | official 0731 decode-benchmark capture (2048 tok, concurrency sweep) |
| [`docs/ENVS.md`](docs/ENVS.md) | Anemll vs Stage-C env registry matrix (unknown-`VLLM_*` warnings) |
| `docker-compose.stage-c.override.yml` | optional Stage-C-only env injection |
| `start-deepseek-v4-flash-dspark.sh` | worker-first launch and smoke test; image must exist on both nodes |
| `stop-deepseek-v4-flash-dspark.sh` | stops head and worker services |
| `status-deepseek-v4-flash-dspark.sh` | shows head/worker container state |
| `logs-deepseek-v4-flash-dspark.sh` | tails head/worker DSpark logs |
| `smoke-deepseek-v4-flash-dspark.sh` | direct concurrent OpenAI-compatible smoke test |
| `validate-dspark-config.sh` | renders and checks the local DSpark compose/env config |
| `prepare-dspark-model-cache.sh` | downloads/verifies 0731 on head **and** worker |
| `scripts/benchmark-0731.py` | streaming concurrency/prefill sweep for the 0731 endpoint |
| `results/deepseek-v4-flash-0731-2x-dgx-spark.json` | published two-Spark 0731 sweep measurements |
| `build-dspark-vllm-runtime.sh` | optional Stage-C local image build (not required for Anemll) |
| `recipe/overlay/` | Stage-C DSpark vLLM overlay sources for local image builds |
| `recipe/vllm/v1/spec_decode/dspark_proposer.py` | Stage-C/proposer reference; start script may sync to worker |
| `recipe/nvfp4/Dockerfile.stage-*` | Stage A/B/C NVFP4 image layers for local builds |
| `patches/hotfix-nvfp4-ds-mla-issue22.sh` | Issue #22: route `nvfp4_ds_mla` to the fast FP8 MLA kernel (auto on start) |
| `patches/hotfix-encoding-dsv4-issue21.py` | Issue #21: fix dict tool-call args in `encode_arguments_to_dsml` (runs at encoder install) |
| `patches/keys-concurrency.patch` | full path-adjusted Keys concurrency patch reference |
| `scripts/test-encoding-dsv4-issue21.py` | unit test for the Issue #21 encoder hotfix |
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
- `NCCL_SOCKET_IFNAME` (and matching `TP_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`, or leave those unset so compose inherits the NCCL IF)
- `NCCL_IB_GID_INDEX` (not always 0 — match your RoCE GID)
- `HF_CACHE`
- `WORKER_HF_CACHE` if the worker cache path differs from the head
- `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP` for each node's fabric IP

Example cluster fabric values (edit for your nodes — f0 vs f1 and GID index vary):

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

- `ABLITERATED=0` (official) or `1` (Keys abliterated) — see [Checkpoint](#checkpoint-official-vs-abliterated)
- `DSPARK_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb` (official tested pin; clear to follow `main`)
- `SERVED_MODEL_NAME=deepseek-v4-flash-0731`
- `VLLM_HOST=0.0.0.0` if Hermes/OpenClaw or another machine must reach the API
- `VLLM_PORT=8888`
- `MAX_MODEL_LEN=1048576` (**1M**)
- `MAX_NUM_SEQS=6`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `GPU_MEMORY_UTILIZATION_TEXT=0.835`
- `MTP_NUM_TOKENS=5`
- `DEFAULT_THINKING=max`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `HF_HUB_OFFLINE=1` after both nodes have a full model cache
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_USE_B12X_MOE=1`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`

Pull the default runtime image on **head and worker**:

```bash
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

Optional: build the historical Stage-C image instead:

```bash
./build-dspark-vllm-runtime.sh
# then set DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
# and IMAGE_PYTHON=/opt/env/bin/python for prepare-dspark-model-cache.sh
```

Prepare the model cache on both nodes (or rsync a verified hub snapshot):

```bash
./prepare-dspark-model-cache.sh              # asks: official (0) or abliterated (1)
./prepare-dspark-model-cache.sh --official   # non-interactive
./prepare-dspark-model-cache.sh --abliterated
./prepare-dspark-model-cache.sh --yes        # use ABLITERATED from .env.dspark
```

The script writes your choice back to `ABLITERATED=` in `.env.dspark`, then
downloads that checkpoint on head **and** worker.
It forces HF online for the download even when `.env.dspark` has
`HF_HUB_OFFLINE=1` (correct for serve after the cache is warm).
Use `IMAGE_PYTHON=/usr/bin/python3` on the Anemll image (default);
Stage-C needs `IMAGE_PYTHON=/opt/env/bin/python`.

### Checkpoint (official vs abliterated)

| `ABLITERATED` | Weights used by start / prepare |
|---------------|----------------------------------|
| `0` | Official: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ `DSPARK_REVISION` (default **`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`**) |
| `1` | Abliterated: [`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32) (optional `DSPARK_REVISION_ABLITERATED`) |

The abliterated lane uses this Hugging Face checkpoint (do not substitute a
different abliterated id unless you intentionally override
`DSPARK_MODEL_ABLITERATED`):

https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32

Special thanks to [@u1tra_instinct](https://x.com/u1tra_instinct) for that
abliterated weights path.

`start-deepseek-v4-flash-dspark.sh` sets `DSPARK_MODEL` from this flag (do not
set `DSPARK_MODEL` by hand). Prepare and serve both honor the revision pin
(`snapshot_download(revision=…)` + `vllm serve --revision …`, and prepare also
writes `refs/main` → that commit so offline hub lookup stays consistent). Clear
`DSPARK_REVISION=` to follow tip of `main`. Override HF ids with
`DSPARK_MODEL_OFFICIAL` / `DSPARK_MODEL_ABLITERATED` if needed. After flipping
the flag or revision, stop + start (and run prepare if that snapshot is not
cached yet).

Start the service:

```bash
./start-deepseek-v4-flash-dspark.sh
```

The API bind address and port can be overridden for one launch without editing
`.env.dspark`:

```bash
./start-deepseek-v4-flash-dspark.sh --host 0.0.0.0 --port 9000
```

These flags override `VLLM_HOST` and `VLLM_PORT` from `.env.dspark`. When the
bind address is a wildcard, startup health checks still connect through
`127.0.0.1` on the selected port.

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
http://HEAD_NODE_IP:VLLM_PORT/v1
```

`VLLM_PORT` defaults to `8888`. For head-node-only tests, set
`VLLM_HOST=127.0.0.1`. For Hermes/OpenClaw or
another machine to use the endpoint, keep `VLLM_HOST=0.0.0.0` and control
access at the network/firewall layer.

## Pi reasoning controls (off / low / high / max)

The 0731 checkpoint has no Hugging Face Jinja `chat_template`.
`--tokenizer-mode deepseek_v4` instead calls the checkpoint's installed
`encoding/encoding_dsv4.py`, which supports `off`, `low`, `high`, and `max`.
The recipe defaults to `DEFAULT_THINKING=max` (full effort). Set `low` (or
`off` / `high`) in `.env.dspark` if you want a quieter base mode. Clients should
still send an explicit request-level override when they require deterministic
behavior.

A ready-to-copy pi configuration is provided in
[`pi-models.dspark.example.json`](pi-models.dspark.example.json):

```bash
mkdir -p ~/.pi/agent
cp pi-models.dspark.example.json ~/.pi/agent/models.json
# If pi runs away from the head node, replace 127.0.0.1 with HEAD_NODE_IP.
```

Select the supported modes with pi's normal thinking control:

```bash
pi --model local-dspark/deepseek-v4-flash-0731 --thinking off
pi --model local-dspark/deepseek-v4-flash-0731 --thinking low
pi --model local-dspark/deepseek-v4-flash-0731 --thinking high
pi --model local-dspark/deepseek-v4-flash-0731 --thinking max
```

`DEFAULT_THINKING` and pi use the same mapping. The pi configuration hides the
unsupported `minimal`, `medium`, and `xhigh` levels:

| Pi level | vLLM request |
|---|---|
| `off` | `chat_template_kwargs: {"thinking": false}` |
| `low` | `chat_template_kwargs: {"thinking": true, "reasoning_effort": "low"}` |
| `high` | `chat_template_kwargs: {"thinking": true, "reasoning_effort": "high"}` |
| `max` | `chat_template_kwargs: {"thinking": true, "reasoning_effort": "max"}` |

Do not use pi's generic top-level OpenAI `reasoning_effort` mapping for this
endpoint. The specialized DeepSeek V4 tokenizer reads these values from
`chat_template_kwargs`. vLLM returns the generated reasoning in its `reasoning`
stream field; pi recognizes that field, stores it as a thinking block, and
replays it as `reasoning`. vLLM normalizes that to `reasoning_content` before
the custom encoder runs, so tool-call reasoning is not lost.

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
- `--gpu-memory-utilization` from `GPU_MEMORY_UTILIZATION_TEXT` (**0.835**; do not set `GPU_MEMORY_UTILIZATION` by hand)
- `--moe-backend flashinfer_b12x`
- `--async-scheduling`
- `--enable-chunked-prefill`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":${MTP_NUM_TOKENS:-3},"draft_sample_method":"probabilistic"}'`
- `--generation-config vllm`

Key runtime env:

- `ABLITERATED=0` (official 0731) or `1` (Keys abliterated) — start sets `DSPARK_MODEL`
- `GPU_MEMORY_UTILIZATION_TEXT=0.835`
- `DEFAULT_THINKING=max`
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
`VLLM_USE_B12X_WO_PROJECTION=0`. Keep `MTP_NUM_TOKENS=5` unless you are
deliberately running an experiment; the current recipe uses probabilistic
DSpark draft sampling at MTP5 (checkpoint `dspark_block_size` is 5).

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

On the Anemll image with the **text-only** 0731 1M profile
(`GPU_MEMORY_UTILIZATION_TEXT=0.835`), expect roughly (trust the live boot log;
util / MTP / `MAX_NUM_SEQS` change the pool):

```text
Available KV cache memory: approximately 18 GiB
GPU KV cache size: approximately 2.5M tokens
Maximum concurrency for 1,048,576 tokens per request: approximately 2.4x
```

Historical preview Anemll boots at util 0.85 reported ~2.8M tokens / ~2.7x.
Historical Stage-C C12 boots reported ~2–3.2M tokens and ~1.9–3.2x depending on
image and util.

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
- 1M was validated as booted/advertised `max_model_len` with KV headroom.
  PR #14 additionally completed a 900K acceptance request and a
  concurrency/prefill sweep through 128K prompts. This repo still does not
  claim a full 1M-token retrieval or correctness benchmark.
- The measured probes were p256/p512 with g64/g256. Rebenchmark if you change
  sampling, batching, context length, WO projection, compressed MLA, or the
  confidence scheduler.
- The **default** agent-serving profile uses `ABLITERATED=0` →
  `deepseek-ai/DeepSeek-V4-Flash-0731`,
  `SERVED_MODEL_NAME=deepseek-v4-flash-0731`,
  `MAX_MODEL_LEN=1048576` (1M), `MAX_NUM_SEQS=6`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `GPU_MEMORY_UTILIZATION_TEXT=0.835`, `MTP_NUM_TOKENS=5`,
  `DEFAULT_THINKING=max`, text-only 0731 on `:8888`,
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0`,
  `DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
  `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_USE_B12X_MOE=1`, no generation override.
  Local `.env.dspark` may temporarily lower context (for example 512k) or raise
  MTP / util without changing that recipe default.
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

## Experimental: Vision (optional)

> [!WARNING]
> **Not part of the default text-only ship.** The supported product profile is
> 0731 on `:8888` at util **0.835**. Vision coexist on GB10 is memory-tight and
> still experimental — expect OOMs if you raise main util while VL is up, or if
> you start VL before main has fully settled and host RAM has recovered.

0731 stays **text-only**. Optional image input is a separate **Qwen3-VL-4B**
sidecar on `:8889` (TP=2) plus the **`ds4f-vision`** MCP tools (or
`scripts/vision-reason.py`). There is no native MoonViT lane.

### Enable (experimenters)

In `.env.dspark` (see commented knobs in `.env.dspark.example`):

```env
ENABLE_VL_SIDECAR=1
GPU_MEMORY_UTILIZATION_VISION=0.80
PREPARE_VL_SIDECAR_MODEL=1
# typical VL defaults:
# VL_SIDECAR_GPU_UTIL=0.04
# VL_SIDECAR_MAX_MODEL_LEN=32768
# VL_SIDECAR_KV_CACHE_DTYPE=int4_per_token_head
# VL_SIDECAR_ATTENTION_BACKEND=TRITON_ATTN
```

Then:

```bash
./prepare-dspark-model-cache.sh          # 0731 + VL weights (head + worker)
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh      # 0731 @ vision util, then VL, then MCP
```

| Flag | Main util | What starts |
|------|-----------|-------------|
| `0` (default ship) | `GPU_MEMORY_UTILIZATION_TEXT` (**0.835**) | 0731 `:8888` only |
| `1` (experimental) | `GPU_MEMORY_UTILIZATION_VISION` (**0.80**) | 0731 + Qwen3-VL `:8889` + MCP |

Rough measured KV on this cluster (Anemll `0.1.1`): text-only ~**18 GiB / ~2.5M**
tokens; vision mode main ~**13 GiB / ~1.4M** + VL ~**1.5 GiB / ~84k** (util
**0.04**, int4 KV, 32k). True Qwen `--kv-cache-dtype nvfp4` needs FlashInfer
SM100; GB10 is SM12.1 — use `int4_per_token_head` + `TRITON_ATTN`.

**Practical tip:** if VL OOMs on NCCL/CUDA after a full start, leave main up,
`docker compose … -f docker-compose.vl-sidecar.yml down` on **both** nodes
(**without** `--remove-orphans`), wait until worker available RAM recovers, then
bring VL up worker-first. Do not use `--remove-orphans` next to a live 0731.

### More detail

| Path | Purpose |
|------|---------|
| [`docker-compose.vl-sidecar.yml`](docker-compose.vl-sidecar.yml) | VL TP=2 compose |
| [`plugins/dspark_vision_mcp/`](plugins/dspark_vision_mcp/) | MCP package + harness install notes |
| [`scripts/install-ds4f-vision-mcp.sh`](scripts/install-ds4f-vision-mcp.sh) | register `ds4f-vision` (gated on flag; `--force` to override) |
| [`scripts/vision-reason.py`](scripts/vision-reason.py) | CLI two-pass: sidecar extract → 0731 reason |
| [`results/vl-nvfp4-coexist-2026-08-11.md`](results/vl-nvfp4-coexist-2026-08-11.md) | coexist chase notes / evidence |
