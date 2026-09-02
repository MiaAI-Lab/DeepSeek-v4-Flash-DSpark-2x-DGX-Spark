# DeepSeek V4 Flash Vision-Exp DSpark on 2x DGX Spark

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Two-node DGX Spark recipe for **`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`**: vLLM TP=2,
DSpark speculative decoding, **1M-token** ceiling, `nvfp4_ds_mla` KV. Native
**image** support is a startup hotfix on the Anemll 0.1.1 runtime (ViT + Aligner
from the Vision-Exp checkpoint, OpenAI `image_url` / `<image>path</image>`).
There is **no video encoder** in the official weights; GIF is a still frame.
The old Qwen3-VL sidecar / MCP path is removed.

**Default image:** [`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`](https://github.com/Anemll/dspark-vllm-gx10)

**Numbers:** [results/RESULTS-2026-08-14.md](results/RESULTS-2026-08-14.md) (dated
tables, method, historical lanes). Checkpoint / encoder:
[docs/DEEPSEEK_V4_FLASH_0731.md](docs/DEEPSEEK_V4_FLASH_0731.md).

---

## Quick start

Run everything from the **head** node. You need two DGX Sparks, RoCE/NCCL
working, and the same image on both. By default each node has its own
HuggingFace checkpoint (`prepare` copies onto the worker). Set
`DSPARK_WORKER_HF_NFS=1` to keep weights only on the head: the worker
mounts that cache over NFSv4 on the ConnectX link (same pattern as
Qwen3.8-Flash-vLLM).

1. **Env**

   ```bash
   cp .env.dspark.example .env.dspark
   ```

   Set at least: `WORKER_HOST`, `MASTER_ADDR`, `NCCL_IB_HCA`,
   `NCCL_SOCKET_IFNAME` (and matching `TP_` / `GLOO_` IF names),
   `VLLM_HOST_IP`, `WORKER_VLLM_HOST_IP`, `HF_CACHE`.
   `WORKER_HF_CACHE` is the worker checkpoint path. When `DSPARK_WORKER_HF_NFS=1`,
   it is the worker's local JIT overlay on the NFS mount.
   If the worker checkout is not the same path, set `WORKER_DIR` /
   `WORKER_SCRIPT_DIR`.

   Example fabric (edit f0 vs f1 and GID for your ring):

   ```env
   WORKER_HOST=10.0.0.2
   MASTER_ADDR=10.0.0.1
   VLLM_HOST_IP=10.0.0.1
   WORKER_VLLM_HOST_IP=10.0.0.2
   NCCL_IB_HCA=rocep1s0f1
   NCCL_SOCKET_IFNAME=enp1s0f1np1
   DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
   ```

   Leave serving knobs at the defaults unless you mean to change them.
   Meaningful on/off flags (`ABLITERATED`, thinking, hotfixes) are
   listed under [.env.dspark switches](#envdspark-switches).

2. **Image on both nodes**

   ```bash
   docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
   ```

   Repeat on the worker (or pull there via ssh). Start refuses to launch if
   either node is missing the image.

3. **Weights on the head**

   ```bash
   ./prepare-dspark-model-cache.sh --official
   ```

   Use `--abliterated` or `--yes` (reads `ABLITERATED` from `.env.dspark`).
   Abliterated weights are gated (`HF_TOKEN`). Prepare forces HF
   online even if `HF_HUB_OFFLINE=1`, then you can serve offline.
   Default `DSPARK_WORKER_HF_NFS=0` also downloads onto the worker. After
   the cache is complete, keep `HF_HUB_OFFLINE=1`. See
   [Worker weights over NFS](#worker-weights-over-nfs-optional) to skip the
   second copy.

4. **Optional CPU gates** (no GPU; will not measure tok/s)

   ```bash
   bash scripts/ci-validate.sh
   ```


5. **Start** (worker first, then head)

   ```bash
   ./start-deepseek-v4-flash-dspark.sh
   ```

   One-shot bind override: `./start-deepseek-v4-flash-dspark.sh --host 0.0.0.0 --port 9000`.
   After a reboot, dockerd may already have restored the ranks (`restart: unless-stopped`); start then exits **3** (already running), not 1. That is expected — do not `./stop` unless you want a cold start. systemd: `SuccessExitStatus=3`.

   Optional **three Sparks (TP=3)** is a separate launcher so `.env` cannot flip the 2-node path: `./start-tp3.sh` (needs `WORKER2_HOST`; see [docs/TP3.md](docs/TP3.md)).

6. **Check it is up**

   ```bash
   curl -fsS http://127.0.0.1:8888/v1/models
   ./smoke-deepseek-v4-flash-dspark.sh
   ./status-deepseek-v4-flash-dspark.sh
   ```

   Expect `"id": "deepseek-v4-flash-vision-exp"` and `"max_model_len": 1048576`.
   Boot log (trust the live numbers):

   ```text
   Available KV cache memory: 17.04 GiB
   GPU KV cache size: 2,331,430 tokens
   Maximum concurrency for 1,048,576 tokens per request: 2.22x
   ```

API: `http://HEAD_NODE_IP:8888/v1` (`VLLM_HOST=0.0.0.0` by default).
Head-only tests: `VLLM_HOST=127.0.0.1`.

Day-to-day: `./status-…`, `./logs-…`, `./stop-…`. Disable **earlyoom** on both
hosts or it can kill vLLM under deep-context load.

---

## Default profile

| Knob | Default |
| --- | --- |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| Checkpoint | official Vision-Exp @ `86f746b36186f0e567729a5c06a8c918caba82a9` (`ABLITERATED=0`) |
| Served name | `deepseek-v4-flash-vision-exp` |
| Context ceiling | `MAX_MODEL_LEN=1048576` (1M) |
| Concurrent seqs | `MAX_NUM_SEQS=6` |
| Batch tokens | `MAX_NUM_BATCHED_TOKENS=8192` |
| KV | `nvfp4_ds_mla`, **17.04 GiB / 2,331,430 tokens** on this cluster (util 0.83; Vision-Exp ViT takes more weight RAM than 0731) |
| Spec | `MTP_NUM_TOKENS=6` (≥ `dspark_block_size` 5 and divisible by Vision-Exp `n_predict=3`) |
| Thinking | `DEFAULT_THINKING=max` (`off` / `low` / `high` / `max`) |
| Graphs | `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (keep this; unset is slower) |

`start-*.sh` exports `GPU_MEMORY_UTILIZATION` from
`GPU_MEMORY_UTILIZATION_TEXT`. Do not set
`GPU_MEMORY_UTILIZATION` by hand.

`max_model_len` and `max_num_seqs` are **ceilings**, not reservations. The
limit is `sum(live tokens) ≤ KV pool`. Six normal agent turns fit; six
simultaneous full-1M requests do not. See [How the KV cache works](#how-the-kv-cache-works-why-1m--concurrency-is-safe).

Long coding / big prompts (optional, still 1M ceiling):

```env
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=16384
GPU_MEMORY_UTILIZATION_TEXT=0.87
```

---

## .env.dspark switches

Copy [`.env.dspark.example`](.env.dspark.example) → `.env.dspark`. Start syncs
it to the worker. **Restart both ranks** after a flip (`./stop-…` then
`./start-…`). Do not set `DSPARK_MODEL` or `GPU_MEMORY_UTILIZATION` by hand.

NCCL/RoCE, CUDA arch, and compile knobs stay in the example file — they are
cluster wiring, not product switches. Full Anemll vs Stage-C matrix:
[docs/ENVS.md](docs/ENVS.md).

### Weights

| Variable | Default | What it does |
| --- | --- | --- |
| **`ABLITERATED`** | `0` | **`0`** = official [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) @ `DSPARK_REVISION`. **`1`** = [Keys abliterated](https://huggingface.co/drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit). Start and prepare pick the HF id from this flag. Gated; `prepare --abliterated` needs `HF_TOKEN`. |
| `DSPARK_REVISION` | `86f746b36186f0e567729a5c06a8c918caba82a9` | Official Vision-Exp pin. Empty = tip of `main`. |
| `DSPARK_REVISION_ABLITERATED` | empty | Abliterated pin. Empty = tip of that repo. |
| `DSPARK_MODEL_OFFICIAL` / `DSPARK_MODEL_ABLITERATED` | the two HF ids above | Override only if you intentionally swap the repo id. Do not point this at the 0731 ablit dump — that drops `image_url`. |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-vision-exp` | Name clients send as `model`. |
| `HF_HUB_OFFLINE` | `1` | `1` after the hub cache is warm. Prepare forces online for the download. |
| `DSPARK_WORKER_HF_NFS` | `0` | **`0`** (default) = bind `WORKER_HF_CACHE` as a second copy (`prepare` downloads on the worker). **`1`** = worker mounts head `HF_CACHE` over NFSv4 on ConnectX (no local checkpoint). |

Flip `ABLITERATED` like this:

```bash
# in .env.dspark
ABLITERATED=1

./prepare-dspark-model-cache.sh --yes    # or --abliterated / --official
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

`--official` writes `ABLITERATED=0`; `--abliterated` writes `1`.

### Worker weights over NFS (optional)

Default is **off**: `DSPARK_WORKER_HF_NFS=0`. `prepare` downloads the checkpoint
onto the worker as well.

Set `DSPARK_WORKER_HF_NFS=1` in `.env.dspark` to skip that second Hub download.
Start then exports the head `HF_CACHE` via NFSv4 on `NCCL_SOCKET_IFNAME`
(ConnectX). A live exporter on that address is reused (for example Qwen's
`vllm-fn-nfs`); otherwise start brings up `dspark-nfs`. The worker Docker
volume `dspark-hf` mounts the share read-only. Triton, TileLang, vLLM,
FlashInfer, CuTe, and NCCL-FR caches stay on the worker host as overlays
under `WORKER_HF_CACHE`. `./stop-deepseek-v4-flash-dspark.sh --nfs` tears
down only `dspark-nfs`, not Qwen's share.

```env
DSPARK_WORKER_HF_NFS=1
# NFS_SERVER_IP=10.0.22.1   # optional; default is IPv4 on NCCL_SOCKET_IFNAME
```

If official Vision-Exp is already cached, overlay the 26 edited shards
(~87 GiB) instead of re-fetching the full ~157 GiB dump:

```bash
python3 scripts/overlay-vision-exp-ablit-cache.py
```

The 0731 abliterated freeze stays on branch `0731-ablit`.

Images: OpenAI `image_url` (JPEG/PNG/GIF/WebP; GIF is still-frame). No video.
Images belong in **`user` messages only**. A structured `image` / `image_url`
part (or the raw `<｜deepseek_image｜>` token) on `system`, `assistant`,
`tool`, or `function` returns HTTP 400. Tool/function *text* that quotes
`<image>` tags is allowed; putting a vision-tool result on `role: "tool"` as
`image_url` is not. That 400 is not retried by typical OpenAI clients, and
the rejected turn stays in history, so later messages keep failing until
that tool message is dropped ([issue #178](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/178)).
Put screenshots on a **`user`** turn instead. Default cap 8 images
(`LIMIT_MM_PER_PROMPT` is JSON `{"image":8}`; `image=8` is converted). Example:

```json
{"model":"deepseek-v4-flash-vision-exp","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
  {"type":"text","text":"What is in this picture?"}
]}]}
```

Update the pin like this:

```bash
# in .env.dspark
DSPARK_REVISION=<commit>

./prepare-dspark-model-cache.sh --yes
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

### Thinking, API

| Variable | Default | What it does |
| --- | --- | --- |
| **`DEFAULT_THINKING`** | `max` | `off` / `low` / `high` / `max`. Request-level `chat_template_kwargs` still wins. |
| `VLLM_HOST` | `0.0.0.0` | `127.0.0.1` for head-only tests. |
| `VLLM_PORT` | `8888` | Or `./start-… --port 9000` for one launch. |

An explicit `thinking_token_budget` is **off unless you set
`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`** (then opt-in per request). Default
stock V2 rejects the field (HTTP 400). `DEFAULT_THINKING=max` still needs a
generous `max_tokens` or that budget hotfix, or thinking won't end. See
[Thinking-token budgets](#thinking-token-budgets).

### Serve shape (not on/off, but the knobs that change the lane)

| Variable | Default | What it does |
| --- | --- | --- |
| `MAX_MODEL_LEN` | `1048576` | Per-request ceiling (1M). `200000` is the high-concurrency / Keys profile. |
| `MAX_NUM_SEQS` | `6` | Concurrent slots. `16` only with the 200K + Stage-C path. |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill tokens per step. `16384` for big-prompt coding. |
| `LONG_PREFILL_TOKEN_THRESHOLD` | `1024` | Issue **#27** chunk cap. `0` lets one prefill eat the whole batch (decode starves). `2048` costs ~1.5 GB of head-node host RAM on GB10 (measured 2026-09-02), keep 1024. |
| `DSPARK_MAX_INFLIGHT_PREFILLS` | `2` | Issue **#27** in-flight partial prefills (1–3). `2` since the 2026-09-02 A/B ([docs/CLAUDE/ab-results-2026-09-03.md](docs/CLAUDE/ab-results-2026-09-03.md)): a 4 × 8K wave sees first-token spread 9.1 s vs 14.6 s and +12 % aggregate, single-stream and c=6 decode unchanged. `1` restores the strictly serialized default. |
| `GPU_MEMORY_UTILIZATION_TEXT` | `0.835` | Main GPU util / KV pool size. Larger = bigger KV pool. |
| `LIMIT_MM_PER_PROMPT` | `{"image":8}` | Max images per request (Vision-Exp native `image_url`). `image=8` is converted to JSON for Anemll argparse. No video. |
| `MTP_NUM_TOKENS` | `6` | DSpark draft depth. Vision-Exp `n_predict=3` so k must be ≥ 5 and divisible by 3. Capture size = `seqs * (k+1)` padded up to a multiple of 8 (48 at 6×6). |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | **Keep 0.** Unset enables Anemll’s slower breakable graphs. |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `4096` | Issue **#26** SWA prefix-cache spacing. Leave unless you are debugging warm-cache hits. |

### Hotfixes and diagnostics (on by default unless you skip)

| Variable | Default | What it does |
| --- | --- | --- |
| `DSPARK_SUPPRESS_STOPS_IN_REASONING` | `1` | `0` = client `stop` strings can fire inside `<think>` (blank `content`). |
| `DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX` | `0` | `1` = do not apply that patch at all. |
| `DSPARK_SKIP_ISSUE22_HOTFIX` | `0` | `1` = skip the `nvfp4_ds_mla` long-context decode fix. Don’t, on this recipe. |
| `DSPARK_SKIP_HOTFIX` | `0` | `1` = skip the six v0.27 perf backports only (#22 still applies). |
| `DSPARK_SKIP_SPIN_WAIT_HOTFIX` | `0` | `1` = leave vLLM shm `busy_loop_s=1s` (issue **#79** P-core spin on TP=2). |
| `DSPARK_ISSUE43_SCHED_DIAG` | `0` | `1` = one scheduler line per step in the vLLM log (mixed prefill/decode). |
| `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX` | `0` | `1` = apply GPU `thinking_token_budget` at boot (fail-closed). Default stock V2; omit-field clients do not need this ([Issue #66](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/66)). |
| `DSPARK_ENABLE_SP_INDEXER` | `0` | `1` = sequence-parallel Lightning indexer for prefill chunks ≥ `DSPARK_SP_INDEXER_MIN_KEYS` (8192) compressed keys: each TP rank scores half the keys, exact top-k merge. Long-context TTFT lever ([docs/PATCHES.md](docs/PATCHES.md)). |
| `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS` | `0` | `1` = alias DeepGEMM `sm121_*` indexer-logits headers to the shipped `sm120_*` so a cold JIT cache can compile on GB10. |
| `ENABLE_VLLM_GB10_PATCH` | `0` | `1` = experimental hybrid NVFP4 plugin (`--quantization modelopt_gb10_hybrid`). |

Issue **#21 / #26 / #27 / #43** Python hotfixes always run at container start
(they are not skipped by `DSPARK_SKIP_HOTFIX`). `#27` + the 1024 prefill cap
is why six huge cold prompts queue instead of starving decode.

### Stage-C only (no-ops on Anemll `0.1.1`)

These warn `Unknown vLLM environment variable` on the default image. They
matter only after you switch `DSPARK_VLLM_IMAGE` to Stage-C **and** merge
`docker-compose.stage-c.override.yml`:

`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK`, `VLLM_USE_B12X_WO_PROJECTION`,
`VLLM_DSPARK_LOCAL_ARGMAX`, `VLLM_DSPARK_REPLICATE_MARKOV_W1`,
`DSPARK_SLOT_CLAMP`, and the rest of the commented Stage-C block in the
example. See [Optional: Stage-C / 200K-16](#optional-stage-c--200k-16).

---

## What speed to expect

Full tables, method, and older lanes:
**[results/RESULTS-2026-08-14.md](results/RESULTS-2026-08-14.md)**.

On the **default Anemll 1M/6** stack:

| Workload | What you should see |
| --- | --- |
| One chat, any prompt length through 128K | ~62–83 decode tok/s after first token |
| **Six short chats** (hundreds of tokens), 1M still *allowed* | **~160–190 tok/s aggregate** (~30–37 per stream) |
| Six **cold 32K–128K** prompts at once | Prefills are chunked (issue #27), **two in flight** (`DSPARK_MAX_INFLIGHT_PREFILLS=2`), the rest queue. A 4 × 8K wave gets its first tokens at 7.4 / 9.0 / 15.7 / 16.5 s (was 4.7 / 9.4 / 14.3 / 19.3 s at `1`). ~8 tok/s decode floor while prefills run; 128K × 6 TTFT minutes |

That ~170–190 c=6 number is **six streams generating**, not six huge prefills.
Live 2026-08-14 on this cluster: 256 × c=6 = **162** agg; 128K × c=1 still
**75 tok/s** / **80 s** TTFT.

Live 2026-09-02, same lane, sp-indexer on, capture 48, inflight 2
([docs/CLAUDE/ab-results-2026-09-03.md](docs/CLAUDE/ab-results-2026-09-03.md)): 256 × c=1 = **56** tok/s
(8-trial median, 51–67), 256 × c=6 = **139** agg, 128K × c=1 TTFT **78.5 s**.
Decode is 15–25 % under the 14 Aug figures above and no `.env.dspark` knob
accounts for it; treat the 14 Aug numbers as the best seen, not the norm.

**315 / 205 tok/s** (200K context, 16 slots) needs the **Stage-C + Keys**
path. The ~182 1M/6 microbench was also measured with that Keys mask; the
same *ballpark* on **short** prompts is already what Anemll does (1 Aug
256 × c=6 = **191** agg; 14 Aug = **162**). Setting
`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` on Anemll `0.1.1` is a **no-op**
(warning only). See [Optional: Stage-C / 200K-16](#optional-stage-c--200k-16) and
[docs/ENVS.md](docs/ENVS.md).

Capture: [docs/benchmarks.png](docs/benchmarks.png).

---

## Thinking and `max_tokens`

> [!IMPORTANT]
> The #31 GPU budget patch is **opt-in at boot** (`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`)
> and then opt-in per request. Default is stock V2: do not send
> `thinking_token_budget`. When enabled, counters stay on the GPU; omitting the
> field does not inject a server-side default. Leave the flag at `0` unless a
> client actually sends the field ([Issue #66](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/issues/66)).

`max_tokens` counts **think + answer** (reasoning + visible response + tool
markup). With `DEFAULT_THINKING=max`, a harness cap of 256/512/800 often
returns `content: null` / `finish_reason: length` because reasoning eats the
whole budget — `max` ships a checkpoint-level directive ("do not stop reasoning
until … no error remains undiscovered") and produced **~50,000 reasoning chars
(~12.5k tokens) on a moderate prompt** in live measurement. So "size `max_tokens`
accordingly" means **tens of thousands of tokens**, not a small bump. Raise
`max_tokens`, set thinking `low` / `off`, or — with
`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1` — send an explicit `thinking_token_budget`
(see [Thinking-token budgets](#thinking-token-budgets)) so reasoning is
hard-capped and the rest of `max_tokens` is left for the visible answer.

Client `stop` strings used to fire inside `<think>`. The recipe applies
`patches/hotfix-dsv4-suppress-stops-in-reasoning.py` so they wait for
`</think>`. Opt out: `DSPARK_SUPPRESS_STOPS_IN_REASONING=0`.

A tool call cut off by `max_tokens` used to report `finish_reason: "tool_calls"`
with **invalid JSON** `arguments` and silently poison the transcript (HTTP 400 on
the next turn). The recipe applies `patches/hotfix-dsv4-issue55-tool-truncation.py`
so a truncated call reports `finish_reason: "length"` (not `"tool_calls"`) and any
non-JSON-parseable `arguments` are dropped. Clients that read `length` can discard
the in-flight call and retry; normal model-stopped tool calls keep
`finish_reason: "tool_calls"`. Harnesses that **ignore** `finish_reason` and
blindly replay `args` from streaming deltas can still hit a 400 - verify your
client drops an in-progress tool call on `finish_reason: "length"`.


### Thinking-token budgets

`max_tokens` caps **all** new tokens (reasoning + visible answer + tool
markup). `thinking_token_budget` caps only the reasoning portion and forces a
single `</think>` at the boundary, leaving the rest of `max_tokens` available
for the visible answer or tool call. A budget of `0` disables reasoning for
that request. Natural `</think>` remains untouched.

The field is HTTP 400 on stock V2. Set `DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1` and
recreate the containers, then send it when a hard cap is required. Omitting it
keeps `DEFAULT_THINKING` behavior. Keep `max_tokens` comfortably above the
thinking budget so the answer has room.

```json
{
  "max_tokens": 8192,
  "thinking_token_budget": 1024,
  "temperature": 0.6,
  "top_p": 0.95,
  "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}
}
```

Inspect `finish_reason` and `completion_tokens`. `length` + null `content`
means think ate the cap.

### Enabling the budget from a client

`thinking_token_budget` needs the boot flag **and** the request field — the
server injects no default when the field is omitted. To turn it on, set
`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`, recreate, then send it from the client:

**curl / any OpenAI-compatible client** — add `thinking_token_budget` to the
request body. `0` disables reasoning for that one call; `N>0` caps reasoning at
`N` generated tokens and leaves the rest of `max_tokens` for the visible
answer:

```bash
curl :8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4-flash-vision-exp",
  "messages":[{"role":"user","content":"Design a small rate limiter."}],
  "max_tokens":4096,
  "thinking_token_budget":1024,
  "temperature":0.6,"top_p":0.95,
  "chat_template_kwargs":{"thinking":true,"reasoning_effort":"high"}
}'
```

**pi** — the budget needs the boot flag **and** the pi model entry, so
[`pi-models.dspark.example.json`](pi-models.dspark.example.json) ships
`supportsThinkingTokenBudget: false` to match the server default
(`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=0`). Copy it to `~/.pi/agent/models.json`;
once the boot flag is `1`, flip the capability on the `deepseek-v4-flash-vision-exp`
model and pi attaches a `thinking_token_budget` whenever thinking is enabled:

```json
{
  "id": "deepseek-v4-flash-vision-exp",
  "reasoning": true,
  "compat": {
    "supportsThinkingTokenBudget": true,
    "thinkingFormat": "chat-template",
    "chatTemplateKwargs": {
      "thinking":       { "$var": "thinking.enabled" },
      "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true }
    }
  }
}
```

pi sizes the budget from per-level defaults (`minimal` 1024 / `low` 2048 /
`medium` 8192 / `high` 16384), always leaving room for the answer; override
values with the pi `thinkingBudgets` setting. If the capability is `true` while
the boot flag is `0`, every pi request fails with `thinking_token_budget is not
yet supported by the V2 model runner` — so keep the capability `false` unless
`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1`.

---

## How the KV cache works (why 1M + concurrency is safe)

| Knob | Meaning | This build |
| --- | --- | --- |
| KV pool | Shared blocks after weights load | 2,331,430 tokens / 17.04 GiB @ util 0.83 |
| `max_model_len` | Per-request **ceiling** | 1,048,576 |
| `max_num_seqs` | Max **active** sequences | 6 |

```
6 ×  50k  =  300k   easy
6 × 200k  =  1.2M   fits
6 × 500k  =  3.0M   near/over pool
6 ×   1M  =  6.0M   impossible — extras queue
```

The boot line `Maximum concurrency for 1,048,576 tokens … 2.22x` only means a
few *simultaneous full-1M* requests fit.

---

## If output garbles, loops, or leaks XML

Validate **direct** `:8888` first, then the agent harness.

1. Same image digest on both nodes (`docker image inspect $DSPARK_VLLM_IMAGE`).
   Compose must use `/usr/local/bin/vllm` (Anemll), not Stage-C `/opt/env`.
2. Full 0731 hub snapshot on **head and worker**, including
   `encoding/encoding_dsv4.py`.
3. Send `temperature: 0` for deterministic curls. Clear harness fallback lists
   so another model cannot poison the transcript.

If direct vLLM is clean and the agent is not, fix the harness — do not switch
to fp8 or a smaller model to hide it.

---

## Optional: Stage-C / 200K-16

Historical overlay image `vllm-dspark-runtime:dspark-nvfp4-stage-c`
(`./build-dspark-vllm-runtime.sh`). **Both** nodes need that image.
Merge [`docker-compose.stage-c.override.yml`](docker-compose.stage-c.override.yml)
— `./start-*.sh` does **not** add that file by itself — and uncomment the
Stage-C block in `.env.dspark`.

| Profile | Env | Published headline |
| --- | --- | --- |
| Keep 1M / 6 | `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=6`, Keys mask | ~182 agg on a short-prompt microbench |
| High aggregate | `MAX_MODEL_LEN=200000`, `MAX_NUM_SEQS=16`, Keys mask | 315 static / 205 staggered |

Issue **#27** (`LONG_PREFILL_TOKEN_THRESHOLD=1024`, one in-flight long prefill)
still **serializes** huge cold prefills. Stage-C does not turn 6 × 128K into
six parallel 80 s reads. Details: [results/RESULTS-2026-08-14.md](results/RESULTS-2026-08-14.md),
[docs/PATCHES.md](docs/PATCHES.md).

Compose is Anemll-shaped (`/usr/local/bin/vllm`, hotfixes under
`/usr/local/lib/...`). Treat the first Stage-C boot as an experiment.

---

## Runtime flags (default compose)

- `/usr/local/bin/vllm serve` · TP=2 · `mp` · `nnodes 2`
- `--kv-cache-dtype nvfp4_ds_mla` · `--block-size 256`
- `--max-model-len 1048576` · `--max-num-seqs 6` · `--max-num-batched-tokens 8192`
- `--long-prefill-token-threshold 1024` · `--enable-chunked-prefill` · `--async-scheduling`
- `--max-cudagraph-capture-size` = `MAX_NUM_SEQS * (MTP_NUM_TOKENS + 1)` padded to a multiple of 8 → 48 at 6×6 (plain 42 truncates to 40 and costs 12 % at c=6, measured 2026-09-02)
- `--moe-backend flashinfer_b12x` · `--generation-config vllm`
- DSpark: `{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}`

This is the **Stage C padded NVFP4** path (584-byte sparse-MLA envelope via
`nvfp4_ds_mla`). It is not the abandoned 416-byte “true layout” experiment.

Optional GB10 hybrid plugin: `ENABLE_VLLM_GB10_PATCH=1 ./start-…`
(`--quantization modelopt_gb10_hybrid`). Default off.

CI on every pull request and push to `main` ([`.github/workflows/validate.yml`](.github/workflows/validate.yml))
is **CPU-only** (`scripts/ci-validate.sh`). Live tok/s still needs the 2× Spark pair.

### Strict Responses API verification

Stateful `previous_response_id` continuation requires starting vLLM with
`VLLM_ENABLE_RESPONSES_API_STORE=1`. vLLM keeps the Responses API store off
by default; when enabled, stored response state consumes memory and is retained
until the server restarts. A continuation `response_id` 404 while the other
gates pass indicates this configuration is off, not a verifier regression.

Existing live evidence: the stock configuration passed 3/4 gates, with only
the known configuration 404; a controlled
`VLLM_ENABLE_RESPONSES_API_STORE=1` run passed all four gates.

After the server is ready, run the dependency-free live verifier to check
Responses text/SSE, stateful tool continuation, strict JSON schema, reasoning,
invalid-field errors, appended multi-turn prefix reuse, and disconnect cleanup:

```bash
python3 scripts/verify-responses-api-live.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-vision-exp \
  --output results/responses-api-live.json
```

The verifier above is the store/tool/cache and broad no-regression gate. Issue
#138 has a separate mode-strict verifier for stateless **full-history replay**;
it always sends `store: false` and never uses `previous_response_id`. On stock
(default flag `0`), recreate the containers and pin the reported legacy item to
HTTP 400 while the complete canonical output replay remains HTTP 200:

```bash
python3 scripts/verify-issue138-responses-history-live.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-vision-exp \
  --expect-legacy rejected \
  --output results/issue138-stock.json
```

To test compatibility mode, set
`DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT=1` in `.env.dspark`, stop and
start the pair so both containers are recreated, then require the same replay
to succeed by rerunning the command above with `--expect-legacy accepted` and
`--output results/issue138-enabled.json`.

The enabled run also checks assistant semantic continuity, the exact reported
four-item payload, six malformed/ambiguous neighbors that must remain HTTP 400,
and valid easy/canonical controls. Neither expected mode accepts the opposite
outcome. Changing the flag in either direction requires recreation, not
`docker compose restart`, because disabling it cannot undo bytes already
patched in a container writable layer.

The broad verifier run intentionally creates a ~21K-token appended conversation
and a forced client disconnect. Use `--skip-multiturn` or `--skip-disconnect`
only when the corresponding live behavior is outside the test scope.

---

## Files

| Path | Purpose |
| --- | --- |
| [results/RESULTS-2026-08-14.md](results/RESULTS-2026-08-14.md) | Dated benches and how to read them |
| `.env.dspark.example` | Cluster template |
| `docker-compose.dspark.yml` | Anemll serve (installs Vision-Exp encoder + hotfixes) |
| `start-` / `stop-` / `status-` / `logs-` / `smoke-*.sh` | Two-node ops (`stop --nfs` also tears down `dspark-nfs`, not Qwen's `vllm-fn-nfs`) |
| `prepare-dspark-model-cache.sh` | Vision-Exp weights on both nodes (`--official` / `--abliterated`). With `DSPARK_WORKER_HF_NFS=1`, head only; worker mounts over NFS. |
| `files/nfs-share.sh` / `files/nfs-server/` | NFSv4 exporter for the head HF cache (reuses a live share if one is already up) |
| `docker-compose.dspark-nfs.override.yml` | Worker: named NFS volume + local JIT overlays |
| `scripts/overlay-vision-exp-ablit-cache.py` | Hardlink official Vision-Exp blobs + copy the 26 ablit shards |
| `scripts/benchmark-0731.py` | Prompt × concurrency sweep |
| `scripts/verify-responses-api-live.py` | Strict Responses, multi-turn cache, and disconnect gates |
| `scripts/verify-issue138-responses-history-live.py` | Mode-strict stock/enabled full-history replay gate |
| [docs/ENVS.md](docs/ENVS.md) | Anemll vs Stage-C env matrix |
| [docs/PATCHES.md](docs/PATCHES.md) | Keys / #27 / #22 notes |
| `patches/` | Issue hotfixes applied at container start |
| `docker-compose.stage-c.override.yml` | Stage-C-only env injection |
| `build-dspark-vllm-runtime.sh` | Optional local Stage-C image |

---

## Credits

Full list: [`CREDITS.md`](CREDITS.md).

**[drowzeys ("Keys")](https://github.com/drowzeys/)** — DSpark concurrency
patch, ragged `query_start_loc`, `nvfp4_ds_mla` wiring.

**[@u1tra_instinct](https://x.com/u1tra_instinct)** — abliterated Vision-Exp
weights (`ABLITERATED=1`), from the original repo
[`drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit`](https://huggingface.co/drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit).

Also: [tonyd2wild](https://github.com/tonyd2wild/), Rafael Caricio, Fraser Price,
[Anemll](https://github.com/Anemll/dspark-vllm-gx10), MiaAI-Lab packaging.

Repo scripts/docs: this repo’s `LICENSE`. Overlay/runtime: Apache-2.0 / upstream
licenses. Weights and base images have their own terms.
