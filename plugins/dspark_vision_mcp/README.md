# dspark-vision-mcp

Local **vision tool** MCP server for the DeepSeek-V4-Flash-0731 DSpark stack.

0731 stays text-only on `:8888`. This server calls the **Qwen3-VL-4B** sidecar on
`:8889` and returns description / OCR / comparison text so the 0731 agent can
reason natively (including `reasoning_effort=max`) without switching models.

---

## Vision support

This is the **production vision path** for the DSpark stack. Native MoonViT on
0731 is retired; agents see images only through these tools (or the CLI helper).

### Architecture

```text
  Agent harness (pi / OMP / Hermes / goose / grok / openclaw / ZCode Desktop / …)
        │  tool call: describe_image / ocr_image / compare_images
        │  (Prime Agent: await dspark_vision.* from IPython skill)
        ▼
  dspark-vision-mcp  (stdio, launched via uvx)  — or Prime Python skill
        │  OpenAI chat.completions + image_url (base64 data URI)
        ▼
  Qwen3-VL-4B AWQ-4bit sidecar  http://127.0.0.1:8889   (TP=2 head+worker)
        │  factual description / OCR / comparison text
        ▼
  Agent continues on 0731  http://127.0.0.1:8888
        │  text-only reasoning (high / max effort is stable here)
        ▼
  Final answer
```

| Piece | Role |
|-------|------|
| **0731** (`:8888`, `deepseek-v4-flash-0731`) | Reasoning / tools / chat — **text only** |
| **VL sidecar** (`:8889`, `qwen3-vl-4b`, TP=2) | Sees pixels; sharded across both Sparks; `enable_thinking=false`, `temperature=0` |
| **This MCP** | Pass-1 extraction only; returns text for 0731 to reason over |
| **`scripts/vision-reason.py`** | Same two-pass idea without a harness (CLI) |

Why not send images to 0731 directly? Max/high-effort reasoning over image tokens
was unstable on the retired MoonViT lane; text-only max effort is stable. Fusing
vision as a **tool** keeps one conversation on 0731.

### What the stack starts for you

1. `./prepare-dspark-model-cache.sh` caches 0731 **and** `VL_SIDECAR_MODEL` on
   head **and** worker (TP=2). Serve keeps `HF_HUB_OFFLINE=1`.
2. With `ENABLE_VL_SIDECAR=1` (default in `.env.dspark`),
   `./start-deepseek-v4-flash-dspark.sh` brings up 0731 (TP=2), **then** the
   VL sidecar worker-first on a separate NCCL master port (`25100`).
3. When the sidecar lists `qwen3-vl-4b`, start runs
   `scripts/install-dspark-vision-mcp.sh` if `INSTALL_VISION_MCP` is on
   (defaults to follow `ENABLE_VL_SIDECAR`).
4. Detected harnesses get `dspark-vision` registered automatically (ZCode Desktop
   included via `~/.zcode/cli/config.json`).

Compose: [`docker-compose.vl-sidecar.yml`](../../docker-compose.vl-sidecar.yml).
Coexist (measured): 0731 `GPU_MEMORY_UTILIZATION=0.82` + `MAX_NUM_SEQS=4` with
sidecar `VL_SIDECAR_GPU_UTIL=0.03` and **`int4_per_token_head`** KV → ~1.60M main
KV tokens while VL keeps ≥1× of 32k. True `nvfp4` KV needs FlashInfer SM100
(GB10 is SM12.1). Re-check worker free memory if you raise main util.

### Tools

| Tool | Purpose |
|------|---------|
| `describe_image(path_or_url, question?)` | Detailed factual description (focused on `question` when given) |
| `ocr_image(path_or_url)` | Extract visible text |
| `compare_images(paths, question)` | Compare up to 4 images |

Accepts local paths and `http(s)` URLs. Missing files and a down sidecar return
actionable `Error: …` strings. Oversized images are downscaled before upload.
Hard limit: **4 images** per request (sidecar `--limit-mm-per-prompt`).

### Using it from an agent

Stay on **`deepseek-v4-flash-0731`**. Give an absolute image path (or URL) and ask
normally — the skill tells the model to call `describe_image` first, then reason:

```text
Look at /home/mia/pic2.jpg — what color is the sweater and what is the likely
setting? Reason step by step.
```

Do **not** switch to the `qwen3-vl-sidecar` model for the answer; that lane is
extraction-only from the agent’s point of view.

### Env knobs (sidecar + MCP)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_VL_SIDECAR` | `1` in `.env.dspark` | Start Qwen3-VL TP=2 on `:8889` |
| `VL_SIDECAR_MODEL` | `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit` | HF id; cached by `./prepare-dspark-model-cache.sh` on both nodes |
| `VL_SIDECAR_TP_SIZE` / `VL_SIDECAR_NNODES` | `2` / `2` | Shards vision across both Sparks |
| `VL_SIDECAR_MASTER_PORT` | `25100` | NCCL master port (DeepSeek uses `25000`) |
| `VL_SIDECAR_GPU_UTIL` | `0.03` | Per-GPU util after TP shard (0.022 also OK when worker free ≥~4.5 GiB) |
| `VL_SIDECAR_KV_CACHE_DTYPE` | `int4_per_token_head` | 4-bit KV via Triton; `nvfp4` blocked on SM12.1 (needs FlashInfer SM100) |
| `VL_SIDECAR_ATTENTION_BACKEND` | `TRITON_ATTN` | Required for int4 KV; also avoids FlashInfer fp8 `plan()` issues |
| `PREPARE_VL_SIDECAR_MODEL` | `1` | Prepare-cache downloads VL weights on head **and** worker |
| `VL_SIDECAR_PORT` | `8889` | Sidecar listen port (head API rank) |
| `INSTALL_VISION_MCP` | follows sidecar | Auto-register into harnesses on start |
| `VISION_MCP_HARNESSES` | `auto` | `auto` or `pi,omp,hermes,…,zcode,prime` |
| `DSPARK_VL_BASE_URL` | `http://127.0.0.1:8889` | Where this MCP posts completions |
| `DSPARK_VL_MODEL` | `qwen3-vl-4b` | Served model id |
| `DSPARK_VL_MAX_TOKENS` | `1024` | Extraction max tokens |

### Errors you should see

| Situation | Tool returns |
|-----------|----------------|
| Missing file | `Error: image file not found: …` |
| Sidecar down | `Error: vision sidecar unreachable at …` (+ start hint) |
| Too many images | `Error: too many images (N); sidecar limit is 4 …` |

### Retired: native MoonViT

`plugins/dsv4_moonvit_vllm` and the overlay model dir are **not** the production
path. Historical notes: [`docs/VISION.md`](../../docs/VISION.md),
[`docs/PROJECTOR-FINETUNE.md`](../../docs/PROJECTOR-FINETUNE.md).

---

## Seamless harness install (recommended)

When `ENABLE_VL_SIDECAR=1`, `./start-deepseek-v4-flash-dspark.sh` waits for the
sidecar then runs:

```bash
./scripts/install-dspark-vision-mcp.sh
```

That detects installed harnesses and upserts the `dspark-vision` MCP server
(plus skill where applicable). Opt out with `INSTALL_VISION_MCP=0`. Restrict
with `VISION_MCP_HARNESSES=pi,omp` (default `auto` = all supported).

Supported harnesses:

| Harness | Detect | Writes |
|---------|--------|--------|
| **pi** | `pi` + `~/.pi/agent` | `~/.config/mcp/mcp.json`, `~/.pi/agent/mcp.json`, skill, `pi-mcp-adapter` |
| **OMP** | `omp` + `~/.omp` | `~/.omp/agent/mcp.json`, skill under `~/.omp/agent/skills/` |
| **Hermes** | `hermes` + `~/.hermes/config.yaml` | `mcp_servers.dspark-vision` (surgical YAML append), skill |
| **opencode** | `opencode` or `~/.config/opencode` | `~/.config/opencode/opencode.json` `mcp` block + skill |
| **goose** | `goose` or `~/.config/goose/config.yaml` | `extensions.dspark-vision` in `~/.config/goose/config.yaml` + skill ([goose-docs.ai](https://goose-docs.ai/)) |
| **grok** | `grok` / `~/.grok/bin/grok` or `~/.grok/config.toml` | `[mcp_servers.dspark-vision]` in `~/.grok/config.toml` + skill ([Grok Build](https://docs.x.ai/build/features/mcp-servers)) |
| **openclaw** | `openclaw`/`oclaw` or `~/.openclaw` | `mcp.servers.dspark-vision` in `~/.openclaw/openclaw.json` + skill ([OpenClaw MCP](https://docs2.openclaw.ai/tools/mcp)) |
| **zcode** | `zcode` or `~/.zcode` | User-scope `mcp.servers.dspark-vision` in `~/.zcode/cli/config.json` (+ skill). **ZCode Desktop** reads this same file (Settings → MCP Servers); restart/refresh if the app was already open ([ZCode MCP](https://zcode.z.ai/en/docs/mcp-services)) |
| **prime** | `prime-agent` or `~/.prime/agent` | Python skill `~/.prime/agent/skills/dspark-vision` calling `:8889` directly (Prime MCP is HTTP-only; [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)) |

Idempotent; never wipes other MCP entries. Failures are non-fatal unless
`--strict`. Run alone anytime (sidecar should be up for Hermes sessions that
eager-connect):

```bash
./scripts/install-dspark-vision-mcp.sh
./scripts/install-dspark-vision-mcp.sh --dry-run
./scripts/install-dspark-vision-mcp.sh --harnesses pi,hermes
```

## Run (stdio)

```bash
# from repo root — uvx installs deps into an ephemeral env
uvx --from ./plugins/dspark_vision_mcp dspark-vision-mcp
```

## Manual pi registration (if not using the installer)

pi has no built-in MCP — use [`pi-mcp-adapter`](https://www.npmjs.com/package/pi-mcp-adapter)
plus `~/.config/mcp/mcp.json`. Prefer the installer above; example fragment:

```json
{
  "settings": { "toolPrefix": "none" },
  "mcpServers": {
    "dspark-vision": {
      "command": "/home/YOU/.local/bin/uvx",
      "args": [
        "--from",
        "/path/to/deepSeek-v4-Flash-DSpark/plugins/dspark_vision_mcp",
        "dspark-vision-mcp"
      ],
      "directTools": ["describe_image", "ocr_image", "compare_images"]
    }
  }
}
```

Use absolute paths to `uvx` and the plugin so GUI/agent launches find them.

## Smoke (no harness)

```bash
curl -s http://127.0.0.1:8889/v1/models | head -c 200

uv run --directory plugins/dspark_vision_mcp python -c \
  'from dspark_vision_mcp.server import describe_image; print(describe_image("/home/mia/pic2.jpg", "sweater color?"))'
```

CLI two-pass (extract + 0731 max reasoning):

```bash
python3 scripts/vision-reason.py --image /home/mia/pic2.jpg \
  --question "What color is the sweater and what is the likely setting?"
```
