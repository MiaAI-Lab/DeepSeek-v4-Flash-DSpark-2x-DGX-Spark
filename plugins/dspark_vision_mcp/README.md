# dspark-vision-mcp

Local **vision tool** MCP server for the DeepSeek-V4-Flash-0731 DSpark stack.

0731 stays text-only on `:8888`. This server calls the **Qwen3-VL-4B** sidecar on
`:8889` and returns description / OCR / comparison text so the 0731 agent can
reason natively (including `reasoning_effort=max`) without switching models.

Same extraction idea as `scripts/vision-reason.py` pass 1 — the MCP returns the
description; pass 2 is the agent's own reasoning turn.

## Tools

| Tool | Purpose |
|------|---------|
| `describe_image(path_or_url, question?)` | Detailed factual description (focused on `question` when given) |
| `ocr_image(path_or_url)` | Extract visible text |
| `compare_images(paths, question)` | Compare up to 4 images |

Accepts local paths and `http(s)` URLs. Missing files and a down sidecar return
actionable `Error: …` strings. Oversized images are downscaled before upload.
Hard limit: **4 images** per request (sidecar `--limit-mm-per-prompt`).

## Seamless install (recommended)

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

Env overrides (optional):

- `DSPARK_VL_BASE_URL` (default `http://127.0.0.1:8889`)
- `DSPARK_VL_MODEL` (default `qwen3-vl-4b`)
- `DSPARK_VL_MAX_TOKENS` (default `1024`)

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
