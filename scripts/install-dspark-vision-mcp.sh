#!/usr/bin/env bash
# Register dspark-vision MCP into detected agent harnesses
# (pi, OMP, Hermes, opencode, goose, grok, openclaw, zcode, prime).
# Idempotent. Non-fatal by default (use --strict to fail on adapter errors).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_DIR="${VISION_MCP_PLUGIN_DIR:-$REPO_DIR/plugins/dspark_vision_mcp}"
INSTALL_PY="$PLUGIN_DIR/install_harnesses.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--strict] [--dry-run] [--harnesses LIST] [--base-url URL]

Registers the local dspark-vision MCP server into harnesses present on this
machine. Supported: pi, omp, hermes, opencode, goose, grok, openclaw, zcode, prime.

Options:
  --strict         Exit non-zero if a detected harness fails to install
  --dry-run        Detect only; do not write configs
  --harnesses LIST auto (default) or comma list:
                   pi,omp,hermes,opencode,goose,grok,openclaw,zcode,prime
  --base-url URL   VL sidecar base (default DSPARK_VL_BASE_URL / :8889)
  -h, --help       Show this help

Env:
  VISION_MCP_HARNESSES   same as --harnesses
  DSPARK_VL_BASE_URL     sidecar base URL
  VL_SIDECAR_PORT        used when building the default base URL
  VISION_MCP_PLUGIN_DIR  override plugin path
EOF
}

STRICT=0
DRY_RUN=0
HARNESSES="${VISION_MCP_HARNESSES:-auto}"
BASE_URL="${DSPARK_VL_BASE_URL:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --harnesses)
      [ "$#" -ge 2 ] || { echo "--harnesses requires a value" >&2; exit 2; }
      HARNESSES="$2"; shift 2 ;;
    --harnesses=*) HARNESSES="${1#*=}"; shift ;;
    --base-url)
      [ "$#" -ge 2 ] || { echo "--base-url requires a value" >&2; exit 2; }
      BASE_URL="$2"; shift 2 ;;
    --base-url=*) BASE_URL="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ ! -f "$INSTALL_PY" ]; then
  echo "Missing installer: $INSTALL_PY" >&2
  exit 1
fi

# Prefer repo-local / user uvx on PATH; Python helper installs uv if needed.
export PATH="${HOME}/.local/bin:${PATH}"

ARGS=(--plugin-dir "$PLUGIN_DIR" --harnesses "$HARNESSES")
if [ -n "$BASE_URL" ]; then
  ARGS+=(--base-url "$BASE_URL")
fi
if [ "$STRICT" = "1" ]; then
  ARGS+=(--strict)
fi
if [ "$DRY_RUN" = "1" ]; then
  ARGS+=(--dry-run)
fi

exec python3 "$INSTALL_PY" "${ARGS[@]}"
