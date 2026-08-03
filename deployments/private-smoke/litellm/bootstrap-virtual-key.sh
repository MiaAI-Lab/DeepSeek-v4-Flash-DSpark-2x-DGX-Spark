#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_KEY_FILE="${LITELLM_MASTER_KEY_FILE:?set LITELLM_MASTER_KEY_FILE}"
OUTPUT="${LITELLM_VIRTUAL_KEY_FILE:?set LITELLM_VIRTUAL_KEY_FILE}"
BASE_URL="${LITELLM_BASE_URL:-http://${HEAD_TAILSCALE_IP:?set HEAD_TAILSCALE_IP}:4001}"

[ -f "$MASTER_KEY_FILE" ] && [ ! -L "$MASTER_KEY_FILE" ] || exit 1
[ ! -e "$OUTPUT" ] || { echo "Refusing to overwrite virtual key file: $OUTPUT" >&2; exit 1; }
# Exact scope sent to /key/generate: {"models": ["deepseek-v4-flash-0731-smoke"]}
python3 - "$BASE_URL" "$MASTER_KEY_FILE" "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

base_url, master_path, output_path = sys.argv[1:]
master = Path(master_path).read_text().strip()
payload = json.dumps({
    "models": ["deepseek-v4-flash-0731-smoke"],
    "key_alias": "hermes-deepseek-smoke",
    "max_parallel_requests": 1,
    "metadata": {"scope": "synthetic-hermes-smoke"},
}).encode()
request = urllib.request.Request(
    f"{base_url}/key/generate",
    data=payload,
    headers={"Authorization": f"Bearer {master}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    result = json.load(response)
key = result.get("key")
if not isinstance(key, str) or not key.startswith("sk-") or key == master:
    raise SystemExit("LiteLLM returned an invalid virtual key")
target = Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(key + "\n")
target.chmod(0o600)
PY
chmod 0600 "$OUTPUT"
echo "Model-scoped virtual key written to $OUTPUT"
