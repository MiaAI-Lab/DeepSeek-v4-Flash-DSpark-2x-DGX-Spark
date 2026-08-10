#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_KEY_FILE="${LITELLM_MASTER_KEY_FILE:?set LITELLM_MASTER_KEY_FILE}"
OUTPUT="${LITELLM_VIRTUAL_KEY_FILE:?set LITELLM_VIRTUAL_KEY_FILE}"
CONTAINER="dspark-private-litellm-litellm-1"
completed=0
umask 077

[ -f "$MASTER_KEY_FILE" ] && [ ! -L "$MASTER_KEY_FILE" ] || exit 1
[ ! -e "$OUTPUT" ] || { echo "Refusing to overwrite virtual key file: $OUTPUT" >&2; exit 1; }
cleanup() {
  if [ "$completed" -ne 1 ] && [ -e "$OUTPUT" ]; then
    unlink "$OUTPUT" || true
  fi
}
trap cleanup EXIT
# Exact model scope plus the read-only model-detail route used by Hermes.
# LiteLLM classifies /v1/models/{id} as admin-only unless the virtual key
# explicitly permits it; inference remains restricted by the models list.
docker exec -i "$CONTAINER" python3 - >"$OUTPUT" <<'PY'
import json
from pathlib import Path
import urllib.request

master = Path("/run/secrets/master-key").read_text().strip()
payload = json.dumps({
    "models": ["deepseek-v4-flash-0731-smoke"],
    "allowed_routes": ["/v1/models", "/v1/models/*", "/v1/chat/completions"],
    "key_alias": "hermes-deepseek-smoke",
    "metadata": {"scope": "synthetic-hermes-smoke"},
}).encode()
request = urllib.request.Request(
    "http://127.0.0.1:4001/key/generate",
    data=payload,
    headers={"Authorization": f"Bearer {master}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    result = json.load(response)
key = result.get("key")
if not isinstance(key, str) or not key.startswith("sk-") or key == master:
    raise SystemExit("LiteLLM returned an invalid virtual key")
print(key)
PY
chmod 0600 "$OUTPUT"
completed=1
trap - EXIT
echo "Model-scoped virtual key written to $OUTPUT"
