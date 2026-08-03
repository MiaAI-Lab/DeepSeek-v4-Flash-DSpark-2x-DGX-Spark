#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MASTER_KEY_FILE="${LITELLM_MASTER_KEY_FILE:?set LITELLM_MASTER_KEY_FILE}"
OUTPUT="${LITELLM_VIRTUAL_KEY_FILE:?set LITELLM_VIRTUAL_KEY_FILE}"
CONTAINER="dspark-private-litellm-litellm-1"
CONTAINER_OUTPUT="/tmp/hermes-inference.key"
completed=0

[ -f "$MASTER_KEY_FILE" ] && [ ! -L "$MASTER_KEY_FILE" ] || exit 1
[ ! -e "$OUTPUT" ] || { echo "Refusing to overwrite virtual key file: $OUTPUT" >&2; exit 1; }
cleanup() {
  docker exec "$CONTAINER" python3 -c \
    'from pathlib import Path; Path("/tmp/hermes-inference.key").unlink(missing_ok=True)' \
    >/dev/null 2>&1 || true
  if [ "$completed" -ne 1 ] && [ -e "$OUTPUT" ]; then
    unlink "$OUTPUT" || true
  fi
}
trap cleanup EXIT
# Exact scope sent to /key/generate: {"models": ["deepseek-v4-flash-0731-smoke"]}
docker exec -i "$CONTAINER" python3 - "$CONTAINER_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
import urllib.request

output_path = sys.argv[1]
master = Path("/run/secrets/master-key").read_text().strip()
payload = json.dumps({
    "models": ["deepseek-v4-flash-0731-smoke"],
    "key_alias": "hermes-deepseek-smoke",
    "max_parallel_requests": 1,
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
target = Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(key + "\n")
target.chmod(0o600)
PY
docker cp "$CONTAINER:$CONTAINER_OUTPUT" "$OUTPUT"
chmod 0600 "$OUTPUT"
completed=1
cleanup
trap - EXIT
echo "Model-scoped virtual key written to $OUTPUT"
