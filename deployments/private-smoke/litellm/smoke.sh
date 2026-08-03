#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${LITELLM_ENV_FILE:-$SCRIPT_DIR/.env}"
COMPOSE=(docker compose -p dspark-private-litellm --env-file "$ENV_FILE" -f "$SCRIPT_DIR/docker-compose.yml")
MODE="${1:---all-interfaces}"
public_catalog="${ACTIVE_GATEWAY_SNAPSHOT:-$ROOT_DIR/artifacts/active-gateway-snapshot.json}"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

snapshot_public_catalog() {
  python3 - "$public_catalog" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
names = ["plexiz-litellm", "plexiz-litellm-postgres", "plexiz-litellm-cloudflared"]
items = []
for name in names:
    result = subprocess.run(["docker", "inspect", name], text=True, capture_output=True, check=True)
    item = json.loads(result.stdout)[0]
    labels = item.get("Config", {}).get("Labels") or {}
    items.append({
        "name": name,
        "id": item["Id"],
        "image": item["Image"],
        "config_hash": labels.get("com.docker.compose.config-hash"),
        "running": item.get("State", {}).get("Running"),
    })
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps({"public_catalog": items}, indent=2, sort_keys=True) + "\n")
target.chmod(0o600)
PY
}

if [ "$MODE" = "--snapshot-before" ]; then
  snapshot_public_catalog
  echo "Active gateway snapshot written to $public_catalog"
  exit 0
fi
[ "$MODE" = "--all-interfaces" ] || { echo "Usage: smoke.sh --snapshot-before|--all-interfaces" >&2; exit 2; }
[ -f "$public_catalog" ] || { echo "Missing pre-deployment public_catalog snapshot." >&2; exit 1; }

cleanup_on_failure() {
  local status=$?
  trap - ERR
  echo "Private gateway gate failed; removing only the smoke gateway." >&2
  "${COMPOSE[@]}" down --remove-orphans || true
  exit "$status"
}
trap cleanup_on_failure ERR

"$SCRIPT_DIR/egress-policy.sh" --check
binding="$(docker inspect dspark-private-litellm-litellm-1 -f '{{(index (index .HostConfig.PortBindings "4001/tcp") 0).HostIp}}:{{(index (index .HostConfig.PortBindings "4001/tcp") 0).HostPort}}')"
[ "$binding" = "$HEAD_TAILSCALE_IP:4001" ] || { echo "wrong published interface: $binding" >&2; false; }
[ "$(docker inspect dspark-private-litellm-litellm-1 -f '{{.HostConfig.ReadonlyRootfs}}')" = "true" ]
docker inspect dspark-private-litellm-litellm-1 -f '{{json .HostConfig.CapDrop}}' | grep -F 'ALL' >/dev/null
docker inspect dspark-private-litellm-litellm-1 -f '{{json .Mounts}}' | grep -F 'docker.sock' >/dev/null && {
  echo "docker.sock must not be mounted" >&2
  false
}

python3 - "$HEAD_TAILSCALE_IP" "$public_catalog" <<'PY'
import json
from pathlib import Path
import socket
import subprocess
import sys

tail_ip, snapshot_path = sys.argv[1:]

for denied_ip in {"127.0.0.1", *[item.split("/")[0] for item in subprocess.check_output(["hostname", "-I"], text=True).split()]} - {tail_ip}:
    try:
        socket.create_connection((denied_ip, 4001), timeout=1).close()
    except OSError:
        continue
    raise AssertionError(f"port 4001 unexpectedly reachable on {denied_ip}")

before = json.loads(Path(snapshot_path).read_text())
current = []
for name in ("plexiz-litellm", "plexiz-litellm-postgres", "plexiz-litellm-cloudflared"):
    item = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
    labels = item.get("Config", {}).get("Labels") or {}
    current.append({"name": name, "id": item["Id"], "image": item["Image"], "config_hash": labels.get("com.docker.compose.config-hash"), "running": item.get("State", {}).get("Running")})
if before != {"public_catalog": current}:
    raise AssertionError("active public gateway catalog/container identity changed")
PY

container_key="/tmp/smoke-virtual.key"
docker cp "$LITELLM_VIRTUAL_KEY_FILE" "dspark-private-litellm-litellm-1:$container_key"
docker exec -i dspark-private-litellm-litellm-1 python3 - <<'PY'
import json
from pathlib import Path
import urllib.error
import urllib.request

base = "http://127.0.0.1:4001"
key = Path("/tmp/smoke-virtual.key").read_text().strip()

def call(path, *, token=key, body=None, expected=(200,)):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, payload = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        payload = json.loads(error.read() or b"{}")
        error.close()
    if status not in expected:
        raise AssertionError(f"unexpected status for {path}: {status}")
    return payload

models = call("/v1/models")
ids = {item.get("id") for item in models.get("data", [])}
if ids != {"deepseek-v4-flash-0731-smoke"}:
    raise AssertionError(f"virtual key model scope mismatch: {ids}")
chat = call("/v1/chat/completions", body={
    "model": "deepseek-v4-flash-0731-smoke",
    "messages": [{"role": "user", "content": "Reply exactly GATEWAY_OK."}],
    "temperature": 0,
})
if not chat.get("choices"):
    raise AssertionError("gateway response has no choices")
call("/v1/models", token="wrong-key", expected=(401, 403))
call("/v1/chat/completions", body={"model": "wrong-model", "messages": [{"role": "user", "content": "x"}]}, expected=(400, 403, 404))
for admin_path in ("/key/generate", "/config"):
    call(admin_path, body={} if admin_path == "/key/generate" else None, expected=(401, 403, 404, 405))
PY
docker exec dspark-private-litellm-litellm-1 python3 -c \
  'from pathlib import Path; Path("/tmp/smoke-virtual.key").unlink(missing_ok=True)'

docker exec -i dspark-private-litellm-litellm-1 python3 - <<'PY'
import os
import socket
if os.path.exists("/var/run/docker.sock"):
    raise SystemExit("docker.sock is visible")
def reachable(host, port):
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False
if not reachable("172.30.0.1", 8888):
    raise SystemExit("allowed origin is unreachable")
for host, port in (("172.30.0.1", 4000), ("172.30.0.1", 22), ("1.1.1.1", 443)):
    if reachable(host, port):
        raise SystemExit(f"forbidden egress is reachable: {host}:{port}")
PY

grep -F 'num_retries: 0' "$SCRIPT_DIR/config.yaml" >/dev/null
grep -F 'fallbacks: []' "$SCRIPT_DIR/config.yaml" >/dev/null
echo "Private LiteLLM key scope, interfaces, public_catalog, egress, and no fallback gates passed."
trap - ERR
