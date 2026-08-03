#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.dspark}"
OUTPUT="${1:-$ROOT_DIR/artifacts/node-evidence.json}"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

python3 - "$WORKER_HOST" "$DSPARK_VLLM_IMAGE" "${NCCL_SOCKET_IFNAME:-enp1s0f0np0}" \
  "${VLLM_HOST_IP:-10.77.77.1}" "${WORKER_VLLM_HOST_IP:-10.77.77.2}" "$OUTPUT" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

worker, image, interface, head_ip, worker_ip, output = sys.argv[1:]

def run(role, command):
    argv = ["bash", "-lc", command] if role == "head" else ["ssh", worker, command]
    return subprocess.run(argv, text=True, capture_output=True, check=False)

def value(role, command, default=""):
    result = run(role, command)
    return result.stdout.strip() if result.returncode == 0 else default

def collect(role, address):
    image_result = run(role, f"docker image inspect {image!r}")
    link = value(role, f"ip -o link show dev {interface!r}")
    address_text = value(role, f"ip -4 -o address show dev {interface!r}")
    running = value(role, "docker inspect -f '{{.State.Running}}' deepseek-v4-flash-vllm-dspark-1 2>/dev/null", "false")
    restarts = value(role, "docker inspect -f '{{.RestartCount}}' deepseek-v4-flash-vllm-dspark-1 2>/dev/null", "0")
    mtu = value(role, f"cat /sys/class/net/{interface}/mtu", "0")
    return {
        "role": role,
        "kernel": value(role, "uname -r", "unknown"),
        "mem_available_gib": round(float(value(role, "awk '/MemAvailable/ {print $2/1024/1024}' /proc/meminfo", "0")), 2),
        "disk_available_gib": round(float(value(role, "df -BG --output=avail /home/plexiz | tail -1 | tr -dc '0-9'", "0")), 2),
        "docker_version": value(role, "docker version --format '{{.Server.Version}}'", "unknown"),
        "image_present": image_result.returncode == 0,
        "fabric_up": "UP" in link,
        "fabric_mtu": int(mtu or 0),
        "fabric_address_present": address in address_text,
        "rank_running": running == "true",
        "rank_restart_count": int(restarts or 0),
    }

payload = {
    "schema_version": 1,
    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "nodes": [collect("head", head_ip), collect("worker", worker_ip)],
}
target = Path(output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
target.chmod(0o600)
print(f"node-evidence={target}")
PY
