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
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys

worker, image, interface, head_ip, worker_ip, output = sys.argv[1:]

def run(role, command):
    argv = ["bash", "-lc", command] if role == "head" else ["ssh", worker, command]
    return subprocess.run(argv, text=True, capture_output=True, check=False)

def collect(role, address):
    quoted_image = shlex.quote(image)
    quoted_interface = shlex.quote(interface)
    quoted_mtu_path = shlex.quote(f"/sys/class/net/{interface}/mtu")
    command = "\n".join((
        "printf 'kernel=%s\\n' \"$(uname -r)\"",
        "printf 'mem_available_gib=%s\\n' \"$(awk '/MemAvailable/ {print $2/1024/1024}' /proc/meminfo)\"",
        "printf 'memory_psi_full_avg10=%s\\n' \"$(awk '/^full / {for (i=1;i<=NF;i++) if ($i ~ /^avg10=/) {split($i,a,\"=\"); print a[2]}}' /proc/pressure/memory)\"",
        "printf 'disk_available_gib=%s\\n' \"$(df -BG --output=avail /home/plexiz | tail -1 | tr -dc '0-9')\"",
        "printf 'docker_version=%s\\n' \"$(docker version --format '{{.Server.Version}}' 2>/dev/null || printf unknown)\"",
        f"if docker image inspect {quoted_image} >/dev/null 2>&1; then echo image_present=true; else echo image_present=false; fi",
        f"printf 'link=%s\\n' \"$(ip -o link show dev {quoted_interface} 2>/dev/null)\"",
        f"printf 'address=%s\\n' \"$(ip -4 -o address show dev {quoted_interface} 2>/dev/null)\"",
        f"printf 'mtu=%s\\n' \"$(cat {quoted_mtu_path} 2>/dev/null || printf 0)\"",
        "printf 'rank_running=%s\\n' \"$(docker inspect -f '{{.State.Running}}' deepseek-v4-flash-vllm-dspark-1 2>/dev/null || printf false)\"",
        "printf 'rank_restarts=%s\\n' \"$(docker inspect -f '{{.RestartCount}}' deepseek-v4-flash-vllm-dspark-1 2>/dev/null || printf 0)\"",
    ))
    result = run(role, command)
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    return {
        "role": role,
        "kernel": fields.get("kernel", "unknown"),
        "mem_available_gib": round(float(fields.get("mem_available_gib", "0")), 2),
        "memory_psi_full_avg10": round(float(fields["memory_psi_full_avg10"]), 4),
        "disk_available_gib": round(float(fields.get("disk_available_gib", "0")), 2),
        "docker_version": fields.get("docker_version", "unknown"),
        "image_present": fields.get("image_present") == "true",
        "fabric_up": "UP" in fields.get("link", ""),
        "fabric_mtu": int(fields.get("mtu", "0") or 0),
        "fabric_address_present": address in fields.get("address", ""),
        "rank_running": fields.get("rank_running") == "true",
        "rank_restart_count": int(fields.get("rank_restarts", "0") or 0),
    }

with ThreadPoolExecutor(max_workers=2) as pool:
    head_future = pool.submit(collect, "head", head_ip)
    worker_future = pool.submit(collect, "worker", worker_ip)
    nodes = [head_future.result(), worker_future.result()]

payload = {
    "schema_version": 2,
    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "nodes": nodes,
}
target = Path(output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
target.chmod(0o600)
print(f"node-evidence={target}")
PY
