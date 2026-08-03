#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.dspark}"
MANIFEST=""
REPORT=""
RUN_ALL=0

usage() { echo "Usage: preflight.sh --all [--manifest QWEN_JSON] [--report OUTPUT]"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --all) RUN_ALL=1 ;;
    --manifest) shift; MANIFEST="${1:-}" ;;
    --report) shift; REPORT="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[ "$RUN_ALL" -eq 1 ] || { usage >&2; exit 2; }

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST is required}"
: "${VLLM_ORIGIN_KEY_FILE:?VLLM_ORIGIN_KEY_FILE is required}"
MIN_DISK_AVAILABLE_GIB="${MIN_DISK_AVAILABLE_GIB:-300}"
MIN_PRESTOP_MEM_AVAILABLE_GIB="${MIN_PRESTOP_MEM_AVAILABLE_GIB:-32}"
WORKER_DIR="${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$ROOT_DIR}}"
HEAD_MANIFEST="${HEAD_ARTIFACT_MANIFEST:-$ROOT_DIR/artifacts/model-manifest.json}"
WORKER_MANIFEST="${WORKER_ARTIFACT_MANIFEST:-$ROOT_DIR/artifacts/model-manifest.worker.json}"

ENV_FILE="$ENV_FILE" VALIDATE_RENDER=0 "$ROOT_DIR/validate-dspark-config.sh"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" true
command -v mpirun >/dev/null || { echo "mpirun is required on the head for the NCCL gate." >&2; exit 1; }
ssh "$WORKER_HOST" "command -v orted >/dev/null || command -v prte >/dev/null" || {
  echo "An OpenMPI remote launcher (orted/prte) is required on the worker." >&2
  exit 1
}

check_capacity() {
  local role="$1" host="$2" disk mem disk_command mem_command
  disk_command="df -BG --output=avail /home/plexiz | tail -1 | tr -dc '0-9'"
  mem_command="awk '/MemAvailable/ {print int(\$2/1024/1024)}' /proc/meminfo"
  if [ -z "$host" ]; then
    disk="$(bash -c "$disk_command")"
    mem="$(bash -c "$mem_command")"
  else
    disk="$(ssh "$host" "$disk_command")"
    mem="$(ssh "$host" "$mem_command")"
  fi
  [ "$disk" -ge "$MIN_DISK_AVAILABLE_GIB" ] || { echo "$role disk gate failed: ${disk}GiB" >&2; return 1; }
  [ "$mem" -ge "$MIN_PRESTOP_MEM_AVAILABLE_GIB" ] || { echo "$role MemAvailable gate failed: ${mem}GiB" >&2; return 1; }
}
check_capacity head ""
check_capacity worker "$WORKER_HOST"

head_image="$(docker image inspect "$DSPARK_VLLM_IMAGE" -f '{{.Id}}')"
worker_image="$(ssh "$WORKER_HOST" "docker image inspect '$DSPARK_VLLM_IMAGE' -f '{{.Id}}'")"
[ "$head_image" = "$worker_image" ] || { echo "Pinned image IDs differ." >&2; exit 1; }

[ -f "$HEAD_MANIFEST" ] && [ -f "$WORKER_MANIFEST" ] || { echo "Artifact manifests are missing." >&2; exit 1; }
python3 "$ROOT_DIR/scripts/verify-artifact-manifest.py" compare \
  --left "$HEAD_MANIFEST" --right "$WORKER_MANIFEST"

HEAD_HOST=localhost WORKER_HOST="$WORKER_HOST" \
  "$ROOT_DIR/deployments/private-smoke/scripts/verify-fabric.sh" --require-persistent

for port in "${VLLM_PORT:-8889}" "${VLLM_PROXY_PORT:-8888}"; do
  if ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .; then
    echo "Required DSpark port is already listening: $port" >&2
    exit 1
  fi
done
key_mode="$(stat -c '%a' "$VLLM_ORIGIN_KEY_FILE")"
[ "$key_mode" = "600" ] && [ ! -L "$VLLM_ORIGIN_KEY_FILE" ] || { echo "Origin key file gate failed." >&2; exit 1; }

if [ -n "$REPORT" ]; then
  [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] || { echo "--report requires --manifest" >&2; exit 1; }
  python3 - "$MANIFEST" "$REPORT" <<'PY'
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
import sys
manifest, output = map(Path, sys.argv[1:])
item = json.loads(manifest.read_text())
report = {
    "accepted": True,
    "run_id": item["run_id"],
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gates": {"fabric": True, "artifacts": True, "direct": False, "hermes": False},
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
output.chmod(0o600)
PY
fi
echo "All pre-Qwen DSpark preflight gates passed."
