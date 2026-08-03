#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MANIFEST="${QWEN_MANIFEST:-$ROOT_DIR/artifacts/qwen-manifest.json}"
GATE_REPORT=""
QUARANTINE_ROOT="${QWEN_QUARANTINE_ROOT:-/home/plexiz/.dspark-qwen-quarantine}"

usage() { echo "Usage: purge-qwen.sh --gate-report FILE [--manifest FILE]"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) shift; MANIFEST="${1:-}" ;;
    --gate-report) shift; GATE_REPORT="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[ -f "$MANIFEST" ] || { echo "Missing manifest." >&2; exit 1; }
[ -n "$GATE_REPORT" ] && [ -f "$GATE_REPORT" ] || { echo "Missing accepted gate report." >&2; exit 1; }
# The helper compares manifest_sha256, accepted, run_id, realpath, st_dev,
# st_ino, owner, type, and symlink state immediately before quarantine.
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live --manifest "$MANIFEST" --max-age-hours 24
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report \
  --manifest "$MANIFEST" --report "$GATE_REPORT" --max-age-hours 24 \
  --required-gate fabric --required-gate artifacts --required-gate direct --required-gate hermes
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-targets --manifest "$MANIFEST"

docker ps -q --filter name='^/urbanplan-qwen$' | grep -q . && {
  echo "Qwen is still running; purge refused." >&2
  exit 1
}
[ -t 0 ] || { echo "Purge is forbidden without an interactive terminal (-t 0)." >&2; exit 1; }
run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$MANIFEST")"
printf 'Type PURGE QWEN %s to quarantine exact targets: ' "$run_id" >&2
read -r confirmation
[ "$confirmation" = "PURGE QWEN $run_id" ] || { echo "Confirmation mismatch." >&2; exit 1; }

image_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["container"]["image_id"])' "$MANIFEST")"
qwen_id="$(docker inspect -f '{{.Id}}' urbanplan-qwen)"
other_users="$(docker ps -aq --no-trunc --filter "ancestor=$image_id" | grep -Fvx "$qwen_id" || true)"
[ -z "$other_users" ] || { echo "Other containers still reference the Qwen image." >&2; exit 1; }
quarantine="$(python3 "$ROOT_DIR/scripts/qwen_manifest.py" quarantine \
  --manifest "$MANIFEST" --quarantine-root "$QUARANTINE_ROOT")"
docker rm urbanplan-qwen >/dev/null
docker image rm "$image_id" >/dev/null

printf 'Type DELETE QWEN %s to delete the quarantine permanently: ' "$run_id" >&2
read -r delete_confirmation
if [ "$delete_confirmation" != "DELETE QWEN $run_id" ]; then
  echo "Targets remain recoverable in quarantine: $quarantine" >&2
  exit 1
fi
[ "$(realpath "$quarantine")" = "$(realpath "$QUARANTINE_ROOT")/$run_id" ] || exit 1
find "$quarantine" -depth -delete
[ ! -e "$quarantine" ] || { echo "Quarantine deletion was incomplete." >&2; exit 1; }
docker image inspect "$image_id" >/dev/null 2>&1 && { echo "Qwen image remains." >&2; exit 1; }
for path in \
  /home/plexiz/services/urbanplan-spark-api \
  /home/plexiz/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4; do
  [ ! -e "$path" ] || { echo "Qwen target remains: $path" >&2; exit 1; }
done
echo "Qwen filesystem targets were quarantined, deleted, and proven absent."
