#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MANIFEST="${QWEN_MANIFEST:-$ROOT_DIR/artifacts/qwen-manifest.json}"
GATE_REPORT=""
QUARANTINE_ROOT="${QWEN_QUARANTINE_ROOT:-/home/plexiz/.dspark-qwen-quarantine}"
VERIFY_ONLY=0
DSPARK_ENV_FILE="${DSPARK_ENV_FILE:-$ROOT_DIR/.env.dspark}"
LITELLM_ENV_FILE="${LITELLM_ENV_FILE:-$ROOT_DIR/deployments/private-smoke/litellm/.env}"

usage() { echo "Usage: purge-qwen.sh --gate-report FILE [--manifest FILE] | --verify-only"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) shift; MANIFEST="${1:-}" ;;
    --gate-report) shift; GATE_REPORT="${1:-}" ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$VERIFY_ONLY" -eq 1 ]; then
  # Static, nondestructive operator check. If explicit evidence paths are
  # supplied, also validate their hash binding and freshness.
  grep -F 'PURGE QWEN' "$0" >/dev/null
  grep -F 'DELETE QWEN' "$0" >/dev/null
  grep -F 'find "$quarantine" -depth -delete' "$0" >/dev/null
  if [ -n "$GATE_REPORT" ]; then
    [ -f "$MANIFEST" ] && [ -n "$GATE_REPORT" ] && [ -f "$GATE_REPORT" ] || {
      echo "Live verify-only requires both manifest and gate report." >&2
      exit 1
    }
    python3 "$SCRIPT_DIR/sanitize-evidence.py" \
      --schema "$ROOT_DIR/deployments/private-smoke/schemas/acceptance.schema.json" \
      --input "$GATE_REPORT" >/dev/null
    python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report \
      --manifest "$MANIFEST" --report "$GATE_REPORT" --max-age-hours 24 \
      --required-gate fabric --required-gate artifacts --required-gate direct --required-gate hermes
  fi
  echo "Qwen purge contract is present; no target was changed."
  exit 0
fi

[ -f "$MANIFEST" ] || { echo "Missing manifest." >&2; exit 1; }
[ -n "$GATE_REPORT" ] && [ -f "$GATE_REPORT" ] || { echo "Missing accepted gate report." >&2; exit 1; }
# The helper compares manifest_sha256, accepted, run_id, realpath, st_dev,
# st_ino, owner, type, and symlink state immediately before quarantine.
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live --manifest "$MANIFEST" --max-age-hours 24
python3 "$SCRIPT_DIR/sanitize-evidence.py" \
  --schema "$ROOT_DIR/deployments/private-smoke/schemas/acceptance.schema.json" \
  --input "$GATE_REPORT" >/dev/null
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report \
  --manifest "$MANIFEST" --report "$GATE_REPORT" --max-age-hours 24 \
  --required-gate fabric --required-gate artifacts --required-gate direct --required-gate hermes
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-targets --manifest "$MANIFEST"
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-no-supervisors --manifest "$MANIFEST"

# Re-prove the replacement immediately before the irreversible operation.
ENV_FILE="$DSPARK_ENV_FILE" "$ROOT_DIR/status-deepseek-v4-flash-dspark.sh" --expect running
LITELLM_ENV_FILE="$LITELLM_ENV_FILE" \
  "$ROOT_DIR/deployments/private-smoke/litellm/smoke.sh" --all-interfaces

docker ps -q --filter name='^/urbanplan-qwen$' | grep -q . && {
  echo "Qwen is still running; purge refused." >&2
  exit 1
}
[ -t 0 ] || { echo "Purge is forbidden without an interactive terminal (-t 0)." >&2; exit 1; }
run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$MANIFEST")"
manifest_hash="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$MANIFEST")"
printf 'Type PURGE QWEN %s %s to quarantine exact targets: ' "$run_id" "$manifest_hash" >&2
read -r confirmation
[ "$confirmation" = "PURGE QWEN $run_id $manifest_hash" ] || { echo "Confirmation mismatch." >&2; exit 1; }

image_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["container"]["image_id"])' "$MANIFEST")"
qwen_id="$(docker inspect -f '{{.Id}}' urbanplan-qwen)"
other_users="$(docker ps -aq --no-trunc --filter "ancestor=$image_id" | grep -Fvx "$qwen_id" || true)"
[ -z "$other_users" ] || { echo "Other containers still reference the Qwen image." >&2; exit 1; }
quarantine="$(python3 "$ROOT_DIR/scripts/qwen_manifest.py" quarantine \
  --manifest "$MANIFEST" --quarantine-root "$QUARANTINE_ROOT")"
docker rm urbanplan-qwen >/dev/null
docker image rm "$image_id" >/dev/null

printf 'Type DELETE QWEN %s %s to delete the quarantine permanently: ' "$run_id" "$manifest_hash" >&2
read -r delete_confirmation
if [ "$delete_confirmation" != "DELETE QWEN $run_id $manifest_hash" ]; then
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
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-no-supervisors --manifest "$MANIFEST"
echo "Qwen filesystem targets were quarantined, deleted, and proven absent."
