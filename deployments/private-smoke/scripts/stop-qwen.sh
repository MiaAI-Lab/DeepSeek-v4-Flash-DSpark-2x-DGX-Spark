#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MANIFEST="${QWEN_MANIFEST:-$ROOT_DIR/artifacts/qwen-manifest.json}"
GATE_REPORT=""
VERIFY_ONLY=0
STOP_WAIT_SECONDS="${STOP_WAIT_SECONDS:-75}"

usage() { echo "Usage: stop-qwen.sh [--verify-only] [--manifest FILE] [--gate-report FILE]"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1 ;;
    --manifest) shift; MANIFEST="${1:-}" ;;
    --gate-report) shift; GATE_REPORT="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

verify_stopped() {
  if docker ps -q --filter name='^/urbanplan-qwen$' | grep -q .; then
    echo "urbanplan-qwen is still running." >&2
    return 1
  fi
  if ss -ltn '( sport = :8000 )' | tail -n +2 | grep -q .; then
    echo "port 8000 is still listening." >&2
    return 1
  fi
  if systemctl list-units --state=active --plain --no-legend | grep -Ei 'qwen|urbanplan' >/dev/null; then
    echo "an active Qwen/Urbanplan supervisor remains." >&2
    return 1
  fi
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
  verify_stopped
  echo "Qwen is stopped and port 8000 has no listener."
  exit 0
fi

[ -f "$MANIFEST" ] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }
[ -n "$GATE_REPORT" ] && [ -f "$GATE_REPORT" ] || { echo "A gate report is required." >&2; exit 1; }
# The verifier requires accepted: true, an exact manifest_sha256, a matching
# run_id, freshness, and passing fabric/artifacts gates.
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-live --manifest "$MANIFEST" --max-age-hours 24
python3 "$ROOT_DIR/scripts/qwen_manifest.py" verify-report \
  --manifest "$MANIFEST" --report "$GATE_REPORT" --max-age-hours 24 \
  --required-gate fabric --required-gate artifacts

[ -t 0 ] || { echo "Qwen stop requires an interactive terminal." >&2; exit 1; }
printf 'Type STOP urbanplan-qwen to continue: ' >&2
read -r confirmation
[ "$confirmation" = "STOP urbanplan-qwen" ] || { echo "Confirmation mismatch." >&2; exit 1; }

IFS=$'\t' read -r project config_file service < <(
  python3 - "$MANIFEST" <<'PY'
import json, sys
item = json.load(open(sys.argv[1]))["container"]
print(item["project"], item["config_file"], item["service"], sep="\t")
PY
)
[ "$project" = "urbanplan-spark-api" ] && [ "$service" = "qwen" ] || exit 1
[ "$(realpath "$config_file")" = "/home/plexiz/services/urbanplan-spark-api/docker-compose.yml" ] || exit 1
docker compose -p "$project" -f "$config_file" stop "$service"

deadline=$((SECONDS + STOP_WAIT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  verify_stopped
  sleep 5
done
verify_stopped
echo "Qwen stopped and remained stopped for ${STOP_WAIT_SECONDS}s. It was not reactivated."
