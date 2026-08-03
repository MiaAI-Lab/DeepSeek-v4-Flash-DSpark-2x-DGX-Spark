#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT="${QWEN_MANIFEST:-$ROOT_DIR/artifacts/qwen-manifest.json}"
CHECK_ONLY=0

usage() { echo "Usage: inventory-qwen.sh [--check-only] [--output FILE]"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --output) shift; OUTPUT="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$CHECK_ONLY" -eq 1 ]; then
  exec python3 "$ROOT_DIR/scripts/qwen_manifest.py" inventory \
    --container urbanplan-qwen --check-only
fi
python3 "$ROOT_DIR/scripts/qwen_manifest.py" inventory \
  --container urbanplan-qwen --output "$OUTPUT"
echo "Sanitized Qwen manifest written to $OUTPUT"
