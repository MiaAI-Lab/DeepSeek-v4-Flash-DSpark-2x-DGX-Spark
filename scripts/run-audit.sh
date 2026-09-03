#!/usr/bin/env bash
# run-audit.sh — full DS4 DSpark serving audit (methodology documented in scripts/EVAL.md)
#
# Runs, against a LIVE endpoint:
#   Phase 1  Throughput matrix  — scripts/bench-miaai.py (MiaAI 08-14 methodology)
#   Phase 2  Spec-decode health — scripts/spec-acceptance.py (acceptance rate)
#   Phase 3  RULER-lite quality — scripts/ruler-lite.py (retrieval/tracing/aggregation at depth)
#   Phase 4  Tool calling       — scripts/tool-battery.py (incl. issue55 truncation at depth)
#   Phase 5  Garble sweep       — scripts/context-garble-sweep.py (cold prefill, tokenize-verified)
#
# Usage: bash scripts/run-audit.sh [--base-url http://127.0.0.1:8888/v1] [--model deepseek-v4-flash-0731]
#        [--lengths 8192,32768,131072,262144] [--tool-lengths 32768,131072] [--garble-lengths 2048,32768,131072]
#        [--request-timeout SECONDS]
# Exit 0 = all phases pass, 1 = any failure.
#
# --request-timeout raises the RULER-lite client HTTP timeout (phase 3). Its
# 900 s default cannot finish a prefill past ~790k tokens on this cluster, so
# pair it with any --lengths value near the 1M ceiling. Unset = script default.
set -u
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-deepseek-v4-flash-0731}"
LENGTHS="${LENGTHS:-8192,32768,131072,262144}"
TOOL_LENGTHS="${TOOL_LENGTHS:-32768,131072}"
GARBLE_LENGTHS="${GARBLE_LENGTHS:-2048,32768,131072}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --tool-lengths) TOOL_LENGTHS="$2"; shift 2 ;;
    --garble-lengths) GARBLE_LENGTHS="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP=$(date +%Y%m%d-%H%M%S)
REPORT_DIR="results/audit-${STAMP}"
mkdir -p "$REPORT_DIR"
echo "=== DS4 audit $STAMP | $BASE_URL | $MODEL ==="

fail=0
run_phase() {
  local name="$1" log="$2"
  shift 2
  echo; echo "########## Phase: $name ##########"
  set +e
  "$@" | tee "$log"
  local rc="${PIPESTATUS[0]}"
  if [ "$rc" -eq 0 ]; then echo "## $name: PASS"; else echo "## $name: FAIL"; fail=1; fi
}

run_phase "1 throughput" "$REPORT_DIR/throughput.log" python3 "$SCRIPT_DIR/bench-miaai.py" --base-url "$BASE_URL" --model "$MODEL" \
  --prompt 256 --concurrency 1 --repeat 5

run_phase "2 spec-acceptance" "$REPORT_DIR/acceptance.log" python3 "$SCRIPT_DIR/spec-acceptance.py" --base-url "$BASE_URL" \
  --model "$MODEL" --trials 5 --bench-script "$SCRIPT_DIR/bench-miaai.py"

RULER_TIMEOUT_ARG=()
if [ -n "$REQUEST_TIMEOUT" ]; then
  RULER_TIMEOUT_ARG=(--request-timeout "$REQUEST_TIMEOUT")
fi
run_phase "3 ruler-lite quality" "$REPORT_DIR/ruler.log" python3 "$SCRIPT_DIR/ruler-lite.py" --base-url "$BASE_URL" \
  --model "$MODEL" --lengths "$LENGTHS" "${RULER_TIMEOUT_ARG[@]}" --output "$REPORT_DIR/ruler-lite.json"

run_phase "4 tool battery" "$REPORT_DIR/tool.log" python3 "$SCRIPT_DIR/tool-battery.py" "$BASE_URL/chat/completions" "$MODEL"

run_phase "5 deep-context tool" "$REPORT_DIR/deeptool.log" python3 "$SCRIPT_DIR/deepctx-tool-battery.py" "$BASE_URL/chat/completions" "$MODEL" "$TOOL_LENGTHS"

run_phase "6 garble sweep" "$REPORT_DIR/garble.log" python3 "$SCRIPT_DIR/context-garble-sweep.py" --url "$BASE_URL" \
  --model "$MODEL" --lengths "$GARBLE_LENGTHS" --runs 1 --out "$REPORT_DIR/garble.md"

echo; echo "=== AUDIT COMPLETE: $([ $fail -eq 0 ] && echo ALL-PASS || echo FAILURES) — reports in $REPORT_DIR ==="
exit $fail
