#!/usr/bin/env bash
# bench-patches.sh — Quick performance/stability check for v0.27 hotfix patches
#
# Usage:
#   bash scripts/bench-patches.sh [--quick|--full]
#
# What it tests:
#   #49486 (+#52492 capture guard): Short-context TTFT (≤2048 tokens) — should be faster
#   #50312: KV budget — should be UNCHANGED (compute-only saving)
#   #48407: Dormant — no effect expected
#   issue-22: Long-context decode — should not regress
#
# Note: #50004 (adaptive C128A topk width) was removed from the chain — upstream
# vLLM reverted it in #51318 (capture/replay row-stride corruption).
#
# Outputs:
#   results/patch-bench-<timestamp>.txt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
API="http://127.0.0.1:8888"
MODEL="deepseek-v4-flash-0731"
MODE="${1:---quick}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_DIR/patch-bench-${TIMESTAMP}.txt"
mkdir -p "$RESULTS_DIR"

log() { echo "$@" | tee -a "$OUT"; }

separator() { log ""; log "═══════════════════════════════════════════════════════════"; log "$@"; log "═══════════════════════════════════════════════════════════"; }

# ── Pre-flight ──────────────────────────────────────────────────────────────
if ! curl -fsS --max-time 3 "$API/v1/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM API not reachable at $API" >&2; exit 1
fi

log "Patch performance bench — $(date -Is)"
log "Mode: $MODE"
log ""

# ── 1. KV budget check (#50312) ────────────────────────────────────────────
separator "1. KV BUDGET (#50312 — should be UNCHANGED)"
KV_BEFORE="results/hotfix-50312-kv-baseline.txt"
KV_NOW=$(mktemp)
docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 \
  | grep -E "Available KV cache memory:|GPU KV cache size:|Maximum concurrency for" \
  | tail -6 > "$KV_NOW" || true

if [ -f "$KV_BEFORE" ]; then
  log "Baseline (pre-patch):"
  grep -E "Available KV|GPU KV|Maximum" "$KV_BEFORE" | sed 's/^/  /' | tee -a "$OUT"
  log "Current (post-patch):"
  grep -E "Available KV|GPU KV|Maximum" "$KV_NOW" | sed 's/^/  /' | tee -a "$OUT"
  # Extract numbers for comparison
  KV_BEFORE_SIZE=$(grep "GPU KV cache size" "$KV_BEFORE" | grep -oE '[0-9,]+ tokens' | head -1)
  KV_NOW_SIZE=$(grep "GPU KV cache size" "$KV_NOW" | grep -oE '[0-9,]+ tokens' | head -1)
  if [ "$KV_BEFORE_SIZE" = "$KV_NOW_SIZE" ]; then
    log "  ✅ KV cache size UNCHANGED ($KV_NOW_SIZE) — expected for #50312"
  else
    log "  ⚠️  KV cache size CHANGED: $KV_BEFORE_SIZE → $KV_NOW_SIZE"
  fi
else
  log "  (no baseline — first run after restart)"
  grep -E "Available KV|GPU KV|Maximum" "$KV_NOW" | sed 's/^/  /' | tee -a "$OUT"
fi
rm -f "$KV_NOW"

# ── 2. Short-context TTFT (#49486) ────────────────────────────────────────
separator "2. SHORT-CONTEXT TTFT (#49486 — should be ≤ pre-patch)"
log "Testing prompt lengths: 256, 512, 1024, 2048 tokens"
log ""

run_ttft() {
  local prompt_tokens=$1
  local label=$2
  # Generate a prompt of ~prompt_tokens words (rough: 1 token ≈ 1.3 chars)
  local prompt
  prompt=$(python3 -c "print('hello ' * ($prompt_tokens * 4 / 3 // 6))" 2>/dev/null || echo "hello world")
  
  local start_ms end_ms ttft_ms
  start_ms=$(date +%s%3N)
  
  local resp
  resp=$(curl -s --max-time 60 "$API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"$prompt\",
      \"max_tokens\": 1,
      \"temperature\": 0
    }" 2>/dev/null) || { log "  $label: TIMEOUT"; return; }
  
  end_ms=$(date +%s%3N)
  ttft_ms=$((end_ms - start_ms))
  
  local usage
  usage=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null || echo "?")
  
  log "  $label (${usage} tokens): ${ttft_ms} ms"
}

run_ttft 256   "prompt=256"
run_ttft 512   "prompt=512"
run_ttft 1024  "prompt=1024"
run_ttft 2048  "prompt=2048"

if [ "$MODE" = "--full" ]; then
  log ""
  log "Full mode — testing longer prompts:"
  run_ttft 4096  "prompt=4096"
  run_ttft 8192  "prompt=8192"
fi

# ── 3. Decode stability (#50312) ───────────────────────────────────────────
separator "3. DECODE STABILITY (5-turn generate, check for errors)"
for i in 1 2 3 4 5; do
  local_start=$(date +%s%3N)
  resp=$(curl -s --max-time 30 "$API/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Say exactly: turn $i ok\"}],
      \"max_tokens\": 10,
      \"temperature\": 0
    }" 2>/dev/null) || { log "  Turn $i: TIMEOUT"; continue; }
  local_end=$(date +%s%3N)
  local_ms=$((local_end - local_start))
  
  text=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
  log "  Turn $i: ${local_ms} ms  |  $text"
done

# ── 4. Check for errors in logs ────────────────────────────────────────────
separator "4. ERROR CHECK (last 1000 lines of vllm log)"
ERROR_COUNT=$(docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | tail -1000 | grep -c "ERROR\|Traceback\|RuntimeError\|CUDA error" || true)
if [ "$ERROR_COUNT" -eq 0 ]; then
  log "  ✅ No errors in last 1000 log lines"
else
  log "  ⚠️  $ERROR_COUNT error lines found — check docker logs"
  docker logs deepseek-v4-flash-vllm-dspark-1 2>&1 | tail -1000 \
    | grep "ERROR\|Traceback\|RuntimeError\|CUDA error" | tail -5 | sed 's/^/  /' | tee -a "$OUT"
fi

# ── 5. Prompt-length histogram (DSpark acceptance) ─────────────────────────
separator "5. REQUEST HISTOGRAM (from /metrics)"
curl -s --max-time 5 "$API/metrics" 2>/dev/null \
  | grep -E "vllm:request_prompt_tokens_(count|sum|bucket)" \
  | head -20 | sed 's/^/  /' | tee -a "$OUT" || log "  (no metrics available)"

# ── Summary ─────────────────────────────────────────────────────────────────
separator "SUMMARY"
log "Results saved to: $OUT"
log ""
log "Patch-specific expectations:"
log "  #49486: Short-context TTFT should be ≤ pre-patch (indexer skip; #52492 guard keeps captured graphs on the scored path)"
log "  #50312: KV budget UNCHANGED (conditional buffer, no KV impact)"
log "  #48407: No effect (dormant — dense_mha_metadata_layer_name=\"\")"
log "  issue-22: Long-context decode should not regress"
log ""
log "For deeper testing, use: vllm bench serve --model $MODEL <args>"
