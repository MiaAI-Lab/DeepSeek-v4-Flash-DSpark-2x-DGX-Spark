#!/usr/bin/env bash
# bench-baseline-no-patches.sh — Run benchmark WITHOUT any patches applied
#
# This script:
#   1. Stops the current patched containers
#   2. Starts fresh containers (no hotfixes)
#   3. Runs the TTFT benchmark
#   4. Saves results as baseline
#   5. Restarts the patched containers
#
# Usage:
#   bash scripts/bench-baseline-no-patches.sh [--num-prompts 10]
#
# ⚠️  This restarts the vLLM server — expect ~5 min downtime for model reload.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NUM_PROMPTS="${1:---num-prompts}"
if [ "$NUM_PROMPTS" = "--num-prompts" ]; then
  shift 2>/dev/null || true
  NUM_PROMPTS="${1:-10}"
fi
source "$SCRIPT_DIR/.env.dspark" 2>/dev/null || true

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  BASELINE BENCHMARK (no patches)                           ║"
echo "║  This will restart the server without hotfixes.            ║"
echo "║  Expect ~5 min downtime for model reload.                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
read -p "Continue? [y/N] " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# ── Step 1: Stop current containers ────────────────────────────────────────
echo ""
echo "Step 1/4: Stopping current containers..."
env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose -p deepseek-v4-flash --env-file .env.dspark \
  -f docker-compose.dspark.yml down 2>/dev/null || true
sleep 3
echo "  ✓ Containers stopped"

# ── Step 2: Start WITHOUT hotfixes ────────────────────────────────────────
echo ""
echo "Step 2/4: Starting server WITHOUT patches (DSPARK_SKIP_HOTFIX=1 DSPARK_SKIP_ISSUE22_HOTFIX=1)..."
DSPARK_SKIP_HOTFIX=1 DSPARK_SKIP_ISSUE22_HOTFIX=1 bash "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh" &
START_PID=$!

# Wait for API to be ready
echo -n "  Waiting for API..."
for i in $(seq 1 60); do
  sleep 10
  if curl -fsS --max-time 3 http://127.0.0.1:8888/v1/models 2>/dev/null | grep -q "deepseek"; then
    echo " READY (${i}0s)"
    break
  fi
  echo -n "."
done

# Verify no patches applied
echo "  Verifying no patches..."
VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
CHECKS=$(docker exec deepseek-v4-flash-vllm-dspark-1 bash -c "
  c1=\$(grep -c 'PORT #49486' '$VLLM/models/deepseek_v4/attention.py' 2>/dev/null || true); c1=\${c1:-0}
  c2=\$(grep -c 'needs_mtp_hidden_states' '$VLLM/models/deepseek_v4/nvidia/model.py' 2>/dev/null || true); c2=\${c2:-0}
  c3=\$(grep -c 'active_topk_width' '$VLLM/models/deepseek_v4/sparse_mla.py' 2>/dev/null || true); c3=\${c3:-0}
  c4=\$(grep -c 'dense_mha_metadata_layer_name' '$VLLM/model_executor/layers/sparse_attn_indexer.py' 2>/dev/null || true); c4=\${c4:-0}
  echo \$c1 \$c2 \$c3 \$c4
" 2>/dev/null)
echo "  Patch check (should be 0 0 0 0): $CHECKS"

# ── Step 3: Run benchmark ─────────────────────────────────────────────────
echo ""
echo "Step 3/4: Running TTFT benchmark (no patches)..."
python3 "$SCRIPT_DIR/scripts/bench-ttft.py" \
  --prompt-len 256,512,1024,2048,4096,65536,131072,262144 \
  --num-prompts "$NUM_PROMPTS" \
  --output "$SCRIPT_DIR/results/bench-baseline-no-patches.json"

# ── Step 4: Restart with patches ──────────────────────────────────────────
echo ""
echo "Step 4/4: Restarting server WITH patches..."
kill $START_PID 2>/dev/null || true
env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose -p deepseek-v4-flash --env-file .env.dspark \
  -f docker-compose.dspark.yml down 2>/dev/null || true
sleep 3

bash "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh" &
START_PID2=$!

echo -n "  Waiting for API..."
for i in $(seq 1 60); do
  sleep 10
  if curl -fsS --max-time 3 http://127.0.0.1:8888/v1/models 2>/dev/null | grep -q "deepseek"; then
    echo " READY (${i}0s)"
    break
  fi
  echo -n "."
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DONE                                                      ║"
echo "║  Baseline saved: results/bench-baseline-no-patches.json    ║"
echo "║  Patched results: results/bench-20260811-222027.json       ║"
echo "║                                                             ║"
echo "║  Compare with:                                             ║"
echo "║    python3 scripts/compare-bench.py                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
