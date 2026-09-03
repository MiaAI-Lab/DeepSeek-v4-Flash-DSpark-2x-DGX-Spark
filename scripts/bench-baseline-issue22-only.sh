#!/usr/bin/env bash
# bench-baseline-issue22-only.sh — Benchmark with Issue #22 only (no perf patches)
#
# This script:
#   1. Stops the current patched containers
#   2. Starts fresh containers (no hotfixes)
#   3. Applies ONLY Issue #22
#   4. Runs the TTFT benchmark
#   5. Saves results as baseline
#   6. Restarts the patched containers
#
# Usage:
#   bash scripts/bench-baseline-issue22-only.sh [--num-prompts 10]
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
echo "║  BASELINE BENCHMARK (Issue #22 only, no perf patches)      ║"
echo "║  This will restart the server with only the bugfix.        ║"
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
echo "Step 1/5: Stopping current containers..."
env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose -p deepseek-v4-flash --env-file .env.dspark \
  -f docker-compose.dspark.yml down 2>/dev/null || true
sleep 3
echo "  ✓ Containers stopped"

# ── Step 2: Start WITHOUT hotfixes ────────────────────────────────────────
echo ""
echo "Step 2/5: Starting server WITHOUT patches (DSPARK_SKIP_HOTFIX=1)..."
DSPARK_SKIP_HOTFIX=1 bash "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh" &
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

# ── Step 3: Apply ONLY Issue #22 ──────────────────────────────────────────
echo ""
echo "Step 3/5: Applying Issue #22 only..."
docker exec deepseek-v4-flash-vllm-dspark-1 bash /tmp/hotfix-nvfp4-ds-mla-issue22.sh 2>&1 | tail -5

# Verify: Issue #22 applied, others NOT
echo "  Verifying patch state..."
VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
CHECKS=$(docker exec deepseek-v4-flash-vllm-dspark-1 bash -c "
  c22=\$(grep -c 'nvfp4_ds_mla' '$VLLM/models/deepseek_v4/sparse_mla.py' 2>/dev/null || echo 0)
  c49=\$(grep -c 'PORT #49486' '$VLLM/models/deepseek_v4/attention.py' 2>/dev/null || echo 0)
  c50=\$(grep -c 'needs_mtp_hidden_states' '$VLLM/models/deepseek_v4/nvidia/model.py' 2>/dev/null || echo 0)
  c07=\$(grep -c 'dense_mha_metadata_layer_name' '$VLLM/model_executor/layers/sparse_attn_indexer.py' 2>/dev/null || echo 0)
  echo \$c22 \$c49 \$c50 \$c07
" 2>/dev/null)
echo "  Issue #22:$(echo $CHECKS | cut -d' ' -f1)  #49486:$(echo $CHECKS | cut -d' ' -f2)  #50312:$(echo $CHECKS | cut -d' ' -f3)  #48407:$(echo $CHECKS | cut -d' ' -f4)"
echo "  (expect: Issue #22 ≥2, others = 0)"

# ── Step 4: Run benchmark ─────────────────────────────────────────────────
echo ""
echo "Step 4/5: Running TTFT benchmark (Issue #22 only)..."
python3 "$SCRIPT_DIR/scripts/bench-ttft.py" \
  --prompt-len 256,512,1024,2048,4096,65536,131072,262144 \
  --num-prompts "$NUM_PROMPTS" \
  --output "$SCRIPT_DIR/results/bench-baseline-issue22-only.json"

# ── Step 5: Restart with all patches ──────────────────────────────────────
echo ""
echo "Step 5/5: Restarting server WITH all patches..."
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
echo "║  Issue #22 only: results/bench-baseline-issue22-only.json  ║"
echo "║  All patches:    results/bench-patched-full.json           ║"
echo "║  No patches:     results/bench-baseline-no-patches.json    ║"
echo "║                                                             ║"
echo "║  Compare: python3 scripts/compare-bench.py                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
