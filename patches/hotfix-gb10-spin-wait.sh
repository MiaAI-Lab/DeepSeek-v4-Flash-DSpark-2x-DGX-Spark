#!/usr/bin/env bash
# hotfix-gb10-spin-wait.sh — Reduce vLLM IPC spin-wait CPU load / SoC heat on GB10
#
# ROOT CAUSE: vLLM's cross-process IPC wait path (shm_broadcast.SpinCondition)
# spins on performance cores for `busy_loop_s` (default 1s) before sleeping.
# Under decode the message gap is always < 1s, so the "sleep" path never engages
# and 3-4 P-cores spin at max clock indefinitely → severe wasted CPU (333%→89%)
# and SoC temperature rise on single-die GB10 packages (DGX Spark / Ascent GX10).
#
# FIX (one value):
#   vllm/distributed/device_communicators/shm_broadcast.py
#   busy_loop_s: float = 1  →  0.002   (2ms grace, then sleep via notify socket)
#
# Only relevant for multi-process (mp) executor deployments, i.e. TP>=2 (this 2x
# DGX Spark recipe). Single-GPU/world_size=1 uses the uni executor and this wait
# path never forms. Model outputs are unchanged (wait policy only). Throughput and
# first-token latency are unaffected — measurements in nacyot's
# "vllm-spin-wait-gb10" write-up.
#
# Usage:
#   docker exec <container> bash /opt/dspark-patches/hotfix-gb10-spin-wait.sh
#   # Applied automatically at container boot via docker-compose; no restart needed
#   # beyond the normal boot sequence. Idempotent — safe to re-run.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SPIN_WAIT_FILE="$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"
READY_OLD="busy_loop_s: float = 1"
READY_NEW="busy_loop_s: float = 0.002"

if [ ! -f "$SPIN_WAIT_FILE" ]; then
  echo "ERROR: vLLM shm_broadcast.py not found at $SPIN_WAIT_FILE" >&2
  exit 1
fi

echo "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat) ==="
echo "vLLM file: $SPIN_WAIT_FILE"

if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo "  [skip] busy_loop_s already set to 0.002 (already applied)"
elif grep -qF "$READY_OLD" "$SPIN_WAIT_FILE"; then
  python3 - "$SPIN_WAIT_FILE" "$READY_OLD" "$READY_NEW" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text()
old, new = sys.argv[2], sys.argv[3]
assert old in text, f"anchor not found: {old}"
p.write_text(text.replace(old, new, 1))
print("  [OK]   busy_loop_s 1 -> 0.002 (2ms grace before sleep)")
PYEOF
else
  echo "  [WARN] Could not locate 'busy_loop_s: float = 1' default in $SPIN_WAIT_FILE" >&2
  echo "         Inspect manually; skipping." >&2
fi

echo ""
echo "=== Verification ==="
if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo "[OK] busy_loop_s = 0.002 applied — vLLM IPC wait now sleeps after 2ms instead of spinning 1s."
else
  echo "[FAIL] busy_loop_s not set to 0.002" >&2
  exit 1
fi
