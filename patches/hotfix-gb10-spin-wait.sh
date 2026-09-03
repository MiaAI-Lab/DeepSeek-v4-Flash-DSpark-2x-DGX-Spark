#!/usr/bin/env bash
# hotfix-gb10-spin-wait.sh — Reduce vLLM IPC spin-wait CPU load / SoC heat on GB10
#
# ROOT CAUSE: vLLM's cross-process IPC wait path (shm_broadcast.SpinCondition)
# spins on performance cores for `busy_loop_s` (default 1s) before sleeping.
# Under decode the message gap is always < 1s, so the sleep path never engages
# and 3–4 P-cores spin at max clock (issue #79).
#
# FIX (one value):
#   vllm/distributed/device_communicators/shm_broadcast.py
#   busy_loop_s: float = 1 → 0.002  (2ms grace, then sleep via notify socket)
#
# TP>=2 only (this recipe). TP=1 uses the uni executor and never forms this
# path. Wait policy only — model outputs unchanged.
# Skip: DSPARK_SKIP_SPIN_WAIT_HOTFIX=1
#
# Applied from the compose entrypoint before exec vllm. Idempotent.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
SPIN_WAIT_FILE="$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"
READY_OLD="busy_loop_s: float = 1"
READY_NEW="busy_loop_s: float = 0.002"

if [ ! -f "$SPIN_WAIT_FILE" ]; then
  echo "ERROR: vLLM shm_broadcast.py not found at $SPIN_WAIT_FILE" >&2
  exit 1
fi

echo "=== Hotfix: GB10 vLLM IPC spin-wait (CPU load / SoC heat, issue #79) ==="
echo "vLLM file: $SPIN_WAIT_FILE"

if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo " [skip] busy_loop_s already set to 0.002"
elif grep -qF "$READY_OLD" "$SPIN_WAIT_FILE"; then
  python3 - "$SPIN_WAIT_FILE" "$READY_OLD" "$READY_NEW" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text()
old, new = sys.argv[2], sys.argv[3]
count = text.count(old)
assert count == 1, f"expected exactly one '{old}', found {count}"
p.write_text(text.replace(old, new, 1))
print(" [OK] busy_loop_s 1 -> 0.002 (2ms grace before sleep)")
PYEOF
else
  echo " [WARN] Could not locate '${READY_OLD}' in $SPIN_WAIT_FILE" >&2
  echo " Inspect manually; skipping." >&2
fi

echo "=== Verification ==="
if grep -qF "$READY_NEW" "$SPIN_WAIT_FILE"; then
  echo "[OK] busy_loop_s = 0.002 — IPC wait sleeps after 2ms instead of spinning 1s."
else
  echo "[FAIL] busy_loop_s not set to 0.002" >&2
  exit 1
fi
