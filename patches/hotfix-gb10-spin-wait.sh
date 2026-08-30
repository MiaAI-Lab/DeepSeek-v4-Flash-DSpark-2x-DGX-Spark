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

# === [shm-reader-recheck] bounded dispatch-ring reader re-check (#117 family) ===
# Upstream vLLM already ships exactly this fix (SHM_READER_RECHECK_INTERVAL_MS,
# timeout_ms() never returns None); this is a backport with a shorter ceiling
# (1000 ms) so a lost/coalesced PUB-SUB notify self-heals within ~1 s instead
# of black-holing the EngineCore->local-worker ring (dispatch-trace observed:
# "disp seq=27591 idx=0 (wrap 8->9->0) with no recv on rank 0", 600 s NCCL kill).
# https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/device_communicators/shm_broadcast.py
REC_FILE="$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"
if grep -qF "[shm-reader-recheck]" "$REC_FILE"; then
echo " [skip] shm-reader-recheck already applied"
else
python3 - "$REC_FILE" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()

# 1) module-level constant (after VLLM_RINGBUFFER_WARNING_INTERVAL).
CONST_ANCHOR = "VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL"
CONST_ADD = CONST_ANCHOR + (
    "\n\n# [shm-reader-recheck] bounded reader re-check for the EngineCore->local\n"
    "# worker dispatch ring; a lost notify must never hang the reader.\n"
    "# Upstream vLLM default 5000; this backport uses 1000 for a shorter\n"
    "# self-heal ceiling (negligible idle-wakeup cost).\n"
    "SHM_READER_RECHECK_INTERVAL_MS = 1000"
)
assert s.count(CONST_ANCHOR) == 1, "const anchor count != 1"
s = s.replace(CONST_ANCHOR, CONST_ADD, 1)

# 2) timeout_ms(): never return None; cap at the recheck interval.
OLD_METHOD = '''        def timeout_ms(self) -> int | None:
            """Returns a timeout that is:
            - min(time to deadline, time to next warning) if we're logging warnings
            - time to deadline, if we're not logging warnings
            - None if the timeout is None and we're not logging warnings
            - raise TimeoutError if we are past the deadline
            """
            warning_wait_time = self.warning_wait_time_ms
            if self.timeout is None:
                return warning_wait_time

            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError

            if warning_wait_time and warning_wait_time < time_left_ms:
                return warning_wait_time

            return time_left_ms
'''
NEW_METHOD = '''        def timeout_ms(self) -> int:
            """Returns a timeout capped at the reader recheck interval:
            min(time to deadline, time to next warning,
            SHM_READER_RECHECK_INTERVAL_MS). Never returns None, so a lost or
            coalesced notify cannot strand a reader in an indefinite poll.
            """
            wait_ms = SHM_READER_RECHECK_INTERVAL_MS
            if self.warning_wait_time_ms is not None:
                wait_ms = min(wait_ms, self.warning_wait_time_ms)
            if self.timeout is None:
                return wait_ms
            time_left_ms = int((self.deadline - time.monotonic()) * 1000)
            if time_left_ms <= 0:
                raise TimeoutError
            return min(wait_ms, time_left_ms)
'''
assert s.count(OLD_METHOD) == 1, "timeout_ms anchor count != 1"
s = s.replace(OLD_METHOD, NEW_METHOD, 1)

compile(s, str(p), "exec")
p.write_text(s)
print(" [OK] shm-reader-recheck applied (SHM_READER_RECHECK_INTERVAL_MS=1000)")
PYEOF
fi