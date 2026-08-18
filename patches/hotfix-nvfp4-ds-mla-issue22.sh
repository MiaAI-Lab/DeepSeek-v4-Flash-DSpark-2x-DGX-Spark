#!/usr/bin/env bash
# hotfix-nvfp4-ds-mla-issue22.sh — Fix nvfp4_ds_mla long-context decode regression
#
# ACTUAL ROOT CAUSE: flashmla_sparse.py dispatches nvfp4_ds_mla to the slow
# _forward_bf16_kv path instead of the fast _forward_fp8_kv path.  The584-byte
# KV layout is identical for both dtypes; only the kernel dispatch differs.
#
# Usage:
#   docker exec <container> bash /path/to/hotfix-nvfp4-ds-mla-issue22.sh
#   # Then restart the vLLM process inside the container.
#
#   docker exec <container> bash /path/to/hotfix-nvfp4-ds-mla-issue22.sh --status
#   # Report patch state only; makes no changes.
#
# Safe to re-run (idempotent — skips already-applied patches).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
ACTION="${1:-}"

if [ ! -d "$VLLM_ROOT" ]; then
  echo "ERROR: vLLM not found at $VLLM_ROOT" >&2
  exit 1
fi

# Report patch state without touching the tree, in the same
# "<label> : APPLIED|NOT APPLIED" form the other hotfixes here use, so tooling
# can compare status text across ranks.
status() {
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / "v1" / "attention" / "backends" / "mla" / "flashmla_sparse.py"
text = target.read_text() if target.exists() else ""


def chk(label, cond):
    print(f"{label:44} :", "APPLIED" if cond else "NOT APPLIED")


chk("nvfp4_ds_mla routed to fast FP8 path",
    'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' in text)
PY
  exit 0
}

if [ "$ACTION" = "--status" ]; then
  status
fi

echo "=== Hotfix: nvfp4_ds_mla long-context decode (Issue #22) ==="
echo "vLLM root: $VLLM_ROOT"

python3 <<PYEOF
import sys
from pathlib import Path

root = Path("$VLLM_ROOT")
applied = 0
skipped = 0
errors = []

def patch(path: str, old: str, new: str, label: str) -> None:
    global applied, skipped
    p = root / path
    if not p.exists():
        errors.append(f"File not found: {path}")
        return
    text = p.read_text()
    if new in text:
        print(f"  [skip] {label} (already applied)")
        skipped += 1
        return
    if old not in text:
        errors.append(f"[ERR] Anchor not found for {label} in {path}")
        return
    p.write_text(text.replace(old, new, 1))
    print(f"  [OK]   {label}")
    applied += 1


# ============================================================
# THE FIX: Route nvfp4_ds_mla to the fast FP8 kernel path
# ============================================================
# The584-byte KV layout is identical for fp8_ds_mla and nvfp4_ds_mla on DSV4.
# The only difference is the kernel dispatch.  nvfp4_ds_mla was incorrectly
# routed to the slow _forward_bf16_kv path.
patch(
    "v1/attention/backends/mla/flashmla_sparse.py",
    '        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"',
    '        use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")',
    "Route nvfp4_ds_mla to fast FP8 kernel path",
)


# ============================================================
# Summary
# ============================================================
print(f"\nApplied: {applied}, Skipped: {skipped}, Errors: {len(errors)}")
for e in errors:
    print(f"  {e}", file=sys.stderr)

if errors:
    print("\nWARNING: Some patches could not be applied.")
    sys.exit(1)

if applied == 0 and skipped > 0:
    print("Patch already applied. No changes needed.")
elif applied > 0:
    print("\nHotfix applied successfully. Restart the vLLM process to take effect.")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
