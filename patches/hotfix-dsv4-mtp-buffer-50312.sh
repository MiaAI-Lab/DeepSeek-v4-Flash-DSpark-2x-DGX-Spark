#!/usr/bin/env bash
# hotfix-dsv4-mtp-buffer-50312.sh — Free the unused DSV4 MTP hidden-state PP buffer
#
# Backport of upstream vLLM PR #50312 ("Fix redundant memory allocation and copy
# for dsv4 pp buffer, 448 MiB GPU memory saved") applied to the Anemll
# dspark-vllm-gx10 0.1.1 image (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: nvidia/model.py allocates `_mtp_hidden_buffer`
# (max_num_batched_tokens x hc_mult*hidden_size bf16 = 256 MiB per rank for
# DeepSeek-V4-Flash: 8192 x 16384 x 2 B) unconditionally on the last PP rank and
# runs a `copy_` into it on EVERY target forward step. The buffer is only
# consumed by the MTP / Eagle draft path (`get_mtp_target_hidden_states`).
# This deployment runs `method=dspark`; dspark/speculator.py explicitly does NOT
# consume it ("DSpark does not use the same pre-allocated buffer that
# DeepSeek-V4's MTP uses" — it drafts from mean-pooled aux hidden states), so
# on DSpark the allocation is dead weight and the per-step copy_ is wasted GPU
# work. Guarded on `use_eagle() or uses_draft_model()` (both False for dspark),
# matching upstream #50312.
#
# CRITICAL SAFETY NET (not in upstream #50312): gpu/model_runner.py calls
# `get_mtp_target_hidden_states()` and slices the result with NO None-guard at
# two sites (`pre_hc_hidden_states[: n]`). Once the buffer can be None, both
# sites must guard or DSpark decode crashes with `TypeError: 'NoneType' object
# is not subscriptable`. This backport adds those guards (mainline v0.27.0
# ships the same unguarded code — see analysis).
#
# Usage:
#   docker cp hotfix-dsv4-mtp-buffer-50312.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-mtp-buffer-50312.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-mtp-buffer-50312.sh --before   # capture pre-restart KV budget + prompt histogram
#   ... apply + restart yourself ...
#   bash hotfix-dsv4-mtp-buffer-50312.sh --after    # capture + diff vs baseline
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-50312-kv-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-50312-kv-after.txt"

ACTION="${1:-patch}"

# VLLM_ROOT is only needed for patch / --status (which run inside the container
# via docker exec). The host-side --before/--after capture modes only need the
# docker/curl tooling, so skip the check there.
if [ "$ACTION" != "--before" ] && [ "$ACTION" != "--baseline" ] \
   && [ "$ACTION" != "--after" ] && [ "$ACTION" != "--verify" ]; then
  if [ ! -d "$VLLM_ROOT" ]; then
    echo "ERROR: vLLM not found at $VLLM_ROOT" >&2
    exit 1
  fi
fi

# ---- validation helpers (host side) -----------------------------------------
capture_kv_snapshot() {
  local tag="$1" outfile="$2"
  {
    echo "=== KV snapshot ($tag) $(date -Is) ==="
    echo "-- boot log (docker logs ${CONTAINER}) --"
    docker logs "$CONTAINER" 2>&1 \
      | grep -E "Available KV cache memory:|GPU KV cache size:|Maximum concurrency for" \
      | tail -6 || true
    echo "-- live /metrics --"
    curl -s -m 8 "$API_URL/metrics" 2>/dev/null \
      | grep -E "cache_config_info" \
      | grep -oE '(num_gpu_blocks|kv_cache_size_tokens|kv_cache_max_concurrency)="[0-9.]+"' \
      || true
    echo "-- request prompt-length histogram (_count/sum + buckets) --"
    curl -s -m 8 "$API_URL/metrics" 2>/dev/null \
      | grep -E "vllm:request_prompt_tokens_(count|sum|bucket)" \
      || true
  } > "$outfile"
  echo "[OK] $tag snapshot -> $outfile"
}

print_histogram_summary() {
  local f="$1"
  python3 - "$f" <<'PY'
import math, re, sys
f = sys.argv[1]
counts = {}
try:
    for line in open(f):
        m = re.match(r'vllm:request_prompt_tokens_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+([0-9.eE+-]+)', line)
        if m:
            counts[float(m.group(1))] = float(m.group(2))
except FileNotFoundError:
    print("  (no histogram captured)")
    sys.exit(0)
if not counts:
    print("  (no request_prompt_tokens histogram yet on this server)")
    sys.exit(0)
total = max(counts.values(), default=0.0)
prev_c = 0.0
print("  prompt-tokens per request (bucket -> requests -> %):")
for b in sorted(counts):
    n = counts[b] - prev_c
    prev_c = counts[b]
    if n > 0 or b in (100.0, 1000.0, 5000.0, 20000.0, 65536.0, 262144.0, 1048576.0):
        print(f"    <= {b:>10.0f}: {n:>10.0f}  ({100 * n / total:5.1f}%)")
# COUNTERFACTUAL since the #50004 revert (upstream vLLM #51318; this chain no
# longer applies hotfix-dsv4-adaptive-topk-50004.sh): estimates the C128A topk
# width this workload WOULD use if that patch were still applied — within each
# bucket assume prompts spread to the bucket bound, take the midpoint, then
# width = clamp(next_pow2(mid/128), 128, 8192). The live kernel runs the full
# fixed width, so treat the printed percentage as hypothetical headroom only,
# not an active saving.
active_sum = 0.0
prev_c = 0.0
prev_b = 0.0
for b in sorted(counts):
    n = counts[b] - prev_c
    if n > 0:
        mid = (prev_b + b) / 2.0
        w = max(128, 2 ** math.ceil(math.log2(max(mid / 128.0, 1))))
        w = min(w, 8192)
        active_sum += n * w
    prev_b, prev_c = b, counts[b]
if total:
    avg_w = active_sum / total
    print(f"  est. avg C128A kernel loop width: {avg_w:.0f} / 8192"
          f"  (~{100 * (1 - avg_w / 8192):.0f}% fewer metadata-kernel iterations)")
PY
}

# ---- actions ----------------------------------------------------------------
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--baseline" ]; then
  mkdir -p "$RESULTS_DIR"
  capture_kv_snapshot "BEFORE (pre-restart)" "$BASELINE_FILE"
  echo ""; echo "Baseline histogram:"; print_histogram_summary "$BASELINE_FILE"
  echo ""
  echo "Next: apply the patch, restart the server yourself, then run:"
  echo "  bash $(basename "$0") --after"
  exit 0
fi

if [ "$ACTION" = "--after" ] || [ "$ACTION" = "--verify" ]; then
  if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: no baseline at $BASELINE_FILE — run --before first" >&2
    exit 1
  fi
  mkdir -p "$RESULTS_DIR"
  capture_kv_snapshot "AFTER (post-restart)" "$AFTER_FILE"
  echo ""; echo "=== KV budget diff (BEFORE -> AFTER) ==="
  python3 - "$BASELINE_FILE" "$AFTER_FILE" <<'PY'
import sys
def grab(f, pat, end=None):
    try:
        for line in open(f):
            if pat in line:
                val = line.split(pat, 1)[1].strip()
                if end is not None:
                    val = val.split(end, 1)[0].strip()
                return val
    except FileNotFoundError:
        return None
    return None
B, A = sys.argv[1], sys.argv[2]
for label, pat, end in [
    ("Available KV cache memory", "Available KV cache memory: ", None),
    ("GPU KV cache size", "GPU KV cache size: ", None),
    ("num_gpu_blocks", 'num_gpu_blocks="', '"'),
]:
    b, a = grab(B, pat, end), grab(A, pat, end)
    arrow = "" if (b is None or a is None or b == a) else "  <-- CHANGED (expected: KV grows)"
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{arrow}")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #50312 effect: GPU KV cache size / num_gpu_blocks INCREASE"
  echo "(~256 MiB per rank ≈ +1-2k blocks / ~+37k tokens at 148k tokens/GiB)."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
m = (root / "models/deepseek_v4/nvidia/model.py").read_text()
r = (root / "v1/worker/gpu/model_runner.py").read_text()
guard = "needs_mtp_hidden_states"
cp = "if self._mtp_hidden_buffer is not None:"
rg = "if pre_hc_hidden_states is not None:"
print("model.py alloc guard  :", "APPLIED" if guard in m else "NOT APPLIED")
print("model.py copy_ guard  :", "APPLIED" if cp in m else "NOT APPLIED")
print("runner None-guard x2  :", "APPLIED" if r.count(rg) >= 2 else "NOT APPLIED")
PY
  exit 0
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 MTP hidden-state PP buffer (upstream #50312 backport) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

if [ -n "${WORKER_HOST:-}" ]; then
  echo ""
  echo "NOTE: WORKER_HOST is set — this is the HEAD node only."
  echo "Run the same script on the worker container too."
fi

python3 <<PYEOF
import os
import stat
import sys
import tempfile
from pathlib import Path

root = Path("$VLLM_ROOT")
applied = 0
skipped = 0
errors = []

# ---- whole-script transaction ------------------------------------------------
# Hunks validate against an in-memory staged view; nothing touches the tree
# until the last hunk has validated. originals/modes hold the per-run bytes and
# permission bits of every target so a failed commit rolls back byte-exactly.
originals: dict[str, bytes] = {}
modes: dict[str, int] = {}
staged: dict[str, str] = {}


def _stage(path: str) -> str:
    if path not in staged:
        p = root / path
        originals[path] = p.read_bytes()
        modes[path] = stat.S_IMODE(p.stat().st_mode)
        staged[path] = originals[path].decode()
    return staged[path]


def patch(path: str, old: str, new: str, label: str, expect: int = 1) -> None:
    global applied, skipped
    p = root / path
    if not p.exists():
        errors.append(f"File not found: {path}")
        return
    text = _stage(path)
    n_old = text.count(old)
    n_new = text.count(new)
    if n_new == expect and n_old == new.count(old) * expect:
        print(f"  [skip] {label} (already applied)")
        skipped += 1
        return
    if n_old != expect or n_new != 0:
        errors.append(
            f"[ERR] ambiguous state for {label} in {path}: "
            f"old x{n_old} (expect {expect}), new x{n_new}"
        )
        return
    staged[path] = text.replace(old, new)
    print(f"  [stage] {label} (prepared {n_old})")
    applied += 1


# ---- model.py: only allocate the MTP buffer when a speculator consumes it ----
patch(
    "models/deepseek_v4/nvidia/model.py",
    """        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None""",
    """        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None""",
    "model.py: conditional _mtp_hidden_buffer allocation (dspark -> None)",
)

# ---- model.py: skip the per-step copy_ when the buffer does not exist -------
patch(
    "models/deepseek_v4/nvidia/model.py",
    """        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))""",
    """        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))""",
    "model.py: skip copy_ when buffer is None",
)

# ---- model_runner.py: None-guard BOTH draft-feeding sites (crash safety) ----
# The exact same 3-line block appears at model_runner.py:599 and :1463.
patch(
    "v1/worker/gpu/model_runner.py",
    """            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]""",
    """            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                if pre_hc_hidden_states is not None:
                    spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]""",
    "model_runner.py: None-guard at both get_mtp_target_hidden_states sites",
    expect=2,
)


print(f"\nStaged: {applied}, Skipped: {skipped}, Errors: {len(errors)}")
for e in errors:
    print(f"  {e}", file=sys.stderr)

if errors:
    print("\nWARNING: Some patches could not be applied. Nothing was written.")
    sys.exit(1)


# ---- atomic commit -----------------------------------------------------------
# Every hunk validated against the staged view. Publish changed targets in
# deterministic order via same-directory temp files + os.replace (mode kept,
# file fsynced, directory fsynced best-effort), then re-read to verify the
# written bytes. A target is tracked as published the moment its rename lands,
# so any later failure — including interrupts and verification faults — rolls
# back every published target from the per-run originals in reverse order
# through the same safe writer and re-verifies byte identity; a failed
# rollback is loud and exits nonzero.

def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _safe_write(dest: Path, data: bytes, mode: int, done: list[str] | None = None, rel: str | None = None) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix="." + dest.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    if done is not None:
        done.append(rel)
    _fsync_dir(dest.parent)


def _rollback(done: list[str], why: str) -> None:
    print(f"\nERROR: {why}", file=sys.stderr)
    print(f"Restoring {len(done)} published file(s) in reverse order...", file=sys.stderr)
    for rel in reversed(done):
        try:
            _safe_write(root / rel, originals[rel], modes[rel])
            if (root / rel).read_bytes() != originals[rel]:
                raise RuntimeError("restored bytes differ from the originals")
        except Exception as exc:
            print(f"FATAL: rollback of {rel} failed: {exc}", file=sys.stderr)
            print("FATAL: targets above may not match their pre-run state; inspect manually.", file=sys.stderr)
            sys.exit(2)
        print(f"  [restored] {rel}", file=sys.stderr)


pending = [rel for rel in sorted(staged) if staged[rel].encode() != originals[rel]]
if pending:
    done: list[str] = []

    def _track_published(rel: str) -> None:
        # A failure may race the rename (e.g. an interrupt delivered after
        # the syscall completed): if the destination already holds the staged
        # bytes, the target is published and must join the rollback set.
        try:
            published = (root / rel).read_bytes() == staged[rel].encode()
        except OSError:
            published = True
        if published and rel not in done:
            done.append(rel)

    try:
        for rel in pending:
            try:
                _safe_write(root / rel, staged[rel].encode(), modes[rel], done, rel)
            except Exception as exc:
                _track_published(rel)
                raise RuntimeError(f"commit failed for {rel}: {exc}") from exc
            except BaseException as exc:
                _track_published(rel)
                raise

        try:
            bad = [
                rel
                for rel in done
                if (root / rel).read_bytes() != staged[rel].encode()
            ]
        except Exception as exc:
            raise RuntimeError(f"post-commit verification failed: {exc}") from exc
        if bad:
            raise RuntimeError(
                "post-commit verification failed for " + ", ".join(bad)
            )
    except Exception as exc:
        _rollback(done, str(exc))
        sys.exit(1)
    except BaseException as exc:
        _rollback(done, f"transaction interrupted: {exc}")
        raise

    print(f"\nCommitted: {applied} hunk(s) across {len(done)} file(s).")

if applied == 0 and skipped > 0:
    print("Patch already applied. No changes needed.")
elif applied > 0:
    print("\nHotfix applied. Stop + start the vLLM process (or the container) to take effect.")
    print("Validate with: bash <this-script> --before  (pre-restart; run BEFORE restarting)")
    print("             bash <this-script> --after   (post-restart; compare KV budget)")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
