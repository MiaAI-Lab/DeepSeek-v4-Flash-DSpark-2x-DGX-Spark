#!/usr/bin/env bash
# hotfix-dsv4-grammar-advance.sh — Advance grammar across the reasoning boundary
# (Stage-A #44993 backport). Fixes structured-output corruption (think #24:
# `response_format json_schema` + thinking enabled → duplicated prefix).
#
# ROOT CAUSE (this fork only): `should_advance` returns False for JSON/regex/
# choice when reasoning ends mid-step, so the scheduler's
# `trim_reasoning_for_advance()` + `accept_tokens()` never run on the boundary
# step. With speculative decoding, the draft produced right after the
# reasoning_end marker never enters the grammar FSM → the model re-emits the
# opening token next step → `{"city":"Paris{"city":"Paris"}`.
#
# Upstream #44993 (merged) fixes it: should_advance takes the step's
# `new_token_ids` as the exact delta window (placeholder count is wrong under
# async+spec when drafts are rejected), records `reasoning_end_token_index` for
# ALL constraint types, and the scheduler trims + advances with the verified
# post-marker suffix.
#
# Usage (like the other hotfixes):
#   docker cp hotfix-dsv4-grammar-advance.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-grammar-advance.sh
#   bash hotfix-dsv4-grammar-advance.sh --status   (inside container)
#   bash hotfix-dsv4-grammar-advance.sh --before    (host-side; doc only)
#   bash hotfix-dsv4-grammar-advance.sh --after     (host-side; doc only)
#
# Idempotent — re-running skips already-applied hunks.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
ACTION="${1:-}"

if [ ! -d "$VLLM_ROOT" ]; then
  echo "ERROR: vLLM not found at $VLLM_ROOT (run inside the container)" >&2
  exit 1
fi

status() {
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
so = root / "v1" / "structured_output" / "__init__.py"
sch = root / "v1" / "core" / "sched" / "scheduler.py"
t_so = so.read_text()
t_sch = sch.read_text()

def chk(label, cond):
    print(f"{label:44} :", "APPLIED" if cond else "NOT APPLIED")

chk("should_advance new_token_ids param", "new_token_ids: list[int] | None = None" in t_so)
chk("delta window uses new_token_ids", "delta_ids: Iterable[int] = new_token_ids" in t_so)
chk("boundary records end_index (all types)", "end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)" in t_so)
chk("old STRUCTURAL_TAG-only gate removed",
    '== StructuredOutputOptions.STRUCTURAL_TAG' not in t_so and
    "self.vllm_config.speculative_config is not None" not in t_so)
chk("scheduler passes new_token_ids", "should_advance(\n                request, new_token_ids=new_token_ids\n            )" in t_sch)
PY
  exit 0
}

if [ "$ACTION" = "--status" ]; then
  status
fi

# host-side before/after are documentation-only for this hotfix (no KV change
# expected: pure correctness fix). Provide the same interface as siblings.
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--after" ]; then
  if [ "$ACTION" = "--before" ]; then
    echo "No KV change expected: #44993 backport is a correctness fix (grammar"
    echo "advance across the reasoning boundary). Validate via issue #24 repro."
  else
    echo "No KV change expected. Run the #24 repro instead of --after."
  fi
  exit 0
fi

echo "=== Hotfix: DSV4 grammar advance across reasoning boundary (upstream #44993 backport) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

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


# ---- structured_output/__init__.py: should_advance signature ----------------
patch(
    "v1/structured_output/__init__.py",
    """    def should_advance(self, request: "Request") -> bool:""",
    """    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:""",
    "__init__.py: should_advance new_token_ids param (upstream #44993)",
)

# ---- structured_output/__init__.py: delta-window + boundary record ----------
patch(
    "v1/structured_output/__init__.py",
    """        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                # The scheduler will advance the grammar with this step's
                # tokens right away, but the step still contains reasoning
                # content up to and including the end marker. Record where
                # it ends so trim_reasoning_for_advance() can drop it.
                structured_req.reasoning_end_token_index = (
                    self._find_reasoning_end_index(reasoner, all_token_ids, start)
                )
                return True

        return False""",
    """        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids: Iterable[int] = new_token_ids
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)
        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True

            # Record the boundary so the scheduler can exclude reasoning tokens.
            end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)

            structured_req.reasoning_end_token_index = end_index
            return True

        return False""",
    "__init__.py: delta window + boundary record, all constraint types (upstream #44993)",
)

# ---- scheduler.py: pass the step tokens as the exact delta window ------------
patch(
    "v1/core/sched/scheduler.py",
    """            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar""",
    """            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar""",
    "scheduler.py: pass new_token_ids into should_advance (upstream #44993)",
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
    print("\nHotfix applied. Restart the vLLM process (or container) to take effect.")
    print("Validate: issue #24 repro (json_schema + thinking) x5 must be clean JSON.")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
