#!/usr/bin/env bash
# hotfix-vllm-redact-api-key-log.sh — keep configured API keys out of the
# vLLM startup/residual Docker log.
#
# ROOT CAUSE (upstream vLLM, still unfixed as of the pinned image
# 0.25.2.dev0+g752a3a504.d20260714): `vllm/entrypoints/serve/utils/api_utils.py`
# has two sibling helpers over `get_non_default_args()`:
#
#   jsonify_non_default_args(args, *, exclude=None)   <-- has redaction
#   log_non_default_args(args)                        <-- does NOT
#
# `log_non_default_args` hands the whole resolved dict to
# `logger.info("non-default args: %s", ...)`. `--api-key` is a dataclass
# `list[str]` field, so every configured key lands in that dict and is printed
# verbatim at server startup:
#
#   [api_utils.py] non-default args: {..., 'api_key': ['sk-...', ...]}
#
# The `exclude` set on the jsonify path is exactly the mechanism needed here;
# it is simply never wired into the logging path. This patch redacts in
# `log_non_default_args` itself, which covers BOTH call sites (the api_server
# and llm entrypoints) with one edit.
#
# The redaction PRESERVES THE COUNT (`<redacted:N value(s)>`). The count is the
# documented way to confirm that a single `--api-key` flag carried N keys
# (repeating the flag overwrites instead of appends), so the diagnostic keeps
# working without printing the secrets.
#
# SCOPE — this closes the *log* channel only. Keys also remain readable in the
# host process cmdline (`/proc/<pid>/cmdline`) because `--api-key` is argv;
# that is argv/env exposure by design (documented in .env.dspark.example) and
# no Python patch can fix it.
#
# Usage (same shape as the sibling hotfixes):
#   bash hotfix-vllm-redact-api-key-log.sh          # apply (inside container)
#   bash hotfix-vllm-redact-api-key-log.sh --status # report (inside container)
#   bash hotfix-vllm-redact-api-key-log.sh --before | --after  (host-side; doc only)
#
# --status exits nonzero unless every applied-state check passes. Keyed starts
# require this patch and verify it fail-closed, with no DSPARK_SKIP_HOTFIX
# bypass. Re-running skips only a fully applied hunk; partial states fail.
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
import typing
from pathlib import Path

root = Path(sys.argv[1])
api_utils = root / "entrypoints" / "serve" / "utils" / "api_utils.py"
text = api_utils.read_text()
failed = []

def chk(label, cond):
    print(f"{label:44} :", "APPLIED" if cond else "NOT APPLIED")
    if not cond:
        failed.append(label)

chk("redact set defined", "_DSPARK_REDACT_LOG_ARGS" in text)
chk("log path calls redactor",
    "_dspark_redact_non_default(get_non_default_args(args))" in text)
# The pre-patch body passed the raw dict straight to the logger. Assert that
# exact two-line sequence is gone rather than just that a logger call exists
# (the patched version still logs -- it logs the redacted dict).
chk("raw dict no longer logged", """    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)""" not in text)

# Behavioural checks: exec the injected helper standalone and prove it actually
# redacts. String matching alone would report APPLIED for a patch that parsed
# but did the wrong thing -- and a hotfix whose --status lies is worse than no
# hotfix.
ns = {"Any": typing.Any}
start = text.find("_DSPARK_REDACT_LOG_ARGS")
end = text.find("def log_non_default_args")
redacts = no_mutate = keeps_others = False
if start != -1 and end > start:
    try:
        exec(compile(text[start:end], "<hotfix-status>", "exec"), ns)
        redact = ns["_dspark_redact_non_default"]
        out = redact({"api_key": ["sk-probe-a", "sk-probe-b"], "port": 8888})
        rendered = repr(out)
        redacts = "sk-probe-a" not in rendered and "2 value(s)" in rendered
        keeps_others = out.get("port") == 8888
        # The fresh dict's values are the live parsed args, so the redactor
        # must rebind the entry, never mutate the list in place.
        live = ["sk-probe-a", "sk-probe-b"]
        redact({"api_key": live})
        no_mutate = live == ["sk-probe-a", "sk-probe-b"]
    except Exception:
        pass
chk("redaction verified behaviourally", redacts)
chk("non-secret args untouched", keeps_others)
chk("live args list not mutated", no_mutate)
raise SystemExit(1 if failed else 0)
PY
  exit 0
}

if [ "$ACTION" = "--status" ]; then
  status
fi

# host-side before/after are documentation-only: this is a logging-redaction fix
# with no KV/throughput effect. Same interface as the siblings.
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--after" ]; then
  if [ "$ACTION" = "--before" ]; then
    echo "No KV or perf change expected: pure log-redaction fix."
    echo "Verify with: docker logs <container> | grep 'non-default args'"
  else
    echo "No KV or perf change expected. Confirm the log line shows"
    echo "'api_key': ['<redacted:N value(s)>'] and no sk- prefix anywhere."
  fi
  exit 0
fi

echo "=== Hotfix: redact api_key in vLLM non-default-args log (no upstream fix) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

python3 <<PYEOF
from pathlib import Path

root = Path("$VLLM_ROOT")
applied = 0
skipped = 0
errors = []
RAW_ANCHOR = '''    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)'''

def patch(path: str, old: str, new: str, label: str, expect: int = 1) -> None:
    global applied, skipped
    p = root / path
    if not p.exists():
        errors.append(f"File not found: {path}")
        return
    text = p.read_text()
    if new in text:
        if RAW_ANCHOR in text:
            errors.append("partial patch: replacement present but raw logger still defined")
            return
        print(f"  [skip] {label} (already applied)")
        skipped += 1
        return
    n = text.count(old)
    if n == 0 or (expect and n != expect):
        errors.append(f"[ERR] anchor x{n} (expect {expect}) for {label} in {path}")
        return
    text = text.replace(old, new)
    p.write_text(text)
    print(f"  [OK]   {label} (replaced {n})")
    applied += 1


# ---- api_utils.py: redact secret-bearing args before logging ---------------
# get_non_default_args() builds a FRESH dict per call, so rebinding an entry
# here cannot affect the real parsed args -- and we must never mutate the
# list in place for the same reason.
patch(
    "entrypoints/serve/utils/api_utils.py",
    '''def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)''',
    '''# Args whose values are secrets and must never reach the log. The sibling
# jsonify_non_default_args() already takes an \`exclude\` set for this; the
# logging path had no equivalent, so the keys were printed verbatim.
_DSPARK_REDACT_LOG_ARGS = frozenset({"api_key"})


def _dspark_redact_non_default(non_default_args: dict[str, Any]) -> dict[str, Any]:
    """Replace secret values with a count-preserving placeholder.

    The count is kept because it is the documented way to verify that one
    \`--api-key\` flag carried N keys rather than being overwritten down to one.
    """
    for key in _DSPARK_REDACT_LOG_ARGS:
        value = non_default_args.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            # Rebind, never mutate in place -- the list is the live args value.
            non_default_args[key] = [f"<redacted:{len(value)} value(s)>"]
        else:
            non_default_args[key] = "<redacted>"
    return non_default_args


def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = _dspark_redact_non_default(get_non_default_args(args))
    logger.info("non-default args: %s", non_default_args)''',
    "api_utils.py: redact api_key in non-default-args log",
)

print()
if errors:
    for e in errors:
        print(e)
    print(f"FAILED: applied={applied} skipped={skipped} errors={len(errors)}")
    raise SystemExit(1)
print(f"OK: applied={applied} skipped={skipped}")
PYEOF

echo
echo "Done. Verify after restart:"
echo "  docker logs <container> 2>&1 | grep 'non-default args'"
echo "  -> expect \"'api_key': ['<redacted:N value(s)>']\", no sk- prefix."
