#!/usr/bin/env bash
# CPU-only recipe/patch gates. Same script as .github/workflows/validate.yml.
# Does NOT run the live 2× Spark serve, decode bench, or tool-eval-bench.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0

ok() { printf '  ok  %s\n' "$*"; }
bad() { printf '  FAIL %s\n' "$*" >&2; fail=1; }

echo "== shell syntax =="
for f in \
  start-deepseek-v4-flash-dspark.sh \
  stop-deepseek-v4-flash-dspark.sh \
  validate-dspark-config.sh \
  prepare-dspark-model-cache.sh \
  smoke-deepseek-v4-flash-dspark.sh \
  scripts/ci-validate.sh \
  scripts/verify-overlay-sources.sh \
  patches/*.sh
do
  [ -e "$f" ] || continue
  bash -n "$f" || bad "bash -n $f"
  ok "bash -n $f"
done

echo "== python compile (patches + unit scripts) =="
mapfile -t py_files < <(find patches -name '*.py' -not -path '*/__pycache__/*' | sort)
py_files+=(
  scripts/test-issue26-swa-min-v2.py
  scripts/test-issue27-adaptive-prefill.py
  scripts/test-encoding-dsv4-issue21.py
  scripts/test-suppress-stops-in-reasoning.py
  scripts/verify-dsv4-027-equality-gate.py
)
python3 -m py_compile "${py_files[@]}"
ok "py_compile ${#py_files[@]} files"

echo "== unit tests (no GPU) =="
python3 scripts/test-issue26-swa-min-v2.py -q
ok "test-issue26-swa-min-v2"
python3 scripts/test-issue27-adaptive-prefill.py -q
ok "test-issue27-adaptive-prefill"
python3 scripts/test-encoding-dsv4-issue21.py -q
ok "test-encoding-dsv4-issue21"
python3 scripts/test-suppress-stops-in-reasoning.py -q
ok "test-suppress-stops-in-reasoning"
python3 scripts/verify-dsv4-027-equality-gate.py
ok "verify-dsv4-027-equality-gate"
bash scripts/verify-overlay-sources.sh
ok "verify-overlay-sources"

echo "== recipe guards (do not re-ship known regressions) =="

# #31/#34 thinking-budget path must stay gone (decode tok/s cliff).
if compgen -G 'patches/hotfix-dsv4-issue31*' > /dev/null; then
  bad "thinking-budget patch file present under patches/"
else
  ok "no patches/hotfix-dsv4-issue31*"
fi

launch_files=(
  docker-compose.dspark.yml
  start-deepseek-v4-flash-dspark.sh
  stop-deepseek-v4-flash-dspark.sh
  .env.dspark.example
)
if grep -nE 'hotfix-dsv4-issue31|ThinkingBudgetState|thinking_budget\.py|DEFAULT_THINKING_TOKEN_BUDGET|DEFAULT_MAX_TOKENS=131072' \
  "${launch_files[@]}" >/tmp/ci-budget-hits.txt 2>/dev/null; then
  bad "thinking-budget still wired into launch/example:"
  cat /tmp/ci-budget-hits.txt >&2 || true
else
  ok "launch path does not apply thinking-budget"
fi

# #26 v1 continue must not be the applied patch (warm-prefix garble / tool names).
i26=patches/hotfix-dsv4-issue26-hybrid-swa-min.py
if [ ! -f "$i26" ]; then
  bad "missing $i26"
else
  if ! grep -q 'issue26-hotfix-v2' "$i26"; then
    bad "$i26 is not marked v2"
  else
    ok "$i26 is v2"
  fi
  if grep -q 'SWA groups must not shrink the hybrid common hit' "$i26" \
    && grep -q 'if isinstance(spec, SlidingWindowSpec):' "$i26"; then
    # v1 text may exist as V1_INJECT for revert tests — applied block must not be v1.
    if grep -A8 'V2_BLOCK' "$i26" | grep -q 'if isinstance(spec, SlidingWindowSpec):'; then
      bad "$i26 V2_BLOCK still has SlidingWindowSpec continue"
    else
      ok "$i26 keeps v1 only as revert source, not V2_BLOCK"
    fi
  fi
fi

# Compose must still apply #26 v2 + #27 and keep restart policy.
if grep -q 'hotfix-dsv4-issue26-hybrid-swa-min.py' docker-compose.dspark.yml \
  && grep -q 'hotfix-dsv4-issue27-partial-prefill-concurrency.py' docker-compose.dspark.yml; then
  ok "compose mounts #26 + #27"
else
  bad "compose missing #26 or #27 mount"
fi
if grep -q 'python3 /opt/hotfix-dsv4-issue26-hybrid-swa-min.py' docker-compose.dspark.yml \
  && grep -q 'python3 /opt/hotfix-dsv4-issue27-partial-prefill-concurrency.py' docker-compose.dspark.yml; then
  ok "compose entrypoint applies #26 + #27"
else
  bad "compose entrypoint does not apply #26 + #27"
fi
if grep -q 'hotfix-dsv4-suppress-stops-in-reasoning.py' docker-compose.dspark.yml; then
  ok "compose applies suppress-stops-in-reasoning"
else
  bad "compose missing suppress-stops-in-reasoning"
fi
if grep -q 'restart: ${DSPARK_RESTART_POLICY:-unless-stopped}' docker-compose.dspark.yml; then
  ok "compose restart unless-stopped"
else
  bad "compose missing restart: unless-stopped"
fi

# Mounted hotfix files must exist.
for p in \
  patches/hotfix-encoding-dsv4-issue21.py \
  patches/hotfix-dsv4-issue26-hybrid-swa-min.py \
  patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py \
  patches/hotfix-nvfp4-ds-mla-issue22.sh \
  patches/hotfix-dsv4-suppress-stops-in-reasoning.py
do
  if [ -f "$p" ]; then
    ok "present $p"
  else
    bad "missing required $p"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "CI validate FAILED" >&2
  exit 1
fi
echo "CI validate passed (CPU recipe gates only)."
