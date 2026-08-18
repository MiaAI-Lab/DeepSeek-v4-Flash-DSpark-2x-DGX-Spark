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
  scripts/test-issue31-thinking-budget-gpu.py
  scripts/test-issue55-tool-truncation.py
  scripts/test-responses-api-live.py
  scripts/test-encoding-dsv4-issue21.py
  scripts/test-suppress-stops-in-reasoning.py
  scripts/test-assistant-final-continuation.py
  scripts/test-ruler-lite-pad.py
  scripts/ruler-lite.py
  scripts/verify-dsv4-027-equality-gate.py
)
python3 -m py_compile "${py_files[@]}"
ok "py_compile ${#py_files[@]} files"

echo "== unit tests (no GPU) =="
python3 scripts/test-issue26-swa-min-v2.py -q
ok "test-issue26-swa-min-v2"
python3 scripts/test-issue31-thinking-budget-gpu.py -q
ok "test-issue31-thinking-budget-gpu"
python3 scripts/test-issue55-tool-truncation.py -q
ok "test-issue55-tool-truncation"
python3 scripts/test-responses-api-live.py -q
ok "test-responses-api-live"
python3 scripts/test-encoding-dsv4-issue21.py -q
ok "test-encoding-dsv4-issue21"
python3 scripts/test-suppress-stops-in-reasoning.py -q
ok "test-suppress-stops-in-reasoning"
python3 scripts/test-assistant-final-continuation.py -q
ok "test-assistant-final-continuation"
python3 scripts/test-ruler-lite-pad.py -q
ok "test-ruler-lite-pad"
python3 scripts/verify-dsv4-027-equality-gate.py
ok "verify-dsv4-027-equality-gate"
bash scripts/verify-overlay-sources.sh
ok "verify-overlay-sources"

echo "== recipe guards (do not re-ship known regressions) =="

# The withdrawn #31/#34 CPU-scanning path must stay gone (decode tok/s cliff).
old_i31=patches/hotfix-dsv4-issue31-v2-thinking-budget.py
gpu_i31=patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py
if [ -e "$old_i31" ]; then
  bad "withdrawn CPU-scanning thinking-budget patch returned: $old_i31"
else
  ok "withdrawn CPU-scanning thinking-budget patch stays absent"
fi
if grep -nE '\.cpu\(|\.tolist\(|\.detach\(|all_token_ids|DEFAULT_THINKING_TOKEN_BUDGET' \
  "$gpu_i31" >/tmp/ci-budget-hotpath-hits.txt 2>/dev/null; then
  bad "GPU thinking-budget patch contains a forbidden decode-path scan/sync:"
  cat /tmp/ci-budget-hotpath-hits.txt >&2 || true
else
  ok "GPU thinking-budget hot path has no CPU sync or token-buffer scan"
fi

launch_files=(
  docker-compose.dspark.yml
  start-deepseek-v4-flash-dspark.sh
  stop-deepseek-v4-flash-dspark.sh
  .env.dspark.example
)
if grep -nE 'hotfix-dsv4-issue31-v2-thinking-budget\.py|thinking_budget\.py|DEFAULT_THINKING_TOKEN_BUDGET|DEFAULT_MAX_TOKENS=131072' \
  "${launch_files[@]}" >/tmp/ci-budget-hits.txt 2>/dev/null; then
  bad "withdrawn thinking-budget implementation still wired into launch/example:"
  cat /tmp/ci-budget-hits.txt >&2 || true
else
  ok "launch path does not apply withdrawn thinking-budget implementation"
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
if grep -q 'hotfix-dsv4-issue31-v2-thinking-budget-gpu.py' docker-compose.dspark.yml \
  && grep -q 'python3 /opt/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py' docker-compose.dspark.yml; then
  ok "compose applies GPU-resident V2 thinking budget"
else
  bad "compose missing GPU-resident V2 thinking budget"
fi
if grep -q 'hotfix-dsv4-issue55-tool-truncation.py' docker-compose.dspark.yml \
  && grep -q 'python3 /opt/hotfix-dsv4-issue55-tool-truncation.py' docker-compose.dspark.yml; then
  ok "compose applies issue #55 tool-call truncation safety"
else
  bad "compose missing issue #55 tool-call truncation safety"
fi
# Assistant-final continuation (#52/PR53): default OFF (stock renderer);
# ON must be an exactly-1 gate with a fail-closed invocation.
if grep -Fq 'DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX: "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-assistant-final-continuation.py || exit 1; fi;' docker-compose.dspark.yml; then
  ok "compose gates assistant-final hotfix behind =1, fail-closed"
else
  bad "compose must invoke assistant-final hotfix only when DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX=1, with || exit 1"
fi
if grep -q 'restart: ${DSPARK_RESTART_POLICY:-unless-stopped}' docker-compose.dspark.yml; then
  ok "compose restart unless-stopped"
else
  bad "compose missing restart: unless-stopped"
fi
if grep -q 'exit 3' start-deepseek-v4-flash-dspark.sh \
  && grep -q 'SuccessExitStatus=3' start-deepseek-v4-flash-dspark.sh \
  && grep -q 'SuccessExitStatus=3' docs/ENVS.md; then
  ok "start already-running is exit 3 (#72)"
else
  bad "start missing already-running exit 3 (#72)"
fi

# Mounted hotfix files must exist.
for p in \
  patches/hotfix-encoding-dsv4-issue21.py \
  patches/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py \
  patches/hotfix-dsv4-issue55-tool-truncation.py \
  patches/hotfix-dsv4-issue26-hybrid-swa-min.py \
  patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py \
  patches/hotfix-nvfp4-ds-mla-issue22.sh \
  patches/hotfix-gb10-spin-wait.sh \
  patches/hotfix-dsv4-suppress-stops-in-reasoning.py \
  patches/hotfix-dsv4-assistant-final-continuation.py
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
