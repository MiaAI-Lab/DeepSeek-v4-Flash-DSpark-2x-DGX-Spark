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
  start-tp3.sh \
  stop-deepseek-v4-flash-dspark.sh \
  validate-dspark-config.sh \
  prepare-dspark-model-cache.sh \
  build-dspark-vllm-runtime.sh \
  files/nfs-share.sh \
  files/nfs-server/entrypoint.sh \
  smoke-deepseek-v4-flash-dspark.sh \
  status-deepseek-v4-flash-dspark.sh \
  logs-deepseek-v4-flash-dspark.sh \
  scripts/ci-validate.sh \
  scripts/verify-overlay-sources.sh \
  scripts/test-draft-sample-method-gate.sh \
  scripts/test-nccl-fabric-passthrough.sh \
  scripts/test-nccl-ib-hca-gid-resolve.sh \
  scripts/boot-shape-warmup.sh \
  scripts/test-boot-shape-warmup.sh \
  scripts/validate_tp3.sh \
  lmcache/run-lmcache-server.sh \
  scripts/test-lmcache-compose-gate.sh \
  patches/*.sh
do
  [ -e "$f" ] || continue
  bash -n "$f" || bad "bash -n $f"
  ok "bash -n $f"
done

if "$ROOT/build-dspark-vllm-runtime.sh" --tag-selftest; then
  ok "build-dspark-vllm-runtime digest is not a docker -t (issue #173)"
else
  bad "build-dspark-vllm-runtime --tag-selftest"
fi

echo "== python compile (patches + unit scripts) =="
mapfile -t py_files < <(find patches -name '*.py' -not -path '*/__pycache__/*' | sort)
py_files+=(
  scripts/test-issue26-swa-min-v2.py
  scripts/test-issue31-thinking-budget-gpu.py
  scripts/test-issue55-tool-truncation.py
  scripts/test-responses-store-hotfix.py
  scripts/test-responses-api-live.py
  scripts/verify-issue138-responses-history-live.py
  scripts/test-issue138-responses-history-hotfix.py
  scripts/test-issue138-responses-history-live.py
  scripts/test-codex-agent-message-compat.py
  scripts/test-encoding-dsv4-issue21.py
  scripts/test-suppress-stops-in-reasoning.py
  scripts/test-assistant-final-continuation.py
  scripts/spec-acceptance.py
  scripts/test-spec-acceptance.py
  scripts/test-ruler-lite-pad.py
  scripts/test-env-normalisation.py
  scripts/test-served-model-alias.py
  scripts/test-dspark-api-keys.py
  scripts/test-redact-api-key-log.py
  scripts/test-hotfix-atomic-transaction.py
  scripts/test-python-hotfix-failclosed.py
  scripts/test-dsv4-vision-exp-hotfix.py
  scripts/test-issue141-sparse-mla-decode-chunk.py
  scripts/test-issue136-xgrammar-termination.py
  scripts/test-issue117-shm-ring-buffer.py
  scripts/verify-issue136-xgrammar-live.py
  scripts/test-empty-encoder-output-hotfix.py
  scripts/ruler-lite.py
  scripts/verify-dsv4-027-equality-gate.py
  scripts/ab-issue133-triton-specialization.py
  tests/test_dspark_stacked_mapping.py
  tests/test_issue133_triton_specialization.py
  lmcache/patch-compose-lmcache.py
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
python3 scripts/test-responses-store-hotfix.py -q
ok "test-responses-store-hotfix"
python3 scripts/test-responses-api-live.py -q
ok "test-responses-api-live"
python3 scripts/test-issue138-responses-history-hotfix.py -q
ok "test-issue138-responses-history-hotfix"
python3 scripts/test-issue138-responses-history-live.py -q
ok "test-issue138-responses-history-live"
python3 scripts/test-codex-agent-message-compat.py -q
ok "test-codex-agent-message-compat"
python3 scripts/test-encoding-dsv4-issue21.py -q
ok "test-encoding-dsv4-issue21"
python3 scripts/test-suppress-stops-in-reasoning.py -q
ok "test-suppress-stops-in-reasoning"
python3 scripts/test-assistant-final-continuation.py -q
ok "test-assistant-final-continuation"
python3 scripts/test-spec-acceptance.py -q
ok "test-spec-acceptance"
python3 scripts/test-ruler-lite-pad.py -q
ok "test-ruler-lite-pad"
python3 scripts/test-numeric-knob-validation.py -q
ok "test-numeric-knob-validation"
python3 scripts/test-env-normalisation.py -q
ok "test-env-normalisation"
python3 scripts/test-served-model-alias.py -q
ok "test-served-model-alias"
python3 scripts/test-dspark-api-keys.py -q
ok "test-dspark-api-keys"
python3 scripts/test-redact-api-key-log.py -q
ok "test-redact-api-key-log"
python3 scripts/test-hotfix-atomic-transaction.py -q
ok "test-hotfix-atomic-transaction"
python3 scripts/test-python-hotfix-failclosed.py -q
ok "test-python-hotfix-failclosed"
python3 scripts/test-dsv4-vision-exp-hotfix.py -q
ok "test-dsv4-vision-exp-hotfix"
python3 scripts/test-issue141-sparse-mla-decode-chunk.py -q
ok "test-issue141-sparse-mla-decode-chunk"
python3 scripts/test-issue136-xgrammar-termination.py -q
ok "test-issue136-xgrammar-termination"
python3 scripts/test-issue117-shm-ring-buffer.py -q
ok "test-issue117-shm-ring-buffer"
python3 scripts/test-empty-encoder-output-hotfix.py -q
ok "test-empty-encoder-output-hotfix"
python3 tests/test_issue27_inflight_cap.py -q
ok "test_issue27_inflight_cap"
python3 tests/test_adaptive_prefill_chunk.py -q
ok "test_adaptive_prefill_chunk"
python3 tests/test_replicate_markov_head.py
ok "test_replicate_markov_head"
python3 tests/test_sp_indexer_prefill.py
ok "test_sp_indexer_prefill"
python3 tests/test_dspark_stacked_mapping.py -q
ok "test_dspark_stacked_mapping"
python3 tests/test_issue133_triton_specialization.py -q
ok "test_issue133_triton_specialization"
python3 scripts/verify-dsv4-027-equality-gate.py
ok "verify-dsv4-027-equality-gate"
bash scripts/verify-overlay-sources.sh
ok "verify-overlay-sources"
bash scripts/test-draft-sample-method-gate.sh -q
ok "test-draft-sample-method-gate"
bash scripts/test-nccl-fabric-passthrough.sh -q
ok "test-nccl-fabric-passthrough"
bash scripts/test-boot-shape-warmup.sh -q
ok "test-boot-shape-warmup"
bash scripts/test-nccl-ib-hca-gid-resolve.sh -q
ok "test-nccl-ib-hca-gid-resolve"
bash scripts/test-lmcache-compose-gate.sh -q
ok "test-lmcache-compose-gate"

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
if grep -Fq 'python3 /opt/hotfix-dsv4-issue26-hybrid-swa-min.py || exit 1' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/hotfix-dsv4-issue27-partial-prefill-concurrency.py || exit 1' docker-compose.dspark.yml; then
  ok "compose applies #26 + #27 fail-closed"
else
  bad "compose must apply #26 + #27 with || exit 1"
fi
# The safe #27 cap must agree across the fresh-clone env and Compose fallback.
if grep -Fxq 'DSPARK_MAX_INFLIGHT_PREFILLS=2' .env.dspark.example \
  && grep -Fq 'DSPARK_MAX_INFLIGHT_PREFILLS: "${DSPARK_MAX_INFLIGHT_PREFILLS:-2}"' docker-compose.dspark.yml; then
  ok "issue27 in-flight prefill cap defaults to 2 (A/B 2026-09-02)"
else
  bad "issue27 in-flight prefill cap must default to 2 in env example and compose"
fi
if grep -Fq 'hotfix-dsv4-adaptive-prefill-chunk.py}:/opt/hotfix-dsv4-adaptive-prefill-chunk.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ADAPTIVE_CHUNK:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-adaptive-prefill-chunk.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_ADAPTIVE_CHUNK_HOTFIX"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'hotfix-dsv4-replicate-markov-head.py}:/opt/hotfix-dsv4-replicate-markov-head.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_REPLICATE_MARKOV:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-replicate-markov-head.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_REPLICATE_MARKOV_HOTFIX"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'B12X_W4A16_TC_DECODE: "${B12X_W4A16_TC_DECODE:-0}"' docker-compose.dspark.yml \
  && grep -Fq -- '--max-cudagraph-capture-size $$(( ( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-6} + 1) + 7 ) / 8 * 8 ))' docker-compose.dspark.yml; then
  ok "fable5-1 easy knobs: adaptive chunk, replicate Markov, TC-decode env, capture-size round-up"
else
  bad "fable5-1 easy knobs wiring is incomplete"
fi
if grep -Fq 'hotfix-dsv4-sp-indexer-prefill.py}:/opt/hotfix-dsv4-sp-indexer-prefill.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_SP_INDEXER:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-sp-indexer-prefill.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_ENABLE_SP_INDEXER: "${DSPARK_ENABLE_SP_INDEXER:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_SP_INDEXER_MIN_KEYS: "${DSPARK_SP_INDEXER_MIN_KEYS:-8192}"' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_SP_INDEXER_HOTFIX"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq "DSPARK_ENABLE_SP_INDEXER='\$DSPARK_SP_INDEXER_EFFECTIVE'" start-deepseek-v4-flash-dspark.sh \
  && grep -Fxq 'DSPARK_ENABLE_SP_INDEXER=0' .env.dspark.example; then
  ok "item 6 SP indexer prefill: opt-in gate, env passthrough, worker sync, example default 0"
else
  bad "item 6 SP indexer prefill wiring is incomplete"
fi
if grep -Fq 'if [ "$${DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS:-0}" = "1" ]; then bash /opt/dspark-patches/hotfix-deepgemm-sm121-mqa-header-alias.sh || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS: "${DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_DEEPGEMM_ALIAS_HOTFIX"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fxq 'DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=0' .env.dspark.example; then
  ok "DeepGEMM sm121 header alias: opt-in gate, env passthrough, worker sync, example default 0"
else
  bad "DeepGEMM sm121 header alias wiring is incomplete"
fi
if grep -Fq 'hotfix-dsv4-issue133-triton-specialization.py}:/opt/hotfix-dsv4-issue133-triton-specialization.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/hotfix-dsv4-issue133-triton-specialization.py || exit 1' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_ISSUE133_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue133-triton-specialization.py"' start-deepseek-v4-flash-dspark.sh; then
  ok "issue133 triton specialization hotfix is mounted, fail-closed, and worker-synced"
else
  bad "issue133 hotfix wiring is incomplete"
fi
# Issue #141: default OFF; exact 1 mounts and runs the source-locked
# fixed-64 workaround fail-closed on both identically synced ranks.
if grep -Fq 'hotfix-dsv4-issue141-sparse-mla-decode-chunk.py}:/opt/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK: "${DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq 'if [ ! -f "$DSPARK_ISSUE141_HOTFIX" ]; then' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'issue141 sparse-MLA fixed-64 workaround: $DSPARK_ISSUE141_EFFECTIVE' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq "DSPARK_ENABLE_ISSUE141_SPARSE_MLA_CHUNK='\$DSPARK_ISSUE141_EFFECTIVE'" start-deepseek-v4-flash-dspark.sh \
  && grep -Fq "DSPARK_ISSUE141_HOTFIX='./patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py'" start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'scp "$DSPARK_ISSUE141_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py"' start-deepseek-v4-flash-dspark.sh; then
  ok "issue141 workaround is default-off, exact-1, fail-closed, preflighted, reported, and worker-synced"
else
  bad "issue141 sparse-MLA workaround wiring is incomplete"
fi
if grep -Fq 'python3 /opt/hotfix-dsv4-suppress-stops-in-reasoning.py || exit 1' docker-compose.dspark.yml; then
  ok "compose applies suppress-stops-in-reasoning fail-closed"
else
  bad "compose must apply suppress-stops-in-reasoning with || exit 1"
fi
# Issue #66: GPU V2 thinking budget default OFF (stock sampler);
# ON must be an exactly-1 gate with a fail-closed invocation.
if grep -Fq 'DSPARK_ENABLE_ISSUE31_GPU_HOTFIX: "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py || exit 1; fi;' docker-compose.dspark.yml; then
  ok "compose gates issue31 GPU thinking-budget hotfix behind =1, fail-closed"
else
  bad "compose must invoke issue31 GPU hotfix only when DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=1, with || exit 1"
fi
if grep -q 'hotfix-dsv4-issue55-tool-truncation.py' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/hotfix-dsv4-issue55-tool-truncation.py || exit 1' docker-compose.dspark.yml; then
  ok "compose applies issue #55 tool-call truncation safety fail-closed"
else
  bad "compose must apply issue #55 with || exit 1"
fi
if grep -Fq 'hotfix-vllm-empty-encoder-output.py}:/opt/hotfix-vllm-empty-encoder-output.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/hotfix-vllm-empty-encoder-output.py || exit 1' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_EMPTY_ENCODER_OUTPUT_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-empty-encoder-output.py"' start-deepseek-v4-flash-dspark.sh; then
  ok "empty encoder output hotfix is mounted, fail-closed, and worker-synced"
else
  bad "empty encoder output hotfix wiring is incomplete"
fi
if grep -Fq 'hotfix-dsv4-vision-exp.py}:/opt/hotfix-dsv4-vision-exp.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/hotfix-dsv4-vision-exp.py || exit 1' docker-compose.dspark.yml \
  && grep -Fq 'scp "$DSPARK_VISION_EXP_HOTFIX"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'scp -r "$SCRIPT_DIR/patches/vision_exp/."' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq "rm -rf '\${REMOTE_WORKER_DIR}/patches/vision_exp'" start-deepseek-v4-flash-dspark.sh \
  && [ -f patches/hotfix-dsv4-vision-exp.py ] \
  && [ -f patches/vision_exp/apply.py ]; then
  ok "Vision-Exp native image hotfix is mounted, fail-closed, and worker-synced"
else
  bad "Vision-Exp native image hotfix wiring is incomplete"
fi
# Anemll argparse parses --limit-mm-per-prompt with json.loads. Bare `image=8`
# is ArgumentTypeError at exec. Convert image=N in the entrypoint; pass JSON.
if grep -Fq 'LIMIT_MM_ARGS=(--limit-mm-per-prompt "$${LIMIT_MM_JSON}")' docker-compose.dspark.yml \
  && grep -Fq '"$${LIMIT_MM_ARGS[@]}"' docker-compose.dspark.yml \
  && ! grep -Fq -- '--limit-mm-per-prompt ${LIMIT_MM_PER_PROMPT:-image=8}' docker-compose.dspark.yml; then
  ok "limit-mm-per-prompt is converted to JSON before vllm argparse"
else
  bad "compose must not pass bare image=8 to --limit-mm-per-prompt (JSON only)"
fi
# Assistant-final continuation (#52/PR53): default OFF (stock renderer);
# ON must be an exactly-1 gate with a fail-closed invocation.
if grep -Fq 'DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX: "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1" ]; then python3 /opt/hotfix-dsv4-assistant-final-continuation.py || exit 1; fi;' docker-compose.dspark.yml; then
  ok "compose gates assistant-final hotfix behind =1, fail-closed"
else
  bad "compose must invoke assistant-final hotfix only when DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX=1, with || exit 1"
fi
# Issue #138 Responses history replay: default OFF, exact-1/fail-closed on
# both ranks, with launcher preflight/reporting and canonical worker sync.
issue138_worker_env="DSPARK_ISSUE138_HOTFIX='./patches/hotfix-vllm-issue138-responses-history.py'"
issue138_worker_count="$(grep -Fc "$issue138_worker_env" start-deepseek-v4-flash-dspark.sh || true)"
if grep -Fq 'hotfix-vllm-issue138-responses-history.py}:/opt/hotfix-vllm-issue138-responses-history.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT: "${DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT:-0}" = "1" ]; then python3 /opt/hotfix-vllm-issue138-responses-history.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq '# Issue #138 Responses history compatibility pre-flight (begin).' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'issue138 Responses history compatibility: 0 (stock)' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'issue138 Responses history compatibility: 1 (apply)' start-deepseek-v4-flash-dspark.sh \
  && [ "$issue138_worker_count" -eq 4 ] \
  && grep -Fq 'scp "$DSPARK_ISSUE138_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-issue138-responses-history.py"' start-deepseek-v4-flash-dspark.sh; then
  ok "issue138 hotfix is default-off, exact-1 fail-closed, preflighted, reported, and propagated to worker1 and worker2"
else
  bad "issue138 Responses history hotfix wiring is incomplete"
fi
# Codex agent_message replay: default OFF, exact-1/fail-closed, applied after
# issue138, and propagated to both ranks.
codex_agent_worker_env="DSPARK_CODEX_AGENT_MESSAGE_HOTFIX='./patches/hotfix-vllm-codex-agent-message.py'"
codex_agent_worker_count="$(grep -Fc "$codex_agent_worker_env" start-deepseek-v4-flash-dspark.sh || true)"
if grep -Fq 'hotfix-vllm-codex-agent-message.py}:/opt/hotfix-vllm-codex-agent-message.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT: "${DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT:-0}" = "1" ]; then python3 /opt/hotfix-vllm-codex-agent-message.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fq '# Codex agent_message compatibility pre-flight (begin).' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'Codex agent_message compatibility: 0 (stock)' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'Codex agent_message compatibility: 1 (apply)' start-deepseek-v4-flash-dspark.sh \
  && [ "$codex_agent_worker_count" -eq 4 ] \
  && grep -Fq 'scp "$DSPARK_CODEX_AGENT_MESSAGE_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-vllm-codex-agent-message.py"' start-deepseek-v4-flash-dspark.sh; then
  ok "Codex agent_message hotfix is default-off, exact-1 fail-closed, and propagated to worker1 and worker2"
else
  bad "Codex agent_message hotfix wiring is incomplete"
fi
if grep -q 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"' docker-compose.dspark.yml \
  && grep -q 'TILELANG_CACHE_DIR: "${TILELANG_CACHE_DIR:-/cache/huggingface/tilelang-cache}"' docker-compose.dspark.yml; then
  ok "compose JIT timeout 1800s + persistent TileLang cache (#65/#87)"
else
  bad "compose missing VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 or TILELANG_CACHE_DIR"
fi
if grep -Fq 'DSPARK_WORKER_HF_NFS="${DSPARK_WORKER_HF_NFS:-0}"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'source "$SCRIPT_DIR/files/nfs-share.sh"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'dspark-hf:/cache/huggingface:ro' docker-compose.dspark-nfs.override.yml \
  && grep -Fq 'docker-compose.dspark-nfs.override.yml' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'DSPARK_WORKER_HF_NFS:-0' prepare-dspark-model-cache.sh \
  && grep -Fq -- '--nfs' stop-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'vllm-fn-nfs' files/nfs-share.sh; then
  ok "worker HF NFS share is opt-in (default 0), override-mounted, prepare copies worker unless =1, stop --nfs safe vs Qwen"
else
  bad "worker HF NFS wiring is incomplete"
fi
if grep -Fq 'hotfix-dsv4-responses-store.py}:/opt/hotfix-dsv4-responses-store.py:ro' docker-compose.dspark.yml \
  && grep -Fq 'VLLM_ENABLE_RESPONSES_API_STORE: "${VLLM_ENABLE_RESPONSES_API_STORE:-0}"' docker-compose.dspark.yml \
  && grep -Fq 'DSPARK_RESPONSES_STORE_MAX_ENTRIES: "${DSPARK_RESPONSES_STORE_MAX_ENTRIES:-256}"' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${VLLM_ENABLE_RESPONSES_API_STORE:-0}" != "1" ]; then export VLLM_ENABLE_RESPONSES_API_STORE=0; fi;' docker-compose.dspark.yml \
  && grep -Fq 'if [ "$${VLLM_ENABLE_RESPONSES_API_STORE}" = "1" ]; then python3 /opt/hotfix-dsv4-responses-store.py || exit 1; fi;' docker-compose.dspark.yml \
  && grep -Fxq 'VLLM_ENABLE_RESPONSES_API_STORE=0' .env.dspark.example \
  && grep -Fxq 'DSPARK_RESPONSES_STORE_MAX_ENTRIES=256' .env.dspark.example \
  && grep -Fq '# Bounded Responses API store pre-flight (begin).' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'Responses API store: disabled (stock source)' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'Responses API store: enabled (terminal cap $DSPARK_RESPONSES_STORE_MAX_ENTRIES)' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'scp "$DSPARK_RESPONSES_STORE_HOTFIX" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-responses-store.py"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq '/opt/hotfix-dsv4-responses-store.py --check' start-deepseek-v4-flash-dspark.sh; then
  ok "Responses store is exact-1, fail-closed, source-checked, reported, and propagated to every rank"
else
  bad "bounded Responses store wiring is incomplete"
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
  patches/hotfix-dsv4-responses-store.py \
  patches/hotfix-dsv4-issue26-hybrid-swa-min.py \
  patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py \
  patches/hotfix-dsv4-adaptive-prefill-chunk.py \
  patches/hotfix-dsv4-replicate-markov-head.py \
  patches/hotfix-dsv4-issue133-triton-specialization.py \
  patches/hotfix-dsv4-issue141-sparse-mla-decode-chunk.py \
  patches/hotfix-vllm-empty-encoder-output.py \
  patches/hotfix-dsv4-vision-exp.py \
  patches/hotfix-vllm-issue136-xgrammar-termination.py \
  patches/hotfix-nvfp4-ds-mla-issue22.sh \
  patches/hotfix-gb10-spin-wait.sh \
  patches/hotfix-dsv4-suppress-stops-in-reasoning.py \
  patches/hotfix-dsv4-assistant-final-continuation.py \
  patches/hotfix-vllm-issue138-responses-history.py \
  patches/hotfix-vllm-codex-agent-message.py \
  patches/hotfix-vllm-redact-api-key-log.sh
do
  if [ -f "$p" ]; then
    ok "present $p"
  else
    bad "missing required $p"
  fi
done

# Multi-key auth: keyed starts apply and verify redaction fail-closed outside
# the optional performance-hotfix loop, while the worker sync keeps shipping it.
if grep -Fq 'bash /opt/dspark-patches/hotfix-vllm-redact-api-key-log.sh || exit 1' docker-compose.dspark.yml \
  && grep -Fq 'hotfix-vllm-redact-api-key-log.sh --status || exit 1' docker-compose.dspark.yml \
  && ! grep -E 'for _hf in .*hotfix-vllm-redact-api-key-log.sh' docker-compose.dspark.yml >/dev/null \
  && grep -E 'for _hf_sync in .*hotfix-vllm-redact-api-key-log.sh' start-deepseek-v4-flash-dspark.sh >/dev/null; then
  ok "compose redaction gate is fail-closed and worker sync retains the patch"
else
  bad "redact-api-key-log must apply + verify outside the optional loop and remain in worker sync"
fi

# Optional TP=3: pad only at TP_SIZE=3; default start still forces two nodes.
if [ -f start-tp3.sh ] && grep -Fq 'export DSPARK_TP3=1' start-tp3.sh \
  && grep -Fq 'exec "$SCRIPT_DIR/start-deepseek-v4-flash-dspark.sh"' start-tp3.sh \
  && grep -Fq -- '--max-num-seqs' start-tp3.sh; then
  ok "start-tp3.sh is an opt-in exec wrapper with --max-num-seqs"
else
  bad "start-tp3.sh must exec the 2-node start with DSPARK_TP3=1 and --max-num-seqs"
fi

if grep -Fq 'DSPARK_TP3="${DSPARK_TP3:-0}"' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'TP_SIZE=2' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq 'NNODES=2' start-deepseek-v4-flash-dspark.sh \
  && grep -Fq ': "${WORKER2_HOST:?WORKER2_HOST must be set' start-deepseek-v4-flash-dspark.sh; then
  ok "2-node start forces TP=2/NNODES=2; TP=3 requires WORKER2_HOST"
else
  bad "start-deepseek-v4-flash-dspark.sh must force TP=2 unless DSPARK_TP3=1"
fi

if grep -Fq 'if [ "${TP_SIZE:-2}" = "3" ]; then' docker-compose.dspark.yml \
  && grep -Fq 'python3 /opt/dsv4-tp3/apply_tp3_patch.py' docker-compose.dspark.yml \
  && grep -Fq -- '--tensor-parallel-size ${TP_SIZE:-2}' docker-compose.dspark.yml \
  && grep -Fq -- '--nnodes ${NNODES:-2}' docker-compose.dspark.yml \
  && grep -Fq '/opt/dsv4-tp3:ro' docker-compose.dspark.yml; then
  ok "compose interpolates TP_SIZE/NNODES and gates the TP=3 pad"
else
  bad "compose must interpolate TP_SIZE/NNODES and apply the pad only at TP_SIZE=3"
fi

if [ -f patches/tp3/apply_tp3_patch.py ] && [ -f patches/dsv4_tp_pad.py ] && [ -f scripts/validate_tp3.sh ]; then
  ok "TP=3 pad + validate_tp3.sh present"
else
  bad "missing patches/tp3/apply_tp3_patch.py, patches/dsv4_tp_pad.py, or scripts/validate_tp3.sh"
fi

if [ "$fail" -ne 0 ]; then
  echo "CI validate FAILED" >&2
  exit 1
fi

# Healthcheck present, worker-gated, and probing the compose-time VLLM_HOST
# (rank 1 is headless; a hardcoded 127.0.0.1 probe is wrong for a LAN-IP bind).
if grep -q "healthcheck:" docker-compose.dspark.yml \
  && grep -qF 'if [ -n \"$$HEADLESS\" ]' docker-compose.dspark.yml \
  && grep -qF "urlhost='\${VLLM_HOST:-127.0.0.1}'" docker-compose.dspark.yml; then
  ok "compose healthcheck present, HEADLESS-gated, and VLLM_HOST-aware"
else
  bad "compose healthcheck missing, not HEADLESS-gated, or still hardcoded to 127.0.0.1"
fi

if grep -qF -- '--tensor-parallel-size ${TP_SIZE:-2}' docker-compose.dspark.yml \
  && grep -qF -- '--nnodes ${NNODES:-2}' docker-compose.dspark.yml \
  && grep -q 'apply_tp3_patch.py' docker-compose.dspark.yml \
  && grep -q 'DSPARK_TP3=1' start-tp3.sh \
  && grep -q -- '--max-num-seqs' start-tp3.sh \
  && grep -q 'TP3_MAX_NUM_SEQS' start-deepseek-v4-flash-dspark.sh \
  && grep -q 'apply_tp3_bootstrap_ifaces' start-deepseek-v4-flash-dspark.sh \
  && [ -f patches/tp3/apply_tp3_patch.py ]; then
  ok "optional TP=3: compose parameterizes TP/nnodes, start-tp3.sh sets DSPARK_TP3=1, Gloo bootstrap on 10.0.0.x"
else
  bad "TP=3 optional path missing (compose TP_SIZE/NNODES, apply_tp3_patch, start-tp3.sh, or bootstrap ifaces)"
fi

echo "CI validate passed (CPU recipe gates only)."
