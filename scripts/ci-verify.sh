#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

section() {
  printf '\n== %s ==\n' "$*"
}

section "shell syntax"
bash -n \
  build-dspark-vllm-runtime.sh \
  run-dspark-dual-gb10.sh \
  validate-dspark-config.sh \
  start-deepseek-v4-flash-dspark.sh \
  stop-deepseek-v4-flash-dspark.sh \
  status-deepseek-v4-flash-dspark.sh \
  logs-deepseek-v4-flash-dspark.sh \
  smoke-deepseek-v4-flash-dspark.sh \
  scripts/build-production-image.sh \
  scripts/entrypoint.sh \
  scripts/verify-overlay-sources.sh \
  scripts/ci-verify.sh

section "python benchmark compile"
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  benchmarks/bench_concurrent.py \
  benchmarks/staggered_bench.py \
  benchmarks/correctness_test.py \
  benchmarks/gsm8k_eval.py \
  benchmarks/needle_acceptance_sweep.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/audit_runtime.py \
  scripts/benchmark_dspark.py \
  scripts/long_context_gate.py \
  scripts/public_safety_scan.py \
  scripts/runtime_gate.py \
  scripts/verify_model_checkpoint.py \
  scripts/verify_release.py \
  tests/test_release_contract.py \
  tests/test_launch_contract.py \
  tests/test_benchmark_harness_scaffold.py
rm -rf benchmarks/__pycache__ scripts/__pycache__ tests/__pycache__

section "contract tests"
python3 tests/test_release_contract.py
python3 tests/test_launch_contract.py
python3 tests/test_benchmark_harness_scaffold.py
python3 tests/test_model_checkpoint.py
rm -rf benchmarks/__pycache__ scripts/__pycache__ tests/__pycache__

section "required reproducibility files"
required_files=(
  AGENTS.md
  PRIVACY.md
  README.md
  .env.dspark.example
  docker-compose.dspark.yml
  build-dspark-vllm-runtime.sh
  validate-dspark-config.sh
  run-dspark-dual-gb10.sh
  recipe/Dockerfile.production
  docker/runtime-manifest.production.json
  scripts/audit_runtime.py
  scripts/entrypoint.sh
  scripts/benchmark_dspark.py
  scripts/long_context_gate.py
  scripts/public_safety_scan.py
  scripts/runtime_gate.py
  scripts/verify_model_checkpoint.py
  scripts/verify_release.py
  recipe/Dockerfile.dspark-runtime-overlay
  recipe/nvfp4/Dockerfile.stage-a
  recipe/nvfp4/Dockerfile.stage-b
  recipe/nvfp4/Dockerfile.stage-c
  docs/CONTAINER_REPRODUCIBILITY.md
  docs/DSPARK_R0B0TLAB_1M.md
  docs/DSPARK_R0B0TLAB_384K.md
  profiles/dspark-r0b0tlab-1m.env
  profiles/dspark-r0b0tlab-production.env
  profiles/dspark-r0b0tlab-384k.env
  publication/DSpark-r0b0tlab-test-results.html
  publication/DSpark-r0b0tlab-384K.tar.gz
  publication/DSpark-r0b0tlab-384K.tar.gz.sha256
)
for f in "${required_files[@]}"; do
  [[ -f "$f" ]] || fail "missing required file: $f"
done

section "overlay source presence"
scripts/verify-overlay-sources.sh

section "config render"
render_out="$(ENV_FILE=.env.dspark.example ./validate-dspark-config.sh)"
printf '%s\n' "$render_out"
for expected in \
  "max model len: 200000" \
  "max num seqs: 16" \
  "gpu memory utilization: 0.84" \
  "image: dspark-r0b0tlab:production-candidate" \
  "--kv-cache-dtype nvfp4_ds_mla" \
  "--max-model-len 200000" \
  "--max-num-seqs 16" \
  "--gpu-memory-utilization 0.84" \
  "--master-port 25000"; do
  grep -F -- "$expected" <<<"$render_out" >/dev/null || fail "config render missing: $expected"
done

section "profile assertions"
grep -Fx 'MAX_MODEL_LEN=1048576' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing MAX_MODEL_LEN=1048576"
grep -Fx 'MAX_NUM_SEQS=2' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing MAX_NUM_SEQS=2"
grep -Fx 'GPU_MEMORY_UTILIZATION=0.88' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing GPU_MEMORY_UTILIZATION=0.88"
grep -Fx 'KV_CACHE_DTYPE=nvfp4_ds_mla' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile lost NVFP4 KV"
grep -Fx 'DSPARK_RUNTIME_LANE=legacy-stage-c' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing legacy Stage-C lane selector"
grep -Fx 'DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile must remain on the admitted Stage-C compatibility lane"
grep -Fx 'MAX_MODEL_LEN=200000' profiles/dspark-r0b0tlab-production.env >/dev/null || fail "production profile missing MAX_MODEL_LEN=200000"
grep -Fx 'MAX_NUM_SEQS=16' profiles/dspark-r0b0tlab-production.env >/dev/null || fail "production profile missing MAX_NUM_SEQS=16"
grep -Fx 'KV_CACHE_DTYPE=nvfp4_ds_mla' profiles/dspark-r0b0tlab-production.env >/dev/null || fail "production profile lost NVFP4 KV"
grep -Fx 'DSPARK_RUNTIME_LANE=native-v025' profiles/dspark-r0b0tlab-production.env >/dev/null || fail "production profile missing native-v025 lane selector"
grep -Fx 'MAX_MODEL_LEN=384000' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile missing MAX_MODEL_LEN=384000"
grep -Fx 'MAX_NUM_SEQS=4' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile missing MAX_NUM_SEQS=4"
grep -Fx 'KV_CACHE_DTYPE=nvfp4_ds_mla' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile lost NVFP4 KV"
grep -Fx 'DSPARK_RUNTIME_LANE=legacy-stage-c' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile missing legacy Stage-C lane selector"
grep -Fx 'DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile must remain on Stage-C"

section "production evidence"
sha256sum -c results/production-candidate/SHA256SUMS
python3 - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("results/production-candidate/summary.json").read_text())
assert summary["candidate"]["kv_cache_dtype"] == "nvfp4_ds_mla"
assert summary["candidate"]["mtp_tokens_selected"] == 5
assert summary["runtime_gate"]["status"] == "PASS"
assert summary["long_context_uncached"]["status"] == "PASS"
assert summary["long_context_uncached"]["prompt_tokens"] >= 100000
assert summary["staggered_c16"]["requests_ok"] == 16
assert summary["decode_k5_confirmation"]["1"]["requests_ok"] == 5
assert summary["decode_k5_confirmation"]["16"]["requests_ok"] == 80
print("production evidence contract passed")
PY
python3 scripts/verify_release.py --allow-dirty --source-only

section "publication artifact integrity"
tar -tzf publication/DSpark-r0b0tlab-384K.tar.gz >/dev/null
sha256sum -c publication/DSpark-r0b0tlab-384K.tar.gz.sha256

grep -F 'DSpark-r0b0tlab' publication/DSpark-r0b0tlab-test-results.html >/dev/null || fail "HTML report missing DSpark-r0b0tlab"
grep -F 'strict 1M sweep: 3/3 pass' publication/DSpark-r0b0tlab-test-results.html >/dev/null || fail "HTML report missing strict sweep summary"

section "sanitization"
python3 scripts/public_safety_scan.py .
scan_targets=(
  AGENTS.md
  README.md
  CREDITS.md
  .env.dspark.example
  docker-compose.dspark.yml
  build-dspark-vllm-runtime.sh
  validate-dspark-config.sh
  run-dspark-dual-gb10.sh
  docs
  profiles
  publication
  benchmarks/*.py
  recipe/nvfp4/Dockerfile.stage-a
  recipe/nvfp4/Dockerfile.stage-b
  recipe/nvfp4/Dockerfile.stage-c
)
# Intentionally excludes recipe/overlay and patches, which preserve upstream source names/comments.
if grep -RInE 'Mia|MIA|mia-|mia-dspark|/home/r0b0tdgx|/home/zurih|169\.254|10\.100|192\.168\.0\.1|192\.168\.0\.2|AKIA|BEGIN (RSA|OPENSSH|PRIVATE)|hf_[A-Za-z0-9]{20,}' "${scan_targets[@]}"; then
  fail "sanitization scan found forbidden strings"
fi

section "git cleanliness"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
  echo "Tracked changes are present; this is allowed when verifying an in-progress commit." >&2
  git status --short --untracked-files=no >&2
fi
# Fail only on unexpected generated files that canonical verification itself should never create.
if find . -path './.git' -prune -o -name '__pycache__' -print -quit | grep -q .; then
  fail "__pycache__ left behind; remove generated Python cache directories"
fi

section "canonical green"
echo "DSpark-r0b0tlab canonical repository verification passed."
