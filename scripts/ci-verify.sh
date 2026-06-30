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
  scripts/verify-overlay-sources.sh \
  scripts/ci-verify.sh

section "python benchmark compile"
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  benchmarks/bench_concurrent.py \
  benchmarks/staggered_bench.py \
  benchmarks/correctness_test.py \
  benchmarks/gsm8k_eval.py \
  benchmarks/needle_acceptance_sweep.py
rm -rf benchmarks/__pycache__

section "required reproducibility files"
required_files=(
  AGENTS.md
  README.md
  .env.dspark.example
  docker-compose.dspark.yml
  build-dspark-vllm-runtime.sh
  validate-dspark-config.sh
  run-dspark-dual-gb10.sh
  recipe/Dockerfile.dspark-runtime-overlay
  recipe/nvfp4/Dockerfile.stage-a
  recipe/nvfp4/Dockerfile.stage-b
  recipe/nvfp4/Dockerfile.stage-c
  docs/CONTAINER_REPRODUCIBILITY.md
  docs/DSPARK_R0B0TLAB_1M.md
  docs/DSPARK_R0B0TLAB_384K.md
  profiles/dspark-r0b0tlab-1m.env
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
  "max model len: 1048576" \
  "max num seqs: 2" \
  "gpu memory utilization: 0.88" \
  "image: vllm-dspark-runtime:dspark-nvfp4-stage-c" \
  "--kv-cache-dtype nvfp4_ds_mla" \
  "--max-model-len 1048576" \
  "--max-num-seqs 2" \
  "--gpu-memory-utilization 0.88" \
  "--master-port 25000"; do
  grep -F -- "$expected" <<<"$render_out" >/dev/null || fail "config render missing: $expected"
done

section "profile assertions"
grep -Fx 'MAX_MODEL_LEN=1048576' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing MAX_MODEL_LEN=1048576"
grep -Fx 'MAX_NUM_SEQS=2' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing MAX_NUM_SEQS=2"
grep -Fx 'GPU_MEMORY_UTILIZATION=0.88' profiles/dspark-r0b0tlab-1m.env >/dev/null || fail "1M profile missing GPU_MEMORY_UTILIZATION=0.88"
grep -Fx 'MAX_MODEL_LEN=384000' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile missing MAX_MODEL_LEN=384000"
grep -Fx 'MAX_NUM_SEQS=4' profiles/dspark-r0b0tlab-384k.env >/dev/null || fail "384K profile missing MAX_NUM_SEQS=4"

section "publication artifact integrity"
tar -tzf publication/DSpark-r0b0tlab-384K.tar.gz >/dev/null
sha256sum -c publication/DSpark-r0b0tlab-384K.tar.gz.sha256

grep -F 'DSpark-r0b0tlab' publication/DSpark-r0b0tlab-test-results.html >/dev/null || fail "HTML report missing DSpark-r0b0tlab"
grep -F 'strict 1M sweep: 3/3 pass' publication/DSpark-r0b0tlab-test-results.html >/dev/null || fail "HTML report missing strict sweep summary"

section "sanitization"
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
