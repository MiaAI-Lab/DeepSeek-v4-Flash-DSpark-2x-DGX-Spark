#!/usr/bin/env bash
# boot-shape-warmup.sh — burn spec-decode/prefill Triton shape buckets at boot.
#
# Why (issue #117): under live concurrent traffic, batch shapes that the single
# smoke request never materializes JIT-compile mid-serve. jit_monitor warns
# about the latency spike, but the real hazard on TP=2 is worse: a rank stalled
# in compilation leaves its peer waiting in a collective, and torch's
# ProcessGroupNCCL watchdog (600 s, NOT covered by
# VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS) kills the pair. Observed kernel:
# _prepare_dflash_inputs_kernel, whose shape key is
#   BLOCK_SIZE = min(256, next_pow2(max_tokens_per_req))
# so the reachable buckets are small and enumerable: spec-decode steps sit at
# next_pow2(K+1), any prefill chunk >=256 tokens caps at 256, and small
# prefill tails fill the low buckets. This sweep materializes them at boot,
# before traffic: concurrency C=1/2/4/6 (multi-request decode batches), a
# medium and a multi-chunk long prefill (8192-chunk + odd tail), and a
# thinking-off arm. Prompts carry a per-request nonce so prefix caching cannot
# skip the prefill compute being warmed.
#
# Non-fatal by design: the cost of a missed shape is a mid-serve JIT (what this
# script exists to reduce), not an outage — the launcher must treat a warmup
# failure as WARN, never as a boot failure. Pair with a persistent
# TRITON_CACHE_DIR so each bucket is compiled once per image, not once per boot.
#
# Usage: boot-shape-warmup.sh [base_url] [model]
#   base_url default http://127.0.0.1:8888 ; model default deepseek-v4-flash-0731
# Env:
#   DSPARK_WARMUP_REQ_TIMEOUT  per-request curl --max-time, seconds (default 240
#                              — first-ever boot pays real compiles here)
#   VLLM_API_KEY               added as Bearer auth when non-empty
#   WARMUP_CURL                test seam: overrides the curl binary
set -u

BASE="${1:-http://127.0.0.1:8888}"
MODEL="${2:-deepseek-v4-flash-0731}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${DSPARK_WARMUP_REQ_TIMEOUT:-240}"
NONCE="$$-$(date +%s)"

AUTH_ARGS=()
[ -n "${VLLM_API_KEY:-}" ] && AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mk_prompt() { # $1 = approx token count (repeated filler words), $2 = tag
  local n=$1 tag=$2 body
  body=$(printf 'warm %.0s' $(seq 1 "$n"))
  printf '[warmup %s %s] The following is filler context, ignore it: %s Reply with OK.' \
    "$NONCE" "$tag" "$body"
}

fire() { # $1 = tag, $2 = words, $3 = thinking(true|false), $4 = result file
  local tag=$1 words=$2 thinking=$3 out=$4 prompt
  prompt=$(mk_prompt "$words" "$tag")
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"thinking":'"$thinking"',"reasoning_effort":"low"}}' \
      >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$out"
  else
    echo fail > "$out"
  fi
}

burst() { # $1 = arm name, $2 = concurrency, $3 = words-per-request
  local arm=$1 c=$2 words=$3 i t0 t1
  t0=$(date +%s)
  for i in $(seq 1 "$c"); do
    fire "${arm}-${i}" "$words" true "$tmpdir/${arm}-${i}" &
  done
  wait
  t1=$(date +%s)
  echo "  arm ${arm}: C=${c} x ~${words} tok, $((t1 - t0))s"
}

if ! "$CURL_BIN" -fsS --max-time 10 "${AUTH_ARGS[@]}" "$BASE/v1/models" >/dev/null 2>&1; then
  echo "boot-shape-warmup: API not reachable at $BASE — skipping sweep" >&2
  exit 1
fi

echo "boot-shape-warmup: sweeping spec-decode/prefill shape buckets (issue #117)"
total_t0=$(date +%s)

burst c1        1 300
burst c2        2 420
burst c4        4 380
burst c6        6 350
burst mid       1 2600
burst longchunk 1 9500          # crosses the 8192-token chunk boundary + odd tail
t0=$(date +%s)
fire nothink-1 300 false "$tmpdir/nothink-1"
t1=$(date +%s)
echo "  arm nothink: C=1 x ~300 tok, thinking=false, $((t1 - t0))s"

total=0 ok_count=0
for f in "$tmpdir"/*-*; do
  [ -f "$f" ] || continue
  total=$((total + 1))
  [ "$(cat "$f")" = "ok" ] && ok_count=$((ok_count + 1))
done
total_t1=$(date +%s)
echo "boot-shape-warmup: ${ok_count}/${total} requests ok in $((total_t1 - total_t0))s"

if [ "$ok_count" -lt "$total" ]; then
  echo "boot-shape-warmup: $((total - ok_count)) request(s) failed — uncovered shapes may JIT mid-serve" >&2
  sed -n '1,5p' "$tmpdir/errors" >&2 2>/dev/null || true
  exit 1
fi
exit 0
