#!/usr/bin/env bash
# boot-shape-warmup.sh — burn spec-decode/prefill Triton shape buckets at boot.
#
# Why (issue #117): under live traffic, shapes that the single smoke request
# never materializes JIT-compile mid-serve. jit_monitor warns about the latency
# spike, but the real hazard on TP=2 is worse: a rank stalled in compilation
# leaves its peer waiting in a collective, and torch's ProcessGroupNCCL
# watchdog (600 s, NOT covered by VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS) kills
# the pair. Target kernel: _prepare_dflash_inputs_kernel. Its compile key is
#   BLOCK_SIZE = min(256, next_pow2(scheduled_tokens + 6))
# (+6 = 1 + num_speculative_tokens at the deployed num_speculative_tokens=5).
# Request concurrency does NOT enter this key at all, so the empirically live
# BLOCK keys {8, 16, 32, 64, 128, 256} are reached only through exact
# scheduled-token counts s with next_pow2(s+6) = B — never through chat-batch
# concurrency (all our chat prompts land BLOCK 256), and the longchunk tail
# does not help either (~1342 tokens -> BLOCK 256, not a low bucket).
#
# Two mechanisms:
# - Bucket ladder: six plain POST /v1/completions requests whose prompts are
#   built to encode exactly s tokens, s = {1, 6, 20, 45, 100, 200}, mapping via
#   next_pow2(s+6) onto every live BLOCK key {8,16,32,64,128,256}. The
#   deployed tokenizer encodes 'hello' + (s-1)x' hello' as exactly s tokens,
#   but that heuristic is never trusted blindly: each rung is verified at
#   runtime with an authenticated POST /tokenize BEFORE its completion fires,
#   and any count mismatch fails the rung — and the nonfatal warmup — with a
#   precise, secret-free diagnostic instead of silently warming a wrong shape.
# - Chat arms C=1/2/4/6 (bounded by the launcher's resolved
#   --max-num-seqs) cover both bounded longer prompts and ordinary short
#   requests with client-default generation settings. Medium + multi-chunk
#   long prefill and one thinking-off arm cover other batch-keyed variants;
#   none of these arms contributes to the low buckets above.
# - Sampling arms (2026-08-24 incident follow-up): every arm above runs
#   greedy (temperature 0), so vllm/v1/sample/ops/topk_topp_triton.py's
#   _topk_topp_kernel — dispatched only when a request carries top_k and/or
#   top_p — was never warmed and JIT-compiled mid-serve (persistent-cache
#   entries born 15:26/15:27 UTC on a serving box prove it). The pinned
#   runtime's enumerable compile-key axes are the TOPK_ENABLED x TOPP_ENABLED
#   constexpr pair. BATCH_SIZE remains a plain TTIR runtime argument. Each
#   combo runs once at C=1 and, when the profile permits it, in one C=3 burst:
#   live cold-cache validation showed the repeated dispatch is needed to
#   materialize every combo reliably, but it must not require a second cache
#   class. The resulting combos are verified as an observable postcondition.
#   (The second family from the same incident,
#   _compute_global_topk_indices_and_lens_kernel's pointer-alignment keys,
#   is closed engine-side by #135's do_not_specialize_on_alignment fix and
#   needs no request-side arms.)
#
# Non-fatal by design: the cost of a missed shape is a mid-serve JIT (what this
# script exists to reduce), not an outage — the launcher must treat a warmup
# failure as WARN, never as a boot failure. Pair with a persistent
# TRITON_CACHE_DIR so each bucket is compiled once per image, not once per boot.
#
# Usage: boot-shape-warmup.sh [base_url] [model]
#   base_url default http://127.0.0.1:8888 ; model default deepseek-v4-flash-0731
# Env:
#   DSPARK_WARMUP_REQ_TIMEOUT  per-request curl --max-time for chat arms and
#                              ladder completions, seconds (default 240 —
#                              first-ever boot pays real compiles here)
#   DSPARK_WARMUP_BEARER       bearer handed over by the launcher (first parsed
#                              DSPARK_API_KEYS key, else VLLM_API_KEY); preferred
#                              over VLLM_API_KEY. Never logged by this script.
#   VLLM_API_KEY               added as Bearer auth when non-empty and no
#                              DSPARK_WARMUP_BEARER was provided
#   DSPARK_WARMUP_MAX_CONCURRENCY
#                              resolved --max-num-seqs from the launcher
#                              (default 6); C=1/2/3/4/6 arms above it are skipped
#   DSPARK_WARMUP_TRITON_CACHE_DIR
#                              host path of THIS node's persistent Triton cache
#                              (launcher derives it from HF_CACHE when
#                              TRITON_CACHE_DIR sits on the HF volume). Used
#                              only by the sampler-cache postcondition; empty
#                              or missing dir skips that check with a note.
#   WARMUP_CURL                test seam: overrides the curl binary
set -u

BASE="${1:-http://127.0.0.1:8888}"
MODEL="${2:-deepseek-v4-flash-0731}"
CURL_BIN="${WARMUP_CURL:-curl}"
REQ_TIMEOUT="${DSPARK_WARMUP_REQ_TIMEOUT:-240}"
MAX_CONCURRENCY="${DSPARK_WARMUP_MAX_CONCURRENCY:-6}"
case "$MAX_CONCURRENCY" in
  ''|*[!0-9]*|0)
    echo "boot-shape-warmup: invalid DSPARK_WARMUP_MAX_CONCURRENCY=${MAX_CONCURRENCY@Q}; using 6" >&2
    MAX_CONCURRENCY=6
    ;;
esac
NONCE="$$-$(date +%s)"

AUTH_ARGS=()
if [ -n "${DSPARK_WARMUP_BEARER:-}" ]; then
  # Launcher-provided bearer wins: it is the same credential the smoke probe
  # authenticated with, so keyed clusters cannot 401 the whole sweep away.
  AUTH_ARGS=(-H "Authorization: Bearer ${DSPARK_WARMUP_BEARER}")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

# Deterministic bucket ladder: exact prompt-token counts -> live BLOCK keys.
LADDER_S=(1 6 20 45 100 200)    # next_pow2(s+6) = 8 16 32 64 128 256
next_pow2() { # smallest power of two >= $1
  local n=$1 p=1
  while [ "$p" -lt "$n" ]; do p=$((p * 2)); done
  printf '%s' "$p"
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mk_prompt() { # $1 = approx token count (repeated filler words), $2 = tag
  local n=$1 tag=$2 body
  body=$(printf 'warm %.0s' $(seq 1 "$n"))
  printf '[warmup %s %s] The following is filler context, ignore it: %s Reply with OK.' \
    "$NONCE" "$tag" "$body"
}

fire() { # $1 = tag, $2 = words, $3 = thinking, $4 = result file, $5 = request profile
  local tag=$1 words=$2 thinking=$3 out=$4 profile=${5:-bounded} prompt payload sample_fields
  prompt=$(mk_prompt "$words" "$tag")
  if [ "$profile" = "serve-default" ]; then
    # Mirror an ordinary short client request: no explicit max_tokens or
    # chat-template override. These scheduler defaults have distinct Triton
    # variants from the bounded long-context arms below.
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"temperature":0}'
  elif [ "${profile#sampling}" != "$profile" ]; then
    # Sampling arms: temperature must be > 0 or vLLM drops the k/p tensors and
    # _topk_topp_kernel never dispatches. Bounded max_tokens keeps each arm to
    # a few sampler steps — the kernel key does not depend on output length.
    case "$profile" in
      sampling-k)  sample_fields='"top_k":40' ;;
      sampling-p)  sample_fields='"top_p":0.9' ;;
      *)           sample_fields='"top_k":40,"top_p":0.9' ;;
    esac
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":24,"temperature":0.8,'"$sample_fields"',"chat_template_kwargs":{"thinking":'"$thinking"',"reasoning_effort":"low"}}'
  else
    payload='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"'"$prompt"'"}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"thinking":'"$thinking"',"reasoning_effort":"low"}}'
  fi
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
      -d "$payload" >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$out"
  else
    echo fail > "$out"
  fi
}

burst() { # $1 = arm name, $2 = concurrency, $3 = words-per-request, $4 = request profile
  local arm=$1 c=$2 words=$3 profile=${4:-bounded} i t0 t1
  # Pre-create every result file in the parent before forking so a subshell
  # that dies before writing still tallies as a failed outcome: the summary
  # can never claim n/n over fewer outcomes than requests scheduled.
  for i in $(seq 1 "$c"); do : > "$tmpdir/${arm}-${i}"; done
  t0=$(date +%s)
  for i in $(seq 1 "$c"); do
    fire "${arm}-${i}" "$words" true "$tmpdir/${arm}-${i}" "$profile" &
  done
  wait
  t1=$(date +%s)
  echo "  arm ${arm}: C=${c} x ~${words} tok, profile=${profile}, $((t1 - t0))s"
}

SAMPLER_KERNEL=_topk_topp_kernel

sampler_c3_arms() {
  # Re-dispatch each constexpr combo three times. This is reliability
  # redundancy, not a BATCH_SIZE compile-key axis.
  burst samp-k-c3  3 8 sampling-k
  burst samp-p-c3  3 8 sampling-p
  burst samp-kp-c3 3 8 sampling-kp
}

sampler_cache_combos() { # $1 = cache root; emits one line per distinct constexpr combo
  local root=$1 ttir kuse puse combo
  for ttir in "$root"/*/"$SAMPLER_KERNEL.ttir"; do
    [ -f "$ttir" ] || continue
    # TOPK_ENABLED/TOPP_ENABLED are constexprs, folded out of the TTIR
    # signature — observe them through argument USE instead: a disabled side
    # never touches its tensor, so its %K / %P symbol appears exactly once
    # (the declaration); an enabled side is referenced again in the body.
    # ([^A-Za-z0-9_] guards %P against matching %PERCENTILE_TO_STD_TABLE.)
    kuse=$(grep -oE '%K[^A-Za-z0-9_]' "$ttir" | wc -l)
    puse=$(grep -oE '%P[^A-Za-z0-9_]' "$ttir" | wc -l)
    if [ "$kuse" -gt 1 ] && [ "$puse" -gt 1 ]; then combo=k+p
    elif [ "$kuse" -gt 1 ]; then combo=k-only
    elif [ "$puse" -gt 1 ]; then combo=p-only
    else combo=neither; fi
    printf '%s\n' "$combo"
  done | sort -u
}

verify_sampler_cache() { # postcondition; returns 0 = met or skipped, 1 = unmet
  # Every TP rank executes the sampler. This host-side script can inspect only
  # the head rank's cache, so the local check is a dispatch-shape proxy, not a
  # worker-filesystem attestation; acceptance evidence must inspect both ranks.
  local root="${DSPARK_WARMUP_TRITON_CACHE_DIR:-}" combos combo n missing=""
  if [ -z "$root" ] || [ ! -d "$root" ]; then
    echo "  sampler-cache postcondition: SKIPPED (DSPARK_WARMUP_TRITON_CACHE_DIR unset or not a directory)"
    return 0
  fi
  combos=$(sampler_cache_combos "$root")
  for combo in k-only p-only k+p; do
    n=$(printf '%s\n' "$combos" | grep -cx "$combo")
    [ "$n" -ge 1 ] || missing="${missing} ${combo}:0/1"
  done
  if [ -z "$missing" ]; then
    echo "  sampler-cache postcondition: MET — ${SAMPLER_KERNEL} constexpr combos on this rank:"
    printf '%s\n' "$combos" | sed 's/^/    /'
    return 0
  fi
  echo "  sampler-cache postcondition: unmet —${missing} (constexpr combos)"
  return 1
}

mk_ladder_prompt() { # $1 = exact token count ('hello' + (N-1)x ' hello')
  local n=$1 out="hello" i
  for ((i = 1; i < n; i++)); do out="$out hello"; done
  printf '%s' "$out"
}

verify_ladder_rung() { # $1 = exact token count; tokenize-gated completion
  local s=$1 prompt want_block got resp t0 t1
  : > "$tmpdir/ladder-$s"       # pre-created: counted even on failure paths
  prompt=$(mk_ladder_prompt "$s")
  want_block=$(next_pow2 $((s + 6)))
  # Runtime gate: never trust the word-count heuristic against the served
  # tokenizer. Authenticated POST /tokenize must confirm exactly s tokens
  # before this rung's completion may fire.
  if ! resp=$("$CURL_BIN" -fsS --max-time 30 "${AUTH_ARGS[@]}" \
        "$BASE/tokenize" -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","prompt":"'"$prompt"'"}' \
        2>>"$tmpdir/errors"); then
    echo "boot-shape-warmup: tokenize verify FAILED for rung s=${s}: POST /tokenize errored — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/ladder-$s"
    return 0
  fi
  got=$(printf '%s\n' "$resp" | grep -o '"count"[[:space:]]*:[[:space:]]*[0-9]*' | head -n 1 | grep -o '[0-9]*$')
  if [ -z "$got" ]; then
    echo "boot-shape-warmup: tokenize verify FAILED for rung s=${s}: no usable \"count\" in /tokenize response — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/ladder-$s"
    return 0
  fi
  if [ "$got" -ne "$s" ]; then
    echo "boot-shape-warmup: tokenize verify FAILED for rung s=${s}: /tokenize reported ${got} tokens, need exactly ${s} — rung skipped, BLOCK ${want_block} NOT warmed" >&2
    echo fail > "$tmpdir/ladder-$s"
    return 0
  fi
  t0=$(date +%s)
  if "$CURL_BIN" -fsS --max-time "$REQ_TIMEOUT" "${AUTH_ARGS[@]}" \
      "$BASE/v1/completions" -H "Content-Type: application/json" \
      -d '{"model":"'"$MODEL"'","prompt":"'"$prompt"'","max_tokens":1,"temperature":0}' \
      >/dev/null 2>>"$tmpdir/errors"; then
    echo ok > "$tmpdir/ladder-$s"
    t1=$(date +%s)
    echo "  ladder s=${s}: tokenize ${got}/${s} -> BLOCK ${want_block} fired ($((t1 - t0))s)"
  else
    echo fail > "$tmpdir/ladder-$s"
    echo "  ladder s=${s}: tokenize ${got}/${s} -> BLOCK ${want_block} request FAILED"
  fi
}

ladder() {
  local s
  for s in "${LADDER_S[@]}"; do
    verify_ladder_rung "$s"
  done
}

if ! "$CURL_BIN" -fsS --max-time 10 "${AUTH_ARGS[@]}" "$BASE/v1/models" >/dev/null 2>&1; then
  echo "boot-shape-warmup: API not reachable at $BASE — skipping sweep" >&2
  exit 1
fi

echo "boot-shape-warmup: sweeping spec-decode/prefill shape buckets (issue #117)"
total_t0=$(date +%s)

# Kernel-critical first: deterministic bucket ladder (exact-token plain
# completions), then the batch/chat arms.
ladder

EXPECTED_CHAT_REQUESTS=8        # c1 + short-c1 + samp-k + samp-p + samp-kp + mid + longchunk + nothink
burst c1        1 300
burst short-c1  1 8 serve-default
# _topk_topp_kernel constexpr combos: k-only, p-only, k+p.
burst samp-k    1 8 sampling-k
burst samp-p    1 8 sampling-p
burst samp-kp   1 8 sampling-kp
if [ "$MAX_CONCURRENCY" -ge 2 ]; then burst c2 2 420; burst short-c2 2 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 4)); fi
if [ "$MAX_CONCURRENCY" -ge 3 ]; then
  sampler_c3_arms
  EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 9))
fi
if [ "$MAX_CONCURRENCY" -ge 4 ]; then burst c4 4 380; burst short-c4 4 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 8)); fi
if [ "$MAX_CONCURRENCY" -ge 6 ]; then burst c6 6 340; burst short-c6 6 8 serve-default; EXPECTED_CHAT_REQUESTS=$((EXPECTED_CHAT_REQUESTS + 12)); fi
if [ "$MAX_CONCURRENCY" -gt 6 ]; then
  echo "boot-shape-warmup: WARN: MAX_NUM_SEQS=${MAX_CONCURRENCY}; batch shapes above C=6 are not pre-warmed" >&2
fi
burst mid       1 2600
burst longchunk 1 9500          # crosses the 8192-token chunk boundary (BLOCK 256); its tail does NOT reach low buckets
t0=$(date +%s)
: > "$tmpdir/nothink-1"          # pre-created: counted even on subshell death
fire nothink-1 300 false "$tmpdir/nothink-1"
t1=$(date +%s)
echo "  arm nothink: C=1 x ~300 tok, thinking=false, $((t1 - t0))s"

# Observable postcondition for the sampler arms: the cache must actually hold
# each _topk_topp_kernel constexpr combo. Request success alone cannot prove
# that every sampling branch dispatched.
SAMPLER_POSTCOND=ok
verify_sampler_cache || SAMPLER_POSTCOND=fail

total=0 ok_count=0
for f in "$tmpdir"/*-*; do
  [ -f "$f" ] || continue
  total=$((total + 1))
  [ "$(cat "$f")" = "ok" ] && ok_count=$((ok_count + 1))
done
EXPECTED_REQUESTS=$(( ${#LADDER_S[@]} + EXPECTED_CHAT_REQUESTS ))
if [ "$total" -ne "$EXPECTED_REQUESTS" ]; then
  echo "boot-shape-warmup: internal error: tallied $total outcomes for $EXPECTED_REQUESTS scheduled requests" >&2
  exit 1
fi
total_t1=$(date +%s)
echo "boot-shape-warmup: ${ok_count}/${total} requests ok in $((total_t1 - total_t0))s"

if [ "$ok_count" -lt "$total" ]; then
  echo "boot-shape-warmup: $((total - ok_count)) request(s) failed — uncovered shapes may JIT mid-serve" >&2
  sed -n '1,5p' "$tmpdir/errors" >&2 2>/dev/null || true
  exit 1
fi
if [ "$SAMPLER_POSTCOND" != ok ]; then
  echo "boot-shape-warmup: sampler-cache postcondition UNMET — ${SAMPLER_KERNEL} variants may JIT mid-serve" >&2
  exit 1
fi
exit 0
