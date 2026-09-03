#!/usr/bin/env bash
# test-boot-shape-warmup.sh — CPU behavioral tests for scripts/boot-shape-warmup.sh.
#
# Uses the WARMUP_CURL seam to substitute a recording curl stub; every
# assertion is behavioral (exit codes, recorded request bodies/headers across
# /v1/models, /tokenize, /v1/completions and /v1/chat/completions, outcome
# tallies, stdout bucket-ladder lines). No GPU, no network.
# Run: bash scripts/test-boot-shape-warmup.sh [-q]
set -u

QUIET="${1:-}"
here="$(cd "$(dirname "$0")" && pwd)"
target="$here/boot-shape-warmup.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

pass=0 fail=0
ok()  { pass=$((pass + 1)); [ "$QUIET" = "-q" ] || printf '  ok  %s\n' "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Recording curl stub.
# STUB_MODE: ok | chatfail | probefail | killparent | tokfail45 | badcount45 |
#            nocount45.
cat > "$work/curl-stub" <<'STUB'
#!/usr/bin/env bash
rec="${STUB_REC:?}"
n=$(date +%s%N)-$$-$RANDOM
is_probe=0; is_chat=0; is_tok=0; is_comp=0; body=""; hdrs=""
prev=""
for a in "$@"; do
  case "$a" in
    */v1/models*) is_probe=1 ;;
    */v1/chat/completions*) is_chat=1 ;;
    */tokenize*) is_tok=1 ;;
    */v1/completions*) is_comp=1 ;;
  esac
  [ "$prev" = "-d" ] && body="$a"
  [ "$prev" = "-H" ] && hdrs="${hdrs}${a}"$'\n'
  prev="$a"
done
if [ "$is_probe" = 1 ]; then
  [ "${STUB_MODE:-ok}" = "probefail" ] && exit 22
  printf '%s\n' "$hdrs" > "$rec/probe-$n"
  exit 0
fi
if [ "$is_tok" = 1 ]; then
  printf '%s%s\n' "$hdrs" "$body" > "$rec/tok-$n"
  p=$(printf '%s' "$body" | sed -n 's/.*"prompt":"\([^"]*\)".*/\1/p')
  cnt=$(printf '%s' "$p" | wc -w | tr -d ' ')
  [ "${STUB_MODE:-ok}" = "tokfail45" ] && [ "$cnt" = 45 ] && exit 22
  [ "${STUB_MODE:-ok}" = "badcount45" ] && [ "$cnt" = 45 ] && cnt=$((cnt + 1))
  [ "${STUB_MODE:-ok}" = "nocount45" ] && [ "$cnt" = 45 ] && { printf '{"tokens":[]}\n'; exit 0; }
  printf '{"tokens":[],"count":%s}\n' "$cnt"
  exit 0
fi
if [ "$is_comp" = 1 ]; then
  printf '%s%s\n' "$hdrs" "$body" > "$rec/comp-$n"
  exit 0
fi
# killparent mode: the stub's PPID is the fire() subshell running this curl;
# killing it simulates a request whose outcome is never written, leaving the
# pre-created result file empty for the tally.
case "$body" in *" c4-3]"*)
  [ "${STUB_MODE:-ok}" = "killparent" ] && { kill -9 "$PPID" 2>/dev/null; exit 137; } ;;
esac
printf '%s%s\n' "$hdrs" "$body" > "$rec/chat-$n"
[ "${STUB_MODE:-ok}" = "chatfail" ] && exit 22
exit 0
STUB
chmod +x "$work/curl-stub"

run_sweep() { # $1 = mode, $2 = record dir, rest: extra KEY=VALUE env; returns script exit code
  local mode=$1 dir=$2
  shift 2
  mkdir -p "$dir"
  env STUB_MODE="$mode" STUB_REC="$dir" WARMUP_CURL="$work/curl-stub" "$@" \
    bash "$target" http://stub:0 test-model > "$dir/stdout" 2> "$dir/stderr"
}

comp_prompt() { # $1 = recorded comp file -> echoed plain prompt text
  sed -n 's/.*"prompt":"\([^"]*\)".*/\1/p' "$1"
}

file_ts() { # $1 = recorded file path -> leading nanosecond timestamp of its name
  local b=${1##*/}
  b=${b#*-}          # strip probe-/chat-/tok-/comp- prefix
  printf '%s' "${b%%-*}"
}

# ---- 1. shipped-default run: 41 chats + 6 completions + 6 tokenizes ----------
rec="$work/r1"
if run_sweep ok "$rec"; then ok "all-success sweep exits 0"; else bad "all-success sweep exited nonzero"; fi

chat_n=$(ls "$rec"/chat-* 2>/dev/null | wc -l)
[ "$chat_n" -eq 41 ] && ok "41 chat requests fired" || bad "expected 41 chat requests, got $chat_n"
comp_n=$(ls "$rec"/comp-* 2>/dev/null | wc -l)
[ "$comp_n" -eq 6 ] && ok "6 ladder completions fired" || bad "expected 6 ladder completions, got $comp_n"
tok_n=$(ls "$rec"/tok-* 2>/dev/null | wc -l)
[ "$tok_n" -eq 6 ] && ok "6 tokenize verifications performed" || bad "expected 6 tokenize calls, got $tok_n"

for tag in c1-1 c2-1 c2-2 c4-1 c4-2 c4-3 c4-4 c6-1 c6-2 c6-3 c6-4 c6-5 c6-6 mid-1 longchunk-1 short-c1-1 short-c2-1 short-c2-2 short-c4-1 short-c4-2 short-c4-3 short-c4-4 short-c6-1 short-c6-2 short-c6-3 short-c6-4 short-c6-5 short-c6-6 samp-k-1 samp-p-1 samp-kp-1 samp-k-c3-1 samp-k-c3-2 samp-k-c3-3 samp-p-c3-1 samp-p-c3-2 samp-p-c3-3 samp-kp-c3-1 samp-kp-c3-2 samp-kp-c3-3; do
  grep -lq "warmup .* ${tag}\]" "$rec"/chat-* || bad "arm ${tag} missing from recorded bodies"
done
grep -lq 'warmup .* nothink-1\]' "$rec"/chat-* && ok "all 41 arm tags present" \
  || bad "nothink arm missing"
grep -q "sampler-cache postcondition: SKIPPED" "$rec/stdout" \
  && ok "postcondition skips with a note when no cache dir is provided" \
  || bad "missing SKIPPED note for unset DSPARK_WARMUP_TRITON_CACHE_DIR"

# ---- 2. thinking profiles: one explicit off; short arms use serve defaults ----
tf=$(grep -l '"thinking":false' "$rec"/chat-* | wc -l)
[ "$tf" -eq 1 ] && ok "exactly one explicit thinking:false arm" || bad "thinking:false arms: $tf (want 1)"
tt=$(grep -l '"thinking":true' "$rec"/chat-* | wc -l)
[ "$tt" -eq 27 ] && ok "27 bounded/sampling thinking:true arms" || bad "thinking:true arms: $tt (want 27)"

# Sampling arms: exact top_k/top_p placement per combo, temperature 0.8, and
# no leak of k/p into any non-sampling arm.
k_only=$(grep -l '"top_k":40' "$rec"/chat-* | xargs -r grep -L '"top_p"' | wc -l)
[ "$k_only" -eq 4 ] && ok "four k-only sampling requests" \
  || bad "k-only sampling requests: $k_only (want 4)"
p_only=$(grep -l '"top_p":0.9' "$rec"/chat-* | xargs -r grep -L '"top_k"' | wc -l)
[ "$p_only" -eq 4 ] && ok "four p-only sampling requests" \
  || bad "p-only sampling requests: $p_only (want 4)"
kp_both=$(grep -l '"top_k":40,"top_p":0.9' "$rec"/chat-* | wc -l)
[ "$kp_both" -eq 4 ] && ok "four k+p sampling requests" \
  || bad "k+p sampling requests: $kp_both (want 4)"
samp_temp_bad=0
for f in $(grep -l 'warmup .* samp-' "$rec"/chat-*); do
  grep -q '"temperature":0.8' "$f" || samp_temp_bad=$((samp_temp_bad + 1))
done
[ "$samp_temp_bad" -eq 0 ] && ok "every sampling arm uses temperature 0.8 (greedy would drop k/p tensors)" \
  || bad "$samp_temp_bad sampling request(s) not at temperature 0.8"
leak=0
for f in "$rec"/chat-*; do
  grep -q 'warmup .* samp-' "$f" && continue
  grep -qE '"top_k"|"top_p"' "$f" && leak=$((leak + 1))
done
[ "$leak" -eq 0 ] && ok "top_k/top_p never leak into non-sampling arms" \
  || bad "$leak non-sampling request(s) carry top_k/top_p"
serve_default=0
for f in "$rec"/chat-*; do
  grep -q 'warmup .* short-c' "$f" || continue
  serve_default=$((serve_default + 1))
  grep -q '"max_tokens"' "$f" && bad "serve-default arm unexpectedly pins max_tokens"
  grep -q '"chat_template_kwargs"' "$f" && bad "serve-default arm unexpectedly pins chat template"
done
[ "$serve_default" -eq 13 ] && ok "13 short arms mirror ordinary client defaults" \
  || bad "serve-default short arms: $serve_default (want 13)"

# ---- 3. long arm actually crosses the 8192 chunk boundary --------------------
longlen=$(wc -c < "$(grep -l 'longchunk-1\]' "$rec"/chat-*)")
[ "$longlen" -gt 40000 ] && ok "longchunk body is ${longlen} bytes (> one 8192-token chunk)" \
  || bad "longchunk body only ${longlen} bytes"

# ---- 4. prefix-cache busting: per-run nonce differs, per-request tag differs -
rec2="$work/r2"
run_sweep ok "$rec2" >/dev/null 2>&1 || true
n1=$(grep -oh 'warmup [^ ]*' "$rec"/chat-*  | sort -u | head -1)
n2=$(grep -oh 'warmup [^ ]*' "$rec2"/chat-* | sort -u | head -1)
[ -n "$n1" ] && [ "$n1" != "$n2" ] && ok "nonce differs across runs ($n1 vs $n2)" \
  || bad "nonce identical across runs — prefix cache would skip the warmed prefill"
uniq_tags=$(grep -oh 'warmup [^]]*\]' "$rec"/chat-* | sort -u | wc -l)
[ "$uniq_tags" -eq 41 ] && ok "41 distinct chat tags within a run" || bad "distinct chat tags: $uniq_tags (want 41)"

# ---- 5. bucket ladder: exact declared token counts, all six live BLOCK keys --
counts=$(for f in "$rec"/comp-*; do comp_prompt "$f" | wc -w; done | sort -un | tr '\n' ' ')
[ "$counts" = "1 6 20 45 100 200 " ] \
  && ok "completions requested at exactly the six declared token counts" \
  || bad "completion prompt counts wrong: [$counts] want [1 6 20 45 100 200]"
for b in 8 16 32 64 128 256; do
  grep -q "BLOCK ${b} fired" "$rec/stdout" \
    && ok "BLOCK ${b} rung mapped and fired" \
    || bad "no stdout evidence that BLOCK ${b} was mapped/fired"
done
for tag in c6-1 c6-2 c6-3 c6-4 c6-5 c6-6 short-c6-1 short-c6-2 short-c6-3 short-c6-4 short-c6-5 short-c6-6; do
  grep -lq "warmup .* ${tag}\]" "$rec"/chat-* || bad "shipped MAX_NUM_SEQS=6 arm ${tag} missing"
done
ok "C=6 bounded and serve-default arms run at shipped MAX_NUM_SEQS=6"

rec_cap4="$work/r-cap4"
run_sweep ok "$rec_cap4" DSPARK_WARMUP_MAX_CONCURRENCY=4 >/dev/null 2>&1 \
  || bad "MAX_NUM_SEQS=4 profile sweep exited nonzero"
cap4_chat_n=$(ls "$rec_cap4"/chat-* 2>/dev/null | wc -l)
[ "$cap4_chat_n" -eq 29 ] && ok "MAX_NUM_SEQS=4 profile fires 29 chat requests" \
  || bad "MAX_NUM_SEQS=4 profile fired $cap4_chat_n chat requests (want 29)"
if grep -rqE 'arm (short-)?c6| (short-)?c6-[0-9]' "$rec_cap4" 2>/dev/null; then
  bad "C=6 arm fired above the MAX_NUM_SEQS=4 profile"
else
  ok "C=6 bounded and serve-default arms are skipped at MAX_NUM_SEQS=4"
fi
grep -q "^boot-shape-warmup: 35/35 requests ok" "$rec_cap4/stdout" \
  && ok "MAX_NUM_SEQS=4 profile tallies exactly 35/35" \
  || bad "MAX_NUM_SEQS=4 summary wrong: $(grep 'requests ok' "$rec_cap4/stdout" || true)"

rec_cap2="$work/r-cap2"
run_sweep ok "$rec_cap2" DSPARK_WARMUP_MAX_CONCURRENCY=2 >/dev/null 2>&1 \
  || bad "MAX_NUM_SEQS=2 profile sweep exited nonzero"
cap2_chat_n=$(ls "$rec_cap2"/chat-* 2>/dev/null | wc -l)
[ "$cap2_chat_n" -eq 12 ] && ok "MAX_NUM_SEQS=2 profile fires 12 chat requests" \
  || bad "MAX_NUM_SEQS=2 profile fired $cap2_chat_n chat requests (want 12)"
if grep -rq 'samp-k-c3\|samp-p-c3\|samp-kp-c3' "$rec_cap2" 2>/dev/null; then
  bad "C=3 sampler redundancy fired above the MAX_NUM_SEQS=2 profile"
else
  ok "C=3 sampler redundancy is skipped at MAX_NUM_SEQS=2"
fi
grep -q "^boot-shape-warmup: 18/18 requests ok" "$rec_cap2/stdout" \
  && ok "MAX_NUM_SEQS=2 profile tallies exactly 18/18" \
  || bad "MAX_NUM_SEQS=2 summary wrong: $(grep 'requests ok' "$rec_cap2/stdout" || true)"

# ---- 6. tokenize gate: verified, authenticated, BEFORE its completion --------
for s in 1 6 20 45 100 200; do
  p="hello"; i=1
  while [ "$i" -lt "$s" ]; do p="$p hello"; i=$((i + 1)); done
  tfile=$(grep -lF "\"prompt\":\"$p\"" "$rec"/tok-* 2>/dev/null | head -1)
  cfile=$(grep -lF "\"prompt\":\"$p\"" "$rec"/comp-* 2>/dev/null | head -1)
  if [ -n "$tfile" ] && [ -n "$cfile" ] && [ "$(file_ts "$tfile")" -lt "$(file_ts "$cfile")" ]; then
    ok "s=${s}: /tokenize verified before its completion fired"
  else
    bad "s=${s}: tokenize-before-completion ordering violated or records missing"
  fi
done

# ---- 7. auth propagation: launcher bearer preferred over ambient VLLM_API_KEY -
rec5="$work/r5"
run_sweep ok "$rec5" DSPARK_WARMUP_BEARER=warmup-first-key VLLM_API_KEY=ambient-vllm-key >/dev/null 2>&1 \
  || bad "bearer-preference sweep exited nonzero"
missing_auth=0 req_n=0
for f in "$rec5"/chat-* "$rec5"/comp-* "$rec5"/tok-* "$rec5"/probe-*; do
  [ -f "$f" ] || continue
  req_n=$((req_n + 1))
  [ "$(head -n 1 "$f")" = "Authorization: Bearer warmup-first-key" ] || missing_auth=$((missing_auth + 1))
done
[ "$req_n" -eq 54 ] && ok "probe + 41 chats + 6 tokenizes + 6 completions recorded" \
  || bad "expected 54 recorded requests, got $req_n"
[ "$missing_auth" -eq 0 ] && ok "every request (incl. /tokenize gate) carries the launcher-provided bearer" \
  || bad "$missing_auth request(s) missing/mismatching Authorization header"
grep -rq "ambient-vllm-key" "$rec5" && bad "ambient VLLM_API_KEY value leaked into requests or logs" \
  || ok "ambient VLLM_API_KEY never sent nor logged"

# ---- 8. VLLM_API_KEY-only fallback preserved; open cluster sends no header ----
rec6="$work/r6"
run_sweep ok "$rec6" VLLM_API_KEY=solo-vllm-key >/dev/null 2>&1 || true
solo_bad=0
for f in "$rec6"/chat-* "$rec6"/comp-* "$rec6"/tok-*; do
  [ -f "$f" ] || continue
  [ "$(head -n 1 "$f")" = "Authorization: Bearer solo-vllm-key" ] || solo_bad=$((solo_bad + 1))
done
[ "$solo_bad" -eq 0 ] && ok "VLLM_API_KEY fallback authenticates every chat/ladder request" \
  || bad "$solo_bad requests unauthenticated under fallback"
rec7="$work/r7"
run_sweep ok "$rec7" DSPARK_WARMUP_BEARER= VLLM_API_KEY= >/dev/null 2>&1 || true
grep -rq "Authorization:" "$rec7" && bad "open cluster sent an Authorization header" \
  || ok "no keys configured -> no Authorization header anywhere"

# ---- 9. chat failures: exit 1, precise failure count on stderr ----------------
rec3="$work/r3"
if run_sweep chatfail "$rec3"; then bad "chatfail sweep exited 0"; else ok "chatfail sweep exits nonzero"; fi
grep -q "41 request(s) failed" "$rec3/stderr" && ok "failure count named on stderr" \
  || bad "expected '41 request(s) failed' on stderr"

# ---- 10. unreachable API: exit 1 fast, no requests fired ----------------------
rec4="$work/r4"
if run_sweep probefail "$rec4"; then bad "probefail sweep exited 0"; else ok "probefail sweep exits nonzero"; fi
post_probe=$(ls "$rec4"/chat-* "$rec4"/comp-* "$rec4"/tok-* 2>/dev/null | wc -l)
[ "$post_probe" -eq 0 ] && ok "nothing fired after failed probe" \
  || bad "$post_probe request(s) fired after failed probe"

# ---- 11. subshell dying before writing still tallies: 46/47, never n/n short ---
rec8="$work/r8"
if run_sweep killparent "$rec8"; then bad "killparent sweep exited 0"; else ok "lost outcome fails the sweep"; fi
grep -q "^boot-shape-warmup: 46/47 requests ok" "$rec8/stdout" \
  && ok "request whose subshell died before writing counted as failed" \
  || bad "summary not 46/47: $(grep 'requests ok' "$rec8/stdout" || true)"
grep -q "1 request(s) failed" "$rec8/stderr" || bad "failure count missing on stderr"

# ---- 12. tokenize endpoint error: rung skipped, counted, precise diagnostic ----
rec9="$work/r9"
if run_sweep tokfail45 "$rec9"; then bad "tokfail45 sweep exited 0"; else ok "tokenize-error rung fails the sweep"; fi
grep -q "rung s=45: POST /tokenize errored" "$rec9/stderr" \
  && ok "tokenize transport failure diagnosed precisely (rung s=45)" \
  || bad "missing precise diagnostic for tokenize transport failure"
tf_comp=$(ls "$rec9"/comp-* 2>/dev/null | wc -l)
[ "$tf_comp" -eq 5 ] && ok "failed-gate rung did NOT fire its completion" \
  || bad "expected 5 completions after skipped rung, got $tf_comp"
grep -q "^boot-shape-warmup: 46/47 requests ok" "$rec9/stdout" \
  && ok "skipped rung still counted (46/47)" \
  || bad "summary not 46/47 after tokenize transport failure"

# ---- 13. tokenize count mismatch: rung skipped, counted, secret-free diag ------
rec10="$work/r10"
if run_sweep badcount45 "$rec10"; then bad "badcount45 sweep exited 0"; else ok "count-mismatch rung fails the sweep"; fi
grep -q "/tokenize reported 46 tokens, need exactly 45" "$rec10/stderr" \
  && ok "count mismatch diagnosed precisely (46 vs 45)" \
  || bad "missing precise mismatch diagnostic"
bc_comp=$(ls "$rec10"/comp-* 2>/dev/null | wc -l)
[ "$bc_comp" -eq 5 ] && ok "mismatched rung did NOT fire its completion" \
  || bad "expected 5 completions after mismatched rung, got $bc_comp"
grep -q "^boot-shape-warmup: 46/47 requests ok" "$rec10/stdout" \
  && ok "mismatched rung still counted (46/47)" \
  || bad "summary not 46/47 after count mismatch"

# ---- 14. tokenize response without count: rung skipped and diagnosed ----------
rec11="$work/r11"
if run_sweep nocount45 "$rec11"; then bad "nocount45 sweep exited 0"; else ok "missing-count rung fails the sweep"; fi
grep -q 'rung s=45: no usable "count"' "$rec11/stderr" \
  && ok "missing tokenize count diagnosed precisely (rung s=45)" \
  || bad "missing precise diagnostic for absent tokenize count"
nc_comp=$(ls "$rec11"/comp-* 2>/dev/null | wc -l)
[ "$nc_comp" -eq 5 ] && ok "missing-count rung did NOT fire its completion" \
  || bad "expected 5 completions after missing-count rung, got $nc_comp"
grep -q "^boot-shape-warmup: 46/47 requests ok" "$rec11/stdout" \
  && ok "missing-count rung still counted (46/47)" \
  || bad "summary not 46/47 after absent tokenize count"

# ---- 15+. sampler-cache postcondition against fixture TTIR caches -------------
# Fixture entries reproduce the observable constexpr axes of the real
# _topk_topp_kernel cache. TOPK_ENABLED/TOPP_ENABLED show up only through
# argument USE (%K/%P referenced beyond their declaration).
# %PERCENTILE_TO_STD_TABLE guards against a naive %P substring match.
mk_fixture_ttir() { # $1 = cache root, $2 = entry dir, $3 = k-used 0/1, $4 = p-used 0/1
  local d="$1/$2"
  mkdir -p "$d"
  {
    printf 'tt.func public @_topk_topp_kernel(%%LOGITS: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %%PERCENTILE_TO_STD_TABLE: !tt.ptr<f32>, %%K: !tt.ptr<f32>, %%P: !tt.ptr<f32>, %%BATCH_SIZE: i32 loc("BATCH_SIZE")) {\n'
    printf '  %%9 = tt.load %%PERCENTILE_TO_STD_TABLE : !tt.ptr<f32>\n'
    [ "$3" = 1 ] && printf '  %%10 = tt.load %%K : !tt.ptr<f32>\n'
    [ "$4" = 1 ] && printf '  %%11 = tt.load %%P : !tt.ptr<f32>\n'
    printf '  tt.return\n}\n'
  } > "$d/_topk_topp_kernel.ttir"
}
seed_combos() {
  mk_fixture_ttir "$1" K  1 0
  mk_fixture_ttir "$1" P  0 1
  mk_fixture_ttir "$1" KP 1 1
}

# 15. all three constexpr combos present -> postcondition MET, exit 0
rec12="$work/r12"; cache12="$work/cache12"
seed_combos "$cache12"
if run_sweep ok "$rec12" DSPARK_WARMUP_TRITON_CACHE_DIR="$cache12"; then
  ok "postcondition MET with all three constexpr combos -> exit 0"
else
  bad "sweep exited nonzero despite complete sampler-cache combos"
fi
grep -q "sampler-cache postcondition: MET" "$rec12/stdout" || bad "missing MET line"
met_combos=$(sed -n '/postcondition: MET/,/^[^ ]/p' "$rec12/stdout" | grep -cE '^    (k-only|p-only|k\+p)$')
[ "$met_combos" -eq 3 ] && ok "all three observed combos listed on stdout" \
  || bad "expected 3 listed combos, got $met_combos"

# 16. one constexpr combo absent -> precise UNMET, requests still tally exactly
rec13="$work/r13"; cache13="$work/cache13"
mk_fixture_ttir "$cache13" K 1 0
mk_fixture_ttir "$cache13" KP 1 1
if run_sweep ok "$rec13" DSPARK_WARMUP_TRITON_CACHE_DIR="$cache13"; then
  bad "sweep exited 0 with p-only combo missing"
else
  ok "unmet postcondition fails the nonfatal sweep"
fi
grep -q "p-only:0/1" "$rec13/stdout" \
  && ok "missing combo diagnosed precisely" || bad "missing p-only diagnostic"
grep -q "sampler-cache postcondition UNMET" "$rec13/stderr" \
  && ok "UNMET diagnosed on stderr" || bad "missing UNMET diagnostic on stderr"
grep -q "^boot-shape-warmup: 47/47 requests ok" "$rec13/stdout" \
  && ok "postcondition failure leaves request tally exact (47/47)" \
  || bad "request tally drifted: $(grep 'requests ok' "$rec13/stdout" || true)"

# 17. concurrency profile does not alter the sampler compile-key requirement
rec14="$work/r14"; cache14="$work/cache14"
seed_combos "$cache14"
if run_sweep ok "$rec14" DSPARK_WARMUP_TRITON_CACHE_DIR="$cache14" DSPARK_WARMUP_MAX_CONCURRENCY=2; then
  ok "MAX_NUM_SEQS=2: the same three combos satisfy the postcondition"
else
  bad "MAX_NUM_SEQS=2 sweep exited nonzero with complete combos"
fi
grep -q "sampler-cache postcondition: MET" "$rec14/stdout" || bad "missing MET line at C=2"

printf 'test-boot-shape-warmup: %d ok, %d failed\n' "$pass" "$fail"
exit "$((fail > 0 ? 1 : 0))"
