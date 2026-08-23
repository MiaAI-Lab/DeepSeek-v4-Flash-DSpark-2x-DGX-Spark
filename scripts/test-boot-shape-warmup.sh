#!/usr/bin/env bash
# test-boot-shape-warmup.sh — CPU behavioral tests for scripts/boot-shape-warmup.sh.
#
# Uses the WARMUP_CURL seam to substitute a recording stub for curl; every
# assertion is behavioral (exit codes, recorded request bodies). No GPU, no
# network. Run: bash scripts/test-boot-shape-warmup.sh [-q]
set -u

QUIET="${1:-}"
here="$(cd "$(dirname "$0")" && pwd)"
target="$here/boot-shape-warmup.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

pass=0 fail=0
ok()  { pass=$((pass + 1)); [ "$QUIET" = "-q" ] || printf '  ok  %s\n' "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Recording curl stub. STUB_MODE: ok | chatfail | probefail.
cat > "$work/curl-stub" <<'STUB'
#!/usr/bin/env bash
rec="${STUB_REC:?}"
n=$(date +%s%N)-$$-$RANDOM
is_probe=0; body=""
prev=""
for a in "$@"; do
  case "$a" in */v1/models*) is_probe=1 ;; esac
  [ "$prev" = "-d" ] && body="$a"
  prev="$a"
done
if [ "$is_probe" = 1 ]; then
  [ "${STUB_MODE:-ok}" = "probefail" ] && exit 22
  exit 0
fi
printf '%s\n' "$body" > "$rec/chat-$n"
[ "${STUB_MODE:-ok}" = "chatfail" ] && exit 22
exit 0
STUB
chmod +x "$work/curl-stub"

run_sweep() { # $1 = mode, $2 = record dir; returns script exit code
  mkdir -p "$2"
  STUB_MODE="$1" STUB_REC="$2" WARMUP_CURL="$work/curl-stub" \
    bash "$target" http://stub:0 test-model > "$2/stdout" 2> "$2/stderr"
}

# ---- 1. all-success run: exit 0, 16 chat requests, every arm present -------
rec="$work/r1"
if run_sweep ok "$rec"; then ok "all-success sweep exits 0"; else bad "all-success sweep exited nonzero"; fi

count=$(ls "$rec"/chat-* 2>/dev/null | wc -l)
[ "$count" -eq 16 ] && ok "16 chat requests fired" || bad "expected 16 chat requests, got $count"

for tag in c1-1 c2-1 c2-2 c4-1 c4-2 c4-3 c4-4 c6-1 c6-2 c6-3 c6-4 c6-5 c6-6 mid-1 longchunk-1; do
  grep -lq "warmup .* ${tag}\]" "$rec"/chat-* || bad "arm ${tag} missing from recorded bodies"
done
grep -lq 'warmup .* nothink-1\]' "$rec"/chat-* && ok "all 16 arm tags present" \
  || bad "nothink arm missing"

# ---- 2. thinking-off arm is exactly one --------------------------------------
tf=$(grep -l '"thinking":false' "$rec"/chat-* | wc -l)
[ "$tf" -eq 1 ] && ok "exactly one thinking:false arm" || bad "thinking:false arms: $tf (want 1)"
tt=$(grep -l '"thinking":true' "$rec"/chat-* | wc -l)
[ "$tt" -eq 15 ] && ok "15 thinking:true arms" || bad "thinking:true arms: $tt (want 15)"

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
[ "$uniq_tags" -eq 16 ] && ok "16 distinct request tags within a run" || bad "distinct tags: $uniq_tags (want 16)"

# ---- 5. chat failures: exit 1, failure named on stderr -----------------------
rec3="$work/r3"
if run_sweep chatfail "$rec3"; then bad "chatfail sweep exited 0"; else ok "chatfail sweep exits nonzero"; fi
grep -q "failed" "$rec3/stderr" && ok "failure named on stderr" || bad "no failure message on stderr"

# ---- 6. unreachable API: exit 1 fast, no chat requests fired -----------------
rec4="$work/r4"
if run_sweep probefail "$rec4"; then bad "probefail sweep exited 0"; else ok "probefail sweep exits nonzero"; fi
c4count=$(ls "$rec4"/chat-* 2>/dev/null | wc -l)
[ "$c4count" -eq 0 ] && ok "no chat requests after failed probe" || bad "$c4count chat requests fired after failed probe"

printf 'test-boot-shape-warmup: %d ok, %d failed\n' "$pass" "$fail"
exit "$((fail > 0 ? 1 : 0))"
