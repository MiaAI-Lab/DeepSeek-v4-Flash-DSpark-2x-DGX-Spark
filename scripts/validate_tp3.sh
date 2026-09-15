#!/usr/bin/env bash
# Validate a TP=3 DeepSeek-V4-Flash deployment.
#
# "It started" is NOT evidence of correctness. A wrong o_groups shard produces
# fluent, plausible text with no error anywhere -- that is the entire hazard
# this patch exists to avoid. So this checks reasoning that a subtly-broken
# attention would fail, not just that tokens come out.
#
# Usage: validate_tp3.sh [host:port]        (default 127.0.0.1:8888)
#
set -uo pipefail

EP="${1:-127.0.0.1:8888}"
BASE="http://$EP"
pass=0; fail=0

ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
info() { printf '  ---- %s\n' "$1"; }

echo "=== 1. server up + model identity ==="
models="$(curl -s -m 20 "$BASE/v1/models" 2>/dev/null)"
if [[ -z "$models" ]]; then
  bad "no response from $BASE/v1/models"
  echo; echo "Server is not answering. Nothing else can be validated."; exit 1
fi
mid="$(printf '%s' "$models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
[[ -n "$mid" ]] && ok "serving: $mid" || bad "could not parse model id"

ask() {  # $1=prompt  $2=max_tokens
  curl -s -m 180 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],
                  "max_tokens":int(sys.argv[3]),"temperature":0,
                  "chat_template_kwargs":{"thinking":False}}))' "$mid" "$1" "${2:-64}")" \
  | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    m=d["choices"][0]["message"]
    t=(m.get("content") or m.get("reasoning_content") or m.get("reasoning") or "")
    print(t.strip())
except Exception as e:
    print(f"__ERROR__ {e}")' 2>/dev/null
}

echo
echo "=== 2. deterministic factual recall ==="
# Degenerate attention usually survives token-level fluency but loses precise
# recall first. These have exactly one right answer.
for probe in "What is the capital of Japan?|Tokyo" \
             "What is 17 multiplied by 23? Reply with only the number.|391" \
             "Complete exactly: The quick brown fox jumps over the lazy|dog"; do
  q="${probe%%|*}"; want="${probe##*|}"
  a="$(ask "$q" 32)"
  if [[ "$a" == __ERROR__* ]]; then bad "request failed: $q ($a)"
  elif grep -qi -- "$want" <<<"$a"; then ok "$q -> $(head -c 60 <<<"$a")"
  else bad "$q -> expected '$want', got: $(head -c 90 <<<"$a")"; fi
done

echo
echo "=== 3. multi-step reasoning (fails first under a bad shard) ==="
a="$(ask 'A shelf holds 3 red books and 5 blue books. I remove 2 blue books, then add 4 red books. How many red and how many blue remain? Answer in the form: red=N blue=N' 96)"
if [[ "$a" == __ERROR__* ]]; then bad "reasoning request failed ($a)"
elif grep -qE 'red=7' <<<"$a" && grep -qE 'blue=3' <<<"$a"; then ok "arithmetic reasoning: $(head -c 70 <<<"$a")"
else bad "expected red=7 blue=3, got: $(head -c 120 <<<"$a")"; fi

echo
echo "=== 4. long-range coherence (attention over distance) ==="
needle="The passphrase is CRIMSON-MERIDIAN-42."
filler="$(python3 -c 'print(" ".join(["The weather report for that day was unremarkable."]*120))')"
a="$(ask "$needle $filler What exactly is the passphrase? Reply with only the passphrase." 32)"
if [[ "$a" == __ERROR__* ]]; then bad "needle test failed ($a)"
elif grep -qi 'CRIMSON-MERIDIAN-42' <<<"$a"; then ok "recalled the needle across ~1.5k tokens"
else bad "lost the needle -> $(head -c 90 <<<"$a")"; fi

echo
echo "=== 5. degeneration check (repetition / gibberish) ==="
a="$(ask 'Write two sentences about the ocean.' 80)"
if [[ "$a" == __ERROR__* ]]; then bad "generation failed ($a)"
else
  words=$(wc -w <<<"$a"); uniq=$(tr ' ' '\n' <<<"$a" | sort -u | wc -l)
  ratio=$(python3 -c "print(round($uniq/max($words,1),2))")
  info "words=$words unique=$uniq ratio=$ratio"
  # Degenerate output loops: unique/total collapses well below ~0.5.
  python3 -c "import sys; sys.exit(0 if $ratio >= 0.5 else 1)" \
    && ok "no degeneration (unique-word ratio $ratio)" \
    || bad "possible degeneration, ratio $ratio: $(head -c 100 <<<"$a")"
fi

echo
echo "==============================================="
printf '  %d passed, %d failed\n' "$pass" "$fail"
if (( fail )); then
  echo "  ⚠️  DO NOT TRUST THIS DEPLOYMENT until these are explained."
  echo "     A wrong o_groups shard is silent -- fluent output is not proof."
  exit 1
fi
echo "  All checks passed."
