#!/usr/bin/env bash
# CPU-only gate for the issue #32 diagnostic switches.
#
# DSPARK_FLASHINFER_AUTOTUNE selects --enable-flashinfer-autotune (unset/empty/1,
# the shipped default) or --no-enable-flashinfer-autotune (0). The independent
# DSPARK_DIAG_FULL_DECODE_ONLY=1 adds exactly one argv value:
# --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'. Unset must keep
# the historical argv bit-for-bit, and anything but 0/1/unset must exit 2
# before vLLM exec.
#
# Like test-draft-sample-method-gate.sh, this extracts the shipped switch
# block from docker-compose.dspark.yml itself (no copy of the logic), executes
# validate-dspark-config.sh against a stub docker, pins the start launcher's
# per-variable ambient guards (an OS-env-only switch value would split the TP
# pair: compose interpolates the head's process env while the worker only sees
# the streamed .env.dspark), and — when docker compose is available — renders
# BOTH node ranks with `compose config` (never up) to assert switch
# env/command symmetry for the shipped defaults AND for both switches flipped.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
START_LAUNCHER="${START_LAUNCHER:-$ROOT/start-deepseek-v4-flash-dspark.sh}"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

ENABLE=$'A:--enable-flashinfer-autotune'
NOENABLE=$'A:--no-enable-flashinfer-autotune'
JSON=$'C:--compilation-config\nC:{"cudagraph_mode":"FULL_DECODE_ONLY"}'
NOENABLE_JSON=$'A:--no-enable-flashinfer-autotune\nC:--compilation-config\nC:{"cudagraph_mode":"FULL_DECODE_ONLY"}'

# ---------------------------------------------------------------------------
# Layer 1: the shipped entrypoint switch block, lifted verbatim from compose.
#
# Probes print one line per emitted argv element (A: autotune, C: compilation
# config), so exact-output comparison simultaneously proves flag choice, JSON
# single-token quoting, and that no empty argv token ever appears.
# ---------------------------------------------------------------------------
fragment="$(sed -n '/^ *AUTOTUNE_ARGS=();/,/^ *REVISION_ARGS=/p' "$COMPOSE" \
  | sed '/^ *REVISION_ARGS=/d; s/\$\$/$/g')"
if [ -z "$fragment" ] \
  || ! printf '%s' "$fragment" | grep -q 'AUTOTUNE_ARGS=(' \
  || ! printf '%s' "$fragment" | grep -q 'COMPILATION_CONFIG_ARGS=(' \
  || ! printf '%s' "$fragment" | grep -q 'exit 2'; then
  echo "FAIL could not extract the issue #32 switch block from $COMPOSE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' "$fragment" >"$tmp/fragment.sh"
cat >>"$tmp/fragment.sh" <<'EOF'
for _a in "${AUTOTUNE_ARGS[@]}"; do printf 'A:%s\n' "$_a"; done
for _c in "${COMPILATION_CONFIG_ARGS[@]}"; do printf 'C:%s\n' "$_c"; done
EOF

canary_dir="$tmp/canaries"

run_case() { # $1=AT value or __unset__ $2=DG value or __unset__
  local -a pre=()
  [ "$1" != "__unset__" ] && pre+=("DSPARK_FLASHINFER_AUTOTUNE=$1")
  [ "$2" != "__unset__" ] && pre+=("DSPARK_DIAG_FULL_DECODE_ONLY=$2")
  RC=0
  if [ "${#pre[@]}" -gt 0 ]; then
    OUT="$(env "${pre[@]}" bash "$tmp/fragment.sh" 2>"$tmp/frag.err")" || RC=$?
  else
    OUT="$(env -u DSPARK_FLASHINFER_AUTOTUNE -u DSPARK_DIAG_FULL_DECODE_ONLY \
      bash "$tmp/fragment.sh" 2>"$tmp/frag.err")" || RC=$?
  fi
  ERR="$(cat "$tmp/frag.err")"
}

expect_valid() { # $1=label $2=AT $3=DG $4=want-stdout
  run_case "$2" "$3"
  if [ "$RC" -eq 0 ] && [ "$OUT" = "$4" ]; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$(printf '%q' "$OUT")"
  fi
}

expect_reject() { # $1=label $2=AT $3=DG $4=variable named in error
  run_case "$2" "$3"
  if [ "$RC" -eq 2 ] && [ -z "$OUT" ] \
    && printf '%s' "$ERR" | grep -Fq "$4 must be unset, 0 or 1"; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$(printf '%q' "$OUT") err=$(printf '%q' "$ERR")"
  fi
}

expect_valid "unset == default: enable flag only, no compilation config" __unset__ __unset__ "$ENABLE"
expect_valid "empty == default: enable flag only" '' '' "$ENABLE"
expect_valid "explicit 1: enable flag only" 1 __unset__ "$ENABLE"
expect_valid "autotune 0: proven no-enable form" 0 __unset__ "$NOENABLE"
expect_valid "graph diag 0 (or unset): no compilation-config argument" __unset__ 0 "$ENABLE"
expect_valid "graph diag 1: enable flag + compilation config" __unset__ 1 "$ENABLE
$JSON"
expect_valid "combined 0/1: no-enable + compilation config" 0 1 "$NOENABLE_JSON"

# The FULL_DECODE_ONLY JSON must arrive as ONE argv value (compact, unsplit).
run_case __unset__ 1
if [ "$(printf '%s\n' "$OUT" | grep -c '^C:')" = "2" ] \
  && [ "$(printf '%s\n' "$OUT" | grep '^C:' | tail -n 1)" = 'C:{"cudagraph_mode":"FULL_DECODE_ONLY"}' ]; then
  ok "graph diag 1: JSON reaches vLLM as exactly one argv value"
else
  bad "graph diag 1 JSON tokenization: out=$(printf '%q' "$OUT")"
fi

# Empty-array expansion must stay silent even under set -u (entrypoint hardening).
if OUT="$(env -u DSPARK_FLASHINFER_AUTOTUNE -u DSPARK_DIAG_FULL_DECODE_ONLY \
    bash -uc "set -u; source '$tmp/fragment.sh'" 2>/dev/null)" \
  && [ "$OUT" = "$ENABLE" ]; then
  ok "unset arrays expand to no token under set -u"
else
  bad "set -u probe: out=$(printf '%q' "${OUT:-}")"
fi

expect_reject "invalid autotune value exits 2 naming the variable" random __unset__ DSPARK_FLASHINFER_AUTOTUNE
expect_reject "true is not 1" true __unset__ DSPARK_FLASHINFER_AUTOTUNE
expect_reject "padded 1 is not accepted" ' 1' __unset__ DSPARK_FLASHINFER_AUTOTUNE
expect_reject "01 is not accepted" 01 __unset__ DSPARK_FLASHINFER_AUTOTUNE
expect_reject "invalid graph value exits 2 naming the variable" __unset__ yes DSPARK_DIAG_FULL_DECODE_ONLY
expect_reject "graph value 2 exits 2" __unset__ 2 DSPARK_DIAG_FULL_DECODE_ONLY
expect_reject "embedded newline rejected" "$(printf '1\n0')" __unset__ DSPARK_FLASHINFER_AUTOTUNE

# Injection probes: payloads must be rejected as data and never executed.
mkdir -p "$canary_dir"
expect_reject "command substitution rejected" "\$(touch $canary_dir/at)" __unset__ DSPARK_FLASHINFER_AUTOTUNE
expect_reject "backtick substitution rejected" __unset__ "\`touch $canary_dir/dg\`" DSPARK_DIAG_FULL_DECODE_ONLY
expect_reject "JSON key-injection payload rejected" __unset__ '{"cudagraph_mode":"NONE"}","mode":"' DSPARK_DIAG_FULL_DECODE_ONLY
if [ ! -e "$canary_dir/at" ] && [ ! -e "$canary_dir/dg" ]; then
  ok "injection payloads never executed"
else
  bad "injection canary file was created"
fi

# ---------------------------------------------------------------------------
# Layer 2: validate-dspark-config.sh, executed rather than grepped.
#
# Valid values must pass (rc 0), be REPORTED with their resolved meaning, and
# cost exactly one docker call (the final render). Invalid values must exit 2
# BEFORE any docker/compose invocation: zero docker calls.
# ---------------------------------------------------------------------------
VAL="$ROOT/validate-dspark-config.sh"
val_env="$tmp/val.env"
val_bin="$tmp/val-bin"
docker_log="$tmp/docker-calls.log"
mkdir -p "$val_bin"
cat >"$val_bin/docker" <<EOF
#!/usr/bin/env bash
printf '%s\n' "called: \$*" >>"$docker_log"
# Emit a line matching the validator's rendered-command grep so the final
# 'config | grep -E' pipeline succeeds under the validator's pipefail.
printf '%s\n' "--enable-flashinfer-autotune"
EOF
chmod +x "$val_bin/docker"

# Fixtures, not the caller's shell, own both values in Layers 2-4.
export DSPARK_FLASHINFER_AUTOTUNE=1
export DSPARK_DIAG_FULL_DECODE_ONLY=0

write_val_env() { # extra .env lines as arguments
  {
    echo 'WORKER_HOST=stub-worker'
    echo 'MASTER_ADDR=127.0.0.1'
    echo 'MASTER_PORT=29500'
    echo 'DSPARK_VLLM_IMAGE=stub:latest'
    for _line in "$@"; do printf '%s\n' "$_line"; done
  } >"$val_env"
}

reset_docker_log() { : >"$docker_log"; }
docker_calls() { wc -l <"$docker_log"; }

run_val() {
  VRC=0
  VOUT="$(env -u DSPARK_FLASHINFER_AUTOTUNE -u DSPARK_DIAG_FULL_DECODE_ONLY \
    PATH="$val_bin:$PATH" ENV_FILE="$val_env" COMPOSE_FILE="$COMPOSE" \
    bash "$VAL" 2>"$tmp/val.err")" || VRC=$?
  VERR="$(cat "$tmp/val.err")"
}

val_accepts() { # $1=label $2..=.env lines
  local label="$1"; shift
  reset_docker_log
  write_val_env "$@"
  run_val
  if [ "$VRC" -eq 0 ] && [ "$(docker_calls)" -eq 1 ]; then
    ok "$label"
  else
    bad "$label: rc=$VRC docker_calls=$(docker_calls) err=$VERR"
  fi
}

val_reports() { # $1=label $2=grep -E pattern expected in validator stdout
  if printf '%s\n' "$VOUT" | grep -Eq "$2"; then
    ok "$1"
  else
    bad "$1: pattern '$2' not in validator output"
  fi
}

val_rejects() { # $1=label $2=variable $3=.env line
  local label="$1" var="$2"; shift 2
  reset_docker_log
  write_val_env "$@"
  run_val
  if [ "$VRC" -eq 2 ] && [ "$(docker_calls)" -eq 0 ] \
    && printf '%s' "$VERR" | grep -Fq "$var must be unset, 0 or 1" \
    && ! printf '%s\n' "$VOUT" | grep -q 'switch:'; then
    ok "$label"
  else
    bad "$label: rc=$VRC (want 2) docker_calls=$(docker_calls) err=$VERR"
  fi
}

val_accepts "validator: unset switches pass and render once"
val_reports "validator: unset autotune reported as <unset = enabled>" 'flashinfer autotune switch: <unset = enabled>'
val_reports "validator: unset graph diag reported as <unset = off>" 'FULL_DECODE_ONLY diag switch: <unset = off>'
val_accepts "validator: autotune 0 passes and renders once" 'DSPARK_FLASHINFER_AUTOTUNE=0'
val_reports "validator: autotune 0 reported" 'flashinfer autotune switch: 0'
val_accepts "validator: autotune 1 passes" 'DSPARK_FLASHINFER_AUTOTUNE=1'
val_reports "validator: autotune 1 reported" 'flashinfer autotune switch: 1'
val_accepts "validator: graph diag 0 passes" 'DSPARK_DIAG_FULL_DECODE_ONLY=0'
val_reports "validator: graph diag 0 reported" 'FULL_DECODE_ONLY diag switch: 0'
val_accepts "validator: graph diag 1 passes" 'DSPARK_DIAG_FULL_DECODE_ONLY=1'
val_reports "validator: graph diag 1 reported" 'FULL_DECODE_ONLY diag switch: 1'
val_rejects "validator: autotune random exits 2 with ZERO docker calls" DSPARK_FLASHINFER_AUTOTUNE 'DSPARK_FLASHINFER_AUTOTUNE=random'
val_rejects "validator: graph diag yes exits 2 with ZERO docker calls" DSPARK_DIAG_FULL_DECODE_ONLY 'DSPARK_DIAG_FULL_DECODE_ONLY=yes'

unset DSPARK_FLASHINFER_AUTOTUNE DSPARK_DIAG_FULL_DECODE_ONLY

# ---------------------------------------------------------------------------
# Layer 3: the start launcher's per-variable ambient guards (Integration M1).
#
# An OS-environment-only switch value silently splits the TP pair: compose
# interpolates the head's process env, while the worker only ever receives the
# streamed .env.dspark. The launcher therefore mirrors its DSPARK_API_KEYS
# guard for BOTH switches: ambient-only and ambient-vs-env-file mismatches
# must exit 2 naming the variable, an equal value passes, and the whole guard
# range sits BEFORE COMPOSE_ENV_FILE — i.e. before every docker/ssh consumer.
#
# Anchors agreed with the lifecycle owner (per-variable markers inside/next to
# the DSPARK_API_KEYS guard range; both switch ambients snapshotted before the
# FIRST `set -a source` of the env file — a later capture would only see
# file-shadowed values). Rejection contract, per variable: a NON-EMPTY ambient
# absent from .env.dspark exits 2 ("is set in the environment but not in
# .env.dspark"); present-but-differing exits 2 ("does not match"); ambient
# unset, empty, equal, or file-only configuration passes silently. If the
# markers move or reshape, these assertions fail loudly and the two owners
# update together.
# Assertions execute the REAL guard blocks (lifted, Layer-1 style) — no
# docker, no ssh, fully deterministic.
# ---------------------------------------------------------------------------
if [ -f "$START_LAUNCHER" ]; then
  # Lift from the FIRST guard-begin marker (the switch guards may legitimately
  # sit before or after the DSPARK_API_KEYS guard — differing-ambient detection
  # requires their capture to precede the first `set -a source` either way)
  # through the DIAG end marker.
  _first_begin="$(grep -nF -e '# DSPARK_FLASHINFER_AUTOTUNE ambient guard (begin)' \
    -e '# DSPARK_API_KEYS ambient guard (begin)' "$START_LAUNCHER" | cut -d: -f1 | sort -n | sed -n 1p || true)"
  _last_end="$(grep -nF '# DSPARK_DIAG_FULL_DECODE_ONLY ambient guard (end)' "$START_LAUNCHER" | cut -d: -f1 || true)"
  if [ -n "${_first_begin:-}" ] && [ -n "${_last_end:-}" ] && [ "$_first_begin" -le "$_last_end" ]; then
    guards_src="$(sed -n "${_first_begin},${_last_end}p" "$START_LAUNCHER")"
  else
    guards_src=""
  fi

  guards_marked=1
  for _var in DSPARK_FLASHINFER_AUTOTUNE DSPARK_DIAG_FULL_DECODE_ONLY; do
    if [ -n "$guards_src" ] \
      && printf '%s\n' "$guards_src" | grep -qF "# ${_var} ambient guard (begin)" \
      && printf '%s\n' "$guards_src" | grep -qF "# ${_var} ambient guard (end)"; then
      ok "launcher: $_var ambient guard block present (marker-anchored)"
    else
      bad "launcher: $_var ambient guard block missing/reshaped (update anchors with lifecycle owner)"
      guards_marked=0
    fi
  done

  # The guard range must close before COMPOSE_ENV_FILE is assigned: that
  # assignment feeds every later compose render and remote stream, so this
  # line-ordering pin places the guards ahead of all docker/ssh activity.
  # `|| true` keeps a missing marker from tripping set -o pipefail before the
  # assertion below can report it as a loud failure.
  guard_end_line="$_last_end"
  compose_env_line="$(grep -n '^COMPOSE_ENV_FILE=' "$START_LAUNCHER" | cut -d: -f1 || true)"
  if [ -n "$guard_end_line" ] && [ -n "$compose_env_line" ] \
    && [ "$guard_end_line" -lt "$compose_env_line" ] \
    && ! printf '%s\n' "$guards_src" \
      | grep -Eq '(^|[[:space:]])(docker|ssh|scp)([[:space:]]|$)'; then
    ok "launcher: ambient guards precede COMPOSE_ENV_FILE and invoke no docker/ssh/scp"
  else
    bad "launcher: guard ordering broken (end=$guard_end_line COMPOSE_ENV_FILE=$compose_env_line or side-effecting command inside guards)"
  fi

  if [ "$guards_marked" -eq 1 ]; then
    # Execute the lifted guard blocks verbatim. The wrapper replicates the
    # launcher's own mktemp/chmod preamble that sits above the begin marker.
    cat >"$tmp/guards.sh" <<EOF
set -euo pipefail
_dspark_env_clean="\$(mktemp)"
chmod 600 "\$_dspark_env_clean"
_cleanup_guard_env() { [ -z "\$_dspark_env_clean" ] || rm -f -- "\$_dspark_env_clean"; }
trap _cleanup_guard_env EXIT
$(printf '%s\n' "$guards_src")
printf 'RESOLVED_AT=<%s>\n' "\${DSPARK_FLASHINFER_AUTOTUNE-__unset__}"
printf 'RESOLVED_DG=<%s>\n' "\${DSPARK_DIAG_FULL_DECODE_ONLY-__unset__}"
EOF
    guard_env_file="$tmp/guard.env"
    GUARD_RC=0
    GUARD_OUT=""
    run_guard_case() { # $1="VAR=value" injected into the guard env, or "none"; $2=.env.dspark text
      local inject="$1"
      printf '%s' "$2" >"$guard_env_file"
      local -a cmd=(env -u DSPARK_FLASHINFER_AUTOTUNE -u DSPARK_DIAG_FULL_DECODE_ONLY)
      local k
      for k in $(compgen -e); do
        case "$k" in *API_KEY*) cmd+=(-u "$k") ;; esac  # never inherit key material
      done
      if [ "$inject" != "none" ]; then cmd+=("$inject"); fi
      GUARD_RC=0
      GUARD_OUT="$("${cmd[@]}" ENV_FILE="$guard_env_file" TMPDIR="$tmp" \
        bash "$tmp/guards.sh" 2>"$tmp/guard.err")" || GUARD_RC=$?
    }
    guard_expect() { # $1=label $2=want-rc $3=stderr fragment wanted ("" = guard message must be absent) $4=stdout grep -E (optional)
      local label="$1" want_rc="$2" want_err="$3" want_out="${4:-}"
      if [ "$GUARD_RC" -ne "$want_rc" ]; then
        bad "$label: rc=$GUARD_RC (want $want_rc) err=$(tr '\n' ' ' <"$tmp/guard.err")"
        return
      fi
      if [ -n "$want_err" ]; then
        if ! grep -qF "$want_err" "$tmp/guard.err"; then
          bad "$label: stderr missing '$want_err': $(tr '\n' ' ' <"$tmp/guard.err")"
          return
        fi
      elif grep -qF 'does not match .env.dspark' "$tmp/guard.err"; then
        bad "$label: unexpected guard rejection: $(tr '\n' ' ' <"$tmp/guard.err")"
        return
      fi
      if [ -n "$want_out" ] && ! printf '%s' "$GUARD_OUT" | grep -Eq "$want_out"; then
        bad "$label: stdout missing /$want_out/: $(printf '%q' "$GUARD_OUT")"
        return
      fi
      ok "$label"
    }
    # Agreed two-message scheme: an ambient value ABSENT from .env.dspark is
    # reported as "not in"; a present-but-differing one as "does not match".
    GUARD_MSG_NOT_IN='is set in the environment but not in .env.dspark'
    GUARD_MSG_MISMATCH='is set in the environment but does not match .env.dspark'

    # A passing guard must leave neither switch with a stray value: resolved
    # is either unset (<__unset__>) or defined-empty (<>), never a value.
    run_guard_case none 'WORKER_HOST=stub-worker'
    guard_expect "launcher: clean env passes guards, autotune resolves empty/unset" 0 "" \
      '^RESOLVED_AT=(<__unset__>|<>)$'
    guard_expect "launcher: clean env leaves the graph switch empty/unset" 0 "" \
      '^RESOLVED_DG=(<__unset__>|<>)$'

    for _var in DSPARK_FLASHINFER_AUTOTUNE DSPARK_DIAG_FULL_DECODE_ONLY; do
      case "$_var" in
        DSPARK_FLASHINFER_AUTOTUNE) _res='RESOLVED_AT' ;;
        *) _res='RESOLVED_DG' ;;
      esac
      run_guard_case "$_var=1" 'WORKER_HOST=stub-worker'
      guard_expect "launcher: $_var ambient-only (absent from .env.dspark) exits 2" 2 \
        "error: $_var $GUARD_MSG_NOT_IN"
      run_guard_case "$_var=1" "$_var=0"
      guard_expect "launcher: $_var ambient 1 vs .env.dspark 0 exits 2" 2 \
        "error: $_var $GUARD_MSG_MISMATCH"
      run_guard_case "$_var=0" "$_var=1"
      guard_expect "launcher: $_var ambient 0 vs .env.dspark 1 exits 2" 2 \
        "error: $_var $GUARD_MSG_MISMATCH"
      run_guard_case "$_var=1" "$_var=1"
      guard_expect "launcher: $_var ambient equal to .env.dspark passes" 0 "" "^${_res}=<1>$"
      run_guard_case "$_var=" 'WORKER_HOST=stub-worker'
      guard_expect "launcher: $_var ambient empty behaves like unset (passes)" 0 "" "^${_res}=(<__unset__>|<>)$"
      # File-only config is the DOCUMENTED flip path (uncomment in
      # .env.dspark; nothing exported): never reject it as a mismatch just
      # because sourcing put the value into the process env.
      run_guard_case none "$_var=0"
      guard_expect "launcher: $_var file-only flip (ambient untouched) passes" 0 "" "^${_res}=<0>$"
      run_guard_case "$_var=" "$_var=1"
      guard_expect "launcher: $_var explicit-empty ambient defers to .env.dspark" 0 "" "^${_res}=<1>$"
    done

    # Adjacency must not have weakened the original API_KEYS guard.
    run_guard_case 'DSPARK_API_KEYS=zz-sentinel-ambient' 'WORKER_HOST=stub-worker'
    guard_expect "launcher: DSPARK_API_KEYS ambient-only still exits 2" 2 \
      "error: DSPARK_API_KEYS $GUARD_MSG_MISMATCH"
  fi
else
  bad "start launcher not found: $START_LAUNCHER"
fi

# ---------------------------------------------------------------------------
# Layer 4: real rendered Compose expansion, BOTH node ranks, default AND
# flipped switch values (K3 L6: flipped render symmetry).
#
# Runs only `docker compose config` (pure interpolation, never up/start).
# Both ranks must receive identical switch env values and an identical switch
# command slice — the TP pair fails to boot if ranks disagree. Asserted for
# the shipped defaults (defined-empty) AND with both switches flipped
# (AUTOTUNE=0, FULL_DECODE_ONLY=1). The rendered command slice is static shell
# source, so variant strength comes from the interpolated env values, which
# Layer 1 proves select the flipped argv arms.
# ---------------------------------------------------------------------------
if docker compose version >/dev/null 2>&1; then
  cat >"$tmp/rendered.py" <<'PYEOF'
import hashlib, json, os, subprocess, sys

compose, envfile, node_rank, headless, slice_out = sys.argv[1:6]

env = {k: v for k, v in os.environ.items()
       if k not in ("NODE_RANK", "HEADLESS",
                    "DSPARK_FLASHINFER_AUTOTUNE", "DSPARK_DIAG_FULL_DECODE_ONLY")}
env.update(COMPOSE_DISABLE_ENV_FILE="1", NODE_RANK=node_rank)
if headless != "-":
    env["HEADLESS"] = headless
else:
    env.pop("HEADLESS", None)

p = subprocess.run(
    ["docker", "compose", "--env-file", envfile, "-f", compose,
     "config", "--format", "json"],
    capture_output=True, text=True, env=env, cwd=os.path.dirname(compose) or ".",
)
if p.returncode != 0:
    tail = (p.stderr.strip().splitlines() or ["render failed"])[-1]
    print("render failed: " + tail)
    sys.exit(1)

svc = json.loads(p.stdout)["services"]["vllm-dspark"]
renv, script = svc["environment"], svc["command"][2]
start = script.find("AUTOTUNE_ARGS=();")
end = script.find("REVISION_ARGS=", start)
if start < 0 or end < 0:
    print("issue #32 switch block missing from rendered command", file=sys.stderr)
    sys.exit(1)

# The raw switch command slice, for textual assertions by the caller.
with open(slice_out, "w", encoding="utf-8") as fh:
    fh.write(script[start:end])

# stdout carries exactly three data lines per rank: AT value, DG value,
# sha256 of the switch command slice.
print(renv.get("DSPARK_FLASHINFER_AUTOTUNE", "<absent>"))
print(renv.get("DSPARK_DIAG_FULL_DECODE_ONLY", "<absent>"))
print(hashlib.sha256(script[start:end].encode()).hexdigest())

if '"$${AUTOTUNE_ARGS[@]}"' not in script or '"$${COMPILATION_CONFIG_ARGS[@]}"' not in script:
    print("quoted argv-array expansions missing from rendered command", file=sys.stderr)
    sys.exit(2)
PYEOF

  render_rank() { # $1=node_rank $2=headless(-|1) $3=out-file $4=slice-out-file
    python3 "$tmp/rendered.py" "$COMPOSE" "$val_env" "$1" "$2" "$4" >"$3" 2>"$3.err"
  }

  R_OK=0
  R_AT=""
  R_DG=""
  render_pair() { # $1=label; $val_env already carries the wanted switch lines
    local label="$1"
    local r0="$tmp/pair-rank0.out" r1="$tmp/pair-rank1.out"
    if ! render_rank 0 - "$r0" "$tmp/pair-slice0.txt" \
      || ! render_rank 1 1 "$r1" "$tmp/pair-slice1.txt"; then
      bad "$label: render failed: $(cat "$r0.err" "$r1.err" 2>/dev/null | sort -u | sed -n 1,2p)"
      R_OK=0
      return
    fi
    if ! cmp -s "$r0" "$r1"; then
      bad "$label: rank asymmetry: $(diff "$r0" "$r1" | tr '\n' ' ')"
      R_OK=0
      return
    fi
    if ! cmp -s "$tmp/pair-slice0.txt" "$tmp/pair-slice1.txt"; then
      bad "$label: switch command slice differs across ranks"
      R_OK=0
      return
    fi
    ok "$label: ranks 0/1 render identical switch env values and command slice"
    R_AT="$(sed -n 1p "$r0")"
    R_DG="$(sed -n 2p "$r0")"
    R_OK=1
  }

  write_val_env
  render_pair "rendered default .env"
  if [ "$R_OK" -eq 1 ]; then
    if [ -z "$R_AT" ] && [ -z "$R_DG" ]; then
      ok "rendered default .env: both switches reach the entrypoint defined-empty (current behavior)"
    else
      bad "default .env renders non-empty switches: AT='$R_AT' DG='$R_DG'"
    fi
    if grep -qF 'AUTOTUNE_ARGS=(--enable-flashinfer-autotune)' "$tmp/pair-slice0.txt"; then
      ok "rendered default .env: command slice ships the enable-flag arm"
    else
      bad "default command slice lost the enable-flag arm"
    fi
  fi

  write_val_env 'DSPARK_FLASHINFER_AUTOTUNE=0' 'DSPARK_DIAG_FULL_DECODE_ONLY=1'
  render_pair "rendered flipped switches (autotune=0, full-decode-only=1)"
  if [ "$R_OK" -eq 1 ]; then
    if [ "$R_AT" = "0" ] && [ "$R_DG" = "1" ]; then
      ok "rendered flipped switches: both ranks receive AT=0 / DG=1 verbatim"
    else
      bad "flipped render values: AT='$R_AT' DG='$R_DG' (want AT=0 DG=1)"
    fi
    if grep -qF 'AUTOTUNE_ARGS=(--no-enable-flashinfer-autotune)' "$tmp/pair-slice0.txt" \
      && grep -qF 'COMPILATION_CONFIG_ARGS=(--compilation-config '"'"'{"cudagraph_mode":"FULL_DECODE_ONLY"}'"'"')' \
        "$tmp/pair-slice0.txt"; then
      ok "rendered flipped switches: command slice carries the flipped argv arms"
    else
      bad "flipped command slice lost the flipped argv arms"
    fi
  fi
else
  say "SKIP rendered-compose layer (docker compose unavailable); layers 1-3 still cover the contract"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
