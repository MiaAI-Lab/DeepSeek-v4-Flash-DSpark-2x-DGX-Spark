#!/usr/bin/env bash
# Issue #32 lifecycle hook tests (deterministic, CPU-only, no live calls).
#
# Strategy: extract the REAL observer helper blocks from the four launchers,
# source them in a sandbox whose ssh/scp/docker/python3 are stubs. The ssh
# stub mimics real ssh: it %q-logs its exact argv, then EXECUTES the shipped
# remote command string locally under a sanitized worker-like environment
# (own HOME/XDG, no inherited DSPARK_GB10_* config). The python3 stub records
# its argv with \x1f separators, so spaced WORKER_DIR/state-dir values are
# proven to arrive as ONE intact argv token through the remote shell. The
# scp destination is recovered verbatim (python3 shlex) and evaluated to
# prove the spaced path resolves correctly. Plus static source-order/guard
# assertions and a standalone execution of the real ambient-switch guard
# region. Nothing here touches Docker, SSH, or any service.
#
# Usage: bash scripts/test-issue32-observer-lifecycle.sh [-q]
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

QUIET=0
if [ "${1:-}" = "-q" ] || [ "${1:-}" = "--quiet" ]; then
  QUIET=1
fi

PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); [ "$QUIET" = "1" ] || echo "ok   - $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL - $1"; }
check() { if [ "$2" -eq 0 ]; then ok "$1"; else fail "$1"; fi; }

WORK="$(mktemp -d)"
trap 'rm -rf -- "$WORK"' EXIT
BIN="$WORK/bin"
LOG="$WORK/shim.log"
ARGV_LOG="$WORK/observer-argv.log"
OUT="$WORK/block-output.txt"
mkdir -p "$BIN"
: >"$LOG"
: >"$ARGV_LOG"

# ---- command shims ---------------------------------------------------------
cat >"$BIN/timeout" <<'EOF'
#!/usr/bin/env bash
shift
exec "$@"
EOF
chmod +x "$BIN/timeout"

# docker: list the running project container so the already-running exit-3
# fragment fires; otherwise just log argv.
cat >"$BIN/docker" <<EOF
#!/usr/bin/env bash
{ printf 'docker'; printf ' %q' "\$@"; printf '\n'; } >>'$LOG'
printf '%s\n' 'deepseek-v4-flash-vllm-dspark-1'
exit 0
EOF
chmod +x "$BIN/docker"

for c in scp curl ss xargs ip mkdir; do
  cat >"$BIN/$c" <<EOF
#!/usr/bin/env bash
{ printf '$c'; printf ' %q' "\$@"; printf '\n'; } >>'$LOG'
[ "$c" = scp ] && [ "\${SCP_FAIL:-0}" = "1" ] && exit 7
exit 0
EOF
  chmod +x "$BIN/$c"
done


# ssh: log exact %q argv, then behave like real ssh — skip option pairs,
# drop the host token, and EXECUTE the remaining remote command string under
# bash. The execution environment is sanitized to a worker-like identity:
# worker HOME/XDG (no head leakage) and NO inherited DSPARK_GB10_* config,
# so anything the remote command sees traveled inside the shipped string.
cat >"$BIN/ssh" <<EOF
#!/usr/bin/env bash
{ printf 'ssh'; printf ' %q' "\$@"; printf '\n'; } >>'$LOG'
if [ "\${SSH_FAIL:-0}" = "1" ]; then exit 255; fi
while [ "\$#" -gt 0 ]; do
  case "\$1" in
    -o|-p|-i|-l) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
[ "\$#" -gt 0 ] && shift   # host token
[ "\$#" -eq 0 ] && exit 0
_cmd="\$*"
export HOME="\${SANITIZED_HOME:-\$HOME}"
export XDG_STATE_HOME="\${SANITIZED_XDG:-\$HOME}"
unset DSPARK_GB10_OBSERVER DSPARK_GB10_OBSERVER_INTERVAL
unset DSPARK_GB10_OBSERVER_STATE_DIR DSPARK_GB10_OBSERVER_AUTOSTOP
if [ -n "\${SSH_CAPTURE_STDIN:-}" ]; then
  cat >"\${SSH_CAPTURE_STDIN}.last"
  bash -c "\$_cmd" <"\${SSH_CAPTURE_STDIN}.last"
else
  bash -c "\$_cmd" </dev/null
fi
exit 0
EOF
chmod +x "$BIN/ssh"

REAL_PY="$(command -v python3)"
cat >"$BIN/python3" <<EOF
#!/usr/bin/env bash
_obs=0
for arg in "\$@"; do
  case "\$arg" in *gb10-memory-observer.py) _obs=1 ;; esac
done
if [ "\$_obs" = "1" ]; then
  { printf 'observer'; printf '\x1f%s' "\$@"; \\
    printf '\x1f[STATE_DIR=%s]' "\${DSPARK_GB10_OBSERVER_STATE_DIR-}"; \\
    printf '\n'; } >>'$ARGV_LOG'
  _last="\${@: -1}"
  [ "\${OBS_FAIL_START:-0}" = "1" ] && [ "\$_last" = "start" ] && exit 7
  [ "\${OBS_FAIL_STATUS:-0}" = "1" ] && [ "\$_last" = "status" ] && exit 4
  [ "\${OBS_FAIL_STOP:-0}" = "1" ] && [ "\$_last" = "stop" ] && exit 9
  exit 0
fi
exec '$REAL_PY' "\$@"
EOF
chmod +x "$BIN/python3"

# ---- sandbox repo copy hosting the observer path ----------------------------
FIX="$WORK/fixture"
WORKER_STATE_HOME="$WORK/worker-state-home"   # simulates the WORKER's own XDG
mkdir -p "$FIX/scripts" "$WORKER_STATE_HOME"
printf '# stub\n' >"$FIX/scripts/gb10-memory-observer.py"

extract_block() { # $1=launcher -> stdout: the marked observer block
  awk '/# Issue #32 GB10 memory\/NVRM observer \(begin\)/{f=1}
       f{print}
       /# Issue #32 GB10 memory\/NVRM observer \(end\)/{f=0}' "$1"
}

run_block() { # $1=launcher; config via caller environment; sets RB_RC
  extract_block "$1" >"$WORK/block.sh"
  (
    cd "$FIX" || exit 99
    export PATH="$BIN:$PATH" SSH_FAIL SCP_FAIL OBS_FAIL_START OBS_FAIL_STATUS OBS_FAIL_STOP
    export SCRIPT_DIR="$FIX" WORKER_HOST="w-host" PROJECT_NAME="deepseek-v4-flash"
    export WORKER_DIR="/srv/deepseek dspark" TAIL=160
    export XDG_STATE_HOME="$WORK/state-home"
    export SANITIZED_HOME="$WORKER_STATE_HOME" SANITIZED_XDG="$WORKER_STATE_HOME"
    bash -c 'source "$1"' _ "$WORK/block.sh"
  ) >"$OUT" 2>&1
  RB_RC=$?
}

count_log()   { grep -c "$1" "$LOG" 2>/dev/null || true; }
count_argv()  { grep -cF -- "$1" "$ARGV_LOG" 2>/dev/null || true; }
log_line_no() { grep -n "$1" "$LOG" 2>/dev/null | head -1 | cut -d: -f1; }

# Recover the exact %q-encoded argv of log line $2 into array LOG_ARGS ($1
# names it). Uses python3 shlex so backslash-escaped spaces survive verbatim;
# naive word splitting would collapse them and corrupt spaced paths.
log_args() {
  local -a _argv
  mapfile -t _argv < <(python3 -c 'import shlex, sys
for tok in shlex.split(sys.stdin.readline()):
    print(tok)' < <(sed -n "${2}p" "$LOG"))
  local _joined="" _t
  for _t in "${_argv[@]}"; do
    _joined="${_joined:+$_joined }$(printf '%q' "$_t")"
  done
  eval "$1=($_joined)"
}

SEP=$'\x1f'
SPACED_PY="/srv/deepseek dspark/scripts/gb10-memory-observer.py"

# ===========================================================================
# Dynamic: real start-hook block, sandboxed
# ===========================================================================
START_L="$REPO_DIR/start-deepseek-v4-flash-dspark.sh"
STOP_L="$REPO_DIR/stop-deepseek-v4-flash-dspark.sh"
STATUS_L="$REPO_DIR/status-deepseek-v4-flash-dspark.sh"
LOGS_L="$REPO_DIR/logs-deepseek-v4-flash-dspark.sh"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER= run_block "$START_L"
RB=$RB_RC
check "off: start hooks issue zero observer commands and stay silent" \
  "$([ "$(count_log '^observer')" -eq 0 ] && [ "$(count_log '^ssh ')" -eq 0 ] && [ "$RB" -eq 0 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=1 run_block "$START_L"
check "on: head observer started exactly once and worker reached start" \
  "$([ "$(count_argv "${SEP}start${SEP}")" -eq 2 ] && [ "$(count_log '^scp .*gb10-memory-observer')" -eq 1 ] && [ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"
head_ln="$(grep -nF "${SEP}start${SEP}" "$ARGV_LOG" | head -1 | cut -d: -f1)"
scp_ln=$(grep -n '^scp .*gb10-memory-observer' "$LOG" | head -1 | cut -d: -f1)
check "on: head local start precedes any shipping (argv record is first evidence)" \
  "$([ -n "$head_ln" ] && [ "$head_ln" = "1" ] && [ -n "$scp_ln" ] && echo 0 || echo 1)"

# Real-argv proof: the scp destination must be ONE argv token whose remote-
# shell evaluation yields the spaced worker path (what scp actually delivers).
log_args SCP_ARGS "$scp_ln"
scp_dest="${SCP_ARGS[${#SCP_ARGS[@]}-1]}"
remote_path="${scp_dest#*:}"
resolved="$(bash -c "printf '%s' $remote_path")"
check "on: scp destination is one token evaluating to the spaced worker path" \
  "$([ "${#SCP_ARGS[@]}" -eq 4 ] && [ "$resolved" = "/srv/deepseek dspark/scripts/gb10-memory-observer.py" ] && echo 0 || echo 1)"

check "on: shipped worker start parses; spaced script path arrives as ONE argv token" \
  "$([ "$(count_argv "${SEP}${SPACED_PY}${SEP}start")" -ge 1 ] && echo 0 || echo 1)"

sl=$(grep -n '^scp .*gb10-memory-observer' "$LOG" | head -1 | cut -d: -f1)
rl=$(grep -n 'timeout.*python3.*start' "$LOG" | tail -1 | cut -d: -f1)
check "on: scp precedes the remote worker start in shipped order" \
  "$([ -n "$sl" ] && [ -n "$rl" ] && [ "$sl" -lt "$rl" ] && echo 0 || echo 1)"

# Spaced custom state dir must travel %q-escaped INSIDE the shipped remote
# command and arrive INTACT (worker-side stub observes the value).
: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=1 DSPARK_GB10_OBSERVER_STATE_DIR="/x/sp aced" run_block "$START_L"
check "on: spaced custom state dir reaches the WORKER observer as one value" \
  "$([ "$(grep -F "${SPACED_PY}" "$ARGV_LOG" | grep -cF '[STATE_DIR=/x/sp aced]')" -ge 1 ] && echo 0 || echo 1)"

# Strict enable semantics (Integration L2): junk values exit 2 loudly.
: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=true run_block "$START_L"
err="$(grep -o 'DSPARK_GB10_OBSERVER must be 0 or 1' "$OUT" | head -1)"
check "on: invalid DSPARK_GB10_OBSERVER=true rejects with exit 2 naming the variable" \
  "$([ "$RB_RC" -eq 2 ] && [ -n "$err" ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
OBS_FAIL_START=1 DSPARK_GB10_OBSERVER=1 run_block "$START_L"
check "on: head observer start failure stays non-fatal (WARN/continue, exit 0)" \
  "$([ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"

# Worker unreachable: first worker ssh probe fails -> WARN, no scp, head keeps recording.
: >"$LOG"; : >"$ARGV_LOG"
SSH_FAIL=1 DSPARK_GB10_OBSERVER=1 run_block "$START_L"
check "on: worker unreachable warns and continues; head observer still started; no scp" \
  "$([ "$RB_RC" -eq 0 ] && [ "$(count_argv "${SEP}start${SEP}")" -eq 1 ] && [ "$(count_log '^scp ')" -eq 0 ] && echo 0 || echo 1)"

# scp failure: reachability + mkdir succeed, copy fails -> WARN skip, no remote start.
: >"$LOG"; : >"$ARGV_LOG"
SCP_FAIL=1 DSPARK_GB10_OBSERVER=1 run_block "$START_L"
check "on: scp failure skips worker observer without failing the launch" \
  "$([ "$RB_RC" -eq 0 ] && [ "$(count_log 'timeout.*python3.*start')" -eq 0 ] && [ "$(count_argv "${SEP}start${SEP}")" -eq 1 ] && echo 0 || echo 1)"

# ===========================================================================
# Dynamic: already-running attachment fires the hooks on the exit-3 path
# ===========================================================================
attach_awk="$WORK/attach.awk"
cat >"$attach_awk" <<'AWKEOF'
/^already_running_hint\(\) \{/{f=1}
f{print}
f && /^  exit 3$/{exit}
AWKEOF
attach_slice="$(awk -f "$attach_awk" "$START_L")"
extract_block "$START_L" >"$WORK/block-attach-pre.sh"
printf '\n%s\n' "$attach_slice" >>"$WORK/block-attach-pre.sh"
: >"$LOG"; : >"$ARGV_LOG"
(
  cd "$FIX" || exit 99
  export PATH="$BIN:$PATH" SCRIPT_DIR="$FIX" WORKER_HOST="w-host" \
    WORKER_DIR="/srv/deepseek dspark" PROJECT_NAME="deepseek-v4-flash" \
    XDG_STATE_HOME="$WORK/state-home"
  export SANITIZED_HOME="$WORKER_STATE_HOME" SANITIZED_XDG="$WORKER_STATE_HOME"
  export DSPARK_GB10_OBSERVER=1
  bash -c 'source "$1"' _ "$WORK/block-attach-pre.sh" >/dev/null 2>&1
)
ATTACH_RC=$?
check "attach: dockerd-restored cluster path (exit 3) still starts head+worker observers" \
  "$([ "$(count_argv "${SEP}start${SEP}")" -ge 2 ] && [ "$(count_argv "${SEP}${SPACED_PY}${SEP}start")" -ge 1 ] && [ "$(count_log '^scp .*gb10-memory-observer')" -ge 1 ] && echo 0 || echo 1)"

# ===========================================================================
# Dynamic: real stop-hook block, sandboxed
# ===========================================================================
: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=1 run_block "$STOP_L"
check "stop: default autostop stops head once and worker over ssh" \
  "$([ "$(count_argv "${SEP}stop${SEP}")" -eq 2 ] && [ "$(count_log 'timeout.*python3.*stop')" -ge 1 ] && [ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"
check "stop: shipped worker stop parses with spaced path as one token" \
  "$([ "$(count_argv "${SEP}${SPACED_PY}${SEP}stop")" -ge 1 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=1 DSPARK_GB10_OBSERVER_AUTOSTOP=0 run_block "$STOP_L"
check "stop: AUTOSTOP=0 issues zero observer commands" \
  "$([ "$(count_log '^ssh ')" -eq 0 ] && [ "$(count_argv "${SEP}")" -eq 0 ] && [ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=1 DSPARK_GB10_OBSERVER_AUTOSTOP=true run_block "$STOP_L"
err="$(grep -o 'DSPARK_GB10_OBSERVER_AUTOSTOP must be 0 or 1' "$OUT" | head -1)"
check "stop: invalid AUTOSTOP=true rejects with exit 2 naming the variable" \
  "$([ "$RB_RC" -eq 2 ] && [ -n "$err" ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=0 DSPARK_GB10_OBSERVER_AUTOSTOP=true run_block "$STOP_L"
check "stop: explicit 0 short-circuits; junk AUTOSTOP never evaluated; exit 0" \
  "$([ "$RB_RC" -eq 0 ] && [ "$(count_log '^observer ')" -eq 0 ] && [ "$(count_argv "${SEP}")" -eq 0 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
OBS_FAIL_STOP=1 DSPARK_GB10_OBSERVER=1 run_block "$STOP_L"
check "stop: observer stop failure stays non-fatal (never affects service verdict)" \
  "$([ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER=bogus run_block "$STOP_L"
err="$(grep -o 'DSPARK_GB10_OBSERVER must be 0 or 1' "$OUT" | head -1)"
check "stop: invalid DSPARK_GB10_OBSERVER=bogus rejects with exit 2" \
  "$([ "$RB_RC" -eq 2 ] && [ -n "$err" ] && echo 0 || echo 1)"

# ===========================================================================
# Dynamic: status/logs exit neutrality + worker-local state-dir resolution
# ===========================================================================
run_section() { # $1=launcher; sources ONLY the marked opportunistic section
  extract_block "$1" >"$WORK/section.sh"
  (
    cd "$FIX" || exit 99
    export PATH="$BIN:$PATH" SCRIPT_DIR="$FIX" WORKER_HOST="w-host" \
      WORKER_DIR="/srv/deepseek dspark" TAIL=160
    export XDG_STATE_HOME="$WORK/state-home"
    export SANITIZED_HOME="$WORKER_STATE_HOME" SANITIZED_XDG="$WORKER_STATE_HOME"
    export SSH_CAPTURE_STDIN="$WORK/remote-stdin.txt"
    bash -c 'source "$1"' _ "$WORK/section.sh"
  ) >"$OUT" 2>&1
  RB_RC=$?
}

: >"$LOG"; : >"$ARGV_LOG"; rm -f "$WORK/remote-stdin.txt"* 
OBS_FAIL_STATUS=1 run_section "$STATUS_L"
check "status: failing observer status + missing records never change exit status" \
  "$([ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"
check "status: default setup forwards NO head state dir to the worker" \
  "$([ "$(count_log 'DS_OBS_STATE_DIR=')" -eq 0 ] && echo 0 || echo 1)"

# Execute the shipped remote body the way the worker would (empty forwarded
# state dir): it must resolve the WORKER's own XDG default and tail records
# found there. The ssh stub already executed the body under the sanitized
# worker environment; prove both the resolution and the status call.
mkdir -p "$WORKER_STATE_HOME/dspark-observer"
printf '{"schema":1,"event":"sample"}\n' >"$WORKER_STATE_HOME/dspark-observer/records.ndjson"
: >"$LOG"; : >"$ARGV_LOG"
run_section "$STATUS_L"
grep -F 'dspark-observer/records.ndjson' "$OUT" | grep -q "$WORKER_STATE_HOME" && wd=0 || wd=1
tail_showed_records=1
grep -q '"event": *"sample"' "$OUT" || tail_showed_records=0
check "status: remote body resolves the WORKER-local XDG default (never head's)" "$wd"
check "status: worker records found by the resolved default are displayed" \
  "$([ "$tail_showed_records" -eq 1 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
rm -rf "$WORKER_STATE_HOME/dspark-observer"
DSPARK_GB10_OBSERVER_STATE_DIR="/cust/state" run_section "$STATUS_L"
check "status: explicitly set state dir is forwarded to the worker" \
  "$([ "$RB_RC" -eq 0 ] && [ "$(count_log 'DS_OBS_STATE_DIR=/cust/state')" -ge 1 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"; : >"$OUT"
OBS_FAIL_STATUS=1 run_section "$LOGS_L"
check "logs: failing observer status + missing records never change exit status" \
  "$([ "$RB_RC" -eq 0 ] && echo 0 || echo 1)"
check "logs: default setup forwards NO head state dir to the worker" \
  "$([ "$(count_log 'DS_OBS_STATE_DIR=')" -eq 0 ] && echo 0 || echo 1)"

: >"$LOG"; : >"$ARGV_LOG"
DSPARK_GB10_OBSERVER_STATE_DIR="/cust/state logs" run_section "$LOGS_L"
check "logs: custom (spaced) state dir forwarded intact to the worker body" \
  "$([ "$RB_RC" -eq 0 ] && grep -qF 'no records in /cust/state logs yet' "$OUT" && echo 0 || echo 1)"

# Worker ssh down: status degrade to a printed note, still exit 0.
: >"$LOG"; : >"$ARGV_LOG"
SSH_FAIL=1 run_section "$STATUS_L"
check "status: unreachable worker degrades gracefully with exit 0" \
  "$([ "$RB_RC" -eq 0 ] && grep -q 'worker unreachable' "$OUT" && echo 0 || echo 1)"

# ===========================================================================
# Dynamic: ambient-vs-env-file switch guards (real region, standalone)
# ===========================================================================
guard_awk="$WORK/guard.awk"
cat >"$guard_awk" <<'AWKEOF'
/# DSPARK_API_KEYS ambient guard \(begin\)/{f=1}
f{print}
/# DSPARK_DIAG_FULL_DECODE_ONLY ambient guard \(end\)/{f=0}
AWKEOF
guard_region="$(awk -f "$guard_awk" "$START_L")"
printf '%s\nprintf "GUARD_OK\\n"\n' "$guard_region" >"$WORK/guard.sh"

run_guard() {
  local envf="$WORK/guard.env"
  : >"$WORK/guard-out.txt"
  {
    printf '%s\n' "${GUARD_ENV_LINES[@]}"
  } >"$envf"
  (
    cd "$FIX" || exit 99
    export ENV_FILE="$envf"
    _dspark_env_clean="$WORK/guard-clean.env"
    export _dspark_env_clean
    unset DSPARK_FLASHINFER_AUTOTUNE DSPARK_DIAG_FULL_DECODE_ONLY DSPARK_API_KEYS
    [ -n "${AMB_AUTOTUNE+x}" ] && export DSPARK_FLASHINFER_AUTOTUNE="$AMB_AUTOTUNE"
    [ -n "${AMB_DIAG+x}" ] && export DSPARK_DIAG_FULL_DECODE_ONLY="$AMB_DIAG"
    bash -c 'source "$1"' _ "$WORK/guard.sh" >"$WORK/guard-out.txt" 2>&1
  )
}

GUARD_ENV_LINES=("DSPARK_FLASHINFER_AUTOTUNE=0" "DSPARK_DIAG_FULL_DECODE_ONLY=1")
AMB_AUTOTUNE=1 AMB_DIAG= run_guard
grc=$?
check "guards: differing ambient vs .env.dspark rejected (does not match)" \
  "$([ "$grc" -eq 2 ] && grep -q 'does not match .env.dspark' "$WORK/guard-out.txt" && echo 0 || echo 1)"

GUARD_ENV_LINES=("X=1")
AMB_AUTOTUNE=1 AMB_DIAG= run_guard
grc=$?
check "guards: ambient-only value (absent from .env.dspark) rejected (not in)" \
  "$([ "$grc" -eq 2 ] && grep -q 'not in .env.dspark' "$WORK/guard-out.txt" && echo 0 || echo 1)"

GUARD_ENV_LINES=("DSPARK_FLASHINFER_AUTOTUNE=1")
AMB_AUTOTUNE=1 AMB_DIAG= run_guard
grc=$?
check "guards: ambient equal to env file passes silently" \
  "$([ "$grc" -eq 0 ] && grep -q 'GUARD_OK' "$WORK/guard-out.txt" && echo 0 || echo 1)"

GUARD_ENV_LINES=("DSPARK_FLASHINFER_AUTOTUNE=1" "DSPARK_DIAG_FULL_DECODE_ONLY=0")
AMB_AUTOTUNE= AMB_DIAG= run_guard
grc=$?
check "guards: explicitly EMPTY ambient values pass (rank-consistent no-op)" \
  "$([ "$grc" -eq 0 ] && grep -q 'GUARD_OK' "$WORK/guard-out.txt" && echo 0 || echo 1)"

GUARD_ENV_LINES=("DSPARK_FLASHINFER_AUTOTUNE=0" "DSPARK_DIAG_FULL_DECODE_ONLY=1")
AMB_AUTOTUNE= AMB_DIAG= run_guard
grc=$?
check "guards: file-only flip (documented restart-required flow) passes" \
  "$([ "$grc" -eq 0 ] && grep -q 'GUARD_OK' "$WORK/guard-out.txt" && echo 0 || echo 1)"

GUARD_ENV_LINES=("X=1")
unset AMB_AUTOTUNE AMB_DIAG
run_guard
grc=$?
check "guards: fully unset ambients pass untouched" \
  "$([ "$grc" -eq 0 ] && grep -q 'GUARD_OK' "$WORK/guard-out.txt" && echo 0 || echo 1)"

# ===========================================================================
# Static: source order + guards in the real launchers
# ===========================================================================
lnum() { grep -n "$2" "$1" | head -1 | cut -d: -f1; }

sb=$(lnum "$START_L" "NVRM observer (end)")
s_exit3=$(grep -n 'already exists for project' "$START_L" | head -1 | cut -d: -f1)
s_trap=$(lnum "$START_L" "^trap on_error ERR\$")
check "static: observer hooks precede already-running exit-3 AND the ERR trap" \
  "$([ -n "$sb" ] && [ -n "$s_exit3" ] && [ -n "$s_trap" ] && [ "$sb" -lt "$s_exit3" ] && [ "$sb" -lt "$s_trap" ] && echo 0 || echo 1)"

pb=$(lnum "$STOP_L" "NVRM observer (end)")
p_acct=$(lnum "$STOP_L" 'STOP_FAILURES" -gt 0')
check "static: stop hooks precede STOP_FAILURES accounting" \
  "$([ -n "$pb" ] && [ -n "$p_acct" ] && [ "$pb" -lt "$p_acct" ] && echo 0 || echo 1)"

violations=$( { extract_block "$START_L"; extract_block "$STOP_L"; \
                extract_block "$STATUS_L"; extract_block "$LOGS_L"; } \
              | grep -cE '\bdocker\b|\bkill\b|\bcurl\b|\brestart\b' || true)
guarded=$(( $(extract_block "$STATUS_L" | grep -c 'status || true') + \
            $(extract_block "$LOGS_L"   | grep -c 'status || true') ))
check "static: report-only tokens clean; status calls || true-guarded" \
  "$([ "$violations" -eq 0 ] && [ "$guarded" -ge 4 ] && echo 0 || echo 1)"

# Failed serving start intentionally leaves an opted-in observer recording.
grep -q 'does NOT auto-stop an opted-in observer' "$START_L" && l3_doc=0 || l3_doc=1
check "static: failed-start evidence retention documented in the launcher" "$l3_doc"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
