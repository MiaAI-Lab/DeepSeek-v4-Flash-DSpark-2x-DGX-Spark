#!/usr/bin/env bash
# Behavioral exit-code gate for scripts/ci-validate.sh (PR #253 review).
#
# ci-validate.sh records every guard failure in one $fail accumulator and only
# turns it into an exit status where that accumulator is checked. The check used
# to sit above the healthcheck / TP=3 / issue191 / hotfix-passthrough tail, so a
# guard failing there still printed "CI validate passed" and exited 0.
#
# This gate breaks one late-guard target at a time in a sandbox copy of the tree
# and asserts the shipped script fails closed. docker/python3/bash are stubbed
# inside the sandbox so the ~40 external test commands the script runs cannot
# decide the exit status; the guard logic under test is the real script. This is
# an exit-code regression test, not CI qualification.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

BASH_BIN="${BASH:-/bin/bash}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TREE="$WORK/tree"
STUBS="$WORK/stubs"
mkdir -p "$TREE" "$STUBS"

for cmd in docker python3 bash; do
  printf '#!/bin/sh\nexit 0\n' >"$STUBS/$cmd"
  chmod +x "$STUBS/$cmd"
done

reset_tree() {
  rm -rf "$TREE"
  mkdir -p "$TREE"
  tar -C "$ROOT" --exclude=./.git -cf - . | tar -xf - -C "$TREE"
}

RC=0
run_ci() {
  RC=0
  ( cd "$TREE" && PATH="$STUBS:$PATH" "$BASH_BIN" scripts/ci-validate.sh ) >"$1" 2>&1 || RC=$?
}

expect() {
  if [ "$RC" = "$1" ]; then
    ok "$2"
  else
    bad "$2 (exit $RC, want $1)"
    tail -n 12 "$3" >&2
  fi
}

# A clean tree has to pass, otherwise the failure cases below prove nothing.
reset_tree
run_ci "$WORK/clean.log"
expect 0 "clean tree exits 0" "$WORK/clean.log"

# Healthcheck: the first guard after the old accumulator check.
reset_tree
sed -i "s|urlhost='\${VLLM_HOST:-127.0.0.1}'|urlhost='127.0.0.1'|" "$TREE/docker-compose.dspark.yml"
run_ci "$WORK/healthcheck.log"
expect 1 "late healthcheck guard fails closed" "$WORK/healthcheck.log"

# Passthrough: the last guard in the script (remote_compose/remote_compose2).
reset_tree
sed -i "s|DSPARK_ENABLE_ROPE_SWA_FIX=\$REMOTE_ROPE_SWA_FIX|DSPARK_ENABLE_ROPE_SWA_FIX=0|g" \
  "$TREE/start-deepseek-v4-flash-dspark.sh"
run_ci "$WORK/passthrough.log"
expect 1 "final passthrough guard fails closed" "$WORK/passthrough.log"

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
