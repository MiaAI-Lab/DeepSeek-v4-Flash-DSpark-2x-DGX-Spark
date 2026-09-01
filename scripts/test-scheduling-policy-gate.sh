#!/usr/bin/env bash
# CPU-only contract test for SCHEDULING_POLICY -> --scheduling-policy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
VALIDATOR="$ROOT/validate-dspark-config.sh"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

fragment="$(
  sed -n '/case "\$\${SCHEDULING_POLICY/,/^ *esac;/p' "$COMPOSE" |
    sed 's/\$\$/$/g'
)"
if [ -z "$fragment" ]; then
  echo "FAIL could not extract SCHEDULING_POLICY gate from $COMPOSE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\nprintf "%%s" "$SCHEDULING_POLICY"\n' "$fragment" >"$tmp/gate.sh"

run_gate() { # $1=unset|set $2=value
  GATE_RC=0
  if [ "$1" = "unset" ]; then
    GATE_OUT="$(env -u SCHEDULING_POLICY bash "$tmp/gate.sh" 2>"$tmp/gate.err")" || GATE_RC=$?
  else
    GATE_OUT="$(env SCHEDULING_POLICY="$2" bash "$tmp/gate.sh" 2>"$tmp/gate.err")" || GATE_RC=$?
  fi
}

expect_gate_value() { # $1=label $2=unset|set $3=value $4=expected
  run_gate "$2" "$3"
  if [ "$GATE_RC" -eq 0 ] && [ "$GATE_OUT" = "$4" ]; then
    ok "$1"
  else
    bad "$1: rc=$GATE_RC out=$GATE_OUT"
  fi
}

expect_gate_reject() { # $1=label $2=value
  run_gate set "$2"
  if [ "$GATE_RC" -eq 2 ] && [ -z "$GATE_OUT" ]; then
    ok "$1"
  else
    bad "$1: rc=$GATE_RC out=$GATE_OUT"
  fi
}

expect_gate_value "unset defaults to fcfs" unset '' fcfs
expect_gate_value "empty defaults to fcfs" set '' fcfs
expect_gate_value "explicit fcfs accepted" set fcfs fcfs
expect_gate_value "priority accepted" set priority priority
expect_gate_reject "unknown policy rejected" fair
expect_gate_reject "case variant rejected" Priority
expect_gate_reject "embedded newline rejected" "$(printf 'priority\nfcfs')"
expect_gate_reject "shell metacharacters rejected" 'priority; id'

canary="$tmp/injection-canary"
run_gate set "\$(touch $canary)"
if [ "$GATE_RC" -eq 2 ] && [ ! -e "$canary" ]; then
  ok "command substitution rejected without execution"
else
  bad "command substitution handling: rc=$GATE_RC canary=$([ -e "$canary" ] && echo present || echo absent)"
fi

if grep -Fq -- '--scheduling-policy "$${SCHEDULING_POLICY}"' "$COMPOSE"; then
  ok "validated value is passed as one quoted CLI argument"
else
  bad "compose does not pass the validated scheduling policy"
fi

val_env="$tmp/validator.env"
val_bin="$tmp/bin"
mkdir -p "$val_bin"
cat >"$val_bin/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "--scheduling-policy ${SCHEDULING_POLICY:-missing}"
EOF
chmod +x "$val_bin/docker"

write_env() { # optional assignment
  {
    printf '%s\n' \
      'WORKER_HOST=stub-worker' \
      'MASTER_ADDR=127.0.0.1' \
      'MASTER_PORT=29500' \
      'DSPARK_VLLM_IMAGE=stub:latest' \
      'MAX_NUM_SEQS=6' \
      'MAX_NUM_BATCHED_TOKENS=8192' \
      'MTP_NUM_TOKENS=6'
    if [ "$#" -eq 1 ]; then
      printf '%s\n' "$1"
    fi
  } >"$val_env"
}

run_validator() {
  VAL_RC=0
  VAL_OUT="$(
    env -u SCHEDULING_POLICY PATH="$val_bin:$PATH" ENV_FILE="$val_env" \
      COMPOSE_FILE="$COMPOSE" bash "$VALIDATOR" 2>"$tmp/validator.err"
  )" || VAL_RC=$?
  VAL_ERR="$(<"$tmp/validator.err")"
}

validator_accepts() { # $1=label $2=assignment-or-empty $3=expected
  if [ -n "$2" ]; then write_env "$2"; else write_env; fi
  run_validator
  if [ "$VAL_RC" -eq 0 ] \
    && printf '%s\n' "$VAL_OUT" | grep -Fq "scheduling policy: $3" \
    && printf '%s\n' "$VAL_OUT" | grep -Fq -- "--scheduling-policy $3"; then
    ok "$1"
  else
    bad "$1: rc=$VAL_RC err=$VAL_ERR"
  fi
}

validator_accepts "validator defaults to fcfs" '' fcfs
validator_accepts "validator accepts priority" 'SCHEDULING_POLICY=priority' priority

write_env 'SCHEDULING_POLICY=urgent'
run_validator
if [ "$VAL_RC" -eq 2 ] \
  && printf '%s\n' "$VAL_ERR" | grep -Fq 'SCHEDULING_POLICY must be one of: fcfs, priority' \
  && ! printf '%s\n' "$VAL_OUT" | grep -Fq 'DSpark config:'; then
  ok "validator rejects invalid policy before summary/render"
else
  bad "validator invalid policy: rc=$VAL_RC out=$VAL_OUT err=$VAL_ERR"
fi

if grep -Fq 'SCHEDULING_POLICY=fcfs' "$ROOT/.env.dspark.example" \
  && grep -Fq '| `SCHEDULING_POLICY` |' "$ROOT/docs/ENVS.md"; then
  ok "operator template and env reference document the knob"
else
  bad "operator-facing scheduling policy documentation is incomplete"
fi

[ "$QUIET" = "1" ] || printf '%s passed; %s failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
