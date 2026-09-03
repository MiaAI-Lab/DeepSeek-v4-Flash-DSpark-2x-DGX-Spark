#!/usr/bin/env bash
# CPU-only gate for the DRAFT_SAMPLE_METHOD -> --speculative-config boundary.
#
# The compose entrypoint interpolates the env value into JSON. Without a gate,
# a crafted value can use JSON escapes or duplicate keys to stay valid JSON and
# silently rewrite known fields (e.g. num_speculative_tokens), so vLLM never
# sees an invalid config. The entrypoint now resolves the value through a shell
# `case` that accepts exactly probabilistic|greedy before the JSON is built.
#
# This test extracts that gate + the SPECULATIVE_CONFIG assignment from
# docker-compose.dspark.yml itself (no copy of the logic, no docker needed) and
# runs it against the full input matrix: valid values must reproduce the exact
# JSON the compose file used to hardcode; everything else must exit nonzero
# without building a config and without executing embedded shell.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Extract the shipped gate + assignment (entrypoint text uses $$ for literal $).
fragment="$(sed -n '/case "\$\${DRAFT_SAMPLE_METHOD/,/^ *SPECULATIVE_CONFIG=/p' "$COMPOSE" | sed 's/\$\$/$/g')"
if [ -z "$fragment" ] || ! printf '%s' "$fragment" | grep -q 'SPECULATIVE_CONFIG='; then
  echo "FAIL could not extract the DRAFT_SAMPLE_METHOD gate from $COMPOSE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' "$fragment" >"$tmp/fragment.sh"
# Probe appended after the fragment: only reached when the gate accepts.
printf 'printf "%%s" "$SPECULATIVE_CONFIG"\n' >>"$tmp/fragment.sh"

OLD_DEFAULT='{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}'
GREEDY_JSON='{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'

run_case() { # $1=mode(unset|set) $2=value-if-set -> stdout; rc in $RC
  RC=0
  if [ "$1" = "unset" ]; then
    OUT="$(env -u DRAFT_SAMPLE_METHOD -u MTP_NUM_TOKENS bash "$tmp/fragment.sh" 2>/dev/null)" || RC=$?
  else
    OUT="$(env MTP_NUM_TOKENS='' DRAFT_SAMPLE_METHOD="$2" bash "$tmp/fragment.sh" 2>/dev/null)" || RC=$?
  fi
}

expect_valid() { # $1=label $2=mode $3=value $4=want-json
  run_case "$2" "$3"
  if [ "$RC" -eq 0 ] && [ "$OUT" = "$4" ]; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$OUT"
  fi
}

expect_reject() { # $1=label $2=value
  run_case set "$2"
  if [ "$RC" -ne 0 ] && [ -z "$OUT" ]; then
    ok "$1"
  else
    bad "$1: rc=$RC out=$OUT (must exit nonzero with no config)"
  fi
}

expect_valid "unset -> exact old hardcoded JSON" unset '' "$OLD_DEFAULT"
expect_valid "empty -> exact old hardcoded JSON" set '' "$OLD_DEFAULT"
expect_valid "explicit probabilistic -> old JSON" set 'probabilistic' "$OLD_DEFAULT"
expect_valid "greedy -> greedy JSON" set 'greedy' "$GREEDY_JSON"

expect_reject "normal invalid value rejected" 'random'
expect_reject "case variant rejected" 'Greedy'
expect_reject "JSON escape alias rejected" 'gree\u0064y'
expect_reject "duplicate-key payload rejected" 'probabilistic","num_speculative_tokens":9999,"draft_sample_method":"greedy'
expect_reject "embedded newline rejected" "$(printf 'probabilistic\ngreedy')"
expect_reject "shell metacharacters rejected" '$(id); `id`; ;&|'

# Injection probe: a command-substitution payload must not execute.
canary="$tmp/executed-canary"
run_case set "\$(touch $canary)"
if [ "$RC" -ne 0 ] && [ ! -e "$canary" ]; then
  ok "command substitution payload rejected and not executed"
else
  bad "command substitution payload: rc=$RC canary_exists=$([ -e "$canary" ] && echo yes || echo no)"
fi

# ---------------------------------------------------------------------------
# Layer 2: validate-dspark-config.sh, executed rather than grepped.
#
# The pre-flight validator must enforce the same contract before compose runs,
# and must report the *resolved* value rather than a hardcoded one. Both are
# observable by running it against a stub env file, so assert on behavior.
VAL="$ROOT/validate-dspark-config.sh"
val_env="$tmp/val.env"
val_bin="$tmp/val-bin"
mkdir -p "$val_bin"
cat >"$val_bin/docker" <<'EOF'
#!/usr/bin/env bash
printf 'DRAFT_SAMPLE_METHOD=%s\n' "${DRAFT_SAMPLE_METHOD:-}"
EOF
chmod +x "$val_bin/docker"

# Fixtures, not the caller's shell, own both values in Layers 2-3.
export DRAFT_SAMPLE_METHOD=greedy
export MTP_NUM_TOKENS=7

write_val_env() { # $1 = optional DRAFT_SAMPLE_METHOD assignment line
  {
    echo 'WORKER_HOST=stub-worker'
    echo 'MASTER_ADDR=127.0.0.1'
    echo 'MASTER_PORT=29500'
    echo 'DSPARK_VLLM_IMAGE=stub:latest'
    if [ "$#" -eq 1 ]; then printf '%s\n' "$1"; fi
  } >"$val_env"
}

run_val() {
  VRC=0
  VOUT="$(env -u DRAFT_SAMPLE_METHOD -u MTP_NUM_TOKENS PATH="$val_bin:$PATH" ENV_FILE="$val_env" COMPOSE_FILE="$COMPOSE" bash "$VAL" 2>"$tmp/val.err")" || VRC=$?
  VERR="$(cat "$tmp/val.err")"
}

# A deterministic docker stub keeps this validator layer CPU-only while making
# every accepted case require a successful validator exit.
val_accepts() { # $1=label $2=env-line ('' for unset) $3=expected resolved value
  if [ -n "$2" ]; then write_val_env "$2"; else write_val_env; fi
  run_val
  if [ "$VRC" -eq 0 ] && printf '%s\n' "$VOUT" | grep -Fq "draft_sample_method=$3"; then
    ok "$1"
  else
    bad "$1: rc=$VRC resolved=$(printf '%s\n' "$VOUT" | grep -Eo 'draft_sample_method=[^ ]*' || echo none)"
  fi
}

val_rejects() { # $1=label $2=env-line
  write_val_env "$2"
  run_val
  if [ "$VRC" -eq 2 ] \
    && printf '%s\n' "$VERR" | grep -Fq 'DRAFT_SAMPLE_METHOD must be one of: probabilistic, greedy' \
    && ! printf '%s\n' "$VOUT" | grep -Fq 'draft_sample_method='; then
    ok "$1"
  else
    bad "$1: rc=$VRC (want 2) err=$VERR"
  fi
}

val_accepts "validator: unset resolves to probabilistic" '' 'probabilistic'
val_accepts "validator: empty resolves to probabilistic" 'DRAFT_SAMPLE_METHOD=' 'probabilistic'
val_accepts "validator: explicit probabilistic reported as probabilistic" 'DRAFT_SAMPLE_METHOD=probabilistic' 'probabilistic'
val_accepts "validator: greedy reported as greedy (not the old hardcoded value)" 'DRAFT_SAMPLE_METHOD=greedy' 'greedy'
val_rejects "validator: invalid enum exits 2 before any summary" 'DRAFT_SAMPLE_METHOD=random'
val_rejects "validator: case variant exits 2" 'DRAFT_SAMPLE_METHOD=Greedy'
val_rejects "validator: duplicate-key payload exits 2" 'DRAFT_SAMPLE_METHOD='"'"'probabilistic","num_speculative_tokens":9999,"draft_sample_method":"greedy'"'"''

# ---------------------------------------------------------------------------
# Layer 3: the real rendered Compose expansion + entrypoint gate.
#
# Layer 1 runs a fragment lifted out of the compose file with sed, which proves
# the gate logic but not that a value survives `.env` parsing and `${VAR:-...}`
# interpolation intact. Compose applies its own escape processing (a
# double-quoted `\n` in .env becomes a real newline), so the value the
# entrypoint sees is not always the value the operator typed. This layer writes
# a real .env, renders it with `docker compose config`, takes the gate straight
# out of the rendered entrypoint, and runs it under the rendered environment.
if docker compose version >/dev/null 2>&1; then
  cat >"$tmp/rendered.py" <<'PYEOF'
import json, os, subprocess, sys

compose, envfile, want_rc, want_out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

env = {k: v for k, v in os.environ.items() if k not in ("NODE_RANK", "HEADLESS", "DRAFT_SAMPLE_METHOD", "MTP_NUM_TOKENS")}
env.update(COMPOSE_DISABLE_ENV_FILE="1", NODE_RANK="0")
p = subprocess.run(
    ["docker", "compose", "--env-file", envfile, "-f", compose, "config", "--format", "json"],
    capture_output=True, text=True, env=env, cwd=os.path.dirname(compose),
)
if p.returncode != 0:
    print("render failed: " + p.stderr.strip().splitlines()[-1:][0] if p.stderr.strip() else "render failed")
    sys.exit(1)

svc = list(json.loads(p.stdout)["services"].values())[0]
rendered_env, script = svc["environment"], svc["command"][2]

start = script.find('case "$${DRAFT_SAMPLE_METHOD')
spec = script.find("SPECULATIVE_CONFIG=", start)
end = script.find('; case "$${DEFAULT_THINKING', spec)
if min(start, spec, end) < 0:
    print("could not locate the gate in the rendered entrypoint")
    sys.exit(1)
# `$$` is compose's escape for a literal `$`; the container gets a single `$`.
gate = script[start:end + 1].replace("$$", "$") + '\nprintf %s "$SPECULATIVE_CONFIG"\n'

run_env = {"PATH": env.get("PATH", "")}
for key in ("DRAFT_SAMPLE_METHOD", "MTP_NUM_TOKENS"):
    val = rendered_env.get(key)
    if val is not None:
        run_env[key] = val

r = subprocess.run(["bash", "-c", gate], capture_output=True, text=True, env=run_env)
if r.returncode == want_rc and (want_rc != 0 or r.stdout == want_out) and (want_rc == 0 or not r.stdout):
    sys.exit(0)
print("rc=%d (want %d) rendered=%r out=%r" % (r.returncode, want_rc, rendered_env.get("DRAFT_SAMPLE_METHOD"), r.stdout))
sys.exit(1)
PYEOF

  rendered_case() { # $1=label $2=env-line ('' for unset) $3=want-rc $4=want-json
    if [ -n "$2" ]; then write_val_env "$2"; else write_val_env; fi
    if detail="$(python3 "$tmp/rendered.py" "$COMPOSE" "$val_env" "$3" "$4" 2>&1)"; then
      ok "$1"
    else
      bad "$1: $detail"
    fi
  }

  rendered_case "rendered: unset -> exact old hardcoded JSON" '' 0 "$OLD_DEFAULT"
  rendered_case "rendered: empty -> exact old hardcoded JSON" 'DRAFT_SAMPLE_METHOD=' 0 "$OLD_DEFAULT"
  rendered_case "rendered: probabilistic -> old JSON" 'DRAFT_SAMPLE_METHOD=probabilistic' 0 "$OLD_DEFAULT"
  rendered_case "rendered: greedy -> greedy JSON" 'DRAFT_SAMPLE_METHOD=greedy' 0 "$GREEDY_JSON"
  rendered_case "rendered: .env double quotes stripped, greedy still accepted" 'DRAFT_SAMPLE_METHOD="greedy"' 0 "$GREEDY_JSON"
  rendered_case "rendered: invalid enum rejected" 'DRAFT_SAMPLE_METHOD=random' 2 ''
  rendered_case "rendered: compose-expanded newline escape rejected" 'DRAFT_SAMPLE_METHOD="probabilistic\ngreedy"' 2 ''
  rendered_case "rendered: backslash value rejected" 'DRAFT_SAMPLE_METHOD=gree\dy' 2 ''
  rendered_case "rendered: JSON escape alias rejected" "DRAFT_SAMPLE_METHOD='gree\\u0064y'" 2 ''
  rendered_case "rendered: embedded double quote rejected" "DRAFT_SAMPLE_METHOD='gre\"edy'" 2 ''
  rendered_case "rendered: duplicate-key payload rejected" 'DRAFT_SAMPLE_METHOD='"'"'probabilistic","num_speculative_tokens":9999,"draft_sample_method":"greedy'"'"'' 2 ''
else
  say "SKIP rendered-compose layer (docker compose unavailable); layers 1-2 still cover the contract"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
