#!/usr/bin/env bash
# CPU-only gate for the kv-disk-tier default-off isolation contract (PR #220).
#
#   1. switch off, legacy .env.dspark (predates KV_DISK_CACHE_ENABLE): the REAL
#      launcher must live under `set -u` and reach the head `docker compose`
#      call with the operator's COMPOSE_FILE as the only -f, verbatim (a path
#      with spaces stays one argv element);
#   2. switch on: the same base file plus docker-compose.dspark-disk-tier
#      .override.yml, in that order;
#   3. rendered service env (needs docker compose; skipped with a note where
#      unavailable): off renders the stock environment, on adds exactly the
#      documented tier keys and the tier mounts.
#
# Runs the shipped start script under argv recorders for docker/ssh/scp/ip;
# no container, GPU, network, or .env.dspark in the repo is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
START="$ROOT/start-deepseek-v4-flash-dspark.sh"
BASE_COMPOSE="$ROOT/docker-compose.dspark.yml"
TIER_OVERRIDE="$ROOT/docker-compose.dspark-disk-tier.override.yml"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/compose dir"
REAL_DOCKER="$(command -v docker || true)"

# --- recorders ---------------------------------------------------------------
# docker: log every argv (tab-separated, one call per line) and abort the run
# at the first head compose call (compose_base always passes --env-file), so
# nothing later in the launcher needs faking. Version/probe calls pass through.
cat >"$tmp/bin/docker" <<'SH'
#!/usr/bin/env bash
{ printf 'CALL'; for a in "$@"; do printf '\t%s' "$a"; done; printf '\n'; } >>"$KV_GATE_LOG"
if [ "${1:-}" = "compose" ]; then
  for a in "$@"; do [ "$a" = "--env-file" ] && exit 42; done
fi
exit 0
SH
for cmd in ssh scp; do
  cat >"$tmp/bin/$cmd" <<'SH'
#!/usr/bin/env bash
exit 0
SH
done
cat >"$tmp/bin/ip" <<'SH'
#!/usr/bin/env bash
echo "1: lo    inet 127.0.0.1/8 scope host lo"
SH
cat >"$tmp/bin/curl" <<'SH'
#!/usr/bin/env bash
exit 0
SH
# ss: the head port preflight reads real listeners; an empty list is the
# "nothing conflicting" answer the test needs to reach compose.
cat >"$tmp/bin/ss" <<'SH'
#!/usr/bin/env bash
exit 0
SH
chmod +x "$tmp/bin/"*
export PATH="$tmp/bin:$PATH"

# --- fixture -----------------------------------------------------------------
# Minimal two-node config that reaches compose file selection. NCCL_IB_GID_AUTO=0
# keeps the GID resolver on its pinned-index branch (no sysfs probing).
CUSTOM_COMPOSE="$tmp/compose dir/docker-compose custom.yml"
cp "$BASE_COMPOSE" "$CUSTOM_COMPOSE"

legacy_env="$tmp/legacy.env"
cat >"$legacy_env" <<EOF
MASTER_ADDR=10.0.0.1
MASTER_PORT=29500
NCCL_IB_HCA=rocep1s0f0
NCCL_SOCKET_IFNAME=enp1s0f0np0
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
WORKER_HOST=gate-worker.invalid
NCCL_IB_GID_AUTO=0
NCCL_IB_GID_INDEX=3
WORKER_NCCL_IB_GID_INDEX=3
DSPARK_BOOT_SHAPE_WARMUP=0
EOF
enabled_env="$tmp/enabled.env"
{
  cat "$legacy_env"
  echo "KV_DISK_CACHE_ENABLE=1"
  echo "KV_DISK_CACHE_SRC=$tmp/dsv4-kv"
  echo "KV_DISK_CACHE_DIR=$tmp/kvdisk"
} >"$enabled_env"

# run_launcher <env-file> <log> -> rc; stdout+stderr in $RUN_OUT
run_launcher() {
  local env_file="$1" log="$2"
  : >"$log"
  set +e
  RUN_OUT="$(env -u KV_DISK_CACHE_ENABLE KV_GATE_LOG="$log" ENV_FILE="$env_file" \
    COMPOSE_FILE="$CUSTOM_COMPOSE" PROJECT_NAME=kv-gate-proj WAIT_ATTEMPTS=1 \
    bash "$START" 2>&1)"
  RUN_RC=$?
  set -e
}

# head_args <log> -> one argv element per line for the first head compose call
# (the first recorded call carrying --env-file, i.e. compose_base).
head_args() {
  awk -F'\t' '
    {
      is_head = 0
      for (i = 2; i <= NF; i++) if ($i == "--env-file") is_head = 1
      if (!is_head) next
      for (i = 2; i <= NF; i++) print $i
      exit
    }' "$1"
}

# file_args <log> -> the -f values of the head compose call, one per line.
file_args() {
  head_args "$1" | awk '/^-f$/ { getline; print }'
}

# --- 1. switch off, legacy .env.dspark ---------------------------------------
run_launcher "$legacy_env" "$tmp/off.log"

if [ -s "$tmp/off.log" ]; then
  ok "off: launcher reaches the head docker compose call (no set -u exit on the unset knob)"
else
  bad "off: launcher never invoked docker compose — output was:
$(printf '%s\n' "$RUN_OUT" | sed 's/^/      /')"
fi
if printf '%s' "$RUN_OUT" | grep -q 'unbound variable'; then
  bad "off: launcher died on an unbound variable"
fi

mapfile -t off_files < <(file_args "$tmp/off.log")
mapfile -t off_head < <(head_args "$tmp/off.log" | head -5)
if [ "${off_head[0]:-}" = "compose" ]; then
  ok "off: recorded head call is a docker compose invocation"
else
  bad "off: head docker call is not compose: ${off_head[*]:-<none>}"
fi
if [ "${#off_files[@]}" -eq 1 ] && [ "${off_files[0]}" = "$CUSTOM_COMPOSE" ]; then
  ok "off: exactly one compose file, the configured path verbatim ($CUSTOM_COMPOSE)"
else
  bad "off: expected exactly [$CUSTOM_COMPOSE], got [${off_files[*]:-<none>}]"
fi
if [ "$RUN_RC" -eq 42 ]; then
  ok "off: the run stopped at the recorded compose call (stub sentinel)"
else
  bad "off: expected the stub sentinel 42 as exit status, got $RUN_RC:
$(printf '%s\n' "$RUN_OUT" | tail -20 | sed 's/^/      /')"
fi

# --- 2. switch on ------------------------------------------------------------
run_launcher "$enabled_env" "$tmp/on.log"

mapfile -t on_files < <(file_args "$tmp/on.log")
if [ "${#on_files[@]}" -eq 2 ] \
  && [ "${on_files[0]}" = "$CUSTOM_COMPOSE" ] \
  && [ "${on_files[1]}" = "$TIER_OVERRIDE" ]; then
  ok "on: base file stays first, tier override appended as a second -f"
else
  bad "on: expected [$CUSTOM_COMPOSE, $TIER_OVERRIDE], got [${on_files[*]:-<none>}]"
fi

# --- 3. rendered service env -------------------------------------------------
# What the container actually receives, through compose's own merge/interpolation.
if [ -z "$REAL_DOCKER" ] || ! "$REAL_DOCKER" compose version >/dev/null 2>&1; then
  say "rendered-config layer skipped: real docker compose unavailable"
else
  render() { # <env-file> <override?> -> JSON
    local env_file="$1" with_override="$2"
    local args=(-f "$BASE_COMPOSE")
    [ "$with_override" = "1" ] && args+=(-f "$TIER_OVERRIDE")
    "$REAL_DOCKER" compose "${args[@]}" -p kv-gate-proj --env-file "$env_file" \
      config --format json 2>/dev/null
  }
  if ! render "$legacy_env" 0 >"$tmp/off.json" || ! [ -s "$tmp/off.json" ]; then
    say "rendered-config layer skipped: docker compose config failed here"
  else
    render "$enabled_env" 1 >"$tmp/on.json"
    if python3 - "$tmp/off.json" "$tmp/on.json" "$tmp/dsv4-kv" "$tmp/kvdisk" <<'PY'
import json, sys

off = json.load(open(sys.argv[1]))["services"]["vllm-dspark"]
on = json.load(open(sys.argv[2]))["services"]["vllm-dspark"]
TIER_SRC, TIER_DIR = sys.argv[3], sys.argv[4]

# The tier delta the override is allowed to add, and nothing else.
DELTA = {
    "KV_DISK_CACHE_ENABLE": "1",
    "KV_DISK_CACHE_CPU_BYTES": "4294967296",
    "KV_DISK_CACHE_BYTES": "150000000000",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_SKIP_INIT_MEMORY_CHECK": "1",
    "KV_DISK_CACHE_SHARD_HEAD": "tcp://10.0.0.1:25055",
    "KV_DISK_CACHE_SHARD_PORT": "25055",
    "KV_DISK_CACHE_SG_THRESHOLD": "20000",
    "KV_DISK_CACHE_SG_SO": "/usr/local/lib/libdsv4_batch_copy.so",
    "KV_DISK_CACHE_MAX_COPIES_PER_BATCH": "8192",
    "KV_DISK_CACHE_MAX_OFFLOAD_BLOCKS_PER_REQUEST": "0",
    "KV_DISK_CACHE_DIRECT_IO": "0",
    "KV_DISK_CACHE_DIRECT_IO_SO": "/usr/local/lib/libdsv4_host_kv.so",
    "KV_DISK_CACHE_DIRECT_VERIFY": "0",
}
off_env, on_env = off["environment"], on["environment"]


def tier_sources(service):
    return sorted(
        str(v.get("source", ""))
        for v in service["volumes"]
        if str(v.get("source", "")).startswith(TIER_SRC + "/")
        or str(v.get("source", "")) == TIER_DIR
    )


bad = []
for key, value in DELTA.items():
    if key in off_env:
        bad.append(f"off render still carries {key}={off_env[key]}")
    if on_env.get(key) != value:
        bad.append(f"on render {key}={on_env.get(key)!r}, want {value!r}")

added = {k: v for k, v in on_env.items() if off_env.get(k) != v}
if added != DELTA:
    for key in sorted(set(added) | set(DELTA)):
        if added.get(key) != DELTA.get(key):
            bad.append(
                f"delta mismatch {key}: render={added.get(key)!r} documented={DELTA.get(key)!r}"
            )

off_tier, on_tier = tier_sources(off), tier_sources(on)
if off_tier:
    bad.append(f"off render already mounts the tier: {off_tier}")
if len(on_tier) != 7:
    bad.append(f"on render has {len(on_tier)} tier mounts, want 7: {on_tier}")

if bad:
    print("\n".join(bad))
    sys.exit(1)
print(
    f"off env stock ({len(off_env)} keys, {len(off['volumes'])} mounts), "
    f"on env = off + {len(DELTA)} documented tier keys, +{len(on_tier)} tier mounts"
)
PY
    then
      ok "rendered env: off adds no tier key, on adds exactly the documented delta"
    else
      bad "rendered env: delta check failed (see lines above)"
    fi
  fi
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
