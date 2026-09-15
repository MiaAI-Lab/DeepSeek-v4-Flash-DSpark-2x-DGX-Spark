#!/usr/bin/env bash
# CPU-only behavioral gates for the non-root runtime/JIT cache ownership
# contract of prepare-dspark-model-cache.sh:
#
#   * a legacy unwritable cache path stops prepare before any download, keeps
#     the documented fail-closed message, and no ownership is mutated behind
#     the operator's back;
#   * after the one-time migration (or on a fresh install) prepare proceeds as
#     DSPARK_RUNTIME_UID:GID with the same two --user invocations and no
#     ownership change of its own;
#   * the explicit `--migrate-runtime-cache-ownership` mode refuses unsafe
#     roots, symlinks (target or nested), non-directories, and a
#     DSPARK_TMP_HOST that would drag the checkpoint tree in, while its plan
#     covers exactly the seven named caches plus DSPARK_TMP_HOST and never
#     lists HF_CACHE/hub.
#
# The suite runs unprivileged on purpose: CI and ordinary developer runs are
# non-root, so "a root:root 0755 legacy cache directory" is simulated with the
# 0555 directory the runtime identity observes the same way — `test -w` fails,
# which is the single observation the preflight makes. Real chown (root-only
# CAP_CHOWN) is never exercised here; the migration's own plan, refusals and
# no-mutation guarantees are what this suite pins.
set -euo pipefail
unset BASH_ENV

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREPARE="$ROOT/prepare-dspark-model-cache.sh"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

if [ "$(id -u)" = "0" ]; then
  echo "skip: runtime cache ownership suite needs an unprivileged user (the 0555 legacy simulation is meaningless as root)"
  exit 0
fi

UID_="$(id -u)"
GID_="$(id -g)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

names=(runtime-home flashinfer tilelang-cache triton-cache b12x-cute-cache vllm-cache nccl-fr)

# docker stub: records argv, drains the stdin the heredoc-fed `docker run`s pipe
# in, and never touches an image (the CPU lane has no runtime image).
mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${DOCKER_LOG:?}"
case " $* " in *" -i "*) cat >/dev/null ;; esac
exit 0
EOF
chmod +x "$tmp/bin/docker"
export DOCKER_LOG="$tmp/docker.log"
: > "$DOCKER_LOG"

# Per-case path overrides (the refusal cases point the migration at unsafe
# roots); reset by refusal_case.
HF_OVERRIDE=""
TMP_OVERRIDE=""

scenario() {
  # scenario <mode: fresh|existing|legacy>
  rm -rf "$tmp/hf" "$tmp/tmpdir"
  mkdir -p "$tmp/tmpdir"
  case "$1" in
    fresh) ;;
    existing|legacy)
      mkdir -p "$tmp/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
      : > "$tmp/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/config.json"
      for n in "${names[@]}"; do mkdir -p "$tmp/hf/$n"; done
      if [ "$1" = "legacy" ]; then chmod 0555 "$tmp/hf/flashinfer"; fi
      ;;
  esac
  : > "$DOCKER_LOG"
}

# Run the real script with only the environment it needs.
prepare() {
  env -i PATH="$tmp/bin:$PATH" HOME="$tmp/home" ENV_FILE=/dev/null \
    DOCKER_LOG="$DOCKER_LOG" HF_CACHE="$tmp/hf" DSPARK_TMP_HOST="$tmp/tmpdir" \
    DSPARK_RUNTIME_UID="$UID_" DSPARK_RUNTIME_GID="$GID_" PREPARE_WORKER=0 \
    bash "$PREPARE" "$@"
}

migrate() {
  env -i PATH="$PATH" HOME="$tmp/home" ENV_FILE=/dev/null \
    HF_CACHE="${HF_OVERRIDE:-$tmp/hf}" DSPARK_TMP_HOST="${TMP_OVERRIDE:-$tmp/tmpdir}" \
    DSPARK_RUNTIME_UID="$UID_" DSPARK_RUNTIME_GID="$GID_" \
    bash "$PREPARE" --migrate-runtime-cache-ownership "$@"
}

# Ownership+mode fingerprint of the runtime paths, for the no-silent-mutation
# assertions. Missing paths are recorded as absent, which is itself a change.
fingerprint() {
  local n seen=""
  for n in "${names[@]}"; do
    seen="$seen $n=$(stat -c '%u:%g:%a' "$tmp/hf/$n" 2>/dev/null || echo absent)"
  done
  printf 'root=%s tmp=%s%s\n' "$(stat -c '%u:%g:%a' "$tmp/hf" 2>/dev/null || echo absent)" \
    "$(stat -c '%u:%g:%a' "$tmp/tmpdir" 2>/dev/null || echo absent)" "$seen"
}

contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }

# --- 1. migration without root refuses; only the plan is allowed unprivileged
migrate 2>"$tmp/err" >"$tmp/out" && rc=0 || rc=$?
if [ "$rc" = "2" ] \
  && contains "$(cat "$tmp/err")" "Ownership migration needs CAP_CHOWN" \
  && contains "$(cat "$tmp/err")" "sudo ./prepare-dspark-model-cache.sh --migrate-runtime-cache-ownership"; then
  ok "unprivileged migration refuses with the sudo invocation"
else
  bad "unprivileged migration must exit 2 with the CAP_CHOWN/sudo message (rc=$rc)"
fi
if ! contains "$(cat "$tmp/out")" "would chown"; then
  ok "unprivileged non-dry migration changes nothing"
else
  bad "unprivileged migration emitted a chown plan without --dry-run"
fi

# --- 2. prepare refuses a legacy unwritable runtime path, fail-closed
scenario legacy
before="$(fingerprint)"
prepare --yes >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
after="$(fingerprint)"
err="$(cat "$tmp/err")"
if [ "$rc" = "1" ] && contains "$err" "Runtime path must be writable by $UID_:$GID_: $tmp/hf/flashinfer"; then
  ok "prepare refuses the legacy unwritable cache path with the documented message"
else
  bad "prepare must exit 1 naming $tmp/hf/flashinfer as unwritable (rc=$rc)"
fi
if contains "$err" "Fix legacy root-owned cache paths before retrying; do not run this downloader with sudo." \
  && contains "$err" "sudo ./prepare-dspark-model-cache.sh --migrate-runtime-cache-ownership"; then
  ok "refusal message carries the migration remediation"
else
  bad "refusal message must keep the no-sudo warning and point at --migrate-runtime-cache-ownership"
fi
if [ "$before" = "$after" ]; then
  ok "refused prepare mutated no ownership or mode"
else
  bad "refused prepare changed the cache tree: $before -> $after"
fi
if [ ! -s "$DOCKER_LOG" ]; then
  ok "refused prepare never invoked docker (no download attempted)"
else
  bad "refused prepare must not reach the download step: $(tr '\n' ';' < "$DOCKER_LOG")"
fi

# --- 3. prepare proceeds once the runtime paths are writable (migrated state)
scenario existing
before="$(fingerprint)"
prepare --yes >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
after="$(fingerprint)"
if [ "$rc" = "0" ]; then
  ok "prepare proceeds on runtime paths owned by the runtime identity"
else
  bad "prepare must exit 0 on a writable cache tree (rc=$rc): $(tail -3 "$tmp/err")"
fi
if [ "$(grep -Fc -- "--user $UID_:$GID_" "$DOCKER_LOG")" = "2" ]; then
  ok "both download and verify containers run as $UID_:$GID_"
else
  bad "expected exactly two --user $UID_:$GID_ invocations: $(grep -c -- '--user' "$DOCKER_LOG" || true)"
fi
if [ "$before" = "$after" ]; then
  ok "successful prepare mutated no ownership or mode"
else
  bad "successful prepare changed the cache tree: $before -> $after"
fi
if [ -d "$tmp/hf/runtime-home/.cache" ] && [ -w "$tmp/hf/runtime-home/.cache" ]; then
  ok "runtime-home/.cache is created for XDG_CACHE_HOME"
else
  bad "prepare must create the writable runtime-home/.cache"
fi

# --- 4. fresh install: nothing to migrate, prepare creates the named paths
scenario fresh
migrate --dry-run >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
if [ "$rc" = "0" ] && contains "$(cat "$tmp/out")" "nothing to migrate"; then
  ok "fresh install reports nothing to migrate"
else
  bad "fresh install migration must be a no-op success (rc=$rc): $(cat "$tmp/err")"
fi
prepare --yes >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
missing=""
for n in "${names[@]}"; do
  [ -d "$tmp/hf/$n" ] || missing="$missing $n"
done
if [ "$rc" = "0" ] && [ -z "$missing" ] && [ -d "$tmp/tmpdir" ]; then
  ok "fresh prepare creates all seven named caches plus DSPARK_TMP_HOST"
else
  bad "fresh prepare must create the runtime paths (rc=$rc missing:$missing)"
fi

# --- 5. migration plan is exactly the seven named caches plus DSPARK_TMP_HOST
scenario existing
mkdir -p "$tmp/hf/hub" && : > "$tmp/hf/hub/index.json"
before="$(fingerprint)"
migrate --dry-run >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
after="$(fingerprint)"
plan="$(cat "$tmp/out")"
if [ "$rc" = "0" ] && [ "$(grep -c '^would chown -R' "$tmp/out")" = "8" ]; then
  ok "plan recurses into the seven named caches and DSPARK_TMP_HOST"
else
  bad "plan must list exactly 8 recursive targets (rc=$rc, got $(grep -c '^would chown -R' "$tmp/out" || true))"
fi
if contains "$plan" "would chown (cache root directory entry only, checkpoint tree untouched): $tmp/hf"; then
  ok "plan chowns only the cache root directory entry"
else
  bad "plan must state the cache root entry is the only non-recursive target"
fi
if grep -Eq '^would chown.*/hub' "$tmp/out"; then
  bad "plan must not target HF_CACHE/hub"
else
  ok "plan never includes the checkpoint tree"
fi
if contains "$plan" "dry run: no ownership changed" && [ "$before" = "$after" ]; then
  ok "dry run reports no ownership change and makes none"
else
  bad "dry run must report and make no ownership change: $before -> $after"
fi
if contains "$plan" "$tmp/hf/hub is owned by $UID_:$GID_"; then
  bad "checkpoint-tree note must only appear when its owner differs"
else
  ok "plan says nothing about an already-matching checkpoint tree"
fi

# --- 6. refusals: every unsafe shape exits 2 before any mutation
refusal_case() {
  # refusal_case <label> <expected message substring> <setup>
  local label="$1" expect="$2" setup="$3" rc before
  scenario existing
  eval "$setup"
  before="$(fingerprint)"
  migrate --dry-run >"$tmp/out" 2>"$tmp/err" && rc=0 || rc=$?
  if [ "$rc" = "2" ] && contains "$(cat "$tmp/err")" "$expect"; then
    ok "refuses $label"
  else
    bad "must refuse $label with '$expect' (rc=$rc): $(cat "$tmp/err")"
  fi
  if [ "$before" = "$(fingerprint)" ]; then
    ok "refusal for $label changed nothing"
  else
    bad "refusal for $label mutated the tree"
  fi
  HF_OVERRIDE=""
  TMP_OVERRIDE=""
}

refusal_case "a symlinked cache directory" "symbolic link (pass the real path): $tmp/hf/flashinfer" \
  'rm -rf "$tmp/hf/flashinfer"; ln -s "$tmp/outside" "$tmp/hf/flashinfer"'
refusal_case "a symlink nested in a cache directory" "symbolic link inside $tmp/hf/triton-cache" \
  'ln -s /etc/hostname "$tmp/hf/triton-cache/escape"'
refusal_case "a cache path that is not a directory" "not a directory" \
  'rm -rf "$tmp/hf/vllm-cache"; : > "$tmp/hf/vllm-cache"'
refusal_case "a top-level HF_CACHE" "unsafe root (top-level system directory): /" \
  'HF_OVERRIDE=/'
refusal_case "a top-level DSPARK_TMP_HOST" "unsafe root (top-level system directory): /tmp" \
  'TMP_OVERRIDE=/tmp'
refusal_case "a DSPARK_TMP_HOST containing the cache root" "DSPARK_TMP_HOST contains the cache root" \
  'TMP_OVERRIDE=$tmp'

echo ""
if [ "$fail" -ne 0 ]; then
  echo "$fail check(s) failed, $pass passed" >&2
  exit 1
fi
echo "runtime cache ownership gates passed ($pass checks)"
