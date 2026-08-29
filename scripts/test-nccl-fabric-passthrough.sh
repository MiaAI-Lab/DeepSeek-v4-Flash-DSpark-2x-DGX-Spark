#!/usr/bin/env bash
# CPU-only gate for the NCCL fabric passthrough contract:
#   - a knob NOT configured in .env.dspark must be truly ABSENT in the exec'd
#     serving process (NCCL loads config-file values with overwrite=0, so a
#     defined-but-empty variable would mask /etc/nccl.conf / NCCL_CONF_FILE);
#   - a configured value must pass through UNCHANGED.
#
# Compose cannot conditionally omit a map key, so the entrypoint unsets empty
# definitions before exec. This test extracts that shipped normalization from
# docker-compose.dspark.yml (not a copy of it) and asserts both directions
# behaviorally, then checks the wiring: the map entries exist and the
# normalization runs before the exec line. If docker compose is available, the
# rendered config is checked too; otherwise that layer is skipped with a note.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.dspark.yml"
QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

KNOBS=(NCCL_IB_MERGE_NICS NCCL_NET_GDR_LEVEL NCCL_NET_GDR_READ NCCL_DMABUF_ENABLE)

pass=0
fail=0
say() { [ "$QUIET" = "1" ] || printf '  ok  %s\n' "$*"; }
ok() { pass=$((pass + 1)); say "$*"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$*" >&2; }

# Extract the shipped normalization block (entrypoint text uses $$ for $).
fragment="$(sed -n '/if \[ -z "\$\${NCCL_IB_MERGE_NICS/,/unset NCCL_DMABUF_ENABLE; fi;$/p' "$COMPOSE" | sed 's/\$\$/$/g')"
if [ "$(printf '%s\n' "$fragment" | grep -c 'unset NCCL_')" -ne 4 ]; then
  echo "FAIL could not extract the 4-knob unset normalization from $COMPOSE" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
{
  printf '%s\n' "$fragment"
  # Probe: one line per knob — "NAME=<value>" if defined, "NAME absent" if not.
  for k in "${KNOBS[@]}"; do
    printf 'if [ -z "${%s+x}" ]; then echo "%s absent"; else echo "%s=${%s}"; fi\n' "$k" "$k" "$k" "$k"
  done
} >"$tmp/fragment.sh"

# Direction 1: empty is normalized to absent. Defined-but-empty is exactly what
# compose injects when .env leaves the knob unconfigured; it must end up unset,
# never forwarded as an empty setting.
out="$(env NCCL_IB_MERGE_NICS='' NCCL_NET_GDR_LEVEL='' NCCL_NET_GDR_READ='' NCCL_DMABUF_ENABLE='' bash "$tmp/fragment.sh")"
want="$(printf '%s absent\n' "${KNOBS[@]}")"
if [ "$out" = "$want" ]; then
  ok "empty is normalized to absent (defined-empty from compose becomes truly unset)"
else
  bad "unconfigured knobs still defined after normalization: $out"
fi

# Direction 2: configured non-empty values pass through byte-identical.
out="$(env NCCL_IB_MERGE_NICS='0' NCCL_NET_GDR_LEVEL='SYS' NCCL_NET_GDR_READ='1' NCCL_DMABUF_ENABLE='0' bash "$tmp/fragment.sh")"
want="$(printf '%s\n' 'NCCL_IB_MERGE_NICS=0' 'NCCL_NET_GDR_LEVEL=SYS' 'NCCL_NET_GDR_READ=1' 'NCCL_DMABUF_ENABLE=0')"
if [ "$out" = "$want" ]; then
  ok "configured non-empty values pass through unchanged"
else
  bad "configured knobs altered: $out"
fi

# Contract boundary: only *empty* is normalized. A value that is non-empty but
# unusual (whitespace, punctuation, mixed case) is still a configured value and
# must reach NCCL byte-identical rather than being trimmed or dropped.
out="$(env NCCL_IB_MERGE_NICS=' ' NCCL_NET_GDR_LEVEL='PHB ' NCCL_NET_GDR_READ='0' NCCL_DMABUF_ENABLE='Yes' bash "$tmp/fragment.sh")"
want="$(printf '%s\n' 'NCCL_IB_MERGE_NICS= ' 'NCCL_NET_GDR_LEVEL=PHB ' 'NCCL_NET_GDR_READ=0' 'NCCL_DMABUF_ENABLE=Yes')"
if [ "$out" = "$want" ]; then
  ok "non-empty boundary values (whitespace/case preserved) pass through unchanged"
else
  bad "non-empty boundary values altered: $out"
fi

# Mixed: one configured, three unconfigured.
out="$(env NCCL_IB_MERGE_NICS='' NCCL_NET_GDR_LEVEL='LOC' NCCL_NET_GDR_READ='' NCCL_DMABUF_ENABLE='' bash "$tmp/fragment.sh")"
want="$(printf '%s\n' 'NCCL_IB_MERGE_NICS absent' 'NCCL_NET_GDR_LEVEL=LOC' 'NCCL_NET_GDR_READ absent' 'NCCL_DMABUF_ENABLE absent')"
if [ "$out" = "$want" ]; then
  ok "mixed configuration: set stays set, unset stays absent"
else
  bad "mixed configuration wrong: $out"
fi

# Wiring: every knob must be declared in the compose environment map (so the
# .env value reaches the container at all) and the normalization must run
# before the exec line hands off to vLLM.
wiring_missing=""
for k in "${KNOBS[@]}"; do
  grep -Fq "$k: \"\${$k:-}\"" "$COMPOSE" || wiring_missing="$wiring_missing $k"
done
if [ -z "$wiring_missing" ]; then
  ok "compose declares all four knobs in the environment map"
else
  bad "compose environment map missing:$wiring_missing"
fi
unset_line="$(grep -n 'unset NCCL_DMABUF_ENABLE' "$COMPOSE" | head -1 | cut -d: -f1)"
exec_line="$(grep -n '/usr/local/bin/vllm serve' "$COMPOSE" | head -1 | cut -d: -f1)"
if [ -n "$unset_line" ] && [ -n "$exec_line" ] && [ "$unset_line" -lt "$exec_line" ]; then
  ok "normalization runs in the entrypoint before exec vllm"
else
  bad "normalization not found before the exec line (unset=$unset_line exec=$exec_line)"
fi

# Rendered-config layer (needs docker compose; skip quietly where absent).
if docker compose version >/dev/null 2>&1; then
  envfile="$tmp/env"
  {
    echo "NCCL_IB_HCA=stub0"
    echo "NCCL_SOCKET_IFNAME=stub0"
    echo "NCCL_NET_GDR_LEVEL=SYS"
  } >"$envfile"
  rendered="$(cd "$ROOT" && env -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 \
    docker compose --env-file "$envfile" -f "$COMPOSE" config 2>/dev/null || true)"
  if printf '%s' "$rendered" | grep -Eq 'NCCL_NET_GDR_LEVEL: *"?SYS"?' \
    && printf '%s' "$rendered" | grep -Eq 'NCCL_IB_MERGE_NICS: *""'; then
    ok "docker compose config renders configured=verbatim, unconfigured=empty (then unset by entrypoint)"
  else
    bad "rendered compose config does not show the expected environment values"
  fi
else
  say "SKIP rendered-config check (docker compose unavailable); fragment checks above still cover the contract"
fi

printf 'RESULT: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
