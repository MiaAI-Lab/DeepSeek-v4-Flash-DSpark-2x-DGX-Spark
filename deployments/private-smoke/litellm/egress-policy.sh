#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
COMMENT="dspark-smoke-litellm-egress"
IPTABLES_HELPER_IMAGE="ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8"

iptables_cmd() {
  if sudo -n true 2>/dev/null; then
    sudo iptables "$@"
    return
  fi
  command -v docker >/dev/null || {
    echo "iptables requires passwordless sudo or the local Docker daemon" >&2
    return 1
  }
  docker run --rm --network none --privileged --pid host --entrypoint nsenter \
    "$IPTABLES_HELPER_IMAGE" -t 1 -m -n -- /usr/sbin/iptables "$@"
}

rule() {
  local action="$1" source="$2" destination="$3" port="${4:-}"
  local args=(INPUT -s "$source" -d "$destination")
  [ -z "$port" ] || args+=(-p tcp --dport "$port")
  args+=(-m comment --comment "$COMMENT" -j "$action")
  printf '%s\n' "${args[@]}"
}

apply_rule() {
  local action="$1" source="$2" destination="$3" port="${4:-}"
  local args=(INPUT -s "$source" -d "$destination")
  [ -z "$port" ] || args+=(-p tcp --dport "$port")
  args+=(-m comment --comment "$COMMENT" -j "$action")
  iptables_cmd -C "${args[@]}" 2>/dev/null || iptables_cmd -I "${args[@]}"
}

check_rule() {
  local action="$1" source="$2" destination="$3" port="${4:-}"
  local args=(INPUT -s "$source" -d "$destination")
  [ -z "$port" ] || args+=(-p tcp --dport "$port")
  args+=(-m comment --comment "$COMMENT" -j "$action")
  iptables_cmd -C "${args[@]}"
}

case "$MODE" in
  --install)
    [ -t 0 ] || { echo "egress policy install requires an interactive terminal" >&2; exit 1; }
    apply_rule REJECT 172.31.0.10 172.31.0.1
    apply_rule REJECT 172.30.0.10 172.30.0.1
    apply_rule ACCEPT 172.30.0.10 172.30.0.1 8888
    ;;
  --check)
    check_rule ACCEPT 172.30.0.10 172.30.0.1 8888
    check_rule REJECT 172.30.0.10 172.30.0.1
    check_rule REJECT 172.31.0.10 172.31.0.1
    ;;
  --remove)
    [ -t 0 ] || { echo "egress policy removal requires an interactive terminal" >&2; exit 1; }
    while iptables_cmd -D INPUT -s 172.30.0.10 -d 172.30.0.1 -p tcp --dport 8888 -m comment --comment "$COMMENT" -j ACCEPT 2>/dev/null; do :; done
    while iptables_cmd -D INPUT -s 172.30.0.10 -d 172.30.0.1 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    while iptables_cmd -D INPUT -s 172.31.0.10 -d 172.31.0.1 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    ;;
  *) echo "Usage: egress-policy.sh --install|--check|--remove" >&2; exit 2 ;;
esac
