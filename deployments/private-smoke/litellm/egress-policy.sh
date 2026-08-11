#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
COMMENT="dspark-smoke-litellm-egress"

iptables_cmd() {
  sudo -n iptables "$@" || {
    echo "iptables requires narrowly scoped passwordless sudo on this host" >&2
    return 1
  }
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

apply_forward_rule() {
  local action="$1" state="${2:-}"
  local args=(DOCKER-USER -s 172.29.0.10)
  [ -z "$state" ] || args+=(-m conntrack --ctstate "$state")
  args+=(-m comment --comment "$COMMENT" -j "$action")
  iptables_cmd -C "${args[@]}" 2>/dev/null || iptables_cmd -I "${args[@]}"
}

check_forward_rule() {
  local action="$1" state="${2:-}"
  local args=(DOCKER-USER -s 172.29.0.10)
  [ -z "$state" ] || args+=(-m conntrack --ctstate "$state")
  args+=(-m comment --comment "$COMMENT" -j "$action")
  iptables_cmd -C "${args[@]}"
}

case "$MODE" in
  --install)
    [ -t 0 ] || { echo "egress policy install requires an interactive terminal" >&2; exit 1; }
    apply_rule REJECT 172.31.0.10 172.31.0.1
    apply_rule REJECT 172.30.0.10 172.30.0.1
    apply_rule ACCEPT 172.30.0.10 172.30.0.1 8888
    apply_rule REJECT 172.29.0.10 172.29.0.1
    # Insert the blanket reject first so the established-response allow lands above it.
    apply_forward_rule REJECT
    apply_forward_rule ACCEPT ESTABLISHED,RELATED
    ;;
  --check)
    check_rule ACCEPT 172.30.0.10 172.30.0.1 8888
    check_rule REJECT 172.30.0.10 172.30.0.1
    check_rule REJECT 172.31.0.10 172.31.0.1
    check_rule REJECT 172.29.0.10 172.29.0.1
    check_forward_rule ACCEPT ESTABLISHED,RELATED
    check_forward_rule REJECT
    ;;
  --remove)
    [ -t 0 ] || [ "${DSPARK_EGRESS_NONINTERACTIVE_REMOVE:-0}" = "1" ] || {
      echo "egress policy removal requires an interactive terminal" >&2
      exit 1
    }
    while iptables_cmd -D INPUT -s 172.30.0.10 -d 172.30.0.1 -p tcp --dport 8888 -m comment --comment "$COMMENT" -j ACCEPT 2>/dev/null; do :; done
    while iptables_cmd -D INPUT -s 172.30.0.10 -d 172.30.0.1 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    while iptables_cmd -D INPUT -s 172.31.0.10 -d 172.31.0.1 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    while iptables_cmd -D INPUT -s 172.29.0.10 -d 172.29.0.1 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    while iptables_cmd -D DOCKER-USER -s 172.29.0.10 -m conntrack --ctstate ESTABLISHED,RELATED -m comment --comment "$COMMENT" -j ACCEPT 2>/dev/null; do :; done
    while iptables_cmd -D DOCKER-USER -s 172.29.0.10 -m comment --comment "$COMMENT" -j REJECT 2>/dev/null; do :; done
    ;;
  *) echo "Usage: egress-policy.sh --install|--check|--remove" >&2; exit 2 ;;
esac
