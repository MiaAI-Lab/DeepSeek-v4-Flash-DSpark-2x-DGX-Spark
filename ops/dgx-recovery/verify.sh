#!/usr/bin/env bash
set -uo pipefail

role="${1:-}"
case "$role" in
  head|worker) ;;
  *) echo "usage: $0 head|worker [--wait seconds]" >&2; exit 2 ;;
esac
wait_seconds=0
if [ "${2:-}" = --wait ]; then
  wait_seconds="${3:-900}"
  [[ "$wait_seconds" =~ ^[0-9]+$ ]] || { echo "wait duration must be seconds" >&2; exit 2; }
fi

container="${MIA_CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
api_host="${MIA_API_HOST:-100.125.193.22}"
api_port="${MIA_API_PORT:-8888}"
fail=0
check() { if "$@" >/dev/null 2>&1; then printf 'ok   %s\n' "$*"; else printf 'FAIL %s\n' "$*"; fail=1; fi; }

iface="$(ip route show default | awk 'NR==1 {for (i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}')"
case "$iface" in ""|wl*) echo "FAIL wired default route (found: ${iface:-none})"; fail=1;; *) echo "ok   wired default route ($iface)";; esac
check test -f /etc/modprobe.d/blacklist-mt7925-dgx.conf
check sh -c '! lsmod | grep -q "^mt7925e"'
check test "$(sysctl -n kernel.panic_on_oops)" = 1
check test "$(sysctl -n kernel.panic)" = 10
check sh -c 'systemctl show --property=RuntimeWatchdogUSec --value | grep -Eq "^(1min|60000000)$"'
check systemctl is-enabled docker.service
check systemctl is-active docker.service
check test "$(docker inspect "$container" --format '{{.HostConfig.Init}}' 2>/dev/null)" = true
check test "$(docker inspect "$container" --format '{{.State.Running}}' 2>/dev/null)" = true

if [ "$role" = worker ]; then
  check systemctl is-enabled mia-worker-orphan-guard.service
  check systemctl is-active mia-worker-orphan-guard.service
fi

if [ "$wait_seconds" -gt 0 ]; then
  api="http://${api_host}:${api_port}/v1/models"
  started=$SECONDS
  while ! curl -fsS --max-time 3 "$api" >/dev/null 2>&1; do
    if [ $((SECONDS - started)) -ge "$wait_seconds" ]; then
      echo "FAIL API was not ready after ${wait_seconds}s"
      fail=1
      break
    fi
    sleep 5
  done
  [ "$fail" -ne 0 ] || echo "ok   API ready after $((SECONDS - started))s"
fi

exit "$fail"
