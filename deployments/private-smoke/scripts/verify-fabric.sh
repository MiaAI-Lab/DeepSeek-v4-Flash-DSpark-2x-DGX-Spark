#!/usr/bin/env bash
set -euo pipefail

HEAD_HOST="${HEAD_HOST:-spark-api}"
WORKER_HOST="${WORKER_HOST:-spark-lab}"
HEAD_IP="10.77.77.1"
WORKER_IP="10.77.77.2"
IFACE="enp1s0f0np0"
HCA="rocep1s0f0"
NCCL_TESTS_COMMIT="da0b547b1b9c6e3b1d4c15578087874522ae3761"
NCCL_TESTS_IMAGE="${NCCL_TESTS_IMAGE:-dspark-nccl-tests:$NCCL_TESTS_COMMIT}"
REQUIRE_PERSISTENT=0

usage() {
  echo "Usage: verify-fabric.sh [--require-persistent]";
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --require-persistent) REQUIRE_PERSISTENT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

for command in ssh; do
  command -v "$command" >/dev/null 2>&1 || { echo "Missing command: $command" >&2; exit 1; }
done

run_host() {
  local host="$1" command="$2"
  case "$host" in
    localhost|127.0.0.1) bash -lc "$command" ;;
    *) ssh "$host" "$command" ;;
  esac
}

check_node() {
  local host="$1" address="$2"
  run_host "$host" \
    "set -eu; ip -4 address show dev '$IFACE' | grep -F '$address/30'; ip link show dev '$IFACE' | grep -F 'mtu 9000'; ! ip route show default | grep -F 'dev $IFACE'; command -v show_gids >/dev/null; show_gids | grep -F '$HCA' | grep -F '$address'; command -v ib_write_bw >/dev/null"
  if [ "$REQUIRE_PERSISTENT" = "1" ]; then
    run_host "$host" \
      "set -eu; test -f /etc/netplan/90-dspark-cx7.yaml; test \"\$(nmcli -g GENERAL.CONNECTION device show '$IFACE')\" = 'netplan-$IFACE'; test \"\$(nmcli -g connection.autoconnect connection show 'netplan-$IFACE')\" = yes"
  fi
}

check_node "$HEAD_HOST" "$HEAD_IP"
check_node "$WORKER_HOST" "$WORKER_IP"
run_host "$HEAD_HOST" "ping -M do -c 3 -s 8972 '$WORKER_IP'"
run_host "$WORKER_HOST" "ping -M do -c 3 -s 8972 '$HEAD_IP'"

run_ib_write_bw() {
  local server_host="$1" server_ip="$2" client_host="$3"
  local token="dspark-ib-$RANDOM-$$"
  run_host "$server_host" "umask 077; timeout 45 ib_write_bw -d '$HCA' -F --report_gbits >'/tmp/$token.log' 2>&1 & echo \$! >'/tmp/$token.pid'"
  sleep 2
  if ! run_host "$client_host" "timeout 30 ib_write_bw -d '$HCA' -F --report_gbits '$server_ip'"; then
    run_host "$server_host" "test -f '/tmp/$token.pid' && kill \$(cat '/tmp/$token.pid') 2>/dev/null || true; find '/tmp/$token.pid' '/tmp/$token.log' -maxdepth 0 -type f -delete"
    return 1
  fi
  run_host "$server_host" "for _ in 1 2 3 4 5; do test -f '/tmp/$token.pid' && kill -0 \$(cat '/tmp/$token.pid') 2>/dev/null || break; sleep 1; done; cat '/tmp/$token.log'; find '/tmp/$token.pid' '/tmp/$token.log' -maxdepth 0 -type f -delete"
}

run_ib_write_bw "$WORKER_HOST" "$WORKER_IP" "$HEAD_HOST"
run_ib_write_bw "$HEAD_HOST" "$HEAD_IP" "$WORKER_HOST"

launcher="$(cd "$(dirname "$0")/../network" && pwd)/run-nccl-tests.sh"
[ -x "$launcher" ] || { echo "Missing NCCL launcher: $launcher" >&2; exit 1; }
NCCL_TESTS_IMAGE="$NCCL_TESTS_IMAGE" HEAD_HOST="$HEAD_HOST" WORKER_HOST="$WORKER_HOST" \
  HEAD_IP="$HEAD_IP" WORKER_IP="$WORKER_IP" HCA="$HCA" IFACE="$IFACE" \
  "$launcher" all_reduce_perf -b 8M -e 128M -f 2 -g 1

echo "fabric verification passed"
