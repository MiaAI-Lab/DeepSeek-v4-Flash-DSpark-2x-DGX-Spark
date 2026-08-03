#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NETWORK_DIR="$(cd "$SCRIPT_DIR/../network" && pwd)"
MODE="check"
ROLE=""
IFACE="enp1s0f0np0"
TARGET="/etc/netplan/90-dspark-cx7.yaml"
CONNECTION="netplan-$IFACE"

usage() {
  cat <<'EOF'
Usage: apply-cx7-network.sh [--check|--apply] --role head|worker

Runs locally on the selected Spark. Check mode is the default and makes no
persistent change. Apply mode requires an interactive confirmation and uses
Netplan generation, targeted NetworkManager activation, and a timed rollback.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --apply) MODE="apply" ;;
    --role) shift; ROLE="${1:-}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$ROLE" in
  head) TEMPLATE="$NETWORK_DIR/head-cx7.yaml"; ADDRESS="10.77.77.1/30"; PEER_HOST="dgx-spark-2.tailc62fd7.ts.net" ;;
  worker) TEMPLATE="$NETWORK_DIR/worker-cx7.yaml"; ADDRESS="10.77.77.2/30"; PEER_HOST="dgx-spark-1.tailc62fd7.ts.net" ;;
  *) echo "--role must be head or worker" >&2; exit 2 ;;
esac

for command in ip sudo netplan nmcli systemd-run tailscale; do
  command -v "$command" >/dev/null 2>&1 || { echo "Missing command: $command" >&2; exit 1; }
done
[ -f "$TEMPLATE" ] || { echo "Missing template: $TEMPLATE" >&2; exit 1; }
ip link show dev "$IFACE" >/dev/null 2>&1 || { echo "Missing interface: $IFACE" >&2; exit 1; }

sudo -v
default_before="$(ip -4 route show default)"
[ -n "$default_before" ] || { echo "No IPv4 default route is active; refusing network work." >&2; exit 1; }
tailscale status --peers=false >/dev/null
connection_before="$(nmcli -g GENERAL.CONNECTION device show "$IFACE")"
if [ ! -f "$TARGET" ] && [ -n "$connection_before" ] && [ "$connection_before" != "--" ]; then
  echo "$IFACE already has an active unmanaged NetworkManager connection: $connection_before" >&2
  exit 1
fi
if ip -4 route show table all | grep -F "10.77.77.0/30" >/dev/null; then
  current="$(ip -4 address show dev "$IFACE")"
  printf '%s\n' "$current" | grep -F "$ADDRESS" >/dev/null || {
    echo "Route collision for 10.77.77.0/30." >&2
    exit 1
  }
fi

check_root="$(mktemp -d "${TMPDIR:-/tmp}/dspark-netplan-check.XXXXXX")"
trap 'sudo find "$check_root" -depth -delete 2>/dev/null || true' EXIT
mkdir -p "$check_root/etc/netplan"
sudo cp -a /etc/netplan/. "$check_root/etc/netplan/"
sudo install -m 0600 "$TEMPLATE" "$check_root$TARGET"
sudo netplan generate --root-dir "$check_root"
merged="$(sudo netplan get --root-dir "$check_root" "ethernets.$IFACE")"
for expected in "$ADDRESS" "mtu: 9000" "dhcp4: false" "dhcp6: false" "accept-ra: false" "link-local: []"; do
  printf '%s\n' "$merged" | grep -F "$expected" >/dev/null || {
    echo "Merged netplan did not preserve required setting: $expected" >&2
    exit 1
  }
done
if printf '%s\n' "$merged" | grep -E '^[[:space:]]*(gateway4|gateway6|nameservers|routes):' >/dev/null; then
  echo "Merged CX-7 configuration unexpectedly adds a gateway, DNS, or route." >&2
  exit 1
fi

echo "Validated role=$ROLE interface=$IFACE address=$ADDRESS"
if [ -f "$TARGET" ]; then
  sudo diff -u "$TARGET" "$TEMPLATE" || true
else
  echo "New merged netplan fragment: $TARGET"
  sed -n '1,120p' "$TEMPLATE"
fi
[ "$MODE" = "apply" ] || exit 0

printf 'Type APPLY %s to continue: ' "$ROLE" >&2
read -r confirmation
[ "$confirmation" = "APPLY $ROLE" ] || { echo "Confirmation did not match; no change made." >&2; exit 1; }

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="/var/backups/dspark-netplan/$stamp-$ROLE"
sudo install -d -m 0700 "$backup_dir"
if [ -f "$TARGET" ]; then
  sudo cp -a "$TARGET" "$backup_dir/original.yaml"
  rollback="cp '$backup_dir/original.yaml' '$TARGET'; netplan generate; nmcli connection reload; nmcli connection up '$CONNECTION' ifname '$IFACE'"
else
  sudo touch "$backup_dir/target-was-absent"
  rollback="find '$TARGET' -maxdepth 0 -type f -delete; netplan generate; nmcli connection reload; nmcli device disconnect '$IFACE' || true; ip link set dev '$IFACE' up"
fi
sudo install -m 0600 "$TEMPLATE" "$TARGET"
sudo netplan generate

rollback_unit="dspark-cx7-rollback-$stamp"
sudo systemd-run --unit "$rollback_unit" --on-active=120s /bin/sh -c "$rollback"
echo "Timed rollback armed as $rollback_unit for 120 seconds."
sudo nmcli connection reload
sudo nmcli connection up "$CONNECTION" ifname "$IFACE"
ip -4 address show dev "$IFACE" | grep -F "$ADDRESS" >/dev/null
test "$(ip -4 route show default)" = "$default_before" || {
  echo "The IPv4 default route changed; rollback remains armed." >&2
  exit 1
}
ip route show default | grep -F "dev $IFACE" >/dev/null && {
  echo "Unexpected default route on $IFACE; rollback remains armed." >&2
  exit 1
}
tailscale status --peers=false >/dev/null
tailscale ping --timeout=5s "$PEER_HOST" >/dev/null
sudo netplan get "ethernets.$IFACE" | grep -F "$ADDRESS" >/dev/null
sudo systemctl stop "$rollback_unit.timer" 2>/dev/null || true
sudo systemctl reset-failed "$rollback_unit.service" 2>/dev/null || true
echo "Applied $TARGET; backup=$backup_dir"
