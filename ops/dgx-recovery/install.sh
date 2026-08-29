#!/usr/bin/env bash
set -euo pipefail

role="${1:-}"
case "$role" in
  head|worker) ;;
  *) echo "usage: sudo $0 head|worker" >&2; exit 2 ;;
esac
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
route="$(ip route show default | head -n1)"
iface="$(awk '{for (i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}' <<<"$route")"
[ -n "$iface" ] || { echo "no default route; refusing to disable Wi-Fi" >&2; exit 1; }
case "$iface" in wl*) echo "default route uses Wi-Fi ($iface); configure wired networking first" >&2; exit 1;; esac

grep -Fq 'init: true' "$root/docker-compose.dspark.yml"
grep -Eq 'stop_grace_period:.*30s' "$root/docker-compose.dspark.yml"

install -m 0644 "$here/blacklist-mt7925-dgx.conf" /etc/modprobe.d/blacklist-mt7925-dgx.conf
install -m 0644 "$here/90-dgx-recovery.conf" /etc/sysctl.d/90-dgx-recovery.conf
install -D -m 0644 "$here/90-dgx-watchdog.conf" /etc/systemd/system.conf.d/90-dgx-watchdog.conf
install -m 0644 "$here/99-nccl-memlock.conf" /etc/security/limits.d/99-nccl-memlock.conf

systemctl enable docker.service
sysctl --system >/dev/null
update-initramfs -u

if [ "$role" = worker ]; then
  install -m 0755 "$here/mia-worker-orphan-guard" /usr/local/sbin/mia-worker-orphan-guard
  install -m 0644 "$here/mia-worker-orphan-guard.service" /etc/systemd/system/mia-worker-orphan-guard.service
  systemctl daemon-reload
  systemctl enable mia-worker-orphan-guard.service
fi

echo "installed DGX recovery controls for $role; reboot once, then run: $here/verify.sh $role --wait 900"
