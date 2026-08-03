#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/config.yaml"
MODE="create"
HERMES_HOME=""
BASE_URL=""
KEY_FILE=""
REQUEST_ID=""
REQUEST_TIMEOUT=900

usage() {
  echo "Usage: $0 --home PATH --base-url URL --key-file PATH --request-id ID [--request-timeout SEC]" >&2
  echo "       $0 --verify-only" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --home) HERMES_HOME="${2:?missing --home value}"; shift 2 ;;
    --base-url) BASE_URL="${2:?missing --base-url value}"; shift 2 ;;
    --key-file) KEY_FILE="${2:?missing --key-file value}"; shift 2 ;;
    --request-id) REQUEST_ID="${2:?missing --request-id value}"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="${2:?missing --request-timeout value}"; shift 2 ;;
    --verify-only) MODE="verify"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

python3 - "$TEMPLATE" "$MODE" "$HERMES_HOME" "$BASE_URL" "$KEY_FILE" "$REQUEST_ID" "$REQUEST_TIMEOUT" <<'PY'
import ipaddress
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlparse

template_path, mode, home_raw, base_url, key_raw, request_id, timeout_raw = sys.argv[1:]
template = Path(template_path).read_text(encoding="utf-8")
required = {
    "__BASE_URL__": 2,
    "__REQUEST_ID__": 1,
    "__REQUEST_TIMEOUT__": 1,
}
for token, count in required.items():
    if template.count(token) != count:
        raise SystemExit(f"template token {token} count is not {count}")
for forbidden in ("api_key:", "mcp_servers", "telegram:", "slack:", "docker.sock"):
    if forbidden in template.lower():
        raise SystemExit(f"forbidden profile surface in template: {forbidden}")
if mode == "verify":
    print("Hermes profile template is isolated and contains no secret values.")
    raise SystemExit(0)

if not all((home_raw, base_url, key_raw, request_id)):
    raise SystemExit("--home, --base-url, --key-file, and --request-id are required")
if not re.fullmatch(r"hermes-smoke-[a-f0-9-]{8,64}", request_id):
    raise SystemExit("invalid request id")
try:
    timeout = int(timeout_raw)
except ValueError as exc:
    raise SystemExit("request timeout must be an integer") from exc
if not 1 <= timeout <= 1800:
    raise SystemExit("request timeout must be between 1 and 1800 seconds")

parsed = urlparse(base_url)
if parsed.scheme != "http" or parsed.path.rstrip("/") != "/v1" or parsed.username or parsed.password:
    raise SystemExit("base URL must be an unauthenticated http://HOST:PORT/v1 URL")
host = parsed.hostname or ""
allowed = host.endswith(".ts.net")
try:
    address = ipaddress.ip_address(host)
    allowed = allowed or address in ipaddress.ip_network("100.64.0.0/10")
    allowed = allowed or (os.environ.get("ALLOW_LOOPBACK_PROVIDER") == "1" and address.is_loopback)
except ValueError:
    pass
if not allowed:
    raise SystemExit("provider host must be a Tailscale address/name")

home = Path(home_raw).expanduser().resolve(strict=False)
default_root = (Path.home() / ".hermes").resolve(strict=False)
if home == default_root or default_root in home.parents:
    raise SystemExit("isolated HERMES_HOME must be outside ~/.hermes")
if home.exists() or home.is_symlink():
    raise SystemExit("refusing to reuse or overwrite HERMES_HOME")

key_path = Path(key_raw).expanduser()
if key_path.is_symlink() or not key_path.is_file():
    raise SystemExit("key source must be a regular non-symlink file")
key_mode = stat.S_IMODE(key_path.stat().st_mode)
if key_mode & 0o077:
    raise SystemExit("key source permissions must be 0600 or stricter")
key = key_path.read_text(encoding="utf-8").strip()
if not key.startswith("sk-") or "\n" in key or "\r" in key:
    raise SystemExit("invalid virtual inference key")

old_umask = os.umask(0o077)
try:
    home.mkdir(mode=0o700, parents=True)
    rendered = (template
        .replace("__BASE_URL__", base_url.rstrip("/"))
        .replace("__REQUEST_ID__", request_id)
        .replace("__REQUEST_TIMEOUT__", str(timeout)))
    config_path = home / "config.yaml"
    config_path.write_text(rendered, encoding="utf-8")
    config_path.chmod(0o600)
    env_path = home / ".env"
    env_path.write_text(f"DEEPSEEK_SMOKE_API_KEY={key}\n", encoding="utf-8")
    env_path.chmod(0o600)
    marker = home / ".no-bundled-skills"
    marker.write_text("Isolated smoke profile: bundled skills disabled.\n", encoding="utf-8")
    marker.chmod(0o600)
    (home / "home").mkdir(mode=0o700)
finally:
    os.umask(old_umask)

if stat.S_IMODE(home.stat().st_mode) != 0o700:
    raise SystemExit("HERMES_HOME mode is not 0700")
for path in (home / "config.yaml", home / ".env", home / ".no-bundled-skills"):
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit(f"{path.name} mode is not 0600")
print(home)
PY

if [ "$MODE" = "create" ]; then
  # Literal commands retained as an operator-auditable contract.
  chmod 0700 "$HERMES_HOME"
  chmod 0600 "$HERMES_HOME/.env" "$HERMES_HOME/config.yaml"
fi
