#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CREATE_PROFILE="$SCRIPT_DIR/create-profile.sh"
FIXTURE="$SCRIPT_DIR/fixtures/transform-input.json"
CONTRACT="$SCRIPT_DIR/fixtures/tool-contract.json"
MODEL="deepseek-v4-flash-0731-smoke"
PROVIDER="custom:deepseek-smoke"
TERMINAL_IMAGE="python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
REPEAT=1
BASE_URL="${HERMES_SMOKE_BASE_URL:-}"
KEY_FILE="${HERMES_SMOKE_KEY_FILE:-}"
OUTPUT_DIR="${HERMES_SMOKE_OUTPUT_DIR:-$ROOT_DIR/artifacts/hermes}"
COLIMA_BIN="${COLIMA_BIN:-colima}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
HERMES_BIN="${HERMES_BIN:-hermes}"
HERMES_PROCESS_TIMEOUT="${HERMES_SMOKE_PROCESS_TIMEOUT:-600}"
HERMES_DUMP_REQUESTS="${HERMES_SMOKE_DUMP_REQUESTS:-0}"
RUN_TMP=""
COLIMA_PROFILE=""
DOCKER_HOST=""
DOCKER_CONFIG_DIR=""
STUB_PID=""
CAPTURE_PID=""

usage() {
  echo "Usage: HERMES_SMOKE_BASE_URL=http://TAILNET_IP:4001/v1 HERMES_SMOKE_KEY_FILE=PATH $0 [--repeat N]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repeat) REPEAT="${2:?missing --repeat value}"; shift 2 ;;
    --base-url) BASE_URL="${2:?missing --base-url value}"; shift 2 ;;
    --key-file) KEY_FILE="${2:?missing --key-file value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

[[ "$REPEAT" =~ ^[1-9][0-9]*$ ]] && [ "$REPEAT" -le 10 ] || { echo "--repeat must be 1..10" >&2; exit 2; }
[[ "$HERMES_PROCESS_TIMEOUT" =~ ^[1-9][0-9]*$ ]] \
  || { echo "HERMES_SMOKE_PROCESS_TIMEOUT must be a positive integer" >&2; exit 2; }
[[ "$HERMES_DUMP_REQUESTS" =~ ^[01]$ ]] \
  || { echo "HERMES_SMOKE_DUMP_REQUESTS must be 0 or 1" >&2; exit 2; }
[ -n "$BASE_URL" ] && [ -n "$KEY_FILE" ] || { usage; exit 2; }

command -v "$COLIMA_BIN" >/dev/null || { echo "Colima is required." >&2; exit 1; }
command -v "$DOCKER_BIN" >/dev/null || { echo "Docker CLI is required; Colima alone does not provide the client." >&2; exit 1; }
command -v "$HERMES_BIN" >/dev/null || { echo "Hermes CLI is required." >&2; exit 1; }
command -v python3 >/dev/null || { echo "Python 3 is required." >&2; exit 1; }

safe_remove_run_tmp() {
  [ -n "$RUN_TMP" ] || return 0
  case "$RUN_TMP" in
    "${TMPDIR:-/tmp}"/dspark-hermes-smoke.*) rm -rf -- "$RUN_TMP" ;;
    *) echo "Refusing to remove unexpected temp path: $RUN_TMP" >&2; return 1 ;;
  esac
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "$STUB_PID" ]; then
    kill "$STUB_PID" >/dev/null 2>&1 || true
    wait "$STUB_PID" 2>/dev/null || true
  fi
  if [ -n "$CAPTURE_PID" ]; then
    kill "$CAPTURE_PID" >/dev/null 2>&1 || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
  if [ -n "$DOCKER_HOST" ]; then
    DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" ps -aq --filter label=hermes-agent=1 2>/dev/null \
      | while IFS= read -r container; do
          [ -z "$container" ] || DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" rm -f -- "$container" >/dev/null 2>&1 || true
        done
  fi
  if [ -n "$COLIMA_PROFILE" ]; then
    "$COLIMA_BIN" stop --profile "$COLIMA_PROFILE" >/dev/null 2>&1 || true
    "$COLIMA_BIN" delete --profile "$COLIMA_PROFILE" --force >/dev/null 2>&1 || true
  fi
  safe_remove_run_tmp || true
  exit "$status"
}
trap cleanup EXIT INT TERM

RUN_TMP="$(mktemp -d "${TMPDIR:-/tmp}/dspark-hermes-smoke.XXXXXX")"
chmod 0700 "$RUN_TMP"
DOCKER_CONFIG_DIR="$RUN_TMP/docker-config"
mkdir -m 0700 "$DOCKER_CONFIG_DIR"
export DOCKER_CONFIG="$DOCKER_CONFIG_DIR"
RUN_ID="hermes-smoke-$(python3 - <<'PY'
import secrets
print(secrets.token_hex(8))
PY
)"
COLIMA_PROFILE="dspark-hermes-smoke-${RUN_ID##*-}"
DOCKER_HOST="unix://${HOME}/.colima/${COLIMA_PROFILE}/docker.sock"

snapshot_shared_state() {
  local destination="$1"
  python3 - "$HOME" "$destination" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

home, destination = map(Path, sys.argv[1:])
hermes_root = home / ".hermes"
targets = [
    hermes_root,
    hermes_root / "profiles",
    hermes_root / "config.yaml",
    hermes_root / ".env",
    hermes_root / "active_profile",
    hermes_root / "profiles.json",
    hermes_root / "registry.json",
    hermes_root / "profiles" / "default",
    hermes_root / "profiles" / "hermesia",
    home / ".docker" / "config.json",
    home / ".colima" / "default" / "colima.yaml",
    home / ".colima" / "plexiz-inference" / "colima.yaml",
]
targets.extend(sorted((home / ".local" / "bin").glob("*hermes*")))
targets.extend(sorted((home / "Library" / "LaunchAgents").glob("*hermes*")))

def record(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "type": "missing"}
    item = {
        "path": str(path),
        "mode": oct(stat.S_IMODE(info.st_mode)),
        "uid": info.st_uid,
        "gid": info.st_gid,
    }
    if path.is_symlink():
        item.update(type="symlink", target=os.readlink(path))
    elif path.is_file():
        item.update(
            type="file", size=info.st_size, mtime_ns=info.st_mtime_ns,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    elif path.is_dir():
        item.update(type="directory")
    else:
        item.update(type="other")
    return item

rows = []
seen = set()
stable_profile_names = {
    ".env", "SOUL.md", "auth.json", "channel_directory.json", "config.yaml",
    "profile.yaml", "slack-manifest.json",
}
stable_profile_roots = {"bin", "hooks", "memories", "skills"}
for target in targets:
    key = str(target)
    if key not in seen:
        rows.append(record(target)); seen.add(key)
    # Hermesia is live. Fingerprint its credentials, rules, config, skills,
    # hooks, memory and executable registry, but exclude logs, sessions,
    # SQLite WALs, gateway heartbeats and caches that its existing service
    # legitimately changes while this isolated suite runs.
    if target.is_dir() and target.parent.name == "profiles":
        for child in sorted(target.rglob("*")):
            relative = child.relative_to(target)
            root_name = relative.parts[0]
            if root_name not in stable_profile_roots and str(relative) not in stable_profile_names:
                continue
            if child.name in {".usage.json", ".usage.json.lock"}:
                continue
            key = str(child)
            if key not in seen:
                rows.append(record(child)); seen.add(key)

encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
payload = {"sha256": hashlib.sha256(encoded).hexdigest(), "entries": rows}
destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
destination.chmod(0o600)
PY
}

assert_profile_inventory() {
  local profile_home="$1"
  python3 - "$profile_home" <<'PY'
from pathlib import Path
import os
import stat
import sys

home = Path(sys.argv[1])
if stat.S_IMODE(home.stat().st_mode) != 0o700:
    raise SystemExit("isolated home mode is not 0700")
for name in (".env", "config.yaml", ".no-bundled-skills"):
    path = home / name
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit(f"unsafe isolated profile file: {name}")
for forbidden in (
    "skills", "optional-skills", "mcp.json", "mcp_servers.json", "memory",
    "memories", "gateway.pid", "gateway_state.json", "cron", "plugins",
    "platforms", "active_profile",
):
    path = home / forbidden
    if path.is_symlink():
        raise SystemExit(f"forbidden inherited symlink: {forbidden}")
    if path.is_dir() and not any(path.rglob("*")):
        continue
    if path.exists():
        raise SystemExit(f"forbidden populated inherited surface: {forbidden}")
env_lines = (home / ".env").read_text(encoding="utf-8").splitlines()
if len(env_lines) != 1 or not env_lines[0].startswith("DEEPSEEK_SMOKE_API_KEY=sk-"):
    raise SystemExit("isolated .env must contain exactly the inference key")
PY
}

capture_containers() {
  local output="$1"
  : >"$output"
  DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" ps -aq --filter label=hermes-agent=1 \
    | while IFS= read -r container; do
        [ -z "$container" ] || DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" inspect \
          --format '{{json .}}' "$container" >>"$output"
      done
}

capture_default_container_observation() {
  local writer_observation="$1"
  local container observation_candidate
  observation_candidate="${writer_observation}.candidate"
  DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" ps -aq \
    --filter label=hermes-agent=1 --filter label=hermes-task-id=default \
    | while IFS= read -r container; do
        [ -n "$container" ] || continue
        rm -f -- "$observation_candidate"
        # Configuration is immutable and the tmpfs writer may exist for less
        # than one polling interval. Capture confinement as soon as the
        # default backend exists; the tool result independently proves that
        # this backend completed the synthetic program.
        if DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" inspect \
          --format '{{json .}}' "$container" >"$observation_candidate" 2>/dev/null; then
          chmod 0600 "$observation_candidate"
          mv -f -- "$observation_candidate" "$writer_observation"
        fi
      done
  rm -f -- "$observation_candidate"
}

recover_tool_evidence() {
  local profile_home="$1" destination="$2"
  python3 - "$profile_home" "$destination" <<'PY'
import json
from pathlib import Path
import sys

profile, destination = map(Path, sys.argv[1:])
expected = {
    "created": True,
    "read": True,
    "transformed": True,
    "personal_centavos": 97150,
    "plexiz_centavos": 276875,
    "grand_total_centavos": 374025,
    "host_paths_blocked": True,
    "network_blocked": True,
}
evidence = []
for dump_path in sorted(profile.rglob("request_dump_*.json")):
    body = json.loads(dump_path.read_text(encoding="utf-8"))["request"]["body"]
    for message in body.get("messages") or []:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        if result.get("exit_code") != 0:
            continue
        for line in str(result.get("output") or "").splitlines():
            if line.startswith("TERMINAL_EVIDENCE_JSON="):
                evidence.append(json.loads(line.split("=", 1)[1]))
canonical = {json.dumps(item, sort_keys=True) for item in evidence}
if canonical != {json.dumps(expected, sort_keys=True)}:
    raise SystemExit("Hermes tool-result evidence is missing, conflicting, or incomplete")
totals = {key: expected[key] for key in (
    "personal_centavos", "plexiz_centavos", "grand_total_centavos"
)}
destination.write_text(json.dumps(totals, sort_keys=True) + "\n", encoding="utf-8")
destination.chmod(0o600)
PY
}

merge_workspace_writer_observation() {
  local observations="$1" writer_observation="$2"
  python3 - "$observations" "$writer_observation" <<'PY'
import json
from pathlib import Path
import sys

observations, writer_path = map(Path, sys.argv[1:])
writer = json.loads(writer_path.read_text(encoding="utf-8"))
writer_id = writer.get("Id")
if not writer_id:
    raise SystemExit("workspace writer observation lacks a container ID")
writer["_dspark_workspace_writer"] = True
rows = [
    json.loads(line)
    for line in observations.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
merged = [row for row in rows if row.get("Id") != writer_id]
merged.append(writer)
observations.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
    encoding="utf-8",
)
PY
}

summarize_request_diagnostics() {
  local profile_home="$1"
  [ "$HERMES_DUMP_REQUESTS" = "1" ] || return 0
  python3 - "$profile_home" <<'PY'
import json
from pathlib import Path
import sys

dumps = sorted(Path(sys.argv[1]).rglob("request_dump_*.json"))
print(f"Hermes request diagnostics: dumps={len(dumps)}", file=sys.stderr)
if not dumps:
    raise SystemExit(0)
body = json.loads(dumps[-1].read_text(encoding="utf-8"))["request"]["body"]
for index, message in enumerate(body.get("messages") or []):
    role = message.get("role")
    if role == "assistant":
        summaries = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw = function.get("arguments") or ""
            try:
                parsed = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            summaries.append({
                "name": function.get("name"),
                "argument_chars": len(raw),
                "argument_keys": sorted(parsed) if isinstance(parsed, dict) else [],
                "command_chars": len(parsed.get("command") or "") if isinstance(parsed, dict) else 0,
                "nested_arguments": isinstance(parsed, dict) and "arguments" in parsed,
            })
        print(json.dumps({
            "message": index,
            "role": role,
            "content_chars": len(message.get("content") or ""),
            "calls": summaries,
        }, sort_keys=True), file=sys.stderr)
    elif role == "tool":
        content = str(message.get("content") or "")
        print(json.dumps({
            "message": index,
            "role": role,
            "content_chars": len(content),
            "content_prefix": content[:400],
        }, sort_keys=True), file=sys.stderr)
PY
}

summarize_failed_hermes() {
  local status="$1" usage_file="$2" response_file="$3" stderr_file="$4" profile_home="$5"
  python3 - "$status" "$usage_file" "$response_file" "$stderr_file" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

status, usage_path, response_path, stderr_path = sys.argv[1:]
usage_file = Path(usage_path)
usage = json.loads(usage_file.read_text()) if usage_file.exists() else {}
response = Path(response_path).read_text(errors="replace") if Path(response_path).exists() else ""
stderr = Path(stderr_path).read_text(errors="replace") if Path(stderr_path).exists() else ""

def scrub(value):
    value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted-key>", value)
    value = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", value)
    value = re.sub(r"/Users/[^/\s]+", "<home>", value)
    value = re.sub(r"\b100\.(?:6[4-9]|[789][0-9]|1[01][0-9]|12[0-7])(?:\.\d{1,3}){2}\b", "<tailnet-ip>", value)
    return value

safe_usage = {
    key: usage.get(key)
    for key in ("api_calls", "completed", "failed", "input_tokens", "output_tokens", "total_tokens")
}
failure = scrub(str(usage.get("failure") or ""))[-500:]
stderr_tail = scrub(stderr)[-1000:]
print(json.dumps({
    "status": int(status),
    "usage": safe_usage,
    "failure": failure,
    "response_chars": len(response),
    "response_contract": bool(re.fullmatch(r'HERMES_SMOKE_RESULT=(\{.*\})\s*', response)),
    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    "stderr_tail": stderr_tail,
}, sort_keys=True), file=sys.stderr)
PY
  HERMES_DUMP_REQUESTS=1 summarize_request_diagnostics "$profile_home"
}

remove_hermes_containers() {
  DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" ps -aq --filter label=hermes-agent=1 \
    | while IFS= read -r container; do
        [ -z "$container" ] || DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" rm -f -- "$container" >/dev/null
      done
}

validate_container_observations() {
  local observations="$1" profile_home="$2"
  python3 - "$observations" "$profile_home" <<'PY'
import json
from pathlib import Path
import sys

observations, profile_home = sys.argv[1:]
profile = Path(profile_home).resolve()
legacy_cache_names = {
    "document_cache", "image_cache", "audio_cache", "video_cache",
    "browser_screenshots", "web_cache", "delegation_cache",
}

def mount_is_safe(mount):
    source = Path(str(mount.get("Source", ""))).resolve()
    destination = str(mount.get("Destination", ""))
    if mount.get("RW") is not False or not source.is_dir() or any(source.rglob("*")):
        return False
    try:
        relative = source.relative_to(profile)
    except ValueError:
        return False
    if destination == "/root/.hermes/skills":
        return relative == Path("skills")
    return (
        destination.startswith("/root/.hermes/cache/")
        and (relative.parts[:1] == ("cache",) or str(relative) in legacy_cache_names)
    )
rows = []
for line in Path(observations).read_text(encoding="utf-8").splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
if not rows:
    raise SystemExit("no Hermes terminal container was observed")
task_rows = {}
for row in rows:
    labels = row.get("Config", {}).get("Labels") or {}
    task_id = labels.get("hermes-task-id", "")
    network_mode = row.get("HostConfig", {}).get("NetworkMode")
    if task_id == "default" and network_mode != "none":
        raise SystemExit("default work container did not use --network=none")
    if task_id == "prompt-backend-probe" and network_mode not in {"none", "default", "bridge"}:
        raise SystemExit(f"unsafe prompt backend probe network mode: {network_mode}")
    for mount in row.get("Mounts") or []:
        if not mount_is_safe(mount):
            raise SystemExit(
                "unsafe terminal mount: "
                f"{mount.get('Source', '')} -> {mount.get('Destination', '')}"
            )
    if labels.get("hermes-agent") != "1" or "hermes-profile" not in labels:
        raise SystemExit("terminal container lacks Hermes isolation labels")
    task_rows.setdefault(labels.get("hermes-task-id", ""), []).append(row)
default_ids = {row.get("Id") for row in task_rows.get("default", [])}
writer_rows = [row for row in task_rows.get("default", []) if row.get("_dspark_workspace_writer")]
if not (1 <= len(default_ids) <= 2) or len(writer_rows) != 1:
    raise SystemExit("expected one isolated workspace writer and at most one rotated default container")
if set(task_rows) - {"default", "prompt-backend-probe"}:
    raise SystemExit(f"unexpected Hermes task containers: {sorted(task_rows)}")
PY
}

run_hermes() {
  local profile_home="$1" prompt_file="$2" usage_file="$3" response_file="$4" observations="$5"
  local workspace_evidence="$6" writer_observation="$7"
  # Always record the redacted synthetic request history: Hermes destroys
  # the tmpfs immediately after the tool result, so this is the durable
  # source for the executed program's stdout contract.
  env -i \
    PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" TERM="${TERM:-dumb}" NO_COLOR=1 \
    HERMES_HOME="$profile_home" HERMES_SAFE_MODE=1 HERMES_IGNORE_RULES=1 \
    HERMES_TELEMETRY_DISABLED=1 HERMES_STREAM_RETRIES=0 \
    HERMES_DUMP_REQUESTS=1 \
    DOCKER_HOST="$DOCKER_HOST" DOCKER_CONFIG="$DOCKER_CONFIG_DIR" \
    "$HERMES_BIN" -z "$(<"$prompt_file")" \
      --usage-file "$usage_file" \
      --provider "$PROVIDER" --model "$MODEL" --toolsets terminal --ignore-rules \
      >"$response_file" 2>"$RUN_TMP/hermes-stderr.log" &
  local hermes_pid=$!
  local status=0 elapsed=0 process_state="" started_at=$SECONDS
  while kill -0 "$hermes_pid" >/dev/null 2>&1; do
    # Hermes may rotate a tmpfs-backed terminal container while it finalizes
    # a turn. Recover the synthetic output while the writer is still alive;
    # the captured inspect row proves the writer itself had the required
    # network and mount confinement.
    capture_default_container_observation "$writer_observation"
    # An exited, unreaped child is still visible to kill -0 as a zombie.
    # Break so wait can collect its real exit status instead of timing out.
    process_state="$(ps -p "$hermes_pid" -o state= 2>/dev/null | tr -d '[:space:]')"
    [ -n "$process_state" ] || break
    case "$process_state" in Z*) break ;; esac
    elapsed=$((SECONDS - started_at))
    if [ "$elapsed" -ge "$HERMES_PROCESS_TIMEOUT" ]; then
      echo "Hermes one-shot exceeded ${HERMES_PROCESS_TIMEOUT}s; terminating test process $hermes_pid." >&2
      kill -TERM "$hermes_pid" >/dev/null 2>&1 || true
      for _ in $(seq 1 10); do
        kill -0 "$hermes_pid" >/dev/null 2>&1 || break
        sleep 1
      done
      if kill -0 "$hermes_pid" >/dev/null 2>&1; then
        kill -KILL "$hermes_pid" >/dev/null 2>&1 || true
      fi
      wait "$hermes_pid" 2>/dev/null || true
      status=124
      break
    fi
    sleep 0.1
  done
  if [ "$status" -eq 0 ]; then
    wait "$hermes_pid" || status=$?
  fi
  [ "$status" -eq 0 ] || {
    echo "Hermes one-shot failed with status $status" >&2
    summarize_failed_hermes "$status" "$usage_file" "$response_file" "$RUN_TMP/hermes-stderr.log" "$profile_home"
    return "$status"
  }
  capture_default_container_observation "$writer_observation"
  recover_tool_evidence "$profile_home" "$workspace_evidence" || {
    summarize_request_diagnostics "$profile_home"
    return 1
  }
  [ -s "$workspace_evidence" ] && [ -s "$writer_observation" ] \
    || {
      echo "Hermes did not leave recoverable terminal workspace evidence." >&2
      summarize_request_diagnostics "$profile_home"
      return 1
    }
  capture_containers "$observations"
  merge_workspace_writer_observation "$observations" "$writer_observation"
}

write_prompt() {
  local destination="$1"
  python3 - "$FIXTURE" "$CONTRACT" "$destination" <<'PY_PROMPT'
import json
from pathlib import Path
import sys

fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
contract = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
fixture_literal = repr(json.dumps(fixture, sort_keys=True))
expected_literal = repr(contract["expected_totals"])
program = f"""import json
import socket
from decimal import Decimal
from pathlib import Path

workspace = Path('/workspace')
fixture = json.loads({fixture_literal})
expected = {expected_literal}
input_path = workspace / 'input.json'
output_path = workspace / 'output.json'
input_path.write_text(json.dumps(fixture, sort_keys=True), encoding='utf-8')
loaded = json.loads(input_path.read_text(encoding='utf-8'))
assert loaded == fixture
assert loaded['dataset'] == 'synthetic-expenses'
assert loaded['currency'] == 'MXN'
assert [row['id'] for row in loaded['rows']] == ['syn-001', 'syn-002', 'syn-003', 'syn-004']
personal = sum(Decimal(row['amount']) for row in loaded['rows'] if row['scope'] == 'personal')
plexiz = sum(Decimal(row['amount']) for row in loaded['rows'] if row['scope'] == 'plexiz')
computed = {{
    'personal_centavos': int(personal * 100),
    'plexiz_centavos': int(plexiz * 100),
    'grand_total_centavos': int((personal + plexiz) * 100),
}}
assert computed == expected
for host_path in ('/Users/gunter/.ssh', '/Users/gunter/.hermes', '/home/gunter/.ssh', '/root/.ssh'):
    assert not Path(host_path).exists(), host_path
probe = socket.socket()
probe.settimeout(2)
network_blocked = False
try:
    probe.connect(('1.1.1.1', 53))
except OSError:
    network_blocked = True
finally:
    probe.close()
assert network_blocked
output_path.write_text(json.dumps(computed, sort_keys=True), encoding='utf-8')
assert json.loads(output_path.read_text(encoding='utf-8')) == expected
evidence = dict(computed)
evidence.update({{
    'created': True,
    'read': True,
    'transformed': True,
    'host_paths_blocked': True,
    'network_blocked': network_blocked,
}})
print('TERMINAL_EVIDENCE_JSON=' + json.dumps(evidence, sort_keys=True))"""
prompt = f"""You are running a synthetic, isolated acceptance task. You have exactly one toolset: terminal.
Your FIRST action MUST be exactly one terminal call. Do not run exploratory commands and do not nest an arguments object.
Pass this exact multiline string as terminal's command argument:

python3 - <<'PY'
{program}
PY

The command creates, reads, validates, and transforms the synthetic expense fixture with Decimal arithmetic. It also proves the four host paths are absent and the network connection is blocked.
Only after terminal returns TERMINAL_EVIDENCE_JSON with every expected field, return the fixed result below.

Your final response must be exactly one line, with no markdown, in this form:
HERMES_SMOKE_RESULT={{"created":true,"read":true,"transformed":true,"personal_centavos":97150,"plexiz_centavos":276875,"grand_total_centavos":374025,"host_paths_blocked":true,"network_blocked":true}}
Only emit true after terminal evidence proves it.
"""
Path(sys.argv[3]).write_text(prompt, encoding="utf-8")
PY_PROMPT
}

assemble_result() {
  local request_id="$1" usage="$2" response="$3" observations="$4" workspace_evidence="$5" before="$6" after="$7" output="$8"
  local profile_home="$9" capture_evidence="${10}"
  python3 - "$request_id" "$usage" "$response" "$observations" "$workspace_evidence" "$before" "$after" "$output" "$MODEL" "$PROVIDER" "$SCRIPT_DIR/config.yaml" "$FIXTURE" "$CONTRACT" "$profile_home" "$capture_evidence" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

request_id, usage_path, response_path, observations, workspace_path, before_path, after_path, output_path, model, provider, config_path, fixture_path, contract_path, profile_home, capture_path = sys.argv[1:]
usage = json.loads(Path(usage_path).read_text(encoding="utf-8"))
response = Path(response_path).read_text(encoding="utf-8").strip()
match = re.fullmatch(r'HERMES_SMOKE_RESULT=(\{.*\})', response)
if not match:
    raise SystemExit("Hermes did not return the fixed result contract")
reported = json.loads(match.group(1))
expected = {
    "created": True, "read": True, "transformed": True,
    "personal_centavos": 97150, "plexiz_centavos": 276875,
    "grand_total_centavos": 374025, "host_paths_blocked": True,
    "network_blocked": True,
}
if reported != expected:
    raise SystemExit("Hermes result did not match the synthetic fixture contract")
workspace = json.loads(Path(workspace_path).read_text(encoding="utf-8"))
workspace_expected = json.loads(Path(contract_path).read_text(encoding="utf-8"))["expected_totals"]
if workspace != workspace_expected:
    raise SystemExit("independently recovered terminal output does not match the fixture contract")
if usage.get("failed") or not usage.get("completed") or int(usage.get("api_calls") or 0) < 1:
    raise SystemExit("Hermes usage report is incomplete or failed")
if usage.get("model") not in (model, f"openai/{model}"):
    raise SystemExit("Hermes usage report changed model")
if str(usage.get("provider", "")) not in {provider, "custom", "deepseek-smoke"}:
    raise SystemExit("Hermes usage report changed provider")
capture_rows = [
    json.loads(line)
    for line in Path(capture_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
api_calls = int(usage.get("api_calls") or 0)
if len(capture_rows) != api_calls:
    raise SystemExit("captured inference count does not match Hermes usage")
origin_response_ids = []
for row in capture_rows:
    ids = row.get("response_ids") or []
    if row.get("request_id") != request_id or row.get("status") != 200 or len(ids) != 1:
        raise SystemExit("captured gateway response evidence is incomplete")
    origin_response_ids.extend(ids)
if len(set(origin_response_ids)) != api_calls or not all(
    re.fullmatch(r"chatcmpl-[a-f0-9]{16}", item) for item in origin_response_ids
):
    raise SystemExit("captured gateway response IDs are missing, duplicated, or invalid")
before = json.loads(Path(before_path).read_text(encoding="utf-8"))["sha256"]
after = json.loads(Path(after_path).read_text(encoding="utf-8"))["sha256"]
if before != after:
    raise SystemExit("shared default/hermesia state changed during isolated run")
container_rows = [json.loads(line) for line in Path(observations).read_text().splitlines() if line.strip()]
default_rows = [
    row for row in container_rows
    if (row.get("Config", {}).get("Labels") or {}).get("hermes-task-id") == "default"
]
default_ids = {row["Id"] for row in default_rows}
writer_rows = [row for row in default_rows if row.get("_dspark_workspace_writer")]
if not (1 <= len(default_ids) <= 2) or len(writer_rows) != 1:
    raise SystemExit("missing unique workspace-writer confinement evidence")
profile = Path(profile_home).resolve()
legacy_cache_names = {
    "document_cache", "image_cache", "audio_cache", "video_cache",
    "browser_screenshots", "web_cache", "delegation_cache",
}

def mount_is_safe(mount):
    source = Path(str(mount.get("Source", ""))).resolve()
    destination = str(mount.get("Destination", ""))
    if mount.get("RW") is not False or not source.is_dir() or any(source.rglob("*")):
        return False
    try:
        relative = source.relative_to(profile)
    except ValueError:
        return False
    if destination == "/root/.hermes/skills":
        return relative == Path("skills")
    return (
        destination.startswith("/root/.hermes/cache/")
        and (relative.parts[:1] == ("cache",) or str(relative) in legacy_cache_names)
    )

container_isolated = all(
    row.get("HostConfig", {}).get("NetworkMode") == "none"
    and all(mount_is_safe(mount) for mount in row.get("Mounts") or [])
    for row in default_rows
)
if not container_isolated:
    raise SystemExit("terminal container isolation evidence failed")
workspace_sha256 = hashlib.sha256(Path(workspace_path).read_bytes()).hexdigest()
result = {
    "schema_version": 1,
    "run_id": request_id,
    "accepted": True,
    "provider": provider,
    "model": model,
    "tasks": {"create": True, "read": True, "transform": True},
    "negative_checks": {
        "skills": True, "mcp": True, "memory": True, "gateway": True,
        "host_paths": container_isolated, "network": container_isolated, "fallback": True,
    },
    "shared_state": {"before_sha256": before, "after_sha256": after, "unchanged": True},
    "usage": usage,
    "request_ids": [request_id],
    "origin_response_ids": origin_response_ids,
    "tool_evidence_sha256": workspace_sha256,
    "suite_pin_sha256": hashlib.sha256(
        b"".join(Path(path).read_bytes() for path in (config_path, fixture_path, contract_path))
    ).hexdigest(),
}
target = Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
target.chmod(0o600)
PY
}

start_capture_proxy() {
  local evidence_file="$1" port_file="$2"
  python3 "$SCRIPT_DIR/capture-proxy.py" --target-base-url "$BASE_URL" \
    --evidence-file "$evidence_file" --port-file "$port_file" &
  CAPTURE_PID=$!
  for _ in $(seq 1 50); do [ -s "$port_file" ] && break; sleep 0.1; done
  [ -s "$port_file" ] || { echo "Hermes capture proxy did not start." >&2; return 1; }
}

stop_capture_proxy() {
  [ -n "$CAPTURE_PID" ] || return 0
  kill "$CAPTURE_PID" >/dev/null 2>&1 || true
  wait "$CAPTURE_PID" 2>/dev/null || true
  CAPTURE_PID=""
}

run_failure_probe() {
  local mode="$1" timeout="$2"
  local probe_id="${RUN_ID}-${mode}"
  local port_file="$RUN_TMP/${mode}.port"
  local count_file="$RUN_TMP/${mode}.count"
  local key_file="$RUN_TMP/${mode}.key"
  local profile_home="$RUN_TMP/profile-${mode}"
  local usage_file="$RUN_TMP/usage-${mode}.json"
  local response_file="$RUN_TMP/response-${mode}.txt"
  local stderr_file="$RUN_TMP/stderr-${mode}.txt"

  printf 'sk-%s\n' 'synthetic-failure-probe-only' >"$key_file"
  chmod 0600 "$key_file"
  : >"$count_file"
  python3 - "$mode" "$port_file" "$count_file" <<'PY' &
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time

mode, port_path, count_path = sys.argv[1:]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = self.rfile.read(length)
        try:
            stream_requested = bool(json.loads(payload).get("stream", False))
        except (AttributeError, json.JSONDecodeError):
            stream_requested = False
        with Path(count_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "method": "POST",
                "path": self.path,
                "request_id": self.headers.get("X-Request-ID", ""),
                "sdk_retry_count": self.headers.get("X-Stainless-Retry-Count", ""),
                "stream": stream_requested,
                "user_agent": self.headers.get("User-Agent", ""),
            }, sort_keys=True) + "\n")
        if mode == "timeout":
            time.sleep(4)
            body = b'{}'
            self.send_response(504)
            self.send_header("content-type", "application/json")
        elif mode == "malformed":
            body = b'{not-json'
            self.send_response(200)
            self.send_header("content-type", "application/json")
        else:
            body = json.dumps({"error": {"message": "synthetic invalid model", "type": "invalid_request_error"}}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, *_args):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
Path(port_path).write_text(str(server.server_port), encoding="utf-8")
server.serve_forever()
PY
  STUB_PID=$!
  for _ in $(seq 1 50); do [ -s "$port_file" ] && break; sleep 0.1; done
  [ -s "$port_file" ] || { echo "Failure probe server did not start." >&2; return 1; }
  local port
  port="$(<"$port_file")"
  ALLOW_LOOPBACK_PROVIDER=1 "$CREATE_PROFILE" --home "$profile_home" \
    --base-url "http://127.0.0.1:${port}/v1" --key-file "$key_file" \
    --request-id "$probe_id" --request-timeout "$timeout" >/dev/null

  # Profile verification may probe provider metadata (for example /api/show).
  # The failure gate measures only the subsequent agent inference attempts.
  : >"$count_file"

  local status=0
  env -i \
    PATH="$PATH" HOME="$HOME" TMPDIR="${TMPDIR:-/tmp}" TERM="${TERM:-dumb}" NO_COLOR=1 \
    HERMES_HOME="$profile_home" HERMES_SAFE_MODE=1 HERMES_IGNORE_RULES=1 \
    HERMES_TELEMETRY_DISABLED=1 HERMES_STREAM_RETRIES=0 \
    DOCKER_HOST="$DOCKER_HOST" DOCKER_CONFIG="$DOCKER_CONFIG_DIR" \
    "$HERMES_BIN" -z "Return the word impossible; do not call tools." \
      --usage-file "$usage_file" --provider "$PROVIDER" --model "$MODEL" \
      --toolsets terminal --ignore-rules >"$response_file" 2>"$stderr_file" || status=$?
  kill "$STUB_PID" >/dev/null 2>&1 || true
  wait "$STUB_PID" 2>/dev/null || true
  STUB_PID=""

  python3 - "$mode" "$status" "$count_file" "$usage_file" "$stderr_file" <<'PY'
import json
from pathlib import Path
import sys

mode, status, count_path, usage_path, stderr_path = sys.argv[1:]
request_lines = [line for line in Path(count_path).read_text().splitlines() if line.strip()]
requests = [json.loads(line) for line in request_lines]
inference_requests = [
    request for request in requests
    if request.get("path", "").rstrip("/").endswith("/chat/completions")
]
attempts = len(inference_requests)
if attempts != 1:
    stderr_tail = Path(stderr_path).read_text(errors="replace")[-4000:]
    raise SystemExit(
        f"{mode} probe made {attempts} inference attempts; expected exactly one; "
        f"requests={request_lines!r}; stderr_tail={stderr_tail!r}"
    )
usage_file = Path(usage_path)
usage = json.loads(usage_file.read_text()) if usage_file.exists() else {}
if int(status) == 0 and not usage.get("failed"):
    raise SystemExit(f"{mode} probe did not fail explicitly")
if int(usage.get("api_calls") or 0) > 1:
    raise SystemExit(f"{mode} probe reported a retry")
PY
  remove_hermes_containers
}

snapshot_shared_state "$RUN_TMP/shared-before.json"

echo "Starting isolated Colima profile $COLIMA_PROFILE (existing profiles are not modified)."
"$COLIMA_BIN" start --profile "$COLIMA_PROFILE" --runtime docker --cpu 2 --memory 4 --disk 20
for _ in $(seq 1 60); do
  [ -S "${DOCKER_HOST#unix://}" ] && DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" version >/dev/null 2>&1 && break
  sleep 1
done
[ -S "${DOCKER_HOST#unix://}" ] || { echo "Dedicated Colima Docker socket did not appear." >&2; exit 1; }
DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" version >/dev/null
DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" pull "$TERMINAL_IMAGE" >/dev/null

# Fail-closed provider behavior: the installed Hermes version must surface an
# invalid request, a timeout, and malformed JSON after exactly one inference attempt.
# fallback_providers is empty, so any recovery through another model is a gate failure.
run_failure_probe invalid 5
run_failure_probe malformed 5
run_failure_probe timeout 1

mkdir -p "$OUTPUT_DIR"
chmod 0700 "$OUTPUT_DIR"

for iteration in $(seq 1 "$REPEAT"); do
  request_id="${RUN_ID}-${iteration}"
  profile_home="$RUN_TMP/profile-${iteration}"
  usage_file="$RUN_TMP/usage-${iteration}.json"
  response_file="$RUN_TMP/response-${iteration}.txt"
  prompt_file="$RUN_TMP/prompt-${iteration}.txt"
  observations="$RUN_TMP/docker-${iteration}.jsonl"
  workspace_evidence="$RUN_TMP/workspace-output-${iteration}.json"
  writer_observation="$RUN_TMP/workspace-writer-${iteration}.json"
  capture_evidence="$RUN_TMP/gateway-response-ids-${iteration}.jsonl"
  capture_port_file="$RUN_TMP/gateway-capture-${iteration}.port"
  result_file="$OUTPUT_DIR/${request_id}.json"

  start_capture_proxy "$capture_evidence" "$capture_port_file"
  capture_port="$(<"$capture_port_file")"
  ALLOW_LOOPBACK_PROVIDER=1 "$CREATE_PROFILE" --home "$profile_home" \
    --base-url "http://127.0.0.1:${capture_port}/v1" \
    --key-file "$KEY_FILE" --request-id "$request_id"
  assert_profile_inventory "$profile_home"
  write_prompt "$prompt_file"
  run_hermes "$profile_home" "$prompt_file" "$usage_file" "$response_file" "$observations" \
    "$workspace_evidence" "$writer_observation"
  stop_capture_proxy
  validate_container_observations "$observations" "$profile_home"

  # The process-scoped environment and profile must not leave a gateway,
  # cron scheduler, MCP registry, skills, or memory. The terminal container
  # persists only long enough to recover its work product, then is removed.
  assert_profile_inventory "$profile_home"
  snapshot_shared_state "$RUN_TMP/shared-after-${iteration}.json"
  assemble_result "$request_id" "$usage_file" "$response_file" "$observations" "$workspace_evidence" \
    "$RUN_TMP/shared-before.json" "$RUN_TMP/shared-after-${iteration}.json" "$result_file" "$profile_home" \
    "$capture_evidence"
  remove_hermes_containers
  [ -z "$(DOCKER_HOST="$DOCKER_HOST" "$DOCKER_BIN" ps -aq --filter label=hermes-agent=1)" ] \
    || { echo "Hermes terminal container cleanup was incomplete." >&2; exit 1; }
  echo "Hermes isolated suite accepted: $result_file"
done

# Final before/after guard covers the whole repeated suite, including every
# independent HERMES_HOME. readlink/stat/sha256 metadata are in each snapshot.
snapshot_shared_state "$RUN_TMP/shared-after.json"
python3 - "$RUN_TMP/shared-before.json" "$RUN_TMP/shared-after.json" <<'PY'
import json
from pathlib import Path
import sys
before = json.loads(Path(sys.argv[1]).read_text())["sha256"]
after = json.loads(Path(sys.argv[2]).read_text())["sha256"]
if before != after:
    raise SystemExit("protected default/hermesia/active_profile/LaunchAgents state changed")
print(f"shared-state-sha256={after}")
PY

echo "Hermes smoke suite passed $REPEAT consecutive isolated run(s)."
