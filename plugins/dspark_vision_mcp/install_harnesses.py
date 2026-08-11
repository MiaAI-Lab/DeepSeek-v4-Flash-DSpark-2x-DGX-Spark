#!/usr/bin/env python3
"""Idempotent vision-MCP registration for pi / OMP / Hermes / opencode / goose /
grok / openclaw / zcode / prime-agent.

Used by scripts/install-dspark-vision-mcp.sh. Does not wipe existing MCP
entries — only upserts the ``dspark-vision`` server key (and copies the skill).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SUPPORTED = (
    "pi",
    "omp",
    "hermes",
    "opencode",
    "goose",
    "grok",
    "openclaw",
    "zcode",
    "prime",
)
SERVER_KEY = "dspark-vision"


def _log(msg: str) -> None:
    print(f"[vision-mcp] {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"[vision-mcp] WARN: {msg}", file=sys.stderr, flush=True)


def which(name: str) -> Path | None:
    path = shutil.which(name)
    if path:
        return Path(path).resolve()
    # Grok Build ships under ~/.grok/bin even when the current shell PATH is thin.
    if name == "grok":
        cand = Path.home() / ".grok" / "bin" / "grok"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def ensure_uvx() -> Path:
    uvx = which("uvx")
    if uvx:
        return uvx
    local = Path.home() / ".local" / "bin" / "uvx"
    if local.is_file() and os.access(local, os.X_OK):
        return local.resolve()
    _log("uvx not found; installing uv via astral.sh…")
    subprocess.run(
        ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        check=True,
    )
    if local.is_file() and os.access(local, os.X_OK):
        return local.resolve()
    uvx = which("uvx")
    if uvx:
        return uvx
    raise RuntimeError("uvx still missing after install")


def copy_skill(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    skill_src = src / "SKILL.md"
    if not skill_src.is_file():
        raise FileNotFoundError(f"missing skill: {skill_src}")
    # Replace tree contents so updates land cleanly.
    if dest_dir.exists():
        for child in dest_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    shutil.copy2(skill_src, dest_dir / "SKILL.md")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # OpenClaw uses JSON5 (comments / trailing commas). Prefer json5 if present.
        try:
            import json5  # type: ignore

            data = json5.loads(text)
        except ImportError:
            stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
            stripped = re.sub(r"(?m)//.*?$", "", stripped)
            stripped = re.sub(r",(\s*[}\]])", r"\1", stripped)
            data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def stdio_entry(uvx: Path, plugin: Path, base_url: str, *, style: str) -> dict[str, Any]:
    """Build a harness-specific stdio MCP server entry."""
    args = ["--from", str(plugin), "dspark-vision-mcp"]
    env = {"DSPARK_VL_BASE_URL": base_url}
    if style == "pi":
        return {
            "command": str(uvx),
            "args": args,
            "env": env,
            "directTools": ["describe_image", "ocr_image", "compare_images"],
        }
    if style == "omp":
        return {
            "type": "stdio",
            "command": str(uvx),
            "args": args,
            "env": env,
        }
    if style == "opencode":
        return {
            "type": "local",
            "command": [str(uvx), *args],
            "environment": env,
            "enabled": True,
        }
    if style == "hermes":
        return {
            "command": str(uvx),
            "args": args,
            "env": env,
            "enabled": True,
            "timeout": 180,
            "connect_timeout": 60,
        }
    if style == "goose":
        # https://goose-docs.ai/ — extensions in ~/.config/goose/config.yaml
        return {
            "type": "stdio",
            "name": SERVER_KEY,
            "display_name": "DSpark Vision",
            "description": (
                "Local Qwen3-VL sidecar tools (describe_image / ocr_image / "
                "compare_images) for DeepSeek-V4-Flash-0731"
            ),
            "enabled": True,
            "cmd": str(uvx),
            "args": args,
            "envs": env,
            "env_keys": [],
            "timeout": 300,
        }
    if style == "openclaw":
        # https://docs2.openclaw.ai/tools/mcp — mcp.servers in ~/.openclaw/openclaw.json
        return {
            "command": str(uvx),
            "args": args,
            "env": env,
            "enabled": True,
        }
    if style == "zcode":
        # https://zcode.z.ai/en/docs/mcp-services — mcp.servers in ~/.zcode/cli/config.json
        return {
            "command": str(uvx),
            "args": args,
            "env": env,
        }
    raise ValueError(f"unknown style: {style}")


def upsert_mcp_servers_json(
    path: Path,
    entry: dict[str, Any],
    *,
    servers_key: str = "mcpServers",
    settings: dict[str, Any] | None = None,
) -> None:
    data = load_json(path)
    if settings:
        cur = data.get("settings")
        if not isinstance(cur, dict):
            cur = {}
        cur.update(settings)
        data["settings"] = cur
    servers = data.get(servers_key)
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_KEY] = entry
    data[servers_key] = servers
    write_json(path, data)


# --- harness adapters -------------------------------------------------------


def detect_pi() -> bool:
    return which("pi") is not None and (Path.home() / ".pi" / "agent").is_dir()


def install_pi(ctx: dict[str, Any]) -> str:
    agent = Path.home() / ".pi" / "agent"
    settings_path = agent / "settings.json"
    settings = load_json(settings_path)
    packages = settings.get("packages")
    if not isinstance(packages, list):
        packages = []
    if "npm:pi-mcp-adapter" not in packages:
        _log("pi: installing npm:pi-mcp-adapter…")
        try:
            subprocess.run(
                ["pi", "install", "npm:pi-mcp-adapter", "--approve"],
                check=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            # Fall back to recording the package name; user can install later.
            _warn(f"pi install pi-mcp-adapter failed ({exc}); recording package in settings")
            packages.append("npm:pi-mcp-adapter")
            settings["packages"] = packages
            write_json(settings_path, settings)
    else:
        # Re-read in case pi install mutated settings.
        settings = load_json(settings_path)
        packages = settings.get("packages") if isinstance(settings.get("packages"), list) else packages

    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="pi")
    settings_extra = {"toolPrefix": "none"}
    for path in (Path.home() / ".config" / "mcp" / "mcp.json", agent / "mcp.json"):
        upsert_mcp_servers_json(path, entry, settings=settings_extra)
    copy_skill(ctx["skill"], agent / "skills" / "dspark-vision")
    return "installed (mcp.json + skill + pi-mcp-adapter)"


def detect_omp() -> bool:
    return which("omp") is not None and (Path.home() / ".omp").is_dir()


def install_omp(ctx: dict[str, Any]) -> str:
    agent = Path.home() / ".omp" / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="omp")
    upsert_mcp_servers_json(agent / "mcp.json", entry)
    # OMP discovers user skills under ~/.omp/agent/skills (and project .omp/skills).
    copy_skill(ctx["skill"], agent / "skills" / "dspark-vision")
    return "installed (~/.omp/agent/mcp.json + skill)"


def detect_hermes() -> bool:
    return which("hermes") is not None and (Path.home() / ".hermes" / "config.yaml").is_file()


def _yaml_keyed_block(entry: dict[str, Any], *, indent: int = 2) -> str:
    """Render ``{SERVER_KEY: entry}`` indented for nesting under a parent key."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML required for YAML harness merges (pip install pyyaml)"
        ) from exc

    dumped = yaml.safe_dump(
        {SERVER_KEY: entry},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()
    prefix = " " * indent
    lines = [(prefix + line) if line else line for line in dumped.splitlines()]
    return "\n".join(lines) + "\n"


def _upsert_yaml_mapping_entry(
    path: Path,
    *,
    parent_key: str,
    entry: dict[str, Any],
    header_comment: str,
) -> None:
    """Insert/replace ``parent_key.dspark-vision`` without rewriting the whole file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = _yaml_keyed_block(entry, indent=2)
    marker_begin = f"  # BEGIN {SERVER_KEY}"
    marker_end = f"  # END {SERVER_KEY}"

    if path.is_file():
        bak = path.with_suffix(path.suffix + ".bak-dspark-vision")
        if not bak.is_file():
            shutil.copy2(path, bak)

    if marker_begin in text and marker_end in text:
        pre, rest = text.split(marker_begin, 1)
        _, post = rest.split(marker_end, 1)
        path.write_text(
            pre + marker_begin + "\n" + block + marker_end + post, encoding="utf-8"
        )
        return

    parent_re = re.compile(rf"(?m)^{re.escape(parent_key)}:\s*$")
    if parent_re.search(text):
        m = parent_re.search(text)
        assert m is not None
        insert_at = m.end()
        injection = "\n" + marker_begin + "\n" + block + marker_end + "\n"
        path.write_text(text[:insert_at] + injection + text[insert_at:], encoding="utf-8")
        return

    appendix = (
        f"\n{header_comment}\n"
        f"{parent_key}:\n"
        f"{marker_begin}\n"
        f"{block}"
        f"{marker_end}\n"
    )
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + appendix, encoding="utf-8")


def _hermes_server_yaml(entry: dict[str, Any]) -> str:
    return _yaml_keyed_block(entry, indent=2)


def _upsert_hermes_mcp_servers(path: Path, entry: dict[str, Any]) -> None:
    _upsert_yaml_mapping_entry(
        path,
        parent_key="mcp_servers",
        entry=entry,
        header_comment=(
            "# ── DSpark local vision MCP (auto-installed by "
            "install-dspark-vision-mcp.sh) ──"
        ),
    )


def install_hermes(ctx: dict[str, Any]) -> str:
    path = Path.home() / ".hermes" / "config.yaml"
    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="hermes")
    _upsert_hermes_mcp_servers(path, entry)
    copy_skill(ctx["skill"], Path.home() / ".hermes" / "skills" / "dspark-vision")
    return "installed (mcp_servers in config.yaml + skill)"


def detect_opencode() -> bool:
    if which("opencode") is not None:
        return True
    cfg = Path.home() / ".config" / "opencode"
    return cfg.is_dir()


def install_opencode(ctx: dict[str, Any]) -> str:
    cfg_dir = Path.home() / ".config" / "opencode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Prefer opencode.json; fall back to creating it.
    path = cfg_dir / "opencode.json"
    if not path.is_file() and (cfg_dir / "config.json").is_file():
        path = cfg_dir / "config.json"

    data = load_json(path)
    mcp = data.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    mcp[SERVER_KEY] = stdio_entry(
        ctx["uvx"], ctx["plugin"], ctx["base_url"], style="opencode"
    )
    data["mcp"] = mcp
    write_json(path, data)
    copy_skill(ctx["skill"], cfg_dir / "skills" / "dspark-vision")
    return f"installed ({path} + skill)"


def detect_goose() -> bool:
    """goose CLI or an existing goose config dir (https://goose-docs.ai/)."""
    if which("goose") is not None:
        return True
    cfg = Path.home() / ".config" / "goose" / "config.yaml"
    return cfg.is_file()


def install_goose(ctx: dict[str, Any]) -> str:
    path = Path.home() / ".config" / "goose" / "config.yaml"
    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="goose")
    _upsert_yaml_mapping_entry(
        path,
        parent_key="extensions",
        entry=entry,
        header_comment=(
            "# ── DSpark local vision MCP extension (auto-installed by "
            "install-dspark-vision-mcp.sh) ──"
        ),
    )
    copy_skill(ctx["skill"], Path.home() / ".config" / "goose" / "skills" / "dspark-vision")
    return "installed (~/.config/goose/config.yaml extensions + skill)"


def detect_grok() -> bool:
    """Grok Build (https://docs.x.ai/build/) — CLI or ~/.grok/config.toml."""
    if which("grok") is not None:
        return True
    return (Path.home() / ".grok" / "config.toml").is_file()


def _grok_mcp_toml_block(uvx: Path, plugin: Path, base_url: str) -> str:
    # TOML inline table for env; quote paths that may contain specials.
    def q(s: str) -> str:
        return json.dumps(s)  # JSON strings are valid TOML basic strings

    return (
        f"[mcp_servers.{SERVER_KEY}]\n"
        f"command = {q(str(uvx))}\n"
        f"args = [{q('--from')}, {q(str(plugin))}, {q('dspark-vision-mcp')}]\n"
        f"env = {{ DSPARK_VL_BASE_URL = {q(base_url)} }}\n"
        "enabled = true\n"
        "startup_timeout_sec = 120\n"
        "tool_timeout_sec = 6000\n"
    )


def _upsert_grok_mcp_servers(path: Path, uvx: Path, plugin: Path, base_url: str) -> None:
    """Insert/replace [mcp_servers.dspark-vision] without rewriting the whole file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = _grok_mcp_toml_block(uvx, plugin, base_url)
    marker_begin = f"# BEGIN {SERVER_KEY}"
    marker_end = f"# END {SERVER_KEY}"

    if path.is_file():
        bak = path.with_suffix(".toml.bak-dspark-vision")
        if not bak.is_file():
            shutil.copy2(path, bak)

    if marker_begin in text and marker_end in text:
        pre, rest = text.split(marker_begin, 1)
        _, post = rest.split(marker_end, 1)
        path.write_text(
            pre + marker_begin + "\n" + block + marker_end + post, encoding="utf-8"
        )
        return

    # Remove a prior unmarked section if present (from manual edits / grok mcp add).
    section_re = re.compile(
        rf"(?ms)^\[mcp_servers\.{re.escape(SERVER_KEY)}\][^\[]*"
    )
    text = section_re.sub("", text)

    appendix = (
        "\n# ── DSpark local vision MCP (auto-installed by "
        "install-dspark-vision-mcp.sh) ──\n"
        f"{marker_begin}\n"
        f"{block}"
        f"{marker_end}\n"
    )
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + appendix, encoding="utf-8")


def install_grok(ctx: dict[str, Any]) -> str:
    path = Path.home() / ".grok" / "config.toml"
    _upsert_grok_mcp_servers(path, ctx["uvx"], ctx["plugin"], ctx["base_url"])
    copy_skill(ctx["skill"], Path.home() / ".grok" / "skills" / "dspark-vision")
    return "installed (~/.grok/config.toml mcp_servers + skill)"


def detect_openclaw() -> bool:
    """OpenClaw — CLI (openclaw/oclaw) or ~/.openclaw config."""
    if which("openclaw") is not None or which("oclaw") is not None:
        return True
    home = Path.home() / ".openclaw"
    return home.is_dir() or (home / "openclaw.json").is_file()


def install_openclaw(ctx: dict[str, Any]) -> str:
    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="openclaw")
    # Prefer CLI when available — does not connect during config edits.
    cli = which("openclaw") or which("oclaw")
    if cli is not None:
        try:
            subprocess.run(
                [
                    str(cli),
                    "mcp",
                    "set",
                    SERVER_KEY,
                    json.dumps(entry),
                ],
                check=True,
                timeout=60,
                capture_output=True,
                text=True,
            )
            copy_skill(ctx["skill"], Path.home() / ".openclaw" / "skills" / "dspark-vision")
            return f"installed (via {cli.name} mcp set + skill)"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            _warn(f"openclaw mcp set failed ({exc}); falling back to openclaw.json upsert")

    path = Path.home() / ".openclaw" / "openclaw.json"
    data = load_json(path)
    mcp = data.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    servers = mcp.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_KEY] = entry
    mcp["servers"] = servers
    data["mcp"] = mcp
    if path.is_file():
        bak = path.with_suffix(".json.bak-dspark-vision")
        if not bak.is_file():
            shutil.copy2(path, bak)
    write_json(path, data)
    copy_skill(ctx["skill"], Path.home() / ".openclaw" / "skills" / "dspark-vision")
    return "installed (~/.openclaw/openclaw.json mcp.servers + skill)"


def detect_zcode() -> bool:
    """ZCode (https://zcode.z.ai/en) — CLI or ~/.zcode tree."""
    if which("zcode") is not None:
        return True
    home = Path.home() / ".zcode"
    return home.is_dir()


def install_zcode(ctx: dict[str, Any]) -> str:
    # Native user scope: ~/.zcode/cli/config.json → mcp.servers
    # https://zcode.z.ai/en/docs/mcp-services
    path = Path.home() / ".zcode" / "cli" / "config.json"
    entry = stdio_entry(ctx["uvx"], ctx["plugin"], ctx["base_url"], style="zcode")
    data = load_json(path)
    mcp = data.get("mcp")
    if not isinstance(mcp, dict):
        mcp = {}
    servers = mcp.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_KEY] = entry
    mcp["servers"] = servers
    data["mcp"] = mcp
    if path.is_file():
        bak = path.with_suffix(".json.bak-dspark-vision")
        if not bak.is_file():
            shutil.copy2(path, bak)
    write_json(path, data)
    # ZCode-only skill path. Do NOT also write ~/.agents/skills/dspark-vision:
    # pi (and other agents) scan that tree and report a name collision with
    # ~/.pi/agent/skills/dspark-vision from install_pi().
    copy_skill(ctx["skill"], Path.home() / ".zcode" / "cli" / "skills" / "dspark-vision")
    return "installed (~/.zcode/cli/config.json mcp.servers + skill)"


def detect_prime() -> bool:
    """Prime Agent (https://github.com/PrimeIntellect-ai/prime-agent)."""
    if which("prime-agent") is not None:
        return True
    home = Path.home() / ".prime" / "agent"
    return home.is_dir() or (home / "settings.json").is_file()


def install_prime(ctx: dict[str, Any]) -> str:
    """Prime Agent: HTTP-only MCP — install a Python skill that hits the sidecar.

    https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/mcp-integrations.md
    """
    src = ctx["plugin"] / "prime_skill"
    if not (src / "SKILL.md").is_file() or not (src / "pyproject.toml").is_file():
        raise FileNotFoundError(f"missing prime skill template under {src}")

    dest = Path.home() / ".prime" / "agent" / "skills" / "dspark-vision"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    # Record sidecar base for the skill (settings.env is not standard; drop a
    # small sidecar.env next to the skill that __init__ can optionally read).
    env_path = dest / "sidecar.env"
    env_path.write_text(
        f"DSPARK_VL_BASE_URL={ctx['base_url']}\n",
        encoding="utf-8",
    )

    # Ensure agent dir exists even if prime-agent was never launched.
    agent = Path.home() / ".prime" / "agent"
    agent.mkdir(parents=True, exist_ok=True)
    # Optional note in settings.json skills list (additive path).
    settings_path = agent / "settings.json"
    settings = load_json(settings_path)
    skills = settings.get("skills")
    if not isinstance(skills, list):
        skills = []
    skill_ref = str(dest)
    if skill_ref not in skills and "dspark-vision" not in skills:
        skills.append(skill_ref)
        settings["skills"] = skills
        write_json(settings_path, settings)

    return (
        "installed (~/.prime/agent/skills/dspark-vision Python skill; "
        "Prime MCP is HTTP-only so this calls :8889 directly)"
    )


ADAPTERS: dict[str, tuple[Callable[[], bool], Callable[[dict[str, Any]], str]]] = {
    "pi": (detect_pi, install_pi),
    "omp": (detect_omp, install_omp),
    "hermes": (detect_hermes, install_hermes),
    "opencode": (detect_opencode, install_opencode),
    "goose": (detect_goose, install_goose),
    "grok": (detect_grok, install_grok),
    "openclaw": (detect_openclaw, install_openclaw),
    "zcode": (detect_zcode, install_zcode),
    "prime": (detect_prime, install_prime),
}


def parse_harnesses(raw: str) -> list[str]:
    raw = (raw or "auto").strip().lower()
    if raw in ("", "auto", "all"):
        return list(SUPPORTED)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in parts if p not in SUPPORTED]
    if unknown:
        raise SystemExit(f"unknown harness(es): {', '.join(unknown)} (supported: {', '.join(SUPPORTED)})")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plugin-dir",
        type=Path,
        default=None,
        help="path to plugins/dspark_vision_mcp (default: next to this file)",
    )
    ap.add_argument(
        "--harnesses",
        default=os.environ.get("VISION_MCP_HARNESSES", "auto"),
        help="auto | comma list: pi,omp,hermes,opencode,goose,grok,openclaw,zcode,prime",
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get(
            "DSPARK_VL_BASE_URL",
            f"http://127.0.0.1:{os.environ.get('VL_SIDECAR_PORT', '8889')}",
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any selected+detected harness fails",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="detect only; do not write configs",
    )
    args = ap.parse_args()

    plugin = (args.plugin_dir or Path(__file__).resolve().parent).resolve()
    skill = plugin / "skill"
    if not (plugin / "pyproject.toml").is_file():
        _warn(f"plugin dir looks wrong: {plugin}")
        return 1 if args.strict else 0
    if not (skill / "SKILL.md").is_file():
        _warn(f"missing skill at {skill / 'SKILL.md'}")
        return 1 if args.strict else 0

    try:
        uvx = ensure_uvx()
    except Exception as exc:  # noqa: BLE001
        _warn(f"cannot resolve uvx: {exc}")
        return 1 if args.strict else 0

    ctx = {
        "uvx": uvx,
        "plugin": plugin,
        "skill": skill,
        "base_url": args.base_url.rstrip("/"),
    }
    selected = parse_harnesses(args.harnesses)

    installed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for name in selected:
        detect, install = ADAPTERS[name]
        if not detect():
            skipped.append(name)
            _log(f"{name}: not detected — skip")
            continue
        if args.dry_run:
            _log(f"{name}: detected (dry-run)")
            installed.append(name)
            continue
        try:
            detail = install(ctx)
            _log(f"{name}: {detail}")
            installed.append(name)
        except Exception as exc:  # noqa: BLE001
            _warn(f"{name}: failed — {exc}")
            failed.append(name)

    _log(
        "summary: installed=["
        + ", ".join(installed)
        + "] skipped=["
        + ", ".join(skipped)
        + "] failed=["
        + ", ".join(failed)
        + "]"
    )
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
