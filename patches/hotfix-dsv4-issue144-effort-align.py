#!/usr/bin/env python3
"""Issue #144: relocate the reasoning-effort directive so prefix-cache blocks are shared across effort buckets (opt-in).

Symptom
-------
The checkpoint encoder (``encoding/encoding_dsv4.py``, installed by the compose
entrypoint as ``vllm/tokenizers/deepseek_v4_encoding.py``) front-inserts the
reasoning-effort directive immediately after BOS and before all system content
whenever ``thinking_mode == "thinking"``.  The directive is a static,
non-256-aligned segment: BOS+directive is 93 tokens for ``max``/``xhigh``, 80
for ``high`` (the live lane's ``DEFAULT_THINKING=high`` default), and 0 extra
tokens for ``low``/``off``/``medium`` (empty directive).  vLLM's prefix cache
hashes 256-token blocks chained on the parent hash, so requests that differ
only in reasoning effort diverge at block 0 and share **zero** blocks: the
cache is partitioned into the effort buckets {low,off,medium} /
{high,DEFAULT} / {max,xhigh}.  Measured on the live 2xGB10 lane: cross-bucket
hit rate exactly 0, intra-bucket 96-98%.

Fix
---
Render the directive at the END of the leading run of system messages instead
of in front of it.  The bytes of BOS + system prompt + tools are then identical
for every effort, all their full blocks hash identically across buckets, and
only the short directive tail (0/~80/~93 tokens) plus the junction block
diverges.  Rendering changes ONLY for ``high``/``max`` conversations that begin
with at least one system message:

  stock:   BOS + directive + system-region + rest
  aligned: BOS + system-region + "\n\n" + directive + rest

``low`` renders (empty directive) and conversations with no leading system
message are byte-identical to stock.  Measured with the live tokenizer on a
4646-token agent-shaped prompt: stock shares 0 full blocks across buckets;
aligned shares 18/18 cacheable blocks (shared token prefix 4630; the BPE
junction merge costs exactly 1 token).

Gating and fail-closed operation
--------------------------------
The compose entrypoint invokes this script only when
``DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN`` is exactly ``1`` (default ``0`` = stock
renderer, this script never runs) and chains it with ``|| exit 1``, AFTER the
encoder copy and after the other encoder co-patchers (issue #21, Vision-Exp,
assistant-final).  The anchored region (effort prefix + system branch of
``render_message``) is disjoint from all of them and must appear exactly once;
the region constants are themselves sha256-pinned.  Known whole-file
identities are recognized and reported (snapshot ``b4bbb74b…``, live chain
``07432ce4…``, and their patched forms); an unrecognized file with an intact
anchor is patchable (the encoder is co-owned by gated patchers and the
checkpoint revision is unpinned by default), while a missing or duplicated
anchor fails closed.  After writing, a render self-check must pass or the
original bytes are restored and the boot fails:

- ``low``/``None``-effort and chat-mode renders must be byte-identical to the
  pre-patch renderer (reconstructed by reversing the region replace);
- ``high``/``max`` renders must equal exactly
  ``BOS + system-region + "\n\n" + directive + rest``;
- conversations without a leading system message must render byte-identically
  to stock at every effort;
- context continuations must stay byte-identical.

Usage (inside container, after encoder copy):
  python3 hotfix-dsv4-issue144-effort-align.py            # apply (or verify if already applied)
  python3 hotfix-dsv4-issue144-effort-align.py --status   # classify the served encoder copy
  python3 hotfix-dsv4-issue144-effort-align.py --check    # preflight: classify the bytes the next boot will copy
                                                          # (env-resolved snapshot; falls back to the served copy)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)
MARK = "[dspark-issue144-effort-align]"

# Known whole-file identities (recognition + reporting; the anchor governs).
SNAPSHOT_SHA256 = "b4bbb74bbb11a9c8ada04daa30cc7de7dba3abba08e9ade06d38b51a3d0d1701"  # 48095b345 encoding_dsv4.py, 36707 B
LIVE_CHAIN_SHA256 = "07432ce4758143247b80ef269f5855cb4536fd610422e008cfeae36de65ad547"  # + issue21 + vision-exp + assistant-final, 39960 B
PATCHED_SNAPSHOT_SHA256 = "f99de710937b44fc2af915a50f4835874fccc79a77869a69e8540788db88ce8c"
PATCHED_LIVE_CHAIN_SHA256 = "a976ae86f15ea0b7692e219d72ae655c3f7aa83a1c6a2b3d9202a722fb5cb21d"
KNOWN_IDENTITIES = {
    SNAPSHOT_SHA256: "stock (pristine 48095b345 snapshot)",
    LIVE_CHAIN_SHA256: "stock (48095b345 + issue21 + vision-exp + assistant-final)",
    PATCHED_SNAPSHOT_SHA256: "patched (pristine 48095b345 snapshot)",
    PATCHED_LIVE_CHAIN_SHA256: "patched (48095b345 + issue21 + vision-exp + assistant-final)",
}

REGION_OLD = (
    "    # Reasoning effort prefix (only at index 0 in thinking mode; \"low\" adds nothing)\n"
    "    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT\n"
    "    assert reasoning_effort in REASONING_EFFORT_PROMPTS, \\\n"
    "        f\"Invalid reasoning effort: {reasoning_effort}, expected one of {list(REASONING_EFFORT_PROMPTS)}\"\n"
    "    if index == 0 and thinking_mode == \"thinking\":\n"
    "        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]\n"
    "\n"
    "    if role == \"system\":\n"
    "        prompt += system_msg_template.format(content=content or \"\")\n"
    "        if tools:\n"
    "            prompt += \"\\n\\n\" + render_tools(tools)\n"
    "        if response_format:\n"
    "            prompt += \"\\n\\n\" + response_format_template.format(schema=to_json(response_format))\n"
)
REGION_NEW = (
    "    # Reasoning effort prefix (only at index 0 in thinking mode; \"low\" adds nothing)\n"
    "    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT\n"
    "    assert reasoning_effort in REASONING_EFFORT_PROMPTS, \\\n"
    "        f\"Invalid reasoning effort: {reasoning_effort}, expected one of {list(REASONING_EFFORT_PROMPTS)}\"\n"
    "    # [dspark-issue144-effort-align] Issue #144: the effort directive renders at\n"
    "    # the END of the leading run of system messages instead of in front of\n"
    "    # it, so the bytes of BOS + system prompt + tools are identical for\n"
    "    # every reasoning effort and their prefix-cache blocks are shared\n"
    "    # across the effort buckets ({low,off,medium} / {high,DEFAULT} /\n"
    "    # {max,xhigh}); only the short directive tail (0/~79/~93 tokens)\n"
    "    # diverges. A conversation with no leading system message keeps the\n"
    "    # stock front position.\n"
    "    _effort_directive_tail = \"\"\n"
    "    if thinking_mode == \"thinking\":\n"
    "        _leading_system_end = 0\n"
    "        while _leading_system_end < len(messages) and \\\n"
    "                messages[_leading_system_end].get(\"role\") == \"system\":\n"
    "            _leading_system_end += 1\n"
    "        if _leading_system_end == 0:\n"
    "            if index == 0:\n"
    "                prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]\n"
    "        elif index == _leading_system_end - 1:\n"
    "            _effort_directive_tail = REASONING_EFFORT_PROMPTS[reasoning_effort]\n"
    "\n"
    "    if role == \"system\":\n"
    "        prompt += system_msg_template.format(content=content or \"\")\n"
    "        if tools:\n"
    "            prompt += \"\\n\\n\" + render_tools(tools)\n"
    "        if response_format:\n"
    "            prompt += \"\\n\\n\" + response_format_template.format(schema=to_json(response_format))\n"
    "        if _effort_directive_tail:\n"
    "            prompt += (\"\\n\\n\" if prompt else \"\") + _effort_directive_tail\n"
)

# Self-pins: the region constants must not drift inside this file.
REGION_OLD_SHA256 = "602770183715f792eb1d15906ff057497dc7ba53781fbf3c492ef4fcaba0ec7b"
REGION_NEW_SHA256 = "16cbbc7f7d95cf67f6ba0a05f17b4f7f4003193457a0c5e0cdc9845e4acc22d4"


class HotfixError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_self() -> None:
    if _sha256(REGION_OLD.encode()) != REGION_OLD_SHA256:
        raise HotfixError("REGION_OLD does not match its pinned sha256")
    if _sha256(REGION_NEW.encode()) != REGION_NEW_SHA256:
        raise HotfixError("REGION_NEW does not match its pinned sha256")


def resolve_encoding_source(environ=os.environ) -> Path | None:
    """Mirror the compose entrypoint's ENCODING_SOURCE resolution."""
    explicit = environ.get("DSPARK_ENCODING_FILE")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    model = environ.get("DSPARK_MODEL", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
    hub_dir = model.replace("/", "--")
    revision = environ.get("DSPARK_REVISION")
    if revision:
        p = Path(
            f"/cache/huggingface/hub/models--{hub_dir}/snapshots/{revision}/encoding/encoding_dsv4.py"
        )
        if p.is_file():
            return p
    for pattern in (
        f"/cache/huggingface/hub/models--{hub_dir}/snapshots/*/encoding/encoding_dsv4.py",
        "/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*/encoding/encoding_dsv4.py",
    ):
        for candidate in sorted(glob.glob(pattern)):
            if Path(candidate).is_file():
                return Path(candidate)
    fallback = Path("/models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py")
    return fallback if fallback.is_file() else None


def inspect_bytes(data: bytes) -> str:
    """Classify encoder bytes; anything but exactly-one anchor fails closed."""
    _verify_self()
    old_n = data.count(REGION_OLD.encode())
    new_n = data.count(REGION_NEW.encode())
    marked = MARK.encode() in data
    if new_n == 1 and old_n == 0 and marked:
        return "patched"
    if old_n == 1 and new_n == 0 and not marked:
        return "stock"
    raise HotfixError(
        f"unsupported encoder bytes (anchor x{old_n}, patched region x{new_n}, "
        f"mark={marked}, sha256={_sha256(data)}); expected exactly one of either"
    )


def inspect(target: Path) -> tuple[str, bytes]:
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError(f"target is missing: {target}")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError(f"target is not a regular file: {target}")
    data = target.read_bytes()
    return inspect_bytes(data), data


def transform(stock: bytes) -> bytes:
    """Stock bytes -> patched bytes; refuses anything but exactly one site."""
    if inspect_bytes(stock) != "stock":
        raise HotfixError("transform requires stock bytes")
    patched = stock.replace(REGION_OLD.encode(), REGION_NEW.encode(), 1)
    compile(patched, "deepseek_v4_encoding.py", "exec")
    return patched


def _load_module(src: bytes, name: str, tmpdir: str):
    path = Path(tmpdir) / f"{name}.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HotfixError(f"cannot load module spec for {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHECK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
_CHECK_SYS = "You are a precise agent.\nFollow the rules.\n- keep diffs minimal"
_CHECK_FIXTURES = (
    ("sys+user", [
        {"role": "system", "content": _CHECK_SYS},
        {"role": "user", "content": "hello"},
    ]),
    ("toolsys+sys+user", [
        {"role": "system", "tools": _CHECK_TOOLS},
        {"role": "system", "content": _CHECK_SYS},
        {"role": "user", "content": "run it"},
    ]),
    ("user-only", [
        {"role": "user", "content": "no system here"},
    ]),
    ("sys+multiturn", [
        {"role": "system", "content": _CHECK_SYS},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]),
)


def _self_check(patched_src: bytes) -> tuple[bool, str]:
    """Render-level proof: relocation for high/max, byte parity for the rest."""
    stock_src = patched_src.replace(REGION_NEW.encode(), REGION_OLD.encode(), 1)
    with tempfile.TemporaryDirectory(prefix="dspark-issue144-check-") as tmpdir:
        try:
            stock = _load_module(stock_src, "enc_stock_i144", tmpdir)
            patched = _load_module(patched_src, "enc_patched_i144", tmpdir)
            bos = patched.bos_token
            prompts = patched.REASONING_EFFORT_PROMPTS
            for name, fixture in _CHECK_FIXTURES:
                msgs = lambda: [dict(m) for m in fixture]  # noqa: E731
                k = 0
                while k < len(fixture) and fixture[k].get("role") == "system":
                    k += 1
                low_stock = stock.encode_messages(msgs(), "thinking", reasoning_effort="low")
                low_patched = patched.encode_messages(msgs(), "thinking", reasoning_effort="low")
                if low_stock != low_patched:
                    return False, f"{name}: low render diverged"
                if stock.encode_messages(msgs(), "thinking") != patched.encode_messages(msgs(), "thinking"):
                    return False, f"{name}: default-effort render diverged"
                if stock.encode_messages(msgs(), "chat", reasoning_effort="high") != \
                        patched.encode_messages(msgs(), "chat", reasoning_effort="high"):
                    return False, f"{name}: chat-mode render diverged"
                sysregion = "".join(
                    stock.render_message(i, msgs(), "thinking", True, "low")
                    for i in range(k)
                )
                for effort in ("high", "max"):
                    got = patched.encode_messages(msgs(), "thinking", reasoning_effort=effort)
                    ref = stock.encode_messages(msgs(), "thinking", reasoning_effort=effort)
                    if k == 0:
                        if got != ref:
                            return False, f"{name}/{effort}: no-system render changed"
                        continue
                    rest = low_stock[len(bos) + len(sysregion):]
                    sep = "\n\n" if sysregion else ""
                    if ref != bos + prompts[effort] + sysregion + rest:
                        return False, f"{name}/{effort}: stock structure assumption broken"
                    if got != bos + sysregion + sep + prompts[effort] + rest:
                        return False, f"{name}/{effort}: relocated render malformed"
            ctx = [{"role": "system", "content": _CHECK_SYS}, {"role": "user", "content": "q1"}]
            tail = [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
            for effort in (None, "low", "high", "max"):
                a = stock.encode_messages(
                    [dict(m) for m in tail], "thinking",
                    context=[dict(m) for m in ctx], reasoning_effort=effort,
                )
                b = patched.encode_messages(
                    [dict(m) for m in tail], "thinking",
                    context=[dict(m) for m in ctx], reasoning_effort=effort,
                )
                if a != b:
                    return False, f"context continuation diverged at effort={effort}"
        except HotfixError:
            raise
        except Exception as err:  # broken/unimportable patch must fail closed
            return False, f"self-check raised {type(err).__name__}: {err}"
    return True, "relocation verified; low/chat/no-system/context renders byte-identical"


def apply(target: Path) -> str:
    state, data = inspect(target)
    if state == "patched":
        ok, why = _self_check(data)
        if not ok:
            raise HotfixError(f"already patched but self-check failed: {why}")
        return "already-patched"
    patched = transform(data)
    ok, why = _self_check(patched)
    if not ok:
        raise HotfixError(f"self-check failed before write: {why}")
    fd, tmp_name = tempfile.mkstemp(prefix=".dspark-issue144-", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(patched)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    verify_state, verify_data = inspect(target)
    if verify_state != "patched":
        raise HotfixError("post-apply verification failed")
    ok, why = _self_check(verify_data)
    if not ok:
        # Fail closed: never leave a written-but-unverified encoder behind.
        target.write_bytes(data)
        raise HotfixError(f"post-apply self-check failed, original restored: {why}")
    return "applied"


def _describe(state: str, data: bytes, target: Path) -> str:
    identity = KNOWN_IDENTITIES.get(_sha256(data), "unrecognized identity, anchor intact")
    return f"dspark-issue144-effort-align: {state} [{identity}] ({target})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="preflight: classify the bytes the next boot will copy")
    parser.add_argument("--status", action="store_true",
                        help="classify the served encoder copy")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.check:
            target = args.target or resolve_encoding_source() or PRODUCTION_TARGET
        else:
            target = args.target or PRODUCTION_TARGET
        if args.check or args.status:
            state, data = inspect(target)
            ok, why = _self_check(data if state == "patched" else transform(data))
            if not ok:
                raise HotfixError(f"render self-check failed: {why}")
            print(_describe(state, data, target))
            return 0
        outcome = apply(target)
        _, data = inspect(target)
        print(f"dspark-issue144-effort-align: {outcome} [{KNOWN_IDENTITIES.get(_sha256(data), 'unrecognized identity, anchor intact')}] ({target})")
        return 0
    except HotfixError as error:
        print(f"dspark-issue144-effort-align: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
