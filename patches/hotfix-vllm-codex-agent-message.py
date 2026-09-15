#!/usr/bin/env python3
"""Fail-closed compatibility patch for Codex ``agent_message`` history.

Codex may replay internal sub-agent chat as ``agent_message`` input items,
which are outside the pinned OpenAI SDK input union used by vLLM. When this
patch is explicitly enabled, it recognizes only the evidenced dictionary
shape with exactly one ``input_text`` content part and converts it to a minimal
assistant message. All other shapes are left for stock validation to reject.

The conversion intentionally and irreversibly drops ``id``, ``author``,
``recipient``, and ``internal_chat_message_metadata_passthrough``. It preserves
the text as conversational history, but it cannot preserve or reconstruct who
said it to whom. This compatibility layer is therefore unsuitable anywhere
that routing, attribution, or audit provenance must reach the model.

The complete pinned validator method is source-locked by SHA-256. Both the
stock method and the separately source-locked issue #138 postimage are accepted
as preimages, so the two opt-in patches compose in that order. This patcher
does not import vLLM or its GPU/runtime dependencies. Publication is atomic,
and a failed post-publication check atomically restores the original bytes and
mode.

Usage:
  hotfix-vllm-codex-agent-message.py [--status] [TARGET]
"""
from __future__ import annotations

import ast
import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py"
)
TAG = "[codex-agent-message-hotfix]"
MARKER = (
    "# [codex-agent-message-hotfix] Convert the evidenced Codex agent_message."
)
TYPE_ALIAS_GUARD = (
    "ResponseInputOutputItem: TypeAlias = "
    "ResponseInputItemParam | ResponseOutputItem"
)
INPUT_FIELD_GUARD = "\n    input: str | list[ResponseInputOutputItem]\n"

# vLLM 752a3a504485790a2e8491cacbb35c137339ad34 stock method and the
# independent issue #138 patch postimage accepted by this recipe.
PREIMAGE_HASHES = {
    "stock": "2412484a81e8679cedf1934287f1b4187a72bf6e8c910c8ecad463b29b79d9d7",
    "issue138": "536f3a305821445328c1f2131b898bef8a8f0c7d278cef4ba29701501eaf3d78",
}
POSTIMAGE_HASHES = {
    "stock-applied": "010ba578d77333a3e3d2d6b31d1a7b692521a48d899503459ebfede9d09c6851",
    "issue138-applied": "a5af28655454f4f1cdd22e89eaaa0c5400578cdd0efbb6498412d0b8639efc7e",
}

BRANCH_ANCHOR = '''            if item_type == "function_call":
'''

AGENT_BRANCH = '''            codex_agent_message = False
            agent_message_content = None
            if item_type == "agent_message":
                agent_message_content = item.get("content")
                metadata_key = "internal_chat_message_metadata_passthrough"
                has_metadata = metadata_key in item
                agent_message_metadata = item.get(metadata_key)
                codex_agent_message = (
                    len(item) == (6 if has_metadata else 5)
                    and isinstance(item.get("id"), str)
                    and bool(item["id"])
                    and isinstance(item.get("author"), str)
                    and bool(item["author"])
                    and isinstance(item.get("recipient"), str)
                    and bool(item["recipient"])
                    and isinstance(agent_message_content, list)
                    and len(agent_message_content) == 1
                    and isinstance(agent_message_content[0], dict)
                    and len(agent_message_content[0]) == 2
                    and agent_message_content[0].get("type") == "input_text"
                    and isinstance(agent_message_content[0].get("text"), str)
                    and (
                        not has_metadata
                        or (
                            isinstance(agent_message_metadata, dict)
                            and len(agent_message_metadata) == 2
                            and isinstance(
                                agent_message_metadata.get("turn_id"), str
                            )
                            and bool(agent_message_metadata["turn_id"])
                            and isinstance(
                                agent_message_metadata.get("create_time"),
                                (int, float),
                            )
                            and not isinstance(
                                agent_message_metadata.get("create_time"), bool
                            )
                        )
                    )
                )

            if codex_agent_message:
                # [codex-agent-message-hotfix] Convert the evidenced Codex agent_message.
                processed_input.append({
                    "type": "message",
                    "role": "assistant",
                    "content": agent_message_content,
                })

            elif item_type == "function_call":
'''

STATUS_ANCHORS = (
    '''            codex_agent_message = False
            agent_message_content = None
            if item_type == "agent_message":
''',
    '''                    and isinstance(agent_message_content, list)
                    and len(agent_message_content) == 1
                    and isinstance(agent_message_content[0], dict)
                    and len(agent_message_content[0]) == 2
                    and agent_message_content[0].get("type") == "input_text"
                    and isinstance(agent_message_content[0].get("text"), str)
''',
    '''            if codex_agent_message:
                # [codex-agent-message-hotfix] Convert the evidenced Codex agent_message.
                processed_input.append({
                    "type": "message",
                    "role": "assistant",
                    "content": agent_message_content,
                })
''',
)


class PatchError(RuntimeError):
    """Source identity or transactional publication failure."""


def _mode(target: Path) -> int:
    return stat.S_IMODE(target.stat().st_mode)


def _decode(raw: bytes, target: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatchError(f"target is not UTF-8: {target}") from error


def _compile(source: str, target: Path) -> None:
    try:
        compile(source, str(target), "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise PatchError(
            f"target does not compile: {type(error).__name__}: {error}"
        ) from error


def _extract_method(source: str, target: Path) -> str:
    try:
        tree = ast.parse(source, filename=str(target))
    except SyntaxError as error:
        raise PatchError(f"target does not parse: {error}") from error

    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "input_item_parsing"
    ]
    if len(matches) != 1:
        raise PatchError(
            f"expected one input_item_parsing method, found {len(matches)}"
        )
    method_node = matches[0]
    if not method_node.decorator_list or method_node.end_lineno is None:
        raise PatchError("input_item_parsing method boundaries are unavailable")
    start_line = min(decorator.lineno for decorator in method_node.decorator_list)
    lines = source.splitlines(keepends=True)
    method = "".join(lines[start_line - 1 : method_node.end_lineno])
    if not method.endswith("\n") or source.count(method) != 1:
        raise PatchError("input_item_parsing method boundary is not exact")
    return method


def _method_state(method: str, marker_count: int) -> str:
    digest = hashlib.sha256(method.encode("utf-8")).hexdigest()
    for state, expected in PREIMAGE_HASHES.items():
        if digest == expected and marker_count == 0:
            return state
    for state, expected in POSTIMAGE_HASHES.items():
        if digest == expected and marker_count == 1:
            return state
    raise PatchError(
        "complete input_item_parsing source lock mismatch "
        f"(sha256={digest}, marker={marker_count})"
    )


def _source_state(source: str, target: Path) -> tuple[str, str]:
    alias_count = source.count(TYPE_ALIAS_GUARD)
    field_count = source.count(INPUT_FIELD_GUARD)
    if alias_count != 1 or field_count != 1:
        raise PatchError(
            "outer Responses input-union guards drifted "
            f"(alias={alias_count}, input_field={field_count})"
        )
    alias_at = source.find(TYPE_ALIAS_GUARD)
    field_at = source.find(INPUT_FIELD_GUARD)
    method = _extract_method(source, target)
    method_at = source.find(method)
    if not alias_at < field_at < method_at:
        raise PatchError("outer Responses input-union guards are out of order")
    marker_count = source.count(MARKER)
    state = _method_state(method, marker_count)
    _compile(source, target)
    return state, method


def _status_source_state(source: str, target: Path) -> str:
    alias_count = source.count(TYPE_ALIAS_GUARD)
    field_count = source.count(INPUT_FIELD_GUARD)
    if alias_count != 1 or field_count != 1:
        raise PatchError(
            "outer Responses input-union guards drifted "
            f"(alias={alias_count}, input_field={field_count})"
        )
    alias_at = source.find(TYPE_ALIAS_GUARD)
    field_at = source.find(INPUT_FIELD_GUARD)
    method = _extract_method(source, target)
    method_at = source.find(method)
    if not alias_at < field_at < method_at:
        raise PatchError("outer Responses input-union guards are out of order")

    marker_count = source.count(MARKER)
    try:
        state = _method_state(method, marker_count)
    except PatchError:
        anchor_counts = tuple(method.count(anchor) for anchor in STATUS_ANCHORS)
        anchor_positions = tuple(method.find(anchor) for anchor in STATUS_ANCHORS)
        if (
            marker_count == 1
            and all(count == 1 for count in anchor_counts)
            and list(anchor_positions) == sorted(anchor_positions)
        ):
            state = "applied-with-extensions"
        else:
            raise PatchError(
                "status input_item_parsing source lock mismatch "
                f"(marker={marker_count}, anchors={anchor_counts})"
            )
    _compile(source, target)
    return state


def _patched_method(method: str, target: Path) -> str:
    if method.count(BRANCH_ANCHOR) != 1:
        raise PatchError("agent_message insertion anchor drifted")
    patched = method.replace(BRANCH_ANCHOR, AGENT_BRANCH, 1)
    _compile("class _PatchedHarness:\n" + patched, target)
    return patched


def _fsync_parent(parent: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_publish(target: Path, raw: bytes, mode: int) -> None:
    staged: Path | None = None
    fd = -1
    try:
        fd, name = tempfile.mkstemp(
            prefix=f".{target.name}.codex-agent-message.", dir=str(target.parent)
        )
        staged = Path(name)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, target)
        staged = None
        _fsync_parent(target.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        if staged is not None:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass


def _rollback(
    target: Path, original: bytes, original_mode: int, original_state: str
) -> None:
    _atomic_publish(target, original, original_mode)
    restored = target.read_bytes()
    if restored != original:
        raise PatchError("rollback byte verification failed")
    if _mode(target) != original_mode:
        raise PatchError("rollback mode verification failed")
    restored_state, _method = _source_state(_decode(restored, target), target)
    if restored_state != original_state:
        raise PatchError("rollback source-state verification failed")


def status(target: Path) -> int:
    try:
        if not target.is_file():
            raise PatchError(f"target file not found: {target}")
        state = _status_source_state(_decode(target.read_bytes(), target), target)
    except (OSError, PatchError) as error:
        print(f"[FAIL] {TAG} {error}", file=sys.stderr)
        return 1
    if state.endswith("-applied") or state == "applied-with-extensions":
        print(f"[OK] {TAG} already applied and verified ({state}): {target}")
    else:
        print(f"[OK] {TAG} compatible {state} source: {target}")
    return 0


def apply(target: Path) -> int:
    try:
        if not target.is_file():
            raise PatchError(f"target file not found: {target}")
        original = target.read_bytes()
        original_mode = _mode(target)
        source = _decode(original, target)
        state, method = _source_state(source, target)
    except (OSError, PatchError) as error:
        print(f"[FAIL] {TAG} {error}", file=sys.stderr)
        return 1

    if state.endswith("-applied"):
        print(f"[OK] {TAG} already applied and verified ({state}): {target}")
        return 0

    expected_state = f"{state}-applied"
    try:
        patched_method = _patched_method(method, target)
        updated_source = source.replace(method, patched_method, 1)
        updated = updated_source.encode("utf-8")
        new_state, _new_method = _source_state(updated_source, target)
        if new_state != expected_state:
            raise PatchError(
                f"replacement reached {new_state}, expected {expected_state}"
            )
    except PatchError as error:
        print(
            f"[FAIL] {TAG} replacement rejected before publication: {error}",
            file=sys.stderr,
        )
        return 1

    publication_started = False
    try:
        publication_started = True
        _atomic_publish(target, updated, original_mode)
        committed = target.read_bytes()
        if committed != updated:
            raise PatchError("post-publication byte verification failed")
        if _mode(target) != original_mode:
            raise PatchError("post-publication mode verification failed")
        committed_state, _method = _source_state(
            _decode(committed, target), target
        )
        if committed_state != expected_state:
            raise PatchError("post-publication source-state verification failed")
    except BaseException as error:
        rollback_error: BaseException | None = None
        if publication_started:
            try:
                _rollback(target, original, original_mode, state)
            except BaseException as restore_error:
                rollback_error = restore_error
        if rollback_error is None:
            print(
                f"[FAIL] {TAG} publication failed; original restored: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
        else:
            print(
                f"[FAIL] {TAG} publication failed and rollback verification failed: "
                f"{type(error).__name__}: {error}; rollback: "
                f"{type(rollback_error).__name__}: {rollback_error}",
                file=sys.stderr,
            )
        return 1

    print(f"[OK] {TAG} patched and verified ({expected_state}): {target}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    check_only = False
    if args and args[0] == "--status":
        check_only = True
        args.pop(0)
    if len(args) > 1 or (args and args[0].startswith("--")):
        print(f"usage: {Path(argv[0]).name} [--status] [TARGET]", file=sys.stderr)
        return 2
    target = Path(args[0]) if args else DEFAULT_TARGET
    return status(target) if check_only else apply(target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
