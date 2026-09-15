#!/usr/bin/env python3
"""Fail-closed compatibility patch for issue #138 Responses history replay.

The pinned vLLM Responses request validator already canonicalizes assistant
output messages when their item-level ``type`` is ``message``.  Some Responses
clients replay a singleton assistant ``output_text`` part without that one
item-level discriminator.  When explicitly enabled by the recipe, this patch
recognizes only that lossless singleton shape, inserts ``type='message'``, and
lets the existing pinned coercion supply the missing id, status, and
annotations.

This patcher source-locks the complete pinned validator method and its outer
input-union declarations.  It does not import vLLM or its GPU/runtime
dependencies.  Publication is atomic and a failed post-publication check
atomically restores the original bytes and mode.

Usage:
  hotfix-vllm-issue138-responses-history.py [--status] [TARGET]
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
TAG = "[issue138-hotfix]"
MARKER = (
    "# [issue138-hotfix] Normalize the observed singleton type-less "
    "assistant output replay."
)
CODEX_MARKER = (
    "# [codex-agent-message-hotfix] Convert the evidenced Codex agent_message."
)
CODEX_COMBINED_METHOD_SHA256 = (
    "a5af28655454f4f1cdd22e89eaaa0c5400578cdd0efbb6498412d0b8639efc7e"
)
TYPE_ALIAS_GUARD = (
    "ResponseInputOutputItem: TypeAlias = "
    "ResponseInputItemParam | ResponseOutputItem"
)
INPUT_FIELD_GUARD = "\n    input: str | list[ResponseInputOutputItem]\n"

# vLLM 752a3a504485790a2e8491cacbb35c137339ad34
# vllm/entrypoints/openai/responses/protocol.py, complete pinned method.
OLD_METHOD = '''    @model_validator(mode="before")
    @classmethod
    def input_item_parsing(cls, data):
        """Parse input items that are missing required fields or that Pydantic
        cannot disambiguate in a Union of TypedDict / BaseModel types.

        Specifically handles:
        - function_call -> ResponseFunctionToolCall
        - reasoning     -> ResponseReasoningItem (auto-generates id)
        - message(role=assistant) -> ResponseOutputMessage (auto-generates
          id/status and annotations)

        Invalid structures are left for Pydantic to reject.
        """
        input_data = data.get("input")

        # Early return for None, strings, or bytes
        if input_data is None or isinstance(input_data, (str, bytes)):
            return data

        # Convert iterators (like ValidatorIterator) to list
        if not isinstance(input_data, list):
            try:
                input_data = list(input_data)
            except TypeError:
                # Not iterable, leave as-is for Pydantic to handle
                return data

        processed_input = []
        for item in input_data:
            if not isinstance(item, dict):
                processed_input.append(item)
                continue

            item_type = item.get("type")

            if item_type == "function_call":
                try:
                    processed_input.append(ResponseFunctionToolCall(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse function_call to ResponseFunctionToolCall, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)

            elif item_type == "reasoning":
                if "id" not in item:
                    item = {**item, "id": f"rs_{random_uuid()}"}
                try:
                    processed_input.append(ResponseReasoningItem(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse reasoning to ResponseReasoningItem, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)

            elif item_type == "message" and item.get("role") == "assistant":
                content = item.get("content")
                if not isinstance(content, list):
                    # String content is a valid EasyInputMessageParam,
                    # do not coerce it to ResponseOutputMessage
                    processed_input.append(item)
                    continue

                original_item = item
                item = dict(item)
                if "id" not in item:
                    item["id"] = f"msg_{random_uuid()}"
                if "status" not in item:
                    item["status"] = "completed"
                # ResponseOutputText requires annotations
                new_content = []
                for c in content:
                    if (
                        isinstance(c, dict)
                        and c.get("type") == "output_text"
                        and "annotations" not in c
                    ):
                        c = {**c, "annotations": []}
                    new_content.append(c)
                item["content"] = new_content
                try:
                    processed_input.append(ResponseOutputMessage(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse assistant message to ResponseOutputMessage, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(original_item)

            else:
                processed_input.append(item)

        data["input"] = processed_input
        return data
'''

NEW_METHOD = '''    @model_validator(mode="before")
    @classmethod
    def input_item_parsing(cls, data):
        """Parse input items that are missing required fields or that Pydantic
        cannot disambiguate in a Union of TypedDict / BaseModel types.

        Specifically handles:
        - function_call -> ResponseFunctionToolCall
        - reasoning     -> ResponseReasoningItem (auto-generates id)
        - message(role=assistant) -> ResponseOutputMessage (auto-generates
          id/status and annotations)

        Invalid structures are left for Pydantic to reject.
        """
        input_data = data.get("input")

        # Early return for None, strings, or bytes
        if input_data is None or isinstance(input_data, (str, bytes)):
            return data

        # Convert iterators (like ValidatorIterator) to list
        if not isinstance(input_data, list):
            try:
                input_data = list(input_data)
            except TypeError:
                # Not iterable, leave as-is for Pydantic to handle
                return data

        processed_input = []
        for item in input_data:
            if not isinstance(item, dict):
                processed_input.append(item)
                continue

            item_type = item.get("type")
            content = item.get("content")
            legacy_assistant_output = (
                "type" not in item
                and item.get("role") == "assistant"
                and isinstance(content, list)
                and len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "output_text"
                and isinstance(content[0].get("text"), str)
            )

            if item_type == "function_call":
                try:
                    processed_input.append(ResponseFunctionToolCall(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse function_call to ResponseFunctionToolCall, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)

            elif item_type == "reasoning":
                if "id" not in item:
                    item = {**item, "id": f"rs_{random_uuid()}"}
                try:
                    processed_input.append(ResponseReasoningItem(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse reasoning to ResponseReasoningItem, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(item)

            elif (
                item.get("role") == "assistant"
                and (item_type == "message" or legacy_assistant_output)
            ):
                if not isinstance(content, list):
                    # String content is a valid EasyInputMessageParam,
                    # do not coerce it to ResponseOutputMessage
                    processed_input.append(item)
                    continue

                original_item = item
                item = dict(item)
                if legacy_assistant_output:
                    # [issue138-hotfix] Normalize the observed singleton type-less assistant output replay.
                    item["type"] = "message"
                if "id" not in item:
                    item["id"] = f"msg_{random_uuid()}"
                if "status" not in item:
                    item["status"] = "completed"
                # ResponseOutputText requires annotations
                new_content = []
                for c in content:
                    if (
                        isinstance(c, dict)
                        and c.get("type") == "output_text"
                        and "annotations" not in c
                    ):
                        c = {**c, "annotations": []}
                    new_content.append(c)
                item["content"] = new_content
                try:
                    processed_input.append(ResponseOutputMessage(**item))
                except ValidationError:
                    logger.debug(
                        "Failed to parse assistant message to ResponseOutputMessage, "
                        "leaving for Pydantic validation"
                    )
                    processed_input.append(original_item)

            else:
                processed_input.append(item)

        data["input"] = processed_input
        return data
'''


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
        raise PatchError(f"target does not compile: {type(error).__name__}: {error}") from error

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



def _source_state(source: str, target: Path) -> str:
    alias_count = source.count(TYPE_ALIAS_GUARD)
    field_count = source.count(INPUT_FIELD_GUARD)
    if alias_count != 1 or field_count != 1:
        raise PatchError(
            "outer Responses input-union guards drifted "
            f"(alias={alias_count}, input_field={field_count})"
        )

    alias_at = source.find(TYPE_ALIAS_GUARD)
    field_at = source.find(INPUT_FIELD_GUARD)
    old_count = source.count(OLD_METHOD)
    new_count = source.count(NEW_METHOD)
    marker_count = source.count(MARKER)
    codex_marker_count = source.count(CODEX_MARKER)
    if not alias_at < field_at:
        raise PatchError("outer Responses input-union guards are out of order")

    state = (old_count, new_count, marker_count)
    if state == (1, 0, 0) and codex_marker_count == 0:
        method_at = source.find(OLD_METHOD)
        result = "stock"
    elif state == (0, 1, 1) and codex_marker_count == 0:
        method_at = source.find(NEW_METHOD)
        result = "applied"
    elif state == (0, 0, 1) and codex_marker_count == 1:
        method = _extract_method(source, target)
        digest = hashlib.sha256(method.encode("utf-8")).hexdigest()
        if digest != CODEX_COMBINED_METHOD_SHA256:
            raise PatchError(
                "combined issue138/Codex method source lock mismatch "
                f"(sha256={digest})"
            )
        method_at = source.find(method)
        result = "applied-with-codex"
    else:
        raise PatchError(
            "complete input_item_parsing source lock mismatch "
            f"(old={old_count}, new={new_count}, marker={marker_count}, "
            f"codex_marker={codex_marker_count})"
        )
    if field_at >= method_at:
        raise PatchError("outer Responses input-union guards are out of order")
    _compile(source, target)
    return result


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
            prefix=f".{target.name}.issue138.", dir=str(target.parent)
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


def _rollback(target: Path, original: bytes, original_mode: int) -> None:
    _atomic_publish(target, original, original_mode)
    restored = target.read_bytes()
    if restored != original:
        raise PatchError("rollback byte verification failed")
    if _mode(target) != original_mode:
        raise PatchError("rollback mode verification failed")
    source = _decode(restored, target)
    if _source_state(source, target) != "stock":
        raise PatchError("rollback source-state verification failed")


def status(target: Path) -> int:
    try:
        if not target.is_file():
            raise PatchError(f"target file not found: {target}")
        source = _decode(target.read_bytes(), target)
        state = _source_state(source, target)
    except (OSError, PatchError) as error:
        print(f"[FAIL] {TAG} {error}", file=sys.stderr)
        return 1
    if state == "stock":
        print(f"[OK] {TAG} compatible stock source: {target}")
    else:
        print(f"[OK] {TAG} already applied and verified: {target}")
    return 0


def apply(target: Path) -> int:
    try:
        if not target.is_file():
            raise PatchError(f"target file not found: {target}")
        original = target.read_bytes()
        original_mode = _mode(target)
        source = _decode(original, target)
        state = _source_state(source, target)
    except (OSError, PatchError) as error:
        print(f"[FAIL] {TAG} {error}", file=sys.stderr)
        return 1

    if state in ("applied", "applied-with-codex"):
        print(f"[OK] {TAG} already applied and verified: {target}")
        return 0

    updated_source = source.replace(OLD_METHOD, NEW_METHOD, 1)
    updated = updated_source.encode("utf-8")
    try:
        if _source_state(updated_source, target) != "applied":
            raise PatchError("replacement did not reach the exact applied state")
    except PatchError as error:
        print(f"[FAIL] {TAG} replacement rejected before publication: {error}", file=sys.stderr)
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
        committed_source = _decode(committed, target)
        if _source_state(committed_source, target) != "applied":
            raise PatchError("post-publication source-state verification failed")
    except BaseException as error:
        rollback_error: BaseException | None = None
        if publication_started:
            try:
                _rollback(target, original, original_mode)
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

    print(f"[OK] {TAG} patched and verified: {target}")
    return 0


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    check_only = False
    if args and args[0] == "--status":
        check_only = True
        args.pop(0)
    if len(args) > 1 or (args and args[0].startswith("--")):
        print(
            f"usage: {Path(argv[0]).name} [--status] [TARGET]",
            file=sys.stderr,
        )
        return 2
    target = Path(args[0]) if args else DEFAULT_TARGET
    return status(target) if check_only else apply(target)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
