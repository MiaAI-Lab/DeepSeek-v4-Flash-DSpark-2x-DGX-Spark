#!/usr/bin/env python3
"""Issue #191: fail-closed named/required tool_choice contract (opt-in).

Non-streaming Chat Completions with ``tool_choice`` = named function or
``"required"`` serialise ``tool_calls or []`` straight from the parser.  When
the XGrammar structural-tag mask is missed for a step (spec decode + async
scheduling, vLLM #49694 / #54437 family), the engine can return HTTP 200 with
zero calls, a foreign tool name, non-JSON arguments, or arguments that violate
a ``strict`` schema.  This patcher adds a terminal contract check on the head
node's serving layer:

* violation -> WARNING ``[issue191-toolcall]`` line with request id and reason;
* ``DSPARK_ISSUE191_TOOLCALL_MODE=failclosed`` (default): re-generate the same
  engine input up to ``DSPARK_ISSUE191_TOOLCALL_RETRIES`` (default 2) times and,
  if the contract still fails, return HTTP 500 instead of a wrong 200;
* ``DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK=1`` (default): the *last* of
  those retries swaps the prompt's trailing ``<think>`` marker for ``</think>``
  (exactly what ``thinking=false`` renders) and parses the reply with a
  thinking-off parser, so a request whose reasoning outran ``max_tokens``
  (the measured cause of the residual gate failures: ``finish_reason=length``
  with zero or a salvaged partial call) still answers a grammar-constrained
  tool call inside the same budget; ``0`` keeps every retry identical;
* ``DSPARK_ISSUE191_TOOLCALL_MODE=log``: only log (measurement mode).

The opt-in Compose gate runs this before ``vllm serve``.  It accepts only the
pinned Anemll 0.1.1 vLLM version and the exact post-issue55 source identity of
``entrypoints/openai/chat_completion/serving.py`` (issue #55 truncation hotfix
runs unconditionally before this one).  Applying is one same-directory atomic
replace; an already-patched target is verified but never rewritten.
Streaming responses are not covered (nothing can be retried after chunks were
sent); the raw violation rate is still visible through the ``log`` mode of the
non-streaming path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from pathlib import Path

PRODUCTION_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/"
    "chat_completion/serving.py"
)
EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
# Pristine image bytes (before the unconditional issue #55 hotfix): accepted by
# --check (the launcher preflights a fresh container), refused by apply.
PRISTINE_SHA256 = "6239fae503211193942d0e2037f7b0edcf71ed70271604701283bfc0453d202b"
PRISTINE_SIZE = 49_931
# Post-issue55 bytes (issue #55 hotfix is unconditional in Compose).
STOCK_SHA256 = "08ddb5f3b6cd8dd465208e787a7cdb45da308ffeb4e7bc5f8d40ccdec8e15f77"
STOCK_SIZE = 51_928
PATCHED_SHA256 = "bfeccebf2f304e4e018198ea785c94a39e782dda4d6feada548b15eddf7a4916"
PATCHED_SIZE = 63087
MARK = "# [issue191-hotfix] fail-closed named/required tool_choice contract"
ISSUE55_MARK = "# [issue55-hotfix] tool-call truncation safety"

# Hunk A: helper block, inserted right after the issue #55 helper.
HELPER_ANCHOR = b'''        _j.loads(s)
        return True
    except Exception:
        return False
import io
'''
HELPER_NEW = b'''        _j.loads(s)
        return True
    except Exception:
        return False
# [issue191-hotfix] fail-closed named/required tool_choice contract
import os as _issue191_os

_ISSUE191_MARK = "[issue191-toolcall]"
_ISSUE191_JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _issue191_env_int(name, default):
    raw = _issue191_os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(0, min(value, 5))


def _issue191_mode():
    raw = _issue191_os.environ.get("DSPARK_ISSUE191_TOOLCALL_MODE") or "failclosed"
    mode = raw.strip().lower()
    return mode if mode in ("failclosed", "log") else "failclosed"


def _issue191_thinkoff_fallback():
    """1 (default) = the last failclosed retry regenerates with thinking off."""
    raw = _issue191_os.environ.get("DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK")
    if raw is None or raw.strip() == "":
        return True
    return raw.strip() == "1"


def _issue191_reasoning_marker_ids(parser):
    """(think_start_id, think_end_id) taken from the request parser, else None."""
    reasoning = getattr(parser, "reasoning_parser", None)
    engine = getattr(reasoning, "_parser_engine", None)
    start = getattr(engine, "_reasoning_start_token_id", None)
    end = getattr(engine, "_reasoning_end_token_id", None)
    if isinstance(start, int) and isinstance(end, int) and start != end:
        return start, end
    return None


def _issue191_thinkoff_engine_input(engine_input, parser):
    """Copy of ``engine_input`` whose trailing <think> marker becomes </think>.

    That is byte-identical to rendering the same chat with ``thinking=false``
    (the DeepSeek-V4 encoder ends the generation header with <think> when
    thinking is on and with </think> when it is off), so the grammar constrains
    the reply from its first token. Returns None when
    the prompt does not end with the reasoning-start marker (thinking already
    off, unknown template, embeds/encoder input); the caller then retries as is.
    """
    markers = _issue191_reasoning_marker_ids(parser)
    if markers is None or not isinstance(engine_input, dict):
        return None
    ids = engine_input.get("prompt_token_ids")
    if not isinstance(ids, list) or not ids or ids[-1] != markers[0]:
        return None
    swapped = dict(engine_input)
    swapped["prompt_token_ids"] = list(ids[:-1]) + [markers[1]]
    for stale in ("prompt", "prompt_token_offsets", "assistant_tokens_mask"):
        swapped.pop(stale, None)
    return swapped


def _issue191_tool_specs(request):
    """name -> (parameters schema or None, strict) for the request tools."""
    specs = {}
    for tool in getattr(request, "tools", None) or ():
        function = getattr(tool, "function", None)
        name = getattr(function, "name", None)
        if not isinstance(name, str):
            continue
        specs[name] = (
            getattr(function, "parameters", None),
            bool(getattr(function, "strict", False)),
        )
    return specs


def _issue191_fallback_schema_error(value, schema):
    if not isinstance(value, dict):
        return "schema:<root>:type"
    properties = schema.get("properties") or {}
    for key in schema.get("required") or ():
        if key not in value:
            return f"schema:{key}:required"
    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in properties:
                return f"schema:{key}:additionalProperties"
    for key, sub in properties.items():
        if key not in value or not isinstance(sub, dict):
            continue
        expected = sub.get("type")
        types = _ISSUE191_JSON_TYPES.get(expected) if isinstance(expected, str) else None
        if types is None:
            continue
        item = value[key]
        if isinstance(item, bool) and expected in ("integer", "number"):
            return f"schema:{key}:type"
        if not isinstance(item, types):
            return f"schema:{key}:type"
    return None


def _issue191_schema_error(value, schema):
    """Short reason when ``value`` violates ``schema``; None when it conforms."""
    if not isinstance(schema, dict):
        return None
    try:
        import jsonschema as _js
    except Exception:
        return _issue191_fallback_schema_error(value, schema)
    try:
        validator_cls = _js.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        error = next(iter(validator_cls(schema).iter_errors(value)), None)
    except Exception:  # malformed schema or validator error: not a client violation
        return None
    if error is None:
        return None
    path = "/".join(str(part) for part in error.absolute_path)
    return f"schema:{path or '<root>'}:{error.validator}"


def _issue191_tool_contract_violation(request, response):
    """None when ``response`` satisfies the named/required tool_choice contract.

    Only non-streaming named or ``required`` tool_choice requests are checked.
    Returns a short machine-readable reason for the first violation found.
    """
    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice != "required":
            return None
        expected_name = None
    else:
        expected_name = getattr(getattr(tool_choice, "function", None), "name", None)
        if not isinstance(expected_name, str):
            return None
    choices = getattr(response, "choices", None)
    if not choices:
        return "no-choices"
    specs = _issue191_tool_specs(request)
    parallel = getattr(request, "parallel_tool_calls", True)
    for choice in choices:
        message = getattr(choice, "message", None)
        calls = list(getattr(message, "tool_calls", None) or ())
        if not calls:
            if str(getattr(choice, "finish_reason", "")) == "length":
                return "tool-call-truncated"
            return "tool-call-cardinality:0"
        if parallel is False and len(calls) != 1:
            return f"tool-call-cardinality:{len(calls)}"
        for call in calls:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None)
            if expected_name is not None and name != expected_name:
                return "tool-call-name"
            if expected_name is None and name not in specs:
                return "tool-call-name"
            arguments = getattr(function, "arguments", None)
            if not isinstance(arguments, str):
                return "tool-arguments-type"
            try:
                import json as _j
                parsed = _j.loads(arguments)
            except Exception:
                return "tool-arguments-json"
            if not isinstance(parsed, dict):
                return "tool-arguments-json"
            schema, strict = specs.get(name, (None, False))
            if strict and schema is not None:
                reason = _issue191_schema_error(parsed, schema)
                if reason is not None:
                    return "tool-arguments-" + reason
    return None
import io
'''

# Hunk B: terminal contract check + bounded regeneration in
# ``_create_chat_completion``.
TAIL_OLD = b'''        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
'''
TAIL_NEW = b'''        response = await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )
        # [issue191-hotfix] fail-closed named/required tool_choice contract
        violation = (
            _issue191_tool_contract_violation(request, response)
            if not isinstance(response, ErrorResponse)
            and not isinstance(sampling_params, BeamSearchParams)
            else None
        )
        if violation is None:
            return response
        mode = _issue191_mode()
        retries = _issue191_env_int("DSPARK_ISSUE191_TOOLCALL_RETRIES", 2)
        attempt = 0
        while True:
            logger.warning(
                "%s contract violation request=%s attempt=%d mode=%s reason=%s",
                _ISSUE191_MARK,
                request_id,
                attempt,
                mode,
                violation,
            )
            if mode != "failclosed":
                return response
            if attempt >= retries:
                return self.create_error_response(
                    f"{_ISSUE191_MARK} tool_choice contract violated after "
                    f"{attempt + 1} attempt(s): {violation}",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            attempt += 1
            retry_input = engine_input
            retry_parser = parser
            retry_kwargs = chat_template_kwargs
            retry_reasoning_ended = reasoning_ended
            fallback = "none"
            if (
                attempt >= retries
                and _issue191_thinkoff_fallback()
                and parser is not None
                and parser.reasoning_parser is not None
            ):
                swapped = _issue191_thinkoff_engine_input(engine_input, parser)
                if swapped is not None:
                    retry_input = swapped
                    retry_kwargs = dict(chat_template_kwargs or {})
                    retry_kwargs["thinking"] = False
                    retry_kwargs.pop("enable_thinking", None)
                    retry_parser = self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=retry_kwargs,
                        model_config=self.model_config,
                    )
                    retry_reasoning_ended = True
                    fallback = "thinkoff"
            logger.warning(
                "%s regenerating request=%s attempt=%d fallback=%s",
                _ISSUE191_MARK,
                request_id,
                attempt,
                fallback,
            )
            retry_generator = self.engine_client.generate(
                retry_input,
                sampling_params,
                f"{request_id}-issue191r{attempt}",
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
                reasoning_ended=retry_reasoning_ended,
                reasoning_parser_kwargs={
                    "chat_template_kwargs": retry_kwargs,
                }
                if retry_parser is not None and retry_parser.reasoning_parser is not None
                else None,
            )
            response = await self.chat_completion_full_generator(
                request,
                retry_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                parser=retry_parser,
                mm_token_counts=mm_token_counts,
            )
            violation = (
                _issue191_tool_contract_violation(request, response)
                if not isinstance(response, ErrorResponse)
                else None
            )
            if violation is None:
                return response

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
'''


class HotfixError(RuntimeError):
    """Expected compatibility or transaction failure."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(stock: bytes) -> bytes:
    """Pure transformation of the pinned post-issue55 bytes."""
    if stock.count(HELPER_ANCHOR) != 1 or stock.count(TAIL_OLD) != 1:
        raise HotfixError("anchor regions are not unique in the target")
    if MARK.encode() in stock:
        raise HotfixError("target already carries the issue191 mark")
    patched = stock.replace(HELPER_ANCHOR, HELPER_NEW, 1).replace(TAIL_OLD, TAIL_NEW, 1)
    compile(patched.decode("utf-8"), "serving.py", "exec")
    return patched


def _vllm_version(provider=importlib.metadata.version) -> str:
    try:
        value = provider("vllm")
    except Exception as error:
        raise HotfixError(f"vllm package metadata unavailable ({type(error).__name__})")
    if value != EXPECTED_VLLM_VERSION:
        raise HotfixError(
            f"unsupported vllm version {value!r}; expected {EXPECTED_VLLM_VERSION!r}"
        )
    return value


def inspect(target: Path, *, provider=importlib.metadata.version) -> tuple[str, bytes]:
    _vllm_version(provider)
    try:
        st = target.lstat()
    except FileNotFoundError:
        raise HotfixError("target is missing")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HotfixError("target is not a regular file")
    data = target.read_bytes()
    digest = _sha256(data)
    if digest == PATCHED_SHA256 and len(data) == PATCHED_SIZE:
        return "patched", data
    if digest == STOCK_SHA256 and len(data) == STOCK_SIZE:
        if ISSUE55_MARK.encode() not in data:
            raise HotfixError("issue #55 mark missing from the pinned bytes")
        return "stock-compatible", data
    if digest == PRISTINE_SHA256 and len(data) == PRISTINE_SIZE:
        if ISSUE55_MARK.encode() in data:
            raise HotfixError("pristine digest carries an issue #55 mark")
        return "stock-pristine", data
    raise HotfixError(
        f"unsupported target bytes sha256={digest} size={len(data)}; "
        "expected the pinned pristine, post-issue55 or post-issue191 identity"
    )


def apply(target: Path, *, provider=importlib.metadata.version) -> str:
    state, data = inspect(target, provider=provider)
    if state == "patched":
        return "already-patched"
    if state == "stock-pristine":
        raise HotfixError(
            "issue #55 hotfix has not been applied to this target yet; "
            "issue #191 must run after it (Compose ordering error)"
        )
    patched = transform(data)
    if _sha256(patched) != PATCHED_SHA256 or len(patched) != PATCHED_SIZE:
        raise HotfixError("transformed bytes do not match the pinned patched identity")
    fd, tmp_name = tempfile.mkstemp(prefix=".issue191-", dir=str(target.parent))
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
    verify_state, _ = inspect(target, provider=provider)
    if verify_state != "patched":
        raise HotfixError("post-apply verification failed")
    return "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify compatibility only")
    parser.add_argument("--status", action="store_true", help="print the target state")
    parser.add_argument("--target", type=Path, default=PRODUCTION_TARGET)
    args = parser.parse_args(argv)
    try:
        if args.check or args.status:
            state, _ = inspect(args.target)
            print(f"issue191-toolcall-failclosed: {state} ({args.target})")
            return 0
        outcome = apply(args.target)
        print(f"issue191-toolcall-failclosed: {outcome} ({args.target})")
        return 0
    except HotfixError as error:
        print(f"issue191-toolcall-failclosed: FAIL-CLOSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
