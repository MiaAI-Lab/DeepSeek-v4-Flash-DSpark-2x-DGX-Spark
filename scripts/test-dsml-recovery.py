#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-dsml-recovery.py (no vLLM needed).

Fixture identities, transform pins, patcher fail-closed/idempotency/rollback,
and the upstream vllm#52645 regression matrix replayed against the pinned
fixtures: DSML foreign-tool leak cases must recover, and the normal DSML and
tool-call paths must behave byte-identically to stock.

The behavior tests exec the pinned fixture modules under a synthetic ``vllm``
package with stdlib-level stubs for the serving-layer imports.  Only one
synthetic stack may be live at a time: the engine resolves some imports
lazily through ``sys.modules``, so stacks are built, used, and torn down
strictly sequentially.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXDIR = ROOT / "scripts" / "fixtures" / "dspark-dsml-recovery"
PATCHER = ROOT / "patches" / "hotfix-vllm-dsml-recovery.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"
ENVS_DOC = ROOT / "docs" / "ENVS.md"
PATCHES_DOC = ROOT / "docs" / "PATCHES.md"


def _load():
    spec = importlib.util.spec_from_file_location("dspark_dsml_recovery", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DR = _load()
GOOD = lambda name: DR.EXPECTED_VLLM_VERSION  # noqa: E731

FIXTURE = {
    spec.label: FIXDIR / f"{spec.label[:-3]}-752a3a504-stock.py" for spec in DR.SPECS
}
SUPPORT = {
    "vllm.parser.engine.events": FIXDIR / "support" / "events-752a3a504.py",
    "vllm.parser.engine.incremental_lexer": FIXDIR
    / "support"
    / "incremental_lexer-752a3a504.py",
    "vllm.parser.engine.token_id_scanner": FIXDIR
    / "support"
    / "token_id_scanner-752a3a504.py",
    "vllm.tool_parsers.utils": FIXDIR / "support" / "tool_parsers_utils-752a3a504.py",
}
# The harness is only meaningful against the exact pinned engine internals.
SUPPORT_SHA256 = {
    "vllm.parser.engine.events": (
        "493543e5832b721c67640c09a6ad664823423a98afe0bb53ba78518608047f0a",
        626,
    ),
    "vllm.parser.engine.incremental_lexer": (
        "c58797ba16d60a6a4db365dc678b38b756df83eea566c6e34dac2785f55c4cce",
        7_766,
    ),
    "vllm.parser.engine.token_id_scanner": (
        "c9db6d6a29865d65ba1adc523015488aad4d7dbef46b7539041e9efbe81bd034",
        11_349,
    ),
    "vllm.tool_parsers.utils": (
        "1e43768164c8907b29afd601ae7df09bf23418062a60a3e02da522810e31e10f",
        27_284,
    ),
}


def _stock_sources() -> dict[str, bytes]:
    return {spec.label: FIXTURE[spec.label].read_bytes() for spec in DR.SPECS}


def _patched_sources() -> dict[str, bytes]:
    return {
        spec.label: DR.transform(spec, FIXTURE[spec.label].read_bytes())
        for spec in DR.SPECS
    }


# ── Synthetic vllm stack ──────────────────────────────────────────────

PATCHABLE_MOD = {
    "vllm.parser.abstract_parser": "abstract_parser.py",
    "vllm.parser.deepseek_v4": "deepseek_v4.py",
    "vllm.parser.engine.adapters": "adapters.py",
    "vllm.parser.engine.parser_engine": "parser_engine.py",
    "vllm.parser.engine.parser_engine_config": "parser_engine_config.py",
    "vllm.parser.engine.streaming_parser_engine": "streaming_parser_engine.py",
}
EXEC_ORDER = [
    "vllm.parser.engine.events",
    "vllm.parser.engine.parser_engine_config",
    "vllm.parser.engine.incremental_lexer",
    "vllm.parser.engine.token_id_scanner",
    "vllm.parser.engine.streaming_parser_engine",
    "vllm.tool_parsers.utils",
    "vllm.parser.abstract_parser",
    "vllm.parser.engine.parser_engine",
    "vllm.parser.engine.adapters",
    "vllm.parser.deepseek_v4",
]


def _mk(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _ProtoBase:
    """Keyword-init stub for the vLLM pydantic protocol models."""

    _defaults: dict = {}

    def __init__(self, **kwargs):
        for key, value in self._defaults.items():
            setattr(self, key, list(value) if isinstance(value, list) else value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        pairs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({pairs})"


class DeltaFunctionCall(_ProtoBase):
    _defaults = {"name": None, "arguments": None}


class DeltaToolCall(_ProtoBase):
    _defaults = {"index": 0, "id": None, "type": None, "function": None}


class DeltaMessage(_ProtoBase):
    _defaults = {"role": None, "content": None, "reasoning": None, "tool_calls": []}


class FunctionCall(_ProtoBase):
    _defaults = {"id": None, "name": None, "arguments": None}


class ToolCall(_ProtoBase):
    _defaults = {"id": None, "type": "function", "function": None}


class ExtractedToolCallInformation(_ProtoBase):
    _defaults = {"tools_called": False, "tool_calls": [], "content": None}


class FunctionDefinition(_ProtoBase):
    _defaults = {"name": None, "parameters": None}


class ChatCompletionNamedToolChoiceParam(_ProtoBase):
    pass


class ChatCompletionRequest(_ProtoBase):
    pass


class ResponsesRequest(_ProtoBase):
    pass


class FunctionDef(_ProtoBase):
    _defaults = {"name": None, "parameters": None}


class ChatCompletionToolsParam(_ProtoBase):
    _defaults = {"type": "function", "function": None}

    def __init__(self, **kwargs):
        fn = kwargs.get("function")
        if isinstance(fn, dict):
            kwargs["function"] = FunctionDef(**fn)
        super().__init__(**kwargs)


class _StubReasoningParser:
    engine_based_streaming = False

    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer


class _StubToolParser:
    engine_based_streaming = False
    supports_required_and_named = True
    structural_tag_model = None

    def __init__(self, tokenizer, tools=None):
        self.model_tokenizer = tokenizer
        self.tools = tools

    def adjust_request(self, request):
        return request

    def get_remaining_unstreamed_args(self):
        return ""


_STACK_LIVE = False


def _build_stack(sources: dict[str, bytes]):
    """Exec the fixture modules under an isolated synthetic vllm package."""
    global _STACK_LIVE
    assert not _STACK_LIVE, "synthetic vllm stacks must not coexist"
    _STACK_LIVE = True

    prefixes = ("vllm", "openai", "partial_json_parser", "pydantic", "regex")
    saved = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name.startswith(prefixes)
    }
    created: list[str] = []

    def put(name: str, mod: types.ModuleType) -> types.ModuleType:
        sys.modules[name] = mod
        created.append(name)
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, mod)
        return mod

    try:
        importlib.import_module("regex")
    except ImportError:
        import re as _re

        put("regex", _re)
    try:
        importlib.import_module("pydantic")
    except ImportError:

        class ValidationError(Exception):
            pass

        class TypeAdapter:
            def __init__(self, *args, **kwargs):
                pass

            def validate_json(self, *args, **kwargs):
                raise ValidationError("stub")

        put(
            "pydantic",
            _mk("pydantic", TypeAdapter=TypeAdapter, ValidationError=ValidationError),
        )
    put("partial_json_parser", _mk("partial_json_parser"))
    put("partial_json_parser.core", _mk("partial_json_parser.core"))
    put(
        "partial_json_parser.core.options",
        _mk("partial_json_parser.core.options", Allow=object()),
    )
    # Sentinel classes: the fixture tools are ChatCompletionToolsParam stubs,
    # so every isinstance check against these resolves False, taking the
    # chat-completion tool route through vllm.tool_parsers.utils.
    put("openai", _mk("openai"))
    put("openai.types", _mk("openai.types"))
    put(
        "openai.types.responses",
        _mk(
            "openai.types.responses",
            FunctionTool=type("FunctionTool", (), {}),
            NamespaceTool=type("NamespaceTool", (), {}),
            ToolChoiceFunction=type("ToolChoiceFunction", (), {}),
        ),
    )
    put(
        "openai.types.responses.tool",
        _mk("openai.types.responses.tool", Tool=type("ResponsesTool", (), {})),
    )

    put("vllm", _mk("vllm"))
    put("vllm.logger", _mk("vllm.logger", init_logger=logging.getLogger))
    put("vllm.entrypoints", _mk("vllm.entrypoints"))
    put(
        "vllm.entrypoints.chat_utils",
        _mk(
            "vllm.entrypoints.chat_utils",
            get_tool_call_id_type=lambda model_config: "random",
            make_tool_call_id=lambda id_type="random", func_name=None, idx=None: (
                f"call_{func_name or 'fn'}_{idx or 0}"
            ),
        ),
    )
    put("vllm.entrypoints.openai", _mk("vllm.entrypoints.openai"))
    put("vllm.entrypoints.openai.engine", _mk("vllm.entrypoints.openai.engine"))
    put(
        "vllm.entrypoints.openai.engine.protocol",
        _mk(
            "vllm.entrypoints.openai.engine.protocol",
            DeltaFunctionCall=DeltaFunctionCall,
            DeltaMessage=DeltaMessage,
            DeltaToolCall=DeltaToolCall,
            ExtractedToolCallInformation=ExtractedToolCallInformation,
            FunctionCall=FunctionCall,
            ToolCall=ToolCall,
            FunctionDefinition=FunctionDefinition,
        ),
    )
    put(
        "vllm.entrypoints.openai.chat_completion",
        _mk("vllm.entrypoints.openai.chat_completion"),
    )
    put(
        "vllm.entrypoints.openai.chat_completion.protocol",
        _mk(
            "vllm.entrypoints.openai.chat_completion.protocol",
            ChatCompletionNamedToolChoiceParam=ChatCompletionNamedToolChoiceParam,
            ChatCompletionRequest=ChatCompletionRequest,
            ChatCompletionToolsParam=ChatCompletionToolsParam,
        ),
    )
    put("vllm.entrypoints.openai.responses", _mk("vllm.entrypoints.openai.responses"))
    put(
        "vllm.entrypoints.openai.responses.protocol",
        _mk("vllm.entrypoints.openai.responses.protocol", ResponsesRequest=ResponsesRequest),
    )
    put("vllm.parser", _mk("vllm.parser"))
    put(
        "vllm.parser.metrics",
        _mk("vllm.parser.metrics", record_tool_parser_invocation=lambda **kwargs: None),
    )
    put(
        "vllm.parser.utils",
        _mk("vllm.parser.utils", count_history_tool_calls=lambda request: 0),
    )
    put("vllm.parser.engine", _mk("vllm.parser.engine"))
    put("vllm.reasoning", _mk("vllm.reasoning"))
    put(
        "vllm.reasoning.abs_reasoning_parsers",
        _mk("vllm.reasoning.abs_reasoning_parsers", ReasoningParser=_StubReasoningParser),
    )
    put(
        "vllm.sampling_params",
        _mk(
            "vllm.sampling_params",
            StructuredOutputsParams=type(
                "StructuredOutputsParams", (), {"__init__": lambda self, **kw: None}
            ),
        ),
    )
    put("vllm.tokenizers", _mk("vllm.tokenizers", TokenizerLike=object))
    put("vllm.tool_parsers", _mk("vllm.tool_parsers"))
    put(
        "vllm.tool_parsers.abstract_tool_parser",
        _mk("vllm.tool_parsers.abstract_tool_parser", Tool=object, ToolParser=_StubToolParser),
    )

    def _no_named_required(*args, **kwargs):
        raise AssertionError("named/required streaming path must not run")

    put(
        "vllm.tool_parsers.streaming",
        _mk(
            "vllm.tool_parsers.streaming",
            extract_named_tool_call_streaming=_no_named_required,
            extract_required_tool_call_streaming=_no_named_required,
        ),
    )

    module_sources = {name: path.read_bytes() for name, path in SUPPORT.items()}
    for name, label in PATCHABLE_MOD.items():
        module_sources[name] = sources[label]
    for name in EXEC_ORDER:
        mod = put(name, _mk(name))
        mod.__file__ = name.replace(".", "/") + ".py"
        exec(compile(module_sources[name], mod.__file__, "exec"), mod.__dict__)

    return saved, created


def _teardown_stack(token) -> None:
    global _STACK_LIVE
    saved, created = token
    for name in created:
        sys.modules.pop(name, None)
    for name, mod in saved.items():
        if mod is not None:
            sys.modules[name] = mod
    _STACK_LIVE = False


# ── Test doubles and helpers (house port of upstream tests/parser/engine) ──

THINK_START_ID, THINK_END_ID = 50, 51


class MockTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self._id_to_text = {v: k for k, v in vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    @property
    def all_special_tokens(self):
        return list(self._vocab.keys())

    @property
    def all_special_ids(self):
        return list(self._vocab.values())

    def encode(self, text, **kwargs):
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            self._id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
        )


class MockRequest:
    def __init__(self, tools=None, tool_choice="auto"):
        self.tools = tools or []
        self.tool_choice = tool_choice
        self.include_reasoning = True
        self.skip_special_tokens = True


def _make_env(sources: dict[str, bytes]) -> types.SimpleNamespace:
    token = _build_stack(sources)
    ns = types.SimpleNamespace(token=token)
    d = sys.modules["vllm.parser.deepseek_v4"]
    ns.d = d
    ns.DeepSeekV4Parser = d.DeepSeekV4Parser
    reasoning_cls, tool_cls = sys.modules["vllm.parser.engine.adapters"].make_adapters(
        d.DeepSeekV4Parser
    )
    ns.Delegating = type(
        "_DeepSeekV4Delegating",
        (sys.modules["vllm.parser.abstract_parser"].DelegatingParser,),
        {"reasoning_parser_cls": reasoning_cls, "tool_parser_cls": tool_cls},
    )

    def mock_tokenizer(extra=None):
        vocab = {d.DSML_THINK_START: THINK_START_ID, d.DSML_THINK_END: THINK_END_ID}
        vocab.update(extra or {})
        return MockTokenizer(vocab)

    def make_tool(name, properties):
        return ChatCompletionToolsParam(
            type="function",
            function={
                "name": name,
                "parameters": {"type": "object", "properties": properties},
            },
        )

    def param(name, is_str, value):
        return (
            f'<｜DSML｜parameter name="{name}" string="{is_str}">'
            f"{value}</｜DSML｜parameter>"
        )

    def invoke(name, *params):
        body = "\n".join(param(n, s, v) for n, s, v in params)
        return (
            f"{d.DSML_INVOKE_PREFIX}{name}{d.DSML_INVOKE_NAME_END}\n{body}\n"
            f"{d.DSML_INVOKE_END}"
        )

    def content_parser(tokenizer, *tools):
        return ns.DeepSeekV4Parser(
            tokenizer, tools=list(tools), chat_template_kwargs={"thinking": False}
        )

    ns.mock_tokenizer = mock_tokenizer
    ns.make_tool = make_tool
    ns.param = param
    ns.invoke = invoke
    ns.tool_calls = lambda *iv: d.DSML_TOOL_START + "\n".join(iv) + d.DSML_TOOL_END
    ns.recovery_tool = lambda: make_tool("get_weather", {"city": {"type": "string"}})
    ns.recovery_invoke = lambda name="get_weather", city="Seoul": invoke(
        name, ("city", "true", city)
    )
    ns.content_parser = content_parser
    return ns


@contextlib.contextmanager
def stack_env(sources: dict[str, bytes]):
    env = _make_env(sources)
    try:
        yield env
    finally:
        _teardown_stack(env.token)


def _token_id_map(parser):
    cfg, vocab = parser.parser_engine_config, parser.vocab
    return {t: vocab[t] for t in (cfg.token_id_terminals or {}).values() if t in vocab}


def simulate_tool_streaming(parser, request, chunks):
    tmap = _token_id_map(parser)
    results, prev_text, prev_ids = [], "", []
    for chunk in chunks:
        cur_text = prev_text + chunk
        d_ids = [tid for text, tid in tmap.items() if text in chunk]
        cur_ids = prev_ids + d_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=prev_text,
            current_text=cur_text,
            delta_text=chunk,
            previous_token_ids=tuple(prev_ids),
            current_token_ids=tuple(cur_ids),
            delta_token_ids=tuple(d_ids),
            request=request,
        )
        results.append((delta, cur_text))
        prev_text, prev_ids = cur_text, list(cur_ids)
    return results


def collect_tool_arguments(results):
    out = ""
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.arguments:
                    out += tc.function.arguments
    return out


def collect_content(results):
    return "".join(d.content for d, _ in results if d and d.content)


def collect_function_name(results):
    for delta, _ in results:
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    return tc.function.name
    return None


def collect_output(deltas):
    reasoning, content, calls = "", "", {}
    for delta in deltas:
        if not delta:
            continue
        if delta.reasoning:
            reasoning += delta.reasoning
        if delta.content:
            content += delta.content
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"name": None, "arguments": ""})
            if tc.function:
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
    return types.SimpleNamespace(
        reasoning=reasoning,
        content=content,
        tool_calls=[calls[i] for i in sorted(calls)],
    )


# ── Fixture and transform identity ────────────────────────────────────


class FixtureIdentity(unittest.TestCase):
    def test_fixtures_match_stock_pins(self):
        for spec in DR.SPECS:
            data = FIXTURE[spec.label].read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), spec.stock_sha256, spec.label)
            self.assertEqual(len(data), spec.stock_size, spec.label)

    def test_support_fixtures_match_pins(self):
        for name, path in SUPPORT.items():
            data = path.read_bytes()
            sha, size = SUPPORT_SHA256[name]
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha, name)
            self.assertEqual(len(data), size, name)


class Transform(unittest.TestCase):
    def test_transform_is_pinned_and_compiles(self):
        for spec in DR.SPECS:
            stock = FIXTURE[spec.label].read_bytes()
            patched = DR.transform(spec, stock)
            self.assertEqual(
                hashlib.sha256(patched).hexdigest(), spec.patched_sha256, spec.label
            )
            self.assertEqual(len(patched), spec.patched_size, spec.label)
            compile(patched, spec.label, "exec")
            self.assertGreaterEqual(patched.count(DR.MARK.encode()), 1, spec.label)
            # sequential single-site replaces: nothing else changed
            rebuilt = stock
            for old, new in spec.regions:
                self.assertEqual(rebuilt.count(old), 1, spec.label)
                rebuilt = rebuilt.replace(old, new, 1)
            self.assertEqual(patched, rebuilt, spec.label)

    def test_transform_refuses_foreign_or_patched_bytes(self):
        for spec in DR.SPECS:
            with self.assertRaises(DR.HotfixError):
                DR.transform(spec, b"def nothing():\n    pass\n")
            patched = DR.transform(spec, FIXTURE[spec.label].read_bytes())
            with self.assertRaises(DR.HotfixError):
                DR.transform(spec, patched)


# ── Patcher filesystem behavior ───────────────────────────────────────


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dspark-dsml-recovery-"))
        self.root = self.tmp / "vllm"
        for spec in DR.SPECS:
            target = self.root / spec.rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FIXTURE[spec.label], target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _states(self):
        return {
            spec.label: DR.inspect(spec, self.root / spec.rel_path, provider=GOOD)[0]
            for spec in DR.SPECS
        }

    def test_apply_then_idempotent(self):
        outcomes = DR.apply(self.root, provider=GOOD)
        self.assertEqual(set(outcomes.values()), {"applied"})
        snapshot = {
            spec.label: (self.root / spec.rel_path).read_bytes() for spec in DR.SPECS
        }
        for spec in DR.SPECS:
            self.assertEqual(
                hashlib.sha256(snapshot[spec.label]).hexdigest(), spec.patched_sha256
            )
        self.assertEqual(set(self._states().values()), {"patched"})
        again = DR.apply(self.root, provider=GOOD)
        self.assertEqual(set(again.values()), {"already-patched"})
        for spec in DR.SPECS:
            self.assertEqual(
                (self.root / spec.rel_path).read_bytes(), snapshot[spec.label]
            )
        # no temp files left anywhere in the tree
        leftovers = [
            p for p in self.root.rglob("*") if p.name.startswith(".dspark-dsml-recovery-")
        ]
        self.assertEqual(leftovers, [])

    def test_one_foreign_file_blocks_every_write(self):
        victim = DR.SPECS[-1]
        target = self.root / victim.rel_path
        target.write_bytes(b"x = 1\n")
        before = {
            spec.label: (self.root / spec.rel_path).read_bytes() for spec in DR.SPECS
        }
        with self.assertRaises(DR.HotfixError):
            DR.apply(self.root, provider=GOOD)
        for spec in DR.SPECS:
            self.assertEqual(
                (self.root / spec.rel_path).read_bytes(), before[spec.label], spec.label
            )

    def test_late_failure_rolls_back_written_files(self):
        real_publish = DR._publish
        calls = {"n": 0}

        def failing_publish(target, data):
            calls["n"] += 1
            if calls["n"] == 4:
                raise OSError("disk full")
            real_publish(target, data)

        DR._publish = failing_publish
        try:
            with self.assertRaises(OSError):
                DR.apply(self.root, provider=GOOD)
        finally:
            DR._publish = real_publish
        # rollback republishes the first three files, so > 4 total calls
        self.assertGreater(calls["n"], 4)
        self.assertEqual(set(self._states().values()), {"stock"})

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(DR.HotfixError):
            DR.apply(self.root, provider=lambda name: "0.26.0")
        self.assertEqual(set(self._states().values()), {"stock"})

    def test_refuses_symlink(self):
        spec = DR.SPECS[0]
        target = self.root / spec.rel_path
        link = target.with_name("link.py")
        os.symlink(target, link)
        with self.assertRaises(DR.HotfixError):
            DR.inspect(spec, link, provider=GOOD)

    def test_refuses_missing_target(self):
        spec = DR.SPECS[0]
        (self.root / spec.rel_path).unlink()
        with self.assertRaises(DR.HotfixError):
            DR.apply(self.root, provider=GOOD)

    def test_cli_check_and_status_do_not_write(self):
        before = {
            spec.label: (self.root / spec.rel_path).read_bytes() for spec in DR.SPECS
        }
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [sys.executable, str(PATCHER), flag, "--root", str(self.root)],
                capture_output=True,
                text=True,
            )
            # the real vllm is not installed here -> fail closed, no write
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("FAIL-CLOSED", proc.stderr)
            for spec in DR.SPECS:
                self.assertEqual(
                    (self.root / spec.rel_path).read_bytes(), before[spec.label]
                )


# ── Upstream vllm#52645 regression matrix against the pinned fixtures ──


class Recovery(unittest.TestCase):
    """Non-streaming, streaming, and engine-level recovery semantics."""

    @classmethod
    def setUpClass(cls):
        cls.env = _make_env(_patched_sources())
        cls.d = cls.env.d

    @classmethod
    def tearDownClass(cls):
        _teardown_stack(cls.env.token)

    def test_foreign_wrapper_does_not_recover_inner_invoke(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = (
            d.DSML_FOREIGN_TOOL_START + env.recovery_invoke() + d.DSML_FOREIGN_TOOL_END
        )
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_missing_start_wrapper_recovers_declared_tool(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.DeepSeekV4Parser(env.mock_tokenizer(), tools=[tool])
        result = parser.extract_tool_calls(
            env.recovery_invoke() + d.DSML_TOOL_END, MockRequest(tools=[tool])
        )
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments), {"city": "Seoul"}
        )

    def test_corrupted_start_wrapper_still_recovers_invoke(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.DeepSeekV4Parser(env.mock_tokenizer(), tools=[tool])
        text = "<｜DSML｜toolcalls>\n" + env.recovery_invoke() + "\n" + d.DSML_TOOL_END
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            json.loads(result.tool_calls[0].function.arguments), {"city": "Seoul"}
        )

    def test_undeclared_orphan_invoke_stays_content(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = env.recovery_invoke(name="not_declared") + d.DSML_TOOL_END
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_orphan_invoke_without_tools_stays_content(self):
        env, d = self.env, self.d
        parser = env.content_parser(env.mock_tokenizer())
        text = env.recovery_invoke() + d.DSML_TOOL_END
        result = parser.extract_tool_calls(text, MockRequest())
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_request_without_tools_does_not_reuse_prior_tool_names(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = env.recovery_invoke() + d.DSML_TOOL_END
        first = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        second = parser.extract_tool_calls(text, MockRequest(tools=[]))
        self.assertTrue(first.tools_called)
        self.assertFalse(second.tools_called)
        self.assertEqual(second.tool_calls, [])
        self.assertEqual(second.content, text)

    def test_tool_choice_none_disables_recovery(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = env.recovery_invoke() + d.DSML_TOOL_END
        result = parser.extract_tool_calls(
            text, MockRequest(tools=[tool], tool_choice="none")
        )
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_truncated_recovery_candidate_flushes_as_content(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = "Docs quote " + d.DSML_INVOKE_PREFIX + "get_wea"
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_valid_name_without_invoke_end_stays_content(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = (
            f"{d.DSML_INVOKE_PREFIX}get_weather{d.DSML_INVOKE_NAME_END}\n"
            f"{env.param('city', 'true', 'Seoul')}"
        )
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_truncated_recovery_drops_eos_special_token(self):
        env, d = self.env, self.d
        eos_text, eos_id = "<｜end▁of▁sentence｜>", 128801
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer({eos_text: eos_id}), tool)
        parser._check_skip_tool_parsing(MockRequest(tools=[tool]))
        text = (
            f"{d.DSML_INVOKE_PREFIX}get_weather{d.DSML_INVOKE_NAME_END}\n"
            f"{env.param('city', 'true', 'Seoul')}"
        )
        self.assertEqual(parser._engine.feed(text, []), [])
        events = parser._engine.feed(eos_text, [eos_id])
        events.extend(parser._engine.finish())
        self.assertEqual("".join(e.value for e in events if e.value), text)

    def test_tool_end_without_invoke_end_stays_content(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        text = (
            f"{d.DSML_INVOKE_PREFIX}get_weather{d.DSML_INVOKE_NAME_END}\n"
            f"{env.param('city', 'true', 'Seoul')}\n{d.DSML_TOOL_END}"
        )
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.content, text)

    def test_recovered_invoke_preserves_trailing_content_without_tool_end(self):
        env = self.env
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        result = parser.extract_tool_calls(
            env.recovery_invoke() + " Done.", MockRequest(tools=[tool])
        )
        self.assertTrue(result.tools_called)
        self.assertEqual(
            [c.function.name for c in result.tool_calls], ["get_weather"]
        )
        self.assertEqual(result.content, " Done.")

    def test_recovered_parallel_invokes_validate_each_declared_tool(self):
        env, d = self.env, self.d
        weather = env.recovery_tool()
        forecast = env.make_tool("get_forecast", {"city": {"type": "string"}})
        parser = env.content_parser(env.mock_tokenizer(), weather, forecast)
        text = (
            env.recovery_invoke()
            + env.recovery_invoke(name="get_forecast")
            + d.DSML_TOOL_END
        )
        result = parser.extract_tool_calls(
            text, MockRequest(tools=[weather, forecast])
        )
        self.assertTrue(result.tools_called)
        self.assertEqual(
            [c.function.name for c in result.tool_calls],
            ["get_weather", "get_forecast"],
        )

    def test_recovered_parallel_invoke_rejects_undeclared_second_tool(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        rejected = env.recovery_invoke(name="not_declared")
        text = env.recovery_invoke() + rejected + d.DSML_TOOL_END
        result = parser.extract_tool_calls(text, MockRequest(tools=[tool]))
        self.assertTrue(result.tools_called)
        self.assertEqual([c.function.name for c in result.tool_calls], ["get_weather"])
        self.assertEqual(result.content, rejected)

    def test_streaming_orphan_invoke_recovers_after_split_marker(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        chunks = [
            "Checking.\n",
            "<｜DSML",
            '｜invoke name="get_weather">',
            f"\n{env.param('city', 'true', 'Seoul')}\n",
            d.DSML_INVOKE_END,
            d.DSML_TOOL_END,
        ]
        results = simulate_tool_streaming(parser, MockRequest(tools=[tool]), chunks)
        self.assertEqual(collect_function_name(results), "get_weather")
        self.assertEqual(
            json.loads(collect_tool_arguments(results)), {"city": "Seoul"}
        )
        self.assertEqual(collect_content(results), "Checking.\n")

    def test_streaming_tool_end_without_invoke_end_stays_content(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        parser = env.content_parser(env.mock_tokenizer(), tool)
        chunks = [
            f"{d.DSML_INVOKE_PREFIX}get_weather{d.DSML_INVOKE_NAME_END}\n",
            f"{env.param('city', 'true', 'Seoul')}\n",
            d.DSML_TOOL_END,
        ]
        results = simulate_tool_streaming(parser, MockRequest(tools=[tool]), chunks)
        self.assertIsNone(collect_function_name(results))
        self.assertEqual(collect_tool_arguments(results), "")
        self.assertEqual(collect_content(results), "".join(chunks))


class DelegatingRecovery(unittest.TestCase):
    """Serving-style DelegatingParser flow (reasoning adapter + tool adapter)."""

    @classmethod
    def setUpClass(cls):
        cls.env = _make_env(_patched_sources())
        cls.d = cls.env.d

    @classmethod
    def tearDownClass(cls):
        _teardown_stack(cls.env.token)

    def _parse(self, request, text):
        env = self.env
        parser = env.Delegating(
            env.mock_tokenizer(),
            tools=list(request.tools),
            chat_template_kwargs={"thinking": True},
        )
        delta = parser.parse_delta(text, [], request, prompt_token_ids=[], finished=True)
        return collect_output([delta])

    def test_foreign_wrapper_in_reasoning_cannot_execute_inner_invoke(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        text = "Still thinking.\n" + (
            d.DSML_FOREIGN_TOOL_START + env.recovery_invoke() + d.DSML_FOREIGN_TOOL_END
        )
        out = self._parse(MockRequest(tools=[tool]), text)
        self.assertEqual(out.reasoning, text)
        self.assertEqual(out.content, "")
        self.assertEqual(out.tool_calls, [])

    def test_unclosed_foreign_wrapper_finishes_as_reasoning(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        text = "Still thinking.\n" + d.DSML_FOREIGN_TOOL_START + "quoted output"
        out = self._parse(MockRequest(tools=[tool]), text)
        self.assertEqual(out.reasoning, text)
        self.assertEqual(out.content, "")
        self.assertEqual(out.tool_calls, [])

    def test_native_wrapper_escapes_unclosed_foreign_reasoning_block(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        text = d.DSML_FOREIGN_TOOL_START + env.tool_calls(env.recovery_invoke())
        out = self._parse(MockRequest(tools=[tool]), text)
        self.assertEqual(out.reasoning, d.DSML_FOREIGN_TOOL_START)
        self.assertEqual(out.content, "")
        self.assertEqual([c["name"] for c in out.tool_calls], ["get_weather"])

    def test_complete_declared_invoke_commits_before_think_end(self):
        env = self.env
        tool = env.recovery_tool()
        out = self._parse(
            MockRequest(tools=[tool]), "Still thinking.\n" + env.recovery_invoke()
        )
        self.assertEqual(out.reasoning, "Still thinking.")
        self.assertEqual(out.content, "")
        self.assertEqual([c["name"] for c in out.tool_calls], ["get_weather"])

    def test_rejected_invoke_rolls_back_to_reasoning(self):
        env, d = self.env, self.d
        tool = env.recovery_tool()
        cases = {
            "undeclared": (env.recovery_invoke(name="not_declared"), "auto"),
            "truncated": (
                f"{d.DSML_INVOKE_PREFIX}get_weather{d.DSML_INVOKE_NAME_END}\n"
                f"{env.param('city', 'true', 'Seoul')}",
                "auto",
            ),
            "tool_choice_none": (env.recovery_invoke(), "none"),
        }
        for case, (candidate, tool_choice) in cases.items():
            with self.subTest(case=case):
                text = "Still thinking.\n" + candidate
                out = self._parse(
                    MockRequest(tools=[tool], tool_choice=tool_choice), text
                )
                self.assertEqual(out.reasoning, text)
                self.assertEqual(out.content, "")
                self.assertEqual(out.tool_calls, [])


# ── Stock parity on the normal DSML and tool-call paths ───────────────


def _snap_extract(env, text, tools=(), tool_choice="auto"):
    parser = env.DeepSeekV4Parser(
        env.mock_tokenizer(), tools=list(tools), chat_template_kwargs={"thinking": False}
    )
    r = parser.extract_tool_calls(text, MockRequest(tools=list(tools), tool_choice=tool_choice))
    return {
        "tools_called": r.tools_called,
        "content": r.content,
        "calls": [(c.function.name, c.function.arguments) for c in r.tool_calls],
    }


def _snap_stream(env, chunks, tools=()):
    parser = env.content_parser(env.mock_tokenizer(), *tools)
    results = simulate_tool_streaming(parser, MockRequest(tools=list(tools)), chunks)
    fin = parser.finish_streaming()
    if fin is not None:
        results.append((fin, ""))
    return {
        "name": collect_function_name(results),
        "args": collect_tool_arguments(results),
        "content": collect_content(results),
    }


def _snap_delegating(env, text, tools=(), tool_choice="auto"):
    parser = env.Delegating(
        env.mock_tokenizer(), tools=list(tools), chat_template_kwargs={"thinking": True}
    )
    delta = parser.parse_delta(
        text, [], MockRequest(tools=list(tools), tool_choice=tool_choice),
        prompt_token_ids=[], finished=True,
    )
    out = collect_output([delta])
    return {"reasoning": out.reasoning, "content": out.content, "calls": out.tool_calls}


def _snap_parse(env, text, tools=()):
    parser = env.DeepSeekV4Parser(
        env.mock_tokenizer(), tools=list(tools), chat_template_kwargs={"thinking": True}
    )
    reasoning, content, calls = parser.parse(text, MockRequest(tools=list(tools)))
    return {
        "reasoning": reasoning,
        "content": content,
        "calls": [(c.name, c.arguments) for c in (calls or [])],
    }


def _normal_matrix(env):
    d = env.d
    weather = env.make_tool("get_weather", {"location": {"type": "string"}})
    ttime = env.make_tool("get_time", {"timezone": {"type": "string"}})
    wrapped = env.tool_calls(env.invoke("get_weather", ("location", "true", "NYC")))
    wrapped2 = env.tool_calls(
        env.invoke("get_weather", ("location", "true", "NYC")),
        env.invoke("get_time", ("timezone", "true", "EST")),
    )
    unwrap = env.tool_calls(
        env.invoke("get_weather", ("arguments", "false", '{"location": "SF"}'))
    )
    return {
        "plain_content": _snap_extract(env, "Just a normal answer.", (weather,)),
        "wrapped_single": _snap_extract(env, wrapped, (weather,)),
        "wrapped_parallel": _snap_extract(env, wrapped2, (weather, ttime)),
        "wrapped_undeclared_tool_still_parses": _snap_extract(
            env, env.tool_calls(env.invoke("mystery", ("x", "true", "1"))), (weather,)
        ),
        "wrapped_no_request_tools": _snap_extract(env, wrapped, ()),
        "wrapped_tool_choice_none": _snap_extract(
            env, wrapped, (weather,), tool_choice="none"
        ),
        "wrapper_args_unwrap": _snap_extract(env, unwrap, (weather,)),
        "trailing_content": _snap_extract(env, wrapped + "\nDone.", (weather,)),
        "reasoning_then_tool_parse": _snap_parse(
            env, f"Thinking hard.\n{d.DSML_THINK_END}\n{wrapped}", (weather,)
        ),
        "reasoning_only_parse": _snap_parse(env, "Only thoughts here", (weather,)),
        "bare_think_end_absorbed": _snap_extract(
            env, f"{d.DSML_THINK_END}Answer.", (weather,)
        ),
        "stream_wrapped": _snap_stream(
            env,
            [
                d.DSML_TOOL_START,
                '<｜DSML｜invoke name="get_weather">',
                "\n",
                env.param("location", "true", "NYC"),
                "\n",
                d.DSML_INVOKE_END,
                d.DSML_TOOL_END,
                " tail",
            ],
            (weather,),
        ),
        "stream_plain": _snap_stream(env, ["Hello ", "world."], (weather,)),
        "delegating_reasoning_tool": _snap_delegating(
            env, f"Let me check.\n{d.DSML_THINK_END}\n{wrapped}", (weather,)
        ),
        "delegating_reasoning_only": _snap_delegating(
            env, "Deep thoughts only", (weather,)
        ),
        "delegating_none_choice": _snap_delegating(
            env, f"Hmm.\n{d.DSML_THINK_END}\n{wrapped}", (weather,), tool_choice="none"
        ),
    }


class StockParity(unittest.TestCase):
    def test_normal_paths_byte_identical_to_stock(self):
        with stack_env(_stock_sources()) as env:
            stock_matrix = _normal_matrix(env)
        with stack_env(_patched_sources()) as env:
            patched_matrix = _normal_matrix(env)
        self.assertEqual(set(stock_matrix), set(patched_matrix))
        for case in stock_matrix:
            self.assertEqual(stock_matrix[case], patched_matrix[case], case)

    def test_stock_leaks_and_patched_recovers(self):
        """The bug this port fixes: stock returns the invoke as content."""

        def leak_cases(env):
            tool = env.recovery_tool()
            d = env.d
            corrupted = (
                "<｜DSML｜toolcalls>\n" + env.recovery_invoke() + "\n" + d.DSML_TOOL_END
            )
            bare = env.recovery_invoke() + d.DSML_TOOL_END
            return {
                "corrupted": _snap_extract(env, corrupted, (tool,)),
                "bare": _snap_extract(env, bare, (tool,)),
            }, corrupted, bare

        with stack_env(_stock_sources()) as env:
            stock_out, corrupted, bare = leak_cases(env)
        with stack_env(_patched_sources()) as env:
            patched_out, _, _ = leak_cases(env)

        self.assertFalse(stock_out["corrupted"]["tools_called"])
        self.assertEqual(stock_out["corrupted"]["content"], corrupted)
        self.assertFalse(stock_out["bare"]["tools_called"])
        self.assertEqual(stock_out["bare"]["content"], bare)

        for case in ("corrupted", "bare"):
            self.assertTrue(patched_out[case]["tools_called"], case)
            name, args = patched_out[case]["calls"][0]
            self.assertEqual(name, "get_weather")
            self.assertEqual(json.loads(args), {"city": "Seoul"})


# ── Recipe wiring ─────────────────────────────────────────────────────


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_DSML_RECOVERY: "${DSPARK_ENABLE_DSML_RECOVERY:-0}"', compose
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_DSML_RECOVERY:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-dsml-recovery.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_DSML_RECOVERY_HOTFIX:-./patches/hotfix-vllm-dsml-recovery.py}"
            ":/opt/hotfix-vllm-dsml-recovery.py:ro",
            compose,
        )

    def test_launcher_passthrough_sync_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_DSML_RECOVERY_HOTFIX='./patches/hotfix-vllm-dsml-recovery.py'", start
        )
        self.assertIn("DSPARK_ENABLE_DSML_RECOVERY=$REMOTE_DSML_RECOVERY", start)
        self.assertIn("/opt/hotfix-vllm-dsml-recovery.py --check", start)
        self.assertIn('patches/hotfix-vllm-dsml-recovery.py"', start)
        self.assertIn(
            "export DSPARK_DSML_RECOVERY_HOTFIX DSPARK_ENABLE_DSML_RECOVERY", start
        )

    def test_env_example_docs_and_ci(self):
        self.assertIn("DSPARK_ENABLE_DSML_RECOVERY=0", ENV_EXAMPLE.read_text())
        ci = CI.read_text()
        self.assertIn("scripts/test-dsml-recovery.py", ci)
        self.assertIn("hotfix-vllm-dsml-recovery.py", ci)
        self.assertIn("DSPARK_ENABLE_DSML_RECOVERY", ENVS_DOC.read_text())
        self.assertIn("hotfix-vllm-dsml-recovery.py", PATCHES_DOC.read_text())


if __name__ == "__main__":
    unittest.main()
