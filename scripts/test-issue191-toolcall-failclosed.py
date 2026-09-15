#!/usr/bin/env python3
"""Hermetic CPU acceptance for the issue #191 tool-call fail-closed hotfix.

Covers the pinned fixture identity, the pure transformation, the atomic
patcher (apply / idempotence / refusals), the contract-check helper semantics
(exercised from the exact bytes the patcher installs), and the Compose /
launcher / env wiring.  No Docker, vLLM, torch, GPU, or network access.
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "fixtures" / "issue191" / "chat_completion_serving-752a3a504-post-issue55.py"
PRISTINE = ROOT / "scripts" / "fixtures" / "issue191" / "chat_completion_serving-752a3a504-pristine.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-issue191-toolcall-failclosed.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("hotfix_issue191", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HF = _load_patcher()
GOOD_VERSION = lambda name: HF.EXPECTED_VLLM_VERSION  # noqa: E731


def _helpers():
    """Execute the helper block exactly as installed and return its namespace."""
    block = HF.HELPER_NEW.decode("utf-8")
    start = block.index("# [issue191-hotfix]")
    end = block.rindex("import io\n")
    namespace: dict = {}
    exec(compile(block[start:end], "issue191-helpers", "exec"), namespace)
    return namespace


def _tool(name="record_event", strict=True, parameters=None):
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {"event_id": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["event_id", "count"],
            "additionalProperties": False,
        }
    return NS(type="function", function=NS(name=name, parameters=parameters, strict=strict))


def _request(tool_choice, parallel=False, tools=None):
    if tools is None:
        tools = [_tool()]
    return NS(tool_choice=tool_choice, parallel_tool_calls=parallel, tools=tools)


def _named(name="record_event"):
    return NS(type="function", function=NS(name=name))


def _response(calls, finish_reason="stop"):
    tool_calls = [NS(function=NS(name=n, arguments=a)) for n, a in calls]
    return NS(choices=[NS(finish_reason=finish_reason, message=NS(tool_calls=tool_calls or None))])


class FixtureAndTransform(unittest.TestCase):
    def test_fixture_identity_matches_patcher_pins(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), HF.STOCK_SHA256)
        self.assertEqual(len(data), HF.STOCK_SIZE)
        self.assertIn(HF.ISSUE55_MARK.encode(), data)
        self.assertNotIn(HF.MARK.encode(), data)

    def test_transform_is_pinned_and_compiles(self):
        patched = HF.transform(FIXTURE.read_bytes())
        self.assertEqual(hashlib.sha256(patched).hexdigest(), HF.PATCHED_SHA256)
        self.assertEqual(len(patched), HF.PATCHED_SIZE)
        self.assertEqual(patched.count(HF.MARK.encode()), 2)
        self.assertNotIn(HF.TAIL_OLD, patched)
        self.assertIn(b"DSPARK_ISSUE191_TOOLCALL_RETRIES", patched)
        compile(patched.decode("utf-8"), "serving.py", "exec")

    def test_transform_refuses_already_patched(self):
        patched = HF.transform(FIXTURE.read_bytes())
        with self.assertRaises(HF.HotfixError):
            HF.transform(patched)


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="issue191-"))
        self.target = self.tmp / "serving.py"
        shutil.copy(FIXTURE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(HF.inspect(self.target, provider=GOOD_VERSION)[0], "stock-compatible")
        self.assertEqual(HF.apply(self.target, provider=GOOD_VERSION), "applied")
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), HF.PATCHED_SHA256)
        self.assertEqual(HF.apply(self.target, provider=GOOD_VERSION), "already-patched")
        self.assertEqual(HF.inspect(self.target, provider=GOOD_VERSION)[0], "patched")
        self.assertFalse(list(self.tmp.glob(".issue191-*")), "temp file leaked")

    def test_pristine_image_bytes_pass_check_but_refuse_apply(self):
        data = PRISTINE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), HF.PRISTINE_SHA256)
        self.assertEqual(len(data), HF.PRISTINE_SIZE)
        self.assertNotIn(HF.ISSUE55_MARK.encode(), data)
        self.target.write_bytes(data)
        self.assertEqual(HF.inspect(self.target, provider=GOOD_VERSION)[0], "stock-pristine")
        with self.assertRaises(HF.HotfixError):
            HF.apply(self.target, provider=GOOD_VERSION)
        self.assertEqual(self.target.read_bytes(), data)

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(self.target.read_bytes() + b"\n# drift\n")
        with self.assertRaises(HF.HotfixError):
            HF.inspect(self.target, provider=GOOD_VERSION)

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(HF.HotfixError):
            HF.inspect(self.target, provider=lambda name: "0.25.2.dev0+gdeadbeef")

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        link.symlink_to(self.target)
        with self.assertRaises(HF.HotfixError):
            HF.inspect(link, provider=GOOD_VERSION)

    def test_cli_check_and_status_do_not_write(self):
        before = self.target.read_bytes()
        self.assertEqual(HF.main(["--check", "--target", str(self.target)]) in (0, 1), True)
        self.assertEqual(self.target.read_bytes(), before)


class ContractSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _helpers()

    def violation(self, request, response):
        return self.ns["_issue191_tool_contract_violation"](request, response)

    def test_non_forcing_tool_choice_is_ignored(self):
        for choice in (None, "auto", "none"):
            self.assertIsNone(self.violation(_request(choice), _response([])))

    def test_named_ok(self):
        resp = _response([("record_event", '{"event_id":"e1","count":7}')])
        self.assertIsNone(self.violation(_request(_named()), resp))

    def test_required_ok(self):
        resp = _response([("record_event", '{"event_id":"e1","count":7}')])
        self.assertIsNone(self.violation(_request("required"), resp))

    def test_zero_calls_is_cardinality_violation(self):
        self.assertEqual(self.violation(_request(_named()), _response([])), "tool-call-cardinality:0")
        self.assertEqual(self.violation(_request("required"), _response([])), "tool-call-cardinality:0")

    def test_two_calls_with_parallel_false(self):
        resp = _response([("record_event", '{"event_id":"a","count":1}')] * 2)
        self.assertEqual(self.violation(_request("required", parallel=False), resp), "tool-call-cardinality:2")
        self.assertIsNone(self.violation(_request("required", parallel=True), resp))

    def test_wrong_name(self):
        resp = _response([("other", '{"event_id":"a","count":1}')])
        self.assertEqual(self.violation(_request(_named()), resp), "tool-call-name")
        self.assertEqual(self.violation(_request("required"), resp), "tool-call-name")

    def test_bad_json_arguments(self):
        self.assertEqual(
            self.violation(_request(_named()), _response([("record_event", '{"event_id":')])),
            "tool-arguments-json",
        )
        self.assertEqual(
            self.violation(_request(_named()), _response([("record_event", "[1,2]")])),
            "tool-arguments-json",
        )
        self.assertEqual(
            self.violation(_request(_named()), _response([("record_event", None)])),
            "tool-arguments-type",
        )

    def test_strict_schema_violation_and_non_strict_pass(self):
        missing = _response([("record_event", '{"event_id":"a"}')])
        wrong_type = _response([("record_event", '{"event_id":"a","count":"7"}')])
        self.assertTrue(self.violation(_request(_named()), missing).startswith("tool-arguments-schema:"))
        self.assertTrue(self.violation(_request(_named()), wrong_type).startswith("tool-arguments-schema:"))
        loose = _request(_named(), tools=[_tool(strict=False)])
        self.assertIsNone(self.violation(loose, missing))

    def test_length_with_complete_call_is_not_a_violation(self):
        resp = _response([("record_event", '{"event_id":"a","count":1}')], finish_reason="length")
        self.assertIsNone(self.violation(_request(_named()), resp))

    def test_length_without_any_call_is_truncated(self):
        self.assertEqual(self.violation(_request(_named()), _response([], finish_reason="length")), "tool-call-truncated")
        self.assertEqual(self.violation(_request("required"), _response([], finish_reason="length")), "tool-call-truncated")

    def test_no_choices(self):
        self.assertEqual(self.violation(_request(_named()), NS(choices=[])), "no-choices")

    def test_fallback_schema_checker(self):
        fallback = self.ns["_issue191_fallback_schema_error"]
        schema = _tool().function.parameters
        self.assertIsNone(fallback({"event_id": "a", "count": 1}, schema))
        self.assertEqual(fallback({"count": 1}, schema), "schema:event_id:required")
        self.assertEqual(fallback({"event_id": "a", "count": True}, schema), "schema:count:type")
        self.assertEqual(fallback({"event_id": "a", "count": 1, "x": 1}, schema), "schema:x:additionalProperties")
        self.assertEqual(fallback([], schema), "schema:<root>:type")

    def test_schema_error_uses_jsonschema_when_available(self):
        calls = []

        class _Validator:
            def __init__(self, schema):
                self.schema = schema

            @staticmethod
            def check_schema(schema):
                calls.append("check")

            def iter_errors(self, value):
                if "count" not in value:
                    yield NS(absolute_path=["count"], validator="required")

        fake = types.ModuleType("jsonschema")
        fake.validators = NS(validator_for=lambda schema: _Validator)
        saved = sys.modules.get("jsonschema")
        sys.modules["jsonschema"] = fake
        try:
            err = self.ns["_issue191_schema_error"]
            self.assertIsNone(err({"event_id": "a", "count": 1}, _tool().function.parameters))
            self.assertEqual(err({"event_id": "a"}, _tool().function.parameters), "schema:count:required")
        finally:
            if saved is None:
                del sys.modules["jsonschema"]
            else:
                sys.modules["jsonschema"] = saved
        self.assertTrue(calls)
    def test_schema_validator_error_is_not_a_violation(self):
        # D1: a malformed schema or jsonschema validator/meta failure must
        # never fail the request — the exception arm returns None, not a
        # "schema-validator-error" contract violation.
        class _ExplodingValidator:
            def __init__(self, schema):
                raise RuntimeError("meta-validation exploded")

        fake = types.ModuleType("jsonschema")
        fake.validators = NS(validator_for=lambda schema: _ExplodingValidator)
        saved = sys.modules.get("jsonschema")
        sys.modules["jsonschema"] = fake
        try:
            err = self.ns["_issue191_schema_error"]
            self.assertIsNone(err({"event_id": "a"}, _tool().function.parameters))
            self.assertIsNone(err({"event_id": "a"}, {"type": "definitely not a schema"}))
        finally:
            if saved is None:
                del sys.modules["jsonschema"]
            else:
                sys.modules["jsonschema"] = saved

    def test_env_knobs(self):
        import os

        env_int = self.ns["_issue191_env_int"]
        mode = self.ns["_issue191_mode"]
        saved = {
            k: os.environ.get(k)
            for k in (
                "DSPARK_ISSUE191_TOOLCALL_RETRIES",
                "DSPARK_ISSUE191_TOOLCALL_MODE",
                "DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK",
            )
        }
        try:
            os.environ.pop("DSPARK_ISSUE191_TOOLCALL_RETRIES", None)
            self.assertEqual(env_int("DSPARK_ISSUE191_TOOLCALL_RETRIES", 2), 2)
            os.environ["DSPARK_ISSUE191_TOOLCALL_RETRIES"] = "9"
            self.assertEqual(env_int("DSPARK_ISSUE191_TOOLCALL_RETRIES", 2), 5)
            os.environ["DSPARK_ISSUE191_TOOLCALL_RETRIES"] = "-1"
            self.assertEqual(env_int("DSPARK_ISSUE191_TOOLCALL_RETRIES", 2), 0)
            os.environ["DSPARK_ISSUE191_TOOLCALL_RETRIES"] = "x"
            self.assertEqual(env_int("DSPARK_ISSUE191_TOOLCALL_RETRIES", 2), 2)
            os.environ.pop("DSPARK_ISSUE191_TOOLCALL_MODE", None)
            self.assertEqual(mode(), "failclosed")
            os.environ["DSPARK_ISSUE191_TOOLCALL_MODE"] = " LOG "
            self.assertEqual(mode(), "log")
            os.environ["DSPARK_ISSUE191_TOOLCALL_MODE"] = "bogus"
            self.assertEqual(mode(), "failclosed")
            fallback = self.ns["_issue191_thinkoff_fallback"]
            os.environ.pop("DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK", None)
            self.assertTrue(fallback())
            os.environ["DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK"] = "0"
            self.assertFalse(fallback())
            os.environ["DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK"] = " 1 "
            self.assertTrue(fallback())
            os.environ["DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK"] = "yes"
            self.assertFalse(fallback())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_thinkoff_marker_ids_come_from_the_request_parser(self):
        markers = self.ns["_issue191_reasoning_marker_ids"]

        class Engine:
            _reasoning_start_token_id = 128821
            _reasoning_end_token_id = 128822

        class Reasoning:
            _parser_engine = Engine()

        class Parser:
            reasoning_parser = Reasoning()

        self.assertEqual(markers(Parser()), (128821, 128822))
        self.assertIsNone(markers(None))
        self.assertIsNone(markers(types.SimpleNamespace(reasoning_parser=None)))
        Engine._reasoning_end_token_id = 128821  # degenerate: same id
        self.assertIsNone(markers(Parser()))

    def test_thinkoff_engine_input_swaps_only_a_trailing_think_marker(self):
        swap = self.ns["_issue191_thinkoff_engine_input"]

        class Engine:
            _reasoning_start_token_id = 128821
            _reasoning_end_token_id = 128822

        parser = types.SimpleNamespace(reasoning_parser=types.SimpleNamespace(_parser_engine=Engine()))
        ids = [1, 2, 3, 128804, 128821]
        engine_input = {
            "type": "token",
            "prompt_token_ids": ids,
            "prompt": "rendered text",
            "prompt_token_offsets": [(0, 1)] * 5,
            "assistant_tokens_mask": [0] * 5,
            "cache_salt": "s",
        }
        swapped = swap(engine_input, parser)
        self.assertEqual(swapped["prompt_token_ids"], [1, 2, 3, 128804, 128822])
        self.assertEqual(swapped["type"], "token")
        self.assertEqual(swapped["cache_salt"], "s")
        for stale in ("prompt", "prompt_token_offsets", "assistant_tokens_mask"):
            self.assertNotIn(stale, swapped)
        # the original engine input is untouched (the earlier attempts used it)
        self.assertEqual(engine_input["prompt_token_ids"], ids)
        self.assertIn("prompt", engine_input)
        # thinking already off (prompt ends with </think>) -> no fallback
        self.assertIsNone(swap({"prompt_token_ids": [1, 128822]}, parser))
        # no marker knowledge / not a token prompt -> no fallback
        self.assertIsNone(swap(engine_input, None))
        self.assertIsNone(swap({"prompt_embeds": object()}, parser))
        self.assertIsNone(swap({"prompt_token_ids": []}, parser))
        self.assertIsNone(swap("raw string", parser))

    def test_transform_wires_the_fallback_into_the_last_retry(self):
        patched = HF.transform(FIXTURE.read_bytes()).decode("utf-8")
        self.assertIn("attempt >= retries", patched)
        self.assertIn("_issue191_thinkoff_engine_input(engine_input, parser)", patched)
        self.assertIn('retry_kwargs["thinking"] = False', patched)
        self.assertIn("reasoning_ended=retry_reasoning_ended", patched)
        self.assertIn("parser=retry_parser", patched)
        self.assertIn('fallback = "thinkoff"', patched)
        self.assertIn("regenerating request=%s attempt=%d fallback=%s", patched)
        # the first attempt and the mode=log path are untouched
        self.assertIn("contract violation request=%s attempt=%d mode=%s reason=%s", patched)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn(
            'DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED: "${DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED:-0}"',
            compose,
        )
        self.assertIn('DSPARK_ISSUE191_TOOLCALL_RETRIES: "${DSPARK_ISSUE191_TOOLCALL_RETRIES:-2}"', compose)
        self.assertIn('DSPARK_ISSUE191_TOOLCALL_MODE: "${DSPARK_ISSUE191_TOOLCALL_MODE:-failclosed}"', compose)
        self.assertIn(
            'DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK: "${DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK:-1}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-issue191-toolcall-failclosed.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ISSUE191_TOOLCALL_HOTFIX:-./patches/hotfix-vllm-issue191-toolcall-failclosed.py}"
            ":/opt/hotfix-vllm-issue191-toolcall-failclosed.py:ro",
            compose,
        )

    def test_compose_async_scheduling_knob(self):
        compose = COMPOSE.read_text()
        self.assertIn('DSPARK_ASYNC_SCHEDULING: "${DSPARK_ASYNC_SCHEDULING:-1}"', compose)
        self.assertNotIn("\n        --async-scheduling\n", compose)
        self.assertIn("$${ASYNC_SCHEDULING_ARGS}", compose)
        self.assertIn('ASYNC_SCHEDULING_ARGS="--async-scheduling"', compose)

    def test_launcher_passthrough_sync_and_preflight(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_ISSUE191_TOOLCALL_HOTFIX='./patches/hotfix-vllm-issue191-toolcall-failclosed.py'", start
        )
        self.assertIn("DSPARK_ASYNC_SCHEDULING=$REMOTE_ASYNC_SCHEDULING", start)
        self.assertIn("DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK=$REMOTE_ISSUE191_THINKOFF", start)
        self.assertIn("/opt/hotfix-vllm-issue191-toolcall-failclosed.py --check", start)
        self.assertIn("patches/hotfix-vllm-issue191-toolcall-failclosed.py\"", start)
        self.assertIn("DSPARK_ASYNC_SCHEDULING must be 0 or 1", start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ISSUE191_TOOLCALL_FAILCLOSED=0", env)
        self.assertIn("DSPARK_ISSUE191_TOOLCALL_THINKOFF_FALLBACK=1", env)
        self.assertIn("DSPARK_ASYNC_SCHEDULING=1", env)
        self.assertIn("scripts/test-issue191-toolcall-failclosed.py", CI.read_text())


if __name__ == "__main__":
    unittest.main()
