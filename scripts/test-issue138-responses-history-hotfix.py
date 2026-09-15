#!/usr/bin/env python3
"""Hermetic CPU tests for the issue #138 Responses history hotpatch.

The OLD/NEW method fixtures are loaded from the patcher module and frozen by
the independent SHA-256 pins below: the preimage hash covers vLLM commit
752a3a504485790a2e8491cacbb35c137339ad34
``vllm/entrypoints/openai/responses/protocol.py`` lines 461-557 (complete file
Git blob ``ba8bc5a40f1bcffe8073cfdb4f0a8995da5e02e4``); the postimage hash
freezes the expected patched method. Any patcher constant drift fails the pins.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "patches" / "hotfix-vllm-issue138-responses-history.py"
PINNED_VLLM_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
PINNED_PROTOCOL_BLOB = "ba8bc5a40f1bcffe8073cfdb4f0a8995da5e02e4"
OLD_METHOD_SHA256 = "2412484a81e8679cedf1934287f1b4187a72bf6e8c910c8ecad463b29b79d9d7"
NEW_METHOD_SHA256 = "536f3a305821445328c1f2131b898bef8a8f0c7d278cef4ba29701501eaf3d78"
TYPE_ALIAS_GUARD = (
    "ResponseInputOutputItem: TypeAlias = "
    "ResponseInputItemParam | ResponseOutputItem"
)
INPUT_FIELD_GUARD = "\n    input: str | list[ResponseInputOutputItem]\n"
MARKER = (
    "# [issue138-hotfix] Normalize the observed singleton type-less "
    "assistant output replay."
)


def load_patcher():
    spec = importlib.util.spec_from_file_location("issue138_patcher", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PATCHER = load_patcher()

# The patcher's exact preimage/postimage method constants, frozen by the
# independent SHA-256 pins above.
OLD_METHOD = _PATCHER.OLD_METHOD
NEW_METHOD = _PATCHER.NEW_METHOD


def protocol_source(method: str) -> str:
    return (
        TYPE_ALIAS_GUARD
        + "\n\nclass ResponsesRequest:\n"
        + "    input: str | list[ResponseInputOutputItem]\n\n"
        + method
    )


def invoke(function, *args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = function(*args)
    return result, stdout.getvalue(), stderr.getvalue()


class ValidationError(Exception):
    pass


class Record:
    def __init__(self, **values):
        self.values = values
        for key, value in values.items():
            setattr(self, key, value)


class OutputPart(Record):
    pass


class ResponseOutputMessage(Record):
    def __init__(self, **values):
        if values.get("type") != "message" or values.get("role") != "assistant":
            raise ValidationError("bad output-message discriminator")
        if not isinstance(values.get("id"), str) or not values["id"]:
            raise ValidationError("bad id")
        if values.get("status") not in {"completed", "in_progress", "incomplete"}:
            raise ValidationError("bad status")
        raw_content = values.get("content")
        if not isinstance(raw_content, list) or not raw_content:
            raise ValidationError("bad content")
        content = []
        for part in raw_content:
            if not isinstance(part, dict):
                raise ValidationError("bad part")
            if part.get("type") == "output_text":
                if not isinstance(part.get("text"), str):
                    raise ValidationError("bad output text")
                if not isinstance(part.get("annotations"), list):
                    raise ValidationError("bad annotations")
            elif part.get("type") == "refusal":
                if not isinstance(part.get("refusal"), str):
                    raise ValidationError("bad refusal")
            else:
                raise ValidationError("bad part discriminator")
            content.append(OutputPart(**part))
        super().__init__(**{**values, "content": content})


class ResponseReasoningItem(Record):
    def __init__(self, **values):
        if values.get("type") != "reasoning" or not isinstance(values.get("id"), str):
            raise ValidationError("bad reasoning")
        super().__init__(**values)


class ResponseFunctionToolCall(Record):
    def __init__(self, **values):
        if values.get("type") != "function_call":
            raise ValidationError("bad function call")
        for key in ("call_id", "name", "arguments"):
            if not isinstance(values.get(key), str):
                raise ValidationError(f"bad {key}")
        super().__init__(**values)


class Logger:
    def debug(self, *_args, **_kwargs):
        pass


def model_validator(**_kwargs):
    return lambda value: value


def build_harness():
    namespace = {
        "model_validator": model_validator,
        "ResponseOutputMessage": ResponseOutputMessage,
        "ResponseReasoningItem": ResponseReasoningItem,
        "ResponseFunctionToolCall": ResponseFunctionToolCall,
        "ValidationError": ValidationError,
        "logger": Logger(),
        "random_uuid": lambda: "deterministic",
    }
    exec("class Harness:\n" + NEW_METHOD, namespace)
    return namespace["Harness"]


class SourceLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = _PATCHER

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.target = self.root / "protocol.py"

    def write(self, raw: bytes, mode: int = 0o640):
        self.target.write_bytes(raw)
        self.target.chmod(mode)

    def stock(self) -> bytes:
        return protocol_source(OLD_METHOD).encode()

    def assert_rejected_without_write(self, raw: bytes):
        self.write(raw)
        before = self.target.stat()
        result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        after = self.target.stat()
        self.assertEqual(result, 1)
        self.assertIn("[FAIL]", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(after.st_mode, before.st_mode)
        self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(after.st_ino, before.st_ino)

    def test_patcher_constants_match_independent_sha256_pins_and_guards(self):
        self.assertEqual(hashlib.sha256(OLD_METHOD.encode()).hexdigest(), OLD_METHOD_SHA256)
        self.assertEqual(hashlib.sha256(NEW_METHOD.encode()).hexdigest(), NEW_METHOD_SHA256)
        self.assertEqual(self.patcher.TYPE_ALIAS_GUARD, TYPE_ALIAS_GUARD)
        self.assertEqual(self.patcher.INPUT_FIELD_GUARD, INPUT_FIELD_GUARD)
        self.assertEqual(self.patcher.MARKER, MARKER)
        for method in (OLD_METHOD, NEW_METHOD):
            source = protocol_source(method)
            self.assertEqual(source.count(TYPE_ALIAS_GUARD), 1)
            self.assertEqual(source.count(INPUT_FIELD_GUARD), 1)
            self.assertLess(source.index(TYPE_ALIAS_GUARD), source.index(INPUT_FIELD_GUARD))
            self.assertLess(source.index(INPUT_FIELD_GUARD), source.index(method))
        self.assertEqual(NEW_METHOD.count(MARKER), 1)
        self.assertNotIn(MARKER, OLD_METHOD)

    def test_exact_preimage_becomes_exact_postimage_and_preserves_every_outer_byte(self):
        prefix = "# pinned fixture prefix\n"
        suffix = "\n# pinned fixture suffix\n"
        old = prefix + protocol_source(OLD_METHOD) + suffix
        expected = prefix + protocol_source(NEW_METHOD) + suffix
        self.write(old.encode(), 0o754)
        result, stdout, stderr = invoke(self.patcher.apply, self.target)
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("patched and verified", stdout)
        self.assertEqual(self.target.read_bytes(), expected.encode())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o754)

    def test_applied_state_is_idempotent_without_rewrite(self):
        self.write(protocol_source(NEW_METHOD).encode(), 0o604)
        before = self.target.stat()
        raw = self.target.read_bytes()
        result, stdout, stderr = invoke(self.patcher.apply, self.target)
        after = self.target.stat()
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("already applied and verified", stdout)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(after.st_mode, before.st_mode)
        self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(after.st_ino, before.st_ino)

    def test_status_distinguishes_stock_applied_and_drift_without_writing(self):
        for label, method, expected, rc in (
            ("stock", OLD_METHOD, "compatible stock source", 0),
            ("applied", NEW_METHOD, "already applied and verified", 0),
            ("drift", OLD_METHOD.replace("Specifically", "SpecificalLy", 1), "source lock mismatch", 1),
        ):
            with self.subTest(label=label):
                raw = protocol_source(method).encode()
                self.write(raw)
                before = self.target.stat()
                result, stdout, stderr = invoke(
                    self.patcher.main,
                    ["patcher", "--status", str(self.target)],
                )
                self.assertEqual(result, rc)
                self.assertIn(expected, stdout + stderr)
                self.assertEqual(self.target.read_bytes(), raw)
                after = self.target.stat()
                self.assertEqual((after.st_ino, after.st_mtime_ns), (before.st_ino, before.st_mtime_ns))

    def test_missing_target_and_invalid_utf8_fail(self):
        missing = self.root / "missing.py"
        result, _stdout, stderr = invoke(self.patcher.apply, missing)
        self.assertEqual(result, 1)
        self.assertIn("not found", stderr)
        self.assert_rejected_without_write(b"\xff\xfe")

    def test_drift_duplicate_mixed_and_marker_only_states_are_rejected(self):
        cases = {
            "one-character drift": protocol_source(
                OLD_METHOD.replace("Specifically handles", "Specifically handleS", 1)
            ),
            "duplicate old": protocol_source(OLD_METHOD + "\n" + OLD_METHOD),
            "duplicate new": protocol_source(NEW_METHOD + "\n" + NEW_METHOD),
            "old plus new": protocol_source(OLD_METHOD + "\n" + NEW_METHOD),
            "marker only": protocol_source(OLD_METHOD) + "\n" + MARKER + "\n",
            "alias duplicate": TYPE_ALIAS_GUARD + "\n" + protocol_source(OLD_METHOD),
            "input guard missing": protocol_source(OLD_METHOD).replace(
                "    input: str | list[ResponseInputOutputItem]",
                "    input: object",
                1,
            ),
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                self.assert_rejected_without_write(source.encode())

    def test_invalid_compiled_replacement_never_publishes(self):
        raw = self.stock()
        self.write(raw)
        invalid = NEW_METHOD + "\n    this is ! invalid python\n"
        with mock.patch.object(self.patcher, "NEW_METHOD", invalid):
            result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        self.assertEqual(result, 1)
        self.assertIn("before publication", stderr)
        self.assertEqual(self.target.read_bytes(), raw)

    def test_replace_failure_restores_original_bytes_and_mode(self):
        raw = self.stock()
        self.write(raw, 0o605)
        real = self.patcher.os.replace
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("simulated replace failure")
            return real(*args, **kwargs)

        with mock.patch.object(self.patcher.os, "replace", side_effect=fail_once):
            result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        self.assertEqual(result, 1)
        self.assertIn("original restored", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o605)

    def test_post_publication_verification_failure_rolls_back(self):
        raw = self.stock()
        self.write(raw, 0o744)
        real = self.patcher._source_state

        def fail_after_commit(source, target):
            if target.read_bytes() != raw:
                raise self.patcher.PatchError(
                    "simulated post-publication verification failure"
                )
            return real(source, target)

        with mock.patch.object(self.patcher, "_source_state", side_effect=fail_after_commit):
            result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        self.assertEqual(result, 1)
        self.assertIn("original restored", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o744)


class InjectedMethodSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = build_harness()

    def parse(self, items):
        data = {"input": items}
        result = self.harness.input_item_parsing(data)
        self.assertIs(result, data)
        return result["input"]

    def test_exact_issue_item_normalizes_to_stock_output_object(self):
        original = {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello there! 👋"}],
        }
        result = self.parse([original])
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertIsInstance(item, ResponseOutputMessage)
        self.assertEqual(item.type, "message")
        self.assertEqual(item.id, "msg_deterministic")
        self.assertEqual(item.status, "completed")
        self.assertEqual(item.role, "assistant")
        self.assertEqual(item.content[0].text, "Hello there! 👋")
        self.assertEqual(item.content[0].annotations, [])
        self.assertNotIn("type", original)
        self.assertNotIn("annotations", original["content"][0])

    def test_exact_reported_four_item_payload_changes_only_error_index_two(self):
        items = [
            {
                "role": "system",
                "content": (
                    "Current model: deepseek-v4-flash-0731\n"
                    "Current date: 2026-08-25\n"
                    " Additional info for this conversation:"
                ),
            },
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": (
                        "Hello there! 👋 It's great to hear from you. \n\n"
                        "How can I help you today? Whether you have a question, need "
                        "assistance with a task, or just want to chat about something, "
                        "I'm all ears. What's on your mind?"
                    ),
                }],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "your version"}]},
        ]
        original_assistant = items[2]
        result = self.parse(items)
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], items[0])
        self.assertIs(result[1], items[1])
        self.assertIsInstance(result[2], ResponseOutputMessage)
        self.assertIs(result[3], items[3])
        self.assertEqual(result[2].content[0].text, original_assistant["content"][0]["text"])
        self.assertNotIn("type", original_assistant)

    def test_supplied_identity_phase_annotations_logprobs_and_optional_fields_survive(self):
        original = {
            "role": "assistant",
            "id": "msg_client",
            "status": "in_progress",
            "phase": "commentary",
            "content": [{
                "type": "output_text",
                "text": "kept",
                "annotations": [{"type": "url_citation", "url": "https://example.invalid"}],
                "logprobs": [{"token": "kept", "logprob": -0.1}],
            }],
        }
        item = self.parse([original])[0]
        self.assertEqual(item.id, "msg_client")
        self.assertEqual(item.status, "in_progress")
        self.assertEqual(item.phase, "commentary")
        self.assertEqual(item.content[0].annotations, original["content"][0]["annotations"])
        self.assertEqual(item.content[0].logprobs, original["content"][0]["logprobs"])

    def test_two_legacy_items_remain_two_ordered_objects(self):
        items = [
            {"role": "assistant", "content": [{"type": "output_text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "second"}]},
        ]
        result = self.parse(items)
        self.assertEqual(len(result), 2)
        self.assertEqual([item.content[0].text for item in result], ["first", "second"])

    def test_canonical_output_message_keeps_stock_semantics(self):
        canonical = {
            "type": "message",
            "id": "msg_control",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{
                "type": "output_text", "text": "valid output",
                "annotations": [], "logprobs": [],
            }],
        }
        item = self.parse([canonical])[0]
        self.assertIsInstance(item, ResponseOutputMessage)
        self.assertEqual(item.values["type"], "message")
        self.assertEqual(item.values["id"], "msg_control")
        self.assertEqual(item.values["phase"], "final_answer")
        self.assertEqual(item.content[0].logprobs, [])

    def test_canonical_message_without_annotations_uses_existing_stock_fill(self):
        canonical = {
            "type": "message",
            "id": "msg_stock_fill",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "stock"}],
        }
        item = self.parse([canonical])[0]
        self.assertIsInstance(item, ResponseOutputMessage)
        self.assertEqual(item.content[0].annotations, [])

    def test_easy_assistant_forms_remain_raw_without_generated_metadata(self):
        string_item = {"role": "assistant", "content": "easy string"}
        typed_string = {"type": "message", "role": "assistant", "content": "easy string"}
        input_text = {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "valid easy input"}],
        }
        result = self.parse([string_item, typed_string, input_text])
        self.assertIs(result[0], string_item)
        self.assertIs(result[1], typed_string)
        self.assertIs(result[2], input_text)
        for item in result:
            self.assertNotIn("id", item)
            self.assertNotIn("status", item)

    def test_mixed_typed_history_retains_cardinality_and_order(self):
        reasoning = {"type": "reasoning", "summary": [], "id": "rs_control"}
        call = {
            "type": "function_call", "call_id": "call_1",
            "name": "lookup", "arguments": "{}",
        }
        result_item = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        legacy = {"role": "assistant", "content": [{"type": "output_text", "text": "legacy"}]}
        reference = {"type": "item_reference", "id": "msg_prior"}
        user = {"role": "user", "content": [{"type": "input_text", "text": "next"}]}
        result = self.parse([reasoning, call, result_item, legacy, reference, user])
        self.assertEqual(len(result), 6)
        self.assertIsInstance(result[0], ResponseReasoningItem)
        self.assertIsInstance(result[1], ResponseFunctionToolCall)
        self.assertIs(result[2], result_item)
        self.assertIsInstance(result[3], ResponseOutputMessage)
        self.assertIs(result[4], reference)
        self.assertIs(result[5], user)

    def test_iterator_and_non_dictionary_behavior_remains_stock(self):
        user = {"role": "user", "content": "hello"}
        result = self.parse(iter(["bare", 7, user]))
        self.assertEqual(result, ["bare", 7, user])
        self.assertIs(result[2], user)
        marker = object()
        data = {"input": marker}
        self.assertIs(self.harness.input_item_parsing(data)["input"], marker)

    def test_help_must_reject_neighbors_are_left_raw_and_unmutated(self):
        cases = [
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "one"},
                {"type": "output_text", "text": "two"},
            ]},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "one"},
                {"type": "refusal", "refusal": "no"},
            ]},
            {"role": "assistant", "content": [
                {"type": "output_text", "text": "one"},
                {"type": "input_text", "text": "two"},
            ]},
            {"role": "assistant", "content": [{"type": "output_text"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": 123}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": None}]},
            {"role": "user", "content": [{"type": "output_text", "text": "one"}]},
            {"type": None, "role": "assistant", "content": [{"type": "output_text", "text": "one"}]},
            {"type": "", "role": "assistant", "content": [{"type": "output_text", "text": "one"}]},
            {"type": "bogus", "role": "assistant", "content": [{"type": "output_text", "text": "one"}]},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]},
            {"role": "assistant", "content": ["bare string part"]},
        ]
        result = self.parse(cases)
        self.assertEqual(len(result), len(cases))
        for original, processed in zip(cases, result):
            self.assertIs(processed, original)
            self.assertNotIn("id", processed)
            self.assertNotIn("status", processed)

    def test_constructor_failures_restore_exact_original_candidate(self):
        cases = [
            {
                "role": "assistant", "id": None,
                "content": [{"type": "output_text", "text": "one"}],
            },
            {
                "role": "assistant", "id": "",
                "content": [{"type": "output_text", "text": "one"}],
            },
            {
                "role": "assistant", "status": None,
                "content": [{"type": "output_text", "text": "one"}],
            },
            {
                "role": "assistant", "status": "",
                "content": [{"type": "output_text", "text": "one"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "one", "annotations": "bad"}],
            },
        ]
        snapshots = [repr(item) for item in cases]
        result = self.parse(cases)
        for original, processed, snapshot in zip(cases, result, snapshots):
            self.assertIs(processed, original)
            self.assertEqual(repr(original), snapshot)
            self.assertNotIn("type", original)


if __name__ == "__main__":
    unittest.main()
