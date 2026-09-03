#!/usr/bin/env python3
"""Hermetic CPU tests for the Codex ``agent_message`` compatibility patch."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATCHER_PATH = ROOT / "patches" / "hotfix-vllm-codex-agent-message.py"
ISSUE138_PATCHER_PATH = (
    ROOT / "patches" / "hotfix-vllm-issue138-responses-history.py"
)
TYPE_ALIAS_GUARD = (
    "ResponseInputOutputItem: TypeAlias = "
    "ResponseInputItemParam | ResponseOutputItem"
)
INPUT_FIELD_GUARD = "\n    input: str | list[ResponseInputOutputItem]\n"
EXPECTED_PREIMAGE_HASHES = {
    "stock": "2412484a81e8679cedf1934287f1b4187a72bf6e8c910c8ecad463b29b79d9d7",
    "issue138": "536f3a305821445328c1f2131b898bef8a8f0c7d278cef4ba29701501eaf3d78",
}
EXPECTED_POSTIMAGE_HASHES = {
    "stock-applied": "010ba578d77333a3e3d2d6b31d1a7b692521a48d899503459ebfede9d09c6851",
    "issue138-applied": "a5af28655454f4f1cdd22e89eaaa0c5400578cdd0efbb6498412d0b8639efc7e",
}


def load_module(path: Path, name: str):
    if not path.is_file():
        raise AssertionError(f"required patcher is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ISSUE138 = load_module(ISSUE138_PATCHER_PATH, "issue138_fixture")


def protocol_source(method: str) -> str:
    return (
        TYPE_ALIAS_GUARD
        + "\n\nclass ResponsesRequest:\n"
        + "    input: str | list[ResponseInputOutputItem]\n\n"
        + method
    )


def extract_method(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "input_item_parsing":
                start = min(decorator.lineno for decorator in node.decorator_list)
                lines = source.splitlines(keepends=True)
                return "".join(lines[start - 1 : node.end_lineno]) + "\n"
    raise AssertionError("input_item_parsing fixture not found")


def invoke(function, *args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = function(*args)
    return result, stdout.getvalue(), stderr.getvalue()


class PatchTransactionTests(unittest.TestCase):
    def setUp(self):
        self.patcher = load_module(PATCHER_PATH, "codex_agent_message_patcher")
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.target = Path(self.tempdir.name) / "protocol.py"

    def write_method(self, method: str, mode: int = 0o640) -> bytes:
        raw = protocol_source(method).encode()
        self.target.write_bytes(raw)
        self.target.chmod(mode)
        return raw

    def test_source_lock_hashes_match_independent_pins(self):
        self.assertEqual(self.patcher.PREIMAGE_HASHES, EXPECTED_PREIMAGE_HASHES)
        self.assertEqual(self.patcher.POSTIMAGE_HASHES, EXPECTED_POSTIMAGE_HASHES)
        self.assertEqual(
            ISSUE138.CODEX_COMBINED_METHOD_SHA256,
            EXPECTED_POSTIMAGE_HASHES["issue138-applied"],
        )
        for label, method in (
            ("stock", ISSUE138.OLD_METHOD),
            ("issue138", ISSUE138.NEW_METHOD),
        ):
            self.assertEqual(
                hashlib.sha256(method.encode()).hexdigest(),
                EXPECTED_PREIMAGE_HASHES[label],
            )
            patched = method.replace(
                self.patcher.BRANCH_ANCHOR,
                self.patcher.AGENT_BRANCH,
                1,
            )
            self.assertEqual(
                hashlib.sha256(patched.encode()).hexdigest(),
                EXPECTED_POSTIMAGE_HASHES[f"{label}-applied"],
            )

    def test_stock_and_issue138_preimages_apply_and_preserve_outer_bytes(self):
        for label, method in (
            ("stock", ISSUE138.OLD_METHOD),
            ("issue138", ISSUE138.NEW_METHOD),
        ):
            with self.subTest(label=label):
                prefix = f"# {label} prefix\n"
                suffix = f"\n# {label} suffix\n"
                raw = (prefix + protocol_source(method) + suffix).encode()
                self.target.write_bytes(raw)
                self.target.chmod(0o754)
                result, stdout, stderr = invoke(self.patcher.apply, self.target)
                self.assertEqual((result, stderr), (0, ""))
                self.assertIn("patched and verified", stdout)
                updated = self.target.read_text()
                self.assertTrue(updated.startswith(prefix))
                self.assertTrue(updated.endswith(suffix))
                self.assertEqual(updated.count(self.patcher.MARKER), 1)
                self.assertEqual(self.target.stat().st_mode & 0o777, 0o754)

    def test_issue138_then_agent_message_patch_survives_second_boot(self):
        self.write_method(ISSUE138.OLD_METHOD, 0o754)
        for patcher in (ISSUE138, self.patcher):
            result, _stdout, stderr = invoke(patcher.apply, self.target)
            self.assertEqual((result, stderr), (0, ""))
        first_boot = self.target.read_bytes()
        first_mode = self.target.stat().st_mode

        for patcher in (ISSUE138, self.patcher):
            result, _stdout, stderr = invoke(patcher.apply, self.target)
            self.assertEqual((result, stderr), (0, ""))

        self.assertEqual(self.target.read_bytes(), first_boot)
        self.assertEqual(self.target.stat().st_mode, first_mode)
        method = extract_method(self.target.read_text())
        self.assertIn(ISSUE138.MARKER, method)
        self.assertIn(self.patcher.MARKER, method)

    def test_issue138_rejects_tampered_combined_postimage_without_writing(self):
        self.write_method(ISSUE138.OLD_METHOD)
        self.assertEqual(invoke(ISSUE138.apply, self.target)[0], 0)
        self.assertEqual(invoke(self.patcher.apply, self.target)[0], 0)
        tampered = self.target.read_text().replace(
            '"role": "assistant"',
            '"role": "user"',
            1,
        )
        self.target.write_text(tampered)
        before = self.target.read_bytes()

        result, _stdout, stderr = invoke(ISSUE138.apply, self.target)

        self.assertEqual(result, 1)
        self.assertIn("combined issue138/Codex method source lock mismatch", stderr)
        self.assertEqual(self.target.read_bytes(), before)

    def test_applied_state_is_idempotent_and_status_is_read_only(self):
        self.write_method(ISSUE138.OLD_METHOD, 0o604)
        self.assertEqual(invoke(self.patcher.apply, self.target)[0], 0)
        raw = self.target.read_bytes()
        before = self.target.stat()
        result, stdout, stderr = invoke(
            self.patcher.main,
            ["patcher", "--status", str(self.target)],
        )
        after = self.target.stat()
        self.assertEqual((result, stderr), (0, ""))
        self.assertIn("already applied and verified", stdout)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(
            (after.st_ino, after.st_mtime_ns, after.st_mode),
            (before.st_ino, before.st_mtime_ns, before.st_mode),
        )

    def test_status_accepts_later_extension_but_rejects_broken_agent_branch(self):
        applied = ISSUE138.NEW_METHOD.replace(
            self.patcher.BRANCH_ANCHOR,
            self.patcher.AGENT_BRANCH,
            1,
        )
        extended = applied.replace(
            '                agent_message_content = item.get("content")\n',
            '                future_patch_value = item.get("future_extension")\n'
            '                agent_message_content = item.get("content")\n',
            1,
        )
        broken = extended.replace(
            '                    "type": "message",',
            '                    "type": "broken_message",',
            1,
        )

        for label, method, expected, rc in (
            ("extended", extended, "applied-with-extensions", 0),
            ("broken", broken, "source lock mismatch", 1),
        ):
            with self.subTest(label=label):
                raw = self.write_method(method)
                before = self.target.stat()
                result, stdout, stderr = invoke(
                    self.patcher.main,
                    ["patcher", "--status", str(self.target)],
                )
                after = self.target.stat()
                self.assertEqual(result, rc)
                self.assertIn(expected, stdout + stderr)
                self.assertEqual(self.target.read_bytes(), raw)
                self.assertEqual(
                    (after.st_ino, after.st_mtime_ns, after.st_mode),
                    (before.st_ino, before.st_mtime_ns, before.st_mode),
                )

    def test_apply_still_rejects_later_extension_without_writing(self):
        applied = ISSUE138.NEW_METHOD.replace(
            self.patcher.BRANCH_ANCHOR,
            self.patcher.AGENT_BRANCH,
            1,
        )
        extended = applied.replace(
            '                agent_message_content = item.get("content")\n',
            '                future_patch_value = item.get("future_extension")\n'
            '                agent_message_content = item.get("content")\n',
            1,
        )
        raw = self.write_method(extended)
        before = self.target.stat()
        result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        after = self.target.stat()
        self.assertEqual(result, 1)
        self.assertIn("source lock mismatch", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(
            (after.st_ino, after.st_mtime_ns, after.st_mode),
            (before.st_ino, before.st_mtime_ns, before.st_mode),
        )

    def test_status_accepts_both_preimages_without_writing(self):
        for label, method in (
            ("stock", ISSUE138.OLD_METHOD),
            ("issue138", ISSUE138.NEW_METHOD),
        ):
            with self.subTest(label=label):
                raw = self.write_method(method)
                before = self.target.stat()
                result, stdout, stderr = invoke(
                    self.patcher.main,
                    ["patcher", "--status", str(self.target)],
                )
                after = self.target.stat()
                self.assertEqual((result, stderr), (0, ""))
                self.assertIn(f"compatible {label} source", stdout)
                self.assertEqual(self.target.read_bytes(), raw)
                self.assertEqual(
                    (after.st_ino, after.st_mtime_ns, after.st_mode),
                    (before.st_ino, before.st_mtime_ns, before.st_mode),
                )

    def test_missing_target_and_invalid_utf8_fail_closed(self):
        missing = Path(self.tempdir.name) / "missing.py"
        result, _stdout, stderr = invoke(self.patcher.apply, missing)
        self.assertEqual(result, 1)
        self.assertIn("not found", stderr)

        raw = b"\xff\xfe"
        self.target.write_bytes(raw)
        before = self.target.stat()
        result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        after = self.target.stat()
        self.assertEqual(result, 1)
        self.assertIn("not UTF-8", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(
            (after.st_ino, after.st_mtime_ns, after.st_mode),
            (before.st_ino, before.st_mtime_ns, before.st_mode),
        )

    def test_drift_and_mixed_marker_states_fail_without_write(self):
        cases = {
            "method drift": ISSUE138.OLD_METHOD.replace(
                "Specifically handles", "Specifically handleS", 1
            ),
            "foreign marker": ISSUE138.OLD_METHOD + "\n" + self.patcher.MARKER,
        }
        for label, method in cases.items():
            with self.subTest(label=label):
                raw = self.write_method(method)
                before = self.target.stat()
                result, _stdout, stderr = invoke(self.patcher.apply, self.target)
                after = self.target.stat()
                self.assertEqual(result, 1)
                self.assertIn("[FAIL]", stderr)
                self.assertEqual(self.target.read_bytes(), raw)
                self.assertEqual(
                    (after.st_ino, after.st_mtime_ns, after.st_mode),
                    (before.st_ino, before.st_mtime_ns, before.st_mode),
                )

    def test_post_publication_failure_restores_original_bytes_and_mode(self):
        raw = self.write_method(ISSUE138.OLD_METHOD, 0o705)
        real = self.patcher._source_state

        def fail_after_publish(source, target):
            if target.read_bytes() != raw:
                raise self.patcher.PatchError("simulated verification failure")
            return real(source, target)

        with mock.patch.object(
            self.patcher, "_source_state", side_effect=fail_after_publish
        ):
            result, _stdout, stderr = invoke(self.patcher.apply, self.target)
        self.assertEqual(result, 1)
        self.assertIn("original restored", stderr)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o705)


class InjectedSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patcher = None
        cls.harness = None
        if not PATCHER_PATH.is_file():
            return
        cls.patcher = load_module(PATCHER_PATH, "codex_agent_message_semantics")
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "protocol.py"
            target.write_text(protocol_source(ISSUE138.OLD_METHOD))
            result, _stdout, stderr = invoke(cls.patcher.apply, target)
            if result != 0:
                raise AssertionError(stderr)
            method = extract_method(target.read_text())
        namespace = {
            "model_validator": lambda **_kwargs: lambda value: value,
            "ResponseOutputMessage": object,
            "ResponseReasoningItem": object,
            "ResponseFunctionToolCall": object,
            "ValidationError": Exception,
            "logger": type("Logger", (), {"debug": lambda *_a, **_kw: None})(),
            "random_uuid": lambda: "deterministic",
        }
        exec("class Harness:\n" + method, namespace)
        cls.harness = namespace["Harness"]

    def parse(self, items):
        self.assertIsNotNone(self.harness, f"required patcher is missing: {PATCHER_PATH}")
        data = {"input": items}
        result = self.harness.input_item_parsing(data)
        self.assertIs(result, data)
        return result["input"]

    def test_real_codex_item_becomes_minimal_assistant_message(self):
        content = [{"type": "input_text", "text": "Message Type: NEW_TASK\nhello"}]
        original = {
            "type": "agent_message",
            "id": "amsg_01a05588-2eee-7e73-a581-43876d51f7e4",
            "author": "/root",
            "recipient": "/root/crm_code_history",
            "content": content,
            "internal_chat_message_metadata_passthrough": {
                "turn_id": "01a05588-258a-7523-a935-81688b8a3774",
                "create_time": 1788141383.40663,
            },
        }
        result = self.parse([original])
        self.assertEqual(
            result,
            [{"type": "message", "role": "assistant", "content": content}],
        )
        self.assertIs(result[0]["content"], content)
        self.assertEqual(original["author"], "/root")
        self.assertIn("internal_chat_message_metadata_passthrough", original)

    def test_minimal_single_part_input_text_form_is_accepted(self):
        content = [{"type": "input_text", "text": "hello"}]
        original = {
            "type": "agent_message",
            "id": "amsg_1",
            "author": "/root",
            "recipient": "/root/sub",
            "content": content,
        }
        self.assertEqual(
            self.parse([original]),
            [{"type": "message", "role": "assistant", "content": content}],
        )

    def test_malformed_or_extended_agent_messages_are_left_for_stock_rejection(self):
        valid = {
            "type": "agent_message",
            "id": "amsg_1",
            "author": "/root",
            "recipient": "/root/sub",
            "content": [{"type": "input_text", "text": "hello"}],
        }
        cases = [
            {**valid, "extra": True},
            {key: value for key, value in valid.items() if key != "id"},
            {**valid, "id": ""},
            {**valid, "author": None},
            {**valid, "recipient": ""},
            {**valid, "content": []},
            {
                **valid,
                "content": [
                    {"type": "input_text", "text": "first"},
                    {"type": "input_text", "text": "second"},
                ],
            },
            {**valid, "content": "hello"},
            {**valid, "content": [{"type": "output_text", "text": "hello"}]},
            {**valid, "content": [{"type": "input_text"}]},
            {**valid, "content": [{"type": "input_text", "text": 7}]},
            {
                **valid,
                "content": [
                    {"type": "input_text", "text": "hello", "unexpected": True}
                ],
            },
            {
                **valid,
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": "turn",
                    "create_time": 1.0,
                    "unexpected": True,
                },
            },
            {
                **valid,
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": "",
                    "create_time": 1.0,
                },
            },
            {
                **valid,
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": "turn",
                    "create_time": True,
                },
            },
        ]
        result = self.parse(cases)
        self.assertEqual(len(result), len(cases))
        for original, processed in zip(cases, result):
            self.assertIs(processed, original)

    def test_unknown_types_and_standard_items_are_unchanged(self):
        items = [
            {"type": "future_item", "payload": "keep"},
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
        result = self.parse(items)
        self.assertEqual(result, items)
        for original, processed in zip(items, result):
            self.assertIs(processed, original)


class WiringTests(unittest.TestCase):
    def test_default_off_mount_env_and_fail_closed_apply_are_wired(self):
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        env_example = (ROOT / ".env.dspark.example").read_text()
        self.assertIn(
            "hotfix-vllm-codex-agent-message.py}:/opt/"
            "hotfix-vllm-codex-agent-message.py:ro",
            compose,
        )
        self.assertIn(
            'DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT: '
            '"${DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT:-0}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT:-0}" = "1" ]; '
            "then python3 /opt/hotfix-vllm-codex-agent-message.py || exit 1; fi;",
            compose,
        )
        self.assertIn("DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT=0", env_example)


if __name__ == "__main__":
    unittest.main()
