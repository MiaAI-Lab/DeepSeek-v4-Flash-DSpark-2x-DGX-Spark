#!/usr/bin/env python3
"""CPU regression tests for the issue #55 tool-call truncation hotfix.

These are CPU-only gates: they verify stock and previously patched source
upgrades, idempotence, the JSON helper, and the injected truncation branches
against vLLM's list-valued tool-call serialization contract.

Live behavior (finish_reason=length on truncation, no invalid-JSON args) is
verified separately against a running serve; this gate just guards the
recipe integrity.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue55-tool-truncation.py"
SERVING = ROOT / "docker-compose.dspark.yml"  # not used; recipe present-check


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("hotfix_issue55", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A minimal but faithful snippet of the stock serving.py shapes the hotfix
# anchors on. Mirrors upstream vLLM's chat serving.py at Anemll 0.1.1.
STOCK_SERVING = '''import asyncio
import functools


async def gen():
    finish_reason_sent = [False]
    tools_streamed = [True]
    tool_choice_function_name = None
    delta_message = None
    output = type("O", (), {"finish_reason": None})
    for i in range(1):
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )


def chat():
    auto_tools_called = True
    request = type("R", (), {"tool_choice": None})()
    output = type("O", (), {"finish_reason": None, "index": 0})
    message = type("M", (), {"tool_calls": []})()
    is_finish_reason_tool_calls = auto_tools_called or (
        request.tool_choice
        and request.tool_choice == "required"
        and output.finish_reason == "stop"
    )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
            )
'''


class Issue55HotfixTest(unittest.TestCase):
    def setUp(self):
        self.hf = _load_hotfix()
        # stock with both anchors
        self.stock = STOCK_SERVING

    def test_anchors_present_in_stock(self):
        self.assertIn(self.hf.HELPER_ANCHOR, self.stock)
        self.assertIn(self.hf.STREAMING_OLD, self.stock)
        self.assertIn(self.hf.NOSTREAM_OLD, self.stock)

    def test_apply_writes_all_three_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vllm"
            (root / "entrypoints/openai/chat_completion").mkdir(parents=True)
            path = root / "entrypoints/openai/chat_completion/serving.py"
            path.write_text(self.stock, encoding="utf-8")
            # run main() with argv pointing at this root
            import sys
            old = sys.argv
            sys.argv = ["hf", str(root)]
            try:
                rc = self.hf.main()
            finally:
                sys.argv = old
            self.assertEqual(rc, 0)
            patched = path.read_text(encoding="utf-8")
            self.assertIn(self.hf.MARK, patched)
            self.assertIn(self.hf.HELPER_NEW, patched)
            self.assertIn(self.hf.STREAMING_NEW, patched)
            self.assertIn(self.hf.NOSTREAM_NEW, patched)

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vllm"
            (root / "entrypoints/openai/chat_completion").mkdir(parents=True)
            path = root / "entrypoints/openai/chat_completion/serving.py"
            path.write_text(self.stock, encoding="utf-8")
            import sys
            old = sys.argv
            sys.argv = ["hf", str(root)]
            try:
                self.assertEqual(self.hf.main(), 0)
                self.assertEqual(self.hf.main(), 0)
            finally:
                sys.argv = old
            # ensure file size doesn't keep growing
            n = path.read_text(encoding="utf-8").count(self.hf.MARK)
            self.assertEqual(n, 3)  # one mark per applied edit

    def test_upgrade_previous_hotfix_is_idempotent(self):
        # Include mixed old/new installs and the streaming prefix-match trap.
        for old_stream, old_chat in [(True, True), (True, False), (False, True)]:
            with self.subTest(stream=old_stream, chat=old_chat), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / self.hf.SERVING
                path.parent.mkdir(parents=True)
                source = self.stock.replace(self.hf.HELPER_ANCHOR, self.hf.HELPER_NEW)
                source = source.replace(self.hf.STREAMING_OLD,
                    self.hf.STREAMING_PREVIOUS if old_stream else self.hf.STREAMING_NEW)
                source = source.replace(self.hf.NOSTREAM_OLD,
                    self.hf.NOSTREAM_PREVIOUS if old_chat else self.hf.NOSTREAM_NEW)
                path.write_text(source, encoding="utf-8")
                import sys
                from unittest.mock import patch
                with patch.object(sys, "argv", ["hf", str(root)]):
                    self.assertEqual(self.hf.main(), 0)
                    upgraded = path.read_text(encoding="utf-8")
                    self.assertNotIn(self.hf.STREAMING_PREVIOUS, upgraded)
                    self.assertNotIn(self.hf.NOSTREAM_PREVIOUS, upgraded)
                    self.assertIn(self.hf.STREAMING_NEW, upgraded)
                    self.assertIn(self.hf.NOSTREAM_NEW, upgraded)
                    self.assertEqual(self.hf.main(), 0)
                    self.assertEqual(path.read_text(encoding="utf-8"), upgraded)

    def test_truncation_preserves_serializable_lists_in_both_response_paths(self):
        valid = SimpleNamespace(function=SimpleNamespace(arguments='{"ok":true}'))
        invalid = SimpleNamespace(function=SimpleNamespace(arguments='{"unfinished":'))
        no_function = SimpleNamespace(function=None)
        cases = [
            ("length", [invalid, no_function], []),
            ("length", [invalid, valid], [valid]),
            ("length", [valid], [valid]),
            ("length", [], []),
            ("stop", [invalid, valid], [invalid, valid]),
        ]
        for streaming in (True, False):
            for reason, calls, expected in cases:
                with self.subTest(streaming=streaming, reason=reason, calls=calls):
                    message = SimpleNamespace(tool_calls=list(calls))
                    ns = dict(output=SimpleNamespace(finish_reason=reason, index=0),
                        delta_message=message, message=message, tools_streamed=[True], i=0,
                        tool_choice_function_name=None, is_finish_reason_tool_calls=True,
                        logprobs=None, ChatCompletionResponseChoice=SimpleNamespace)
                    exec(self.hf.HELPER_NEW, ns)
                    code = self.hf.STREAMING_NEW if streaming else self.hf.NOSTREAM_NEW + "\n            )"
                    exec(textwrap.dedent(code), ns)
                    self.assertEqual(message.tool_calls, expected)
                    finish = ns["finish_reason_"] if streaming else ns["choice_data"].finish_reason
                    self.assertEqual(finish, "length" if reason == "length" else "tool_calls")
                    # ChatMessage and DeltaMessage serializers in the pinned vLLM
                    # both use this len(...) check before dropping empty lists.
                    data = {"tool_calls": message.tool_calls}
                    if len(data.get("tool_calls", [])) == 0:
                        data.pop("tool_calls", None)
                    encoded = json.dumps(data, default=vars)
                    self.assertNotIn('"tool_calls": null', encoded)

    def test_helper_json_ok(self):
        # We can't import the inserted helper directly; re-execute its body
        # to verify its behavior. Extract from HELPER_NEW.
        src = self.hf.HELPER_NEW
        start = src.index("def _dsml_issue55_json_ok")
        helper_src = src[start:]
        ns: dict = {}
        exec(helper_src, ns)
        ok = ns["_dsml_issue55_json_ok"]
        self.assertTrue(ok(None))
        self.assertTrue(ok(""))
        self.assertTrue(ok('{"a":1}'))
        self.assertTrue(ok('[1,2,3]'))
        self.assertFalse(ok('{"a":1'))
        self.assertFalse(ok('{"path": "/f", "content": "'))
        self.assertFalse(ok("plain text"))

    def test_missing_anchor_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vllm"
            (root / "entrypoints/openai/chat_completion").mkdir(parents=True)
            path = root / "entrypoints/openai/chat_completion/serving.py"
            # stock with the streaming anchor mutated
            src = self.stock.replace(
                'if tools_streamed[i] and not tool_choice_function_name:',
                'if tools_streamed[i] and not DO_NOT_MATCH:',
            )
            path.write_text(src, encoding="utf-8")
            import sys
            old = sys.argv
            sys.argv = ["hf", str(root)]
            try:
                rc = self.hf.main()
            finally:
                sys.argv = old
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
