#!/usr/bin/env python3
"""CPU regression tests for the assistant-final continuation hotfix (issue #52).

These are CPU-only gates: they verify the patcher transforms a faithful
minimal stock encoder shape (trailing-assistant generation gains exactly one
header; the same shape without a generation prompt and every unrelated shape
render byte-identically),
that apply is idempotent, that a gated-ON invocation fails closed on missing
target / missing anchor / failed self-check and restores the original bytes,
and that the compose/.env/start wiring invokes the patch only when
DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX is exactly 1 with `|| exit 1`.

Live render/rescue behavior against a real checkpoint encoder is verified
separately on a running serve; this gate guards recipe integrity.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-assistant-final-continuation.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"

# Minimal but faithful control flow and call signature from the stock checkpoint
# encoder (encoding_dsv4.py, installed as deepseek_v4_encoding.py). The role is
# rendered first; then the separate transition branch appends a generation
# header for user/developer. A trailing assistant therefore closes with EOS and
# falls past that branch. The hotfix widens only the transition condition.
STOCK_ENCODER = '''\
eos_token = "<|end of sentence|>"
user_sp_token = "<|User|>"
assistant_sp_token = "<|Assistant|>"
thinking_start_token = "<think>"


def render_message(index, messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
    role = messages[index].get("role")
    content = messages[index].get("content", "")
    if role == "system":
        out = [content]
    elif role == "tool":
        out = [content]
    elif role in ["user", "developer"]:
        out = []
        if role == "user":
            out.append(user_sp_token)
        out.append(content)
    elif role == "assistant":
        out = [content, eos_token]
    else:
        raise NotImplementedError(role)

    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return out

    task = messages[index].get("task")
    if task is not None:
        out.append("<task>")
    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
        out.append(assistant_sp_token)
        if thinking_mode == "thinking":
            out.append(thinking_start_token)
    return out


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None):
    tokens = []
    for index in range(len(messages)):
        tokens.extend(render_message(
            index, messages, thinking_mode, drop_thinking, reasoning_effort
        ))
    return tokens
'''

TRAILING_ASSISTANT = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
    {"role": "assistant", "content": "A finished answer."},
]

# Shapes whose final message is NOT assistant: the hotfix must not touch
# their rendering at all.
UNRELATED_SHAPES = [
    [],
    [{"role": "system", "content": "s"}],
    [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "developer", "content": "d"},
    ],
    # assistant mid-transcript, including consecutive assistant turns
    [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ],
    [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
    ],
    # trailing non-user roles
    [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "t"},
    ],
    [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
    ],
]


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("hotfix_assistant_final", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module_from(path: Path):
    spec = importlib.util.spec_from_file_location(f"enc_{id(path)}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssistantFinalHotfixTest(unittest.TestCase):
    def setUp(self):
        self.hf = _load_hotfix()

    def _write(self, tmp: str, text: str) -> Path:
        path = Path(tmp) / "deepseek_v4_encoding.py"
        path.write_text(text, encoding="utf-8")
        return path

    def _run(self, path: Path) -> int:
        return self.hf.main(["hotfix", str(path)])

    def test_anchors_present_in_stock(self):
        self.assertIn(self.hf.OLD, STOCK_ENCODER)

    def test_trailing_assistant_gains_exactly_one_generation_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, STOCK_ENCODER)
            self.assertEqual(self._run(path), 0)
            patched = _load_module_from(path)
            stock = _exec_dict(STOCK_ENCODER)
            out = patched.encode_messages(
                TRAILING_ASSISTANT, "thinking", reasoning_effort="high"
            )
            stock_out = stock["encode_messages"](
                TRAILING_ASSISTANT, "thinking", reasoning_effort="high"
            )
            # Stock dead state: trailing assistant closes with EOS, no fresh
            # header. Patched rendering preserves those stock bytes and appends
            # exactly one generation header (assistant speaker + thinking token).
            self.assertEqual(stock_out[-1], "<|end of sentence|>")
            self.assertEqual(
                out,
                stock_out + ["<|Assistant|>", "<think>"],
            )
            self.assertEqual(len(out), len(stock_out) + 2)


    def test_unrelated_shapes_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, STOCK_ENCODER)
            self.assertEqual(self._run(path), 0)
            patched = _load_module_from(path)
            stock = _exec_dict(STOCK_ENCODER)
            for messages in UNRELATED_SHAPES:
                self.assertEqual(
                    patched.encode_messages(messages, "thinking"),
                    stock["encode_messages"](messages, "thinking"),
                    f"render changed for {messages}",
                )

    def test_idempotent_and_already_patched_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, STOCK_ENCODER)
            self.assertEqual(self._run(path), 0)
            once = path.read_text(encoding="utf-8")
            self.assertEqual(once.count(self.hf.MARK), 1)
            # Second run: already-patched state must still validate, rc 0,
            # no further rewrite.
            self.assertEqual(self._run(path), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), once)

    def test_missing_target_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist.py"
            self.assertEqual(self._run(missing), 1)

    def test_missing_anchor_fails_without_writing(self):
        drifted = STOCK_ENCODER.replace(
            'elif messages[index].get("role") in ["user", "developer"]:',
            'elif messages[index].get("role") in ["user", "dev"]:',
        )
        self.assertNotIn(self.hf.OLD, drifted)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, drifted)
            self.assertEqual(self._run(path), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), drifted)

    def test_failed_self_check_restores_original_bytes(self):
        # The anchor is present and the patch applies, but the branch body
        # suppresses the header for assistant turns, so the post-write
        # self-check must fail and the original bytes must be restored.
        sabotaged = STOCK_ENCODER.replace(
            "        out.append(assistant_sp_token)\n"
            "        if thinking_mode == \"thinking\":\n"
            "            out.append(thinking_start_token)\n",
            "        if messages[index].get(\"role\") != \"assistant\":\n"
            "            out.append(assistant_sp_token)\n"
            "            if thinking_mode == \"thinking\":\n"
            "                out.append(thinking_start_token)\n",
        )
        self.assertIn(self.hf.OLD, sabotaged)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, sabotaged)
            self.assertEqual(self._run(path), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), sabotaged)

    def test_self_check_exception_restores_original_bytes(self):
        broken = STOCK_ENCODER.replace(
            "    return tokens\n", "    raise RuntimeError('encoder exploded')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, broken)
            self.assertEqual(self._run(path), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), broken)


def _exec_dict(source: str) -> dict:
    ns: dict = {}
    exec(compile(source, "stock_encoder", "exec"), ns)
    return ns


class ComposeWiringTest(unittest.TestCase):
    """Static OFF-default wiring: the patch ships to the worker but is
    invoked only when DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX is exactly 1,
    fail-closed via `|| exit 1`."""

    def setUp(self):
        self.compose = COMPOSE.read_text(encoding="utf-8")
        self.env_example = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.start = START.read_text(encoding="utf-8")

    def test_env_passthrough_defaults_off(self):
        self.assertIn(
            'DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX: "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}"',
            self.compose,
        )

    def test_entrypoint_invocation_gated_and_fail_closed(self):
        gated = (
            'if [ "$${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1" ]; then '
            "python3 /opt/hotfix-dsv4-assistant-final-continuation.py || exit 1; fi;"
        )
        self.assertIn(gated, self.compose)
        # No ungated invocation anywhere in compose.
        for line in self.compose.splitlines():
            if "python3 /opt/hotfix-dsv4-assistant-final-continuation.py" in line:
                self.assertIn('DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1"', line)
                self.assertIn("|| exit 1", line)

    def test_patch_mounted_read_only_in_worker_image(self):
        self.assertIn(
            "${DSPARK_ASSISTANT_FINAL_HOTFIX:-./patches/hotfix-dsv4-assistant-final-continuation.py}"
            ":/opt/hotfix-dsv4-assistant-final-continuation.py:ro",
            self.compose,
        )

    def test_env_example_documents_default_off(self):
        self.assertIn("DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX=0", self.env_example)

    def test_start_script_keeps_path_override_and_worker_sync(self):
        # Path override for the patch file itself is kept, and the start
        # script still syncs it to the worker (available, never executed by
        # merely syncing).
        self.assertIn(
            'DSPARK_ASSISTANT_FINAL_HOTFIX="${DSPARK_ASSISTANT_FINAL_HOTFIX:-'
            '$SCRIPT_DIR/patches/hotfix-dsv4-assistant-final-continuation.py}"',
            self.start,
        )
        self.assertIn(
            'scp "$DSPARK_ASSISTANT_FINAL_HOTFIX" "${WORKER_HOST}:'
            '${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-assistant-final-continuation.py"',
            self.start,
        )


if __name__ == "__main__":
    unittest.main()
