#!/usr/bin/env python3
"""The LMCache flag in the head shell is copied into every worker compose."""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ASSIGN = "REMOTE_LMCACHE=\"$(printf '%q' \"${DSPARK_ENABLE_LMCACHE:-0}\")\""
TOKEN = "DSPARK_ENABLE_LMCACHE=$REMOTE_LMCACHE"


def _bash() -> str:
    return os.environ.get("DSPARK_TEST_BASH", "bash")


def extract_fn(name: str) -> str:
    text = START.read_text()
    marker = f"{name}() {{"
    start = text.index(marker)
    lines = text[start:].splitlines(keepends=True)
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if i > 0 and line == "}\n":
            return "".join(out)
    raise AssertionError(f"{name} is not closed")


class LmcacheRemoteEnvTest(unittest.TestCase):
    def test_assignment_is_printf_q_and_both_functions_carry_it_once(self):
        text = START.read_text()
        self.assertEqual(text.count(ASSIGN), 1)
        self.assertLess(text.index(ASSIGN), text.index("remote_compose() {"))
        for name, prefix in (
            ("remote_compose", "$REMOTE_COMPOSE "),
            ("remote_compose2", "$REMOTE_COMPOSE2 "),
        ):
            body = extract_fn(name)
            self.assertEqual(body.count(TOKEN), 1)
            self.assertIn(prefix + TOKEN + " ", body)

    def _remote_lines(self, value: str | None) -> tuple[str, str]:
        script = "\n".join(
            [
                "set -u",
                "dssh() { printf '%s\\n' \"$2\"; }",
                "remote_nccl_env() { :; }",
                "remote_nccl_env2() { :; }",
                "WORKER_HOST=worker1",
                "WORKER2_HOST=worker2",
                "REMOTE_COMPOSE='cd /w && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1'",
                "REMOTE_COMPOSE2='cd /w2 && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1'",
                "REMOTE_C128A_PREFILL_CACHE=0",
                "REMOTE_ISSUE136_ENABLE=0",
                "REMOTE_ISSUE191_ENABLE=0",
                "REMOTE_ISSUE191_RETRIES=2",
                "REMOTE_ISSUE191_MODE=failclosed",
                "REMOTE_ISSUE191_THINKOFF=1",
                "REMOTE_ASYNC_SCHEDULING=1",
                "REMOTE_DSPARK_BLOCK_K=0",
                "REMOTE_ROPE_SWA_FIX=0",
                "REMOTE_DSPARK_SWA_PREFIX=0",
                "REMOTE_DSML_RECOVERY=0",
                "REMOTE_MXFP4_INDEXER=0",
                "REMOTE_ISSUE144_EFFORT_ALIGN=0",
                "TP_SIZE=2",
                "NNODES=2",
                ASSIGN,
                extract_fn("remote_compose"),
                extract_fn("remote_compose2"),
                'remote_compose "docker compose up -d"',
                "printf '\\n---\\n'",
                'remote_compose2 "docker compose up -d"',
            ]
        )
        env = os.environ.copy()
        env.pop("DSPARK_ENABLE_LMCACHE", None)
        if value is not None:
            env["DSPARK_ENABLE_LMCACHE"] = value
        proc = subprocess.run(
            [_bash(), "-c", script],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        worker, worker2 = proc.stdout.split("\n---\n", 1)
        return worker, worker2

    def test_unset_empty_and_zero_forward_zero(self):
        for value in (None, "", "0"):
            for line in self._remote_lines(value):
                self.assertIn(
                    "COMPOSE_DISABLE_ENV_FILE=1 DSPARK_ENABLE_LMCACHE=0 ",
                    line,
                )
                self.assertIn(" docker compose up -d", line)
                self.assertNotIn("DSPARK_ENABLE_LMCACHE=1", line)

    def test_one_is_forwarded_on_both_ranks_before_compose(self):
        for line in self._remote_lines("1"):
            self.assertIn(
                "COMPOSE_DISABLE_ENV_FILE=1 DSPARK_ENABLE_LMCACHE=1 ",
                line,
            )
            self.assertLess(
                line.index("DSPARK_ENABLE_LMCACHE=1"),
                line.index("docker compose up -d"),
            )

    def test_metachar_value_stays_one_env_word(self):
        for line in self._remote_lines("1; id"):
            probe = subprocess.run(
                [
                    _bash(),
                    "-c",
                    "id() { echo RAN_ID; }\n"
                    "env() { printf '%s\\n' \"$@\"; }\n"
                    "cd() { :; }\n"
                    'eval "$1"\n',
                    "bash",
                    line,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr + line)
            self.assertNotIn("RAN_ID", probe.stdout)
            self.assertIn("DSPARK_ENABLE_LMCACHE=1; id", probe.stdout)


if __name__ == "__main__":
    unittest.main()
