#!/usr/bin/env python3
"""Parser regressions for issue #80's injected mixed-prefill cap constant."""
from __future__ import annotations

import ast
import os
import pathlib
import unittest
from unittest import mock

from tests.sim.test_issue43_scheduler_sim import Req, step


ROOT = pathlib.Path(__file__).resolve().parent.parent
HOTFIX = ROOT / "patches/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
ENV_NAME = "DSPARK_MIXED_PREFILL_TOKEN_CAP"
CONSTANT = "_ISSUE80_MIXED_PREFILL_TOKEN_CAP"


def _injected_constants_source() -> str:
    """Evaluate only the string expression assigned to A1_NEW in the patcher."""
    tree = ast.parse(HOTFIX.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "A1_NEW"
                for target in node.targets)
    )

    def evaluate(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id == "A1_OLD":
            return "logger = init_logger(__name__)\n"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise AssertionError(f"unexpected A1_NEW expression: {ast.dump(node)}")

    generated = evaluate(assignment.value)
    # The logger anchor is pre-existing target code and has unrelated imports.
    return generated.split("\n", 1)[1]


def _load_cap(value: str | None = None) -> int:
    env = {} if value is None else {ENV_NAME: value}
    with mock.patch.dict(os.environ, env, clear=True):
        namespace: dict[str, object] = {"os": os}
        exec(compile(_injected_constants_source(), "<issue80-cap-constants>", "exec"),
             namespace)
        cap_value = namespace[CONSTANT]
        assert isinstance(cap_value, int)
        return cap_value


class MixedPrefillCapParserTests(unittest.TestCase):
    def test_default_is_256_and_zero_disables(self):
        self.assertEqual(_load_cap(), 256)
        self.assertEqual(_load_cap("0"), 0)

    def test_valid_upper_boundary_is_accepted(self):
        self.assertEqual(_load_cap("8192"), 8192)

    def test_invalid_values_fail_closed(self):
        for value in ("-1", "8193", "not-an-int", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _load_cap(value)

    def test_zero_rollback_disables_mixed_cap(self):
        prefill = Req("prefill", prompt_tokens=4096, max_tokens=128)
        prefill.is_prefill_chunk = True
        decode = Req("decode", prompt_tokens=32, max_tokens=128,
                     sampled_tokens_per_step=6)
        decode.num_computed_tokens = decode.num_prompt_tokens

        scheduled, _ = step(
            [prefill, decode],
            token_budget=8192,
            current_step=7,
            long_threshold=1024,
            num_sampled_tokens_per_step=6,
            mixed_prefill_cap=_load_cap("0"),
        )

        self.assertEqual(
            {request.request_id: tokens for request, tokens in scheduled},
            {"prefill": 1024, "decode": 6},
        )


if __name__ == "__main__":
    unittest.main()
