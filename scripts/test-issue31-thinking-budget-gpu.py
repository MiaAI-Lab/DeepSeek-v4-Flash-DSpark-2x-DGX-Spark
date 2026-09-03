#!/usr/bin/env python3
"""CPU guards for the GPU-resident V2 thinking-budget hotfix."""
from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue31-v2-thinking-budget-gpu.py"


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("issue31_gpu_hotfix", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThinkingBudgetGpuHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hotfix = _load_hotfix()
        cls.source = cls.hotfix.THINKING_BUDGET_PY
        cls.tree = ast.parse(cls.source)

    def test_embedded_module_compiles(self) -> None:
        compile(self.source, "thinking_budget_gpu.py", "exec")

    def test_decode_path_has_no_device_to_host_conversion(self) -> None:
        forbidden = {"cpu", "tolist", "numpy", "detach", "item"}
        attributes = {
            node.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(attributes), attributes & forbidden)
        self.assertNotIn("all_token_ids", self.source)

    def test_unbudgeted_requests_have_a_cpu_only_fast_path(self) -> None:
        self.assertIn(
            "not np.any(self.use_budget[idx_mapping_np])", self.source
        )
        self.assertIn("self.use_budget[req_idx] = False", self.source)
        self.assertNotIn("DEFAULT_THINKING_TOKEN_BUDGET", self.source)

    def test_forces_only_the_exact_boundary(self) -> None:
        self.assertIn("generated_index = pos + 1 - prompt_len", self.source)
        self.assertIn("if generated_index != budget:", self.source)
        self.assertIn("tl.debug_barrier()", self.source)
        self.assertIn("1.0e9", self.source)

    def test_triton_loops_do_not_return_from_inside_the_loop(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.For, ast.While)):
                self.assertFalse(
                    any(isinstance(child, ast.Return) for child in ast.walk(node))
                )

    def test_observes_only_accepted_speculative_tokens(self) -> None:
        self.assertIn("num_sampled = tl.load(num_sampled_ptr", self.source)
        self.assertIn("for i in range(num_sampled):", self.source)
        self.assertIn("tl.store(active_ptr + req_idx, 0)", self.source)
        self.assertIn("sampler_output.num_sampled", HOTFIX.read_text())

    def test_only_generated_suffix_is_checked_on_resume(self) -> None:
        self.assertIn(
            "generated_suffix = prefill_token_ids[int(prompt_len) :]", self.source
        )
        self.assertNotIn("prefill_token_ids.index", self.source)


class ComposeWiringTest(unittest.TestCase):
    """Static OFF-default wiring: the patch ships to the worker but is
    invoked only when DSPARK_ENABLE_ISSUE31_GPU_HOTFIX is exactly 1,
    fail-closed via `|| exit 1`."""

    def setUp(self):
        self.compose = (ROOT / "docker-compose.dspark.yml").read_text(encoding="utf-8")
        self.env_example = (ROOT / ".env.dspark.example").read_text(encoding="utf-8")
        self.start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text(
            encoding="utf-8"
        )

    def test_env_passthrough_defaults_off(self):
        self.assertIn(
            'DSPARK_ENABLE_ISSUE31_GPU_HOTFIX: "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}"',
            self.compose,
        )

    def test_entrypoint_invocation_gated_and_fail_closed(self):
        gated = (
            'if [ "$${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then '
            "python3 /opt/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py || exit 1; fi;"
        )
        self.assertIn(gated, self.compose)
        for line in self.compose.splitlines():
            if "python3 /opt/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py" in line:
                self.assertIn('DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1"', line)
                self.assertIn("|| exit 1", line)

    def test_env_example_documents_default_off(self):
        self.assertIn("DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=0", self.env_example)

    def test_start_smoke_omits_budget_unless_flag_on(self):
        self.assertIn(
            'if [ "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then',
            self.start,
        )
        self.assertIn(
            "Running minimal OpenAI-compatible chat request (stock V2; no thinking_token_budget)...",
            self.start,
        )


if __name__ == "__main__":
    unittest.main()
