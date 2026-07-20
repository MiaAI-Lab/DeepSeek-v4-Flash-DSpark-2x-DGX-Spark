#!/usr/bin/env python3
"""Scaffold tests for the DSpark benchmark harness.

These tests use fake HTTP/SSE and synthetic Prometheus data.  They are not
runtime proof of a live server, GPU, telemetry device, or Docker container.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_dspark as harness  # noqa: E402
import long_context_gate as long_gate  # noqa: E402


class FakeResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


class BenchmarkHarnessScaffoldTests(unittest.TestCase):
    """Synthetic contract coverage, explicitly not live runtime evidence."""

    def test_fake_openai_sse_records_include_usage_ttft_and_itl(self) -> None:
        captured: list[dict[str, object]] = []

        def opener(request, timeout):
            captured.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n',
                    b"data: [DONE]\n",
                ]
            )

        result = harness.stream_chat(
            "http://127.0.0.1:8888",
            "demo-model",
            "say hello",
            max_tokens=8,
            include_usage=True,
            opener=opener,
            clock=IncrementingClock(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["prompt_tokens"], 7)
        self.assertEqual(result["token_count_source"], "usage")
        self.assertEqual(result["response_text"], "hello world")
        self.assertIsNotNone(result["ttft_seconds"])
        self.assertEqual(len(result["itl_seconds"]), 1)
        self.assertEqual(result["tpot_seconds"], result["itl_seconds"][0])
        self.assertEqual(captured[0]["stream"], True)
        self.assertEqual(captured[0]["stream_options"], {"include_usage": True})

    def test_missing_usage_is_an_explicit_error_with_labelled_fallback(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"token"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )

        result = harness.stream_chat(
            "http://127.0.0.1:8888",
            "demo-model",
            "prompt",
            opener=opener,
            clock=IncrementingClock(),
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["token_count_source"], "nonempty_chunks_fallback")
        self.assertTrue(any("no usage event" in error for error in result["errors"]))

    def test_vllm_reasoning_delta_is_measured_as_stream_output(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"133"}}]}\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}\n',
                    b"data: [DONE]\n",
                ]
            )

        result = harness.stream_chat(
            "http://127.0.0.1:8888",
            "demo-model",
            "compute",
            opener=opener,
            clock=IncrementingClock(),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["response_text"], "think133")
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(len(result["itl_seconds"]), 1)

    def test_incomplete_usage_is_an_explicit_error(self) -> None:
        def opener(request, timeout):
            return FakeResponse(
                [
                    b'data: {"choices":[{"delta":{"content":"token"}}]}\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":7}}\n',
                    b"data: [DONE]\n",
                ]
            )

        result = harness.stream_chat(
            "http://127.0.0.1:8888",
            "demo-model",
            "prompt",
            opener=opener,
            clock=IncrementingClock(),
        )
        self.assertFalse(result["ok"])
        self.assertTrue(any("did not include integer" in error for error in result["errors"]))

    def test_usage_token_counts_reject_floats(self) -> None:
        with self.assertRaises(harness.BenchmarkError):
            harness._usage_int({"completion_tokens": 2.5}, "completion_tokens")

    def test_synthetic_prometheus_deltas_ignore_label_order_and_other_models(self) -> None:
        before_text = """
# HELP vllm:prompt_tokens_total prompt counter
vllm:prompt_tokens_total{request_id="a",model_name="demo-model"} 100
vllm:generation_tokens_total{model_name="demo-model",request_id="a"} 40
vllm:spec_decode_num_draft_tokens_total{request_id="a",model_name="demo-model"} 12
vllm:spec_decode_num_accepted_tokens_total{model_name="demo-model",request_id="a"} 9
vllm:prompt_tokens_total{model_name="other-model",request_id="b"} 999
"""
        after_text = """
vllm:prompt_tokens_total{model_name="demo-model",request_id="a"} 105
vllm:generation_tokens_total{request_id="a",model_name="demo-model"} 43
vllm:spec_decode_num_draft_tokens_total{model_name="demo-model",request_id="a"} 14
vllm:spec_decode_num_accepted_tokens_total{request_id="a",model_name="demo-model"} 10
"""

        before = harness.prometheus_counter_snapshot(before_text, "demo-model")
        after = harness.prometheus_counter_snapshot(after_text, "demo-model")
        deltas, errors = harness.counter_deltas(before, after)

        self.assertEqual(errors, [])
        self.assertEqual(deltas["prompt_tokens"], 5)
        self.assertEqual(deltas["generation_tokens"], 3)
        self.assertEqual(deltas["speculative_drafted_tokens"], 2)
        self.assertEqual(deltas["speculative_accepted_tokens"], 1)

    def test_exact_container_name_is_the_only_log_target(self) -> None:
        completed = subprocess.CompletedProcess(
            ["docker"],
            0,
            stdout="GPU KV cache size: 10,000 tokens\n",
            stderr="",
        )
        with patch.object(harness.subprocess, "run", return_value=completed) as run:
            facts = harness.container_log_facts("exact-container")

        self.assertEqual(facts["kv_tokens"], "10,000")
        self.assertEqual(
            run.call_args.args[0],
            ["docker", "logs", "--tail", "2000", "exact-container"],
        )

    def test_default_cli_has_supported_levels_and_three_repeats(self) -> None:
        args = harness.build_parser().parse_args(
            ["--model", "demo-model", "--container-name", "run-owned-container"]
        )
        self.assertEqual(args.concurrency, list(harness.CONCURRENCY_LEVELS))
        self.assertIn(2, args.concurrency)
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.warmup, 1)
        self.assertTrue(args.include_usage)

    def test_synthetic_prompt_and_unique_prefix_cli_contract(self) -> None:
        args = harness.build_parser().parse_args(
            [
                "--model",
                "demo-model",
                "--container-name",
                "run-owned-container",
                "--synthetic-prompt-words",
                "17",
                "--unique-prefix-per-repeat",
            ]
        )
        harness.validate_args(args)
        self.assertEqual(len(args.prompt.split()), 17)
        self.assertTrue(args.unique_prefix_per_repeat)

    def test_synthetic_prompt_rejects_prompt_file_combination(self) -> None:
        args = harness.build_parser().parse_args(
            [
                "--model",
                "demo-model",
                "--container-name",
                "run-owned-container",
                "--prompt-file",
                "prompt.txt",
                "--synthetic-prompt-words",
                "17",
            ]
        )
        with self.assertRaisesRegex(harness.BenchmarkError, "mutually exclusive"):
            harness.validate_args(args)

    def test_long_context_prompt_has_nonce_needle_and_requested_filler(self) -> None:
        prompt = long_gate.build_prompt(17, "fresh-run", "NEEDLE-123")
        self.assertIn("Retrieval run nonce: fresh-run", prompt)
        self.assertIn("secret retrieval code is NEEDLE-123", prompt)
        filler = prompt.split("Unrelated filler begins: ", 1)[1].split("\n", 1)[0]
        self.assertEqual(len(filler.split()), 17)

    def test_long_context_response_accepts_vllm_reasoning_field(self) -> None:
        text, prompt_tokens = long_gate.response_text_and_prompt_tokens(
            {
                "choices": [{"message": {"content": None, "reasoning": "NEEDLE-123"}}],
                "usage": {"prompt_tokens": 101305},
            }
        )
        self.assertEqual(text, "NEEDLE-123")
        self.assertEqual(prompt_tokens, 101305)

    def test_model_catalog_requires_requested_model(self) -> None:
        result = harness._model_catalog({"data": [{"id": "other-model"}]}, "demo-model")
        self.assertIn("error", result)

    def test_telemetry_summary_exposes_host_memory_and_swap(self) -> None:
        summary = harness.telemetry_summary(
            [
                {
                    "power_watts": 100.0,
                    "temperature_c": 55.0,
                    "utilization_gpu_percent": 80.0,
                    "host_mem_available_bytes": 4096.0,
                    "host_swap_total_bytes": 2048.0,
                    "host_swap_free_bytes": 1024.0,
                    "host_swap_used_bytes": 1024.0,
                }
            ]
        )
        self.assertEqual(summary["host_mem_available_bytes"]["p50"], 4096.0)
        self.assertEqual(summary["host_swap_used_bytes"]["mean"], 1024.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
