#!/usr/bin/env python3
"""CPU tests for the live Responses API verifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "verify-responses-api-live.py"


def load_target():
    spec = importlib.util.spec_from_file_location("responses_live", TARGET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContinuationFailureClient:
    api_path = "/v1"

    def __init__(self, error):
        self.error = error
        self.json_calls = 0

    def json(self, method, path, body=None):
        self.json_calls += 1
        if self.json_calls == 1:
            return {
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": "RESPONSES"},
                ]}],
            }
        if self.json_calls == 2:
            return {
                "id": "resp-1",
                "output": [{"type": "function_call", "name": "ping",
                            "call_id": "call-1", "arguments": "{}"}],
            }
        if self.json_calls == 3:
            raise self.error
        raise AssertionError("unexpected JSON request")

    def request(self, method, path, body=None):
        frames = [
            ("response.output_text.delta", {
                "type": "response.output_text.delta", "delta": "RESPONSES",
            }),
            ("response.completed", {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            }),
        ]
        return 200, b"".join(
            f"event: {name}\ndata: {json.dumps(value)}\n\n".encode()
            for name, value in frames
        )


class LruStoreClient:
    api_path = "/v1"

    def __init__(self, capacity):
        self.capacity = capacity
        self.next_id = 0
        self.responses = {}

    def _get(self, response_id):
        if response_id not in self.responses:
            return 404, {}
        response = self.responses.pop(response_id)
        self.responses[response_id] = response
        return 200, response

    def json(self, method, path, body=None):
        if method == "POST":
            self.next_id += 1
            response = {"id": f"resp-{self.next_id}"}
            self.responses[response["id"]] = response
            while len(self.responses) > self.capacity:
                del self.responses[next(iter(self.responses))]
            return response
        status, response = self._get(path.rsplit("/", 1)[-1])
        if status != 200:
            raise AssertionError(f"unexpected missing response: {path}")
        return response

    def request(self, method, path, body=None):
        status, response = self._get(path.rsplit("/", 1)[-1])
        return status, json.dumps(response).encode()


class ResponsesApiLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_target()

    def test_extracts_only_output_text_parts(self):
        value = {
            "output": [
                {"type": "reasoning", "content": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "RES"},
                    {"type": "refusal", "refusal": "ignored"},
                    {"type": "output_text", "text": "PONSES"},
                ]},
            ]
        }
        self.assertEqual(self.module.responses_output_text(value), "RESPONSES")
        with self.assertRaisesRegex(ValueError, "output text missing"):
            self.module.responses_output_text({"output": []})

    def test_sse_requires_exact_terminal_completed_event_and_usage(self):
        frames = [
            ("response.created", {"type": "response.created"}),
            ("response.output_text.delta", {
                "type": "response.output_text.delta", "delta": "RESPONSES",
            }),
            ("response.completed", {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            }),
        ]
        raw = b"".join(
            f"event: {name}\ndata: {json.dumps(value)}\n\n".encode()
            for name, value in frames
        )
        self.assertEqual(self.module.parse_responses_sse(raw), "RESPONSES")
        with self.assertRaises(ValueError):
            self.module.parse_responses_sse(raw.replace(b"response.completed", b"response.evil", 1))
        with self.assertRaises(ValueError):
            self.module.parse_responses_sse(raw.rsplit(b"\n\n", 2)[0] + b"\n\n")
        with self.assertRaises(ValueError):
            self.module.parse_responses_sse(b"garbage\n\n" + raw)

    def test_function_call_requires_exact_name_and_json_arguments(self):
        value = {
            "id": "resp-1",
            "output": [{
                "type": "function_call", "name": "ping",
                "call_id": "call-1", "arguments": "{}",
            }],
        }
        self.assertEqual(
            self.module.response_function_call(value, "ping"),
            ("call-1", "resp-1"),
        )
        value["output"][0]["arguments"] = "{"
        with self.assertRaises(json.JSONDecodeError):
            self.module.response_function_call(value, "ping")

    def test_safe_prefix_floor_respects_swa_and_block_size(self):
        self.assertEqual(self.module.safe_cached_floor(21_000), 16_896)
        self.assertEqual(self.module.safe_cached_floor(4_000), 0)

    def test_metrics_parser_handles_labelled_prometheus_rows(self):
        text = """
vllm:num_requests_running{model_name="x"} 1
vllm:num_requests_waiting{model_name="x"} 0
vllm:generation_tokens_total{model_name="x"} 12
vllm:request_success_total{finished_reason="stop"} 3
vllm:request_success_total{finished_reason="abort"} 2
vllm:prefix_cache_queries_total{model_name="x"} 100
vllm:prefix_cache_hits_total{model_name="x"} 80
"""
        self.assertEqual(self.module.metric_snapshot(text), {
            "running": 1.0, "waiting": 0.0, "generation_tokens": 12.0,
            "successful_requests": 5.0, "cache_queries": 100.0,
            "cache_hits": 80.0,
        })

    def test_disconnect_counter_chain_requires_running_then_idle_without_success(self):
        before = {"running": 0.0, "waiting": 0.0, "generation_tokens": 10.0,
                  "successful_requests": 2.0, "cache_queries": 0.0, "cache_hits": 0.0}
        opened = {**before, "running": 1.0, "generation_tokens": 11.0}
        idle = {**before, "generation_tokens": 15.0}
        self.assertTrue(self.module.disconnect_counter_chain_valid(before, opened, [idle, idle]))
        completed = {**idle, "successful_requests": 3.0}
        self.assertFalse(self.module.disconnect_counter_chain_valid(before, opened, [completed, completed]))

    def test_continuation_404_names_store_remediation(self):
        original = self.module.ResponseHTTPError(
            "POST", "/v1/responses", 404, b'{"error":"missing"}')
        client = ContinuationFailureClient(original)
        with self.assertRaisesRegex(
                RuntimeError,
                "previous_response_id.*VLLM_ENABLE_RESPONSES_API_STORE=1") as caught:
            self.module.check_responses(client, "model")
        self.assertIs(caught.exception.__cause__, original)

    def test_unrelated_continuation_http_failure_is_unchanged(self):
        original = self.module.ResponseHTTPError(
            "POST", "/v1/responses", 500, b"upstream")
        client = ContinuationFailureClient(original)
        with self.assertRaises(self.module.ResponseHTTPError) as caught:
            self.module.check_responses(client, "model")
        self.assertIs(caught.exception, original)
        self.assertEqual(
            str(caught.exception),
            "POST /v1/responses returned HTTP 500: b'upstream'")
        self.assertNotIn(
            "VLLM_ENABLE_RESPONSES_API_STORE", str(caught.exception))

    def test_store_probe_verifies_touch_and_terminal_eviction(self):
        client = LruStoreClient(capacity=2)
        result = self.module.check_store_lru(client, "model", 2)
        self.assertEqual(result, {
            "capacity": 2,
            "lru_touch": True,
            "terminal_eviction": True,
        })
        self.assertEqual(list(client.responses), ["resp-1", "resp-3"])


if __name__ == "__main__":
    unittest.main()
