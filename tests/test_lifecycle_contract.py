import contextlib
import http.client
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def load_script(name):
    path = SCRIPTS / name
    module_name = name.removesuffix(".py").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_semantic_smoke():
    return load_script("smoke-openai-compat.py")


class Upstream(http.server.BaseHTTPRequestHandler):
    seen_authorization = None

    def do_GET(self):
        type(self).seen_authorization = self.headers.get("Authorization")
        body = json.dumps({"data": [{"id": "deepseek-v4-flash-0731"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class RecordingUpstream(http.server.BaseHTTPRequestHandler):
    requests = []
    first_stream_chunk_sent = threading.Event()
    release_stream = threading.Event()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).requests.append((self.path, body))
        if self.headers.get("X-Stream-Test") == "1":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            type(self).first_stream_chunk_sent.set()
            type(self).release_stream.wait(timeout=3)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextlib.contextmanager
def running_origin_proxy(*, max_body_bytes=32 * 1024 * 1024):
    with tempfile.TemporaryDirectory() as tmp:
        key_file = Path(tmp) / "origin.key"
        key_file.write_text("unit-secret\n")
        key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        upstream_port = free_port()
        proxy_port = free_port()
        RecordingUpstream.requests = []
        RecordingUpstream.first_stream_chunk_sent = threading.Event()
        RecordingUpstream.release_stream = threading.Event()
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", upstream_port), RecordingUpstream
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxy = subprocess.Popen(
            [
                "python3", str(SCRIPTS / "origin-auth-proxy.py"),
                "--listen-host", "127.0.0.1", "--listen-port", str(proxy_port),
                "--upstream-host", "127.0.0.1", "--upstream-port", str(upstream_port),
                "--key-file", str(key_file), "--allow-cidr", "127.0.0.1/32",
                "--max-body-bytes", str(max_body_bytes),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.02)
            else:
                raise RuntimeError("origin proxy did not start")
            yield proxy_port
        finally:
            RecordingUpstream.release_stream.set()
            proxy.terminate()
            proxy.wait(timeout=5)
            server.shutdown()
            server.server_close()


def proxy_post(port, target, body, *, content_length=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    headers = {
        "Authorization": "Bearer unit-secret",
        "Content-Type": "application/json",
        "Content-Length": str(len(body) if content_length is None else content_length),
    }
    connection.request("POST", target, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read()
    status = response.status
    connection.close()
    return status, payload


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LifecycleContractTest(unittest.TestCase):
    def test_runtime_hotfixes_are_fail_closed_before_vllm_import(self):
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        copy_at = compose.index('cp "$${ENCODING_SOURCE}"')
        issue21_at = compose.index(
            "python3 /opt/dspark-hotfixes/hotfix-encoding-dsv4-issue21.py"
        )
        serve_at = compose.index("exec /usr/local/bin/vllm serve")
        self.assertLess(copy_at, issue21_at)
        self.assertLess(issue21_at, serve_at)
        self.assertNotIn("hotfix-encoding-dsv4-issue21.py:ro", compose)
        self.assertNotIn("DSPARK_SKIP_HOTFIX", start)
        self.assertNotIn("restart vllm-dspark", start)
        self.assertNotIn("hotfix-nvfp4-ds-mla-issue22.sh", start)

    def test_origin_proxy_authenticates_from_file_and_strips_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "origin.key"
            key_file.write_text("unit-secret\n")
            key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
            upstream_port = free_port()
            proxy_port = free_port()
            server = http.server.ThreadingHTTPServer(("127.0.0.1", upstream_port), Upstream)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            proxy = subprocess.Popen(
                [
                    "python3",
                    str(SCRIPTS / "origin-auth-proxy.py"),
                    "--listen-host", "127.0.0.1",
                    "--listen-port", str(proxy_port),
                    "--upstream-host", "127.0.0.1",
                    "--upstream-port", str(upstream_port),
                    "--key-file", str(key_file),
                    "--allow-cidr", "127.0.0.1/32",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                url = f"http://127.0.0.1:{proxy_port}/v1/models"
                for _ in range(50):
                    try:
                        urllib.request.urlopen(url, timeout=0.1)
                    except urllib.error.HTTPError as ready:
                        ready.close()
                        break
                    except OSError:
                        time.sleep(0.02)
                with self.assertRaises(urllib.error.HTTPError) as missing:
                    urllib.request.urlopen(url, timeout=2)
                self.assertEqual(missing.exception.code, 401)
                missing.exception.close()
                wrong = urllib.request.Request(url, headers={"Authorization": "Bearer wrong"})
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    urllib.request.urlopen(wrong, timeout=2)
                self.assertEqual(denied.exception.code, 401)
                denied.exception.close()
                valid = urllib.request.Request(
                    url, headers={"Authorization": "Bearer unit-secret"}
                )
                with urllib.request.urlopen(valid, timeout=2) as response:
                    payload = json.load(response)
                self.assertEqual(payload["data"][0]["id"], "deepseek-v4-flash-0731")
                self.assertIsNone(Upstream.seen_authorization)
                proxy_source = (SCRIPTS / "origin-auth-proxy.py").read_text()
                self.assertIn("self.client_address[0]", proxy_source)
                self.assertIn("allow_network", proxy_source)
                self.assertIn("response.read1(65536)", proxy_source)
                self.assertNotIn("response.read(65536)", proxy_source)
            finally:
                proxy.terminate()
                proxy.wait(timeout=5)
                server.shutdown()
                server.server_close()

    def test_origin_proxy_validates_only_canonical_chat_target(self):
        valid = json.dumps({
            "model": "deepseek-v4-flash-0731",
            "messages": [{"role": "user", "content": "OK"}],
            "temperature": 0,
        }).encode()
        with running_origin_proxy() as port:
            status, echoed = proxy_post(port, "/v1/chat/completions", valid)
            self.assertEqual(status, 200)
            self.assertEqual(echoed, valid)
            self.assertEqual(RecordingUpstream.requests, [("/v1/chat/completions", valid)])

            invalid = json.dumps({
                "model": "deepseek-v4-flash-0731",
                "messages": [],
                "invented_top_level_field": True,
            }).encode()
            status, payload = proxy_post(port, "/v1/chat/completions", invalid)
            self.assertEqual(status, 400)
            self.assertIn(b"unsupported chat completion field", payload)
            self.assertEqual(len(RecordingUpstream.requests), 1)

            remote_media = json.dumps({
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": [{
                    "type": "image_url",
                    "image_url": {"url": "http://127.0.0.1/private"},
                }]}],
            }).encode()
            status, payload = proxy_post(port, "/v1/chat/completions", remote_media)
            self.assertEqual(status, 400)
            self.assertIn(b"must contain text only", payload)
            self.assertEqual(len(RecordingUpstream.requests), 1)

            text_parts = json.dumps({
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "OK"},
                ]}],
            }).encode()
            status, echoed = proxy_post(port, "/v1/chat/completions", text_parts)
            self.assertEqual(status, 200)
            self.assertEqual(echoed, text_parts)

            for malformed in (b"{", b"[]"):
                with self.subTest(body=malformed):
                    status, _ = proxy_post(port, "/v1/chat/completions", malformed)
                    self.assertEqual(status, 400)
                    self.assertEqual(len(RecordingUpstream.requests), 2)

            non_chat = b'{"invented_top_level_field":true}'
            status, echoed = proxy_post(port, "/v1/embeddings", non_chat)
            self.assertEqual(status, 200)
            self.assertEqual(echoed, non_chat)
            self.assertEqual(RecordingUpstream.requests[-1], ("/v1/embeddings", non_chat))

    def test_origin_proxy_rejects_noncanonical_chat_request_targets(self):
        body = b'{"model":"m","messages":[]}'
        targets = (
            "/v1/chat/completions?stream=false",
            "/v1/chat/completions/",
            "/v1//chat/completions",
            "//v1/chat/completions",
            "/v1/%63hat/completions",
            "/v1/chat%2fcompletions",
            "/v1/chat%252fcompletions",
            "http://example.invalid/v1/chat/completions",
            "//[broken/v1/chat/completions",
        )
        with running_origin_proxy() as port:
            for target in targets:
                with self.subTest(target=target):
                    status, payload = proxy_post(port, target, body)
                    self.assertEqual(status, 400)
                    self.assertIn(b"noncanonical request target", payload)
            self.assertEqual(RecordingUpstream.requests, [])

    def test_origin_proxy_preserves_body_limit_and_streaming_incrementality(self):
        with running_origin_proxy(max_body_bytes=16) as port:
            status, _ = proxy_post(port, "/v1/chat/completions", b"{}", content_length=17)
            self.assertEqual(status, 413)
            self.assertEqual(RecordingUpstream.requests, [])

        # Streaming remains an incremental proxy property after canonical chat
        # validation. The upstream waits for the client to observe its first
        # event before it sends the terminal event, so buffering would deadlock.
        body = b'{"model":"m","messages":[]}'
        with running_origin_proxy() as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request(
                "POST", "/v1/chat/completions", body=body,
                headers={
                    "Authorization": "Bearer unit-secret",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "X-Stream-Test": "1",
                },
            )
            response = connection.getresponse()
            first_line = []

            def read_first_line():
                first_line.append(response.readline())

            reader = threading.Thread(target=read_first_line, daemon=True)
            reader.start()
            self.assertTrue(RecordingUpstream.first_stream_chunk_sent.wait(timeout=1))
            reader.join(timeout=0.75)
            self.assertFalse(reader.is_alive(), "proxy buffered the first stream event")
            self.assertEqual(first_line, [b"data: first\n"])
            RecordingUpstream.release_stream.set()
            response.read()
            connection.close()

    def test_chat_completion_allowlist_contract_is_checked_in_and_reproducible(self):
        contract_path = SCRIPTS / "chat-completion-request-fields.json"
        contract = json.loads(contract_path.read_text())
        self.assertIn("@sha256:", contract["pinned_runtime_image"])
        self.assertEqual(contract["repository_extensions"], [])
        self.assertEqual(
            contract["allowed_fields"],
            sorted(set(contract["schema_fields"]) | set(contract["repository_extensions"])),
        )
        for required in (
            "model", "messages", "max_tokens", "max_completion_tokens", "temperature",
            "top_p", "seed", "stream", "stream_options", "tools", "tool_choice",
            "reasoning_effort", "chat_template_kwargs",
        ):
            self.assertIn(required, contract["allowed_fields"])
        verifier = SCRIPTS / "verify-chat-completion-request-fields.py"
        result = subprocess.run(
            ["python3", str(verifier), "--check"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = verifier.read_text()
        self.assertIn("ChatCompletionRequest.model_fields", source)
        self.assertIn("--check-container", source)
        self.assertNotIn("pydantic", (SCRIPTS / "origin-auth-proxy.py").read_text())

    def test_start_uses_split_worker_env_and_failure_cleanup(self):
        text = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("generate-node-env.py", text)
        self.assertIn("WORKER_ENV_FILE", text)
        self.assertNotIn('scp "$ENV_FILE"', text)
        self.assertIn("cleanup_partial_start", text)
        self.assertIn("verify_stopped", text)
        self.assertIn("head-proxy", text)
        self.assertIn("--internal", text)
        smoke = (ROOT / "smoke-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("stop-deepseek-v4-flash-dspark.sh", smoke)
        self.assertIn("--expect stopped", smoke)

    def test_compose_has_head_only_secret_proxy_and_loopback_vllm(self):
        text = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertIn("origin-auth-proxy:", text)
        self.assertIn('profiles: ["head-proxy"]', text)
        self.assertIn("/run/secrets/origin.key:ro", text)
        self.assertIn("origin-auth-proxy.py", text)
        self.assertNotIn("VLLM_API_KEY", text)
        example = (ROOT / ".env.dspark.example").read_text()
        self.assertIn("VLLM_HOST=127.0.0.1", example)
        self.assertIn("VLLM_PORT=8889", example)

    def test_stop_and_status_are_fail_closed(self):
        stop = (ROOT / "stop-deepseek-v4-flash-dspark.sh").read_text()
        status = (ROOT / "status-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("verify_stopped", stop)
        self.assertIn("exit 1", stop)
        self.assertIn("--expect", status)
        self.assertIn("VLLM_ORIGIN_KEY_FILE", status)
        self.assertIn("SERVED_MODEL_NAME", status)

    def test_semantic_smoke_covers_stream_reasoning_and_tools(self):
        text = (SCRIPTS / "smoke-openai-compat.py").read_text()
        for required in (
            "/v1/models", "stream", "reasoning", "tools", "tool_calls",
            "choices", "message", "--runs",
        ):
            self.assertIn(required, text)
        self.assertIn('message.get("reasoning") or message.get("reasoning_content")', text)

    def test_semantic_smoke_asserts_every_reasoning_mode_response(self):
        smoke = load_semantic_smoke()
        seen = []

        def fake_request(_base_url, _key, path, payload=None, timeout=300):
            self.assertEqual(path, "/chat/completions")
            kwargs = payload["chat_template_kwargs"]
            seen.append(kwargs)
            effort = kwargs["reasoning_effort"]
            message = {"content": f"{effort.upper()}_OK"}
            if effort != "none":
                message["reasoning_content"] = "brief reasoning"
            return {"choices": [{"message": message}]}

        with mock.patch.object(smoke, "request_json", side_effect=fake_request):
            smoke.run_reasoning_modes("http://origin/v1", "secret", "model")
        self.assertEqual(seen, list(smoke.THINKING_MODE_KWARGS.values()))
        self.assertEqual(tuple(smoke.THINKING_MODE_KWARGS), ("off", "low", "high", "max"))

        for field in ("reasoning", "reasoning_content"):
            with self.subTest(response_field=field):
                smoke.assert_message(
                    {"choices": [{"message": {"content": "OK", field: "thinking"}}]},
                    require_reasoning=True,
                    require_content=True,
                )

        bad_off = {"choices": [{"message": {"content": "OK", "reasoning": "still thinking"}}]}
        with self.assertRaisesRegex(AssertionError, "unexpectedly emitted reasoning"):
            smoke.assert_message(bad_off, forbid_reasoning=True)

    def test_history_render_characterizes_default_drop_and_preserves_both_fields(self):
        smoke = load_semantic_smoke()
        seen = []

        def fake_render(url, _key, payload=None, timeout=300):
            self.assertEqual(url, "http://origin/tokenize")
            assistant = payload["messages"][1]
            field = next(name for name in smoke.HISTORY_REASONING_FIELDS if name in assistant)
            marker = assistant[field]
            preserve = payload["chat_template_kwargs"].get("drop_thinking") is False
            seen.append((field, preserve))
            rendered = (
                f"prefix<think>{marker}</think>suffix"
                if preserve else "prefix<think></think>suffix"
            )
            return {"token_strs": [rendered], "count": 1}

        with mock.patch.object(smoke, "request_json_url", side_effect=fake_render):
            smoke.run_history_preservation(
                "http://origin/v1", "secret", "model",
                fields=smoke.HISTORY_REASONING_FIELDS,
            )
        self.assertEqual(
            seen,
            [
                ("reasoning", False), ("reasoning", True),
                ("reasoning_content", False), ("reasoning_content", True),
            ],
        )

    def test_tool_history_render_matches_string_and_dictionary_arguments(self):
        smoke = load_semantic_smoke()
        seen = []

        def fake_render(url, _key, payload=None, timeout=300):
            self.assertEqual(url, "http://origin/tokenize")
            arguments = payload["messages"][1]["tool_calls"][0]["function"]["arguments"]
            seen.append(type(arguments))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            rendered = "\n".join(f'{key}={value}' for key, value in arguments.items())
            return {"token_strs": [rendered], "count": 2}

        with mock.patch.object(smoke, "request_json_url", side_effect=fake_render):
            smoke.run_tool_history_render("http://origin/v1", "secret", "model")
        self.assertEqual(seen, [str, dict])

    def test_semantic_smoke_requires_unknown_field_http_400(self):
        smoke = load_semantic_smoke()
        error = urllib.error.HTTPError(
            "http://origin/v1/chat/completions", 400, "Bad Request", {}, io.BytesIO(b"{}")
        )
        with mock.patch.object(smoke, "request_json", side_effect=error) as request:
            smoke.run_unknown_field_rejection("http://origin/v1", "secret", "model")
        self.assertEqual(request.call_count, 1)
        payload = request.call_args.args[3]
        self.assertTrue(payload["dspark_invented_top_level_field"])

        with mock.patch.object(smoke, "request_json", return_value={"choices": []}):
            with self.assertRaisesRegex(AssertionError, "reached the model boundary"):
                smoke.run_unknown_field_rejection("http://origin/v1", "secret", "model")

    def test_cap_hit_classification_does_not_depend_on_reasoning_presence(self):
        smoke = load_semantic_smoke()
        for message in (
            {"content": "", "reasoning": "budget consumed by reasoning"},
            {"content": None},
        ):
            with self.subTest(message=message):
                state = smoke.classify_completion({
                    "choices": [{"finish_reason": "length", "message": message}]
                })
                self.assertEqual(state["state"], "truncated")
                self.assertEqual(state["finish_reason"], "length")
        with self.assertRaisesRegex(AssertionError, "empty final content"):
            smoke.classify_completion({
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}]
            })

    def test_full_context_gate_enforces_memory_pressure_headroom_and_restart_limits(self):
        probe = load_script("probe-full-context.py")
        healthy = [
            {"node": "head", "mem_available_bytes": 9 * 1024**3, "pressure_full_avg10": 0.0},
            {"node": "worker", "mem_available_bytes": 10 * 1024**3, "pressure_full_avg10": 0.0},
        ]
        result = probe.evaluate_full_context_gate(
            memory_samples=healthy,
            target_prompt_tokens=1048320,
            observed_prompt_tokens=1048320,
            completion_tokens=1,
            restart_delta=0,
            oom_detected=False,
        )
        self.assertTrue(result["passed"])
        transient = healthy * 9 + [
            {"node": "head", "mem_available_bytes": 9 * 1024**3, "pressure_full_avg10": 0.98},
            {"node": "worker", "mem_available_bytes": 10 * 1024**3, "pressure_full_avg10": 0.76},
        ]
        transient_result = probe.evaluate_full_context_gate(
            memory_samples=transient,
            target_prompt_tokens=1048320,
            observed_prompt_tokens=1048320,
            completion_tokens=1,
            restart_delta=0,
            oom_detected=False,
        )
        self.assertTrue(transient_result["checks"]["memory_pressure"])
        self.assertLessEqual(
            transient_result["metrics"]["pressured_sample_ratio"], 0.10
        )
        self.assertEqual(probe.MAX_MEMORY_PSI_FULL_AVG10, 5.0)
        pressured = healthy + [
            {"node": "worker", "mem_available_bytes": 7 * 1024**3, "pressure_full_avg10": 2.0}
        ]
        result = probe.evaluate_full_context_gate(
            memory_samples=pressured,
            target_prompt_tokens=1048320,
            observed_prompt_tokens=1048320,
            completion_tokens=1,
            restart_delta=1,
            oom_detected=True,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["memory_headroom"])
        self.assertFalse(result["checks"]["memory_pressure"])
        self.assertFalse(result["checks"]["no_restart_or_oom"])

    def test_scheduler_baseline_rejects_wrong_provenance_or_configuration(self):
        benchmark = load_script("benchmark-scheduler.py")
        payload = {
            "schema": "dspark-scheduler-baseline/v1",
            "baseline_capture": True,
            "model": "model",
            "configuration": {
                "concurrency": 6,
                "mtp": 5,
                "target_prompt_tokens": 8192,
                "max_num_batched_tokens": 8192,
            },
            "requests": [{"correct": True} for _ in range(6)],
            "gate": {
                "passed": True,
                "metrics": {"p95_ttft_s": 4.0, "p95_decode_latency_s": 0.1},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(payload))
            metrics, digest = benchmark.load_baseline(
                path, model="model", concurrency=6, mtp=5,
                target_prompt_tokens=8192,
            )
            self.assertEqual(metrics["p95_ttft_s"], 4.0)
            self.assertRegex(digest, r"^[a-f0-9]{64}$")
            payload["configuration"]["max_num_batched_tokens"] = 8216
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "provenance/configuration"):
                benchmark.load_baseline(
                    path, model="model", concurrency=6, mtp=5,
                    target_prompt_tokens=8192,
                )

    def test_scheduler_gate_enforces_six_correct_requests_and_latency_regression(self):
        benchmark = load_script("benchmark-scheduler.py")
        requests = [
            {
                "index": index,
                "correct": True,
                "ttft_s": 10 + index,
                "decode_latency_s": 0.2 + index / 100,
                "output_tokens": 2,
            }
            for index in range(6)
        ]
        result = benchmark.evaluate_scheduler_gate(
            requests=requests,
            concurrency=6,
            mtp=5,
            max_num_batched_tokens=8216,
            baseline={"p95_ttft_s": 16.0, "p95_decode_latency_s": 0.3},
            restart_delta=0,
            oom_detected=False,
            eager_fallback_detected=False,
        )
        self.assertTrue(result["passed"])
        requests[0]["correct"] = False
        result = benchmark.evaluate_scheduler_gate(
            requests=requests,
            concurrency=6,
            mtp=5,
            max_num_batched_tokens=8216,
            baseline={"p95_ttft_s": 1.0, "p95_decode_latency_s": 0.01},
            restart_delta=0,
            oom_detected=False,
            eager_fallback_detected=True,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["correctness"])
        self.assertFalse(result["checks"]["ttft_regression_within_40_percent"])
        self.assertFalse(result["checks"]["decode_regression_within_25_percent"])
        self.assertFalse(result["checks"]["no_eager_fallback"])
        source = (SCRIPTS / "benchmark-scheduler.py").read_text()
        self.assertIn("warmup_results = run_batch()", source)
        self.assertIn("results = run_batch()", source)

    def test_u3_prompt_calibration_accepts_tokenizer_rounding(self):
        scheduler = (SCRIPTS / "benchmark-scheduler.py").read_text()
        full_context = (SCRIPTS / "probe-full-context.py").read_text()
        self.assertIn("target - 8 <= observed", scheduler)
        self.assertIn("target - 64 <= observed", full_context)
        probe = load_script("probe-full-context.py")
        self.assertEqual(probe.positive_usage_token_count({}, "prompt_tokens"), 0)
        self.assertEqual(
            probe.positive_usage_token_count({"prompt_tokens": True}, "prompt_tokens"), 0
        )
        self.assertEqual(
            probe.positive_usage_token_count(
                {"prompt_tokens": 1048317}, "prompt_tokens"
            ),
            1048317,
        )
        healthy = [
            {"node": "head", "mem_available_bytes": 9 * 1024**3, "pressure_full_avg10": 0.0},
            {"node": "worker", "mem_available_bytes": 9 * 1024**3, "pressure_full_avg10": 0.0},
        ]
        result = probe.evaluate_full_context_gate(
            memory_samples=healthy, target_prompt_tokens=1048320,
            observed_prompt_tokens=1048256, completion_tokens=1,
            restart_delta=0, oom_detected=False,
        )
        self.assertTrue(result["checks"]["near_max_prefill"])

    def test_long_context_decode_gate_requires_repeated_64_token_streams(self):
        probe = load_script("probe-long-context-decode.py")
        healthy = [
            {"post_first_tokens": 64, "decode_tps": 17.3},
            {"post_first_tokens": 64, "decode_tps": 17.1},
        ]
        accepted = probe.evaluate_decode_gate(healthy, baseline_tps=1.0)
        self.assertTrue(accepted["passed"])
        too_short = probe.evaluate_decode_gate(
            [{"post_first_tokens": 63, "decode_tps": 20.0}, healthy[1]],
            baseline_tps=1.0,
        )
        self.assertFalse(too_short["passed"])
        slow = probe.evaluate_decode_gate(
            [{"post_first_tokens": 64, "decode_tps": 9.9}] * 2,
            baseline_tps=1.0,
        )
        self.assertFalse(slow["passed"])
        source = (SCRIPTS / "probe-long-context-decode.py").read_text()
        self.assertIn('"stream": True', source)
        self.assertIn('"temperature": 0', source)
        self.assertIn('"reasoning_effort": "none"', source)
        self.assertIn("identical_prompt_sha256", source)

    def test_u3_remote_observation_preserves_shell_command_boundaries(self):
        for name in ("probe-full-context.py", "benchmark-scheduler.py"):
            with self.subTest(script=name):
                source = (SCRIPTS / name).read_text()
                self.assertIn("shlex.quote(part)", source)
                self.assertIn('" ".join(', source)

    def test_full_context_proxy_timeout_is_explicit_and_preflighted(self):
        compose = (ROOT / "docker-compose.dspark.yml").read_text()
        env_example = (ROOT / ".env.dspark.example").read_text()
        start = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("VLLM_PROXY_UPSTREAM_TIMEOUT:-3600", compose)
        self.assertIn("VLLM_PROXY_UPSTREAM_TIMEOUT=3600", env_example)
        self.assertIn("VLLM_PROXY_UPSTREAM_TIMEOUT must be an integer", start)

    def test_u3_operator_scripts_require_mode_0600_key_files(self):
        for script_name in (
            "probe-full-context.py",
            "probe-long-context-decode.py",
            "benchmark-scheduler.py",
        ):
            with self.subTest(script=script_name), tempfile.TemporaryDirectory() as tmp:
                script = load_script(script_name)
                path = Path(tmp) / "origin.key"
                path.write_text("do-not-print-this-secret\n")
                path.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "0600"):
                    script.read_key_file(path)
                path.chmod(0o600)
                self.assertEqual(script.read_key_file(path), "do-not-print-this-secret")
                link = Path(tmp) / "link.key"
                link.symlink_to(path)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    script.read_key_file(link)

    def test_semantic_canary_is_bounded_and_classifies_readiness_failures(self):
        smoke = load_semantic_smoke()
        models = {"data": [{"id": "model"}]}
        completion = {
            "choices": [{"finish_reason": "stop", "message": {"content": "READY."}}]
        }

        with mock.patch.object(smoke, "request_json", side_effect=[models, completion]) as request:
            result = smoke.probe_semantic_readiness(
                "http://origin/v1", "secret", "model", wall_timeout=120
            )
        self.assertEqual(result["state"], "semantic-ready")
        self.assertTrue(result["ready"])
        generation = request.call_args_list[1]
        self.assertEqual(generation.args[2], "/chat/completions")
        self.assertLessEqual(generation.args[3]["max_completion_tokens"], 16)
        self.assertEqual(generation.args[3]["chat_template_kwargs"]["reasoning_effort"], "none")
        self.assertEqual(result["wall_timeout_seconds"], 120)

        unauthorized = urllib.error.HTTPError(
            "http://origin/v1/models", 401, "Unauthorized", {}, io.BytesIO(b"{}")
        )
        with mock.patch.object(smoke, "request_json", side_effect=unauthorized):
            result = smoke.probe_semantic_readiness(
                "http://origin/v1", "secret", "model", wall_timeout=120
            )
        self.assertEqual(result["state"], "auth-required")

        with mock.patch.object(
            smoke, "request_json", side_effect=urllib.error.URLError("connection refused")
        ):
            result = smoke.probe_semantic_readiness(
                "http://origin/v1", "secret", "model", wall_timeout=120
            )
        self.assertEqual(result["state"], "host-down")

        active_metrics = """vllm:num_requests_running{model_name="model"} 1
vllm:num_requests_waiting{model_name="model"} 2
"""
        with mock.patch.object(
            smoke, "request_json", side_effect=[models, TimeoutError("deadline")]
        ), mock.patch.object(smoke, "request_text", return_value=active_metrics):
            result = smoke.probe_semantic_readiness(
                "http://origin/v1", "secret", "model", wall_timeout=120
            )
        self.assertEqual(result["state"], "busy/degraded")
        self.assertFalse(result["ready"])

        idle_metrics = """vllm:num_requests_running 0
vllm:num_requests_waiting 0
"""
        with mock.patch.object(
            smoke, "request_json", side_effect=[models, TimeoutError("deadline")]
        ), mock.patch.object(smoke, "request_text", return_value=idle_metrics):
            result = smoke.probe_semantic_readiness(
                "http://origin/v1", "secret", "model", wall_timeout=120
            )
        self.assertEqual(result["state"], "unavailable/not-ready")

        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / "origin.key"
            key_file.write_text("synthetic-secret\n")
            key_file.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "0600"):
                smoke.read_key_file(key_file)
            key_file.chmod(0o600)
            self.assertEqual(smoke.read_key_file(key_file), "synthetic-secret")

    def test_prometheus_metric_sum_matches_exact_name_across_labels(self):
        smoke = load_semantic_smoke()
        metrics = """vllm:spec_decode_num_accepted_tokens_total{rank="0"} 4
vllm:spec_decode_num_accepted_tokens_total{rank="1"} 7
vllm:spec_decode_num_accepted_tokens_total_created 999
vllm:spec_decode_num_draft_tokens_total{rank="0"} 20
"""
        self.assertEqual(
            smoke.prometheus_metric_sum(
                metrics, "vllm:spec_decode_num_accepted_tokens_total"
            ),
            11,
        )

    def test_status_semantic_gate_never_invokes_lifecycle_actions(self):
        status = (ROOT / "status-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn("--semantic", status)
        self.assertIn("SEMANTIC_DEADLINE_SECONDS=120", status)
        self.assertIn("--semantic-canary", status)
        self.assertNotIn("stop-deepseek-v4-flash-dspark.sh", status)
        self.assertNotIn("docker restart", status)
        self.assertNotIn("docker stop", status)

    def test_cap_probe_is_bounded_and_reports_truncation_without_retry(self):
        smoke = load_semantic_smoke()
        seen = []

        def fake_request(_base_url, _key, path, payload=None, timeout=300):
            seen.append((path, payload, timeout))
            return {
                "choices": [{
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "still thinking"},
                }]
            }

        with mock.patch.object(smoke, "request_json", side_effect=fake_request):
            result = smoke.run_cap_probe("http://origin/v1", "secret", "model")
        self.assertEqual(result["state"], "truncated")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "/chat/completions")
        self.assertLessEqual(seen[0][1]["max_completion_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
