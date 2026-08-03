import http.server
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LifecycleContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
