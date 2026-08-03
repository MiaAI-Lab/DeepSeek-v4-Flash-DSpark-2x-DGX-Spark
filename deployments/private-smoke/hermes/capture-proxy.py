#!/usr/bin/env python3
"""Ephemeral loopback proxy that records only provider response IDs."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import signal
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.error
import urllib.request
from urllib.parse import urlparse


HOP_HEADERS = {
    "accept-encoding", "connection", "content-length", "host", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
}
MAX_REQUEST_BYTES = 16 * 1024 * 1024


def validate_target(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme != "http" or parsed.path.rstrip("/") != "/v1":
        raise SystemExit("capture target must be http://TAILNET_HOST:PORT/v1")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SystemExit("capture target must not contain credentials or query data")
    host = parsed.hostname or ""
    allowed = host.endswith(".ts.net")
    try:
        allowed = allowed or ipaddress.ip_address(host) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        pass
    if not allowed:
        raise SystemExit("capture target must be a Tailscale address/name")
    return f"{parsed.scheme}://{parsed.netloc}"


def response_ids(body: bytes) -> list[str]:
    ids: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        value = item.get("id") if isinstance(item, dict) else None
        if isinstance(value, str) and value not in ids:
            ids.append(value)
    if ids:
        return ids
    try:
        item = json.loads(body)
    except json.JSONDecodeError:
        return []
    value = item.get("id") if isinstance(item, dict) else None
    return [value] if isinstance(value, str) else []


def build_handler(target_origin: str, evidence_path: Path):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802
            self._forward()

        def do_POST(self):  # noqa: N802
            self._forward()

        def _forward(self):
            length = int(self.headers.get("content-length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                self.send_error(413)
                return
            request_body = self.rfile.read(length) if length else None
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in HOP_HEADERS
            }
            headers["Accept-Encoding"] = "identity"
            upstream = urllib.request.Request(
                target_origin + self.path,
                data=request_body,
                headers=headers,
                method=self.command,
            )
            try:
                response = urllib.request.urlopen(upstream, timeout=900)
            except urllib.error.HTTPError as exc:
                response = exc
            except Exception:
                body = b'{"error":{"message":"capture proxy upstream failure"}}'
                self.send_response(502)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self._record(502, [])
                return
            with response:
                body = response.read()
                status = int(response.status)
                response_headers = list(response.headers.items())
            self.send_response(status)
            for key, value in response_headers:
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if self.path.rstrip("/").endswith("/chat/completions"):
                self._record(status, response_ids(body))

        def _record(self, status: int, ids: list[str]):
            row = {
                "path": self.path,
                "request_id": self.headers.get("X-Request-ID", ""),
                "response_ids": ids,
                "status": status,
            }
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            with lock, evidence_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)

        def log_message(self, *_args):
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--evidence-file", type=Path, required=True)
    args = parser.parse_args()
    target_origin = validate_target(args.target_base_url)
    old_umask = os.umask(0o077)
    try:
        args.evidence_file.write_text("", encoding="utf-8")
        args.port_file.write_text("", encoding="utf-8")
    finally:
        os.umask(old_umask)
    for path in (args.evidence_file, args.port_file):
        path.chmod(0o600)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise SystemExit("capture evidence files must be mode 0600")
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(target_origin, args.evidence_file))
    args.port_file.write_text(str(server.server_port), encoding="utf-8")

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
