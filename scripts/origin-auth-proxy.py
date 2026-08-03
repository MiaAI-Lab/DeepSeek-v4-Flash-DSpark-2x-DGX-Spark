#!/usr/bin/env python3
"""Small bridge-scoped bearer proxy for the loopback-only vLLM origin."""

from __future__ import annotations

import argparse
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import stat
import sys


HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


def read_key(path: Path) -> str:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f"key file must be mode 0600 or stricter: {path}")
    key = path.read_text().strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError("key file must contain one non-empty line")
    return key


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dspark-origin-proxy/1"

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "proxy_error"}}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _proxy(self) -> None:
        client_ip = ipaddress.ip_address(self.client_address[0])
        if client_ip not in self.server.allow_network:  # type: ignore[attr-defined]
            self._json_error(403, "source is outside the private model bridge")
            return
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.origin_key}"  # type: ignore[attr-defined]
        if not hmac.compare_digest(supplied, expected):
            self._json_error(401, "missing or invalid bearer token")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length > self.server.max_body_bytes:  # type: ignore[attr-defined]
            self._json_error(413, "request body is too large")
            return
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP | {"authorization", "host", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))

        upstream = http.client.HTTPConnection(
            self.server.upstream_host,  # type: ignore[attr-defined]
            self.server.upstream_port,  # type: ignore[attr-defined]
            timeout=self.server.upstream_timeout,  # type: ignore[attr-defined]
        )
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            # ``HTTPResponse.read(n)`` blocks until n bytes or EOF.  A normal
            # 512-token SSE completion is smaller than 64 KiB, so using read()
            # here silently buffered the entire stream and turned TTFT into
            # total request time.  read1() performs at most one underlying read
            # and lets each available SSE fragment cross the authenticated
            # bridge immediately.
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (OSError, http.client.HTTPException) as exc:
            if not self.wfile.closed:
                self._json_error(502, f"upstream unavailable: {type(exc).__name__}")
        finally:
            upstream.close()

    do_GET = _proxy
    do_POST = _proxy

    def log_message(self, format_string: str, *args: object) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.client_address[0], self.log_date_time_string(), format_string % args)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8889)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--allow-cidr", default="172.30.0.0/24")
    parser.add_argument("--upstream-timeout", type=float, default=900)
    parser.add_argument("--max-body-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.origin_key = read_key(args.key_file)
    server.allow_network = ipaddress.ip_network(args.allow_cidr)
    server.upstream_host = args.upstream_host
    server.upstream_port = args.upstream_port
    server.upstream_timeout = args.upstream_timeout
    server.max_body_bytes = args.max_body_bytes
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
