"""Content-free vLLM HTTP lifecycle observer.

This middleware is useful caller-side evidence on John, but it is deliberately
tagged ``http_lifecycle`` and is never rank-progress proof. The custom TP=2
runtime does not expose a supported per-rank iteration hook through FastAPI.
Real worker instrumentation may use ``write_rank_progress`` with the
``rank_worker`` scope; absent that artifact, the rank sentinel fails closed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import threading
import time


MAX_JSON_BYTES = 16_384
GENERATION_PATHS = frozenset(
    {"/v1/chat/completions", "/v1/completions", "/v1/responses"}
)


def _bounded_json(value):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("observer payload too large")
    return raw


def _atomic_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class EventEmitter:
    def __init__(self, path, now=time.time):
        self.path = Path(path)
        self.now = now
        self.instance_id = secrets.token_urlsafe(18)
        self.sequence = 0
        self.lock = threading.Lock()

    def emit(self, request_id, lifecycle):
        with self.lock:
            self.sequence += 1
            row = {
                "contractVersion": 1,
                "scope": "http_lifecycle",
                "observerInstanceId": self.instance_id,
                "sourceSequence": self.sequence,
                "eventAt": self.now(),
                "requestId": request_id,
                "lifecycle": lifecycle,
            }
            raw = _bounded_json(row) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
            )
            try:
                os.write(descriptor, raw)
            finally:
                os.close(descriptor)


def write_rank_progress(path, payload):
    """Persist a snapshot supplied by genuine rank-worker instrumentation."""
    if payload.get("contractVersion") != 1 or payload.get("scope") != "rank_worker":
        raise ValueError("rank progress requires rank_worker scope")
    if payload.get("sourceRank") not in (0, 1):
        raise ValueError("rank progress requires source rank")
    _atomic_write(path, _bounded_json(payload) + b"\n")


async def observe_request_with_emitter(request, call_next, emitter):
    if request.method != "POST" or request.url.path not in GENERATION_PATHS:
        return await call_next(request)
    request_id = secrets.token_urlsafe(18)
    emitter.emit(request_id, "received")
    emitter.emit(request_id, "awaiting_first_output")
    try:
        response = await call_next(request)
    except BaseException:
        emitter.emit(request_id, "failed_disconnected")
        raise
    iterator = getattr(response, "body_iterator", None)
    if iterator is None:
        emitter.emit(request_id, "completed")
        return response

    async def observed_body():
        emitted_streaming = False
        try:
            async for chunk in iterator:
                if not emitted_streaming:
                    emitted_streaming = True
                    emitter.emit(request_id, "streaming")
                yield chunk
        except BaseException:
            emitter.emit(request_id, "failed_disconnected")
            raise
        else:
            emitter.emit(request_id, "completed")

    response.body_iterator = observed_body()
    return response


_DEFAULT_EMITTER = None
if os.environ.get("MONITOR_OBSERVER_ENABLED") == "1" and os.environ.get("HEADLESS") != "1":
    _DEFAULT_EMITTER = EventEmitter(
        os.environ.get(
            "MONITOR_HTTP_LIFECYCLE_PATH",
            "/run/model-serving-monitor/http-lifecycle.ndjson",
        )
    )


async def observe_request(request, call_next):
    if _DEFAULT_EMITTER is None:
        return await call_next(request)
    return await observe_request_with_emitter(request, call_next, _DEFAULT_EMITTER)
