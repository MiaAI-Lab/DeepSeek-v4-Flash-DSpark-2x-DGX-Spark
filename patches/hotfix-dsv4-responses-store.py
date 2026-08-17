#!/usr/bin/env python3
"""Bound vLLM 752a3's opt-in in-memory Responses API store.

The pinned Anemll image supports ``VLLM_ENABLE_RESPONSES_API_STORE`` but keeps
responses, rendered messages, and background events forever.  This exact-source
backport adds a configurable bound while protecting active work. Stateful
``previous_response_id`` reuse refreshes that response's recency.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py"
)
PREIMAGE_SHA256 = "fe3a48ab09c516835ce6dd1471c06cc784ae7504eaa7af7f10574704106830d8"
POSTIMAGE_SHA256 = "caeeb0ed55517f933def7ea0a590125d5c727369318ce20d4cc1821279fb864d"
MARK = "# [dspark-responses-store] bounded active-safe store"

INIT_ANCHOR = """        self.background_tasks: dict[str, asyncio.Task] = {}

        self.tool_server = tool_server
"""
INIT_REPLACEMENT = """        self.background_tasks: dict[str, asyncio.Task] = {}
        self.active_store_requests: set[str] = set()
        self.responses_store_max_entries = int(
            __import__("os").environ.get("DSPARK_RESPONSES_STORE_MAX_ENTRIES", "256")
        )
        if self.enable_store and self.responses_store_max_entries <= 0:
            raise ValueError("DSPARK_RESPONSES_STORE_MAX_ENTRIES must be greater than 0")

        self.tool_server = tool_server
"""
METHOD_ANCHOR = """        self.tool_server = tool_server

    def _effective_chat_template_kwargs(
"""
METHOD_REPLACEMENT = f"""        self.tool_server = tool_server

    {MARK}
    def _prune_response_store_locked(self) -> None:
        protected = self.active_store_requests | set(self.background_tasks)
        for response_id in tuple(self.msg_store):
            if response_id not in self.response_store and response_id not in protected:
                self.msg_store.pop(response_id, None)
        for response_id in tuple(self.event_store):
            if response_id not in self.response_store and response_id not in protected:
                self.event_store.pop(response_id, None)
        terminal = [response_id for response_id, response in self.response_store.items()
                    if response_id not in protected
                    and response.status not in ("queued", "in_progress")]
        excess = len(terminal) - self.responses_store_max_entries
        for response_id in terminal[:max(0, excess)]:
            self.response_store.pop(response_id, None)
            self.msg_store.pop(response_id, None)
            self.event_store.pop(response_id, None)

    async def _background_store_task_done(self, response_id: str, task) -> None:
        async with self.response_store_lock:
            response = self.response_store.get(response_id)
            if response is not None and response.status in ("queued", "in_progress"):
                if task.cancelled():
                    response.status = "cancelled"
                else:
                    task.exception()  # retrieve producer failure before cleanup
                    response.status = "failed"
            self.background_tasks.pop(response_id, None)
            self.active_store_requests.discard(response_id)
            self._prune_response_store_locked()

    async def _stored_stream_lifecycle(self, request, generator):
        try:
            async for event in generator:
                yield event
        finally:
            async with self.response_store_lock:
                self.active_store_requests.discard(request.request_id)
                self._prune_response_store_locked()

    def _effective_chat_template_kwargs(
"""
PREVIOUS_ANCHOR = """            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
"""
PREVIOUS_REPLACEMENT = """            async with self.response_store_lock:
                prev_response = self.response_store.pop(prev_response_id, None)
                if prev_response is not None:
                    self.response_store[prev_response_id] = prev_response
            if prev_response is None:
"""
MESSAGE_ANCHOR = """        # Store the input messages.
        if request.store:
            self.msg_store[request.request_id] = messages

        if request.background:
"""
MESSAGE_REPLACEMENT = """        # Store the input messages and protect the request until its
        # terminal response or background task cleanup is durable.
        if request.store:
            async with self.response_store_lock:
                self.active_store_requests.add(request.request_id)
                self.msg_store[request.request_id] = messages

        if request.background:
"""
BACKGROUND_ANCHOR = """            async with self.response_store_lock:
                self.response_store[response.id] = response

            # Run the request in the background.
"""
BACKGROUND_REPLACEMENT = """            async with self.response_store_lock:
                self.response_store[response.id] = response
                self._prune_response_store_locked()

            # Run the request in the background.
"""
BACKGROUND_DONE_ANCHOR = """            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda _: self.background_tasks.pop(response_id, None)
            )

            if request.stream:
"""
BACKGROUND_DONE_REPLACEMENT = """            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda completed_task: asyncio.create_task(
                    self._background_store_task_done(response_id, completed_task)
                )
            )
            async with self.response_store_lock:
                self.active_store_requests.discard(response_id)
                self._prune_response_store_locked()

            if request.stream:
"""
FOREGROUND_ANCHOR = """        if request.stream:
            return self.responses_stream_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )

        return await self.responses_full_generator(
            request,
            sampling_params,
            result_generator,
            context,
            model_name,
            tokenizer,
            request_metadata,
        )
"""
FOREGROUND_REPLACEMENT = """        if request.stream:
            generator = self.responses_stream_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )
            return self._stored_stream_lifecycle(request, generator)

        try:
            return await self.responses_full_generator(
                request,
                sampling_params,
                result_generator,
                context,
                model_name,
                tokenizer,
                request_metadata,
            )
        finally:
            if request.store:
                async with self.response_store_lock:
                    self.active_store_requests.discard(request.request_id)
                    self._prune_response_store_locked()
"""
FINAL_ANCHOR = """                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response
        return response
"""
FINAL_REPLACEMENT = """                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store.pop(response.id, None)
                    self.response_store[response.id] = response
                self._prune_response_store_locked()
        return response
"""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transformed(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    if MARK in text:
        return raw
    for name, old, new in (
        ("init", INIT_ANCHOR, INIT_REPLACEMENT),
        ("methods", METHOD_ANCHOR, METHOD_REPLACEMENT),
        ("previous", PREVIOUS_ANCHOR, PREVIOUS_REPLACEMENT),
        ("messages", MESSAGE_ANCHOR, MESSAGE_REPLACEMENT),
        ("background", BACKGROUND_ANCHOR, BACKGROUND_REPLACEMENT),
        ("background-done", BACKGROUND_DONE_ANCHOR, BACKGROUND_DONE_REPLACEMENT),
        ("foreground", FOREGROUND_ANCHOR, FOREGROUND_REPLACEMENT),
        ("final", FINAL_ANCHOR, FINAL_REPLACEMENT),
    ):
        if text.count(old) != 1:
            raise RuntimeError(f"Responses store patch anchor drift: {name}")
        text = text.replace(old, new)
    return text.encode("utf-8")


def write_atomic(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.responses-store.{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o644)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def apply(path: Path) -> str:
    raw = path.read_bytes()
    current = sha256(raw)
    if current == POSTIMAGE_SHA256:
        return current
    if current != PREIMAGE_SHA256:
        raise RuntimeError("Responses serving source hash drift")
    updated = transformed(raw)
    updated_hash = sha256(updated)
    if updated_hash != POSTIMAGE_SHA256:
        raise RuntimeError("Responses serving postimage hash drift")
    compile(updated, str(path), "exec")
    write_atomic(path, updated)
    return updated_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    print(apply(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
