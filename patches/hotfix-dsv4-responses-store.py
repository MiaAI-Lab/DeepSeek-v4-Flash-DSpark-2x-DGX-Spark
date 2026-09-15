#!/usr/bin/env python3
"""Bound the pinned vLLM Responses API terminal-state store.

The patch is invoked only when ``VLLM_ENABLE_RESPONSES_API_STORE=1``. It keeps
one LRU order for response, rendered-message, and background-event state.
Queued, in-progress, pinned-continuation, and tracked-producer entries are not
evicted; an active reader retains its captured event state if the dictionary
bundle is later evicted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
from pathlib import Path


DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py"
)
PREIMAGE_SHA256 = "fe3a48ab09c516835ce6dd1471c06cc784ae7504eaa7af7f10574704106830d8"
POSTIMAGE_SHA256 = "1b0033131a34e03a2e129743258f5da81b3e60e979072920153f6d09bf4e5d8f"
MARK = "# [dspark-responses-store] bounded terminal store"

WARNING_ANCHOR = """        self.enable_store = envs.VLLM_ENABLE_RESPONSES_API_STORE
        if self.enable_store:
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled. This may "
                "cause a memory leak since we never remove responses from "
                "the store."
            )
"""
WARNING_REPLACEMENT = """        self.enable_store = envs.VLLM_ENABLE_RESPONSES_API_STORE
        if self.enable_store:
            self.responses_store_max_entries = int(
                __import__("os").environ.get(
                    "DSPARK_RESPONSES_STORE_MAX_ENTRIES", "256"
                )
            )
            if self.responses_store_max_entries <= 0:
                raise ValueError(
                    "DSPARK_RESPONSES_STORE_MAX_ENTRIES must be greater than 0"
                )
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled; retaining at "
                "most %d terminal responses in memory.",
                self.responses_store_max_entries,
            )
        else:
            self.responses_store_max_entries = 256
"""
STORE_DECLARATIONS_ANCHOR = """        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove responses from the store.
        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()

        # HACK(woosuk): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove messages from the store.
        self.msg_store: dict[str, list[ChatCompletionMessageParam]] = {}

        # HACK(wuhang): This is a hack. We should use a better store.
        # FIXME: If enable_store=True, this may cause a memory leak since we
        # never remove events from the store.
        self.event_store: dict[
            str, tuple[deque[StreamingResponsesResponse], asyncio.Event]
        ] = {}

        self.background_tasks: dict[str, asyncio.Task] = {}
"""
STORE_DECLARATIONS_REPLACEMENT = """        self.response_store: dict[str, ResponsesResponse] = {}
        self.response_store_lock = asyncio.Lock()
        self.response_store_pins: dict[str, int] = {}
        self.msg_store: dict[str, list[ChatCompletionMessageParam]] = {}
        self.event_store: dict[
            str,
            tuple[
                deque[StreamingResponsesResponse],
                asyncio.Event,
                asyncio.Event,
            ],
        ] = {}
        self.background_tasks: dict[str, asyncio.Task] = {}
"""
METHOD_ANCHOR = """        self.tool_server = tool_server

    def _effective_chat_template_kwargs(
"""
METHOD_REPLACEMENT = """        self.tool_server = tool_server

    # [dspark-responses-store] bounded terminal store
    def _remove_stored_response(self, response_id: str) -> None:
        self.response_store.pop(response_id, None)
        self.msg_store.pop(response_id, None)
        self.event_store.pop(response_id, None)

    def _evict_stored_responses_locked(self) -> None:
        terminal_ids = [
            response_id
            for response_id, response in self.response_store.items()
            if response.status not in ("queued", "in_progress")
            and response_id not in self.response_store_pins
            and response_id not in self.background_tasks
        ]
        excess = len(terminal_ids) - self.responses_store_max_entries
        for response_id in terminal_ids[:max(0, excess)]:
            self._remove_stored_response(response_id)

    async def _store_response(
        self,
        response: ResponsesResponse,
        *,
        preserve_cancelled: bool = False,
    ) -> None:
        async with self.response_store_lock:
            stored = self.response_store.get(response.id)
            if (
                preserve_cancelled
                and stored is not None
                and stored.status == "cancelled"
            ):
                return
            self.response_store.pop(response.id, None)
            self.response_store[response.id] = response
            self._evict_stored_responses_locked()

    async def _acquire_stored_response(
        self, response_id: str
    ) -> ResponsesResponse | None:
        async with self.response_store_lock:
            response = self.response_store.pop(response_id, None)
            if response is not None:
                self.response_store[response_id] = response
                self.response_store_pins[response_id] = (
                    self.response_store_pins.get(response_id, 0) + 1
                )
            return response

    async def _release_stored_response(self, response_id: str) -> None:
        async with self.response_store_lock:
            pins = self.response_store_pins[response_id] - 1
            if pins:
                self.response_store_pins[response_id] = pins
            else:
                self.response_store_pins.pop(response_id)
            self._evict_stored_responses_locked()

    async def _foreground_stream_with_cleanup(
        self, response_id: str, messages, generator
    ):
        async with self.response_store_lock:
            self.msg_store[response_id] = messages
        try:
            async for event in generator:
                yield event
        finally:
            async with self.response_store_lock:
                if response_id not in self.response_store:
                    self.msg_store.pop(response_id, None)

    def _background_task_done(self, response_id: str, task) -> None:
        if task.cancelled():
            fallback_status = "cancelled"
        else:
            error = task.exception()
            fallback_status = "failed"
            if error is not None:
                logger.error(
                    "Background Responses request %s failed: %r",
                    response_id,
                    error,
                )

        response = self.response_store.get(response_id)
        if response is not None:
            if response.status in ("queued", "in_progress"):
                response.status = fallback_status
            self.response_store.pop(response_id, None)
            self.response_store[response_id] = response
        self.background_tasks.pop(response_id, None)
        event_state = self.event_store.get(response_id)
        if event_state is not None:
            event_state[2].set()
            event_state[1].set()
        self._evict_stored_responses_locked()

    def _effective_chat_template_kwargs(
"""
PREVIOUS_ANCHOR = """            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
"""
PREVIOUS_REPLACEMENT = """            prev_response = await self._acquire_stored_response(prev_response_id)
            if prev_response is None:
"""
PREPARATION_ANCHOR = """        lora_request = self._maybe_get_adapters(request)
        model_name = self.models.model_name(lora_request)

        if self.use_harmony:
            messages, engine_inputs = self._make_request_with_harmony(
                request, prev_response
            )
        else:
            messages, engine_inputs = await self._make_request(request, prev_response)
"""
PREPARATION_REPLACEMENT = """        try:
            lora_request = self._maybe_get_adapters(request)
            model_name = self.models.model_name(lora_request)

            if self.use_harmony:
                messages, engine_inputs = self._make_request_with_harmony(
                    request, prev_response
                )
            else:
                messages, engine_inputs = await self._make_request(
                    request, prev_response
                )
        finally:
            if prev_response is not None:
                await self._release_stored_response(prev_response.id)
"""
MESSAGE_ANCHOR = """        # Store the input messages.
        if request.store:
            self.msg_store[request.request_id] = messages

        if request.background:
"""
MESSAGE_REPLACEMENT = """        # Background and foreground non-streaming requests retain messages now.
        # Foreground streams defer retention until their generator is consumed.
        if request.store and (request.background or not request.stream):
            self.msg_store[request.request_id] = messages

        if request.background:
"""
BACKGROUND_ANCHOR = """            async with self.response_store_lock:
                self.response_store[response.id] = response

            # Run the request in the background.
            if request.stream:
                task = asyncio.create_task(
                    self._run_background_request_stream(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{request.request_id}",
                )
            else:
                task = asyncio.create_task(
                    self._run_background_request(
                        request,
                        sampling_params,
                        result_generator,
                        context,
                        model_name,
                        tokenizer,
                        request_metadata,
                        created_time,
                    ),
                    name=f"create_{response.id}",
                )

            # For cleanup.
            response_id = response.id
            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda _: self.background_tasks.pop(response_id, None)
            )
"""
BACKGROUND_REPLACEMENT = """            await self._store_response(response)

            # Publish background-stream state before returning its lazy reader.
            if request.stream:
                self.event_store[response.id] = (
                    deque(),
                    asyncio.Event(),
                    asyncio.Event(),
                )
                request_coro = self._run_background_request_stream(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                )
            else:
                request_coro = self._run_background_request(
                    request,
                    sampling_params,
                    result_generator,
                    context,
                    model_name,
                    tokenizer,
                    request_metadata,
                    created_time,
                )

            response_id = response.id
            task = asyncio.create_task(request_coro, name=f"create_{response_id}")
            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda completed: self._background_task_done(
                    response_id, completed
                )
            )
"""
BACKGROUND_RETURN_ANCHOR = """            if request.stream:
                return self.responses_background_stream_generator(request.request_id)
            return response
"""
BACKGROUND_RETURN_REPLACEMENT = """            if request.stream:
                return self.responses_background_stream_generator(
                    request.request_id,
                    event_state=self.event_store[request.request_id],
                )
            return response
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
            if request.store:
                return self._foreground_stream_with_cleanup(
                    request.request_id, messages, generator
                )
            return generator

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
                    if request.request_id not in self.response_store:
                        self.msg_store.pop(request.request_id, None)
"""
FINAL_ANCHOR = """        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it.
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response
        return response
"""
FINAL_REPLACEMENT = """        if request.store:
            await self._store_response(response, preserve_cancelled=True)
        return response
"""
PRODUCER_ANCHOR = """        event_deque: deque[StreamingResponsesResponse] = deque()
        new_event_signal = asyncio.Event()
        self.event_store[request.request_id] = (event_deque, new_event_signal)
        generator = self.responses_stream_generator(request, *args, **kwargs)
"""
PRODUCER_REPLACEMENT = """        event_deque, new_event_signal, _ = self.event_store[request.request_id]
        generator = self.responses_stream_generator(request, *args, **kwargs)
"""
READER_ANCHOR = """    async def responses_background_stream_generator(
        self,
        response_id: str,
        starting_after: int | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        if response_id not in self.event_store:
            raise VLLMValidationError(
                f"Unknown response_id: {response_id}",
                parameter="response_id",
                value=response_id,
            )

        event_deque, new_event_signal = self.event_store[response_id]
        start_index = 0 if starting_after is None else starting_after + 1
        current_index = start_index

        while True:
            new_event_signal.clear()

            # Yield existing events from start_index
            while current_index < len(event_deque):
                event = event_deque[current_index]
                yield event
                if getattr(event, "type", "unknown") == "response.completed":
                    return
                current_index += 1

            await new_event_signal.wait()
"""
READER_REPLACEMENT = """    async def responses_background_stream_generator(
        self,
        response_id: str,
        starting_after: int | None = None,
        *,
        event_state: tuple[
            deque[StreamingResponsesResponse],
            asyncio.Event,
            asyncio.Event,
        ] | None = None,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        if event_state is None:
            event_state = self.event_store.get(response_id)
            if event_state is None:
                raise VLLMValidationError(
                    f"Unknown response_id: {response_id}",
                    parameter="response_id",
                    value=response_id,
                )
        event_deque, new_event_signal, done_signal = event_state
        current_index = 0 if starting_after is None else starting_after + 1

        while True:
            new_event_signal.clear()
            while current_index < len(event_deque):
                event = event_deque[current_index]
                current_index += 1
                yield event
                if getattr(event, "type", "unknown") == "response.completed":
                    return
            if done_signal.is_set():
                return
            await new_event_signal.wait()
"""
RETRIEVE_ANCHOR = """        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)

        if stream:
            return self.responses_background_stream_generator(
                response_id,
                starting_after,
            )
        return response
"""
RETRIEVE_REPLACEMENT = """        async with self.response_store_lock:
            response = self.response_store.pop(response_id, None)
            event_state = self.event_store.get(response_id)
            if response is not None:
                self.response_store[response_id] = response

        if response is None:
            return self._make_not_found_error(response_id)

        if stream:
            if event_state is None:
                return self.responses_background_stream_generator(
                    response_id,
                    starting_after,
                )
            return self.responses_background_stream_generator(
                response_id,
                starting_after,
                event_state=event_state,
            )
        return response
"""
CANCEL_ANCHOR = """            # Update the status to "cancelled".
            response.status = "cancelled"

        # Abort the request.
"""
CANCEL_REPLACEMENT = """            # Update the status to "cancelled" and terminalize its LRU entry.
            response.status = "cancelled"
            self.response_store.pop(response_id, None)
            self.response_store[response_id] = response
            self._evict_stored_responses_locked()

        # Abort the request.
"""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def transformed(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    for name, old, new in (
        ("warning", WARNING_ANCHOR, WARNING_REPLACEMENT),
        ("store-declarations", STORE_DECLARATIONS_ANCHOR, STORE_DECLARATIONS_REPLACEMENT),
        ("methods", METHOD_ANCHOR, METHOD_REPLACEMENT),
        ("previous", PREVIOUS_ANCHOR, PREVIOUS_REPLACEMENT),
        ("preparation", PREPARATION_ANCHOR, PREPARATION_REPLACEMENT),
        ("messages", MESSAGE_ANCHOR, MESSAGE_REPLACEMENT),
        ("background", BACKGROUND_ANCHOR, BACKGROUND_REPLACEMENT),
        ("background-return", BACKGROUND_RETURN_ANCHOR, BACKGROUND_RETURN_REPLACEMENT),
        ("foreground", FOREGROUND_ANCHOR, FOREGROUND_REPLACEMENT),
        ("final", FINAL_ANCHOR, FINAL_REPLACEMENT),
        ("background-producer", PRODUCER_ANCHOR, PRODUCER_REPLACEMENT),
        ("background-reader", READER_ANCHOR, READER_REPLACEMENT),
        ("retrieve", RETRIEVE_ANCHOR, RETRIEVE_REPLACEMENT),
        ("cancel", CANCEL_ANCHOR, CANCEL_REPLACEMENT),
    ):
        if text.count(old) != 1:
            raise RuntimeError(f"Responses store patch anchor drift: {name}")
        text = text.replace(old, new, 1)
    if text.count(MARK) != 1:
        raise RuntimeError("Responses store patch marker drift")
    return text.encode("utf-8")


def write_atomic(path: Path, raw: bytes, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.responses-store.{os.getpid()}")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        if temporary.read_bytes() != raw:
            raise RuntimeError("Responses serving staged write verification failed")
        if stat.S_IMODE(temporary.stat().st_mode) != mode:
            raise RuntimeError("Responses serving staged mode verification failed")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verify_file(path: Path, expected: bytes, mode: int) -> None:
    if path.read_bytes() != expected:
        raise RuntimeError("Responses serving published bytes verification failed")
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise RuntimeError("Responses serving published mode verification failed")


def source_state(path: Path) -> str:
    current = sha256(path.read_bytes())
    if current == PREIMAGE_SHA256:
        return "stock"
    if current == POSTIMAGE_SHA256:
        return "applied"
    raise RuntimeError(f"Responses serving source hash drift: sha256={current}")


def apply(path: Path) -> str:
    raw = path.read_bytes()
    current = sha256(raw)
    if current == POSTIMAGE_SHA256:
        return current
    if current != PREIMAGE_SHA256:
        raise RuntimeError(f"Responses serving source hash drift: sha256={current}")

    updated = transformed(raw)
    updated_hash = sha256(updated)
    if updated_hash != POSTIMAGE_SHA256:
        raise RuntimeError(
            f"Responses serving postimage hash drift: sha256={updated_hash}"
        )
    compile(updated, str(path), "exec")

    mode = stat.S_IMODE(path.stat().st_mode)
    try:
        write_atomic(path, updated, mode)
        verify_file(path, updated, mode)
    except Exception:
        try:
            if path.read_bytes() != raw or stat.S_IMODE(path.stat().st_mode) != mode:
                write_atomic(path, raw, mode)
            verify_file(path, raw, mode)
        except Exception as rollback_error:
            raise RuntimeError(
                "Responses serving publication failed and rollback could not be verified"
            ) from rollback_error
        raise
    return updated_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--check",
        "--status",
        action="store_true",
        help="verify exact stock or applied source without modifying it",
    )
    args = parser.parse_args()
    if args.check:
        print(source_state(args.path))
    else:
        print(apply(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
