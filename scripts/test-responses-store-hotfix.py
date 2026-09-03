#!/usr/bin/env python3
"""CPU regression tests for the bounded Responses API store backport."""

from __future__ import annotations

import ast
import asyncio
from collections import deque
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-responses-store.py"


FIXTURE = '''class OpenAIServingResponses:
    def __init__(self):
        self.enable_store = envs.VLLM_ENABLE_RESPONSES_API_STORE
        if self.enable_store:
            logger.warning_once(
                "`VLLM_ENABLE_RESPONSES_API_STORE` is enabled. This may "
                "cause a memory leak since we never remove responses from "
                "the store."
            )

        self.use_harmony = False

        # HACK(woosuk): This is a hack. We should use a better store.
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

        self.tool_server = tool_server

    def _effective_chat_template_kwargs(
        self,
    ):
        pass

    async def _prepare(self, request, prev_response_id):
        if prev_response_id is not None:
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return self._make_not_found_error(prev_response_id)
        else:
            prev_response = None

        lora_request = self._maybe_get_adapters(request)
        model_name = self.models.model_name(lora_request)

        if self.use_harmony:
            messages, engine_inputs = self._make_request_with_harmony(
                request, prev_response
            )
        else:
            messages, engine_inputs = await self._make_request(request, prev_response)

        # Store the input messages.
        if request.store:
            self.msg_store[request.request_id] = messages

        if request.background:
            response = response_factory()
            async with self.response_store_lock:
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

            if request.stream:
                return self.responses_background_stream_generator(request.request_id)
            return response
        if request.stream:
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

    async def responses_full_generator(self, request, response):
        if request.store:
            async with self.response_store_lock:
                stored_response = self.response_store.get(response.id)
                # If the response is already cancelled, don't update it.
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response
        return response

    async def _run_background_request_stream(self, request, *args, **kwargs):
        event_deque: deque[StreamingResponsesResponse] = deque()
        new_event_signal = asyncio.Event()
        self.event_store[request.request_id] = (event_deque, new_event_signal)
        generator = self.responses_stream_generator(request, *args, **kwargs)
        try:
            async for event in generator:
                event_deque.append(event)
                new_event_signal.set()  # Signal new event available
        finally:
            new_event_signal.set()

    async def responses_background_stream_generator(
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

    async def retrieve_responses(self, response_id, starting_after, stream):
        async with self.response_store_lock:
            response = self.response_store.get(response_id)

        if response is None:
            return self._make_not_found_error(response_id)

        if stream:
            return self.responses_background_stream_generator(
                response_id,
                starting_after,
            )
        return response

    async def cancel_responses(self, response_id):
        async with self.response_store_lock:
            response = self.response_store[response_id]
            # Update the status to "cancelled".
            response.status = "cancelled"

        # Abort the request.
        return response
'''


def load_hotfix():
    spec = importlib.util.spec_from_file_location("responses_store_hotfix", HOTFIX)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def method_from_source(source: str, name: str, namespace=None):
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingResponses"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    globals_dict = {
        "AsyncGenerator": object,
        "ResponsesResponse": object,
        "StreamingResponsesResponse": object,
        "asyncio": asyncio,
        "deque": deque,
        "logger": types.SimpleNamespace(warning_once=lambda *args: None),
        "tool_server": None,
    }
    if namespace:
        globals_dict.update(namespace)
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<responses-store-method>", "exec"), globals_dict)
    return globals_dict[name]


def response(response_id: str, status: str = "completed"):
    return types.SimpleNamespace(id=response_id, status=status)


async def collect_async(generator):
    return [item async for item in generator]


class ResponsesStoreHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotfix = load_hotfix()
        cls.source = cls.hotfix.transformed(FIXTURE.encode()).decode()
        compile(cls.source, "<responses-store-fixture>", "exec")

    def make_store(self, capacity=2):
        probe = types.SimpleNamespace(
            response_store={},
            response_store_lock=asyncio.Lock(),
            response_store_pins={},
            msg_store={},
            event_store={},
            background_tasks={},
            responses_store_max_entries=capacity,
        )
        for name in (
            "_remove_stored_response",
            "_evict_stored_responses_locked",
            "_store_response",
            "_acquire_stored_response",
            "_release_stored_response",
            "_background_task_done",
        ):
            method = method_from_source(self.source, name)
            setattr(probe, name, types.MethodType(method, probe))
        return probe

    def test_constructor_ignores_disabled_capacity_and_validates_enabled(self):
        initializer = method_from_source(
            self.source,
            "__init__",
            {"envs": types.SimpleNamespace(VLLM_ENABLE_RESPONSES_API_STORE=False)},
        )
        with mock.patch.dict(
            os.environ,
            {"DSPARK_RESPONSES_STORE_MAX_ENTRIES": "not-an-int"},
            clear=False,
        ):
            disabled = types.SimpleNamespace()
            initializer(disabled)
        self.assertEqual(disabled.responses_store_max_entries, 256)

        enabled_init = method_from_source(
            self.source,
            "__init__",
            {"envs": types.SimpleNamespace(VLLM_ENABLE_RESPONSES_API_STORE=True)},
        )
        with mock.patch.dict(
            os.environ, {"DSPARK_RESPONSES_STORE_MAX_ENTRIES": "0"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "greater than 0"):
                enabled_init(types.SimpleNamespace())

    def test_eviction_removes_one_complete_bundle_and_keeps_active(self):
        probe = self.make_store(capacity=2)
        for response_id in ("old", "middle", "new"):
            probe.response_store[response_id] = response(response_id)
            probe.msg_store[response_id] = [response_id]
            probe.event_store[response_id] = (deque(), asyncio.Event(), asyncio.Event())
        probe.response_store["active"] = response("active", "in_progress")
        probe.msg_store["active"] = ["active"]

        probe._evict_stored_responses_locked()

        self.assertEqual(list(probe.response_store), ["middle", "new", "active"])
        self.assertEqual(set(probe.msg_store), set(probe.response_store))
        self.assertEqual(set(probe.event_store), {"middle", "new"})

    def test_continuation_pin_prevents_eviction_until_preprocessing_finishes(self):
        async def exercise():
            probe = self.make_store(capacity=1)
            await probe._store_response(response("continued"))
            probe.msg_store["continued"] = ["history"]
            pinned = await probe._acquire_stored_response("continued")
            self.assertIsNotNone(pinned)
            await probe._store_response(response("new"))
            self.assertEqual(probe.msg_store["continued"], ["history"])
            self.assertEqual(set(probe.response_store), {"continued", "new"})
            await probe._release_stored_response("continued")
            self.assertEqual(set(probe.response_store), {"new"})
            self.assertNotIn("continued", probe.msg_store)

        asyncio.run(exercise())

    def test_retrieve_refreshes_lru_before_terminal_eviction(self):
        async def exercise():
            probe = self.make_store(capacity=2)
            await probe._store_response(response("old"))
            await probe._store_response(response("new"))
            retrieve = method_from_source(self.source, "retrieve_responses")

            retrieved = await retrieve(probe, "old", None, False)
            self.assertEqual(retrieved.id, "old")
            self.assertEqual(list(probe.response_store), ["new", "old"])

            await probe._store_response(response("rollover"))
            self.assertEqual(list(probe.response_store), ["old", "rollover"])

            class ValidationError(Exception):
                def __init__(self, message, **_):
                    super().__init__(message)

            reader = method_from_source(
                self.source,
                "responses_background_stream_generator",
                {"VLLMValidationError": ValidationError},
            )
            probe.responses_background_stream_generator = types.MethodType(
                reader, probe
            )
            missing_event_stream = await retrieve(probe, "old", None, True)
            with self.assertRaisesRegex(ValidationError, "Unknown response_id"):
                await collect_async(missing_event_stream)

        asyncio.run(exercise())

    def test_terminal_completion_does_not_overwrite_cancelled_status(self):
        async def exercise():
            probe = self.make_store(capacity=1)
            probe.response_store["request"] = response("request", "cancelled")
            await probe._store_response(
                response("request", "completed"),
                preserve_cancelled=True,
            )
            self.assertEqual(probe.response_store["request"].status, "cancelled")

        asyncio.run(exercise())


    def test_tracked_terminal_producer_is_pruned_only_after_callback(self):
        async def exercise():
            probe = self.make_store(capacity=1)
            probe.response_store["producer"] = response("producer")
            probe.msg_store["producer"] = ["producer"]
            probe.event_store["producer"] = (
                deque(),
                asyncio.Event(),
                asyncio.Event(),
            )

            release = asyncio.Event()

            async def pending_producer():
                await release.wait()

            task = asyncio.create_task(pending_producer())
            probe.background_tasks["producer"] = task
            task.add_done_callback(
                lambda completed: probe._background_task_done(
                    "producer", completed
                )
            )

            await probe._store_response(response("other"))
            self.assertEqual(set(probe.response_store), {"producer", "other"})

            release.set()
            await task
            await asyncio.sleep(0)
            self.assertEqual(set(probe.response_store), {"producer"})
            self.assertNotIn("producer", probe.background_tasks)

        asyncio.run(exercise())

    def test_foreground_stream_retains_only_successful_complete_history(self):
        async def exercise():
            wrapper = method_from_source(
                self.source, "_foreground_stream_with_cleanup"
            )
            probe = self.make_store()

            async def never_started():
                yield "unused"

            idle = wrapper(probe, "idle", ["history"], never_started())
            self.assertNotIn("idle", probe.msg_store)
            await idle.aclose()
            self.assertNotIn("idle", probe.msg_store)

            async def closes_early():
                yield "first"
                yield "second"

            early = wrapper(probe, "early", ["history"], closes_early())
            self.assertEqual(await anext(early), "first")
            self.assertEqual(probe.msg_store["early"], ["history"])
            await early.aclose()
            self.assertNotIn("early", probe.msg_store)

            async def fails():
                yield "first"
                raise RuntimeError("stream failed")

            failed = wrapper(probe, "failed", ["history"], fails())
            self.assertEqual(await anext(failed), "first")
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await anext(failed)
            self.assertNotIn("failed", probe.msg_store)

            async def completes():
                yield "first"
                probe.response_store["complete"] = response("complete")

            complete = wrapper(probe, "complete", ["history"], completes())
            self.assertEqual([item async for item in complete], ["first"])
            self.assertEqual(probe.msg_store["complete"], ["history"])

        asyncio.run(exercise())


    def test_background_cleanup_terminalizes_and_wakes_failed_stream(self):
        async def exercise():
            probe = self.make_store(capacity=1)
            probe.response_store["request"] = response("request", "queued")
            events = (
                deque([types.SimpleNamespace(type="response.output")]),
                asyncio.Event(),
                asyncio.Event(),
            )
            probe.event_store["request"] = events

            async def ends_without_terminal_event():
                return None

            task = asyncio.create_task(ends_without_terminal_event())
            probe.background_tasks["request"] = task
            task.add_done_callback(
                lambda completed: probe._background_task_done("request", completed)
            )
            await task
            await asyncio.sleep(0)
            self.assertEqual(probe.response_store["request"].status, "failed")
            self.assertNotIn("request", probe.background_tasks)
            self.assertTrue(events[1].is_set())
            self.assertTrue(events[2].is_set())

            reader = method_from_source(
                self.source, "responses_background_stream_generator"
            )
            yielded = [
                item.type
                async for item in reader(
                    probe,
                    "request",
                    event_state=events,
                )
            ]
            self.assertEqual(yielded, ["response.output"])

        asyncio.run(exercise())

    def test_background_cancellation_terminalizes_and_unblocks_waiting_reader(self):
        async def exercise():
            probe = self.make_store(capacity=1)
            probe.response_store["cancelled"] = response("cancelled", "in_progress")
            events = (deque(), asyncio.Event(), asyncio.Event())
            probe.event_store["cancelled"] = events
            reader = method_from_source(
                self.source, "responses_background_stream_generator"
            )
            reader_task = asyncio.create_task(
                collect_async(
                    reader(probe, "cancelled", event_state=events)
                )
            )
            await asyncio.sleep(0)

            task = asyncio.create_task(asyncio.sleep(60))
            probe.background_tasks["cancelled"] = task
            task.add_done_callback(
                lambda completed: probe._background_task_done(
                    "cancelled", completed
                )
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            self.assertEqual(probe.response_store["cancelled"].status, "cancelled")
            self.assertNotIn("cancelled", probe.background_tasks)
            self.assertTrue(events[2].is_set())
            self.assertEqual(await asyncio.wait_for(reader_task, 1), [])

        asyncio.run(exercise())

    def test_completed_background_stream_reader_uses_captured_state(self):
        async def exercise():
            reader = method_from_source(
                self.source, "responses_background_stream_generator"
            )
            state = (
                deque(
                    [
                        types.SimpleNamespace(type="response.output"),
                        types.SimpleNamespace(type="response.completed"),
                    ]
                ),
                asyncio.Event(),
                asyncio.Event(),
            )
            state[2].set()
            yielded = [
                item.type
                async for item in reader(
                    types.SimpleNamespace(), "evicted", event_state=state
                )
            ]
            self.assertEqual(
                yielded, ["response.output", "response.completed"]
            )

        asyncio.run(exercise())

    def test_atomic_apply_is_idempotent_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "serving.py"
            before = b"value = 1\n"
            after = b"value = 2\n"
            target.write_bytes(before)
            target.chmod(0o640)
            with (
                mock.patch.object(
                    self.hotfix, "PREIMAGE_SHA256", hashlib.sha256(before).hexdigest()
                ),
                mock.patch.object(
                    self.hotfix, "POSTIMAGE_SHA256", hashlib.sha256(after).hexdigest()
                ),
                mock.patch.object(self.hotfix, "transformed", return_value=after),
            ):
                self.assertEqual(
                    self.hotfix.apply(target), hashlib.sha256(after).hexdigest()
                )
                self.assertEqual(
                    self.hotfix.apply(target), hashlib.sha256(after).hexdigest()
                )
            self.assertEqual(target.read_bytes(), after)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            self.assertEqual(list(target.parent.glob(".*.responses-store.*")), [])

    def test_publication_verification_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "serving.py"
            before = b"value = 1\n"
            after = b"value = 2\n"
            target.write_bytes(before)
            original_verify = self.hotfix.verify_file
            calls = 0

            def fail_once(path, expected, mode):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("injected verification failure")
                original_verify(path, expected, mode)

            with (
                mock.patch.object(
                    self.hotfix, "PREIMAGE_SHA256", hashlib.sha256(before).hexdigest()
                ),
                mock.patch.object(
                    self.hotfix, "POSTIMAGE_SHA256", hashlib.sha256(after).hexdigest()
                ),
                mock.patch.object(self.hotfix, "transformed", return_value=after),
                mock.patch.object(self.hotfix, "verify_file", side_effect=fail_once),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self.hotfix.apply(target)
            self.assertEqual(target.read_bytes(), before)

    def test_source_drift_and_anchor_drift_are_non_mutating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "serving.py"
            target.write_text("drifted = True\n")
            before = target.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "source hash drift"):
                self.hotfix.apply(target)
            self.assertEqual(target.read_bytes(), before)
        with self.assertRaisesRegex(RuntimeError, "anchor drift"):
            self.hotfix.transformed(b"class OpenAIServingResponses:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
