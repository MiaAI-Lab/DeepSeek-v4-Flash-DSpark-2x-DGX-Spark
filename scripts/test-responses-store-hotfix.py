#!/usr/bin/env python3
"""CPU regression tests for the bounded Responses store backport."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-responses-store.py"


def load_hotfix():
    spec = importlib.util.spec_from_file_location("responses_store_hotfix", HOTFIX)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def method_from_source(source: str, name: str):
    tree = ast.parse(source)
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingResponses")
    method = next(node for node in owner.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name == name)
    namespace: dict[str, object] = {}
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<responses-store-method>", "exec"), namespace)
    return namespace[name]


class ResponsesStoreHotfixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotfix = load_hotfix()
        fixture = '''class OpenAIServingResponses:
    def __init__(self):
        self.background_tasks: dict[str, asyncio.Task] = {}

        self.tool_server = tool_server

    def _effective_chat_template_kwargs(
        self,
    ):
        pass

    async def _prepare(self, request, messages, prev_response_id):
        if prev_response_id is not None:
            async with self.response_store_lock:
                prev_response = self.response_store.get(prev_response_id)
            if prev_response is None:
                return None

        # Store the input messages.
        if request.store:
            self.msg_store[request.request_id] = messages

        if request.background:
            async with self.response_store_lock:
                self.response_store[response.id] = response

            # Run the request in the background.
            self.background_tasks[response_id] = task
            task.add_done_callback(
                lambda _: self.background_tasks.pop(response_id, None)
            )

            if request.stream:
                return stream

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

    async def responses_full_generator(self, response, stored_response):
        async with self.response_store_lock:
                if stored_response is None or stored_response.status != "cancelled":
                    self.response_store[response.id] = response
        return response
'''
        cls.source = cls.hotfix.transformed(fixture.encode()).decode()

    def test_prune_bounds_terminal_records_and_keeps_active_work(self):
        prune = method_from_source(self.source, "_prune_response_store_locked")
        probe = types.SimpleNamespace(
            response_store={}, msg_store={}, event_store={}, background_tasks={},
            active_store_requests=set(), responses_store_max_entries=3,
            response_store_lock=asyncio.Lock(),
        )
        for index in range(5):
            response_id = f"resp-{index}"
            probe.response_store[response_id] = types.SimpleNamespace(status="completed")
            probe.msg_store[response_id] = [response_id]
            probe.event_store[response_id] = ([], None)
        probe.response_store["active"] = types.SimpleNamespace(status="in_progress")
        probe.msg_store["active"] = ["active"]
        probe.active_store_requests.add("active")
        prune(probe)
        self.assertEqual(list(probe.response_store), ["resp-2", "resp-3", "resp-4", "active"])
        self.assertEqual(set(probe.msg_store), set(probe.response_store))
        self.assertEqual(set(probe.event_store), {"resp-2", "resp-3", "resp-4"})

    def test_background_exception_and_cancel_reach_terminal_status(self):
        done = method_from_source(self.source, "_background_store_task_done")
        probe = types.SimpleNamespace(
            response_store={}, msg_store={}, event_store={}, background_tasks={},
            active_store_requests=set(), responses_store_max_entries=3,
            response_store_lock=asyncio.Lock(),
        )
        prune = method_from_source(self.source, "_prune_response_store_locked")
        probe._prune_response_store_locked = types.MethodType(prune, probe)

        async def exercise():
            async def raises():
                raise RuntimeError("producer failed")

            failed = asyncio.create_task(raises())
            await asyncio.sleep(0)
            probe.response_store["failed"] = types.SimpleNamespace(status="queued")
            probe.background_tasks["failed"] = failed
            await done(probe, "failed", failed)
            self.assertEqual(probe.response_store["failed"].status, "failed")

            cancelled = asyncio.create_task(asyncio.sleep(60))
            probe.response_store["cancelled"] = types.SimpleNamespace(status="in_progress")
            probe.background_tasks["cancelled"] = cancelled
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            await done(probe, "cancelled", cancelled)
            self.assertEqual(probe.response_store["cancelled"].status, "cancelled")

        asyncio.run(exercise())

    def test_capacity_is_read_from_dspark_env_and_must_be_positive(self):
        self.assertIn('DSPARK_RESPONSES_STORE_MAX_ENTRIES', self.source)
        self.assertIn('responses_store_max_entries <= 0', self.source)

    def test_anchor_drift_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "anchor drift"):
            self.hotfix.transformed(b"class OpenAIServingResponses:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
