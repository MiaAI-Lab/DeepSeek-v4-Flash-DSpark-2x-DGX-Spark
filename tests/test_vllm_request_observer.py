import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "monitor" / "observer" / "vllm_request_observer.py"
SITECUSTOMIZE = ROOT / "monitor" / "sitecustomize.py"


def load_script():
    spec = importlib.util.spec_from_file_location("vllm_request_observer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sitecustomize():
    spec = importlib.util.spec_from_file_location("dspark_sitecustomize", SITECUSTOMIZE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Request:
    method = "POST"

    class URL:
        path = "/v1/chat/completions"

    url = URL()

    async def body(self):
        raise AssertionError("observer must never read request content")


class Response:
    def __init__(self):
        async def chunks():
            yield b"data: first\n\n"
            yield b"data: [DONE]\n\n"

        self.body_iterator = chunks()


class VllmRequestObserverTest(unittest.TestCase):
    def setUp(self):
        self.observer = load_script()

    def test_middleware_records_content_free_lifecycle_without_claiming_rank_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.ndjson"
            emitter = self.observer.EventEmitter(events, now=lambda: 100.0)

            async def run():
                response = await self.observer.observe_request_with_emitter(
                    Request(), lambda _request: asyncio.sleep(0, result=Response()), emitter
                )
                return [chunk async for chunk in response.body_iterator]

            chunks = asyncio.run(run())
            rows = [json.loads(line) for line in events.read_text().splitlines()]

        self.assertEqual(chunks, [b"data: first\n\n", b"data: [DONE]\n\n"])
        self.assertEqual(
            [row["lifecycle"] for row in rows],
            ["received", "awaiting_first_output", "streaming", "completed"],
        )
        self.assertTrue(all(row["scope"] == "http_lifecycle" for row in rows))
        serialized = json.dumps(rows)
        for forbidden in ("messages", "prompt", "authorization", "api_key", "data: first"):
            self.assertNotIn(forbidden, serialized.lower())

    def test_atomic_rank_progress_requires_real_rank_instrumentation_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank-progress.json"
            self.observer.write_rank_progress(
                path,
                {
                    "contractVersion": 1,
                    "scope": "rank_worker",
                    "observerRevision": "dspark-rank-observer-v1",
                    "sourceRank": 1,
                    "processGeneration": "worker-generation-2",
                    "sourceSequence": 8,
                    "observedAt": "2026-07-23T20:05:00.000Z",
                    "lifecycle": "serving",
                    "metrics": {
                        "runningRequests": 1,
                        "waitingRequests": 0,
                        "iterationTokens": 120,
                        "generatedTokens": 10,
                        "completedRequests": 2,
                        "requestAttributedKvActivity": 4,
                    },
                },
            )
            payload = json.loads(path.read_text())
        self.assertEqual(payload["scope"], "rank_worker")
        self.assertEqual(payload["sourceRank"], 1)

    def test_rank_hook_counts_only_successful_worker_steps(self):
        hook = load_sitecustomize()
        writes = []

        class SchedulerOutput:
            total_num_scheduled_tokens = 12
            num_scheduled_tokens = {"request-a": 7, "request-b": 5}
            finished_req_ids = {"request-z"}

        class Worker:
            def execute_model(self, scheduler_output):
                return f"completed:{scheduler_output.total_num_scheduled_tokens}"

        installed = hook.install_rank_observer(
            vllm_version="0.25.2.dev0+g752a3a504.d20260714",
            worker_class=Worker,
            source_rank=1,
            observer_revision="dspark-rank-observer-v1",
            capability="c" * 32,
            writer=writes.append,
            now=lambda: "2026-07-23T20:05:00.000Z",
            process_generation="ofus-generation-2",
        )
        result = Worker().execute_model(SchedulerOutput())

        self.assertTrue(installed)
        self.assertEqual(result, "completed:12")
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["metrics"]["iterationTokens"], 12)
        self.assertEqual(writes[0]["metrics"]["runningRequests"], 2)
        self.assertEqual(writes[0]["metrics"]["completedRequests"], 1)
        self.assertNotIn("request-a", json.dumps(writes[0]))

    def test_rank_hook_exception_does_not_publish_progress(self):
        hook = load_sitecustomize()
        writes = []

        class Worker:
            def execute_model(self, scheduler_output):
                raise RuntimeError("worker failed")

        self.assertTrue(
            hook.install_rank_observer(
                vllm_version="0.25.2.dev0+g752a3a504.d20260714",
                worker_class=Worker,
                source_rank=0,
                observer_revision="dspark-rank-observer-v1",
                capability="c" * 32,
                writer=writes.append,
                now=lambda: "2026-07-23T20:05:00.000Z",
                process_generation="john-generation-2",
            )
        )
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            Worker().execute_model(object())
        self.assertEqual(writes, [])

    def test_version_signature_or_capability_mismatch_fails_closed(self):
        hook = load_sitecustomize()

        class Worker:
            def execute_model(self, scheduler_output):
                return None

        original = Worker.execute_model
        for version, capability, worker_class in (
            ("0.25.1", "c" * 32, Worker),
            ("0.25.2.dev0+g752a3a504.d20260714", "short", Worker),
            (
                "0.25.2.dev0+g752a3a504.d20260714",
                "c" * 32,
                type("WrongWorker", (), {"execute_model": lambda self, value, extra: None}),
            ),
        ):
            with self.subTest(version=version, capability=capability):
                self.assertFalse(
                    hook.install_rank_observer(
                        vllm_version=version,
                        worker_class=worker_class,
                        source_rank=0,
                        observer_revision="dspark-rank-observer-v1",
                        capability=capability,
                        writer=lambda _payload: self.fail("must not publish"),
                        now=lambda: "2026-07-23T20:05:00.000Z",
                        process_generation="generation",
                    )
                )
        self.assertIs(Worker.execute_model, original)


if __name__ == "__main__":
    unittest.main()
