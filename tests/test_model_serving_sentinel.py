import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "monitor" / "sentinel" / "model_serving_sentinel.py"


def load_script():
    spec = importlib.util.spec_from_file_location("model_serving_sentinel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelServingSentinelTest(unittest.TestCase):
    def setUp(self):
        self.sentinel = load_script()
        self.now = datetime(2026, 7, 23, 20, 5, tzinfo=timezone.utc)
        self.progress = {
            "contractVersion": 1,
            "scope": "rank_worker",
            "observerRevision": "dspark-rank-observer-v1",
            "sourceRank": 1,
            "processGeneration": "worker-generation-2",
            "sourceSequence": 8,
            "observedAt": self.now.isoformat().replace("+00:00", "Z"),
            "lifecycle": "serving",
            "metrics": {
                "runningRequests": 1,
                "waitingRequests": 1,
                "iterationTokens": 120,
                "generatedTokens": 10,
                "completedRequests": 2,
                "requestAttributedKvActivity": 4,
            },
        }

    def build(self, progress=None, *, rank=1, generation="worker-generation-2", state=None):
        return self.sentinel.build_evidence(
            progress=self.progress if progress is None else progress,
            target_id="deepseek-v4-flash-dspark",
            observer_id=f"ofus-rank{rank}",
            observer_revision="dspark-rank-observer-v1",
            source_host="ofus",
            source_rank=rank,
            source_boot_id="boot-ofus",
            process_generation=generation,
            configuration_revision="b" * 64,
            manifest_hash="a" * 64,
            prior_state=state or {},
            now=self.now,
        )

    def test_emits_exact_independent_u4_rank_evidence(self):
        evidence, state = self.build()
        self.assertEqual(evidence["topology"], "deepseek_tp2")
        self.assertEqual(evidence["sourceRank"], 1)
        self.assertEqual(evidence["observerId"], "ofus-rank1")
        self.assertEqual(evidence["processGeneration"], "worker-generation-2")
        self.assertEqual(evidence["metrics"]["iterationTokens"], 120)
        self.assertEqual(evidence["sourceSequence"], 1)
        self.assertEqual(state["sourceSequence"], 1)
        self.assertNotIn("scope", evidence)

    def test_stale_rank_progress_fails_closed(self):
        stale = dict(self.progress)
        stale["observedAt"] = (self.now - timedelta(seconds=46)).isoformat().replace("+00:00", "Z")
        evidence, _ = self.build(stale)
        self.assertEqual(evidence["lifecycle"], "unknown")
        self.assertEqual(evidence["metrics"]["iterationTokens"], 0)

    def test_lifecycle_only_or_asymmetric_revision_cannot_synthesize_rank_progress(self):
        for mutation in (
            {"scope": "http_lifecycle"},
            {"observerRevision": "other"},
            {"sourceRank": 0},
            {"processGeneration": "old-generation"},
        ):
            with self.subTest(mutation=mutation):
                progress = dict(self.progress)
                progress.update(mutation)
                evidence, _ = self.build(progress)
                self.assertEqual(evidence["lifecycle"], "unknown")
                self.assertEqual(evidence["metrics"]["generatedTokens"], 0)

    def test_atomic_output_is_bounded_and_contains_no_payload(self):
        evidence, _ = self.build()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            self.sentinel.atomic_write_json(output, evidence)
            self.assertEqual(json.loads(output.read_text()), evidence)
            with self.assertRaises(ValueError):
                self.sentinel.atomic_write_json(output, {"prompt": "x" * 20_000})


if __name__ == "__main__":
    unittest.main()
