import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "monitor" / "recovery" / "recover_deepseek_serving.py"


def load_script():
    spec = importlib.util.spec_from_file_location("recover_deepseek_serving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeExecutor:
    def __init__(self, inspections, failures=None, ambiguous=None):
        self.inspections = inspections
        self.failures = set(failures or ())
        self.ambiguous = set(ambiguous or ())
        self.calls = []
        self.generations = {"john": "john-old", "ofus": "ofus-old"}

    def __call__(self, phase, host, context):
        self.calls.append((phase, host))
        key = (phase, host)
        if key in self.ambiguous:
            raise RuntimeError("ambiguous_transport_result")
        if key in self.failures:
            raise RuntimeError(f"{phase}_failed")
        if phase == "inspect":
            return dict(self.inspections[host])
        if phase == "confirm_stopped":
            return {"stopped": True}
        if phase == "start":
            self.generations[host] = f"{host}-new"
            return {"started": True}
        if phase == "wait_fresh_generation":
            return {"processGeneration": self.generations[host]}
        if phase == "stream_verify":
            return {"firstEvent": True, "completed": True, "requestCount": 1}
        return {"ok": True}


class RecoverDeepseekServingTest(unittest.TestCase):
    def setUp(self):
        self.recovery = load_script()
        self.now = datetime(2026, 7, 23, 20, 5, tzinfo=timezone.utc)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.expected = {
            "repositoryRevision": "05a35c5766b2568ad92f31c4a8d1f2d691ca3329",
            "imageDigest": "sha256:" + "1" * 64,
            "configurationHash": "2" * 64,
            "artifactRevision": "ds4-artifact-v1",
            "observerRevision": "dspark-rank-observer-v1",
            "adapterRevision": "deepseek-recovery-v1",
        }
        self.manifest = {
            "contractVersion": 1,
            "targetId": "deepseek-v4-flash-dspark",
            "topology": "deepseek_tp2",
            **self.expected,
            "ranks": {
                "john": {"sourceRank": 0},
                "ofus": {"sourceRank": 1},
            },
        }
        self.manifest_hash = self.recovery.hash_manifest(self.manifest)
        self.authorization = {
            "contractVersion": 1,
            "incidentGeneration": 7,
            "targetId": "deepseek-v4-flash-dspark",
            "manifestHash": self.manifest_hash,
            "expiresAt": (self.now + timedelta(minutes=15)).isoformat().replace("+00:00", "Z"),
            "commandNonce": "nonce-1234567890",
        }
        self.inspections = {
            host: {
                **self.expected,
                "hostId": host,
                "sourceRank": rank,
                "targetId": "deepseek-v4-flash-dspark",
                "manifestHash": self.manifest_hash,
                "incidentGeneration": 7,
                "commandNonce": "nonce-1234567890",
                "observedAt": self.now.isoformat().replace("+00:00", "Z"),
                "processGeneration": f"{host}-old",
            }
            for host, rank in (("john", 0), ("ofus", 1))
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self):
        manifest = self.root / "manifest.json"
        authorization = self.root / "authorization.json"
        manifest.write_text(json.dumps(self.manifest))
        authorization.write_text(json.dumps(self.authorization))
        return manifest, authorization

    def recover(self, executor, **kwargs):
        manifest, authorization = self.write_inputs()
        return self.recovery.recover(
            manifest_path=manifest,
            authorization_path=authorization,
            receipt_dir=self.root / "receipts",
            now=self.now,
            execute=True,
            executor=executor,
            **kwargs,
        )

    def test_dry_run_preflights_both_ranks_and_performs_no_mutation(self):
        manifest, authorization = self.write_inputs()
        executor = FakeExecutor(self.inspections)
        result = self.recovery.recover(
            manifest_path=manifest,
            authorization_path=authorization,
            receipt_dir=self.root / "receipts",
            now=self.now,
            executor=executor,
        )
        self.assertEqual(result["status"], "validated_dry_run")
        self.assertEqual(executor.calls, [("inspect", "john"), ("inspect", "ofus")])
        self.assertFalse((self.root / "receipts").exists())

    def test_drift_on_either_rank_cancels_before_stop(self):
        for field in ("repositoryRevision", "imageDigest", "configurationHash", "artifactRevision"):
            with self.subTest(field=field):
                inspections = {host: dict(value) for host, value in self.inspections.items()}
                inspections["ofus"][field] = "drift"
                executor = FakeExecutor(inspections)
                with self.assertRaisesRegex(self.recovery.RecoveryError, field):
                    self.recover(executor)
                self.assertFalse(any(phase == "stop" for phase, _host in executor.calls))

    def test_stale_or_invocation_unbound_inspection_cancels_before_stop(self):
        for field, value in (
            (
                "observedAt",
                (self.now - timedelta(seconds=46)).isoformat().replace("+00:00", "Z"),
            ),
            ("manifestHash", "f" * 64),
            ("incidentGeneration", 8),
            ("commandNonce", "different-nonce"),
        ):
            with self.subTest(field=field):
                inspections = {host: dict(value) for host, value in self.inspections.items()}
                inspections["ofus"][field] = value
                executor = FakeExecutor(inspections)
                with self.assertRaises(self.recovery.RecoveryError):
                    self.recover(executor)
                self.assertFalse(any(phase == "stop" for phase, _host in executor.calls))

    def test_stop_failure_opens_circuit_and_never_starts(self):
        executor = FakeExecutor(self.inspections, failures={("confirm_stopped", "ofus")})
        with self.assertRaisesRegex(self.recovery.RecoveryError, "confirm_stopped_failed"):
            self.recover(executor)
        self.assertFalse(any(phase == "start" for phase, _host in executor.calls))
        receipt = json.loads(next((self.root / "receipts").glob("*.json")).read_text())
        self.assertEqual(receipt["status"], "circuit_open")

    def test_worker_first_success_has_fresh_generations_and_one_stream(self):
        executor = FakeExecutor(self.inspections)
        result = self.recover(executor)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [call for call in executor.calls if call[0] in {"start", "stream_verify"}],
            [("start", "ofus"), ("start", "john"), ("stream_verify", "john")],
        )
        self.assertEqual(result["rankGenerations"], {"john": "john-new", "ofus": "ofus-new"})
        receipt = json.loads(next((self.root / "receipts").glob("*.json")).read_text())
        self.assertEqual(receipt["streamVerification"]["requestCount"], 1)

        with self.assertRaisesRegex(self.recovery.RecoveryError, "generation_already_claimed"):
            self.recover(executor)
        self.assertEqual(
            [call for call in executor.calls if call[0] == "stream_verify"],
            [("stream_verify", "john")],
        )

    def test_worker_readiness_failure_prevents_head_start(self):
        executor = FakeExecutor(
            self.inspections, failures={("wait_fresh_generation", "ofus")}
        )
        with self.assertRaisesRegex(self.recovery.RecoveryError, "wait_fresh_generation_failed"):
            self.recover(executor)
        self.assertNotIn(("start", "john"), executor.calls)

    def test_ambiguous_action_is_receipted_and_never_retried(self):
        executor = FakeExecutor(self.inspections, ambiguous={("start", "ofus")})
        with self.assertRaisesRegex(self.recovery.RecoveryError, "ambiguous_outcome"):
            self.recover(executor)
        receipt = json.loads(next((self.root / "receipts").glob("*.json")).read_text())
        self.assertEqual(receipt["status"], "ambiguous")
        with self.assertRaisesRegex(self.recovery.RecoveryError, "generation_already_claimed"):
            self.recover(executor)
        self.assertEqual(executor.calls.count(("start", "ofus")), 1)

    def test_expired_authorization_performs_no_action(self):
        self.authorization["expiresAt"] = (self.now - timedelta(seconds=1)).isoformat().replace(
            "+00:00", "Z"
        )
        executor = FakeExecutor(self.inspections)
        with self.assertRaisesRegex(self.recovery.RecoveryError, "authorization_expired"):
            self.recover(executor)
        self.assertEqual(executor.calls, [])

    def test_deadline_expiry_after_claim_opens_circuit_and_cannot_retry(self):
        executor = FakeExecutor(self.inspections)
        after_expiry = self.now + timedelta(minutes=16)
        with self.assertRaisesRegex(self.recovery.RecoveryError, "authorization_expired"):
            self.recover(executor, clock=lambda: after_expiry)
        receipt = json.loads(next((self.root / "receipts").glob("*.json")).read_text())
        self.assertEqual(receipt["status"], "circuit_open")
        self.assertFalse(any(phase == "start" for phase, _host in executor.calls))
        with self.assertRaisesRegex(self.recovery.RecoveryError, "generation_already_claimed"):
            self.recover(executor)


if __name__ == "__main__":
    unittest.main()
