#!/usr/bin/env python3
"""Fake-client contract tests for the issue #138 live verifier."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify-issue138-responses-history-live.py"
NONCE = "fixed_nonce"
T1 = f"I138_T1_{NONCE}"
T2 = f"I138_T2_{NONCE}"


def load_verifier():
    spec = importlib.util.spec_from_file_location("issue138_live", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def output_message(text):
    return {
        "type": "message",
        "id": "msg_fake",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
        }],
    }


def success(text, *, output=None):
    return 200, {
        "id": "resp_fake",
        "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
        "output": output if output is not None else [output_message(text)],
    }


def rejection():
    return 400, {
        "error": {
            "code": 400,
            "message": (
                "240 validation errors: assistant output_text item is missing "
                "the output message type"
            ),
            "type": "BadRequestError",
        }
    }


class FakeClient:
    api_path = "/v1"

    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.turn1_output = [
            {
                "type": "reasoning",
                "id": "rs_fake",
                "summary": [],
                "status": "completed",
            },
            output_message(T1),
        ]

    def request_json(self, method, path, body=None):
        self.calls.append(json.loads(json.dumps(body)))
        index = len(self.calls) - 1
        if method != "POST" or path != "/v1/responses":
            raise AssertionError((method, path))
        if index == 0:
            return success(T1, output=self.turn1_output)
        if index == 1:
            return success(T2)
        if index == 2:
            return rejection() if self.mode == "rejected" else success(T2)
        if index == 3:
            return rejection() if self.mode == "rejected" else success("DeepSeek V4")
        if self.mode == "rejected":
            raise AssertionError(f"unexpected rejected-mode call {index}")
        if index == 4:
            return success(NONCE)
        if 5 <= index <= 10:
            return rejection()
        if 11 <= index <= 13:
            return success(T2)
        raise AssertionError(f"unexpected accepted-mode call {index}")


class WrongLegacyOutcomeClient(FakeClient):
    def request_json(self, method, path, body=None):
        status, value = super().request_json(method, path, body)
        if len(self.calls) == 3:
            return success(T2) if self.mode == "rejected" else rejection()
        return status, value


class Issue138LiveVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_verifier()

    def test_rejected_mode_freezes_issue_payload_and_exact_400_contract(self):
        client = FakeClient("rejected")
        result = self.module.verify(
            client, "deepseek-v4-flash-0731", "rejected", nonce=NONCE
        )
        self.assertEqual(result["expected_legacy"], "rejected")
        self.assertEqual(result["store"], False)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(
            set(result["gates"]),
            {"turn1", "canonical_replay", "legacy_replay", "reported_issue_fixture"},
        )
        self.assertEqual(client.calls[3]["input"], self.module.ISSUE_INPUT)
        issue_assistant = client.calls[3]["input"][2]
        self.assertNotIn("type", issue_assistant)
        self.assertNotIn("id", issue_assistant)
        self.assertNotIn("status", issue_assistant)
        self.assertNotIn("annotations", issue_assistant["content"][0])

    def test_accepted_mode_freezes_two_turn_semantics_negatives_and_controls(self):
        client = FakeClient("accepted")
        result = self.module.verify(
            client, "deepseek-v4-flash-0731", "accepted", nonce=NONCE
        )
        self.assertEqual(result["expected_legacy"], "accepted")
        self.assertEqual(len(client.calls), 14)
        self.assertEqual(
            client.calls[1]["input"][2:4],
            client.turn1_output,
            "canonical replay must retain the complete real output array",
        )

        legacy = client.calls[2]
        self.assertEqual(len(legacy["input"]), 4)
        legacy_item = legacy["input"][2]
        self.assertEqual(
            legacy_item,
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": T1}],
            },
        )

        semantic = client.calls[4]["input"]
        self.assertEqual(len(semantic), 3)
        semantic_legacy_item = semantic[1]
        self.assertEqual(
            semantic_legacy_item,
            {
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": f"The launch code is {NONCE}.",
                }],
            },
        )
        self.assertEqual(
            semantic[2],
            {
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "What launch code did the immediately preceding assistant "
                        "state? Reply with exactly the code and no other text."
                    ),
                }],
            },
        )
        self.assertIn("assistant_semantic_continuity", result["gates"])

        negative_names = list(self.module.NEGATIVE_ITEMS)
        self.assertEqual(len(negative_names), 6)
        for offset, name in enumerate(negative_names, start=5):
            self.assertEqual(
                client.calls[offset]["input"][1],
                self.module.NEGATIVE_ITEMS[name],
            )
            self.assertEqual(result["gates"][f"negative_{name}"], "PASS")

        control_names = list(self.module.POSITIVE_CONTROL_ITEMS)
        for offset, name in enumerate(control_names, start=11):
            self.assertEqual(
                client.calls[offset]["input"][1],
                self.module.POSITIVE_CONTROL_ITEMS[name],
            )
            self.assertEqual(result["gates"][f"positive_{name}"], "PASS")

        for body in client.calls:
            self.assertIs(body["store"], False)
            self.assertEqual(body["model"], "deepseek-v4-flash-0731")
            self.assertEqual(body["max_output_tokens"], 256)
            self.assertEqual(body["reasoning"], {"effort": "none"})
            self.assertEqual(body["temperature"], 0)

    def test_expected_mode_is_an_assertion_not_either_outcome_passes(self):
        with self.assertRaisesRegex(self.module.VerificationError, "expected HTTP 400"):
            self.module.verify(
                WrongLegacyOutcomeClient("rejected"),
                "model",
                "rejected",
                nonce=NONCE,
            )
        with self.assertRaisesRegex(self.module.VerificationError, "expected HTTP 200"):
            self.module.verify(
                WrongLegacyOutcomeClient("accepted"),
                "model",
                "accepted",
                nonce=NONCE,
            )
        with self.assertRaises(ValueError):
            self.module.verify(FakeClient("accepted"), "model", "either", nonce=NONCE)

    def test_response_shape_requires_completed_single_text_and_integer_usage(self):
        _status, valid = success("ok")
        self.assertEqual(self.module.completed_output_text(valid), "ok")
        malformed = json.loads(json.dumps(valid))
        malformed["usage"]["input_tokens"] = True
        with self.assertRaisesRegex(self.module.VerificationError, "integer usage"):
            self.module.completed_output_text(malformed)
        malformed = json.loads(json.dumps(valid))
        malformed["output"].append(output_message("second"))
        with self.assertRaisesRegex(self.module.VerificationError, "exactly one"):
            self.module.completed_output_text(malformed)

    def test_validation_error_contract_requires_structured_numeric_400_and_locus(self):
        status, value = rejection()
        self.module.require_validation_400(status, value)
        self.module.require_legacy_locus(value)
        bad_code = json.loads(json.dumps(value))
        bad_code["error"]["code"] = "400"
        with self.assertRaisesRegex(self.module.VerificationError, "numeric 400"):
            self.module.require_validation_400(400, bad_code)
        with self.assertRaisesRegex(self.module.VerificationError, "expected HTTP 400"):
            self.module.require_validation_400(422, value)
        no_locus = json.loads(json.dumps(value))
        no_locus["error"]["message"] = "bad request"
        with self.assertRaisesRegex(self.module.VerificationError, "locus"):
            self.module.require_legacy_locus(no_locus)


if __name__ == "__main__":
    unittest.main()
