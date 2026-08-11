from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ISSUE21 = ROOT / "patches" / "hotfix-encoding-dsv4-issue21.py"
ISSUE22 = ROOT / "patches" / "hotfix-nvfp4-ds-mla-issue22.py"
VERIFIER = ROOT / "scripts" / "verify-runtime-hotfixes.py"
LONG_CONTEXT_PROBE = ROOT / "scripts" / "probe-long-context-decode.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_encoder(broken: bool = True) -> str:
    body = (
        '''    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = {"arguments": tool_call["arguments"]}'''
        if broken
        else '''    raw = tool_call["arguments"]
    if isinstance(raw, dict):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            arguments = {"arguments": raw}'''
    )
    return f'''import json

def encode_arguments_to_dsml(tool_call):
    parts = []
{body}
    for key, value in arguments.items():
        parts.append((key, value))
    return parts
'''


class RuntimeHotfixTest(unittest.TestCase):
    def test_long_context_prompt_forces_enough_deterministic_decode(self):
        probe = load_module("long_context_probe", LONG_CONTEXT_PROBE)
        with mock.patch.object(probe, "tokenize", return_value=600_000):
            prompt, observed = probe.build_prompt(
                "http://127.0.0.1:8888/v1",
                "secret",
                "model",
                600_000,
                1,
                "rollout-57f75ac",
            )
        self.assertEqual(observed, 600_000)
        self.assertTrue(prompt.startswith(" dspark-probe-nonce-rollout-57f75ac"))
        self.assertTrue(prompt.endswith(probe.PROMPT_SUFFIX))
        self.assertIn("integers 1 through 80", probe.PROMPT_SUFFIX)

    def test_issue21_dict_and_string_arguments_render_equivalently(self):
        hotfix = load_module("issue21", ISSUE21)
        updated, status = hotfix.patch_text(minimal_encoder())
        self.assertEqual(status, "applied")
        namespace: dict = {}
        exec(updated, namespace)
        encode = namespace["encode_arguments_to_dsml"]
        expected = [("kind", "synthetic"), ("limit", 2)]
        self.assertEqual(
            encode({"arguments": '{"kind":"synthetic","limit":2}'}), expected
        )
        self.assertEqual(
            encode({"arguments": {"kind": "synthetic", "limit": 2}}), expected
        )
        self.assertNotIn("arguments", dict(expected))

    def test_issue21_is_idempotent_and_unknown_anchor_fails_closed(self):
        hotfix = load_module("issue21_idempotent", ISSUE21)
        updated, status = hotfix.patch_text(minimal_encoder())
        self.assertEqual(status, "applied")
        again, second_status = hotfix.patch_text(updated)
        self.assertEqual((again, second_status), (updated, "skipped"))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "encoding.py"
            target.write_text("# unexpected encoder\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ISSUE21), str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pattern not found", result.stderr)

    def test_issue22_routes_nvfp4_to_fp8_path_and_is_idempotent(self):
        hotfix = load_module("issue22", ISSUE22)
        source = '        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"\n'
        updated, status = hotfix.patch_text(source)
        self.assertEqual(status, "applied")
        self.assertIn(
            'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")', updated
        )
        self.assertEqual(hotfix.patch_text(updated), (updated, "skipped"))

    def test_issue22_unknown_anchor_fails_closed(self):
        hotfix = load_module("issue22_unknown", ISSUE22)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "flashmla_sparse.py"
            target.write_text("# unexpected backend\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ISSUE22), str(target)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pattern not found", result.stderr)

    def test_derived_image_uses_pinned_base_and_applies_issue22_at_build(self):
        dockerfile = (ROOT / "recipe" / "Dockerfile.anemll-runtime-hotfixes").read_text()
        manifest = json.loads(
            (ROOT / "recipe" / "runtime-hotfixes.manifest.json").read_text()
        )
        self.assertRegex(manifest["base_image"], r"@sha256:[0-9a-f]{64}$")
        self.assertIn("hotfix-nvfp4-ds-mla-issue22.py", dockerfile)
        self.assertIn("RUN python3 /opt/dspark-hotfixes/hotfix-nvfp4", dockerfile)
        self.assertEqual(manifest["issue21_upstream_commit"], "94baabf")
        self.assertEqual(manifest["issue22_upstream_commit"], "6c42a7a")

    def test_image_verifier_distinguishes_patcher_from_effective_runtime(self):
        verifier = load_module("runtime_verifier", VERIFIER)
        with mock.patch.object(
            verifier,
            "docker_markers",
            return_value={
                "issue21_patcher_present": False,
                "issue22_runtime_present": True,
            },
        ):
            with self.assertRaisesRegex(ValueError, "Issue #21 patcher"):
                verifier.verify_image("synthetic:image")
        source = VERIFIER.read_text()
        self.assertIn("issue21_patcher_present", source)
        self.assertNotIn('return {"image": image, "image_id": image_id, "issue21": True', source)


if __name__ == "__main__":
    unittest.main()
