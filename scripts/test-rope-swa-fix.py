#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-rope-swa-fix.py (no vLLM needed)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "fixtures" / "dspark-rope-swa" / "rope-752a3a504-stock.py"
PATCHER = ROOT / "patches" / "hotfix-vllm-rope-swa-fix.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"


def _load():
    spec = importlib.util.spec_from_file_location("dspark_rope_swa", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RS = _load()
GOOD = lambda name: RS.EXPECTED_VLLM_VERSION  # noqa: E731

_STUBS = (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.rotary_embedding",
    "vllm.model_executor.layers.rotary_embedding.base",
)


def _build_rope_fn(source: bytes):
    """Exec a rope.py variant against stubbed vLLM imports.

    Returns ``(build_deepseek_v4_rope, calls)`` where ``calls`` collects the
    rope_parameters dict of every ``get_rope`` invocation.
    """
    calls: list[dict] = []
    saved = {name: sys.modules.get(name) for name in _STUBS}
    try:
        for name in _STUBS:
            sys.modules[name] = types.ModuleType(name)
        rotary = sys.modules["vllm.model_executor.layers.rotary_embedding"]

        def get_rope(head_dim, *, max_position, rope_parameters, is_neox_style):
            snapshot = dict(rope_parameters)
            calls.append(snapshot)
            return ("ROPE", head_dim, max_position, is_neox_style, snapshot)

        rotary.get_rope = get_rope
        sys.modules["vllm.model_executor.layers.rotary_embedding.base"].RotaryEmbedding = object
        namespace: dict = {}
        exec(compile(source, "rope.py", "exec"), namespace)
        return namespace["build_deepseek_v4_rope"], calls
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _vision_exp_config():
    """rope config exactly as vLLM normalizes the served abliterated checkpoint."""
    config = types.SimpleNamespace()
    config.rope_parameters = {
        "rope_type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    config.rope_theta = 10000
    config.compress_rope_theta = 160000
    return config


_KW = dict(head_dim=576, rope_head_dim=64, max_position_embeddings=1048576)


class Transform(unittest.TestCase):
    def test_fixture_matches_stock_pin(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), RS.STOCK_SHA256)
        self.assertEqual(len(data), RS.STOCK_SIZE)

    def test_transform_is_pinned_and_compiles(self):
        stock = FIXTURE.read_bytes()
        patched = RS.transform(stock)
        self.assertEqual(hashlib.sha256(patched).hexdigest(), RS.PATCHED_SHA256)
        self.assertEqual(len(patched), RS.PATCHED_SIZE)
        compile(patched, "rope.py", "exec")
        self.assertEqual(patched.count(RS.MARK.encode()), 1)
        # single-site replace: everything outside the region is byte-identical
        self.assertEqual(stock.count(RS.REGION_OLD), 1)
        self.assertEqual(
            patched, stock.replace(RS.REGION_OLD, RS.REGION_NEW, 1)
        )
        self.assertIn(b'if compress_ratio > 1 and rope_parameters["rope_type"] != "default":', patched)

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(RS.HotfixError):
            RS.transform(b"def nothing():\n    pass\n")
        patched = RS.transform(FIXTURE.read_bytes())
        with self.assertRaises(RS.HotfixError):
            RS.transform(patched)


class Semantics(unittest.TestCase):
    """Layer-type routing with the served checkpoint's real rope values."""

    def _params_by_ratio(self, source: bytes) -> dict[int, dict]:
        build, calls = _build_rope_fn(source)
        config = _vision_exp_config()
        for ratio in (1, 4, 128):
            build(config, compress_ratio=ratio, **_KW)
        return dict(zip((1, 4, 128), calls))

    def test_stock_applies_yarn_to_swa_layers(self):
        """The bug this port fixes: stock YaRN-scales compress_ratio=1 layers."""
        params = self._params_by_ratio(FIXTURE.read_bytes())
        swa = params[1]
        self.assertEqual(swa["rope_type"], "deepseek_yarn")
        self.assertEqual(swa["factor"], 16)
        self.assertEqual(swa["original_max_position_embeddings"], 65536)
        self.assertEqual(swa["rope_theta"], 10000)

    def test_patched_swa_layers_use_plain_rope(self):
        params = self._params_by_ratio(RS.transform(FIXTURE.read_bytes()))
        swa = params[1]
        self.assertEqual(swa["rope_type"], "deepseek_yarn")
        self.assertEqual(swa["factor"], 1.0)  # identity scaling = plain RoPE
        self.assertEqual(swa["original_max_position_embeddings"], _KW["max_position_embeddings"])
        self.assertEqual(swa["rope_theta"], 10000)
        self.assertEqual(swa["mscale"], 0)
        self.assertEqual(swa["mscale_all_dim"], 0)
        self.assertTrue(swa["is_deepseek_v4"])
        self.assertEqual(swa["rope_dim"], 64)

    def test_patched_compressor_layers_are_unchanged(self):
        stock_params = self._params_by_ratio(FIXTURE.read_bytes())
        patched_params = self._params_by_ratio(RS.transform(FIXTURE.read_bytes()))
        for ratio in (4, 128):
            self.assertEqual(patched_params[ratio], stock_params[ratio])
            self.assertEqual(patched_params[ratio]["factor"], 16)
            self.assertEqual(patched_params[ratio]["rope_theta"], 160000)

    def test_patched_does_not_mutate_shared_config_dict(self):
        build, _ = _build_rope_fn(RS.transform(FIXTURE.read_bytes()))
        config = _vision_exp_config()
        for ratio in (1, 4):
            build(config, compress_ratio=ratio, **_KW)
        self.assertEqual(config.rope_parameters, _vision_exp_config().rope_parameters)

    def test_patched_routes_nested_main_compress_dicts(self):
        build, calls = _build_rope_fn(RS.transform(FIXTURE.read_bytes()))
        config = _vision_exp_config()
        config.rope_parameters = {
            "main": {"rope_type": "default"},
            "compress": {"rope_type": "yarn", "factor": 16,
                         "original_max_position_embeddings": 65536},
        }
        build(config, compress_ratio=1, **_KW)
        build(config, compress_ratio=4, **_KW)
        swa, compress = calls
        self.assertEqual(swa["factor"], 1.0)
        self.assertEqual(swa["rope_theta"], 10000)
        self.assertEqual(compress["rope_type"], "deepseek_yarn")
        self.assertEqual(compress["factor"], 16)
        self.assertEqual(compress["rope_theta"], 160000)


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dspark-rope-swa-"))
        self.target = self.tmp / "rope.py"
        shutil.copyfile(FIXTURE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(RS.apply(self.target, provider=GOOD), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), RS.PATCHED_SHA256)
        self.assertEqual(RS.inspect(self.target, provider=GOOD)[0], "patched")
        self.assertEqual(RS.apply(self.target, provider=GOOD), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["rope.py"])

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(RS.HotfixError):
            RS.inspect(self.target, provider=GOOD)
        with self.assertRaises(RS.HotfixError):
            RS.apply(self.target, provider=GOOD)
        self.assertEqual(self.target.read_bytes(), b"x = 1\n")

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(RS.HotfixError):
            RS.inspect(self.target, provider=lambda name: "0.26.0")
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), RS.STOCK_SHA256)

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(RS.HotfixError):
            RS.inspect(link, provider=GOOD)

    def test_cli_check_and_status_do_not_write(self):
        import subprocess

        env = dict(os.environ)
        before = self.target.read_bytes()
        for flag in ("--check", "--status"):
            proc = subprocess.run(
                [sys.executable, str(PATCHER), flag, "--target", str(self.target)],
                capture_output=True,
                text=True,
                env=env,
            )
            # the real vllm is not installed here -> fail closed, no write
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("FAIL-CLOSED", proc.stderr)
            self.assertEqual(self.target.read_bytes(), before)


class Wiring(unittest.TestCase):
    def test_compose_gate_default_off_fail_closed(self):
        compose = COMPOSE.read_text()
        self.assertIn('DSPARK_ENABLE_ROPE_SWA_FIX: "${DSPARK_ENABLE_ROPE_SWA_FIX:-0}"', compose)
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_ROPE_SWA_FIX:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-rope-swa-fix.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_ROPE_SWA_FIX_HOTFIX:-./patches/hotfix-vllm-rope-swa-fix.py}"
            ":/opt/hotfix-vllm-rope-swa-fix.py:ro",
            compose,
        )

    def test_launcher_passthrough_sync_and_preflight(self):
        start = START.read_text()
        self.assertIn("DSPARK_ROPE_SWA_FIX_HOTFIX='./patches/hotfix-vllm-rope-swa-fix.py'", start)
        self.assertIn("DSPARK_ENABLE_ROPE_SWA_FIX=$REMOTE_ROPE_SWA_FIX", start)
        self.assertIn("/opt/hotfix-vllm-rope-swa-fix.py --check", start)
        self.assertIn('patches/hotfix-vllm-rope-swa-fix.py"', start)
        self.assertIn("export DSPARK_ROPE_SWA_FIX_HOTFIX DSPARK_ENABLE_ROPE_SWA_FIX", start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_ROPE_SWA_FIX=0", env)
        ci = CI.read_text()
        self.assertIn("scripts/test-rope-swa-fix.py", ci)
        self.assertIn("hotfix-vllm-rope-swa-fix.py", ci)


if __name__ == "__main__":
    unittest.main()
