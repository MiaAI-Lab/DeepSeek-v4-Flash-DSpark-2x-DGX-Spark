#!/usr/bin/env python3
"""CPU tests for patches/hotfix-vllm-mxfp4-indexer-cache.py (no vLLM needed)."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "scripts" / "fixtures" / "dspark-mxfp4-indexer" / "indexer-752a3a504-stock.py"
)
PATCHER = ROOT / "patches" / "hotfix-vllm-mxfp4-indexer-cache.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
CI = ROOT / "scripts" / "ci-validate.sh"


def _load():
    spec = importlib.util.spec_from_file_location("dspark_mxfp4_indexer", PATCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MX = _load()
GOOD = lambda name: MX.EXPECTED_VLLM_VERSION  # noqa: E731


def _run_gate(region: bytes, *, family: int | None, flag: bool) -> bool:
    """Exec the gate region under a stubbed platform; True = boot proceeds."""
    platform = types.SimpleNamespace(
        is_device_capability_family=lambda fam: fam == family
    )
    namespace = {
        "current_platform": platform,
        "self": types.SimpleNamespace(use_fp4_indexer_cache=flag),
    }
    try:
        exec(textwrap.dedent(region.decode("utf-8")), namespace)
    except AssertionError:
        return False
    return True


class Transform(unittest.TestCase):
    def test_fixture_matches_stock_pin(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), MX.STOCK_SHA256)
        self.assertEqual(len(data), MX.STOCK_SIZE)

    def test_transform_is_pinned_minimal_and_compiles(self):
        stock = FIXTURE.read_bytes()
        patched = MX.transform(stock)
        self.assertEqual(hashlib.sha256(patched).hexdigest(), MX.PATCHED_SHA256)
        self.assertEqual(len(patched), MX.PATCHED_SIZE)
        compile(patched, "indexer.py", "exec")
        self.assertEqual(patched.count(MX.MARK.encode()), 1)
        # one added condition line plus its two comment lines; the assert
        # message is reworded; everything else byte-identical
        stock_lines = stock.splitlines()
        patched_lines = patched.splitlines()
        self.assertEqual(len(patched_lines), len(stock_lines) + 3)
        added = [line for line in patched_lines if line not in stock_lines]
        removed = [line for line in stock_lines if line not in patched_lines]
        self.assertEqual(
            [line.strip() for line in added],
            [
                MX.MARK.encode(),
                b"# kernels; consumer Blackwell (sm_12x, e.g. GB10) runs them too.",
                b"or current_platform.is_device_capability_family(120)",
                b'"use_fp4_indexer_cache requires Blackwell GPUs (sm_10x "',
                b'"datacenter, e.g. B200/GB200, or sm_12x consumer, e.g. GB10); "',
            ],
        )
        self.assertEqual(
            [line.strip() for line in removed],
            [
                b'"use_fp4_indexer_cache requires Blackwell datacenter GPUs "',
                b'"(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "',
            ],
        )
        # the rest of the builder (flattening rule included) is untouched
        self.assertIn(
            b"self.use_flattening = not current_platform.is_device_capability_family",
            patched,
        )

    def test_transform_refuses_foreign_or_patched_bytes(self):
        with self.assertRaises(MX.HotfixError):
            MX.transform(b"def nothing():\n    pass\n")
        patched = MX.transform(FIXTURE.read_bytes())
        with self.assertRaises(MX.HotfixError):
            MX.transform(patched)

    def test_gate_semantics(self):
        """Exec the real region bytes: sm_12x passes only when patched."""
        # stock: datacenter (family 100) only
        self.assertTrue(_run_gate(MX.REGION_OLD, family=100, flag=True))
        self.assertFalse(_run_gate(MX.REGION_OLD, family=120, flag=True))
        # patched: sm_10x and sm_12x pass; older architectures still fail closed
        self.assertTrue(_run_gate(MX.REGION_NEW, family=100, flag=True))
        self.assertTrue(_run_gate(MX.REGION_NEW, family=120, flag=True))
        self.assertFalse(_run_gate(MX.REGION_NEW, family=90, flag=True))
        self.assertFalse(_run_gate(MX.REGION_NEW, family=None, flag=True))
        # flag off never trips the gate on either variant
        self.assertTrue(_run_gate(MX.REGION_OLD, family=None, flag=False))
        self.assertTrue(_run_gate(MX.REGION_NEW, family=None, flag=False))


class Patcher(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dspark-mxfp4-indexer-"))
        self.target = self.tmp / "indexer.py"
        shutil.copyfile(FIXTURE, self.target)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_then_idempotent(self):
        self.assertEqual(MX.apply(self.target, provider=GOOD), "applied")
        data = self.target.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), MX.PATCHED_SHA256)
        self.assertEqual(MX.inspect(self.target, provider=GOOD)[0], "patched")
        self.assertEqual(MX.apply(self.target, provider=GOOD), "already-patched")
        self.assertEqual(self.target.read_bytes(), data)
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["indexer.py"])

    def test_refuses_foreign_bytes(self):
        self.target.write_bytes(b"x = 1\n")
        with self.assertRaises(MX.HotfixError):
            MX.inspect(self.target, provider=GOOD)
        with self.assertRaises(MX.HotfixError):
            MX.apply(self.target, provider=GOOD)
        self.assertEqual(self.target.read_bytes(), b"x = 1\n")

    def test_refuses_wrong_vllm_version(self):
        with self.assertRaises(MX.HotfixError):
            MX.inspect(self.target, provider=lambda name: "0.26.0")
        self.assertEqual(
            hashlib.sha256(self.target.read_bytes()).hexdigest(), MX.STOCK_SHA256
        )

    def test_refuses_symlink(self):
        link = self.tmp / "link.py"
        os.symlink(self.target, link)
        with self.assertRaises(MX.HotfixError):
            MX.inspect(link, provider=GOOD)

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
        self.assertIn(
            'DSPARK_ENABLE_MXFP4_INDEXER_CACHE: "${DSPARK_ENABLE_MXFP4_INDEXER_CACHE:-0}"',
            compose,
        )
        self.assertIn(
            'if [ "$${DSPARK_ENABLE_MXFP4_INDEXER_CACHE:-0}" = "1" ]; then '
            "python3 /opt/hotfix-vllm-mxfp4-indexer-cache.py || exit 1; fi;",
            compose,
        )
        self.assertIn(
            "${DSPARK_MXFP4_INDEXER_CACHE_HOTFIX:-./patches/hotfix-vllm-mxfp4-indexer-cache.py}"
            ":/opt/hotfix-vllm-mxfp4-indexer-cache.py:ro",
            compose,
        )

    def test_compose_emits_attention_config_only_when_enabled(self):
        compose = COMPOSE.read_text()
        # the arg is assembled by the same 0/1-exact case pattern as
        # ASYNC_SCHEDULING_ARGS and expands to exactly two words
        self.assertIn(
            'case "$${DSPARK_ENABLE_MXFP4_INDEXER_CACHE:-0}" in '
            '1) MXFP4_INDEXER_ARGS="--attention-config '
            '{\\"use_fp4_indexer_cache\\":true}" ;; '
            '0) MXFP4_INDEXER_ARGS="" ;;',
            compose,
        )
        self.assertIn("$${MXFP4_INDEXER_ARGS}", compose)
        serve = compose.index("exec /usr/local/bin/vllm serve")
        self.assertGreater(compose.index("$${MXFP4_INDEXER_ARGS}", serve), serve)

    def test_launcher_passthrough_sync_preflight_and_alias_requirement(self):
        start = START.read_text()
        self.assertIn(
            "DSPARK_MXFP4_INDEXER_CACHE_HOTFIX='./patches/hotfix-vllm-mxfp4-indexer-cache.py'",
            start,
        )
        self.assertIn(
            "DSPARK_ENABLE_MXFP4_INDEXER_CACHE=$REMOTE_MXFP4_INDEXER", start
        )
        self.assertIn("/opt/hotfix-vllm-mxfp4-indexer-cache.py --check", start)
        self.assertIn('patches/hotfix-vllm-mxfp4-indexer-cache.py"', start)
        # the fp4 logits kernels are not in the persisted JIT cache: enabling
        # the cache without the sm121 alias headers must fail the launch
        self.assertIn(
            'if [ "${DSPARK_ENABLE_MXFP4_INDEXER_CACHE:-0}" = "1" ] && '
            '[ "${DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS:-0}" != "1" ]; then',
            start,
        )
        self.assertIn("requires DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1", start)

    def test_env_example_and_ci(self):
        env = ENV_EXAMPLE.read_text()
        self.assertIn("DSPARK_ENABLE_MXFP4_INDEXER_CACHE=0", env)
        ci = CI.read_text()
        self.assertIn("scripts/test-mxfp4-indexer-cache.py", ci)
        self.assertIn("hotfix-vllm-mxfp4-indexer-cache.py", ci)


if __name__ == "__main__":
    unittest.main()
