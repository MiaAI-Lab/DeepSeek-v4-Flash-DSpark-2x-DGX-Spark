#!/usr/bin/env python3
"""Hermetic CPU acceptance for the issue #117 SHM ring backport.

The fixture is the complete pinned vLLM source.  Tests exercise the generated
reader methods without importing vLLM, then cover exact-state classification,
transactional publication, rollback, and two-rank startup wiring.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "scripts"
    / "fixtures"
    / "issue117"
    / "shm_broadcast-752a3a504.py"
)
UPSTREAM_MERGE_FIXTURE = (
    ROOT
    / "scripts"
    / "fixtures"
    / "issue117"
    / "shm_broadcast-10c75477.py"
)
UPSTREAM_MERGE_GIT_BLOB_SHA1 = "6b9dd4068b9a82b46fe12a47e0479fe2cb0ae2ad"
PATCHER_PATH = ROOT / "patches" / "hotfix-vllm-issue117-shm-ring-buffer.py"
SPIN_PATCHER = ROOT / "patches" / "hotfix-gb10-spin-wait.sh"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
EXPECTED_VLLM = "0.25.2.dev0+g752a3a504.d20260714"
STOCK_SIZE = 39_864
STOCK_SHA256 = "7ff67c2ef6b8a33a13b11aa3cb202da7887d1d44eed27c6a02d817ea24807d61"
STOCK_ISSUE79_SIZE = 39_868
STOCK_ISSUE79_SHA256 = "423234a203429b4d74aa48021a3ea02f3811be7d6a1938369feed298254fd51f"
PATCHED_SIZE = 40_312
PATCHED_SHA256 = "911e0dd65e0a0c6346e4f8f2120d2417fefe431ce5f2618ba9e9c9e1986faf23"
PATCHED_ISSUE79_SIZE = 40_316
PATCHED_ISSUE79_SHA256 = "30d8b62817adab4fabde8ddc6ce9a0f4b71899b80f70e8a50c688f5e63a46b0f"
FRAGMENT_SHA256 = {
    "OLD_CONSTANT": "201ddc055849864bddabcbc67c51c4ad1e6c178b31142fdd0aa92246cbe4c996",
    "NEW_CONSTANT": "35f58ffaf2813cd4cea2ccde15c7f315304774d36ef4f368f5fd80b9a81efc71",
    "OLD_TIMEOUT": "9fe3cf184e904605d7a42bd46196ad97a62a8688cb89675ca669fde46703e3c2",
    "NEW_TIMEOUT": "f8d805d27efa02e92a841f61fa6aa1ed00510babae23824ddefabc259f3d2e7f",
    "OLD_RELEASE": "21ef5e27ec5f1ed81088288cba11fdca7eb851b5240be61d126720472fff09ad",
    "NEW_RELEASE": "458b4dd119f1db0cce130b579aaa8ca10efeaea0206d8113b6b7d714fdc014b4",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def load_patcher():
    spec = importlib.util.spec_from_file_location("issue117_hotfix", PATCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load issue117 patcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PATCHER = load_patcher()


def provider(version: str = EXPECTED_VLLM, *, missing: bool = False):
    def get_version(package: str) -> str:
        if missing:
            raise importlib.metadata.PackageNotFoundError(package)
        if package != "vllm":
            raise KeyError(package)
        return version

    return get_version


def nested_source(source: bytes, name: str) -> str:
    text = source.decode("utf-8")
    tree = ast.parse(text)
    queue = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MessageQueue"
    )
    node = next(
        child
        for child in queue.body
        if isinstance(child, (ast.ClassDef, ast.FunctionDef))
        and child.name == name
    )
    if node.end_lineno is None:
        raise RuntimeError(f"cannot extract MessageQueue.{name}")
    start_lineno = min(
        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
    )
    lines = text.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[start_lineno - 1 : node.end_lineno]))


def extracted_message_queue(source: bytes):
    timeout_class = nested_source(source, "ReadTimeoutWithWarnings")
    acquire_read = nested_source(source, "acquire_read")
    harness = f'''from contextlib import contextmanager
import sys
import time

VLLM_RINGBUFFER_WARNING_INTERVAL = 60
SHM_READER_RECHECK_INTERVAL_MS = 5000
SPINLOOP_EXT_ENABLED = False
SPINLOOP_TIMEOUT_SECONDS = 0.1
LONG_WAIT_TIME_LOG_MSG = "long wait"

class Logger:
    def info(self, *args):
        pass
logger = Logger()

def memory_fence():
    pass

class MessageQueue:
{textwrap.indent(timeout_class, "    ")}
{textwrap.indent(acquire_read, "    ")}
'''
    namespace: dict[str, object] = {}
    exec(compile(harness, "<issue117-reader-harness>", "exec"), namespace)
    return namespace["MessageQueue"]


class IndefinitePoll(RuntimeError):
    pass


class FakeSpinCondition:
    def __init__(self, metadata: list[int]):
        self.metadata = metadata
        self.waits: list[int | None] = []
        self.reads = 0

    def wait(self, timeout_ms: int | None = None) -> None:
        self.waits.append(timeout_ms)
        if timeout_ms is None:
            raise IndefinitePoll("lost notification leaves poll unbounded")
        self.metadata[0] = 1

    def record_read(self) -> None:
        self.reads += 1


class FakeBuffer:
    max_chunks = 2

    def __init__(self, metadata: list[int], value: int = 123):
        self.metadata = metadata
        self.value = value

    @contextmanager
    def get_metadata(self, _index: int):
        yield self.metadata

    @contextmanager
    def get_data(self, _index: int):
        yield bytearray([self.value])


def queue_instance(queue_class, metadata: list[int]):
    queue = object.__new__(queue_class)
    queue._is_local_reader = True
    queue.local_reader_rank = 0
    queue.current_idx = 0
    queue.shutting_down = False
    queue.buffer = FakeBuffer(metadata)
    queue._spin_condition = FakeSpinCondition(metadata)
    return queue


def temp_artifacts(directory: Path) -> list[Path]:
    return sorted(directory.glob(".shm_broadcast.py.issue117-*.tmp"))


class FixtureAndBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = FIXTURE.read_bytes()
        cls.upstream_post = UPSTREAM_MERGE_FIXTURE.read_bytes()
        cls.post = PATCHER.build_candidate(cls.stock)
        cls.issue79_stock = cls.stock.replace(
            PATCHER.SPIN_STOCK, PATCHER.SPIN_ISSUE79, 1
        )
        cls.issue79_post = PATCHER.build_candidate(cls.issue79_stock)

    def test_fixture_raw_size_and_hash_match_pinned_github_blob(self):
        self.assertEqual(
            (len(self.stock), sha256(self.stock)), (STOCK_SIZE, STOCK_SHA256)
        )

    def test_issue117_postimage_matches_official_upstream_merge_blob(self):
        self.assertEqual(
            git_blob_sha1(self.upstream_post), UPSTREAM_MERGE_GIT_BLOB_SHA1
        )
        self.assertEqual(
            (len(self.upstream_post), sha256(self.upstream_post)),
            (PATCHED_SIZE, PATCHED_SHA256),
        )
        self.assertEqual(self.post, self.upstream_post)

    def test_derived_images_and_upstream_regions_are_independently_pinned(self):
        self.assertEqual(
            (len(self.issue79_stock), sha256(self.issue79_stock)),
            (STOCK_ISSUE79_SIZE, STOCK_ISSUE79_SHA256),
        )
        self.assertEqual(
            (len(self.post), sha256(self.post)), (PATCHED_SIZE, PATCHED_SHA256)
        )
        self.assertEqual(
            (len(self.issue79_post), sha256(self.issue79_post)),
            (PATCHED_ISSUE79_SIZE, PATCHED_ISSUE79_SHA256),
        )
        for name, expected in FRAGMENT_SHA256.items():
            with self.subTest(name=name):
                fragment = getattr(PATCHER, name)
                self.assertEqual(sha256(fragment), expected)
                expected_count = 1 if name.startswith("OLD_") else 0
                self.assertEqual(self.stock.count(fragment), expected_count)
                expected_post_count = 1 - expected_count
                self.assertEqual(self.post.count(fragment), expected_post_count)

    def test_patcher_has_no_optimization_sensitive_compatibility_checks(self):
        tree = ast.parse(PATCHER_PATH.read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_lost_notify_stock_path_is_unbounded_but_patch_rechecks_at_5000ms(self):
        stock_queue = queue_instance(extracted_message_queue(self.stock), [0, 0])
        with self.assertRaises(IndefinitePoll):
            with stock_queue.acquire_read(indefinite=True):
                pass
        self.assertEqual(stock_queue._spin_condition.waits, [None])

        patched_queue = queue_instance(extracted_message_queue(self.post), [0, 0])
        with patched_queue.acquire_read(indefinite=True) as buffer:
            value = buffer[0]
        self.assertEqual(value, 123)
        self.assertEqual(patched_queue._spin_condition.waits, [5000])
        self.assertEqual(patched_queue.buffer.metadata, [1, 1])
        self.assertEqual(patched_queue._spin_condition.reads, 1)

    def test_consumer_exception_releases_slot_only_after_patch(self):
        stock_queue = queue_instance(extracted_message_queue(self.stock), [1, 0])
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with stock_queue.acquire_read(timeout=0.1):
                raise RuntimeError("consumer failed")
        self.assertEqual(stock_queue.buffer.metadata, [1, 0])
        self.assertEqual(stock_queue.current_idx, 0)
        self.assertEqual(stock_queue._spin_condition.reads, 0)

        patched_queue = queue_instance(extracted_message_queue(self.post), [1, 0])
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with patched_queue.acquire_read(timeout=0.1):
                raise RuntimeError("consumer failed")
        self.assertEqual(patched_queue.buffer.metadata, [1, 1])
        self.assertEqual(patched_queue.current_idx, 1)
        self.assertEqual(patched_queue._spin_condition.reads, 1)


class PatcherTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = FIXTURE.read_bytes()
        cls.post = PATCHER.build_candidate(cls.stock)
        cls.issue79_stock = cls.stock.replace(
            PATCHER.SPIN_STOCK, PATCHER.SPIN_ISSUE79, 1
        )
        cls.issue79_post = PATCHER.build_candidate(cls.issue79_stock)

    def write_target(self, directory: Path, data: bytes, mode: int = 0o640) -> Path:
        target = directory / "shm_broadcast.py"
        target.write_bytes(data)
        target.chmod(mode)
        return target

    def snapshot(self, target: Path):
        current = target.lstat()
        return (
            target.read_bytes(),
            current.st_ino,
            current.st_mtime_ns,
            stat.S_IMODE(current.st_mode),
            current.st_uid,
            current.st_gid,
        )


class PatcherCompatibilityTests(PatcherTestBase):
    def test_stock_variants_apply_and_preserve_mode_and_issue79_state(self):
        cases = (
            (self.stock, self.post, False, 0o604),
            (self.issue79_stock, self.issue79_post, True, 0o751),
        )
        for initial, expected, issue79, mode in cases:
            with self.subTest(issue79=issue79), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, initial, mode)
                result = PATCHER.apply(target, provider())
                self.assertEqual(result.outcome, "applied")
                self.assertEqual(result.issue79, issue79)
                self.assertEqual(target.read_bytes(), expected)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), mode)
                self.assertEqual(temp_artifacts(directory), [])

    def test_apply_is_idempotent_without_rewrite_for_both_postimages(self):
        for post in (self.post, self.issue79_post):
            with self.subTest(digest=sha256(post)), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, post)
                before = self.snapshot(target)
                result = PATCHER.apply(target, provider())
                self.assertEqual(result.outcome, "already-patched")
                self.assertEqual(self.snapshot(target), before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_check_and_status_classify_without_writes(self):
        cases = (
            (self.stock, "stock-compatible"),
            (self.issue79_stock, "stock-compatible"),
            (self.post, "patched"),
            (self.issue79_post, "patched"),
        )
        for data, expected in cases:
            with self.subTest(state=expected, digest=sha256(data)), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), data)
                before = self.snapshot(target)
                inspection = PATCHER.inspect_target(target, provider())
                self.assertEqual(inspection.state, expected)
                self.assertEqual(self.snapshot(target), before)

    def test_cli_check_and_status_have_fail_closed_exit_codes(self):
        cases = (
            (self.stock, ["--check"], 0, "compatible: stock-compatible"),
            (self.post, ["--check"], 0, "compatible: patched"),
            (self.stock, ["--status"], 1, "stock-compatible"),
            (self.post, ["--status"], 0, "patched"),
            (self.stock + b"\n# drift\n", ["--status"], 2, "incompatible"),
        )
        for data, argv, expected_rc, output in cases:
            with self.subTest(argv=argv, rc=expected_rc), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), data)
                before = self.snapshot(target)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    mock.patch.object(PATCHER, "PRODUCTION_TARGET", target),
                    mock.patch.object(
                        PATCHER.importlib.metadata, "version", provider()
                    ),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    rc = PATCHER.main(argv)
                self.assertEqual(rc, expected_rc)
                self.assertIn(output, stdout.getvalue() + stderr.getvalue())
                self.assertEqual(self.snapshot(target), before)

    def test_cli_check_accepts_exact_live_issue79_identity_without_writing(self):
        self.assertEqual(
            (len(self.issue79_stock), sha256(self.issue79_stock)),
            (STOCK_ISSUE79_SIZE, STOCK_ISSUE79_SHA256),
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = self.write_target(Path(tmp), self.issue79_stock)
            before = self.snapshot(target)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(PATCHER, "PRODUCTION_TARGET", target),
                mock.patch.object(PATCHER.importlib.metadata, "version", provider()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                rc = PATCHER.main(["--check"])
            output = stdout.getvalue() + stderr.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("compatible: stock-compatible", output)
            self.assertIn("issue79=patched", output)
            self.assertEqual(self.snapshot(target), before)

    def test_version_missing_nonregular_and_symlink_targets_fail_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            for metadata_provider in (
                provider("0.25.2"),
                provider(missing=True),
            ):
                before = self.snapshot(target)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, metadata_provider)
                self.assertEqual(self.snapshot(target), before)

            missing = directory / "missing.py"
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(missing, provider())
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(directory, provider())

            link = directory / "link.py"
            link.symlink_to(target)
            before = self.snapshot(target)
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(link, provider())
            self.assertEqual(self.snapshot(target), before)

    def test_independent_old_and_new_anchor_drifts_all_fail_without_writes(self):
        def flip_inside(data: bytes, fragment: bytes) -> bytes:
            start = data.index(fragment) + min(12, len(fragment) - 1)
            replacement = b"X" if data[start : start + 1] != b"X" else b"Y"
            return data[:start] + replacement + data[start + 1 :]

        variants = {}
        for name in ("CONSTANT", "TIMEOUT", "RELEASE"):
            variants[f"old-{name.lower()}"] = flip_inside(
                self.stock, getattr(PATCHER, f"OLD_{name}")
            )
            variants[f"new-{name.lower()}"] = flip_inside(
                self.post, getattr(PATCHER, f"NEW_{name}")
            )
        variants["issue79-overlay"] = self.issue79_stock.replace(
            PATCHER.SPIN_ISSUE79, b"busy_loop_s: float = 0.003", 1
        )

        for name, data in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, data)
                before = self.snapshot(target)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, provider())
                self.assertEqual(self.snapshot(target), before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_marker_only_partial_mixed_and_invalid_states_all_fail_closed(self):
        variants = {
            "marker-only": self.stock + b"\n# [shm-reader-recheck]\n",
            "marker-with-constant-drift": self.post.replace(
                b"SHM_READER_RECHECK_INTERVAL_MS = 5000",
                b"SHM_READER_RECHECK_INTERVAL_MS = 4999",
                1,
            )
            + b"\n# [shm-reader-recheck]\n",
            "constant-only": self.stock.replace(
                PATCHER.OLD_CONSTANT, PATCHER.NEW_CONSTANT, 1
            ),
            "timeout-only": self.stock.replace(
                PATCHER.OLD_TIMEOUT, PATCHER.NEW_TIMEOUT, 1
            ),
            "release-only": self.stock.replace(
                PATCHER.OLD_RELEASE, PATCHER.NEW_RELEASE, 1
            ),
            "constant-timeout": self.stock.replace(
                PATCHER.OLD_CONSTANT, PATCHER.NEW_CONSTANT, 1
            ).replace(PATCHER.OLD_TIMEOUT, PATCHER.NEW_TIMEOUT, 1),
            "duplicate-old": self.stock + PATCHER.OLD_TIMEOUT,
            "duplicate-new": self.post + PATCHER.NEW_RELEASE,
            "invalid-utf8": self.stock[:-1] + b"\xff",
        }
        for name, data in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, data)
                before = self.snapshot(target)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, provider())
                self.assertEqual(self.snapshot(target), before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_python_optimize_does_not_recreate_removed_constant_failopen(self):
        drifted = self.stock.replace(PATCHER.OLD_CONSTANT, b"", 1)
        code = textwrap.dedent(
            """
            import importlib.util
            import pathlib
            import sys

            spec = importlib.util.spec_from_file_location("optimized_issue117", sys.argv[1])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            try:
                module.apply(pathlib.Path(sys.argv[2]), lambda _name: sys.argv[3])
            except module.CompatibilityError:
                raise SystemExit(7)
            raise SystemExit(0)
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = self.write_target(Path(tmp), drifted)
            before = self.snapshot(target)
            env = os.environ.copy()
            env["PYTHONOPTIMIZE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    "-c",
                    code,
                    str(PATCHER_PATH),
                    str(target),
                    EXPECTED_VLLM,
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(self.snapshot(target), before)


class PatcherFailureRecoveryTests(PatcherTestBase):
    def test_staging_and_prepublication_failures_leave_original(self):
        failures = (
            mock.patch.object(
                PATCHER.tempfile, "mkstemp", side_effect=OSError("injected")
            ),
            mock.patch.object(
                PATCHER, "_write_all", side_effect=OSError("injected")
            ),
            mock.patch.object(
                PATCHER.os, "replace", side_effect=OSError("injected")
            ),
        )
        for patch_context in failures:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, self.stock, 0o605)
                before = self.snapshot(target)
                with patch_context, self.assertRaises(OSError):
                    PATCHER.apply(target, provider())
                self.assertEqual(self.snapshot(target), before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_exception_after_successful_replace_restores_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.issue79_stock, 0o605)
            original = target.read_bytes()
            original_stat = target.stat()
            real_replace = PATCHER.os.replace
            calls = 0

            def replace_then_raise(src, dst):
                nonlocal calls
                calls += 1
                real_replace(src, dst)
                if calls == 1:
                    raise OSError("injected after rename")

            with mock.patch.object(
                PATCHER.os, "replace", side_effect=replace_then_raise
            ), self.assertRaises(OSError):
                PATCHER.apply(target, provider())
            restored = target.stat()
            self.assertEqual(calls, 2)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(
                (stat.S_IMODE(restored.st_mode), restored.st_uid, restored.st_gid),
                (
                    stat.S_IMODE(original_stat.st_mode),
                    original_stat.st_uid,
                    original_stat.st_gid,
                ),
            )
            self.assertEqual(temp_artifacts(directory), [])

    def test_postpublication_failure_restores_bytes_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o604)
            original = target.read_bytes()
            with mock.patch.object(
                PATCHER,
                "_verify_published",
                side_effect=KeyboardInterrupt("injected verification interrupt"),
            ), self.assertRaises(KeyboardInterrupt):
                PATCHER.apply(target, provider())
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)
            self.assertEqual(temp_artifacts(directory), [])

    def test_failed_rollback_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            real_replace = PATCHER.os.replace
            calls = 0

            def fail_second_replace(src, dst):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected rollback failure")
                return real_replace(src, dst)

            with (
                mock.patch.object(
                    PATCHER.os, "replace", side_effect=fail_second_replace
                ),
                mock.patch.object(
                    PATCHER,
                    "_verify_published",
                    side_effect=OSError("injected verification failure"),
                ),
                self.assertRaises(PATCHER.RollbackError),
            ):
                PATCHER.apply(target, provider())
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(temp_artifacts(directory), [])


class StartupWiringTests(unittest.TestCase):
    def test_compose_applies_and_verifies_after_issue79_before_exec(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        issue79 = "bash /opt/dspark-patches/hotfix-gb10-spin-wait.sh"
        apply = "python3 /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py || exit 1"
        status = "python3 /opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py --status || exit 1"
        env_default = (
            'DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX: '
            '"${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}"'
        )
        self.assertEqual(compose.count(env_default), 1)
        self.assertEqual(compose.count(apply), 1)
        self.assertEqual(compose.count(status), 1)
        positions = [
            compose.index(issue79),
            compose.index(apply),
            compose.index(status),
            compose.index("exec /usr/local/bin/vllm serve"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_launcher_syncs_and_checks_both_ranks_before_either_start(self):
        source = START.read_text(encoding="utf-8")
        regular_check = (
            '[ "${DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX:-0}" != "1" ] '
            '&& { [ ! -f "$DSPARK_ISSUE117_HOTFIX" ] '
            '|| [ -L "$DSPARK_ISSUE117_HOTFIX" ]; }'
        )
        sync = (
            'scp "$DSPARK_ISSUE117_HOTFIX" '
            '"${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/'
            'hotfix-vllm-issue117-shm-ring-buffer.py"'
        )
        worker_check = (
            'run --rm --no-deps --entrypoint python3 vllm-dspark '
            '/opt/dspark-patches/hotfix-vllm-issue117-shm-ring-buffer.py --check'
        )
        head_check = (
            'compose_base 0 "" run --rm --no-deps --entrypoint python3 '
            'vllm-dspark /opt/dspark-patches/'
            'hotfix-vllm-issue117-shm-ring-buffer.py --check'
        )
        worker_start = 'echo "Starting DSpark worker on ${WORKER_HOST}..."'
        head_start = 'echo "Starting DSpark head..."'
        for token in (regular_check, sync, worker_check, head_check):
            self.assertIn(token, source)
        positions = [
            source.index(sync),
            source.index(worker_check),
            source.index(head_check),
            source.index(worker_start),
            source.index(head_start),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_issue117_rollback_switch_is_independent_and_not_tunable(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        start = START.read_text(encoding="utf-8")
        spin = SPIN_PATCHER.read_text(encoding="utf-8")
        self.assertIn("DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX", compose)
        self.assertIn("DSPARK_SKIP_ISSUE117_RECHECK_HOTFIX", start)
        self.assertNotIn("ISSUE117", spin)
        self.assertNotIn("DSPARK_ISSUE117_RECHECK_INTERVAL", compose + start)
        self.assertIn("SHM_READER_RECHECK_INTERVAL_MS = 5000", PATCHER_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
