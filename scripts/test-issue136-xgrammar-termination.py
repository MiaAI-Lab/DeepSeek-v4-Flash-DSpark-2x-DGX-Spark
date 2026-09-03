#!/usr/bin/env python3
"""Hermetic CPU acceptance for the issue #136 XGrammar backport.

The checked-in files are exact upstream-derived fixtures.  Tests execute the
real XgrammarGrammar methods extracted from each fixture, then exercise the
production patcher and startup wiring without Docker, vLLM, xgrammar, torch, a
GPU, or network access.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "scripts" / "fixtures" / "issue136"
STOCK_FIXTURE = FIXTURES / "backend_xgrammar-752a3a504.py"
POST_FIXTURE = FIXTURES / "backend_xgrammar-752a3a504-pr52805.py"
PATCHER_PATH = ROOT / "patches" / "hotfix-vllm-issue136-xgrammar-termination.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
GRAMMAR_ADVANCE = ROOT / "patches" / "hotfix-dsv4-grammar-advance.sh"

STOCK_SHA256 = "231f6b9d7dab5e8d68aba486fa5912db99f8bdd3f9d8842ee3e0bb12bdb7cb67"
POST_SHA256 = "6c7e23c0ae5c6836d0d56862c6e825c49727fa2409b881b44ea2526f1fd03f04"
STOCK_REGION_SHA256 = "9677073da0986c345f8fa36c787248ff5b3a1b0fbe999da31a91491f3267a149"
POST_REGION_SHA256 = "2a7417bbe9e32179c3de8a5750358339320bec672b388fc0ede978e2270b72f4"
STOCK_BYTES = 12_699
POST_BYTES = 12_983
EXPECTED_VLLM = "0.25.2.dev0+g752a3a504.d20260714"
EXPECTED_XGRAMMAR = "0.2.3"
REGION_START = b"    def accept_tokens("
REGION_SENTINEL = (
    b"# cf https://github.com/mlc-ai/xgrammar/blob/"
    b"a32ac892676d2eedc0327416105b9b06edfb94b2/"
    b"cpp/json_schema_converter.cc\n"
)

# Independent literal description of the three upstream #52805 hunks.  Applying
# only these replacements to the pinned fixture must produce the complete
# checked-in post fixture byte-for-byte.
ACCEPT_OLD = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True
'''
ACCEPT_NEW = b'''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True
'''
VALIDATE_OLD = b'''    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''
VALIDATE_NEW = b'''    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''
RESET_OLD = b'''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''
RESET_NEW = b'''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False
'''

STOP = 2
TRAILING = 3
ORDINARY = 1
REJECT = 9


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def method_region(data: bytes) -> tuple[bytes, bytes, bytes]:
    start = data.index(REGION_START)
    end = data.index(REGION_SENTINEL, start) + len(REGION_SENTINEL)
    return data[:start], data[start:end], data[end:]


def load_patcher():
    spec = importlib.util.spec_from_file_location("issue136_hotfix", PATCHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load issue136 patcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PATCHER = load_patcher()


def provider(
    vllm: str = EXPECTED_VLLM,
    xgrammar: str = EXPECTED_XGRAMMAR,
    missing: str | None = None,
):
    values = {"vllm": vllm, "xgrammar": xgrammar}

    def get_version(name: str) -> str:
        if name == missing:
            raise importlib.metadata.PackageNotFoundError(name)
        return values[name]

    return get_version


class FakeLogger:
    def __init__(self):
        self.errors: list[tuple[object, ...]] = []

    def error(self, *args: object) -> None:
        self.errors.append(args)


class FakeMatcher:
    def __init__(self):
        self.calls: list[int] = []
        self.rollback_calls: list[int] = []
        self.accepted: list[int] = []
        self.terminated = False
        self.reset_calls = 0

    def accept_token(self, token: int) -> bool:
        self.calls.append(token)
        if self.terminated or token == REJECT:
            return False
        self.accepted.append(token)
        if token == STOP:
            self.terminated = True
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, count: int) -> None:
        self.rollback_calls.append(count)
        if count < 0 or count > len(self.accepted):
            raise AssertionError("invalid fake rollback")
        if count:
            del self.accepted[-count:]
        self.terminated = STOP in self.accepted

    def reset(self) -> None:
        self.reset_calls += 1
        self.calls.clear()
        self.accepted.clear()
        self.terminated = False


def grammar_class(source: bytes):
    tree = ast.parse(source.decode("utf-8"))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "XgrammarGrammar"
    )
    method_names = {
        "accept_tokens",
        "validate_tokens",
        "rollback",
        "is_terminated",
        "reset",
    }
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    minimal = ast.Module(
        body=[
            ast.ClassDef(
                name="XgrammarGrammar",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(minimal)
    namespace = {"__name__": "fixture_methods", "logger": FakeLogger()}
    exec(compile(minimal, "fixture-methods", "exec"), namespace)
    return namespace["XgrammarGrammar"], namespace["logger"]


def grammar_instance(source: bytes):
    cls, logger = grammar_class(source)
    grammar = cls()
    grammar.matcher = FakeMatcher()
    grammar.num_processed_tokens = 0
    grammar._is_terminated = False
    return grammar, logger


def temp_artifacts(directory: Path) -> list[Path]:
    return sorted(directory.glob(".backend_xgrammar.py.issue136-*.tmp"))


class FixtureProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = STOCK_FIXTURE.read_bytes()
        cls.post = POST_FIXTURE.read_bytes()

    def test_full_fixture_identities(self):
        self.assertEqual((len(self.stock), len(self.post)), (STOCK_BYTES, POST_BYTES))
        self.assertEqual((sha256(self.stock), sha256(self.post)), (STOCK_SHA256, POST_SHA256))
        compile(self.stock.decode("utf-8"), STOCK_FIXTURE.name, "exec")
        compile(self.post.decode("utf-8"), POST_FIXTURE.name, "exec")

    def test_only_the_three_upstream_hunks_change(self):
        generated = self.stock
        for old, new in (
            (ACCEPT_OLD, ACCEPT_NEW),
            (VALIDATE_OLD, VALIDATE_NEW),
            (RESET_OLD, RESET_NEW),
        ):
            self.assertEqual(generated.count(old), 1)
            self.assertEqual(generated.count(new), 0)
            generated = generated.replace(old, new, 1)
        self.assertEqual(generated, self.post)

    def test_region_hashes_and_outside_bytes(self):
        stock_prefix, stock_region, stock_suffix = method_region(self.stock)
        post_prefix, post_region, post_suffix = method_region(self.post)
        self.assertEqual(sha256(stock_region), STOCK_REGION_SHA256)
        self.assertEqual(sha256(post_region), POST_REGION_SHA256)
        self.assertEqual(stock_prefix, post_prefix)
        self.assertEqual(stock_suffix, post_suffix)

    def test_production_patcher_generates_checked_in_post_fixture(self):
        self.assertEqual(PATCHER.build_candidate(self.stock), self.post)

    def test_issue44993_patch_targets_are_disjoint(self):
        grammar_advance = GRAMMAR_ADVANCE.read_text(encoding="utf-8")
        self.assertNotIn("backend_xgrammar.py", grammar_advance)
        self.assertIn("v1/structured_output/__init__.py", grammar_advance)
        self.assertIn("v1/core/sched/scheduler.py", grammar_advance)
        stock_prefix, _, stock_suffix = method_region(self.stock)
        post_prefix, _, post_suffix = method_region(self.post)
        self.assertEqual((stock_prefix, stock_suffix), (post_prefix, post_suffix))


class ExtractedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = STOCK_FIXTURE.read_bytes()
        cls.post = POST_FIXTURE.read_bytes()

    def test_stock_negative_control_desynchronizes_after_stop(self):
        grammar, logger = grammar_instance(self.stock)
        self.assertFalse(grammar.accept_tokens("stock", [STOP, TRAILING]))
        self.assertEqual(grammar.matcher.calls, [STOP, TRAILING])
        self.assertEqual(grammar.num_processed_tokens, 1)
        self.assertTrue(grammar.matcher.is_terminated())
        self.assertFalse(grammar._is_terminated)
        self.assertEqual(len(logger.errors), 1)

    def test_patched_accept_stops_batch_and_later_accept_is_noop(self):
        grammar, logger = grammar_instance(self.post)
        self.assertTrue(grammar.accept_tokens("post", [ORDINARY, STOP, TRAILING]))
        self.assertEqual(grammar.matcher.calls, [ORDINARY, STOP])
        self.assertEqual(grammar.num_processed_tokens, 2)
        self.assertTrue(grammar.is_terminated())
        before = list(grammar.matcher.calls)
        self.assertTrue(grammar.accept_tokens("post", [TRAILING]))
        self.assertEqual(grammar.matcher.calls, before)
        self.assertEqual(grammar.num_processed_tokens, 2)
        self.assertEqual(logger.errors, [])

    def test_patched_pretermination_rejection_remains_false(self):
        grammar, logger = grammar_instance(self.post)
        self.assertFalse(grammar.accept_tokens("reject", [REJECT]))
        self.assertEqual(grammar.matcher.calls, [REJECT])
        self.assertEqual(grammar.num_processed_tokens, 0)
        self.assertFalse(grammar.is_terminated())
        self.assertEqual(len(logger.errors), 1)

    def test_patched_validation_stops_at_stop_and_rolls_back(self):
        grammar, _ = grammar_instance(self.post)
        self.assertEqual(grammar.validate_tokens([STOP, TRAILING]), [STOP])
        self.assertEqual(grammar.matcher.calls, [STOP])
        self.assertEqual(grammar.matcher.rollback_calls, [1])
        self.assertEqual(grammar.matcher.accepted, [])
        self.assertFalse(grammar.matcher.is_terminated())
        self.assertFalse(grammar.is_terminated())

    def test_patched_validation_after_cached_termination_is_noop(self):
        grammar, _ = grammar_instance(self.post)
        self.assertTrue(grammar.accept_tokens("terminate", [STOP]))
        grammar.matcher.calls.clear()
        grammar.matcher.rollback_calls.clear()
        self.assertEqual(grammar.validate_tokens([TRAILING]), [])
        self.assertEqual(grammar.matcher.calls, [])
        self.assertEqual(grammar.matcher.rollback_calls, [])

    def test_reset_clears_matcher_counter_and_cached_flag(self):
        grammar, _ = grammar_instance(self.post)
        grammar.accept_tokens("terminate", [ORDINARY, STOP])
        grammar.reset()
        self.assertEqual(grammar.matcher.reset_calls, 1)
        self.assertEqual(grammar.num_processed_tokens, 0)
        self.assertFalse(grammar._is_terminated)
        self.assertFalse(grammar.matcher.is_terminated())

    def test_termination_state_is_instance_local(self):
        first, _ = grammar_instance(self.post)
        second, _ = grammar_instance(self.post)
        first.accept_tokens("first", [STOP, TRAILING])
        second.accept_tokens("second", [ORDINARY])
        self.assertTrue(first.is_terminated())
        self.assertFalse(second.is_terminated())
        self.assertEqual((first.num_processed_tokens, second.num_processed_tokens), (1, 1))
        self.assertEqual(first.matcher.calls, [STOP])
        self.assertEqual(second.matcher.calls, [ORDINARY])


class PatcherTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = STOCK_FIXTURE.read_bytes()
        cls.post = POST_FIXTURE.read_bytes()

    def write_target(self, directory: Path, data: bytes, mode: int = 0o640) -> Path:
        target = directory / "backend_xgrammar.py"
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
        )

    def assert_snapshot(self, target: Path, expected) -> None:
        self.assertEqual(self.snapshot(target), expected)

class PatcherCompatibilityTests(PatcherTestBase):
    def test_exact_stock_applies_atomically_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o604)
            result = PATCHER.apply(target, provider())
            self.assertEqual(result.outcome, "applied")
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(sha256(target.read_bytes()), POST_SHA256)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)
            self.assertEqual(temp_artifacts(directory), [])

    def test_second_apply_and_exact_post_are_successful_nowrites(self):
        for initial, first_apply in ((self.stock, True), (self.post, False)):
            with self.subTest(first_apply=first_apply), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, initial)
                if first_apply:
                    PATCHER.apply(target, provider())
                before = self.snapshot(target)
                result = PATCHER.apply(target, provider())
                self.assertEqual(result.outcome, "already-patched")
                self.assert_snapshot(target, before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_check_and_status_classify_without_writes(self):
        for data, expected_state in ((self.stock, "stock-compatible"), (self.post, "patched")):
            with self.subTest(state=expected_state), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), data)
                before = self.snapshot(target)
                inspection = PATCHER.inspect_target(target, provider())
                self.assertEqual(inspection.state, expected_state)
                self.assert_snapshot(target, before)

    def test_cli_check_and_status_exit_codes(self):
        cases = (
            (self.stock, ["--check"], 0, "compatible: stock-compatible"),
            (self.post, ["--check"], 0, "compatible: patched"),
            (self.stock, ["--status"], 1, "stock-compatible"),
            (self.post, ["--status"], 0, "patched"),
            (self.stock + b"# drift\n", ["--status"], 2, "incompatible"),
        )
        for data, argv, expected_rc, output in cases:
            with self.subTest(argv=argv, rc=expected_rc), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), data)
                before = self.snapshot(target)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with mock.patch.object(PATCHER, "PRODUCTION_TARGET", target), mock.patch.object(
                    PATCHER.importlib.metadata, "version", provider()
                ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    rc = PATCHER.main(argv)
                self.assertEqual(rc, expected_rc)
                self.assertIn(output, stdout.getvalue())
                self.assert_snapshot(target, before)

    def test_wrong_or_missing_metadata_fails_unchanged(self):
        cases = (
            provider(vllm="0.25.2"),
            provider(xgrammar="0.2.4"),
            provider(missing="vllm"),
            provider(missing="xgrammar"),
        )
        for metadata_provider in cases:
            with self.subTest(provider=metadata_provider), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), self.stock)
                before = self.snapshot(target)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, metadata_provider)
                self.assert_snapshot(target, before)

    def test_missing_directory_and_symlink_targets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            missing = directory / "missing.py"
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(missing, provider())

            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(directory, provider())

            real = self.write_target(directory, self.stock)
            link = directory / "link.py"
            link.symlink_to(real)
            before = self.snapshot(real)
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(link, provider())
            self.assert_snapshot(real, before)

    def test_every_drift_class_fails_without_write(self):
        prefix, old_region, suffix = method_region(self.stock)
        _, new_region, _ = method_region(self.post)

        def flipped(data: bytes, offset: int) -> bytes:
            replacement = b"X" if data[offset : offset + 1] != b"X" else b"Y"
            return data[:offset] + replacement + data[offset + 1 :]

        variants = {
            "before-region": flipped(self.stock, max(0, len(prefix) - 2)),
            "inside-old-anchor": flipped(self.stock, len(prefix) + 10),
            "after-region": flipped(self.stock, len(prefix) + len(old_region) + 2),
            "inside-new-anchor": flipped(self.post, len(prefix) + 10),
            "duplicate-old": self.stock + old_region,
            "duplicate-new": self.post + new_region,
            "mixed-old-new": self.stock.replace(VALIDATE_OLD, VALIDATE_NEW, 1),
            "partial-accept-only": self.stock.replace(ACCEPT_OLD, ACCEPT_NEW, 1),
            "invalid-utf8": self.stock[:-1] + b"\xff",
        }
        for name, data in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, data)
                before = self.snapshot(target)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, provider())
                self.assert_snapshot(target, before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_candidate_syntax_failure_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            before = self.snapshot(target)
            real_compile = PATCHER._compile_source

            def fail_post(data: bytes, label: str) -> None:
                if sha256(data) == POST_SHA256:
                    raise PATCHER.CompatibilityError("injected candidate syntax failure")
                real_compile(data, label)

            with mock.patch.object(PATCHER, "_compile_source", side_effect=fail_post):
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(target, provider())
            self.assert_snapshot(target, before)
            self.assertEqual(temp_artifacts(directory), [])


class PatcherFailureRecoveryTests(PatcherTestBase):
    def test_staging_creation_and_write_failures_leave_original(self):
        failures = (
            ("mkstemp", mock.patch.object(PATCHER.tempfile, "mkstemp", side_effect=OSError("injected"))),
            ("write", mock.patch.object(PATCHER, "_write_all", side_effect=OSError("injected"))),
        )
        for name, patch_context in failures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, self.stock)
                before = self.snapshot(target)
                with patch_context:
                    with self.assertRaises(OSError):
                        PATCHER.apply(target, provider())
                self.assert_snapshot(target, before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_replace_failure_leaves_original_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            before = self.snapshot(target)
            with mock.patch.object(PATCHER.os, "replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    PATCHER.apply(target, provider())
            self.assert_snapshot(target, before)
            self.assertEqual(temp_artifacts(directory), [])

    def test_exception_after_successful_replace_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o605)
            original = target.read_bytes()
            real_replace = PATCHER.os.replace
            calls = 0

            def replace_then_raise(src, dst):
                nonlocal calls
                calls += 1
                real_replace(src, dst)
                if calls == 1:
                    raise OSError("injected after rename")

            with mock.patch.object(PATCHER.os, "replace", side_effect=replace_then_raise):
                with self.assertRaises(OSError):
                    PATCHER.apply(target, provider())
            self.assertEqual(calls, 2)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o605)
            self.assertEqual(temp_artifacts(directory), [])

    def test_post_publish_verification_failure_rolls_back(self):
        # KeyboardInterrupt doubles as the breadth probe: only the patcher's
        # ``except BaseException`` rollback path restores the published target.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o604)
            original = target.read_bytes()
            with mock.patch.object(
                PATCHER,
                "_verify_published",
                side_effect=KeyboardInterrupt("injected verification interrupt"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    PATCHER.apply(target, provider())
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)
            self.assertEqual(temp_artifacts(directory), [])

    def test_failed_rollback_is_fatal_and_cleans_staging_files(self):
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

            with mock.patch.object(PATCHER.os, "replace", side_effect=fail_second_replace), mock.patch.object(
                PATCHER, "_verify_published", side_effect=OSError("injected verification failure")
            ):
                with self.assertRaises(PATCHER.RollbackError):
                    PATCHER.apply(target, provider())
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(temp_artifacts(directory), [])


class StartupWiringTests(unittest.TestCase):
    # Gate execution behavior (disabled/non-"1" values skip, enabled invokes,
    # failure blocks exec) is covered once, in scripts/test-python-hotfix-failclosed.py;
    # these tests pin the static compose and launcher wiring.
    def compose_gate(self) -> str:
        token = "python3 /opt/hotfix-vllm-issue136-xgrammar-termination.py"
        matches = [line.strip() for line in COMPOSE.read_text(encoding="utf-8").splitlines() if token in line]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_compose_mount_default_gate_and_exec_order(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        mount = (
            "${DSPARK_ISSUE136_XGRAMMAR_HOTFIX:-./patches/"
            "hotfix-vllm-issue136-xgrammar-termination.py}:"
            "/opt/hotfix-vllm-issue136-xgrammar-termination.py:ro"
        )
        env_default = (
            'DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX: '
            '"${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}"'
        )
        gate = self.compose_gate()
        self.assertEqual(compose.count(mount), 1)
        self.assertEqual(compose.count(env_default), 1)
        self.assertEqual(
            gate,
            'if [ "$${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}" = "1" ]; then '
            'python3 /opt/hotfix-vllm-issue136-xgrammar-termination.py || exit 1; fi;',
        )
        self.assertLess(compose.index(gate), compose.index("exec /usr/local/bin/vllm serve"))
        hotfix_loop = next(line for line in compose.splitlines() if "for _hf in" in line)
        self.assertNotIn("issue136", hotfix_loop.lower())

    def test_launcher_syncs_and_preflights_both_nodes_before_any_up(self):
        source = START.read_text(encoding="utf-8")
        regular_check = (
            '[ "${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}" = "1" ] '
            '&& { [ ! -f "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ] '
            '|| [ -L "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" ]; }'
        )
        sync = (
            'scp "$DSPARK_ISSUE136_XGRAMMAR_HOTFIX" '
            '"${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/'
            'hotfix-vllm-issue136-xgrammar-termination.py"'
        )
        worker_check = (
            'run --rm --no-deps --entrypoint python3 vllm-dspark '
            '/opt/hotfix-vllm-issue136-xgrammar-termination.py --check'
        )
        head_check = (
            'compose_base 0 "" run --rm --no-deps --entrypoint python3 '
            'vllm-dspark /opt/hotfix-vllm-issue136-xgrammar-termination.py --check'
        )
        worker_up = 'echo "Starting DSpark worker on ${WORKER_HOST}..."'
        head_up = 'echo "Starting DSpark head..."'
        for token in (regular_check, sync, worker_check, head_check):
            self.assertIn(token, source)
        self.assertIn(
            'issue136 XGrammar termination hotfix: ${DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX:-0}',
            source,
        )
        self.assertIn(
            "DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX=$REMOTE_ISSUE136_ENABLE",
            source,
        )
        self.assertIn(
            "DSPARK_ISSUE136_XGRAMMAR_HOTFIX='./patches/hotfix-vllm-issue136-xgrammar-termination.py'",
            source,
        )
        positions = [source.index(token) for token in (sync, worker_check, head_check, worker_up, head_up)]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
