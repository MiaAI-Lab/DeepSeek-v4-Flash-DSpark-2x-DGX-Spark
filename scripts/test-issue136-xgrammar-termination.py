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
MANAGER_STOCK_FIXTURE = FIXTURES / "structured_output_init-752a3a504.py"
MANAGER_POST_FIXTURE = FIXTURES / "structured_output_init-752a3a504-pr53046.py"
MANAGER_PRISTINE_FIXTURE = FIXTURES / "structured_output_init-752a3a504-pristine.py"
MANAGER_PRISTINE_POST_FIXTURE = FIXTURES / "structured_output_init-752a3a504-pristine-pr53046.py"
# Fixture provenance: extracted from the pinned image (registry manifest
# sha256:a8394849… = local config/image ID sha256:3430d661…). The non-pristine
# pair additionally carries the default #44993 grammar-advance train.
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
MANAGER_STOCK_SHA256 = "e782163b8a83d58e61a655df042d3126cde8c913a2eeaf9d4a061148cd8e5c77"
MANAGER_POST_SHA256 = "3dff0e1e35f04f35e8c50c17d9efa65cd5fc8db1f25d4eb5d536b6e61114a616"
MANAGER_STOCK_BYTES = 21_979
MANAGER_POST_BYTES = 22_271
MANAGER_PRISTINE_SHA256 = "fd23813a4e0d8cdc93fa1e6687e5a4f4e514b0ae37dec707d50d840771390818"
MANAGER_PRISTINE_POST_SHA256 = "53186ccf86e3d620a9aa91af8c541516f0b45a3f640d937607a252bc42f376e6"
MANAGER_PRISTINE_BYTES = 22_076
MANAGER_PRISTINE_POST_BYTES = 22_368
# Independent literal description of the single upstream #53046 hunk (issue
# 210): inside grammar_bitmask's speculative window, a post-reasoning-end
# draft is validated before it is accepted, so a grammar-invalid draft that
# predates the bitmask no longer trips a spurious FSM error.
MANAGER_OLD = (
    b"                    if advance_grammar and not grammar.is_terminated():\n"
    b"                        accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        if accepted:\n"
    b"                            state_advancements += 1\n"
    b"                        elif not post_reasoning_end_in_window:\n"
    b"                            raise AssertionError(\n"
    b"                                (token, req_id, scheduled_spec_decode_tokens)\n"
    b"                            )\n"
)
MANAGER_NEW = (
    b"                    if advance_grammar and not grammar.is_terminated():\n"
    b"                        if post_reasoning_end_in_window:\n"
    b"                            accepted = bool(grammar.validate_tokens([token]))\n"
    b"                            if accepted:\n"
    b"                                accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        else:\n"
    b"                            accepted = grammar.accept_tokens(req_id, [token])\n"
    b"                        if accepted:\n"
    b"                            state_advancements += 1\n"
    b"                        elif not post_reasoning_end_in_window:\n"
    b"                            raise AssertionError(\n"
    b"                                (token, req_id, scheduled_spec_decode_tokens)\n"
    b"                            )\n"
)
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
BACKEND_SPEC = next(t for t in PATCHER.TARGETS if t.name == "backend_xgrammar")
MANAGER_SPEC = next(t for t in PATCHER.TARGETS if t.name == "structured_output_init")

def load_transaction_parser():
    spec = importlib.util.spec_from_file_location(
        "hotfix_atomic_transaction",
        ROOT / "scripts" / "test-hotfix-atomic-transaction.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the atomic-transaction harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def grammar_advance_hunks():
    """The production #44993 __init__.py hunks, parsed from the shell script."""
    module = load_transaction_parser()
    found = [
        h
        for h in module.hunks("hotfix-dsv4-grammar-advance.sh")
        if h.path.endswith("v1/structured_output/__init__.py")
    ]
    if len(found) != 2:
        raise RuntimeError("expected exactly two #44993 __init__.py hunks")
    return found


def apply_hunks(text: str, found) -> str:
    for hunk in found:
        if text.count(hunk.old) != hunk.expect:
            raise AssertionError(f"hunk anchor count != expect={hunk.expect} ({hunk.label})")
        text = text.replace(hunk.old, hunk.new, hunk.expect)
    return text


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
    return sorted(directory.glob("*.issue136-*.tmp"))


class FixtureProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock = STOCK_FIXTURE.read_bytes()
        cls.post = POST_FIXTURE.read_bytes()
        cls.manager_stock = MANAGER_STOCK_FIXTURE.read_bytes()
        cls.manager_post = MANAGER_POST_FIXTURE.read_bytes()

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
    def test_manager_fixture_identities(self):
        self.assertEqual(
            (len(self.manager_stock), len(self.manager_post)),
            (MANAGER_STOCK_BYTES, MANAGER_POST_BYTES),
        )
        self.assertEqual(
            (sha256(self.manager_stock), sha256(self.manager_post)),
            (MANAGER_STOCK_SHA256, MANAGER_POST_SHA256),
        )
        compile(self.manager_stock.decode("utf-8"), MANAGER_STOCK_FIXTURE.name, "exec")
        compile(self.manager_post.decode("utf-8"), MANAGER_POST_FIXTURE.name, "exec")

    def test_only_the_53046_hunk_changes_in_the_manager(self):
        self.assertEqual(self.manager_stock.count(MANAGER_OLD), 1)
        self.assertEqual(self.manager_stock.count(MANAGER_NEW), 0)
        generated = self.manager_stock.replace(MANAGER_OLD, MANAGER_NEW, 1)
        self.assertEqual(generated, self.manager_post)
        self.assertEqual(self.manager_post.count(MANAGER_NEW), 1)
        self.assertEqual(self.manager_post.count(MANAGER_OLD), 0)
    def test_pristine_variant_hunk_and_identity(self):
        pristine = MANAGER_PRISTINE_FIXTURE.read_bytes()
        pristine_post = MANAGER_PRISTINE_POST_FIXTURE.read_bytes()
        self.assertEqual((len(pristine), len(pristine_post)), (MANAGER_PRISTINE_BYTES, MANAGER_PRISTINE_POST_BYTES))
        self.assertEqual((sha256(pristine), sha256(pristine_post)), (MANAGER_PRISTINE_SHA256, MANAGER_PRISTINE_POST_SHA256))
        # the #53046 anchor is present exactly once in both stock variants
        self.assertEqual(pristine.count(MANAGER_OLD), 1)
        self.assertEqual(pristine.replace(MANAGER_OLD, MANAGER_NEW, 1), pristine_post)
        compile(pristine.decode("utf-8"), MANAGER_PRISTINE_FIXTURE.name, "exec")
        compile(pristine_post.decode("utf-8"), MANAGER_PRISTINE_POST_FIXTURE.name, "exec")

    def test_both_application_orders_converge(self):
        # Apply the PRODUCTION #44993 hunks (parsed from the shell script, not
        # fixture-derived) around the #53046 hunk in both orders; both must
        # yield the checked-in post-#44993 + #53046 post-image byte-for-byte.
        pristine = MANAGER_PRISTINE_FIXTURE.read_text()
        pristine_post = MANAGER_PRISTINE_POST_FIXTURE.read_text()
        hunks44993 = grammar_advance_hunks()
        # the production #44993 literals transform pristine into the post-#44993
        # stock fixture exactly (also proves the parsed hunks are complete)
        trained = apply_hunks(pristine, hunks44993)
        self.assertEqual(trained, self.manager_stock.decode("utf-8"))
        manager_old = MANAGER_OLD.decode("utf-8")
        manager_new = MANAGER_NEW.decode("utf-8")
        # production boot order: #44993 train first, then this chain
        self.assertEqual(
            trained.replace(manager_old, manager_new, 1),
            self.manager_post.decode("utf-8"),
        )
        # reverse order: #53046 on pristine, then the same production hunks
        self.assertEqual(
            apply_hunks(pristine.replace(manager_old, manager_new, 1), hunks44993),
            self.manager_post.decode("utf-8"),
        )

    def test_region_hashes_and_outside_bytes(self):
        stock_prefix, stock_region, stock_suffix = method_region(self.stock)
        post_prefix, post_region, post_suffix = method_region(self.post)
        self.assertEqual(sha256(stock_region), STOCK_REGION_SHA256)
        self.assertEqual(sha256(post_region), POST_REGION_SHA256)
        self.assertEqual(stock_prefix, post_prefix)
        self.assertEqual(stock_suffix, post_suffix)

    def test_production_patcher_generates_checked_in_post_fixture(self):
        self.assertEqual(PATCHER.build_candidate(BACKEND_SPEC, BACKEND_SPEC.variants[0], self.stock), self.post)

    def test_issue44993_patch_targets_are_disjoint(self):
        # File-level disjointness no longer holds: the #136+#210 chain and #44993
        # both touch v1/structured_output/__init__.py.  Region-level disjointness
        # does: the production #44993 hunks and the #53046 hunk anchor on
        # non-overlapping text (proven constructively in
        # test_both_application_orders_converge), and #44993's shell script does
        # not touch the chain's backend file.
        grammar_advance = GRAMMAR_ADVANCE.read_text(encoding="utf-8")
        self.assertNotIn("backend_xgrammar.py", grammar_advance)
        self.assertIn("v1/structured_output/__init__.py", grammar_advance)
        self.assertIn("v1/core/sched/scheduler.py", grammar_advance)
        stock = MANAGER_PRISTINE_FIXTURE.read_text()
        for hunk in grammar_advance_hunks():
            self.assertNotIn(hunk.old, MANAGER_NEW.decode("utf-8"))
            self.assertNotIn(hunk.new, MANAGER_NEW.decode("utf-8"))


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
        cls.manager_stock = MANAGER_STOCK_FIXTURE.read_bytes()
        cls.manager_post = MANAGER_POST_FIXTURE.read_bytes()

    def write_manager(self, directory: Path, data: bytes, mode: int = 0o640) -> Path:
        target = directory / "structured_output_init.py"
        target.write_bytes(data)
        target.chmod(mode)
        return target

    def targets(self, backend: Path, manager: Path) -> dict:
        return {
            "backend_xgrammar": backend,
            "structured_output_init": manager,
        }

    def publish_one(self, spec, target, data):
        inspection = PATCHER.inspect_target(spec, target)
        candidate = PATCHER.build_candidate(spec, inspection.variant, inspection.data)
        PATCHER._publish_one(spec, target, inspection, candidate)
        return inspection, candidate

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
            manager = self.write_manager(directory, self.manager_stock, 0o604)
            result = PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(result.outcome, "applied")
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(manager.read_bytes(), self.manager_post)
            self.assertEqual(sha256(target.read_bytes()), POST_SHA256)
            self.assertEqual(sha256(manager.read_bytes()), MANAGER_POST_SHA256)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)
            self.assertEqual(stat.S_IMODE(manager.stat().st_mode), 0o604)
            self.assertEqual(temp_artifacts(directory), [])

    def test_second_apply_and_exact_post_are_successful_nowrites(self):
        for (backend_data, manager_data), first_apply in (
            ((self.stock, self.manager_stock), True),
            ((self.post, self.manager_post), False),
        ):
            with self.subTest(first_apply=first_apply), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, backend_data)
                manager = self.write_manager(directory, manager_data)
                if first_apply:
                    PATCHER.apply(self.targets(target, manager), provider())
                before = self.snapshot(target)
                before_manager = self.snapshot(manager)
                result = PATCHER.apply(self.targets(target, manager), provider())
                self.assertEqual(result.outcome, "already-patched")
                self.assert_snapshot(target, before)
                self.assert_snapshot(manager, before_manager)
                self.assertEqual(temp_artifacts(directory), [])

    def test_check_and_status_classify_without_writes(self):
        for data, expected_state in ((self.stock, "stock"), (self.post, "patched")):
            with self.subTest(state=expected_state), tempfile.TemporaryDirectory() as tmp:
                target = self.write_target(Path(tmp), data)
                before = self.snapshot(target)
                inspection = PATCHER.inspect_target(BACKEND_SPEC, target)
                self.assertEqual(inspection.state, expected_state)
                self.assert_snapshot(target, before)

    def test_cli_check_and_status_exit_codes(self):
        cases = (
            ((self.stock, self.manager_stock), ["--check"], 0, "compatible: stock"),
            ((self.post, self.manager_post), ["--check"], 0, "compatible: patched"),
            ((self.post, self.manager_stock), ["--check"], 0, "compatible: partial-legacy"),
            ((self.stock, self.manager_post), ["--check"], 2, "incompatible: partial-invalid"),
            ((self.stock, self.manager_stock), ["--status"], 1, "stock"),
            ((self.post, self.manager_post), ["--status"], 0, "patched"),
            ((self.post, self.manager_stock), ["--status"], 1, "partial-invalid"),
            ((self.stock, self.manager_post), ["--status"], 1, "partial-invalid"),
            ((self.stock + b"# drift\n", self.manager_stock), ["--status"], 2, "incompatible"),
        )
        for (backend_data, manager_data), argv, expected_rc, output in cases:
            with self.subTest(argv=argv, rc=expected_rc), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, backend_data)
                manager = self.write_manager(directory, manager_data)
                before = self.snapshot(target)
                before_manager = self.snapshot(manager)
                stdout = io.StringIO()
                stderr = io.StringIO()
                with mock.patch.object(
                    PATCHER, "_production_targets", lambda: self.targets(target, manager)
                ), mock.patch.object(
                    PATCHER.importlib.metadata, "version", provider()
                ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    rc = PATCHER.main(argv)
                self.assertEqual(rc, expected_rc)
                self.assertIn(output, stdout.getvalue())
                self.assert_snapshot(target, before)
                self.assert_snapshot(manager, before_manager)

    def test_wrong_or_missing_metadata_fails_unchanged(self):
        cases = (
            provider(vllm="0.25.2"),
            provider(xgrammar="0.2.4"),
            provider(missing="vllm"),
            provider(missing="xgrammar"),
        )
        for metadata_provider in cases:
            with self.subTest(provider=metadata_provider), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                target = self.write_target(directory, self.stock)
                manager = self.write_manager(directory, self.manager_stock)
                before = self.snapshot(target)
                before_manager = self.snapshot(manager)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(self.targets(target, manager), metadata_provider)
                self.assert_snapshot(target, before)
                self.assert_snapshot(manager, before_manager)

    def test_missing_directory_and_symlink_targets_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manager = self.write_manager(directory, self.manager_stock)
            missing = directory / "missing.py"
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(self.targets(missing, manager), provider())

            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(self.targets(directory, manager), provider())

            real = self.write_target(directory, self.stock)
            link = directory / "link.py"
            link.symlink_to(real)
            before = self.snapshot(real)
            before_manager = self.snapshot(manager)
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(self.targets(link, manager), provider())
            self.assert_snapshot(real, before)
            self.assert_snapshot(manager, before_manager)

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
                manager = self.write_manager(directory, self.manager_stock)
                before = self.snapshot(target)
                before_manager = self.snapshot(manager)
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(self.targets(target, manager), provider())
                self.assert_snapshot(target, before)
                self.assert_snapshot(manager, before_manager)
                self.assertEqual(temp_artifacts(directory), [])

        with self.subTest(name="manager-drift"), tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_stock + b"# drift\n")
            before = self.snapshot(target)
            before_manager = self.snapshot(manager)
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(self.targets(target, manager), provider())
            self.assert_snapshot(target, before)
            self.assert_snapshot(manager, before_manager)
            self.assertEqual(temp_artifacts(directory), [])

    def test_candidate_syntax_failure_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_stock)
            before = self.snapshot(target)
            before_manager = self.snapshot(manager)
            real_compile = PATCHER._compile_source

            def fail_post(data: bytes, label: str) -> None:
                if sha256(data) == POST_SHA256:
                    raise PATCHER.CompatibilityError("injected candidate syntax failure")
                real_compile(data, label)

            with mock.patch.object(PATCHER, "_compile_source", side_effect=fail_post):
                with self.assertRaises(PATCHER.CompatibilityError):
                    PATCHER.apply(self.targets(target, manager), provider())
            self.assert_snapshot(target, before)
            self.assert_snapshot(manager, before_manager)
            self.assertEqual(temp_artifacts(directory), [])

class PatcherFailureRecoveryTests(PatcherTestBase):
    # Single-file publication/failure semantics are exercised through the
    # per-target transaction helper directly; apply() itself is reserved for
    # valid chain states (see ChainTransactionTests).

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
                        self.publish_one(BACKEND_SPEC, target, self.stock)
                self.assert_snapshot(target, before)
                self.assertEqual(temp_artifacts(directory), [])

    def test_replace_failure_leaves_original_and_cleans_temps(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            before = self.snapshot(target)
            with mock.patch.object(PATCHER.os, "replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    self.publish_one(BACKEND_SPEC, target, self.stock)
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
                    self.publish_one(BACKEND_SPEC, target, self.stock)
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
                    self.publish_one(BACKEND_SPEC, target, self.stock)
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
                    self.publish_one(BACKEND_SPEC, target, self.stock)
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(temp_artifacts(directory), [])


class ChainTransactionTests(PatcherTestBase):
    def test_chain_publishes_both_files_in_chain_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_stock)
            order = []
            real_publish = PATCHER._publish_one

            def recording(spec, *args, **kwargs):
                order.append(spec.name)
                return real_publish(spec, *args, **kwargs)

            with mock.patch.object(PATCHER, "_publish_one", side_effect=recording):
                result = PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(result.outcome, "applied")
            self.assertEqual(order, ["backend_xgrammar", "structured_output_init"])
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(manager.read_bytes(), self.manager_post)
            self.assertEqual(temp_artifacts(directory), [])

    def test_chain_second_file_failure_rolls_back_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o604)
            manager = self.write_manager(directory, self.manager_stock)
            real_replace = PATCHER.os.replace

            def fail_manager_candidate(src, dst):
                if Path(dst) == manager and "issue136-" in Path(src).name:
                    raise OSError("injected manager publish failure")
                return real_replace(src, dst)

            with mock.patch.object(PATCHER.os, "replace", side_effect=fail_manager_candidate):
                with self.assertRaises(OSError):
                    PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(target.read_bytes(), self.stock)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)
            self.assertEqual(manager.read_bytes(), self.manager_stock)
            self.assertEqual(temp_artifacts(directory), [])

    def test_chain_rollback_refuses_to_clobber_concurrent_change(self):
        # The manager publish fails after a concurrent writer changes the
        # already-published backend bytes: rollback must refuse to clobber.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_stock)
            concurrent = b"# concurrent change\n" + self.stock
            real_verify = PATCHER._verify_published

            def corrupt_backend_then_fail(spec, *args, **kwargs):
                if spec.name == "structured_output_init":
                    target.write_bytes(concurrent)
                    raise OSError("injected manager verification failure")
                return real_verify(spec, *args, **kwargs)

            with mock.patch.object(PATCHER, "_verify_published", side_effect=corrupt_backend_then_fail):
                with self.assertRaises(PATCHER.RollbackError):
                    PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(target.read_bytes(), concurrent)
            self.assertEqual(manager.read_bytes(), self.manager_stock)
            self.assertEqual(temp_artifacts(directory), [])
    def test_rollback_refuses_after_metadata_only_change(self):
        # chmod lands on the already-published backend before the manager
        # publish fails: the rollback must refuse (metadata mismatch), leaving
        # the patched bytes with their new mode unclobbered.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock, 0o640)
            manager = self.write_manager(directory, self.manager_stock)
            real_verify = PATCHER._verify_published

            def chmod_backend_then_fail(spec, *args, **kwargs):
                if spec.name == "structured_output_init":
                    target.chmod(0o600)
                    raise OSError("injected manager verification failure")
                return real_verify(spec, *args, **kwargs)

            with mock.patch.object(PATCHER, "_verify_published", side_effect=chmod_backend_then_fail):
                with self.assertRaises(PATCHER.RollbackError):
                    PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(manager.read_bytes(), self.manager_stock)

    def test_rollback_refuses_after_symlink_swap_with_same_bytes(self):
        # The backend path is swapped for a symlink whose target holds the
        # exact published bytes: content compare alone would be fooled; the
        # regular-file guard must refuse the rollback.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_stock)
            real_verify = PATCHER._verify_published

            def swap_backend_then_fail(spec, *args, **kwargs):
                if spec.name == "structured_output_init":
                    twin = directory / "twin.py"
                    twin.write_bytes(self.post)
                    target.unlink()
                    target.symlink_to(twin)
                    raise OSError("injected manager verification failure")
                return real_verify(spec, *args, **kwargs)

            with mock.patch.object(PATCHER, "_verify_published", side_effect=swap_backend_then_fail):
                with self.assertRaises(PATCHER.RollbackError):
                    PATCHER.apply(self.targets(target, manager), provider())
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(manager.read_bytes(), self.manager_stock)

    def test_legacy_partial_completes_manager_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.post, 0o604)
            manager = self.write_manager(directory, self.manager_stock)
            before = self.snapshot(target)
            result = PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(result.outcome, "applied")
            self.assert_snapshot(target, before)
            self.assertEqual(manager.read_bytes(), self.manager_post)
            self.assertEqual(temp_artifacts(directory), [])

    def test_inverse_partial_is_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, self.manager_post)
            before = self.snapshot(target)
            before_manager = self.snapshot(manager)
            with self.assertRaises(PATCHER.CompatibilityError):
                PATCHER.apply(self.targets(target, manager), provider())
            self.assert_snapshot(target, before)
            self.assert_snapshot(manager, before_manager)
            self.assertEqual(temp_artifacts(directory), [])
    def test_pristine_manager_variant_applies_to_its_own_postimage(self):
        # DSPARK_SKIP_HOTFIX=1 boots present the pristine pinned image bytes;
        # the chain must select the pristine variant and publish its postimage.
        pristine = MANAGER_PRISTINE_FIXTURE.read_bytes()
        pristine_post = MANAGER_PRISTINE_POST_FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = self.write_target(directory, self.stock)
            manager = self.write_manager(directory, pristine)
            result = PATCHER.apply(self.targets(target, manager), provider())
            self.assertEqual(result.outcome, "applied")
            self.assertEqual(target.read_bytes(), self.post)
            self.assertEqual(manager.read_bytes(), pristine_post)
            self.assertEqual(temp_artifacts(directory), [])

    def test_inspect_classifies_both_manager_variants(self):
        for fixture, variant_name, state in (
            (self.manager_stock, "post-44993", "stock"),
            (self.manager_post, "post-44993", "patched"),
            (MANAGER_PRISTINE_FIXTURE.read_bytes(), "pristine", "stock"),
            (MANAGER_PRISTINE_POST_FIXTURE.read_bytes(), "pristine", "patched"),
        ):
            with self.subTest(variant=variant_name, state=state), tempfile.TemporaryDirectory() as tmp:
                manager = self.write_manager(Path(tmp), fixture)
                inspection = PATCHER.inspect_target(MANAGER_SPEC, manager)
                self.assertEqual(inspection.state, state)
                self.assertEqual(inspection.variant.name, variant_name)


def manager_window_step(region: bytes):
    """Build a callable from the exact grammar_bitmask window region bytes.

    The region lives at a 20-space base indent inside the manager's spec-decode
    loop; it is re-homed into a function whose parameters provide the region's
    free names.  The text under test is byte-exact from the fixtures.
    """
    text = region.decode("utf-8")
    body = "".join(line[16:] for line in text.splitlines(keepends=True))
    source = (
        "def _step(grammar, token, req_id, scheduled_spec_decode_tokens,"
        " post_reasoning_end_in_window, state_advancements, advance_grammar):\n"
        + body
        + "    return state_advancements\n"
    )
    namespace: dict = {}
    exec(compile(source, "manager-region", "exec"), namespace)
    return namespace["_step"]


class FakeWindowGrammar:
    def __init__(self, validate_result=None, accept_result=True, terminated=False):
        self._validate_result = [] if validate_result is None else validate_result
        self._accept_result = accept_result
        self._terminated = terminated
        self.calls = []

    def is_terminated(self):
        return self._terminated

    def validate_tokens(self, tokens):
        self.calls.append(("validate", list(tokens)))
        return self._validate_result

    def accept_tokens(self, req_id, tokens):
        self.calls.append(("accept", req_id, list(tokens)))
        return self._accept_result


class ManagerWindowBehaviorTests(unittest.TestCase):
    def test_patched_post_reasoning_invalid_draft_is_validated_not_accepted(self):
        step = manager_window_step(MANAGER_NEW)
        grammar = FakeWindowGrammar(validate_result=[], accept_result=True)
        state = step(grammar, 9, "r1", (), True, 0, True)
        self.assertEqual(state, 0)
        self.assertEqual(grammar.calls, [("validate", [9])])

    def test_patched_post_reasoning_valid_draft_validates_then_accepts(self):
        step = manager_window_step(MANAGER_NEW)
        grammar = FakeWindowGrammar(validate_result=[9], accept_result=True)
        state = step(grammar, 9, "r1", (), True, 0, True)
        self.assertEqual(state, 1)
        self.assertEqual(grammar.calls, [("validate", [9]), ("accept", "r1", [9])])

    def test_patched_post_reasoning_accept_failure_after_validate_is_tolerated(self):
        step = manager_window_step(MANAGER_NEW)
        grammar = FakeWindowGrammar(validate_result=[9], accept_result=False)
        state = step(grammar, 9, "r1", (), True, 0, True)
        self.assertEqual(state, 0)
        self.assertEqual(grammar.calls, [("validate", [9]), ("accept", "r1", [9])])

    def test_patched_outside_window_accepts_directly_and_raises_on_reject(self):
        step = manager_window_step(MANAGER_NEW)
        grammar = FakeWindowGrammar(accept_result=False)
        with self.assertRaises(AssertionError):
            step(grammar, 9, "r1", (), False, 0, True)
        self.assertEqual(grammar.calls, [("accept", "r1", [9])])

    def test_patched_terminated_grammar_is_untouched(self):
        step = manager_window_step(MANAGER_NEW)
        grammar = FakeWindowGrammar(terminated=True)
        state = step(grammar, 9, "r1", (), True, 0, True)
        self.assertEqual(state, 0)
        self.assertEqual(grammar.calls, [])

    def test_stock_region_accepts_blindly_in_window(self):
        # Negative control: the stock region calls accept_tokens without
        # validating first — the upstream source of the spurious "Failed to
        # advance FSM" errors.  The patched region must not do this.
        step = manager_window_step(MANAGER_OLD)
        grammar = FakeWindowGrammar(accept_result=False)
        state = step(grammar, 9, "r1", (), True, 0, True)
        self.assertEqual(state, 0)
        self.assertEqual(grammar.calls, [("accept", "r1", [9])])

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
