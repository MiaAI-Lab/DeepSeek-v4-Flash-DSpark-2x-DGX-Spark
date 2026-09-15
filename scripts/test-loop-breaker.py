#!/usr/bin/env python3
"""Behavioral tests for the issue #82 loop-breaker hotfix (CPU only).

vLLM is not importable here, so the tests apply the patch to a synthetic module
that carries the exact production anchors and then *exec* the patched module:
firing thresholds, the counting-window bound, the fence/indent/structural
false-positive controls, the DSML hold and the runtime knob parsing are
exercised as code. CLI tests cover apply/--check/--status, the default-OFF and
skip gates, complete injected-block classification, atomic same-directory
writes with mode preservation, the fail-closed restore paths (including a
post-write re-read/decode failure) and the bounded knob parsing. Rank-parity
cases run the launcher's resolved-control/forwarding slices and worker sync
statements against recording ssh/scp stubs, so the argv and the transferred
bytes each rank receives are asserted without touching a host or a container.

    python3 scripts/test-loop-breaker.py -q
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import re
import shlex
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-loop-breaker.py"

# The patched module warns through its own logger when a knob is malformed (it
# runs inside the import block, before the module logger exists). Keep that out
# of the test output; test_malformed_knob_warning_names_the_variable asserts on
# the record itself.
logging.getLogger("vllm.loop-breaker").addHandler(logging.NullHandler())


def _load():
    spec = importlib.util.spec_from_file_location("hotfix_loop_breaker", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_source(mod, header: str = "stock") -> str:
    """Synthetic detokenizer carrying the exact production anchors.

    The class mirrors the pinned ``BaseIncrementalDetokenizer`` shape the patch
    anchors on (``__init__`` generation block, ``update()`` stop-string tail,
    module import block), so applying the patch here exercises the same
    transformations the container boot performs.
    """
    import_block = (
        mod.IMPORT_OLD_STOCK
        if header == "stock"
        else mod.IMPORT_OLD_WITH_SUPPRESS
    )
    return (
        'import logging\n\n'
        'logger = logging.getLogger("loop-breaker-fixture")\n\n'
        + import_block
        + '\nclass DetokenizerFixture:\n'
        '    def __init__(self):\n'
        '        self.tokens = 0\n'
        + mod.INIT_OLD
        + '\n'
        '    def num_output_tokens(self):\n'
        '        return self.tokens\n'
        '\n'
        '    def update(self, stop=None):\n'
        '        stop_string = None\n'
        '        if stop is not None:\n'
        + mod.RET_OLD
        + '\n'
        '    def feed(self, text, tokens=10):\n'
        '        self.output_text += text\n'
        '        self.tokens += tokens\n'
    )


LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
PARITY_BEGIN = "# Issue #82 loop-breaker rank parity (begin)."
PARITY_END = "# Issue #82 loop-breaker rank parity (end)."
FORWARD_BEGIN = "remote_compose() {"
FORWARD_END = "log_since() {"
WORKER_SYNC_BEGIN = 'DSPARK_LOOP_BREAKER_HOTFIX="${DSPARK_LOOP_BREAKER_HOTFIX:-'
WORKER_SYNC_END = 'DSPARK_ASSISTANT_FINAL_HOTFIX="${DSPARK_ASSISTANT_FINAL_HOTFIX:-'
WORKER2_SYNC_BEGIN = '  if [ -f "$DSPARK_C128A_PREFILL_CACHE_HOTFIX" ]'
WORKER2_SYNC_END = '  sync_tp3_patch_dir "$WORKER2_HOST" "$REMOTE_WORKER2_DIR"'
CONTROL_KEYS = (
    "DSPARK_LOOP_BREAKER",
    "DSPARK_SKIP_LOOP_BREAKER_HOTFIX",
    "DSPARK_LOOP_BREAKER_REPEATS",
    "DSPARK_LOOP_BREAKER_SHORT_REPEATS",
    "DSPARK_LOOP_BREAKER_MIN_TOKENS",
    "DSPARK_LOOP_BREAKER_HOTFIX",
)
CANONICAL_REMOTE_PATCH = "./patches/hotfix-dsv4-loop-breaker.py"

SSH_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$SSH_RECORD"
"""

SCP_STUB = """#!/usr/bin/env bash
n=$(wc -l < "$SCP_RECORD")
printf '%s\\t%s\\n' "$1" "${@: -1}" >> "$SCP_RECORD"
cp -- "$1" "$SCP_STAGE/$n"
"""


def _launcher_region(begin: str, end: str) -> str:
    """Executable launcher slice between two anchors that must be unique."""
    text = LAUNCHER.read_text()
    for anchor in (begin, end):
        if text.count(anchor) != 1:
            raise AssertionError(f"launcher anchor is not unique: {anchor!r}")
    start = text.index(begin)
    return text[start : text.index(end, start)]


def _remote_assignments(line: str) -> dict[str, str]:
    """Loop-breaker assignments the ssh recorder saw on a remote command."""
    found = {}
    for key in CONTROL_KEYS:
        match = re.search(rf"(?<![A-Z_]){key}=('[^']*'|\S+)", line)
        if match is not None:
            found[key] = match.group(1).strip("'\"")
    return found


class PatchTextTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_patched_source_from_both_header_states_is_executable(self):
        for header in ("stock", "suppress"):
            with self.subTest(header=header):
                patched, status = self.mod.apply_text(
                    fixture_source(self.mod, header)
                )
                self.assertEqual(status, "applied")
                compile(patched, "<fixture>", "exec")

    def test_second_apply_is_skipped_and_byte_identical(self):
        once, status = self.mod.apply_text(fixture_source(self.mod))
        self.assertEqual(status, "applied")
        twice, status = self.mod.apply_text(once)
        self.assertEqual(status, "skipped")
        self.assertEqual(once, twice)

    def test_partial_patch_is_not_an_idempotent_hit(self):
        once, _ = self.mod.apply_text(fixture_source(self.mod))
        partial = once.replace("        if stop_string is None and _LB_ENABLED:\n", "")
        self.assertNotEqual(partial, once)
        new, status = self.mod.apply_text(partial)
        self.assertEqual(status, "partial")
        self.assertEqual(new, partial)
        self.assertEqual(self.mod.classify(partial), "partial")
        self.assertEqual(self.mod.classify(once), "applied")

    def test_missing_anchors_are_named_and_never_written(self):
        drifted = 'print("not a detokenizer")\n'
        new, status = self.mod.apply_text(drifted)
        self.assertTrue(status.startswith("missing:"), status)
        self.assertEqual(new, drifted)
        for part in ("import", "init", "return"):
            self.assertIn(part, status)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _fixture(self, mode: int = 0o640) -> Path:
        target = self.root / "detokenizer.py"
        target.write_text(fixture_source(self.mod))
        target.chmod(mode)
        return target

    def _run(self, argv, env) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env, clear=True):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = self.mod.main(["hotfix-dsv4-loop-breaker.py", *argv])
        return code, out.getvalue(), err.getvalue()

    def _leftovers(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file())

    def test_disabled_boot_ignores_knobs_and_writes_nothing(self):
        target = self._fixture()
        original = target.read_bytes()
        for value in (None, "0", "true", "2", ""):
            with self.subTest(value=value):
                env = {"DSPARK_LOOP_BREAKER_REPEATS": "not-an-int"}
                if value is not None:
                    env["DSPARK_LOOP_BREAKER"] = value
                code, out, _ = self._run([str(target)], env)
                self.assertEqual(code, 0, out)
                self.assertIn("disabled", out)
        code, out, _ = self._run(["--check", str(target)], {"DSPARK_LOOP_BREAKER_REPEATS": "abc"})
        self.assertEqual(code, 0, out)
        self.assertIn("disabled", out)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_skip_flag_skips_apply_and_check_but_not_status(self):
        target = self._fixture()
        original = target.read_bytes()
        env = {"DSPARK_LOOP_BREAKER": "1", "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1"}
        code, out, _ = self._run([str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("DSPARK_SKIP_LOOP_BREAKER_HOTFIX=1", out)
        self.assertEqual(target.read_bytes(), original)
        code, out, _ = self._run(["--check", str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("skipped", out)
        self.assertEqual(target.read_bytes(), original)

        # --status stays a byte-state query: the skip flag must not mask it.
        code, out, _ = self._run(["--status", str(target)], env)
        self.assertEqual(code, 1, out)
        self.assertIn("STOCK", out)
        applied, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied)
        code, out, _ = self._run(["--status", str(target)], env)
        self.assertEqual(code, 0, out)
        self.assertIn("APPLIED", out)

    def test_enabled_apply_writes_atomically_and_preserves_mode(self):
        target = self._fixture(0o640)
        seen_dir = []
        real_mkstemp = tempfile.mkstemp

        def spy(*args, **kwargs):
            seen_dir.append(kwargs.get("dir"))
            return real_mkstemp(*args, **kwargs)

        with mock.patch.object(tempfile, "mkstemp", spy):
            code, out, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertEqual(seen_dir, [str(target.parent)])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
        self.assertEqual(self.mod.classify(target.read_text()), "applied")
        self.assertEqual(self._leftovers(), ["detokenizer.py"])
        compile(target.read_text(), str(target), "exec")

    def test_write_failure_leaves_target_untouched(self):
        target = self._fixture()
        original = target.read_bytes()
        with mock.patch.object(
            tempfile, "mkstemp", side_effect=OSError("read-only filesystem")
        ):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_uncompilable_patch_never_reaches_the_target(self):
        target = self._fixture()
        original = target.read_bytes()
        try:
            self.mod.RET_NEW = "        return (\n"
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        finally:
            reloaded = _load()
            self.mod.RET_NEW = reloaded.RET_NEW
        self.assertEqual(code, 1)
        self.assertIn("does not compile", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_failed_postcondition_restores_the_original(self):
        target = self._fixture()
        original = target.read_bytes()
        real_write = self.mod.write_atomically
        calls = []

        def tampering_write(path, payload, mode):
            calls.append(payload)
            if len(calls) == 1:
                # Simulate a corrupted first write: the postcondition must see
                # it, restore the original and fail closed.
                real_write(path, payload.replace("def _lb_check(self)", "def _lb_check("), mode)
            else:
                real_write(path, payload, mode)

        with mock.patch.object(self.mod, "write_atomically", tampering_write):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("post-apply verification failed", err)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_post_write_read_failure_restores_the_original(self):
        failures = (
            ("OSError", OSError("simulated re-read failure")),
            (
                "UnicodeDecodeError",
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "simulated decode failure"),
            ),
        )
        for name, failure in failures:
            with self.subTest(name=name):
                target = self._fixture()
                original = target.read_bytes()
                real_read_text = Path.read_text
                reads = []

                def failing_read_text(path, *args, **kwargs):
                    reads.append(path)
                    if len(reads) == 2:
                        raise failure
                    return real_read_text(path, *args, **kwargs)

                with mock.patch.object(Path, "read_text", failing_read_text):
                    code, _, err = self._run(
                        [str(target)], {"DSPARK_LOOP_BREAKER": "1"}
                    )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn("original restored", err)
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(self._leftovers(), ["detokenizer.py"])

    def test_post_write_read_failure_reports_a_failed_restore(self):
        target = self._fixture()
        real_read_text = Path.read_text
        real_write = self.mod.write_atomically
        reads, writes = [], []

        def failing_read_text(path, *args, **kwargs):
            reads.append(path)
            if len(reads) == 2:
                raise OSError("simulated re-read failure")
            return real_read_text(path, *args, **kwargs)

        def failing_restore(path, payload, mode):
            writes.append(payload)
            if len(writes) == 2:
                raise OSError("simulated restore failure")
            real_write(path, payload, mode)

        with mock.patch.object(Path, "read_text", failing_read_text), \
                mock.patch.object(self.mod, "write_atomically", failing_restore):
            code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("cannot restore", err)
        # The restore failed, so the written patch is still on disk; the CLI
        # must have said so instead of reporting success.
        self.assertEqual(self.mod.classify(target.read_text()), "applied")

    def test_huge_digit_knobs_fail_closed_without_conversion_errors(self):
        target = self._fixture()
        original = target.read_bytes()
        huge = "9" * 5000
        for name in (
            "DSPARK_LOOP_BREAKER_REPEATS",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS",
        ):
            with self.subTest(name=name):
                code, _, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", name: huge}
                )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn(name, err)
                self.assertEqual(target.read_bytes(), original)

    def test_zero_padded_in_range_knob_still_applies(self):
        target = self._fixture()
        code, out, err = self._run(
            [str(target)],
            {
                "DSPARK_LOOP_BREAKER": "1",
                "DSPARK_LOOP_BREAKER_REPEATS": "0" * 5000 + "6",
            },
        )
        self.assertEqual(code, 0, out + err)
        self.assertEqual(self.mod.classify(target.read_text()), "applied")

    def test_enabled_rejects_malformed_and_out_of_range_knobs(self):
        target = self._fixture()
        original = target.read_bytes()
        rows = (
            ("DSPARK_LOOP_BREAKER_REPEATS", "abc"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "-3"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "1"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "1025"),
            ("DSPARK_LOOP_BREAKER_REPEATS", " 6"),
            ("DSPARK_LOOP_BREAKER_REPEATS", "6.0"),
            ("DSPARK_LOOP_BREAKER_SHORT_REPEATS", "0"),
            ("DSPARK_LOOP_BREAKER_MIN_TOKENS", "1000001"),
            ("DSPARK_LOOP_BREAKER_MIN_TOKENS", "\uff16"),
        )
        for name, value in rows:
            with self.subTest(name=name, value=value):
                code, _, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", name: value}
                )
                self.assertEqual(code, 1)
                self.assertIn("FAIL-CLOSED", err)
                self.assertIn(name, err)
                self.assertEqual(target.read_bytes(), original)

    def test_enabled_accepts_documented_defaults_and_bounds(self):
        rows = (
            {},
            {"DSPARK_LOOP_BREAKER_REPEATS": "2"},
            {"DSPARK_LOOP_BREAKER_SHORT_REPEATS": "1024"},
            {"DSPARK_LOOP_BREAKER_MIN_TOKENS": "0"},
            {"DSPARK_LOOP_BREAKER_MIN_TOKENS": "1000000"},
        )
        for extra in rows:
            with self.subTest(extra=extra):
                target = self._fixture()
                code, out, err = self._run(
                    [str(target)], {"DSPARK_LOOP_BREAKER": "1", **extra}
                )
                self.assertEqual(code, 0, out + err)
                self.assertEqual(self.mod.classify(target.read_text()), "applied")
        # A second run over a fully patched target stays a byte-identical no-op.
        target = self._fixture()
        self.assertEqual(self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})[0], 0)
        once = target.read_bytes()
        code, out, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertIn("skipped", out)
        self.assertEqual(target.read_bytes(), once)

    def test_check_mode_reports_ready_drift_and_partial(self):
        ready = self._fixture()
        code, out, err = self._run(["--check", str(ready)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out + err)
        self.assertIn("READY", out)
        self.assertIn("repeats=6", out)
        self.assertIn("min_tokens=64", out)

        applied_text, _ = self.mod.apply_text(fixture_source(self.mod))
        ready.write_text(applied_text)
        code, out, _ = self._run(["--check", str(ready)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 0, out)
        self.assertIn("already applied", out)

        drifted = self.root / "drifted.py"
        drifted.write_text('print("not a detokenizer")\n')
        code, _, err = self._run(["--check", str(drifted)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("expected anchors", err)

        partial = self.root / "partial.py"
        partial.write_text(
            applied_text.replace("        if stop_string is None and _LB_ENABLED:\n", "")
        )
        code, _, err = self._run(["--check", str(partial)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("partial patch", err)
        missing = self.root / "absent.py"
        code, out, _ = self._run(["--status", str(missing)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("NOT APPLIED", out)

    def test_status_is_nonzero_on_partial_and_missing_state(self):
        target = self._fixture()
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("STOCK", out)

        applied_text, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied_text)
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 0, out)
        self.assertIn("APPLIED", out)

        target.write_text(
            applied_text.replace("    def _lb_check(self) -> str | None:\n", "")
        )
        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

    def test_incomplete_injected_code_outside_the_markers_is_refused(self):
        target = self._fixture()
        applied_text, _ = self.mod.apply_text(target.read_text())
        # Executable injected state is deleted while every substring the old
        # classifier counted would still match: the module raises on first use.
        gutted = applied_text
        for line in (
            "        self._lb_counts: dict[str, int] = {}\n",
            "        self._lb_hist: deque[str] = deque()\n",
            "        self._lb_scan: int = 0\n",
        ):
            gutted = gutted.replace(line, "")
        self.assertNotEqual(gutted, applied_text)
        target.write_text(gutted)

        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

        code, _, err = self._run(["--check", str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("partial patch", err)

        code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)
        self.assertEqual(target.read_text(), gutted)

    def test_uncompilable_applied_source_is_not_reported_applied(self):
        target = self._fixture()
        applied_text, _ = self.mod.apply_text(target.read_text())
        target.write_text(applied_text + "def broken(:\n")

        code, out, _ = self._run(["--status", str(target)], {})
        self.assertEqual(code, 1)
        self.assertIn("PARTIAL", out)

        code, _, err = self._run([str(target)], {"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(code, 1)
        self.assertIn("FAIL-CLOSED", err)


class DetectorTest(unittest.TestCase):
    """Exec the patched fixture and drive the detector through update()."""

    def setUp(self):
        self.mod = _load()
        self.patched, status = self.mod.apply_text(fixture_source(self.mod))
        self.assertEqual(status, "applied")

    def _exec(self, env: dict[str, str]) -> dict:
        with mock.patch.dict(os.environ, env, clear=True):
            namespace: dict = {"__name__": "loop_breaker_fixture"}
            exec(compile(self.patched, "<fixture>", "exec"), namespace)
        return namespace

    def _detector(self, env: dict[str, str] | None = None):
        namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", **(env or {})})
        return namespace["DetokenizerFixture"]()

    @staticmethod
    def _repeat(fixture, line: str, times: int, tokens: int = 20) -> str | None:
        result = None
        for _ in range(times):
            fixture.feed(line + "\n", tokens)
            result = fixture.update()
        return result

    def test_prose_repeat_fires_exactly_at_the_configured_bound(self):
        for count, expected in ((5, None), (6, "loop-breaker")):
            with self.subTest(count=count):
                fixture = self._detector()
                self.assertEqual(self._repeat(fixture, "Doing it.", count), expected)

    def test_short_sentence_tier_needs_its_own_bound(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Now.", 14), None)
        self.assertEqual(self._repeat(fixture, "Now.", 1), "loop-breaker")
        # A six-repeat short line is prose noise, not a loop.
        self.assertEqual(self._repeat(self._detector(), "Yes.", 6), None)
        # Below the short tier's length floor and never sentence-terminated.
        self.assertEqual(self._repeat(self._detector(), "ok", 40), None)

    def test_token_floor_defers_fire_until_the_bound_is_passed(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Doing it.", 6, tokens=5), None)
        fixture.tokens = 64
        self.assertEqual(self._repeat(fixture, "Doing it.", 1), "loop-breaker")

    def test_configured_bounds_and_window_follow_the_knobs(self):
        fixture = self._detector(
            {"DSPARK_LOOP_BREAKER_REPEATS": "2", "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0"}
        )
        self.assertEqual(self._repeat(fixture, "Doing it.", 2), "loop-breaker")

    def test_fenced_code_repetition_never_fires(self):
        for fence in ("```", "~~~", "```python"):
            with self.subTest(fence=fence):
                fixture = self._detector()
                fixture.feed(fence + "\n", 5)
                # Unindented, so only the fence state can suppress the count.
                self.assertIsNone(
                    self._repeat(fixture, "return value_that_repeats()", 40)
                )
                fixture.feed(fence + "\n", 5)
                fixture.feed("Closing prose line.\n", 5)
                self.assertIsNone(fixture.update())

    def test_indented_code_repetition_never_fires(self):
        for line in ("    value = compute(queue)", "\tvalue = compute(queue)"):
            with self.subTest(line=line.split("=")[0].strip() or "tab"):
                fixture = self._detector()
                self.assertIsNone(self._repeat(fixture, line, 40))

    def test_structural_and_marker_lines_are_never_counted(self):
        rows = (
            '{"tool": "call", "name": "bash"}',
            "const x = `template ${value}`;",
            "<\uff5cDSML\uff5cfunction_calls>",
            "---",
            "| column a | column b |",
            "x" * 200,
        )
        for line in rows:
            with self.subTest(line=line[:24]):
                fixture = self._detector()
                self.assertIsNone(self._repeat(fixture, line, 40))

    def test_dsml_tail_holds_fire_until_the_tool_block_closes(self):
        fixture = self._detector()
        self.assertEqual(self._repeat(fixture, "Doing it.", 6), "loop-breaker")
        fixture.feed("<\uff5cDSML\uff5cinvoke name=\"bash\">\n", 10)
        self.assertIsNone(fixture.update())
        fixture.feed("x" * 500 + "\n", 20)
        self.assertEqual(self._repeat(fixture, "Doing it.", 1), "loop-breaker")

    def test_counting_window_decays_and_stays_bounded(self):
        knobs = {
            "DSPARK_LOOP_BREAKER_REPEATS": "2",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "2",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0",
        }
        within = self._detector(knobs)
        within.feed("alpha repeated line\n", 5)
        within.feed("".join(f"filler {i}\n" for i in range(60)), 100)
        within.feed("alpha repeated line\n", 5)
        self.assertEqual(within.update(), "loop-breaker")

        decayed = self._detector(knobs)
        decayed.feed("alpha repeated line\n", 5)
        decayed.feed("".join(f"filler {i}\n" for i in range(200)), 300)
        decayed.feed("alpha repeated line\n", 5)
        self.assertIsNone(decayed.update())

        long_generation_ns = self._exec({"DSPARK_LOOP_BREAKER": "1", **knobs})
        window = long_generation_ns["_LB_WINDOW"]
        long_generation = long_generation_ns["DetokenizerFixture"]()
        for i in range(500):
            long_generation.feed(f"line {i} of a long answer\n", 3)
            long_generation.update()
        self.assertLessEqual(len(long_generation._lb_counts), window)
        self.assertLessEqual(len(long_generation._lb_hist), window)

    def test_alternating_two_line_babble_fires(self):
        fixture = self._detector()
        result = None
        for _ in range(6):
            fixture.feed("Doing it.\n", 10)
            result = fixture.update()
            fixture.feed("Running.\n", 10)
            result = fixture.update()
        self.assertEqual(result, "loop-breaker")

    def test_healthy_verbose_turn_is_never_touched(self):
        fixture = self._detector()
        transcript = [
            "# Result",
            "",
            "The run completed with the expected shape.",
            "",
            "```python",
            'print("value")',
            "print(value)",
            "```",
            "",
            '{"tool": "bash", "command": "ls"}',
            "",
            "Next step:",
            "- Inspect the log.",
            "- Inspect the log.",
            "- Rerun the sampler.",
            "",
            "The sampler produced the expected sequence.",
        ]
        for line in transcript:
            fixture.feed(line + "\n", 20)
            self.assertIsNone(fixture.update(), line)
        # Four prose repeats of the same line, inside the bound.
        self.assertIsNone(self._repeat(fixture, "Retrying the request now.", 4))

    def test_disabled_module_is_inert_and_parses_no_knob(self):
        for value in (None, "0", "true", "2", ""):
            with self.subTest(value=value):
                env = {"DSPARK_LOOP_BREAKER_REPEATS": "not-an-int"}
                if value is not None:
                    env["DSPARK_LOOP_BREAKER"] = value
                namespace = self._exec(env)
                self.assertFalse(namespace["_LB_ENABLED"])
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 50))

    def test_malformed_knobs_disarm_instead_of_aborting_import(self):
        rows = ("abc", "1", "0", "-3", "1025", " 6", "6.0", "\uff16")
        for value in rows:
            with self.subTest(value=value):
                namespace = self._exec(
                    {"DSPARK_LOOP_BREAKER": "1", "DSPARK_LOOP_BREAKER_REPEATS": value}
                )
                self.assertFalse(namespace["_LB_ENABLED"])
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 40))

    def test_malformed_knob_warning_names_the_variable(self):
        with self.assertLogs("vllm.loop-breaker", level="WARNING") as captured:
            namespace = self._exec(
                {
                    "DSPARK_LOOP_BREAKER": "1",
                    "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "many",
                }
            )
        self.assertFalse(namespace["_LB_ENABLED"])
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn("DSPARK_LOOP_BREAKER_SHORT_REPEATS", message)
        self.assertIn("'many'", message)
        self.assertIn("[loop-breaker]", message)

    def test_huge_digit_knobs_disarm_the_import(self):
        huge = "9" * 5000
        for name in (
            "DSPARK_LOOP_BREAKER_REPEATS",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS",
        ):
            with self.subTest(name=name):
                with self.assertLogs("vllm.loop-breaker", level="WARNING") as captured:
                    namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", name: huge})
                self.assertFalse(namespace["_LB_ENABLED"])
                self.assertEqual(len(captured.records), 1)
                self.assertIn(name, captured.records[0].getMessage())
                fixture = namespace["DetokenizerFixture"]()
                self.assertIsNone(self._repeat(fixture, "Doing it.", 40))

    def test_zero_padded_knob_arms_with_the_normalized_value(self):
        namespace = self._exec(
            {"DSPARK_LOOP_BREAKER": "1", "DSPARK_LOOP_BREAKER_REPEATS": "0" * 5000 + "6"}
        )
        self.assertTrue(namespace["_LB_ENABLED"])
        self.assertEqual(namespace["_LB_REPEATS"], 6)

    def test_cli_and_runtime_knob_bounds_agree(self):
        specs = self.mod.KNOB_SPECS
        defaults = {name: default for name, default, _, _ in specs}
        for name, _, low, high in specs:
            for value in (str(low), str(high), str(low - 1), str(high + 1)):
                with self.subTest(name=name, value=value):
                    namespace = self._exec({"DSPARK_LOOP_BREAKER": "1", name: value})
                    runtime_armed = namespace["_LB_ENABLED"]
                    in_range = low <= int(value) <= high
                    self.assertEqual(runtime_armed, in_range, (name, value))
                    try:
                        self.mod.resolve_knobs({name: value})
                        cli_accepted = True
                    except self.mod.LoopBreakerConfigError:
                        cli_accepted = False
                    self.assertEqual(cli_accepted, runtime_armed, (name, value))
                    observed = tuple(
                        namespace[key]
                        for key in (
                            "_LB_REPEATS",
                            "_LB_SHORT_REPEATS",
                            "_LB_MIN_TOKENS",
                        )
                    )
                    if runtime_armed:
                        expected = dict(defaults)
                        expected[name] = int(value)
                        self.assertEqual(
                            observed,
                            tuple(expected[spec[0]] for spec in specs),
                        )
                    else:
                        self.assertEqual(observed, (0, 0, 0))

    def test_defaults_are_used_when_no_knob_is_set(self):
        namespace = self._exec({"DSPARK_LOOP_BREAKER": "1"})
        self.assertEqual(
            (
                namespace["_LB_REPEATS"],
                namespace["_LB_SHORT_REPEATS"],
                namespace["_LB_MIN_TOKENS"],
            ),
            tuple(spec[1] for spec in self.mod.KNOB_SPECS),
        )


class ComposeWiringTest(unittest.TestCase):
    """Defect guard: the knobs must actually reach the container.

    scripts/test-python-hotfix-failclosed.py executes the real compose command
    line, but it sets the variables directly: only the service environment block
    can put them into the container (the original defect was a chain gate with no
    matching environment entries, i.e. an inert kill switch). This guard checks
    those declarations with their documented defaults, supplementing — not
    replacing — the behavioral chain and detector tests.
    """

    DECLARED = {
        "DSPARK_LOOP_BREAKER": "0",
        "DSPARK_LOOP_BREAKER_REPEATS": "6",
        "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "15",
        "DSPARK_LOOP_BREAKER_MIN_TOKENS": "64",
        "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "0",
    }

    def test_service_declares_every_knob_with_its_default(self):
        text = (ROOT / "docker-compose.dspark.yml").read_text()
        for knob, default in self.DECLARED.items():
            with self.subTest(knob=knob):
                match = re.search(
                    rf'^\s+{knob}: "\$\{{{knob}:-([^}}]*)\}}"$', text, re.M
                )
                self.assertIsNotNone(match, f"{knob} has no environment entry")
                self.assertEqual(match.group(1), default)

    def test_example_env_documents_the_flag_as_default_off(self):
        text = (ROOT / ".env.dspark.example").read_text()
        match = re.search(r"^DSPARK_LOOP_BREAKER=(\S+)$", text, re.M)
        self.assertIsNotNone(match, "DSPARK_LOOP_BREAKER is not documented")
        self.assertEqual(match.group(1), "0")


class RankParityForwardingTest(unittest.TestCase):
    """Defect guard: both ranks must run the same resolved loop-breaker state.

    ComposeWiringTest pins the service declarations and
    scripts/test-python-hotfix-failclosed.py executes the real Compose command
    line; neither runs the launcher's remote commands. These cases execute the
    launcher's resolved-control block, its remote Compose wrappers and its
    worker sync statements against recording ssh/scp stubs, so a control that
    lives only in the head environment (absent from the pushed env file) or an
    override the worker2 tar would shadow is caught on the argv each rank
    receives and on the transferred bytes.
    """

    PREFLIGHT = (
        "NODE_RANK={rank} HEADLESS=1 VLLM_HOST_IP='10.0.0.2{rank}' docker compose "
        "-p dspark --env-file .env.dspark -f docker-compose.dspark.yml run --rm "
        "--no-deps --entrypoint python3 vllm-dspark "
        "/opt/hotfix-dsv4-loop-breaker.py --check"
    )
    RUNTIME = (
        "NODE_RANK={rank} HEADLESS=1 VLLM_HOST_IP='10.0.0.2{rank}' docker compose "
        "-p dspark --env-file .env.dspark -f docker-compose.dspark.yml up -d"
    )
    CALLS = (
        ("pre-flight --check", "remote_compose", "worker-one", 1),
        ("pre-flight --check", "remote_compose2", "worker-two", 2),
        ("runtime up -d", "remote_compose", "worker-one", 1),
        ("runtime up -d", "remote_compose2", "worker-two", 2),
    )

    def _remote_command_record(self, ambient: dict[str, str]) -> list[str]:
        """Run the real forwarding slices; return the ssh argv per rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            ssh_stub = bin_dir / "ssh"
            ssh_stub.write_text(SSH_STUB)
            ssh_stub.chmod(0o755)
            record = root / "ssh-record.txt"
            record.touch()
            calls = []
            for kind, wrapper, _, rank in self.CALLS:
                shape = self.PREFLIGHT if kind.startswith("pre-flight") else self.RUNTIME
                calls.append(f'{wrapper} "{shape.format(rank=rank)}"')
            script = "\n".join(
                [
                    "set -euo pipefail",
                    "WORKER_HOST=worker-one",
                    "WORKER2_HOST=worker-two",
                    "REMOTE_WORKER_DIR=/srv/dspark",
                    "REMOTE_WORKER2_DIR=/srv/dspark2",
                    'REMOTE_COMPOSE="cd /srv/dspark && env -u NODE_RANK '
                    'COMPOSE_DISABLE_ENV_FILE=1"',
                    'REMOTE_COMPOSE2="cd /srv/dspark2 && env -u NODE_RANK '
                    'COMPOSE_DISABLE_ENV_FILE=1"',
                    "TP_SIZE=2",
                    "NNODES=2",
                    "REMOTE_C128A_PREFILL_CACHE=0",
                    "REMOTE_ISSUE136_ENABLE=0",
                    "REMOTE_ISSUE191_ENABLE=0",
                    "REMOTE_ISSUE191_RETRIES=2",
                    "REMOTE_ISSUE191_MODE=failclosed",
                    "REMOTE_ISSUE191_THINKOFF=1",
                    "REMOTE_ASYNC_SCHEDULING=1",
                    "REMOTE_DSPARK_BLOCK_K=0",
                    "REMOTE_ROPE_SWA_FIX=0",
                    "REMOTE_DSPARK_SWA_PREFIX=0",
                    "REMOTE_DSML_RECOVERY=0",
                    "REMOTE_MXFP4_INDEXER=0",
                    "REMOTE_ISSUE144_EFFORT_ALIGN=0",
                    "remote_nccl_env() { printf \"NCCL_IB_HCA='rocep1s0f0'\"; }",
                    "remote_nccl_env2() { printf \"NCCL_IB_HCA='rocep2s0f0'\"; }",
                    _launcher_region(PARITY_BEGIN, PARITY_END),
                    _launcher_region(FORWARD_BEGIN, FORWARD_END),
                    *calls,
                ]
            )
            base = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("DSPARK_LOOP_BREAKER")
                and key != "DSPARK_SKIP_LOOP_BREAKER_HOTFIX"
            }
            env = {
                **base,
                **ambient,
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "SSH_RECORD": str(record),
            }
            proc = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return record.read_text().splitlines()

    def test_both_ranks_resolve_and_receive_the_same_controls(self):
        # Ambient-only controls (no .env.dspark entry): the head runs enabled
        # while a rank that only saw the pushed file would stay disabled.
        record = self._remote_command_record(
            {
                "DSPARK_LOOP_BREAKER": "1",
                "DSPARK_LOOP_BREAKER_REPEATS": "9",
                "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0",
            }
        )
        expected = {
            "DSPARK_LOOP_BREAKER": "1",
            "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "0",
            "DSPARK_LOOP_BREAKER_REPEATS": "9",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "15",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS": "0",
            "DSPARK_LOOP_BREAKER_HOTFIX": CANONICAL_REMOTE_PATCH,
        }
        self.assertEqual(len(record), len(self.CALLS), record)
        for (kind, _, host, _), line in zip(self.CALLS, record):
            where = f"{kind} on {host}: {line}"
            self.assertEqual(line.split()[0], host, where)
            self.assertEqual(_remote_assignments(line), expected, where)

    def test_skip_flag_and_selected_override_keep_the_canonical_path(self):
        record = self._remote_command_record(
            {
                "DSPARK_LOOP_BREAKER": "1",
                "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1",
                "DSPARK_LOOP_BREAKER_HOTFIX": "/opt/head/selected-loop-breaker.py",
            }
        )
        expected = {
            "DSPARK_LOOP_BREAKER": "1",
            "DSPARK_SKIP_LOOP_BREAKER_HOTFIX": "1",
            "DSPARK_LOOP_BREAKER_REPEATS": "6",
            "DSPARK_LOOP_BREAKER_SHORT_REPEATS": "15",
            "DSPARK_LOOP_BREAKER_MIN_TOKENS": "64",
            "DSPARK_LOOP_BREAKER_HOTFIX": CANONICAL_REMOTE_PATCH,
        }
        self.assertEqual(len(record), len(self.CALLS), record)
        for line in record:
            self.assertEqual(_remote_assignments(line), expected, line)

    def test_selected_override_bytes_reach_both_worker_canonical_paths(self):
        # The repository copy stands for what the worker2 tar ships; only a
        # selected override may be what the pre-flight reads on either rank.
        repository_copy = b'"""repository loop-breaker copy"""\n'
        selected_override = b'"""selected loop-breaker override"""\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_dir = root / "head"
            patches = script_dir / "patches"
            patches.mkdir(parents=True)
            (patches / "hotfix-dsv4-loop-breaker.py").write_bytes(repository_copy)
            selected = root / "selected-loop-breaker.py"
            selected.write_bytes(selected_override)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name, body in (("ssh", SSH_STUB), ("scp", SCP_STUB)):
                stub = bin_dir / name
                stub.write_text(body)
                stub.chmod(0o755)
            stage = root / "stage"
            stage.mkdir()
            ssh_record = root / "ssh-record.txt"
            ssh_record.touch()
            scp_record = root / "scp-record.txt"
            scp_record.touch()
            script = "\n".join(
                [
                    "set -euo pipefail",
                    f"SCRIPT_DIR={shlex.quote(str(script_dir))}",
                    "WORKER_HOST=worker-one",
                    "WORKER2_HOST=worker-two",
                    "WORKER_DIR=/srv/dspark",
                    "REMOTE_WORKER_DIR=/srv/dspark",
                    "REMOTE_WORKER2_DIR=/srv/dspark2",
                    f"DSPARK_LOOP_BREAKER_HOTFIX={shlex.quote(str(selected))}",
                    f"DSPARK_C128A_PREFILL_CACHE_HOTFIX="
                    f"{shlex.quote(str(root / 'absent-c128a-prefill-cache.py'))}",
                    "ENABLE_VLLM_GB10_PATCH=0",
                    _launcher_region(WORKER_SYNC_BEGIN, WORKER_SYNC_END),
                    _launcher_region(WORKER2_SYNC_BEGIN, WORKER2_SYNC_END),
                ]
            )
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "SSH_RECORD": str(ssh_record),
                "SCP_RECORD": str(scp_record),
                "SCP_STAGE": str(stage),
            }
            proc = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            delivered = {}
            for index, line in enumerate(scp_record.read_text().splitlines()):
                _, destination = line.split("\t")
                delivered[destination] = (stage / str(index)).read_bytes()
        self.assertEqual(
            sorted(delivered),
            [
                "worker-one:/srv/dspark/patches/hotfix-dsv4-loop-breaker.py",
                "worker-two:/srv/dspark2/patches/hotfix-dsv4-loop-breaker.py",
            ],
        )
        for destination, payload in delivered.items():
            self.assertEqual(payload, selected_override, destination)


if __name__ == "__main__":
    unittest.main()
