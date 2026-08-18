#!/usr/bin/env python3
"""CPU regression tests for the Q/KV RMSNorm split-blocks hotfix (vLLM #49283).

No GPU, no container, no network: everything runs against the two byte-frozen
upstream blobs in `tests/fixtures/vllm-49283/` (see PROVENANCE.md there).

What is actually proven here:

  * the frozen fixtures still hash to the upstream SHAs they claim;
  * all five anchors occur **exactly once** in upstream's pre-image, so
    `str.replace(old, new, 1)` cannot silently patch one of two sites;
  * applying the patcher to the pre-image reproduces upstream's own post-image
    **byte for byte** — the backport is upstream's diff, not a paraphrase;
  * the transform adds exactly one module-level definition and touches no KV
    budget or GPU-memory knob;
  * re-running re-validates instead of double-patching, and leaves bytes stable;
  * anchor drift, duplicated anchors and a spoofed "already applied" comment are
    all handled without corrupting the target;
  * a post-write self-check failure restores the original bytes;
  * `--status` and `--dry-run` report the right state, exit 0/1/2, and never
    write;
  * the compose/start/env/CI wiring is default-OFF, exactly-`1` gated,
    fail-closed, and reaches **both** ranks.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-qk-rmsnorm-split-blocks.py"
FIXTURES = ROOT / "tests" / "fixtures" / "vllm-49283"
BASE = FIXTURES / "fused_qk_rmsnorm.base.py"
HEAD = FIXTURES / "fused_qk_rmsnorm.head.py"
COMPOSE = ROOT / "docker-compose.dspark.yml"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
START = ROOT / "start-deepseek-v4-flash-dspark.sh"
CI_VALIDATE = ROOT / "scripts" / "ci-validate.sh"
ENVS_DOC = ROOT / "docs" / "ENVS.md"
REGISTRY_DOC = ROOT / "docs" / "vllm-027-new-patches.md"

# Pinned upstream vllm-project/vllm blobs.
BASE_SHA256 = "8ea5fd82ab09db66872be1fbd5e830022bef97f75719a009fef5ed2a9f70fbb8"
HEAD_SHA256 = "cb5262282376c5c4d51e6cc423ff0fb5f4068ea406d8c0b61e2f700764909fb8"

# Names that would mean this compute-only backport had started touching memory
# planning. #50004 states the same invariant as a comment; here it is a gate.
BUDGET_NAMES = (
    "gpu_memory_utilization",
    "GPU_MEMORY_UTILIZATION",
    "max_num_batched_tokens",
    "num_gpu_blocks",
    "kv_cache",
)


def _load_hotfix():
    spec = importlib.util.spec_from_file_location("hotfix_qk_rmsnorm_split", HOTFIX)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _module_level_defs(src: str) -> list[str]:
    return [
        f"{type(n).__name__}:{getattr(n, 'name', '')}" for n in ast.parse(src).body
    ]


class _PatcherCase(unittest.TestCase):
    """Shared plumbing: a scratch copy of a fixture plus a captured run."""

    def setUp(self):
        self.hf = _load_hotfix()
        self.base = BASE.read_text(encoding="utf-8")
        self.head = HEAD.read_text(encoding="utf-8")
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write(self, text: str, name: str = "fused_qk_rmsnorm.py") -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.hf.main(["hotfix", *args])
        return rc, out.getvalue(), err.getvalue()


class UpstreamEquivalenceTest(_PatcherCase):
    def test_fixtures_are_byte_frozen(self):
        self.assertEqual(_sha256(self.base), BASE_SHA256)
        self.assertEqual(_sha256(self.head), HEAD_SHA256)

    def test_every_anchor_occurs_exactly_once_in_upstream_base(self):
        self.assertEqual(self.hf.anchor_counts(self.base), [1, 1, 1, 1, 1])

    def test_apply_reproduces_upstream_post_image_byte_for_byte(self):
        target = self.write(self.base)
        rc, out, _ = self.run_cli("--target", str(target))
        self.assertEqual(rc, 0, out)
        self.assertEqual(target.read_text(encoding="utf-8"), self.head)
        self.assertEqual(_sha256(target.read_text(encoding="utf-8")), HEAD_SHA256)

    def test_upstream_post_image_passes_the_self_check(self):
        self.assertEqual(self.hf.structure_errors(self.head), [])
        self.assertEqual(self.hf.classify(self.head)[0], "applied")

    def test_stock_pre_image_is_classified_stock_not_applied(self):
        state, errors, counts = self.hf.classify(self.base)
        self.assertEqual(state, "stock")
        self.assertEqual(counts, [1, 1, 1, 1, 1])
        self.assertTrue(errors)

    def test_patched_output_compiles(self):
        compile(self.hf.apply_hunks(self.base), "patched", "exec")

    def test_transform_adds_exactly_one_module_level_definition(self):
        before = _module_level_defs(self.base)
        after = _module_level_defs(self.head)
        self.assertEqual(
            [d for d in after if d not in before], ["FunctionDef:_rmsnorm_row"]
        )
        self.assertEqual([d for d in before if d not in after], [])

    def test_backport_touches_no_kv_budget_or_gpu_memory_knob(self):
        # A block-width change is pure compute. If any of these names ever
        # appears in the patcher or in either upstream image, the "no KV-budget
        # change" claim in docs/vllm-027-new-patches.md has stopped being true.
        for label, text in (
            ("patcher", HOTFIX.read_text(encoding="utf-8")),
            ("upstream base", self.base),
            ("upstream head", self.head),
        ):
            for name in BUDGET_NAMES:
                self.assertNotIn(name, text, f"{name} appeared in {label}")


class IdempotenceTest(_PatcherCase):
    def test_second_run_revalidates_and_leaves_bytes_stable(self):
        target = self.write(self.base)
        self.assertEqual(self.run_cli("--target", str(target))[0], 0)
        once = target.read_text(encoding="utf-8")
        rc, out, _ = self.run_cli("--target", str(target))
        self.assertEqual(rc, 0)
        self.assertIn("already applied and re-validated", out)
        self.assertEqual(target.read_text(encoding="utf-8"), once)

    def test_already_patched_target_is_never_double_patched(self):
        target = self.write(self.head)
        self.assertEqual(self.run_cli("--target", str(target))[0], 0)
        patched = target.read_text(encoding="utf-8")
        self.assertEqual(patched, self.head)
        self.assertEqual(patched.count("def _rmsnorm_row("), 1)

    def test_already_applied_but_broken_target_is_refused_not_repatched(self):
        # Re-validation, not a bare skip: an applied file that no longer
        # satisfies the self-check must fail closed instead of reporting OK.
        broken = self.head.replace("        Q_BLOCK,\n", "")
        target = self.write(broken)
        rc, _, err = self.run_cli("--target", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("refusing to patch", err)
        self.assertEqual(target.read_text(encoding="utf-8"), broken)


class DriftTest(_PatcherCase):
    def test_missing_anchor_is_refused_without_writing(self):
        drifted = self.base.replace(
            "    pid_task = tl.program_id(1)",
            "    pid_task = tl.program_id(1)  # vendor note",
            1,
        )
        self.assertEqual(self.hf.anchor_counts(drifted)[2], 0)
        target = self.write(drifted)
        rc, _, err = self.run_cli("--target", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("nothing was written", err)
        self.assertEqual(target.read_text(encoding="utf-8"), drifted)

    def test_duplicated_anchor_is_refused_without_writing(self):
        # replace(old, new, 1) would patch the first site and leave the second
        # one stock. Exact-occurrence counting is what makes that impossible.
        duplicated = self.base.replace(
            "        BLOCK_SIZE=block_size,",
            "        BLOCK_SIZE=block_size,\n        BLOCK_SIZE=block_size,",
            1,
        )
        self.assertEqual(self.hf.anchor_counts(duplicated)[4], 2)
        target = self.write(duplicated)
        rc, _, err = self.run_cli("--target", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("anchor5=2", err)
        self.assertEqual(target.read_text(encoding="utf-8"), duplicated)

    def test_comment_mentioning_the_tokens_does_not_spoof_applied(self):
        # A token-presence guard ("Q_BLOCK" in src and "_rmsnorm_row" in src)
        # certifies this stock file as already patched. The structural check
        # must patch it instead.
        spoof = self.base.replace(
            "import torch",
            "import torch\n# vendor note: see _rmsnorm_row / Q_BLOCK upstream",
            1,
        )
        target = self.write(spoof)
        self.assertEqual(self.hf.classify(spoof)[0], "stock")
        rc, out, _ = self.run_cli("--target", str(target))
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.hf.structure_errors(target.read_text("utf-8")), [])

    def test_unreadable_target_fails_closed(self):
        missing = self.tmp / "does-not-exist.py"
        self.assertEqual(self.run_cli("--target", str(missing))[0], 1)
        self.assertEqual(self.run_cli("--status", "--target", str(missing))[0], 2)
        self.assertEqual(self.run_cli("--dry-run", "--target", str(missing))[0], 2)


class SelfCheckRestoreTest(_PatcherCase):
    # All five anchors match, so the patch applies — but a second use of the
    # removed `block_size` survives the rewrite and dangles. The post-write
    # self-check must catch that and put the original bytes back.
    TRAP_FROM = "    return qr_out, kv_out"
    TRAP_TO = "    _ = block_size\n    return qr_out, kv_out"

    def test_failed_self_check_restores_original_bytes(self):
        trap = self.base.replace(self.TRAP_FROM, self.TRAP_TO, 1)
        self.assertEqual(self.hf.anchor_counts(trap), [1, 1, 1, 1, 1])
        target = self.write(trap)
        rc, _, err = self.run_cli("--target", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("original bytes restored", err)
        self.assertIn("block_size", err)
        self.assertEqual(target.read_text(encoding="utf-8"), trap)

    def test_dry_run_predicts_the_self_check_failure(self):
        trap = self.base.replace(self.TRAP_FROM, self.TRAP_TO, 1)
        target = self.write(trap)
        rc, _, err = self.run_cli("--dry-run", "--target", str(target))
        self.assertEqual(rc, 2, "anchor tally alone would have said 'would apply'")
        self.assertIn("fails the self-check", err)
        self.assertEqual(target.read_text(encoding="utf-8"), trap)


class StatusAndDryRunTest(_PatcherCase):
    def test_status_on_stock_reports_not_applied_and_never_writes(self):
        target = self.write(self.base)
        rc, out, _ = self.run_cli("--status", "--target", str(target))
        self.assertEqual(rc, 1)
        self.assertIn("not-applied", out)
        self.assertIn(BASE_SHA256, out)
        self.assertIn("anchor1=1 anchor2=1 anchor3=1 anchor4=1 anchor5=1", out)
        self.assertEqual(target.read_text(encoding="utf-8"), self.base)

    def test_status_on_patched_reports_applied_and_never_writes(self):
        target = self.write(self.head)
        rc, out, _ = self.run_cli("--status", "--target", str(target))
        self.assertEqual(rc, 0)
        self.assertIn("status   applied", out)
        self.assertIn(HEAD_SHA256, out)
        self.assertEqual(target.read_text(encoding="utf-8"), self.head)

    def test_status_on_drift_reports_drifted_and_never_writes(self):
        drifted = self.base.replace("    pid_task = tl.program_id(1)", "", 1)
        target = self.write(drifted)
        rc, _, err = self.run_cli("--status", "--target", str(target))
        self.assertEqual(rc, 2)
        self.assertIn("drifted", err)
        self.assertEqual(target.read_text(encoding="utf-8"), drifted)

    def test_status_reports_the_image_vllm_version_from_root(self):
        root = self.tmp / "vllm"
        (root / "models/deepseek_v4/common/ops").mkdir(parents=True)
        shutil.copyfile(BASE, root / self.hf.TARGET_REL)
        (root / "_version.py").write_text(
            "__version__ = version = '0.25.2.dev0+g752a3a504.d20260714'\n",
            encoding="utf-8",
        )
        rc, out, _ = self.run_cli("--status", "--root", str(root))
        self.assertEqual(rc, 1)
        self.assertIn("0.25.2.dev0+g752a3a504.d20260714", out)

    def test_status_reports_unknown_version_when_root_has_none(self):
        target = self.write(self.base)
        _, out, _ = self.run_cli("--status", "--target", str(target))
        self.assertIn("vllm     unknown", out)

    def test_dry_run_on_stock_says_would_apply_and_never_writes(self):
        target = self.write(self.base)
        rc, out, _ = self.run_cli("--dry-run", "--target", str(target))
        self.assertEqual(rc, 0)
        self.assertIn("would apply cleanly", out)
        self.assertEqual(target.read_text(encoding="utf-8"), self.base)

    def test_dry_run_on_patched_says_nothing_to_do_and_never_writes(self):
        target = self.write(self.head)
        rc, out, _ = self.run_cli("--dry-run", "--target", str(target))
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", out)
        self.assertEqual(target.read_text(encoding="utf-8"), self.head)

    def test_dry_run_on_drift_exits_two_and_never_writes(self):
        drifted = self.base.replace("    block_size = triton", "    blk = triton", 1)
        target = self.write(drifted)
        rc, _, err = self.run_cli("--dry-run", "--target", str(target))
        self.assertEqual(rc, 2)
        self.assertIn("would refuse to patch", err)
        self.assertEqual(target.read_text(encoding="utf-8"), drifted)


class WiringTest(unittest.TestCase):
    """Default-OFF, exactly-`1`, fail-closed, and present on BOTH ranks."""

    MOUNT = (
        "${DSPARK_QK_RMSNORM_SPLIT_HOTFIX:-./patches/hotfix-dsv4-qk-rmsnorm-split-blocks.py}"
        ":/opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py:ro"
    )
    ENV_PASSTHROUGH = (
        'DSPARK_ENABLE_QK_RMSNORM_SPLIT: "${DSPARK_ENABLE_QK_RMSNORM_SPLIT:-0}"'
    )
    GATED = (
        'if [ "$${DSPARK_ENABLE_QK_RMSNORM_SPLIT:-0}" = "1" ]; then '
        "python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py || exit 1; fi;"
    )

    def setUp(self):
        self.compose = COMPOSE.read_text(encoding="utf-8")
        self.env_example = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.start = START.read_text(encoding="utf-8")
        self.ci = CI_VALIDATE.read_text(encoding="utf-8")

    def test_patch_mounted_read_only(self):
        self.assertIn(self.MOUNT, self.compose)

    def test_env_passthrough_defaults_off(self):
        self.assertIn(self.ENV_PASSTHROUGH, self.compose)

    def test_entrypoint_invocation_gated_and_fail_closed(self):
        self.assertIn(self.GATED, self.compose)

    def test_no_ungated_invocation_anywhere_in_compose(self):
        for line in self.compose.splitlines():
            if "python3 /opt/hotfix-dsv4-qk-rmsnorm-split-blocks.py" in line:
                self.assertIn('DSPARK_ENABLE_QK_RMSNORM_SPLIT:-0}" = "1"', line)
                self.assertIn("|| exit 1", line)

    def test_not_added_to_the_fail_open_sh_allowlist(self):
        # The `:-0}" != "1"` .sh loop swallows failures with `|| true`; putting
        # this patcher there would silently discard its fail-closed exit.
        for line in self.compose.splitlines():
            if "hotfix-dsv4-qk-rmsnorm-split-blocks" in line:
                self.assertNotIn("|| true", line)

    def test_env_example_documents_default_off(self):
        self.assertIn("\nDSPARK_ENABLE_QK_RMSNORM_SPLIT=0\n", self.env_example)

    def test_start_script_syncs_the_patch_to_the_worker_rank(self):
        self.assertIn(
            'DSPARK_QK_RMSNORM_SPLIT_HOTFIX="${DSPARK_QK_RMSNORM_SPLIT_HOTFIX:-'
            '$SCRIPT_DIR/patches/hotfix-dsv4-qk-rmsnorm-split-blocks.py}"',
            self.start,
        )
        self.assertIn(
            'scp "$DSPARK_QK_RMSNORM_SPLIT_HOTFIX" "${WORKER_HOST}:'
            '${REMOTE_WORKER_DIR}/patches/hotfix-dsv4-qk-rmsnorm-split-blocks.py"',
            self.start,
        )

    def test_both_ranks_read_one_flag_value(self):
        # The worker's compose reads the head's .env.dspark, scp'd verbatim, so
        # a single DSPARK_ENABLE_QK_RMSNORM_SPLIT governs both containers and
        # cannot go asymmetric on a TP-replicated latent.
        self.assertIn(
            'scp "$ENV_FILE" "${WORKER_HOST}:${REMOTE_ENV_FILE}"', self.start
        )

    def test_ci_validate_requires_the_patch_file(self):
        required = self.ci.split("# Mounted hotfix files must exist.", 1)[1]
        self.assertIn(
            "  patches/hotfix-dsv4-qk-rmsnorm-split-blocks.py\n",
            required.split("\ndo\n", 1)[0] + "\n",
        )

    def test_ci_validate_asserts_the_gate_and_the_worker_sync(self):
        self.assertIn(self.ENV_PASSTHROUGH, self.ci)
        self.assertIn(self.GATED, self.ci)
        self.assertIn(
            "scripts/test-qk-rmsnorm-split-blocks.py", self.ci
        )

    def test_docs_document_the_flag_and_the_evidence_status(self):
        self.assertIn("DSPARK_ENABLE_QK_RMSNORM_SPLIT", ENVS_DOC.read_text("utf-8"))
        registry = " ".join(REGISTRY_DOC.read_text(encoding="utf-8").split())
        self.assertIn("hotfix-dsv4-qk-rmsnorm-split-blocks.py", registry)
        self.assertIn("No end-to-end measurement is claimed", registry)
        self.assertIn("KV_BLOCK` = 512", registry)

    def test_no_unattributable_speed_claim_survives(self):
        # The withdrawn figures must not creep back into the patcher docstring;
        # the CHANGELOG is the single place that records what was removed.
        patcher = HOTFIX.read_text(encoding="utf-8")
        for stale in ("+4.0%", "+12.2%", "49.7", "47.8", "52.0", "46.3", "Welch"):
            self.assertNotIn(stale, patcher)
        self.assertIn("no end-to-end measurement is claimed", patcher.lower())


if __name__ == "__main__":
    unittest.main()
