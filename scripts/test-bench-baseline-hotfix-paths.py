#!/usr/bin/env python3
"""CPU regression: bench-baseline scripts may only exec hotfixes that exist.

bench-baseline-issue22-only.sh used to run
    docker exec ... bash /tmp/hotfix-nvfp4-ds-mla-issue22.sh
Nothing is mounted at /tmp/hotfix-* in the container — compose mounts the repo's
patches directory read-only at /opt/dspark-patches (docker-compose.dspark.yml),
so the step could never succeed and `set -euo pipefail` aborted the baseline
mid-run. These tests pin every in-container hotfix path the bench-baseline
scripts exec to a file that actually exists in patches/.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = (ROOT / "docker-compose.dspark.yml").read_text()
SCRIPTS = [
    ROOT / "scripts" / "bench-baseline-issue22-only.sh",
    ROOT / "scripts" / "bench-baseline-no-patches.sh",
]

EXEC_RE = re.compile(r"docker exec\s+\S+\s+bash\s+(/\S+)")


class HotfixPaths(unittest.TestCase):
    def test_no_tmp_hotfix_references(self):
        for script in SCRIPTS:
            with self.subTest(script=script.name):
                self.assertNotIn("/tmp/hotfix-", script.read_text())

    def test_execd_hotfixes_exist_in_patches_dir(self):
        found = 0
        for script in SCRIPTS:
            for path in EXEC_RE.findall(script.read_text()):
                found += 1
                with self.subTest(script=script.name, path=path):
                    self.assertTrue(
                        path.startswith("/opt/dspark-patches/"),
                        f"{path} is not the compose-mounted patches dir",
                    )
                    patch = ROOT / "patches" / Path(path).name
                    self.assertTrue(patch.is_file(), f"{patch} missing")
        self.assertGreaterEqual(found, 1, "no docker-exec hotfix found to check")

    def test_compose_mount_point_unchanged(self):
        # The path contract this PR relies on: patches dir -> /opt/dspark-patches.
        self.assertIn(":/opt/dspark-patches:ro", COMPOSE)

    def test_issue22_hotfix_declares_idempotency(self):
        # Step 3 re-runs the hotfix after a boot that already applied it; the
        # hotfix header must keep promising "Safe to re-run (idempotent …)".
        header = (ROOT / "patches" / "hotfix-nvfp4-ds-mla-issue22.sh").read_text()
        self.assertRegex(header, r"Safe to re-run \(idempotent")


if __name__ == "__main__":
    unittest.main()
