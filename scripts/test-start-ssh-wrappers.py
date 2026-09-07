#!/usr/bin/env python3
"""CPU regressions for fail-fast ssh/scp in the start launcher.

Only the two preflight probes carried `-o BatchMode=yes -o ConnectTimeout=10`;
the ~100 ssh/scp invocations after preflight had neither, so a worker that
dropped or stalled mid-start left the launcher hanging on TCP retransmits —
indistinguishable from a slow start, at the exact moment restart:unless-stopped
makes the cluster state ambiguous. All invocations now go through dssh()/dscp()
wrappers defined next to the launcher's other globals.

These tests pin: both wrappers carry the options, every call site uses them
(no bare ssh/scp invocation survives), the wrappers do not recurse (the
mechanical rewrite once produced `dssh() { dssh "$@"; }`), and a dssh/dscp
call against fake ssh/scp binaries actually passes the options first.
"""
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "start-deepseek-v4-flash-dspark.sh"
SOURCE = LAUNCHER.read_text()


def code_lines():
    return [ln for ln in SOURCE.splitlines() if not ln.strip().startswith("#")]


class Wrappers(unittest.TestCase):
    def test_wrappers_carry_options(self):
        self.assertIn('dssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }', SOURCE)
        self.assertIn('dscp() { scp -o BatchMode=yes -o ConnectTimeout=10 "$@"; }', SOURCE)

    def test_wrappers_do_not_recurse(self):
        self.assertNotIn('dssh() { dssh', SOURCE)
        self.assertNotIn('dscp() { dscp', SOURCE)

    def test_no_bare_invocations_remain(self):
        bare = []
        for ln in code_lines():
            if re.search(r'(?<![\w])ssh "', ln) or re.search(r'(?<![\w])scp "', ln):
                if "dssh() {" not in ln and "dscp() {" not in ln:
                    bare.append(ln.strip())
        self.assertEqual(bare, [])

    def test_preflight_probes_use_wrappers(self):
        self.assertIn('dssh "$WORKER_HOST" "true"', SOURCE)
        self.assertIn('dssh "$WORKER2_HOST" "true"', SOURCE)
        # No duplicated inline options next to the wrapper calls.
        for ln in code_lines():
            if "dssh " in ln or "dscp " in ln:
                self.assertNotIn("ConnectTimeout", ln, ln)

    def test_echo_messages_untouched(self):
        # The "ssh exit $worker_rc" diagnostic text must survive verbatim.
        self.assertIn("(ssh exit $worker_rc)", SOURCE)


class WrapperBehavior(unittest.TestCase):
    def run_wrapper(self, wrapper: str) -> str:
        workdir = Path(tempfile.mkdtemp())
        log = workdir / "argv.log"
        for tool in ("ssh", "scp"):
            f = workdir / tool
            f.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" >> {shlex.quote(str(log))}\n')
            f.chmod(0o755)
        wrapper_def = next(ln for ln in SOURCE.splitlines() if ln.startswith(f"{wrapper}() {{"))
        script = f"""set -euo pipefail
PATH={shlex.quote(str(workdir))}:/usr/bin:/bin
{wrapper_def}
{wrapper} worker.example "remote command"
"""
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        lines = log.read_text().splitlines() if log.exists() else []
        shutil.rmtree(workdir)
        assert r.returncode == 0, r.stderr
        return lines

    def test_dssh_passes_options_first(self):
        self.assertEqual(self.run_wrapper("dssh"),
                         ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                          "worker.example", "remote command"])

    def test_dscp_passes_options_first(self):
        self.assertEqual(self.run_wrapper("dscp"),
                         ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                          "worker.example", "remote command"])


if __name__ == "__main__":
    unittest.main()
