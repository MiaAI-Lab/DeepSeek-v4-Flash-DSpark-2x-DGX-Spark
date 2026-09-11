#!/usr/bin/env python3
"""CPU regressions for the NFS export hardening.

The DSpark NFS exporter shares the head's HuggingFace cache root (which also
holds the 0600 `huggingface-cli` token file) on a privileged, host-networked
container. Three defaults weakened that:

1. `no_root_squash` mapped client-side root to server root, so the token file
   was readable regardless of its mode. Default is now root_squash: remote
   root maps to nobody — world-readable blobs stay readable, the token does not.
2. `insecure` allowed mounts from unprivileged client ports; dropped (the
   worker's kernel NFS client in the docker volume mounts privileged).
3. `NFS_CLIENTS:-*` (entrypoint) and `${worker_ip:-*}` (nfs_clients fallback)
   wrote a WORLD export when configuration or CIDR detection failed. Both now
   fail closed with a clear message.

Tests run the shipped guard block and the shipped nfs_clients() against stub
binaries; they do not need nfsd.
"""
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = (ROOT / "files" / "nfs-server" / "entrypoint.sh").read_text()
SHARE = (ROOT / "files" / "nfs-share.sh").read_text()

# The guard block: variable resolution + empty-client refusal + opts default,
# stopping before the first nfsd filesystem touch.
GUARD_BLOCK = ENTRYPOINT[ENTRYPOINT.index("EXPORT_DIR="):ENTRYPOINT.index("\nmkdir ")]

_fn_start = SHARE.index("nfs_clients() {")
CLIENTS_FN = SHARE[_fn_start:SHARE.index("\n}", _fn_start) + 2]

FAKE_IP = """#!/usr/bin/env bash
if [ -n "${IP_CIDR:-}" ]; then
  echo "3: dev inet ${IP_CIDR} brd 10.0.22.255 scope global dev"
fi
exit 0
"""


def run_guard(env_over: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("NFS_CLIENTS", None)
    env.pop("NFS_OPTS", None)
    env.update(env_over)
    return subprocess.run(["bash", "-c", GUARD_BLOCK], capture_output=True,
                          text=True, env=env)


def run_clients(cidr: str, worker_ip: str) -> subprocess.CompletedProcess:
    workdir = Path(tempfile.mkdtemp())
    ip = workdir / "ip"
    ip.write_text(FAKE_IP)
    ip.chmod(0o755)
    script = f"""set -euo pipefail
PATH={shlex.quote(str(workdir))}:/usr/bin:/bin
export IP_CIDR={shlex.quote(cidr)}
IFACE=enp1s0f1np1
WORKER_IP={shlex.quote(worker_ip)}
WORKER_HOST=worker.example
nfs_err() {{ echo "$*" >&2; exit 1; }}
{CLIENTS_FN}
nfs_clients
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    shutil.rmtree(workdir)
    return r


class EntrypointGuard(unittest.TestCase):
    def test_empty_clients_refused(self):
        r = run_guard({})
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to export to '*'", r.stderr)
        self.assertIn("NFS_CLIENTS", r.stderr)

    def test_explicit_clients_pass(self):
        r = run_guard({"NFS_CLIENTS": "10.0.22.2,10.0.22.0/24"})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_default_opts_are_squashed_and_secure(self):
        for source, name in ((ENTRYPOINT, "entrypoint.sh"), (SHARE, "nfs-share.sh")):
            with self.subTest(file=name):
                for line in source.splitlines():
                    if "NFS_OPTS:-" in line:
                        self.assertIn("root_squash", line)
                        self.assertNotIn("no_root_squash", line)
                        self.assertNotIn("insecure", line)

    def test_no_wildcard_default_remains(self):
        for source, name in ((ENTRYPOINT, "entrypoint.sh"), (SHARE, "nfs-share.sh")):
            code = "\n".join(ln for ln in source.splitlines()
                             if not ln.strip().startswith("#"))
            with self.subTest(file=name):
                self.assertNotIn('NFS_CLIENTS:-*}', code)
                self.assertNotIn('${worker_ip:-*}', code)


class NfsClients(unittest.TestCase):
    def test_cidr_and_worker_ip(self):
        r = run_clients("10.0.22.1/24", "10.0.22.2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.2,10.0.22.0/24")

    def test_cidr_only(self):
        r = run_clients("10.0.22.1/24", "")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.0/24")

    def test_worker_ip_only(self):
        r = run_clients("", "10.0.22.2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "10.0.22.2")

    def test_nothing_detected_fails_closed(self):
        r = run_clients("", "")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Refusing to export to '*'", r.stderr)
        self.assertNotIn("*", r.stdout)


if __name__ == "__main__":
    unittest.main()
