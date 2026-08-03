from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
NETWORK = ROOT / "deployments/private-smoke/network"
SCRIPTS = ROOT / "deployments/private-smoke/scripts"


class NetworkTemplatesTest(unittest.TestCase):
    def test_netplan_templates_are_static_and_route_free(self):
        expected = {
            "head-cx7.yaml": "10.77.77.1/30",
            "worker-cx7.yaml": "10.77.77.2/30",
        }
        for name, address in expected.items():
            with self.subTest(name=name):
                text = (NETWORK / name).read_text()
                self.assertIn("enp1s0f0np0:", text)
                self.assertIn(address, text)
                self.assertIn("mtu: 9000", text)
                self.assertIn("dhcp4: false", text)
                self.assertIn("dhcp6: false", text)
                self.assertIn("accept-ra: false", text)
                self.assertIn("link-local: []", text)
                self.assertIn("optional: true", text)
                self.assertIn("renderer: NetworkManager", text)
                self.assertIn('connection.autoconnect: "true"', text)
                for forbidden in ("gateway4:", "gateway6:", "nameservers:", "routes:", "never-default"):
                    self.assertNotIn(forbidden, text)

    def test_apply_defaults_to_check_and_has_guarded_apply(self):
        script = (SCRIPTS / "apply-cx7-network.sh").read_text()
        self.assertIn('MODE="check"', script)
        self.assertIn("--apply", script)
        self.assertIn("netplan generate", script)
        self.assertIn('netplan get --root-dir "$check_root"', script)
        self.assertIn("Merged CX-7 configuration unexpectedly adds", script)
        self.assertIn('nmcli connection up "$CONNECTION" ifname "$IFACE"', script)
        self.assertIn("nmcli device disconnect", script)
        self.assertIn("connection.autoconnect", script)
        self.assertNotIn("netplan try", script)
        self.assertNotIn("netplan apply", script)
        self.assertIn("systemd-run", script)
        self.assertIn("/etc/netplan", script)
        self.assertIn("ip route", script)
        self.assertIn("default_before", script)
        self.assertIn("tailscale ping", script)
        self.assertIn("dgx-spark-1.tailc62fd7.ts.net", script)
        self.assertIn("dgx-spark-2.tailc62fd7.ts.net", script)
        self.assertIn("enp1s0f0np0", script)
        self.assertNotIn("rm -rf", script)
        result = subprocess.run(
            ["bash", str(SCRIPTS / "apply-cx7-network.sh"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_verify_requires_rdma_and_collective_proofs(self):
        script = (SCRIPTS / "verify-fabric.sh").read_text()
        for required in (
            "ping -M do",
            "ib_write_bw",
            "show_gids",
            "rocep1s0f0",
            "all_reduce_perf",
            "NCCL_TESTS_COMMIT",
            "--require-persistent",
        ):
            self.assertIn(required, script)

    def test_nccl_image_is_immutable(self):
        dockerfile = (NETWORK / "Dockerfile.nccl-tests").read_text()
        self.assertIn(
            "ghcr.io/anemll/dspark-vllm-gx10@sha256:"
            "a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8",
            dockerfile,
        )
        self.assertIn("da0b547b1b9c6e3b1d4c15578087874522ae3761", dockerfile)
        self.assertIn("MPI=1", dockerfile)

    def test_head_ssh_config_pins_the_private_worker_identity(self):
        text = (NETWORK / "head-ssh-config").read_text()
        for required in (
            "Host 10.77.77.2", "User plexiz", "id_ed25519_dgx_cluster",
            "IdentitiesOnly yes", "BatchMode yes", "StrictHostKeyChecking yes",
        ):
            self.assertIn(required, text)

    def test_nccl_launcher_stages_remote_wrapper_and_cleans_up(self):
        script = (NETWORK / "run-nccl-tests.sh").read_text()
        self.assertIn("scp -q", script)
        self.assertIn('remote_wrapper="/tmp/dspark-nccl-wrapper-', script)
        self.assertIn("trap cleanup EXIT", script)
        self.assertIn("--volume /tmp:/tmp", script)
        self.assertIn("--volume /run:/run", script)
        self.assertNotIn('  "$wrapper" "$@"', script)


if __name__ == "__main__":
    unittest.main()
