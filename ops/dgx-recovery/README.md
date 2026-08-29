# DGX Spark recovery controls

The MT7925 driver came from the stock `linux-modules-*-nvidia` package. Kernel
6.17 is affected by CVE-2026-68307, whose reset-path crash matches these nodes.
Keep Wi-Fi disabled until NVIDIA ships a DGX-tested kernel containing the fix.

Install idempotently after configuring the wired default route:

```bash
sudo ./ops/dgx-recovery/install.sh head    # head Spark
sudo ./ops/dgx-recovery/install.sh worker  # worker Spark
sudo reboot
./ops/dgx-recovery/verify.sh head --wait 900
```

The installer enables Docker at boot, disables the failing Wi-Fi driver, sets
kernel-oops recovery and the system watchdog, raises NCCL memlock limits, and
installs the worker orphan guard. Compose uses an init process and a bounded
30-second shutdown grace period.

The worker guard defaults match the reference two-Spark deployment. Override
them for another pair in `/etc/default/mia-worker-orphan-guard`:

```bash
MIA_CONTAINER=deepseek-v4-flash-vllm-dspark-1
MIA_API_HOST=100.125.193.22
MIA_API_PORT=8888
MIA_MASTER_HOST=192.168.200.12
MIA_MASTER_PORT=25000
```

The Wi-Fi driver is disabled because its reset work oopsed twice in
`mt7925_mac_reset_work`, wedging RTNL and every network-dependent service.
A wired interface must be the default route or installation fails closed.

`--wait` is the boot-readiness hook: it checks the recovery contract and waits
up to 15 minutes for the API without starting another model load. `verify.sh`
accepts `MIA_CONTAINER`, `MIA_API_HOST`, and `MIA_API_PORT` overrides too.

Run the installed two-node NCCL proof from the head after adapting hosts and
interfaces to the target pair:

```bash
mpirun --allow-run-as-root -np 2 -H 192.168.200.12:1,192.168.200.13:1 \
  -x NCCL_SOCKET_IFNAME=enP7s7 -x NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 \
  ~/.local/src/nccl-tests/build/all_reduce_perf -b 8M -e 1G -f 2 -g 1
```
