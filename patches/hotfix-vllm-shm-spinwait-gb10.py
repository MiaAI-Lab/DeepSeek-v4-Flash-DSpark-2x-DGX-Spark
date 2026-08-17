#!/usr/bin/env python3
"""GB10 shm-broadcast spin-wait hotfix — busy_loop_s 1s -> 2ms.

On GB10 (DGX Spark / GX10) with TP>=2 vLLM uses its multiprocessing executor,
whose shm ring-buffer ``SpinCondition`` busy-loops via ``sched_yield()`` for
``busy_loop_s`` seconds after each read before falling back to an efficient
sleep. During decode a new IPC message arrives every few ms — always inside the
default 1s window — so the sleep path is NEVER taken. The cost:

  * ~4 Grace P-cores pinned at full clock doing no work (~333% CPU) while the
    GPU sips ~38W at ~96% util (measured: cores 6/9/17/19 at ~90-100%, rest idle;
    VLLM::Worker ~300% + EngineCore ~100%);
  * the CPU cluster runs ~84C, roughly half of GB10's shared thermal budget
    (hard-shutdown at 104.8C) — a likely contributor to thermal blackouts.

Lowering ``busy_loop_s`` to 2ms lets the ring buffer sleep between decode
steps: CPU ~333% -> ~89%, SoC ~-11C, decode throughput unchanged (GPU stays
~96%). Single-unit TP=1 boxes use the in-process executor and are unaffected.

Analysis and measurements:
    https://nacyot.github.io/artifacts/vllm-spin-wait-gb10-en/

Idempotent. Refuses to patch (leaving stock behaviour) if the upstream default
has changed. Called from the compose entrypoint before ``exec vllm serve``.
"""
from __future__ import annotations

from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/"
    "device_communicators/shm_broadcast.py"
)

MARK = "[gb10-spinwait]"
STOCK = "        busy_loop_s: float = 1,\n"
PATCHED = (
    "        busy_loop_s: float = 0.002,"
    f"  # {MARK} was 1s: never slept during decode, pinned ~4 P-cores\n"
)


def main() -> None:
    if not P.exists():
        print(f"{MARK} {P} not found; skipping")
        return
    src = P.read_text()
    if MARK in src or "busy_loop_s: float = 0.002" in src:
        print(f"{MARK} already applied in {P}")
        return
    if STOCK not in src:
        print(
            f"{MARK} anchor 'busy_loop_s: float = 1' not found in {P}; "
            "refusing to patch (upstream default may have changed)"
        )
        return
    P.write_text(src.replace(STOCK, PATCHED, 1))
    print(f"{MARK} patched busy_loop_s 1s -> 2ms in {P}")


if __name__ == "__main__":
    main()
