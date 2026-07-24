"""Revision-pinned per-rank progress hook for the custom DSpark vLLM image.

Python imports ``sitecustomize`` during interpreter startup because the monitor
root is on ``PYTHONPATH``. Any version, module, signature, revision, capability,
or output-path mismatch leaves the observer disabled. Inference is never made
dependent on observer publication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import functools
import inspect
import json
import os
from pathlib import Path
import stat
import threading


EXPECTED_VLLM_VERSION = "0.25.2.dev0+g752a3a504.d20260714"
EXPECTED_MODULE_SUFFIX = "/vllm/v1/worker/gpu_worker.py"
EXPECTED_OBSERVER_REVISION = "dspark-rank-observer-v1"
MAX_JSON_BYTES = 16_384


def _atomic_write(path, payload):
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("rank progress payload too large")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _iso_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _process_generation():
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    start_ticks = Path("/proc/self/stat").read_text().split()[21]
    return f"{boot_id}:{os.getpid()}:{start_ticks}"


def install_rank_observer(
    *,
    vllm_version,
    worker_class,
    source_rank,
    observer_revision,
    capability,
    writer,
    now,
    process_generation,
):
    if (
        vllm_version != EXPECTED_VLLM_VERSION
        or observer_revision != EXPECTED_OBSERVER_REVISION
        or source_rank not in (0, 1)
        or not isinstance(capability, str)
        or len(capability) < 32
    ):
        return False
    original = getattr(worker_class, "execute_model", None)
    if original is None:
        return False
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != ("self", "scheduler_output"):
        return False

    lock = threading.Lock()
    counters = {
        "sourceSequence": 0,
        "iterationTokens": 0,
        "completedRequests": 0,
        "requestAttributedKvActivity": 0,
    }

    @functools.wraps(original)
    def observed_execute_model(self, scheduler_output):
        result = original(self, scheduler_output)
        try:
            scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            per_request = scheduler_output.num_scheduled_tokens
            finished = scheduler_output.finished_req_ids
            if (
                not isinstance(scheduled_tokens, int)
                or isinstance(scheduled_tokens, bool)
                or scheduled_tokens < 0
                or not hasattr(per_request, "__len__")
                or not hasattr(finished, "__len__")
            ):
                return result
            with lock:
                counters["sourceSequence"] += 1
                counters["iterationTokens"] += scheduled_tokens
                counters["completedRequests"] += len(finished)
                counters["requestAttributedKvActivity"] += scheduled_tokens
                payload = {
                    "contractVersion": 1,
                    "scope": "rank_worker",
                    "observerRevision": observer_revision,
                    "sourceRank": source_rank,
                    "processGeneration": process_generation,
                    "sourceSequence": counters["sourceSequence"],
                    "observedAt": now(),
                    "lifecycle": "serving",
                    "metrics": {
                        "runningRequests": len(per_request),
                        "waitingRequests": 0,
                        "iterationTokens": counters["iterationTokens"],
                        "generatedTokens": 0,
                        "completedRequests": counters["completedRequests"],
                        "requestAttributedKvActivity": counters[
                            "requestAttributedKvActivity"
                        ],
                    },
                }
                try:
                    writer(payload)
                except Exception:
                    pass
        except Exception:
            pass
        return result

    worker_class.execute_model = observed_execute_model
    return True


def _install_from_runtime():
    if os.environ.get("MONITOR_OBSERVER_ENABLED") != "1":
        return False
    capability_path = Path(
        os.environ.get(
            "MONITOR_OBSERVER_CAPABILITY_FILE", "/run/gx10-monitor/capability"
        )
    )
    try:
        capability_stat = capability_path.stat()
        if stat.S_IMODE(capability_stat.st_mode) & 0o077:
            return False
        capability = capability_path.read_text(encoding="utf-8").strip()
        import vllm
        from vllm.v1.worker import gpu_worker
    except (ImportError, OSError):
        return False
    module_path = str(Path(gpu_worker.__file__).resolve())
    if not module_path.endswith(EXPECTED_MODULE_SUFFIX):
        return False
    try:
        source_rank = int(os.environ["NODE_RANK"])
        process_generation = _process_generation()
    except (KeyError, OSError, ValueError):
        return False
    output = Path(
        os.environ.get(
            "MONITOR_RANK_PROGRESS_PATH",
            "/run/model-serving-monitor/rank-progress.json",
        )
    )
    return install_rank_observer(
        vllm_version=vllm.__version__,
        worker_class=gpu_worker.Worker,
        source_rank=source_rank,
        observer_revision=os.environ.get("MONITOR_OBSERVER_REVISION", ""),
        capability=capability,
        writer=lambda payload: _atomic_write(output, payload),
        now=_iso_now,
        process_generation=process_generation,
    )


try:
    _install_from_runtime()
except Exception:
    # A telemetry hook must never prevent the serving interpreter from starting.
    pass
