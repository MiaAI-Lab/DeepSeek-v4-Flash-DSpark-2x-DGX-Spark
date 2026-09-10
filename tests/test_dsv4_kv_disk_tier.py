#!/usr/bin/env python3
"""CPU regression tests for the kv-disk-tier accounting / eviction logic.

These exercise the tier managers' pure-Python bookkeeping (byte counters, LRU
pinning, job completion) by constructing instances with ``__new__`` and stubbing
the parent methods, so no GPU, ZMQ socket, or ``/kvdisk`` is needed. They run
anywhere vLLM + zmq + msgspec are importable (the CI container or the serving
image); on a host without those they are skipped, never failed.

The cases pin the specific bugs fixed over the life of this module:
  * ``_forget`` never decremented ``_bytes`` (membership-only ``OrderedDict``).
  * a failed job evicted blocks still pinned by a concurrent job.
  * ``submit_load`` lacked the ``_ready`` guard that ``submit_store`` had.
  * ``has_pending_work`` ignored the already-completed ``_done`` queue.
  * ``CappedFileSystemTierManager.lookup`` used truthiness on a ``LookupResult``.
  * ``CappedFileSystemTierManager`` kept phantom entries after a failed store.
"""
from __future__ import annotations

import sys
import threading
import unittest
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv-disk-tier"))


def _missing_external_dependency(err: Exception) -> bool:
    """True only when a top-level third-party dep is absent on this host.

    A ``ModuleNotFoundError`` naming ``vllm`` / ``zmq`` / ``msgspec`` (or one of
    their submodules) means the environment genuinely lacks the dependency and
    the suite should skip. Anything else -- a syntax error, a ``NameError``, an
    ``ImportError: cannot import name ...`` from our own modules or a drifted
    vLLM API -- is a real bug and must fail the suite, not hide it.
    """
    name = getattr(err, "name", None)
    return (
        isinstance(err, ModuleNotFoundError)
        and bool(name)
        and name.split(".")[0] in ("vllm", "zmq", "msgspec")
    )


try:
    import dsv4_shard_tier  # noqa: F401
    import dsv4_kv_disk_tier  # noqa: F401
except ImportError as e:
    if _missing_external_dependency(e):
        # vllm/zmq/msgspec unavailable on this host: skip the whole suite.
        dsv4_shard_tier = None
        dsv4_kv_disk_tier = None
        _HAVE_TIER = False
    else:
        raise  # our own import bug: fail loudly
else:
    _HAVE_TIER = True

if _HAVE_TIER:
    # Our own symbols: an import failure here is a real bug, never a missing dep.
    from dsv4_kv_disk_tier import CappedFileSystemTierManager, _load_block_buffered

    try:
        from vllm.v1.kv_offload.tiering.base import LookupResult
        from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    except ImportError as e:
        if _missing_external_dependency(e):
            LookupResult = None
            FileSystemTierManager = None
        else:
            raise
else:
    CappedFileSystemTierManager = None
    _load_block_buffered = None
    LookupResult = None
    FileSystemTierManager = None

SLICE = 10


def make_tier(keys):
    t = dsv4_shard_tier.DistributedShardTier.__new__(
        dsv4_shard_tier.DistributedShardTier
    )
    t._slice_bytes = SLICE
    t._present = OrderedDict()
    t._bytes = 0
    for k in keys:
        t._present[k] = None
        t._bytes += SLICE
    t._max_bytes = 10_000
    t._pinned = {}
    t._jobs = {}
    t._done = []
    t._broadcast = lambda payload: True
    return t


def _acc(keys):
    return type("Acc", (), {"keys": keys, "is_promotion": False})()


def _jm(job_id, keys):
    return type(
        "JM", (), {"job_id": job_id, "keys": keys, "block_ids": [1] * len(keys)}
    )()


@unittest.skipIf(dsv4_shard_tier is None, "vllm/zmq not importable")
class TestShardTierAccounting(unittest.TestCase):
    def test_forget_decrements_bytes(self):
        t = make_tier([b"a", b"b", b"c"])
        t._forget([b"a", b"nope"])
        self.assertEqual(t._bytes, 2 * SLICE)
        self.assertNotIn(b"a", t._present)
        self.assertIn(b"b", t._present)
        self.assertIn(b"c", t._present)

    def test_finish_failure_keeps_pinned(self):
        t = make_tier([b"a", b"b"])
        t._pinned[b"b"] = 2  # this job + one concurrent job
        t._finish(1, _acc([b"a", b"b"]), False)
        self.assertNotIn(b"a", t._present)  # unpinned -> evicted
        self.assertIn(b"b", t._present)     # still pinned -> survives
        self.assertEqual(t._pinned.get(b"b"), 1)

    def test_submit_load_ready_guard(self):
        t = make_tier([b"a"])
        t._ready = False
        t.submit_load(_jm(7, [b"a"]))
        self.assertEqual(len(t._done), 1)
        self.assertFalse(t._done[0].success)

    def test_has_pending_work_includes_done(self):
        t = make_tier([])
        t._jobs = {}
        t._done = []
        self.assertFalse(t.has_pending_work())
        t._done = [object()]
        self.assertTrue(t.has_pending_work())
        t._jobs = {1: object()}
        t._done = []
        self.assertTrue(t.has_pending_work())


def make_head():
    t = dsv4_shard_tier.DistributedShardTier.__new__(
        dsv4_shard_tier.DistributedShardTier
    )
    t._num_agents = 2
    t._identities = {}
    t._hellos = {}
    t._ready = False
    t._epoch = 0
    t._slice_bytes = SLICE
    t._present = OrderedDict()
    t._bytes = 0
    t._pinned = {}
    t._jobs = {}
    t._done = []
    t._n_timeouts = 0
    t._sent = []
    t._broadcast = lambda payload: (t._sent.append(payload), True)[1]
    return t


def _jacc(need, keys, deadline, abandoned=False):
    return type(
        "Acc",
        (),
        {
            "need": need,
            "acks": 0,
            "ok": True,
            "deadline": deadline,
            "keys": keys,
            "is_promotion": False,
            "abandoned": abandoned,
        },
    )()


@unittest.skipIf(dsv4_shard_tier is None, "vllm/zmq not importable")
class TestRecoveryTransitions(unittest.TestCase):
    def test_timeout_holds_then_fails(self):
        t = make_head()
        t._drain = lambda: None
        t._job_timeout_s = 10.0
        t._pinned = {b"a": 1}
        t._present = OrderedDict([(b"a", None)])
        t._bytes = SLICE
        acc = _jacc(2, [b"a"], deadline=0.0)
        t._jobs = {1: acc}

        # First call: past deadline, not yet abandoned -> hold, no result.
        out = list(t.get_finished())
        self.assertEqual(out, [])
        self.assertTrue(acc.abandoned)
        self.assertIn(1, t._jobs)  # still held so peer I/O can drain

        # Simulate the grace window expiring with the agents still silent.
        acc.deadline = 0.0
        # Second call: still past deadline -> force-fail and release.
        out = list(t.get_finished())
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].success)
        self.assertEqual(t._n_timeouts, 1)
        self.assertNotIn(1, t._jobs)

    def test_reconcile_drops_partial_shards(self):
        t = make_head()
        t._on_hello(b"id0", {"rank": 0, "keys": [b"shared", b"only0"]})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"shared", b"only1"]})
        self.assertTrue(t._ready)
        self.assertIn(b"shared", t._present)
        self.assertNotIn(b"only0", t._present)
        self.assertNotIn(b"only1", t._present)
        evicts = [p for p in t._sent if p.get("t") == "evict"]
        self.assertTrue(evicts)

    def test_reconnect_drops_stale_hellos(self):
        t = make_head()
        # Round 0: both agents register, reconcile -> READY.
        t._on_hello(b"id0", {"rank": 0, "keys": [b"a"]})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"]})
        self.assertTrue(t._ready)
        self.assertEqual(t._epoch, 0)

        # Agent 0 reconnects with a changed inventory.
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"]})
        self.assertFalse(t._ready)
        self.assertEqual(t._epoch, 1)
        self.assertEqual(t._hellos, {})  # triggering hello dropped
        self.assertTrue(
            any(p.get("t") == "rehello" and p.get("gen") == 1 for p in t._sent)
        )

        # A late hello from the OLD round (gen 0) must not start a new round.
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"], "gen": 0})
        self.assertNotIn(1, t._hellos)
        self.assertFalse(t._ready)

        # Fresh round-1 hellos reconcile again.
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"], "gen": 1})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"b"], "gen": 1})
        self.assertTrue(t._ready)
        self.assertIn(b"b", t._present)
        self.assertNotIn(b"a", t._present)


@unittest.skipIf(
    CappedFileSystemTierManager is None, "vllm/zmq not importable"
)
class TestFsCapped(unittest.TestCase):
    def _make(self):
        cm = CappedFileSystemTierManager.__new__(CappedFileSystemTierManager)
        cm._lock = threading.RLock()
        cm._block_size = 100
        cm._lru = OrderedDict()
        cm._total_bytes = 0
        cm._pinned = {}
        cm._load_job_keys = {}
        cm._store_job_keys = {}
        cm.file_mapper = type(
            "FM", (), {"get_file_name": lambda self, k: f"/x/{k}"}
        )()
        return cm

    def test_lookup_marks_recent_only_on_hit(self):
        cm = self._make()
        cm._lru["/x/k"] = 100
        recent = []
        cm._mark_recent = lambda paths: recent.append(list(paths))
        orig = FileSystemTierManager.lookup
        try:
            FileSystemTierManager.lookup = lambda self, key, req: LookupResult.MISS
            self.assertIs(cm.lookup("k", None), LookupResult.MISS)
            self.assertEqual(recent, [])

            FileSystemTierManager.lookup = lambda self, key, req: LookupResult.HIT
            self.assertIs(cm.lookup("k", None), LookupResult.HIT)
            self.assertEqual(recent, [["/x/k"]])
        finally:
            FileSystemTierManager.lookup = orig

    def test_failed_store_reconciles(self):
        cm = self._make()
        cm._store_job_keys = {1: ["/a", "/b"]}
        cm._lru = OrderedDict([("/a", 100), ("/b", 100), ("/c", 100)])
        cm._total_bytes = 300
        Res = type("Res", (), {"job_id": 1, "success": False})
        orig = FileSystemTierManager.get_finished_jobs
        try:
            FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
            results = list(cm.get_finished_jobs())
        finally:
            FileSystemTierManager.get_finished_jobs = orig
        self.assertEqual(len(results), 1)
        self.assertNotIn("/a", cm._lru)
        self.assertNotIn("/b", cm._lru)
        self.assertIn("/c", cm._lru)
        self.assertEqual(cm._total_bytes, 100)


@unittest.skipIf(_load_block_buffered is None, "vllm/zmq not importable")
class TestLoadBlockBuffered(unittest.TestCase):
    def test_bounds_check_rejects_out_of_range(self):
        view = memoryview(bytearray(100))
        with self.assertRaises(ValueError):
            _load_block_buffered("/nonexistent", view, 90, 20)  # 90+20 > 100


if __name__ == "__main__":
    unittest.main()
