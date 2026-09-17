#!/usr/bin/env python3
"""CPU regression tests for the kv-disk-tier accounting / eviction logic.

These exercise the tier managers' pure-Python bookkeeping (byte counters, LRU
pinning, job completion) by constructing instances with ``__new__`` and stubbing
the parent methods, so no GPU, ZMQ socket, or ``/kvdisk`` is needed. They run
anywhere vLLM + zmq + msgspec are importable (the CI container or the serving
image); a host that lacks one of those top-level packages skips the suite, while
a missing vLLM submodule, a drifted symbol or a broken import fails it -- see
``_missing_external_dependency``.

The cases pin the specific bugs fixed over the life of this module:
  * ``_forget`` never decremented ``_bytes`` (membership-only ``OrderedDict``).
  * a failed job evicted blocks still pinned by a concurrent job.
  * ``submit_load`` lacked the ``_ready`` guard that ``submit_store`` had.
  * ``has_pending_work`` ignored the already-completed ``_done`` queue.
  * ``CappedFileSystemTierManager.lookup`` used truthiness on a ``LookupResult``.
  * ``CappedFileSystemTierManager`` kept phantom entries after a failed store.
  * load completion must release pins so the fs_capped byte cap stays enforceable.
  * a failed load left its unreadable block file accounted (and on disk).
  * a failed-store reconcile un-accounted a path that was present on disk.
  * an out-of-range block id clamped a memoryview instead of failing the job.
  * a mid-round agent (re)connect was dropped, stalling the tier below READY.
  * a gen-less spontaneous hello and a stale ``gen=0`` reply were conflated, so
    a late round-0 reply discarded the reconciled index after READY.
  * an agent-side job failure sent no ack, holding the head's job to timeout.
  * direct-I/O mappings survived their owning job (stale slot -> GPU block ids)
    and used 0 both as a real GPU block id and as the null-sub-block sentinel.
  * slot invalidation took the truth of a real numpy ``block_ids``, which raised
    for a multi-slot spec and skipped the single slot 0.
  * a failed load whose unreadable file could not be retired stayed eligible for
    lookup (parent is a bare os.path.exists) and was re-promoted forever; it is
    now forgotten and quarantined until a fresh store rewrites it.
  * a job completed when enough ACK frames arrived, letting a single/duplicate
    sender satisfy the all-nodes barrier; completion now requires ``need``
    distinct sender identities.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kv-disk-tier"))


_OPTIONAL_DEPENDENCIES = ("vllm", "zmq", "msgspec")


def _missing_external_dependency(err: Exception) -> bool:
    """True only when the dependency ITSELF is absent on this host.

    ``ModuleNotFoundError.name`` is the first name in the requested import chain
    that could not be found, so only an exact top-level match -- ``vllm``, never
    ``vllm.v1.kv_offload...`` -- means the environment genuinely lacks the
    dependency and the suite may skip. A dotted name means the package IS
    installed and the submodule or API the tier needs has drifted, which is a
    real failure and must fail the suite. Same for a syntax error, a
    ``NameError``, or an ``ImportError: cannot import name ...`` raised inside
    our own modules or inside vLLM.
    """
    name = getattr(err, "name", None)
    return isinstance(err, ModuleNotFoundError) and name in _OPTIONAL_DEPENDENCIES


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
    import dsv4_vllm_patches
    from dsv4_kv_disk_tier import CappedFileSystemTierManager, _load_block_buffered

    try:
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
        from vllm.v1.kv_offload.tiering.base import LookupResult
        from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
    except ImportError as e:
        if _missing_external_dependency(e):
            GPULoadStoreSpec = None
            CPULoadStoreSpec = None
            LookupResult = None
            FileSystemTierManager = None
        else:
            raise
else:
    CappedFileSystemTierManager = None
    _load_block_buffered = None
    GPULoadStoreSpec = None
    CPULoadStoreSpec = None
    LookupResult = None
    FileSystemTierManager = None
    dsv4_vllm_patches = None

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
    t._send_rehello = lambda ident, gen: t._sent.append(
        {"t": "rehello", "gen": gen, "to": ident}
    )
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
            "acked": set(),
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

    def test_completion_requires_distinct_senders(self):
        # Completion must mean the required distinct participants completed, not
        # that enough ACK frames arrived. A single (or duplicate) identity
        # re-acking must not satisfy the all-nodes barrier and release the
        # primary-tier slots before every peer finished its I/O.
        t = make_head()
        t._pinned = {b"a": 1}
        t._present = OrderedDict([(b"a", None)])
        t._bytes = SLICE
        acc = _jacc(2, [b"a"], deadline=1e9)
        t._jobs = {1: acc}
        # Two ACKs from the same identity: only the first counts.
        t._on_ack(1, True, b"id0")
        t._on_ack(1, True, b"id0")
        self.assertEqual(acc.acks, 1)
        self.assertIn(1, t._jobs)  # not finished on a duplicate sender
        # The lone distinct identity alone cannot satisfy need=2 either.
        t._on_ack(1, True, b"id1")
        self.assertNotIn(1, t._jobs)  # now 2 distinct senders -> done

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

    def test_mid_round_reconnect_is_repolled(self):
        t = make_head()
        # Round 1 is in flight (a reconnect already bumped the epoch) and only
        # rank 0 has reported.
        t._ready = False
        t._epoch = 1
        t._hellos = {0: [b"a"]}
        # Rank 1's agent restarted, so its hello carries no gen: it is not a
        # reply to a rehello that was broadcast before it registered. Dropping
        # it stalls _hellos below _num_agents forever -- the tier never serves
        # again -- so it must be re-polled instead.
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"]})
        self.assertNotIn(1, t._hellos)
        self.assertFalse(t._ready)
        self.assertIn({"t": "rehello", "gen": 1, "to": b"id1"}, t._sent)
        # Its answer for the current round is counted normally.
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"], "gen": 1})
        self.assertTrue(t._ready)
        self.assertIn(b"a", t._present)

    def test_restart_after_a_round_is_not_ignored(self):
        t = make_head()
        t._on_hello(b"id0", {"rank": 0, "keys": [b"a"]})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"]})
        self.assertTrue(t._ready)
        # Round 1: rank 0 reconnects and everyone re-reports.
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"]})
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"], "gen": 1})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"b"], "gen": 1})
        self.assertTrue(t._ready)
        self.assertEqual(t._epoch, 1)
        # Rank 1 now restarts: gen-less hello with gen (0) < epoch (1). It is
        # not a late reply, so ignoring it would keep serving an index built
        # from the previous inventory of that node.
        t._on_hello(b"id1", {"rank": 1, "keys": [b"c"]})
        self.assertFalse(t._ready)
        self.assertEqual(t._epoch, 2)
        self.assertEqual(t._hellos, {})

    def test_stale_zero_reply_while_ready_is_ignored(self):
        t = make_head()
        # Round 0 reconciles (gen-less hellos), then rank 0 reconnects and
        # everyone answers for epoch 1.
        t._on_hello(b"id0", {"rank": 0, "keys": [b"a"]})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"]})
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"]})
        t._on_hello(b"id0", {"rank": 0, "keys": [b"b"], "gen": 1})
        t._on_hello(b"id1", {"rank": 1, "keys": [b"b"], "gen": 1})
        self.assertTrue(t._ready)
        self.assertEqual(t._epoch, 1)
        sent = len(t._sent)
        # A round-0 reply (gen=0, explicitly present) arriving late is stale --
        # not a spontaneous restart. Conflating "gen absent" with "gen=0" would
        # discard the reconciled index and re-poll every agent for nothing.
        t._on_hello(b"id1", {"rank": 1, "keys": [b"zzz"], "gen": 0})
        self.assertTrue(t._ready)
        self.assertEqual(t._epoch, 1)
        self.assertEqual(len(t._sent), sent)
        self.assertIn(b"b", t._present)
        self.assertNotIn(b"zzz", t._present)

    def test_duplicate_rank_cannot_satisfy_the_barrier(self):
        t = make_head()
        t._on_hello(b"id0", {"rank": 0, "keys": [b"a"]})
        t._on_hello(b"id9", {"rank": 0, "keys": [b"a"]})
        self.assertFalse(t._ready)
        self.assertEqual(t.lookup(b"a", None), LookupResult.MISS)
        t._on_hello(b"id1", {"rank": 1, "keys": [b"a"]})
        self.assertTrue(t._ready)
        self.assertEqual(t.lookup(b"a", None), LookupResult.HIT)


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
        cm._store_all_keys = {}
        cm._store_in_flight = set()
        cm._unreadable = set()
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

    def test_completion_releases_load_pins(self):
        # The pinned image polls get_finished_jobs() (tiering/base.py's abstract
        # hook, implemented by FileSystemTierManager). The pins a load takes in
        # submit_load() must be released when that job is reported finished: a
        # pin that outlives its job is skipped by _evict_for() forever, so the
        # byte cap is bypassed while the disk fills. Both published hook names
        # must reach that same body.
        for hook in ("get_finished_jobs", "get_finished"):
            with self.subTest(hook=hook):
                cm = self._make()
                cm._load_job_keys = {1: ["/x/k"]}
                cm._pinned = {"/x/k": 1}
                Res = type("Res", (), {"job_id": 1, "success": True})
                orig = FileSystemTierManager.get_finished_jobs
                try:
                    FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
                    self.assertEqual(len(list(getattr(cm, hook)())), 1)
                finally:
                    FileSystemTierManager.get_finished_jobs = orig
                self.assertEqual(cm._pinned, {})

    def test_failed_load_drops_retired_block_accounting(self):
        cm = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "retired.bin")
            with open(p, "wb") as f:
                f.write(b"x" * 100)
            cm._lru = OrderedDict([(p, 100)])
            cm._total_bytes = 100
            cm._pinned = {p: 1}
            cm._load_job_keys = {1: [p]}
            os.unlink(p)  # _load_block_buffered retired it on the I/O thread
            Res = type("Res", (), {"job_id": 1, "success": False})
            orig = FileSystemTierManager.get_finished_jobs
            try:
                FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
                list(cm.get_finished())
            finally:
                FileSystemTierManager.get_finished_jobs = orig
        self.assertNotIn(p, cm._lru)
        self.assertEqual(cm._total_bytes, 0)
        self.assertEqual(cm._pinned, {})

    def test_failed_load_quarantines_unremovable_file(self):
        # A block can remain eligible for lookup if retiring an unreadable file
        # fails (e.g. os.remove raises): the parent's lookup() is a bare
        # os.path.exists(), so the surviving file keeps reporting a HIT and the
        # same unusable block is re-promoted forever. It must be forgotten AND
        # quarantined so lookup() returns MISS until a fresh store rewrites it.
        cm = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "stuck.bin")
            with open(p, "wb") as f:
                f.write(b"x" * 100)
            cm._lru = OrderedDict([(p, 100)])
            cm._total_bytes = 100
            cm._pinned = {p: 1}
            cm._load_job_keys = {1: [p]}
            # The file survives because retirement FAILED (not retired like the
            # test above), so it is still present on disk -- the dangerous case.
            Res = type("Res", (), {"job_id": 1, "success": False})
            orig = FileSystemTierManager.get_finished_jobs
            try:
                FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
                list(cm.get_finished())
            finally:
                FileSystemTierManager.get_finished_jobs = orig
            self.assertNotIn(p, cm._lru)
            self.assertEqual(cm._total_bytes, 0)
            self.assertIn(p, cm._unreadable)
            # lookup() must report MISS for the quarantined path, not a HIT from
            # the surviving unreadable file (whose bytes are present on disk).
            cm.file_mapper = type(
                "FM", (), {"get_file_name": lambda self, k: p}
            )()
            self.assertIs(cm.lookup("stuck.bin", None), LookupResult.MISS)
            self.assertTrue(os.path.exists(p))  # the file really is still there

    def test_store_unquarantines_a_failed_load_block(self):
        # A fresh store rewrites the file, so a previous quarantine must not
        # keep the manager blind to the now-readable block.
        cm = self._make()
        cm._unreadable = {"/x/k"}
        cm._account(["/x/k"])
        self.assertEqual(cm._unreadable, set())
        self.assertIn("/x/k", cm._lru)
        self.assertEqual(cm._total_bytes, 100)

    def test_failed_load_skips_concurrent_store_path(self):
        # A concurrent store of the same key is about to rewrite the file; the
        # failed load must not quarantine it (the store's fresh write is
        # readable) and must not un-account its already-reserved bytes.
        cm = self._make()
        cm._lru = OrderedDict([("/x/k", 100)])
        cm._total_bytes = 100
        cm._pinned = {"/x/k": 1}
        cm._load_job_keys = {1: ["/x/k"]}
        cm._store_in_flight = {"/x/k"}
        Res = type("Res", (), {"job_id": 1, "success": False})
        orig = FileSystemTierManager.get_finished_jobs
        try:
            FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
            list(cm.get_finished())
        finally:
            FileSystemTierManager.get_finished_jobs = orig
        self.assertIn("/x/k", cm._lru)
        self.assertEqual(cm._total_bytes, 100)
        self.assertNotIn("/x/k", cm._unreadable)

    def test_failed_store_keeps_present_file_accounted(self):
        cm = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "restored.bin")
            # A later store of the same key re-created the file: un-accounting
            # it here would orphan real bytes (invisible to the cap, never
            # evicted, and never rewritten because lookup() keeps hitting them).
            with open(p, "wb") as f:
                f.write(b"x" * 100)
            cm._store_job_keys = {1: [p]}
            cm._lru = OrderedDict([(p, 100)])
            cm._total_bytes = 100
            Res = type("Res", (), {"job_id": 1, "success": False})
            orig = FileSystemTierManager.get_finished_jobs
            try:
                FileSystemTierManager.get_finished_jobs = lambda self: iter([Res()])
                list(cm.get_finished())
            finally:
                FileSystemTierManager.get_finished_jobs = orig
        self.assertIn(p, cm._lru)
        self.assertEqual(cm._total_bytes, 100)


@unittest.skipIf(_load_block_buffered is None, "vllm/zmq not importable")
class TestLoadBlockBuffered(unittest.TestCase):
    def test_bounds_check_rejects_out_of_range(self):
        view = memoryview(bytearray(100))
        with self.assertRaises(ValueError):
            _load_block_buffered("/nonexistent", view, 90, 20)  # 90+20 > 100

    def test_unreadable_block_is_retired(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "short.bin")
            with open(p, "wb") as f:
                f.write(b"abc")  # shorter than the block being promoted
            with self.assertRaises(OSError):
                _load_block_buffered(p, memoryview(bytearray(100)), 0, 100)
            # The parent's lookup() is a bare os.path.exists(): a block that can
            # never be read must not stay on disk, or every request whose prefix
            # touches it re-promotes it forever.
            self.assertFalse(os.path.exists(p))


def make_agent():
    a = dsv4_shard_tier.ShardAgent.__new__(dsv4_shard_tier.ShardAgent)
    a._direct_layout = None
    a._zero_buf = None
    a._zero_addr = 0
    a._zero_discard_addr = 0
    return a


@unittest.skipIf(dsv4_shard_tier is None, "vllm/zmq not importable")
class TestShardSliceIo(unittest.TestCase):
    def test_slice_bounds_are_rejected_not_clamped(self):
        # memoryview slicing clamps: an out-of-range offset would leave the
        # write loop spinning on an empty buffer (os.write returns 0) instead of
        # failing the job, and a partially clamped one would publish a truncated
        # file that the size-only inventory scan accepts.
        mv = memoryview(bytearray(100))
        with self.assertRaises(OSError):
            dsv4_shard_tier._store_one("/nonexistent", mv, 90, 20)
        with self.assertRaises(OSError):
            dsv4_shard_tier._load_one("/nonexistent", mv, 90, 20)

    def test_gpu_block_zero_is_not_the_null_sentinel(self):
        a = make_agent()
        a._direct_layout = [(0x1000, 16, 8)]
        a._zero_addr = 0xDEAD
        a._zero_discard_addr = 0xBEEF
        self.assertEqual(
            a._direct_iovecs([0, 2], for_write=True),
            [(0x1000, 8), (0x1000 + 2 * 16, 8)],
        )
        self.assertEqual(a._direct_iovecs([-1], for_write=False), [(0xBEEF, 8)])

    def test_failed_frame_is_acked(self):
        # Without the ack the head holds the job for the full double timeout
        # with its primary-tier slots pinned, because a job that is never acked
        # never completes.
        a = make_agent()
        sent = []

        class _Sock:
            def send(self, payload):
                sent.append(payload)

        frame = dsv4_shard_tier._ENC.encode(
            {"t": "store", "job": 7, "keys": [], "bids": []}
        )
        a._ack_frame_failure(_Sock(), frame, RuntimeError("no GPU block ids"))
        self.assertEqual(
            [dsv4_shard_tier._DEC.decode(p) for p in sent],
            [{"t": "ack", "job": 7, "ok": False}],
        )
        # Frames that carry no job must not produce an ack.
        a._ack_frame_failure(
            _Sock(),
            dsv4_shard_tier._ENC.encode({"t": "evict", "keys": []}),
            RuntimeError("x"),
        )
        self.assertEqual(len(sent), 1)


@unittest.skipIf(dsv4_shard_tier is None, "vllm/zmq not importable")
class TestShardInventory(unittest.TestCase):
    def test_keys_that_do_not_round_trip_are_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            from vllm.v1.kv_offload.file_mapper import FileMapper

            mapper = FileMapper(
                root_dir=td, model_name="inert", hash_block_size=16,
                gpu_blocks_per_file=1, tp_size=2, pp_size=1, pcp_size=1,
                dcp_size=1, rank=0, dtype="float32",
            )
            root = mapper.base_path + "_r0"
            good_key = bytes.fromhex("01020304") + (0).to_bytes(4, "big")
            good = mapper.get_file_name(good_key)
            stray = os.path.join(root, "bbb", "bb_g0", "05060708.bin")
            for p in (good, stray):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(b"abcd")


            a = dsv4_shard_tier.ShardAgent.__new__(dsv4_shard_tier.ShardAgent)
            a._rank = 0
            a._slice_bytes = 4
            a._direct_layout = None
            a._direct_cell = 0
            a._mapper = mapper
            # A key that does not round-trip through FileMapper would be counted
            # as resident by the head but can never be served: re-storing the
            # block is cheap, adopting a phantom is not.
            self.assertEqual(a._inventory(), [good_key])


@unittest.skipIf(dsv4_vllm_patches is None, "vllm/zmq not importable")
class TestDirectIoMapping(unittest.TestCase):
    """Scheduler-side staging-slot -> GPU-block map bookkeeping.

    The fixtures are the pinned image's real specs, whose ``block_ids`` is a
    numpy array built by ``BlockIDsLoadStoreSpec.__init__``
    (vllm/v1/kv_offload/base.py:355-356). That type is the whole point: a list
    fixture cannot reproduce either the multi-slot ``ValueError`` or the
    single-element ``[0]`` false negative of a truthiness test.
    """

    def setUp(self):
        self._saved = dict(dsv4_vllm_patches._GPU_BLOCK_MAP)

    def tearDown(self):
        dsv4_vllm_patches._GPU_BLOCK_MAP.clear()
        dsv4_vllm_patches._GPU_BLOCK_MAP.update(self._saved)

    @staticmethod
    def _gpu(block_ids, group_sizes, block_indices):
        return GPULoadStoreSpec(block_ids, group_sizes, block_indices)

    @staticmethod
    def _staging(block_ids):
        return CPULoadStoreSpec(block_ids)

    def test_new_owner_of_a_slot_replaces_the_old_mapping(self):
        staging = self._staging([5])
        dsv4_vllm_patches._map_staging_to_gpu(
            self._gpu([10, 11], [2], [0]), staging, 2
        )
        self.assertEqual(dsv4_vllm_patches.get_gpu_blocks(5), [10, 11])
        # The next job reuses slot 5 but carries no usable spec: its slots are
        # forgotten, so the head cannot read the previous job's GPU blocks as
        # this key's (that would gather/scatter KV belonging to another request).
        dsv4_vllm_patches._map_staging_to_gpu(object(), staging, 2)
        self.assertIsNone(dsv4_vllm_patches.get_gpu_blocks(5))

    def test_shifted_mapping_is_not_recorded(self):
        staging = self._staging([7])  # a 4-block group at F=2 needs 2 slots
        dsv4_vllm_patches._map_staging_to_gpu(
            self._gpu([10, 11, 12, 13], [4], [0]), staging, 2
        )
        self.assertIsNone(dsv4_vllm_patches.get_gpu_blocks(7))

    def test_null_sub_blocks_use_a_negative_sentinel(self):
        dsv4_vllm_patches._map_staging_to_gpu(
            self._gpu([10], [1], [1]), self._staging([3]), 2
        )
        self.assertEqual(
            dsv4_vllm_patches.get_gpu_blocks(3),
            [dsv4_vllm_patches._NULL_BLOCK, 10],
        )

    def test_multi_slot_invalidation_does_not_raise(self):
        # A two-element block_ids array is ambiguous as a condition: a
        # truthiness test raises inside TransferJob.__init__ (the engine's
        # scheduler path), and neither slot is invalidated.
        staging = self._staging([0, 5])
        dsv4_vllm_patches._map_staging_to_gpu(
            self._gpu([10, 11, 12, 13], [2, 2], [0, 0]), staging, 2
        )
        self.assertEqual(dsv4_vllm_patches.get_gpu_blocks(0), [10, 11])
        self.assertEqual(dsv4_vllm_patches.get_gpu_blocks(5), [12, 13])
        dsv4_vllm_patches._forget_staging_slots(staging)
        self.assertIsNone(dsv4_vllm_patches.get_gpu_blocks(0))
        self.assertIsNone(dsv4_vllm_patches.get_gpu_blocks(5))

    def test_slot_zero_is_invalidated_not_skipped(self):
        # [0] is a one-element array, i.e. falsy: a truthiness test names no
        # slot, so the mapping of real GPU block 0's slot survives its owner
        # and is read as the next job's blocks.
        dsv4_vllm_patches._map_staging_to_gpu(
            self._gpu([42, 43], [2], [0]), self._staging([0]), 2
        )
        self.assertEqual(dsv4_vllm_patches.get_gpu_blocks(0), [42, 43])
        dsv4_vllm_patches._forget_staging_slots(self._staging([0]))
        self.assertIsNone(dsv4_vllm_patches.get_gpu_blocks(0))


class TestOptionalDependencySkipContract(unittest.TestCase):
    """The suite may skip only when a top-level optional dependency is absent.

    A skipped suite reads as success to a plain unittest exit status, so an
    over-broad guard turns a drifted vLLM API into a green qualification run.
    Needs no vllm on purpose: this contract must hold on every host.
    """

    def test_only_top_level_absence_skips(self):
        self.assertTrue(
            _missing_external_dependency(
                ModuleNotFoundError("No module named 'vllm'", name="vllm")
            )
        )
        for err in (
            # vLLM is installed; the module the tier imports is not there.
            ModuleNotFoundError("No module named 'vllm.v1'", name="vllm.v1"),
            ModuleNotFoundError(
                "No module named 'vllm.v1.kv_offload'", name="vllm.v1.kv_offload"
            ),
            # Some other dependency of vLLM is missing: not this suite's skip.
            ModuleNotFoundError("No module named 'torch'", name="torch"),
            ImportError("cannot import name 'LookupResult' from 'vllm'"),
            NameError("name 'LookupResult' is not defined"),
            ModuleNotFoundError("No module named 'x'", name=None),
        ):
            with self.subTest(err=repr(err)):
                self.assertFalse(_missing_external_dependency(err))


if __name__ == "__main__":
    unittest.main()
