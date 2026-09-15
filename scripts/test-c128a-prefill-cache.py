#!/usr/bin/env python3
"""Behavioral regression checks for C128A prefill metadata reuse (CPU only)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c128a_patch", ROOT / "patches/hotfix-vllm-c128a-prefill-cache.py"
)
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)
FIXTURE = ROOT / "scripts/fixtures/c128a-prefill-cache/flashinfer-prefill.py"


class Tokens(list):
    device = "cpu"


class PrefillCacheTests(unittest.TestCase):
    def setUp(self):
        self.conversions = 0
        self.observed = []

        def convert(topk, requests, blocks, block_size, valid):
            self.conversions += 1
            indices = [
                [blocks[request][index // block_size] * block_size + index % block_size
                 if index >= 0 else -1 for index in row]
                for row, request in zip(topk, requests)
            ]
            lengths = [sum(index >= 0 for index in row) if ok else 0
                       for row, ok in zip(topk, valid)]
            return indices, lengths

        def attend(**kwargs):
            self.observed.append((kwargs["extra_sparse_indices"],
                                  kwargs["extra_sparse_topk_lens"]))

        namespace = {
            "compute_global_topk_indices_and_lens": convert,
            "flashinfer_trtllm_batch_decode_sparse_mla_dsv4": attend,
        }
        exec(compile(PATCH.transform(FIXTURE.read_text(), PATCH.FS), str(FIXTURE), "exec"), namespace)
        self.attention_type = namespace["PrefillFixture"]

    def metadata(self, blocks=7):
        return SimpleNamespace(
            c128a_prefill_topk_indices=[[0, 2], [1, -1]],
            c128a_prefill_global_topk=None,
            block_size=512,
            block_table=[[blocks, blocks + 4]],
        )

    def forward(self, metadata, ratio=128, topk=None, decode=0, valid=None):
        attention = self.attention_type()
        attention.compress_ratio = ratio
        attention.topk_indices_buffer = topk
        attention.PREFILL_CHUNK_SIZE = 1
        attention.scale = 1.0
        attention.attn_sink = None
        attention._prepare_query = lambda query, output: query
        attention._as_sparse_cache = lambda cache: cache
        attention._get_workspace = lambda device: None
        swa = SimpleNamespace(
            num_prefills=1,
            num_decodes=int(decode > 0),
            num_decode_tokens=decode,
            num_prefill_tokens=2,
            query_start_loc_cpu=[0, decode, decode + 2] if decode else [0, 2],
            token_to_req_indices=[0] * decode + [int(decode > 0)] * 2,
            is_valid_token=[True] * decode + (valid or [True, True]),
            prefill_swa_indices=[[0], [1]],
            prefill_swa_lens=[1, 1],
        )
        attention._forward_prefill(Tokens([0, 1]), [], [], [None, None], metadata, swa)
        return self.observed[-1]

    def test_shared_metadata_reuses_conversion_without_changing_indices(self):
        metadata = self.metadata()
        expected = ([[28, 30], [29, -1]], [2, 1])
        self.assertEqual(self.forward(metadata), expected)
        self.assertEqual(self.forward(metadata), expected)
        self.assertEqual(self.conversions, 1)

    def test_new_step_does_not_reuse_previous_physical_blocks(self):
        self.forward(self.metadata())
        self.assertEqual(self.forward(self.metadata(13)), ([[52, 54], [53, -1]], [2, 1]))
        self.assertEqual(self.conversions, 2)

    def test_c4_indices_remain_layer_dependent(self):
        metadata = self.metadata()
        self.assertEqual(self.forward(metadata, 4, [[0], [1]]), ([[896], [897]], [1, 1]))
        self.assertEqual(self.forward(metadata, 4, [[2], [3]]), ([[898], [899]], [1, 1]))
        self.assertEqual(self.conversions, 2)

    def test_mixed_batch_uses_prefill_request_and_validity_slice(self):
        metadata = self.metadata()
        metadata.block_table.append([13, 17])
        expected = ([[52, 54], [53, -1]], [2, 0])
        self.assertEqual(self.forward(metadata, decode=2, valid=[True, False]), expected)
        self.assertEqual(self.forward(metadata, decode=2, valid=[True, False]), expected)
        self.assertEqual(self.conversions, 1)

    def test_duplicate_or_damaged_regions_are_rejected(self):
        source = FIXTURE.read_text()
        updated = PATCH.transform(source, PATCH.FS)
        for invalid in (source + source, updated.replace(PATCH.MARK, "# damaged", 1),
                        source.replace(PATCH.CALL_OLD, "pass", 1)):
            with self.subTest(source=invalid[:40]), self.assertRaises(ValueError):
                PATCH.transform(invalid, PATCH.FS)


if __name__ == "__main__":
    unittest.main()
