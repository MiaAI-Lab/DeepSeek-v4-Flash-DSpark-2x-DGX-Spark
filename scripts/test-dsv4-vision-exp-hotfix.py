#!/usr/bin/env python3
"""CPU tests for Vision-Exp image layout + fail-closed hotfix text patches."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patches"))

from vision_exp.image_processor import (  # noqa: E402
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    IMAGE_TOKEN_ID,
    as_pil,
    build_image_block,
    compress_pad_tokens,
    grid_tokens,
    image_block_num_tokens,
    is_unregistered_router_bias,
    is_vision_exp_weight_name,
    looks_like_chw,
    salt_mm_image_hash,
    token_routing_kind,
    vision_args_from_config,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image as _PILImage  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

_spec = importlib.util.spec_from_file_location(
    "hotfix_dsv4_vision_exp",
    ROOT / "patches" / "hotfix-dsv4-vision-exp.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
patch_encoding_text = _mod.patch_encoding_text
patch_model_text = _mod.patch_model_text
patch_dspark_text = _mod.patch_dspark_text
ENC_MARK = _mod.ENC_MARK
ENC_ROLE_MARK = _mod.ENC_ROLE_MARK
ENC_ROLE_PAIRED_MARK = _mod.ENC_ROLE_PAIRED_MARK
ENC_ROLE_TOOL_MARK = _mod.ENC_ROLE_TOOL_MARK
ENC_ROLE_QUOTE_MARK = _mod.ENC_ROLE_QUOTE_MARK
MODEL_MARK = _mod.MODEL_MARK
DSPARK_MARK = _mod.DSPARK_MARK


class VisionExpLayoutTest(unittest.TestCase):
    def test_grid_tokens_square_is_bounded(self):
        n_h, n_w, n_tok = grid_tokens(756, 756, 14, 3)
        self.assertEqual((n_h, n_w), (18, 18))
        self.assertLessEqual(n_tok, 384)

    def test_issue172_40x19_block_lengths_are_122_plus_compress_pad(self):
        n_h, n_w, base = grid_tokens(19 * 14, 40 * 14, 14, 3)
        self.assertEqual((n_h, n_w, base), (7, 14, 122))
        expected = {0: 125, 1: 124, 2: 123, 3: 122}
        for start, ntok in expected.items():
            self.assertEqual(compress_pad_tokens(start), 3 - start)
            self.assertEqual(image_block_num_tokens(7, 14, start), ntok)
            self.assertEqual(image_block_num_tokens(7, 14, start + 4), ntok)

    def test_issue172_encoder_cache_hash_includes_block_length(self):
        a = salt_mm_image_hash("deadbeef", 125)
        b = salt_mm_image_hash("deadbeef", 124)
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith("vlexp-ntok125"))
        self.assertTrue(b.endswith("vlexp-ntok124"))

    def test_issue172_processor_salts_encoder_cache_hash(self):
        text = (ROOT / "patches" / "vision_exp" / "processor.py").read_text()
        self.assertIn("salt_mm_image_hash", text)
        self.assertIn("_salt_image_mm_hashes", text)
        self.assertIn("mm_info._replace(hashes=salted)", text)

    def test_issue172_embed_rejects_placeholder_mismatch(self):
        text = (ROOT / "patches" / "vision_exp" / "apply.py").read_text()
        self.assertIn("placeholder/embedding mismatch", text)
        self.assertIn("issue #172", text)

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_issue172_build_image_block_matches_num_tokens_helper(self):
        for start in range(8):
            types, perm = build_image_block(7, 14, start_pos=start)
            self.assertEqual(int(types.numel()), image_block_num_tokens(7, 14, start))
            self.assertEqual(int(perm.numel()), 7 * 14)

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_build_image_block_starts_and_ends(self):
        types, perm = build_image_block(4, 4, start_pos=3)
        self.assertEqual(int(types[0].item()), IMAGE_START)
        self.assertEqual(int(types[-1].item()), IMAGE_END)
        self.assertGreater(int((types == IMAGE).sum()), 0)
        self.assertEqual(int(perm.numel()), 16)

    @unittest.skipUnless(HAS_TORCH and HAS_PIL, "torch+Pillow not installed on this host")
    def test_pil_to_patches_respects_max_tokens(self):
        from PIL import Image

        from vision_exp.image_processor import pil_to_patches

        args = vision_args_from_config(
            type("C", (), {"vision_max_n_token": 384, "hidden_size": 4096})()
        )
        image = Image.new("RGB", (2048, 2048), (20, 40, 80))
        patches, n_h, n_w, n_llm_h, n_llm_w = pil_to_patches(image, args)
        types, perm = build_image_block(n_llm_h, n_llm_w, start_pos=0)
        self.assertEqual(patches.shape[0], n_h * n_w)
        self.assertLessEqual(types.numel(), 384)
        self.assertEqual(int(perm.numel()), int((types == IMAGE).sum()))

    @unittest.skipUnless(HAS_PIL, "Pillow not installed on this host")
    def test_as_pil_accepts_pil_hwc_chw_and_dict(self):
        from PIL import Image

        rgb = Image.new("RGB", (4, 6), (10, 20, 30))
        got = as_pil(rgb)
        self.assertEqual(got.size, (4, 6))
        self.assertEqual(got.getpixel((0, 0)), (10, 20, 30))
        wrapped = as_pil({"image": rgb})
        self.assertEqual(wrapped.size, (4, 6))
        try:
            import numpy as np
        except ImportError:
            return
        hwc = np.zeros((6, 4, 3), dtype="uint8")
        hwc[..., 0] = 11
        self.assertEqual(as_pil(hwc).size, (4, 6))
        self.assertEqual(as_pil(hwc).getpixel((0, 0)), (11, 0, 0))
        chw = np.zeros((3, 6, 4), dtype="uint8")
        chw[1] = 22
        got_chw = as_pil(chw)
        self.assertEqual(got_chw.size, (4, 6))
        self.assertEqual(got_chw.getpixel((0, 0)), (0, 22, 0))
        chw_w3 = np.zeros((3, 8, 3), dtype="uint8")
        chw_w3[1] = 22
        self.assertEqual(as_pil(chw_w3).size, (3, 8))
        self.assertEqual(as_pil(chw_w3).getpixel((0, 0)), (0, 22, 0))
        chw_wide = np.zeros((3, 8, 5), dtype="uint8")
        chw_wide[1] = 22
        self.assertEqual(as_pil(chw_wide).getpixel((0, 0)), (0, 22, 0))

    def test_looks_like_chw_width_in_channel_set(self):
        self.assertTrue(looks_like_chw((3, 6, 4)))
        self.assertTrue(looks_like_chw((3, 8, 3)))
        self.assertTrue(looks_like_chw((3, 8, 1)))
        self.assertTrue(looks_like_chw((3, 8, 5)))
        self.assertTrue(looks_like_chw((3, 8, 8)))
        self.assertTrue(looks_like_chw((4, 100, 200)))
        self.assertTrue(looks_like_chw((1, 100, 200)))
        self.assertFalse(looks_like_chw((8, 5, 3)))
        self.assertFalse(looks_like_chw((6, 4, 3)))
        self.assertFalse(looks_like_chw((4, 100, 3)))
        self.assertFalse(looks_like_chw((1, 100, 3)))
        self.assertFalse(looks_like_chw((6, 4)))

    def test_vision_weight_names_bypass_stacked_w1(self):
        self.assertTrue(is_vision_exp_weight_name("aligner.w1.bias"))
        self.assertTrue(is_vision_exp_weight_name("vision.blocks.0.mlp.w1.weight"))
        self.assertTrue(is_vision_exp_weight_name("image_start"))
        self.assertFalse(is_vision_exp_weight_name("layers.0.ffn.w1.weight"))
        self.assertFalse(is_vision_exp_weight_name("model.layers.0.ffn.w1.weight"))

    def test_hash_layer_gate_bias_is_skipped(self):
        routed = {"layers.3.ffn.gate.e_score_correction_bias"}
        self.assertTrue(
            is_unregistered_router_bias(
                "layers.0.ffn.gate.e_score_correction_bias", routed
            )
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "layers.3.ffn.gate.e_score_correction_bias", routed
            )
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "layers.0.ffn.gate.e_score_correction_bias_vl",
                {"layers.0.ffn.gate.e_score_correction_bias_vl"},
            )
        )
        self.assertTrue(
            is_unregistered_router_bias("model.layers.0.ffn.gate.bias_vl", {})
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "model.layers.0.ffn.gate.e_score_correction_bias_vl",
                {"model.layers.0.ffn.gate.e_score_correction_bias_vl"},
            )
        )

    def test_eagle3_aux_target_is_causal_lm_model(self):
        """DSpark/Eagle3 need get_language_model().model (the inner tower)."""

        class Inner:
            def __init__(self):
                self.aux = None
                self.layers = [None] * 43

            def _set_aux_hidden_state_layers(self, layers):
                self.aux = layers

            def embed_input_ids(self, x):
                return x

        class CausalLM:
            def __init__(self):
                self.model = Inner()

            def get_language_model(self):
                return self

        lm = CausalLM()
        parent_ref = lm.get_language_model()
        self.assertTrue(hasattr(parent_ref, "model"))
        self.assertIs(parent_ref.model, lm.model)
        parent_ref.model._set_aux_hidden_state_layers((41, 42, 43))
        self.assertEqual(lm.model.aux, (41, 42, 43))

    def test_hash_moe_keeps_raw_input_ids(self):
        text = (ROOT / "patches" / "vision_exp" / "apply.py").read_text()
        self.assertIn("requires_raw_input_tokens = True", text)
        self.assertIn("multimodal_embeddings", text)
        self.assertIn("_merge_multimodal_embeddings", text)

    def test_issue175_placeholder_id_is_in_vocab_tail(self):
        self.assertEqual(IMAGE_TOKEN_ID, 129264)
        proc = (ROOT / "patches" / "vision_exp" / "processor.py").read_text()
        self.assertIn("return IMAGE_TOKEN_ID", proc)

    def test_issue175_token_routing_kind_splits_placeholder_rows(self):
        img = IMAGE_TOKEN_ID
        self.assertEqual(token_routing_kind(None), "text")
        self.assertEqual(token_routing_kind([]), "text")
        self.assertEqual(token_routing_kind([1, 2, 3]), "text")
        self.assertEqual(token_routing_kind([img, img]), "image")
        self.assertEqual(token_routing_kind([1, img, 2]), "mixed")

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_issue175_token_routing_kind_accepts_tensors(self):
        import torch

        img = IMAGE_TOKEN_ID
        self.assertEqual(token_routing_kind(torch.tensor([7, 8])), "text")
        self.assertEqual(token_routing_kind(torch.tensor([img, img])), "image")
        self.assertEqual(token_routing_kind(torch.tensor([[1, img]])), "mixed")

    def test_issue175_overlay_routes_image_rows_with_bias_vl(self):
        text = (ROOT / "patches" / "vision_exp" / "apply.py").read_text()
        self.assertIn("def fused_topk_bias_split_vl", text)
        self.assertIn("def _wrap_router_compute_routing", text)
        self.assertIn("_wrap_router_compute_routing(router, self.gate)", text)
        self.assertIn("e_score_correction_bias_vl", text)
        self.assertIn('kind == "image"', text)
        self.assertIn("return _call(hidden_states, gating_output, vl, None, None)", text)
        self.assertIn("nvidia_mod.fused_topk_bias = _split_ftb", text)
        self.assertIn("is_current_stream_capturing", text)


class VisionExpHotfixTextTest(unittest.TestCase):
    def test_model_inject_is_idempotent(self):
        src = (
            "class DeepseekV4MoE:\n    pass\n\n"
            "class DeepseekV4ForCausalLM:\n    pass\n"
        )
        first, st1 = patch_model_text(src)
        second, st2 = patch_model_text(first)
        self.assertEqual(st1, "applied")
        self.assertEqual(st2, "skipped")
        self.assertIn(MODEL_MARK, first)
        self.assertEqual(first, second)

    def test_model_missing_class_is_drift(self):
        _, status = patch_model_text("class Other:\n    pass\n")
        self.assertTrue(status.startswith("drift"))

    def test_encoding_relaxes_placeholder_checks(self):
        src = '''
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

def _validate_no_image_sp_tokens(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("bad")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("bad-reason")

def _process_image_blocks(blocks):
    text = blocks[0].get("text") or ""
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("bad-text")
'''
        updated, status = patch_encoding_text(src)
        self.assertEqual(status, "applied")
        self.assertIn(ENC_MARK, updated)
        self.assertIn(ENC_ROLE_MARK, updated)
        self.assertIn(ENC_ROLE_QUOTE_MARK, updated)
        skipped, status2 = patch_encoding_text(updated)
        self.assertEqual(status2, "skipped")
        self.assertEqual(updated, skipped)
        ns: dict = {}
        exec(compile(updated, "encoding.py", "exec"), ns)
        placeholder = ns["IMAGE_PLACEHOLDER"]
        ns["_validate_no_image_sp_tokens"](
            {"role": "user", "content": placeholder}
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        )
        with self.assertRaises(ValueError) as system_err:
            ns["_validate_no_image_sp_tokens"](
                {
                    "role": "system",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            )
        self.assertIn("user messages only", str(system_err.exception))
        with self.assertRaises(ValueError) as assistant_err:
            ns["_validate_no_image_sp_tokens"](
                {"role": "assistant", "content": placeholder}
            )
        self.assertIn("assistant", str(assistant_err.exception))
        ns["_validate_no_image_sp_tokens"]({"role": "system", "content": "text only"})
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "system",
                "content": "You are a helpful assistant. Markdown images look like <image> tags.",
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {"role": "assistant", "content": "example: <image> in markdown"}
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "system",
                "content": "load <image>/tmp/shot.png</image> now",
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "assistant",
                "content": "yes — OpenAI `image_url` parts and `<image>path</image>` tags",
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "tool",
                "tool_call_id": "chatcmpl-tool-9fce887f3da696ed",
                "content": (
                    "The 0.1.1 image's DeepseekV4ForCausalLM is text-only. "
                    "Vision-Exp ships a 32-layer ViT + Aligner and "
                    + placeholder
                    + " prompt tokens."
                ),
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "tool",
                "content": (
                    "aa25c89ea 2026-08-31 fix(vision): require paired "
                    "<image> tags in the role check (issue #165)"
                ),
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "function",
                "content": [
                    {
                        "type": "text",
                        "text": "docs mention " + placeholder + " and <image>x</image>",
                    }
                ],
            }
        )
        with self.assertRaises(ValueError) as tool_img_err:
            ns["_validate_no_image_sp_tokens"](
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://x/y.png"},
                        }
                    ],
                }
            )
        self.assertIn("tool", str(tool_img_err.exception))

    def test_encoding_role_check_upgrades_bare_image_substring(self):
        src = '''
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

def _validate_no_image_sp_tokens(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("bad")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("bad-reason")

def _process_image_blocks(blocks):
    text = blocks[0].get("text") or ""
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("bad-text")
'''
        first, status = patch_encoding_text(src)
        self.assertEqual(status, "applied")
        stale_inject = f'''
{ENC_ROLE_MARK}
def _dspark_vision_value_has_image(value) -> bool:
    if isinstance(value, str):
        return IMAGE_PLACEHOLDER in value or "<image>" in value
    return False

def _validate_no_image_sp_tokens(msg):
    role = msg.get("role")
    if role in ("user", "developer"):
        return
    if _dspark_vision_value_has_image(msg.get("content")):
        raise ValueError(
            "Images are supported in user messages only: "
            "images in " + repr(role) + " messages return a 400 error."
        )
'''
        stale = first[: first.rfind(ENC_ROLE_MARK)] + stale_inject
        self.assertNotIn(ENC_ROLE_PAIRED_MARK, stale)
        ns_stale: dict = {}
        exec(compile(stale, "encoding.py", "exec"), ns_stale)
        with self.assertRaises(ValueError):
            ns_stale["_validate_no_image_sp_tokens"](
                {
                    "role": "system",
                    "content": "Markdown images look like <image> tags.",
                }
            )
        upgraded, status2 = patch_encoding_text(stale)
        self.assertEqual(status2, "applied")
        self.assertIn(ENC_ROLE_PAIRED_MARK, upgraded)
        self.assertIn(ENC_ROLE_TOOL_MARK, upgraded)
        self.assertIn(ENC_ROLE_QUOTE_MARK, upgraded)
        ns: dict = {}
        exec(compile(upgraded, "encoding.py", "exec"), ns)
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "system",
                "content": "Markdown images look like <image> tags.",
            }
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "assistant",
                "content": "yes — `<image>path</image>` tags",
            }
        )

    def test_encoding_role_check_upgrades_tool_text_scan(self):
        src = '''
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

def _validate_no_image_sp_tokens(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("bad")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("bad-reason")

def _process_image_blocks(blocks):
    text = blocks[0].get("text") or ""
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("bad-text")
'''
        first, status = patch_encoding_text(src)
        self.assertEqual(status, "applied")
        stale_inject = f'''
{ENC_ROLE_MARK}
{ENC_ROLE_PAIRED_MARK}
def _dspark_vision_text_has_image(text: str) -> bool:
    if IMAGE_PLACEHOLDER in text:
        return True
    needle, close = "<image>", "</image>"
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return False
        if text.find(close, i + len(needle)) >= 0:
            return True
        start = i + len(needle)


def _dspark_vision_value_has_image(value) -> bool:
    if isinstance(value, str):
        return _dspark_vision_text_has_image(value)
    return False


def _validate_no_image_sp_tokens(msg):
    role = msg.get("role")
    if role in ("user", "developer"):
        return
    if _dspark_vision_value_has_image(msg.get("content")):
        raise ValueError(
            "Images are supported in user messages only: "
            "images in " + repr(role) + " messages return a 400 error."
        )
'''
        stale = first[: first.rfind(ENC_ROLE_MARK)] + stale_inject
        self.assertIn(ENC_ROLE_PAIRED_MARK, stale)
        self.assertNotIn(ENC_ROLE_TOOL_MARK, stale)
        ns_stale: dict = {}
        exec(compile(stale, "encoding.py", "exec"), ns_stale)
        placeholder = ns_stale["IMAGE_PLACEHOLDER"]
        with self.assertRaises(ValueError):
            ns_stale["_validate_no_image_sp_tokens"](
                {"role": "tool", "content": "mentions " + placeholder}
            )
        upgraded, status2 = patch_encoding_text(stale)
        self.assertEqual(status2, "applied")
        self.assertIn(ENC_ROLE_TOOL_MARK, upgraded)
        self.assertIn(ENC_ROLE_QUOTE_MARK, upgraded)
        skipped, status3 = patch_encoding_text(upgraded)
        self.assertEqual(status3, "skipped")
        self.assertEqual(upgraded, skipped)
        ns: dict = {}
        exec(compile(upgraded, "encoding.py", "exec"), ns)
        ns["_validate_no_image_sp_tokens"](
            {"role": "tool", "content": "mentions " + ns["IMAGE_PLACEHOLDER"]}
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "assistant",
                "content": "yes — OpenAI `image_url` parts and `<image>path</image>` tags",
            }
        )
        with self.assertRaises(ValueError):
            ns["_validate_no_image_sp_tokens"](
                {"role": "assistant", "content": ns["IMAGE_PLACEHOLDER"]}
            )

    def test_encoding_role_check_upgrades_quoted_paired_tags(self):
        src = '''
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

def _validate_no_image_sp_tokens(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("bad")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("bad-reason")

def _process_image_blocks(blocks):
    text = blocks[0].get("text") or ""
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("bad-text")
'''
        first, status = patch_encoding_text(src)
        self.assertEqual(status, "applied")
        stale_inject = f'''
{ENC_ROLE_MARK}
{ENC_ROLE_PAIRED_MARK}
{ENC_ROLE_TOOL_MARK}
def _dspark_vision_text_has_image(text: str) -> bool:
    if IMAGE_PLACEHOLDER in text:
        return True
    needle, close = "<image>", "</image>"
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            return False
        if text.find(close, i + len(needle)) >= 0:
            return True
        start = i + len(needle)


def _dspark_vision_value_has_image(value, scan_text: bool = True) -> bool:
    if isinstance(value, str):
        return bool(scan_text) and _dspark_vision_text_has_image(value)
    return False


def _validate_no_image_sp_tokens(msg):
    role = msg.get("role")
    if role in ("user", "developer"):
        return
    scan_text = role not in ("tool", "function")
    if _dspark_vision_value_has_image(msg.get("content"), scan_text):
        raise ValueError(
            "Images are supported in user messages only: "
            "images in " + repr(role) + " messages return a 400 error."
        )
'''
        stale = first[: first.rfind(ENC_ROLE_MARK)] + stale_inject
        self.assertIn(ENC_ROLE_TOOL_MARK, stale)
        self.assertNotIn(ENC_ROLE_QUOTE_MARK, stale)
        ns_stale: dict = {}
        exec(compile(stale, "encoding.py", "exec"), ns_stale)
        quoted = "yes — OpenAI `image_url` parts and `<image>path</image>` tags"
        with self.assertRaises(ValueError):
            ns_stale["_validate_no_image_sp_tokens"](
                {"role": "assistant", "content": quoted}
            )
        upgraded, status2 = patch_encoding_text(stale)
        self.assertEqual(status2, "applied")
        self.assertIn(ENC_ROLE_QUOTE_MARK, upgraded)
        skipped, status3 = patch_encoding_text(upgraded)
        self.assertEqual(status3, "skipped")
        self.assertEqual(upgraded, skipped)
        ns: dict = {}
        exec(compile(upgraded, "encoding.py", "exec"), ns)
        ns["_validate_no_image_sp_tokens"]({"role": "assistant", "content": quoted})
        ns["_validate_no_image_sp_tokens"](
            {"role": "system", "content": "load <image>/tmp/shot.png</image> now"}
        )
        with self.assertRaises(ValueError):
            ns["_validate_no_image_sp_tokens"](
                {"role": "assistant", "content": ns["IMAGE_PLACEHOLDER"]}
            )
        with self.assertRaises(ValueError):
            ns["_validate_no_image_sp_tokens"](
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            )

    def test_dspark_remaps_bias_vl_before_lookup(self):
        src = '''
class DSparkDeepseekV4ForCausalLM:
    def load_weights(self, weights):
        params_dict = {}
        for name, loaded_weight in weights:
            if False:
                pass
            else:
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
'''
        updated, status = patch_dspark_text(src)
        self.assertEqual(status, "applied")
        self.assertIn(DSPARK_MARK, updated)
        self.assertIn(".ffn.gate.bias_vl", updated)
        self.assertIn("if name not in params_dict:", updated)
        skipped, status2 = patch_dspark_text(updated)
        self.assertEqual(status2, "skipped")
        self.assertEqual(updated, skipped)
        _, drift = patch_dspark_text("def load_weights(self, weights): pass\n")
        self.assertTrue(drift.startswith("drift"))


class VisionExpComposeWiringTest(unittest.TestCase):
    def test_limit_mm_is_json_not_bare_image_eq(self):
        text = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertNotIn(
            "--limit-mm-per-prompt ${LIMIT_MM_PER_PROMPT:-image=8}",
            text,
        )
        self.assertIn('LIMIT_MM_ARGS=(--limit-mm-per-prompt "$${LIMIT_MM_JSON}")', text)
        self.assertIn('"$${LIMIT_MM_ARGS[@]}"', text)
        self.assertIn("${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}", text)

    def test_worker_vision_exp_sync_does_not_nest(self):
        text = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn('scp -r "$SCRIPT_DIR/patches/vision_exp/."', text)
        self.assertIn("rm -rf '${REMOTE_WORKER_DIR}/patches/vision_exp'", text)


if __name__ == "__main__":
    unittest.main()
