"""Unit tests for shipped MoonViT math/routing (no mocks of the units under test)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Allow `plugins/dsv4_moonvit_vllm` import without install.
ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "dsv4_moonvit_vllm"
sys.path.insert(0, str(PLUGIN))

from dsv4_moonvit_vllm.config import (  # noqa: E402
    PROJECTOR_SHA256,
    TOWER_SHA256,
    verify_artifact_shas,
)
from dsv4_moonvit_vllm.preprocess import (  # noqa: E402
    navit_resize_image,
    pil_to_pixel_values_and_grid,
)
from dsv4_moonvit_vllm.projector import (  # noqa: E402
    PatchMerger,
    count_parameters,
    expected_projector_param_count,
    project_tower_merged_features,
)
from dsv4_moonvit_vllm.routing import (  # noqa: E402
    DEFAULT_ROUTING_PALETTE,
    IMAGE_TOKEN_ID_DEFAULT,
    apply_palette_cycle,
    palette_cycle_replacements,
    routing_replacements_for_spans,
)
from dsv4_moonvit_vllm.wrapper import (  # noqa: E402
    TransparentLanguageModelProxy,
    assert_dspark_transparency,
)


class TestRouting:
    def test_text_ids_unchanged(self):
        ids = [100, 200, 300, 400]
        out = apply_palette_cycle(ids, image_token_id=IMAGE_TOKEN_ID_DEFAULT)
        assert out == ids

    def test_palette_cycle_on_image_span_only(self):
        # text + 5 image placeholders + text
        img = IMAGE_TOKEN_ID_DEFAULT
        ids = [10, 20, img, img, img, img, img, 30]
        out = apply_palette_cycle(ids)
        assert out[0] == 10 and out[1] == 20 and out[-1] == 30
        expected = list(DEFAULT_ROUTING_PALETTE[:5])
        assert out[2:7] == expected

    def test_palette_wraps_after_64(self):
        img = IMAGE_TOKEN_ID_DEFAULT
        n = 70
        ids = [img] * n
        out = apply_palette_cycle(ids)
        for i in range(n):
            assert out[i] == DEFAULT_ROUTING_PALETTE[i % 64]

    def test_tensor_path(self):
        img = IMAGE_TOKEN_ID_DEFAULT
        t = torch.tensor([1, img, img, 2], dtype=torch.long)
        out = apply_palette_cycle(t)
        assert out.tolist() == [
            1,
            DEFAULT_ROUTING_PALETTE[0],
            DEFAULT_ROUTING_PALETTE[1],
            2,
        ]

    def test_span_chunked_prefill_phase(self):
        # image at absolute [10, 14] inclusive; chunk starts at prefix=12, len=5
        # overlap absolute 12,13,14 → phase 2,3,4
        reps = routing_replacements_for_spans(
            extend_prefix_lens=[12],
            extend_seq_lens=[5],
            image_spans=[[(10, 14)]],
            palette=DEFAULT_ROUTING_PALETTE,
        )
        assert reps == [
            (0, DEFAULT_ROUTING_PALETTE[2]),
            (1, DEFAULT_ROUTING_PALETTE[3]),
            (2, DEFAULT_ROUTING_PALETTE[4]),
        ]

    def test_replacements_helper(self):
        ids = [IMAGE_TOKEN_ID_DEFAULT, 5, IMAGE_TOKEN_ID_DEFAULT]
        reps = palette_cycle_replacements(ids)
        # two separate image spans of length 1 each → both phase 0
        assert reps == [(0, DEFAULT_ROUTING_PALETTE[0]), (2, DEFAULT_ROUTING_PALETTE[0])]

    def test_multi_span_palette_restart(self):
        """Two image spans each restart palette at index 0."""
        img = IMAGE_TOKEN_ID_DEFAULT
        ids = [10, img, img, 20, img, img, img, 30]
        out = apply_palette_cycle(ids)
        # Text unchanged
        assert out[0] == 10 and out[3] == 20 and out[-1] == 30
        # First span: 2 image tokens → palette[0], palette[1]
        assert out[1] == DEFAULT_ROUTING_PALETTE[0]
        assert out[2] == DEFAULT_ROUTING_PALETTE[1]
        # Second span: 3 image tokens → palette restarts at 0
        assert out[4] == DEFAULT_ROUTING_PALETTE[0]
        assert out[5] == DEFAULT_ROUTING_PALETTE[1]
        assert out[6] == DEFAULT_ROUTING_PALETTE[2]

    def test_replacements_multi_span(self):
        """palette_cycle_replacements returns phase-0 restart per span."""
        img = IMAGE_TOKEN_ID_DEFAULT
        ids = [img, img, 5, img, img, img]
        reps = palette_cycle_replacements(ids)
        # Span 1: positions 0,1 → palette[0], palette[1]
        assert reps[0] == (0, DEFAULT_ROUTING_PALETTE[0])
        assert reps[1] == (1, DEFAULT_ROUTING_PALETTE[1])
        # Span 2: positions 3,4,5 → palette restarts at 0
        assert reps[2] == (3, DEFAULT_ROUTING_PALETTE[0])
        assert reps[3] == (4, DEFAULT_ROUTING_PALETTE[1])
        assert reps[4] == (5, DEFAULT_ROUTING_PALETTE[2])



class TestProjector:
    def test_shape_from_merged_groups(self):
        m = PatchMerger()
        # 7 merged tokens, each 2x2x1152
        x = torch.randn(7, 4, 1152)
        y = m(x)
        assert y.shape == (7, 4096)

    def test_shape_from_packed_patches(self):
        m = PatchMerger()
        x = torch.randn(28, 1152)  # 7 groups * 4
        y = m(x)
        assert y.shape == (7, 4096)

    def test_max_512_tokens(self):
        m = PatchMerger()
        x = torch.randn(512, 4, 1152)
        y = m(x)
        assert y.shape == (512, 4096)
        assert y.shape[0] <= 512

    def test_param_count_matches_webbrain(self):
        m = PatchMerger()
        assert count_parameters(m) == expected_projector_param_count()

    def test_load_real_projector_weights_if_present(self):
        path = os.environ.get(
            "DSV4_MOONVIT_PROJECTOR",
            str(
                Path.home()
                / ".cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors"
            ),
        )
        if not Path(path).is_file():
            pytest.skip(f"projector weights not at {path}")
        m = PatchMerger()
        loaded = m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        assert len(loaded) == 6
        x = torch.randn(3, 4, 1152, dtype=torch.bfloat16)
        y = m(x)
        assert y.shape == (3, 4096)
        assert y.dtype == torch.bfloat16
        # Determinism: same input → same output
        y2 = m(x)
        assert torch.equal(y, y2)

    def test_project_tower_merged_features(self):
        m = PatchMerger()
        outs = [torch.randn(2, 4, 1152), torch.randn(5, 4, 1152)]
        y = project_tower_merged_features(m, outs)
        assert y.shape == (7, 4096)

    def test_projector_binding_exact_if_present(self):
        """Every WebBrain file tensor must land bit-exact on a named param."""
        path = os.environ.get(
            "DSV4_MOONVIT_PROJECTOR",
            str(
                Path.home()
                / ".cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors"
            ),
        )
        if not Path(path).is_file():
            pytest.skip(f"projector weights not at {path}")
        from safetensors.torch import load_file

        file_tensors = load_file(str(path), device="cpu")
        m = PatchMerger()
        m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        params = dict(m.named_parameters())
        mapping = {
            "pre_norm.weight": "pre_norm.weight",
            "pre_norm.bias": "pre_norm.bias",
            "proj.0.weight": "linear_1.weight",
            "proj.0.bias": "linear_1.bias",
            "proj.2.weight": "linear_2.weight",
            "proj.2.bias": "linear_2.bias",
        }
        assert set(file_tensors) == set(mapping)
        for file_key, param_name in mapping.items():
            assert params[param_name].shape == file_tensors[file_key].shape
            assert torch.equal(params[param_name], file_tensors[file_key])


class TestPreprocess:
    def test_navit_token_budget(self):
        r = navit_resize_image(1024, 768, max_image_tokens=512)
        assert r.num_tokens <= 512
        assert r.grid_h * r.grid_w == r.num_patches
        assert r.num_tokens == (r.padded_height // 28) * (r.padded_width // 28)

    def test_pil_pipeline_shape(self):
        from PIL import Image

        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        pv, grid, ntok = pil_to_pixel_values_and_grid(img, max_image_tokens=512)
        assert pv.ndim == 4 and pv.shape[1] == 3
        assert grid[0] == 1
        assert pv.shape[0] == grid[1] * grid[2]
        assert ntok <= 512
        assert ntok == (grid[1] // 2) * (grid[2] // 2)

    def test_channel_order_rgb(self):
        """Solid red must light up channel 0; blue channel 2 (RGB, not BGR)."""
        from PIL import Image

        red = Image.new("RGB", (256, 256), color=(255, 0, 0))
        pv, _, _ = pil_to_pixel_values_and_grid(red, max_image_tokens=512)
        means = pv.mean(dim=(0, 2, 3))
        assert means[0] > means[1] and means[0] > means[2]
        assert abs(float(means[1] - means[2])) < 1e-4  # G and B identical for pure red

        blue = Image.new("RGB", (256, 256), color=(0, 0, 255))
        pv, _, _ = pil_to_pixel_values_and_grid(blue, max_image_tokens=512)
        means = pv.mean(dim=(0, 2, 3))
        assert means[2] > means[0] and means[2] > means[1]

    def test_no_padding_on_merge_multiple(self):
        """280x280 (multiple of 28) must not pad; 256x256 pads to 280."""
        r = navit_resize_image(280, 280, max_image_tokens=512)
        assert r.pad_width == 0 and r.pad_height == 0
        assert r.grid_h == 20 and r.grid_w == 20
        r = navit_resize_image(256, 256, max_image_tokens=512)
        assert r.padded_width == 280 and r.padded_height == 280


class TestTransparency:
    def test_proxy_lm_head_and_kwargs(self):
        class FakeLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 4)
                self.dspark_marker = "ok"

            def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
                return {"ids": input_ids, "kwargs": kwargs}

            def compute_logits(self, h):
                return h

        class Wrap(TransparentLanguageModelProxy, torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.language_model = FakeLM()

            def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
                return self.language_model(
                    input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
                )

            def get_language_model(self):
                return self.language_model

        w = Wrap()
        assert w.lm_head is w.language_model.lm_head
        assert w.dspark_marker == "ok"
        out = w.forward(torch.tensor([1]), torch.tensor([0]), foo=123)
        assert out["kwargs"]["foo"] == 123
        report = assert_dspark_transparency(w)
        assert report["all_ok"], report


class TestSmokeGate:
    def _load_harness(self):
        import importlib.util

        path = ROOT / "scripts" / "smoke-moonvit-colors.py"
        if not path.is_file():
            pytest.skip(f"smoke harness not at {path}")
        spec = importlib.util.spec_from_file_location("smoke_moonvit_colors", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_answer_matching_synonyms(self):
        mod = self._load_harness()
        assert mod.answer_matches("red", "Red.")
        assert mod.answer_matches("red", "crimson")
        assert mod.answer_matches("red", "The image is scarlet.")
        assert not mod.answer_matches("red", "Black.")
        assert mod.answer_matches("black", "Black.")
        assert not mod.answer_matches("black", "Red.")
        assert mod.answer_matches("white", "white")
        assert mod.answer_matches("green", "Green.")
        assert mod.answer_matches("blue", "blue")
        # Cross-contamination guards
        assert not mod.answer_matches("blue", "Red.")
        assert not mod.answer_matches("green", "Red.")

    def test_thresholds_cover_goal_colors(self):
        mod = self._load_harness()
        assert set(mod.FIXTURES) >= {"red", "black", "white", "green"}
        assert "crimson" in mod.SYNONYMS["red"]


class TestArtifactShas:
    def test_verify_if_present(self):
        base = Path.home() / ".cache/huggingface/webbrain-0731-moonvit-src"
        tower = base / "vision_tower.safetensors"
        proj = base / "mm_projector.safetensors"
        if not (tower.is_file() and proj.is_file()):
            pytest.skip("vision artifacts not staged")
        rep = verify_artifact_shas(tower, proj)
        assert rep["tower_ok"], rep
        assert rep["projector_ok"], rep
        assert rep["tower_sha256"] == TOWER_SHA256
        assert rep["projector_sha256"] == PROJECTOR_SHA256


class TestProjectorFinetuned:
    """Tests for fine-tuned projector (drop-in replacement)."""

    def _finetuned_path(self) -> Path:
        """Return path to fine-tuned projector if it exists.

        Preference: env override → v3 (embedding-aligned) → v2 (color CE).
        """
        env = os.environ.get("DSV4_MOONVIT_FINETUNED_PROJECTOR")
        if env:
            return Path(env)
        hf = Path.home() / ".cache/huggingface"
        for candidate in (
            hf / "webbrain-0731-moonvit-src/mm_projector-v3-0731.safetensors",
            hf / "projector-v3/mm_projector-v3-0731.safetensors",
            hf / "mm_projector-finetuned-0731.safetensors",
        ):
            if candidate.is_file():
                return candidate
        return hf / "mm_projector-finetuned-0731.safetensors"

    def test_projector_finetuned_loads(self):
        """New projector file loads, shapes match, missing=0."""
        path = self._finetuned_path()
        if not path.is_file():
            pytest.skip(f"fine-tuned projector not at {path}")
        m = PatchMerger()
        loaded = m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        assert len(loaded) == 6, f"expected 6 tensors, got {len(loaded)}"
        # Verify shapes match original
        assert m.pre_norm.weight.shape == (1152,)
        assert m.linear_1.weight.shape == (4608, 4608)
        assert m.linear_2.weight.shape == (4096, 4608)

    def test_projector_finetuned_shape(self):
        """Forward pass produces correct (T, 4096) output."""
        path = self._finetuned_path()
        if not path.is_file():
            pytest.skip(f"fine-tuned projector not at {path}")
        m = PatchMerger()
        m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        # Test various input shapes
        for n_tokens in [1, 7, 50, 512]:
            x = torch.randn(n_tokens, 4, 1152, dtype=torch.bfloat16)
            y = m(x)
            assert y.shape == (n_tokens, 4096), f"failed for {n_tokens} tokens"
            assert y.dtype == torch.bfloat16

    def test_projector_finetuned_deterministic(self):
        """Same input produces same output (deterministic)."""
        path = self._finetuned_path()
        if not path.is_file():
            pytest.skip(f"fine-tuned projector not at {path}")
        m = PatchMerger()
        m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        x = torch.randn(10, 4, 1152, dtype=torch.bfloat16)
        y1 = m(x)
        y2 = m(x)
        assert torch.equal(y1, y2)

    def test_projector_finetuned_param_count(self):
        """Fine-tuned projector has same param count as original."""
        path = self._finetuned_path()
        if not path.is_file():
            pytest.skip(f"fine-tuned projector not at {path}")
        m = PatchMerger()
        m.load_webbrain_safetensors(path, device="cpu", dtype=torch.bfloat16)
        count = count_parameters(m)
        assert count == expected_projector_param_count(), (
            f"param count mismatch: {count} != {expected_projector_param_count()}"
        )

    def test_projector_finetuned_differs_from_original(self):
        """Fine-tuned weights differ from original (training happened)."""
        path = self._finetuned_path()
        orig_path = os.environ.get(
            "DSV4_MOONVIT_PROJECTOR",
            str(
                Path.home()
                / ".cache/huggingface/webbrain-0731-moonvit-src/mm_projector.safetensors"
            ),
        )
        if not path.is_file():
            pytest.skip(f"fine-tuned projector not at {path}")
        if not Path(orig_path).is_file():
            pytest.skip(f"original projector not at {orig_path}")

        from safetensors.torch import load_file

        orig_state = load_file(str(orig_path), device="cpu")
        ft_state = load_file(str(path), device="cpu")

        # At least one tensor should differ
        any_different = False
        for key in orig_state:
            if key in ft_state:
                if not torch.equal(orig_state[key], ft_state[key]):
                    any_different = True
                    break
        assert any_different, "fine-tuned weights are identical to original - no training occurred"
