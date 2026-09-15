#!/usr/bin/env python3
"""Install native DeepSeek-V4-Flash-Vision-Exp image support into Anemll vLLM.

The 0.1.1 image's ``DeepseekV4ForCausalLM`` is text-only. Vision-Exp ships a
32-layer ViT + Aligner and ``<｜deepseek_image｜>`` prompt tokens. This
startup patch:

1. Appends a fail-closed import hook to ``nvidia/model.py`` that constructs
   the tower, maps ``vision.*`` / ``aligner.*`` / ``image_*`` / ``bias_vl``
   weights, and registers a vLLM multimodal processor.
2. Remaps DSpark draft ``ffn.gate.bias_vl`` → ``e_score_correction_bias_vl``
   (Anemll only rewrote names ending in ``.ffn.gate.bias``). Image placeholder
   rows then route with that tensor and skip the hash table (issue #175);
   text rows keep ``e_score_correction_bias`` + ``tid2eid``.
3. Relaxes the Vision-Exp encoder's rejection of already-substituted
   ``<｜deepseek_image｜>`` text so OpenAI ``image_url`` parts survive
   vLLM's chat parser, then restores the official Chat Completions rule:
   images in ``user`` messages only (``system`` / ``assistant`` → 400).
   ``tool`` / ``function`` *result text* is not scanned for image markers
   (a ``cat`` of this file must not 400). Quoted ``<image>…</image>`` in
   ``system`` / ``assistant`` prose is also not an image (the served path
   only accepts structured ``image`` / ``image_url`` parts). Structured
   parts and the raw placeholder token in those roles still 400.

Video is not wired: the official weights, ``encoding/``, and ``inference/``
have no video encoder. GIF is decoded as a still RGB frame.

Usage (inside the container, after the encoder copy):
  python3 hotfix-dsv4-vision-exp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_MODEL = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
)
DEFAULT_ENCODING = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)
DEFAULT_DSPARK = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)
DEFAULT_PATCHES = Path("/opt/dspark-patches/vision_exp")

MODEL_MARK = "# [vision-exp-hotfix] native DeepSeek-V4-Flash-Vision-Exp image tower"
ENC_MARK = "# [vision-exp-hotfix] allow vLLM-inserted image placeholders"
ENC_ROLE_MARK = "# [vision-exp-hotfix] images only in user messages"
ENC_ROLE_PAIRED_MARK = "# [vision-exp-hotfix] paired <image> tag (issue 165)"
ENC_ROLE_TOOL_MARK = "# [vision-exp-hotfix] tool text is not an image (issue 167)"
ENC_ROLE_QUOTE_MARK = (
    "# [vision-exp-hotfix] quoted paired tags are prose (issue 181)"
)
DSPARK_MARK = "# [vision-exp-hotfix] remap ffn.gate.bias_vl"

DSPARK_GATE_BIAS_OLD = '''                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]'''

DSPARK_GATE_BIAS_NEW = f'''                if name.endswith(".ffn.gate.bias_vl"):
                    name = name.replace(
                        ".ffn.gate.bias_vl",
                        ".ffn.gate.e_score_correction_bias_vl",
                    )
                elif name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                if name not in params_dict:
                    continue  {DSPARK_MARK}
                param = params_dict[name]'''

MODEL_INJECT = f'''
{MODEL_MARK}
import sys as _dspark_vision_sys
if "/opt/dspark-patches" not in _dspark_vision_sys.path:
    _dspark_vision_sys.path.insert(0, "/opt/dspark-patches")
from vision_exp.apply import apply_vision_exp as _dspark_apply_vision_exp
_dspark_apply_vision_exp(
    DeepseekV4Model=DeepseekV4Model,
    DeepseekV4ForCausalLM=DeepseekV4ForCausalLM,
    DeepseekV4MoE=DeepseekV4MoE,
)
'''

CONTENT_CHECK = (
    "if isinstance(content, str) and IMAGE_PLACEHOLDER in content:"
)
CONTENT_CHECK_NEW = (
    "if False and isinstance(content, str) and IMAGE_PLACEHOLDER in content:"
    f"  {ENC_MARK}"
)
REASONING_CHECK = (
    "if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:"
)
REASONING_CHECK_NEW = (
    "if False and isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:"
    f"  {ENC_MARK}"
)
TEXT_CHECK = "if IMAGE_PLACEHOLDER in text:"
TEXT_CHECK_NEW = f"if False and IMAGE_PLACEHOLDER in text:  {ENC_MARK}"

ENC_ROLE_INJECT = f'''
{ENC_ROLE_MARK}
{ENC_ROLE_PAIRED_MARK}
{ENC_ROLE_TOOL_MARK}
{ENC_ROLE_QUOTE_MARK}
def _dspark_vision_text_has_image(text: str) -> bool:
    # Serve path has no tagged-text expander: only the raw placeholder
    # token in non-user prose is an image. Quoted <image>…</image> is not.
    return IMAGE_PLACEHOLDER in text


def _dspark_vision_value_has_image(value, scan_text: bool = True) -> bool:
    if isinstance(value, str):
        return bool(scan_text) and _dspark_vision_text_has_image(value)
    if not isinstance(value, list):
        return False
    for block in value:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("image", "image_url"):
            return True
        if scan_text:
            text = block.get("text") or ""
            if isinstance(text, str) and _dspark_vision_text_has_image(text):
                return True
        nested = block.get("content")
        if isinstance(nested, list) and _dspark_vision_value_has_image(
            nested, scan_text
        ):
            return True
    return False


def _validate_no_image_sp_tokens(msg):
    """Official restriction: images in user messages only (system/assistant → 400)."""
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            "reasoning_content contains image special token "
            + repr(IMAGE_PLACEHOLDER)
        )
    role = msg.get("role")
    if role in ("user", "developer"):
        return
    scan_text = role not in ("tool", "function")
    if _dspark_vision_value_has_image(
        msg.get("content"), scan_text
    ) or _dspark_vision_value_has_image(msg.get("content_blocks"), scan_text):
        raise ValueError(
            "Images are supported in user messages only: "
            "images in " + repr(role) + " messages return a 400 error."
        )
'''


def patch_model_text(source: str) -> tuple[str, str]:
    if MODEL_MARK in source:
        return source, "skipped"
    if "class DeepseekV4ForCausalLM" not in source or "class DeepseekV4MoE" not in source:
        return source, "drift:missing-dsv4-class"
    updated = source.rstrip() + "\n" + MODEL_INJECT
    compile(updated, "model.py", "exec")
    return updated, "applied"


def _encoding_role_complete(source: str) -> bool:
    return (
        ENC_ROLE_MARK in source
        and ENC_ROLE_PAIRED_MARK in source
        and ENC_ROLE_TOOL_MARK in source
        and ENC_ROLE_QUOTE_MARK in source
    )


def patch_encoding_text(source: str) -> tuple[str, str]:
    if (
        ENC_MARK in source
        and _encoding_role_complete(source)
        and CONTENT_CHECK_NEW in source
    ):
        return source, "skipped"
    if "IMAGE_PLACEHOLDER" not in source:
        return source, "drift:no-image-placeholder"
    if ENC_MARK not in source:
        missing = []
        for old, new in (
            (CONTENT_CHECK, CONTENT_CHECK_NEW),
            (REASONING_CHECK, REASONING_CHECK_NEW),
            (TEXT_CHECK, TEXT_CHECK_NEW),
        ):
            if source.count(old) != 1:
                missing.append(f"{old!r}={source.count(old)}")
                continue
            source = source.replace(old, new, 1)
        if missing:
            return source, "drift:" + ",".join(missing)
    if ENC_ROLE_MARK in source and not _encoding_role_complete(source):
        source = source[: source.rfind(ENC_ROLE_MARK)].rstrip() + "\n"
    if ENC_ROLE_MARK not in source:
        source = source.rstrip() + "\n" + ENC_ROLE_INJECT
    compile(source, "encoding.py", "exec")
    return source, "applied"


def patch_dspark_text(source: str) -> tuple[str, str]:
    if DSPARK_MARK in source and DSPARK_GATE_BIAS_NEW in source:
        return source, "skipped"
    if source.count(DSPARK_GATE_BIAS_OLD) != 1:
        return source, (
            "drift:dspark-gate-bias-remap="
            f"{source.count(DSPARK_GATE_BIAS_OLD)}"
        )
    updated = source.replace(DSPARK_GATE_BIAS_OLD, DSPARK_GATE_BIAS_NEW, 1)
    compile(updated, "dspark.py", "exec")
    return updated, "applied"


def _write(path: Path, original: str, updated: str, status: str) -> None:
    if status == "applied":
        path.write_text(updated)
    elif status != "skipped":
        raise SystemExit(f"FATAL: {path} {status}")
    print(f"vision-exp hotfix {path.name:40s}: {status}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        model = DEFAULT_MODEL.read_text() if DEFAULT_MODEL.is_file() else ""
        encoding = DEFAULT_ENCODING.read_text() if DEFAULT_ENCODING.is_file() else ""
        dspark = DEFAULT_DSPARK.read_text() if DEFAULT_DSPARK.is_file() else ""
        print(
            "vision-exp model.py                    :",
            "APPLIED" if MODEL_MARK in model else "NOT APPLIED",
        )
        print(
            "vision-exp encoding.py                 :",
            "APPLIED"
            if ENC_MARK in encoding and _encoding_role_complete(encoding)
            else "NOT APPLIED",
        )
        print(
            "vision-exp dspark.py                   :",
            "APPLIED" if DSPARK_MARK in dspark else "NOT APPLIED",
        )
        ok = (
            MODEL_MARK in model
            and ENC_MARK in encoding
            and _encoding_role_complete(encoding)
            and DSPARK_MARK in dspark
        )
        return 0 if ok else 1

    patches = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATCHES
    model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MODEL
    encoding_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ENCODING
    dspark_path = Path(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_DSPARK

    if not (patches / "apply.py").is_file() or not (patches / "vision.py").is_file():
        print(f"FATAL: Vision-Exp overlay missing under {patches}", file=sys.stderr)
        return 1
    if not model_path.is_file():
        print(f"FATAL: {model_path} missing", file=sys.stderr)
        return 1
    if not encoding_path.is_file():
        print(
            f"FATAL: {encoding_path} missing (encoder copy must run first)",
            file=sys.stderr,
        )
        return 1
    if not dspark_path.is_file():
        print(f"FATAL: {dspark_path} missing", file=sys.stderr)
        return 1

    model_src = model_path.read_text()
    model_new, model_status = patch_model_text(model_src)
    _write(model_path, model_src, model_new, model_status)

    enc_src = encoding_path.read_text()
    enc_new, enc_status = patch_encoding_text(enc_src)
    _write(encoding_path, enc_src, enc_new, enc_status)

    dspark_src = dspark_path.read_text()
    dspark_new, dspark_status = patch_dspark_text(dspark_src)
    _write(dspark_path, dspark_src, dspark_new, dspark_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
