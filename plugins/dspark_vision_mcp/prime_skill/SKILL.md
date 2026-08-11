---
name: dspark-vision
description: >-
  Local vision for DeepSeek-V4-Flash-0731 via the on-prem Qwen3-VL sidecar.
  Use when the user attaches or mentions an image path/URL and needs visual
  facts, OCR, or image comparison — stay on 0731; do not switch to a VL model.
  Call from IPython: await dspark_vision.describe_image(...), ocr_image(...),
  compare_images(...).
---

# DSpark local vision (Prime Agent)

Prime Agent only supports **HTTP** MCP integrations today, so this skill talks
to the Qwen3-VL sidecar (`http://127.0.0.1:8889`) directly from the kernel.

## When to use

- User gives a local path or image URL and asks what is in it, colors, text, etc.
- Prefer these helpers over guessing from filenames.

## IPython usage

```python
import dspark_vision

print(await dspark_vision.describe_image("/abs/path.jpg", question="What color is the sweater?"))
print(await dspark_vision.ocr_image("/abs/path.jpg"))
print(await dspark_vision.compare_images(["/a.jpg", "/b.jpg"], question="What changed?"))
```

## Notes

- Sidecar must be running (`./start-deepseek-v4-flash-dspark.sh`, `ENABLE_VL_SIDECAR=1`).
- Override base with env `DSPARK_VL_BASE_URL` (default `http://127.0.0.1:8889`).
- Max 4 images for compare. Huge images are auto-downscaled.
