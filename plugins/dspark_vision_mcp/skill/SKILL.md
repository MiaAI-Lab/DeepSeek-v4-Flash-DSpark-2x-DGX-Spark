---
name: dspark-vision
description: >-
  Local vision for DeepSeek-V4-Flash-0731 via Qwen3-VL sidecar MCP tools.
  Use when the user attaches or mentions an image path/URL and needs visual
  facts, OCR, or image comparison — stay on 0731; do not switch to a VL model.
---

# DSpark local vision

The default model (`deepseek-v4-flash-0731`) is **text-only**. Vision comes from
MCP tools that call the on-prem **Qwen3-VL-4B** sidecar (`http://127.0.0.1:8889`,
TP=2 across both Sparks).

## When to use

- User gives a local path (`/home/mia/pic2.jpg`) or image URL and asks what is
  in it, colors, setting, text, differences between photos, etc.
- Prefer tools over guessing from filenames.

## Tools

1. **`describe_image(path_or_url, question?)`** — detailed factual description.
   Pass the user's question as `question` so the description focuses correctly.
2. **`ocr_image(path_or_url)`** — extract visible text only.
3. **`compare_images(paths, question)`** — up to 4 images.

## Workflow

1. Call `describe_image` (or OCR/compare) with the **absolute** path or URL.
2. Read the returned text (or the `Error: …` message if the file/sidecar failed).
3. Reason over that text yourself on 0731 — including high/max thinking. Do
   **not** switch to `qwen3-vl-sidecar` for the answer.

## Notes

- Sidecar must be running (`./start-deepseek-v4-flash-dspark.sh`, `ENABLE_VL_SIDECAR=1`).
- Max 4 images per compare call. Huge images are auto-downscaled.
- Do not send `image_url` to `:8888` — 0731 is text-only.
