# Bundled refusal direction

`direction_r1.pt` — a unit-norm 4096-dim refusal-direction tensor for
the optional runtime ablation path (`ABLATE=1`, see the main README).

Thank you [`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors`](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors)
for publishing this direction.

Byte-identical (sha256 `6e4d8a8f3aa9e21795faab2c5b14d29b019acdf2ddbfbd8238430458a5837fe0`)
to `results/refusal_direction_r1.pt` from
[drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks](https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks),
captured on the FP8 DSpark sibling of `deepseek-ai/DeepSeek-V4-Flash-0731`.
For `ABLATE=1`, the two-node launcher SHA-256-verifies and stages this file
into each node's HF cache at `dspark-ablation/direction_r1.pt` (container path
`/cache/huggingface/dspark-ablation/direction_r1.pt`). Set the host-side
`DSPARK_ABLATE_SOURCE_FILE` knob to stage a different direction deliberately.

Redistributed under that repository's MIT license:

---

MIT License

Copyright (c) 2026 drowzeys / keys

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

NOTE: Model weights are derived from deepseek-ai/DeepSeek-V4-Flash-DSpark and
remain subject to the original model license terms.
