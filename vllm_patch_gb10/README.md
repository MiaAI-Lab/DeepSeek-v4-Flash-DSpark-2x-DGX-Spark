# vllm-gb10-hybrid-nvfp4

Experimental vLLM quantization plugin for NVIDIA DGX Spark / GB10 (`sm_121`).
It dispatches NVFP4 linear layers by token count:

- `M < GB10_HYBRID_NVFP4_M_THRESHOLD`: Marlin W4A16 path
- `M >= GB10_HYBRID_NVFP4_M_THRESHOLD`: CUTLASS W4A4 path

This composes stock vLLM kernels. It does **not** implement new CUDA kernels,
and it will not fix a vLLM/CUDA stack where the CUTLASS NVFP4 path is missing,
slow, or disabled for `sm_121`.

## Install

```bash
python -m pip install -e . --no-deps
```

## Use

```bash
export VLLM_PLUGINS=gb10_hybrid_nvfp4
export GB10_HYBRID_NVFP4_M_THRESHOLD=128

vllm serve /path/to/your/NVFP4-W4A4-model \
  --quantization modelopt_gb10_hybrid \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 4
```

## Notes

- Requires a ModelOpt NVFP4 checkpoint whose layers have W4A4 metadata and
  activation/input scale tensors.
- Uses extra memory because it keeps a Marlin-prepared copy and a CUTLASS copy
  of FP4 weights.
- Re-benchmark `GB10_HYBRID_NVFP4_M_THRESHOLD`; `128` is only a starting point.
- Tested here only for syntax packaging, not on a live GB10.
