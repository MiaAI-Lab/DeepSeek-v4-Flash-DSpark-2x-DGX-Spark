# Credits

This repo combines several public efforts. Please credit the upstream authors
when reusing the recipe, the patch, or benchmark numbers.

## DSpark Concurrency Patch

The in-server DSpark concurrency breakthrough comes from Keys / drowzeys:

- Repo: https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- Tested commit in this repo: `5da4dc7a1b1f64cb479bf33252d3598860bee783`

Keys' patch fixes the two core blockers for `max_num_seqs > 1`:

- Request-stable DSpark main-KV slots, so persistent DSpark draft KV follows
  request identity instead of condensed vLLM batch-row position.
- Ragged `query_start_loc` handling for real independent-arrival batches where
  prefill and decode rows mix in the same scheduler step.
- Patch 2B ragged detection independent of rejection, validated by the GSM8K
  quality-certification path.

The validated concurrency numbers in this repo depend directly on that patch.

## DSpark vLLM Integration

Rafael Caricio published the DSpark vLLM integration and deployment work this
recipe builds on:

- https://github.com/rafaelcaricio/vllm/pull/1
- https://github.com/rafaelcaricio/spark_vllm_docker/pull/1

## Model And Runtime Work

Fraser Price published the DeepSeek V4 Flash DSpark model/runtime work used by
this recipe:

- https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark
- https://github.com/fraserprice/dspark-vllm

## DSpark-r0b0tlab Model Implementation

This repository now carries the DSpark-r0b0tlab model implementation, draft-head
runtime path, two-node DGX Spark launch flow, and publication artifacts.

## Upstream Foundations

This work also relies on:

- vLLM
- FlashInfer
- NVIDIA CUDA/NCCL/Blackwell tooling
- DeepSeek V4 Flash
- DeepSeek-AI DeepSpec / DSpark speculative decoding research

## Repository Contribution

This repo contributes the validated 2x DGX Spark NVFP4-KV recipe, Stage A/B/C
runtime packaging, sanitized two-node launch flow, application of Keys'
concurrency patch to the NVFP4 profile, and benchmark artifacts from the
validated runs.

## License Notes

Repo-local scripts and docs are MIT licensed via `LICENSE`.

The vLLM overlay files and `patches/keys-concurrency.patch` are vLLM/DSpark
derived and retain their Apache-2.0 lineage from the upstream sources and
Keys' patch repo. Model weights, base images, CUDA/NCCL, FlashInfer, TileLang,
and Triton are separate upstream artifacts with their own licenses and terms.
