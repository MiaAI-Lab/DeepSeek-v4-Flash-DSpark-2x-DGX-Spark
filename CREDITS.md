# Credits

This repo combines several public efforts. Please credit the upstream authors
when reusing the recipe, the patch, or benchmark numbers.

## Special Thanks

**[drowzeys ("Keys")](https://github.com/drowzeys/)** — special thanks for
publishing the work that made real DSpark concurrency possible on DGX Spark.
Keys' public repos and patches provided:

- the in-server DSpark concurrency patch used in this overlay
- request-stable DSpark main-KV slot mapping for `max_num_seqs > 1`
- ragged `query_start_loc` handling for mixed prefill/decode batches
- early `nvfp4_ds_mla` KV-cache recipe wiring on Spark hardware

This repo's concurrency results, overlay proposer, and NVFP4 launch path all
depend directly on that contribution.

## DSpark Concurrency Patch

The in-server DSpark concurrency breakthrough comes from Keys / drowzeys:

- Repo: https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- Tested commit in this repo: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

Keys' patch fixes the two core blockers for `max_num_seqs > 1`:

- Request-stable DSpark main-KV slots, so persistent DSpark draft KV follows
  request identity instead of condensed vLLM batch-row position.
- Ragged `query_start_loc` handling for real independent-arrival batches where
  prefill and decode rows mix in the same scheduler step.

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

## Two-Node DGX Spark Packaging

MiaAI-Lab published the two-node DGX Spark packaging and launch lineage this
repo builds from:

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

## Upstream Foundations

This work also relies on:

- vLLM
- FlashInfer
- NVIDIA CUDA/NCCL/Blackwell tooling
- DeepSeek V4 Flash
- DeepSeek-AI DeepSpec / DSpark speculative decoding research

## TonyD2Wild NVFP4 Recipe Lineage

TonyD2Wild's public NVFP4 recipe work informed this fork's garble-fix launcher
defaults, runtime documentation, and the non-uniform batch guard merged into the
bind-mounted `dspark_proposer.py`.

- https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark

## MiaAI-Lab Contribution

MiaAI-Lab maintains this fork's validated 2x DGX Spark NVFP4-KV recipe, Stage
A/B/C runtime packaging, sanitized two-node launch flow, Keys concurrency patch
integration, runtime proposer bind-mount, and benchmark artifacts from the
validated runs.

## License Notes

Repo-local scripts and docs are MIT licensed via `LICENSE`.

The vLLM overlay files and `patches/keys-concurrency.patch` are vLLM/DSpark
derived and retain their Apache-2.0 lineage from the upstream sources and
Keys' patch repo. Model weights, base images, CUDA/NCCL, FlashInfer, TileLang,
and Triton are separate upstream artifacts with their own licenses and terms.
