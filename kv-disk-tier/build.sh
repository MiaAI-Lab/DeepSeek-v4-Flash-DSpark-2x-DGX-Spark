#!/usr/bin/env bash
# Build the disk-tier CUDA kernels. Must run on each node (or build once and
# copy the .so, which is what up-dsv4-diskcache.sh does).
#
# libdsv4_batch_copy.so is what makes large restores possible at all: the NVIDIA
# driver segfaults inside libcuda.so.1 above ~23,000 cuMemcpyBatchAsync
# descriptors, and a 1M-token restore needs ~246,000. See ./README.md.
#
# libdsv4_host_kv.so backs the EXPERIMENTAL DSV4_HOST_KV=1 path (KV cache
# allocated from cudaHostAlloc so the disk tier can DMA straight into it).
#
# IMG must contain the CUDA toolkit (nvcc). The serving image
# (ghcr.io/anemll/dspark-vllm-gx10:0.1.1) has no nvcc, so a separate build
# image is required here.
set -e
KV_SRC="${KV_SRC:-/opt/dsv4-kv}"
IMG="${IMG:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
ARCH="${ARCH:-sm_121a}"     # GB10. Use sm_120a on consumer Blackwell.

docker run --rm --gpus all -v "$KV_SRC":/src --entrypoint bash "$IMG" -lc "
  export PATH=/opt/env/bin:\$PATH
  export LD_LIBRARY_PATH=/opt/env/lib:/opt/env/targets/sbsa-linux/lib:\$LD_LIBRARY_PATH
  nvcc -O3 -arch=$ARCH -shared -Xcompiler -fPIC \
       -o /src/libdsv4_batch_copy.so /src/dsv4_batch_copy.cu
  nvcc -O3 -arch=$ARCH -shared -Xcompiler -fPIC \
       -o /src/libdsv4_host_kv.so /src/dsv4_host_kv_alloc.cu
  ls -la /src/libdsv4_batch_copy.so /src/libdsv4_host_kv.so
"
echo "built $KV_SRC/libdsv4_batch_copy.so $KV_SRC/libdsv4_host_kv.so"
