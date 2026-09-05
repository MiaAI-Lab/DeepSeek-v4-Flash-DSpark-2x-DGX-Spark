// Scatter-gather batch copy that replaces cuMemcpyBatchAsync for large KV restores.
//
// WHY. vLLM's CPU->GPU offload load issues one cuMemcpyBatchAsync with one descriptor
// per (block, canonical tensor) pair -- ~243 per staging cell, so ~0.237 * tokens.
// Above roughly 23,000 descriptors the NVIDIA driver segfaults INSIDE libcuda.so.1
// (confirmed by core dump: PC in libcuda's range, x0 = 0, consistent with an
// unchecked NULL from a failed internal allocation; dmesg shows NVRM
// NV_ERR_NO_MEMORY on both nodes at the same instant). It is not general host
// memory -- reproduced with 11-13 GB free -- so it is a fixed internal driver pool.
//
// Measured wall:
//     69,048 tok  ~18,000 ops  -> OK
//     91,697 tok   22,562 ops  -> segfault (OK with chunk+sync)
//    141,405 tok   34,714 ops  -> segfault even with chunk=2048 + per-chunk sync
//
// Chunking with a stream sync bounds *pending* copies and bought only ~25%, so the
// driver's limit tracks the TOTAL it is asked to handle, not the in-flight count.
// A 1M-token restore needs ~244,000 descriptors -- an order of magnitude past the
// wall -- so the driver path cannot get there by tuning.
//
// This kernel has no such limit: one CUDA block per descriptor, grid-strided so any
// count works. It is ~1.2-1.3x slower than the driver's batch path (measured
// previously on this box), which is irrelevant here because NVMe feeds the restore
// 8-12x slower than either.
//
// Build inside the container:
//   nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC \
//        -o /opt/env/lib/libdsv4_batch_copy.so dsv4_batch_copy.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// One block per copy. Threads within a block stride over the payload.
// 16-byte vectorised path when src, dst and size are all 16B-aligned (true for the
// nvfp4_ds_mla page sizes 8,640 and 37,440, both multiples of 16); byte fallback
// otherwise so an unaligned layout degrades instead of corrupting.
__global__ void dsv4_sg_copy_kernel(const int64_t* __restrict__ src,
                                    const int64_t* __restrict__ dst,
                                    const int64_t* __restrict__ sizes,
                                    int64_t n) {
    for (int64_t i = blockIdx.x; i < n; i += gridDim.x) {
        const char* s = reinterpret_cast<const char*>(src[i]);
        char* d = reinterpret_cast<char*>(dst[i]);
        int64_t nb = sizes[i];

        bool aligned = ((reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(d)
                         | static_cast<uintptr_t>(nb)) & 0xF) == 0;
        if (aligned) {
            const uint4* s4 = reinterpret_cast<const uint4*>(s);
            uint4* d4 = reinterpret_cast<uint4*>(d);
            int64_t n4 = nb >> 4;
            for (int64_t j = threadIdx.x; j < n4; j += blockDim.x) d4[j] = s4[j];
        } else {
            for (int64_t j = threadIdx.x; j < nb; j += blockDim.x) d[j] = s[j];
        }
    }
}

extern "C" {

// Descriptor arrays must already be in device-accessible memory holding the same
// (src, dst, size) triples the driver path would have consumed.
// Returns 0 on success, else the cudaError_t.
int dsv4_batch_copy(const int64_t* d_src, const int64_t* d_dst, const int64_t* d_sizes,
                    int64_t n, void* stream) {
    if (n <= 0) return 0;
    // Cap the grid so very large n stays within launch limits; the kernel is
    // grid-strided, so a smaller grid just means each block does more copies.
    int64_t grid = n < 65535 ? n : 65535;
    dsv4_sg_copy_kernel<<<static_cast<int>(grid), 256, 0,
                          static_cast<cudaStream_t>(stream)>>>(d_src, d_dst, d_sizes, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[dsv4-sg-copy] launch failed for n=%lld: %s\n",
                static_cast<long long>(n), cudaGetErrorString(e));
        fflush(stderr);
        return static_cast<int>(e);
    }
    return 0;
}

}  // extern "C"
