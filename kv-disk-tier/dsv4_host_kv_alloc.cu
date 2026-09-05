// Pluggable CUDA allocator that backs "device" tensors with cudaHostAlloc memory.
//
// WHY. A cudaMalloc pointer lives in a PROT_NONE anonymous VA reservation with no CPU
// page-table mapping and no struct page, so get_user_pages() rejects it and a file read
// into it returns EFAULT. That single fact is the entire reason the KV offload path needs
// a CPU staging tier: bytes cannot go NVMe -> KV cache directly.
//
// cudaHostAlloc memory has none of those problems. O_DIRECT reads land in it at
// 7.25 GB/s, and -- measured on GB10 with a paged gather at the real nvfp4_ds_mla page
// sizes -- the GPU reads it at 100.8% / 101.4% of cudaMalloc speed. It is NOT the
// pageable/ATS path (that one really does cost 22-31%); cudaHostAlloc gets real GPU
// page-table mappings.
//
// So if the KV cache is allocated through here, the staging buffer and the KV cache
// become the same allocation: the 8.58 GB/node staging region disappears, and with it
// the whole-prefix residency cliff (there are no staging cells left to run out of).
//
// Build inside the container:
//   nvcc -O3 -arch=sm_121a -shared -Xcompiler -fPIC \
//        -o /opt/env/lib/libdsv4_host_kv.so dsv4_host_kv_alloc.cu
//
// Used via torch.cuda.memory.CUDAPluggableAllocator(path, "dsv4_host_malloc",
// "dsv4_host_free"), wrapped in a MemPool around vLLM's KV cache allocation.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" {

// torch calls this as: void* malloc(ssize_t size, int device, cudaStream_t stream)
void* dsv4_host_malloc(ssize_t size, int device, cudaStream_t stream) {
    (void)stream;
    // Bind the allocation to the right device before pinning, so the mapping lands in
    // this device's page tables rather than whichever context happens to be current.
    int prev = -1;
    cudaGetDevice(&prev);
    if (device >= 0 && device != prev) cudaSetDevice(device);

    void* p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, (size_t)size, cudaHostAllocDefault);

    if (prev >= 0 && device >= 0 && device != prev) cudaSetDevice(prev);

    if (e != cudaSuccess || p == nullptr) {
        // Returning null makes torch raise a clean OOM instead of corrupting silently.
        fprintf(stderr, "[dsv4-host-kv] cudaHostAlloc(%zd) failed on device %d: %s\n",
                size, device, cudaGetErrorString(e));
        fflush(stderr);
        return nullptr;
    }
    if (getenv("DSV4_HOST_KV_VERBOSE")) {
        fprintf(stderr, "[dsv4-host-kv] alloc %.1f MiB -> %p (device %d)\n",
                (double)size / (1024.0 * 1024.0), p, device);
        fflush(stderr);
    }
    return p;
}

// torch calls this as: void free(void* ptr, ssize_t size, int device, cudaStream_t stream)
void dsv4_host_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    if (ptr) cudaFreeHost(ptr);
}

}  // extern "C"
