#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
#include <atomic>
#include <mutex>
#include <vector>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;
typedef struct GpuEventSet GpuEventSet;

int gpu_context_create(
    uint32_t logN,
    uint32_t L,
    uint32_t dnum,
    const uint64_t *moduli,
    size_t moduli_len,
    const int *gpu_ids,
    size_t gpu_ids_len,
    size_t stream_pool_size,
    uint32_t vram_percent,
    GpuContext **out_ctx);

void gpu_context_destroy(GpuContext *ctx);
int gpu_context_fence_releases(const GpuContext *ctx);
int gpu_context_get_N(const GpuContext *ctx, int *out_N);
int gpu_context_get_vram_budget_bytes(const GpuContext *ctx, size_t *out_bytes);
int gpu_default_mempool_get_usage(
    int device,
    size_t *out_used_current_bytes,
    size_t *out_used_high_bytes,
    size_t *out_reserved_current_bytes);
int gpu_default_mempool_reset_used_high(int device);
int gpu_device_context_state(int device, size_t *out_count, uint64_t *out_generation);
int gpu_device_get_identity(
    int device,
    char *out_name,
    size_t name_capacity,
    int *out_compute_major,
    int *out_compute_minor,
    size_t *out_total_global_memory);

/// Transfers ownership of pinned host pointers to the context-owned
/// reclaimer. The reclaimer records a completion event on `stream`, waits
/// for that event on its worker thread, and only then calls cudaFreeHost.
/// A non-zero return means that ownership was retained as a fail-closed leak.
int gpu_defer_pinned_frees(
    GpuContext *ctx,
    int device,
    cudaStream_t stream,
    void *const *ptrs,
    size_t count);

int gpu_event_set_wait(GpuEventSet *events);
void gpu_event_set_destroy(GpuEventSet *events);

int gpu_device_count(int *out_count);
int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total);
int gpu_device_synchronize();
int gpu_device_reset();

const char *gpu_last_error();

void *gpu_pinned_alloc(size_t bytes);
void gpu_pinned_free(void *ptr);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
constexpr size_t GPU_RUNTIME_MAX_LIMBS = 64;
constexpr size_t GPU_RUNTIME_MAX_DIGITS = 8;

enum GpuLimbType : uint8_t
{
    GPU_LIMB_U32 = 0,
    GPU_LIMB_U64 = 1,
};

struct GpuNttDeviceConstants
{
    int device;
    size_t limb_count;
    uint32_t ring_dimension;
    uint64_t *twiddle_forward; // limb-major layout: [limb][exponent]
    uint64_t *twiddle_inverse; // limb-major layout: [limb][exponent]
    uint64_t *twiddle_shoup_forward;
    uint64_t *twiddle_shoup_inverse;
    uint64_t *moduli;
    uint64_t *n_inv;
    uint64_t *n_inv_shoup;
};

struct PinnedHostReclaimer;

struct GpuContext
{
    std::vector<uint64_t> moduli;
    std::vector<uint64_t> ntt_n_inv_by_prime;
    std::vector<uint64_t> ntt_root_by_prime;
    std::vector<uint64_t> ntt_inv_root_by_prime;
    std::vector<GpuNttDeviceConstants> ntt_device_constants;
    int N;
    int level;
    std::vector<int> gpu_ids;
    uint32_t dnum;
    size_t max_aux_limbs;
    size_t vram_budget_bytes;
    std::vector<uint64_t> garner_inverse_table;
    std::vector<dim3> limb_gpu_ids;
    std::vector<int> limb_prime_ids;
    std::vector<GpuLimbType> limb_types;
    std::vector<uint8_t> limb_coeff_bytes;
    std::vector<size_t> decomp_counts_by_partition;
    std::mutex transform_mutex;
    std::vector<std::vector<cudaStream_t>> compute_streams_by_partition;
    std::vector<cudaStream_t> release_streams_by_partition;
    std::vector<cudaEvent_t> release_fence_events_by_partition;
    PinnedHostReclaimer *pinned_host_reclaimer = nullptr;
    std::atomic<size_t> next_compute_stream{0};
};

struct GpuEventSet
{
    struct Entry
    {
        cudaEvent_t event;
        int device;
    };
    std::vector<Entry> entries;
};

extern "C" int gpu_set_last_error(const char *msg);
#endif
