#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
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
    uint32_t batch,
    GpuContext **out_ctx);

void gpu_context_destroy(GpuContext *ctx);
int gpu_context_get_N(const GpuContext *ctx, int *out_N);

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

struct GpuContext
{
    std::vector<uint64_t> moduli;
    std::vector<uint64_t> ntt_n_inv_by_prime;
    std::vector<uint64_t> ntt_root_by_prime;
    std::vector<uint64_t> ntt_inv_root_by_prime;
    int N;
    int level;
    std::vector<int> gpu_ids;
    uint32_t batch;
    uint32_t dnum;
    size_t max_aux_limbs;
    std::vector<uint64_t> garner_inverse_table;
    std::vector<dim3> limb_gpu_ids;
    std::vector<int> limb_prime_ids;
    std::vector<GpuLimbType> limb_types;
    std::vector<size_t> decomp_counts_by_partition;
    std::mutex transform_mutex;
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
