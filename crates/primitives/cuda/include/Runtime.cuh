#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;
typedef struct GpuEventSet GpuEventSet;
typedef struct GpuDeviceTiming GpuDeviceTiming;

typedef enum GpuAllocationEpochBoundary {
    GPU_ALLOCATION_INITIAL_SETUP = 0,
    GPU_ALLOCATION_REFRESH = 1,
} GpuAllocationEpochBoundary;

typedef enum GpuAllocationEpochUnverified {
    GPU_ALLOCATION_EPOCH_VERIFIED = 0,
    GPU_ALLOCATION_EXTERNAL_EXCLUSIVITY_REQUIRED = 1,
    GPU_ALLOCATION_UNSUPPORTED_ACTIVITY = 2,
    GPU_ALLOCATION_HOST_ACTIVITY = 3,
    GPU_ALLOCATION_PENDING_WORK = 4,
    GPU_ALLOCATION_NONEXCLUSIVE_OWNER = 5,
    GPU_ALLOCATION_CHANGED = 6,
} GpuAllocationEpochUnverified;

typedef struct GpuAllocationEpochEvidence {
    int device;
    int boundary;
    uint64_t execution_identity;
    uint64_t context_generation;
    uint64_t owner_revision;
    uint64_t device_revision;
    size_t total_bytes;
    size_t free_bytes;
    size_t pool_reserved_bytes;
    size_t pool_used_bytes;
    size_t resident_bytes;
} GpuAllocationEpochEvidence;

// The external assertion excludes unrelated, uninstrumented default-pool users
// for this boundary; it is an operating contract, not a CUDA-detectable fact.
// Initial setup may wait on owner events. Refresh only queries readiness.
// A zero return with a nonzero reason publishes no verified evidence.
// UnsupportedActivity may expose identity/generation/revision diagnostics from
// atomic loads only; memory fields remain zero. Only Verified grants accounting.
int gpu_context_observe_allocation_epoch(
    const GpuContext *ctx, int device, GpuAllocationEpochBoundary boundary,
    int external_pool_exclusive, GpuAllocationEpochEvidence *out,
    GpuAllocationEpochUnverified *out_reason);
int gpu_context_validate_allocation_epoch(
    const GpuContext *ctx, const GpuAllocationEpochEvidence *evidence,
    int *out_current);
int gpu_context_admission_is_required(const GpuContext *ctx);

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
    const GpuContext *related_context,
    GpuContext **out_ctx);

void gpu_context_destroy(GpuContext *ctx);
int gpu_context_fence_releases(const GpuContext *ctx);
int gpu_context_record_releases(const GpuContext *ctx, GpuEventSet **out_events);
int gpu_context_query_releases(const GpuContext *ctx, const GpuEventSet *events, int *out_ready);
// Explicit benchmark boundaries only. Begin/stop enqueue event dependencies;
// elapsed waits for the recorded stops. Include every execution-owner stream,
// including streams used by related parameter contexts. The caller coordinates
// all submissions on the owner while the measurement is active.
int gpu_context_begin_device_timing(const GpuContext *ctx, GpuDeviceTiming **out_timing);
int gpu_device_timing_stop(GpuDeviceTiming *timing);
int gpu_device_timing_elapsed(
    GpuDeviceTiming *timing, int *out_devices, double *out_seconds, size_t count);
void gpu_device_timing_destroy(GpuDeviceTiming *timing);
int gpu_context_get_N(const GpuContext *ctx, int *out_N);
int gpu_context_get_vram_budget_bytes(const GpuContext *ctx, size_t *out_bytes);
uint64_t gpu_context_execution_identity(const GpuContext *ctx);
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
/// for that event on its worker thread, then recycles a managed transfer buffer
/// or calls cudaFreeHost for an external allocation.
/// A non-zero return means that ownership was retained as a fail-closed leak.
int gpu_defer_pinned_frees(
    GpuContext *ctx,
    int device,
    cudaStream_t stream,
    void *const *ptrs,
    size_t count);

/// Consumes both the event set and pinned pointer without a host wait.
int gpu_event_set_defer_pinned_free(GpuContext *ctx, GpuEventSet *events, void *pointer);

int gpu_event_set_wait(GpuEventSet *events);
void gpu_event_set_destroy(GpuEventSet *events);

int gpu_device_count(int *out_count);
int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total);
int gpu_device_synchronize();
int gpu_device_reset();

const char *gpu_last_error();

void *gpu_pinned_alloc(GpuContext *ctx, size_t bytes, size_t alignment);
int gpu_pinned_free(void *ptr);

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

struct GpuRingDeviceConstants
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
    uint64_t *garner_inverses; // immutable full-basis [source][target] table
};

struct PinnedHostReclaimer;

// One setup-owned measurement group per physical partition. The existing
// timing_active lease permits only one host observer to reuse these handles.
// Context-owned driver handles, resolved for each physical device at setup.
// These are borrowed CUDA module functions; the runtime owns their lifetime.
struct GpuKernelPartition
{
    void *launch_entry = nullptr;
    cudaFunction_t tensor_row_sum[2]{};
};

// Called on the selected device during initial context provisioning, before
// accepting a physical budget. Loads the finite compiled kernel inventory;
// does not certify queued launch or remaining worker resource bounds.
int gpu_matrix_prepare_kernels(GpuKernelPartition *partition);

struct GpuDeviceTimingPartition
{
    int device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<cudaStream_t> participants;
    std::vector<cudaEvent_t> before;
    std::vector<cudaEvent_t> after;
};

// Related rings share execution resources explicitly, while all CRT and NTT
// metadata below remain ring-specific. Independent executions stay isolated.
struct GpuExecutionOwner
{
    uint64_t identity = 0;
    std::vector<int> gpu_ids;
    size_t vram_budget_bytes = 0;
    uint32_t vram_percent = 0;
    bool registered = false;
    std::vector<std::vector<cudaStream_t>> compute_streams_by_partition;
    std::vector<cudaStream_t> release_streams_by_partition;
    // One immutable stream assignment per setup-created fence event. Concurrent
    // fences on that stream can only extend its captured completion boundary.
    std::vector<std::vector<cudaEvent_t>> completion_events_by_partition;
    std::vector<GpuDeviceTimingPartition> timing_partitions;
    std::vector<GpuKernelPartition> kernel_partitions;
    PinnedHostReclaimer *pinned_host_reclaimer = nullptr;
    std::atomic<size_t> next_compute_stream{0};
    // A failed free cannot be reclaimed by admission, even if a later event
    // completes. Retain this failure for the lifetime of the execution owner.
    std::atomic<bool> memory_release_failed{false};
    // CUDA could not establish dependencies for already submitted work. Keep
    // every related context/allocation alive instead of freeing unknown readers.
    std::atomic<bool> unretired_work{false};
    // Measurement boundaries may not overlap on one execution owner. This is
    // instrumentation bookkeeping only; ordinary submission takes no lock.
    std::atomic<bool> timing_active{false};
    // Tracking is certified only after the supported admission surface rejects
    // uninstrumented entrypoints. Counter equality alone never enables it.
    std::atomic<bool> admission_required{false};
    // A separate resource domain: ordinary prepared storage alone never claims
    // coverage of host transfer buffers. A successful pinned reservation makes
    // later managed pinned allocations require their own exact prepared claim.
    std::atomic<bool> pinned_admission_required{false};
    // Transfer preparation is independent of arithmetic preparation. A context
    // that has not admitted this domain remains an unsealed standalone caller;
    // it cannot certify complete invocation coverage from arithmetic slots.
    std::atomic<bool> transfer_admission_required{false};
    std::atomic<bool> resource_admission_required{false};
    // Enabled only by complete-inventory setup after all audited managed
    // domains are closed. This does not bound opaque CUDA consumption.
    std::atomic<bool> allocation_tracking_complete{false};
    std::atomic<bool> allocation_activity_unknown{false};
    std::atomic<uint64_t> allocation_revision{0};
    std::atomic<size_t> active_allocation_calls{0};
    // One logical device-span domain across prepared stores of related rings.
    // These are actual claims, not reservations or physical CUDA residency.
    std::atomic<size_t> prepared_storage_count{0};
    std::atomic<size_t> prepared_active_reservations{0};
    std::atomic<size_t> prepared_occupied_bytes{0};
    std::atomic<size_t> prepared_high_water_bytes{0};
    ~GpuExecutionOwner();
};

// Covers the complete host submission/resource-mutation scope, including error
// cleanup. Concurrent writers never wait for one another. A null owner tracks
// device-only resources (construction and thread-local CUDA events). Device -1
// covers every device of a nonnull owner; the owner must outlive the scope.
struct GpuAllocationActivity
{
    GpuAllocationActivity(GpuExecutionOwner *owner, int device);
    ~GpuAllocationActivity();
    GpuAllocationActivity(const GpuAllocationActivity &) = delete;
    GpuAllocationActivity &operator=(const GpuAllocationActivity &) = delete;
private:
    GpuExecutionOwner *owner_;
    int device_;
};

void gpu_execution_mark_allocation_unknown(GpuExecutionOwner *owner);
// Resource cleanup can outlive a context (for example, a thread-local event).
// Preserve that uncertainty for every later observation of the same device.
void gpu_device_mark_allocation_unknown(int device);

struct GpuBarrettReciprocal
{
    uint64_t lo;
    uint64_t hi;
};

struct GpuContext
{
    std::vector<uint64_t> moduli;
    std::vector<GpuBarrettReciprocal> barrett_reciprocals;
    std::vector<uint64_t> ntt_n_inv_by_prime;
    std::vector<uint64_t> ntt_root_by_prime;
    std::vector<uint64_t> ntt_inv_root_by_prime;
    std::vector<GpuRingDeviceConstants> ring_device_constants;
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
    std::shared_ptr<GpuExecutionOwner> execution;
};

struct GpuCudaResource;

struct GpuEventSet
{
    struct Entry
    {
        cudaEvent_t event;
        int device;
        std::shared_ptr<GpuCudaResource> resource;
    };
    // Event sets may outlive their parameter wrapper. Retain attribution until
    // the last event is retired, including destruction of this owner reference.
    std::shared_ptr<GpuExecutionOwner> execution;
    std::vector<Entry> entries;
};

// Error-path retirement only: join this submitted stream into the execution
// owner's compute and release streams using one event, without waiting on the host.
extern "C" int gpu_context_retire_stream(const GpuContext *ctx, int device, cudaStream_t stream);

extern "C" int gpu_set_last_error(const char *msg);
#endif
