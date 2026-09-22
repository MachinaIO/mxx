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

// Status returned by the CUDA C ABI for a device allocation that cannot be
// satisfied.  Keep this distinct from the generic failure status so Rust can
// classify allocation failure without inspecting the human-readable message.
#define GPU_STATUS_OUT_OF_MEMORY 2
#define GPU_STATUS_CONDITIONAL_UNSUPPORTED 3
// Graph launch returned after submission was attempted but stream completion
// could not be established. Callers must retain all bound owners and treat
// this as a distinct fail-closed outcome; it is never safe to infer that no
// device work was queued.
#define GPU_STATUS_LAUNCH_UNCERTAIN 4

typedef struct GpuContext GpuContext;
typedef struct GpuEventSet GpuEventSet;
typedef struct MxxGpuGraphCapture MxxGpuGraphCapture;
typedef struct MxxGpuGraphBodyCapture MxxGpuGraphBodyCapture;
typedef struct MxxGpuGraphExec MxxGpuGraphExec;
typedef struct MxxGpuNativeEvent MxxGpuNativeEvent;
typedef struct MxxGpuCaptureEvent MxxGpuCaptureEvent;
typedef struct MxxGpuDeviceBuffer MxxGpuDeviceBuffer;

// Transfer an allocation released during capture to the executable lifetime.
// Returns one when ownership was transferred, zero outside capture.
int mxx_graph_retain_released_resource(GpuContext *ctx, void *resource, void (*destroy)(void *));
int mxx_gpu_graph_capture_resolve_fixed_addresses(MxxGpuGraphCapture *capture);

#ifdef __cplusplus
// The owner layout is shared by Runtime.cu and allocation-free native
// primitives.  Keeping it in the private CUDA header lets control kernels
// enqueue work while preserving the owner's producer event without exposing
// CUDA types across the Rust ABI.
struct MxxGpuDeviceBuffer
{
    int device = -1;
    cudaStream_t allocation_stream = nullptr;
    uint8_t *address = nullptr;
    size_t bytes = 0;
    cudaEvent_t producer = nullptr;
    bool producer_valid = false;
};
#endif

// Stable sampler control/status records shared by the runtime adapter and
// matrix launch wrappers. Keeping these in the base runtime header avoids a
// dependency from Runtime.cu onto MatrixUtils' error helpers.
typedef struct MxxPreimageStatus
{
    uint32_t attempts;
    uint32_t accepted;
    uint32_t error_code;
    uint32_t reserved;
} MxxPreimageStatus;

typedef struct MxxPreimageLaunchControl
{
    uint8_t execution_nonce[32];
    uint64_t logical_instance;
    uint64_t global_column_start;
    uint32_t max_attempts;
    union
    {
        uint32_t attempt;
        uint32_t reserved;
    };
    uint64_t binding_table;
} MxxPreimageLaunchControl;

enum MxxPreimageErrorCode : uint32_t
{
    MXX_PREIMAGE_SUCCESS = 0,
    MXX_PREIMAGE_EXHAUSTED = 1,
};

// Static metadata for a sampler-owned conditional retry body. The sampler
// supplies fixed scratch/control/status storage and allocation-free kernels.
typedef struct MxxPreimageRetrySpec
{
    uint32_t max_attempts;
    uint32_t attempt_binding_index;
    uint32_t control_binding_index;
    uint32_t status_binding_index;
    uint32_t reserved;
} MxxPreimageRetrySpec;

// Flat values are the only values that cross the primitives/runtime graph
// boundary.  The native implementation validates the width of every value
// against the patch record before changing a graph node.
enum MxxGraphBindingKind
{
    MXX_GRAPH_BINDING_DEVICE_ADDRESS = 0,
    MXX_GRAPH_BINDING_U64 = 1,
    MXX_GRAPH_BINDING_I64 = 2,
    MXX_GRAPH_BINDING_BYTES32 = 3,
};

struct MxxGraphBindingValue
{
    uint32_t kind;
    uint32_t byte_count;
    uint8_t bytes[32];
};

// A capture-local launch schema may use compact operation-local binding IDs
// while the runtime replay vector uses arbitrary region-global IDs.  The
// mapping is explicit and immutable for the current launch-registration
// scope; native capture never infers identities from addresses.
struct MxxGraphBindingMapEntry
{
    uint32_t local_binding;
    uint32_t global_binding;
};

struct MxxGraphMemorySnapshot
{
    uint64_t used_current;
    uint64_t used_high;
    uint64_t reserved_current;
    uint64_t reserved_high;
};

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
int gpu_context_get_N(const GpuContext *ctx, int *out_N);
int gpu_context_get_vram_budget_bytes(const GpuContext *ctx, size_t *out_bytes);
int gpu_context_get_compute_stream(const GpuContext *ctx, int physical_device, void **out_stream);
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
    char *out_uuid,
    size_t uuid_capacity,
    int *out_compute_major,
    int *out_compute_minor,
    size_t *out_total_global_memory,
    int *out_driver_version,
    int *out_runtime_version,
    uint64_t *out_context_generation);

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

// Store a CUDA runtime error and return its stable ABI status.  In particular,
// cudaErrorMemoryAllocation maps to GPU_STATUS_OUT_OF_MEMORY; all other CUDA
// errors remain generic failures.
int gpu_set_last_error_cuda(int cuda_error);

void *gpu_pinned_alloc(size_t bytes);
void gpu_pinned_free(void *ptr);

// Stream-ordered device buffer owners used by allocation-free graph bodies.
// The opaque owner retains the allocation and its producer completion event;
// callers may create borrowed interior views without duplicating ownership.
int gpu_device_buffer_alloc(void *stream, size_t bytes, MxxGpuDeviceBuffer **out);
int gpu_device_buffer_address(
    const MxxGpuDeviceBuffer *buffer,
    size_t offset,
    size_t bytes,
    void **out_address);
int gpu_device_buffer_free(MxxGpuDeviceBuffer *buffer);
int gpu_device_buffer_upload(
    MxxGpuDeviceBuffer *buffer,
    size_t offset,
    const void *source,
    size_t bytes);
int gpu_device_buffer_download(
    const MxxGpuDeviceBuffer *buffer,
    size_t offset,
    void *destination,
    size_t bytes);
int gpu_device_buffer_wait_compiled_inputs(
    const MxxGpuDeviceBuffer *buffer,
    int consumer_device,
    void *consumer_stream,
    bool read_only);
int gpu_device_buffer_track_compiled_consumer(
    const MxxGpuDeviceBuffer *buffer,
    int consumer_device,
    void *consumer_stream,
    void *completion_event,
    bool read_only);
int gpu_device_buffer_record_compiled_write(
    MxxGpuDeviceBuffer *buffer,
    void *stream);
int gpu_device_buffer_wait(const MxxGpuDeviceBuffer *buffer);
int gpu_device_buffer_prepare_external_for_capture(MxxGpuDeviceBuffer *buffer);
int gpu_device_buffer_gather_u64(
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t source_count,
    const MxxGpuDeviceBuffer *indices,
    size_t indices_offset,
    size_t index_count,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    void *stream);
// Enqueue a device-to-device range copy on the caller-provided stream.  The
// destination producer event is updated so later owners retain the copy
// dependency.  Different physical devices use CUDA peer copy; no host
// staging is permitted by this API.
int gpu_device_buffer_copy_range(
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    size_t bytes,
    void *stream);


// CUDA graph capture/replay is deliberately a flat primitives-owned API.  A
// capture consumes its opaque handle on both success and failure; callers must
// not use the handle after finish/abort.  The launch stream is a non-owning
// view into the existing execution owner and is represented as void* in this
// C ABI so Rust never needs to know CUDA's stream handle layout.
int mxx_gpu_graph_capture_begin(
    GpuContext *ctx,
    int physical_device,
    void *stream,
    MxxGpuGraphCapture **out_capture);
int mxx_gpu_graph_capture_stream(
    MxxGpuGraphCapture *capture,
    void **out_stream);
int mxx_gpu_graph_capture_finish(
    MxxGpuGraphCapture *capture,
    MxxGpuGraphExec **out_exec);
int mxx_gpu_graph_capture_abort(MxxGpuGraphCapture *capture);
// Claim a contiguous region-global binding range before lowering one
// operation. The range is stable for the lifetime of the capture and is used
// to translate primitive-local launch-site patch indices into the frozen
// runtime schema.
int mxx_gpu_graph_capture_claim_binding_range(
    MxxGpuGraphCapture *capture,
    uint32_t count,
    uint32_t *out_offset);
int mxx_gpu_graph_capture_set_binding_offset(
    MxxGpuGraphCapture *capture,
    uint32_t offset);
int mxx_gpu_graph_capture_set_binding_map(
    MxxGpuGraphCapture *capture,
    const MxxGraphBindingMapEntry *entries,
    size_t entry_count);
// Register an immutable schema identity. Re-registering the exact
// (binding,address,bytes) triple is idempotent; reusing a binding ID for a
// different range fails. Different IDs may alias the same address range.
int mxx_gpu_graph_capture_bind_resident_address(
    MxxGpuGraphCapture *capture, uint64_t address, size_t bytes, uint32_t binding);
bool mxx_gpu_graph_control_status_needs_reset(GpuContext *ctx, void *stream, void *status);
int mxx_gpu_graph_body_capture_begin(
    MxxGpuGraphCapture *capture,
    MxxGpuGraphBodyCapture **out_body);
int mxx_gpu_graph_body_capture_stream(
    MxxGpuGraphBodyCapture *body,
    void **out_stream);
int mxx_gpu_graph_body_capture_finish(
    MxxGpuGraphBodyCapture *body,
    void **out_graph);
int mxx_gpu_graph_body_capture_abort(MxxGpuGraphBodyCapture *body);
void mxx_gpu_graph_body_destroy(void *graph);
int mxx_gpu_graph_upload(MxxGpuGraphExec *exec, void *launch_stream);
int mxx_gpu_graph_bind(
    MxxGpuGraphExec *exec,
    const MxxGraphBindingValue *values,
    size_t count);
int mxx_gpu_graph_launch(
    MxxGpuGraphExec *exec,
    void *launch_stream,
    MxxGpuNativeEvent **out_event);
int mxx_gpu_graph_add_preimage_retry_body(
    MxxGpuGraphCapture *capture,
    const MxxPreimageRetrySpec *spec,
    void *fixed_scratch,
    void *device_control,
    void *device_status,
    void *body_graph);
void mxx_gpu_graph_exec_destroy(MxxGpuGraphExec *exec);

int mxx_gpu_native_event_wait(MxxGpuNativeEvent *event);
int mxx_gpu_native_event_enqueue_wait(
    MxxGpuNativeEvent *event,
    void *stream);
// Expose the event handle only to primitives-owned native lifetime adapters.
// The opaque event remains owned by the Rust wrapper and must not be destroyed
// by the caller of this accessor.
int mxx_gpu_native_event_raw(
    MxxGpuNativeEvent *event,
    void **out_event);
int mxx_gpu_native_event_query(MxxGpuNativeEvent *event, int *out_complete);
void mxx_gpu_native_event_destroy(MxxGpuNativeEvent *event);

int mxx_gpu_graph_memory_snapshot(
    int physical_device,
    MxxGraphMemorySnapshot *out_snapshot);

// These registration functions are intentionally native/internal.  Existing
// kernel launch sites provide exact argument layouts and captured node handles;
// replay never searches graph bytes or infers nodes from their order.
enum MxxGraphPatchTarget
{
    MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD = 0,
    MXX_GRAPH_PATCH_MEMCPY_1D_SRC = 1,
    MXX_GRAPH_PATCH_MEMCPY_1D_DST = 2,
    MXX_GRAPH_PATCH_MEMSET_1D_DST = 3,
    MXX_GRAPH_PATCH_INTEGER_ENCODING = 4,
};

struct MxxGraphPatch
{
    void *node;
    uint32_t target;
    uint32_t argument_index;
    uint32_t byte_offset;
    uint32_t byte_count;
    uint32_t binding_index;
    uint64_t address_addend;
};

int mxx_graph_register_kernel_update(
    MxxGpuGraphCapture *capture,
    void *node,
    const void *launch,
    const size_t *argument_sizes,
    size_t argument_count,
    const MxxGraphPatch *patches,
    size_t patch_count);
// Launch-site adapter: the caller invokes this immediately after a kernel
// launch on the supplied stream.  Native capture introspection resolves the
// just-created top-level kernel node and records its exact argument layout;
// no graph scan or node-order inference is performed at replay.
int mxx_graph_register_kernel_update_for_stream(
    GpuContext *ctx,
    void *stream,
    const size_t *argument_sizes,
    size_t argument_count,
    const MxxGraphPatch *patches,
    size_t patch_count);
int mxx_graph_register_resident_descriptor_for_stream(
    GpuContext *ctx, void *stream, const size_t *argument_sizes, size_t argument_count);
int mxx_graph_register_memcpy1d_update(
    MxxGpuGraphCapture *capture,
    void *node,
    size_t fixed_bytes,
    int fixed_kind,
    const MxxGraphPatch *patches,
    size_t patch_count);
int mxx_graph_register_memcpy1d_update_for_stream(
    GpuContext *ctx,
    void *stream,
    size_t fixed_bytes,
    int fixed_kind,
    const MxxGraphPatch *patches,
    size_t patch_count);
int mxx_graph_register_memset1d_update(
    MxxGpuGraphCapture *capture,
    void *node,
    const void *fixed_params,
    const MxxGraphPatch *destination_patch);
int mxx_graph_register_memset1d_update_for_stream(
    GpuContext *ctx,
    void *stream,
    const void *fixed_params,
    const MxxGraphPatch *destination_patch);

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
    PinnedHostReclaimer *pinned_host_reclaimer = nullptr;
    std::atomic<size_t> next_compute_stream{0};
    // During planning capture, matrix dispatch is temporarily routed to the
    // origin stream. This keeps every internal compute/release dependency in
    // one captured stream while preserving the ordinary pool after finish or
    // abort. Access is guarded because capture orchestration and dispatch may
    // run on different host threads.
    std::mutex capture_mutex;
    cudaStream_t capture_stream = nullptr;
    int capture_device = -1;
    bool capture_active = false;
    uint64_t capture_generation = 0;
    // While a sampler lowers a conditional body, matrix launch sites route to
    // this separate stream so candidate generation is captured into the child
    // graph rather than accidentally appended to the root capture.
    cudaStream_t capture_body_stream = nullptr;
    int capture_body_device = -1;
    bool capture_body_active = false;
    // Non-owning pointer to the active conditional-body capture.  Launch
    // adapters use this to register body-node patches against the child graph
    // instead of silently treating the body stream as an ordinary stream.
    MxxGpuGraphBodyCapture *capture_body_handle = nullptr;
    // Non-owning pointer to the active capture record. It is valid only while
    // capture_active is true and is used by launch-site registration helpers.
    MxxGpuGraphCapture *capture_handle = nullptr;
    ~GpuExecutionOwner();
};

// An event recorded on a CUDA graph-capture stream has two owners: the
// matrix state that records the event and the graph executable that retains
// the corresponding event node.  Keep that lifetime explicit so a capture
// event is never exposed as a replay completion event or destroyed while its
// graph still contains the node.
#ifdef __cplusplus
struct MxxGpuCaptureEvent
{
    cudaEvent_t event = nullptr;
    uint64_t capture_generation = 0;
    size_t references = 1;
};

MxxGpuCaptureEvent *mxx_gpu_capture_event_create(GpuContext *ctx, int device);
void mxx_gpu_capture_event_release(MxxGpuCaptureEvent *event);
#endif

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
    std::shared_ptr<GpuExecutionOwner> execution;
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
