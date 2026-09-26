#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "SubgraphKernel.cuh"

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
typedef struct MxxGpuGraphExec MxxGpuGraphExec;
typedef struct MxxGpuGraphBuilder MxxGpuGraphBuilder;
struct MxxGraphPatch;
typedef struct MxxGpuNativeEvent MxxGpuNativeEvent;
typedef struct MxxGpuDeviceBuffer MxxGpuDeviceBuffer;
typedef struct GpuModulusConversionPlan GpuModulusConversionPlan;
typedef struct GpuRawPreimageCutoffPlan GpuRawPreimageCutoffPlan;
typedef struct GpuIndexedMatrixTable GpuIndexedMatrixTable;

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
    MXX_PREIMAGE_INVALID_ATTEMPT = 2,
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

int gpu_context_create(
    uint32_t logN,
    uint32_t L,
    uint32_t dnum,
    const uint64_t *moduli,
    size_t moduli_len,
    const int *gpu_ids,
    size_t gpu_ids_len,
    size_t stream_pool_size,
    const GpuContext *related_context,
    GpuContext **out_ctx);

void gpu_context_destroy(GpuContext *ctx);
int gpu_context_fence_releases(const GpuContext *ctx);
int gpu_context_get_N(const GpuContext *ctx, int *out_N);
int gpu_context_get_compute_stream(const GpuContext *ctx, int physical_device, void **out_stream);
uint64_t gpu_context_execution_identity(const GpuContext *ctx);
int gpu_default_mempool_get_usage(
    int device,
    size_t *out_used_current_bytes,
    size_t *out_used_high_bytes,
    size_t *out_reserved_current_bytes);

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

// Logical device ids name the devices of a GPU fleet. They map to physical
// CUDA devices through one process-wide table, the identity until configured;
// several logical devices may share one physical device. Configure the table
// once, before any context or allocation exists.
int gpu_configure_logical_devices(const int *physical, size_t count);
int mxx_physical_device(int logical);
cudaError_t mxx_set_device(int logical);
cudaError_t mxx_get_device(int *logical);

int gpu_device_count(int *out_count);
int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total);
int gpu_device_synchronize();

const char *gpu_last_error();

// Store a CUDA runtime error and return its stable ABI status.  In particular,
// cudaErrorMemoryAllocation maps to GPU_STATUS_OUT_OF_MEMORY; all other CUDA
// errors remain generic failures.
int gpu_set_last_error_cuda(int cuda_error);

void *gpu_pinned_alloc(size_t bytes);
void gpu_pinned_free(void *ptr);
// One-shot mapped export slot. The CPU may read metadata and payload only
// after gpu_export_slot_ready observes the device's system-scope publication.
struct alignas(8) MxxExportSlotHeader
{
    uint64_t ready;
    uint64_t occurrence;
    uint64_t artifact_offset;
    uint64_t payload_bytes;
    uint32_t site;
    uint32_t flags;
};
int gpu_export_slot_alloc(int physical_device, size_t payload_capacity,
    void **out_host, void **out_device);
int gpu_export_slot_ready(const void *host_header, int *out_ready);
int gpu_export_slot_reset(void *host_header);

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

int gpu_device_buffer_wait(const MxxGpuDeviceBuffer *buffer);

// Blocking D2H read of device memory after the runtime has validated its
// owner/view bounds and awaited its producer completion event.
int gpu_context_download_address(
    GpuContext *ctx, int physical_device, const void *address,
    void *destination, size_t bytes);

struct MxxRawSmallMatrixView
{
    uint64_t payload_address;
    int32_t physical_device;
    uint32_t degree;
    uint64_t rows;
    uint64_t columns;
    uint64_t storage_columns;
    uint64_t column_offset;
    uint32_t magnitude_bytes;
    uint32_t bound_domain;
    uint32_t crt_depth;
    uint32_t reserved;
};
// The NTT tables of each limb of `view`, which must be on `physical_device`.
int gpu_context_ntt_tables(GpuContext *ctx, const MxxRawMatrixView *view,
    MxxNttTables *out_tables);
int gpu_raw_matrix_ntt(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    int inverse, uint32_t source_binding_base, uint32_t destination_binding_base);
int gpu_raw_matrix_add_sub(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination, int subtract,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_matrix_mul(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination, int accumulate,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_matrix_mul_transpose_rhs(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_matrix_transpose(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base);
int gpu_raw_matrix_tensor(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_matrix_copy(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *sources, const MxxRawMatrixView *destinations, size_t count,
    const uint32_t *source_binding_bases, const uint32_t *destination_binding_bases);
int gpu_raw_matrix_scale(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const uint64_t *scalar_residues, size_t residue_count,
    uint32_t source_binding_base, uint32_t destination_binding_base);
int gpu_raw_matrix_scale_dynamic(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *scalar, int scalar_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t scalar_binding, uint32_t status_binding);
int gpu_raw_ring_automorphism(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *index_value, int index_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t index_binding, uint32_t status_binding);
int gpu_raw_lift_integer_constant(GpuContext *ctx, void *stream,
    const void *value, int value_encoding,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t value_binding, uint32_t destination_binding_base,
    uint32_t status_binding);
int gpu_indexed_matrix_table_create(GpuContext *ctx, void *stream,
    int32_t physical_device, size_t family_count, size_t limb_count,
    GpuIndexedMatrixTable **out_table);
int gpu_indexed_matrix_table_upload(GpuIndexedMatrixTable *table, void *stream,
    const MxxRawMatrixLimb *limbs, size_t limb_count);
uint64_t gpu_indexed_matrix_table_address(const GpuIndexedMatrixTable *table);
void gpu_indexed_matrix_table_destroy(GpuIndexedMatrixTable *table);
int gpu_raw_matrix_indexed_copy(GpuContext *ctx, void *stream,
    const void *index_address, int index_encoding,
    const GpuIndexedMatrixTable *table, const MxxRawMatrixView *destination,
    uint32_t *status, uint32_t index_binding,
    uint32_t destination_binding_base, uint32_t status_binding);
int gpu_raw_preimage_derive_attempt_seed(GpuContext *ctx, void *stream,
    const uint8_t *base_seed, const uint64_t *attempt, uint64_t domain,
    uint8_t *derived_seed, uint32_t base_binding,
    uint32_t attempt_binding, uint32_t derived_binding);
int gpu_raw_rns_conversion_prepare(GpuContext *ctx, int32_t physical_device,
    void *stream, const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count, size_t digit_size,
    int normalize, const uint64_t *plaintext_words, size_t plaintext_word_count,
    GpuModulusConversionPlan **out_plan);
int gpu_raw_block_mod_switch_prepare(GpuContext *ctx, int32_t physical_device,
    void *stream, const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    const uint64_t *plaintext_words, size_t word_count,
    GpuModulusConversionPlan **out_plan);
int gpu_raw_conversion_plan_wait(const GpuModulusConversionPlan *plan,
    void *stream);
int gpu_raw_rns_conversion_emit(GpuModulusConversionPlan *plan,
    GpuContext *ctx, void *stream, const MxxRawMatrixView *source,
    const MxxRawMatrixView *destination, uint32_t source_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_block_mod_switch_emit(GpuModulusConversionPlan *plan,
    GpuContext *ctx, void *stream, const MxxRawMatrixView *source,
    const MxxRawMatrixView *destination, uint32_t source_binding_base,
    uint32_t destination_binding_base);
int gpu_raw_matrix_identity_fill(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *destination, uint64_t square_size,
    uint64_t column_base, uint32_t destination_binding_base);
int gpu_raw_matrix_gadget_fill(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *destination, uint64_t rows,
    uint32_t digits_per_tower, const uint64_t *base_residues,
    uint64_t column_base, uint32_t destination_binding_base);
int gpu_raw_p1_covariance_refresh(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *a, const MxxRawMatrixView *b,
    const MxxRawMatrixView *d, double sigma, double s,
    double dgg_stddev, void *cov_workspace, double *sqrt_var,
    double *update_coeff, uint32_t a_binding, uint32_t b_binding,
    uint32_t d_binding, uint32_t cov_binding, uint32_t sqrt_binding,
    uint32_t update_binding);
int gpu_raw_p1_sample(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *tp2, const MxxRawMatrixView *output,
    const void *seed, int64_t *sampled,
    size_t sampled_bytes, void *workspace, size_t workspace_bytes,
    const double *sqrt_var, const double *update_coeff,
    double sigma, double s, uint32_t tp2_binding,
    uint32_t output_binding_base, uint32_t seed_binding,
    uint32_t sampled_binding, uint32_t workspace_binding,
    uint32_t sqrt_binding, uint32_t update_binding);
int gpu_raw_gq_sample(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *seed, int64_t *sampled, size_t sampled_bytes,
    uint32_t base_bits, double c, uint32_t source_binding_base,
    uint32_t destination_binding_base, uint32_t seed_binding,
    uint32_t sampled_binding);
int gpu_raw_preimage_cutoff_prepare(GpuContext *ctx,
    int32_t physical_device, void *stream, const uint64_t *bound_words,
    size_t bound_word_count, uint32_t magnitude_bytes,
    GpuRawPreimageCutoffPlan **out_plan);
int gpu_raw_preimage_cutoff_plan_wait(
    const GpuRawPreimageCutoffPlan *plan, void *consumer_stream);
int gpu_raw_preimage_cutoff_metadata_ranges(
    const GpuRawPreimageCutoffPlan *plan, uint64_t *addresses,
    size_t *bytes, size_t capacity, size_t *out_count);
void gpu_raw_preimage_cutoff_destroy(GpuRawPreimageCutoffPlan *plan);
int gpu_raw_preimage_add_correction(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *candidate_eval,
    const MxxRawMatrixView *r_eval, const MxxRawMatrixView *e_eval,
    const MxxRawMatrixView *z_eval, uint32_t candidate_binding_base,
    uint32_t r_binding_base, uint32_t e_binding_base,
    uint32_t z_binding_base);
int gpu_raw_preimage_hard_cutoff(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *candidate_coeff,
    const GpuRawPreimageCutoffPlan *plan, void *staging,
    size_t staging_bytes, const uint64_t *attempt,
    MxxPreimageStatus *status, uint32_t candidate_binding_base,
    uint32_t staging_binding, uint32_t control_binding,
    uint32_t status_binding);
int gpu_raw_preimage_publish_accepted(GpuContext *ctx, void *stream,
    const MxxRawSmallMatrixView *destination,
    const GpuRawPreimageCutoffPlan *plan, const void *staging,
    size_t staging_bytes, const MxxPreimageStatus *status,
    uint32_t destination_binding, uint32_t staging_binding,
    uint32_t status_binding);
int gpu_raw_matrix_decompose_coeff(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t base_bits, size_t dropped_moduli,
    uint32_t source_binding_base, uint32_t destination_binding_base);
int gpu_raw_matrix_decompose_compact(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawSmallMatrixView *destination,
    uint32_t base_bits, size_t dropped_moduli, int full_basis_small,
    uint32_t source_binding_base, uint32_t destination_binding);
int gpu_raw_modulus_conversion_prepare(
    GpuContext *ctx, int physical_device, void *stream,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    int round_scale, GpuModulusConversionPlan **out_plan);
int gpu_raw_centered_rebase_prepare(
    GpuContext *ctx, int physical_device, void *stream,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    GpuModulusConversionPlan **out_plan);
int gpu_raw_centered_round_divide_prepare(
    GpuContext *ctx, int physical_device, void *stream,
    const uint64_t *moduli, size_t modulus_count,
    const uint64_t *divisor_words, size_t divisor_word_count,
    GpuModulusConversionPlan **out_plan);
int gpu_raw_crt_recompose_level_prepare(
    GpuContext *ctx, int32_t physical_device, void *stream,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    const uint64_t *plaintext_words, size_t plaintext_count,
    const uint64_t *reconstruction_residues, size_t reconstruction_count,
    GpuModulusConversionPlan **out_plan);
int gpu_raw_crt_recompose_level_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *target,
    int initialize, uint32_t source_binding_base,
    uint32_t target_binding_base);
int gpu_raw_compact_pack_prepare(
    GpuContext *ctx, int32_t physical_device, void *stream,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *bound_words, size_t bound_word_count,
    uint32_t magnitude_bytes, GpuModulusConversionPlan **out_plan);
int gpu_raw_compact_pack_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source,
    const MxxRawSmallMatrixView *destination, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding,
    uint32_t status_binding);
int gpu_raw_matrix_dynamic_slice(
    GpuContext *ctx, void *stream, const MxxRawMatrixView *source,
    const MxxRawMatrixView *destination,
    const void *row_start, int row_start_encoding,
    const void *row_end, int row_end_encoding,
    const void *column_start, int column_start_encoding,
    const void *column_end, int column_end_encoding,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t destination_binding_base, uint32_t row_start_binding,
    uint32_t row_end_binding, uint32_t column_start_binding,
    uint32_t column_end_binding, uint32_t status_binding);
int gpu_raw_modulus_conversion_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base);
int gpu_raw_centered_round_divide_dynamic_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *divisor, int divisor_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t divisor_binding, uint32_t status_binding);
int gpu_raw_matrix_sample(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *destination, int distribution, double sigma,
    uint64_t max_coefficient_bound, uint64_t coefficient_modulus,
    int64_t interval_minimum, int64_t interval_maximum,
    const void *device_seed, uint64_t full_columns, uint64_t sample_domain,
    uint32_t destination_binding_base, uint32_t seed_binding);
int gpu_raw_polynomial_from_values(GpuContext *ctx, void *stream,
    const uint64_t *source, size_t source_count, size_t magnitude_words,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t source_binding, uint32_t destination_binding_base,
    uint32_t status_binding);
int gpu_raw_matrix_mul_scalar(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *matrix, const MxxRawMatrixView *scalar,
    const MxxRawMatrixView *destination, uint32_t matrix_binding_base,
    uint32_t scalar_binding_base, uint32_t destination_binding_base);
int gpu_raw_matrix_mul_small_rhs(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *left, const MxxRawSmallMatrixView *right,
    const MxxRawMatrixView *workspace, const MxxRawMatrixView *destination,
    const MxxRawMatrixView *addend,
    uint32_t left_binding_base, uint32_t right_binding,
    uint32_t workspace_binding_base, uint32_t destination_binding_base,
    uint32_t addend_binding_base);
int gpu_raw_small_rhs_expand(GpuContext *ctx, void *stream,
    const MxxRawSmallMatrixView *source,
    const MxxRawMatrixView *destination,
    uint32_t source_binding, uint32_t destination_binding_base);

// Explicit graph construction. Operation predecessors are terminal tokens
// returned by finish_operation; each operation's internal nodes are ordered.
int mxx_gpu_graph_builder_create(GpuContext *ctx, int physical_device,
    void *stream, MxxGpuGraphBuilder **out_builder);
int gpu_device_release_cached_memory(const GpuContext *ctx, int device);
int gpu_device_graph_memory_reserved(int device, size_t *out_reserved_bytes);
int gpu_graph_allocation_free_async(uint64_t address, void *stream);
int mxx_gpu_graph_builder_add_memory_alloc(MxxGpuGraphBuilder *builder, int device,
    size_t bytes, const uint32_t *after, size_t after_count, uint32_t *out_token,
    uint64_t *out_address);
int mxx_gpu_graph_builder_add_memory_free(MxxGpuGraphBuilder *builder, uint64_t address,
    const uint32_t *operations, size_t operation_count, const uint32_t *after,
    size_t after_count, uint32_t *out_token);
int mxx_gpu_graph_builder_set_pending_memory_dependencies(MxxGpuGraphBuilder *builder,
    const uint32_t *tokens, size_t count);
int mxx_gpu_graph_builder_begin_operation(MxxGpuGraphBuilder *builder,
    uint32_t operation_index, const uint32_t *predecessors, size_t predecessor_count);
int mxx_gpu_graph_builder_finish_operation(MxxGpuGraphBuilder *builder,
    uint32_t *out_terminal);
int mxx_gpu_graph_builder_bind_resident_address(MxxGpuGraphBuilder *builder,
    uint64_t address, size_t bytes, uint32_t binding);

int mxx_gpu_graph_builder_add_kernel(MxxGpuGraphBuilder *builder, const void *function,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    size_t shared_bytes, const void *const *arguments, const size_t *argument_sizes,
    size_t argument_count, const MxxGraphPatch *patches, size_t patch_count);
// Copy between resident allocations of two (logical) devices. The copy is
// one node when both are on one physical GPU or the destination GPU can
// access the source GPU's memory; otherwise it is staged through a pinned
// host buffer the executable owns, as a device-to-host and a host-to-device
// node. Staged copies between one GPU pair in one graph alternate between two
// buffers, each copy ordered after the host-to-device node of the copy that
// used its buffer last, so pinned memory does not grow with the copy count.
// Each node is created and updated with the GPU of the memory it
// touches current, which pool and graph allocations require.
// MXX_GPU_HOST_STAGED_COPIES=1 stages every copy between distinct logical
// devices (a diagnostic for GPUs without peer access).
int mxx_gpu_graph_builder_add_device_copy(MxxGpuGraphBuilder *builder,
    void *destination, int destination_device, const void *source, int source_device,
    size_t bytes, const MxxGraphPatch *patches, size_t patch_count);
int mxx_gpu_graph_builder_add_memcpy(MxxGpuGraphBuilder *builder,
    void *destination, const void *source, size_t bytes, int copy_kind,
    const MxxGraphPatch *patches, size_t patch_count);
int mxx_gpu_graph_builder_add_memset(MxxGpuGraphBuilder *builder,
    void *destination, int value, size_t bytes, const MxxGraphPatch *patch);
int mxx_gpu_graph_builder_add_export_publish(MxxGpuGraphBuilder *builder,
    void *device_header, uint64_t occurrence, uint64_t artifact_offset,
    uint64_t payload_bytes, uint32_t site, uint32_t flags,
    uint32_t header_binding);

int mxx_gpu_graph_builder_begin_if(MxxGpuGraphBuilder *builder,
    const uint64_t *predicate, uint32_t predicate_binding);
int mxx_gpu_graph_builder_begin_while(MxxGpuGraphBuilder *builder,
    uint64_t *index, const uint64_t *limit, uint64_t max_iterations,
    uint32_t *status_word, uint32_t index_binding, uint32_t limit_binding,
    uint32_t status_binding);
int mxx_gpu_graph_builder_finish_generic_body(MxxGpuGraphBuilder *builder);
int mxx_gpu_graph_builder_finish(MxxGpuGraphBuilder *builder, MxxGpuGraphExec **out_exec);
void mxx_gpu_graph_builder_destroy(MxxGpuGraphBuilder *builder);
MxxGpuGraphBuilder *mxx_gpu_graph_builder_for_stream(GpuContext *ctx, void *stream);

int mxx_gpu_graph_upload(MxxGpuGraphExec *exec, void *launch_stream);
int mxx_gpu_graph_bind(
    MxxGpuGraphExec *exec,
    const MxxGraphBindingValue *values,
    size_t count);
int mxx_gpu_graph_launch(
    MxxGpuGraphExec *exec,
    void *launch_stream,
    MxxGpuNativeEvent **out_event);
void mxx_gpu_graph_exec_destroy(MxxGpuGraphExec *exec);

int mxx_gpu_stream_record_event(void *stream, int device, MxxGpuNativeEvent **out_event);
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

void mxx_gpu_native_event_destroy(MxxGpuNativeEvent *event);
// Enqueue a device-to-device copy from a validated resident address into an
// owned buffer after the caller has enqueued the source producer dependencies
// on `stream`. The returned event completes with the copy; the host never waits.
int gpu_device_buffer_copy_from_address(
    const void *source,
    int source_device,
    MxxGpuDeviceBuffer *destination,
    size_t bytes,
    void *stream,
    MxxGpuNativeEvent **out_event);

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
    bool registered = false;
    std::vector<std::vector<cudaStream_t>> compute_streams_by_partition;
    std::vector<cudaStream_t> release_streams_by_partition;
    PinnedHostReclaimer *pinned_host_reclaimer = nullptr;
    std::atomic<size_t> next_compute_stream{0};
    std::mutex graph_mutex;
    MxxGpuGraphBuilder *explicit_builder = nullptr;
    ~GpuExecutionOwner();
};

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
