#pragma once

#include "matrix/Matrix.cuh"
#include "matrix/MatrixData.cuh"
#include "ChaCha.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct GpuSmallMatrix GpuSmallMatrix;

// Derive a canonical per-stage preimage seed from the fixed device control
// record.  Stage IDs 0, 1, and 2 use the byte-compatible host v1 transcript
// for p2, p1, and z respectively. Both pointers are device addresses and the call only
// enqueues one allocation-free kernel on `stream`; it performs no host copy,
// event operation, or synchronization.  `device_seed` must point at a
// retained 32-byte gpu_chacha::GpuRngSeed workspace owned by the sampler.
int gpu_preimage_seed_from_control_stage_async(
    GpuContext *ctx,
    const void *device_control,
    void *device_seed,
    uint32_t stage_id,
    void *stream);

typedef struct GpuSmallMatrixBindingDescriptor
{
    int physical_device;
    size_t rows;
    size_t columns;
    size_t n;
    size_t magnitude_bytes;
    uint32_t bound_domain;
    size_t crt_depth;
    size_t payload_bytes;
    size_t storage_columns;
    size_t column_offset;
    size_t row_stride_bytes;
    size_t column_stride_bytes;
    size_t coefficient_stride_bytes;
    size_t limb_stride_bytes;
    void *payload;
    void *device_status;
    void *host_status;
    void *hard_cutoff_staging;
    size_t hard_cutoff_staging_bytes;
} GpuSmallMatrixBindingDescriptor;

typedef struct GpuSmallMatrixAllocationReport
{
    size_t lhs_eval_bytes;
    size_t compact_rhs_bytes;
    size_t full_output_bytes;
    size_t expanded_rhs_workspace_bytes;
    size_t event_overhead_bytes;
    size_t high_water_bytes;
    size_t full_expanded_rhs_bytes;
    size_t workspace_word_bytes;
    size_t ntt_preparation_launches;
    size_t u32_workspace_limb_count;
    size_t u64_workspace_limb_count;
} GpuSmallMatrixAllocationReport;

int gpu_small_matrix_create(
    GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    uint32_t bound_domain,
    const uint64_t *bound_words,
    size_t bound_word_count,
    GpuSmallMatrix **out);
// Side-effect-free exact size query for the compact owner created by
// gpu_small_matrix_create.  This uses the same native payload-size helper as
// the allocating path and deliberately performs no CUDA allocation.
int gpu_small_matrix_query_allocation_bytes(
    const GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    uint32_t bound_domain,
    GpuMatrixAllocationBytes *out);
int gpu_small_matrix_binding_descriptor(
    const GpuSmallMatrix *mat,
    GpuSmallMatrixBindingDescriptor *out);
void gpu_small_matrix_destroy(GpuSmallMatrix *mat);
int gpu_small_matrix_wait(const GpuSmallMatrix *mat);
// Asynchronous compiled-runtime lifetime adapters. Handles are non-owning
// CUDA stream/event values represented as void* at the C ABI boundary.
int gpu_small_matrix_wait_compiled_inputs(
    const GpuSmallMatrix *mat,
    int consumer_device,
    void *consumer_stream);
int gpu_small_matrix_track_compiled_consumer(
    const GpuSmallMatrix *mat,
    void *consumer_stream,
    void *completion_event);
int gpu_small_matrix_record_compiled_write(
    GpuSmallMatrix *mat,
    void *stream);
int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src);
int gpu_small_matrix_copy_cross_context(GpuSmallMatrix *out, const GpuSmallMatrix *src);
int gpu_small_matrix_copy_columns(
    GpuSmallMatrix *out,
    const GpuSmallMatrix *src,
    size_t source_column_start);
int gpu_small_matrix_view_columns(
    const GpuSmallMatrix *src,
    size_t source_column_start,
    size_t columns,
    GpuSmallMatrix **out);
int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat,
    const uint8_t *payload,
    size_t payload_len);
int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat,
    uint8_t *payload,
    size_t payload_len);
struct MxxDecomposeFragmentRange {
    size_t source_column, destination_row, destination_column, columns;
};
int gpu_small_matrix_decompose_base(
    const GpuMatrix *const *sources,
    size_t block_count,
    uint32_t base_bits,
    int small_mode,
    const uint64_t *max_coefficient_bound,
    size_t bound_word_count,
    GpuSmallMatrix *out,
    size_t dropped_moduli,
    const MxxDecomposeFragmentRange *ranges);
int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat);
int gpu_small_matrix_prepare_preimage_hard_cutoff_for_tile(
    GpuSmallMatrix *mat,
    size_t rows,
    size_t cols);
// Submit one fixed-workspace cutoff attempt. This function only enqueues
// kernels and records native lifetime events; it does not allocate, copy to
// the host, synchronize, or inspect acceptance.
int gpu_small_matrix_submit_preimage_hard_cutoff_tile(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    uint32_t attempt);
// Explicit stream variant used while a sampler captures a conditional-body
// child graph. The stream belongs to the same primitives execution owner and
// is never retained by this call.
int gpu_small_matrix_submit_preimage_hard_cutoff_tile_on_stream(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    uint32_t attempt,
    void *stream);
int gpu_small_matrix_submit_preimage_hard_cutoff_tile_control(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    void *device_control,
    void *stream);
int gpu_small_matrix_copy_preimage_status_async(GpuSmallMatrix *mat);
int gpu_small_matrix_mark_preimage_exhausted(GpuSmallMatrix *mat);
int gpu_small_matrix_wait_preimage_status(
    GpuSmallMatrix *mat,
    MxxPreimageStatus *out_status);
int gpu_small_matrix_try_pack_preimage_hard_cutoff_tile(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    int32_t *accepted_out);
int gpu_matrix_mul_small_rhs(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report);
// Capture-safe compact multiplication into preallocated evaluation matrices.
// The binding IDs are operation-local graph identities: one source and one
// destination descriptor binding per row block, plus the compact RHS payload
// binding.  The caller maps these local IDs to its outer capture schema before
// launch; this entry point only enqueues work on the supplied active stream.
int gpu_matrix_mul_small_rhs_into_bound(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report,
    void *stream,
    const uint32_t *source_binding_indices,
    const uint32_t *destination_binding_indices,
    uint32_t rhs_payload_binding_index,
    const size_t *block_fragment_ends);
#ifdef __cplusplus
}
#endif
