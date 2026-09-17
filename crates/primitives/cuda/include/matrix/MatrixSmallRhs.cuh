#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_prepared_plan.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct GpuSmallMatrix GpuSmallMatrix;
typedef struct GpuPreparedSmallRhs GpuPreparedSmallRhs;
typedef struct GpuPreparedSmallUpload GpuPreparedSmallUpload;
typedef struct GpuPreparedPreimageCutoff GpuPreparedPreimageCutoff;
int gpu_preimage_cutoff_batch_layout(GpuSmallMatrix *output, size_t job_count,
    GpuPreparedWorkspaceLayout *layouts, size_t capacity, size_t *count);
int gpu_preimage_cutoff_batch_layout_shape(size_t ring_dimension, size_t rows, size_t columns,
    size_t magnitude_bytes, size_t job_count, GpuPreparedWorkspaceLayout *layouts,
    size_t capacity, size_t *count);
int gpu_preimage_cutoff_is_ready(const GpuPreparedPreimageCutoff *plan, bool *ready);

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
    const uint64_t *bound_words,
    size_t bound_word_count,
    bool initialize_zero,
    GpuSmallMatrix **out);
void gpu_small_matrix_destroy(GpuSmallMatrix *mat);
int gpu_small_matrix_wait(const GpuSmallMatrix *mat);
int gpu_small_matrix_prepare_readback(GpuSmallMatrix *mat, uint8_t *payload, size_t bytes);
int gpu_small_matrix_read_prepared(const GpuSmallMatrix *mat);
int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src);
int gpu_small_matrix_copy_columns(
    GpuSmallMatrix *out,
    const GpuSmallMatrix *src,
    size_t source_column_start);
int gpu_small_matrix_copy_range(
    GpuSmallMatrix *out,
    size_t destination_column_start,
    const GpuSmallMatrix *src,
    size_t source_column_start,
    size_t columns);
int gpu_small_matrix_view_columns(
    const GpuSmallMatrix *src,
    size_t source_column_start,
    size_t columns,
    GpuSmallMatrix **out);
int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat,
    const uint8_t *payload,
    size_t payload_len);
int gpu_matrix_prepare_small_upload(
    GpuSmallMatrix *mat, const uint8_t *payload, size_t payload_len,
    const GpuPreparedPlanDescriptor *plan,
    GpuPreparedSmallUpload **out_plan);
int gpu_matrix_submit_small_upload(GpuPreparedSmallUpload *plan);
int gpu_matrix_query_small_upload(const GpuPreparedSmallUpload *plan, int *out_ready);
int gpu_matrix_wait_small_upload(const GpuPreparedSmallUpload *plan);
void gpu_matrix_destroy_small_upload(GpuPreparedSmallUpload *plan);
int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat,
    uint8_t *payload,
    size_t payload_len);
int gpu_small_matrix_decompose_base(
    const GpuMatrix *const *sources,
    size_t block_count,
    uint32_t base_bits,
    int small_mode,
    const uint64_t *max_coefficient_bound,
    size_t bound_word_count,
    GpuSmallMatrix *out,
    size_t dropped_moduli,
    const GpuMatrixRange *source_views,
    const GpuMatrixRange *destination_view);
int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat);
int gpu_small_matrix_prepare_preimage_cutoff(
    GpuSmallMatrix *const *destinations, const GpuMatrix *const *sources,
    const size_t *dst_rows, const size_t *dst_columns, size_t count,
    int32_t *host_status, const GpuPreparedWorkspaceLayout *layouts, size_t layout_count,
    GpuPreparedPreimageCutoff **out);
int gpu_small_matrix_begin_preimage_cutoff(GpuPreparedPreimageCutoff *plan);
int gpu_small_matrix_submit_preimage_cutoff(GpuPreparedPreimageCutoff *plan);
int gpu_small_matrix_finish_preimage_cutoff(GpuPreparedPreimageCutoff *plan);
int gpu_small_matrix_wait_preimage_cutoff(const GpuPreparedPreimageCutoff *plan);
void gpu_small_matrix_destroy_preimage_cutoff(GpuPreparedPreimageCutoff *plan);

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
int gpu_matrix_query_small_rhs_workspace_bytes(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t *narrow_bytes, size_t *wide_bytes);
int gpu_matrix_mul_small_rhs(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report, const GpuMatrixBatchView *views);
// Prepare one fixed compact-RHS multiplication.  The compact RHS NTT and its
// typed workspace are built once; submit only rebinds a same-shape evaluation
// input and replays the fixed accumulation launch.
int gpu_matrix_prepare_small_rhs(
    const GpuMatrix *input_template,
    GpuMatrix *output,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    const GpuPreparedPlanDescriptor *plan,
    GpuPreparedSmallRhs **out_plan);
int gpu_matrix_submit_small_rhs(
    const GpuPreparedSmallRhs *plan,
    const GpuMatrix *input);
void gpu_matrix_destroy_prepared_small_rhs(GpuPreparedSmallRhs *plan);
#ifdef __cplusplus
}
#endif
