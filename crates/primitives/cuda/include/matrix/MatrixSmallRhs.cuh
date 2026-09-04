#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct GpuSmallMatrix GpuSmallMatrix;

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
    GpuSmallMatrix **out);
void gpu_small_matrix_destroy(GpuSmallMatrix *mat);
int gpu_small_matrix_wait(const GpuSmallMatrix *mat);
int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src);
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
int gpu_small_matrix_decompose_base(
    const GpuMatrix *src,
    uint32_t base_bits,
    int small_mode,
    const uint64_t *max_coefficient_bound,
    size_t bound_word_count,
    GpuSmallMatrix *out);
int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat);
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
    GpuMatrix *out,
    const GpuMatrix *lhs_eval,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report);
#ifdef __cplusplus
}
#endif
