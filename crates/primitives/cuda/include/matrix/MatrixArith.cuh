#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum GpuMatrixBatchOperation
    {
        GPU_MATRIX_BATCH_BINARY = 0,
        GPU_MATRIX_BATCH_NEGATE = 1,
        GPU_MATRIX_BATCH_AUTOMORPHISM = 2,
        GPU_MATRIX_BATCH_SCALAR = 3,
        GPU_MATRIX_BATCH_MULTIPLY = 4,
        GPU_MATRIX_BATCH_ACCUMULATE = 5,
    } GpuMatrixBatchOperation;

    typedef struct GpuMatrixBatchWorkspaceBytes
    {
        size_t workspace_bytes;
        size_t additional_bytes;
        size_t alignment;
    } GpuMatrixBatchWorkspaceBytes;

    // Rectangular ranges refer to the original allocation owner. Empty ranges
    // are handled by Rust without native submission. No events/payload are copied.
    typedef struct GpuMatrixRange
    {
        size_t row_start;
        size_t row_end;
        size_t column_start;
        size_t column_end;
    } GpuMatrixRange;

    // Null selects complete owners. Otherwise left/output shapes agree across
    // the batch; right is used only for binary add/sub. Output rectangles may
    // share an owner only when disjoint, and output/input owners never alias.
    typedef struct GpuMatrixBatchView
    {
        GpuMatrixRange left;
        GpuMatrixRange right;
        GpuMatrixRange output;
    } GpuMatrixBatchView;

    // Out-of-place batches use the first output's exclusive auxiliary storage.
    // matrix_views is 0 or 1; add/sub, negate, scalar multiplication, and
    // coefficient automorphism accept 1 and reserve each owner's row geometry.
    // output_rows/output_cols describe the full first output owner, whose
    // auxiliary capacity is shared by the complete batch.
    int gpu_matrix_query_batch_workspace_bytes(
        const GpuContext *ctx, int level, size_t output_rows, size_t output_cols,
        size_t matrix_count, size_t product_count, GpuMatrixBatchOperation operation,
        int matrix_views,
        GpuMatrixBatchWorkspaceBytes *out);

    // Protect release and reuse after an interrupted batch completion update.
    // This retires submitted work by events; it does not wait for device completion.
    int gpu_matrix_retire_submitted_work(const GpuMatrix *output);

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_transpose(GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view);
    int gpu_matrix_sum_rows(
        GpuMatrix *out, const GpuMatrix *source, const size_t *rows, const size_t *offsets,
        size_t group_count, size_t term_count, const GpuMatrixBatchView *view);
    int gpu_matrix_add_row_blocks(
        GpuMatrix *out, const GpuMatrix *const *lhs_blocks, size_t block_count, const GpuMatrix *rhs,
        const GpuMatrixRange *block_views, const GpuMatrixBatchView *view);
    int gpu_matrix_add_block(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols);
    int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_tensor(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
        const GpuMatrixBatchView *view, size_t column_start);
    int gpu_matrix_tensor_sum_rows(
        GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
        const size_t *rows, const size_t *offsets, size_t group_count, size_t term_count,
        const GpuMatrixBatchView *view, size_t column_start);
    // One result span per active device, reused across its CRT limbs.
    int gpu_matrix_query_equality_workspace(GpuPreparedWorkspaceLayout *out);

    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_scalar(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar);
    int gpu_matrix_binary_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        const GpuMatrixBatchView *views,
        size_t matrix_count,
        int operation);
    int gpu_matrix_negate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        const GpuMatrixBatchView *views,
        size_t matrix_count);
    int gpu_matrix_mul_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        const GpuMatrixBatchView *views,
        size_t matrix_count);
    int gpu_matrix_validate_ring_automorphism(
        size_t ring_dimension,
        const size_t *indices,
        size_t matrix_count);
    int gpu_matrix_ring_automorphism_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        const size_t *indices,
        const GpuMatrixBatchView *views,
        size_t matrix_count);
    int gpu_matrix_mul_accumulate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        const GpuMatrix *const *coefficients,
        const GpuMatrix *const *biases,
        const size_t *inner_dimensions,
        size_t matrix_count,
        size_t product_count,
        const GpuMatrixBatchView *views,
        const GpuMatrixRange *bias_view,
        const uint64_t *integer_residues);
    int gpu_matrix_mul_scalar_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *matrices,
        const GpuMatrix *const *scalars,
        const GpuMatrixBatchView *views,
        size_t matrix_count,
        const uint64_t *integer_residues);
    // Null inputs transform the exclusively owned outputs in place.
    int gpu_matrix_intt_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        size_t matrix_count);
    int gpu_matrix_ntt_in_place_batch(
        GpuMatrix *const *matrices,
        size_t matrix_count);

#ifdef __cplusplus
}
#endif
