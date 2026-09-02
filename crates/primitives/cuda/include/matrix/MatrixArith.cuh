#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
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
    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_scalar(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar);
    int gpu_matrix_binary_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count,
        int operation);
    int gpu_matrix_negate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        size_t matrix_count);
    int gpu_matrix_mul_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count);
    int gpu_matrix_mul_accumulate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        const GpuMatrix *const *coefficients,
        const GpuMatrix *const *biases,
        const size_t *inner_dimensions,
        size_t matrix_count,
        size_t product_count);
    int gpu_matrix_mul_scalar_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *matrices,
        const GpuMatrix *const *scalars,
        size_t matrix_count);
    int gpu_matrix_intt_out_of_place_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        size_t matrix_count);

#ifdef __cplusplus
}
#endif
