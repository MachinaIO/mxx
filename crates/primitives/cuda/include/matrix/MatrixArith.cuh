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

#ifdef __cplusplus
}
#endif
