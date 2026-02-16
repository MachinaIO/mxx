#pragma once

#include "Poly.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_block_add(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count);
    int gpu_block_sub(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count);
    int gpu_block_entrywise_mul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t count);
    int gpu_block_mul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols);
    int gpu_block_mul_timed(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        double *out_kernel_ms);

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_timed(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, double *out_kernel_ms);
    int gpu_matrix_mul_scalar(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuPoly *scalar);

#ifdef __cplusplus
}
#endif
