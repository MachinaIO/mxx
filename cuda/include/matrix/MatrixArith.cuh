#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_timed(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, double *out_kernel_ms);
    int gpu_matrix_mul_scalar(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar);

#ifdef __cplusplus
}
#endif
