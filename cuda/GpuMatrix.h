#pragma once

#include <stddef.h>

#include "GpuPoly.h"

#ifdef __cplusplus
extern "C" {
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

#ifdef __cplusplus
}
#endif
