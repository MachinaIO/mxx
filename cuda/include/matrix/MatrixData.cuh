#pragma once

#include "Poly.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_create(
        GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrix **out);
    void gpu_matrix_destroy(GpuMatrix *mat);
    int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src);
    int gpu_matrix_copy_block(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols);

#ifdef __cplusplus
}
#endif
