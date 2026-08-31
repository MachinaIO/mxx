#pragma once

#include "matrix/Matrix.cuh"

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
    /// Creates homogeneous matrix views over one packed allocation per GPU
    /// partition.  The views remain ordinary GpuMatrix handles to every
    /// existing operation; their shared owner releases the packed bases only
    /// after the final view's stream dependencies have completed.
    int gpu_matrix_create_batch(
        GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        size_t output_count,
        GpuMatrix **outputs);
    void gpu_matrix_destroy(GpuMatrix *mat);
    int gpu_matrix_wait(const GpuMatrix *mat);
    int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src);
    int gpu_matrix_copy_peer(GpuMatrix *dst, const GpuMatrix *src, int *out_copied);
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
