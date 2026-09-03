#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuMatrixAllocationBytes
    {
        size_t data_bytes;
        size_t aux_bytes;
        size_t event_bytes;
        size_t total_bytes;
    } GpuMatrixAllocationBytes;

    int gpu_matrix_query_allocation_bytes(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrixAllocationBytes *out);

    int gpu_matrix_create(
        GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrix **out);
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
