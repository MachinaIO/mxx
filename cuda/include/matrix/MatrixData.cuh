#pragma once

#include "matrix/Matrix.h"

namespace
{
    int get_matrix_u64_limb_table(
        const GpuMatrix *matrix,
        int limb,
        const GpuMatrix::LimbU64Table **out_table,
        const char *unsupported_msg);

    int try_get_matrix_u64_limb_table(
        const GpuMatrix *matrix,
        int limb,
        const GpuMatrix::LimbU64Table **out_table,
        bool *out_available);
} // namespace

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
    int gpu_matrix_entry_clone(
        const GpuMatrix *mat,
        size_t row,
        size_t col,
        GpuPoly **out_poly);
    int gpu_matrix_copy_entry(
        GpuMatrix *mat,
        size_t row,
        size_t col,
        const GpuPoly *src);
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
