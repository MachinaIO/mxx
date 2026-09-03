#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_fill_identity_columns(
        GpuMatrix *out,
        size_t full_size,
        size_t global_column_start);
    int gpu_matrix_fill_unit_row_columns(
        GpuMatrix *out,
        size_t total_columns,
        size_t unit_index,
        size_t global_column_start);
    int gpu_matrix_fill_gadget_columns(
        GpuMatrix *out,
        uint32_t base_bits,
        int small,
        size_t full_size,
        size_t global_column_start);
    int gpu_matrix_fill_small_decomposed_identity_chunk(
        GpuMatrix *out,
        const GpuMatrix *scalar_by_digit,
        size_t chunk_idx);
    int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out);
    int gpu_matrix_decompose_base_small(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out);

#ifdef __cplusplus
}
#endif
