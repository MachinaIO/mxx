#pragma once

#include "matrix/Matrix.cuh"
#include "matrix/MatrixData.cuh"
#include "matrix/MatrixCrt.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuMatrixDecomposeWorkspaceBytes
    {
        // Empty when the coefficient source can be read directly.
        GpuMatrixAllocationBytes coefficient_copy;
        // Correction reuses coefficient_copy auxiliary storage when it fits.
        GpuMatrixTransformWorkspaceBytes correction;
        size_t output_rows;
    } GpuMatrixDecomposeWorkspaceBytes;

    int gpu_matrix_query_decompose_workspace_bytes(
        const GpuContext *ctx, int level, size_t rows, size_t cols, int format,
        uint32_t base_bits, int small, size_t dropped_moduli,
        GpuMatrixDecomposeWorkspaceBytes *out);
    int gpu_matrix_query_gadget_correction_workspace_bytes(
        const GpuContext *ctx, int level, size_t rows, size_t cols,
        size_t dropped_moduli, GpuMatrixTransformWorkspaceBytes *out);

    // mode: zero, identity, unit row, gadget, unit column. All rows of the
    // logical constant are retained; range locates them in the output owner.
    int gpu_matrix_fill_constant_columns(
        GpuMatrix *out, const GpuMatrixRange *range, size_t global_column_start,
        int mode, size_t total_columns, size_t unit_index,
        uint32_t base_bits, int small, size_t dropped_moduli);
    int gpu_matrix_fill_small_decomposed_identity_chunk(
        GpuMatrix *out,
        const GpuMatrix *scalar_by_digit,
        size_t chunk_idx);
    int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out, size_t dropped_moduli);
    int gpu_matrix_correct_gadget_residues(GpuMatrix *src, size_t dropped_moduli);
    int gpu_matrix_decompose_base_small(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out);

#ifdef __cplusplus
}
#endif
