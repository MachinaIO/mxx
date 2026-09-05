#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_crt_recompose(
        GpuMatrix *out,
        const GpuMatrix *const *levels,
        size_t level_count,
        const uint64_t *plaintext_moduli,
        const uint64_t *reconstruction_residues,
        size_t reconstruction_stride);

#ifdef __cplusplus
}
#endif
