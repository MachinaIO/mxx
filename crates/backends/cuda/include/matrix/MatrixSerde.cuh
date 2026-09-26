#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_load_rns_batch(
        GpuMatrix *mat,
        const uint8_t *bytes,
        size_t bytes_per_poly,
        GpuEventSet **out_events);

    enum GpuValuesEncoding
    {
        GPU_VALUES_SIGNED_I64 = 0,
        GPU_VALUES_CANONICAL_U64 = 1,
    };

    int gpu_matrix_load_compact_bytes(
        GpuMatrix *mat,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

    int gpu_poly_load_compact_bytes(
        GpuMatrix *poly,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

#ifdef __cplusplus
}
#endif
