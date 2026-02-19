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
        int format,
        GpuEventSet **out_events);

    int gpu_matrix_store_rns_batch(
        const GpuMatrix *mat,
        uint8_t *bytes_out,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events);

    int gpu_matrix_store_compact_bytes(
        GpuMatrix *mat,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    int gpu_matrix_load_compact_bytes(
        GpuMatrix *mat,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

    int gpu_poly_store_compact_bytes(
        GpuMatrix *poly,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    int gpu_poly_load_compact_bytes(
        GpuMatrix *poly,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

#ifdef __cplusplus
}
#endif
