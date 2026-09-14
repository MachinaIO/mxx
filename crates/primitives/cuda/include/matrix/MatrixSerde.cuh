#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"

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

    int gpu_matrix_rns_store_completion_events(const GpuMatrix *mat, size_t *out_count);
    int gpu_matrix_store_rns_batch(
        const GpuMatrix *mat,
        uint8_t *bytes_out,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events);

    int gpu_matrix_store_const_coeff_batch(
        const GpuMatrix *mat,
        uint64_t *words_out,
        size_t words_per_poly,
        GpuEventSet **out_events);


    int gpu_matrix_store_compact_bytes(
        GpuMatrix *mat,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    int gpu_matrix_store_compact_bytes_batch(
        GpuMatrix *const *matrices,
        size_t matrix_count,
        uint8_t *const *payload_outputs,
        const size_t *payload_capacities,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_lengths);

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

// Exact unpack/pack span for one single-device RNS matrix transfer. This excludes
// the retained matrix, pinned host bytes and completion resources.
extern "C" int gpu_matrix_query_rns_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns,
    GpuPreparedWorkspaceLayout *out);

// kind: 0 scalar store, 1 homogeneous batch store, 2 scalar load.
// Store capacity uses the codec's modulus-derived signed-width bound; load uses
// the validated artifact width. Streams, events and host buffers are separate.
extern "C" int gpu_matrix_query_compact_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns, size_t matrices,
    int kind, uint16_t max_coeff_bits, GpuPreparedWorkspaceLayout *out);
