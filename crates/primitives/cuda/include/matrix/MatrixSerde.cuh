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

    enum GpuValuesEncoding
    {
        GPU_VALUES_SIGNED_I64 = 0,
        GPU_VALUES_CANONICAL_U64 = 1,
    };

    // Fill a preallocated scalar polynomial from a device-resident integer
    // vector. The caller owns the vector and keeps it live until the recorded
    // matrix write completes; this entry point performs no host serialization
    // and is safe to enqueue from a graph body.
    int gpu_matrix_write_values(
        GpuMatrix *mat,
        const void *values_device,
        size_t values_count,
        int format,
        int encoding,
        int output_format,
        size_t constant_column_start,
        void *stream_raw,
        uint32_t source_binding_index,
        uint32_t destination_binding_index);

    int gpu_matrix_store_values_into(
        const GpuMatrix *mat,
        void *values_device,
        size_t values_count,
        size_t words,
        int format,
        int output_device,
        void *output_stream);

    // Capture-safe variant.  When coefficient values are requested from an
    // evaluation-form source, the source is transformed in place on the
    // capture stream, extracted, and transformed back before return.
    int gpu_matrix_store_values_into_bound(
        GpuMatrix *mat,
        void *values_device,
        size_t values_count,
        size_t words,
        int format,
        int output_device,
        void *output_stream,
        uint32_t descriptor_binding_index,
        uint32_t output_binding_index);

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
