#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

// Direct explicit-Graph variants consume one owner-derived FullCoeff or
// FullEval scalar matrix physical view and caller-owned SignedWords storage.
int gpu_raw_polynomial_values(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, void *output,
    size_t output_magnitude_words, uint32_t source_binding_base,
    uint32_t output_binding);
int gpu_raw_extract_coefficient(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const void *position,
    int position_encoding, void *output, size_t output_magnitude_words,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t position_binding, uint32_t output_binding,
    uint32_t status_binding);
int gpu_raw_pack_polynomial_coefficients(GpuContext *ctx, void *stream,
    const void *bits, size_t bit_count, int bits_encoding,
    const void *coefficient_bits, int coefficient_bits_encoding,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t bits_binding, uint32_t coefficient_bits_binding,
    uint32_t destination_binding_base, uint32_t status_binding);
int gpu_raw_threshold_decode(GpuContext *ctx, void *stream,
    const MxxRawMatrixView *source, const void *plaintext_modulus,
    size_t plaintext_words, const void *length_value, int length_encoding,
    void *workspace, size_t workspace_bytes, void *output,
    size_t output_count, size_t output_magnitude_words, bool output_bool,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t plaintext_binding, uint32_t length_binding,
    uint32_t workspace_binding, uint32_t output_binding,
    uint32_t status_binding);

#ifdef __cplusplus
}
#endif
