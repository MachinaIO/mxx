#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

// These entry points operate only on already resident allocations.  The
// descriptor array is the device-side array owned by GpuMatrix; values and
// status are caller-owned device buffers.  No allocation or host transfer is
// performed by any entry point.
// Status points to an eight-byte integer owner. The low word is the atomic
// error code; every submission resets the entire owner, including its high word.
int gpu_primitive_extract_coefficient(
    GpuContext *ctx,
    const void *descriptors,
    size_t limb_count,
    size_t ring_dimension,
    size_t position,
    size_t output_words,
    void *output,
    uint32_t *status,
    void *stream);

int gpu_primitive_threshold_decode(
    GpuContext *ctx,
    const void *descriptors,
    size_t limb_count,
    size_t ring_dimension,
    const uint64_t *plaintext_modulus,
    size_t plaintext_words,
    uint64_t *workspace,
    size_t length,
    bool output_bool,
    void *output,
    uint32_t *status,
    void *stream);

int gpu_primitive_pack_polynomial_coefficients(
    GpuContext *ctx,
    const void *bits,
    size_t bit_count,
    size_t coefficient_bits,
    int bits_encoding,
    void *packed_output,
    uint32_t *status,
    void *stream);

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
