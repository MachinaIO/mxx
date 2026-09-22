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
    void *stream,
    uint32_t descriptor_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index);

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
    void *stream,
    uint32_t descriptor_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index);

int gpu_primitive_pack_polynomial_coefficients(
    GpuContext *ctx,
    const void *bits,
    size_t bit_count,
    size_t coefficient_bits,
    int bits_encoding,
    void *packed_output,
    uint32_t *status,
    void *stream,
    uint32_t bits_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index);

#ifdef __cplusplus
}
#endif
