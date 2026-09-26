#pragma once

#include "Runtime.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// All pointers are preallocated resident owners. `status` is a separately
// reset four-byte word: 2 means invalid domain/encoding, 3 means non-finite.
int gpu_real_emit(
    GpuContext *ctx, void *stream, uint32_t operation,
    double *output, const void *left, const double *right,
    int left_integer_encoding, uint64_t constant_bits, uint32_t *status,
    uint32_t output_binding, uint32_t left_binding,
    uint32_t right_binding, uint32_t status_binding);

#ifdef __cplusplus
}
#endif
