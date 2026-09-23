#pragma once

#include <stddef.h>
#include <stdint.h>

#include "Runtime.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Device-side status values are written only when the corresponding optional
// status pointer is supplied.  They are intentionally small and stable: the
// lowering layer can keep the status resident and inspect it at its existing
// error boundary.
enum MxxGpuControlStatus : uint32_t
{
    MXX_GPU_CONTROL_OK = 0,
    MXX_GPU_CONTROL_DIVISION_BY_ZERO = 1,
    MXX_GPU_CONTROL_OVERFLOW = 2,
    MXX_GPU_CONTROL_INVALID_INDEX = 3,
    MXX_GPU_CONTROL_INEXACT_DIVISION = 4,
    MXX_GPU_CONTROL_INVALID_RING_PROPERTY = 5,
};

int gpu_control_integer_operation_direct(
    GpuContext *ctx, void *out, const void *lhs, const void *rhs,
    void *aux, uint32_t *status, size_t count, size_t lhs_count,
    size_t rhs_count, int output_encoding, int lhs_encoding,
    int rhs_encoding, unsigned operation, uint64_t argument,
    void *stream_raw, uint32_t out_binding, uint32_t lhs_binding,
    uint32_t rhs_binding, uint32_t aux_binding, uint32_t status_binding);

#ifdef __cplusplus
}
#endif
