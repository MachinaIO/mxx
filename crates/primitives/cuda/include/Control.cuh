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
};

void *gpu_control_launch_stream(GpuContext *ctx, int device, void *fallback);
int gpu_control_integer_operation(
    GpuContext *ctx, void *out, const void *lhs, const void *rhs, void *aux, uint32_t *status,
    size_t count, size_t lhs_count, size_t rhs_count, int output_encoding, int lhs_encoding, int rhs_encoding,
    unsigned operation, uint64_t argument, void *stream_raw);
int gpu_control_wait_input(
    const MxxGpuDeviceBuffer *buffer, int device, void *stream, bool read_only);
int gpu_control_record_compiled_write(MxxGpuDeviceBuffer *buffer, void *stream);
int gpu_control_read_status(
    const MxxGpuDeviceBuffer *status,
    size_t status_offset,
    void *completion_event,
    void *stream,
    uint32_t *host_status);

int gpu_control_fill_constant_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    size_t count,
    int64_t value,
    void *stream,
    uint32_t *status);
int gpu_control_fill_loop_index_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    size_t count,
    int64_t start,
    int64_t step,
    void *stream,
    uint32_t *status);

int gpu_control_add_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream,
    uint32_t *status);
int gpu_control_sub_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream,
    uint32_t *status);
int gpu_control_mul_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream,
    uint32_t *status);

// Euclidean division uses the positive modulus |denominator|, matching
// mxx-ir-core's div_mod_floor contract.  Both outputs are produced by one
// kernel so a captured loop never needs a host-side temporary.
int gpu_control_div_rem_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *quotient,
    size_t quotient_offset,
    MxxGpuDeviceBuffer *remainder,
    size_t remainder_offset,
    const MxxGpuDeviceBuffer *numerator,
    size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator,
    size_t denominator_offset,
    size_t count,
    void *stream,
    uint32_t *status);

// IntExpr::Div is exact division: a non-zero remainder is an execution
// error, unlike ordinary IntBinary division which is Euclidean.
int gpu_control_exact_div_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *numerator,
    size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator,
    size_t denominator_offset,
    size_t count,
    void *stream,
    uint32_t *status);

// IntExpr::FloorDiv/Rem share one kernel so a captured expression does not
// need a host-side temporary.  The remainder has the denominator's sign,
// matching BigInt::div_floor/mod_floor rather than Euclidean |denominator|.
int gpu_control_floor_div_rem_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *quotient,
    size_t quotient_offset,
    MxxGpuDeviceBuffer *remainder,
    size_t remainder_offset,
    const MxxGpuDeviceBuffer *numerator,
    size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator,
    size_t denominator_offset,
    size_t count,
    void *stream,
    uint32_t *status);

int gpu_control_compare_eq_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream);
int gpu_control_compare_lt_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream);
int gpu_control_compare_le_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs,
    size_t rhs_offset,
    size_t count,
    void *stream);
int gpu_control_bit_extract_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t count,
    uint64_t bit,
    void *stream);
int gpu_control_bool_to_int_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t count,
    void *stream);

int gpu_control_copy_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t count,
    void *stream);
// Pack is the resident, stream-ordered contiguous form of copy.  Higher
// layers may issue several such segments when packing a family; each segment
// remains allocation-free and graph-capturable.
int gpu_control_pack_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t count,
    void *stream);

// Encoding-agnostic resident transport.  The byte offsets are part of the
// launch schema and are also applied as graph address addends, so replay can
// rebind the owner bases without retaining exemplar pointers.
int gpu_control_copy_range(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    uint64_t destination_addend,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    uint64_t source_addend,
    size_t bytes,
    void *stream);

// Encoding-agnostic u64-word gather used for CanonicalU64 families. Indices
// remain signed i64 words, but source and destination payloads are copied
// without arithmetic or signed interpretation.
int gpu_control_gather_u64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t source_count,
    const MxxGpuDeviceBuffer *indices,
    size_t indices_offset,
    size_t count,
    size_t value_words,
    void *stream, uint32_t *status);

int gpu_control_gather_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *source,
    size_t source_offset,
    size_t source_count,
    const MxxGpuDeviceBuffer *indices,
    size_t indices_offset,
    size_t count,
    void *stream,
    uint32_t *status);
int gpu_control_select_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    const MxxGpuDeviceBuffer *selector,
    size_t selector_offset,
    size_t selector_count,
    const MxxGpuDeviceBuffer *when_false,
    size_t when_false_offset,
    size_t when_false_count,
    const MxxGpuDeviceBuffer *when_true,
    size_t when_true_offset,
    size_t when_true_count,
    size_t count,
    void *stream,
    uint32_t *status);

#ifdef __cplusplus
}
#endif
