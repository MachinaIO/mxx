#pragma once

#include "matrix/Matrix.cuh"
#include "matrix/MatrixData.cuh"
#include "ChaCha.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct GpuSmallMatrix GpuSmallMatrix;

typedef struct GpuSmallMatrixBindingDescriptor
{
    int physical_device;
    size_t rows;
    size_t columns;
    size_t n;
    size_t magnitude_bytes;
    uint32_t bound_domain;
    size_t crt_depth;
    size_t payload_bytes;
    size_t row_stride_bytes;
    size_t column_stride_bytes;
    size_t coefficient_stride_bytes;
    size_t limb_stride_bytes;
    void *payload;
} GpuSmallMatrixBindingDescriptor;

int gpu_small_matrix_create(
    GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    uint32_t bound_domain,
    const uint64_t *bound_words,
    size_t bound_word_count,
    GpuSmallMatrix **out);
// Side-effect-free exact size query for the compact owner created by
// gpu_small_matrix_create.  This uses the same native payload-size helper as
// the allocating path and deliberately performs no CUDA allocation.
int gpu_small_matrix_query_allocation_bytes(
    const GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    uint32_t bound_domain,
    GpuMatrixAllocationBytes *out);
int gpu_small_matrix_binding_descriptor(
    const GpuSmallMatrix *mat,
    GpuSmallMatrixBindingDescriptor *out);
void gpu_small_matrix_destroy(GpuSmallMatrix *mat);
int gpu_small_matrix_wait(const GpuSmallMatrix *mat);

int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat,
    const uint8_t *payload,
    size_t payload_len);

#ifdef __cplusplus
}
#endif
