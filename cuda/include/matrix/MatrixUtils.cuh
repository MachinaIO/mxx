#pragma once

#include "matrix/Matrix.h"

#include <cuda_runtime.h>

namespace
{
    int set_error(const char *msg);
    int set_error(cudaError_t err);
    int default_batch(const GpuContext *ctx);
    bool parse_format(int format, PolyFormat &out);
    int sync_poly_limb_streams(const GpuPoly *poly, const char *context);
    int sync_poly_partition_streams(const GpuPoly *poly, const char *context);
    int sync_context_devices(const GpuContext *ctx, const char *context);
    int transform_matrix_format_sync(
        GpuMatrix *matrix,
        PolyFormat target_format,
        const char *context);
    uint32_t bit_width_u64(uint64_t v);

    __host__ __device__ __forceinline__ size_t matrix_index(size_t row, size_t col, size_t cols);
    __device__ __forceinline__ uint32_t mul_mod_u32(uint32_t a, uint32_t b, uint32_t mod);
    __device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod);
    __device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b, uint32_t mod);
    __device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod);
} // namespace

