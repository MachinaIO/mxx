// Inline packed-limb definitions shared by the matrix and primitive translation units.
#include "matrix/MatrixUtils.h"

__host__ __device__ __forceinline__ uint64_t matrix_load_packed_u64_at(
    const uint8_t *ptr,
    uint8_t coeff_bytes)
{
    uint64_t out = 0;
    for (uint8_t i = 0; i < coeff_bytes; ++i)
    {
        out |= static_cast<uint64_t>(ptr[i]) << (static_cast<uint32_t>(i) * 8U);
    }
    return out;
}

__host__ __device__ __forceinline__ void matrix_store_packed_u64_at(
    uint8_t *ptr,
    uint8_t coeff_bytes,
    uint64_t value)
{
    for (uint8_t i = 0; i < coeff_bytes; ++i)
    {
        ptr[i] = static_cast<uint8_t>((value >> (static_cast<uint32_t>(i) * 8U)) & 0xFFU);
    }
}

__host__ __device__ __forceinline__ uint64_t matrix_load_limb_u64(
    const uint8_t *base,
    size_t poly_idx,
    size_t coeff_idx,
    size_t stride_bytes,
    uint8_t coeff_bytes)
{
    const size_t byte_offset = poly_idx * stride_bytes + coeff_idx * static_cast<size_t>(coeff_bytes);
    return matrix_load_packed_u64_at(base + byte_offset, coeff_bytes);
}

__host__ __device__ __forceinline__ void matrix_store_limb_u64(
    uint8_t *base,
    size_t poly_idx,
    size_t coeff_idx,
    size_t stride_bytes,
    uint8_t coeff_bytes,
    uint64_t value)
{
    const size_t byte_offset = poly_idx * stride_bytes + coeff_idx * static_cast<size_t>(coeff_bytes);
    matrix_store_packed_u64_at(base + byte_offset, coeff_bytes, value);
}
