#pragma once

#include "matrix/Matrix.cuh"

int set_error(const char *msg);
int set_error(cudaError_t err);
bool parse_format(int format, GpuPolyFormat &out);
size_t matrix_poly_count(const GpuMatrix *mat);
int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device);
int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, cudaStream_t *out_stream);
uint8_t *matrix_limb_ptr_by_id(GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id);
const uint8_t *matrix_limb_ptr_by_id(const GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id);
bool matrix_limb_metadata_by_id(
    const GpuMatrix *mat,
    const dim3 &limb_id,
    size_t *out_stride_bytes,
    uint8_t *out_coeff_bytes);
int matrix_wait_limb_stream(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream);
int matrix_track_limb_consumer(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream);
int matrix_record_limb_write(GpuMatrix *dst, const dim3 &limb_id, cudaStream_t stream);
bool matrix_aux_slice_for_limb(const GpuMatrix *mat, const dim3 &limb_id, size_t bytes, void **out_ptr);
size_t matrix_align_up_size(size_t value, size_t alignment);
int matrix_acquire_aux_workspace(
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id,
    size_t bytes,
    void **out_ptr,
    bool *out_shared,
    cudaStream_t stream);
int matrix_release_aux_workspace(void *ptr, bool from_shared, cudaStream_t stream);

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
