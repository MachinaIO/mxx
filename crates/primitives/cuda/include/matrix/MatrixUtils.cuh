#pragma once

#include "matrix/Matrix.cuh"

// Exact for every uint128 input. For 1 < modulus < 2^63, reciprocal is
// floor(2^128/modulus); wider moduli retain the general remainder operation.
__device__ __forceinline__ uint64_t matrix_reduce_barrett_u128(
    unsigned __int128 value, uint64_t modulus, uint64_t reciprocal_lo, uint64_t reciprocal_hi);

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
// When device_already_selected is true, the calling thread must already have
// selected the device of every affected limb. These helpers preserve that device
// on success; ordinary callers leave the argument false.
// Only read-only operands may retain host-observed writer readiness and skip
// its dependency. Destinations use the default and invalidate before dispatch.
int matrix_wait_limb_stream(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream,
    bool device_already_selected = false, bool read_only = false);
// Optional completion must already cover the consumer on consumer_device.
// It remains caller-owned and must stay valid until this call has queued its wait.
int matrix_track_limb_consumer(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream,
    cudaEvent_t completion = nullptr, bool device_already_selected = false);
// Register a read-only consumer without modifying the source's producer event.
// The source release stream waits on a per-consumer event, so concurrent
// consumers can safely share a const/Arc-owned matrix.
int matrix_track_limb_consumer_readonly(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream);
int matrix_record_limb_write(
    GpuMatrix *dst, const dim3 &limb_id, cudaStream_t stream,
    bool device_already_selected = false);
// Whole operations must wait all affected limbs before dispatch; completion
// aliasing stays matrix-owned and later per-limb writes restore individual slots.
int matrix_wait_all_limb_streams(
    const GpuMatrix *src, int consumer_device, cudaStream_t consumer_stream,
    bool device_already_selected = false, bool read_only = false);
int matrix_track_all_limb_consumers(
    const GpuMatrix *src, int consumer_device, cudaStream_t consumer_stream,
    cudaEvent_t completion = nullptr, bool device_already_selected = false);
int matrix_record_all_limb_writes(
    GpuMatrix *dst, cudaStream_t stream, bool device_already_selected = false);
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
