#pragma once

#include "matrix/Matrix.h"

int set_error(const char *msg);
int set_error(gpuError_t err);

size_t matrix_poly_count(const GpuMatrix *mat);
int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device);
int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, gpuStream_t *out_stream);
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
    gpuStream_t consumer_stream,
    bool device_already_selected = false, bool read_only = false);

int matrix_record_limb_write(
    GpuMatrix *dst, const dim3 &limb_id, gpuStream_t stream,
    bool device_already_selected = false);
// Whole operations must wait all affected limbs before dispatch; completion
// aliasing stays matrix-owned and later per-limb writes restore individual slots.
int matrix_wait_all_limb_streams(
    const GpuMatrix *src, int consumer_device, gpuStream_t consumer_stream,
    bool device_already_selected = false, bool read_only = false);

int matrix_record_all_limb_writes(
    GpuMatrix *dst, gpuStream_t stream, bool device_already_selected = false);

__host__ __device__ __forceinline__ uint64_t matrix_load_packed_u64_at(
    const uint8_t *ptr,
    uint8_t coeff_bytes);

__host__ __device__ __forceinline__ void matrix_store_packed_u64_at(
    uint8_t *ptr,
    uint8_t coeff_bytes,
    uint64_t value);

__host__ __device__ __forceinline__ uint64_t matrix_load_limb_u64(
    const uint8_t *base,
    size_t poly_idx,
    size_t coeff_idx,
    size_t stride_bytes,
    uint8_t coeff_bytes);

__host__ __device__ __forceinline__ void matrix_store_limb_u64(
    uint8_t *base,
    size_t poly_idx,
    size_t coeff_idx,
    size_t stride_bytes,
    uint8_t coeff_bytes,
    uint64_t value);
