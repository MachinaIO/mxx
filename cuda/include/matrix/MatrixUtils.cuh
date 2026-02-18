#pragma once

#include "matrix/Matrix.cuh"

int set_error(const char *msg);
int set_error(cudaError_t err);
int default_batch(const GpuContext *ctx);
bool parse_format(int format, GpuPolyFormat &out);
size_t matrix_poly_count(const GpuMatrix *mat);
int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device);
int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, cudaStream_t *out_stream);
uint64_t *matrix_limb_ptr_by_id(GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id);
const uint64_t *matrix_limb_ptr_by_id(const GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id);
int matrix_wait_limb_stream(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream);
int matrix_wait_limb(const GpuMatrix *dst, const GpuMatrix *src, const dim3 &limb_id);
int matrix_record_limb_write(GpuMatrix *dst, const dim3 &limb_id, cudaStream_t stream);
int sync_matrix_limb_streams(const GpuMatrix *mat, const char *context);
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
