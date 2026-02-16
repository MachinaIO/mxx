#pragma once

#include "matrix/Matrix.h"

namespace
{
    template <typename PtrType>
    int ensure_device_pointer_capacity(
        int device,
        PtrType **ptr,
        size_t *capacity,
        size_t required_count);
} // namespace

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_block_add_u64_ptrs(
        uint64_t *const *out,
        uint64_t *const *lhs,
        uint64_t *const *rhs,
        size_t count,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream);
    int gpu_block_sub_u64_ptrs(
        uint64_t *const *out,
        uint64_t *const *lhs,
        uint64_t *const *rhs,
        size_t count,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream);
    int gpu_block_mul_u64_ptrs(
        uint64_t *const *out,
        uint64_t *const *lhs,
        uint64_t *const *rhs,
        size_t count,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream);
    int gpu_block_equal_u64_ptrs(
        const uint64_t *const *lhs,
        const uint64_t *const *rhs,
        size_t count,
        size_t n,
        cudaStream_t stream,
        int *out_equal);
    int gpu_block_mul_u64_device_ptrs(
        uint64_t **d_out,
        uint64_t **d_lhs,
        uint64_t **d_rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream,
        double *out_kernel_ms);

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_timed(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, double *out_kernel_ms);
    int gpu_matrix_mul_scalar(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *scalar);

#ifdef __cplusplus
}
#endif
