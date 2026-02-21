#pragma once

#include "matrix/Matrix.cuh"

int launch_fill_gadget_multi_limb_kernel(
    uint64_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride,
    uint64_t modulus,
    uint32_t limb_idx,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits,
    cudaStream_t stream,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id);

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_fill_gadget(
        GpuMatrix *out,
        uint32_t base_bits);
    int gpu_matrix_fill_small_gadget(
        GpuMatrix *out,
        uint32_t base_bits);
    int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out);
    int gpu_matrix_decompose_base_small(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out);

#ifdef __cplusplus
}
#endif
