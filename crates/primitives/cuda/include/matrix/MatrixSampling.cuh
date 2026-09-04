#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"

int launch_sample_distribution_multi_limb_kernel(
    uint8_t *dst_base,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    gpu_chacha::GpuRngSeed seed,
    cudaStream_t stream);

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_sample_distribution(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t max_coefficient_bound,
        uint64_t coefficient_modulus,
        gpu_chacha::GpuRngSeed seed);

    int gpu_matrix_sample_distribution_columns(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t max_coefficient_bound,
        uint64_t coefficient_modulus,
        gpu_chacha::GpuRngSeed seed,
        size_t full_ncol,
        size_t col_offset);

#ifdef __cplusplus
}
#endif
