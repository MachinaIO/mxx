#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"

int launch_sample_distribution_multi_limb_kernel(
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    const std::vector<uint64_t> &moduli,
    const std::vector<uint32_t> &limb_indices,
    int dist_type,
    double sigma,
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
        gpu_chacha::GpuRngSeed seed);

    int gpu_matrix_sample_distribution_columns(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        gpu_chacha::GpuRngSeed seed,
        size_t full_ncol,
        size_t col_offset);

#ifdef __cplusplus
}
#endif
