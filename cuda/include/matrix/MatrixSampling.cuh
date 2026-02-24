#pragma once

#include "matrix/Matrix.cuh"

int launch_sample_distribution_multi_limb_kernel(
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &moduli,
    const std::vector<uint32_t> &limb_indices,
    int dist_type,
    double sigma,
    uint64_t seed,
    cudaStream_t stream);

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_sample_distribution(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t seed);
    int gpu_matrix_sample_distribution_decompose_base(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t seed,
        uint32_t base_bits);
    int gpu_matrix_sample_distribution_decompose_base_small(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t seed,
        uint32_t base_bits);

#ifdef __cplusplus
}
#endif
