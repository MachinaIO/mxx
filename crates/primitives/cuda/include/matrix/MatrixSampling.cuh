#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"
#include "matrix/MatrixArith.cuh"

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
        size_t col_offset,
        const GpuMatrixRange *range);

#ifdef __cplusplus
}
#endif
