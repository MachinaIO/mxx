#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"
#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixNTT.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_sample_gaussian_batch(
        GpuMatrix *const *outputs, const gpu_chacha::GpuRngSeed *seeds,
        size_t count, double sigma);

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

    typedef struct GpuPreparedSampling GpuPreparedSampling;

    int gpu_matrix_prepare_sampling(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t max_coefficient_bound,
        uint64_t coefficient_modulus,
        size_t full_ncol,
        size_t col_offset,
        const GpuMatrixRange *range,
        GpuPreparedSampling **plan);
    int gpu_matrix_submit_sampling(
        const GpuPreparedSampling *plan,
        gpu_chacha::GpuRngSeed seed);
    void gpu_matrix_destroy_sampling(GpuPreparedSampling *plan);

#ifdef __cplusplus
}
#endif
