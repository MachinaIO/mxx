#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"
#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixNTT.cuh"

typedef struct GpuPreparedPlanDescriptor GpuPreparedPlanDescriptor;

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

    typedef struct GpuPreparedSamplingLayout
    {
        size_t rows;
        size_t columns;
        size_t ring_dimension;
        size_t limb_count;
        size_t polynomial_count;
        size_t blocks;
        size_t workspace_bytes;
        size_t alignment;
        size_t event_count;
        size_t transform_stage_count;
        int device;
        int format;
        int dist_type;
        int stream_role;
    } GpuPreparedSamplingLayout;

    int gpu_matrix_query_sampling_layout(
        size_t ring_dimension, size_t limb_count, size_t rows, size_t columns,
        size_t full_ncol, size_t col_offset, int format, int dist_type,
        int device, GpuPreparedSamplingLayout *out);

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
    int gpu_matrix_prepare_sampling_with_layout(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t max_coefficient_bound,
        uint64_t coefficient_modulus,
        size_t full_ncol,
        size_t col_offset,
        const GpuMatrixRange *range,
        const GpuPreparedPlanDescriptor *layout,
        GpuPreparedSampling **plan);
    int gpu_matrix_submit_sampling(
        const GpuPreparedSampling *plan,
        gpu_chacha::GpuRngSeed seed);
    void gpu_matrix_destroy_sampling(GpuPreparedSampling *plan);

#ifdef __cplusplus
}
#endif
