#pragma once

#include "matrix/Matrix.h"

#include <cuda_runtime.h>
#include <vector>

namespace
{
    int launch_gauss_samp_gq_arb_base_multi_kernel(
        const std::vector<const uint64_t *> &src_ptrs,
        const std::vector<uint64_t *> &dst_ptrs,
        size_t poly_count,
        size_t n,
        const std::vector<uint64_t> &tower_moduli,
        uint32_t base_bits,
        uint32_t digits_per_tower,
        const std::vector<uint32_t> &digit_indices,
        double c,
        const std::vector<uint32_t> &tower_indices,
        uint64_t seed,
        const std::vector<uint64_t> &out_moduli,
        int device,
        cudaStream_t stream);

    int launch_sample_p1_integer_kernel(
        const std::vector<const uint64_t *> &a_entries,
        const std::vector<const uint64_t *> &b_entries,
        const std::vector<const uint64_t *> &d_entries,
        const std::vector<const uint64_t *> &tp2_entries,
        size_t d,
        size_t cols,
        size_t n,
        uint64_t modulus,
        double sigma,
        double s,
        double dgg_stddev,
        uint64_t seed,
        cudaStream_t stream,
        int device_id,
        std::vector<int64_t> &sampled_out_host);

    int launch_scatter_p1_integer_to_limb_kernel(
        const std::vector<int64_t> &sampled_in_host,
        const std::vector<uint64_t *> &out_entries,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream,
        int device_id);
} // namespace

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_gauss_samp_gq_arb_base(
        const GpuMatrix *src,
        uint32_t base_bits,
        double c,
        double dgg_stddev,
        uint64_t seed,
        GpuMatrix *out);
    int gpu_matrix_sample_p1_full(
        const GpuMatrix *a_mat,
        const GpuMatrix *b_mat,
        const GpuMatrix *d_mat,
        const GpuMatrix *tp2,
        double sigma,
        double s,
        double dgg_stddev,
        uint64_t seed,
        GpuMatrix *out);

#ifdef __cplusplus
}
#endif
