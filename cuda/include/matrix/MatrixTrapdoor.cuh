#pragma once

#include "matrix/Matrix.cuh"

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
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    const uint8_t *tp2_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    size_t tp2_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    uint8_t tp2_coeff_bytes,
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
    int64_t **sampled_out_device,
    cudaEvent_t sampled_ready_event);

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    uint8_t *out_base,
    size_t out_stride_bytes,
    uint8_t out_coeff_bytes,
    size_t entry_count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id);

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_gauss_samp_gq_arb_base(
        GpuMatrix *src,
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
