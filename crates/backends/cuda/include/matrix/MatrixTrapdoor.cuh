#pragma once

#include "ChaCha.cuh"
#include "matrix/Matrix.cuh"

typedef struct GpuP1CovarianceCache GpuP1CovarianceCache;

#ifdef __cplusplus
extern "C"
{
#endif

    // Reset and enqueue the one final status transfer for a retry region.
    // Neither function waits for device work. The caller records/uses its
    // existing stream event as the host success-gate boundary.
    int gpu_preimage_status_reset(MxxPreimageStatus *device_status, cudaStream_t stream);
    int gpu_preimage_status_copy_async(
        const MxxPreimageStatus *device_status,
        MxxPreimageStatus *host_status,
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

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
    gpu_chacha::GpuRngSeed seed,
    const std::vector<uint64_t> &out_moduli,
    int device,
    cudaStream_t stream);

int launch_sample_p1_integer_kernel(
    GpuContext *ctx,
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
    gpu_chacha::GpuRngSeed seed,
    cudaStream_t stream,
    int device_id,
    int64_t **sampled_out_device,
    cudaEvent_t sampled_ready_event);

int launch_scatter_p1_integer_to_limb_kernel_device(
    GpuContext *ctx,
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
        gpu_chacha::GpuRngSeed seed,
        GpuMatrix *out);

    int gpu_matrix_sample_p1_full(
        const GpuMatrix *a_mat,
        const GpuMatrix *b_mat,
        const GpuMatrix *d_mat,
        const GpuMatrix *tp2,
        double sigma,
        double s,
        double dgg_stddev,
        gpu_chacha::GpuRngSeed seed,
        GpuMatrix *out);

    int gpu_matrix_create_p1_covariance_cache(
        const GpuMatrix *a_mat,
        const GpuMatrix *b_mat,
        const GpuMatrix *d_mat,
        double sigma,
        double s,
        double dgg_stddev,
        GpuP1CovarianceCache **out_cache);

    void gpu_matrix_destroy_p1_covariance_cache(GpuP1CovarianceCache *cache);
    int gpu_matrix_initialize_preimage_retry(
        GpuP1CovarianceCache *cache, void *device_control, void *device_status,
        const void *control, void *stream);

    int gpu_matrix_sample_p1_full_cached(
        const GpuP1CovarianceCache *cache,
        const GpuMatrix *tp2,
        gpu_chacha::GpuRngSeed seed,
        GpuMatrix *out);

    // Allocation-free capture-body form.  sampled_out and workspace are
    // caller-owned fixed device storage prepared before graph capture.  The
    // seed is read on device for every retry iteration; no event, allocation,
    // free, host callback, or host transfer is performed here.
    int gpu_matrix_sample_p1_full_cached_into(
        const GpuP1CovarianceCache *cache,
        const GpuMatrix *tp2,
        const gpu_chacha::GpuRngSeed *device_seed,
        int64_t *sampled_out,
        void *workspace,
        size_t workspace_bytes,
        GpuMatrix *out,
        void *stream,
        uint32_t tp2_binding_index,
        uint32_t seed_binding_index,
        uint32_t sampled_binding_index,
        uint32_t workspace_binding_index);

    int gpu_matrix_gauss_samp_gq_arb_base_into(
        const GpuMatrix *src,
        GpuMatrix *out,
        const gpu_chacha::GpuRngSeed *device_seed,
        int64_t *sampled_digits,
        size_t sampled_capacity_bytes,
        uint32_t base_bits,
        double c,
        uint8_t *const *dst_bases_device,
        const size_t *dst_strides_device,
        const uint8_t *dst_coeff_bytes_device,
        const uint64_t *dst_moduli_device,
        void *stream,
        uint32_t src_binding_index,
        uint32_t out_bases_binding_index,
        uint32_t out_strides_binding_index,
        uint32_t out_coeff_bytes_binding_index,
        uint32_t out_moduli_binding_index,
        uint32_t seed_binding_index,
        uint32_t sampled_binding_index);

    int gpu_matrix_mul_vertical_pair(
        GpuMatrix *out,
        const GpuMatrix *top,
        const GpuMatrix *bottom,
        const GpuMatrix *rhs);
    int gpu_matrix_mul_vertical_pair_on_stream(
        GpuMatrix *out,
        const GpuMatrix *top,
        const GpuMatrix *bottom,
        const GpuMatrix *rhs,
        cudaStream_t stream);

    int gpu_matrix_preimage_residual(
        GpuMatrix *out,
        const GpuMatrix *target,
        const GpuMatrix *public_matrix,
        const GpuMatrix *p1,
        const GpuMatrix *p2);
    int gpu_matrix_preimage_residual_on_stream(
        GpuMatrix *out,
        const GpuMatrix *target,
        const GpuMatrix *public_matrix,
        const GpuMatrix *p1,
        const GpuMatrix *p2,
        cudaStream_t stream);

    int gpu_matrix_preimage_add_correction(
        GpuMatrix *out,
        const GpuMatrix *r,
        const GpuMatrix *e,
        const GpuMatrix *z);
    int gpu_matrix_preimage_add_correction_on_stream(
        GpuMatrix *out,
        const GpuMatrix *r,
        const GpuMatrix *e,
        const GpuMatrix *z,
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif
