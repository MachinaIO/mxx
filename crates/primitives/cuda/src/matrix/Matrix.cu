#include "matrix/Matrix.cuh"
#include "ChaCha.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>

#include "../ChaCha.cu"
#include "MatrixUtils.cu"
#include "MatrixNTT.cu"
#include "MatrixNTTBatch.cu"
#include "MatrixArith.cu"
#include "MatrixArithBatch.cu"
#include "MatrixData.cu"
#include "MatrixDecompose.cu"
#include "MatrixSampling.cu"
#include "MatrixTrapdoor.cu"
#include "MatrixSerde.cu"
#include "MatrixSerdeBatch.cu"
#include "MatrixCrt.cu"
#include "MatrixSmallRhs.cu" // compact bounded RHS implementation and staged kernels

// Every native kernel specialization reachable from this translation unit is
// loaded before setup residency is accepted. Keep this explicit inventory in
// sync with launch sites, including inferred Word/Layout template arguments.
// cudaFuncGetAttributes forces kernel loading without executing a sample:
// https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
int gpu_matrix_prepare_kernels(GpuKernelPartition *partition)
{
    if (!partition) return set_error("missing kernel provisioning output");
    const void *const kernels[] = {
        // MatrixNTT.cu
        reinterpret_cast<const void *>(ntt_twist_all_limbs_kernel),
        reinterpret_cast<const void *>(ntt_scale_all_limbs_kernel),
        reinterpret_cast<const void *>(ntt_stage_all_limbs_kernel<false>),
        reinterpret_cast<const void *>(ntt_stage_all_limbs_kernel<true>),
        reinterpret_cast<const void *>(ntt_fused_local_stages_kernel<false, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_local_stages_kernel<false, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_local_stages_kernel<true, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_local_stages_kernel<true, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 2, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 2, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 4, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 4, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 8, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 8, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 16, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<false, 16, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 2, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 2, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 4, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 4, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 8, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 8, MatrixNttBatchDescriptors>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 16, MatrixNttDescriptorView>),
        reinterpret_cast<const void *>(ntt_fused_top_stages_kernel<true, 16, MatrixNttBatchDescriptors>),
        // MatrixNTTBatch.cu
        reinterpret_cast<const void *>(batch_ntt_twist_kernel),
        reinterpret_cast<const void *>(batch_ntt_bit_reverse_kernel),
        reinterpret_cast<const void *>(batch_ntt_stage_kernel),
        reinterpret_cast<const void *>(batch_ntt_first_stage_out_of_place_kernel),
        reinterpret_cast<const void *>(batch_ntt_scale_twist_kernel),
        // MatrixArith.cu
        reinterpret_cast<const void *>(transpose_all_limbs_kernel),
        reinterpret_cast<const void *>(sum_rows_all_limbs_kernel),
        reinterpret_cast<const void *>(add_row_blocks_all_limbs_kernel),
        reinterpret_cast<const void *>(small_dot_all_limbs_kernel),
        reinterpret_cast<const void *>(tensor_all_limbs_kernel),
        reinterpret_cast<const void *>(tensor_sum_rows_all_limbs_kernel<false>),
        reinterpret_cast<const void *>(tensor_sum_rows_all_limbs_kernel<true>),
        reinterpret_cast<const void *>(block_elementwise_all_limbs_kernel),
        reinterpret_cast<const void *>(block_copy_rect_all_limbs_kernel),
        reinterpret_cast<const void *>(block_add_rect_all_limbs_kernel),
        reinterpret_cast<const void *>(block_matmul_kernel),
        reinterpret_cast<const void *>(block_thin_row_matmul_kernel),
        reinterpret_cast<const void *>(block_equal_kernel),
        // MatrixArithBatch.cu
        reinterpret_cast<const void *>(matrix_binary_batch_kernel),
        reinterpret_cast<const void *>(matrix_negate_batch_kernel),
        reinterpret_cast<const void *>(matrix_ring_automorphism_batch_kernel),
        reinterpret_cast<const void *>(matrix_scalar_mul_batch_kernel),
        reinterpret_cast<const void *>(matrix_matmul_batch_kernel),
        reinterpret_cast<const void *>(matrix_thin_row_matmul_batch_kernel),
        reinterpret_cast<const void *>(matrix_thin_row_mul_accumulate_batch_kernel),
        reinterpret_cast<const void *>(matrix_accumulate_range_kernel),
        // MatrixData.cu
        reinterpret_cast<const void *>(initialize_device_descriptors),
        // MatrixDecompose.cu
        reinterpret_cast<const void *>(matrix_decompose_all_slots_kernel),
        reinterpret_cast<const void *>(matrix_fill_constant_columns_kernel),
        reinterpret_cast<const void *>(matrix_fill_small_decomposed_identity_chunk_all_limbs_kernel),
        reinterpret_cast<const void *>(gadget_low_constants_kernel),
        reinterpret_cast<const void *>(gadget_correct_residues_kernel),
        // MatrixSampling.cu
        reinterpret_cast<const void *>(matrix_sample_distribution_multi_limb_kernel),
        // MatrixTrapdoor.cu
        reinterpret_cast<const void *>(matrix_mul_vertical_pair_kernel),
        reinterpret_cast<const void *>(matrix_preimage_residual_kernel),
        reinterpret_cast<const void *>(matrix_preimage_add_correction_top_kernel),
        reinterpret_cast<const void *>(matrix_preimage_add_correction_bottom_kernel),
        reinterpret_cast<const void *>(matrix_precompute_p1_covariance_kernel),
        reinterpret_cast<const void *>(matrix_sample_p1_integer_cached_kernel_small),
        reinterpret_cast<const void *>(matrix_sample_p1_integer_cached_kernel_large),
        reinterpret_cast<const void *>(matrix_sample_p1_integer_kernel_small),
        reinterpret_cast<const void *>(matrix_sample_p1_integer_kernel_large),
        reinterpret_cast<const void *>(matrix_scatter_p1_integer_to_limb_kernel),
        reinterpret_cast<const void *>(matrix_gauss_samp_gq_arb_base_sample_kernel),
        reinterpret_cast<const void *>(matrix_gauss_samp_gq_arb_base_scatter_kernel),
        // MatrixSerde.cu
        reinterpret_cast<const void *>(serde_pack_u64_limbs_to_packed_kernel),
        reinterpret_cast<const void *>(serde_unpack_packed_limbs_to_u64_kernel),
        reinterpret_cast<const void *>(serde_reconstruct_rns_to_words_kernel),
        reinterpret_cast<const void *>(serde_center_coeff_words_kernel),
        reinterpret_cast<const void *>(serde_check_centered_bound_batch_kernel),
        reinterpret_cast<const void *>(serde_pack_centered_coeff_words_bits_kernel),
        reinterpret_cast<const void *>(serde_unpack_packed_coeffs_mod_kernel),
        // MatrixSerdeBatch.cu
        reinterpret_cast<const void *>(serde_reconstruct_rns_batch_to_words_kernel),
        reinterpret_cast<const void *>(serde_center_coeff_words_batch_kernel),
        reinterpret_cast<const void *>(serde_pack_centered_coeff_words_batch_kernel),
        // MatrixCrt.cu
        reinterpret_cast<const void *>(convert_modulus_kernel),
        reinterpret_cast<const void *>(convert_modulus_range_kernel),
        reinterpret_cast<const void *>(centered_rebase_kernel),
        reinterpret_cast<const void *>(crt_recompose_kernel),
        reinterpret_cast<const void *>(crt_recompose_range_kernel),
        reinterpret_cast<const void *>(crt_recompose_setup_kernel),
        reinterpret_cast<const void *>(rns_setup_kernel),
        reinterpret_cast<const void *>(rns_compact_conversion_kernel),
        reinterpret_cast<const void *>(rns_conversion_kernel),
        // MatrixSmallRhs.cu
        reinterpret_cast<const void *>(compact_decompose_kernel),
        reinterpret_cast<const void *>(compact_rhs_dif_first_kernel<uint32_t>),
        reinterpret_cast<const void *>(compact_rhs_dif_first_kernel<uint64_t>),
        reinterpret_cast<const void *>(compact_ntt_dif_stage_kernel<uint32_t>),
        reinterpret_cast<const void *>(compact_ntt_dif_stage_kernel<uint64_t>),
        reinterpret_cast<const void *>(compact_rhs_dif_all_shared_kernel<uint32_t>),
        reinterpret_cast<const void *>(compact_rhs_dif_all_shared_kernel<uint64_t>),
        reinterpret_cast<const void *>(compact_rhs_dif_suffix_kernel<uint32_t>),
        reinterpret_cast<const void *>(compact_rhs_dif_suffix_kernel<uint64_t>),
        reinterpret_cast<const void *>(compact_accumulate_kernel<uint32_t>),
        reinterpret_cast<const void *>(compact_accumulate_kernel<uint64_t>),
        reinterpret_cast<const void *>(compact_check_pack_preimage_kernel),
        reinterpret_cast<const void *>(compact_commit_preimage_tile_kernel),
    };
    for (const void *kernel : kernels) {
        cudaFuncAttributes attributes{};
        const cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
        if (error != cudaSuccess) return set_error(error);
    }
    cudaError_t error = cudaGetDriverEntryPointByVersion(
        "cuLaunchKernel", &partition->launch_entry, 12000, cudaEnableLegacyStream);
    if (error != cudaSuccess) return set_error(error);
    if (!partition->launch_entry) return set_error("cuLaunchKernel entry point unavailable");
    error = cudaGetFuncBySymbol(&partition->tensor_row_sum[0],
        reinterpret_cast<const void *>(tensor_sum_rows_all_limbs_kernel<false>));
    if (error == cudaSuccess)
        error = cudaGetFuncBySymbol(&partition->tensor_row_sum[1],
            reinterpret_cast<const void *>(tensor_sum_rows_all_limbs_kernel<true>));
    return error == cudaSuccess ? 0 : set_error(error);
}
