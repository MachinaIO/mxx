#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_transpose(GpuMatrix *out, const GpuMatrix *source);
    int gpu_matrix_sum_rows(
        GpuMatrix *out, const GpuMatrix *const *sources, const size_t *source_starts,
        const size_t *widths, const size_t *destination_starts, size_t source_count,
        const size_t *rows, const size_t *offsets,
        size_t group_count, size_t term_count);
    struct MxxRowAddFragment {
        const GpuMatrix *lhs;
        const GpuMatrix *rhs;
        size_t lhs_column, rhs_row, rhs_column;
        size_t destination_row, destination_column, rows, columns;
    };
    int gpu_matrix_add_row_blocks(
        GpuMatrix *out, const MxxRowAddFragment *fragments, size_t fragment_count);
    int gpu_matrix_add_block(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols);
    int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_tensor(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs);
    int gpu_matrix_tensor_sum_rows(
        GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
        const size_t *rows, const size_t *offsets, size_t group_count, size_t term_count);
    // Fused tensor row sums for several planned output owners.  The row
    // groups are concatenated in `rows`/`offsets`; `output_offsets` maps each
    // output owner to its contiguous range of row groups.  The native local
    // graph ABI is lhs=0, rhs=1, outputs=2+j.
    int gpu_matrix_tensor_sum_rows_batch(
        GpuMatrix *const *outputs, size_t output_count,
        const GpuMatrix *lhs, const GpuMatrix *rhs,
        const size_t *rows, const size_t *offsets, const size_t *output_offsets,
        size_t group_count, size_t term_count);
    int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal);
    int gpu_matrix_mul_scalar(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar);
    int gpu_matrix_binary_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count,
        int operation);

    // A physical lane layout is a typed view of one matrix owner.  The
    // descriptor/data pair is deliberately carried together per CRT limb so
    // capture replay cannot accidentally bind a descriptor table from one
    // allocation to the data slab of another allocation.  `limbs` is a host
    // array owned by the caller and is copied into capture-owned metadata by
    // gpu_matrix_binary_lane_batch_layout.
    typedef struct GpuMatrixLaneLimbLayout
    {
        const uint8_t *data;
        const void *descriptors;
        size_t stride_bytes;
        uint8_t coefficient_bytes;
    } GpuMatrixLaneLimbLayout;

    typedef struct GpuMatrixLanePhysicalLayout
    {
        const GpuMatrix *owner;
        const GpuMatrixLaneLimbLayout *limbs;
        size_t limb_count;
        size_t rows;
        size_t columns;
    } GpuMatrixLanePhysicalLayout;

    int gpu_matrix_lane_physical_layout(
        const GpuMatrix *owner,
        GpuMatrixLaneLimbLayout *limbs,
        size_t limb_capacity,
        size_t *out_limb_count,
        size_t *out_rows,
        size_t *out_columns);

    // Add/subtract active lanes in one launch.  `lane_count` is the capacity
    // of the preallocated wave, `active_count` is the number of published
    // lanes, and `active_lane_mask` identifies exactly which lanes are active
    // (including tail waves).  For broadcast RHS, right_layouts contains one
    // layout and is reused for every active lane; otherwise it contains one
    // layout per lane.  Inactive output layouts may be null and are untouched.
    // No host result transfer is performed; the return value is the native
    // submission/status code.
    int gpu_matrix_binary_lane_batch_layout(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *left_layouts,
        const GpuMatrixLanePhysicalLayout *right_layouts,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        int operation,
        int rhs_broadcast,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count);

    int gpu_matrix_mul_lane_batch_layout(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *left_layouts,
        const GpuMatrixLanePhysicalLayout *right_layouts,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count);

    int gpu_matrix_negate_lane_batch_layout(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *inputs,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count);

    int gpu_matrix_scalar_mul_lane_batch_layout(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *inputs,
        const GpuMatrixLanePhysicalLayout *scalars,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        int scalar_broadcast,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count);

    int gpu_matrix_transpose_lane_batch_layout(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *inputs,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count);

    // Capture-safe refresh of a family descriptor pointer table.  Each table
    // entry is patched from the corresponding live source binding on replay;
    // no host pointer-array rebuild is needed after capture.
    int gpu_matrix_family_descriptor_table_bind_live_sources(
        GpuContext *ctx,
        void *destination,
        const uint64_t *source_descriptors,
        size_t source_count,
        const uint32_t *source_binding_indices,
        size_t source_binding_count,
        cudaStream_t stream);
    int gpu_matrix_negate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        size_t matrix_count);
    int gpu_matrix_mul_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count);
    int gpu_matrix_validate_ring_automorphism(
        size_t ring_dimension,
        const size_t *indices,
        size_t matrix_count);
    int gpu_matrix_ring_automorphism_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        const size_t *indices,
        size_t matrix_count);
    int gpu_matrix_mul_accumulate_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        const GpuMatrix *const *coefficients,
        const GpuMatrix *const *biases,
        const size_t *inner_dimensions,
        size_t matrix_count,
        size_t product_count);
    int gpu_matrix_mul_scalar_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *matrices,
        const GpuMatrix *const *scalars,
        size_t matrix_count);
    // Null inputs transform the exclusively owned outputs in place.
    int gpu_matrix_intt_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *inputs,
        size_t matrix_count);
    int gpu_matrix_ntt_in_place_batch(
        GpuMatrix *const *matrices,
        size_t matrix_count);

#ifdef __cplusplus
}
#endif
