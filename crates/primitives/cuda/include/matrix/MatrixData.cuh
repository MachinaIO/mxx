#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuMatrixAllocationBytes
    {
        size_t data_bytes;
        size_t aux_bytes;
        size_t event_bytes;
        size_t total_bytes;
    } GpuMatrixAllocationBytes;

    // Stable owner-free projection of one physical matrix allocation. The
    // addresses are borrowed for the lifetime of the GpuMatrix and are used
    // only to resolve registered graph bindings at replay time.
    typedef struct GpuMatrixBindingComponent
    {
        int physical_device;
        size_t limb_count;
        size_t bytes_per_poly;
        size_t data_bytes;
        size_t n;
        size_t device_descriptor_stride;
        void *data;
        void *device_descriptors;
        void *auxiliary;
        size_t aux_slots_per_poly;
        size_t aux_slots_total;
    } GpuMatrixBindingComponent;

    int gpu_matrix_query_allocation_bytes(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrixAllocationBytes *out);
    int gpu_matrix_binding_component_count(
        const GpuMatrix *mat,
        size_t *out_count);
    int gpu_matrix_binding_component(
        const GpuMatrix *mat,
        size_t component_index,
        GpuMatrixBindingComponent *out);

    int gpu_matrix_create(
        GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrix **out,
        bool initialize_descriptors = true);
    void gpu_matrix_destroy(GpuMatrix *mat);
    int gpu_matrix_wait(const GpuMatrix *mat);
    // Asynchronous compiled-runtime lifetime adapters. The stream and event
    // handles are non-owning CUDA handles represented as void* at this ABI
    // boundary; ownership remains with the execution context/native event.
    int gpu_matrix_wait_compiled_inputs(
        const GpuMatrix *mat,
        int consumer_device,
        void *consumer_stream,
        bool read_only);
    int gpu_matrix_track_compiled_consumer(
        const GpuMatrix *mat,
        int consumer_device,
        void *consumer_stream,
        void *completion_event,
        bool read_only);
    int gpu_matrix_record_compiled_write(
        GpuMatrix *mat,
        void *stream);
    // Preparation-only capture helper. Resolves current writer events before
    // capture and invalidates only those completed pre-capture dependencies.
    int gpu_matrix_prepare_external_for_capture(GpuMatrix *mat);
    int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src);
    int gpu_matrix_copy_peer(GpuMatrix *dst, const GpuMatrix *src, int *out_copied);
    // Check whether a matrix created for `dst_ctx` would satisfy the exact
    // peer-copy contract without allocating a destination, enabling peer
    // access, enqueueing a copy, or mutating either matrix/context.
    int gpu_matrix_copy_peer_query(
        const GpuMatrix *src,
        const GpuContext *dst_ctx,
        int *out_compatible);
    int gpu_matrix_copy_block(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols);
    int gpu_matrix_copy_block_on_stream(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols,
        cudaStream_t stream);
    // Capture-only variant. Every source and destination limb pointer is
    // graph-bound using contiguous source_binding_base + limb and
    // destination_binding_base + limb local identities.
    int gpu_matrix_copy_block_on_capture_stream(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t dst_row,
        size_t dst_col,
        size_t src_row,
        size_t src_col,
        size_t rows,
        size_t cols,
        cudaStream_t stream,
        uint32_t source_binding_base,
        uint32_t destination_binding_base);

    // Gather matrix lanes from a device-resident family descriptor table.
    // `family_descriptors` points to a device array of `family_count` device
    // pointers; each pointer addresses one owner's local-limb descriptor
    // array.  The table is intentionally independent from GpuMatrix owners,
    // so Arc-backed owners can be retained by the Rust wrapper without
    // rebuilding host pointer arrays on graph replay.
    //
    // The destination is laid out as `lane_count` adjacent source matrices:
    // destination columns = lane_count * source_columns.  For lane `l`, the
    // signed index is read at `wave_base + lane_offset + l`.  `width` bounds
    // the one-dimensional grid (zero is rejected).  Invalid signed arithmetic
    // or bounds write zero for the complete lane and atomically latch
    // MXX_GPU_CONTROL_INVALID_INDEX when status is supplied.  This operation
    // never clears status, allowing the latch to survive later kernels.
    int gpu_matrix_gather_family(
        GpuMatrix *destination,
        const void *family_descriptors,
        size_t family_count,
        const int64_t *indices,
        size_t index_count,
        int64_t wave_base,
        int64_t lane_offset,
        size_t lane_count,
        size_t source_columns,
        size_t width,
        cudaStream_t stream,
        uint32_t *status,
        size_t destination_partition,
        uint32_t destination_binding_index,
        uint32_t indices_binding_index,
        uint32_t status_binding_index);

    // Gather one source-family element into each independent lane destination
    // in a single launch. `indices[lane]` is local to the wave; semantic_offset
    // is applied to the selected value, never to the index pointer. The
    // destination binding array supplies one explicit graph binding identity
    // for each lane descriptor pointer.
    int gpu_matrix_gather_family_lanes(
        const void *family_descriptors,
        size_t family_count,
        const GpuMatrix *const *destinations,
        size_t destination_count,
        const int64_t *indices,
        size_t index_count,
        size_t active_count,
        uint64_t active_lane_mask,
        int64_t semantic_offset,
        size_t width,
        int physical_device,
        cudaStream_t stream,
        uint32_t *status,
        uint32_t indices_binding_index,
        uint32_t status_binding_index,
        const uint32_t *destination_binding_indices);

#ifdef __cplusplus
}
#endif
