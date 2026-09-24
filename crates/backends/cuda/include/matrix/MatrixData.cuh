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

    // Exact owner-derived address of one ordered CRT limb. The first half of
    // each polynomial record is the observable coefficient/evaluation value;
    // the second half is native NTT scratch, not another persistent format.
    typedef struct GpuMatrixBindingLimb
    {
        int physical_device;
        size_t crt_limb_index;
        size_t component_index;
        size_t local_limb_index;
        size_t byte_offset;
        size_t coefficient_bytes;
        size_t poly_stride_bytes;
        size_t row_stride_bytes;
        size_t scratch_offset_bytes;
        size_t data_bytes;
        uint64_t modulus;
        void *data;
    } GpuMatrixBindingLimb;

    int gpu_matrix_query_allocation_bytes(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        GpuMatrixAllocationBytes *out);
    int gpu_matrix_binding_component_count(
        const GpuMatrix *mat,
        size_t *out_count);
    int gpu_matrix_binding_component(
        const GpuMatrix *mat,
        size_t component_index,
        GpuMatrixBindingComponent *out);
    int gpu_matrix_binding_limb_count(const GpuMatrix *mat, size_t *out_count);
    int gpu_matrix_binding_limb(const GpuMatrix *mat,
        size_t crt_limb_index, GpuMatrixBindingLimb *out);
    int gpu_matrix_binding_layout(
        const GpuContext *ctx, int level, size_t rows, size_t cols,
        GpuMatrixBindingLimb *out_limbs, size_t capacity,
        size_t *out_limb_count, size_t *out_data_bytes);

    int gpu_matrix_create(
        GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
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

    int gpu_matrix_record_compiled_write(
        GpuMatrix *mat,
        void *stream);

#ifdef __cplusplus
}
#endif
