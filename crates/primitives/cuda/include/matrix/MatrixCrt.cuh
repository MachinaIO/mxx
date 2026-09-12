#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuMatrixTransformWorkspaceBytes
    {
        size_t workspace_bytes;
        size_t additional_bytes;
        size_t pinned_bytes;
        size_t alignment;
    } GpuMatrixTransformWorkspaceBytes;

    typedef enum GpuMatrixCrtOperation
    {
        GPU_MATRIX_CRT_CONVERT_MODULUS = 0,
        GPU_MATRIX_CRT_CENTERED_REBASE = 1,
        GPU_MATRIX_CRT_RECOMPOSE = 2,
        GPU_MATRIX_CRT_RNS_CONVERSION = 3,
    } GpuMatrixCrtOperation;

    // Workspace only: parameter tables, inputs and retained outputs are separate.
    int gpu_matrix_query_crt_workspace_bytes(
        const GpuContext *ctx, int level, size_t rows, size_t cols,
        GpuMatrixCrtOperation operation, size_t source_limb_count, size_t level_count,
        bool with_views, GpuMatrixTransformWorkspaceBytes *out);

    int gpu_matrix_crt_recompose(
        GpuMatrix *out,
        const GpuMatrix *const *levels,
        size_t level_count,
        const uint64_t *plaintext_moduli,
        const uint64_t *reconstruction_residues,
        size_t reconstruction_stride, const GpuMatrixRange *input_views, const GpuMatrixRange *output_view);

    int gpu_matrix_rns_conversion(
        GpuMatrix *out, const GpuMatrix *source, size_t digit_size,
        uint64_t plaintext_modulus, const uint64_t *scales,
        const uint64_t *inverses, const uint64_t *weights, const GpuMatrixBatchView *view);

    int gpu_matrix_centered_rebase(GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view);

    int gpu_matrix_convert_modulus(
        GpuMatrix *out,
        const GpuMatrix *source,
        int conversion,
        const uint64_t *division_inverses,
        size_t inverse_count,
        uint64_t plaintext_modulus,
        const uint64_t *input_scales, const GpuMatrixBatchView *view);

#ifdef __cplusplus
}

// One output-owned arena, reusable after all previous auxiliary consumers.
// Incomplete submissions retire their stream before any owner can be released.
struct MatrixTransformWorkspace
{
    MatrixTransformWorkspace();
    MatrixTransformWorkspace(const MatrixTransformWorkspace &) = delete;
    MatrixTransformWorkspace &operator=(const MatrixTransformWorkspace &) = delete;
    ~MatrixTransformWorkspace();
    int acquire(GpuMatrix *output, int device, cudaStream_t stream,
                const GpuMatrixTransformWorkspaceBytes &requirements);
    int upload();
    int complete();
    int retire();
    uint8_t *base;
    uint8_t *pinned;

private:
    GpuMatrix *owner;
    int device;
    cudaStream_t stream;
    size_t bytes;
    bool separate;
    bool completed;
    GpuDeviceWorkspace device_workspace;
    int release();
};
#endif
