#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    struct GpuPreparedModulusConversion;
    struct GpuPreparedCrtRecompose;
    struct GpuPreparedCenteredRebase;

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

    int gpu_matrix_prepare_crt_recompose(
        const GpuMatrix *const *levels, size_t level_count,
        const uint64_t *plaintext_moduli, const uint64_t *reconstruction_residues,
        size_t reconstruction_stride, GpuMatrix *out,
        GpuPreparedCrtRecompose **plan);
    int gpu_matrix_submit_crt_recompose(
        const GpuPreparedCrtRecompose *plan,
        const GpuMatrix *const *levels, size_t level_count);
    void gpu_matrix_destroy_prepared_crt_recompose(GpuPreparedCrtRecompose *plan);

    int gpu_matrix_rns_conversion(
        GpuMatrix *out, const GpuMatrix *source, size_t digit_size,
        uint64_t plaintext_modulus, const uint64_t *scales,
        const uint64_t *inverses, const uint64_t *weights, const GpuMatrixBatchView *view);

    int gpu_matrix_centered_rebase(GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view);
    int gpu_matrix_prepare_centered_rebase(
        GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
        GpuPreparedCenteredRebase **plan);
    int gpu_matrix_submit_centered_rebase(const GpuPreparedCenteredRebase *plan);
    void gpu_matrix_destroy_centered_rebase(GpuPreparedCenteredRebase *plan);

    int gpu_matrix_convert_modulus(
        GpuMatrix *out,
        const GpuMatrix *source,
        int conversion,
        const uint64_t *division_inverses,
        size_t inverse_count,
        uint64_t plaintext_modulus,
        const uint64_t *input_scales, const GpuMatrixBatchView *view);

    int gpu_matrix_prepare_modulus_conversion(
        const GpuMatrix *source, const GpuMatrix *out, int conversion,
        const uint64_t *division_inverses, size_t inverse_count,
        uint64_t plaintext_modulus, const uint64_t *input_scales,
        GpuPreparedModulusConversion **plan);

    int gpu_matrix_prepare_rns_conversion(
        const GpuMatrix *source, const GpuMatrix *out, size_t digit_size,
        uint64_t plaintext_modulus, const uint64_t *scales,
        const uint64_t *inverses, size_t inverse_count,
        GpuPreparedModulusConversion **plan);

    int gpu_matrix_submit_modulus_conversion(
        const GpuPreparedModulusConversion *plan, GpuMatrix *out,
        const GpuMatrix *source, const GpuMatrixBatchView *view,
        bool apply_output_transform);

    void gpu_matrix_destroy_prepared_modulus_conversion(
        GpuPreparedModulusConversion *plan);

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
    int acquire_persistent(GpuMatrix *output, int device, cudaStream_t stream,
                           const GpuMatrixTransformWorkspaceBytes &requirements);
    int begin_replay();
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
    bool persistent;
    GpuDeviceWorkspace device_workspace;
    int release();
};
#endif
