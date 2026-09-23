#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuModulusConversionPlan GpuModulusConversionPlan;

    int gpu_matrix_crt_recompose(
        GpuMatrix *out,
        const GpuMatrix *const *levels,
        size_t level_count,
        const uint64_t *plaintext_moduli,
        const uint64_t *reconstruction_residues,
        size_t reconstruction_stride);

    int gpu_matrix_rns_conversion(
        GpuMatrix *out, const GpuMatrix *source, size_t digit_size,
        uint64_t plaintext_modulus, const uint64_t *scales,
        const uint64_t *inverses, const uint64_t *weights);

    int gpu_matrix_centered_rebase(GpuMatrix *out, const GpuMatrix *source);

    int gpu_matrix_centered_rebase_prepare(
        const GpuMatrix *out,
        const GpuMatrix *source,
        GpuModulusConversionPlan **out_plan);

    int gpu_matrix_centered_rebase_submit(
        GpuModulusConversionPlan *plan,
        GpuMatrix *out,
        const GpuMatrix *source,
        uint32_t source_binding_index,
        uint32_t output_binding_index);

    int gpu_matrix_centered_round_divide_prepare(
        const GpuMatrix *out,
        const GpuMatrix *source,
        const uint64_t *divisor_words,
        size_t divisor_word_count,
        GpuModulusConversionPlan **out_plan);

    int gpu_matrix_centered_round_divide_submit(
        GpuModulusConversionPlan *plan,
        GpuMatrix *out,
        const GpuMatrix *source,
        uint32_t source_binding_index,
        uint32_t output_binding_index);

    // Prepare the immutable CRT metadata before entering CUDA graph capture.
    // The returned plan owns stable pinned host and device copies until it is
    // destroyed. Submit reuses this metadata and patches only the dynamic
    // source/output descriptor pointers into the captured kernel arguments.
    int gpu_matrix_modulus_conversion_prepare(
        const GpuMatrix *out,
        const GpuMatrix *source,
        int round_scale,
        const uint64_t *division_inverses,
        size_t inverse_count,
        GpuModulusConversionPlan **out_plan);

    int gpu_matrix_modulus_conversion_submit(
        GpuModulusConversionPlan *plan,
        GpuMatrix *out,
        const GpuMatrix *source,
        uint32_t source_binding_index,
        uint32_t output_binding_index);

    // Arm the plan's stream-ordered release with the completion event for a
    // compiled graph replay. The wait is queued on the plan's free stream, so
    // destroying the plan remains asynchronous while protecting its opaque
    // metadata from an in-flight replay.
    int gpu_matrix_modulus_conversion_plan_protect_compiled_submission(
        GpuModulusConversionPlan *plan,
        GpuContext *context,
        int physical_device,
        void *launch_stream,
        void *completion_event);

    void gpu_matrix_modulus_conversion_plan_destroy(GpuModulusConversionPlan *plan);
    int gpu_matrix_modulus_conversion_plan_allocation_range(
        const GpuModulusConversionPlan *plan, uint64_t *address,
        size_t *bytes);

#ifdef __cplusplus
}
#endif
