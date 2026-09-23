#pragma once

#include "matrix/Matrix.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuModulusConversionPlan GpuModulusConversionPlan;

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
