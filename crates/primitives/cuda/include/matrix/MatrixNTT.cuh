#pragma once

typedef struct GpuMatrix GpuMatrix;
typedef struct GpuMatrixTransformPlan GpuMatrixTransformPlan;

typedef struct GpuMatrixRange GpuMatrixRange;

#ifdef __cplusplus
extern "C" {
#endif

int gpu_matrix_ntt_all(GpuMatrix *mat);
int gpu_matrix_intt_all(GpuMatrix *mat);

int gpu_matrix_prepare_ntt_plan(
    const GpuMatrix *mat, const GpuMatrixRange *range,
    bool forward, GpuMatrixTransformPlan **plan);
int gpu_matrix_submit_ntt_plan(
    const GpuMatrixTransformPlan *plan, GpuMatrix *mat);
void gpu_matrix_destroy_ntt_plan(GpuMatrixTransformPlan *plan);

#ifdef __cplusplus
}
#endif
