#pragma once

typedef struct GpuMatrix GpuMatrix;
typedef struct GpuMatrixTransformPlan GpuMatrixTransformPlan;

typedef struct GpuMatrixRange GpuMatrixRange;
typedef struct GpuPreparedPlanDescriptor GpuPreparedPlanDescriptor;

typedef struct GpuPreparedNttLayout
{
    size_t ring_dimension;
    size_t limb_count;
    size_t polynomial_count;
    size_t launch_count;
    size_t stage_count;
    size_t workspace_bytes;
    size_t alignment;
    size_t event_count;
    size_t max_grid_x;
    size_t max_grid_y;
    size_t max_grid_z;
    int device;
    int forward;
} GpuPreparedNttLayout;

#ifdef __cplusplus
extern "C" {
#endif

int gpu_matrix_ntt_all(GpuMatrix *mat);
int gpu_matrix_intt_all(GpuMatrix *mat);

int gpu_matrix_query_ntt_layout(
    size_t ring_dimension, size_t limb_count, size_t polynomial_count,
    int device, bool forward, GpuPreparedNttLayout *out);

int gpu_matrix_prepare_ntt_plan(
    const GpuMatrix *mat, const GpuMatrixRange *range,
    bool forward, GpuMatrixTransformPlan **plan);
int gpu_matrix_prepare_ntt_plan_with_layout(
    const GpuMatrix *mat, const GpuMatrixRange *range, bool forward,
    const GpuPreparedPlanDescriptor *layout, GpuMatrixTransformPlan **plan);
int gpu_matrix_submit_ntt_plan(
    const GpuMatrixTransformPlan *plan, GpuMatrix *mat);
void gpu_matrix_destroy_ntt_plan(GpuMatrixTransformPlan *plan);

#ifdef __cplusplus
}
#endif
