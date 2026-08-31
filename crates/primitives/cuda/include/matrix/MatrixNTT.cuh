#pragma once

typedef struct GpuMatrix GpuMatrix;

#ifdef __cplusplus
extern "C" {
#endif

int gpu_matrix_ntt_all(GpuMatrix *mat);
int gpu_matrix_intt_all(GpuMatrix *mat);
/// Forward-transforms packed homogeneous matrix views, independently for each
/// device partition.  The implementation derives each output address from
/// one packed base and an output stride, without host pointer arrays.
int gpu_matrix_ntt_contiguous_batch(GpuMatrix *const *matrices, size_t matrix_count);

#ifdef __cplusplus
}
#endif
