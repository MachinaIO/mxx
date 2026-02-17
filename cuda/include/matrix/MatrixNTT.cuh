#pragma once

typedef struct GpuMatrix GpuMatrix;

#ifdef __cplusplus
extern "C" {
#endif

int gpu_matrix_ntt_all(GpuMatrix *mat, int batch);
int gpu_matrix_intt_all(GpuMatrix *mat, int batch);

#ifdef __cplusplus
}
#endif
