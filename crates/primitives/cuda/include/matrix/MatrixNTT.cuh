#pragma once

typedef struct GpuMatrix GpuMatrix;

#ifdef __cplusplus
extern "C" {
#endif

int gpu_matrix_ntt_all(GpuMatrix *mat);
int gpu_matrix_ntt_all_on_stream(GpuMatrix *mat, cudaStream_t stream);
int gpu_matrix_intt_all_on_stream(GpuMatrix *mat, cudaStream_t stream);
int gpu_matrix_ntt_all_on_stream_bound(
    GpuMatrix *mat,
    cudaStream_t stream,
    uint32_t binding_index);
int gpu_matrix_intt_all_on_stream_bound(
    GpuMatrix *mat,
    cudaStream_t stream,
    uint32_t binding_index);
// Capture-aware in-place transforms. The binding index is forwarded to every
// descriptor patch registered by the transform; UINT32_MAX disables patching
// for ordinary non-captured execution.
int gpu_matrix_ntt_all_bound(GpuMatrix *mat, uint32_t binding_index);
int gpu_matrix_intt_all_bound(GpuMatrix *mat, uint32_t binding_index);
int gpu_matrix_intt_all(GpuMatrix *mat);

#ifdef __cplusplus
}
#endif
