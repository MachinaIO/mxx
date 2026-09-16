#pragma once
#include "matrix/Matrix.cuh"
#include "ChaCha.cuh"

struct GpuPreparedPreimagePhases;
struct GpuPreparedPreimageCutoff;
struct GpuPreparedSampling;
extern "C" {
int gpu_preimage_phase_layout(GpuContext *ctx, size_t rows, size_t columns, GpuPreparedWorkspaceLayout *layouts);
int gpu_preimage_prepare_phases(const GpuMatrix *a, const GpuMatrix *b, const GpuMatrix *d,
    const GpuMatrix *product, GpuMatrix *p1, const GpuMatrix *residual, GpuMatrix *gadget,
    uint32_t base_bits, double c, double smoothing, double sigma,
    GpuPreparedPreimagePhases **out);
int gpu_preimage_refresh_covariance(GpuPreparedPreimagePhases *plan);
int gpu_preimage_submit_p1(GpuPreparedPreimagePhases *plan, GpuRngSeed seed);
int gpu_preimage_submit_gadget(GpuPreparedPreimagePhases *plan, GpuRngSeed seed);
void gpu_preimage_destroy_phases(GpuPreparedPreimagePhases *plan);
int gpu_preimage_mask_phases(GpuPreparedPreimagePhases *plan,
    const GpuPreparedPreimageCutoff *cutoff, size_t job);
int gpu_preimage_mask_sampling(GpuPreparedSampling *plan,
    const GpuPreparedPreimageCutoff *cutoff, size_t job);
}
