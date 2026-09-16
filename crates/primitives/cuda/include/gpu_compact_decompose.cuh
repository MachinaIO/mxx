#pragma once
#include "matrix/Matrix.cuh"

struct GpuPreparedPlanDescriptor;

struct GpuPreparedCompactDecompose;
extern "C" {
// source is exclusive reusable scratch, fully overwritten by its producer on
// each invocation. Correction and inverse NTT may modify it in place.
int gpu_small_matrix_prepare_decompose(GpuMatrix *source, GpuSmallMatrix *output,
    uint32_t base_bits, bool small, size_t dropped_moduli,
    GpuPreparedCompactDecompose **out);
int gpu_small_matrix_prepare_decompose_with_layout(
    GpuMatrix *source, GpuSmallMatrix *output, uint32_t base_bits, bool small,
    size_t dropped_moduli, const GpuPreparedPlanDescriptor *layout,
    GpuPreparedCompactDecompose **out);
int gpu_small_matrix_submit_decompose(const GpuPreparedCompactDecompose *plan);
void gpu_small_matrix_destroy_decompose(GpuPreparedCompactDecompose *plan);
}
