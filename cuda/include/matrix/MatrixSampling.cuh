#pragma once

#include "matrix/Matrix.h"

#include <cuda_runtime.h>
#include <vector>

namespace
{
    int ensure_multi_limb_sampling_scratch(
        int device,
        size_t ptr_count,
        size_t limb_count,
        uint64_t ***dst_table,
        uint64_t **moduli_table,
        uint32_t **limb_indices_table);
} // namespace

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_sample_distribution(
        GpuMatrix *out,
        int dist_type,
        double sigma,
        uint64_t seed);

#ifdef __cplusplus
}
#endif
