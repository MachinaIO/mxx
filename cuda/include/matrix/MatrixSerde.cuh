#pragma once

#include "matrix/Matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

    int gpu_matrix_load_rns_batch(
        GpuMatrix *mat,
        const uint8_t *bytes,
        size_t bytes_per_poly,
        int format);
    int gpu_matrix_store_rns_batch(
        const GpuMatrix *mat,
        uint8_t *bytes_out,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events);

#ifdef __cplusplus
}
#endif
