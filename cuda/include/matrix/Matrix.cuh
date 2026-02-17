#pragma once

#include <vector>

#include "Poly.h"

#ifdef __cplusplus
struct GpuMatrix
{
    GpuContext *ctx;
    size_t rows;
    size_t cols;
    int level;
    PolyFormat format;
    struct SharedLimbBuffer
    {
        int device;
        uint64_t *ptr;
        size_t limb_count;
        size_t words_per_poly;
        size_t words_total;
        size_t n;
    };
    struct SharedAuxBuffer
    {
        int device;
        void **ptr;
        size_t slots_per_poly;
        size_t slots_total;
    };
    std::vector<SharedLimbBuffer> shared_limb_buffers;
    std::vector<SharedAuxBuffer> shared_aux_buffers;
};
#endif

#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixData.cuh"
#include "matrix/MatrixDecompose.cuh"
#include "matrix/MatrixSampling.cuh"
#include "matrix/MatrixSerde.cuh"
#include "matrix/MatrixTrapdoor.cuh"
#include "matrix/MatrixUtils.cuh"
