#pragma once

#include <memory>
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
    std::vector<std::unique_ptr<GpuPoly>> polys;
    struct SharedLimbBuffer
    {
        int device;
        uint64_t *ptr;
    };
    struct SharedAuxBuffer
    {
        int device;
        void **ptr;
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
