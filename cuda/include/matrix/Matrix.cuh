#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "Runtime.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuMatrix GpuMatrix;

    typedef enum GpuPolyFormat
    {
        GPU_POLY_FORMAT_COEFF = 0,
        GPU_POLY_FORMAT_EVAL = 1,
    } GpuPolyFormat;

    typedef enum GpuMatrixSampleDist
    {
        GPU_MATRIX_DIST_UNIFORM = 0,
        GPU_MATRIX_DIST_GAUSS = 1,
        GPU_MATRIX_DIST_BIT = 2,
        GPU_MATRIX_DIST_TERNARY = 3,
    } GpuMatrixSampleDist;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
struct GpuMatrix
{
    GpuContext *ctx;
    size_t rows;
    size_t cols;
    int level;
    GpuPolyFormat format;
    struct LimbExecState
    {
        int device;
        cudaStream_t stream;
        cudaEvent_t write_done;
        bool write_done_valid;
    };
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
    std::vector<std::vector<LimbExecState>> exec_limb_states;
};
#endif

#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixData.cuh"
#include "matrix/MatrixDecompose.cuh"
#include "matrix/MatrixNTT.cuh"
#include "matrix/MatrixSampling.cuh"
#include "matrix/MatrixSerde.cuh"
#include "matrix/MatrixTrapdoor.cuh"
#include "matrix/MatrixUtils.cuh"
