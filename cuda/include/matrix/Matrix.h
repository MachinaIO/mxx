#pragma once

#include <stddef.h>
#include <stdint.h>

#include "Runtime.cuh"

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
#include <vector>

#include <cuda_runtime.h>

namespace FIDESlib
{
    namespace CKKS
    {
        class RNSPoly;
    }
}

enum class PolyFormat
{
    Coeff,
    Eval,
};

struct GpuPoly
{
    CKKS::RNSPoly *poly;
    GpuContext *ctx;
    int level;
    PolyFormat format;
};

struct GpuMatrix
{
    struct LimbU64Table
    {
        int limb;
        bool available;
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> entry_ptrs;
        uint64_t **device_entry_ptrs;
    };
    GpuContext *ctx;
    size_t rows;
    size_t cols;
    int level;
    PolyFormat format;
    std::vector<GpuPoly *> polys;
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
    std::vector<LimbU64Table> limb_u64_tables;
};
#else
typedef struct GpuPoly GpuPoly;
typedef struct GpuMatrix GpuMatrix;
#endif
