#pragma once

#include <stdint.h>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

#include "Poly.h"

// FIDESlib headers (resolved via include path).
#include "CKKS/Context.cuh"
#include "CKKS/Parameters.cuh"
#include "LimbUtils.cuh"
#include "CKKS/RNSPoly.cuh"

namespace CKKS = FIDESlib::CKKS;

enum class PolyFormat
{
    Coeff,
    Eval,
};

struct GpuContext
{
    CKKS::Context *ctx;
    std::vector<uint64_t> moduli;
    int N;
    std::vector<int> gpu_ids;
    uint32_t batch;
    std::vector<uint64_t> garner_inverse_table;
    std::mutex transform_mutex;
};

struct GpuPoly
{
    CKKS::RNSPoly *poly;
    GpuContext *ctx;
    int level;
    PolyFormat format;
};

struct GpuEventSet
{
    struct Entry
    {
        cudaEvent_t event;
        int device;
    };
    std::vector<Entry> entries;
};

struct GpuMatrix
{
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
};

extern "C" int gpu_set_last_error(const char *msg);
