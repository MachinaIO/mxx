#pragma once

#include <stdint.h>
#include <vector>

#include <cuda_runtime.h>

#include "GpuPoly.h"

// FIDESlib headers (expected under third_party/FIDESlib/include).
#include "../third_party/FIDESlib/include/CKKS/Context.cuh"
#include "../third_party/FIDESlib/include/CKKS/Parameters.cuh"
#include "../third_party/FIDESlib/include/LimbUtils.cuh"
#include "../third_party/FIDESlib/include/CKKS/RNSPoly.cuh"

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
};

extern "C" int gpu_set_last_error(const char *msg);
