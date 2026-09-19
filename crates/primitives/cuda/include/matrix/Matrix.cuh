#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "Runtime.cuh"

#ifdef __cplusplus
extern "C" {
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
        // This event is physically owned here and destroyed exactly once.
        cudaEvent_t write_done;
        // Local state index whose owned event currently dominates this limb.
        uint32_t completion_owner;
        // Producer ownership and last event-recording stream can differ.
        cudaStream_t last_write_stream;
        bool write_done_valid;
    };
    struct SharedLimbBuffer
    {
        struct DeviceDescriptor
        {
            uint8_t *base;
            size_t stride;
            uint8_t width;
        };
        int device;
        uint8_t *ptr;
        DeviceDescriptor *device_descriptors;
        size_t limb_count;
        size_t bytes_per_poly;
        size_t bytes_total;
        size_t n;
        std::vector<uint8_t> limb_coeff_bytes;
        std::vector<size_t> limb_offsets_bytes;
    };
    struct SharedAuxBuffer
    {
        int device;
        // Non-owning interior view of the corresponding SharedLimbBuffer allocation.
        void **ptr;
        size_t slots_per_poly;
        size_t slots_total;
    };
    std::vector<SharedLimbBuffer> shared_limb_buffers;
    std::vector<SharedAuxBuffer> shared_aux_buffers;
    std::vector<std::vector<LimbExecState>> exec_limb_states;
    // A deferred allocation stays private until its filling kernel is submitted.
    bool descriptors_initialized = true;
    // Actual writes invalidate this; reader lifetime joins remain independent.
    mutable std::atomic<bool> host_observed_writer_ready{false};
};
#endif

#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixCrt.cuh"
#include "matrix/MatrixData.cuh"
#include "matrix/MatrixDecompose.cuh"
#include "matrix/MatrixNTT.cuh"
#include "matrix/MatrixSampling.cuh"
#include "matrix/MatrixSerde.cuh"
#include "matrix/MatrixTrapdoor.cuh"
#include "matrix/MatrixUtils.cuh"
#include "matrix/MatrixSmallRhs.cuh"
