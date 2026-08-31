#pragma once

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <mutex>
#include <utility>
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
        cudaEvent_t write_done;
        bool write_done_valid;
    };
    struct SharedAllocation;
    struct SharedLimbBuffer
    {
        int device;
        uint8_t *ptr;
        size_t limb_count;
        size_t bytes_per_poly;
        size_t bytes_total;
        size_t n;
        std::vector<uint8_t> limb_coeff_bytes;
        std::vector<size_t> limb_offsets_bytes;
        std::shared_ptr<SharedAllocation> allocation;
    };
    struct SharedAuxBuffer
    {
        int device;
        void **ptr;
        size_t slots_per_poly;
        size_t slots_total;
        std::shared_ptr<SharedAllocation> allocation;
    };
    /// A packed allocation may back multiple matrix handles.  Each handle
    /// stores only its view pointer; the allocation base is released once the
    /// final view has been destroyed, after all recorded write events.
    struct SharedAllocation
    {
        int device = -1;
        uint8_t *limb_base = nullptr;
        void **aux_base = nullptr;
        size_t limb_bytes = 0;
        size_t aux_bytes = 0;
        cudaEvent_t allocation_ready = nullptr;
        cudaStream_t release_stream = nullptr;
        size_t live_views = 1;
        bool free_queued = false;
        // Set when CUDA cannot provide a completion event after work was
        // submitted.  In that fail-closed case the storage is intentionally
        // leaked rather than risking an early free/use-after-free.
        bool release_blocked = false;
        bool release_stream_owned = false;
        std::mutex mutex;
        std::vector<std::pair<int, cudaEvent_t>> pending_write_events;

        ~SharedAllocation();
    };
    std::vector<SharedLimbBuffer> shared_limb_buffers;
    std::vector<SharedAuxBuffer> shared_aux_buffers;
    std::vector<std::vector<LimbExecState>> exec_limb_states;
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
