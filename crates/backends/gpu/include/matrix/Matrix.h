#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "Runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuMatrix GpuMatrix;

// Compiled storage bindings identify one logical owning partition, separately
// from the consumer device. Reads may join a remote producer; writes require
// the same physical GPU. These adapters leave other partitions untouched.
int gpu_matrix_wait_compiled_storage(const GpuMatrix *mat, int storage_device,
    int consumer_device, void *consumer_stream, bool read_only);
// Protect submitted use on the selected owner's release stream. Only actual
// writes refresh producer readiness. Failure is launch-uncertain: callers must
// retain the submitted graph and bound owners until completion is established.
int gpu_matrix_record_compiled_storage_use(GpuMatrix *mat, int storage_device,
    int consumer_device, void *consumer_stream, void *completion_event, bool written);

typedef enum GpuMatrixSampleDist
{
    GPU_MATRIX_DIST_UNIFORM = 0,
    GPU_MATRIX_DIST_GAUSS = 1,
    GPU_MATRIX_DIST_BIT = 2,
    GPU_MATRIX_DIST_TERNARY = 3,
    // Uniform over an inclusive signed interval; raw Graph sampler only.
    GPU_MATRIX_DIST_INTERVAL = 4,
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
    struct LimbExecState
    {
        int device;
        gpuStream_t stream;
        // This event is physically owned here and destroyed exactly once.
        gpuEvent_t write_done;
        // Local state index whose owned event currently dominates this limb.
        uint32_t completion_owner;
        // Producer ownership and last event-recording stream can differ.
        gpuStream_t last_write_stream;
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

#include "matrix/MatrixCrt.h"
#include "matrix/MatrixData.h"
#include "matrix/MatrixNTT.h"
#include "matrix/MatrixSerde.h"
#include "matrix/MatrixUtils.h"
#include "matrix/MatrixSmallRhs.h"
