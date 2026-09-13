#include "gpu_test.cuh"
#include "matrix/MatrixUtils.cuh"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {
struct StreamGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool released = false;
    size_t submitted = 0;
    size_t finished = 0;
};

void CUDART_CB wait_for_release(void *pointer)
{
    auto &gate = *static_cast<StreamGate *>(pointer);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.changed.wait(lock, [&] { return gate.released; });
    ++gate.finished;
    // Notify under the lock so destruction cannot race a final callback access.
    gate.changed.notify_all();
}
}

extern "C" void gpu_test_release_stream_gate(void *pointer)
{
    if (!pointer) return;
    std::unique_ptr<StreamGate> gate(static_cast<StreamGate *>(pointer));
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->released = true;
    gate->changed.notify_all();
    gate->changed.wait(lock, [&] { return gate->finished == gate->submitted; });
}

static void *install_gates(const std::vector<std::pair<int, cudaStream_t>> &streams)
{
    if (streams.empty()) return nullptr;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) return nullptr;
    auto gate = std::make_unique<StreamGate>();
    cudaError_t error = cudaSuccess;
    for (const auto &[device, stream] : streams) {
        error = cudaSetDevice(device);
        if (error == cudaSuccess) error = cudaLaunchHostFunc(stream, wait_for_release, gate.get());
        if (error != cudaSuccess) break;
        ++gate->submitted;
    }
    const cudaError_t restored = cudaSetDevice(previous);
    if (error != cudaSuccess || restored != cudaSuccess) {
        gpu_test_release_stream_gate(gate.release());
        return nullptr;
    }
    return gate.release();
}

extern "C" void *gpu_test_matrix_stream_gate(GpuMatrix *matrix)
{
    if (!matrix || !matrix->ctx || matrix->ctx->limb_gpu_ids.empty()) return nullptr;
    const dim3 limb = matrix->ctx->limb_gpu_ids[0];
    int device = -1;
    cudaStream_t stream = nullptr;
    if (matrix_limb_device(matrix, limb, &device) != 0 ||
        matrix_limb_stream(matrix, limb, &stream) != 0) return nullptr;
    return install_gates({{device, stream}});
}

extern "C" void *gpu_test_context_stream_gate(GpuContext *context)
{
    if (!context || !context->execution || context->gpu_ids.empty() ||
        context->execution->compute_streams_by_partition.empty() ||
        context->execution->compute_streams_by_partition.front().empty()) return nullptr;
    return install_gates({{context->gpu_ids.front(),
        context->execution->compute_streams_by_partition.front().front()}});
}

extern "C" void *gpu_test_context_streams_gate(GpuContext *context)
{
    if (!context || !context->execution) return nullptr;
    std::vector<std::pair<int, cudaStream_t>> streams;
    const auto &owner = *context->execution;
    for (size_t partition = 0; partition < owner.compute_streams_by_partition.size(); ++partition)
        for (cudaStream_t stream : owner.compute_streams_by_partition[partition])
            streams.emplace_back(owner.gpu_ids[partition], stream);
    return install_gates(streams);
}
