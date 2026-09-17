#include "gpu_test.cuh"
#include "matrix/MatrixUtils.cuh"

#include <condition_variable>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace {
#ifdef MXX_GPU_INSTRUMENTATION
std::atomic<bool> work_gate{false};
std::atomic<size_t> event_creations{0};
std::atomic<size_t> stream_creations{0};
std::atomic<size_t> native_validations{0};
std::atomic<size_t> cuda_allocations{0};
std::atomic<size_t> kernel_launches{0};
std::atomic<size_t> measurement_launches{0};
#endif

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

extern "C" void gpu_test_set_work_gate(bool enabled)
{
#ifdef MXX_GPU_INSTRUMENTATION
    work_gate.store(enabled, std::memory_order_release);
#else
    (void)enabled;
#endif
}

extern "C" void gpu_test_reset_work_counters()
{
#ifdef MXX_GPU_INSTRUMENTATION
    event_creations.store(0, std::memory_order_relaxed);
    stream_creations.store(0, std::memory_order_relaxed);
    native_validations.store(0, std::memory_order_relaxed);
    cuda_allocations.store(0, std::memory_order_relaxed);
    kernel_launches.store(0, std::memory_order_relaxed);
    measurement_launches.store(0, std::memory_order_relaxed);
#endif
}

extern "C" void gpu_test_read_work_counters(
    size_t *events, size_t *streams, size_t *validations,
    size_t *allocations, size_t *kernels, size_t *measurements)
{
#ifdef MXX_GPU_INSTRUMENTATION
    if (events) *events = event_creations.load(std::memory_order_relaxed);
    if (streams) *streams = stream_creations.load(std::memory_order_relaxed);
    if (validations) *validations = native_validations.load(std::memory_order_relaxed);
    if (allocations) *allocations = cuda_allocations.load(std::memory_order_relaxed);
    if (kernels) *kernels = kernel_launches.load(std::memory_order_relaxed);
    if (measurements) *measurements = measurement_launches.load(std::memory_order_relaxed);
#else
    if (events) *events = 0;
    if (streams) *streams = 0;
    if (validations) *validations = 0;
    if (allocations) *allocations = 0;
    if (kernels) *kernels = 0;
    if (measurements) *measurements = 0;
#endif
}

#ifdef MXX_GPU_INSTRUMENTATION
extern "C" void gpu_test_record_event_creation()
{
    if (work_gate.load(std::memory_order_acquire))
        event_creations.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void gpu_test_record_stream_creation()
{
    if (work_gate.load(std::memory_order_acquire))
        stream_creations.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void gpu_test_record_native_validation()
{
    if (work_gate.load(std::memory_order_acquire))
        native_validations.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void gpu_test_record_cuda_allocation()
{
    if (work_gate.load(std::memory_order_acquire))
        cuda_allocations.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void gpu_test_record_kernel_launch()
{
    if (work_gate.load(std::memory_order_acquire))
        kernel_launches.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void gpu_test_record_measurement_launch()
{
    if (work_gate.load(std::memory_order_acquire))
        measurement_launches.fetch_add(1, std::memory_order_relaxed);
}
#endif
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
