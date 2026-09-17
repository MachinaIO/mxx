#pragma once

#include <cuda_runtime_api.h>

struct GpuMatrix;
struct GpuContext;

// Unit-test stream gate. No production Rust wrapper references these symbols.
extern "C" void *gpu_test_matrix_stream_gate(GpuMatrix *matrix);
extern "C" void *gpu_test_context_stream_gate(GpuContext *context);
extern "C" void *gpu_test_context_streams_gate(GpuContext *context);
extern "C" void gpu_test_release_stream_gate(void *gate);

// These counters and their call sites are compiled only into the explicit
// gpu-instrumentation build. Production CUDA translation units see inline
// no-ops, so the steady-state path has no branch or atomic overhead.
#ifdef MXX_GPU_INSTRUMENTATION
extern "C" void gpu_test_set_work_gate(bool enabled);
extern "C" void gpu_test_reset_work_counters();
extern "C" void gpu_test_read_work_counters(
    size_t *events, size_t *streams, size_t *native_validations,
    size_t *allocations, size_t *kernel_launches, size_t *measurement_launches);
extern "C" void gpu_test_record_event_creation();
extern "C" void gpu_test_record_stream_creation();
extern "C" void gpu_test_record_native_validation();
extern "C" void gpu_test_record_cuda_allocation();
extern "C" void gpu_test_record_kernel_launch();
extern "C" void gpu_test_record_measurement_launch();
#else
#define gpu_test_record_event_creation() ((void)0)
#define gpu_test_record_stream_creation() ((void)0)
#define gpu_test_record_native_validation() ((void)0)
#define gpu_test_record_cuda_allocation() ((void)0)
#define gpu_test_record_kernel_launch() ((void)0)
#define gpu_test_record_measurement_launch() ((void)0)
#endif

#ifdef MXX_GPU_INSTRUMENTATION
inline cudaError_t gpu_test_event_create(cudaEvent_t *event)
{
    gpu_test_record_event_creation();
    return cudaEventCreate(event);
}

inline cudaError_t gpu_test_event_create_with_flags(cudaEvent_t *event, unsigned flags)
{
    gpu_test_record_event_creation();
    return cudaEventCreateWithFlags(event, flags);
}

inline cudaError_t gpu_test_stream_create_with_flags(cudaStream_t *stream, unsigned flags)
{
    gpu_test_record_stream_creation();
    return cudaStreamCreateWithFlags(stream, flags);
}

#define cudaEventCreate gpu_test_event_create
#define cudaEventCreateWithFlags gpu_test_event_create_with_flags
#define cudaStreamCreateWithFlags gpu_test_stream_create_with_flags
#endif
