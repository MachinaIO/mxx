#include "Control.cuh"

#include <cuda_runtime.h>

#include <limits>
#include <initializer_list>
#include <utility>


namespace
{

constexpr int CONTROL_THREADS = 256;

__device__ __forceinline__ void control_error(uint32_t *status, uint32_t code)
{
    if (status != nullptr)
        atomicExch(status, code);
}

__device__ __forceinline__ int64_t checked_i64(__int128 value, uint32_t *status)
{
    if (value < static_cast<__int128>(INT64_MIN) ||
        value > static_cast<__int128>(INT64_MAX))
    {
        control_error(status, MXX_GPU_CONTROL_OVERFLOW);
        return 0;
    }
    return static_cast<int64_t>(value);
}

#define CONTROL_FOR_EACH(count) \
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
         index < (count); index += static_cast<size_t>(gridDim.x) * blockDim.x)

__global__ void fill_constant_kernel(int64_t *out, size_t count, int64_t value)
{
    CONTROL_FOR_EACH(count) { out[index] = value; }
}

__global__ void fill_loop_index_kernel(
    int64_t *out,
    size_t count,
    int64_t start,
    int64_t step,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        const __int128 value = static_cast<__int128>(start) +
            static_cast<__int128>(step) * static_cast<__int128>(index);
        out[index] = checked_i64(value, status);
    }
}

enum class BinaryOp : uint8_t { Add, Subtract, Multiply };

__global__ void binary_kernel(
    int64_t *out,
    const int64_t *lhs,
    const int64_t *rhs,
    size_t count,
    BinaryOp operation,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        const __int128 left = lhs[index];
        const __int128 right = rhs[index];
        const __int128 value = operation == BinaryOp::Add
            ? left + right
            : operation == BinaryOp::Subtract ? left - right : left * right;
        out[index] = checked_i64(value, status);
    }
}

__device__ __forceinline__ void euclidean_div_rem(
    int64_t numerator,
    int64_t denominator,
    int64_t &quotient,
    int64_t &remainder,
    uint32_t *status)
{
    if (denominator == 0)
    {
        control_error(status, MXX_GPU_CONTROL_DIVISION_BY_ZERO);
        quotient = 0;
        remainder = 0;
        return;
    }
    // __int128 handles abs(INT64_MIN) and the one representable quotient
    // boundary without relying on signed overflow or implementation-defined
    // shifts.  C++ division truncates toward zero, so adjust a negative
    // non-integral quotient to obtain floor division by |denominator|.
    const __int128 numerator_wide = static_cast<__int128>(numerator);
    const __int128 modulus = denominator < 0
        ? -static_cast<__int128>(denominator)
        : static_cast<__int128>(denominator);
    __int128 q = numerator_wide / modulus;
    __int128 r = numerator_wide % modulus;
    if (r < 0)
    {
        --q;
        r += modulus;
    }
    quotient = static_cast<int64_t>(q);
    remainder = static_cast<int64_t>(r);
}

__global__ void div_rem_kernel(
    int64_t *quotient,
    int64_t *remainder,
    const int64_t *numerator,
    const int64_t *denominator,
    size_t count,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        euclidean_div_rem(
            numerator[index], denominator[index], quotient[index], remainder[index], status);
    }
}

__global__ void exact_div_kernel(
    int64_t *out,
    const int64_t *numerator,
    const int64_t *denominator,
    size_t count,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        const int64_t divisor = denominator[index];
        if (divisor == 0)
        {
            control_error(status, MXX_GPU_CONTROL_DIVISION_BY_ZERO);
            out[index] = 0;
            continue;
        }
        const __int128 numerator_wide = static_cast<__int128>(numerator[index]);
        const __int128 divisor_wide = static_cast<__int128>(divisor);
        const __int128 quotient = numerator_wide / divisor_wide;
        const __int128 remainder = numerator_wide % divisor_wide;
        if (remainder != 0)
        {
            control_error(status, MXX_GPU_CONTROL_INEXACT_DIVISION);
            out[index] = 0;
            continue;
        }
        out[index] = checked_i64(quotient, status);
    }
}

__device__ __forceinline__ void floor_div_rem(
    int64_t numerator,
    int64_t denominator,
    int64_t &quotient,
    int64_t &remainder,
    uint32_t *status)
{
    if (denominator == 0)
    {
        control_error(status, MXX_GPU_CONTROL_DIVISION_BY_ZERO);
        quotient = 0;
        remainder = 0;
        return;
    }
    // Widen before division so INT64_MIN/-1 is represented and reported as
    // overflow instead of invoking signed overflow in the native type.
    const __int128 numerator_wide = static_cast<__int128>(numerator);
    const __int128 denominator_wide = static_cast<__int128>(denominator);
    __int128 q = numerator_wide / denominator_wide;
    __int128 r = numerator_wide % denominator_wide;
    // C++ truncates toward zero.  Adjust when the truncating remainder has a
    // different sign from the divisor to obtain BigInt::div_floor/mod_floor.
    if (r != 0 && ((r < 0) != (denominator_wide < 0)))
    {
        --q;
        r += denominator_wide;
    }
    quotient = checked_i64(q, status);
    remainder = checked_i64(r, status);
}

__global__ void floor_div_rem_kernel(
    int64_t *quotient,
    int64_t *remainder,
    const int64_t *numerator,
    const int64_t *denominator,
    size_t count,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        floor_div_rem(
            numerator[index], denominator[index], quotient[index], remainder[index], status);
    }
}

enum class CompareOp : uint8_t { Equal, Less, LessEqual };

__global__ void compare_kernel(
    int64_t *out,
    const int64_t *lhs,
    const int64_t *rhs,
    size_t count,
    CompareOp operation)
{
    CONTROL_FOR_EACH(count) {
        out[index] = operation == CompareOp::Equal
            ? lhs[index] == rhs[index]
            : operation == CompareOp::Less ? lhs[index] < rhs[index] : lhs[index] <= rhs[index];
    }
}

__global__ void bit_extract_kernel(
    int64_t *out,
    const int64_t *source,
    size_t count,
    uint64_t bit)
{
    CONTROL_FOR_EACH(count) {
        // Signed integer expressions in the IR have unbounded sign extension.
        // Preserve that behavior for bit positions beyond the native word.
        out[index] = bit < 64 ?
            static_cast<int64_t>((static_cast<uint64_t>(source[index]) >> bit) & 1U) :
            static_cast<int64_t>(source[index] < 0);
    }
}

__global__ void bool_to_int_kernel(int64_t *out, const int64_t *source, size_t count)
{
    CONTROL_FOR_EACH(count) {
        out[index] = source[index] == 0 ? 0 : 1;
    }
}

__global__ void gather_kernel(
    int64_t *out,
    const int64_t *source,
    size_t source_count,
    const int64_t *indices,
    size_t count,
    uint32_t *status)
{
    CONTROL_FOR_EACH(count) {
        const int64_t selected = indices[index];
        if (selected < 0 || static_cast<uint64_t>(selected) >= source_count)
        {
            control_error(status, MXX_GPU_CONTROL_INVALID_INDEX);
            out[index] = 0;
            continue;
        }
        out[index] = source[static_cast<size_t>(selected)];
    }
}

__global__ void gather_u64_kernel(
    uint64_t *out,
    const uint64_t *source,
    size_t source_count,
    const uint64_t *indices,
    size_t count, uint32_t *status, size_t value_words)
{
    CONTROL_FOR_EACH(count) {
        const int64_t selected = static_cast<int64_t>(indices[index]);
        if (selected < 0 || static_cast<uint64_t>(selected) >= source_count)
        {
            control_error(status, MXX_GPU_CONTROL_INVALID_INDEX);
            for (size_t word = 0; word < value_words; ++word) out[index * value_words + word] = 0;
            continue;
        }
        for (size_t word = 0; word < value_words; ++word)
            out[index * value_words + word] = source[static_cast<size_t>(selected) * value_words + word];
    }
}

__global__ void select_i64_kernel(
    int64_t *out,
    const int64_t *selector,
    size_t selector_count,
    const int64_t *when_false,
    size_t when_false_count,
    const int64_t *when_true,
    size_t when_true_count,
    size_t count,
    uint32_t *status)
{
    (void)status;
    CONTROL_FOR_EACH(count) {
        const size_t selector_index = selector_count == 1 ? 0 : index;
        const size_t branch_index = when_false_count == 1 ? 0 : index;
        const size_t true_index = when_true_count == 1 ? 0 : index;
        out[index] = selector[selector_index] != 0
            ? when_true[true_index]
            : when_false[branch_index];
    }
}

#undef CONTROL_FOR_EACH

int control_blocks(size_t count)
{
    const size_t needed = (count + CONTROL_THREADS - 1) / CONTROL_THREADS;
    return static_cast<int>(needed == 0 ? 1 : needed > 65535 ? 65535 : needed);
}

template <typename T>
int control_check_range(const MxxGpuDeviceBuffer *buffer, size_t offset, size_t count)
{
    (void)sizeof(T);
    return buffer == nullptr || offset > buffer->bytes ||
        count > (buffer->bytes - offset) / sizeof(T);
}

int control_prepare(
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    size_t count,
    void *stream_raw,
    const MxxGpuDeviceBuffer *first_input,
    const MxxGpuDeviceBuffer *second_input)
{
    if (destination == nullptr || stream_raw == nullptr ||
        control_check_range<int64_t>(destination, destination_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t status = cudaSetDevice(destination->device);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (status == cudaSuccess)
        status = cudaStreamIsCapturing(stream, &capture_status);
    auto wait_producer = [stream, capture_status](const MxxGpuDeviceBuffer *buffer) -> cudaError_t {
        if (buffer == nullptr || !buffer->producer_valid || stream == buffer->allocation_stream)
            return cudaSuccess;
        if (capture_status != cudaStreamCaptureStatusNone)
            return cudaSuccess;
        return cudaStreamWaitEvent(stream, buffer->producer, 0);
    };
    if (status == cudaSuccess)
        status = wait_producer(destination);
    if (status == cudaSuccess)
        status = wait_producer(first_input);
    if (status == cudaSuccess)
        status = wait_producer(second_input);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

int control_prepare_bytes(
    MxxGpuDeviceBuffer *destination,
    size_t destination_offset,
    size_t bytes,
    void *stream_raw,
    const MxxGpuDeviceBuffer *first_input,
    const MxxGpuDeviceBuffer *second_input)
{
    if (destination == nullptr || stream_raw == nullptr ||
        destination_offset > destination->bytes ||
        bytes > destination->bytes - destination_offset)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t status = cudaSetDevice(destination->device);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (status == cudaSuccess)
        status = cudaStreamIsCapturing(stream, &capture_status);
    auto wait_producer = [stream, capture_status](const MxxGpuDeviceBuffer *buffer) -> cudaError_t {
        if (buffer == nullptr || !buffer->producer_valid || stream == buffer->allocation_stream)
            return cudaSuccess;
        // Cross-stream event waits are not legal during this launch-site
        // capture. The owner contract requires external producers to have
        // been made ready before capture begins; ordinary launches retain the
        // event dependency below.
        if (capture_status != cudaStreamCaptureStatusNone)
            return cudaSuccess;
        return cudaStreamWaitEvent(stream, buffer->producer, 0);
    };
    if (status == cudaSuccess)
        status = wait_producer(destination);
    if (status != cudaSuccess)
        return gpu_set_last_error_cuda(status);
    status = wait_producer(first_input);
    if (status != cudaSuccess)
        return gpu_set_last_error_cuda(status);
    status = wait_producer(second_input);
    if (status != cudaSuccess)
        return gpu_set_last_error_cuda(status);
    return 0;
}

int control_wait_extra_input(const MxxGpuDeviceBuffer *input, void *stream_raw)
{
    if (input == nullptr || stream_raw == nullptr)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t status = cudaStreamIsCapturing(stream, &capture_status);
    if (status == cudaSuccess && capture_status == cudaStreamCaptureStatusNone &&
        input->producer_valid && stream != input->allocation_stream)
        status = cudaStreamWaitEvent(stream, input->producer, 0);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

cudaError_t control_record_owner(MxxGpuDeviceBuffer *buffer, void *stream_raw)
{
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t status = cudaStreamIsCapturing(
        reinterpret_cast<cudaStream_t>(stream_raw), &capture_status);
    if (status != cudaSuccess || capture_status != cudaStreamCaptureStatusNone)
        return status;
    status = cudaEventRecord(buffer->producer, reinterpret_cast<cudaStream_t>(stream_raw));
    if (status == cudaSuccess)
        buffer->producer_valid = true;
    return status;
}

int control_finish(MxxGpuDeviceBuffer *destination, void *stream_raw, cudaError_t status)
{
    if (status == cudaSuccess)
        status = control_record_owner(destination, stream_raw);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

int control_finish_two(
    MxxGpuDeviceBuffer *first,
    MxxGpuDeviceBuffer *second,
    void *stream_raw,
    cudaError_t status)
{
    if (status == cudaSuccess)
        status = control_record_owner(first, stream_raw);
    if (status == cudaSuccess)
        status = control_record_owner(second, stream_raw);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

// Graph registration has its own descriptive error channel.  Preserve that
// native status instead of replacing it with cudaErrorUnknown, so a resident
// binding/schema failure reaches the caller with its actual reason.
int control_finish_registered(
    MxxGpuDeviceBuffer *destination, void *stream_raw, int registered)
{
    if (registered != 0) return registered;
    return control_finish(destination, stream_raw, cudaSuccess);
}

int control_finish_registered_two(
    MxxGpuDeviceBuffer *first, MxxGpuDeviceBuffer *second, void *stream_raw, int registered)
{
    if (registered != 0) return registered;
    return control_finish_two(first, second, stream_raw, cudaSuccess);
}

MxxGraphPatch control_kernel_patch(uint32_t argument, uint32_t binding)
{
    MxxGraphPatch patch{};
    patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
    patch.argument_index = argument;
    patch.byte_offset = 0;
    patch.byte_count = sizeof(void *);
    patch.binding_index = binding;
    return patch;
}

int register_control_kernel(
    GpuContext *ctx,
    void *stream,
    const size_t *argument_sizes,
    size_t argument_count,
    std::initializer_list<std::pair<uint32_t, uint32_t>> pointer_bindings)
{
    MxxGraphPatch patches[8]{};
    size_t count = 0;
    for (const auto &[argument, binding] : pointer_bindings)
        patches[count++] = control_kernel_patch(argument, binding);
    return mxx_graph_register_kernel_update_for_stream(
        ctx, stream, argument_sizes, argument_count, patches, count);
}

int clear_control_status(
    GpuContext *ctx,
    void *stream_raw,
    uint32_t *status,
    uint32_t binding)
{
    if (!status)
        return 0;
    if (!mxx_gpu_graph_control_status_needs_reset(ctx, stream_raw, status))
        return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = cudaMemsetAsync(status, 0, sizeof(uint64_t), stream);
    if (error != cudaSuccess)
        return gpu_set_last_error_cuda(error);
    MxxGraphPatch patch{};
    patch.target = MXX_GRAPH_PATCH_MEMSET_1D_DST;
    patch.byte_count = sizeof(uint64_t);
    patch.binding_index = binding;
    return mxx_graph_register_memset1d_update_for_stream(ctx, stream_raw, nullptr, &patch);
}

int register_control_copy(
    GpuContext *ctx,
    void *stream,
    size_t bytes,
    uint32_t destination_binding,
    uint32_t source_binding,
    uint64_t destination_addend,
    uint64_t source_addend)
{
    MxxGraphPatch patches[2]{};
    patches[0].target = MXX_GRAPH_PATCH_MEMCPY_1D_DST;
    patches[0].byte_count = sizeof(void *);
    patches[0].binding_index = destination_binding;
    patches[0].address_addend = destination_addend;
    patches[1].target = MXX_GRAPH_PATCH_MEMCPY_1D_SRC;
    patches[1].byte_count = sizeof(void *);
    patches[1].binding_index = source_binding;
    patches[1].address_addend = source_addend;
    return mxx_graph_register_memcpy1d_update_for_stream(
        ctx, stream, bytes, cudaMemcpyDeviceToDevice, patches, 2);
}

} // namespace

extern "C" void *gpu_control_launch_stream(GpuContext *ctx, int device, void *fallback)
{
    if (ctx == nullptr || !ctx->execution)
        return fallback;
    auto &owner = *ctx->execution;
    std::lock_guard<std::mutex> lock(owner.capture_mutex);
    if (owner.capture_body_active && owner.capture_body_device == device)
        return reinterpret_cast<void *>(owner.capture_body_stream);
    if (owner.capture_active && owner.capture_device == device)
        return reinterpret_cast<void *>(owner.capture_stream);
    return fallback;
}

extern "C" int gpu_control_wait_input(
    const MxxGpuDeviceBuffer *buffer, int device, void *stream, bool read_only)
{
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t status = cudaStreamIsCapturing(
        reinterpret_cast<cudaStream_t>(stream), &capture_status);
    if (status != cudaSuccess)
        return gpu_set_last_error_cuda(status);
    // Planning resolves all external producers before capture begins. Values
    // produced inside capture are ordered by the single resolved stream.
    if (capture_status != cudaStreamCaptureStatusNone)
        return 0;
    return gpu_device_buffer_wait_compiled_inputs(buffer, device, stream, read_only);
}

extern "C" int gpu_control_record_compiled_write(
    MxxGpuDeviceBuffer *buffer, void *stream)
{
    if (buffer == nullptr || stream == nullptr)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const cudaError_t status = control_record_owner(buffer, stream);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

extern "C" int gpu_control_read_status(
    const MxxGpuDeviceBuffer *status,
    size_t status_offset,
    void *completion_event,
    void *stream_raw,
    uint32_t *host_status)
{
    if (status == nullptr || completion_event == nullptr || stream_raw == nullptr ||
        host_status == nullptr || status_offset > status->bytes ||
        sizeof(uint32_t) > status->bytes - status_offset)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t error = cudaStreamIsCapturing(stream, &capture_status);
    if (error != cudaSuccess)
        return gpu_set_last_error_cuda(error);
    if (capture_status != cudaStreamCaptureStatusNone)
        return gpu_set_last_error_cuda(cudaErrorStreamCaptureUnsupported);
    error = cudaSetDevice(status->device);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(
            stream, reinterpret_cast<cudaEvent_t>(completion_event), 0);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(
            host_status,
            status->address + status_offset,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream);
    if (error == cudaSuccess)
        error = cudaStreamSynchronize(stream);
    return error == cudaSuccess ? 0 : gpu_set_last_error_cuda(error);
}

extern "C" int gpu_control_fill_constant_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, size_t count, int64_t value,
    void *stream, uint32_t *status)
{
    (void)status;
    const int prepared = control_prepare(destination, destination_offset, count, stream, nullptr, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fill_constant_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset), count, value);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(size_t), sizeof(int64_t)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 3, {{0, 0}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_fill_loop_index_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, size_t count, int64_t start,
    int64_t step, void *stream, uint32_t *status)
{
    const int prepared = control_prepare(destination, destination_offset, count, stream, nullptr, nullptr);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 1);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fill_loop_index_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset), count, start, step, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess)
        return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(size_t), sizeof(int64_t), sizeof(int64_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 5, {{0, 0}, {4, 1}});
    return control_finish_registered(destination, stream, registered);
}

int gpu_control_binary_impl(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    const MxxGpuDeviceBuffer *lhs, size_t lhs_offset,
    const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count, void *stream,
    uint32_t *status, BinaryOp operation)
{
    if (control_check_range<int64_t>(lhs, lhs_offset, count) ||
        control_check_range<int64_t>(rhs, rhs_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, lhs, rhs);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 3);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    binary_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(lhs->address + lhs_offset),
        reinterpret_cast<const int64_t *>(rhs->address + rhs_offset), count, operation, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(BinaryOp), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 6, {{0, 0}, {1, 1}, {2, 2}, {5, 3}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_add_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count,
    void *stream, uint32_t *status)
{
    return gpu_control_binary_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, status, BinaryOp::Add);
}
extern "C" int gpu_control_sub_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count,
    void *stream, uint32_t *status)
{
    return gpu_control_binary_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, status, BinaryOp::Subtract);
}
extern "C" int gpu_control_mul_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count,
    void *stream, uint32_t *status)
{
    return gpu_control_binary_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, status, BinaryOp::Multiply);
}

extern "C" int gpu_control_div_rem_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *quotient, size_t quotient_offset, MxxGpuDeviceBuffer *remainder,
    size_t remainder_offset, const MxxGpuDeviceBuffer *numerator, size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator, size_t denominator_offset, size_t count,
    void *stream, uint32_t *status)
{
    if (quotient == remainder ||
        control_check_range<int64_t>(quotient, quotient_offset, count) ||
        control_check_range<int64_t>(numerator, numerator_offset, count) ||
        control_check_range<int64_t>(denominator, denominator_offset, count) ||
        control_check_range<int64_t>(remainder, remainder_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(quotient, quotient_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 4);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t capture_query = cudaStreamIsCapturing(cuda_stream, &capture_status);
    if (capture_query != cudaSuccess) return gpu_set_last_error_cuda(capture_query);
    if (remainder->producer_valid && cuda_stream != remainder->allocation_stream &&
        capture_status == cudaStreamCaptureStatusNone)
    {
        const cudaError_t wait = cudaStreamWaitEvent(cuda_stream, remainder->producer, 0);
        if (wait != cudaSuccess) return gpu_set_last_error_cuda(wait);
    }
    div_rem_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(quotient->address + quotient_offset),
        reinterpret_cast<int64_t *>(remainder->address + remainder_offset),
        reinterpret_cast<const int64_t *>(numerator->address + numerator_offset),
        reinterpret_cast<const int64_t *>(denominator->address + denominator_offset), count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish_two(quotient, remainder, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 6, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {5, 4}});
    return control_finish_registered_two(quotient, remainder, stream, registered);
}

extern "C" int gpu_control_exact_div_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    const MxxGpuDeviceBuffer *numerator, size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator, size_t denominator_offset, size_t count,
    void *stream, uint32_t *status)
{
    if (control_check_range<int64_t>(numerator, numerator_offset, count) ||
        control_check_range<int64_t>(denominator, denominator_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 4);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    exact_div_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(numerator->address + numerator_offset),
        reinterpret_cast<const int64_t *>(denominator->address + denominator_offset),
        count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 5, {{0, 0}, {1, 1}, {2, 2}, {4, 3}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_floor_div_rem_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *quotient, size_t quotient_offset, MxxGpuDeviceBuffer *remainder,
    size_t remainder_offset, const MxxGpuDeviceBuffer *numerator, size_t numerator_offset,
    const MxxGpuDeviceBuffer *denominator, size_t denominator_offset, size_t count,
    void *stream, uint32_t *status)
{
    if (quotient == remainder ||
        control_check_range<int64_t>(quotient, quotient_offset, count) ||
        control_check_range<int64_t>(numerator, numerator_offset, count) ||
        control_check_range<int64_t>(denominator, denominator_offset, count) ||
        control_check_range<int64_t>(remainder, remainder_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(quotient, quotient_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 4);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t capture_query = cudaStreamIsCapturing(cuda_stream, &capture_status);
    if (capture_query != cudaSuccess) return gpu_set_last_error_cuda(capture_query);
    if (remainder->producer_valid && cuda_stream != remainder->allocation_stream &&
        capture_status == cudaStreamCaptureStatusNone)
    {
        const cudaError_t wait = cudaStreamWaitEvent(cuda_stream, remainder->producer, 0);
        if (wait != cudaSuccess) return gpu_set_last_error_cuda(wait);
    }
    floor_div_rem_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(quotient->address + quotient_offset),
        reinterpret_cast<int64_t *>(remainder->address + remainder_offset),
        reinterpret_cast<const int64_t *>(numerator->address + numerator_offset),
        reinterpret_cast<const int64_t *>(denominator->address + denominator_offset),
        count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish_two(quotient, remainder, stream, launch);
    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 6, {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {5, 4}});
    return control_finish_registered_two(quotient, remainder, stream, registered);
}

int gpu_control_compare_impl(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count,
    void *stream, CompareOp operation)
{
    if (control_check_range<int64_t>(lhs, lhs_offset, count) ||
        control_check_range<int64_t>(rhs, rhs_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, lhs, rhs);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    compare_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(lhs->address + lhs_offset),
        reinterpret_cast<const int64_t *>(rhs->address + rhs_offset), count, operation);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(CompareOp)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 5, {{0, 0}, {1, 1}, {2, 2}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_compare_eq_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count, void *stream)
{ return gpu_control_compare_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, CompareOp::Equal); }
extern "C" int gpu_control_compare_lt_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count, void *stream)
{ return gpu_control_compare_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, CompareOp::Less); }
extern "C" int gpu_control_compare_le_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *lhs,
    size_t lhs_offset, const MxxGpuDeviceBuffer *rhs, size_t rhs_offset, size_t count, void *stream)
{ return gpu_control_compare_impl(ctx, destination, destination_offset, lhs, lhs_offset, rhs, rhs_offset, count, stream, CompareOp::LessEqual); }

extern "C" int gpu_control_bit_extract_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, uint64_t bit, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bit_extract_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), count, bit);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(uint64_t)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 4, {{0, 0}, {1, 1}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_bool_to_int_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bool_to_int_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), count);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(size_t)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 3, {{0, 0}, {1, 1}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_copy_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaMemcpyAsync(
        destination->address + destination_offset, source->address + source_offset,
        count * sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_stream);
    if (status != cudaSuccess) return control_finish(destination, stream, status);
    const int registered = register_control_copy(
        ctx, stream, count * sizeof(int64_t), 0, 1, 0, 0);
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_pack_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaMemcpyAsync(
        destination->address + destination_offset, source->address + source_offset,
        count * sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_stream);
    if (status != cudaSuccess) return control_finish(destination, stream, status);
    const int registered = register_control_copy(
        ctx, stream, count * sizeof(int64_t), 0, 1, 0, 0);
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_copy_range(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    uint64_t destination_addend,
    const MxxGpuDeviceBuffer *source, size_t source_offset, uint64_t source_addend,
    size_t bytes,
    void *stream)
{
    if (source == nullptr || source_offset > source->bytes ||
        bytes > source->bytes - source_offset)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare_bytes(
        destination, destination_offset, bytes, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaSetDevice(destination->device);
    if (status == cudaSuccess && source->device == destination->device)
    {
        status = cudaMemcpyAsync(
            destination->address + destination_offset,
            source->address + source_offset,
            bytes,
            cudaMemcpyDeviceToDevice,
            cuda_stream);
    }
    else if (status == cudaSuccess)
    {
        int can_access = 0;
        status = cudaDeviceCanAccessPeer(&can_access, destination->device, source->device);
        if (status == cudaSuccess && !can_access)
            return gpu_set_last_error_cuda(cudaErrorPeerAccessUnsupported);
        if (status == cudaSuccess)
        {
            status = cudaDeviceEnablePeerAccess(source->device, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled)
            {
                cudaGetLastError();
                status = cudaSuccess;
            }
        }
        if (status == cudaSuccess)
        {
            status = cudaMemcpyPeerAsync(
                destination->address + destination_offset,
                destination->device,
                source->address + source_offset,
                source->device,
                bytes,
                cuda_stream);
        }
    }
    if (status != cudaSuccess) return control_finish(destination, stream, status);
    const int registered = register_control_copy(
        ctx,
        stream,
        bytes,
        0,
        1,
        destination_addend,
        source_addend);
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_gather_u64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    const MxxGpuDeviceBuffer *source, size_t source_offset, size_t source_count,
    const MxxGpuDeviceBuffer *indices, size_t indices_offset, size_t count, size_t value_words,
    void *stream, uint32_t *status)
{
    if (destination == nullptr || source == nullptr || indices == nullptr ||
        source->device != destination->device || indices->device != destination->device ||
        value_words == 0 || count > SIZE_MAX / sizeof(uint64_t) / value_words || source_count > SIZE_MAX / value_words ||
        control_check_range<uint64_t>(source, source_offset, source_count * value_words) ||
        control_check_range<uint64_t>(indices, indices_offset, count) ||
        control_check_range<uint64_t>(destination, destination_offset, count * value_words))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare_bytes(
        destination,
        destination_offset,
        count * value_words * sizeof(uint64_t),
        stream,
        source,
        indices);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 3);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gather_u64_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<uint64_t *>(destination->address + destination_offset),
        reinterpret_cast<const uint64_t *>(source->address + source_offset),
        source_count,
        reinterpret_cast<const uint64_t *>(indices->address + indices_offset),
        count, status, value_words);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *), sizeof(size_t), sizeof(void *), sizeof(size_t)};
    MxxGraphPatch patches[4]{};
    patches[0] = control_kernel_patch(0, 0);
    patches[1] = control_kernel_patch(1, 1);
    patches[2] = control_kernel_patch(3, 2);
    patches[3] = control_kernel_patch(5, 3);
    const int registered = mxx_graph_register_kernel_update_for_stream(
        ctx, stream, argument_sizes, 7, patches, status ? 4 : 3);
    if (registered != 0)
        return control_finish(destination, stream, cudaErrorUnknown);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_gather_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t source_count, const MxxGpuDeviceBuffer *indices,
    size_t indices_offset, size_t count, void *stream, uint32_t *status)
{
    if (control_check_range<int64_t>(source, source_offset, source_count) ||
        control_check_range<int64_t>(indices, indices_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, source, indices);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(ctx, stream, status, 3);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gather_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), source_count,
        reinterpret_cast<const int64_t *>(indices->address + indices_offset), count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *), sizeof(size_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 6, {{0, 0}, {1, 1}, {3, 2}, {5, 3}});
    return control_finish_registered(destination, stream, registered);
}

extern "C" int gpu_control_select_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    const MxxGpuDeviceBuffer *selector, size_t selector_offset, size_t selector_count,
    const MxxGpuDeviceBuffer *when_false, size_t when_false_offset, size_t when_false_count,
    const MxxGpuDeviceBuffer *when_true, size_t when_true_offset, size_t when_true_count,
    size_t count, void *stream, uint32_t *status)
{
    if (destination == nullptr || selector == nullptr || when_false == nullptr ||
        when_true == nullptr || destination->device != selector->device ||
        destination->device != when_false->device || destination->device != when_true->device ||
        (selector_count != 1 && selector_count != count) ||
        (when_false_count != 1 && when_false_count != count) ||
        (when_true_count != 1 && when_true_count != count) ||
        control_check_range<int64_t>(selector, selector_offset, selector_count) ||
        control_check_range<int64_t>(when_false, when_false_offset, when_false_count) ||
        control_check_range<int64_t>(when_true, when_true_offset, when_true_count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(destination, destination_offset, count, stream, selector, when_false);
    if (prepared != 0) return prepared;
    const int extra = control_wait_extra_input(when_true, stream);
    if (extra != 0) return extra;
    const int reset = clear_control_status(ctx, stream, status, 4);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    select_i64_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(selector->address + selector_offset), selector_count,
        reinterpret_cast<const int64_t *>(when_false->address + when_false_offset), when_false_count,
        reinterpret_cast<const int64_t *>(when_true->address + when_true_offset), when_true_count,
        count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *), sizeof(size_t),
        sizeof(void *), sizeof(size_t), sizeof(size_t), sizeof(void *)};
    const int registered = register_control_kernel(
        ctx, stream, argument_sizes, 9, {{0, 0}, {1, 1}, {3, 2}, {5, 3}, {8, 4}});
    return control_finish_registered(destination, stream, registered);
}

namespace {
__device__ size_t integer_words(int encoding) { return encoding > 2 ? encoding - 2 : 1; }
__device__ const uint64_t *integer_element(const uint64_t *values, int encoding, size_t count, size_t index) {
    return values + (count == 1 ? 0 : index) * (encoding > 2 ? encoding - 1 : 1);
}
__device__ bool integer_negative(const uint64_t *value, int encoding) {
    return encoding == 0 ? static_cast<int64_t>(*value) < 0 : encoding > 2 && value[0] != 0;
}
__device__ uint64_t integer_word(const uint64_t *value, int encoding, size_t word) {
    if (word >= integer_words(encoding)) return 0;
    if (encoding > 2) return value[word + 1];
    return encoding == 0 && static_cast<int64_t>(*value) < 0 ? uint64_t(0) - *value : *value;
}
__device__ int integer_compare_magnitude(const uint64_t *lhs, int le, const uint64_t *rhs, int re) {
    size_t words = max(integer_words(le), integer_words(re));
    while (words) {
        --words;
        const uint64_t left = integer_word(lhs, le, words), right = integer_word(rhs, re, words);
        if (left != right) return left < right ? -1 : 1;
    }
    return 0;
}
__device__ void integer_normalize(uint64_t *out, size_t words) {
    bool nonzero = false;
    for (size_t word = 1; word <= words; ++word) nonzero |= out[word] != 0;
    if (!nonzero) out[0] = 0;
}
__global__ void integer_operation_kernel(
    uint64_t *out, const uint64_t *lhs, const uint64_t *rhs, uint64_t *aux,
    uint32_t *status, size_t count, size_t lc, size_t rc,
    int oe, int le, int re, unsigned operation, uint64_t argument)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (operation == 13 && (index < argument || index - argument >= lc)) continue;
        const uint64_t *left = integer_element(lhs, le, lc, (operation == 10 || operation == 12) ? 0 : operation == 13 ? index - argument : index);
        const uint64_t *right = rhs ? integer_element(rhs, re, rc, index) : nullptr;
        if (operation == 10) {
            bool invalid = integer_negative(right, re);
            for (size_t word = 1; word < integer_words(re); ++word) invalid |= integer_word(right, re, word) != 0;
            const uint64_t selected = integer_word(right, re, 0);
            if (invalid || selected >= lc) { control_error(status, MXX_GPU_CONTROL_INVALID_INDEX); continue; }
            left = integer_element(lhs, le, lc, selected);
        }
        if (operation == 12) {
            if (argument >= lc) { control_error(status, MXX_GPU_CONTROL_INVALID_INDEX); continue; }
            left = integer_element(lhs, le, lc, argument);
        }
        const bool ln = integer_negative(left, le);
        const bool rn = right && integer_negative(right, re);
        if (operation >= 4 && operation <= 6) {
            const int magnitude = integer_compare_magnitude(left, le, right, re);
            const int comparison = ln != rn ? (ln ? -1 : 1) : (ln ? -magnitude : magnitude);
            out[index] = operation == 4 ? comparison == 0 : operation == 5 ? comparison < 0 : comparison <= 0;
            continue;
        }
        if (operation == 7) {
            const size_t word = argument / 64;
            uint64_t value = integer_word(left, le, word);
            if (ln) {
                bool borrow = true;
                for (size_t previous = 0; previous < word; ++previous)
                    if (integer_word(left, le, previous) != 0) { borrow = false; break; }
                value = ~(value - static_cast<uint64_t>(borrow));
            }
            out[index] = (value >> (argument % 64)) & 1;
            continue;
        }
        if (operation == 8) {
            bool invalid = integer_negative(right, re);
            for (size_t word = 1; word < integer_words(re); ++word) invalid |= integer_word(right, re, word) != 0;
            const uint64_t selected = integer_word(right, re, 0);
            if (invalid || selected >= (argument >> 32)) { control_error(status, MXX_GPU_CONTROL_INVALID_INDEX); continue; }
            if (selected != (argument & UINT32_MAX)) continue;
        }
        if ((operation >= 8 && operation <= 13) && oe < 3) {
            const uint64_t magnitude = integer_word(left, le, 0);
            bool overflow = oe == 1 ? ln : magnitude > (ln ? (uint64_t(1) << 63) : INT64_MAX);
            for (size_t word = 1; word < integer_words(le); ++word) overflow |= integer_word(left, le, word) != 0;
            if (overflow) { control_error(status, MXX_GPU_CONTROL_OVERFLOW); continue; }
            out[index] = ln ? uint64_t(0) - magnitude : magnitude;
            continue;
        }
        const size_t words = integer_words(oe);
        uint64_t *result = out + index * (words + 1);
        for (size_t word = 0; word <= words; ++word) result[word] = 0;
        if (operation == 8 || operation == 9 || operation == 10 || operation == 11 || operation == 12 || operation == 13) {
            bool overflow = false;
            for (size_t word = words; word < integer_words(le); ++word) overflow |= integer_word(left, le, word) != 0;
            if (overflow) { control_error(status, MXX_GPU_CONTROL_OVERFLOW); continue; }
            result[0] = ln;
            for (size_t word = 0; word < words; ++word) result[word + 1] = integer_word(left, le, word);
            continue;
        }
        if (operation == 0 || operation == 1) {
            const bool right_negative = rn != (operation == 1);
            if (ln == right_negative) {
                uint64_t carry = 0;
                for (size_t word = 0; word < words; ++word) {
                    const unsigned __int128 term = static_cast<unsigned __int128>(integer_word(left, le, word)) + integer_word(right, re, word) + carry;
                    result[word + 1] = static_cast<uint64_t>(term);
                    carry = static_cast<uint64_t>(term >> 64);
                }
                if (carry) control_error(status, MXX_GPU_CONTROL_OVERFLOW);
                result[0] = ln;
            } else {
                const bool swap = integer_compare_magnitude(left, le, right, re) < 0;
                const uint64_t *larger = swap ? right : left, *smaller = swap ? left : right;
                const int large_encoding = swap ? re : le, small_encoding = swap ? le : re;
                uint64_t borrow = 0;
                for (size_t word = 0; word < words; ++word) {
                    const uint64_t large = integer_word(larger, large_encoding, word);
                    const unsigned __int128 small = static_cast<unsigned __int128>(integer_word(smaller, small_encoding, word)) + borrow;
                    result[word + 1] = large - static_cast<uint64_t>(small);
                    borrow = static_cast<unsigned __int128>(large) < small;
                }
                result[0] = swap ? right_negative : ln;
            }
        } else if (operation == 2) {
            result[0] = ln != rn;
            for (size_t a = 0; a < integer_words(le); ++a) {
                uint64_t carry = 0;
                for (size_t b = 0; b < integer_words(re); ++b) {
                    const size_t word = a + b;
                    const unsigned __int128 term = static_cast<unsigned __int128>(integer_word(left, le, a)) * integer_word(right, re, b) + carry + (word < words ? result[word + 1] : 0);
                    if (word < words) result[word + 1] = static_cast<uint64_t>(term);
                    else if (static_cast<uint64_t>(term)) control_error(status, MXX_GPU_CONTROL_OVERFLOW);
                    carry = static_cast<uint64_t>(term >> 64);
                }
                const size_t word = a + integer_words(re);
                if (word < words) result[word + 1] = carry;
                else if (carry) control_error(status, MXX_GPU_CONTROL_OVERFLOW);
            }
        } else if (operation == 3) {
            uint64_t *remainder = aux + index * (words + 1);
            for (size_t word = 0; word <= words; ++word) remainder[word] = 0;
            bool nonzero = false;
            for (size_t word = 0; word < integer_words(re); ++word) nonzero |= integer_word(right, re, word) != 0;
            if (!nonzero) { control_error(status, MXX_GPU_CONTROL_DIVISION_BY_ZERO); continue; }
            for (size_t bit = integer_words(le) * 64; bit != 0;) {
                --bit;
                uint64_t carry = (integer_word(left, le, bit / 64) >> (bit % 64)) & 1;
                for (size_t word = 1; word <= words; ++word) {
                    const uint64_t next = remainder[word] >> 63;
                    remainder[word] = (remainder[word] << 1) | carry;
                    carry = next;
                }
                if (integer_compare_magnitude(remainder, oe, right, re) >= 0) {
                    uint64_t borrow = 0;
                    for (size_t word = 0; word < words; ++word) {
                        const unsigned __int128 sub = static_cast<unsigned __int128>(integer_word(right, re, word)) + borrow;
                        const uint64_t value = remainder[word + 1];
                        remainder[word + 1] = value - static_cast<uint64_t>(sub);
                        borrow = static_cast<unsigned __int128>(value) < sub;
                    }
                    if (bit / 64 < words) result[bit / 64 + 1] |= uint64_t(1) << (bit % 64);
                    else control_error(status, MXX_GPU_CONTROL_OVERFLOW);
                }
            }
            bool has_remainder = false;
            for (size_t word = 1; word <= words; ++word) has_remainder |= remainder[word] != 0;
            if (ln && has_remainder) {
                uint64_t carry = 1, borrow = 0;
                for (size_t word = 0; word < words; ++word) {
                    const unsigned __int128 quotient_word = static_cast<unsigned __int128>(result[word + 1]) + carry;
                    result[word + 1] = static_cast<uint64_t>(quotient_word);
                    carry = static_cast<uint64_t>(quotient_word >> 64);
                    const unsigned __int128 sub = static_cast<unsigned __int128>(remainder[word + 1]) + borrow;
                    const uint64_t modulus_word = integer_word(right, re, word);
                    remainder[word + 1] = modulus_word - static_cast<uint64_t>(sub);
                    borrow = static_cast<unsigned __int128>(modulus_word) < sub;
                }
                if (carry) control_error(status, MXX_GPU_CONTROL_OVERFLOW);
            }
            result[0] = ln;
            remainder[0] = 0;
            integer_normalize(remainder, words);
        }
        integer_normalize(result, words);
    }
}

struct HashIntegerBytes32
{
    uint8_t bytes[32];
};

__device__ __forceinline__ uint64_t hash_integer_rotl64(uint64_t value, int count)
{
    return count == 0 ? value : (value << count) | (value >> (64 - count));
}

__device__ __forceinline__ void hash_integer_keccak_f(uint64_t state[25])
{
    constexpr uint64_t round_constants[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL,
    };
    constexpr int rotation[25] = {
        0, 1, 62, 28, 27,
        36, 44, 6, 55, 20,
        3, 10, 43, 25, 39,
        41, 45, 15, 21, 8,
        18, 2, 61, 56, 14,
    };
    for (int round = 0; round < 24; ++round)
    {
        uint64_t column_parity[5];
        for (int x = 0; x < 5; ++x)
            column_parity[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^
                               state[x + 15] ^ state[x + 20];
        uint64_t theta[5];
        for (int x = 0; x < 5; ++x)
            theta[x] = column_parity[(x + 4) % 5] ^
                       hash_integer_rotl64(column_parity[(x + 1) % 5], 1);
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y) state[x + 5 * y] ^= theta[x];
        uint64_t rho_pi[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
            {
                const int destination_x = y;
                const int destination_y = (2 * x + 3 * y) % 5;
                rho_pi[destination_x + 5 * destination_y] =
                    hash_integer_rotl64(state[x + 5 * y], rotation[x + 5 * y]);
            }
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                state[x + 5 * y] = rho_pi[x + 5 * y] ^
                    ((~rho_pi[(x + 1) % 5 + 5 * y]) & rho_pi[(x + 2) % 5 + 5 * y]);
        state[0] ^= round_constants[round];
    }
}

__device__ __forceinline__ void hash_integer_keccak256(
    const uint8_t *message, size_t length, uint64_t digest[4])
{
    uint8_t block[136] = {};
    for (size_t index = 0; index < length; ++index) block[index] = message[index];
    block[length] ^= 0x01U;
    block[135] ^= 0x80U;
    uint64_t state[25] = {};
    for (size_t lane = 0; lane < 17; ++lane)
    {
        uint64_t word = 0;
        for (int byte = 0; byte < 8; ++byte)
            word |= static_cast<uint64_t>(block[lane * 8 + byte]) << (8 * byte);
        state[lane] = word;
    }
    hash_integer_keccak_f(state);
    for (int word = 0; word < 4; ++word) digest[word] = state[word];
}

__global__ void hash_integer_family_kernel(
    uint64_t *output,
    size_t count,
    size_t modulus_bits,
    size_t words,
    HashIntegerBytes32 key,
    HashIntegerBytes32 tag_digest)
{
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t value_base = index * (words + 1);
        output[value_base] = 0;
        for (size_t word = 0; word < words; ++word) output[value_base + word + 1] = 0;
        if (modulus_bits == 0) continue;

        constexpr char key_prefix[] = "mxx/hash-int-family/key/v1";
        uint8_t seed_message[sizeof(key_prefix) - 1 + 64];
        size_t offset = 0;
        for (size_t byte = 0; byte < sizeof(key_prefix) - 1; ++byte)
            seed_message[offset++] = static_cast<uint8_t>(key_prefix[byte]);
        for (int byte = 0; byte < 32; ++byte) seed_message[offset++] = key.bytes[byte];
        for (int byte = 0; byte < 32; ++byte) seed_message[offset++] = tag_digest.bytes[byte];
        uint64_t seed[4];
        hash_integer_keccak256(seed_message, offset, seed);

        for (size_t word = 0; word < words; ++word)
        {
            if (word % 4 == 0)
            {
                uint8_t draw_message[48];
                for (int seed_word = 0; seed_word < 4; ++seed_word)
                    for (int byte = 0; byte < 8; ++byte)
                        draw_message[seed_word * 8 + byte] =
                            static_cast<uint8_t>(seed[seed_word] >> (8 * byte));
                for (int byte = 0; byte < 8; ++byte)
                    draw_message[32 + byte] = static_cast<uint8_t>(index >> (8 * byte));
                const size_t block = word / 4;
                for (int byte = 0; byte < 8; ++byte)
                    draw_message[40 + byte] = static_cast<uint8_t>(block >> (8 * byte));
                uint64_t draw[4];
                hash_integer_keccak256(draw_message, sizeof(draw_message), draw);
                for (int limb = 0; limb < 4 && word + limb < words; ++limb)
                {
                    const size_t output_word = word + limb;
                    uint64_t value = draw[limb];
                    if (output_word + 1 == words && modulus_bits % 64 != 0)
                        value &= (uint64_t(1) << (modulus_bits % 64)) - 1;
                    output[value_base + output_word + 1] = value;
                }
            }
        }
    }
}
}

extern "C" int gpu_control_hash_integer_family(
    GpuContext *ctx,
    void *out,
    size_t count,
    size_t modulus_bits,
    size_t words,
    const uint8_t key_bytes[32],
    const uint8_t tag_digest_bytes[32],
    void *stream_raw)
{
    if (!ctx || !out || !key_bytes || !tag_digest_bytes || !stream_raw || words == 0 ||
        words > (SIZE_MAX / 8 - 1) || count > SIZE_MAX / (words + 1) / 8 ||
        modulus_bits > words * 64 || (modulus_bits != 0 && modulus_bits <= (words - 1) * 64))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    if (count == 0) return 0;

    HashIntegerBytes32 key{};
    HashIntegerBytes32 tag_digest{};
    for (int byte = 0; byte < 32; ++byte)
    {
        key.bytes[byte] = key_bytes[byte];
        tag_digest.bytes[byte] = tag_digest_bytes[byte];
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    hash_integer_family_kernel<<<control_blocks(count), CONTROL_THREADS, 0, stream>>>(
        static_cast<uint64_t *>(out), count, modulus_bits, words, key, tag_digest);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return gpu_set_last_error_cuda(launch);

    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(size_t), sizeof(size_t), sizeof(size_t),
        sizeof(HashIntegerBytes32), sizeof(HashIntegerBytes32)};
    MxxGraphPatch patches[2]{};
    patches[0] = control_kernel_patch(0, 0);
    patches[1].target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
    patches[1].argument_index = 4;
    patches[1].byte_offset = 0;
    patches[1].byte_count = sizeof(HashIntegerBytes32);
    patches[1].binding_index = 1;
    return mxx_graph_register_kernel_update_for_stream(ctx, stream_raw, argument_sizes, 6, patches, 2);
}

extern "C" int gpu_control_integer_operation(
    GpuContext *ctx, void *out, const void *lhs, const void *rhs, void *aux, uint32_t *status,
    size_t count, size_t lhs_count, size_t rhs_count, int output_encoding, int lhs_encoding, int rhs_encoding,
    unsigned operation, uint64_t argument, void *stream_raw)
{
    if (!ctx || !out || !lhs || !stream_raw || count == 0 ||
        (operation != 10 && operation != 12 && operation != 13 && lhs_count != 1 && lhs_count != count) ||
        (rhs && rhs_count != 1 && rhs_count != count) ||
        (operation < 4 && (!status || output_encoding <= 2)) ||
        (operation == 3 && !aux) || (operation == 8 && (!rhs || !status)))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int reset = clear_control_status(ctx, stream_raw, status, 3);
    if (reset != 0) return reset;
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    integer_operation_kernel<<<control_blocks(count), CONTROL_THREADS, 0, stream>>>(
        static_cast<uint64_t *>(out), static_cast<const uint64_t *>(lhs), static_cast<const uint64_t *>(rhs),
        static_cast<uint64_t *>(aux), status, count, lhs_count, rhs_count, output_encoding, lhs_encoding, rhs_encoding, operation, argument);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return gpu_set_last_error_cuda(error);
    const size_t sizes[] = {sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(int), sizeof(int), sizeof(int), sizeof(unsigned), sizeof(uint64_t)};
    MxxGraphPatch patches[8];
    size_t patch_count = 0;
    patches[patch_count++] = control_kernel_patch(0, 0);
    if (operation != 9) patches[patch_count++] = control_kernel_patch(1, 1);
    if (rhs) patches[patch_count++] = control_kernel_patch(2, 2);
    if (status) patches[patch_count++] = control_kernel_patch(4, 3);
    if (aux) patches[patch_count++] = control_kernel_patch(3, 4);
    patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_INTEGER_ENCODING, 8, 0, sizeof(int), 0, reinterpret_cast<uint64_t>(out)};
    if (operation != 9) patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_INTEGER_ENCODING, 9, 0, sizeof(int), 1, reinterpret_cast<uint64_t>(lhs)};
    if (rhs) patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_INTEGER_ENCODING, 10, 0, sizeof(int), 2, reinterpret_cast<uint64_t>(rhs)};
    return mxx_graph_register_kernel_update_for_stream(ctx, stream_raw, sizes, 13, patches, patch_count);
}
