#include "Control.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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
    GpuContext *ctx,
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
    if (!ctx || !ctx->execution ||
        std::find(ctx->execution->gpu_ids.begin(), ctx->execution->gpu_ids.end(),
            destination->device) == ctx->execution->gpu_ids.end())
        return gpu_set_last_error_cuda(cudaErrorInvalidDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t status = cudaSetDevice(destination->device);
    auto wait_producer = [stream](const MxxGpuDeviceBuffer *buffer) -> cudaError_t {
        if (buffer == nullptr || !buffer->producer_valid || stream == buffer->allocation_stream)
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
    GpuContext *ctx,
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
    if (!ctx || !ctx->execution ||
        std::find(ctx->execution->gpu_ids.begin(), ctx->execution->gpu_ids.end(),
            destination->device) == ctx->execution->gpu_ids.end())
        return gpu_set_last_error_cuda(cudaErrorInvalidDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t status = cudaSetDevice(destination->device);
    auto wait_producer = [stream](const MxxGpuDeviceBuffer *buffer) -> cudaError_t {
        if (buffer == nullptr || !buffer->producer_valid || stream == buffer->allocation_stream)
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
    if (!input->producer_valid || stream == input->allocation_stream)
        return 0;
    const cudaError_t status = cudaStreamWaitEvent(stream, input->producer, 0);
    return status == cudaSuccess ? 0 : gpu_set_last_error_cuda(status);
}

cudaError_t control_record_owner(MxxGpuDeviceBuffer *buffer, void *stream_raw)
{
    cudaError_t status = cudaEventRecord(buffer->producer, reinterpret_cast<cudaStream_t>(stream_raw));
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

int clear_control_status(void *stream_raw, uint32_t *status)
{
    if (!status)
        return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const cudaError_t error = cudaMemsetAsync(status, 0, sizeof(uint64_t), stream);
    return error == cudaSuccess ? 0 : gpu_set_last_error_cuda(error);
}

} // namespace

extern "C" int gpu_control_wait_input(
    const MxxGpuDeviceBuffer *buffer, int device, void *stream, bool read_only)
{
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
    cudaError_t error = cudaSetDevice(status->device);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, nullptr, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fill_constant_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset), count, value);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_fill_loop_index_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, size_t count, int64_t start,
    int64_t step, void *stream, uint32_t *status)
{
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, nullptr, nullptr);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fill_loop_index_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset), count, start, step, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess)
        return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, lhs, rhs);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    binary_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(lhs->address + lhs_offset),
        reinterpret_cast<const int64_t *>(rhs->address + rhs_offset), count, operation, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, quotient, quotient_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (remainder->producer_valid && cuda_stream != remainder->allocation_stream)
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
    return control_finish_two(quotient, remainder, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    exact_div_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(numerator->address + numerator_offset),
        reinterpret_cast<const int64_t *>(denominator->address + denominator_offset),
        count, status);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, quotient, quotient_offset, count, stream, numerator, denominator);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (remainder->producer_valid && cuda_stream != remainder->allocation_stream)
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
    return control_finish_two(quotient, remainder, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, lhs, rhs);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    compare_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(lhs->address + lhs_offset),
        reinterpret_cast<const int64_t *>(rhs->address + rhs_offset), count, operation);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bit_extract_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), count, bit);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_bool_to_int_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bool_to_int_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), count);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return control_finish(destination, stream, launch);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_copy_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaMemcpyAsync(
        destination->address + destination_offset, source->address + source_offset,
        count * sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_stream);
    if (status != cudaSuccess) return control_finish(destination, stream, status);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_pack_i64(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset, const MxxGpuDeviceBuffer *source,
    size_t source_offset, size_t count, void *stream)
{
    if (control_check_range<int64_t>(source, source_offset, count))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, source, nullptr);
    if (prepared != 0) return prepared;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaError_t status = cudaMemcpyAsync(
        destination->address + destination_offset, source->address + source_offset,
        count * sizeof(int64_t), cudaMemcpyDeviceToDevice, cuda_stream);
    if (status != cudaSuccess) return control_finish(destination, stream, status);
    return control_finish(destination, stream, cudaSuccess);
}

extern "C" int gpu_control_copy_range(
    GpuContext *ctx,
    MxxGpuDeviceBuffer *destination, size_t destination_offset,
    const MxxGpuDeviceBuffer *source, size_t source_offset,
    size_t bytes,
    void *stream)
{
    if (source == nullptr || source_offset > source->bytes ||
        bytes > source->bytes - source_offset)
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int prepared = control_prepare_bytes(
        ctx, destination, destination_offset, bytes, stream, source, nullptr);
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
    return control_finish(destination, stream, cudaSuccess);
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
        ctx, destination,
        destination_offset,
        count * value_words * sizeof(uint64_t),
        stream,
        source,
        indices);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, source, indices);
    if (prepared != 0) return prepared;
    const int reset = clear_control_status(stream, status);
    if (reset != 0) return reset;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gather_kernel<<<control_blocks(count), CONTROL_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<int64_t *>(destination->address + destination_offset),
        reinterpret_cast<const int64_t *>(source->address + source_offset), source_count,
        reinterpret_cast<const int64_t *>(indices->address + indices_offset), count, status);
    const cudaError_t launch = cudaGetLastError();
    return control_finish(destination, stream, launch);
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
    const int prepared = control_prepare(ctx, destination, destination_offset, count, stream, selector, when_false);
    if (prepared != 0) return prepared;
    const int extra = control_wait_extra_input(when_true, stream);
    if (extra != 0) return extra;
    const int reset = clear_control_status(stream, status);
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
    return control_finish(destination, stream, cudaSuccess);
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
        const uint64_t *left = integer_element(lhs, le, lc, (operation == 10 || operation == 12 || operation == 18) ? 0 : operation == 13 ? index - argument : index);
        const uint64_t *right = rhs ? integer_element(rhs, re, rc, index) : nullptr;
        if (operation == 10 || operation == 18) {
            bool invalid = integer_negative(right, re);
            for (size_t word = 1; word < integer_words(re); ++word) invalid |= integer_word(right, re, word) != 0;
            const uint64_t selected = integer_word(right, re, 0);
            if (invalid || selected >= lc) {
                control_error(status, operation == 18 ?
                    MXX_GPU_CONTROL_INVALID_RING_PROPERTY : MXX_GPU_CONTROL_INVALID_INDEX);
                continue;
            }
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
        if (operation == 17) {
            control_error(status, MXX_GPU_CONTROL_INVALID_RING_PROPERTY);
            continue;
        }
        if (operation == 16) {
            bool nonzero = false;
            bool power_of_two = true;
            size_t highest_bit = 0;
            for (size_t word = 0; word < integer_words(le); ++word) {
                const uint64_t value = integer_word(left, le, word);
                if (!value) continue;
                if (nonzero || (value & (value - 1)) != 0) power_of_two = false;
                nonzero = true;
                highest_bit = word * 64 + 63 - static_cast<size_t>(__clzll(value));
            }
            if (ln || !nonzero) {
                control_error(status, MXX_GPU_CONTROL_INVALID_INDEX);
                continue;
            }
            result[1] = highest_bit + (power_of_two ? 0 : 1);
            continue;
        }
        if (operation == 8 || operation == 9 || operation == 10 || operation == 11 || operation == 12 || operation == 13 || operation == 18) {
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
        } else if (operation == 3 || operation == 14 || operation == 15) {
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
            if (operation == 15 && has_remainder)
                control_error(status, MXX_GPU_CONTROL_INEXACT_DIVISION);
            const bool adjust = operation != 15 && has_remainder &&
                (operation == 3 ? ln : ln != rn);
            if (adjust) {
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
            result[0] = operation == 3 ? ln : ln != rn;
            remainder[0] = operation == 14 && has_remainder ? rn : 0;
            integer_normalize(remainder, words);
        }
        integer_normalize(result, words);
    }
}
}
extern "C" int gpu_control_integer_operation(
    GpuContext *ctx, void *out, const void *lhs, const void *rhs, void *aux, uint32_t *status,
    size_t count, size_t lhs_count, size_t rhs_count, int output_encoding, int lhs_encoding, int rhs_encoding,
    unsigned operation, uint64_t argument, void *stream_raw)
{
    if (!ctx || !out || !lhs || !stream_raw || count == 0 ||
        (operation != 10 && operation != 12 && operation != 13 && operation != 18 && lhs_count != 1 && lhs_count != count) ||
        (rhs && rhs_count != 1 && rhs_count != count) ||
        ((operation < 4 || operation == 14 || operation == 15) &&
            (!status || output_encoding <= 2 || !rhs)) ||
        (operation == 16 && (!status || output_encoding <= 2 ||
            lhs_encoding <= 2 || count != 1 || lhs_count != 1 || rhs || aux)) ||
        (operation == 17 && (!status || argument != MXX_GPU_CONTROL_INVALID_RING_PROPERTY ||
            output_encoding != 3 || lhs_encoding != 3 || count != 1 ||
            lhs_count != 1 || rhs || aux)) ||
        (operation == 18 && (!status || !rhs || aux || argument != 0 ||
            output_encoding != 3 || lhs_encoding != 3 ||
            (rhs_encoding != 1 && rhs_encoding < 3) ||
            count != 1 || rhs_count != 1)) ||
        ((operation == 3 || operation == 14 || operation == 15) && !aux) ||
        (operation == 8 && (!rhs || !status)))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    const int reset = clear_control_status(stream_raw, status);
    if (reset != 0) return reset;
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    integer_operation_kernel<<<control_blocks(count), CONTROL_THREADS, 0, stream>>>(
        static_cast<uint64_t *>(out), static_cast<const uint64_t *>(lhs), static_cast<const uint64_t *>(rhs),
        static_cast<uint64_t *>(aux), status, count, lhs_count, rhs_count, output_encoding, lhs_encoding, rhs_encoding, operation, argument);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return gpu_set_last_error_cuda(error);
    return 0;
}

extern "C" int gpu_control_integer_operation_direct(
    GpuContext *ctx, void *out, const void *lhs, const void *rhs,
    void *aux, uint32_t *status, size_t count, size_t lhs_count,
    size_t rhs_count, int output_encoding, int lhs_encoding,
    int rhs_encoding, unsigned operation, uint64_t argument,
    void *stream_raw, uint32_t out_binding, uint32_t lhs_binding,
    uint32_t rhs_binding, uint32_t aux_binding, uint32_t status_binding)
{
    if (!ctx || !stream_raw || !out || !lhs || !status || !count || !lhs_count ||
        operation > 18 || (rhs && !rhs_count) ||
        (operation <= 6 && !rhs) ||
        ((operation == 14 || operation == 15) &&
            (!rhs || output_encoding <= 2 ||
            lhs_encoding <= 2 || rhs_encoding <= 2)) ||
        ((operation == 3 || operation == 14 || operation == 15) && !aux) ||
        (operation == 16 && (output_encoding <= 2 || lhs_encoding <= 2 ||
            count != 1 || lhs_count != 1 || rhs || aux)) ||
        (operation == 17 && (argument != MXX_GPU_CONTROL_INVALID_RING_PROPERTY ||
            output_encoding != 3 || lhs_encoding != 3 ||
            count != 1 || lhs_count != 1 || rhs || aux)) ||
        (operation == 18 && (!rhs || aux || argument != 0 ||
            output_encoding != 3 || lhs_encoding != 3 ||
            (rhs_encoding != 1 && rhs_encoding < 3) ||
            count != 1 || rhs_count != 1)) ||
        (operation == 8 && !rhs))
        return gpu_set_last_error_cuda(cudaErrorInvalidValue);
    MxxGraphPatch patches[5];
    size_t patch_count = 0;
    patches[patch_count++] = control_kernel_patch(0, out_binding);
    patches[patch_count++] = control_kernel_patch(1, lhs_binding);
    if (rhs) patches[patch_count++] = control_kernel_patch(2, rhs_binding);
    if (aux) patches[patch_count++] = control_kernel_patch(3, aux_binding);
    patches[patch_count++] = control_kernel_patch(4, status_binding);
    return mxx_gpu_launch_kernel(ctx, reinterpret_cast<cudaStream_t>(stream_raw),
        integer_operation_kernel, dim3(control_blocks(count)), dim3(CONTROL_THREADS),
        0, patches, patch_count,
        static_cast<uint64_t *>(out), static_cast<const uint64_t *>(lhs),
        static_cast<const uint64_t *>(rhs), static_cast<uint64_t *>(aux),
        status, count, lhs_count, rhs_count, output_encoding, lhs_encoding,
        rhs_encoding, operation, argument);
}
