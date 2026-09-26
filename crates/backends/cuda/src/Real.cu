#include "Real.cuh"

#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
    constexpr uint32_t kInvalidRealDomain = 2;
    constexpr uint32_t kNonFiniteReal = 3;

    __device__ __forceinline__ bool real_signed_words_to_double(
        const uint64_t *source, int encoding, double *result)
    {
        const size_t words = static_cast<size_t>(encoding - 2);
        const uint64_t sign = source[0];
        if (sign > 1) return false;
        size_t highest = words;
        while (highest > 0 && source[highest] == 0) --highest;
        if (highest == 0)
        {
            if (sign != 0) return false;
            *result = 0.0;
            return true;
        }
        const uint64_t high = source[highest];
        const size_t bit_length = (highest - 1) * 64 + 64 - __clzll(high);
        if (bit_length > 1024)
        {
            *result = sign == 0 ? INFINITY : -INFINITY;
            return true;
        }
        double magnitude;
        if (bit_length <= 53)
        {
            magnitude = static_cast<double>(high);
        }
        else
        {
            const size_t shift = bit_length - 53;
            const size_t first_word = shift / 64 + 1;
            const unsigned first_bit = static_cast<unsigned>(shift % 64);
            uint64_t significant = source[first_word] >> first_bit;
            if (first_bit != 0 && first_word < words)
                significant |= source[first_word + 1] << (64 - first_bit);
            significant &= (UINT64_C(1) << 53) - 1;
            const size_t round_bit_index = shift - 1;
            const size_t round_word = round_bit_index / 64 + 1;
            const unsigned round_offset = static_cast<unsigned>(round_bit_index % 64);
            const bool round_bit = ((source[round_word] >> round_offset) & 1U) != 0;
            bool sticky = (source[round_word] & ((UINT64_C(1) << round_offset) - 1)) != 0;
            for (size_t word = 1; word < round_word; ++word)
                sticky = sticky || source[word] != 0;
            if (round_bit && (sticky || (significant & 1U))) ++significant;
            magnitude = ldexp(static_cast<double>(significant), static_cast<int>(shift));
        }
        *result = sign == 0 ? magnitude : -magnitude;
        return true;
    }

    __global__ void real_operation_kernel(
        double *output, const void *left, const double *right,
        uint32_t *status, uint32_t operation, int integer_encoding,
        uint64_t constant_bits)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        double value = 0.0;
        bool valid = true;
        if (operation == 0)
        {
            value = __longlong_as_double(static_cast<long long>(constant_bits));
        }
        else if (operation == 1)
        {
            if (integer_encoding == 0)
                value = static_cast<double>(*static_cast<const int64_t *>(left));
            else if (integer_encoding == 1)
                value = static_cast<double>(*static_cast<const uint64_t *>(left));
            else
                valid = real_signed_words_to_double(
                    static_cast<const uint64_t *>(left), integer_encoding, &value);
        }
        else
        {
            const double lhs = *static_cast<const double *>(left);
            const double rhs = right ? *right : 0.0;
            if (!isfinite(lhs) || (right && !isfinite(rhs)))
            {
                atomicCAS(status, 0U, kNonFiniteReal);
                *output = 0.0;
                return;
            }
            if (operation == 7) value = lhs;
            else if (operation == 2) value = lhs + rhs;
            else if (operation == 3) value = lhs - rhs;
            else if (operation == 4) value = lhs * rhs;
            else if (operation == 5)
            {
                if (rhs == 0.0) valid = false;
                else value = lhs / rhs;
            }
            else
            {
                if (lhs < 0.0) valid = false;
                else value = sqrt(lhs);
            }
        }
        if (!valid)
        {
            atomicCAS(status, 0U, kInvalidRealDomain);
            *output = 0.0;
        }
        else if (!isfinite(value))
        {
            atomicCAS(status, 0U, kNonFiniteReal);
            *output = 0.0;
        }
        else *output = value;
    }
}

extern "C" int gpu_real_emit(
    GpuContext *ctx, void *stream_raw, uint32_t operation,
    double *output, const void *left, const double *right,
    int left_integer_encoding, uint64_t constant_bits, uint32_t *status,
    uint32_t output_binding, uint32_t left_binding,
    uint32_t right_binding, uint32_t status_binding)
{
    if (!ctx || !stream_raw || !output || !status || operation > 7 ||
        (operation == 0 && (left || right)) ||
        (operation == 1 && (!left || right || left_integer_encoding == 2 ||
            left_integer_encoding < 0)) ||
        (operation >= 2 && !left) ||
        (operation >= 2 && operation <= 5 && !right) ||
        ((operation == 6 || operation == 7) && right))
        return gpu_set_last_error("invalid raw real operation");
    MxxGraphPatch patches[4];
    size_t patch_count = 0;
    patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
        0, 0, static_cast<uint32_t>(sizeof(void *)), output_binding, 0};
    if (left) patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
        1, 0, static_cast<uint32_t>(sizeof(void *)), left_binding, 0};
    if (right) patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
        2, 0, static_cast<uint32_t>(sizeof(void *)), right_binding, 0};
    patches[patch_count++] = {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
        3, 0, static_cast<uint32_t>(sizeof(void *)), status_binding, 0};
    return mxx_gpu_launch_kernel(ctx, reinterpret_cast<cudaStream_t>(stream_raw),
        real_operation_kernel, dim3(1), dim3(1), 0, patches, patch_count,
        output, left, right, status, operation, left_integer_encoding,
        constant_bits);
}
