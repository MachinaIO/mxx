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

#define CONTROL_FOR_EACH(count) \

enum class BinaryOp : uint8_t { Add, Subtract, Multiply };

enum class CompareOp : uint8_t { Equal, Less, LessEqual };

#undef CONTROL_FOR_EACH

int control_blocks(size_t count)
{
    const size_t needed = (count + CONTROL_THREADS - 1) / CONTROL_THREADS;
    return static_cast<int>(needed == 0 ? 1 : needed > 65535 ? 65535 : needed);
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
