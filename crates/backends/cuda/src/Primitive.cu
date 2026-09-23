#include "Primitive.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "matrix/MatrixUtils.cuh"

namespace
{
constexpr int kMaxLimbs = static_cast<int>(GPU_RUNTIME_MAX_LIMBS);
constexpr int kMaxWords = kMaxLimbs + 1;

// Status values are intentionally small and stable.  They are written to the
// caller's resident status word, not returned as a host-observed result.
enum PrimitiveStatus : uint32_t
{
    kPrimitiveOk = 0,
    kPrimitiveInvalidInput = 1,
    kPrimitiveOverflow = 2,
    kPrimitiveInvalidBit = 3,
    kPrimitivePackedOutOfRange = 4,
};

struct PrimitiveCrtMetadata
{
    int limb_count;
    int word_count;
    uint64_t moduli[kMaxLimbs];
    uint64_t modulus_words[kMaxWords];
    uint64_t half_modulus_words[kMaxWords];
};

struct PrimitiveRawDescriptors
{
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor limbs[kMaxLimbs];
};

__device__ uint64_t primitive_mul_mod_exact(uint64_t lhs, uint64_t rhs, uint64_t modulus)
{
    return static_cast<uint64_t>(
        (static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs)) %
        static_cast<unsigned __int128>(modulus));
}

__device__ uint64_t primitive_inverse_mod(uint64_t value, uint64_t modulus)
{
    // Runtime contexts reject non-coprime CRT bases.  CRT moduli are below
    // 2^63, so signed __int128 is sufficient for extended Euclid products.
    __int128 old_r = static_cast<__int128>(value % modulus);
    __int128 r = static_cast<__int128>(modulus);
    __int128 old_t = 1;
    __int128 t = 0;
    while (r != 0)
    {
        const __int128 quotient = old_r / r;
        const __int128 next_r = old_r - quotient * r;
        old_r = r;
        r = next_r;
        const __int128 next_t = old_t - quotient * t;
        old_t = t;
        t = next_t;
    }
    if (old_t < 0)
    {
        old_t += static_cast<__int128>(modulus);
    }
    return static_cast<uint64_t>(old_t);
}

__device__ __forceinline__ uint64_t primitive_load_residue(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
    int limb,
    size_t coefficient)
{
    const auto descriptor = descriptors[limb];
    return matrix_load_limb_u64(descriptor.base, 0, coefficient, descriptor.stride, descriptor.width);
}

__device__ void primitive_zero_words(uint64_t *words, int count)
{
    for (int i = 0; i < count; ++i)
    {
        words[i] = 0;
    }
}

__device__ bool primitive_words_less(const uint64_t *lhs, const uint64_t *rhs, int count)
{
    for (int i = count - 1; i >= 0; --i)
    {
        if (lhs[i] != rhs[i])
        {
            return lhs[i] < rhs[i];
        }
    }
    return false;
}

__device__ void primitive_words_sub(uint64_t *lhs, const uint64_t *rhs, int count)
{
    uint64_t borrow = 0;
    for (int i = 0; i < count; ++i)
    {
        const unsigned __int128 subtrahend =
            static_cast<unsigned __int128>(rhs[i]) + static_cast<unsigned __int128>(borrow);
        const unsigned __int128 minuend = static_cast<unsigned __int128>(lhs[i]);
        lhs[i] = static_cast<uint64_t>(minuend - subtrahend);
        borrow = minuend < subtrahend ? 1 : 0;
    }
}

__device__ void primitive_words_add_product(
    uint64_t *destination,
    const uint64_t *factor,
    uint64_t digit,
    int count)
{
    uint64_t carry = 0;
    for (int i = 0; i < count; ++i)
    {
        const unsigned __int128 term =
            static_cast<unsigned __int128>(factor[i]) * static_cast<unsigned __int128>(digit) +
            static_cast<unsigned __int128>(destination[i]) + static_cast<unsigned __int128>(carry);
        destination[i] = static_cast<uint64_t>(term);
        carry = static_cast<uint64_t>(term >> 64);
    }
}

__device__ void primitive_words_mul_u64(
    const uint64_t *input,
    uint64_t factor,
    uint64_t *output,
    int count)
{
    uint64_t carry = 0;
    for (int i = 0; i < count; ++i)
    {
        const unsigned __int128 term =
            static_cast<unsigned __int128>(input[i]) * static_cast<unsigned __int128>(factor) +
            static_cast<unsigned __int128>(carry);
        output[i] = static_cast<uint64_t>(term);
        carry = static_cast<uint64_t>(term >> 64);
    }
    output[count] = carry;
}

__device__ uint64_t primitive_words_add_in_place(uint64_t *lhs, const uint64_t *rhs, int count)
{
    uint64_t carry = 0;
    for (int i = 0; i < count; ++i)
    {
        const unsigned __int128 term =
            static_cast<unsigned __int128>(lhs[i]) + static_cast<unsigned __int128>(rhs[i]) +
            static_cast<unsigned __int128>(carry);
        lhs[i] = static_cast<uint64_t>(term);
        carry = static_cast<uint64_t>(term >> 64);
    }
    return carry;
}

__device__ void primitive_words_shift_left_one(uint64_t *words, int count, uint64_t input_bit)
{
    uint64_t carry = input_bit;
    for (int i = 0; i < count; ++i)
    {
        const uint64_t next = words[i] >> 63;
        words[i] = (words[i] << 1) | carry;
        carry = next;
    }
}

__device__ void primitive_reconstruct(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
    const PrimitiveCrtMetadata &metadata,
    size_t coefficient,
    uint64_t *output)
{
    uint64_t mixed_digits[kMaxLimbs];
    uint64_t product[kMaxWords];
    primitive_zero_words(output, metadata.word_count);
    primitive_zero_words(product, metadata.word_count);
    product[0] = 1;

    for (int limb = 0; limb < metadata.limb_count; ++limb)
    {
        const uint64_t modulus = metadata.moduli[limb];
        uint64_t digit = primitive_load_residue(descriptors, limb, coefficient) % modulus;
        for (int previous = 0; previous < limb; ++previous)
        {
            const uint64_t previous_mod = mixed_digits[previous] % modulus;
            digit = digit >= previous_mod ? digit - previous_mod : modulus - (previous_mod - digit);
            const uint64_t inverse = primitive_inverse_mod(
                metadata.moduli[previous] % modulus, modulus);
            digit = primitive_mul_mod_exact(digit, inverse, modulus);
        }
        mixed_digits[limb] = digit;
        primitive_words_add_product(output, product, digit, metadata.word_count);

        uint64_t next_product[kMaxWords];
        primitive_words_mul_u64(product, modulus, next_product, metadata.word_count);
        for (int word = 0; word < metadata.word_count; ++word)
        {
            product[word] = next_product[word];
        }
    }
}

__device__ void primitive_set_status(uint32_t *status, uint32_t value)
{
    if (status)
    {
        atomicMax(status, value);
    }
}

bool primitive_valid_metadata(const GpuContext *ctx, size_t limb_count, PrimitiveCrtMetadata *out)
{
    if (!ctx || !out || limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS ||
        limb_count > ctx->moduli.size())
    {
        return false;
    }
    out->limb_count = static_cast<int>(limb_count);
    out->word_count = 1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        out->moduli[limb] = ctx->moduli[limb];
    }
    std::fill_n(out->modulus_words, kMaxWords, uint64_t(0));
    out->modulus_words[0] = 1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        uint64_t carry = 0;
        for (int word = 0; word < out->word_count; ++word)
        {
            const unsigned __int128 term =
                static_cast<unsigned __int128>(out->modulus_words[word]) * out->moduli[limb] + carry;
            out->modulus_words[word] = static_cast<uint64_t>(term);
            carry = static_cast<uint64_t>(term >> 64);
        }
        if (carry != 0)
        {
            out->modulus_words[out->word_count++] = carry;
        }
    }
    std::copy_n(out->modulus_words, kMaxWords, out->half_modulus_words);
    uint64_t carry = 0;
    for (int word = out->word_count - 1; word >= 0; --word)
    {
        const uint64_t next = out->half_modulus_words[word] & 1;
        out->half_modulus_words[word] = (out->half_modulus_words[word] >> 1) | (carry << 63);
        carry = next;
    }
    return true;
}

bool primitive_raw_view_metadata(GpuContext *ctx,
    const MxxRawMatrixView *source, void *stream,
    PrimitiveRawDescriptors *descriptors, PrimitiveCrtMetadata *metadata)
{
    if (!ctx || !source || !stream || !descriptors || !metadata ||
        !source->limbs || source->rows != 1 || source->columns != 1 ||
        source->row_origin != 0 || source->column_origin != 0 ||
        source->degree != static_cast<uint32_t>(ctx->N) ||
        source->limb_count != ctx->moduli.size() ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream) ||
        !primitive_valid_metadata(ctx, source->limb_count, metadata))
        return false;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const auto &part = source->limbs[limb];
        const unsigned __int128 span =
            static_cast<unsigned __int128>(source->degree - 1) *
            part.coefficient_stride_bytes + part.word_bytes;
        if (!part.address || part.crt_limb_index != limb ||
            part.modulus != ctx->moduli[limb] ||
            (part.word_bytes != 4 && part.word_bytes != 8) ||
            part.coefficient_stride_bytes != part.word_bytes ||
            part.column_stride_bytes < span ||
            limb >= ctx->limb_gpu_ids.size() ||
            ctx->limb_gpu_ids[limb].x >= ctx->gpu_ids.size() ||
            ctx->gpu_ids[ctx->limb_gpu_ids[limb].x] != source->physical_device)
            return false;
        descriptors->limbs[limb] = {
            reinterpret_cast<uint8_t *>(part.address),
            static_cast<size_t>(part.column_stride_bytes),
            static_cast<uint8_t>(part.word_bytes),
        };
    }
    return true;
}

MxxGraphPatch primitive_raw_patch(uint32_t argument, uint32_t offset,
    uint32_t binding)
{
    return {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, argument, offset,
        static_cast<uint32_t>(sizeof(void *)), binding, 0};
}

__global__ void primitive_raw_values_kernel(PrimitiveRawDescriptors descriptors,
    PrimitiveCrtMetadata metadata, uint64_t *output,
    size_t magnitude_words, size_t count)
{
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         coefficient < count; coefficient += step)
    {
        uint64_t words[kMaxWords];
        primitive_reconstruct(descriptors.limbs, metadata, coefficient, words);
        uint64_t *value = output + coefficient * (magnitude_words + 1);
        value[0] = 0;
        for (size_t word = 0; word < magnitude_words; ++word)
            value[word + 1] = word < static_cast<size_t>(metadata.word_count) ?
                words[word] : 0;
    }
}

__device__ bool primitive_position(const uint64_t *encoded, int encoding,
    size_t degree, size_t *position)
{
    if (encoding == 0)
    {
        const int64_t signed_value = static_cast<int64_t>(encoded[0]);
        if (signed_value < 0) return false;
        *position = static_cast<size_t>(signed_value);
    }
    else if (encoding == 1)
        *position = static_cast<size_t>(encoded[0]);
    else if (encoding > 2)
    {
        if (encoded[0] != 0) return false;
        for (int word = encoding - 2; word > 1; --word)
            if (encoded[word] != 0) return false;
        *position = static_cast<size_t>(encoded[1]);
    }
    else return false;
    return *position < degree;
}

__global__ void primitive_raw_extract_kernel(PrimitiveRawDescriptors descriptors,
    PrimitiveCrtMetadata metadata, const uint64_t *position_value,
    int position_encoding, uint64_t *output, size_t magnitude_words,
    uint32_t *status, size_t degree)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    size_t position = 0;
    if (!primitive_position(position_value, position_encoding, degree, &position))
    {
        atomicCAS(status, 0U, kPrimitiveInvalidInput);
        return;
    }
    uint64_t words[kMaxWords];
    primitive_reconstruct(descriptors.limbs, metadata, position, words);
    output[0] = 0;
    for (size_t word = 0; word < magnitude_words; ++word)
        output[word + 1] = word < static_cast<size_t>(metadata.word_count) ? words[word] : 0;
}

__global__ void primitive_raw_pack_kernel(const uint64_t *bits, size_t bit_count,
    int bits_encoding, const uint64_t *coefficient_bits_value,
    int coefficient_bits_encoding, PrimitiveCrtMetadata metadata,
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor destination,
    uint64_t destination_modulus, uint32_t *status, size_t degree)
{
    size_t coefficient_bits = 0;
    const bool width_valid = bit_count < SIZE_MAX &&
        primitive_position(coefficient_bits_value, coefficient_bits_encoding,
            bit_count + 1, &coefficient_bits) &&
        coefficient_bits != 0 && degree <= SIZE_MAX / coefficient_bits &&
        degree * coefficient_bits == bit_count;
    if (!width_valid)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            primitive_set_status(status, kPrimitiveInvalidInput);
        return;
    }
    const size_t lane_words = bits_encoding > 2 ?
        static_cast<size_t>(bits_encoding - 1) : 1;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         coefficient < degree; coefficient += step)
    {
        uint64_t packed[kMaxWords];
        primitive_zero_words(packed, metadata.word_count);
        bool overflow = false;
        for (size_t bit = coefficient_bits; bit-- > 0;)
        {
            const uint64_t *lane = bits + (coefficient * coefficient_bits + bit) * lane_words;
            uint64_t value = lane[bits_encoding > 2 ? 1 : 0];
            bool valid = value <= 1;
            if (bits_encoding == 0 && static_cast<int64_t>(lane[0]) < 0) valid = false;
            if (bits_encoding > 2)
            {
                valid = valid && lane[0] == 0;
                for (size_t word = 2; word < lane_words; ++word)
                    valid = valid && lane[word] == 0;
            }
            if (!valid) primitive_set_status(status, kPrimitiveInvalidBit);
            overflow = overflow || (packed[metadata.word_count - 1] >> 63) != 0;
            primitive_words_shift_left_one(packed, metadata.word_count, value & 1U);
        }
        if (overflow ||
            !primitive_words_less(packed, metadata.modulus_words, metadata.word_count))
            primitive_set_status(status, kPrimitivePackedOutOfRange);
        uint64_t residue = 0;
        for (int word = metadata.word_count; word-- > 0;)
            residue = static_cast<uint64_t>(
                ((static_cast<unsigned __int128>(residue) << 64) | packed[word]) %
                destination_modulus);
        matrix_store_limb_u64(destination.base, 0, coefficient,
            destination.stride, destination.width, residue);
    }
}

__global__ void primitive_raw_threshold_kernel(PrimitiveRawDescriptors descriptors,
    PrimitiveCrtMetadata metadata, const uint64_t *plaintext_modulus,
    size_t plaintext_words, const uint64_t *length_value, int length_encoding,
    uint64_t *workspace, uint64_t *output, size_t output_count,
    size_t output_magnitude_words, bool output_bool, uint32_t *status,
    size_t degree)
{
    size_t actual_length = 0;
    if (!primitive_position(length_value, length_encoding, degree + 1,
            &actual_length) || actual_length != output_count)
    {
        primitive_set_status(status, kPrimitiveInvalidInput);
        return;
    }
    bool positive = plaintext_modulus[0] == 0;
    bool nonzero = false;
    for (size_t word = 0; word < plaintext_words; ++word)
        nonzero |= plaintext_modulus[word + 1] != 0;
    if (!positive || !nonzero)
    {
        primitive_set_status(status, kPrimitiveInvalidInput);
        return;
    }
    const size_t numerator_words = metadata.word_count + plaintext_words + 1;
    const size_t quotient_words = plaintext_words + 1;
    const size_t scratch_words = numerator_words + quotient_words;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         coefficient < output_count; coefficient += step)
    {
        uint64_t *numerator = workspace + coefficient * scratch_words;
        uint64_t *quotient = numerator + numerator_words;
        for (size_t word = 0; word < scratch_words; ++word) numerator[word] = 0;
        uint64_t value[kMaxWords];
        primitive_reconstruct(descriptors.limbs, metadata, coefficient, value);
        for (int left = 0; left < metadata.word_count; ++left)
        {
            uint64_t carry = 0;
            for (size_t right = 0; right < plaintext_words; ++right)
            {
                const unsigned __int128 product =
                    static_cast<unsigned __int128>(value[left]) *
                    plaintext_modulus[right + 1] + numerator[left + right] + carry;
                numerator[left + right] = static_cast<uint64_t>(product);
                carry = static_cast<uint64_t>(product >> 64);
            }
            for (size_t word = left + plaintext_words; carry && word < numerator_words; ++word)
            {
                const unsigned __int128 sum =
                    static_cast<unsigned __int128>(numerator[word]) + carry;
                numerator[word] = static_cast<uint64_t>(sum);
                carry = static_cast<uint64_t>(sum >> 64);
            }
            if (carry) primitive_set_status(status, kPrimitiveOverflow);
        }
        uint64_t carry = primitive_words_add_in_place(
            numerator, metadata.half_modulus_words, metadata.word_count);
        for (size_t word = metadata.word_count; carry && word < numerator_words; ++word)
        {
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(numerator[word]) + carry;
            numerator[word] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
        if (carry) primitive_set_status(status, kPrimitiveOverflow);

        uint64_t remainder[kMaxWords];
        primitive_zero_words(remainder, kMaxWords);
        for (size_t word = numerator_words; word-- > 0;)
        {
            for (int bit = 63; bit >= 0; --bit)
            {
                primitive_words_shift_left_one(remainder, metadata.word_count + 1,
                    (numerator[word] >> bit) & 1U);
                if (!primitive_words_less(remainder, metadata.modulus_words,
                        metadata.word_count + 1))
                {
                    primitive_words_sub(remainder, metadata.modulus_words,
                        metadata.word_count + 1);
                    if (word < quotient_words)
                        quotient[word] |= uint64_t(1) << bit;
                    else primitive_set_status(status, kPrimitiveOverflow);
                }
            }
        }
        bool equal_modulus = quotient[plaintext_words] == 0;
        bool quotient_nonzero = quotient[plaintext_words] != 0;
        for (size_t word = 0; word < plaintext_words; ++word)
        {
            equal_modulus &= quotient[word] == plaintext_modulus[word + 1];
            quotient_nonzero |= quotient[word] != 0;
        }
        if (quotient[plaintext_words]) primitive_set_status(status, kPrimitiveOverflow);
        uint64_t *result = output + coefficient * (output_magnitude_words + 1);
        result[0] = 0;
        for (size_t word = 0; word < output_magnitude_words; ++word)
            result[word + 1] = output_bool ?
                (word == 0 && quotient_nonzero && !equal_modulus ? 1 : 0) :
                (equal_modulus || word >= plaintext_words ? 0 : quotient[word]);
    }
}

}

extern "C" int gpu_raw_polynomial_values(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, void *output,
    size_t output_magnitude_words, uint32_t source_binding_base,
    uint32_t output_binding)
{
    PrimitiveRawDescriptors descriptors{};
    PrimitiveCrtMetadata metadata{};
    if (!output || !primitive_raw_view_metadata(ctx, source, stream_raw,
            &descriptors, &metadata) ||
        output_magnitude_words < static_cast<size_t>(metadata.word_count) ||
        output_magnitude_words > SIZE_MAX / sizeof(uint64_t) - 1 ||
        source->degree > SIZE_MAX / ((output_magnitude_words + 1) * sizeof(uint64_t)) ||
        source_binding_base > UINT32_MAX - source->limb_count)
        return set_error("invalid raw polynomial-values physical view");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    MxxGraphPatch patches[kMaxLimbs + 1];
    size_t patch_count = 0;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        patches[patch_count++] = primitive_raw_patch(0,
            static_cast<uint32_t>(offsetof(PrimitiveRawDescriptors, limbs) +
                limb * sizeof(descriptors.limbs[0]) +
                offsetof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor, base)),
            source_binding_base + static_cast<uint32_t>(limb));
    patches[patch_count++] = primitive_raw_patch(2, 0, output_binding);
    const uint32_t grid = static_cast<uint32_t>(
        std::min<size_t>((static_cast<size_t>(source->degree) + 127) / 128, 65535));
    return mxx_gpu_launch_kernel(ctx,
        reinterpret_cast<cudaStream_t>(stream_raw), primitive_raw_values_kernel,
        dim3(grid), dim3(128), 0, patches, patch_count,
        descriptors, metadata, static_cast<uint64_t *>(output),
        output_magnitude_words, static_cast<size_t>(source->degree));
}

extern "C" int gpu_raw_extract_coefficient(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const void *position,
    int position_encoding, void *output, size_t output_magnitude_words,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t position_binding, uint32_t output_binding,
    uint32_t status_binding)
{
    PrimitiveRawDescriptors descriptors{};
    PrimitiveCrtMetadata metadata{};
    if (!position || !output || !status ||
        position_encoding < 0 || position_encoding == 2 ||
        !primitive_raw_view_metadata(ctx, source, stream_raw,
            &descriptors, &metadata) ||
        output_magnitude_words < static_cast<size_t>(metadata.word_count) ||
        source_binding_base > UINT32_MAX - source->limb_count)
        return set_error("invalid raw extract-coefficient physical view");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    MxxGraphPatch patches[kMaxLimbs + 3];
    size_t patch_count = 0;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        patches[patch_count++] = primitive_raw_patch(0,
            static_cast<uint32_t>(offsetof(PrimitiveRawDescriptors, limbs) +
                limb * sizeof(descriptors.limbs[0]) +
                offsetof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor, base)),
            source_binding_base + static_cast<uint32_t>(limb));
    patches[patch_count++] = primitive_raw_patch(2, 0, position_binding);
    patches[patch_count++] = primitive_raw_patch(4, 0, output_binding);
    patches[patch_count++] = primitive_raw_patch(6, 0, status_binding);
    return mxx_gpu_launch_kernel(ctx,
        reinterpret_cast<cudaStream_t>(stream_raw), primitive_raw_extract_kernel,
        dim3(1), dim3(1), 0, patches, patch_count,
        descriptors, metadata, static_cast<const uint64_t *>(position),
        position_encoding, static_cast<uint64_t *>(output),
        output_magnitude_words, status, static_cast<size_t>(source->degree));
}

extern "C" int gpu_raw_pack_polynomial_coefficients(GpuContext *ctx,
    void *stream_raw, const void *bits, size_t bit_count, int bits_encoding,
    const void *coefficient_bits, int coefficient_bits_encoding,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t bits_binding, uint32_t coefficient_bits_binding,
    uint32_t destination_binding_base, uint32_t status_binding)
{
    PrimitiveCrtMetadata metadata{};
    if (!ctx || !stream_raw || !bits || !coefficient_bits || !status ||
        !destination || !destination->limbs || !bit_count ||
        bits_encoding < 0 || bits_encoding == 2 ||
        coefficient_bits_encoding < 0 || coefficient_bits_encoding == 2 ||
        destination->rows != 1 || destination->columns != 1 ||
        destination->row_origin != 0 || destination->column_origin != 0 ||
        destination->degree != static_cast<uint32_t>(ctx->N) ||
        destination->limb_count != ctx->moduli.size() ||
        destination_binding_base > UINT32_MAX - destination->limb_count ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream_raw) ||
        !primitive_valid_metadata(ctx, destination->limb_count, &metadata))
        return set_error("invalid raw polynomial pack physical view");
    const size_t lane_words = bits_encoding > 2 ?
        static_cast<size_t>(bits_encoding - 1) : 1;
    if (bit_count > SIZE_MAX / lane_words)
        return set_error("raw polynomial bit family size overflows");
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const auto &part = destination->limbs[limb];
        const unsigned __int128 span =
            static_cast<unsigned __int128>(destination->degree - 1) *
            part.coefficient_stride_bytes + part.word_bytes;
        if (!part.address || part.crt_limb_index != limb ||
            part.modulus != ctx->moduli[limb] ||
            (part.word_bytes != 4 && part.word_bytes != 8) ||
            part.coefficient_stride_bytes != part.word_bytes ||
            part.column_stride_bytes < span ||
            (part.word_bytes == 4 && part.modulus > UINT32_MAX) ||
            limb >= ctx->limb_gpu_ids.size() ||
            ctx->limb_gpu_ids[limb].x >= ctx->gpu_ids.size() ||
            ctx->gpu_ids[ctx->limb_gpu_ids[limb].x] != destination->physical_device)
            return set_error("invalid raw polynomial pack CRT limb");
    }
    if (cudaSetDevice(destination->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t grid = static_cast<uint32_t>(
        std::min<size_t>((static_cast<size_t>(destination->degree) + 127) / 128, 65535));
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const auto &part = destination->limbs[limb];
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor descriptor{
            reinterpret_cast<uint8_t *>(part.address),
            static_cast<size_t>(part.column_stride_bytes),
            static_cast<uint8_t>(part.word_bytes),
        };
        const MxxGraphPatch patches[] = {
            primitive_raw_patch(0, 0, bits_binding),
            primitive_raw_patch(3, 0, coefficient_bits_binding),
            primitive_raw_patch(6,
                offsetof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor, base),
                destination_binding_base + static_cast<uint32_t>(limb)),
            primitive_raw_patch(8, 0, status_binding),
        };
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            primitive_raw_pack_kernel, dim3(grid), dim3(128), 0,
            patches, std::size(patches), static_cast<const uint64_t *>(bits),
            bit_count, bits_encoding,
            static_cast<const uint64_t *>(coefficient_bits),
            coefficient_bits_encoding, metadata, descriptor, part.modulus,
            status, static_cast<size_t>(destination->degree));
        if (result != 0) return result;
    }
    return 0;
}

extern "C" int gpu_raw_threshold_decode(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const void *plaintext_modulus,
    size_t plaintext_words, const void *length_value, int length_encoding,
    void *workspace, size_t workspace_bytes, void *output,
    size_t output_count, size_t output_magnitude_words, bool output_bool,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t plaintext_binding, uint32_t length_binding,
    uint32_t workspace_binding, uint32_t output_binding,
    uint32_t status_binding)
{
    PrimitiveRawDescriptors descriptors{};
    PrimitiveCrtMetadata metadata{};
    if (!plaintext_modulus || !length_value || !workspace || !output || !status ||
        !plaintext_words || !output_count ||
        length_encoding < 0 || length_encoding == 2 ||
        !primitive_raw_view_metadata(ctx, source, stream_raw,
            &descriptors, &metadata) ||
        output_count > source->degree ||
        output_magnitude_words < plaintext_words ||
        source_binding_base > UINT32_MAX - source->limb_count)
        return set_error("invalid raw threshold-decode physical view");
    if (plaintext_words > (SIZE_MAX - metadata.word_count - 2) / 2)
        return set_error("raw threshold-decode workspace size overflows");
    const size_t scratch_words = metadata.word_count + 2 * plaintext_words + 2;
    if (
        output_count > SIZE_MAX / scratch_words / sizeof(uint64_t) ||
        workspace_bytes < output_count * scratch_words * sizeof(uint64_t))
        return set_error("raw threshold-decode workspace is too small");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    MxxGraphPatch patches[kMaxLimbs + 5];
    size_t patch_count = 0;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        patches[patch_count++] = primitive_raw_patch(0,
            static_cast<uint32_t>(offsetof(PrimitiveRawDescriptors, limbs) +
                limb * sizeof(descriptors.limbs[0]) +
                offsetof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor, base)),
            source_binding_base + static_cast<uint32_t>(limb));
    patches[patch_count++] = primitive_raw_patch(2, 0, plaintext_binding);
    patches[patch_count++] = primitive_raw_patch(4, 0, length_binding);
    patches[patch_count++] = primitive_raw_patch(6, 0, workspace_binding);
    patches[patch_count++] = primitive_raw_patch(7, 0, output_binding);
    patches[patch_count++] = primitive_raw_patch(11, 0, status_binding);
    const uint32_t grid = static_cast<uint32_t>(
        std::min<size_t>((output_count + 127) / 128, 65535));
    return mxx_gpu_launch_kernel(ctx,
        reinterpret_cast<cudaStream_t>(stream_raw), primitive_raw_threshold_kernel,
        dim3(grid), dim3(128), 0, patches, patch_count,
        descriptors, metadata, static_cast<const uint64_t *>(plaintext_modulus),
        plaintext_words, static_cast<const uint64_t *>(length_value),
        length_encoding, static_cast<uint64_t *>(workspace),
        static_cast<uint64_t *>(output), output_count, output_magnitude_words,
        output_bool, status, static_cast<size_t>(source->degree));
}
