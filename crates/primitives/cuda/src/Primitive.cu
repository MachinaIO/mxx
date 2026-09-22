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
constexpr uint32_t kInvalidBinding = UINT32_MAX;

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

__global__ void primitive_extract_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
    PrimitiveCrtMetadata metadata,
    size_t position,
    size_t output_words,
    uint64_t *output,
    uint32_t *status)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    uint64_t coefficient[kMaxWords];
    primitive_reconstruct(descriptors, metadata, position, coefficient);
    output[0] = 0;
    for (size_t word = 0; word < output_words; ++word)
        output[word + 1] = word < static_cast<size_t>(metadata.word_count) ? coefficient[word] : 0;
}

__global__ void primitive_threshold_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
    PrimitiveCrtMetadata metadata,
    const uint64_t *plaintext_modulus,
    size_t plaintext_words,
    uint64_t *workspace,
    size_t length,
    bool output_bool,
    uint64_t *output,
    uint32_t *status)
{
    const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coefficient >= length) return;
    const size_t numerator_words = metadata.word_count + plaintext_words + 1;
    uint64_t *numerator = workspace + coefficient * (numerator_words + plaintext_words);
    uint64_t *quotient = numerator + numerator_words;
    for (size_t word = 0; word < numerator_words + plaintext_words; ++word) numerator[word] = 0;
    uint64_t value[kMaxWords];
    primitive_reconstruct(descriptors, metadata, coefficient, value);
    // The modulus owner is a positive SignedWords scalar; skip its sign word.
    const uint64_t *modulus = plaintext_modulus + 1;
    for (int left = 0; left < metadata.word_count; ++left) {
        uint64_t carry = 0;
        for (size_t right = 0; right < plaintext_words; ++right) {
            const unsigned __int128 product = static_cast<unsigned __int128>(value[left]) * modulus[right]
                + numerator[left + right] + carry;
            numerator[left + right] = static_cast<uint64_t>(product);
            carry = product >> 64;
        }
        numerator[left + plaintext_words] = carry;
    }
    uint64_t carry = primitive_words_add_in_place(numerator, metadata.half_modulus_words, metadata.word_count);
    for (size_t word = metadata.word_count; carry && word < numerator_words; ++word) {
        const uint64_t previous = numerator[word];
        numerator[word] += carry;
        carry = numerator[word] < previous;
    }
    uint64_t remainder[kMaxWords];
    primitive_zero_words(remainder, kMaxWords);
    for (size_t word = numerator_words; word-- > 0;) {
        for (int bit = 63; bit >= 0; --bit) {
            primitive_words_shift_left_one(remainder, metadata.word_count + 1, (numerator[word] >> bit) & 1);
            if (!primitive_words_less(remainder, metadata.modulus_words, metadata.word_count + 1)) {
                primitive_words_sub(remainder, metadata.modulus_words, metadata.word_count + 1);
                if (word < plaintext_words) quotient[word] |= uint64_t(1) << bit;
                else primitive_set_status(status, kPrimitivePackedOutOfRange);
            }
        }
    }
    bool equal_modulus = true;
    bool nonzero = false;
    for (size_t word = 0; word < plaintext_words; ++word) {
        equal_modulus &= quotient[word] == modulus[word];
        nonzero |= quotient[word] != 0;
    }
    // Rounding can yield p exactly; the decoder returns its canonical residue.
    if (output_bool) output[coefficient] = nonzero && !equal_modulus ? 1 : 0;
    else if (plaintext_words == 1) output[coefficient] = equal_modulus ? 0 : quotient[0];
    else {
        output[coefficient * (plaintext_words + 1)] = 0;
        for (size_t word = 0; word < plaintext_words; ++word)
            output[coefficient * (plaintext_words + 1) + word + 1] = equal_modulus ? 0 : quotient[word];
    }
}

__global__ void primitive_pack_kernel(
    const uint64_t *bits,
    size_t bit_count,
    size_t coefficient_bits,
    PrimitiveCrtMetadata metadata,
    uint64_t *output,
    uint32_t *status,
    int bits_encoding)
{
    const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t coefficient_count = coefficient_bits == 0 ? 0 : bit_count / coefficient_bits;
    if (coefficient >= coefficient_count)
    {
        return;
    }
    uint64_t packed_words[kMaxWords];
    primitive_zero_words(packed_words, metadata.word_count);
    for (size_t bit = 0; bit < coefficient_bits; ++bit)
    {
        const size_t index = coefficient * coefficient_bits + bit;
        const uint64_t *encoded = bits + index * (bits_encoding > 2 ? bits_encoding - 1 : 1);
        const uint64_t value = encoded[bits_encoding > 2 ? 1 : 0];
        bool invalid = value > 1;
        if (bits_encoding > 2) {
            invalid |= encoded[0] != 0;
            for (int word = 2; word < bits_encoding - 1; ++word) invalid |= encoded[word] != 0;
        }
        if (invalid)
        {
            primitive_set_status(status, kPrimitiveInvalidBit);
        }
        if (bit / 64 < static_cast<size_t>(metadata.word_count))
            packed_words[bit / 64] |= (value & 1) << (bit % 64);
        else if (value != 0)
            primitive_set_status(status, kPrimitivePackedOutOfRange);
    }
    if (!primitive_words_less(packed_words, metadata.modulus_words, metadata.word_count))
    {
        primitive_set_status(status, kPrimitivePackedOutOfRange);
    }
    output[coefficient * (metadata.word_count + 1)] = 0;
    for (int word = 0; word < metadata.word_count; ++word)
        output[coefficient * (metadata.word_count + 1) + word + 1] = packed_words[word];
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

int primitive_register_patches(
    GpuContext *ctx,
    void *stream,
    const size_t *argument_sizes,
    size_t argument_count,
    const MxxGraphPatch *patches,
    size_t patch_count)
{
    MxxGraphPatch selected[4]{};
    size_t selected_count = 0;
    for (size_t i = 0; i < patch_count; ++i)
    {
        if (patches[i].binding_index != kInvalidBinding)
        {
            if (selected_count >= std::size(selected))
            {
                return set_error("too many primitive graph pointer patches");
            }
            selected[selected_count++] = patches[i];
        }
    }
    return selected_count == 0 ? 0 : mxx_graph_register_kernel_update_for_stream(
        ctx, stream, argument_sizes, argument_count, selected, selected_count);
}

} // namespace

extern "C" int gpu_primitive_extract_coefficient(
    GpuContext *ctx,
    const void *descriptors,
    size_t limb_count,
    size_t ring_dimension,
    size_t position,
    size_t output_words,
    void *output,
    uint32_t *status,
    void *stream_raw,
    uint32_t descriptor_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index)
{
    if (!ctx || !descriptors || !output || !stream_raw || position >= ring_dimension)
    {
        return set_error("invalid gpu_primitive_extract_coefficient arguments");
    }
    PrimitiveCrtMetadata metadata{};
    if (!primitive_valid_metadata(ctx, limb_count, &metadata) || output_words < static_cast<size_t>(metadata.word_count))
    {
        return set_error("invalid CRT metadata in gpu_primitive_extract_coefficient");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = status ? cudaMemsetAsync(status, 0, sizeof(uint64_t), stream) : cudaSuccess;
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    primitive_extract_kernel<<<1, 1, 0, stream>>>(
        static_cast<const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(descriptors), metadata, position, output_words,
        static_cast<uint64_t *>(output), status);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const size_t argument_sizes[] = {sizeof(void *), sizeof(PrimitiveCrtMetadata), sizeof(size_t), sizeof(size_t), sizeof(void *), sizeof(uint32_t *)};
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *), descriptor_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 4, 0, sizeof(void *), output_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 5, 0, sizeof(uint32_t *), status_binding_index, 0},
    };
    return primitive_register_patches(ctx, stream_raw, argument_sizes, 6, patches, 3);
}

extern "C" int gpu_primitive_threshold_decode(
    GpuContext *ctx,
    const void *descriptors,
    size_t limb_count,
    size_t ring_dimension,
    const uint64_t *plaintext_modulus,
    size_t plaintext_words,
    uint64_t *workspace,
    size_t length,
    bool output_bool,
    void *output,
    uint32_t *status,
    void *stream_raw,
    uint32_t descriptor_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index)
{
    if (!ctx || !descriptors || !output || !stream_raw || !plaintext_modulus || plaintext_words == 0 || !workspace ||
        length == 0 || length > ring_dimension)
    {
        return set_error("invalid gpu_primitive_threshold_decode arguments");
    }
    PrimitiveCrtMetadata metadata{};
    if (!primitive_valid_metadata(ctx, limb_count, &metadata))
    {
        return set_error("invalid CRT metadata in gpu_primitive_threshold_decode");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = status ? cudaMemsetAsync(status, 0, sizeof(uint64_t), stream) : cudaSuccess;
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const int threads = 128;
    const int blocks = static_cast<int>((length + threads - 1) / threads);
    primitive_threshold_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(descriptors), metadata,
        plaintext_modulus, plaintext_words, workspace, length, output_bool, static_cast<uint64_t *>(output), status);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const size_t argument_sizes[] = {sizeof(void *), sizeof(PrimitiveCrtMetadata), sizeof(void *), sizeof(size_t), sizeof(void *), sizeof(size_t), sizeof(bool), sizeof(void *), sizeof(uint32_t *)};
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *), descriptor_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 7, 0, sizeof(void *), output_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 8, 0, sizeof(uint32_t *), status_binding_index, 0},
    };
    return primitive_register_patches(ctx, stream_raw, argument_sizes, 9, patches, 3);
}

extern "C" int gpu_primitive_pack_polynomial_coefficients(
    GpuContext *ctx,
    const void *bits,
    size_t bit_count,
    size_t coefficient_bits,
    int bits_encoding,
    void *packed_output,
    uint32_t *status,
    void *stream_raw,
    uint32_t bits_binding_index,
    uint32_t output_binding_index,
    uint32_t status_binding_index)
{
    if (!ctx || !bits || !packed_output || !stream_raw || coefficient_bits == 0 ||
        bit_count == 0 || bit_count % coefficient_bits != 0)
    {
        return set_error("invalid gpu_primitive_pack_polynomial_coefficients arguments");
    }
    PrimitiveCrtMetadata metadata{};
    if (!primitive_valid_metadata(ctx, (ctx->level + 1), &metadata))
    {
        return set_error("invalid CRT metadata in gpu_primitive_pack_polynomial_coefficients");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = status ? cudaMemsetAsync(status, 0, sizeof(uint64_t), stream) : cudaSuccess;
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const size_t coefficient_count = bit_count / coefficient_bits;
    const int threads = 128;
    const int blocks = static_cast<int>((coefficient_count + threads - 1) / threads);
    primitive_pack_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint64_t *>(bits), bit_count, coefficient_bits, metadata,
        static_cast<uint64_t *>(packed_output), status, bits_encoding);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const size_t argument_sizes[] = {sizeof(void *), sizeof(size_t), sizeof(size_t), sizeof(PrimitiveCrtMetadata), sizeof(void *), sizeof(uint32_t *), sizeof(int)};
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *), bits_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 4, 0, sizeof(void *), output_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 5, 0, sizeof(uint32_t *), status_binding_index, 0},
        {nullptr, MXX_GRAPH_PATCH_INTEGER_ENCODING, 6, 0, sizeof(int), bits_binding_index, reinterpret_cast<uint64_t>(bits)},
    };
    return primitive_register_patches(ctx, stream_raw, argument_sizes, 7, patches, 4);
}
