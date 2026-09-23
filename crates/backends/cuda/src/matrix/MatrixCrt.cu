#include <algorithm>
#include <memory>

// The plan is opaque at the C ABI boundary. Its metadata is prepared before
// graph construction and retained until the caller destroys the plan.
struct GpuModulusConversionPlan
{
    GpuContext *context = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    void *device_metadata = nullptr;
    void *pinned_metadata = nullptr;
    size_t metadata_bytes = 0;
    size_t coefficient_count = 0;
    size_t ring_dimension = 0;
    bool round_scale = false;
    size_t source_count = 0;
    size_t target_count = 0;
    bool centered_rebase = false;
    bool centered_round_divide = false;
    bool round_divide_zero = false;
    bool raw_rns = false;
    bool raw_block = false;
    bool raw_recompose = false;
    bool raw_compact_pack = false;
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor *raw_source_descriptors = nullptr;
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor *raw_target_descriptors = nullptr;
    cudaEvent_t raw_ready = nullptr;
    // Set when a compiled-submission arm fails after graph work may have been
    // queued. Destruction then intentionally leaks the metadata rather than
    // freeing it without a proven completion dependency.
    bool release_blocked = false;
};

extern "C" int gpu_matrix_modulus_conversion_plan_allocation_range(
    const GpuModulusConversionPlan *plan, uint64_t *address, size_t *bytes)
{
    if (!plan || !address || !bytes || !plan->device_metadata || !plan->metadata_bytes)
        return set_error("invalid modulus conversion metadata range query");
    *address = reinterpret_cast<uint64_t>(plan->device_metadata);
    *bytes = plan->metadata_bytes;
    return 0;
}

namespace
{
    __global__ void copy_crt_descriptors_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *destination,
        size_t coefficient_count, size_t ring_dimension, size_t limb_count)
    {
        const size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
        if (index >= coefficient_count * limb_count) return;
        const size_t limb = index / coefficient_count;
        const size_t coefficient = index % ring_dimension;
        const size_t poly = (index % coefficient_count) / ring_dimension;
        const auto input = source[limb];
        const auto output = destination[limb];
        matrix_store_limb_u64(output.base, poly, coefficient, output.stride, output.width,
            matrix_load_limb_u64(input.base, poly, coefficient, input.stride, input.width));
    }

    constexpr int kCrtMaxLimbs = 64;
    constexpr int kCrtMaxWords = 64;

    bool mod_inverse_u64(uint64_t value, uint64_t modulus, uint64_t &inverse)
    {
        if (!modulus || value % modulus == 0) return false;
        // CRT primes are below 2^60 in the native backend, so signed
        // __int128 is sufficient for the extended Euclidean coefficients.
        uint64_t r0 = modulus, r1 = value % modulus;
        __int128 t0 = 0, t1 = 1;
        while (r1 != 0)
        {
            const uint64_t quotient = r0 / r1;
            const uint64_t next_r = r0 - quotient * r1;
            const __int128 next_t = t0 - static_cast<__int128>(quotient) * t1;
            r0 = r1;
            r1 = next_r;
            t0 = t1;
            t1 = next_t;
        }
        if (r0 != 1) return false;
        const __int128 reduced = t0 % static_cast<__int128>(modulus);
        inverse = static_cast<uint64_t>(reduced < 0 ? reduced + modulus : reduced);
        return true;
    }

    struct ModulusConversionMetadata
    {
        size_t source_count;
        size_t target_count;
        size_t discarded_count;
        size_t retained[kCrtMaxLimbs];
        size_t discarded[kCrtMaxLimbs];
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t target_moduli[kCrtMaxLimbs];
        uint64_t division_inverses[kCrtMaxLimbs];
        uint64_t divisor_residues[kCrtMaxLimbs];
        uint64_t garner[kCrtMaxLimbs * kCrtMaxLimbs];
    };

    __global__ void convert_modulus_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const ModulusConversionMetadata *metadata,
        size_t coefficient_count,
        size_t ring_dimension,
        bool round_scale)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t digits[kCrtMaxLimbs];
        bool negative = false;
        if (round_scale)
        {
            for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            {
                const size_t source_index = metadata->discarded[limb];
                const auto descriptor = source[source_index];
                const uint64_t modulus = metadata->source_moduli[source_index];
                uint64_t digit = matrix_load_limb_u64(descriptor.base, poly, coefficient,
                    descriptor.stride, descriptor.width) % modulus;
                for (size_t previous = 0; previous < limb; ++previous)
                {
                    const uint64_t residue = digits[previous] % modulus;
                    const uint64_t difference = digit >= residue ? digit - residue :
                        modulus - (residue - digit);
                    digit = mul_mod_u64(difference,
                        metadata->garner[previous * kCrtMaxLimbs + limb], modulus);
                }
                digits[limb] = digit;
            }
            // For odd radices, (J-1)/2 has mixed-radix digits (q_i-1)/2.
            // Comparing from the most significant digit chooses the exact
            // centered representative of x mod J without a large-integer divide.
            for (size_t limb = metadata->discarded_count; limb-- > 0;)
            {
                const uint64_t half = metadata->source_moduli[metadata->discarded[limb]] / 2;
                if (digits[limb] != half)
                {
                    negative = digits[limb] > half;
                    break;
                }
            }
        }
        for (size_t limb = 0; limb < metadata->target_count; ++limb)
        {
            const uint64_t modulus = metadata->target_moduli[limb];
            const auto input = source[metadata->retained[limb]];
            uint64_t value = matrix_load_limb_u64(input.base, poly, coefficient,
                input.stride, input.width) % modulus;
            if (round_scale)
            {
                uint64_t residual = 0;
                for (size_t digit = metadata->discarded_count; digit-- > 0;)
                {
                    residual = add_mod_u64(mul_mod_u64(residual,
                        metadata->source_moduli[metadata->discarded[digit]] % modulus,
                        modulus), digits[digit] % modulus, modulus);
                }
                if (negative)
                {
                    const uint64_t divisor = metadata->divisor_residues[limb];
                    residual = residual >= divisor ? residual - divisor :
                        modulus - (divisor - residual);
                }
                value = value >= residual ? value - residual : modulus - (residual - value);
                value = mul_mod_u64(value, metadata->division_inverses[limb], modulus);
            }
            const auto output = target[limb];
            matrix_store_limb_u64(output.base, poly, coefficient,
                output.stride, output.width, value);
        }
    }

    __device__ __forceinline__ int crt_compare_words(
        const uint64_t *lhs,
        const uint64_t *rhs,
        int word_count)
    {
        for (int word = word_count - 1; word >= 0; --word)
        {
            if (lhs[word] != rhs[word])
            {
                return lhs[word] > rhs[word] ? 1 : -1;
            }
        }
        return 0;
    }

    __device__ uint64_t crt_rounded_scale(
        uint64_t *value_words,
        const uint64_t *modulus_words,
        int word_count,
        uint64_t plaintext_modulus)
    {
        // Form plaintext_modulus * value + floor(Q / 2) exactly. The extra
        // word contains the multiplication/addition carry.
        uint64_t carry = 0;
        for (int word = 0; word < word_count; ++word)
        {
            const unsigned __int128 product =
                static_cast<unsigned __int128>(value_words[word]) * plaintext_modulus + carry;
            value_words[word] = static_cast<uint64_t>(product);
            carry = static_cast<uint64_t>(product >> 64);
        }
        value_words[word_count] = carry;

        uint64_t add_carry = 0;
        for (int word = 0; word < word_count; ++word)
        {
            const uint64_t current = modulus_words[word];
            const uint64_t next = word + 1 < word_count ? modulus_words[word + 1] : 0;
            const uint64_t half_word = (current >> 1) | ((next & 1) << 63);
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(value_words[word]) + half_word + add_carry;
            value_words[word] = static_cast<uint64_t>(sum);
            add_carry = static_cast<uint64_t>(sum >> 64);
        }
        value_words[word_count] += add_carry;

        // The quotient is at most plaintext_modulus. Binary search avoids a
        // general multi-word division while retaining exact integer rounding.
        uint64_t low = 0;
        uint64_t high = plaintext_modulus;
        uint64_t multiple[kCrtMaxWords + 1];
        while (low < high)
        {
            const uint64_t midpoint = low + (high - low) / 2 + (high - low) % 2;
            uint64_t mul_carry = 0;
            for (int word = 0; word < word_count; ++word)
            {
                const unsigned __int128 product =
                    static_cast<unsigned __int128>(modulus_words[word]) * midpoint + mul_carry;
                multiple[word] = static_cast<uint64_t>(product);
                mul_carry = static_cast<uint64_t>(product >> 64);
            }
            multiple[word_count] = mul_carry;
            if (crt_compare_words(value_words, multiple, word_count + 1) >= 0)
            {
                low = midpoint;
            }
            else
            {
                high = midpoint - 1;
            }
        }
        return low % plaintext_modulus;
    }

    struct CrtLevelMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        size_t limb_count;
        int word_count;
        uint64_t plaintext_modulus;
        uint64_t moduli[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        uint64_t modulus_words[kCrtMaxWords + 1];
        uint64_t reconstruction[kCrtMaxLimbs];
    };

    struct CrtOutputMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        size_t limb_count;
        uint64_t moduli[kCrtMaxLimbs];
    };

    // Recover the exact mixed-radix digits of a CRT value and select its
    // centered representative.  `selected_indices == nullptr` means the
    // source basis is contiguous; otherwise it names the source limbs used
    // by the dropped block.  Keeping both cases in one device helper avoids
    // subtly diverging center-boundary behavior between CRT operations while
    // retaining the caller-owned local digit array.
    __device__ __forceinline__ bool crt_recover_centered_digits(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const size_t *selected_indices,
        const uint64_t *source_moduli,
        const uint64_t *residue_inverses,
        const uint64_t *prefix_inverses,
        size_t count,
        size_t poly,
        size_t coefficient,
        uint64_t *digits)
    {
        for (size_t current = 0; current < count; ++current)
        {
            const size_t source_index = selected_indices ? selected_indices[current] : current;
            const uint64_t modulus = source_moduli[source_index];
            const auto input = source[source_index];
            const uint64_t raw = matrix_load_limb_u64(input.base, poly, coefficient,
                input.stride, input.width) % modulus;
            const uint64_t residue = residue_inverses
                ? mul_mod_u64(raw, residue_inverses[current], modulus)
                : raw;
            uint64_t prefix = 0;
            uint64_t weight = 1;
            for (size_t previous = 0; previous < current; ++previous)
            {
                const size_t previous_index = selected_indices ? selected_indices[previous] : previous;
                const uint64_t previous_modulus = source_moduli[previous_index];
                prefix = add_mod_u64(prefix,
                    mul_mod_u64(digits[previous] % modulus, weight, modulus), modulus);
                weight = mul_mod_u64(weight, previous_modulus % modulus, modulus);
            }
            const uint64_t difference = residue >= prefix ? residue - prefix : modulus - (prefix - residue);
            digits[current] = mul_mod_u64(difference, prefix_inverses[current], modulus);
        }
        for (size_t current = count; current-- > 0;)
        {
            const size_t source_index = selected_indices ? selected_indices[current] : current;
            const uint64_t half = (source_moduli[source_index] - 1) / 2;
            if (digits[current] != half) return digits[current] > half;
        }
        return false;
    }

    struct CenteredRebaseMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        CrtOutputMetadata output;
        size_t source_count;
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        int target_source[kCrtMaxLimbs];
    };

    struct CenteredRoundDivideMetadata
    {
        size_t limb_count;
        int word_count;
        int divisor_word_count;
        uint64_t moduli[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        uint64_t modulus_words[kCrtMaxWords + 1];
        uint64_t divisor_words[kCrtMaxWords];
    };

    __device__ __forceinline__ int compare_words(
        const uint64_t *lhs, const uint64_t *rhs, int count)
    {
        for (int word = count - 1; word >= 0; --word)
        {
            if (lhs[word] != rhs[word]) return lhs[word] > rhs[word] ? 1 : -1;
        }
        return 0;
    }

    __device__ __forceinline__ void subtract_words(
        uint64_t *lhs, const uint64_t *rhs, int count)
    {
        uint64_t borrow = 0;
        for (int word = 0; word < count; ++word)
        {
            const uint64_t subtrahend = rhs[word] + borrow;
            const uint64_t next_borrow = (subtrahend < rhs[word]) || (lhs[word] < subtrahend);
            lhs[word] -= subtrahend;
            borrow = next_borrow;
        }
    }

    __device__ __forceinline__ void divide_centered_words(
        const uint64_t *magnitude, const CenteredRoundDivideMetadata &metadata,
        const uint64_t *divisor_words, int divisor_word_count,
        bool negative, uint64_t *quotient)
    {
        uint64_t remainder[kCrtMaxWords + 1]{};
        uint64_t divisor[kCrtMaxWords + 1]{};
        for (int word = 0; word < divisor_word_count; ++word)
            divisor[word] = divisor_words[word];
        const int total_words = metadata.word_count + 1;
        // A divisor with a nonzero word above the value's maximum width
        // makes the rounded quotient exactly zero.
        if (divisor_word_count > total_words)
        {
            for (int word = total_words; word < divisor_word_count; ++word)
                if (divisor_words[word] != 0) return;
        }
        for (int bit = metadata.word_count * 64 - 1; bit >= 0; --bit)
        {
            uint64_t carry = (magnitude[bit / 64] >> (bit % 64)) & 1;
            for (int word = 0; word < total_words; ++word)
            {
                const uint64_t next = remainder[word] >> 63;
                remainder[word] = (remainder[word] << 1) | carry;
                carry = next;
            }
            if (compare_words(remainder, divisor, total_words) >= 0)
            {
                subtract_words(remainder, divisor, total_words);
                quotient[bit / 64] |= uint64_t(1) << (bit % 64);
            }
        }
        uint64_t doubled[kCrtMaxWords + 1]{};
        uint64_t carry = 0;
        for (int word = 0; word < total_words; ++word)
        {
            doubled[word] = (remainder[word] << 1) | carry;
            carry = remainder[word] >> 63;
        }
        const int rounded = negative ? compare_words(doubled, divisor, total_words) > 0 :
            compare_words(doubled, divisor, total_words) >= 0;
        if (rounded)
        {
            uint64_t add_carry = 1;
            for (int word = 0; word < metadata.word_count && add_carry; ++word)
            {
                const uint64_t next = quotient[word] + add_carry;
                add_carry = next < quotient[word];
                quotient[word] = next;
            }
        }
    }

    __global__ void centered_round_divide_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const CenteredRoundDivideMetadata *metadata,
        size_t coefficient_count, size_t ring_dimension)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t digits[kCrtMaxLimbs]{};
        const bool negative = crt_recover_centered_digits(
            source, nullptr, metadata->moduli, nullptr, metadata->prefix_inverses,
            metadata->limb_count, poly, coefficient, digits);
        uint64_t value[kCrtMaxWords + 1]{};
        for (size_t limb = metadata->limb_count; limb-- > 0;)
        {
            uint64_t carry = digits[limb];
            for (int word = 0; word < metadata->word_count; ++word)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(value[word]) * metadata->moduli[limb] + carry;
                value[word] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
        }
        if (negative)
        {
            uint64_t magnitude[kCrtMaxWords + 1]{};
            for (int word = 0; word <= metadata->word_count; ++word)
                magnitude[word] = metadata->modulus_words[word];
            subtract_words(magnitude, value, metadata->word_count + 1);
            for (int word = 0; word <= metadata->word_count; ++word) value[word] = magnitude[word];
        }
        uint64_t quotient[kCrtMaxWords]{};
        divide_centered_words(value, *metadata, metadata->divisor_words,
            metadata->divisor_word_count, negative, quotient);
        for (size_t limb = 0; limb < metadata->limb_count; ++limb)
        {
            const uint64_t modulus = metadata->moduli[limb];
            uint64_t residue = 0;
            const uint64_t radix = static_cast<uint64_t>((static_cast<unsigned __int128>(1) << 64) % modulus);
            for (int word = metadata->word_count; word-- > 0;)
                residue = static_cast<uint64_t>((static_cast<unsigned __int128>(residue) * radix + quotient[word]) % modulus);
            if (negative && residue) residue = modulus - residue;
            const auto output = target[limb];
            matrix_store_limb_u64(output.base, poly, coefficient, output.stride, output.width, residue);
        }
    }

    __global__ void centered_rebase_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const CenteredRebaseMetadata *metadata,
        size_t coefficient_count, size_t ring_dimension)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * metadata->output.limb_count) return;
        const size_t limb = index / coefficient_count;
        const size_t position = index % coefficient_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        uint64_t digits[kCrtMaxLimbs];
        const bool negative = crt_recover_centered_digits(
            source, nullptr, metadata->source_moduli, nullptr,
            metadata->prefix_inverses, metadata->source_count, poly, coefficient, digits);
        const uint64_t modulus = metadata->output.moduli[limb];
        uint64_t value = 0;
        const int source_index = metadata->target_source[limb];
        if (source_index >= 0)
        {
            const auto input = source[source_index];
            value = matrix_load_limb_u64(input.base, poly, coefficient,
                input.stride, input.width) % modulus;
        }
        else
        {
            for (size_t current = metadata->source_count; current-- > 0;)
                value = (static_cast<unsigned __int128>(value) *
                    (metadata->source_moduli[current] % modulus) + digits[current] % modulus) % modulus;
            if (negative)
            {
                uint64_t source_product = 1;
                for (size_t current = 0; current < metadata->source_count; ++current)
                    source_product = mul_mod_u64(source_product,
                        metadata->source_moduli[current] % modulus, modulus);
                value = value >= source_product ? value - source_product : modulus - (source_product - value);
            }
        }
        const auto output = target[limb];
        matrix_store_limb_u64(output.base, poly, coefficient, output.stride, output.width, value);
    }

    struct BlockModSwitchMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        CrtOutputMetadata output;
        size_t source_count;
        size_t dropped_count;
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t dropped_t_inverses[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        uint64_t target_t[kCrtMaxLimbs];
        uint64_t target_p_inverses[kCrtMaxLimbs];
        size_t dropped_indices[kCrtMaxLimbs];
        int target_source[kCrtMaxLimbs];
    };

    __global__ void block_mod_switch_kernel(
        const BlockModSwitchMetadata metadata_value,
        size_t coefficient_count, size_t ring_dimension)
    {
        const BlockModSwitchMetadata *metadata = &metadata_value;
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * metadata->output.limb_count) return;
        const size_t target_limb = index / coefficient_count;
        const size_t position = index % coefficient_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        uint64_t digits[kCrtMaxLimbs];
        const bool negative = crt_recover_centered_digits(
            metadata->source, metadata->dropped_indices, metadata->source_moduli,
            metadata->dropped_t_inverses, metadata->prefix_inverses, metadata->dropped_count,
            poly, coefficient, digits);
        const uint64_t modulus = metadata->output.moduli[target_limb];
        uint64_t centered = 0;
        for (size_t current = metadata->dropped_count; current-- > 0;)
        {
            const uint64_t p = metadata->source_moduli[metadata->dropped_indices[current]];
            centered = (static_cast<unsigned __int128>(centered) * (p % modulus) +
                digits[current] % modulus) % modulus;
        }
        uint64_t dropped_product = 1;
        for (size_t current = 0; current < metadata->dropped_count; ++current)
            dropped_product = mul_mod_u64(dropped_product,
                metadata->source_moduli[metadata->dropped_indices[current]] % modulus, modulus);
        if (negative)
            centered = centered >= dropped_product ? centered - dropped_product : modulus - (dropped_product - centered);
        const auto retained = metadata->source[metadata->target_source[target_limb]];
        const uint64_t z = matrix_load_limb_u64(retained.base, poly, coefficient,
            retained.stride, retained.width) % modulus;
        const uint64_t correction = mul_mod_u64(metadata->target_t[target_limb], centered, modulus);
        const uint64_t numerator = z >= correction ? z - correction : modulus - (correction - z);
        const uint64_t value = mul_mod_u64(numerator, metadata->target_p_inverses[target_limb], modulus);
        const auto output = metadata->output.descriptors[target_limb];
        matrix_store_limb_u64(output.base, poly, coefficient, output.stride, output.width, value);
    }

    __global__ void crt_recompose_kernel(
        CrtLevelMetadata metadata, CrtOutputMetadata output,
        bool initialize, size_t coefficient_count, size_t ring_dimension)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        {
            uint64_t mixed_digits[kCrtMaxLimbs];
            crt_recover_centered_digits(metadata.descriptors, nullptr, metadata.moduli,
                nullptr, metadata.prefix_inverses, metadata.limb_count, poly, coefficient,
                mixed_digits);
            uint64_t value_words[kCrtMaxWords + 1]{};
            for (size_t limb = metadata.limb_count; limb-- > 0;)
            {
                uint64_t carry = mixed_digits[limb];
                for (int word = 0; word < metadata.word_count; ++word)
                {
                    const unsigned __int128 term =
                        static_cast<unsigned __int128>(value_words[word]) * metadata.moduli[limb] + carry;
                    value_words[word] = static_cast<uint64_t>(term);
                    carry = static_cast<uint64_t>(term >> 64);
                }
            }
            const uint64_t rounded = crt_rounded_scale(value_words, metadata.modulus_words,
                metadata.word_count, metadata.plaintext_modulus);
            for (size_t limb = 0; limb < output.limb_count; ++limb)
            {
                const uint64_t modulus = output.moduli[limb];
                const auto descriptor = output.descriptors[limb];
                const uint64_t previous = initialize ? 0 : matrix_load_limb_u64(
                    descriptor.base, poly, coefficient, descriptor.stride, descriptor.width);
                const uint64_t accumulated = add_mod_u64(previous,
                    mul_mod_u64(rounded % modulus, metadata.reconstruction[limb], modulus), modulus);
                matrix_store_limb_u64(descriptor.base, poly, coefficient,
                    descriptor.stride, descriptor.width, accumulated);
            }
        }
    }

    template <typename T>
    int crt_alloc_and_copy_async(
        T **device,
        const std::vector<T> &host,
        cudaStream_t stream,
        std::vector<void *> *pinned_metadata)
    {
        *device = nullptr;
        if (host.empty() || !pinned_metadata)
        {
            return host.empty() ? 0 : set_error("missing CRT pinned-metadata owner");
        }
        const size_t bytes = host.size() * sizeof(T);
        void *allocation = nullptr;
        cudaError_t error = cudaMallocAsync(&allocation, bytes, stream);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        *device = static_cast<T *>(allocation);
        void *pinned = nullptr;
        error = cudaHostAlloc(&pinned, bytes, cudaHostAllocPortable);
        if (error != cudaSuccess)
        {
            cudaFreeAsync(*device, stream);
            *device = nullptr;
            return set_error(error);
        }
        std::memcpy(pinned, host.data(), bytes);
        pinned_metadata->push_back(pinned);
        error = cudaMemcpyAsync(*device, pinned, bytes, cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess)
        {
            cudaFreeAsync(*device, stream);
            *device = nullptr;
            return set_error(error);
        }
        return 0;
    }

    int build_modulus_conversion_metadata(
        ModulusConversionMetadata *metadata,
        const GpuMatrix *out,
        const GpuMatrix *source,
        int round_scale,
        const uint64_t *division_inverses,
        size_t inverse_count)
    {
        if (!metadata || !out || !source || !out->ctx || !source->ctx ||
            out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
            out->rows != source->rows || out->cols != source->cols ||
            out->format != source->format ||
            (round_scale != 0 && round_scale != 1) || source->level < 0 || out->level < 0 ||
            source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
            return set_error("invalid coefficient modulus conversion layout");
        const size_t source_count = static_cast<size_t>(source->level) + 1;
        const size_t target_count = static_cast<size_t>(out->level) + 1;
        if (source_count > kCrtMaxLimbs || target_count > source_count ||
            source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size() ||
            !division_inverses || inverse_count != target_count)
            return set_error("invalid coefficient modulus conversion basis");

        *metadata = ModulusConversionMetadata{};
        metadata->source_count = source_count;
        metadata->target_count = target_count;
        bool retained[kCrtMaxLimbs]{};
        for (size_t limb = 0; limb < source_count; ++limb)
        {
            metadata->source_moduli[limb] = source->ctx->moduli[limb];
            if (metadata->source_moduli[limb] <= 1 || !(metadata->source_moduli[limb] & 1))
                return set_error("nearest conversion requires odd CRT moduli");
        }
        for (size_t limb = 0; limb < target_count; ++limb)
        {
            const uint64_t modulus = out->ctx->moduli[limb];
            size_t selected = 0;
            while (selected < source_count && metadata->source_moduli[selected] != modulus) ++selected;
            if (selected == source_count || retained[selected])
                return set_error("destination CRT basis is not an exact subset");
            retained[selected] = true;
            metadata->retained[limb] = selected;
            metadata->target_moduli[limb] = modulus;
            metadata->division_inverses[limb] = division_inverses[limb];
        }
        for (size_t limb = 0; limb < source_count; ++limb)
            if (!retained[limb]) metadata->discarded[metadata->discarded_count++] = limb;
        for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            for (size_t previous = 0; previous < limb; ++previous)
                metadata->garner[previous * kCrtMaxLimbs + limb] = source->ctx->garner_inverse_table[
                    metadata->discarded[previous] * source->ctx->moduli.size() + metadata->discarded[limb]];
        for (size_t limb = 0; limb < target_count; ++limb)
        {
            const uint64_t modulus = metadata->target_moduli[limb];
            uint64_t divisor = 1;
            for (size_t discarded = 0; discarded < metadata->discarded_count; ++discarded)
                divisor = static_cast<uint64_t>((static_cast<unsigned __int128>(divisor) *
                    metadata->source_moduli[metadata->discarded[discarded]]) % modulus);
            metadata->divisor_residues[limb] = divisor;
            if (round_scale && (static_cast<unsigned __int128>(divisor) * division_inverses[limb]) % modulus != 1)
                return set_error("invalid exact modulus division inverse");
        }
        const auto &input = source->shared_limb_buffers[0];
        const auto &output = out->shared_limb_buffers[0];
        if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
            input.limb_count < source_count || output.limb_count < target_count)
            return set_error("coefficient conversion requires colocated device descriptors");
        return 0;
    }

    int build_centered_rebase_metadata(
        CenteredRebaseMetadata *metadata,
        const GpuMatrix *out,
        const GpuMatrix *source)
    {
        if (!metadata || !out || !source || !out->ctx || !source->ctx ||
            out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
            source->level < 0 || out->level < 0 || source->rows != out->rows ||
            source->cols != out->cols || source->shared_limb_buffers.size() != 1 ||
            out->shared_limb_buffers.size() != 1)
            return set_error("invalid centered rebase layout");
        const size_t source_count = static_cast<size_t>(source->level) + 1;
        const size_t target_count = static_cast<size_t>(out->level) + 1;
        if (!source_count || source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs ||
            source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size())
            return set_error("invalid centered rebase destination basis");
        for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
        {
            const uint64_t p = source->ctx->moduli[source_limb];
            if (p <= 1 || !(p & 1)) return set_error("centered rebase requires odd source CRT moduli");
        }
        for (size_t target_limb = 0; target_limb < target_count; ++target_limb)
            if (out->ctx->moduli[target_limb] <= 1 || !(out->ctx->moduli[target_limb] & 1))
                return set_error("centered rebase requires odd destination CRT moduli");
        std::memset(metadata, 0, sizeof(*metadata));
        metadata->source = source->shared_limb_buffers[0].device_descriptors;
        metadata->output.descriptors = out->shared_limb_buffers[0].device_descriptors;
        metadata->output.limb_count = target_count;
        metadata->source_count = source_count;
        std::copy_n(out->ctx->moduli.begin(), target_count, metadata->output.moduli);
        std::copy_n(source->ctx->moduli.begin(), source_count, metadata->source_moduli);
        metadata->prefix_inverses[0] = 1;
        for (size_t current = 1; current < source_count; ++current)
        {
            const uint64_t modulus = metadata->source_moduli[current];
            uint64_t product = 1;
            for (size_t previous = 0; previous < current; ++previous)
                product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                    metadata->source_moduli[previous]) % modulus);
            uint64_t inverse = 0;
            if (!mod_inverse_u64(product, modulus, inverse))
                return set_error("centered rebase source CRT basis is not invertible");
            metadata->prefix_inverses[current] = inverse;
        }
        for (size_t target_limb = 0; target_limb < target_count; ++target_limb)
        {
            metadata->target_source[target_limb] = -1;
            for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
                if (out->ctx->moduli[target_limb] == source->ctx->moduli[source_limb])
                    metadata->target_source[target_limb] = static_cast<int>(source_limb);
        }
        return 0;
    }

    int build_centered_round_divide_metadata(
        CenteredRoundDivideMetadata *metadata,
        const GpuMatrix *out,
        const GpuMatrix *source,
        const uint64_t *divisor_words,
        size_t divisor_word_count)
    {
        bool divisor_nonzero = false;
        if (divisor_words)
            for (size_t word = 0; word < divisor_word_count; ++word)
                divisor_nonzero = divisor_nonzero || divisor_words[word] != 0;
        if (!metadata || !out || !source || !out->ctx || !source->ctx ||
            out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
            out->rows != source->rows || out->cols != source->cols ||
            out->format != GPU_POLY_FORMAT_COEFF ||
            source->level < 0 || out->level < 0 || source->level != out->level ||
            source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1 ||
            !divisor_words || !divisor_word_count || divisor_word_count > kCrtMaxWords ||
            !divisor_nonzero)
            return set_error("invalid centered round divide layout");
        const size_t limb_count = static_cast<size_t>(source->level) + 1;
        if (!limb_count || limb_count > kCrtMaxLimbs ||
            limb_count > source->ctx->moduli.size() || limb_count > out->ctx->moduli.size())
            return set_error("invalid centered round divide CRT basis");
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            if (source->ctx->moduli[limb] != out->ctx->moduli[limb] ||
                source->ctx->moduli[limb] <= 1 || !(source->ctx->moduli[limb] & 1))
                return set_error("centered round divide requires the same odd CRT basis");
        }
        const auto &input = source->shared_limb_buffers[0];
        const auto &output = out->shared_limb_buffers[0];
        if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
            input.limb_count < limb_count || output.limb_count < limb_count)
            return set_error("centered round divide requires colocated device descriptors");
        std::memset(metadata, 0, sizeof(*metadata));
        metadata->limb_count = limb_count;
        metadata->divisor_word_count = static_cast<int>(divisor_word_count);
        std::copy_n(divisor_words, divisor_word_count, metadata->divisor_words);
        std::copy_n(source->ctx->moduli.begin(), limb_count, metadata->moduli);
        metadata->prefix_inverses[0] = 1;
        for (size_t current = 1; current < limb_count; ++current)
        {
            uint64_t product = 1;
            for (size_t previous = 0; previous < current; ++previous)
                product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                    metadata->moduli[previous]) % metadata->moduli[current]);
            uint64_t inverse = 0;
            if (!mod_inverse_u64(product, metadata->moduli[current], inverse))
                return set_error("centered round divide CRT basis is not invertible");
            metadata->prefix_inverses[current] = inverse;
        }
        std::vector<uint64_t> moduli(source->ctx->moduli.begin(), source->ctx->moduli.begin() + limb_count);
        std::vector<uint64_t> modulus_words;
        if (!serde_compute_modulus_words_le(moduli, &modulus_words) || modulus_words.empty() ||
            modulus_words.size() > kCrtMaxWords)
            return set_error("unsupported centered round divide modulus size");
        metadata->word_count = static_cast<int>(modulus_words.size());
        std::copy(modulus_words.begin(), modulus_words.end(), metadata->modulus_words);
        return 0;
    }

    int modulus_conversion_shape(
        const GpuMatrix *out,
        size_t *coefficient_count,
        size_t *ring_dimension)
    {
        if (!out || !out->ctx || !coefficient_count || !ring_dimension || out->rows == 0 ||
            out->cols == 0 || out->ctx->N <= 0 ||
            out->rows > std::numeric_limits<size_t>::max() / out->cols ||
            out->rows * out->cols > std::numeric_limits<size_t>::max() /
                static_cast<size_t>(out->ctx->N))
            return set_error("coefficient conversion shape overflow");
        *ring_dimension = static_cast<size_t>(out->ctx->N);
        *coefficient_count = out->rows * out->cols * *ring_dimension;
        return 0;
    }

    bool context_owns_compute_stream(
        const GpuContext *context, int physical_device, cudaStream_t stream)
    {
        if (!context || !context->execution || physical_device < 0 || !stream) return false;
        const auto found = std::find(
            context->execution->gpu_ids.begin(), context->execution->gpu_ids.end(), physical_device);
        if (found == context->execution->gpu_ids.end()) return false;
        const size_t partition = static_cast<size_t>(found - context->execution->gpu_ids.begin());
        if (partition >= context->execution->compute_streams_by_partition.size()) return false;
        const auto &streams = context->execution->compute_streams_by_partition[partition];
        return std::find(streams.begin(), streams.end(), stream) != streams.end();
    }
}

extern "C" int gpu_matrix_modulus_conversion_prepare(
    const GpuMatrix *out,
    const GpuMatrix *source,
    int round_scale,
    const uint64_t *division_inverses,
    size_t inverse_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!out_plan) return set_error("missing modulus conversion plan output");
    *out_plan = nullptr;
    ModulusConversionMetadata metadata{};
    int status = build_modulus_conversion_metadata(
        &metadata, out, source, round_scale, division_inverses, inverse_count);
    if (status != 0) return status;
    size_t coefficient_count = 0;
    size_t ring_dimension = 0;
    status = modulus_conversion_shape(out, &coefficient_count, &ring_dimension);
    if (status != 0) return status;
    const size_t blocks = coefficient_count / 128 + (coefficient_count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("coefficient conversion exceeds CUDA grid capacity");
    const auto &output = out->shared_limb_buffers[0];
    cudaStream_t stream = nullptr;
    status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < static_cast<size_t>(source->level) + 1; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported source CRT limb placement");
    }
    for (size_t limb = 0; limb < static_cast<size_t>(out->level) + 1; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported destination CRT limb placement");
    }
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate modulus conversion plan");
    plan->context = out->ctx;
    plan->device = output.device;
    plan->stream = stream;
    plan->metadata_bytes = sizeof(metadata);
    plan->coefficient_count = coefficient_count;
    plan->ring_dimension = ring_dimension;
    plan->round_scale = round_scale != 0;
    plan->source_count = static_cast<size_t>(source->level) + 1;
    plan->target_count = static_cast<size_t>(out->level) + 1;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, stream);
    if (error != cudaSuccess)
    {
        delete plan;
        return set_error(error);
    }
    error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        plan->device_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, plan->metadata_bytes);
    error = cudaMemcpyAsync(
        plan->device_metadata, plan->pinned_metadata, plan->metadata_bytes,
        cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        cudaFreeHost(plan->pinned_metadata);
        plan->device_metadata = nullptr;
        plan->pinned_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_matrix_modulus_conversion_submit(
    GpuModulusConversionPlan *plan,
    GpuMatrix *out,
    const GpuMatrix *source,
    uint32_t source_binding_index,
    uint32_t output_binding_index)
{
    if (!plan || !out || !source || !out->ctx || !source->ctx ||
        !plan->device_metadata || !plan->context)
        return set_error("invalid modulus conversion plan submission");
    if (out->ctx->execution != plan->context->execution ||
        source->ctx->execution != plan->context->execution)
        return set_error("modulus conversion plan execution mismatch");
    if (source->level < 0 || out->level < 0 ||
        static_cast<size_t>(source->level) + 1 != plan->source_count)
        return set_error("modulus conversion plan source level mismatch");
    if (static_cast<size_t>(out->level) + 1 != plan->target_count)
        return set_error("modulus conversion plan target level mismatch");
    if (out->rows * out->cols * static_cast<size_t>(out->ctx->N) != plan->coefficient_count)
        return set_error("modulus conversion plan shape mismatch");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != plan->device || output.device != plan->device ||
        !input.device_descriptors || !output.device_descriptors)
        return set_error("modulus conversion plan device mismatch");
    cudaError_t error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[limb], plan->device,
            stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < plan->target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status != 0) return status;
    }
    // Metadata preparation is independent of the coefficient domain. For
    // rounded conversion, normalize a private source copy inside the graph
    // so concurrent consumers keep their evaluation-domain input untouched.
    std::unique_ptr<GpuMatrix, decltype(&gpu_matrix_destroy)> coefficients(nullptr, gpu_matrix_destroy);
    const auto *source_descriptors = input.device_descriptors;
    const bool restore_evaluation = plan->round_scale && source->format == GPU_POLY_FORMAT_EVAL;
    if (restore_evaluation)
    {
        GpuMatrix *temporary = nullptr;
        status = gpu_matrix_create(source->ctx, source->level, source->rows, source->cols,
            GPU_POLY_FORMAT_EVAL, &temporary);
        if (status != 0) return status;
        coefficients.reset(temporary);
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            status = matrix_wait_limb_stream(temporary, temporary->ctx->limb_gpu_ids[limb],
                plan->device, stream);
            if (status != 0) return status;
        }
        source_descriptors = temporary->shared_limb_buffers[0].device_descriptors;
        const size_t copy_count = plan->coefficient_count * plan->source_count;
        copy_crt_descriptors_kernel<<<static_cast<unsigned>((copy_count + 127) / 128), 128, 0, stream>>>(
            input.device_descriptors, source_descriptors, plan->coefficient_count,
            plan->ring_dimension, plan->source_count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            status = matrix_record_limb_write(temporary, temporary->ctx->limb_gpu_ids[limb], stream);
            if (status != 0) return status;
        }
        status = gpu_matrix_intt_all_on_stream(temporary, stream);
        if (status != 0) return status;
    }
    const size_t blocks = plan->coefficient_count / 128 + (plan->coefficient_count % 128 != 0);
    convert_modulus_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        source_descriptors, output.device_descriptors,
        static_cast<const ModulusConversionMetadata *>(plan->device_metadata),
        plan->coefficient_count, plan->ring_dimension, plan->round_scale);
    error = cudaGetLastError();
    if (error != cudaSuccess) status = set_error(error);
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        const int tracked = matrix_track_limb_consumer_readonly(
            source, source->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status == 0) status = tracked;
    }
    for (size_t limb = 0; limb < plan->target_count; ++limb)
    {
        const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
        if (status == 0) status = recorded;
    }
    if (coefficients)
    {
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            const int tracked = matrix_track_limb_consumer_readonly(coefficients.get(),
                coefficients->ctx->limb_gpu_ids[limb], plan->device, stream);
            if (status == 0) status = tracked;
        }
    }
    if (status != 0 || !restore_evaluation) return status;
    out->format = GPU_POLY_FORMAT_COEFF;
    return gpu_matrix_ntt_all_bound(out, output_binding_index);
}

extern "C" int gpu_matrix_modulus_conversion_plan_protect_compiled_submission(
    GpuModulusConversionPlan *plan,
    GpuContext *context,
    int physical_device,
    void *launch_stream_raw,
    void *completion_event_raw)
{
    if (!plan)
        return set_error("invalid modulus conversion plan protection arguments");
    const auto reject = [plan](const char *message) {
        plan->release_blocked = true;
        return set_error(message);
    };
    if (!context || !plan->context || !plan->stream || !launch_stream_raw || !completion_event_raw)
        return reject("invalid modulus conversion plan protection arguments");
    if (context->execution != plan->context->execution)
        return reject("modulus conversion plan protection execution mismatch");
    if (physical_device != plan->device)
        return reject("modulus conversion plan protection device mismatch");
    const auto launch_stream = reinterpret_cast<cudaStream_t>(launch_stream_raw);
    if (!context_owns_compute_stream(context, physical_device, launch_stream))
        return reject("modulus conversion plan protection stream mismatch");
    cudaError_t error = cudaSetDevice(physical_device);
    if (error == cudaSuccess)
    {
        error = cudaStreamWaitEvent(
            plan->stream, reinterpret_cast<cudaEvent_t>(completion_event_raw), 0);
    }
    if (error != cudaSuccess)
    {
        plan->release_blocked = true;
        return set_error(error);
    }
    return 0;
}

extern "C" void gpu_matrix_modulus_conversion_plan_destroy(GpuModulusConversionPlan *plan)
{
    if (!plan) return;
    if (plan->release_blocked)
    {
        // The completion dependency could not be established. Keep both
        // metadata allocations live rather than risking a graph use-after-free.
        delete plan;
        return;
    }
    if (plan->device_metadata)
    {
        cudaSetDevice(plan->device);
        cudaFreeAsync(plan->device_metadata, plan->stream);
    }
    if (plan->pinned_metadata)
    {
        void *pinned = plan->pinned_metadata;
        if (plan->context && gpu_defer_pinned_frees(plan->context, plan->device,
                plan->stream, &pinned, 1) != 0)
        {
            // The reclaimer retains ownership on failure. Do not free a
            // pinned pointer which may still be read by the queued copy.
        }
    }
    if (plan->raw_ready) cudaEventDestroy(plan->raw_ready);
    delete plan;
}

extern "C" int gpu_raw_modulus_conversion_prepare(
    GpuContext *ctx, int device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    int round_scale, GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !source_moduli || !target_moduli || !out_plan ||
        !source_count || !target_count || source_count > kCrtMaxLimbs ||
        target_count > source_count || source_count > ctx->moduli.size() ||
        (round_scale != 0 && round_scale != 1))
        return set_error("invalid raw modulus conversion plan arguments");
    *out_plan = nullptr;
    ModulusConversionMetadata metadata{};
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    bool retained[kCrtMaxLimbs]{};
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        if (source_moduli[limb] != ctx->moduli[limb] ||
            source_moduli[limb] <= 1 || !(source_moduli[limb] & 1))
            return set_error("raw modulus source basis/context mismatch");
        metadata.source_moduli[limb] = source_moduli[limb];
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = target_moduli[limb];
        size_t source_index = 0;
        while (source_index < source_count && source_moduli[source_index] != modulus)
            ++source_index;
        if (source_index == source_count || retained[source_index])
            return set_error("raw modulus target is not an ordered source subset");
        retained[source_index] = true;
        metadata.retained[limb] = source_index;
        metadata.target_moduli[limb] = modulus;
    }
    for (size_t limb = 0; limb < source_count; ++limb)
        if (!retained[limb]) metadata.discarded[metadata.discarded_count++] = limb;
    for (size_t limb = 0; limb < metadata.discarded_count; ++limb)
    {
        const uint64_t modulus = source_moduli[metadata.discarded[limb]];
        for (size_t previous = 0; previous < limb; ++previous)
        {
            uint64_t inverse = 0;
            if (!mod_inverse_u64(source_moduli[metadata.discarded[previous]] % modulus,
                modulus, inverse))
                return set_error("raw modulus discarded basis is not invertible");
            metadata.garner[previous * kCrtMaxLimbs + limb] = inverse;
        }
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = target_moduli[limb];
        uint64_t divisor = 1;
        for (size_t index = 0; index < metadata.discarded_count; ++index)
            divisor = static_cast<uint64_t>((static_cast<unsigned __int128>(divisor) *
                (source_moduli[metadata.discarded[index]] % modulus)) % modulus);
        metadata.divisor_residues[limb] = divisor;
        uint64_t inverse = 1;
        if (round_scale && !mod_inverse_u64(divisor, modulus, inverse))
            return set_error("raw modulus divisor is not invertible");
        metadata.division_inverses[limb] = inverse;
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw modulus plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->metadata_bytes = sizeof(metadata);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->round_scale = round_scale != 0;
    plan->source_count = source_count;
    plan->target_count = target_count;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        if (plan->device_metadata) cudaFreeAsync(plan->device_metadata, plan->stream);
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, sizeof(metadata));
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        sizeof(metadata), cudaMemcpyHostToDevice, plan->stream);
    // The compiled graph can launch on another compute stream. Complete this
    // one-time metadata upload before the plan becomes available to emitters.
    if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_centered_rebase_prepare(
    GpuContext *ctx, int device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !source_moduli || !target_moduli || !out_plan ||
        !source_count || !target_count || source_count > kCrtMaxLimbs ||
        target_count > kCrtMaxLimbs || source_count > ctx->moduli.size())
        return set_error("invalid raw centered rebase plan arguments");
    *out_plan = nullptr;
    CenteredRebaseMetadata metadata{};
    metadata.source_count = source_count;
    metadata.output.limb_count = target_count;
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const uint64_t modulus = source_moduli[limb];
        if (modulus != ctx->moduli[limb] || modulus <= 1 || !(modulus & 1))
            return set_error("raw centered rebase source basis mismatch");
        metadata.source_moduli[limb] = modulus;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = target_moduli[limb];
        if (modulus <= 1 || !(modulus & 1))
            return set_error("invalid raw centered rebase target modulus");
        metadata.output.moduli[limb] = modulus;
        metadata.target_source[limb] = -1;
        for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
            if (modulus == source_moduli[source_limb])
                metadata.target_source[limb] = static_cast<int>(source_limb);
    }
    metadata.prefix_inverses[0] = 1;
    for (size_t limb = 1; limb < source_count; ++limb)
    {
        const uint64_t modulus = source_moduli[limb];
        uint64_t prefix = 1;
        for (size_t previous = 0; previous < limb; ++previous)
            prefix = static_cast<uint64_t>((static_cast<unsigned __int128>(prefix) *
                (source_moduli[previous] % modulus)) % modulus);
        if (!mod_inverse_u64(prefix, modulus, metadata.prefix_inverses[limb]))
            return set_error("raw centered rebase basis is not invertible");
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw centered rebase plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->metadata_bytes = sizeof(metadata);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->target_count = target_count;
    plan->centered_rebase = true;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        if (plan->device_metadata) cudaFreeAsync(plan->device_metadata, plan->stream);
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, sizeof(metadata));
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        sizeof(metadata), cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_centered_round_divide_prepare(
    GpuContext *ctx, int device, void *stream_raw,
    const uint64_t *moduli, size_t modulus_count,
    const uint64_t *divisor_words, size_t divisor_word_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !moduli || !divisor_words || !out_plan ||
        !modulus_count || modulus_count > kCrtMaxLimbs ||
        modulus_count > ctx->moduli.size() || !divisor_word_count)
        return set_error("invalid raw centered round divide plan arguments");
    *out_plan = nullptr;
    while (divisor_word_count && divisor_words[divisor_word_count - 1] == 0)
        --divisor_word_count;
    if (!divisor_word_count)
        return set_error("raw centered round divisor must be positive");
    const bool always_zero = divisor_word_count > kCrtMaxWords;
    CenteredRoundDivideMetadata metadata{};
    metadata.limb_count = modulus_count;
    metadata.divisor_word_count = always_zero ? 1 : static_cast<int>(divisor_word_count);
    if (always_zero) metadata.divisor_words[0] = 1;
    else std::copy_n(divisor_words, divisor_word_count, metadata.divisor_words);
    for (size_t limb = 0; limb < modulus_count; ++limb)
    {
        const uint64_t modulus = moduli[limb];
        if (modulus != ctx->moduli[limb] || modulus <= 1 || !(modulus & 1))
            return set_error("raw centered round divide CRT basis mismatch");
        metadata.moduli[limb] = modulus;
        metadata.prefix_inverses[limb] = 1;
        if (limb == 0) continue;
        uint64_t prefix = 1;
        for (size_t previous = 0; previous < limb; ++previous)
            prefix = static_cast<uint64_t>((static_cast<unsigned __int128>(prefix) *
                (moduli[previous] % modulus)) % modulus);
        if (!mod_inverse_u64(prefix, modulus, metadata.prefix_inverses[limb]))
            return set_error("raw centered round divide CRT basis is not invertible");
    }
    std::vector<uint64_t> modulus_words;
    if (!serde_compute_modulus_words_le(
            std::vector<uint64_t>(moduli, moduli + modulus_count), &modulus_words) ||
        modulus_words.empty() || modulus_words.size() > kCrtMaxWords)
        return set_error("unsupported raw centered round divide modulus size");
    metadata.word_count = static_cast<int>(modulus_words.size());
    std::copy(modulus_words.begin(), modulus_words.end(), metadata.modulus_words);
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw centered round divide plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->metadata_bytes = sizeof(metadata);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = modulus_count;
    plan->target_count = modulus_count;
    plan->centered_round_divide = true;
    plan->round_divide_zero = always_zero;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        if (plan->device_metadata) cudaFreeAsync(plan->device_metadata, plan->stream);
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, sizeof(metadata));
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        sizeof(metadata), cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

namespace
{
    struct RawCrtSources
    {
        MxxRawMatrixLimb limbs[kCrtMaxLimbs];
    };
    static_assert(sizeof(RawCrtSources) < 4096, "raw CRT source arguments exceed CUDA limit");

    __global__ void raw_modulus_conversion_kernel(
        RawCrtSources sources, MxxRawMatrixLimb destination,
        const ModulusConversionMetadata *metadata,
        size_t columns, size_t degree, size_t coefficient_count,
        size_t coefficient_offset, size_t target_limb, bool round_scale)
    {
        const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t index = coefficient_offset + local;
        if (index >= coefficient_count) return;
        const size_t poly = index / degree;
        const size_t coefficient = index - poly * degree;
        const uint64_t modulus = metadata->target_moduli[target_limb];
        uint64_t digits[kCrtMaxLimbs];
        bool negative = false;
        if (round_scale)
        {
            for (size_t digit_index = 0; digit_index < metadata->discarded_count; ++digit_index)
            {
                const size_t source_index = metadata->discarded[digit_index];
                const uint64_t source_modulus = metadata->source_moduli[source_index];
                uint64_t digit = raw_matrix_load(sources.limbs[source_index],
                    poly, coefficient, columns) % source_modulus;
                for (size_t previous = 0; previous < digit_index; ++previous)
                {
                    const uint64_t residue = digits[previous] % source_modulus;
                    const uint64_t difference = digit >= residue ? digit - residue :
                        source_modulus - (residue - digit);
                    digit = mul_mod_u64(difference,
                        metadata->garner[previous * kCrtMaxLimbs + digit_index],
                        source_modulus);
                }
                digits[digit_index] = digit;
            }
            for (size_t digit_index = metadata->discarded_count; digit_index-- > 0;)
            {
                const uint64_t half =
                    metadata->source_moduli[metadata->discarded[digit_index]] / 2;
                if (digits[digit_index] != half)
                {
                    negative = digits[digit_index] > half;
                    break;
                }
            }
        }
        uint64_t value = raw_matrix_load(
            sources.limbs[metadata->retained[target_limb]],
            poly, coefficient, columns) % modulus;
        if (round_scale)
        {
            uint64_t residual = 0;
            for (size_t digit_index = metadata->discarded_count; digit_index-- > 0;)
            {
                residual = add_mod_u64(mul_mod_u64(residual,
                    metadata->source_moduli[metadata->discarded[digit_index]] % modulus,
                    modulus), digits[digit_index] % modulus, modulus);
            }
            if (negative)
            {
                const uint64_t divisor = metadata->divisor_residues[target_limb];
                residual = residual >= divisor ? residual - divisor :
                    modulus - (divisor - residual);
            }
            value = value >= residual ? value - residual : modulus - (residual - value);
            value = mul_mod_u64(value,
                metadata->division_inverses[target_limb], modulus);
        }
        raw_matrix_store(destination, poly, coefficient, columns, value);
    }

    __global__ void raw_centered_rebase_kernel(
        RawCrtSources sources, MxxRawMatrixLimb destination,
        const CenteredRebaseMetadata *metadata,
        size_t columns, size_t degree, size_t coefficient_count,
        size_t coefficient_offset, size_t target_limb)
    {
        const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t index = coefficient_offset + local;
        if (index >= coefficient_count) return;
        const size_t poly = index / degree;
        const size_t coefficient = index - poly * degree;
        uint64_t digits[kCrtMaxLimbs];
        for (size_t current = 0; current < metadata->source_count; ++current)
        {
            const uint64_t modulus = metadata->source_moduli[current];
            const uint64_t residue = raw_matrix_load(
                sources.limbs[current], poly, coefficient, columns) % modulus;
            uint64_t prefix = 0;
            uint64_t weight = 1;
            for (size_t previous = 0; previous < current; ++previous)
            {
                prefix = add_mod_u64(prefix,
                    mul_mod_u64(digits[previous] % modulus, weight, modulus), modulus);
                weight = mul_mod_u64(weight,
                    metadata->source_moduli[previous] % modulus, modulus);
            }
            const uint64_t difference = residue >= prefix ? residue - prefix :
                modulus - (prefix - residue);
            digits[current] = mul_mod_u64(difference,
                metadata->prefix_inverses[current], modulus);
        }
        bool negative = false;
        for (size_t current = metadata->source_count; current-- > 0;)
        {
            const uint64_t half = (metadata->source_moduli[current] - 1) / 2;
            if (digits[current] != half)
            {
                negative = digits[current] > half;
                break;
            }
        }
        const uint64_t modulus = metadata->output.moduli[target_limb];
        uint64_t value = 0;
        const int source_index = metadata->target_source[target_limb];
        if (source_index >= 0)
            value = raw_matrix_load(sources.limbs[source_index],
                poly, coefficient, columns) % modulus;
        else
        {
            for (size_t current = metadata->source_count; current-- > 0;)
                value = add_mod_u64(mul_mod_u64(value,
                    metadata->source_moduli[current] % modulus, modulus),
                    digits[current] % modulus, modulus);
            if (negative)
            {
                uint64_t source_product = 1;
                for (size_t current = 0; current < metadata->source_count; ++current)
                    source_product = mul_mod_u64(source_product,
                        metadata->source_moduli[current] % modulus, modulus);
                value = value >= source_product ? value - source_product :
                    modulus - (source_product - value);
            }
        }
        raw_matrix_store(destination, poly, coefficient, columns, value);
    }

    __global__ void raw_centered_round_divide_kernel(
        RawCrtSources sources, RawCrtSources destinations,
        const CenteredRoundDivideMetadata *metadata,
        const uint64_t *runtime_divisor, int runtime_encoding,
        uint32_t *runtime_status,
        size_t columns, size_t degree, size_t coefficient_count,
        size_t coefficient_offset, bool always_zero)
    {
        const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t index = coefficient_offset + local;
        if (index >= coefficient_count) return;
        const size_t poly = index / degree;
        const size_t coefficient = index - poly * degree;
        uint64_t dynamic_divisor[kCrtMaxWords]{};
        int dynamic_word_count = 0;
        if (runtime_divisor)
        {
            bool valid = runtime_status != nullptr;
            if (runtime_encoding == 0)
            {
                const int64_t signed_value = static_cast<int64_t>(runtime_divisor[0]);
                valid = valid && signed_value > 0;
                if (valid) dynamic_divisor[0] = static_cast<uint64_t>(signed_value);
                dynamic_word_count = 1;
            }
            else if (runtime_encoding == 1)
            {
                valid = valid && runtime_divisor[0] != 0;
                if (valid) dynamic_divisor[0] = runtime_divisor[0];
                dynamic_word_count = 1;
            }
            else if (runtime_encoding > 2)
            {
                valid = valid && runtime_divisor[0] == 0;
                dynamic_word_count = runtime_encoding - 2;
                while (dynamic_word_count > 0 && runtime_divisor[dynamic_word_count] == 0)
                    --dynamic_word_count;
                valid = valid && dynamic_word_count > 0;
                if (dynamic_word_count > kCrtMaxWords)
                    always_zero = true;
                else if (valid)
                    for (int word = 0; word < dynamic_word_count; ++word)
                        dynamic_divisor[word] = runtime_divisor[word + 1];
            }
            else valid = false;
            if (!valid)
            {
                if (runtime_status) atomicCAS(runtime_status, 0U, 2U);
                return;
            }
        }
        if (always_zero)
        {
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
                raw_matrix_store(destinations.limbs[limb], poly, coefficient, columns, 0);
            return;
        }
        uint64_t digits[kCrtMaxLimbs]{};
        for (size_t current = 0; current < metadata->limb_count; ++current)
        {
            const uint64_t modulus = metadata->moduli[current];
            const uint64_t residue = raw_matrix_load(
                sources.limbs[current], poly, coefficient, columns) % modulus;
            uint64_t prefix = 0;
            uint64_t weight = 1;
            for (size_t previous = 0; previous < current; ++previous)
            {
                prefix = add_mod_u64(prefix,
                    mul_mod_u64(digits[previous] % modulus, weight, modulus), modulus);
                weight = mul_mod_u64(weight,
                    metadata->moduli[previous] % modulus, modulus);
            }
            const uint64_t difference = residue >= prefix ? residue - prefix :
                modulus - (prefix - residue);
            digits[current] = mul_mod_u64(difference,
                metadata->prefix_inverses[current], modulus);
        }
        bool negative = false;
        for (size_t current = metadata->limb_count; current-- > 0;)
        {
            const uint64_t half = (metadata->moduli[current] - 1) / 2;
            if (digits[current] != half)
            {
                negative = digits[current] > half;
                break;
            }
        }
        uint64_t value[kCrtMaxWords + 1]{};
        for (size_t limb = metadata->limb_count; limb-- > 0;)
        {
            uint64_t carry = digits[limb];
            for (int word = 0; word < metadata->word_count; ++word)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(value[word]) * metadata->moduli[limb] + carry;
                value[word] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
        }
        if (negative)
        {
            uint64_t magnitude[kCrtMaxWords + 1]{};
            for (int word = 0; word <= metadata->word_count; ++word)
                magnitude[word] = metadata->modulus_words[word];
            subtract_words(magnitude, value, metadata->word_count + 1);
            for (int word = 0; word <= metadata->word_count; ++word)
                value[word] = magnitude[word];
        }
        uint64_t quotient[kCrtMaxWords]{};
        divide_centered_words(value, *metadata,
            runtime_divisor ? dynamic_divisor : metadata->divisor_words,
            runtime_divisor ? dynamic_word_count : metadata->divisor_word_count,
            negative, quotient);
        for (size_t limb = 0; limb < metadata->limb_count; ++limb)
        {
            const uint64_t modulus = metadata->moduli[limb];
            uint64_t residue = 0;
            const uint64_t radix = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(1) << 64) % modulus);
            for (int word = metadata->word_count; word-- > 0;)
                residue = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(residue) * radix + quotient[word]) % modulus);
            if (negative && residue) residue = modulus - residue;
            raw_matrix_store(destinations.limbs[limb], poly, coefficient, columns, residue);
        }
    }
}

static int raw_modulus_conversion_emit_impl(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    const uint64_t *runtime_divisor, int runtime_encoding,
    uint32_t *runtime_status, uint32_t divisor_binding, uint32_t status_binding)
{
    if ((runtime_divisor != nullptr) != (runtime_status != nullptr) ||
        (runtime_divisor && (!plan || !plan->centered_round_divide ||
            runtime_encoding < 0 || runtime_encoding == 2)))
        return set_error("invalid dynamic centered round divisor view");
    if (!plan || !plan->device_metadata || plan->context != ctx ||
        validate_raw_view(ctx, source, stream_raw) != 0 ||
        !destination || !destination->limbs || !destination->limb_count ||
        source->physical_device != plan->device ||
        destination->physical_device != plan->device ||
        source->degree != destination->degree ||
        source->row_origin != destination->row_origin ||
        source->column_origin != destination->column_origin ||
        source->rows != destination->rows || source->columns != destination->columns ||
        source->limb_count != plan->source_count ||
        destination->limb_count != plan->target_count ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw modulus conversion emission");
    const auto *metadata = static_cast<const ModulusConversionMetadata *>(plan->pinned_metadata);
    if (!metadata) return set_error("raw modulus conversion metadata is absent");
    const auto *rebase = static_cast<const CenteredRebaseMetadata *>(plan->pinned_metadata);
    const auto *round_divide =
        static_cast<const CenteredRoundDivideMetadata *>(plan->pinned_metadata);
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        if (source->limbs[limb].modulus != (plan->centered_round_divide ?
                round_divide->moduli[limb] : plan->centered_rebase ?
                rebase->source_moduli[limb] : metadata->source_moduli[limb]))
            return set_error("raw modulus conversion source basis changed");
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const auto &target = destination->limbs[limb];
        const uint64_t modulus = plan->centered_round_divide ?
            round_divide->moduli[limb] : plan->centered_rebase ?
            rebase->output.moduli[limb] : metadata->target_moduli[limb];
        const unsigned __int128 column_span =
            static_cast<unsigned __int128>(destination->degree - 1) *
                target.coefficient_stride_bytes + target.word_bytes;
        const unsigned __int128 row_span =
            static_cast<unsigned __int128>(destination->columns - 1) *
                target.column_stride_bytes + column_span;
        if (!target.address || target.modulus != modulus ||
            (target.word_bytes != 4 && target.word_bytes != 8) ||
            (target.word_bytes == 4 && modulus > UINT32_MAX) ||
            target.coefficient_stride_bytes < target.word_bytes ||
            column_span > UINT64_MAX || row_span > UINT64_MAX ||
            target.column_stride_bytes < column_span ||
            target.row_stride_bytes < row_span)
            return set_error("raw modulus conversion target view changed");
    }
    if (cudaSetDevice(plan->device) != cudaSuccess)
        return set_error(cudaGetLastError());
    RawCrtSources sources{};
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        sources.limbs[limb] = source->limbs[limb];
    if (source->rows > SIZE_MAX / source->columns ||
        source->rows * source->columns > SIZE_MAX / source->degree)
        return set_error("raw modulus conversion coefficient count overflow");
    const size_t count = source->rows * source->columns * source->degree;
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    if (plan->centered_round_divide)
    {
        RawCrtSources destinations{};
        std::vector<MxxGraphPatch> patches;
        patches.reserve(source->limb_count + destination->limb_count);
        for (size_t limb = 0; limb < source->limb_count; ++limb)
        {
            destinations.limbs[limb] = destination->limbs[limb];
            patches.push_back(MxxGraphPatch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(RawCrtSources, limbs) +
                    limb * sizeof(MxxRawMatrixLimb)), sizeof(void *),
                static_cast<uint32_t>(source_binding_base + limb), 0});
            patches.push_back(MxxGraphPatch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1,
                static_cast<uint32_t>(offsetof(RawCrtSources, limbs) +
                    limb * sizeof(MxxRawMatrixLimb)), sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0});
        }
        if (runtime_divisor)
        {
            patches.push_back(MxxGraphPatch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 3, 0, sizeof(void *),
                divisor_binding, 0});
            patches.push_back(MxxGraphPatch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 5, 0, sizeof(void *),
                status_binding, 0});
        }
        for (size_t offset = 0; offset < count; offset += 65535 * 128)
        {
            const size_t chunk = std::min<size_t>(65535 * 128, count - offset);
            const dim3 grid(static_cast<uint32_t>((chunk + 127) / 128));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_centered_round_divide_kernel, grid, dim3(128), 0,
                patches.data(), patches.size(), sources, destinations,
                static_cast<const CenteredRoundDivideMetadata *>(plan->device_metadata),
                runtime_divisor, runtime_encoding, runtime_status,
                source->columns, static_cast<size_t>(source->degree), count, offset,
                plan->round_divide_zero);
            if (status != 0) return status;
        }
        return 0;
    }
    for (size_t target_limb = 0; target_limb < destination->limb_count; ++target_limb)
    {
        std::vector<MxxGraphPatch> patches;
        patches.reserve(source->limb_count + 1);
        for (size_t source_limb = 0; source_limb < source->limb_count; ++source_limb)
            patches.push_back(MxxGraphPatch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(RawCrtSources, limbs) +
                    source_limb * sizeof(MxxRawMatrixLimb)), sizeof(void *),
                static_cast<uint32_t>(source_binding_base + source_limb), 0});
        patches.push_back(MxxGraphPatch{nullptr,
            MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
            static_cast<uint32_t>(destination_binding_base + target_limb), 0});
        for (size_t offset = 0; offset < count; offset += 65535 * 128)
        {
            const size_t chunk = std::min<size_t>(65535 * 128, count - offset);
            const dim3 grid(static_cast<uint32_t>((chunk + 127) / 128));
            const int status = plan->centered_rebase ?
                mxx_gpu_launch_kernel(ctx, stream,
                    raw_centered_rebase_kernel, grid, dim3(128), 0,
                    patches.data(), patches.size(),
                    sources, destination->limbs[target_limb],
                    static_cast<const CenteredRebaseMetadata *>(plan->device_metadata),
                    source->columns, static_cast<size_t>(source->degree), count,
                    offset, target_limb) :
                mxx_gpu_launch_kernel(ctx, stream,
                    raw_modulus_conversion_kernel, grid, dim3(128), 0,
                    patches.data(), patches.size(),
                    sources, destination->limbs[target_limb],
                    static_cast<const ModulusConversionMetadata *>(plan->device_metadata),
                    source->columns, static_cast<size_t>(source->degree), count,
                    offset, target_limb, plan->round_scale);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_modulus_conversion_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    return raw_modulus_conversion_emit_impl(plan, ctx, stream_raw,
        source, destination, source_binding_base, destination_binding_base,
        nullptr, -1, nullptr, 0, 0);
}

extern "C" int gpu_raw_centered_round_divide_dynamic_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *divisor, int divisor_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t divisor_binding, uint32_t status_binding)
{
    return raw_modulus_conversion_emit_impl(plan, ctx, stream_raw,
        source, destination, source_binding_base, destination_binding_base,
        static_cast<const uint64_t *>(divisor), divisor_encoding,
        status, divisor_binding, status_binding);
}

extern "C" int gpu_matrix_centered_rebase_prepare(
    const GpuMatrix *out,
    const GpuMatrix *source,
    GpuModulusConversionPlan **out_plan)
{
    if (!out_plan) return set_error("missing centered rebase plan output");
    *out_plan = nullptr;
    CenteredRebaseMetadata metadata{};
    int status = build_centered_rebase_metadata(&metadata, out, source);
    if (status != 0) return status;
    size_t coefficient_count = 0;
    size_t ring_dimension = 0;
    status = modulus_conversion_shape(out, &coefficient_count, &ring_dimension);
    if (status != 0) return status;
    const auto &output = out->shared_limb_buffers[0];
    cudaStream_t stream = nullptr;
    status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < static_cast<size_t>(source->level) + 1; ++limb)
        if (source->ctx->limb_gpu_ids[limb].x != 0 || source->ctx->limb_gpu_ids[limb].y != limb)
            return set_error("unsupported source CRT limb placement");
    for (size_t limb = 0; limb < static_cast<size_t>(out->level) + 1; ++limb)
        if (out->ctx->limb_gpu_ids[limb].x != 0 || out->ctx->limb_gpu_ids[limb].y != limb)
            return set_error("unsupported destination CRT limb placement");
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate centered rebase plan");
    plan->context = out->ctx;
    plan->device = output.device;
    plan->stream = stream;
    plan->metadata_bytes = sizeof(metadata);
    plan->coefficient_count = coefficient_count;
    plan->ring_dimension = ring_dimension;
    plan->source_count = static_cast<size_t>(source->level) + 1;
    plan->target_count = static_cast<size_t>(out->level) + 1;
    plan->centered_rebase = true;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, stream);
    if (error != cudaSuccess) { delete plan; return set_error(error); }
    error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        plan->device_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, plan->metadata_bytes);
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        plan->metadata_bytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        cudaFreeHost(plan->pinned_metadata);
        plan->device_metadata = nullptr;
        plan->pinned_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_matrix_centered_rebase_submit(
    GpuModulusConversionPlan *plan,
    GpuMatrix *out,
    const GpuMatrix *source,
    uint32_t source_binding_index,
    uint32_t output_binding_index)
{
    if (!plan || !plan->centered_rebase || !out || !source || !out->ctx || !source->ctx ||
        !plan->device_metadata || !plan->context)
        return set_error("invalid centered rebase plan submission");
    if (out->ctx->execution != plan->context->execution ||
        source->ctx->execution != plan->context->execution || source->level < 0 || out->level < 0 ||
        static_cast<size_t>(source->level) + 1 != plan->source_count ||
        static_cast<size_t>(out->level) + 1 != plan->target_count ||
        out->rows * out->cols * static_cast<size_t>(out->ctx->N) != plan->coefficient_count)
        return set_error("centered rebase plan shape or execution mismatch");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != plan->device || output.device != plan->device ||
        !input.device_descriptors || !output.device_descriptors)
        return set_error("centered rebase plan device mismatch");
    cudaError_t error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[limb], plan->device,
            stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < plan->target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status != 0) return status;
    }
    // Normalize a private source copy so other consumers keep their input.
    std::unique_ptr<GpuMatrix, decltype(&gpu_matrix_destroy)> coefficients(nullptr, gpu_matrix_destroy);
    const auto *source_descriptors = input.device_descriptors;
    if (source->format == GPU_POLY_FORMAT_EVAL)
    {
        GpuMatrix *temporary = nullptr;
        status = gpu_matrix_create(source->ctx, source->level, source->rows, source->cols,
            GPU_POLY_FORMAT_EVAL, &temporary);
        if (status != 0) return status;
        coefficients.reset(temporary);
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            status = matrix_wait_limb_stream(temporary, temporary->ctx->limb_gpu_ids[limb],
                plan->device, stream);
            if (status != 0) return status;
        }
        source_descriptors = temporary->shared_limb_buffers[0].device_descriptors;
        const size_t copy_count = plan->coefficient_count * plan->source_count;
        copy_crt_descriptors_kernel<<<static_cast<unsigned>((copy_count + 127) / 128), 128, 0, stream>>>(
            input.device_descriptors, source_descriptors, plan->coefficient_count,
            plan->ring_dimension, plan->source_count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            status = matrix_record_limb_write(temporary, temporary->ctx->limb_gpu_ids[limb], stream);
            if (status != 0) return status;
        }
        status = gpu_matrix_intt_all_on_stream(temporary, stream);
        if (status != 0) return status;
    }
    const size_t count = plan->coefficient_count * plan->target_count;
    const size_t blocks = count / 128 + (count % 128 != 0);
    centered_rebase_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        source_descriptors, output.device_descriptors,
        static_cast<const CenteredRebaseMetadata *>(plan->device_metadata),
        plan->coefficient_count, plan->ring_dimension);
    error = cudaGetLastError();
    if (error != cudaSuccess) status = set_error(error);
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        const int tracked = matrix_track_limb_consumer_readonly(
            source, source->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status == 0) status = tracked;
    }
    for (size_t limb = 0; limb < plan->target_count; ++limb)
    {
        const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
        if (status == 0) status = recorded;
    }
    if (coefficients)
    {
        for (size_t limb = 0; limb < plan->source_count; ++limb)
        {
            const int tracked = matrix_track_limb_consumer_readonly(coefficients.get(),
                coefficients->ctx->limb_gpu_ids[limb], plan->device, stream);
            if (status == 0) status = tracked;
        }
    }
    if (status != 0) return status;
    out->format = GPU_POLY_FORMAT_COEFF;
    return gpu_matrix_ntt_all_bound(out, output_binding_index);
}

extern "C" int gpu_matrix_centered_round_divide_prepare(
    const GpuMatrix *out,
    const GpuMatrix *source,
    const uint64_t *divisor_words,
    size_t divisor_word_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!out_plan) return set_error("missing centered round divide plan output");
    *out_plan = nullptr;
    CenteredRoundDivideMetadata metadata{};
    int status = build_centered_round_divide_metadata(
        &metadata, out, source, divisor_words, divisor_word_count);
    if (status != 0) return status;
    size_t coefficient_count = 0;
    size_t ring_dimension = 0;
    status = modulus_conversion_shape(out, &coefficient_count, &ring_dimension);
    if (status != 0) return status;
    const auto &output = out->shared_limb_buffers[0];
    cudaStream_t stream = nullptr;
    status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate centered round divide plan");
    plan->context = out->ctx;
    plan->device = output.device;
    plan->stream = stream;
    plan->metadata_bytes = sizeof(metadata);
    plan->coefficient_count = coefficient_count;
    plan->ring_dimension = ring_dimension;
    plan->source_count = metadata.limb_count;
    plan->target_count = metadata.limb_count;
    plan->centered_round_divide = true;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, stream);
    if (error != cudaSuccess) { delete plan; return set_error(error); }
    error = cudaHostAlloc(&plan->pinned_metadata, plan->metadata_bytes, cudaHostAllocPortable);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        plan->device_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    std::memcpy(plan->pinned_metadata, &metadata, plan->metadata_bytes);
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        plan->metadata_bytes, cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
        cudaFreeAsync(plan->device_metadata, stream);
        cudaFreeHost(plan->pinned_metadata);
        plan->device_metadata = nullptr;
        plan->pinned_metadata = nullptr;
        delete plan;
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_matrix_centered_round_divide_submit(
    GpuModulusConversionPlan *plan,
    GpuMatrix *out,
    const GpuMatrix *source,
    uint32_t source_binding_index,
    uint32_t output_binding_index)
{
    if (!plan || !plan->centered_round_divide || !out || !source || !out->ctx || !source->ctx ||
        !plan->device_metadata || !plan->context ||
        out->format != GPU_POLY_FORMAT_COEFF)
        return set_error("invalid centered round divide plan submission");
    if (out->ctx->execution != plan->context->execution ||
        source->ctx->execution != plan->context->execution || source->level < 0 || out->level < 0 ||
        static_cast<size_t>(source->level) + 1 != plan->source_count ||
        static_cast<size_t>(out->level) + 1 != plan->target_count ||
        out->rows * out->cols * static_cast<size_t>(out->ctx->N) != plan->coefficient_count)
        return set_error("centered round divide plan shape or execution mismatch");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != plan->device || output.device != plan->device ||
        !input.device_descriptors || !output.device_descriptors)
        return set_error("centered round divide plan device mismatch");
    cudaError_t error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[limb], plan->device,
            stream, false, true);
        if (status != 0) return status;
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status != 0) return status;
    }
    std::unique_ptr<GpuMatrix, decltype(&gpu_matrix_destroy)> coefficients(nullptr, gpu_matrix_destroy);
    const auto *source_descriptors = input.device_descriptors;
    if (source->format == GPU_POLY_FORMAT_EVAL)
    {
        GpuMatrix *temporary = nullptr;
        status = gpu_matrix_create(source->ctx, source->level, source->rows, source->cols,
            GPU_POLY_FORMAT_EVAL, &temporary);
        if (status != 0) return status;
        coefficients.reset(temporary);
        status = matrix_wait_all_limb_streams(temporary, plan->device, stream);
        if (status != 0) return status;
        source_descriptors = temporary->shared_limb_buffers[0].device_descriptors;
        const size_t copy_count = plan->coefficient_count * plan->source_count;
        copy_crt_descriptors_kernel<<<static_cast<unsigned>((copy_count + 127) / 128), 128, 0, stream>>>(
            input.device_descriptors, source_descriptors, plan->coefficient_count,
            plan->ring_dimension, plan->source_count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        status = matrix_record_all_limb_writes(temporary, stream);
        if (status != 0) return status;
        status = gpu_matrix_intt_all_on_stream(temporary, stream);
        if (status != 0) return status;
    }
    const size_t blocks = plan->coefficient_count / 128 + (plan->coefficient_count % 128 != 0);
    centered_round_divide_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        source_descriptors, output.device_descriptors,
        static_cast<const CenteredRoundDivideMetadata *>(plan->device_metadata),
        plan->coefficient_count, plan->ring_dimension);
    error = cudaGetLastError();
    if (error != cudaSuccess) status = set_error(error);
    for (size_t limb = 0; limb < plan->source_count; ++limb)
    {
        const int tracked = matrix_track_limb_consumer_readonly(
            source, source->ctx->limb_gpu_ids[limb], plan->device, stream);
        if (status == 0) status = tracked;
        const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
        if (status == 0) status = recorded;
    }
    if (status != 0) return status;
    return gpu_matrix_ntt_all_bound(out, output_binding_index);
}

extern "C" int gpu_matrix_centered_rebase(GpuMatrix *out, const GpuMatrix *source)
{
    if (!out || !source || !out->ctx || !source->ctx ||
        out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
        source->level < 0 || out->level < 0 ||
        source->rows != out->rows || source->cols != out->cols ||
        source->format != GPU_POLY_FORMAT_COEFF || out->format != GPU_POLY_FORMAT_COEFF ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("centered rebase requires colocated coefficient matrices with matching execution");
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (!source_count || source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs ||
        source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size())
        return set_error("invalid centered rebase destination basis");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("centered rebase requires colocated device descriptors");
    for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
    {
        const uint64_t p = source->ctx->moduli[source_limb];
        if (p <= 1 || !(p & 1)) return set_error("centered rebase requires odd source CRT moduli");
    }
    for (size_t target_limb = 0; target_limb < target_count; ++target_limb)
        if (out->ctx->moduli[target_limb] <= 1 || !(out->ctx->moduli[target_limb] & 1))
            return set_error("centered rebase requires odd destination CRT moduli");
    if (!out->rows || !out->cols || out->rows > std::numeric_limits<size_t>::max() / out->cols ||
        out->rows * out->cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(out->ctx->N))
        return set_error("centered rebase shape overflow");
    const size_t coefficient_count = out->rows * out->cols * static_cast<size_t>(out->ctx->N);
    if (coefficient_count > std::numeric_limits<size_t>::max() / target_count)
        return set_error("centered rebase coefficient count overflow");
    const size_t count = coefficient_count * target_count;
    const size_t blocks = count / 128 + (count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("centered rebase exceeds CUDA grid capacity");
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    // CenteredRebase reads every source limb while reconstructing the exact
    // mixed-radix value.  Wait for each producer on the output device before
    // allocating metadata or launching the kernel; waiting only on limb zero
    // leaves later-limb writes racy when source work is asynchronous.
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[limb], output.device,
            stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], output.device, stream);
        if (status != 0) return status;
    }
    CenteredRebaseMetadata host_metadata{};
    host_metadata.source = input.device_descriptors;
    host_metadata.output.descriptors = output.device_descriptors;
    host_metadata.output.limb_count = target_count;
    std::copy_n(out->ctx->moduli.begin(), target_count, host_metadata.output.moduli);
    host_metadata.source_count = source_count;
    std::copy_n(source->ctx->moduli.begin(), source_count, host_metadata.source_moduli);
    host_metadata.prefix_inverses[0] = 1;
    for (size_t current = 1; current < source_count; ++current)
    {
        const uint64_t modulus = host_metadata.source_moduli[current];
        uint64_t product = 1;
        for (size_t previous = 0; previous < current; ++previous)
            product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                host_metadata.source_moduli[previous]) % modulus);
        uint64_t inverse = 0;
        if (!mod_inverse_u64(product, modulus, inverse))
            return set_error("centered rebase source CRT basis is not invertible");
        host_metadata.prefix_inverses[current] = inverse;
    }
    for (size_t target_limb = 0; target_limb < target_count; ++target_limb)
    {
        host_metadata.target_source[target_limb] = -1;
        for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
            if (out->ctx->moduli[target_limb] == source->ctx->moduli[source_limb])
                host_metadata.target_source[target_limb] = static_cast<int>(source_limb);
    }
    CenteredRebaseMetadata *device_metadata = nullptr;
    std::vector<void *> pinned;
    status = crt_alloc_and_copy_async(&device_metadata, std::vector<CenteredRebaseMetadata>{host_metadata}, stream, &pinned);
    if (status == 0)
    {
        centered_rebase_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            input.device_descriptors, output.device_descriptors, device_metadata,
            coefficient_count, static_cast<size_t>(out->ctx->N));
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
    }
    // The temporary INTT source can be dropped immediately after this call.
    for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
    {
        const int tracked = matrix_track_limb_consumer_readonly(source,
            source->ctx->limb_gpu_ids[source_limb], output.device, stream);
        if (status == 0) status = tracked;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
        if (status == 0) status = recorded;
    }
    if (device_metadata) cudaFreeAsync(device_metadata, stream);
    if (!pinned.empty())
    {
        const int deferred = gpu_defer_pinned_frees(out->ctx, output.device, stream, pinned.data(), pinned.size());
        if (status == 0) status = deferred;
    }
    return status;
}

extern "C" int gpu_matrix_block_mod_switch(
    GpuMatrix *out, const GpuMatrix *source,
    const uint64_t *plaintext_modulus_words, size_t plaintext_modulus_word_count)
{
    if (!out || !source || !out->ctx || !source->ctx || !plaintext_modulus_words ||
        !plaintext_modulus_word_count || out->ctx->execution != source->ctx->execution ||
        out->ctx->N != source->ctx->N || source->level < 0 || out->level < 0 ||
        source->rows != out->rows || source->cols != out->cols ||
        source->format != GPU_POLY_FORMAT_COEFF || out->format != GPU_POLY_FORMAT_COEFF ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("invalid BlockModSwitch layout");
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    // BlockModSwitch only needs t modulo each CRT prime.  `words_mod` below
    // folds the complete host-provided little-endian word slice into those
    // residues before the metadata is copied to the device, so this path must
    // not impose the fixed word-array bound used by CRT recomposition.
    if (!source_count || !target_count || source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs ||
        source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size())
        return set_error("invalid BlockModSwitch basis");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("BlockModSwitch requires colocated device descriptors");
    BlockModSwitchMetadata host_metadata{};
    host_metadata.source = input.device_descriptors;
    host_metadata.output.descriptors = output.device_descriptors;
    host_metadata.output.limb_count = target_count;
    host_metadata.source_count = source_count;
    host_metadata.dropped_count = 0;
    std::copy_n(out->ctx->moduli.begin(), target_count, host_metadata.output.moduli);
    std::copy_n(source->ctx->moduli.begin(), source_count, host_metadata.source_moduli);
    int target_source[kCrtMaxLimbs];
    bool retained[kCrtMaxLimbs]{};
    for (size_t target = 0; target < target_count; ++target)
    {
        target_source[target] = -1;
        for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
            if (out->ctx->moduli[target] == source->ctx->moduli[source_limb])
            {
                if (target_source[target] >= 0 || retained[source_limb])
                    return set_error("BlockModSwitch basis contains duplicate limbs");
                target_source[target] = static_cast<int>(source_limb);
                retained[source_limb] = true;
            }
        if (target_source[target] < 0)
            return set_error("BlockModSwitch destination basis is not a source subset");
        host_metadata.target_source[target] = target_source[target];
    }
    for (size_t source_limb = 0; source_limb < source_count; ++source_limb)
        if (!retained[source_limb])
            host_metadata.dropped_indices[host_metadata.dropped_count++] = source_limb;
    if (!host_metadata.dropped_count)
        return set_error("BlockModSwitch requires a strict destination subset");
    auto words_mod = [&](uint64_t modulus) {
        uint64_t value = 0;
        const uint64_t radix = static_cast<uint64_t>((static_cast<unsigned __int128>(1) << 64) % modulus);
        for (size_t word = plaintext_modulus_word_count; word-- > 0;)
            value = static_cast<uint64_t>((static_cast<unsigned __int128>(value) * radix +
                plaintext_modulus_words[word] % modulus) % modulus);
        return value;
    };
    for (size_t current = 0; current < host_metadata.dropped_count; ++current)
    {
        const uint64_t p = host_metadata.source_moduli[host_metadata.dropped_indices[current]];
        uint64_t t_inverse = 0;
        if (!mod_inverse_u64(words_mod(p), p, t_inverse))
            return set_error("BlockModSwitch t is not invertible modulo a dropped prime");
        host_metadata.dropped_t_inverses[current] = t_inverse;
        host_metadata.prefix_inverses[current] = 1;
        if (current)
        {
            uint64_t product = 1;
            for (size_t previous = 0; previous < current; ++previous)
                product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                    host_metadata.source_moduli[host_metadata.dropped_indices[previous]]) % p);
            if (!mod_inverse_u64(product, p, host_metadata.prefix_inverses[current]))
                return set_error("BlockModSwitch dropped basis is not invertible");
        }
    }
    for (size_t target = 0; target < target_count; ++target)
    {
        const uint64_t q = host_metadata.output.moduli[target];
        host_metadata.target_t[target] = words_mod(q);
        uint64_t product = 1;
        for (size_t current = 0; current < host_metadata.dropped_count; ++current)
            product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                host_metadata.source_moduli[host_metadata.dropped_indices[current]]) % q);
        if (!mod_inverse_u64(product, q, host_metadata.target_p_inverses[target]))
            return set_error("BlockModSwitch dropped product is not invertible in destination");
    }
    if (!out->rows || !out->cols || out->rows > std::numeric_limits<size_t>::max() / out->cols ||
        out->rows * out->cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(out->ctx->N))
        return set_error("BlockModSwitch shape overflow");
    const size_t coefficient_count = out->rows * out->cols * static_cast<size_t>(out->ctx->N);
    if (coefficient_count > std::numeric_limits<size_t>::max() / target_count)
        return set_error("BlockModSwitch coefficient count overflow");
    const size_t count = coefficient_count * target_count;
    const size_t blocks = count / 128 + (count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("BlockModSwitch exceeds CUDA grid capacity");
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[limb], output.device, stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], output.device, stream);
        if (status != 0) return status;
    }
    // Scalar CRT metadata fits in the kernel argument buffer.
    static_assert(sizeof(BlockModSwitchMetadata) + 2 * sizeof(size_t) <= 4096);
    block_mod_switch_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        host_metadata, coefficient_count, static_cast<size_t>(out->ctx->N));
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const int tracked = matrix_track_limb_consumer_readonly(source,
            source->ctx->limb_gpu_ids[limb], output.device, stream);
        if (status == 0) status = tracked;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
        if (status == 0) status = recorded;
    }
    return status;
}

extern "C" int gpu_matrix_crt_recompose(
    GpuMatrix *out, const GpuMatrix *const *levels, size_t level_count,
    const uint64_t *plaintext_moduli, const uint64_t *reconstruction_residues,
    size_t reconstruction_stride)
{
    if (!out || !out->ctx || !levels || !plaintext_moduli || !reconstruction_residues ||
        !level_count || out->rows != 1 || !out->cols || out->level < 0 ||
        out->shared_limb_buffers.size() != 1)
        return set_error("invalid CRT recomposition output");
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (target_count > kCrtMaxLimbs || target_count > out->ctx->moduli.size() ||
        reconstruction_stride != target_count)
        return set_error("invalid CRT recomposition output basis");
    const auto &target = out->shared_limb_buffers[0];
    if (!target.device_descriptors || target.limb_count < target_count)
        return set_error("missing CRT output descriptors");
    std::vector<CrtLevelMetadata> metadata(level_count);
    std::vector<CrtOutputMetadata> output_metadata(1);
    output_metadata[0].descriptors = target.device_descriptors;
    output_metadata[0].limb_count = target_count;
    std::copy_n(out->ctx->moduli.begin(), target_count, output_metadata[0].moduli);

    for (size_t level = 0; level < level_count; ++level)
    {
        const auto *source = levels[level];
        if (!source || !source->ctx || source->ctx->execution != out->ctx->execution ||
            source->ctx->N != out->ctx->N || source->rows != 1 || source->cols != out->cols ||
            source->level < 0 || source->format != GPU_POLY_FORMAT_COEFF ||
            source->shared_limb_buffers.size() != 1 || !plaintext_moduli[level])
            return set_error("invalid mixed-modulus CRT input");
        const size_t count = static_cast<size_t>(source->level) + 1;
        const auto &buffer = source->shared_limb_buffers[0];
        if (count > kCrtMaxLimbs || count > source->ctx->moduli.size() ||
            buffer.device != target.device || !buffer.device_descriptors || buffer.limb_count < count)
            return set_error("invalid mixed-modulus CRT basis or placement");
        auto &entry = metadata[level];
        entry.descriptors = buffer.device_descriptors;
        entry.limb_count = count;
        entry.plaintext_modulus = plaintext_moduli[level];
        std::copy_n(source->ctx->moduli.begin(), count, entry.moduli);
        std::copy_n(reconstruction_residues + level * target_count, target_count, entry.reconstruction);
        std::vector<uint64_t> words;
        const std::vector<uint64_t> active(source->ctx->moduli.begin(), source->ctx->moduli.begin() + count);
        if (!serde_compute_modulus_words_le(active, &words) || words.empty() || words.size() > kCrtMaxWords)
            return set_error("unsupported CRT source modulus size");
        entry.word_count = static_cast<int>(words.size());
        std::copy(words.begin(), words.end(), entry.modulus_words);
        for (size_t limb = 0; limb < count; ++limb)
        {
            uint64_t product = 1;
            for (size_t previous = 0; previous < limb; ++previous)
                product = static_cast<uint64_t>((static_cast<unsigned __int128>(product) *
                    entry.moduli[previous]) % entry.moduli[limb]);
            if (!mod_inverse_u64(product, entry.moduli[limb], entry.prefix_inverses[limb]))
                return set_error("CRT source moduli must be pairwise coprime");
        }
    }
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    auto error = cudaSetDevice(target.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], target.device, stream);
        if (status != 0) return status;
    }
    for (size_t level = 0; level < level_count; ++level)
        for (size_t limb = 0; limb < metadata[level].limb_count; ++limb)
        {
            status = matrix_wait_limb_stream(levels[level], levels[level]->ctx->limb_gpu_ids[limb],
                target.device, stream);
            if (status != 0) return status;
        }
    if (out->cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(out->ctx->N))
        return set_error("CRT coefficient shape overflow");
    const size_t count = out->cols * static_cast<size_t>(out->ctx->N);
    const size_t blocks = count / 128 + (count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("CRT recomposition exceeds CUDA grid capacity");

    // One bounded by-value launch per level supports arbitrary level counts.
    static_assert(sizeof(CrtLevelMetadata) + sizeof(CrtOutputMetadata) +
        sizeof(bool) + 2 * sizeof(size_t) <= 4096);
    for (size_t level = 0; status == 0 && level < level_count; ++level)
    {
        crt_recompose_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            metadata[level], output_metadata[0], level == 0, count, static_cast<size_t>(out->ctx->N));
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
        if (status != 0) break;
    }
    {
        for (size_t level = 0; level < level_count; ++level)
            for (size_t limb = 0; limb < metadata[level].limb_count; ++limb)
            {
                const int tracked = matrix_track_limb_consumer_readonly(levels[level],
                    levels[level]->ctx->limb_gpu_ids[limb], target.device, stream);
                if (status == 0) status = tracked;
            }
        for (size_t limb = 0; limb < target_count; ++limb)
        {
            const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
            if (status == 0) status = recorded;
        }
    }
    if (status == 0) out->format = GPU_POLY_FORMAT_COEFF;
    return status;
}

namespace
{
    struct RawRecomposeMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target;
        size_t source_count;
        size_t target_count;
        int source_words;
        int plaintext_words;
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t target_moduli[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        uint64_t source_product[kCrtMaxWords];
        uint64_t plaintext[kCrtMaxWords];
        uint64_t reconstruction[kCrtMaxLimbs];
    };

    __global__ void raw_recompose_descriptor_kernel(
        MxxRawMatrixLimb limb,
        GpuMatrix::SharedLimbBuffer::DeviceDescriptor *destination)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            destination->base = reinterpret_cast<uint8_t *>(limb.address);
            destination->stride = static_cast<size_t>(limb.column_stride_bytes);
            destination->width = static_cast<uint8_t>(limb.word_bytes);
        }
    }

    __device__ __forceinline__ void raw_recompose_divide(
        const uint64_t *numerator, const uint64_t *divisor,
        int divisor_words, uint64_t *quotient)
    {
        uint64_t remainder[kCrtMaxWords + 1]{};
        const int numerator_words = 2 * divisor_words + 1;
        for (int bit = numerator_words * 64 - 1; bit >= 0; --bit)
        {
            uint64_t carry = (numerator[bit / 64] >> (bit % 64)) & 1;
            for (int word = 0; word <= divisor_words; ++word)
            {
                const uint64_t next = remainder[word] >> 63;
                remainder[word] = (remainder[word] << 1) | carry;
                carry = next;
            }
            bool ge = remainder[divisor_words] != 0;
            if (!ge)
                ge = crt_compare_words(remainder, divisor, divisor_words) >= 0;
            if (ge)
            {
                uint64_t borrow = 0;
                for (int word = 0; word < divisor_words; ++word)
                {
                    const uint64_t subtrahend = divisor[word] + borrow;
                    const bool overflow = subtrahend < divisor[word];
                    const uint64_t old = remainder[word];
                    remainder[word] = old - subtrahend;
                    borrow = overflow || old < subtrahend;
                }
                remainder[divisor_words] -= borrow;
                if (bit < divisor_words * 64)
                    quotient[bit / 64] |= uint64_t(1) << (bit % 64);
            }
        }
    }

    __global__ void raw_recompose_level_kernel(
        const RawRecomposeMetadata *metadata, size_t coefficient_count,
        size_t degree, bool initialize)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / degree;
        const size_t coefficient = index % degree;
        uint64_t digits[kCrtMaxLimbs]{};
        crt_recover_centered_digits(metadata->source, nullptr, metadata->source_moduli,
            nullptr, metadata->prefix_inverses, metadata->source_count,
            poly, coefficient, digits);
        uint64_t value[kCrtMaxWords]{};
        for (size_t limb = metadata->source_count; limb-- > 0;)
        {
            uint64_t carry = digits[limb];
            for (int word = 0; word < metadata->source_words; ++word)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(value[word]) *
                    metadata->source_moduli[limb] + carry;
                value[word] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
        }
        uint64_t numerator[2 * kCrtMaxWords + 1]{};
        for (int left = 0; left < metadata->source_words; ++left)
        {
            uint64_t carry = 0;
            for (int right = 0; right < metadata->plaintext_words; ++right)
            {
                const size_t position = static_cast<size_t>(left + right);
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(value[left]) *
                    metadata->plaintext[right] + numerator[position] + carry;
                numerator[position] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
            size_t position = static_cast<size_t>(left + metadata->plaintext_words);
            while (carry)
            {
                const unsigned __int128 sum =
                    static_cast<unsigned __int128>(numerator[position]) + carry;
                numerator[position] = static_cast<uint64_t>(sum);
                carry = static_cast<uint64_t>(sum >> 64);
                ++position;
            }
        }
        uint64_t carry = 0;
        for (int word = 0; word < metadata->source_words; ++word)
        {
            const uint64_t current = metadata->source_product[word];
            const uint64_t next = word + 1 < metadata->source_words ?
                metadata->source_product[word + 1] : 0;
            const uint64_t half = (current >> 1) | ((next & 1) << 63);
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(numerator[word]) + half + carry;
            numerator[word] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
        for (int word = metadata->source_words; carry; ++word)
        {
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(numerator[word]) + carry;
            numerator[word] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
        uint64_t rounded[kCrtMaxWords]{};
        raw_recompose_divide(numerator, metadata->source_product,
            metadata->source_words, rounded);
        bool wraps_plaintext = true;
        for (int word = 0; word < metadata->source_words; ++word)
        {
            const uint64_t plain = word < metadata->plaintext_words ?
                metadata->plaintext[word] : 0;
            if (rounded[word] != plain)
            {
                wraps_plaintext = false;
                break;
            }
        }
        if (wraps_plaintext)
            for (int word = 0; word < metadata->source_words; ++word)
                rounded[word] = 0;
        for (size_t limb = 0; limb < metadata->target_count; ++limb)
        {
            const uint64_t modulus = metadata->target_moduli[limb];
            const uint64_t radix = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(1) << 64) % modulus);
            uint64_t residue = 0;
            for (int word = metadata->source_words; word-- > 0;)
                residue = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(residue) * radix +
                        rounded[word] % modulus) % modulus);
            const auto output = metadata->target[limb];
            const uint64_t previous = initialize ? 0 :
                matrix_load_limb_u64(output.base, poly, coefficient,
                    output.stride, output.width);
            const uint64_t scaled =
                mul_mod_u64(residue, metadata->reconstruction[limb], modulus);
            matrix_store_limb_u64(output.base, poly, coefficient,
                output.stride, output.width,
                add_mod_u64(previous, scaled, modulus));
        }
    }

    bool raw_recompose_view_matches(const MxxRawMatrixView *view,
        const uint64_t *moduli, size_t count, int device, size_t degree,
        bool one_row = true)
    {
        if (!view || !view->limbs || view->physical_device != device ||
            view->limb_count != count || view->degree != degree ||
            view->row_origin || view->column_origin || !view->rows ||
            (one_row && view->rows != 1) ||
            !view->columns || !view->degree ||
            view->rows > SIZE_MAX / view->columns ||
            view->rows * view->columns > SIZE_MAX / view->degree) return false;
        for (size_t index = 0; index < count; ++index)
        {
            const auto &limb = view->limbs[index];
            if (!limb.address || limb.crt_limb_index != index ||
                limb.modulus != moduli[index] ||
                (limb.word_bytes != 4 && limb.word_bytes != 8) ||
                (limb.word_bytes == 4 && limb.modulus > UINT32_MAX) ||
                limb.coefficient_stride_bytes != limb.word_bytes ||
                view->degree > UINT64_MAX / limb.word_bytes ||
                limb.column_stride_bytes < view->degree * limb.word_bytes ||
                view->columns > UINT64_MAX / limb.column_stride_bytes ||
                limb.row_stride_bytes != view->columns * limb.column_stride_bytes)
                return false;
        }
        return true;
    }
}

extern "C" int gpu_raw_crt_recompose_level_prepare(
    GpuContext *ctx, int32_t device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    const uint64_t *plaintext_words, size_t plaintext_count,
    const uint64_t *reconstruction_residues, size_t reconstruction_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !out_plan || !source_moduli || !target_moduli ||
        !plaintext_words || !reconstruction_residues ||
        !source_count || !target_count || source_count > kCrtMaxLimbs ||
        target_count > kCrtMaxLimbs || reconstruction_count != target_count)
        return set_error("invalid raw CRT recomposition plan arguments");
    *out_plan = nullptr;
    while (plaintext_count && plaintext_words[plaintext_count - 1] == 0)
        --plaintext_count;
    if (!plaintext_count || plaintext_count > kCrtMaxWords ||
        (plaintext_count == 1 && plaintext_words[0] <= 1))
        return set_error("invalid raw CRT recomposition plaintext modulus");
    RawRecomposeMetadata metadata{};
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    metadata.plaintext_words = static_cast<int>(plaintext_count);
    std::copy_n(source_moduli, source_count, metadata.source_moduli);
    std::copy_n(target_moduli, target_count, metadata.target_moduli);
    std::copy_n(plaintext_words, plaintext_count, metadata.plaintext);
    std::copy_n(reconstruction_residues, target_count, metadata.reconstruction);
    std::vector<uint64_t> product;
    if (!serde_compute_modulus_words_le(
            std::vector<uint64_t>(source_moduli, source_moduli + source_count),
            &product) || product.empty() || product.size() > kCrtMaxWords ||
        plaintext_count > product.size())
        return set_error("raw CRT recomposition source modulus exceeds supported width");
    metadata.source_words = static_cast<int>(product.size());
    std::copy(product.begin(), product.end(), metadata.source_product);
    for (size_t word = product.size(); word-- > 0;)
    {
        const uint64_t plain = word < plaintext_count ? plaintext_words[word] : 0;
        if (plain != product[word])
        {
            if (plain > product[word])
                return set_error("raw CRT recomposition plaintext exceeds source modulus");
            break;
        }
    }
    for (size_t index = 0; index < source_count; ++index)
    {
        if (source_moduli[index] <= 1 ||
            index >= ctx->moduli.size() ||
            ctx->moduli[index] != source_moduli[index])
            return set_error("raw CRT recomposition source basis mismatch");
        uint64_t prefix = 1;
        for (size_t previous = 0; previous < index; ++previous)
            prefix = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(prefix) *
                    (source_moduli[previous] % source_moduli[index])) %
                source_moduli[index]);
        if (!mod_inverse_u64(prefix, source_moduli[index],
                metadata.prefix_inverses[index]))
            return set_error("raw CRT recomposition source basis is not coprime");
    }
    for (size_t index = 0; index < target_count; ++index)
        if (target_moduli[index] <= 1 ||
            reconstruction_residues[index] >= target_moduli[index])
            return set_error("raw CRT recomposition target residue is invalid");
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw CRT recomposition plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->target_count = target_count;
    plan->raw_recompose = true;
    const size_t descriptor_bytes = (source_count + target_count) *
        sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor);
    plan->metadata_bytes = sizeof(metadata) + descriptor_bytes;
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_metadata,
            sizeof(metadata), cudaHostAllocPortable);
    if (error == cudaSuccess)
        error = cudaEventCreateWithFlags(&plan->raw_ready, cudaEventDisableTiming);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    auto *descriptors = reinterpret_cast<GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(
        static_cast<uint8_t *>(plan->device_metadata) + sizeof(metadata));
    plan->raw_source_descriptors = descriptors;
    plan->raw_target_descriptors = descriptors + source_count;
    metadata.source = plan->raw_source_descriptors;
    metadata.target = plan->raw_target_descriptors;
    std::memcpy(plan->pinned_metadata, &metadata, sizeof(metadata));
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        sizeof(metadata), cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess)
        error = cudaEventRecord(plan->raw_ready, plan->stream);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_crt_recompose_level_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *target,
    int initialize, uint32_t source_binding_base, uint32_t target_binding_base)
{
    if (!plan || !plan->raw_recompose || !ctx || !stream_raw ||
        !plan->context || ctx->execution != plan->context->execution ||
        (initialize != 0 && initialize != 1) ||
        !raw_recompose_view_matches(source,
            static_cast<const RawRecomposeMetadata *>(plan->pinned_metadata)->source_moduli,
            plan->source_count, plan->device, plan->ring_dimension) ||
        !raw_recompose_view_matches(target,
            static_cast<const RawRecomposeMetadata *>(plan->pinned_metadata)->target_moduli,
            plan->target_count, plan->device, plan->ring_dimension) ||
        source->columns != target->columns ||
        source_binding_base > UINT32_MAX - plan->source_count ||
        target_binding_base > UINT32_MAX - plan->target_count)
        return set_error("invalid raw CRT recomposition views");
    const size_t count = source->columns * source->degree;
    if (!count || count > static_cast<size_t>(std::numeric_limits<int>::max() - 1) * 128)
        return set_error("raw CRT recomposition grid exceeds capacity");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t index = 0; index < source->limb_count; ++index)
    {
        const MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            0, static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
            sizeof(void *), source_binding_base + static_cast<uint32_t>(index), 0};
        const int status = mxx_gpu_launch_kernel(ctx, stream,
            raw_recompose_descriptor_kernel, dim3(1), dim3(1), 0,
            &patch, 1, source->limbs[index],
            plan->raw_source_descriptors + index);
        if (status != 0) return status;
    }
    for (size_t index = 0; index < target->limb_count; ++index)
    {
        const MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            0, static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
            sizeof(void *), target_binding_base + static_cast<uint32_t>(index), 0};
        const int status = mxx_gpu_launch_kernel(ctx, stream,
            raw_recompose_descriptor_kernel, dim3(1), dim3(1), 0,
            &patch, 1, target->limbs[index],
            plan->raw_target_descriptors + index);
        if (status != 0) return status;
    }
    return mxx_gpu_launch_kernel(ctx, stream, raw_recompose_level_kernel,
        dim3(static_cast<unsigned>((count + 127) / 128)), dim3(128), 0,
        nullptr, 0, static_cast<const RawRecomposeMetadata *>(plan->device_metadata),
        count, source->degree, initialize != 0);
}

namespace
{
    struct RawCompactPackMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        size_t limb_count;
        int word_count;
        uint32_t magnitude_bytes;
        uint64_t moduli[kCrtMaxLimbs];
        uint64_t prefix_inverses[kCrtMaxLimbs];
        uint64_t product[kCrtMaxWords];
        uint64_t half[kCrtMaxWords];
        uint64_t bound[kCrtMaxWords];
    };

    __global__ void raw_compact_pack_kernel(
        const RawCompactPackMetadata *metadata, MxxRawSmallMatrixView destination,
        uint32_t *status, size_t coefficient_count)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count || *status != 0U) return;
        const size_t poly = index / destination.degree;
        const size_t coefficient = index % destination.degree;
        uint64_t digits[kCrtMaxLimbs]{};
        crt_recover_centered_digits(metadata->source, nullptr, metadata->moduli,
            nullptr, metadata->prefix_inverses, metadata->limb_count,
            poly, coefficient, digits);
        uint64_t value[kCrtMaxWords]{};
        for (size_t limb = metadata->limb_count; limb-- > 0;)
        {
            uint64_t carry = digits[limb];
            for (int word = 0; word < metadata->word_count; ++word)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(value[word]) *
                    metadata->moduli[limb] + carry;
                value[word] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
        }
        const bool negative =
            crt_compare_words(value, metadata->half, metadata->word_count) > 0;
        uint64_t magnitude[kCrtMaxWords]{};
        if (negative)
        {
            uint64_t borrow = 0;
            for (int word = 0; word < metadata->word_count; ++word)
            {
                const unsigned __int128 subtrahend =
                    static_cast<unsigned __int128>(value[word]) + borrow;
                const unsigned __int128 minuend = metadata->product[word];
                magnitude[word] = static_cast<uint64_t>(minuend - subtrahend);
                borrow = minuend < subtrahend;
            }
        }
        else
            for (int word = 0; word < metadata->word_count; ++word)
                magnitude[word] = value[word];
        if (crt_compare_words(magnitude, metadata->bound, metadata->word_count) > 0)
        {
            atomicCAS(status, 0U, 2U);
            return;
        }
        const size_t row = poly / destination.columns;
        const size_t column = poly - row * destination.columns;
        const size_t packed_poly = row * destination.storage_columns +
            destination.column_offset + column;
        const size_t width = static_cast<size_t>(destination.magnitude_bytes) + 1;
        auto *encoded = reinterpret_cast<uint8_t *>(destination.payload_address) +
            (packed_poly * destination.degree + coefficient) * width;
        bool zero = true;
        for (int word = 0; word < metadata->word_count; ++word)
            zero = zero && magnitude[word] == 0;
        encoded[0] = zero ? 0 : (negative ? 2 : 1);
        for (size_t byte = 0; byte < destination.magnitude_bytes; ++byte)
            encoded[1 + byte] = byte / 8 < static_cast<size_t>(metadata->word_count) ?
                static_cast<uint8_t>(magnitude[byte / 8] >> (8 * (byte % 8))) : 0;
    }
}

extern "C" int gpu_raw_compact_pack_prepare(
    GpuContext *ctx, int32_t device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *bound_words, size_t bound_count,
    uint32_t magnitude_bytes, GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !out_plan || !source_moduli || !source_count ||
        source_count > kCrtMaxLimbs || !bound_words || !bound_count ||
        bound_count > kCrtMaxWords || !magnitude_bytes || magnitude_bytes > 64)
        return set_error("invalid raw compact pack plan arguments");
    *out_plan = nullptr;
    RawCompactPackMetadata metadata{};
    metadata.limb_count = source_count;
    metadata.magnitude_bytes = magnitude_bytes;
    std::copy_n(source_moduli, source_count, metadata.moduli);
    std::copy_n(bound_words, bound_count, metadata.bound);
    for (size_t word = (magnitude_bytes + 7) / 8; word < bound_count; ++word)
        if (bound_words[word] != 0)
            return set_error("raw compact bound exceeds magnitude encoding");
    if (magnitude_bytes % 8 &&
        bound_count > magnitude_bytes / 8 &&
        (bound_words[magnitude_bytes / 8] >> (8 * (magnitude_bytes % 8))) != 0)
        return set_error("raw compact bound exceeds magnitude encoding");
    std::vector<uint64_t> product;
    if (!serde_compute_modulus_words_le(
            std::vector<uint64_t>(source_moduli, source_moduli + source_count),
            &product) || product.empty() || product.size() > kCrtMaxWords)
        return set_error("raw compact source modulus exceeds supported width");
    metadata.word_count = static_cast<int>(product.size());
    std::copy(product.begin(), product.end(), metadata.product);
    bool bound_exceeds_source_width = false;
    for (size_t word = product.size(); word < bound_count; ++word)
        bound_exceeds_source_width |= bound_words[word] != 0;
    if (bound_exceeds_source_width)
        std::fill_n(metadata.bound, product.size(), UINT64_MAX);
    for (size_t word = 0; word < product.size(); ++word)
    {
        const uint64_t next = word + 1 < product.size() ? product[word + 1] : 0;
        metadata.half[word] = (product[word] >> 1) | ((next & 1) << 63);
    }
    for (size_t index = 0; index < source_count; ++index)
    {
        if (index >= ctx->moduli.size() || ctx->moduli[index] != source_moduli[index] ||
            source_moduli[index] <= 1)
            return set_error("raw compact source basis mismatch");
        uint64_t prefix = 1;
        for (size_t previous = 0; previous < index; ++previous)
            prefix = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(prefix) *
                    (source_moduli[previous] % source_moduli[index])) %
                source_moduli[index]);
        if (!mod_inverse_u64(prefix, source_moduli[index],
                metadata.prefix_inverses[index]))
            return set_error("raw compact source basis is not coprime");
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw compact pack plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->raw_compact_pack = true;
    plan->metadata_bytes = sizeof(metadata) + source_count *
        sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor);
    error = cudaMallocAsync(&plan->device_metadata, plan->metadata_bytes, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_metadata, sizeof(metadata), cudaHostAllocPortable);
    if (error == cudaSuccess)
        error = cudaEventCreateWithFlags(&plan->raw_ready, cudaEventDisableTiming);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    plan->raw_source_descriptors =
        reinterpret_cast<GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(
            static_cast<uint8_t *>(plan->device_metadata) + sizeof(metadata));
    metadata.source = plan->raw_source_descriptors;
    std::memcpy(plan->pinned_metadata, &metadata, sizeof(metadata));
    error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
        sizeof(metadata), cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess)
        error = cudaEventRecord(plan->raw_ready, plan->stream);
    if (error != cudaSuccess)
    {
        gpu_matrix_modulus_conversion_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_compact_pack_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawSmallMatrixView *destination,
    uint32_t *status_address, uint32_t source_binding_base,
    uint32_t destination_binding, uint32_t status_binding)
{
    if (!plan || !plan->raw_compact_pack || !ctx || !plan->context ||
        ctx->execution != plan->context->execution || !stream_raw ||
        !raw_recompose_view_matches(source,
            static_cast<const RawCompactPackMetadata *>(plan->pinned_metadata)->moduli,
            plan->source_count, plan->device, plan->ring_dimension, false) ||
        !destination || !destination->payload_address || !status_address ||
        destination->physical_device != plan->device ||
        destination->degree != source->degree ||
        destination->rows != source->rows ||
        destination->columns != source->columns ||
        destination->column_offset > destination->storage_columns ||
        destination->columns > destination->storage_columns - destination->column_offset ||
        !destination->magnitude_bytes || destination->magnitude_bytes > 64 ||
        destination->magnitude_bytes !=
            static_cast<const RawCompactPackMetadata *>(plan->pinned_metadata)->magnitude_bytes ||
        source_binding_base > UINT32_MAX - plan->source_count)
        return set_error("invalid raw compact pack views");
    const size_t count = source->rows * source->columns * source->degree;
    if (!count || count > static_cast<size_t>(std::numeric_limits<int>::max() - 1) * 128)
        return set_error("raw compact pack grid exceeds capacity");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t index = 0; index < source->limb_count; ++index)
    {
        const MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            0, static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
            sizeof(void *), source_binding_base + static_cast<uint32_t>(index), 0};
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_recompose_descriptor_kernel, dim3(1), dim3(1), 0,
            &patch, 1, source->limbs[index],
            plan->raw_source_descriptors + index);
        if (result != 0) return result;
    }
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1,
            static_cast<uint32_t>(offsetof(MxxRawSmallMatrixView, payload_address)),
            sizeof(void *), destination_binding, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0,
            sizeof(void *), status_binding, 0},
    };
    return mxx_gpu_launch_kernel(ctx, stream, raw_compact_pack_kernel,
        dim3(static_cast<unsigned>((count + 127) / 128)), dim3(128), 0,
        patches, 2, static_cast<const RawCompactPackMetadata *>(plan->device_metadata),
        *destination, status_address, count);
}

namespace
{
    __global__ void raw_compact_pack_per_crt_limb_kernel(
        MxxRawMatrixLimb source, MxxRawSmallMatrixView destination,
        uint32_t *status, uint64_t bound, size_t limb,
        size_t coefficient_count, size_t coefficient_offset)
    {
        const size_t index = coefficient_offset +
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t coefficient = index % destination.degree;
        const size_t poly = index / destination.degree;
        const uint64_t residue = raw_matrix_load(
            source, poly, coefficient, destination.columns);
        const bool negative = residue > source.modulus / 2;
        const uint64_t magnitude = negative ? source.modulus - residue : residue;
        if (magnitude > bound)
        {
            atomicCAS(status, 0U, 2U);
            return;
        }
        const size_t row = poly / destination.columns;
        const size_t column = poly - row * destination.columns;
        const size_t packed_poly = row * destination.storage_columns +
            destination.column_offset + column;
        const size_t width = static_cast<size_t>(destination.magnitude_bytes) + 1;
        auto *encoded = reinterpret_cast<uint8_t *>(destination.payload_address) +
            ((packed_poly * destination.degree + coefficient) * destination.crt_depth + limb) * width;
        encoded[0] = magnitude == 0 ? 0 : (negative ? 2 : 1);
        for (size_t byte = 0; byte < destination.magnitude_bytes; ++byte)
            encoded[1 + byte] = byte < sizeof(uint64_t) ?
                static_cast<uint8_t>(magnitude >> (8 * byte)) : 0;
    }
}

extern "C" int gpu_raw_compact_pack_per_crt_limb(
    GpuContext *ctx, void *stream_raw, const MxxRawMatrixView *source,
    const MxxRawSmallMatrixView *destination, uint32_t *status,
    uint64_t bound, uint32_t source_binding_base,
    uint32_t destination_binding, uint32_t status_binding)
{
    if (!ctx || !stream_raw || !source || !destination || !status ||
        validate_raw_view(ctx, source, stream_raw) != 0 ||
        source->limb_count == 0 || source->limb_count > kCrtMaxLimbs ||
        destination->bound_domain != 1 ||
        destination->crt_depth != source->limb_count ||
        destination->physical_device != source->physical_device ||
        destination->degree != source->degree ||
        destination->rows != source->rows ||
        destination->columns != source->columns ||
        destination->column_offset > destination->storage_columns ||
        destination->columns > destination->storage_columns - destination->column_offset ||
        !destination->payload_address || !destination->magnitude_bytes ||
        destination->magnitude_bytes > 8 ||
        (destination->magnitude_bytes < 8 &&
            (bound >> (8 * destination->magnitude_bytes)) != 0) ||
        source_binding_base > UINT32_MAX - source->limb_count)
        return set_error("invalid per-CRT-limb compact pack views");
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        if (source->limbs[limb].crt_limb_index != limb ||
            limb >= ctx->moduli.size() ||
            source->limbs[limb].modulus != ctx->moduli[limb])
            return set_error("per-CRT-limb compact pack basis mismatch");
    }
    if (source->rows > SIZE_MAX / source->columns ||
        source->rows * source->columns > SIZE_MAX / source->degree)
        return set_error("per-CRT-limb compact pack coefficient count overflows");
    const size_t count = source->rows * source->columns * source->degree;
    if (!count) return set_error("empty per-CRT-limb compact pack view");
    cudaError_t error = cudaSetDevice(source->physical_device);
    if (error != cudaSuccess) return set_error(error);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    constexpr size_t maximum_chunk = 65535ULL * 256ULL;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), source_binding_base + static_cast<uint32_t>(limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1,
                static_cast<uint32_t>(offsetof(MxxRawSmallMatrixView, payload_address)),
                sizeof(void *), destination_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0,
                sizeof(void *), status_binding, 0},
        };
        for (size_t offset = 0; offset < count; offset += maximum_chunk)
        {
            const size_t chunk = std::min(maximum_chunk, count - offset);
            const int result = mxx_gpu_launch_kernel(ctx, stream,
                raw_compact_pack_per_crt_limb_kernel,
                dim3(static_cast<uint32_t>((chunk + 255) / 256)), dim3(256), 0,
                patches, 3, source->limbs[limb], *destination, status,
                bound, limb, count, offset);
            if (result != 0) return result;
        }
    }
    return 0;
}

namespace
{
    // By-value launch parameters avoid pinned allocation and host metadata
    // transfers. CUDA copies these arguments before the launch returns.
    struct RnsLaunchMetadata
    {
        size_t source_count;
        size_t target_count;
        size_t digit_size;
        size_t group_count;
        size_t retained[kCrtMaxLimbs];
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t target_moduli[kCrtMaxLimbs];
        uint64_t scales[kCrtMaxLimbs];
        uint64_t inverses[kCrtMaxLimbs];
        uint64_t plaintext_modulus;
        uint64_t plaintext_residues[kCrtMaxLimbs];
    };
    static_assert(sizeof(RnsLaunchMetadata) + sizeof(void *) < 4096,
        "RNS setup arguments must fit the baseline CUDA kernel argument limit");

    struct RnsConversionMetadata
    {
        RnsLaunchMetadata plan;
        uint64_t weights[kCrtMaxLimbs * kCrtMaxLimbs];
    };

    __global__ void rns_setup_kernel(RnsConversionMetadata *metadata, RnsLaunchMetadata plan)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index == 0) metadata->plan = plan;
        if (index >= plan.source_count * plan.target_count) return;
        const size_t input = index / plan.target_count;
        const size_t target = index % plan.target_count;
        const bool down = plan.plaintext_modulus != 0;
        const uint64_t modulus = plan.target_moduli[target];
        uint64_t weight = down && plan.scales[input] == 0 ? 0 : 1;
        const size_t begin = down ? 0 : (input / plan.digit_size) * plan.digit_size;
        const size_t end = down ? plan.source_count : min(plan.source_count, begin + plan.digit_size);
        for (size_t limb = begin; limb < end && weight != 0; ++limb)
        {
            // Zero scales mark Q limbs retained by ModDown; P is their complement.
            if (limb != input && (!down || plan.scales[limb] != 0))
                weight = mul_mod_u64(weight, plan.source_moduli[limb] % modulus, modulus);
        }
        metadata->weights[input * kCrtMaxLimbs + target] = weight;
    }

    __device__ __forceinline__ void rns_convert_coefficient(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const RnsLaunchMetadata &plan, const uint64_t *weights, size_t weight_stride,
        size_t coefficient_count, size_t ring_dimension, bool mod_down)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * plan.target_count * plan.group_count) return;
        const size_t position = index % coefficient_count;
        const size_t limb = (index / coefficient_count) % plan.target_count;
        const size_t group = index / coefficient_count / plan.target_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        const uint64_t modulus = plan.target_moduli[limb];
        const size_t begin = mod_down ? 0 : group * plan.digit_size;
        const size_t end = mod_down ? plan.source_count :
            min(plan.source_count, begin + plan.digit_size);
        uint64_t sum = 0;
        for (size_t input_limb = begin; input_limb < end; ++input_limb)
        {
            const uint64_t weight = weights[input_limb * weight_stride + limb];
            if (weight == 0) continue;
            const auto input = source[input_limb];
            const uint64_t source_modulus = plan.source_moduli[input_limb];
            const uint64_t residue = mul_mod_u64(matrix_load_limb_u64(input.base, poly,
                coefficient, input.stride, input.width) % source_modulus,
                plan.scales[input_limb], source_modulus);
            const bool negative = residue > source_modulus / 2;
            const uint64_t magnitude = (negative ? source_modulus - residue : residue) % modulus;
            const uint64_t centered = negative && magnitude != 0 ? modulus - magnitude : magnitude;
            sum = add_mod_u64(sum, mul_mod_u64(centered, weight, modulus), modulus);
        }
        if (mod_down)
        {
            const auto input = source[plan.retained[limb]];
            const uint64_t residue = matrix_load_limb_u64(input.base, poly, coefficient,
                input.stride, input.width) % modulus;
            sum = mul_mod_u64(add_mod_u64(residue,
                mul_mod_u64(plan.plaintext_residues[limb], sum, modulus), modulus),
                plan.inverses[limb], modulus);
        }
        const auto output = target[limb];
        const size_t output_poly = group * (coefficient_count / ring_dimension) + poly;
        matrix_store_limb_u64(output.base, output_poly, coefficient,
            output.stride, output.width, sum);
    }
    struct RnsCompactMetadata
    {
        RnsLaunchMetadata plan;
        uint64_t weights[64];
    };
    static_assert(sizeof(RnsCompactMetadata) + 2 * sizeof(void *) + 2 * sizeof(size_t) + sizeof(bool) <= 4096,
                  "compact RNS launch exceeds portable CUDA parameter budget");

    __global__ void rns_compact_conversion_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        RnsCompactMetadata metadata, size_t count, size_t dimension, bool down)
    {
        rns_convert_coefficient(source, target, metadata.plan, metadata.weights,
                                metadata.plan.target_count, count, dimension, down);
    }

    __global__ void rns_conversion_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const RnsConversionMetadata *metadata, size_t count, size_t dimension, bool down)
    {
        rns_convert_coefficient(source, target, metadata->plan, metadata->weights,
                                kCrtMaxLimbs, count, dimension, down);
    }

}

extern "C" int gpu_matrix_rns_conversion(
    GpuMatrix *out, const GpuMatrix *source, size_t digit_size,
    uint64_t plaintext_modulus, const uint64_t *scales,
    const uint64_t *inverses, const uint64_t *weights)
{
    const bool mod_down = plaintext_modulus != 0;
    if (plaintext_modulus == 1) return set_error("RNS ModDown plaintext modulus must be at least two");
    if (!out || !source || !out->ctx || !source->ctx ||
        out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
        out->format != GPU_POLY_FORMAT_COEFF || source->format != GPU_POLY_FORMAT_COEFF ||
        out->level < 0 || source->level < 0 || !scales || !inverses ||
        digit_size == 0 || source->cols != out->cols ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("invalid fused RNS conversion layout");
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    const size_t groups = mod_down ? 1 : source_count / digit_size + (source_count % digit_size != 0);
    if (source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs ||
        source_count != source->ctx->moduli.size() || target_count != out->ctx->moduli.size() ||
        source->rows > std::numeric_limits<size_t>::max() / groups || out->rows != source->rows * groups)
        return set_error("invalid fused RNS conversion basis or shape");
    RnsLaunchMetadata metadata{};
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    metadata.digit_size = digit_size;
    metadata.group_count = groups;
    metadata.plaintext_modulus = plaintext_modulus;
    for (size_t limb = 0; limb < target_count; ++limb)
        metadata.plaintext_residues[limb] = plaintext_modulus % out->ctx->moduli[limb];
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        metadata.source_moduli[limb] = source->ctx->moduli[limb];
        metadata.scales[limb] = scales[limb];
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        metadata.target_moduli[limb] = out->ctx->moduli[limb];
        metadata.inverses[limb] = inverses[limb];
        if (mod_down)
        {
            size_t retained = 0;
            while (retained < source_count && source->ctx->moduli[retained] != out->ctx->moduli[limb]) ++retained;
            if (retained == source_count) return set_error("ModDown target must be a source subset");
            metadata.retained[limb] = retained;
        }
    }
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("fused RNS conversion requires colocated descriptors");
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb) return set_error("unsupported source RNS limb placement");
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb) return set_error("unsupported target RNS limb placement");
    }
    status = matrix_wait_all_limb_streams(source, output.device, stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, output.device, stream);
    if (status != 0) return status;
    const size_t dimension = static_cast<size_t>(source->ctx->N);
    if (source->rows == 0 || source->cols == 0 ||
        source->rows > std::numeric_limits<size_t>::max() / source->cols ||
        source->rows * source->cols > std::numeric_limits<size_t>::max() / dimension)
        return set_error("fused RNS conversion shape overflow");
    const size_t count = source->rows * source->cols * dimension;
    if (count > std::numeric_limits<size_t>::max() / target_count / groups)
        return set_error("fused RNS conversion grid overflow");
    const size_t total = count * target_count * groups;
    const size_t blocks = total / 128 + (total % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("fused RNS conversion exceeds CUDA grid capacity");
    RnsConversionMetadata *device_metadata = nullptr;
    if (source_count * target_count <= 64)
    {
        if (!weights) return set_error("missing compact RNS weights");
        RnsCompactMetadata compact{};
        compact.plan = metadata;
        std::copy_n(weights, source_count * target_count, compact.weights);
        rns_compact_conversion_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            input.device_descriptors, output.device_descriptors, compact, count, dimension, mod_down);
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
    }
    else
    {
        error = cudaMallocAsync(reinterpret_cast<void **>(&device_metadata), sizeof(RnsConversionMetadata), stream);
        if (error != cudaSuccess) return set_error(error);
        const size_t setup_count = source_count * target_count;
        const size_t setup_blocks = setup_count / 128 + (setup_count % 128 != 0);
        rns_setup_kernel<<<static_cast<int>(setup_blocks), 128, 0, stream>>>(device_metadata, metadata);
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
        if (status == 0)
        {
            rns_conversion_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
                input.device_descriptors, output.device_descriptors, device_metadata, count, dimension, mod_down);
            error = cudaGetLastError();
            if (error != cudaSuccess) status = set_error(error);
        }
    }
    if (status == 0)
    {
        // All source limbs share this partition's allocation/release stream.
        // One readonly consumer fence protects the complete buffer without
        // mutating a coefficient input shared by concurrent host readers.
        const int tracked = matrix_track_limb_consumer_readonly(
            source, source->ctx->limb_gpu_ids[0], output.device, stream);
        if (status == 0) status = tracked;
        const int recorded = matrix_record_all_limb_writes(out, stream);
        if (status == 0) status = recorded;
    }
    if (device_metadata)
    {
        error = cudaFreeAsync(device_metadata, stream);
        if (status == 0 && error != cudaSuccess) status = set_error(error);
    }
    return status;
}
