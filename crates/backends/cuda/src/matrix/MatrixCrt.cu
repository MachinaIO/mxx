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
    // Nonzero when every packed magnitude is below half of every source
    // modulus, so one limb's centered residue is the value itself.
    uint64_t compact_single_limb_bound = 0;
    bool compact_single_limb = false;
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
    cudaError_t error = mxx_set_device(physical_device);
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
        mxx_set_device(plan->device);
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
    cudaError_t error = mxx_set_device(device);
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
    cudaError_t error = mxx_set_device(device);
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
    cudaError_t error = mxx_set_device(device);
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
    if (mxx_set_device(plan->device) != cudaSuccess)
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
    cudaError_t error = mxx_set_device(device);
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
    cudaError_t error = mxx_set_device(plan->device);
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

namespace
{
    // The single-limb form of `raw_compact_pack_kernel`: with the bound below
    // half of the limb modulus, the centered residue is the packed value.
    __global__ void raw_compact_pack_single_limb_kernel(
        MxxRawMatrixLimb source, MxxRawSmallMatrixView destination,
        uint32_t *status, uint64_t bound, size_t coefficient_count)
    {
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < coefficient_count; index += stride)
        {
            const size_t poly = index / destination.degree;
            const size_t coefficient = index - poly * destination.degree;
            const uint64_t residue = raw_matrix_load(
                source, poly, coefficient, destination.columns);
            const bool negative = residue > source.modulus / 2;
            const uint64_t magnitude = negative ? source.modulus - residue : residue;
            if (magnitude > bound)
            {
                atomicCAS(status, 0U, 2U);
                continue;
            }
            const size_t row = poly / destination.columns;
            const size_t column = poly - row * destination.columns;
            const size_t packed_poly = row * destination.storage_columns +
                destination.column_offset + column;
            const size_t width = static_cast<size_t>(destination.magnitude_bytes) + 1;
            auto *encoded = reinterpret_cast<uint8_t *>(destination.payload_address) +
                (packed_poly * destination.degree + coefficient) * width;
            encoded[0] = magnitude == 0 ? 0 : (negative ? 2 : 1);
            for (size_t byte = 0; byte < destination.magnitude_bytes; ++byte)
                encoded[1 + byte] = byte < sizeof(uint64_t) ?
                    static_cast<uint8_t>(magnitude >> (8 * byte)) : 0;
        }
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
    cudaError_t error = mxx_set_device(device);
    if (error != cudaSuccess) return set_error(error);
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw compact pack plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->raw_compact_pack = true;
    {
        bool single = !bound_exceeds_source_width;
        for (size_t word = 1; word < bound_count; ++word)
            single = single && bound_words[word] == 0;
        for (size_t index = 0; index < source_count; ++index)
            single = single && bound_words[0] < source_moduli[index] / 2;
        plan->compact_single_limb = single;
        plan->compact_single_limb_bound = bound_words[0];
    }
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
    cudaError_t error = mxx_set_device(plan->device);
    if (error != cudaSuccess) return set_error(error);
    if (plan->compact_single_limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), source_binding_base, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1,
                static_cast<uint32_t>(offsetof(MxxRawSmallMatrixView, payload_address)),
                sizeof(void *), destination_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0,
                sizeof(void *), status_binding, 0},
        };
        return mxx_gpu_launch_kernel(ctx, stream, raw_compact_pack_single_limb_kernel,
            dim3(static_cast<unsigned>(std::min<size_t>(65535, (count + 255) / 256))),
            dim3(256), 0, patches, 3, source->limbs[0], *destination, status_address,
            plan->compact_single_limb_bound, count);
    }
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
    cudaError_t error = mxx_set_device(source->physical_device);
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

    __global__ void rns_conversion_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const RnsConversionMetadata *metadata, size_t count, size_t dimension, bool down)
    {
        rns_convert_coefficient(source, target, metadata->plan, metadata->weights,
                                kCrtMaxLimbs, count, dimension, down);
    }

}

