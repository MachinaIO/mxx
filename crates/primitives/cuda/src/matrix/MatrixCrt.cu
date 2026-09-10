namespace
{
    constexpr int kCrtMaxLimbs = 64;
    constexpr int kCrtMaxWords = 64;

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
        uint64_t input_scales[kCrtMaxLimbs];
        uint64_t plaintext_modulus;
        uint64_t divisor_residues[kCrtMaxLimbs];
        uint64_t garner[kCrtMaxLimbs * kCrtMaxLimbs];
    };

    __global__ void convert_modulus_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const ModulusConversionMetadata *metadata,
        size_t coefficient_count,
        size_t ring_dimension,
        int conversion)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t digits[kCrtMaxLimbs];
        bool negative = false;
        if (conversion != 0)
        {
            for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            {
                const size_t source_index = metadata->discarded[limb];
                const auto descriptor = source[source_index];
                const uint64_t modulus = metadata->source_moduli[source_index];
                uint64_t digit = matrix_load_limb_u64(descriptor.base, poly, coefficient,
                    descriptor.stride, descriptor.width) % modulus;
                digit = mul_mod_u64(digit, metadata->input_scales[source_index], modulus);
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
            for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            {
                const uint64_t half = metadata->source_moduli[metadata->discarded[limb]] / 2;
                negative = (digits[limb] > half) | ((digits[limb] == half) & negative);
            }
        }
        for (size_t limb = 0; limb < metadata->target_count; ++limb)
        {
            const uint64_t modulus = metadata->target_moduli[limb];
            uint64_t value = 0;
            if (conversion != 2)
            {
                const auto input = source[metadata->retained[limb]];
                value = matrix_load_limb_u64(input.base, poly, coefficient,
                    input.stride, input.width) % modulus;
            }
            if (conversion != 0)
            {
                uint64_t residual = 0;
                for (size_t digit = metadata->discarded_count; digit-- > 0;)
                {
                    residual = add_mod_u64(mul_mod_u64(residual,
                        metadata->source_moduli[metadata->discarded[digit]] % modulus,
                        modulus), digits[digit] % modulus, modulus);
                }
                const uint64_t divisor = metadata->divisor_residues[limb] &
                    (uint64_t{0} - static_cast<uint64_t>(negative));
                residual = residual >= divisor ? residual - divisor : modulus - (divisor - residual);
                if (conversion == 2) value = residual;
                else
                {
                    if (conversion == 3)
                        residual = mul_mod_u64(residual, metadata->plaintext_modulus % modulus, modulus);
                    value = value >= residual ? value - residual : modulus - (residual - value);
                    value = mul_mod_u64(value, metadata->division_inverses[limb], modulus);
                }
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
        uint64_t garner[kCrtMaxLimbs * kCrtMaxLimbs];
        uint64_t modulus_words[kCrtMaxWords];
        uint64_t reconstruction[kCrtMaxLimbs];
    };

    struct CrtOutputMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        size_t limb_count;
        uint64_t moduli[kCrtMaxLimbs];
    };

    __global__ void centered_rebase_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        CrtOutputMetadata output, uint64_t source_modulus,
        size_t coefficient_count, size_t ring_dimension)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * output.limb_count) return;
        const size_t limb = index / coefficient_count;
        const size_t position = index % coefficient_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        const auto input = source[0];
        const uint64_t residue = matrix_load_limb_u64(input.base, poly, coefficient,
            input.stride, input.width) % source_modulus;
        const uint64_t modulus = output.moduli[limb];
        // Keep the magnitude unsigned, including for source moduli above INT64_MAX.
        const bool negative = residue > source_modulus / 2;
        const uint64_t magnitude = (negative ? source_modulus - residue : residue) % modulus;
        const uint64_t value = negative && magnitude != 0 ? modulus - magnitude : magnitude;
        const auto target = output.descriptors[limb];
        matrix_store_limb_u64(target.base, poly, coefficient, target.stride, target.width, value);
    }

    __global__ void crt_recompose_kernel(
        const CrtLevelMetadata *levels, const CrtOutputMetadata *output,
        size_t level_count, size_t coefficient_count, size_t ring_dimension)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t accumulated[kCrtMaxLimbs]{};
        for (size_t level = 0; level < level_count; ++level)
        {
            const auto &metadata = levels[level];
            uint64_t mixed_digits[kCrtMaxLimbs];
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const auto descriptor = metadata.descriptors[limb];
                const uint64_t modulus = metadata.moduli[limb];
                uint64_t digit = matrix_load_limb_u64(descriptor.base, poly, coefficient,
                    descriptor.stride, descriptor.width) % modulus;
                for (size_t previous = 0; previous < limb; ++previous)
                {
                    const uint64_t residue = mixed_digits[previous] % modulus;
                    const uint64_t difference = digit >= residue ? digit - residue :
                        modulus - (residue - digit);
                    digit = mul_mod_u64(difference,
                        metadata.garner[previous * kCrtMaxLimbs + limb], modulus);
                }
                mixed_digits[limb] = digit;
            }
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
            for (size_t limb = 0; limb < output->limb_count; ++limb)
            {
                const uint64_t modulus = output->moduli[limb];
                accumulated[limb] = add_mod_u64(accumulated[limb],
                    mul_mod_u64(rounded % modulus, metadata.reconstruction[limb], modulus), modulus);
            }
        }
        for (size_t limb = 0; limb < output->limb_count; ++limb)
        {
            const auto descriptor = output->descriptors[limb];
            matrix_store_limb_u64(descriptor.base, poly, coefficient,
                descriptor.stride, descriptor.width, accumulated[limb]);
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
}

extern "C" int gpu_matrix_convert_modulus(
    GpuMatrix *out,
    const GpuMatrix *source,
    int conversion,
    const uint64_t *division_inverses,
    size_t inverse_count,
    uint64_t plaintext_modulus,
    const uint64_t *input_scales)
{
    if (!out || !source || !out->ctx || !source->ctx ||
        out->ctx->execution != source->ctx->execution ||
        out->ctx->N != source->ctx->N || out->rows != source->rows || out->cols != source->cols ||
        out->format != source->format ||
        (conversion != 0 && source->format != GPU_POLY_FORMAT_COEFF) ||
        (conversion < 0 || conversion > 3) || source->level < 0 || out->level < 0 ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("invalid coefficient modulus conversion layout");
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs || (conversion != 2 && target_count > source_count) ||
        source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size() ||
        !division_inverses || inverse_count != target_count)
        return set_error("invalid coefficient modulus conversion basis");
    if (conversion == 3 && (plaintext_modulus == 0 || !input_scales || target_count >= source_count))
        return set_error("invalid exact BGV block switch parameters");
    std::vector<ModulusConversionMetadata> host(1);
    auto &metadata = host[0];
    metadata.plaintext_modulus = plaintext_modulus;
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    bool retained[kCrtMaxLimbs]{};
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        metadata.source_moduli[limb] = source->ctx->moduli[limb];
        metadata.input_scales[limb] = conversion == 3 ? input_scales[limb] : 1;
        if (metadata.source_moduli[limb] <= 1 || !(metadata.source_moduli[limb] & 1))
            return set_error("nearest conversion requires odd CRT moduli");
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = out->ctx->moduli[limb];
        size_t selected = 0;
        while (selected < source_count && metadata.source_moduli[selected] != modulus) ++selected;
        if (conversion != 2 && (selected == source_count || retained[selected]))
            return set_error("destination CRT basis is not an exact subset");
        if (selected < source_count) retained[selected] = true;
        metadata.retained[limb] = selected;
        metadata.target_moduli[limb] = modulus;
        metadata.division_inverses[limb] = division_inverses[limb];
    }
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        if (conversion == 2 && !retained[limb])
            return set_error("centered extension requires a containing CRT basis");
        if (conversion == 3 && (static_cast<unsigned __int128>(plaintext_modulus) *
            metadata.input_scales[limb]) % metadata.source_moduli[limb] != 1)
            return set_error("invalid BGV plaintext inverse");
        if (conversion == 2 || !retained[limb]) metadata.discarded[metadata.discarded_count++] = limb;
    }
    for (size_t limb = 0; limb < metadata.discarded_count; ++limb)
        for (size_t previous = 0; previous < limb; ++previous)
            metadata.garner[previous * kCrtMaxLimbs + limb] = source->ctx->garner_inverse_table[
                metadata.discarded[previous] * source->ctx->moduli.size() + metadata.discarded[limb]];
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = metadata.target_moduli[limb];
        uint64_t divisor = 1;
        for (size_t discarded = 0; discarded < metadata.discarded_count; ++discarded)
            divisor = static_cast<uint64_t>((static_cast<unsigned __int128>(divisor) *
                metadata.source_moduli[metadata.discarded[discarded]]) % modulus);
        metadata.divisor_residues[limb] = divisor;
        if ((conversion == 1 || conversion == 3) && (static_cast<unsigned __int128>(divisor) * division_inverses[limb]) % modulus != 1)
            return set_error("invalid exact modulus division inverse");
    }
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("coefficient conversion requires colocated device descriptors");
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported source CRT limb placement");
        status = matrix_wait_limb_stream(source, id, output.device, stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported destination CRT limb placement");
        status = matrix_wait_limb_stream(out, id, output.device, stream);
        if (status != 0) return status;
    }
    if (out->rows == 0 || out->cols == 0 || out->rows > std::numeric_limits<size_t>::max() / out->cols ||
        out->rows * out->cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(out->ctx->N))
        return set_error("coefficient conversion shape overflow");
    const size_t coefficient_count = out->rows * out->cols * static_cast<size_t>(out->ctx->N);
    const size_t blocks = coefficient_count / 128 + (coefficient_count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("coefficient conversion exceeds CUDA grid capacity");
    ModulusConversionMetadata *device_metadata = nullptr;
    std::vector<void *> pinned;
    status = crt_alloc_and_copy_async(&device_metadata, host, stream, &pinned);
    if (status == 0)
    {
        convert_modulus_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            input.device_descriptors, output.device_descriptors, device_metadata,
            coefficient_count, static_cast<size_t>(out->ctx->N), conversion);
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
        // Record consumers even when a launch reports an error: neither ring's
        // buffers may be reclaimed ahead of any work already queued here.
        for (size_t limb = 0; limb < source_count; ++limb)
        {
            const int tracked = matrix_track_limb_consumer_readonly(source, source->ctx->limb_gpu_ids[limb],
                output.device, stream);
            if (status == 0) status = tracked;
        }
        for (size_t limb = 0; limb < target_count; ++limb)
        {
            const int recorded = matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream);
            if (status == 0) status = recorded;
        }
    }
    if (device_metadata) cudaFreeAsync(device_metadata, stream);
    if (!pinned.empty())
    {
        const int deferred = gpu_defer_pinned_frees(out->ctx, output.device, stream, pinned.data(), pinned.size());
        if (status == 0) status = deferred;
    }
    return status;
}

extern "C" int gpu_matrix_centered_rebase(GpuMatrix *out, const GpuMatrix *source)
{
    if (!out || !source || !out->ctx || !source->ctx ||
        out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
        source->level != 0 || source->ctx->moduli.size() != 1 || out->level < 0 ||
        source->rows != out->rows || source->cols != out->cols ||
        source->format != GPU_POLY_FORMAT_COEFF || out->format != GPU_POLY_FORMAT_COEFF ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("centered rebase requires colocated single-limb coefficients and matching execution");
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (target_count > kCrtMaxLimbs || target_count > out->ctx->moduli.size())
        return set_error("invalid centered rebase destination basis");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count != 1 || output.limb_count < target_count)
        return set_error("centered rebase requires colocated device descriptors");
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
    status = matrix_wait_limb_stream(source, source->ctx->limb_gpu_ids[0], output.device, stream, false, true);
    if (status != 0) return status;
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, out->ctx->limb_gpu_ids[limb], output.device, stream);
        if (status != 0) return status;
    }
    CrtOutputMetadata metadata{};
    metadata.descriptors = output.device_descriptors;
    metadata.limb_count = target_count;
    std::copy_n(out->ctx->moduli.begin(), target_count, metadata.moduli);
    centered_rebase_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        input.device_descriptors, metadata, source->ctx->moduli[0],
        coefficient_count, static_cast<size_t>(out->ctx->N));
    error = cudaGetLastError();
    if (error != cudaSuccess) status = set_error(error);
    // The temporary INTT source can be dropped immediately after this call.
    const int tracked = matrix_track_limb_consumer_readonly(source,
        source->ctx->limb_gpu_ids[0], output.device, stream);
    if (status == 0) status = tracked;
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
        out->format != GPU_POLY_FORMAT_COEFF || out->shared_limb_buffers.size() != 1)
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
        for (size_t row = 0; row < count; ++row)
            for (size_t column = 0; column < count; ++column)
                entry.garner[row * kCrtMaxLimbs + column] =
                    source->ctx->garner_inverse_table[row * source->ctx->moduli.size() + column];
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

    CrtLevelMetadata *device_levels = nullptr;
    CrtOutputMetadata *device_output = nullptr;
    std::vector<void *> pinned;
    status = crt_alloc_and_copy_async(&device_levels, metadata, stream, &pinned);
    if (status == 0) status = crt_alloc_and_copy_async(&device_output, output_metadata, stream, &pinned);
    if (status == 0)
    {
        crt_recompose_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            device_levels, device_output, level_count, count, static_cast<size_t>(out->ctx->N));
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
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
    if (device_levels) cudaFreeAsync(device_levels, stream);
    if (device_output) cudaFreeAsync(device_output, stream);
    if (!pinned.empty())
    {
        const int released = gpu_defer_pinned_frees(out->ctx, target.device, stream,
            pinned.data(), pinned.size());
        if (status == 0) status = released;
    }
    return status;
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
                mul_mod_u64(plan.plaintext_modulus % modulus, sum, modulus), modulus),
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
