namespace
{
    constexpr int kCrtMaxLimbs = 64;
    constexpr int kCrtMaxWords = 64;

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

    __global__ void crt_recompose_kernel(
        const uint8_t *const *source_bases,
        const size_t *source_strides,
        const uint8_t *source_coeff_bytes,
        uint8_t *const *output_bases,
        const size_t *output_strides,
        const uint8_t *output_coeff_bytes,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        const uint64_t *modulus_words,
        const uint64_t *plaintext_moduli,
        const uint64_t *reconstruction_residues,
        size_t level_count,
        size_t limb_count,
        size_t poly_count,
        size_t ring_dimension,
        int word_count)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t coefficient_count = poly_count * ring_dimension;
        if (index >= coefficient_count)
        {
            return;
        }
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t accumulated[kCrtMaxLimbs];
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            accumulated[limb] = 0;
        }

        for (size_t level = 0; level < level_count; ++level)
        {
            uint64_t mixed_digits[kCrtMaxLimbs];
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const size_t metadata = level * limb_count + limb;
                mixed_digits[limb] = matrix_load_limb_u64(
                    source_bases[metadata],
                    poly,
                    coefficient,
                    source_strides[metadata],
                    source_coeff_bytes[metadata]) % moduli[limb];
            }
            for (size_t limb = 1; limb < limb_count; ++limb)
            {
                const uint64_t modulus = moduli[limb];
                uint64_t digit = mixed_digits[limb];
                for (size_t previous = 0; previous < limb; ++previous)
                {
                    const uint64_t previous_digit = mixed_digits[previous] % modulus;
                    const uint64_t difference = digit >= previous_digit
                                                    ? digit - previous_digit
                                                    : modulus - (previous_digit - digit);
                    digit = mul_mod_u64(
                        difference,
                        garner_inverses[previous * limb_count + limb],
                        modulus);
                }
                mixed_digits[limb] = digit;
            }

            uint64_t value_words[kCrtMaxWords + 1];
            for (int word = 0; word <= word_count; ++word)
            {
                value_words[word] = 0;
            }
            for (int limb = static_cast<int>(limb_count) - 1; limb >= 0; --limb)
            {
                uint64_t carry = mixed_digits[limb];
                for (int word = 0; word < word_count; ++word)
                {
                    const unsigned __int128 term =
                        static_cast<unsigned __int128>(value_words[word]) * moduli[limb] + carry;
                    value_words[word] = static_cast<uint64_t>(term);
                    carry = static_cast<uint64_t>(term >> 64);
                }
            }

            const uint64_t rounded = crt_rounded_scale(
                value_words,
                modulus_words,
                word_count,
                plaintext_moduli[level]);
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const uint64_t modulus = moduli[limb];
                const uint64_t contribution = mul_mod_u64(
                    rounded % modulus,
                    reconstruction_residues[level * limb_count + limb],
                    modulus);
                accumulated[limb] = add_mod_u64(accumulated[limb], contribution, modulus);
            }
        }

        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            matrix_store_limb_u64(
                output_bases[limb],
                poly,
                coefficient,
                output_strides[limb],
                output_coeff_bytes[limb],
                accumulated[limb]);
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

extern "C" int gpu_matrix_crt_recompose(
    GpuMatrix *out,
    const GpuMatrix *const *levels,
    size_t level_count,
    const uint64_t *plaintext_moduli,
    const uint64_t *reconstruction_residues,
    size_t reconstruction_stride)
{
    if (!out || !levels || !plaintext_moduli || !reconstruction_residues || level_count == 0)
    {
        return set_error("invalid gpu_matrix_crt_recompose arguments");
    }
    const GpuMatrix *first = levels[0];
    if (!first || out->ctx != first->ctx || out->rows != 1 || first->rows != 1 ||
        out->cols != first->cols || out->level != first->level ||
        out->format != GPU_POLY_FORMAT_COEFF || first->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("matrix layout mismatch in gpu_matrix_crt_recompose");
    }
    const size_t limb_count = static_cast<size_t>(out->level + 1);
    if (limb_count == 0 || limb_count > kCrtMaxLimbs || reconstruction_stride != limb_count)
    {
        return set_error("unsupported limb layout in gpu_matrix_crt_recompose");
    }
    for (size_t level = 0; level < level_count; ++level)
    {
        if (!levels[level] || levels[level]->ctx != out->ctx || levels[level]->rows != 1 ||
            levels[level]->cols != out->cols || levels[level]->level != out->level ||
            levels[level]->format != GPU_POLY_FORMAT_COEFF || plaintext_moduli[level] == 0)
        {
            return set_error("input layout mismatch in gpu_matrix_crt_recompose");
        }
    }

    std::vector<uint64_t> modulus_words;
    std::vector<uint64_t> active_moduli(out->ctx->moduli.begin(), out->ctx->moduli.begin() + limb_count);
    if (!serde_compute_modulus_words_le(active_moduli, &modulus_words) ||
        modulus_words.empty() || modulus_words.size() > kCrtMaxWords)
    {
        return set_error("unsupported CRT modulus size in gpu_matrix_crt_recompose");
    }
    const int word_count = static_cast<int>(modulus_words.size());
    modulus_words.resize(static_cast<size_t>(word_count), 0);

    const auto &limb_map = out->ctx->limb_gpu_ids;
    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    std::vector<dim3> limb_ids(limb_count);
    std::vector<uint8_t *> output_bases(limb_count, nullptr);
    std::vector<size_t> output_strides(limb_count, 0);
    std::vector<uint8_t> output_coeff_bytes(limb_count, 0);
    std::vector<const uint8_t *> source_bases(level_count * limb_count, nullptr);
    std::vector<size_t> source_strides(level_count * limb_count, 0);
    std::vector<uint8_t> source_coeff_bytes(level_count * limb_count, 0);

    int status = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        limb_ids[limb] = limb_id;
        int device = -1;
        cudaStream_t stream = nullptr;
        status = matrix_limb_device(out, limb_id, &device);
        if (status != 0) return status;
        status = matrix_limb_stream(out, limb_id, &stream);
        if (status != 0) return status;
        if (limb == 0)
        {
            dispatch_device = device;
            dispatch_stream = stream;
        }
        else if (device != dispatch_device)
        {
            return set_error("gpu_matrix_crt_recompose requires a single-device matrix");
        }
        output_bases[limb] = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!output_bases[limb] ||
            !matrix_limb_metadata_by_id(out, limb_id, &output_strides[limb], &output_coeff_bytes[limb]))
        {
            return set_error("invalid output metadata in gpu_matrix_crt_recompose");
        }
        for (size_t level = 0; level < level_count; ++level)
        {
            int source_device = -1;
            status = matrix_limb_device(levels[level], limb_id, &source_device);
            if (status != 0) return status;
            if (source_device != dispatch_device)
            {
                return set_error("gpu_matrix_crt_recompose requires colocated input matrices");
            }
            const size_t metadata = level * limb_count + limb;
            source_bases[metadata] = matrix_limb_ptr_by_id(levels[level], 0, limb_id);
            if (!source_bases[metadata] || !matrix_limb_metadata_by_id(
                    levels[level], limb_id, &source_strides[metadata], &source_coeff_bytes[metadata]))
            {
                return set_error("invalid input metadata in gpu_matrix_crt_recompose");
            }
        }
    }
    if (dispatch_device < 0 || !dispatch_stream)
    {
        return set_error("invalid dispatch stream in gpu_matrix_crt_recompose");
    }
    cudaError_t error = cudaSetDevice(dispatch_device);
    if (error != cudaSuccess) return set_error(error);

    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        status = matrix_wait_limb_stream(out, limb_ids[limb], dispatch_device, dispatch_stream);
        if (status != 0) return status;
        for (size_t level = 0; level < level_count; ++level)
        {
            status = matrix_wait_limb_stream(
                levels[level], limb_ids[limb], dispatch_device, dispatch_stream);
            if (status != 0) return status;
        }
    }

    const size_t inverse_count = limb_count * limb_count;
    std::vector<uint64_t> garner_inverses(inverse_count, 0);
    for (size_t row = 0; row < limb_count; ++row)
    {
        for (size_t column = 0; column < limb_count; ++column)
        {
            garner_inverses[row * limb_count + column] =
                out->ctx->garner_inverse_table[row * out->ctx->moduli.size() + column];
        }
    }
    std::vector<uint64_t> plaintext(plaintext_moduli, plaintext_moduli + level_count);
    std::vector<uint64_t> reconstruction(
        reconstruction_residues,
        reconstruction_residues + level_count * limb_count);

    const uint8_t **d_source_bases = nullptr;
    size_t *d_source_strides = nullptr;
    uint8_t *d_source_coeff_bytes = nullptr;
    uint8_t **d_output_bases = nullptr;
    size_t *d_output_strides = nullptr;
    uint8_t *d_output_coeff_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_garner = nullptr;
    uint64_t *d_modulus_words = nullptr;
    uint64_t *d_plaintext = nullptr;
    uint64_t *d_reconstruction = nullptr;
    std::vector<void *> pinned_metadata;
    pinned_metadata.reserve(11);
    auto cleanup = [&]()
    {
        cudaSetDevice(dispatch_device);
        if (d_reconstruction) cudaFreeAsync(d_reconstruction, dispatch_stream);
        if (d_plaintext) cudaFreeAsync(d_plaintext, dispatch_stream);
        if (d_modulus_words) cudaFreeAsync(d_modulus_words, dispatch_stream);
        if (d_garner) cudaFreeAsync(d_garner, dispatch_stream);
        if (d_moduli) cudaFreeAsync(d_moduli, dispatch_stream);
        if (d_output_coeff_bytes) cudaFreeAsync(d_output_coeff_bytes, dispatch_stream);
        if (d_output_strides) cudaFreeAsync(d_output_strides, dispatch_stream);
        if (d_output_bases) cudaFreeAsync(d_output_bases, dispatch_stream);
        if (d_source_coeff_bytes) cudaFreeAsync(d_source_coeff_bytes, dispatch_stream);
        if (d_source_strides) cudaFreeAsync(d_source_strides, dispatch_stream);
        if (d_source_bases) cudaFreeAsync(const_cast<uint8_t **>(d_source_bases), dispatch_stream);
        if (!pinned_metadata.empty())
        {
            (void)gpu_defer_pinned_frees(
                out->ctx,
                dispatch_device,
                dispatch_stream,
                pinned_metadata.data(),
                pinned_metadata.size());
            pinned_metadata.clear();
        }
    };
#define CRT_COPY(device, host) \
    do { status = crt_alloc_and_copy_async( \
             &(device), (host), dispatch_stream, &pinned_metadata); \
         if (status != 0) { cleanup(); return status; } } while (false)
    CRT_COPY(d_source_bases, source_bases);
    CRT_COPY(d_source_strides, source_strides);
    CRT_COPY(d_source_coeff_bytes, source_coeff_bytes);
    CRT_COPY(d_output_bases, output_bases);
    CRT_COPY(d_output_strides, output_strides);
    CRT_COPY(d_output_coeff_bytes, output_coeff_bytes);
    CRT_COPY(d_moduli, active_moduli);
    CRT_COPY(d_garner, garner_inverses);
    CRT_COPY(d_modulus_words, modulus_words);
    CRT_COPY(d_plaintext, plaintext);
    CRT_COPY(d_reconstruction, reconstruction);
#undef CRT_COPY

    const size_t coefficient_count = out->cols * static_cast<size_t>(out->ctx->N);
    const int threads = 128;
    const int blocks = static_cast<int>((coefficient_count + threads - 1) / threads);
    crt_recompose_kernel<<<blocks, threads, 0, dispatch_stream>>>(
        d_source_bases,
        d_source_strides,
        d_source_coeff_bytes,
        d_output_bases,
        d_output_strides,
        d_output_coeff_bytes,
        d_moduli,
        d_garner,
        d_modulus_words,
        d_plaintext,
        d_reconstruction,
        level_count,
        limb_count,
        out->cols,
        static_cast<size_t>(out->ctx->N),
        word_count);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cleanup();
        return set_error(error);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        for (size_t level = 0; level < level_count; ++level)
        {
            status = matrix_track_limb_consumer(
                levels[level], limb_ids[limb], dispatch_device, dispatch_stream);
            if (status != 0) { cleanup(); return status; }
        }
        status = matrix_record_limb_write(out, limb_ids[limb], dispatch_stream);
        if (status != 0) { cleanup(); return status; }
    }
    cleanup();
    return 0;
}
