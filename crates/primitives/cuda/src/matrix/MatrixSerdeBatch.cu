namespace
{
    __global__ void serde_reconstruct_rns_batch_to_words_kernel(
        const uint8_t *const *matrix_limb_ptrs,
        const size_t *limb_strides,
        const uint8_t *limb_coeff_bytes,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        int inverse_stride,
        int limb_count,
        size_t coefficients_per_matrix,
        size_t total_coefficients,
        size_t n,
        int words_per_coeff,
        uint64_t *coeff_words_out,
        int *overflow_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_idx = idx / coefficients_per_matrix;
        const size_t local_idx = idx % coefficients_per_matrix;
        const size_t poly_idx = local_idx / n;
        const size_t coeff_idx = local_idx % n;
        const size_t pointer_base = matrix_idx * static_cast<size_t>(limb_count);

        uint64_t mixed_digits[kMaxRnsLimbs];
        uint64_t coeff_words[kMaxCoeffWords];
        for (int i = 0; i < limb_count; ++i)
        {
            const size_t limb = static_cast<size_t>(i);
            mixed_digits[i] = matrix_load_limb_u64(
                                  matrix_limb_ptrs[pointer_base + limb],
                                  poly_idx,
                                  coeff_idx,
                                  limb_strides[limb],
                                  limb_coeff_bytes[limb]) %
                              moduli[i];
        }
        const size_t inverse_stride_size = static_cast<size_t>(inverse_stride);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[i];
            uint64_t t = mixed_digits[i];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t xj_mod_qi = mixed_digits[j] % qi;
                const uint64_t diff =
                    t >= xj_mod_qi
                        ? t - xj_mod_qi
                        : static_cast<uint64_t>(
                              static_cast<unsigned __int128>(t) + qi - xj_mod_qi);
                t = serde_mul_mod_u64_device(
                    diff,
                    garner_inverses[static_cast<size_t>(j) * inverse_stride_size +
                                    static_cast<size_t>(i)],
                    qi);
            }
            mixed_digits[i] = t;
        }
        for (int word = 0; word < words_per_coeff; ++word)
        {
            coeff_words[word] = 0;
        }
        for (int i = limb_count - 1; i >= 0; --i)
        {
            uint64_t carry = mixed_digits[i];
            for (int word = 0; word < words_per_coeff; ++word)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(coeff_words[word]) * moduli[i] + carry;
                coeff_words[word] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
            if (carry != 0)
            {
                atomicExch(overflow_out, 1);
            }
        }
        uint64_t *dst = coeff_words_out + idx * static_cast<size_t>(words_per_coeff);
        for (int word = 0; word < words_per_coeff; ++word)
        {
            dst[word] = coeff_words[word];
        }
    }

    __global__ void serde_center_coeff_words_batch_kernel(
        uint64_t *coeff_words,
        size_t coefficients_per_matrix,
        size_t total_coefficients,
        int words_per_coeff,
        const uint64_t *modulus_words,
        const uint64_t *half_modulus_words,
        uint8_t *sign_bits_out,
        unsigned int *max_abs_bits_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_idx = idx / coefficients_per_matrix;
        uint64_t *coeff = coeff_words + idx * static_cast<size_t>(words_per_coeff);
        const bool negative =
            serde_compare_words_desc_device(coeff, half_modulus_words, words_per_coeff) > 0;
        if (negative)
        {
            uint64_t borrow = 0;
            for (int word = 0; word < words_per_coeff; ++word)
            {
                const unsigned __int128 minuend = modulus_words[word];
                const unsigned __int128 subtrahend =
                    static_cast<unsigned __int128>(coeff[word]) + borrow;
                if (minuend >= subtrahend)
                {
                    coeff[word] = static_cast<uint64_t>(minuend - subtrahend);
                    borrow = 0;
                }
                else
                {
                    coeff[word] = static_cast<uint64_t>(
                        minuend + (static_cast<unsigned __int128>(1) << 64) - subtrahend);
                    borrow = 1;
                }
            }
        }
        sign_bits_out[idx] = static_cast<uint8_t>(negative);
        uint32_t width = 0;
        for (int word = words_per_coeff - 1; word >= 0; --word)
        {
            if (coeff[word] != 0)
            {
                width = static_cast<uint32_t>(word) * 64u +
                        serde_bit_width_u64_device(coeff[word]);
                break;
            }
        }
        atomicMax(&max_abs_bits_out[matrix_idx], width);
    }

    __global__ void serde_pack_centered_coeff_words_batch_kernel(
        const uint64_t *centered_abs_words,
        const uint8_t *sign_bits,
        size_t coefficients_per_matrix,
        int words_per_coeff,
        const uint32_t *signed_widths,
        const size_t *payload_offsets,
        uint8_t *payload_out)
    {
        const size_t matrix_idx = static_cast<size_t>(blockIdx.y);
        const size_t byte_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t payload_len = payload_offsets[matrix_idx + 1] - payload_offsets[matrix_idx];
        if (byte_idx >= payload_len)
        {
            return;
        }
        const uint32_t bit_width = signed_widths[matrix_idx];
        if (bit_width == 0)
        {
            return;
        }
        const size_t total_bits = coefficients_per_matrix * static_cast<size_t>(bit_width);
        const size_t base_bit = byte_idx * 8;
        const size_t coefficient_base = matrix_idx * coefficients_per_matrix;
        uint8_t packed = 0;
        for (size_t bit = 0; bit < 8; ++bit)
        {
            const size_t bit_idx = base_bit + bit;
            if (bit_idx >= total_bits)
            {
                break;
            }
            const size_t local_coefficient = bit_idx / static_cast<size_t>(bit_width);
            const size_t coefficient_bit = bit_idx % static_cast<size_t>(bit_width);
            const size_t coefficient = coefficient_base + local_coefficient;
            uint8_t value = 0;
            if (coefficient_bit == static_cast<size_t>(bit_width - 1))
            {
                value = sign_bits[coefficient] & 1u;
            }
            else
            {
                const size_t word = coefficient_bit / 64;
                const uint32_t bit_in_word = static_cast<uint32_t>(coefficient_bit % 64);
                if (word < static_cast<size_t>(words_per_coeff))
                {
                    value = static_cast<uint8_t>(
                        (centered_abs_words[
                             coefficient * static_cast<size_t>(words_per_coeff) + word] >>
                         bit_in_word) &
                        1u);
                }
            }
            packed |= static_cast<uint8_t>(value << bit);
        }
        payload_out[payload_offsets[matrix_idx] + byte_idx] = packed;
    }
}

extern "C" int gpu_matrix_store_compact_bytes_batch(
    GpuMatrix *const *matrices,
    size_t matrix_count,
    uint8_t *const *payload_outputs,
    const size_t *payload_capacities,
    uint16_t *out_max_coeff_bits,
    uint16_t *out_bytes_per_coeff,
    size_t *out_payload_lengths)
{
    if (!matrices || matrix_count == 0 || !payload_outputs || !payload_capacities ||
        !out_max_coeff_bits || !out_bytes_per_coeff || !out_payload_lengths)
    {
        return set_error("invalid compact serialization batch arguments");
    }
    GpuMatrix *first = matrices[0];
    if (!first || !first->ctx || first->format != GPU_POLY_FORMAT_COEFF || first->level < 0)
    {
        return set_error("compact serialization batch requires coefficient matrices");
    }
    const size_t limb_count = static_cast<size_t>(first->level + 1);
    const size_t poly_count = first->rows * first->cols;
    const size_t n = static_cast<size_t>(first->ctx->N);
    size_t coefficients_per_matrix = 0;
    size_t total_coefficients = 0;
    if (limb_count == 0 || limb_count > static_cast<size_t>(kMaxRnsLimbs) ||
        !serde_checked_mul_size(poly_count, n, &coefficients_per_matrix) ||
        !serde_checked_mul_size(coefficients_per_matrix, matrix_count, &total_coefficients))
    {
        return set_error("invalid compact serialization batch shape");
    }
    size_t total_bits_upper = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        total_bits_upper += bit_width_u64(first->ctx->moduli[limb]);
    }
    const size_t words_per_coeff = std::max<size_t>(1, (total_bits_upper + 63) / 64);
    if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
    {
        return set_error("compact serialization batch coefficient width is unsupported");
    }
    const auto &limb_map = first->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("invalid compact serialization batch limb map");
    }

    std::vector<const uint8_t *> pointers(matrix_count * limb_count);
    std::vector<size_t> strides(limb_count);
    std::vector<uint8_t> coefficient_bytes(limb_count);
    int device = -1;
    cudaStream_t stream = nullptr;
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        GpuMatrix *matrix = matrices[matrix_idx];
        if (!matrix || matrix->ctx != first->ctx || matrix->rows != first->rows ||
            matrix->cols != first->cols || matrix->level != first->level ||
            matrix->format != GPU_POLY_FORMAT_COEFF)
        {
            return set_error("compact serialization batch must be homogeneous");
        }
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 limb_id = limb_map[limb];
            const uint8_t *pointer = matrix_limb_ptr_by_id(matrix, 0, limb_id);
            int limb_device = -1;
            cudaStream_t limb_stream = nullptr;
            size_t stride = 0;
            uint8_t bytes = 0;
            if (!pointer || matrix_limb_device(matrix, limb_id, &limb_device) != 0 ||
                matrix_limb_stream(matrix, limb_id, &limb_stream) != 0 ||
                !matrix_limb_metadata_by_id(matrix, limb_id, &stride, &bytes))
            {
                return set_error("invalid compact serialization batch limb");
            }
            if (device < 0)
            {
                device = limb_device;
                stream = limb_stream;
            }
            else if (device != limb_device)
            {
                return set_error("compact serialization batch requires one placement");
            }
            if (matrix_idx == 0)
            {
                strides[limb] = stride;
                coefficient_bytes[limb] = bytes;
            }
            else if (strides[limb] != stride || coefficient_bytes[limb] != bytes)
            {
                return set_error("compact serialization batch limb layout differs");
            }
            pointers[matrix_idx * limb_count + limb] = pointer;
        }
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess || !stream)
    {
        return set_error(error != cudaSuccess ? error : cudaErrorInvalidResourceHandle);
    }
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const int status = matrix_wait_limb_stream(matrices[matrix_idx], limb_map[limb], device, stream);
            if (status != 0)
            {
                return status;
            }
        }
    }

    std::vector<uint64_t> moduli(first->ctx->moduli.begin(), first->ctx->moduli.begin() + limb_count);
    std::vector<uint64_t> modulus_words;
    if (!serde_compute_modulus_words_le(moduli, &modulus_words))
    {
        return set_error("failed to compute compact serialization batch modulus");
    }
    std::vector<uint64_t> half_modulus_words = modulus_words;
    serde_shift_words_right_one_le(&half_modulus_words);
    modulus_words.resize(words_per_coeff, 0);
    half_modulus_words.resize(words_per_coeff, 0);

    const uint8_t **d_pointers = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_coefficient_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_inverses = nullptr;
    uint64_t *d_modulus_words = nullptr;
    uint64_t *d_half_modulus_words = nullptr;
    uint64_t *d_words = nullptr;
    uint8_t *d_signs = nullptr;
    unsigned int *d_max_bits = nullptr;
    int *d_overflow = nullptr;
    uint32_t *d_signed_widths = nullptr;
    size_t *d_offsets = nullptr;
    uint8_t *d_payload = nullptr;
    unsigned int *host_max_bits = nullptr;
    uint8_t *host_payload = nullptr;
    auto release = [&]() {
        if (d_payload) cudaFreeAsync(d_payload, stream);
        if (d_offsets) cudaFreeAsync(d_offsets, stream);
        if (d_signed_widths) cudaFreeAsync(d_signed_widths, stream);
        if (d_overflow) cudaFreeAsync(d_overflow, stream);
        if (d_max_bits) cudaFreeAsync(d_max_bits, stream);
        if (d_signs) cudaFreeAsync(d_signs, stream);
        if (d_words) cudaFreeAsync(d_words, stream);
        if (d_half_modulus_words) cudaFreeAsync(d_half_modulus_words, stream);
        if (d_modulus_words) cudaFreeAsync(d_modulus_words, stream);
        if (d_inverses) cudaFreeAsync(d_inverses, stream);
        if (d_moduli) cudaFreeAsync(d_moduli, stream);
        if (d_coefficient_bytes) cudaFreeAsync(d_coefficient_bytes, stream);
        if (d_strides) cudaFreeAsync(d_strides, stream);
        if (d_pointers) cudaFreeAsync(d_pointers, stream);
        if (host_payload) cudaFreeHost(host_payload);
        if (host_max_bits) cudaFreeHost(host_max_bits);
    };
    auto allocate = [&](auto **pointer, size_t bytes) {
        return cudaMallocAsync(reinterpret_cast<void **>(pointer), bytes, stream);
    };
    const size_t inverse_count = first->ctx->garner_inverse_table.size();
    const size_t word_count = total_coefficients * words_per_coeff;
    error = allocate(&d_pointers, pointers.size() * sizeof(uint8_t *));
    if (error == cudaSuccess) error = allocate(&d_strides, strides.size() * sizeof(size_t));
    if (error == cudaSuccess) error = allocate(&d_coefficient_bytes, coefficient_bytes.size());
    if (error == cudaSuccess) error = allocate(&d_moduli, moduli.size() * sizeof(uint64_t));
    if (error == cudaSuccess) error = allocate(&d_inverses, inverse_count * sizeof(uint64_t));
    if (error == cudaSuccess) error = allocate(&d_modulus_words, words_per_coeff * sizeof(uint64_t));
    if (error == cudaSuccess) error = allocate(&d_half_modulus_words, words_per_coeff * sizeof(uint64_t));
    if (error == cudaSuccess) error = allocate(&d_words, word_count * sizeof(uint64_t));
    if (error == cudaSuccess) error = allocate(&d_signs, total_coefficients);
    if (error == cudaSuccess) error = allocate(&d_max_bits, matrix_count * sizeof(unsigned int));
    if (error == cudaSuccess) error = allocate(&d_overflow, sizeof(int));
    if (error == cudaSuccess) error = cudaHostAlloc(&host_max_bits, matrix_count * sizeof(unsigned int), cudaHostAllocDefault);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    auto copy = [&](void *dst, const void *src, size_t bytes) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    };
    error = copy(d_pointers, pointers.data(), pointers.size() * sizeof(uint8_t *));
    if (error == cudaSuccess) error = copy(d_strides, strides.data(), strides.size() * sizeof(size_t));
    if (error == cudaSuccess) error = copy(d_coefficient_bytes, coefficient_bytes.data(), coefficient_bytes.size());
    if (error == cudaSuccess) error = copy(d_moduli, moduli.data(), moduli.size() * sizeof(uint64_t));
    if (error == cudaSuccess) error = copy(d_inverses, first->ctx->garner_inverse_table.data(), inverse_count * sizeof(uint64_t));
    if (error == cudaSuccess) error = copy(d_modulus_words, modulus_words.data(), words_per_coeff * sizeof(uint64_t));
    if (error == cudaSuccess) error = copy(d_half_modulus_words, half_modulus_words.data(), words_per_coeff * sizeof(uint64_t));
    if (error == cudaSuccess) error = cudaMemsetAsync(d_max_bits, 0, matrix_count * sizeof(unsigned int), stream);
    if (error == cudaSuccess) error = cudaMemsetAsync(d_overflow, 0, sizeof(int), stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    serde_reconstruct_rns_batch_to_words_kernel<<<blocks, threads, 0, stream>>>(
        d_pointers, d_strides, d_coefficient_bytes, d_moduli, d_inverses,
        static_cast<int>(first->ctx->moduli.size()), static_cast<int>(limb_count),
        coefficients_per_matrix, total_coefficients, n, static_cast<int>(words_per_coeff),
        d_words, d_overflow);
    serde_center_coeff_words_batch_kernel<<<blocks, threads, 0, stream>>>(
        d_words, coefficients_per_matrix, total_coefficients, static_cast<int>(words_per_coeff),
        d_modulus_words, d_half_modulus_words, d_signs, d_max_bits);
    error = cudaGetLastError();
    int host_overflow = 0;
    if (error == cudaSuccess) error = cudaMemcpyAsync(host_max_bits, d_max_bits, matrix_count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(&host_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess || host_overflow != 0)
    {
        release();
        return error != cudaSuccess ? set_error(error) : set_error("compact serialization batch overflow");
    }

    std::vector<uint32_t> signed_widths(matrix_count);
    std::vector<size_t> offsets(matrix_count + 1, 0);
    size_t max_payload = 0;
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        const uint32_t signed_width = host_max_bits[matrix_idx] == 0 ? 0 : host_max_bits[matrix_idx] + 1;
        size_t payload_len = 0;
        if (signed_width > std::numeric_limits<uint16_t>::max() ||
            !serde_compute_payload_len(coefficients_per_matrix, signed_width, &payload_len) ||
            payload_len > payload_capacities[matrix_idx])
        {
            release();
            return set_error("compact serialization batch output capacity is insufficient");
        }
        signed_widths[matrix_idx] = signed_width;
        offsets[matrix_idx + 1] = offsets[matrix_idx] + payload_len;
        max_payload = std::max(max_payload, payload_len);
        out_max_coeff_bits[matrix_idx] = static_cast<uint16_t>(signed_width);
        out_bytes_per_coeff[matrix_idx] = static_cast<uint16_t>((signed_width + 7) / 8);
        out_payload_lengths[matrix_idx] = payload_len;
    }
    const size_t total_payload = offsets.back();
    if (total_payload > 0)
    {
        error = allocate(&d_signed_widths, matrix_count * sizeof(uint32_t));
        if (error == cudaSuccess) error = allocate(&d_offsets, offsets.size() * sizeof(size_t));
        if (error == cudaSuccess) error = allocate(&d_payload, total_payload);
        if (error == cudaSuccess) error = cudaHostAlloc(&host_payload, total_payload, cudaHostAllocDefault);
        if (error == cudaSuccess) error = copy(d_signed_widths, signed_widths.data(), matrix_count * sizeof(uint32_t));
        if (error == cudaSuccess) error = copy(d_offsets, offsets.data(), offsets.size() * sizeof(size_t));
        if (error != cudaSuccess)
        {
            release();
            return set_error(error);
        }
        const dim3 grid(
            static_cast<unsigned int>((max_payload + threads - 1) / threads),
            static_cast<unsigned int>(matrix_count));
        serde_pack_centered_coeff_words_batch_kernel<<<grid, threads, 0, stream>>>(
            d_words, d_signs, coefficients_per_matrix, static_cast<int>(words_per_coeff),
            d_signed_widths, d_offsets, d_payload);
        error = cudaGetLastError();
        if (error == cudaSuccess) error = cudaMemcpyAsync(host_payload, d_payload, total_payload, cudaMemcpyDeviceToHost, stream);
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
        {
            release();
            return set_error(error);
        }
        for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
        {
            const size_t length = out_payload_lengths[matrix_idx];
            if (length > 0)
            {
                std::memcpy(payload_outputs[matrix_idx], host_payload + offsets[matrix_idx], length);
            }
        }
    }
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const int status = matrix_track_limb_consumer(matrices[matrix_idx], limb_map[limb], device, stream);
            if (status != 0)
            {
                release();
                return status;
            }
        }
    }
    release();
    return 0;
}
