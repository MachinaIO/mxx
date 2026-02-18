namespace
{
    constexpr int kMaxRnsLimbs = 64;
    constexpr int kMaxCoeffWords = 64;

    struct SerdeStreamRef
    {
        int device;
        cudaStream_t stream;
    };

    bool serde_checked_mul_size(size_t a, size_t b, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (a != 0 && b > static_cast<size_t>(-1) / a)
        {
            return false;
        }
        *out = a * b;
        return true;
    }

    void serde_append_unique_stream(std::vector<SerdeStreamRef> &streams, int device, cudaStream_t stream)
    {
        if (!stream)
        {
            return;
        }
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                return;
            }
        }
        streams.push_back(SerdeStreamRef{device, stream});
    }

    void serde_destroy_event_set(GpuEventSet *events)
    {
        if (!events)
        {
            return;
        }
        for (const auto &entry : events->entries)
        {
            cudaSetDevice(entry.device);
            cudaEventDestroy(entry.event);
        }
        delete events;
    }

    int serde_build_event_set_from_streams(const std::vector<SerdeStreamRef> &streams, GpuEventSet **out_events)
    {
        if (!out_events)
        {
            return set_error("invalid out_events in serde_build_event_set_from_streams");
        }
        *out_events = nullptr;
        if (streams.empty())
        {
            return 0;
        }

        auto *event_set = new GpuEventSet();
        event_set->entries.reserve(streams.size());
        for (const auto &entry : streams)
        {
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }

            cudaEvent_t ev = nullptr;
            err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }
            err = cudaEventRecord(ev, entry.stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(ev);
                serde_destroy_event_set(event_set);
                return set_error(err);
            }
            event_set->entries.push_back(GpuEventSet::Entry{ev, entry.device});
        }

        *out_events = event_set;
        return 0;
    }

    __device__ __forceinline__ uint64_t serde_mul_mod_u64_device(
        uint64_t a,
        uint64_t b,
        uint64_t modulus)
    {
        const unsigned __int128 prod =
            static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % static_cast<unsigned __int128>(modulus));
    }

    __global__ void serde_reconstruct_rns_to_words_kernel(
        const uint64_t *const *limb_ptrs,
        const size_t *limb_strides,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        int inverse_stride,
        int limb_count,
        size_t coeff_count,
        size_t n,
        int words_per_coeff,
        uint64_t *coeff_words_out,
        int *overflow_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= coeff_count)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;

        uint64_t mixed_digits[kMaxRnsLimbs];
        uint64_t coeff_words[kMaxCoeffWords];

        for (int i = 0; i < limb_count; ++i)
        {
            const size_t stride = limb_strides[static_cast<size_t>(i)];
            mixed_digits[i] =
                limb_ptrs[i][poly_idx * stride + coeff_idx] % moduli[i];
        }

        const size_t inverse_stride_sz = static_cast<size_t>(inverse_stride);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[i];
            uint64_t t = mixed_digits[i];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t xj_mod_qi = mixed_digits[j] % qi;
                const uint64_t diff =
                    t >= xj_mod_qi
                        ? (t - xj_mod_qi)
                        : static_cast<uint64_t>(
                              static_cast<unsigned __int128>(t) +
                              static_cast<unsigned __int128>(qi) -
                              static_cast<unsigned __int128>(xj_mod_qi));
                const uint64_t inv =
                    garner_inverses[static_cast<size_t>(j) * inverse_stride_sz +
                                    static_cast<size_t>(i)];
                t = serde_mul_mod_u64_device(diff, inv, qi);
            }
            mixed_digits[i] = t;
        }

        for (int w = 0; w < words_per_coeff; ++w)
        {
            coeff_words[w] = 0;
        }

        for (int i = limb_count - 1; i >= 0; --i)
        {
            const uint64_t qi = moduli[i];
            uint64_t carry = mixed_digits[i];
            for (int w = 0; w < words_per_coeff; ++w)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(coeff_words[w]) *
                        static_cast<unsigned __int128>(qi) +
                    static_cast<unsigned __int128>(carry);
                coeff_words[w] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
            if (carry != 0)
            {
                atomicExch(overflow_out, 1);
            }
        }

        uint64_t *dst = coeff_words_out + idx * static_cast<size_t>(words_per_coeff);
        for (int w = 0; w < words_per_coeff; ++w)
        {
            dst[w] = coeff_words[w];
        }
    }

    __device__ __forceinline__ uint32_t serde_bit_width_u64_device(uint64_t value)
    {
        return value == 0 ? 0u : static_cast<uint32_t>(64 - __clzll(value));
    }

    __global__ void serde_compute_max_bits_from_words_kernel(
        const uint64_t *coeff_words,
        size_t n,
        int words_per_coeff,
        unsigned int *max_bits_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= n)
        {
            return;
        }

        const uint64_t *coeff = coeff_words + idx * static_cast<size_t>(words_per_coeff);
        uint32_t bit_width = 0;
        for (int w = words_per_coeff - 1; w >= 0; --w)
        {
            const uint64_t word = coeff[w];
            if (word != 0)
            {
                bit_width = static_cast<uint32_t>(w) * 64u + serde_bit_width_u64_device(word);
                break;
            }
        }
        atomicMax(max_bits_out, bit_width);
    }

    __global__ void serde_pack_coeff_words_bits_kernel(
        const uint64_t *coeff_words,
        size_t n,
        int words_per_coeff,
        uint32_t bit_width,
        uint8_t *payload_out,
        size_t payload_len)
    {
        const size_t byte_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (byte_idx >= payload_len)
        {
            return;
        }

        const size_t total_bits = n * static_cast<size_t>(bit_width);
        const size_t base_bit = byte_idx * 8;
        uint8_t out = 0;
        for (size_t k = 0; k < 8; ++k)
        {
            const size_t bit_idx = base_bit + k;
            if (bit_idx >= total_bits)
            {
                break;
            }
            const size_t coeff_idx = bit_idx / static_cast<size_t>(bit_width);
            const size_t coeff_bit = bit_idx % static_cast<size_t>(bit_width);
            const size_t word_idx = coeff_bit / 64;
            const uint32_t bit_in_word = static_cast<uint32_t>(coeff_bit % 64);
            const uint64_t word =
                coeff_words[coeff_idx * static_cast<size_t>(words_per_coeff) + word_idx];
            const uint8_t bit = static_cast<uint8_t>((word >> bit_in_word) & uint64_t{1});
            out |= static_cast<uint8_t>(bit << k);
        }
        payload_out[byte_idx] = out;
    }

    __global__ void serde_unpack_packed_coeffs_mod_kernel(
        const uint8_t *payload,
        size_t coeff_count,
        size_t n,
        uint32_t bit_width,
        const uint64_t *moduli,
        int limb_count,
        uint64_t *const *limb_ptrs,
        const size_t *limb_strides)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= coeff_count)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;
        if (bit_width == 0)
        {
            for (int limb = 0; limb < limb_count; ++limb)
            {
                const size_t stride = limb_strides[static_cast<size_t>(limb)];
                uint64_t *base = limb_ptrs[static_cast<size_t>(limb)];
                base[poly_idx * stride + coeff_idx] = 0;
            }
            return;
        }

        const size_t base_bit = idx * static_cast<size_t>(bit_width);
        for (int limb = 0; limb < limb_count; ++limb)
        {
            const uint64_t modulus = moduli[static_cast<size_t>(limb)];
            uint64_t residue = 0;
            for (int b = static_cast<int>(bit_width) - 1; b >= 0; --b)
            {
                const size_t bit_idx = base_bit + static_cast<size_t>(b);
                const uint8_t byte_val = payload[bit_idx / 8];
                const uint8_t bit = static_cast<uint8_t>((byte_val >> (bit_idx % 8)) & 0x1u);
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(residue) * static_cast<unsigned __int128>(2u) +
                    static_cast<unsigned __int128>(bit);
                residue =
                    static_cast<uint64_t>(term % static_cast<unsigned __int128>(modulus));
            }
            const size_t stride = limb_strides[static_cast<size_t>(limb)];
            uint64_t *base = limb_ptrs[static_cast<size_t>(limb)];
            base[poly_idx * stride + coeff_idx] = residue;
        }
    }

    bool serde_get_matrix_shape(
        const GpuMatrix *mat,
        int *out_level,
        size_t *out_n,
        size_t *out_limb_count,
        size_t *out_poly_count)
    {
        if (!mat || !mat->ctx || !out_level || !out_n ||
            !out_limb_count || !out_poly_count)
        {
            return false;
        }
        if (mat->level < 0)
        {
            return false;
        }
        const int N = mat->ctx->N;
        if (N <= 0)
        {
            return false;
        }
        size_t poly_count = 0;
        if (!serde_checked_mul_size(mat->rows, mat->cols, &poly_count))
        {
            return false;
        }
        *out_level = mat->level;
        *out_n = static_cast<size_t>(N);
        *out_limb_count = static_cast<size_t>(mat->level + 1);
        *out_poly_count = poly_count;
        return true;
    }

    bool serde_compute_payload_len(size_t n, uint32_t max_bits, size_t *out_payload_len)
    {
        if (!out_payload_len)
        {
            return false;
        }
        if (max_bits == 0)
        {
            *out_payload_len = 0;
            return true;
        }
        size_t total_bits = 0;
        if (!serde_checked_mul_size(n, static_cast<size_t>(max_bits), &total_bits))
        {
            return false;
        }
        *out_payload_len = (total_bits + static_cast<size_t>(7)) / static_cast<size_t>(8);
        return true;
    }

}

extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_load_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_load_rns_batch");
    }
    if (count > 0 && !bytes)
    {
        return set_error("null bytes in gpu_matrix_load_rns_batch");
    }

    if (mat->level < 0)
    {
        mat->format = target_format;
        return count == 0 ? 0 : set_error("invalid level in gpu_matrix_load_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_load_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_load_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_load_rns_batch");
    }
    if (count == 0)
    {
        mat->format = target_format;
        return 0;
    }

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_load_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t coeff_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint64_t *dst = matrix_limb_ptr_by_id(mat, 0, limb_id);
        if (!dst)
        {
            return set_error("null matrix limb base pointer in gpu_matrix_load_rns_batch");
        }

        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= mat->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_matrix_load_rns_batch");
        }
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const size_t dst_pitch = buffer.words_per_poly * sizeof(uint64_t);
        const uint8_t *src = bytes + limb * coeff_bytes;

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpy2DAsync(
            dst,
            dst_pitch,
            src,
            bytes_per_poly,
            coeff_bytes,
            count,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = matrix_record_limb_write(mat, limb_id, stream);
        if (status != 0)
        {
            return status;
        }
        serde_append_unique_stream(streams, device, stream);
    }

    mat->format = target_format;
    return serde_build_event_set_from_streams(streams, out_events);
}

extern "C" int gpu_matrix_store_rns_batch(
    const GpuMatrix *mat,
    uint8_t *bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_store_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_store_rns_batch");
    }
    if (target_format != mat->format)
    {
        return set_error("format conversion is not supported in gpu_matrix_store_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_store_rns_batch");
    }
    if (count > 0 && !bytes_out)
    {
        return set_error("null bytes_out in gpu_matrix_store_rns_batch");
    }
    if (count == 0)
    {
        return 0;
    }

    if (mat->level < 0)
    {
        return set_error("invalid level in gpu_matrix_store_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_store_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_store_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_store_rns_batch");
    }

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_store_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t coeff_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        const uint64_t *src = matrix_limb_ptr_by_id(mat, 0, limb_id);
        if (!src)
        {
            return set_error("null matrix limb base pointer in gpu_matrix_store_rns_batch");
        }

        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= mat->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_matrix_store_rns_batch");
        }
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const size_t src_pitch = buffer.words_per_poly * sizeof(uint64_t);
        uint8_t *dst = bytes_out + limb * coeff_bytes;

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = matrix_wait_limb_stream(mat, limb_id, device, stream);
        if (status != 0)
        {
            return status;
        }
        err = cudaMemcpy2DAsync(
            dst,
            bytes_per_poly,
            src,
            src_pitch,
            coeff_bytes,
            count,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = matrix_track_limb_consumer(mat, limb_id, device, stream);
        if (status != 0)
        {
            return status;
        }
        serde_append_unique_stream(streams, device, stream);
    }

    return serde_build_event_set_from_streams(streams, out_events);
}

extern "C" int gpu_poly_store_compact_bytes(
    GpuMatrix *poly,
    uint8_t *payload_out,
    size_t payload_capacity,
    uint16_t *out_max_coeff_bits,
    uint16_t *out_bytes_per_coeff,
    size_t *out_payload_len)
{
    if (!poly || !payload_out || !out_max_coeff_bits || !out_bytes_per_coeff || !out_payload_len)
    {
        return set_error("invalid gpu_poly_store_compact_bytes arguments");
    }
    *out_max_coeff_bits = 0;
    *out_bytes_per_coeff = 0;
    *out_payload_len = 0;

    int level = -1;
    size_t n = 0;
    size_t limb_count = 0;
    size_t poly_count = 0;
    if (!serde_get_matrix_shape(poly, &level, &n, &limb_count, &poly_count))
    {
        return set_error("invalid matrix in gpu_poly_store_compact_bytes");
    }
    if (limb_count > poly->ctx->moduli.size())
    {
        return set_error("unexpected modulus count in gpu_poly_store_compact_bytes");
    }
    if (limb_count == 0 || n == 0 || poly_count == 0)
    {
        return 0;
    }
    if (limb_count > static_cast<size_t>(kMaxRnsLimbs))
    {
        return set_error("limb_count exceeds kernel limit in gpu_poly_store_compact_bytes");
    }
    size_t coeff_count = 0;
    if (!serde_checked_mul_size(poly_count, n, &coeff_count))
    {
        return set_error("coeff_count overflow in gpu_poly_store_compact_bytes");
    }

    const int batch = default_batch(poly->ctx);
    if (poly->format == GPU_POLY_FORMAT_EVAL)
    {
        const int status = gpu_matrix_intt_all(poly, batch);
        if (status != 0)
        {
            return status;
        }
    }
    if (poly->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("gpu_poly_store_compact_bytes expects coeff format");
    }

    size_t total_bits_upper = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        total_bits_upper += static_cast<size_t>(bit_width_u64(poly->ctx->moduli[limb]));
    }
    const size_t words_per_coeff =
        std::max<size_t>(
            1,
            (total_bits_upper + static_cast<size_t>(63)) / static_cast<size_t>(64));
    if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
    {
        return set_error("words_per_coeff exceeds kernel limit in gpu_poly_store_compact_bytes");
    }

    auto &limb_map = poly->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_poly_store_compact_bytes");
    }

    std::vector<const uint64_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    int common_device = -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        const uint64_t *limb_ptr = matrix_limb_ptr_by_id(poly, 0, limb_id);
        if (!limb_ptr)
        {
            return set_error("null matrix limb pointer in gpu_poly_store_compact_bytes");
        }
        int device = -1;
        int status = matrix_limb_device(poly, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= poly->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_poly_store_compact_bytes");
        }
        const auto &buffer = poly->shared_limb_buffers[limb_id.x];
        if (common_device < 0)
        {
            common_device = device;
        }
        else if (common_device != device)
        {
            return set_error("gpu_poly_store_compact_bytes requires all limbs on a single GPU");
        }
        limb_ptrs[limb] = limb_ptr;
        limb_strides[limb] = buffer.words_per_poly;
    }

    const size_t inverse_stride = poly->ctx->moduli.size();
    const std::vector<uint64_t> &inverse_table = poly->ctx->garner_inverse_table;
    if (inverse_table.size() != inverse_stride * inverse_stride)
    {
        return set_error("invalid cached inverse table in gpu_poly_store_compact_bytes");
    }
    std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + limb_count);

    if (words_per_coeff > std::numeric_limits<size_t>::max() / coeff_count)
    {
        return set_error("coeff word length overflow in gpu_poly_store_compact_bytes");
    }
    const size_t coeff_word_len = coeff_count * words_per_coeff;

    cudaError_t err = cudaSetDevice(common_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const uint64_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_garner_inv = nullptr;
    uint64_t *d_coeff_words = nullptr;
    int *d_overflow = nullptr;
    unsigned int *d_max_bits = nullptr;
    uint8_t *d_payload = nullptr;
    cudaStream_t work_stream = nullptr;
    auto release = [&]() {
        if (work_stream)
        {
            cudaStreamDestroy(work_stream);
            work_stream = nullptr;
        }
        if (d_payload)
        {
            cudaFree(d_payload);
            d_payload = nullptr;
        }
        if (d_max_bits)
        {
            cudaFree(d_max_bits);
            d_max_bits = nullptr;
        }
        if (d_overflow)
        {
            cudaFree(d_overflow);
            d_overflow = nullptr;
        }
        if (d_coeff_words)
        {
            cudaFree(d_coeff_words);
            d_coeff_words = nullptr;
        }
        if (d_garner_inv)
        {
            cudaFree(d_garner_inv);
            d_garner_inv = nullptr;
        }
        if (d_moduli)
        {
            cudaFree(d_moduli);
            d_moduli = nullptr;
        }
        if (d_limb_strides)
        {
            cudaFree(d_limb_strides);
            d_limb_strides = nullptr;
        }
        if (d_limb_ptrs)
        {
            cudaFree(d_limb_ptrs);
            d_limb_ptrs = nullptr;
        }
    };

    err = cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int wait_status = matrix_wait_limb_stream(
            poly,
            limb_map[limb],
            common_device,
            work_stream);
        if (wait_status != 0)
        {
            release();
            return wait_status;
        }
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_limb_ptrs), limb_count * sizeof(uint64_t *));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_ptrs,
        limb_ptrs.data(),
        limb_count * sizeof(uint64_t *),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_limb_strides), limb_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_strides,
        limb_strides.data(),
        limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_moduli), moduli_subset.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_moduli,
        moduli_subset.data(),
        moduli_subset.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMalloc(
        reinterpret_cast<void **>(&d_garner_inv),
        inverse_table.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_garner_inv,
        inverse_table.data(),
        inverse_table.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMalloc(
        reinterpret_cast<void **>(&d_coeff_words),
        coeff_word_len * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_overflow), sizeof(int));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemsetAsync(d_overflow, 0, sizeof(int), work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_max_bits), sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemsetAsync(d_max_bits, 0, sizeof(unsigned int), work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks =
        static_cast<int>((coeff_count + static_cast<size_t>(threads) - 1) /
                         static_cast<size_t>(threads));
    serde_reconstruct_rns_to_words_kernel<<<blocks, threads, 0, work_stream>>>(
        d_limb_ptrs,
        d_limb_strides,
        d_moduli,
        d_garner_inv,
        static_cast<int>(inverse_stride),
        static_cast<int>(limb_count),
        coeff_count,
        n,
        static_cast<int>(words_per_coeff),
        d_coeff_words,
        d_overflow);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    serde_compute_max_bits_from_words_kernel<<<blocks, threads, 0, work_stream>>>(
        d_coeff_words,
        coeff_count,
        static_cast<int>(words_per_coeff),
        d_max_bits);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    int h_overflow = 0;
    unsigned int h_max_bits = 0;
    err = cudaMemcpyAsync(
        &h_overflow,
        d_overflow,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        &h_max_bits,
        d_max_bits,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaStreamSynchronize(work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    if (h_overflow != 0)
    {
        release();
        return set_error("overflow in gpu_poly_store_compact_bytes");
    }
    if (h_max_bits > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
    {
        release();
        return set_error("max coeff bits exceed u16 range in gpu_poly_store_compact_bytes");
    }
    const unsigned int h_bytes_per_coeff =
        (h_max_bits + static_cast<unsigned int>(7)) / static_cast<unsigned int>(8);
    if (h_bytes_per_coeff > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
    {
        release();
        return set_error("bytes_per_coeff exceed u16 range in gpu_poly_store_compact_bytes");
    }

    size_t payload_len = 0;
    if (!serde_compute_payload_len(coeff_count, h_max_bits, &payload_len))
    {
        release();
        return set_error("payload length overflow in gpu_poly_store_compact_bytes");
    }
    if (payload_len > payload_capacity)
    {
        release();
        return set_error("payload_capacity too small in gpu_poly_store_compact_bytes");
    }

    if (payload_len > 0)
    {
        err = cudaMalloc(reinterpret_cast<void **>(&d_payload), payload_len);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        const int payload_blocks =
            static_cast<int>((payload_len + static_cast<size_t>(threads) - 1) /
                             static_cast<size_t>(threads));
        serde_pack_coeff_words_bits_kernel<<<payload_blocks, threads, 0, work_stream>>>(
            d_coeff_words,
            coeff_count,
            static_cast<int>(words_per_coeff),
            h_max_bits,
            d_payload,
            payload_len);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            payload_out,
            d_payload,
            payload_len,
            cudaMemcpyDeviceToHost,
            work_stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
    }

    err = cudaStreamSynchronize(work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    *out_max_coeff_bits = static_cast<uint16_t>(h_max_bits);
    *out_bytes_per_coeff = static_cast<uint16_t>(h_bytes_per_coeff);
    *out_payload_len = payload_len;
    release();
    return 0;
}

extern "C" int gpu_poly_load_compact_bytes(
    GpuMatrix *poly,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    if (!poly || !poly->ctx)
    {
        return set_error("invalid gpu_poly_load_compact_bytes arguments");
    }

    int level = -1;
    size_t n = 0;
    size_t limb_count = 0;
    size_t poly_count = 0;
    if (!serde_get_matrix_shape(poly, &level, &n, &limb_count, &poly_count))
    {
        return set_error("invalid matrix in gpu_poly_load_compact_bytes");
    }
    if (limb_count > poly->ctx->moduli.size())
    {
        return set_error("unexpected modulus count in gpu_poly_load_compact_bytes");
    }
    if (limb_count == 0 || n == 0 || poly_count == 0)
    {
        poly->format = GPU_POLY_FORMAT_COEFF;
        return 0;
    }
    if (limb_count > static_cast<size_t>(kMaxRnsLimbs))
    {
        return set_error("limb_count exceeds kernel limit in gpu_poly_load_compact_bytes");
    }

    size_t coeff_count = 0;
    if (!serde_checked_mul_size(poly_count, n, &coeff_count))
    {
        return set_error("coeff_count overflow in gpu_poly_load_compact_bytes");
    }

    if (max_coeff_bits == 0)
    {
        if (payload_len != 0)
        {
            return set_error("payload_len must be zero when max_coeff_bits is zero");
        }
    }
    else if (!payload)
    {
        return set_error("null payload in gpu_poly_load_compact_bytes");
    }

    size_t expected_payload_len = 0;
    if (!serde_compute_payload_len(
            coeff_count,
            static_cast<uint32_t>(max_coeff_bits),
            &expected_payload_len))
    {
        return set_error("payload length overflow in gpu_poly_load_compact_bytes");
    }
    if (payload_len != expected_payload_len)
    {
        return set_error("payload length mismatch in gpu_poly_load_compact_bytes");
    }

    auto &limb_map = poly->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_poly_load_compact_bytes");
    }

    std::vector<uint64_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    int common_device = -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint64_t *limb_ptr = matrix_limb_ptr_by_id(poly, 0, limb_id);
        if (!limb_ptr)
        {
            return set_error("null matrix limb pointer in gpu_poly_load_compact_bytes");
        }
        int device = -1;
        int status = matrix_limb_device(poly, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= poly->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_poly_load_compact_bytes");
        }
        const auto &buffer = poly->shared_limb_buffers[limb_id.x];
        if (common_device < 0)
        {
            common_device = device;
        }
        else if (common_device != device)
        {
            return set_error("gpu_poly_load_compact_bytes requires all limbs on a single GPU");
        }
        limb_ptrs[limb] = limb_ptr;
        limb_strides[limb] = buffer.words_per_poly;
    }

    cudaError_t err = cudaSetDevice(common_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    uint8_t *d_payload = nullptr;
    uint64_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint64_t *d_moduli = nullptr;
    cudaStream_t work_stream = nullptr;
    auto release = [&]() {
        if (work_stream)
        {
            cudaStreamDestroy(work_stream);
            work_stream = nullptr;
        }
        if (d_moduli)
        {
            cudaFree(d_moduli);
            d_moduli = nullptr;
        }
        if (d_limb_strides)
        {
            cudaFree(d_limb_strides);
            d_limb_strides = nullptr;
        }
        if (d_limb_ptrs)
        {
            cudaFree(d_limb_ptrs);
            d_limb_ptrs = nullptr;
        }
        if (d_payload)
        {
            cudaFree(d_payload);
            d_payload = nullptr;
        }
    };

    err = cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int wait_status = matrix_wait_limb_stream(
            poly,
            limb_map[limb],
            common_device,
            work_stream);
        if (wait_status != 0)
        {
            release();
            return wait_status;
        }
    }

    if (payload_len > 0)
    {
        err = cudaMalloc(reinterpret_cast<void **>(&d_payload), payload_len);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            d_payload,
            payload,
            payload_len,
            cudaMemcpyHostToDevice,
            work_stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_limb_ptrs), limb_count * sizeof(uint64_t *));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_ptrs,
        limb_ptrs.data(),
        limb_count * sizeof(uint64_t *),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMalloc(reinterpret_cast<void **>(&d_limb_strides), limb_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_strides,
        limb_strides.data(),
        limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + limb_count);
    err = cudaMalloc(reinterpret_cast<void **>(&d_moduli), moduli_subset.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_moduli,
        moduli_subset.data(),
        moduli_subset.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks =
        static_cast<int>((coeff_count + static_cast<size_t>(threads) - 1) /
                         static_cast<size_t>(threads));
    serde_unpack_packed_coeffs_mod_kernel<<<blocks, threads, 0, work_stream>>>(
        d_payload,
        coeff_count,
        n,
        static_cast<uint32_t>(max_coeff_bits),
        d_moduli,
        static_cast<int>(limb_count),
        d_limb_ptrs,
        d_limb_strides);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int status = matrix_record_limb_write(poly, limb_map[limb], work_stream);
        if (status != 0)
        {
            release();
            return status;
        }
    }

    release();
    poly->format = GPU_POLY_FORMAT_COEFF;
    return 0;
}

extern "C" int gpu_matrix_store_compact_bytes(
    GpuMatrix *mat,
    uint8_t *payload_out,
    size_t payload_capacity,
    uint16_t *out_max_coeff_bits,
    uint16_t *out_bytes_per_coeff,
    size_t *out_payload_len)
{
    return gpu_poly_store_compact_bytes(
        mat,
        payload_out,
        payload_capacity,
        out_max_coeff_bits,
        out_bytes_per_coeff,
        out_payload_len);
}

extern "C" int gpu_matrix_load_compact_bytes(
    GpuMatrix *mat,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    return gpu_poly_load_compact_bytes(mat, payload, payload_len, max_coeff_bits);
}
