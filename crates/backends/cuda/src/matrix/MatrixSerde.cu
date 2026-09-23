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
            mxx_set_device(entry.device);
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
            cudaError_t err = mxx_set_device(entry.device);
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

    __global__ void serde_pack_u64_limb_to_packed_kernel(
        const uint64_t *src_words,
        size_t src_stride_words,
        uint8_t *dst_base,
        size_t dst_stride_bytes,
        uint8_t dst_coeff_bytes,
        size_t poly_count,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;
        const uint64_t value = src_words[poly_idx * src_stride_words + coeff_idx];
        matrix_store_limb_u64(
            dst_base,
            poly_idx,
            coeff_idx,
            dst_stride_bytes,
            dst_coeff_bytes,
            value);
    }

    __device__ __forceinline__ int serde_compare_words_desc_device(
        const uint64_t *lhs,
        const uint64_t *rhs,
        int words_per_coeff)
    {
        for (int w = words_per_coeff - 1; w >= 0; --w)
        {
            const uint64_t a = lhs[w];
            const uint64_t b = rhs[w];
            if (a > b)
            {
                return 1;
            }
            if (a < b)
            {
                return -1;
            }
        }
        return 0;
    }

    __global__ void serde_unpack_packed_coeffs_mod_kernel(
        const uint8_t *payload,
        size_t coeff_count,
        size_t n,
        uint32_t bit_width,
        const uint64_t *moduli,
        int limb_count,
        uint8_t *const *limb_ptrs,
        const size_t *limb_strides,
        const uint8_t *limb_coeff_bytes)
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
                const size_t limb_idx = static_cast<size_t>(limb);
                matrix_store_limb_u64(
                    limb_ptrs[limb_idx],
                    poly_idx,
                    coeff_idx,
                    limb_strides[limb_idx],
                    limb_coeff_bytes[limb_idx],
                    0);
            }
            return;
        }

        const size_t base_bit = idx * static_cast<size_t>(bit_width);
        const uint32_t magnitude_bits = bit_width > 0 ? (bit_width - 1) : 0;
        const size_t sign_bit_idx = base_bit + static_cast<size_t>(bit_width - 1);
        const uint8_t sign_byte = payload[sign_bit_idx / 8];
        const bool is_negative =
            ((sign_byte >> (sign_bit_idx % 8)) & 0x1u) != 0;
        for (int limb = 0; limb < limb_count; ++limb)
        {
            const uint64_t modulus = moduli[static_cast<size_t>(limb)];
            uint64_t residue = 0;
            for (int b = static_cast<int>(magnitude_bits) - 1; b >= 0; --b)
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
            if (is_negative && residue != 0)
            {
                residue = modulus - residue;
            }
            const size_t limb_idx = static_cast<size_t>(limb);
            matrix_store_limb_u64(
                limb_ptrs[limb_idx],
                poly_idx,
                coeff_idx,
                limb_strides[limb_idx],
                limb_coeff_bytes[limb_idx],
                residue);
        }
    }

    bool serde_compute_modulus_words_le(
        const std::vector<uint64_t> &moduli,
        std::vector<uint64_t> *out_words)
    {
        if (!out_words)
        {
            return false;
        }
        out_words->clear();
        out_words->push_back(1u);
        for (uint64_t modulus : moduli)
        {
            unsigned __int128 carry = 0;
            for (size_t i = 0; i < out_words->size(); ++i)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>((*out_words)[i]) *
                        static_cast<unsigned __int128>(modulus) +
                    carry;
                (*out_words)[i] = static_cast<uint64_t>(term);
                carry = term >> 64;
            }
            if (carry != 0)
            {
                out_words->push_back(static_cast<uint64_t>(carry));
            }
        }
        while (out_words->size() > 1 && out_words->back() == 0)
        {
            out_words->pop_back();
        }
        return true;
    }

    void serde_shift_words_right_one_le(std::vector<uint64_t> *words)
    {
        if (!words || words->empty())
        {
            return;
        }
        uint64_t carry = 0;
        for (size_t i = words->size(); i-- > 0;)
        {
            const uint64_t current = (*words)[i];
            (*words)[i] = (current >> 1) | (carry << 63);
            carry = current & 1ULL;
        }
        while (words->size() > 1 && words->back() == 0)
        {
            words->pop_back();
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
    GpuEventSet **out_events)
{
    if (mat) mat->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_load_rns_batch");
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
        return 0;
    }

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_load_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t host_limb_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    const size_t host_limb_words = static_cast<size_t>(N);
    size_t total_coeff = 0;
    size_t staging_bytes = 0;
    if (!serde_checked_mul_size(count, host_limb_words, &total_coeff) ||
        !serde_checked_mul_size(total_coeff, sizeof(uint64_t), &staging_bytes))
    {
        return set_error("staging size overflow in gpu_matrix_load_rns_batch");
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint8_t *dst = matrix_limb_ptr_by_id(mat, 0, limb_id);
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
        size_t dst_pitch = 0;
        uint8_t dst_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(mat, limb_id, &dst_pitch, &dst_coeff_bytes))
        {
            return set_error("invalid limb metadata in gpu_matrix_load_rns_batch");
        }
        const size_t dst_width = static_cast<size_t>(N) * static_cast<size_t>(dst_coeff_bytes);
        if (dst_coeff_bytes == 0 || dst_pitch < dst_width)
        {
            return set_error("invalid destination stride in gpu_matrix_load_rns_batch");
        }
        const uint8_t *src = bytes + limb * host_limb_bytes;

        cudaError_t err = mxx_set_device(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        uint64_t *src_words_device = nullptr;
        err = cudaMallocAsync(reinterpret_cast<void **>(&src_words_device), staging_bytes, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpy2DAsync(
            src_words_device,
            host_limb_bytes,
            src,
            bytes_per_poly,
            host_limb_bytes,
            count,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(src_words_device, stream);
            return set_error(err);
        }
        const int threads = 256;
        const int blocks =
            static_cast<int>((total_coeff + static_cast<size_t>(threads) - 1) /
                             static_cast<size_t>(threads));
        serde_pack_u64_limb_to_packed_kernel<<<blocks, threads, 0, stream>>>(
            src_words_device,
            host_limb_words,
            dst,
            dst_pitch,
            dst_coeff_bytes,
            count,
            static_cast<size_t>(N));
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(src_words_device, stream);
            return set_error(err);
        }
        err = cudaFreeAsync(src_words_device, stream);
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

    return serde_build_event_set_from_streams(streams, out_events);
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

    std::vector<uint8_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    std::vector<uint8_t> limb_coeff_bytes(limb_count);
    int common_device = -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint8_t *limb_ptr = matrix_limb_ptr_by_id(poly, 0, limb_id);
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
        size_t limb_stride = 0;
        uint8_t limb_bytes = 0;
        if (!matrix_limb_metadata_by_id(poly, limb_id, &limb_stride, &limb_bytes))
        {
            return set_error("invalid limb metadata in gpu_poly_load_compact_bytes");
        }
        if (limb_bytes == 0 || limb_stride < n * static_cast<size_t>(limb_bytes))
        {
            return set_error("invalid limb stride in gpu_poly_load_compact_bytes");
        }
        if (common_device < 0)
        {
            common_device = device;
        }
        else if (common_device != device)
        {
            return set_error("gpu_poly_load_compact_bytes requires all limbs on a single GPU");
        }
        limb_ptrs[limb] = limb_ptr;
        limb_strides[limb] = limb_stride;
        limb_coeff_bytes[limb] = limb_bytes;
    }

    cudaError_t err = mxx_set_device(common_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    uint8_t *d_payload = nullptr;
    uint8_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint8_t *d_limb_coeff_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    cudaStream_t work_stream = nullptr;
    auto free_ptr = [&](auto *&ptr) {
        if (!ptr)
        {
            return;
        }
        if (work_stream)
        {
            cudaFreeAsync(ptr, work_stream);
        }
        else
        {
            cudaFree(ptr);
        }
        ptr = nullptr;
    };
    auto release = [&]() {
        free_ptr(d_moduli);
        free_ptr(d_limb_coeff_bytes);
        free_ptr(d_limb_strides);
        free_ptr(d_limb_ptrs);
        free_ptr(d_payload);
        if (work_stream)
        {
            cudaStreamDestroy(work_stream);
            work_stream = nullptr;
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
        err = cudaMallocAsync(
            reinterpret_cast<void **>(&d_payload),
            payload_len,
            work_stream);
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

    err = cudaMallocAsync(
        reinterpret_cast<void **>(&d_limb_ptrs),
        limb_count * sizeof(uint8_t *),
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_ptrs,
        limb_ptrs.data(),
        limb_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = cudaMallocAsync(
        reinterpret_cast<void **>(&d_limb_strides),
        limb_count * sizeof(size_t),
        work_stream);
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
    err = cudaMallocAsync(
        reinterpret_cast<void **>(&d_limb_coeff_bytes),
        limb_count * sizeof(uint8_t),
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_coeff_bytes,
        limb_coeff_bytes.data(),
        limb_count * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + limb_count);
    err = cudaMallocAsync(
        reinterpret_cast<void **>(&d_moduli),
        moduli_subset.size() * sizeof(uint64_t),
        work_stream);
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
        d_limb_strides,
        d_limb_coeff_bytes);
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
    return 0;
}

extern "C" int gpu_matrix_load_compact_bytes(
    GpuMatrix *mat,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    return gpu_poly_load_compact_bytes(mat, payload, payload_len, max_coeff_bits);
}
