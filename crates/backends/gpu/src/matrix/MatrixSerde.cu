namespace
{
    constexpr int kMaxRnsLimbs = 64;
    constexpr int kMaxCoeffWords = 64;

    struct SerdeStreamRef
    {
        int device;
        gpuStream_t stream;
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

    void serde_append_unique_stream(std::vector<SerdeStreamRef> &streams, int device, gpuStream_t stream)
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
            matrix_report_cleanup_error(mxx_set_device(entry.device));
            matrix_report_cleanup_error(gpuEventDestroy(entry.event));
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
            gpuError_t err = mxx_set_device(entry.device);
            if (err != gpuSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }

            gpuEvent_t ev = nullptr;
            err = gpuEventCreateWithFlags(&ev, gpuEventDisableTiming);
            if (err != gpuSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }
            err = gpuEventRecord(ev, entry.stream);
            if (err != gpuSuccess)
            {
                matrix_report_cleanup_error(gpuEventDestroy(ev));
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

    // A compact import uses one partition's dense limb prefix. Derive each
    // packed limb address from its modulus width, including deferred owners
    // whose descriptor table has not been filled by a graph yet.
    __global__ void serde_unpack_packed_coeffs_mod_kernel(
        const uint8_t *payload, size_t coeff_count, size_t n,
        uint32_t bit_width, const uint64_t *moduli, int limb_count,
        uint8_t *base, size_t stride)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= coeff_count) return;
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;
        const size_t base_bit = idx * static_cast<size_t>(bit_width);
        const uint32_t magnitude_bits = bit_width ? bit_width - 1 : 0;
        bool is_negative = false;
        if (bit_width)
        {
            const size_t sign_bit_idx = base_bit + bit_width - 1;
            is_negative = ((payload[sign_bit_idx / 8] >> (sign_bit_idx % 8)) & 1U) != 0;
        }
        size_t offset = 0;
        for (int limb = 0; limb < limb_count; ++limb)
        {
            const uint64_t modulus = moduli[limb];
            const uint8_t width = modulus <= UINT32_MAX ? 4 : 8;
            uint64_t residue = 0;
            for (int b = static_cast<int>(magnitude_bits) - 1; b >= 0; --b)
            {
                const size_t bit_idx = base_bit + static_cast<size_t>(b);
                const uint8_t bit = (payload[bit_idx / 8] >> (bit_idx % 8)) & 1U;
                residue = static_cast<uint64_t>((static_cast<unsigned __int128>(residue) * 2 + bit) % modulus);
            }
            if (is_negative && residue) residue = modulus - residue;
            matrix_store_limb_u64(base + offset, poly_idx, coeff_idx, stride, width, residue);
            offset += n * width;
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
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        // Upload on the partition's transfer stream, which no Graph launches
        // on, ordered after the limb's previous write.
        const auto &transfer = mat->ctx->execution->transfer_streams_by_partition;
        if (limb_id.x >= transfer.size() || !transfer[limb_id.x])
        {
            return set_error("missing transfer stream in gpu_matrix_load_rns_batch");
        }
        gpuStream_t stream = transfer[limb_id.x];
        status = matrix_wait_limb_stream(mat, limb_id, device, stream);
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

        gpuError_t err = mxx_set_device(device);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        uint64_t *src_words_device = nullptr;
        err = gpuMallocAsync(reinterpret_cast<void **>(&src_words_device), staging_bytes, stream);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        err = gpuMemcpy2DAsync(
            src_words_device,
            host_limb_bytes,
            src,
            bytes_per_poly,
            host_limb_bytes,
            count,
            gpuMemcpyHostToDevice,
            stream);
        if (err != gpuSuccess)
        {
            matrix_report_cleanup_error(gpuFreeAsync(src_words_device, stream));
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
        err = gpuGetLastError();
        if (err != gpuSuccess)
        {
            matrix_report_cleanup_error(gpuFreeAsync(src_words_device, stream));
            return set_error(err);
        }
        err = gpuFreeAsync(src_words_device, stream);
        if (err != gpuSuccess)
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

    const size_t partition = limb_map[0].x;
    if (partition >= poly->shared_limb_buffers.size() ||
        partition >= poly->ctx->ntt_device_constants.size())
        return set_error("missing compact import partition");
    const auto &storage = poly->shared_limb_buffers[partition];
    const auto &constants = poly->ctx->ntt_device_constants[partition];
    if (!storage.ptr || storage.limb_count < limb_count || !constants.moduli ||
        constants.limb_count < limb_count || storage.device != constants.device)
        return set_error("invalid compact import storage or constants");
    for (size_t limb = 0; limb < limb_count; ++limb)
        if (limb_map[limb].x != partition || limb_map[limb].y != limb)
            return set_error("compact import requires a dense limb prefix on one GPU");
    const int device = storage.device;
    const auto &transfer = poly->ctx->execution->transfer_streams_by_partition;
    if (partition >= transfer.size() || !transfer[partition])
        return set_error("missing compact import transfer stream");
    const gpuStream_t stream = transfer[partition];
    auto error = mxx_set_device(device);
    if (error != gpuSuccess) return set_error(error);
    if (matrix_wait_all_limb_streams(poly, device, stream, true) != 0) return 1;

    uint8_t *device_payload = nullptr;
    void *pinned_payload = nullptr;
    const auto release = [&]() {
        if (device_payload)
        {
            matrix_report_cleanup_error(gpuFreeAsync(device_payload, stream));
            device_payload = nullptr;
        }
        int status = 0;
        if (pinned_payload)
        {
            status = gpu_defer_pinned_frees(poly->ctx, device, stream, &pinned_payload, 1);
            // The reclaimer owns the pointer even on a fail-closed error.
            pinned_payload = nullptr;
        }
        return status;
    };
    if (payload_len)
    {
        error = gpuMallocAsync(reinterpret_cast<void **>(&device_payload), payload_len, stream);
        if (error == gpuSuccess) error = gpuMallocHost(&pinned_payload, payload_len);
        if (error == gpuSuccess)
        {
            std::memcpy(pinned_payload, payload, payload_len);
            error = gpuMemcpyAsync(device_payload, pinned_payload, payload_len, gpuMemcpyHostToDevice, stream);
        }
        if (error != gpuSuccess)
        {
            release();
            return set_error(error);
        }
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((coeff_count + threads - 1) / threads);
    serde_unpack_packed_coeffs_mod_kernel<<<blocks, threads, 0, stream>>>(
        device_payload, coeff_count, n, max_coeff_bits, constants.moduli,
        static_cast<int>(limb_count), storage.ptr, storage.bytes_per_poly);
    error = gpuGetLastError();
    if (error != gpuSuccess)
    {
        release();
        return set_error(error);
    }
    // One writer completion covers all limbs and queued staging cleanup.
    const int retired = release();
    const int written = matrix_record_all_limb_writes(poly, stream, true);
    return written ? written : retired;
}

extern "C" int gpu_matrix_load_compact_bytes(
    GpuMatrix *mat,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    return gpu_poly_load_compact_bytes(mat, payload, payload_len, max_coeff_bits);
}
