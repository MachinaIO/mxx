namespace
{
    __device__ __forceinline__ uint64_t mul_mod_u64_device(uint64_t a, uint64_t b, uint64_t modulus)
    {
        const unsigned __int128 prod =
            static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % static_cast<unsigned __int128>(modulus));
    }

    __global__ void reconstruct_rns_to_words_kernel(
        const uint64_t *const *limb_ptrs,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        int inverse_stride,
        int limb_count,
        size_t n,
        int words_per_coeff,
        uint64_t *coeff_words_out,
        int *overflow_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= n)
        {
            return;
        }

        uint64_t mixed_digits[kMaxRnsLimbs];
        uint64_t coeff_words[kMaxCoeffWords];

        for (int i = 0; i < limb_count; ++i)
        {
            mixed_digits[i] = limb_ptrs[i][idx] % moduli[i];
        }

        const size_t inverse_stride_sz = static_cast<size_t>(inverse_stride);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[i];
            uint64_t t = mixed_digits[i];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t xj_mod_qi = mixed_digits[j] % qi;
                const uint64_t diff = t >= xj_mod_qi
                                          ? (t - xj_mod_qi)
                                          : static_cast<uint64_t>(
                                                static_cast<unsigned __int128>(t) +
                                                static_cast<unsigned __int128>(qi) -
                                                static_cast<unsigned __int128>(xj_mod_qi));
                const uint64_t inv = garner_inverses[static_cast<size_t>(j) * inverse_stride_sz +
                                                    static_cast<size_t>(i)];
                t = mul_mod_u64_device(diff, inv, qi);
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
    __device__ __forceinline__ uint32_t bit_width_u64_device(uint64_t value)
    {
        return value == 0 ? 0u : static_cast<uint32_t>(64 - __clzll(value));
    }

    __global__ void compute_max_bits_from_words_kernel(
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
                bit_width = static_cast<uint32_t>(w) * 64u + bit_width_u64_device(word);
                break;
            }
        }
        atomicMax(max_bits_out, bit_width);
    }

    __global__ void pack_coeff_words_bits_kernel(
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

    __global__ void unpack_packed_coeffs_mod_kernel(
        const uint8_t *payload,
        size_t n,
        uint32_t bit_width,
        uint64_t modulus,
        uint64_t *residue_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= n)
        {
            return;
        }
        if (bit_width == 0)
        {
            residue_out[idx] = 0;
            return;
        }

        const size_t base_bit = idx * static_cast<size_t>(bit_width);
        uint64_t residue = 0;
        for (int b = static_cast<int>(bit_width) - 1; b >= 0; --b)
        {
            const size_t bit_idx = base_bit + static_cast<size_t>(b);
            const uint8_t byte_val = payload[bit_idx / 8];
            const uint8_t bit = static_cast<uint8_t>((byte_val >> (bit_idx % 8)) & 0x1u);
            const unsigned __int128 term =
                static_cast<unsigned __int128>(residue) * static_cast<unsigned __int128>(2u) +
                static_cast<unsigned __int128>(bit);
            residue = static_cast<uint64_t>(term % static_cast<unsigned __int128>(modulus));
        }
        residue_out[idx] = residue;
    }
}

extern "C"
{
    int gpu_poly_load_rns(GpuPoly *poly, const uint64_t *rns_flat, size_t rns_len, int format)
    {
        try
        {
            if (!poly || !rns_flat)
            {
                return set_error("invalid gpu_poly_load_rns arguments");
            }
            PolyFormat target_format;
            if (!parse_format(format, target_format))
            {
                return set_error("invalid format in gpu_poly_load_rns");
            }
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (rns_len != expected)
            {
                return set_error("rns_len mismatch in gpu_poly_load_rns");
            }

            std::vector<std::vector<uint64_t>> data(level + 1, std::vector<uint64_t>(N));
            for (int limb = 0; limb <= level; ++limb)
            {
                for (int i = 0; i < N; ++i)
                {
                    data[limb][i] = rns_flat[static_cast<size_t>(limb) * N + i];
                }
            }

            std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + level + 1);
            poly->poly->load(data, moduli_subset);
            poly->format = target_format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_load_rns");
        }
    }

    int gpu_poly_load_compact_bytes(
        GpuPoly *poly,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits)
    {
        try
        {
            if (!poly || !poly->ctx)
            {
                return set_error("invalid gpu_poly_load_compact_bytes arguments");
            }

            const int level = poly->level;
            const int limb_count = level + 1;
            const size_t n = static_cast<size_t>(poly->ctx->N);
            if (limb_count <= 0)
            {
                return set_error("invalid level in gpu_poly_load_compact_bytes");
            }
            if (limb_count > static_cast<int>(poly->ctx->moduli.size()))
            {
                return set_error("unexpected modulus count in gpu_poly_load_compact_bytes");
            }

            if (max_coeff_bits == 0)
            {
                if (payload_len != 0)
                {
                    return set_error("payload_len must be zero when max_coeff_bits is zero");
                }
            }
            else
            {
                if (!payload)
                {
                    return set_error("null payload in gpu_poly_load_compact_bytes");
                }
            }

            if (max_coeff_bits > 0 &&
                n > std::numeric_limits<size_t>::max() / static_cast<size_t>(max_coeff_bits))
            {
                return set_error("payload length overflow in gpu_poly_load_compact_bytes");
            }
            const size_t expected_payload_len =
                (n * static_cast<size_t>(max_coeff_bits) + static_cast<size_t>(7)) / static_cast<size_t>(8);
            if (payload_len != expected_payload_len)
            {
                return set_error("payload length mismatch in gpu_poly_load_compact_bytes");
            }

            auto *ctx = poly->ctx->ctx;
            if (ctx->limbGPUid.size() < static_cast<size_t>(limb_count))
            {
                return set_error("unexpected limb mapping size in gpu_poly_load_compact_bytes");
            }

            struct DevicePayloadBuffer
            {
                int device;
                uint8_t *ptr;
                cudaEvent_t copy_ready;
            };
            struct DeviceEvent
            {
                int device;
                cudaEvent_t event;
            };
            std::vector<DevicePayloadBuffer> payload_buffers;
            std::vector<DeviceEvent> completion_events;
            auto release = [&]() {
                for (const auto &entry : completion_events)
                {
                    if (entry.event)
                    {
                        cudaSetDevice(entry.device);
                        cudaEventDestroy(entry.event);
                    }
                }
                completion_events.clear();
                for (auto &entry : payload_buffers)
                {
                    cudaSetDevice(entry.device);
                    if (entry.copy_ready)
                    {
                        cudaEventDestroy(entry.copy_ready);
                        entry.copy_ready = nullptr;
                    }
                    if (entry.ptr)
                    {
                        cudaFree(entry.ptr);
                        entry.ptr = nullptr;
                    }
                }
            };
            auto find_payload_buffer = [&](int device) -> DevicePayloadBuffer * {
                for (auto &entry : payload_buffers)
                {
                    if (entry.device == device)
                    {
                        return &entry;
                    }
                }
                return nullptr;
            };

            const int threads = 256;
            const int blocks = static_cast<int>(
                (n + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

            for (int limb = 0; limb < limb_count; ++limb)
            {
                const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                if (limb_id.x >= poly->poly->GPU.size())
                {
                    release();
                    return set_error("unexpected limb GPU partition in gpu_poly_load_compact_bytes");
                }
                auto &partition = poly->poly->GPU[limb_id.x];
                if (limb_id.y >= partition.limb.size())
                {
                    release();
                    return set_error("unexpected limb index in gpu_poly_load_compact_bytes");
                }
                auto &limb_impl = partition.limb[limb_id.y];
                if (limb_impl.index() != FIDESlib::U64)
                {
                    release();
                    return set_error("unsupported limb type in gpu_poly_load_compact_bytes");
                }
                auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

                cudaError_t err = cudaSetDevice(partition.device);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }

                if (max_coeff_bits == 0)
                {
                    err = cudaMemsetAsync(
                        limb_u64.v.data,
                        0,
                        static_cast<size_t>(poly->ctx->N) * sizeof(uint64_t),
                        limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    cudaEvent_t done = nullptr;
                    err = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    err = cudaEventRecord(done, limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(done);
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    completion_events.push_back(DeviceEvent{partition.device, done});
                    continue;
                }

                DevicePayloadBuffer *payload_entry = find_payload_buffer(partition.device);
                if (!payload_entry)
                {
                    DevicePayloadBuffer entry{partition.device, nullptr, nullptr};
                    uint8_t *d_payload = nullptr;
                    err = cudaMalloc(reinterpret_cast<void **>(&d_payload), payload_len);
                    if (err != cudaSuccess)
                    {
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    err = cudaMemcpyAsync(
                        d_payload,
                        payload,
                        payload_len,
                        cudaMemcpyHostToDevice,
                        limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        cudaFree(d_payload);
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    entry.ptr = d_payload;
                    err = cudaEventCreateWithFlags(&entry.copy_ready, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        cudaFree(d_payload);
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    err = cudaEventRecord(entry.copy_ready, limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(entry.copy_ready);
                        cudaFree(d_payload);
                        release();
                        return set_error(cudaGetErrorString(err));
                    }
                    payload_buffers.push_back(entry);
                    payload_entry = &payload_buffers.back();
                }
                err = cudaStreamWaitEvent(limb_u64.stream.ptr, payload_entry->copy_ready, 0);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }

                unpack_packed_coeffs_mod_kernel<<<blocks, threads, 0, limb_u64.stream.ptr>>>(
                    payload_entry->ptr,
                    n,
                    static_cast<uint32_t>(max_coeff_bits),
                    poly->ctx->moduli[static_cast<size_t>(limb)],
                    limb_u64.v.data);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                cudaEvent_t done = nullptr;
                err = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(done, limb_u64.stream.ptr);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(done);
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                completion_events.push_back(DeviceEvent{partition.device, done});
            }

            for (const auto &entry : completion_events)
            {
                cudaError_t err = cudaSetDevice(entry.device);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventSynchronize(entry.event);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
            }

            release();
            poly->format = PolyFormat::Coeff;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_load_compact_bytes");
        }
    }

    int gpu_poly_store_rns(
        GpuPoly *poly,
        uint64_t *rns_flat_out,
        size_t rns_len,
        int format,
        GpuEventSet **out_events)
    {
        try
        {
            if (!poly || !rns_flat_out || !out_events)
            {
                return set_error("invalid gpu_poly_store_rns arguments");
            }
            PolyFormat target_format;
            if (!parse_format(format, target_format))
            {
                return set_error("invalid format in gpu_poly_store_rns");
            }
            *out_events = nullptr;
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (rns_len != expected)
            {
                return set_error("rns_len mismatch in gpu_poly_store_rns");
            }

            const int batch = default_batch(poly->ctx);
            if (target_format == PolyFormat::Eval)
            {
                ensure_eval(poly, batch);
            }
            else
            {
                ensure_coeff(poly, batch);
            }

            auto *ctx = poly->ctx->ctx;
            if (ctx->limbGPUid.size() < static_cast<size_t>(level + 1))
            {
                return set_error("unexpected limb mapping size in gpu_poly_store_rns");
            }

            auto *event_set = new GpuEventSet();
            event_set->entries.reserve(static_cast<size_t>(level + 1));
            for (int limb = 0; limb <= level; ++limb)
            {
                const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                if (limb_id.x >= poly->poly->GPU.size())
                {
                    destroy_event_set(event_set);
                    return set_error("unexpected limb GPU partition in gpu_poly_store_rns");
                }
                auto &partition = poly->poly->GPU[limb_id.x];
                if (limb_id.y >= partition.limb.size())
                {
                    destroy_event_set(event_set);
                    return set_error("unexpected limb index in gpu_poly_store_rns");
                }
                auto &limb_impl = partition.limb[limb_id.y];
                if (limb_impl.index() != FIDESlib::U64)
                {
                    destroy_event_set(event_set);
                    return set_error("unsupported limb type in gpu_poly_store_rns");
                }
                auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

                cudaError_t err = cudaSetDevice(partition.device);
                if (err != cudaSuccess)
                {
                    destroy_event_set(event_set);
                    return set_error(cudaGetErrorString(err));
                }

                uint64_t *dst = rns_flat_out + static_cast<size_t>(limb) * static_cast<size_t>(N);
                err = cudaMemcpyAsync(
                    dst,
                    limb_u64.v.data,
                    static_cast<size_t>(N) * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    limb_u64.stream.ptr);
                if (err != cudaSuccess)
                {
                    destroy_event_set(event_set);
                    return set_error(cudaGetErrorString(err));
                }

                cudaEvent_t ev = nullptr;
                err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    destroy_event_set(event_set);
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(ev, limb_u64.stream.ptr);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ev);
                    destroy_event_set(event_set);
                    return set_error(cudaGetErrorString(err));
                }
                event_set->entries.push_back(GpuEventSet::Entry{ev, partition.device});
            }

            *out_events = event_set;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_store_rns");
        }
    }

    int gpu_poly_load_rns_batch(
        GpuPoly *const *polys,
        size_t poly_count,
        const uint8_t *bytes,
        size_t bytes_per_poly,
        int format)
    {
        try
        {
            if (!polys || (!bytes && poly_count > 0))
            {
                return set_error("invalid gpu_poly_load_rns_batch arguments");
            }
            if (poly_count == 0)
            {
                return 0;
            }
            PolyFormat target_format;
            if (!parse_format(format, target_format))
            {
                return set_error("invalid format in gpu_poly_load_rns_batch");
            }
            if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
            {
                return set_error("bytes_per_poly must be a non-zero multiple of 8");
            }

            GpuPoly *first = polys[0];
            if (!first || !first->ctx)
            {
                return set_error("null poly in gpu_poly_load_rns_batch");
            }
            const size_t expected_len = expected_rns_len(first);
            const size_t expected_bytes = expected_len * sizeof(uint64_t);
            if (bytes_per_poly < expected_bytes)
            {
                return set_error("bytes_per_poly too small in gpu_poly_load_rns_batch");
            }
            const int level = first->level;
            const int N = first->ctx->N;
            std::vector<uint64_t> moduli_subset(first->ctx->moduli.begin(),
                                                first->ctx->moduli.begin() + level + 1);

            std::vector<std::vector<uint64_t>> data(level + 1);
            for (int limb = 0; limb <= level; ++limb)
            {
                data[limb].resize(N);
            }

            for (size_t i = 0; i < poly_count; ++i)
            {
                GpuPoly *poly = polys[i];
                if (!poly || !poly->ctx)
                {
                    return set_error("null poly in gpu_poly_load_rns_batch");
                }
                if (poly->ctx != first->ctx || poly->level != level)
                {
                    return set_error("mismatched poly context or level in gpu_poly_load_rns_batch");
                }

                const uint8_t *base = bytes + i * bytes_per_poly;
                const uint64_t *rns_flat = reinterpret_cast<const uint64_t *>(base);
                std::vector<uint64_t> tmp;
                if (reinterpret_cast<uintptr_t>(base) % alignof(uint64_t) != 0)
                {
                    tmp.resize(expected_len);
                    std::memcpy(tmp.data(), base, expected_bytes);
                    rns_flat = tmp.data();
                }

                for (int limb = 0; limb <= level; ++limb)
                {
                    const uint64_t *src = rns_flat + static_cast<size_t>(limb) * static_cast<size_t>(N);
                    std::memcpy(data[limb].data(), src, static_cast<size_t>(N) * sizeof(uint64_t));
                }

                poly->poly->load(data, moduli_subset);
                poly->format = target_format;
            }
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_load_rns_batch");
        }
    }

    int gpu_poly_store_rns_batch(
        GpuPoly *const *polys,
        size_t poly_count,
        uint8_t *bytes_out,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events)
    {
        try
        {
            if (!polys || (!bytes_out && poly_count > 0) || !out_events)
            {
                return set_error("invalid gpu_poly_store_rns_batch arguments");
            }
            *out_events = nullptr;
            if (poly_count == 0)
            {
                return 0;
            }
            PolyFormat target_format;
            if (!parse_format(format, target_format))
            {
                return set_error("invalid format in gpu_poly_store_rns_batch");
            }
            if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
            {
                return set_error("bytes_per_poly must be a non-zero multiple of 8");
            }

            GpuPoly *first = polys[0];
            if (!first || !first->ctx)
            {
                return set_error("null poly in gpu_poly_store_rns_batch");
            }
            const size_t expected_len = expected_rns_len(first);
            const size_t expected_bytes = expected_len * sizeof(uint64_t);
            if (bytes_per_poly < expected_bytes)
            {
                return set_error("bytes_per_poly too small in gpu_poly_store_rns_batch");
            }
            const int level = first->level;
            const int N = first->ctx->N;
            auto *ctx = first->ctx->ctx;
            if (ctx->limbGPUid.size() < static_cast<size_t>(level + 1))
            {
                return set_error("unexpected limb mapping size in gpu_poly_store_rns_batch");
            }

            auto *event_set = new GpuEventSet();
            event_set->entries.reserve(static_cast<size_t>(level + 1) * poly_count);

            const int batch = default_batch(first->ctx);

            for (size_t i = 0; i < poly_count; ++i)
            {
                GpuPoly *poly = polys[i];
                if (!poly || !poly->ctx)
                {
                    destroy_event_set(event_set);
                    return set_error("null poly in gpu_poly_store_rns_batch");
                }
                if (poly->ctx != first->ctx || poly->level != level)
                {
                    destroy_event_set(event_set);
                    return set_error("mismatched poly context or level in gpu_poly_store_rns_batch");
                }

                if (target_format == PolyFormat::Eval)
                {
                    ensure_eval(poly, batch);
                }
                else
                {
                    ensure_coeff(poly, batch);
                }

                for (int limb = 0; limb <= level; ++limb)
                {
                    const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                    if (limb_id.x >= poly->poly->GPU.size())
                    {
                        destroy_event_set(event_set);
                        return set_error("unexpected limb GPU partition in gpu_poly_store_rns_batch");
                    }
                    auto &partition = poly->poly->GPU[limb_id.x];
                    if (limb_id.y >= partition.limb.size())
                    {
                        destroy_event_set(event_set);
                        return set_error("unexpected limb index in gpu_poly_store_rns_batch");
                    }
                    auto &limb_impl = partition.limb[limb_id.y];
                    if (limb_impl.index() != FIDESlib::U64)
                    {
                        destroy_event_set(event_set);
                        return set_error("unsupported limb type in gpu_poly_store_rns_batch");
                    }
                    auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

                    cudaError_t err = cudaSetDevice(partition.device);
                    if (err != cudaSuccess)
                    {
                        destroy_event_set(event_set);
                        return set_error(cudaGetErrorString(err));
                    }

                    uint8_t *dst_bytes = bytes_out + i * bytes_per_poly +
                                         static_cast<size_t>(limb) * static_cast<size_t>(N) * sizeof(uint64_t);
                    err = cudaMemcpyAsync(
                        dst_bytes,
                        limb_u64.v.data,
                        static_cast<size_t>(N) * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost,
                        limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        destroy_event_set(event_set);
                        return set_error(cudaGetErrorString(err));
                    }

                    cudaEvent_t ev = nullptr;
                    err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        destroy_event_set(event_set);
                        return set_error(cudaGetErrorString(err));
                    }
                    err = cudaEventRecord(ev, limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(ev);
                        destroy_event_set(event_set);
                        return set_error(cudaGetErrorString(err));
                    }
                    event_set->entries.push_back({ev, partition.device});
                }
            }

            *out_events = event_set;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_store_rns_batch");
        }
    }

    int gpu_poly_store_coeffs_words(
        GpuPoly *poly,
        uint64_t *coeff_words_out,
        size_t coeff_words_len,
        size_t words_per_coeff,
        int format)
    {
        try
        {
            if (!poly || !poly->ctx || !coeff_words_out)
            {
                return set_error("invalid gpu_poly_store_coeffs_words arguments");
            }
            if (words_per_coeff == 0)
            {
                return set_error("words_per_coeff must be non-zero in gpu_poly_store_coeffs_words");
            }
            PolyFormat target_format;
            if (!parse_format(format, target_format))
            {
                return set_error("invalid format in gpu_poly_store_coeffs_words");
            }

            const int batch = default_batch(poly->ctx);
            if (target_format == PolyFormat::Eval)
            {
                ensure_coeff(poly, batch);
            }
            if (poly->format != PolyFormat::Coeff)
            {
                return set_error("gpu_poly_store_coeffs_words expects coeff format");
            }

            const int level = poly->level;
            const int limb_count = level + 1;
            if (limb_count <= 0)
            {
                return 0;
            }
            if (limb_count > static_cast<int>(poly->ctx->moduli.size()))
            {
                return set_error("unexpected modulus count in gpu_poly_store_coeffs_words");
            }
            if (limb_count > kMaxRnsLimbs)
            {
                return set_error("limb_count exceeds kernel limit in gpu_poly_store_coeffs_words");
            }
            if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
            {
                return set_error("words_per_coeff exceeds kernel limit in gpu_poly_store_coeffs_words");
            }

            size_t total_bits_upper = 0;
            for (int limb = 0; limb < limb_count; ++limb)
            {
                total_bits_upper +=
                    static_cast<size_t>(bit_width_u64(poly->ctx->moduli[static_cast<size_t>(limb)]));
            }
            const size_t required_words =
                std::max<size_t>(1, (total_bits_upper + static_cast<size_t>(63)) /
                                        static_cast<size_t>(64));
            if (words_per_coeff < required_words)
            {
                return set_error("words_per_coeff too small in gpu_poly_store_coeffs_words");
            }

            const size_t n = static_cast<size_t>(poly->ctx->N);
            if (n > 0 && words_per_coeff > std::numeric_limits<size_t>::max() / n)
            {
                return set_error("output length overflow in gpu_poly_store_coeffs_words");
            }
            const size_t expected_len = n * words_per_coeff;
            if (coeff_words_len < expected_len)
            {
                return set_error("coeff_words_len too small in gpu_poly_store_coeffs_words");
            }

            auto *ctx = poly->ctx->ctx;
            if (ctx->limbGPUid.size() < static_cast<size_t>(limb_count))
            {
                return set_error("unexpected limb mapping size in gpu_poly_store_coeffs_words");
            }

            std::vector<const uint64_t *> limb_ptrs(static_cast<size_t>(limb_count));
            std::vector<cudaStream_t> limb_streams(static_cast<size_t>(limb_count));
            int common_device = -1;
            for (int limb = 0; limb < limb_count; ++limb)
            {
                const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                if (limb_id.x >= poly->poly->GPU.size())
                {
                    return set_error("unexpected limb GPU partition in gpu_poly_store_coeffs_words");
                }
                auto &partition = poly->poly->GPU[limb_id.x];
                if (limb_id.y >= partition.limb.size())
                {
                    return set_error("unexpected limb index in gpu_poly_store_coeffs_words");
                }
                auto &limb_impl = partition.limb[limb_id.y];
                if (limb_impl.index() != FIDESlib::U64)
                {
                    return set_error("unsupported limb type in gpu_poly_store_coeffs_words");
                }
                if (common_device < 0)
                {
                    common_device = partition.device;
                }
                else if (common_device != partition.device)
                {
                    return set_error(
                        "gpu_poly_store_coeffs_words requires all limbs on a single GPU");
                }
                auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
                limb_ptrs[static_cast<size_t>(limb)] = limb_u64.v.data;
                limb_streams[static_cast<size_t>(limb)] = limb_u64.stream.ptr;
            }

            cudaError_t err = cudaSetDevice(common_device);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }

            std::vector<uint64_t> moduli_subset(
                poly->ctx->moduli.begin(),
                poly->ctx->moduli.begin() + static_cast<size_t>(limb_count));
            const size_t inverse_stride = poly->ctx->moduli.size();
            const std::vector<uint64_t> &inverse_table = poly->ctx->garner_inverse_table;
            if (inverse_table.size() != inverse_stride * inverse_stride)
            {
                return set_error("invalid cached inverse table in gpu_poly_store_coeffs_words");
            }

            const uint64_t **d_limb_ptrs = nullptr;
            uint64_t *d_moduli = nullptr;
            uint64_t *d_garner_inv = nullptr;
            uint64_t *d_coeff_words = nullptr;
            int *d_overflow = nullptr;
            cudaStream_t work_stream = nullptr;
            std::vector<cudaEvent_t> ready_events;
            auto release = [&]() {
                for (cudaEvent_t ev : ready_events)
                {
                    if (ev)
                    {
                        cudaEventDestroy(ev);
                    }
                }
                ready_events.clear();
                if (work_stream)
                {
                    cudaStreamDestroy(work_stream);
                    work_stream = nullptr;
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
                return set_error(cudaGetErrorString(err));
            }
            ready_events.reserve(static_cast<size_t>(limb_count));
            for (int limb = 0; limb < limb_count; ++limb)
            {
                cudaEvent_t ready = nullptr;
                err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(ready, limb_streams[static_cast<size_t>(limb)]);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ready);
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaStreamWaitEvent(work_stream, ready, 0);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ready);
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                ready_events.push_back(ready);
            }

            err = cudaMalloc(
                reinterpret_cast<void **>(&d_limb_ptrs),
                static_cast<size_t>(limb_count) * sizeof(uint64_t *));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMemcpyAsync(
                d_limb_ptrs,
                limb_ptrs.data(),
                static_cast<size_t>(limb_count) * sizeof(uint64_t *),
                cudaMemcpyHostToDevice,
                work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            err = cudaMalloc(reinterpret_cast<void **>(&d_moduli), moduli_subset.size() * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
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
                return set_error(cudaGetErrorString(err));
            }

            err = cudaMalloc(
                reinterpret_cast<void **>(&d_garner_inv),
                inverse_table.size() * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
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
                return set_error(cudaGetErrorString(err));
            }

            if (expected_len > 0)
            {
                err = cudaMalloc(reinterpret_cast<void **>(&d_coeff_words), expected_len * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
            }
            err = cudaMalloc(reinterpret_cast<void **>(&d_overflow), sizeof(int));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMemsetAsync(d_overflow, 0, sizeof(int), work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            if (n > 0)
            {
                const int threads = 256;
                const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1) /
                                                    static_cast<size_t>(threads));
                reconstruct_rns_to_words_kernel<<<blocks, threads, 0, work_stream>>>(
                    d_limb_ptrs,
                    d_moduli,
                    d_garner_inv,
                    static_cast<int>(inverse_stride),
                    limb_count,
                    n,
                    static_cast<int>(words_per_coeff),
                    d_coeff_words,
                    d_overflow);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
            }

            int h_overflow = 0;
            err = cudaMemcpyAsync(
                &h_overflow,
                d_overflow,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            if (expected_len > 0)
            {
                err = cudaMemcpyAsync(
                    coeff_words_out,
                    d_coeff_words,
                    expected_len * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    work_stream);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
            }
            err = cudaStreamSynchronize(work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            if (h_overflow != 0)
            {
                release();
                return set_error("overflow in gpu_poly_store_coeffs_words");
            }

            release();
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_store_coeffs_words");
        }
    }

    int gpu_poly_store_compact_bytes(
        GpuPoly *poly,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len)
    {
        try
        {
            if (!poly || !poly->ctx || !payload_out || !out_max_coeff_bits ||
                !out_bytes_per_coeff || !out_payload_len)
            {
                return set_error("invalid gpu_poly_store_compact_bytes arguments");
            }

            const int batch = default_batch(poly->ctx);
            ensure_coeff(poly, batch);
            if (poly->format != PolyFormat::Coeff)
            {
                return set_error("gpu_poly_store_compact_bytes expects coeff format");
            }

            const int level = poly->level;
            const int limb_count = level + 1;
            const size_t n = static_cast<size_t>(poly->ctx->N);
            if (limb_count <= 0 || n == 0)
            {
                *out_max_coeff_bits = 0;
                *out_bytes_per_coeff = 0;
                *out_payload_len = 0;
                return 0;
            }
            if (limb_count > static_cast<int>(poly->ctx->moduli.size()))
            {
                return set_error("unexpected modulus count in gpu_poly_store_compact_bytes");
            }
            if (limb_count > kMaxRnsLimbs)
            {
                return set_error("limb_count exceeds kernel limit in gpu_poly_store_compact_bytes");
            }

            size_t total_bits_upper = 0;
            for (int limb = 0; limb < limb_count; ++limb)
            {
                total_bits_upper +=
                    static_cast<size_t>(bit_width_u64(poly->ctx->moduli[static_cast<size_t>(limb)]));
            }
            const size_t words_per_coeff =
                std::max<size_t>(1, (total_bits_upper + static_cast<size_t>(63)) /
                                        static_cast<size_t>(64));
            if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
            {
                return set_error("words_per_coeff exceeds kernel limit in gpu_poly_store_compact_bytes");
            }

            auto *ctx = poly->ctx->ctx;
            if (ctx->limbGPUid.size() < static_cast<size_t>(limb_count))
            {
                return set_error("unexpected limb mapping size in gpu_poly_store_compact_bytes");
            }

            std::vector<const uint64_t *> limb_ptrs(static_cast<size_t>(limb_count));
            std::vector<cudaStream_t> limb_streams(static_cast<size_t>(limb_count));
            int common_device = -1;
            for (int limb = 0; limb < limb_count; ++limb)
            {
                const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                if (limb_id.x >= poly->poly->GPU.size())
                {
                    return set_error("unexpected limb GPU partition in gpu_poly_store_compact_bytes");
                }
                auto &partition = poly->poly->GPU[limb_id.x];
                if (limb_id.y >= partition.limb.size())
                {
                    return set_error("unexpected limb index in gpu_poly_store_compact_bytes");
                }
                auto &limb_impl = partition.limb[limb_id.y];
                if (limb_impl.index() != FIDESlib::U64)
                {
                    return set_error("unsupported limb type in gpu_poly_store_compact_bytes");
                }
                if (common_device < 0)
                {
                    common_device = partition.device;
                }
                else if (common_device != partition.device)
                {
                    return set_error(
                        "gpu_poly_store_compact_bytes requires all limbs on a single GPU");
                }
                auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
                limb_ptrs[static_cast<size_t>(limb)] = limb_u64.v.data;
                limb_streams[static_cast<size_t>(limb)] = limb_u64.stream.ptr;
            }

            cudaError_t err = cudaSetDevice(common_device);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }

            if (words_per_coeff > std::numeric_limits<size_t>::max() / n)
            {
                return set_error("coeff word length overflow in gpu_poly_store_compact_bytes");
            }
            const size_t coeff_word_len = n * words_per_coeff;

            std::vector<uint64_t> moduli_subset(
                poly->ctx->moduli.begin(),
                poly->ctx->moduli.begin() + static_cast<size_t>(limb_count));
            const size_t inverse_stride = poly->ctx->moduli.size();
            const std::vector<uint64_t> &inverse_table = poly->ctx->garner_inverse_table;
            if (inverse_table.size() != inverse_stride * inverse_stride)
            {
                return set_error("invalid cached inverse table in gpu_poly_store_compact_bytes");
            }

            const uint64_t **d_limb_ptrs = nullptr;
            uint64_t *d_moduli = nullptr;
            uint64_t *d_garner_inv = nullptr;
            uint64_t *d_coeff_words = nullptr;
            int *d_overflow = nullptr;
            unsigned int *d_max_bits = nullptr;
            uint8_t *d_payload = nullptr;
            cudaStream_t work_stream = nullptr;
            std::vector<cudaEvent_t> ready_events;
            auto release = [&]() {
                for (cudaEvent_t ev : ready_events)
                {
                    if (ev)
                    {
                        cudaEventDestroy(ev);
                    }
                }
                ready_events.clear();
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
                return set_error(cudaGetErrorString(err));
            }
            ready_events.reserve(static_cast<size_t>(limb_count));
            for (int limb = 0; limb < limb_count; ++limb)
            {
                cudaEvent_t ready = nullptr;
                err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(ready, limb_streams[static_cast<size_t>(limb)]);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ready);
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaStreamWaitEvent(work_stream, ready, 0);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ready);
                    release();
                    return set_error(cudaGetErrorString(err));
                }
                ready_events.push_back(ready);
            }

            err = cudaMalloc(
                reinterpret_cast<void **>(&d_limb_ptrs),
                static_cast<size_t>(limb_count) * sizeof(uint64_t *));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMemcpyAsync(
                d_limb_ptrs,
                limb_ptrs.data(),
                static_cast<size_t>(limb_count) * sizeof(uint64_t *),
                cudaMemcpyHostToDevice,
                work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            err = cudaMalloc(reinterpret_cast<void **>(&d_moduli), moduli_subset.size() * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
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
                return set_error(cudaGetErrorString(err));
            }

            err = cudaMalloc(
                reinterpret_cast<void **>(&d_garner_inv),
                inverse_table.size() * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
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
                return set_error(cudaGetErrorString(err));
            }

            err = cudaMalloc(
                reinterpret_cast<void **>(&d_coeff_words),
                coeff_word_len * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMalloc(reinterpret_cast<void **>(&d_overflow), sizeof(int));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMemsetAsync(d_overflow, 0, sizeof(int), work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            const int threads = 256;
            const int blocks = static_cast<int>((n + static_cast<size_t>(threads) - 1) /
                                                static_cast<size_t>(threads));
            reconstruct_rns_to_words_kernel<<<blocks, threads, 0, work_stream>>>(
                d_limb_ptrs,
                d_moduli,
                d_garner_inv,
                static_cast<int>(inverse_stride),
                limb_count,
                n,
                static_cast<int>(words_per_coeff),
                d_coeff_words,
                d_overflow);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            int h_overflow = 0;
            err = cudaMemcpyAsync(
                &h_overflow,
                d_overflow,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaStreamSynchronize(work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            if (h_overflow != 0)
            {
                release();
                return set_error("overflow in gpu_poly_store_compact_bytes");
            }

            err = cudaMalloc(reinterpret_cast<void **>(&d_max_bits), sizeof(unsigned int));
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaMemsetAsync(d_max_bits, 0, sizeof(unsigned int), work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            compute_max_bits_from_words_kernel<<<blocks, threads, 0, work_stream>>>(
                d_coeff_words,
                n,
                static_cast<int>(words_per_coeff),
                d_max_bits);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            unsigned int h_max_bits = 0;
            err = cudaMemcpyAsync(
                &h_max_bits,
                d_max_bits,
                sizeof(unsigned int),
                cudaMemcpyDeviceToHost,
                work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            err = cudaStreamSynchronize(work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }
            if (h_max_bits > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
            {
                release();
                return set_error("max coeff bits exceed u16 range in gpu_poly_store_compact_bytes");
            }
            const unsigned int h_bytes_per_coeff =
                (h_max_bits + static_cast<unsigned int>(7)) / static_cast<unsigned int>(8);
            if (h_bytes_per_coeff >
                static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
            {
                release();
                return set_error("bytes_per_coeff exceed u16 range in gpu_poly_store_compact_bytes");
            }

            if (h_max_bits > 0 &&
                n > std::numeric_limits<size_t>::max() / static_cast<size_t>(h_max_bits))
            {
                release();
                return set_error("payload length overflow in gpu_poly_store_compact_bytes");
            }
            const size_t payload_len =
                (n * static_cast<size_t>(h_max_bits) + static_cast<size_t>(7)) / static_cast<size_t>(8);
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
                    return set_error(cudaGetErrorString(err));
                }

                const int payload_blocks = static_cast<int>((payload_len + threads - 1) / threads);
                pack_coeff_words_bits_kernel<<<payload_blocks, threads, 0, work_stream>>>(
                    d_coeff_words,
                    n,
                    static_cast<int>(words_per_coeff),
                    h_max_bits,
                    d_payload,
                    payload_len);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    release();
                    return set_error(cudaGetErrorString(err));
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
                    return set_error(cudaGetErrorString(err));
                }
            }

            err = cudaStreamSynchronize(work_stream);
            if (err != cudaSuccess)
            {
                release();
                return set_error(cudaGetErrorString(err));
            }

            *out_max_coeff_bits = static_cast<uint16_t>(h_max_bits);
            *out_bytes_per_coeff = static_cast<uint16_t>(h_bytes_per_coeff);
            *out_payload_len = payload_len;
            release();
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_store_compact_bytes");
        }
    }
} // extern "C"
