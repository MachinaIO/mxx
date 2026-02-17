namespace
{
    thread_local std::string last_error;

    int set_error(const char *msg)
    {
        last_error = msg ? msg : "unknown error";
        return 1;
    }

    int set_error(const std::exception &e)
    {
        last_error = e.what();
        return 1;
    }
}

extern "C" int gpu_set_last_error(const char *msg)
{
    return set_error(msg);
}

namespace
{
    int default_batch(const GpuContext *ctx)
    {
        return static_cast<int>(ctx && ctx->batch != 0 ? ctx->batch : 1);
    }

    bool parse_format(int format, PolyFormat &out)
    {
        switch (format)
        {
        case GPU_POLY_FORMAT_COEFF:
            out = PolyFormat::Coeff;
            return true;
        case GPU_POLY_FORMAT_EVAL:
            out = PolyFormat::Eval;
            return true;
        default:
            return false;
        }
    }

    size_t expected_rns_len(const GpuPoly *poly)
    {
        const int level = poly->level;
        const int N = poly->ctx->N;
        return static_cast<size_t>(level + 1) * static_cast<size_t>(N);
    }

    void propagate_partition_stream_to_limbs(CKKS::RNSPoly *poly)
    {
        for (auto &partition : poly->GPU)
        {
            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(err));
            }
            for (auto &limb_impl : partition.limb)
            {
                cudaStream_t limb_stream = nullptr;
                if (limb_impl.index() == FIDESlib::U64)
                {
                    limb_stream = std::get<FIDESlib::U64>(limb_impl).stream.ptr;
                }
                else if (limb_impl.index() == FIDESlib::U32)
                {
                    limb_stream = std::get<FIDESlib::U32>(limb_impl).stream.ptr;
                }
                if (!limb_stream || limb_stream == partition.s.ptr)
                {
                    continue;
                }

                cudaEvent_t ready = nullptr;
                err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(ready, partition.s.ptr);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ready);
                    throw std::runtime_error(cudaGetErrorString(err));
                }
                err = cudaStreamWaitEvent(limb_stream, ready, 0);
                cudaError_t destroy_err = cudaEventDestroy(ready);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
                if (destroy_err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(destroy_err));
                }
            }
        }
    }

    void ensure_eval(GpuPoly *poly, int batch)
    {
        std::lock_guard<std::mutex> guard(poly->ctx->transform_mutex);
        if (poly->format == PolyFormat::Coeff)
        {
            poly->poly->NTT(batch);
            propagate_partition_stream_to_limbs(poly->poly);
            poly->format = PolyFormat::Eval;
        }
    }

    void ensure_coeff(GpuPoly *poly, int batch)
    {
        std::lock_guard<std::mutex> guard(poly->ctx->transform_mutex);
        if (poly->format == PolyFormat::Eval)
        {
            poly->poly->INTT(batch);
            propagate_partition_stream_to_limbs(poly->poly);
            poly->format = PolyFormat::Coeff;
        }
    }

    void convert_format(CKKS::RNSPoly &poly, PolyFormat from, PolyFormat to, int batch)
    {
        if (from == to)
        {
            return;
        }
        if (to == PolyFormat::Eval)
        {
            poly.NTT(batch);
        }
        else
        {
            poly.INTT(batch);
        }
    }

    void destroy_event_set(GpuEventSet *events)
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

    uint32_t bit_width_u64(uint64_t v)
    {
        if (v == 0)
        {
            return 0;
        }
        return static_cast<uint32_t>(64 - __builtin_clzll(v));
    }

    constexpr int kMaxRnsLimbs = 64;
    constexpr int kMaxCoeffWords = 64;

    bool mod_inverse_u64(uint64_t a, uint64_t modulus, uint64_t &out_inv)
    {
        if (modulus == 0)
        {
            return false;
        }
        __int128 t = 0;
        __int128 new_t = 1;
        __int128 r = static_cast<__int128>(modulus);
        __int128 new_r = static_cast<__int128>(a % modulus);
        while (new_r != 0)
        {
            const __int128 q = r / new_r;

            const __int128 tmp_t = t - q * new_t;
            t = new_t;
            new_t = tmp_t;

            const __int128 tmp_r = r - q * new_r;
            r = new_r;
            new_r = tmp_r;
        }
        if (r != 1)
        {
            return false;
        }
        if (t < 0)
        {
            t += static_cast<__int128>(modulus);
        }
        out_inv = static_cast<uint64_t>(t);
        return true;
    }

    std::vector<uint64_t> compute_garner_inverse_table(const std::vector<uint64_t> &moduli, int limb_count)
    {
        const size_t count = static_cast<size_t>(limb_count);
        std::vector<uint64_t> inverse_table(count * count, 0);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[static_cast<size_t>(i)];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t qj = moduli[static_cast<size_t>(j)];
                uint64_t inv = 0;
                if (!mod_inverse_u64(qj % qi, qi, inv))
                {
                    throw std::runtime_error("CRT moduli must be pairwise coprime");
                }
                inverse_table[static_cast<size_t>(j) * count + static_cast<size_t>(i)] = inv;
            }
        }
        return inverse_table;
    }
}

extern "C"
{
    int gpu_context_create(
        uint32_t logN,
        uint32_t L,
        uint32_t dnum,
        const uint64_t *moduli,
        size_t moduli_len,
        const int *gpu_ids,
        size_t gpu_ids_len,
        uint32_t batch,
        GpuContext **out_ctx)
    {
        try
        {
            if (!out_ctx || !moduli || moduli_len == 0)
            {
                return set_error("invalid context arguments");
            }
            if (moduli_len != static_cast<size_t>(L + 1))
            {
                return set_error("moduli_len must equal L + 1");
            }

            std::vector<int> gpu_list;
            if (gpu_ids_len == 0 || !gpu_ids)
            {
                gpu_list.push_back(0);
            }
            else
            {
                gpu_list.assign(gpu_ids, gpu_ids + gpu_ids_len);
            }

            CKKS::Parameters params;
            params.raw = nullptr;
            params.K = 0;
            params.L = static_cast<int>(moduli_len - 1);
            params.logN = logN;
            params.dnum = dnum == 0 ? static_cast<uint32_t>(gpu_list.size()) : dnum;
            params.batch = batch;
            const size_t level_len = static_cast<size_t>(params.L + 1);
            params.ModReduceFactor.assign(level_len, 1.0);
            params.ScalingFactorReal.assign(level_len, 1.0);
            params.ScalingFactorRealBig.assign(level_len, 1.0);
            params.scalingTechnique = lbcrypto::ScalingTechnique::FIXEDMANUAL;

            params.primes.clear();
            params.primes.reserve(moduli_len);
            for (size_t i = 0; i < moduli_len; ++i)
            {
                params.primes.push_back(FIDESlib::PrimeRecord{.p = moduli[i], .type = FIDESlib::U64});
            }
            params.Sprimes.clear();

            std::vector<uint64_t> moduli_vec(moduli, moduli + moduli_len);
            std::vector<uint64_t> inverse_table =
                compute_garner_inverse_table(moduli_vec, static_cast<int>(moduli_len));

            auto *ctx = new CKKS::Context(params, gpu_list, 0);
            auto *gpu_ctx = new GpuContext{
                ctx,
                std::move(moduli_vec),
                1 << logN,
                gpu_list,
                batch,
                std::move(inverse_table)};
            *out_ctx = gpu_ctx;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_context_create");
        }
    }

    void gpu_context_destroy(GpuContext *ctx)
    {
        if (!ctx)
        {
            return;
        }
        delete ctx->ctx;
        delete ctx;
    }

    int gpu_context_get_N(const GpuContext *ctx, int *out_N)
    {
        if (!ctx || !out_N)
        {
            return set_error("invalid gpu_context_get_N arguments");
        }
        *out_N = ctx->N;
        return 0;
    }

    int gpu_poly_create(GpuContext *ctx, int level, GpuPoly **out_poly)
    {
        try
        {
            if (!ctx || !out_poly)
            {
                return set_error("invalid gpu_poly_create arguments");
            }
            auto *poly = new CKKS::RNSPoly(*ctx->ctx, level, true);
            *out_poly = new GpuPoly{poly, ctx, level, PolyFormat::Eval};
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_create");
        }
    }

    void gpu_poly_destroy(GpuPoly *poly)
    {
        if (!poly)
        {
            return;
        }
        delete poly->poly;
        delete poly;
    }

    int gpu_poly_clone(const GpuPoly *src, GpuPoly **out_poly)
    {
        try
        {
            if (!src || !out_poly)
            {
                return set_error("invalid gpu_poly_clone arguments");
            }
            auto *poly = new CKKS::RNSPoly(src->poly->clone());
            *out_poly = new GpuPoly{poly, src->ctx, src->level, src->format};
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_clone");
        }
    }

    int gpu_poly_clone_async(const GpuPoly *src, GpuPoly **out_poly, GpuEventSet **out_events)
    {
        try
        {
            if (!src || !out_poly || !out_events)
            {
                return set_error("invalid gpu_poly_clone_async arguments");
            }
            *out_poly = nullptr;
            *out_events = nullptr;

            auto *poly = new CKKS::RNSPoly(src->poly->clone());
            auto *gpu_poly = new GpuPoly{poly, src->ctx, src->level, src->format};
            auto *event_set = new GpuEventSet();
            event_set->entries.reserve(gpu_poly->poly->GPU.size());

            for (auto &partition : gpu_poly->poly->GPU)
            {
                cudaError_t err = cudaSetDevice(partition.device);
                if (err != cudaSuccess)
                {
                    destroy_event_set(event_set);
                    gpu_poly_destroy(gpu_poly);
                    return set_error(cudaGetErrorString(err));
                }

                cudaEvent_t ev = nullptr;
                err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    destroy_event_set(event_set);
                    gpu_poly_destroy(gpu_poly);
                    return set_error(cudaGetErrorString(err));
                }
                err = cudaEventRecord(ev, partition.s.ptr);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(ev);
                    destroy_event_set(event_set);
                    gpu_poly_destroy(gpu_poly);
                    return set_error(cudaGetErrorString(err));
                }
                event_set->entries.push_back(GpuEventSet::Entry{ev, partition.device});
            }

            *out_poly = gpu_poly;
            *out_events = event_set;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_clone_async");
        }
    }

    int gpu_poly_copy(GpuPoly *dst, const GpuPoly *src)
    {
        try
        {
            if (!dst || !src)
            {
                return set_error("invalid gpu_poly_copy arguments");
            }
            dst->poly->copy(*src->poly);
            // copy() writes via partition stream; propagate completion to limb streams
            // so the next limb-stream consumer observes copied data.
            propagate_partition_stream_to_limbs(dst->poly);
            dst->level = src->level;
            dst->format = src->format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_copy");
        }
    }

    int gpu_poly_get_level(const GpuPoly *poly, int *out_level)
    {
        if (!poly || !out_level)
        {
            return set_error("invalid gpu_poly_get_level arguments");
        }
        *out_level = poly->level;
        return 0;
    }
    int gpu_event_set_wait(GpuEventSet *events)
    {
        if (!events)
        {
            return set_error("invalid gpu_event_set_wait arguments");
        }
        for (const auto &entry : events->entries)
        {
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }
            err = cudaEventSynchronize(entry.event);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }
        }
        return 0;
    }

    void gpu_event_set_destroy(GpuEventSet *events)
    {
        destroy_event_set(events);
    }
    int gpu_device_count(int *out_count)
    {
        if (!out_count)
        {
            return set_error("invalid gpu_device_count arguments");
        }
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice)
        {
            *out_count = 0;
            return 0;
        }
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        *out_count = count;
        return 0;
    }

    int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total)
    {
        if (!out_free || !out_total)
        {
            return set_error("invalid gpu_device_mem_info arguments");
        }
        int current = 0;
        cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        cudaError_t restore_err = cudaSetDevice(current);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        if (restore_err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(restore_err));
        }
        *out_free = free_bytes;
        *out_total = total_bytes;
        return 0;
    }

    int gpu_device_synchronize()
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        return 0;
    }

    int gpu_device_reset()
    {
        cudaError_t err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        return 0;
    }

    const char *gpu_last_error()
    {
        return last_error.c_str();
    }

    void *gpu_pinned_alloc(size_t bytes)
    {
        try
        {
            if (bytes == 0)
            {
                return nullptr;
            }
            void *ptr = nullptr;
            cudaError_t err = cudaMallocHost(&ptr, bytes);
            if (err != cudaSuccess)
            {
                set_error(cudaGetErrorString(err));
                return nullptr;
            }
            return ptr;
        }
        catch (const std::exception &e)
        {
            set_error(e);
            return nullptr;
        }
        catch (...)
        {
            set_error("unknown exception in gpu_pinned_alloc");
            return nullptr;
        }
    }

    void gpu_pinned_free(void *ptr)
    {
        if (!ptr)
        {
            return;
        }
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess)
        {
            set_error(cudaGetErrorString(err));
        }
    }
} // extern "C"
