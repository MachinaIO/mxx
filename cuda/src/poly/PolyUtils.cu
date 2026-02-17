namespace
{
    int set_error(const char *msg)
    {
        return gpu_set_last_error(msg ? msg : "unknown error");
    }

    int set_error(const std::exception &e)
    {
        return gpu_set_last_error(e.what());
    }
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

}

extern "C"
{
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
} // extern "C"
