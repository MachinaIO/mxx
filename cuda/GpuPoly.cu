#include "GpuPolyInternal.h"

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <cuda_runtime.h>

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

    void ensure_eval(GpuPoly *poly, int batch)
    {
        if (poly->format == PolyFormat::Coeff)
        {
            poly->poly->NTT(batch);
            poly->format = PolyFormat::Eval;
        }
    }

    void ensure_coeff(GpuPoly *poly, int batch)
    {
        if (poly->format == PolyFormat::Eval)
        {
            poly->poly->INTT(batch);
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

            auto *ctx = new CKKS::Context(params, gpu_list, 0);
            auto *gpu_ctx = new GpuContext{ctx, std::vector<uint64_t>(moduli, moduli + moduli_len), 1 << logN, gpu_list, batch};
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

    int gpu_poly_copy(GpuPoly *dst, const GpuPoly *src)
    {
        try
        {
            if (!dst || !src)
            {
                return set_error("invalid gpu_poly_copy arguments");
            }
            dst->poly->copy(*src->poly);
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

    int gpu_poly_add(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_add arguments");
            }

            const int batch = default_batch(a->ctx);
            const PolyFormat desired_format = a->format;
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != desired_format)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, desired_format, batch);
                b_poly = b_tmp.get();
            }

            if (out == a && out->level <= b->level && out->format == desired_format)
            {
                out->poly->add(*b_poly);
            }
            else if (out->level == a->level && out->level <= b->level && out->format == desired_format)
            {
                out->poly->add(*a->poly, *b_poly);
            }
            else
            {
                out->poly->copy(*a->poly);
                out->poly->add(*b_poly);
            }
            out->level = a->level;
            out->format = desired_format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_add");
        }
    }

    int gpu_poly_sub(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_sub arguments");
            }

            const int batch = default_batch(a->ctx);
            const PolyFormat desired_format = a->format;
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != desired_format)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, desired_format, batch);
                b_poly = b_tmp.get();
            }

            if (out == a && out->level <= b->level && out->format == desired_format)
            {
                out->poly->sub(*b_poly);
            }
            else
            {
                out->poly->copy(*a->poly);
                out->poly->sub(*b_poly);
            }
            out->level = a->level;
            out->format = desired_format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_sub");
        }
    }

    int gpu_poly_mul(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_mul arguments");
            }
            const int batch = default_batch(a->ctx);
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != PolyFormat::Eval)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, PolyFormat::Eval, batch);
                b_poly = b_tmp.get();
            }

            if (out != a)
            {
                out->poly->copy(*a->poly);
                out->level = a->level;
                out->format = a->format;
            }
            ensure_eval(out, batch);
            out->poly->multElement(*b_poly);
            out->level = a->level;
            out->format = PolyFormat::Eval;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_mul");
        }
    }

    int gpu_poly_ntt(GpuPoly *poly, int batch)
    {
        try
        {
            if (!poly)
            {
                return set_error("invalid gpu_poly_ntt arguments");
            }
            ensure_eval(poly, batch);
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_ntt");
        }
    }

    int gpu_poly_intt(GpuPoly *poly, int batch)
    {
        try
        {
            if (!poly)
            {
                return set_error("invalid gpu_poly_intt arguments");
            }
            ensure_coeff(poly, batch);
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_intt");
        }
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
