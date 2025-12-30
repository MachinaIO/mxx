#include "GpuPoly.h"

#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// FIDESlib headers (expected under third_party/FIDESlib/include).
#include "../third_party/FIDESlib/include/CKKS/Context.cuh"
#include "../third_party/FIDESlib/include/CKKS/Parameters.cuh"
#include "../third_party/FIDESlib/include/LimbUtils.cuh"
#include "../third_party/FIDESlib/include/CKKS/RNSPoly.cuh"

namespace CKKS = FIDESlib::CKKS;

enum class PolyFormat
{
    Coeff,
    Eval,
};

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

struct GpuContext
{
    CKKS::Context *ctx;
    std::vector<uint64_t> moduli;
    int N;
    std::vector<int> gpu_ids;
    uint32_t batch;
};

struct GpuPoly
{
    CKKS::RNSPoly *poly;
    GpuContext *ctx;
    int level;
    PolyFormat format;
};

namespace
{
    int default_batch(const GpuContext *ctx)
    {
        return static_cast<int>(ctx && ctx->batch != 0 ? ctx->batch : 1);
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

    int gpu_poly_load_rns(GpuPoly *poly, const uint64_t *coeffs_flat, size_t coeffs_len)
    {
        try
        {
            if (!poly || !coeffs_flat)
            {
                return set_error("invalid gpu_poly_load_rns arguments");
            }
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (coeffs_len != expected)
            {
                return set_error("coeffs_len mismatch in gpu_poly_load_rns");
            }

            std::vector<std::vector<uint64_t>> data(level + 1, std::vector<uint64_t>(N));
            for (int limb = 0; limb <= level; ++limb)
            {
                for (int i = 0; i < N; ++i)
                {
                    data[limb][i] = coeffs_flat[static_cast<size_t>(limb) * N + i];
                }
            }

            std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + level + 1);
            poly->poly->load(data, moduli_subset);
            poly->format = PolyFormat::Coeff;
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

    int gpu_poly_store_rns(GpuPoly *poly, uint64_t *coeffs_flat_out, size_t coeffs_len)
    {
        try
        {
            if (!poly || !coeffs_flat_out)
            {
                return set_error("invalid gpu_poly_store_rns arguments");
            }
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (coeffs_len != expected)
            {
                return set_error("coeffs_len mismatch in gpu_poly_store_rns");
            }

            const int batch = default_batch(poly->ctx);
            ensure_coeff(poly, batch);

            std::vector<std::vector<uint64_t>> data;
            poly->poly->store(data);
            if (data.size() != static_cast<size_t>(level + 1))
            {
                return set_error("unexpected RNS limb count in gpu_poly_store_rns");
            }

            for (int limb = 0; limb <= level; ++limb)
            {
                if (data[limb].size() != static_cast<size_t>(N))
                {
                    return set_error("unexpected limb size in gpu_poly_store_rns");
                }
                for (int i = 0; i < N; ++i)
                {
                    coeffs_flat_out[static_cast<size_t>(limb) * N + i] = data[limb][i];
                }
            }

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
