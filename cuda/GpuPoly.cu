#include "GpuPoly.h"

#include <exception>
#include <new>
#include <string>
#include <vector>

// FIDESlib headers (expected under third_party/FIDESlib/include).
#include "../third_party/FIDESlib/include/CKKS/Context.cuh"
#include "../third_party/FIDESlib/include/CKKS/Parameters.cuh"
#include "../third_party/FIDESlib/include/LimbUtils.cuh"
#include "../third_party/FIDESlib/include/CKKS/RNSPoly.cuh"

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
};

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
            params.ModReduceFactor = 1.0;
            params.ScalingFactorReal = 1.0;
            params.ScalingFactorRealBig = 1.0;
            params.scalingTechnique = CKKS::ScalingTechnique::FIXEDMANUAL;

            params.primes.clear();
            params.primes.reserve(moduli_len);
            for (size_t i = 0; i < moduli_len; ++i)
            {
                params.primes.emplace_back(moduli[i]);
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
            *out_poly = new GpuPoly{poly, ctx, level};
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
            auto *poly = new CKKS::RNSPoly(*src->poly);
            *out_poly = new GpuPoly{poly, src->ctx, src->level};
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
            *dst->poly = *src->poly;
            dst->level = src->level;
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

    int gpu_poly_store_rns(const GpuPoly *poly, uint64_t *coeffs_flat_out, size_t coeffs_len)
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
            *out->poly = *a->poly;
            out->poly->add(*b->poly);
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
            *out->poly = *a->poly;
            out->poly->sub(*b->poly);
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
            CKKS::RNSPoly tmpA(*a->poly);
            CKKS::RNSPoly tmpB(*b->poly);
            const int batch = static_cast<int>(a->ctx->batch == 0 ? 1 : a->ctx->batch);
            tmpA.NTT(batch);
            tmpB.NTT(batch);
            tmpA.multElement(tmpB);
            tmpA.INTT(batch);
            *out->poly = tmpA;
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
            poly->poly->NTT(batch);
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
            poly->poly->INTT(batch);
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

} // extern "C"
