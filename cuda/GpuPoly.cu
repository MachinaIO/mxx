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

    __global__ void u32_to_u64_kernel(const uint32_t *src, uint64_t *dst, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            dst[idx] = static_cast<uint64_t>(src[idx]);
        }
    }

    __global__ void u64_to_u32_kernel(const uint64_t *src, uint32_t *dst, size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            dst[idx] = static_cast<uint32_t>(src[idx]);
        }
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
    std::vector<uint64_t *> scratch_u64;
    std::vector<int> scratch_device;
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

    uint64_t *ensure_scratch_u64(GpuPoly *poly, size_t limb_idx, size_t n, int device)
    {
        if (limb_idx >= poly->scratch_u64.size())
        {
            poly->scratch_u64.resize(limb_idx + 1, nullptr);
            poly->scratch_device.resize(limb_idx + 1, -1);
        }
        if (poly->scratch_u64[limb_idx] == nullptr || poly->scratch_device[limb_idx] != device)
        {
            if (poly->scratch_u64[limb_idx] != nullptr)
            {
                int old_device = poly->scratch_device[limb_idx];
                if (old_device >= 0)
                {
                    cudaSetDevice(old_device);
                }
                cudaFree(poly->scratch_u64[limb_idx]);
            }
            cudaSetDevice(device);
            cudaMalloc(&poly->scratch_u64[limb_idx], n * sizeof(uint64_t));
            poly->scratch_device[limb_idx] = device;
        }
        return poly->scratch_u64[limb_idx];
    }

    void store_limb_async(GpuPoly *poly, CKKS::LimbImpl &limb, uint64_t *out, size_t n, size_t limb_idx)
    {
        if (limb.index() == FIDESlib::U64)
        {
            auto &l = std::get<CKKS::Limb<uint64_t>>(limb);
            cudaSetDevice(l.v.device);
            cudaMemcpyAsync(out, l.v.data, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, l.stream.ptr);
            return;
        }

        auto &l = std::get<CKKS::Limb<uint32_t>>(limb);
        cudaSetDevice(l.v.device);
        uint64_t *scratch = ensure_scratch_u64(poly, limb_idx, n, l.v.device);
        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        u32_to_u64_kernel<<<grid, block, 0, l.stream.ptr>>>(l.v.data, scratch, n);
        cudaMemcpyAsync(out, scratch, n * sizeof(uint64_t), cudaMemcpyDeviceToHost, l.stream.ptr);
    }

    void load_limb_async(GpuPoly *poly, CKKS::LimbImpl &limb, const uint64_t *in, size_t n, size_t limb_idx)
    {
        if (limb.index() == FIDESlib::U64)
        {
            auto &l = std::get<CKKS::Limb<uint64_t>>(limb);
            cudaSetDevice(l.v.device);
            cudaMemcpyAsync(l.v.data, in, n * sizeof(uint64_t), cudaMemcpyHostToDevice, l.stream.ptr);
            return;
        }

        auto &l = std::get<CKKS::Limb<uint32_t>>(limb);
        cudaSetDevice(l.v.device);
        uint64_t *scratch = ensure_scratch_u64(poly, limb_idx, n, l.v.device);
        cudaMemcpyAsync(scratch, in, n * sizeof(uint64_t), cudaMemcpyHostToDevice, l.stream.ptr);
        const int block = 256;
        const int grid = static_cast<int>((n + block - 1) / block);
        u64_to_u32_kernel<<<grid, block, 0, l.stream.ptr>>>(scratch, l.v.data, n);
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

    int gpu_pinned_alloc(size_t bytes, void **out_ptr)
    {
        try
        {
            if (!out_ptr || bytes == 0)
            {
                return set_error("invalid gpu_pinned_alloc arguments");
            }
            void *ptr = nullptr;
            cudaError_t err = cudaMallocHost(&ptr, bytes);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }
            *out_ptr = ptr;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_pinned_alloc");
        }
    }

    int gpu_pinned_free(void *ptr)
    {
        try
        {
            if (!ptr)
            {
                return 0;
            }
            cudaError_t err = cudaFreeHost(ptr);
            if (err != cudaSuccess)
            {
                return set_error(cudaGetErrorString(err));
            }
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_pinned_free");
        }
    }

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
            auto *gpu_poly = new GpuPoly{
                poly,
                ctx,
                level,
                PolyFormat::Eval,
                std::vector<uint64_t *>(static_cast<size_t>(level + 1), nullptr),
                std::vector<int>(static_cast<size_t>(level + 1), -1),
            };
            *out_poly = gpu_poly;
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
        for (size_t i = 0; i < poly->scratch_u64.size(); ++i)
        {
            if (poly->scratch_u64[i] != nullptr)
            {
                int device = poly->scratch_device[i];
                if (device >= 0)
                {
                    cudaSetDevice(device);
                }
                cudaFree(poly->scratch_u64[i]);
                poly->scratch_u64[i] = nullptr;
                poly->scratch_device[i] = -1;
            }
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
            *out_poly = new GpuPoly{
                poly,
                src->ctx,
                src->level,
                src->format,
                std::vector<uint64_t *>(static_cast<size_t>(src->level + 1), nullptr),
                std::vector<int>(static_cast<size_t>(src->level + 1), -1),
            };
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

            int status = gpu_poly_load_rns_async(poly, coeffs_flat, coeffs_len);
            if (status != 0)
            {
                return status;
            }
            poly->poly->sync();
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

    int gpu_poly_load_rns_async(GpuPoly *poly, const uint64_t *coeffs_flat, size_t coeffs_len)
    {
        try
        {
            if (!poly || !coeffs_flat)
            {
                return set_error("invalid gpu_poly_load_rns_async arguments");
            }
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (coeffs_len != expected)
            {
                return set_error("coeffs_len mismatch in gpu_poly_load_rns_async");
            }

            for (int limb = 0; limb <= level; ++limb)
            {
                const dim3 loc = poly->ctx->ctx->limbGPUid.at(static_cast<size_t>(limb));
                auto &partition = poly->poly->GPU.at(static_cast<size_t>(loc.x));
                auto &limb_variant = partition.limb.at(static_cast<size_t>(loc.y));
                load_limb_async(
                    poly,
                    limb_variant,
                    coeffs_flat + static_cast<size_t>(limb) * N,
                    static_cast<size_t>(N),
                    static_cast<size_t>(limb)
                );
            }

            poly->format = PolyFormat::Coeff;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_load_rns_async");
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

    int gpu_poly_store_rns_async(GpuPoly *poly, uint64_t *coeffs_flat_out, size_t coeffs_len)
    {
        try
        {
            if (!poly || !coeffs_flat_out)
            {
                return set_error("invalid gpu_poly_store_rns_async arguments");
            }
            const int level = poly->level;
            const int N = poly->ctx->N;
            const size_t expected = static_cast<size_t>(level + 1) * static_cast<size_t>(N);
            if (coeffs_len != expected)
            {
                return set_error("coeffs_len mismatch in gpu_poly_store_rns_async");
            }

            const int batch = default_batch(poly->ctx);
            ensure_coeff(poly, batch);

            for (int limb = 0; limb <= level; ++limb)
            {
                const dim3 loc = poly->ctx->ctx->limbGPUid.at(static_cast<size_t>(limb));
                auto &partition = poly->poly->GPU.at(static_cast<size_t>(loc.x));
                auto &limb_variant = partition.limb.at(static_cast<size_t>(loc.y));
                store_limb_async(
                    poly,
                    limb_variant,
                    coeffs_flat_out + static_cast<size_t>(limb) * N,
                    static_cast<size_t>(N),
                    static_cast<size_t>(limb)
                );
            }

            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_store_rns_async");
        }
    }

    int gpu_poly_sync(GpuPoly *poly)
    {
        try
        {
            if (!poly)
            {
                return set_error("invalid gpu_poly_sync arguments");
            }
            poly->poly->sync();
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_sync");
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

} // extern "C"
