#include "Runtime.cuh"

#include <stdexcept>
#include <string>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/RNSPoly.cuh"
#include "LimbUtils.cuh"

namespace
{
    thread_local std::string g_last_error;

    int runtime_set_error(const char *msg)
    {
        g_last_error = msg ? msg : "unknown error";
        return 1;
    }

    int runtime_set_error(cudaError_t err)
    {
        return runtime_set_error(cudaGetErrorString(err));
    }

    void destroy_event_set_impl(GpuEventSet *events)
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

extern "C" int gpu_set_last_error(const char *msg)
{
    return runtime_set_error(msg);
}

extern "C" int gpu_context_create(
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
            return runtime_set_error("invalid context arguments");
        }
        if (moduli_len != static_cast<size_t>(L + 1))
        {
            return runtime_set_error("moduli_len must equal L + 1");
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
            std::move(inverse_table),
            {}};
        *out_ctx = gpu_ctx;
        return 0;
    }
    catch (const std::exception &e)
    {
        return runtime_set_error(e.what());
    }
    catch (...)
    {
        return runtime_set_error("unknown exception in gpu_context_create");
    }
}

extern "C" void gpu_context_destroy(GpuContext *ctx)
{
    if (!ctx)
    {
        return;
    }
    delete ctx->ctx;
    delete ctx;
}

extern "C" int gpu_context_get_N(const GpuContext *ctx, int *out_N)
{
    if (!ctx || !out_N)
    {
        return runtime_set_error("invalid gpu_context_get_N arguments");
    }
    *out_N = ctx->N;
    return 0;
}

extern "C" int gpu_event_set_wait(GpuEventSet *events)
{
    if (!events)
    {
        return runtime_set_error("invalid gpu_event_set_wait arguments");
    }
    for (const auto &entry : events->entries)
    {
        cudaError_t err = cudaSetDevice(entry.device);
        if (err != cudaSuccess)
        {
            return runtime_set_error(err);
        }
        err = cudaEventSynchronize(entry.event);
        if (err != cudaSuccess)
        {
            return runtime_set_error(err);
        }
    }
    return 0;
}

extern "C" void gpu_event_set_destroy(GpuEventSet *events)
{
    destroy_event_set_impl(events);
}

extern "C" int gpu_device_count(int *out_count)
{
    if (!out_count)
    {
        return runtime_set_error("invalid gpu_device_count arguments");
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
        return runtime_set_error(err);
    }
    *out_count = count;
    return 0;
}

extern "C" int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total)
{
    if (!out_free || !out_total)
    {
        return runtime_set_error("invalid gpu_device_mem_info arguments");
    }
    int current = 0;
    cudaError_t err = cudaGetDevice(&current);
    if (err != cudaSuccess)
    {
        return runtime_set_error(err);
    }
    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return runtime_set_error(err);
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    err = cudaMemGetInfo(&free_bytes, &total_bytes);
    cudaError_t restore_err = cudaSetDevice(current);
    if (err != cudaSuccess)
    {
        return runtime_set_error(err);
    }
    if (restore_err != cudaSuccess)
    {
        return runtime_set_error(restore_err);
    }
    *out_free = free_bytes;
    *out_total = total_bytes;
    return 0;
}

extern "C" int gpu_device_synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        return runtime_set_error(err);
    }
    return 0;
}

extern "C" int gpu_device_reset()
{
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        return runtime_set_error(err);
    }
    return 0;
}

extern "C" const char *gpu_last_error()
{
    return g_last_error.c_str();
}

extern "C" void *gpu_pinned_alloc(size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }
    void *ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess)
    {
        runtime_set_error(err);
        return nullptr;
    }
    return ptr;
}

extern "C" void gpu_pinned_free(void *ptr)
{
    if (!ptr)
    {
        return;
    }
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess)
    {
        runtime_set_error(err);
    }
}
