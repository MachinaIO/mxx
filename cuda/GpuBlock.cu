#include "GpuBlock.h"
#include "GpuPolyInternal.h"

#include <exception>
#include <type_traits>
#include <vector>

namespace
{
    enum class BlockOp
    {
        Add,
        Sub,
        Mul,
    };

    int set_error(const char *msg)
    {
        return gpu_set_last_error(msg);
    }

    int set_error(cudaError_t err)
    {
        return gpu_set_last_error(cudaGetErrorString(err));
    }

    template <typename T>
    __global__ void block_add_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        T sum = a + b;
        if (sum >= modulus)
        {
            sum -= modulus;
        }
        out[poly_idx][coeff_idx] = sum;
    }

    template <typename T>
    __global__ void block_sub_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        T diff = a >= b ? (a - b) : (modulus - (b - a));
        out[poly_idx][coeff_idx] = diff;
    }

    __device__ __forceinline__ uint32_t mul_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
    {
        uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
        return static_cast<uint32_t>(prod % mod);
    }

    __device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
    {
        unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % mod);
    }

    template <typename T>
    __global__ void block_mul_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            out[poly_idx][coeff_idx] = mul_mod_u64(a, b, modulus);
        }
        else
        {
            out[poly_idx][coeff_idx] = mul_mod_u32(a, b, modulus);
        }
    }

    template <typename T>
    int launch_block_kernel(
        const std::vector<T *> &out_ptrs,
        const std::vector<const T *> &lhs_ptrs,
        const std::vector<const T *> &rhs_ptrs,
        size_t n,
        T modulus,
        BlockOp op,
        cudaStream_t stream)
    {
        const size_t count = out_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }

        T **d_out = nullptr;
        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        const size_t bytes = count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_out, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_lhs, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            return set_error(err);
        }
        err = cudaMalloc(&d_rhs, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_out, out_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        switch (op)
        {
        case BlockOp::Add:
            block_add_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        case BlockOp::Sub:
            block_sub_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        case BlockOp::Mul:
            block_mul_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        cudaFree(d_out);
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        return 0;
    }

    template <typename T>
    int launch_for_limb(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t count,
        size_t n,
        int limb,
        const dim3 &limb_id,
        BlockOp op)
    {
        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(count);
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);

        cudaStream_t stream = nullptr;
        for (size_t i = 0; i < count; ++i)
        {
            auto &lhs_partition = lhs[i]->poly->GPU[limb_id.x];
            auto &rhs_partition = rhs[i]->poly->GPU[limb_id.x];
            auto &out_partition = out[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= lhs_partition.limb.size() || limb_id.y >= rhs_partition.limb.size() ||
                limb_id.y >= out_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_op");
            }
            auto &lhs_limb = lhs_partition.limb[limb_id.y];
            auto &rhs_limb = rhs_partition.limb[limb_id.y];
            auto &out_limb = out_partition.limb[limb_id.y];
            if (lhs_limb.index() != rhs_limb.index() || lhs_limb.index() != out_limb.index())
            {
                return set_error("mixed limb types in gpu_block_op");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &lhs_u = std::get<FIDESlib::U64>(lhs_limb);
                auto &rhs_u = std::get<FIDESlib::U64>(rhs_limb);
                auto &out_u = std::get<FIDESlib::U64>(out_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
                rhs_ptrs.push_back(rhs_u.v.data);
                out_ptrs.push_back(out_u.v.data);
            }
            else
            {
                auto &lhs_u = std::get<FIDESlib::U32>(lhs_limb);
                auto &rhs_u = std::get<FIDESlib::U32>(rhs_limb);
                auto &out_u = std::get<FIDESlib::U32>(out_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
                rhs_ptrs.push_back(rhs_u.v.data);
                out_ptrs.push_back(out_u.v.data);
            }
        }

        const uint64_t modulus64 = lhs[0]->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, n, modulus, op, stream);
    }

    int gpu_block_op(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count, BlockOp op)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid gpu_block_op arguments");
        }
        if (count == 0)
        {
            return 0;
        }

        const GpuPoly *lhs0 = lhs[0];
        if (!lhs0 || !lhs0->ctx)
        {
            return set_error("null context in gpu_block_op");
        }
        const GpuContext *ctx = lhs0->ctx;
        const int level = lhs0->level;
        const PolyFormat format = lhs0->format;
        if (op == BlockOp::Mul && format != PolyFormat::Eval)
        {
            return set_error("gpu_block_entrywise_mul requires Eval format");
        }

        // for (size_t i = 0; i < count; ++i)
        // {
        //     if (!out[i] || !lhs[i] || !rhs[i])
        //     {
        //         return set_error("null polynomial in gpu_block_op");
        //     }
        //     if (lhs[i]->ctx != ctx || rhs[i]->ctx != ctx || out[i]->ctx != ctx)
        //     {
        //         return set_error("mismatched contexts in gpu_block_op");
        //     }
        //     if (lhs[i]->level != level || rhs[i]->level != level || out[i]->level != level)
        //     {
        //         return set_error("mismatched levels in gpu_block_op");
        //     }
        //     if (lhs[i]->format != format || rhs[i]->format != format)
        //     {
        //         return set_error("mismatched formats in gpu_block_op");
        //     }
        // }

        if (level < 0)
        {
            return set_error("invalid level in gpu_block_op");
        }

        const int N = ctx->N;
        if (N <= 0)
        {
            return 0;
        }

        auto &limb_map = ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error("unexpected limb mapping size in gpu_block_op");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= lhs0->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_block_op");
            }
            const auto &partition = lhs0->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_op");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            const auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() == FIDESlib::U64)
            {
                int status = launch_for_limb<uint64_t>(out, lhs, rhs, count, static_cast<size_t>(N), limb, limb_id, op);
                if (status != 0)
                {
                    return status;
                }
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                int status = launch_for_limb<uint32_t>(out, lhs, rhs, count, static_cast<size_t>(N), limb, limb_id, op);
                if (status != 0)
                {
                    return status;
                }
            }
            else
            {
                return set_error("unsupported limb type in gpu_block_op");
            }
        }

        for (size_t i = 0; i < count; ++i)
        {
            out[i]->level = level;
            out[i]->format = format;
        }
        return 0;
    }
} // namespace

extern "C" int gpu_block_add(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Add);
}

extern "C" int gpu_block_sub(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Sub);
}

extern "C" int gpu_block_entrywise_mul(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Mul);
}
