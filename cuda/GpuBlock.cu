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

    constexpr int kMatmulTileM = 16;
    constexpr int kMatmulTileN = 16;
    constexpr int kMatmulTileK = 8;

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

    __device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
    {
        uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
        if (sum >= mod)
        {
            sum -= mod;
        }
        return static_cast<uint32_t>(sum);
    }

    __device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
    {
        unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
        if (sum >= mod)
        {
            sum -= mod;
        }
        return static_cast<uint64_t>(sum);
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
    __global__ void block_matmul_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        T modulus)
    {
        __shared__ T lhs_tile[kMatmulTileM][kMatmulTileK];
        __shared__ T rhs_tile[kMatmulTileK][kMatmulTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t coeff_idx = static_cast<size_t>(blockIdx.z);
        if (coeff_idx >= n)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;

        T acc = 0;
        for (size_t k0 = 0; k0 < inner; k0 += kMatmulTileK)
        {
            for (int i = tid; i < kMatmulTileM * kMatmulTileK; i += threads)
            {
                const int r = i / kMatmulTileK;
                const int k = i - r * kMatmulTileK;
                const size_t lhs_row = row_base + static_cast<size_t>(r);
                const size_t lhs_k = k0 + static_cast<size_t>(k);
                T val = 0;
                if (lhs_row < rows && lhs_k < inner)
                {
                    const T *lhs_poly = lhs[lhs_row * inner + lhs_k];
                    val = lhs_poly[coeff_idx];
                }
                lhs_tile[r][k] = val;
            }
            for (int i = tid; i < kMatmulTileK * kMatmulTileN; i += threads)
            {
                const int k = i / kMatmulTileN;
                const int c = i - k * kMatmulTileN;
                const size_t rhs_k = k0 + static_cast<size_t>(k);
                const size_t rhs_col = col_base + static_cast<size_t>(c);
                T val = 0;
                if (rhs_k < inner && rhs_col < cols)
                {
                    const T *rhs_poly = rhs[rhs_k * cols + rhs_col];
                    val = rhs_poly[coeff_idx];
                }
                rhs_tile[k][c] = val;
            }
            __syncthreads();

            if (row < rows && col < cols)
            {
                for (int kk = 0; kk < kMatmulTileK; ++kk)
                {
                    T prod;
                    if constexpr (std::is_same_v<T, uint64_t>)
                    {
                        prod = mul_mod_u64(lhs_tile[threadIdx.y][kk], rhs_tile[kk][threadIdx.x], modulus);
                        acc = add_mod_u64(acc, prod, modulus);
                    }
                    else
                    {
                        prod = mul_mod_u32(lhs_tile[threadIdx.y][kk], rhs_tile[kk][threadIdx.x], modulus);
                        acc = add_mod_u32(acc, prod, modulus);
                    }
                }
            }
            __syncthreads();
        }

        if (row < rows && col < cols)
        {
            out[row * cols + col][coeff_idx] = acc;
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
    int launch_block_matmul_kernel(
        const std::vector<T *> &out_ptrs,
        const std::vector<const T *> &lhs_ptrs,
        const std::vector<const T *> &rhs_ptrs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        T modulus,
        cudaStream_t stream)
    {
        const size_t out_count = rows * cols;
        const size_t lhs_count = rows * inner;
        const size_t rhs_count = inner * cols;
        if (out_count == 0 || n == 0)
        {
            return 0;
        }
        if (out_ptrs.size() != out_count || lhs_ptrs.size() != lhs_count || rhs_ptrs.size() != rhs_count)
        {
            return set_error("unexpected pointer counts in gpu_block_mul");
        }

        T **d_out = nullptr;
        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        const size_t out_bytes = out_count * sizeof(T *);
        const size_t lhs_bytes = lhs_count * sizeof(T *);
        const size_t rhs_bytes = rhs_count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_out, out_bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_lhs, lhs_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            return set_error(err);
        }
        err = cudaMalloc(&d_rhs, rhs_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_out, out_ptrs.data(), out_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), lhs_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), rhs_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        const dim3 threads(kMatmulTileN, kMatmulTileM);
        const dim3 blocks(
            static_cast<unsigned int>((cols + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(n));

        block_matmul_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, rows, inner, cols, n, modulus);

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

    template <typename T>
    int launch_for_limb_matmul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        int limb,
        const dim3 &limb_id)
    {
        const size_t out_count = rows * cols;
        const size_t lhs_count = rows * inner;
        const size_t rhs_count = inner * cols;

        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(out_count);
        lhs_ptrs.reserve(lhs_count);
        rhs_ptrs.reserve(rhs_count);

        cudaStream_t stream = nullptr;
        int limb_index = -1;

        for (size_t i = 0; i < lhs_count; ++i)
        {
            auto &lhs_partition = lhs[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= lhs_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &lhs_limb = lhs_partition.limb[limb_id.y];
            if constexpr (std::is_same_v<T, uint64_t>)
            {
                if (lhs_limb.index() != FIDESlib::U64)
                {
                    return set_error("mixed limb types in gpu_block_mul");
                }
                auto &lhs_u = std::get<FIDESlib::U64>(lhs_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
            }
            else
            {
                if (lhs_limb.index() != FIDESlib::U32)
                {
                    return set_error("mixed limb types in gpu_block_mul");
                }
                auto &lhs_u = std::get<FIDESlib::U32>(lhs_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
            }
            if (limb_index == -1)
            {
                limb_index = lhs_limb.index();
            }
        }

        for (size_t i = 0; i < rhs_count; ++i)
        {
            auto &rhs_partition = rhs[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= rhs_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &rhs_limb = rhs_partition.limb[limb_id.y];
            if (rhs_limb.index() != limb_index)
            {
                return set_error("mixed limb types in gpu_block_mul");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &rhs_u = std::get<FIDESlib::U64>(rhs_limb);
                rhs_ptrs.push_back(rhs_u.v.data);
            }
            else
            {
                auto &rhs_u = std::get<FIDESlib::U32>(rhs_limb);
                rhs_ptrs.push_back(rhs_u.v.data);
            }
        }

        for (size_t i = 0; i < out_count; ++i)
        {
            auto &out_partition = out[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= out_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &out_limb = out_partition.limb[limb_id.y];
            if (out_limb.index() != limb_index)
            {
                return set_error("mixed limb types in gpu_block_mul");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &out_u = std::get<FIDESlib::U64>(out_limb);
                out_ptrs.push_back(out_u.v.data);
            }
            else
            {
                auto &out_u = std::get<FIDESlib::U32>(out_limb);
                out_ptrs.push_back(out_u.v.data);
            }
        }

        const uint64_t modulus64 = lhs[0]->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_matmul_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, rows, inner, cols, n, modulus, stream);
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

    int gpu_block_matmul(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t rows, size_t inner, size_t cols)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid gpu_block_mul arguments");
        }
        const size_t out_count = rows * cols;
        if (rows == 0 || inner == 0 || cols == 0)
        {
            return 0;
        }
        if (out_count == 0)
        {
            return 0;
        }

        const GpuPoly *lhs0 = lhs[0];
        if (!lhs0 || !lhs0->ctx)
        {
            return set_error("null context in gpu_block_mul");
        }
        const GpuContext *ctx = lhs0->ctx;
        const int level = lhs0->level;
        const PolyFormat format = lhs0->format;
        if (format != PolyFormat::Eval)
        {
            return set_error("gpu_block_mul requires Eval format");
        }

        if (level < 0)
        {
            return set_error("invalid level in gpu_block_mul");
        }

        const int N = ctx->N;
        if (N <= 0)
        {
            return 0;
        }

        auto &limb_map = ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error("unexpected limb mapping size in gpu_block_mul");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= lhs0->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_block_mul");
            }
            const auto &partition = lhs0->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            const auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() == FIDESlib::U64)
            {
                int status =
                    launch_for_limb_matmul<uint64_t>(out, lhs, rhs, rows, inner, cols, static_cast<size_t>(N), limb, limb_id);
                if (status != 0)
                {
                    return status;
                }
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                int status =
                    launch_for_limb_matmul<uint32_t>(out, lhs, rhs, rows, inner, cols, static_cast<size_t>(N), limb, limb_id);
                if (status != 0)
                {
                    return status;
                }
            }
            else
            {
                return set_error("unsupported limb type in gpu_block_mul");
            }
        }

        for (size_t i = 0; i < out_count; ++i)
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

extern "C" int gpu_block_mul(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t rows,
    size_t inner,
    size_t cols)
{
    return gpu_block_matmul(out, lhs, rhs, rows, inner, cols);
}
