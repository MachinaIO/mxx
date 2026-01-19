#include "GpuMatrix.h"
#include "GpuPolyInternal.h"

#include <algorithm>
#include <exception>
#include <limits>
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

    uint32_t bit_width_u64(uint64_t v)
    {
        if (v == 0)
        {
            return 0;
        }
        return static_cast<uint32_t>(64 - __builtin_clzll(v));
    }

    size_t matrix_index(size_t row, size_t col, size_t cols)
    {
        return row * cols + col;
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
        sum = sum >= modulus ? (sum - modulus) : sum;
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
    __global__ void matrix_decompose_kernel(
        const T **src,
        T **dst,
        size_t poly_count,
        size_t n,
        uint32_t shift,
        T mask)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T residue = src[poly_idx][coeff_idx];
        T digit = shift >= static_cast<uint32_t>(sizeof(T) * 8) ? 0 : ((residue >> shift) & mask);
        dst[poly_idx][coeff_idx] = digit;
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
        cudaStream_t stream,
        double *out_kernel_ms)
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

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (out_kernel_ms)
        {
            err = cudaEventCreate(&start);
            if (err != cudaSuccess)
            {
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventCreate(&stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventRecord(start, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
        }

        block_matmul_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, rows, inner, cols, n, modulus);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            if (start)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        if (out_kernel_ms)
        {
            err = cudaEventRecord(stop, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            float kernel_ms = 0.0f;
            err = cudaEventElapsedTime(&kernel_ms, start, stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            *out_kernel_ms += static_cast<double>(kernel_ms);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        else
        {
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
            {
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
        }

        cudaFree(d_out);
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        return 0;
    }

    template <typename T>
    int launch_decompose_kernel(
        const std::vector<const T *> &src_ptrs,
        const std::vector<T *> &dst_ptrs,
        size_t n,
        uint32_t shift,
        T mask,
        cudaStream_t stream)
    {
        const size_t count = src_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (dst_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in matrix_decompose_kernel");
        }

        const T **d_src = nullptr;
        T **d_dst = nullptr;
        const size_t bytes = count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_src, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_dst, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_src, src_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        matrix_decompose_kernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, count, n, shift, mask);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        cudaFree(d_src);
        cudaFree(d_dst);
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
        const dim3 &limb_id,
        double *out_kernel_ms)
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
        return launch_block_matmul_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, rows, inner, cols, n, modulus, stream, out_kernel_ms);
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

    int gpu_block_matmul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        double *out_kernel_ms = nullptr)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid gpu_block_mul arguments");
        }
        if (out_kernel_ms)
        {
            *out_kernel_ms = 0.0;
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
                    launch_for_limb_matmul<uint64_t>(
                        out,
                        lhs,
                        rhs,
                        rows,
                        inner,
                        cols,
                        static_cast<size_t>(N),
                        limb,
                        limb_id,
                        out_kernel_ms);
                if (status != 0)
                {
                    return status;
                }
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                int status =
                    launch_for_limb_matmul<uint32_t>(
                        out,
                        lhs,
                        rhs,
                        rows,
                        inner,
                        cols,
                        static_cast<size_t>(N),
                        limb,
                        limb_id,
                        out_kernel_ms);
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

extern "C" int gpu_block_mul_timed(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t rows,
    size_t inner,
    size_t cols,
    double *out_kernel_ms)
{
    if (!out_kernel_ms)
    {
        return set_error("null out_kernel_ms in gpu_block_mul_timed");
    }
    return gpu_block_matmul(out, lhs, rhs, rows, inner, cols, out_kernel_ms);
}

extern "C" int gpu_matrix_create(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
    GpuMatrix **out)
{
    if (!ctx || !out)
    {
        return set_error("invalid gpu_matrix_create arguments");
    }
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_create");
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}};
    const size_t count = rows * cols;
    mat->polys.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        GpuPoly *poly = nullptr;
        int status = gpu_poly_create(ctx, level, &poly);
        if (status != 0)
        {
            for (auto *p : mat->polys)
            {
                gpu_poly_destroy(p);
            }
            delete mat;
            return status;
        }
        poly->format = fmt;
        mat->polys.push_back(poly);
    }
    *out = mat;
    return 0;
}

extern "C" void gpu_matrix_destroy(GpuMatrix *mat)
{
    if (!mat)
    {
        return;
    }
    for (auto *poly : mat->polys)
    {
        gpu_poly_destroy(poly);
    }
    delete mat;
}

extern "C" int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src)
{
    if (!dst || !src)
    {
        return set_error("invalid gpu_matrix_copy arguments");
    }
    if (dst->rows != src->rows || dst->cols != src->cols)
    {
        return set_error("size mismatch in gpu_matrix_copy");
    }
    if (dst->level != src->level || dst->ctx != src->ctx)
    {
        return set_error("context mismatch in gpu_matrix_copy");
    }
    const size_t count = src->rows * src->cols;
    for (size_t i = 0; i < count; ++i)
    {
        int status = gpu_poly_copy(dst->polys[i], src->polys[i]);
        if (status != 0)
        {
            return status;
        }
    }
    dst->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_entry_clone(
    const GpuMatrix *mat,
    size_t row,
    size_t col,
    GpuPoly **out_poly)
{
    if (!mat || !out_poly)
    {
        return set_error("invalid gpu_matrix_entry_clone arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_entry_clone");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_clone(mat->polys[idx], out_poly);
}

extern "C" int gpu_matrix_copy_entry(
    GpuMatrix *mat,
    size_t row,
    size_t col,
    const GpuPoly *src)
{
    if (!mat || !src)
    {
        return set_error("invalid gpu_matrix_copy_entry arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_copy_entry");
    }
    if (src->ctx != mat->ctx || src->level != mat->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_entry");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_copy(mat->polys[idx], src);
}

extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format)
{
    if (!mat)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    const size_t count = mat->rows * mat->cols;
    int status = gpu_poly_load_rns_batch(
        mat->polys.data(),
        count,
        bytes,
        bytes_per_poly,
        format);
    if (status != 0)
    {
        return status;
    }
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }
    mat->format = fmt;
    return 0;
}

extern "C" int gpu_matrix_store_rns_batch(
    const GpuMatrix *mat,
    uint8_t *bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    const size_t count = mat->rows * mat->cols;
    return gpu_poly_store_rns_batch(
        const_cast<GpuPoly *const *>(mat->polys.data()),
        count,
        bytes_out,
        bytes_per_poly,
        format,
        out_events);
}

extern "C" int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_add arguments");
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return set_error("size mismatch in gpu_matrix_add");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_add");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_add");
    }
    const size_t count = lhs->rows * lhs->cols;
    int status = gpu_block_add(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_sub arguments");
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return set_error("size mismatch in gpu_matrix_sub");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_sub");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_sub");
    }
    const size_t count = lhs->rows * lhs->cols;
    int status = gpu_block_sub(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}

extern "C" int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_mul arguments");
    }
    if (lhs->cols != rhs->rows)
    {
        return set_error("size mismatch in gpu_matrix_mul");
    }
    if (out->rows != lhs->rows || out->cols != rhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul");
    }
    int status = gpu_block_mul(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        lhs->rows,
        lhs->cols,
        rhs->cols);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_mul_timed(
    GpuMatrix *out,
    const GpuMatrix *lhs,
    const GpuMatrix *rhs,
    double *out_kernel_ms)
{
    if (!out_kernel_ms)
    {
        return set_error("null out_kernel_ms in gpu_matrix_mul_timed");
    }
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_mul_timed arguments");
    }
    if (lhs->cols != rhs->rows)
    {
        return set_error("size mismatch in gpu_matrix_mul_timed");
    }
    if (out->rows != lhs->rows || out->cols != rhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul_timed");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul_timed");
    }
    int status = gpu_block_mul_timed(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        lhs->rows,
        lhs->cols,
        rhs->cols,
        out_kernel_ms);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_mul_scalar(
    GpuMatrix *out,
    const GpuMatrix *lhs,
    const GpuPoly *scalar)
{
    if (!out || !lhs || !scalar)
    {
        return set_error("invalid gpu_matrix_mul_scalar arguments");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul_scalar");
    }
    if (lhs->ctx != out->ctx || lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul_scalar");
    }
    if (scalar->ctx != lhs->ctx || scalar->level != lhs->level)
    {
        return set_error("scalar context mismatch in gpu_matrix_mul_scalar");
    }

    const size_t count = lhs->rows * lhs->cols;
    std::vector<const GpuPoly *> rhs(count, scalar);
    int status = gpu_block_entrywise_mul(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        rhs.data(),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}

extern "C" int gpu_matrix_copy_block(
    GpuMatrix *out,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t src_row,
    size_t src_col,
    size_t rows,
    size_t cols)
{
    if (!out || !src)
    {
        return set_error("invalid gpu_matrix_copy_block arguments");
    }
    if (src_row + rows > src->rows || src_col + cols > src->cols)
    {
        return set_error("source bounds exceeded in gpu_matrix_copy_block");
    }
    if (dst_row + rows > out->rows || dst_col + cols > out->cols)
    {
        return set_error("dest bounds exceeded in gpu_matrix_copy_block");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_block");
    }

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            const size_t src_idx = matrix_index(src_row + i, src_col + j, src->cols);
            const size_t dst_idx = matrix_index(dst_row + i, dst_col + j, out->cols);
            int status = gpu_poly_copy(out->polys[dst_idx], src->polys[src_idx]);
            if (status != 0)
            {
                return status;
            }
        }
    }
    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out)
{
    if (!src || !out)
    {
        return set_error("invalid gpu_matrix_decompose_base arguments");
    }
    if (base_bits == 0)
    {
        return set_error("base_bits must be non-zero in gpu_matrix_decompose_base");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_decompose_base");
    }

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_decompose_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    uint32_t crt_bits = 0;
    for (const auto &modulus : src->ctx->moduli)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_decompose_base");
    }
    const uint32_t digits_per_tower =
        static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_decompose_base");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_decompose_base");
    }
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    std::vector<GpuPoly *> tmp_inputs;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        tmp_inputs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            GpuPoly *clone = nullptr;
            int status = gpu_poly_clone(src->polys[i], &clone);
            if (status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                gpu_poly_destroy(clone);
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            tmp_inputs.push_back(clone);
            inputs.push_back(clone);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            inputs.push_back(src->polys[i]);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        for (auto *p : tmp_inputs)
        {
            gpu_poly_destroy(p);
        }
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("invalid output poly in gpu_matrix_decompose_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb GPU partition in gpu_matrix_decompose_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb index in gpu_matrix_decompose_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unsupported limb type in gpu_matrix_decompose_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    const uint64_t base_mask =
        base_bits >= 64 ? std::numeric_limits<uint64_t>::max() : ((1ULL << base_bits) - 1);
    const uint32_t last_bits =
        static_cast<uint32_t>(crt_bits - base_bits * (digits_per_tower - 1));
    const uint64_t last_mask =
        last_bits >= 64 ? std::numeric_limits<uint64_t>::max() : ((1ULL << last_bits) - 1);

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        if (limb_id.x >= inputs[0]->poly->GPU.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected limb GPU partition in gpu_matrix_decompose_base");
        }
        const auto &partition = inputs[0]->poly->GPU[limb_id.x];
        if (limb_id.y >= partition.limb.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected limb index in gpu_matrix_decompose_base");
        }
        const auto &limb_impl = partition.limb[limb_id.y];
        if (limb_impl.index() != FIDESlib::U64)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unsupported limb type in gpu_matrix_decompose_base");
        }

        cudaError_t err = cudaSetDevice(partition.device);
        if (err != cudaSuccess)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error(err);
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const size_t digit_offset =
                static_cast<size_t>(limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);

            std::vector<const uint64_t *> src_ptrs;
            std::vector<uint64_t *> dst_ptrs;
            src_ptrs.reserve(count);
            dst_ptrs.reserve(count);

            cudaStream_t stream = nullptr;
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[limb_id.x];
                if (limb_id.y >= in_partition.limb.size())
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unexpected input limb index in gpu_matrix_decompose_base");
                }
                const auto &in_limb_impl = in_partition.limb[limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unsupported limb type in gpu_matrix_decompose_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                if (!stream)
                {
                    stream = in_limb_u64.stream.ptr;
                }
                src_ptrs.push_back(in_limb_u64.v.data);

                const size_t row = idx / cols;
                const size_t col = idx % cols;
                const size_t out_row = row * log_base_q + digit_offset;
                const size_t out_idx = matrix_index(out_row, col, out->cols);
                auto &out_partition = out->polys[out_idx]->poly->GPU[limb_id.x];
                if (out_partition.device != in_partition.device)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("input/output limb device mismatch in gpu_matrix_decompose_base");
                }
                if (limb_id.y >= out_partition.limb.size())
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unexpected output limb index in gpu_matrix_decompose_base");
                }
                const auto &out_limb_impl = out_partition.limb[limb_id.y];
                if (out_limb_impl.index() != FIDESlib::U64)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unsupported output limb type in gpu_matrix_decompose_base");
                }
                const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                dst_ptrs.push_back(out_limb_u64.v.data);
            }

            const uint64_t mask =
                (digit_idx + 1 == digits_per_tower && last_mask != 0) ? last_mask : base_mask;
            const uint32_t shift = digit_idx * base_bits;
            int status = launch_decompose_kernel<uint64_t>(
                src_ptrs,
                dst_ptrs,
                static_cast<size_t>(src->ctx->N),
                shift,
                mask,
                stream);
            if (status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    for (auto *p : tmp_inputs)
    {
        gpu_poly_destroy(p);
    }
    return 0;
}
