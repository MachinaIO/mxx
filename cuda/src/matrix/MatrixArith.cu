#include "matrix/MatrixArith.cuh"

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

    __global__ void block_equal_u64_kernel(
        const uint64_t **lhs,
        const uint64_t **rhs,
        size_t poly_count,
        size_t n,
        unsigned int *mismatch)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        if (*mismatch != 0)
        {
            return;
        }

        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        if (lhs[poly_idx][coeff_idx] != rhs[poly_idx][coeff_idx])
        {
            atomicExch(mismatch, 1U);
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
    struct BlockPointerScratch
    {
        int device;
        T **d_out;
        const T **d_lhs;
        const T **d_rhs;
        size_t out_capacity;
        size_t lhs_capacity;
        size_t rhs_capacity;
    };

    template <typename T>
    BlockPointerScratch<T> &block_pointer_scratch_for_device(int device)
    {
        thread_local std::vector<BlockPointerScratch<T>> scratches;
        for (auto &scratch : scratches)
        {
            if (scratch.device == device)
            {
                return scratch;
            }
        }
        scratches.push_back(BlockPointerScratch<T>{
            device,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
            0});
        return scratches.back();
    }

    template <typename PtrType>
    int ensure_device_pointer_capacity(
        int device,
        PtrType **ptr,
        size_t *capacity,
        size_t required_count)
    {
        if (!ptr || !capacity)
        {
            return set_error("invalid pointer capacity arguments");
        }
        if (required_count <= *capacity)
        {
            return 0;
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        if (*ptr)
        {
            err = cudaFree(*ptr);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            *ptr = nullptr;
            *capacity = 0;
        }
        if (required_count == 0)
        {
            return 0;
        }

        err = cudaMalloc(ptr, required_count * sizeof(PtrType));
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        *capacity = required_count;
        return 0;
    }

    template <typename T>
    int ensure_block_pointer_scratch(
        int device,
        size_t out_count,
        size_t lhs_count,
        size_t rhs_count,
        T ***out_table,
        const T ***lhs_table,
        const T ***rhs_table)
    {
        if (!out_table || !lhs_table || !rhs_table)
        {
            return set_error("invalid output tables in ensure_block_pointer_scratch");
        }

        auto &scratch = block_pointer_scratch_for_device<T>(device);
        int status = ensure_device_pointer_capacity<T *>(
            device,
            &scratch.d_out,
            &scratch.out_capacity,
            out_count);
        if (status != 0)
        {
            return status;
        }
        status = ensure_device_pointer_capacity<const T *>(
            device,
            &scratch.d_lhs,
            &scratch.lhs_capacity,
            lhs_count);
        if (status != 0)
        {
            return status;
        }
        status = ensure_device_pointer_capacity<const T *>(
            device,
            &scratch.d_rhs,
            &scratch.rhs_capacity,
            rhs_count);
        if (status != 0)
        {
            return status;
        }

        *out_table = scratch.d_out;
        *lhs_table = scratch.d_lhs;
        *rhs_table = scratch.d_rhs;
        return 0;
    }

    struct BlockEqualScratch
    {
        int device;
        unsigned int *d_mismatch;
        size_t mismatch_capacity;
    };

    BlockEqualScratch &block_equal_scratch_for_device(int device)
    {
        thread_local std::vector<BlockEqualScratch> scratches;
        for (auto &scratch : scratches)
        {
            if (scratch.device == device)
            {
                return scratch;
            }
        }
        scratches.push_back(BlockEqualScratch{
            device,
            nullptr,
            0});
        return scratches.back();
    }

    int ensure_block_equal_scratch(
        int device,
        unsigned int **mismatch_ptr)
    {
        if (!mismatch_ptr)
        {
            return set_error("invalid output in ensure_block_equal_scratch");
        }

        auto &scratch = block_equal_scratch_for_device(device);
        int status = ensure_device_pointer_capacity<unsigned int>(
            device,
            &scratch.d_mismatch,
            &scratch.mismatch_capacity,
            1);
        if (status != 0)
        {
            return status;
        }

        *mismatch_ptr = scratch.d_mismatch;
        return 0;
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

        int device = -1;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        T **d_out = nullptr;
        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        const size_t bytes = count * sizeof(T *);
        int status = ensure_block_pointer_scratch<T>(
            device,
            count,
            count,
            count,
            &d_out,
            &d_lhs,
            &d_rhs);
        if (status != 0)
        {
            return status;
        }

        err = cudaMemcpyAsync(d_out, out_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
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
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_block_matmul_kernel_device_ptrs(
        T **d_out,
        const T **d_lhs,
        const T **d_rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        T modulus,
        cudaStream_t stream,
        double *out_kernel_ms)
    {
        const size_t out_count = rows * cols;
        if (out_count == 0 || n == 0)
        {
            return 0;
        }
        if (!d_out || !d_lhs || !d_rhs)
        {
            return set_error("null device pointer table in launch_block_matmul_kernel_device_ptrs");
        }
        if (!stream)
        {
            return set_error("null stream in launch_block_matmul_kernel_device_ptrs");
        }

        const dim3 threads(kMatmulTileN, kMatmulTileM);
        const dim3 blocks(
            static_cast<unsigned int>((cols + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(n));

        cudaError_t err;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (out_kernel_ms)
        {
            err = cudaEventCreate(&start);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaEventCreate(&stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                return set_error(err);
            }
            err = cudaEventRecord(start, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
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
            return set_error(err);
        }

        if (out_kernel_ms)
        {
            err = cudaEventRecord(stop, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return set_error(err);
            }
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return set_error(err);
            }
            float kernel_ms = 0.0f;
            err = cudaEventElapsedTime(&kernel_ms, start, stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
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
                return set_error(err);
            }
        }

        return 0;
    }

} // namespace

namespace
{
    int gpu_block_u64_ptrs_impl(
        uint64_t *const *out,
        uint64_t *const *lhs,
        uint64_t *const *rhs,
        size_t count,
        size_t n,
        uint64_t modulus,
        cudaStream_t stream,
        BlockOp op)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid pointer arrays in gpu_block_u64_ptrs_impl");
        }
        if (!stream)
        {
            return set_error("null stream in gpu_block_u64_ptrs_impl");
        }

        std::vector<uint64_t *> out_ptrs;
        std::vector<const uint64_t *> lhs_ptrs;
        std::vector<const uint64_t *> rhs_ptrs;
        out_ptrs.reserve(count);
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            if (!out[i] || !lhs[i] || !rhs[i])
            {
                return set_error("null limb pointer in gpu_block_u64_ptrs_impl");
            }
            out_ptrs.push_back(out[i]);
            lhs_ptrs.push_back(lhs[i]);
            rhs_ptrs.push_back(rhs[i]);
        }

        return launch_block_kernel<uint64_t>(out_ptrs, lhs_ptrs, rhs_ptrs, n, modulus, op, stream);
    }
} // namespace

extern "C" int gpu_block_add_u64_ptrs(
    uint64_t *const *out,
    uint64_t *const *lhs,
    uint64_t *const *rhs,
    size_t count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream)
{
    return gpu_block_u64_ptrs_impl(out, lhs, rhs, count, n, modulus, stream, BlockOp::Add);
}

extern "C" int gpu_block_sub_u64_ptrs(
    uint64_t *const *out,
    uint64_t *const *lhs,
    uint64_t *const *rhs,
    size_t count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream)
{
    return gpu_block_u64_ptrs_impl(out, lhs, rhs, count, n, modulus, stream, BlockOp::Sub);
}

extern "C" int gpu_block_mul_u64_ptrs(
    uint64_t *const *out,
    uint64_t *const *lhs,
    uint64_t *const *rhs,
    size_t count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream)
{
    return gpu_block_u64_ptrs_impl(out, lhs, rhs, count, n, modulus, stream, BlockOp::Mul);
}

extern "C" int gpu_block_equal_u64_ptrs(
    const uint64_t *const *lhs,
    const uint64_t *const *rhs,
    size_t count,
    size_t n,
    cudaStream_t stream,
    int *out_equal)
{
    if (!lhs || !rhs || !out_equal)
    {
        return set_error("invalid arguments in gpu_block_equal_u64_ptrs");
    }
    if (!stream)
    {
        return set_error("null stream in gpu_block_equal_u64_ptrs");
    }
    if (count == 0 || n == 0)
    {
        *out_equal = 1;
        return 0;
    }

    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    uint64_t **unused_out = nullptr;
    const uint64_t **d_lhs = nullptr;
    const uint64_t **d_rhs = nullptr;
    int status = ensure_block_pointer_scratch<uint64_t>(
        device,
        0,
        count,
        count,
        &unused_out,
        &d_lhs,
        &d_rhs);
    if (status != 0)
    {
        return status;
    }

    const size_t ptr_bytes = count * sizeof(uint64_t *);
    err = cudaMemcpyAsync(d_lhs, lhs, ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_rhs, rhs, ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    unsigned int *d_mismatch = nullptr;
    status = ensure_block_equal_scratch(device, &d_mismatch);
    if (status != 0)
    {
        return status;
    }
    err = cudaMemsetAsync(d_mismatch, 0, sizeof(unsigned int), stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    block_equal_u64_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, count, n, d_mismatch);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    unsigned int mismatch_host = 0;
    err = cudaMemcpyAsync(&mismatch_host, d_mismatch, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    *out_equal = mismatch_host == 0 ? 1 : 0;
    return 0;
}

extern "C" int gpu_block_mul_u64_device_ptrs(
    uint64_t **d_out,
    uint64_t **d_lhs,
    uint64_t **d_rhs,
    size_t rows,
    size_t inner,
    size_t cols,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    double *out_kernel_ms)
{
    return launch_block_matmul_kernel_device_ptrs<uint64_t>(
        d_out,
        (const uint64_t **)d_lhs,
        (const uint64_t **)d_rhs,
        rows,
        inner,
        cols,
        n,
        modulus,
        stream,
        out_kernel_ms);
}

namespace
{
    enum class MatrixBinaryOp
    {
        Add,
        Sub,
    };

    int wait_stream_on_stream(cudaStream_t waiter, cudaStream_t signaler)
    {
        if (!waiter || !signaler || waiter == signaler)
        {
            return 0;
        }

        cudaEvent_t event = nullptr;
        cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        err = cudaEventRecord(event, signaler);
        if (err == cudaSuccess)
        {
            err = cudaStreamWaitEvent(waiter, event, 0);
        }
        cudaError_t destroy_err = cudaEventDestroy(event);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        if (destroy_err != cudaSuccess)
        {
            return set_error(destroy_err);
        }

        return 0;
    }

    int run_matrix_binary_u64_tables(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        MatrixBinaryOp op)
    {
        const size_t count = lhs->rows * lhs->cols;
        if (count == 0)
        {
            return 0;
        }

        const int level = lhs->level;
        if (level < 0)
        {
            return set_error("invalid level in run_matrix_binary_u64_tables");
        }
        const int N = lhs->ctx->N;
        if (N <= 0)
        {
            return 0;
        }
        const size_t crt_depth = static_cast<size_t>(level + 1);
        if (lhs->ctx->moduli.size() < crt_depth)
        {
            return set_error("unexpected modulus count in run_matrix_binary_u64_tables");
        }
        if (lhs->limb_u64_tables.size() < crt_depth ||
            rhs->limb_u64_tables.size() < crt_depth ||
            out->limb_u64_tables.size() < crt_depth)
        {
            return set_error("unexpected limb table size in run_matrix_binary_u64_tables");
        }

        struct LimbJob
        {
            const GpuMatrix::LimbU64Table *lhs_table;
            const GpuMatrix::LimbU64Table *rhs_table;
            const GpuMatrix::LimbU64Table *out_table;
            uint64_t modulus;
        };
        std::vector<LimbJob> jobs;
        jobs.reserve(crt_depth);

        for (int limb = 0; limb <= level; ++limb)
        {
            const GpuMatrix::LimbU64Table *lhs_table = nullptr;
            const GpuMatrix::LimbU64Table *rhs_table = nullptr;
            const GpuMatrix::LimbU64Table *out_table = nullptr;
            bool lhs_has_u64 = false;
            bool rhs_has_u64 = false;
            bool out_has_u64 = false;

            int status = try_get_matrix_u64_limb_table(lhs, limb, &lhs_table, &lhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(rhs, limb, &rhs_table, &rhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(out, limb, &out_table, &out_has_u64);
            if (status != 0)
            {
                return status;
            }

            if (!lhs_has_u64 || !rhs_has_u64 || !out_has_u64)
            {
                return set_error("non-u64 limb is unsupported in run_matrix_binary_u64_tables");
            }
            if (!lhs_table || !rhs_table || !out_table)
            {
                return set_error("missing limb table in run_matrix_binary_u64_tables");
            }
            if (!lhs_table->stream || !rhs_table->stream || !out_table->stream)
            {
                return set_error("null stream in run_matrix_binary_u64_tables");
            }
            if (lhs_table->device != rhs_table->device || lhs_table->device != out_table->device)
            {
                return set_error("device mismatch in run_matrix_binary_u64_tables");
            }

            jobs.push_back(LimbJob{
                lhs_table,
                rhs_table,
                out_table,
                lhs->ctx->moduli[static_cast<size_t>(limb)]});
        }

        for (const auto &job : jobs)
        {
            cudaError_t err = cudaSetDevice(job.out_table->device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            int status = wait_stream_on_stream(job.out_table->stream, job.lhs_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.out_table->stream, job.rhs_table->stream);
            if (status != 0)
            {
                return status;
            }

            switch (op)
            {
            case MatrixBinaryOp::Add:
                status = gpu_block_add_u64_ptrs(
                    job.out_table->entry_ptrs.data(),
                    job.lhs_table->entry_ptrs.data(),
                    job.rhs_table->entry_ptrs.data(),
                    count,
                    static_cast<size_t>(N),
                    job.modulus,
                    job.out_table->stream);
                break;
            case MatrixBinaryOp::Sub:
                status = gpu_block_sub_u64_ptrs(
                    job.out_table->entry_ptrs.data(),
                    job.lhs_table->entry_ptrs.data(),
                    job.rhs_table->entry_ptrs.data(),
                    count,
                    static_cast<size_t>(N),
                    job.modulus,
                    job.out_table->stream);
                break;
            }
            if (status != 0)
            {
                return status;
            }

            status = wait_stream_on_stream(job.lhs_table->stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.rhs_table->stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }

            // Callers may release temporary pointer tables immediately after return.
            // Ensure matmul work that dereferences those tables is complete.
            err = cudaStreamSynchronize(job.out_table->stream);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }

        return 0;
    }

    int run_matrix_mul_scalar_u64_tables(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar)
    {
        const size_t count = lhs->rows * lhs->cols;
        if (count == 0)
        {
            return 0;
        }

        const int level = lhs->level;
        if (level < 0)
        {
            return set_error("invalid level in run_matrix_mul_scalar_u64_tables");
        }
        const int N = lhs->ctx->N;
        if (N <= 0)
        {
            return 0;
        }
        const size_t crt_depth = static_cast<size_t>(level + 1);
        if (lhs->ctx->moduli.size() < crt_depth)
        {
            return set_error("unexpected modulus count in run_matrix_mul_scalar_u64_tables");
        }
        if (lhs->limb_u64_tables.size() < crt_depth || out->limb_u64_tables.size() < crt_depth)
        {
            return set_error("unexpected limb table size in run_matrix_mul_scalar_u64_tables");
        }
        if (scalar->rows != 1 || scalar->cols != 1)
        {
            return set_error("scalar must be 1x1 in run_matrix_mul_scalar_u64_tables");
        }
        if (scalar->limb_u64_tables.size() < crt_depth)
        {
            return set_error("unexpected scalar limb table size in run_matrix_mul_scalar_u64_tables");
        }

        struct LimbJob
        {
            const GpuMatrix::LimbU64Table *lhs_table;
            const GpuMatrix::LimbU64Table *out_table;
            uint64_t *scalar_ptr;
            cudaStream_t scalar_stream;
            uint64_t modulus;
        };
        std::vector<LimbJob> jobs;
        jobs.reserve(crt_depth);

        for (int limb = 0; limb <= level; ++limb)
        {
            const GpuMatrix::LimbU64Table *lhs_table = nullptr;
            const GpuMatrix::LimbU64Table *out_table = nullptr;
            const GpuMatrix::LimbU64Table *scalar_table = nullptr;
            bool lhs_has_u64 = false;
            bool out_has_u64 = false;
            bool scalar_has_u64 = false;

            int status = try_get_matrix_u64_limb_table(lhs, limb, &lhs_table, &lhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(out, limb, &out_table, &out_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(scalar, limb, &scalar_table, &scalar_has_u64);
            if (status != 0)
            {
                return status;
            }
            if (!lhs_has_u64 || !out_has_u64 || !scalar_has_u64)
            {
                return set_error("non-u64 limb is unsupported in run_matrix_mul_scalar_u64_tables");
            }
            if (!lhs_table || !out_table || !scalar_table)
            {
                return set_error("missing limb table in run_matrix_mul_scalar_u64_tables");
            }
            if (!lhs_table->stream || !out_table->stream || !scalar_table->stream)
            {
                return set_error("null stream in run_matrix_mul_scalar_u64_tables");
            }
            if (scalar_table->entry_ptrs.size() != 1)
            {
                return set_error("unexpected scalar entry count in run_matrix_mul_scalar_u64_tables");
            }
            if (lhs_table->device != out_table->device || lhs_table->device != scalar_table->device)
            {
                return set_error("scalar device mismatch in gpu_matrix_mul_scalar");
            }

            jobs.push_back(LimbJob{
                lhs_table,
                out_table,
                scalar_table->entry_ptrs[0],
                scalar_table->stream,
                lhs->ctx->moduli[static_cast<size_t>(limb)]});
        }

        std::vector<uint64_t *> rhs_ptrs(count, nullptr);
        for (const auto &job : jobs)
        {
            cudaError_t err = cudaSetDevice(job.out_table->device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            int status = wait_stream_on_stream(job.out_table->stream, job.lhs_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.out_table->stream, job.scalar_stream);
            if (status != 0)
            {
                return status;
            }

            std::fill(rhs_ptrs.begin(), rhs_ptrs.end(), job.scalar_ptr);
            status = gpu_block_mul_u64_ptrs(
                job.out_table->entry_ptrs.data(),
                job.lhs_table->entry_ptrs.data(),
                rhs_ptrs.data(),
                count,
                static_cast<size_t>(N),
                job.modulus,
                job.out_table->stream);
            if (status != 0)
            {
                return status;
            }

            status = wait_stream_on_stream(job.lhs_table->stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.scalar_stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }
        }

        return 0;
    }

    int run_matrix_mul_u64_tables(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        double *out_kernel_ms)
    {
        const size_t rows = lhs->rows;
        const size_t inner = lhs->cols;
        const size_t cols = rhs->cols;
        if (rows == 0 || inner == 0 || cols == 0)
        {
            if (out_kernel_ms)
            {
                *out_kernel_ms = 0.0;
            }
            return 0;
        }

        const int level = lhs->level;
        if (level < 0)
        {
            return set_error("invalid level in run_matrix_mul_u64_tables");
        }
        const int N = lhs->ctx->N;
        if (N <= 0)
        {
            if (out_kernel_ms)
            {
                *out_kernel_ms = 0.0;
            }
            return 0;
        }

        const size_t crt_depth = static_cast<size_t>(level + 1);
        if (lhs->ctx->moduli.size() < crt_depth)
        {
            return set_error("unexpected modulus count in run_matrix_mul_u64_tables");
        }
        if (lhs->limb_u64_tables.size() < crt_depth ||
            rhs->limb_u64_tables.size() < crt_depth ||
            out->limb_u64_tables.size() < crt_depth)
        {
            return set_error("unexpected limb table size in run_matrix_mul_u64_tables");
        }

        struct LimbJob
        {
            const GpuMatrix::LimbU64Table *lhs_table;
            const GpuMatrix::LimbU64Table *rhs_table;
            const GpuMatrix::LimbU64Table *out_table;
            uint64_t modulus;
        };
        std::vector<LimbJob> jobs;
        jobs.reserve(crt_depth);

        for (int limb = 0; limb <= level; ++limb)
        {
            const GpuMatrix::LimbU64Table *lhs_table = nullptr;
            const GpuMatrix::LimbU64Table *rhs_table = nullptr;
            const GpuMatrix::LimbU64Table *out_table = nullptr;
            bool lhs_has_u64 = false;
            bool rhs_has_u64 = false;
            bool out_has_u64 = false;

            int status = try_get_matrix_u64_limb_table(lhs, limb, &lhs_table, &lhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(rhs, limb, &rhs_table, &rhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(out, limb, &out_table, &out_has_u64);
            if (status != 0)
            {
                return status;
            }

            if (!lhs_has_u64 || !rhs_has_u64 || !out_has_u64)
            {
                return set_error("non-u64 limb is unsupported in run_matrix_mul_u64_tables");
            }
            if (!lhs_table || !rhs_table || !out_table)
            {
                return set_error("missing limb table in run_matrix_mul_u64_tables");
            }
            if (!lhs_table->stream || !rhs_table->stream || !out_table->stream)
            {
                return set_error("null stream in run_matrix_mul_u64_tables");
            }
            if (lhs_table->device != rhs_table->device || lhs_table->device != out_table->device)
            {
                return set_error("device mismatch in run_matrix_mul_u64_tables");
            }
            if (!lhs_table->device_entry_ptrs ||
                !rhs_table->device_entry_ptrs ||
                !out_table->device_entry_ptrs)
            {
                return set_error("missing device pointer table in run_matrix_mul_u64_tables");
            }

            jobs.push_back(LimbJob{
                lhs_table,
                rhs_table,
                out_table,
                lhs->ctx->moduli[static_cast<size_t>(limb)]});
        }

        if (out_kernel_ms)
        {
            *out_kernel_ms = 0.0;
        }

        for (const auto &job : jobs)
        {
            cudaError_t err = cudaSetDevice(job.out_table->device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            int status = wait_stream_on_stream(job.out_table->stream, job.lhs_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.out_table->stream, job.rhs_table->stream);
            if (status != 0)
            {
                return status;
            }

            status = gpu_block_mul_u64_device_ptrs(
                job.out_table->device_entry_ptrs,
                job.lhs_table->device_entry_ptrs,
                job.rhs_table->device_entry_ptrs,
                rows,
                inner,
                cols,
                static_cast<size_t>(N),
                job.modulus,
                job.out_table->stream,
                out_kernel_ms);
            if (status != 0)
            {
                return status;
            }

            status = wait_stream_on_stream(job.lhs_table->stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.rhs_table->stream, job.out_table->stream);
            if (status != 0)
            {
                return status;
            }

            // Poly wrappers may release temporary pointer tables immediately
            // after matrix-mul returns; force completion before returning.
            err = cudaStreamSynchronize(job.out_table->stream);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }

        return 0;
    }

    int run_matrix_equal_u64_tables(
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        int *out_equal)
    {
        if (!out_equal)
        {
            return set_error("null out_equal in run_matrix_equal_u64_tables");
        }
        *out_equal = 0;

        const size_t count = lhs->rows * lhs->cols;
        if (count == 0)
        {
            *out_equal = 1;
            return 0;
        }

        const int level = lhs->level;
        if (level < 0)
        {
            return set_error("invalid level in run_matrix_equal_u64_tables");
        }
        const int N = lhs->ctx->N;
        if (N <= 0)
        {
            *out_equal = 1;
            return 0;
        }
        const size_t crt_depth = static_cast<size_t>(level + 1);
        if (lhs->limb_u64_tables.size() < crt_depth || rhs->limb_u64_tables.size() < crt_depth)
        {
            return set_error("unexpected limb table size in run_matrix_equal_u64_tables");
        }

        struct LimbJob
        {
            const GpuMatrix::LimbU64Table *lhs_table;
            const GpuMatrix::LimbU64Table *rhs_table;
        };
        std::vector<LimbJob> jobs;
        jobs.reserve(crt_depth);

        for (int limb = 0; limb <= level; ++limb)
        {
            const GpuMatrix::LimbU64Table *lhs_table = nullptr;
            const GpuMatrix::LimbU64Table *rhs_table = nullptr;
            bool lhs_has_u64 = false;
            bool rhs_has_u64 = false;

            int status = try_get_matrix_u64_limb_table(lhs, limb, &lhs_table, &lhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            status = try_get_matrix_u64_limb_table(rhs, limb, &rhs_table, &rhs_has_u64);
            if (status != 0)
            {
                return status;
            }
            if (!lhs_has_u64 || !rhs_has_u64)
            {
                return set_error("non-u64 limb is unsupported in run_matrix_equal_u64_tables");
            }
            if (!lhs_table || !rhs_table)
            {
                return set_error("missing limb table in run_matrix_equal_u64_tables");
            }
            if (!lhs_table->stream || !rhs_table->stream)
            {
                return set_error("null stream in run_matrix_equal_u64_tables");
            }
            if (lhs_table->device != rhs_table->device)
            {
                return set_error("device mismatch in run_matrix_equal_u64_tables");
            }

            jobs.push_back(LimbJob{lhs_table, rhs_table});
        }

        for (const auto &job : jobs)
        {
            cudaError_t err = cudaSetDevice(job.lhs_table->device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            int status = wait_stream_on_stream(job.lhs_table->stream, job.rhs_table->stream);
            if (status != 0)
            {
                return status;
            }

            int limb_equal = 0;
            status = gpu_block_equal_u64_ptrs(
                job.lhs_table->entry_ptrs.data(),
                job.rhs_table->entry_ptrs.data(),
                count,
                static_cast<size_t>(N),
                job.lhs_table->stream,
                &limb_equal);
            if (status != 0)
            {
                return status;
            }
            status = wait_stream_on_stream(job.rhs_table->stream, job.lhs_table->stream);
            if (status != 0)
            {
                return status;
            }
            if (limb_equal == 0)
            {
                *out_equal = 0;
                return 0;
            }
        }

        *out_equal = 1;
        return 0;
    }
} // namespace

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

    int status = run_matrix_binary_u64_tables(out, lhs, rhs, MatrixBinaryOp::Add);
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

    int status = run_matrix_binary_u64_tables(out, lhs, rhs, MatrixBinaryOp::Sub);
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
    int status = run_matrix_mul_u64_tables(out, lhs, rhs, nullptr);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal)
{
    if (!lhs || !rhs || !out_equal)
    {
        return set_error("invalid gpu_matrix_equal arguments");
    }
    *out_equal = 0;

    if (lhs == rhs)
    {
        *out_equal = 1;
        return 0;
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return 0;
    }
    if (lhs->ctx != rhs->ctx || lhs->level != rhs->level)
    {
        return 0;
    }

    return run_matrix_equal_u64_tables(lhs, rhs, out_equal);
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
    int status = run_matrix_mul_u64_tables(out, lhs, rhs, out_kernel_ms);
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
    const GpuMatrix *scalar)
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
    if (scalar->rows != 1 || scalar->cols != 1)
    {
        return set_error("scalar must be 1x1 in gpu_matrix_mul_scalar");
    }

    int status = run_matrix_mul_scalar_u64_tables(out, lhs, scalar);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}
