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
    __global__ void matrix_decompose_multi_kernel(
        const T **src,
        T **dst,
        size_t poly_count,
        size_t n,
        size_t job_count,
        const uint32_t *shifts,
        const T *masks,
        const T *out_moduli)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = job_count * poly_count * n;
        if (idx >= total)
        {
            return;
        }

        const size_t per_job = poly_count * n;
        const size_t job_idx = idx / per_job;
        const size_t rem = idx - job_idx * per_job;
        const size_t poly_idx = rem / n;
        const size_t coeff_idx = rem - poly_idx * n;

        const size_t ptr_idx = job_idx * poly_count + poly_idx;
        T residue = src[ptr_idx][coeff_idx];
        const uint32_t shift = shifts[job_idx];
        const T mask = masks[job_idx];
        T digit = shift >= static_cast<uint32_t>(sizeof(T) * 8) ? 0 : ((residue >> shift) & mask);
        const T out_modulus = out_moduli[job_idx];
        if (out_modulus != 0 && digit >= out_modulus)
        {
            digit %= out_modulus;
        }
        dst[ptr_idx][coeff_idx] = digit;
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
    __global__ void block_copy_kernel(
        const T **src,
        T **dst,
        size_t poly_count,
        size_t n)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        dst[poly_idx][coeff_idx] = src[poly_idx][coeff_idx];
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
    int launch_decompose_multi_kernel(
        const std::vector<const T *> &src_ptrs,
        const std::vector<T *> &dst_ptrs,
        size_t poly_count,
        size_t n,
        const std::vector<uint32_t> &shifts,
        const std::vector<T> &masks,
        const std::vector<T> &out_moduli,
        cudaStream_t stream)
    {
        const size_t job_count = shifts.size();
        if (job_count == 0 || poly_count == 0 || n == 0)
        {
            return 0;
        }
        if (masks.size() != job_count || out_moduli.size() != job_count)
        {
            return set_error("unexpected job parameter counts in matrix_decompose_multi_kernel");
        }
        const size_t ptr_count = job_count * poly_count;
        if (src_ptrs.size() != ptr_count || dst_ptrs.size() != ptr_count)
        {
            return set_error("unexpected pointer counts in matrix_decompose_multi_kernel");
        }

        const T **d_src = nullptr;
        T **d_dst = nullptr;
        uint32_t *d_shifts = nullptr;
        T *d_masks = nullptr;
        T *d_out_moduli = nullptr;

        const size_t ptr_bytes = ptr_count * sizeof(T *);
        const size_t shift_bytes = job_count * sizeof(uint32_t);
        const size_t scalar_bytes = job_count * sizeof(T);

        cudaError_t err = cudaMalloc(&d_src, ptr_bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_dst, ptr_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            return set_error(err);
        }
        err = cudaMalloc(&d_shifts, shift_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }
        err = cudaMalloc(&d_masks, scalar_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            return set_error(err);
        }
        err = cudaMalloc(&d_out_moduli, scalar_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_src, src_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_shifts, shifts.data(), shift_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_masks, masks.data(), scalar_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_out_moduli, out_moduli.data(), scalar_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = ptr_count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        matrix_decompose_multi_kernel<<<blocks, threads, 0, stream>>>(
            d_src,
            d_dst,
            poly_count,
            n,
            job_count,
            d_shifts,
            d_masks,
            d_out_moduli);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            cudaFree(d_shifts);
            cudaFree(d_masks);
            cudaFree(d_out_moduli);
            return set_error(err);
        }

        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_shifts);
        cudaFree(d_masks);
        cudaFree(d_out_moduli);
        return 0;
    }

    template <typename T>
    int launch_copy_kernel(
        const std::vector<const T *> &src_ptrs,
        const std::vector<T *> &dst_ptrs,
        size_t n,
        cudaStream_t stream)
    {
        const size_t count = src_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (dst_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in block_copy_kernel");
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

        block_copy_kernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, count, n);
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

    bool has_stream_ref(const std::vector<FIDESlib::Stream *> &streams, FIDESlib::Stream *candidate)
    {
        for (auto *stream : streams)
        {
            if (stream == candidate)
            {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    int launch_copy_for_limb(
        GpuMatrix *out,
        const GpuMatrix *src,
        const std::vector<size_t> &src_indices,
        const std::vector<size_t> &dst_indices,
        const dim3 &limb_id,
        size_t n)
    {
        const size_t count = src_indices.size();
        if (count == 0)
        {
            return 0;
        }
        if (dst_indices.size() != count)
        {
            return set_error("mismatched index counts in launch_copy_for_limb");
        }

        std::vector<const T *> src_ptrs;
        std::vector<T *> dst_ptrs;
        std::vector<FIDESlib::Stream *> src_streams;
        std::vector<FIDESlib::Stream *> dst_streams;
        src_ptrs.reserve(count);
        dst_ptrs.reserve(count);
        src_streams.reserve(count);
        dst_streams.reserve(count);

        FIDESlib::Stream *exec_stream_ref = nullptr;
        cudaStream_t exec_stream = nullptr;
        int device = -1;

        for (size_t i = 0; i < count; ++i)
        {
            const GpuPoly *src_poly = src->polys[src_indices[i]].get();
            GpuPoly *dst_poly = out->polys[dst_indices[i]].get();
            if (!src_poly || !dst_poly)
            {
                return set_error("null polynomial in launch_copy_for_limb");
            }
            if (limb_id.x >= src_poly->poly->GPU.size() || limb_id.x >= dst_poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in launch_copy_for_limb");
            }

            const auto &src_partition = src_poly->poly->GPU[limb_id.x];
            auto &dst_partition = dst_poly->poly->GPU[limb_id.x];
            if (src_partition.device != dst_partition.device)
            {
                return set_error("source/destination device mismatch in launch_copy_for_limb");
            }
            if (limb_id.y >= src_partition.limb.size() || limb_id.y >= dst_partition.limb.size())
            {
                return set_error("unexpected limb index in launch_copy_for_limb");
            }

            const auto &src_limb = src_partition.limb[limb_id.y];
            auto &dst_limb = dst_partition.limb[limb_id.y];
            if constexpr (std::is_same_v<T, uint64_t>)
            {
                if (src_limb.index() != FIDESlib::U64 || dst_limb.index() != FIDESlib::U64)
                {
                    return set_error("mixed limb types in launch_copy_for_limb");
                }
                const auto &src_u64 = std::get<FIDESlib::U64>(src_limb);
                auto &dst_u64 = std::get<FIDESlib::U64>(dst_limb);
                src_ptrs.push_back(src_u64.v.data);
                dst_ptrs.push_back(dst_u64.v.data);

                auto *src_stream_ref = &src_u64.stream;
                auto *dst_stream_ref = &dst_u64.stream;
                if (!has_stream_ref(src_streams, src_stream_ref))
                {
                    src_streams.push_back(src_stream_ref);
                }
                if (!has_stream_ref(dst_streams, dst_stream_ref))
                {
                    dst_streams.push_back(dst_stream_ref);
                }

                if (!exec_stream_ref)
                {
                    exec_stream_ref = dst_stream_ref;
                    exec_stream = dst_u64.stream.ptr;
                    device = dst_partition.device;
                }
                else if (device != dst_partition.device)
                {
                    return set_error("inconsistent copy device in launch_copy_for_limb");
                }
            }
            else
            {
                if (src_limb.index() != FIDESlib::U32 || dst_limb.index() != FIDESlib::U32)
                {
                    return set_error("mixed limb types in launch_copy_for_limb");
                }
                const auto &src_u32 = std::get<FIDESlib::U32>(src_limb);
                auto &dst_u32 = std::get<FIDESlib::U32>(dst_limb);
                src_ptrs.push_back(src_u32.v.data);
                dst_ptrs.push_back(dst_u32.v.data);

                auto *src_stream_ref = &src_u32.stream;
                auto *dst_stream_ref = &dst_u32.stream;
                if (!has_stream_ref(src_streams, src_stream_ref))
                {
                    src_streams.push_back(src_stream_ref);
                }
                if (!has_stream_ref(dst_streams, dst_stream_ref))
                {
                    dst_streams.push_back(dst_stream_ref);
                }

                if (!exec_stream_ref)
                {
                    exec_stream_ref = dst_stream_ref;
                    exec_stream = dst_u32.stream.ptr;
                    device = dst_partition.device;
                }
                else if (device != dst_partition.device)
                {
                    return set_error("inconsistent copy device in launch_copy_for_limb");
                }
            }
        }

        if (!exec_stream_ref || !exec_stream)
        {
            return set_error("null execution stream in launch_copy_for_limb");
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        for (auto *src_stream_ref : src_streams)
        {
            if (src_stream_ref != exec_stream_ref)
            {
                exec_stream_ref->wait(*src_stream_ref);
            }
        }
        for (auto *dst_stream_ref : dst_streams)
        {
            if (dst_stream_ref != exec_stream_ref)
            {
                exec_stream_ref->wait(*dst_stream_ref);
            }
        }

        int status = launch_copy_kernel(src_ptrs, dst_ptrs, n, exec_stream);
        if (status != 0)
        {
            return status;
        }

        for (auto *dst_stream_ref : dst_streams)
        {
            if (dst_stream_ref != exec_stream_ref)
            {
                dst_stream_ref->wait(*exec_stream_ref);
            }
        }
        for (auto *src_stream_ref : src_streams)
        {
            if (src_stream_ref != exec_stream_ref)
            {
                src_stream_ref->wait(*exec_stream_ref);
            }
        }

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

int launch_fill_gadget_multi_limb_kernel(
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &moduli,
    const std::vector<uint32_t> &limb_indices,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits,
    cudaStream_t stream);

int launch_sample_p1_integer_kernel(
    const std::vector<const uint64_t *> &a_entries,
    const std::vector<const uint64_t *> &b_entries,
    const std::vector<const uint64_t *> &d_entries,
    const std::vector<const uint64_t *> &tp2_entries,
    size_t d,
    size_t cols,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    cudaStream_t stream,
    int device_id,
    std::vector<int64_t> &sampled_out_host);

int launch_scatter_p1_integer_to_limb_kernel(
    const std::vector<int64_t> &sampled_in_host,
    const std::vector<uint64_t *> &out_entries,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id);

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
    auto out_polys = collect_poly_ptrs(out);
    auto lhs_polys = collect_poly_const_ptrs(lhs);
    auto rhs_polys = collect_poly_const_ptrs(rhs);
    int status = gpu_block_add(
        out_polys.data(),
        lhs_polys.data(),
        rhs_polys.data(),
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
    auto out_polys = collect_poly_ptrs(out);
    auto lhs_polys = collect_poly_const_ptrs(lhs);
    auto rhs_polys = collect_poly_const_ptrs(rhs);
    int status = gpu_block_sub(
        out_polys.data(),
        lhs_polys.data(),
        rhs_polys.data(),
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
    auto out_polys = collect_poly_ptrs(out);
    auto lhs_polys = collect_poly_const_ptrs(lhs);
    auto rhs_polys = collect_poly_const_ptrs(rhs);
    int status = gpu_block_mul(
        out_polys.data(),
        lhs_polys.data(),
        rhs_polys.data(),
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

    const size_t count = lhs->rows * lhs->cols;
    for (size_t i = 0; i < count; ++i)
    {
        int poly_equal = 0;
        int status = gpu_poly_equal(lhs->polys[i].get(), rhs->polys[i].get(), &poly_equal);
        if (status != 0)
        {
            return status;
        }
        if (poly_equal == 0)
        {
            return 0;
        }
    }

    *out_equal = 1;
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
    auto out_polys = collect_poly_ptrs(out);
    auto lhs_polys = collect_poly_const_ptrs(lhs);
    auto rhs_polys = collect_poly_const_ptrs(rhs);
    int status = gpu_block_mul_timed(
        out_polys.data(),
        lhs_polys.data(),
        rhs_polys.data(),
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
    auto out_polys = collect_poly_ptrs(out);
    auto lhs_polys = collect_poly_const_ptrs(lhs);
    std::vector<const GpuPoly *> rhs(count, scalar);
    int status = gpu_block_entrywise_mul(
        out_polys.data(),
        lhs_polys.data(),
        rhs.data(),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}
