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
            const GpuPoly *src_poly = src->polys[src_indices[i]];
            GpuPoly *dst_poly = out->polys[dst_indices[i]];
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

int launch_sample_p1_full_kernel(
    const std::vector<const uint64_t *> &a_entries,
    const std::vector<const uint64_t *> &b_entries,
    const std::vector<const uint64_t *> &d_entries,
    const std::vector<const uint64_t *> &tp2_entries,
    const std::vector<uint64_t *> &out_entries,
    size_t d,
    size_t cols,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint32_t limb_idx,
    uint64_t seed,
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

namespace
{
    bool checked_mul_size(size_t a, size_t b, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (a != 0 && b > static_cast<size_t>(-1) / a)
        {
            return false;
        }
        *out = a * b;
        return true;
    }

    size_t limb_count_for_level(const std::vector<FIDESlib::LimbRecord> &meta, int level)
    {
        size_t count = 0;
        for (const auto &record : meta)
        {
            if (record.id > level)
            {
                break;
            }
            ++count;
        }
        return count;
    }

    void free_matrix_shared_buffers(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        for (const auto &entry : mat->shared_limb_buffers)
        {
            if (!entry.ptr)
            {
                continue;
            }
            cudaSetDevice(entry.device);
            cudaFree(entry.ptr);
        }
        mat->shared_limb_buffers.clear();
        for (const auto &entry : mat->shared_aux_buffers)
        {
            if (!entry.ptr)
            {
                continue;
            }
            cudaSetDevice(entry.device);
            cudaFree(entry.ptr);
        }
        mat->shared_aux_buffers.clear();
    }

    void detach_matrix_shared_aux_buffers(GpuMatrix *mat)
    {
        if (!mat || mat->shared_aux_buffers.empty())
        {
            return;
        }
        for (auto *poly : mat->polys)
        {
            if (!poly || !poly->poly)
            {
                continue;
            }
            for (auto &partition : poly->poly->GPU)
            {
                partition.bufferAUXptrs = nullptr;
            }
        }
    }

    void destroy_matrix_contents(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        detach_matrix_shared_aux_buffers(mat);
        for (auto *poly : mat->polys)
        {
            gpu_poly_destroy(poly);
        }
        mat->polys.clear();
        free_matrix_shared_buffers(mat);
    }
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
    *out = nullptr;
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_create");
    }
    if (level < -1 || level > ctx->ctx->L)
    {
        return set_error("invalid level in gpu_matrix_create");
    }

    size_t count = 0;
    if (!checked_mul_size(rows, cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_create");
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}, {}, {}};
    mat->polys.reserve(count);
    const size_t partition_count = ctx->ctx->meta.size();
    if (partition_count != ctx->ctx->GPUid.size())
    {
        destroy_matrix_contents(mat);
        delete mat;
        return set_error("unexpected context partition mapping in gpu_matrix_create");
    }

    std::vector<size_t> limb_counts(partition_count, 0);
    std::vector<size_t> words_per_poly(partition_count, 0);
    std::vector<size_t> words_total(partition_count, 0);
    std::vector<uint64_t *> device_bases(partition_count, nullptr);
    std::vector<void **> aux_device_bases(partition_count, nullptr);
    std::vector<size_t> aux_slots_per_poly(partition_count, 0);
    std::vector<size_t> aux_slots_total(partition_count, 0);
    std::vector<FIDESlib::Stream *> alloc_streams(partition_count, nullptr);
    const size_t n = static_cast<size_t>(ctx->N);

    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
    {
        auto &meta = ctx->ctx->meta[partition_idx];
        const size_t limbs = limb_count_for_level(meta, level);
        limb_counts[partition_idx] = limbs;
        if (limbs == 0 || count == 0)
        {
            continue;
        }

        size_t limb_words = 0;
        size_t poly_words = 0;
        size_t total_words = 0;
        size_t total_bytes = 0;
        if (!checked_mul_size(n, limbs, &limb_words) ||
            !checked_mul_size(limb_words, static_cast<size_t>(2), &poly_words) ||
            !checked_mul_size(poly_words, count, &total_words) ||
            !checked_mul_size(total_words, sizeof(uint64_t), &total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix limb allocation overflow in gpu_matrix_create");
        }

        cudaError_t err = cudaSetDevice(ctx->ctx->GPUid[partition_idx]);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        FIDESlib::Stream *alloc_stream = nullptr;
        for (size_t limb_idx = 0; limb_idx < meta.size(); ++limb_idx)
        {
            if (meta[limb_idx].id > level)
            {
                break;
            }
            alloc_stream = &meta[limb_idx].stream;
            break;
        }
        if (!alloc_stream || !alloc_stream->ptr)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("missing allocation stream in gpu_matrix_create");
        }

        uint64_t *base = nullptr;
        err = cudaMallocAsync(&base, total_bytes, alloc_stream->ptr);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        size_t aux_slots = 0;
        size_t aux_total_slots = 0;
        size_t aux_total_bytes = 0;
        const size_t decomp_count = ctx->ctx->decompMeta[partition_idx].size();
        if (!checked_mul_size(static_cast<size_t>(FIDESlib::MAXP), static_cast<size_t>(4 + 4 * decomp_count), &aux_slots) ||
            !checked_mul_size(aux_slots, count, &aux_total_slots) ||
            !checked_mul_size(aux_total_slots, sizeof(void *), &aux_total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix aux allocation overflow in gpu_matrix_create");
        }

        void **aux_base = nullptr;
        err = cudaMallocAsync(&aux_base, aux_total_bytes, alloc_stream->ptr);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        for (size_t limb_idx = 0; limb_idx < meta.size(); ++limb_idx)
        {
            if (meta[limb_idx].id > level)
            {
                break;
            }
            FIDESlib::Stream *limb_stream = &meta[limb_idx].stream;
            if (limb_stream != alloc_stream)
            {
                limb_stream->wait(*alloc_stream);
            }
        }

        device_bases[partition_idx] = base;
        aux_device_bases[partition_idx] = aux_base;
        aux_slots_per_poly[partition_idx] = aux_slots;
        aux_slots_total[partition_idx] = aux_total_slots;
        alloc_streams[partition_idx] = alloc_stream;
        words_per_poly[partition_idx] = poly_words;
        words_total[partition_idx] = total_words;
        mat->shared_limb_buffers.push_back(GpuMatrix::SharedLimbBuffer{
            ctx->ctx->GPUid[partition_idx],
            base});
        mat->shared_aux_buffers.push_back(GpuMatrix::SharedAuxBuffer{
            ctx->ctx->GPUid[partition_idx],
            aux_base});
    }

    std::vector<size_t> words_used(partition_count, 0);
    std::vector<size_t> aux_slots_used(partition_count, 0);
    for (size_t poly_idx = 0; poly_idx < count; ++poly_idx)
    {
        auto *poly_impl = new CKKS::RNSPoly(*ctx->ctx, -1, false);
        poly_impl->setLevel(level);
        auto *poly = new GpuPoly{poly_impl, ctx, level, fmt};

        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
        {
            if (partition_idx >= poly_impl->GPU.size())
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected poly partition size in gpu_matrix_create");
            }

            auto &partition = poly_impl->GPU[partition_idx];
            if (partition.device != ctx->ctx->GPUid[partition_idx])
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected partition device in gpu_matrix_create");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
            if (alloc_streams[partition_idx])
            {
                partition.s.wait(*alloc_streams[partition_idx]);
            }

            void **aux_base = aux_device_bases[partition_idx];
            if (!aux_base)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("missing shared aux buffer in gpu_matrix_create");
            }
            const size_t aux_poly_slots = aux_slots_per_poly[partition_idx];
            const size_t aux_offset_slots = aux_slots_used[partition_idx];
            if (aux_offset_slots > aux_slots_total[partition_idx] ||
                aux_poly_slots > aux_slots_total[partition_idx] - aux_offset_slots)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("shared aux buffer overflow in gpu_matrix_create");
            }
            if (partition.bufferAUXptrs)
            {
                err = cudaFree(partition.bufferAUXptrs);
                if (err != cudaSuccess)
                {
                    gpu_poly_destroy(poly);
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error(err);
                }
            }
            void **poly_aux_base = aux_base + aux_offset_slots;
            partition.bufferAUXptrs = poly_aux_base;
            const size_t aux_stride = static_cast<size_t>(FIDESlib::MAXP);
            const size_t decomp_ptr_count = partition.DECOMPlimbptr.size();
            if (partition.DECOMPauxptr.size() != decomp_ptr_count ||
                partition.DIGITlimbptr.size() != decomp_ptr_count ||
                partition.DIGITauxptr.size() != decomp_ptr_count)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected aux pointer layout in gpu_matrix_create");
            }
            partition.limbptr.data = poly_aux_base;
            partition.auxptr.data = poly_aux_base + aux_stride;
            partition.SPECIALlimbptr.data = poly_aux_base + 2 * aux_stride;
            partition.SPECIALauxptr.data = poly_aux_base + 3 * aux_stride;
            for (size_t decomp_idx = 0; decomp_idx < decomp_ptr_count; ++decomp_idx)
            {
                partition.DECOMPlimbptr[decomp_idx].data =
                    poly_aux_base + (4 + decomp_idx) * aux_stride;
                partition.DECOMPauxptr[decomp_idx].data =
                    poly_aux_base + (4 + decomp_ptr_count + decomp_idx) * aux_stride;
                partition.DIGITlimbptr[decomp_idx].data =
                    poly_aux_base + (4 + 2 * decomp_ptr_count + decomp_idx) * aux_stride;
                partition.DIGITauxptr[decomp_idx].data =
                    poly_aux_base + (4 + 3 * decomp_ptr_count + decomp_idx) * aux_stride;
            }
            aux_slots_used[partition_idx] = aux_offset_slots + aux_poly_slots;

            const size_t limbs = limb_counts[partition_idx];
            if (limbs == 0)
            {
                continue;
            }

            uint64_t *base = device_bases[partition_idx];
            if (!base)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("missing shared limb buffer in gpu_matrix_create");
            }

            const size_t poly_words = words_per_poly[partition_idx];
            const size_t limb_words = poly_words / 2;
            const size_t offset_words = words_used[partition_idx];
            if (offset_words > words_total[partition_idx] ||
                poly_words > words_total[partition_idx] - offset_words)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("shared limb buffer overflow in gpu_matrix_create");
            }

            partition.generate(
                partition.meta,
                partition.limb,
                partition.limbptr,
                static_cast<int>(limbs) - 1,
                &partition.auxptr,
                base,
                offset_words,
                base,
                offset_words + limb_words);
            words_used[partition_idx] = offset_words + poly_words;
        }

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

    destroy_matrix_contents(mat);
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
        int status = gpu_poly_equal(lhs->polys[i], rhs->polys[i], &poly_equal);
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

    const size_t count = rows * cols;
    if (count == 0)
    {
        out->format = src->format;
        return 0;
    }

    std::vector<size_t> src_indices;
    std::vector<size_t> dst_indices;
    src_indices.reserve(count);
    dst_indices.reserve(count);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            src_indices.push_back(matrix_index(src_row + i, src_col + j, src->cols));
            dst_indices.push_back(matrix_index(dst_row + i, dst_col + j, out->cols));
        }
    }

    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_copy_block");
    }
    const int N = src->ctx->N;
    if (N <= 0)
    {
        out->format = src->format;
        return 0;
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_copy_block");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        const GpuPoly *src_poly0 = src->polys[src_indices[0]];
        if (!src_poly0)
        {
            return set_error("null source polynomial in gpu_matrix_copy_block");
        }
        if (limb_id.x >= src_poly0->poly->GPU.size())
        {
            return set_error("unexpected limb GPU partition in gpu_matrix_copy_block");
        }
        const auto &src_partition0 = src_poly0->poly->GPU[limb_id.x];
        if (limb_id.y >= src_partition0.limb.size())
        {
            return set_error("unexpected limb index in gpu_matrix_copy_block");
        }

        const auto &limb_impl = src_partition0.limb[limb_id.y];
        int status = 0;
        if (limb_impl.index() == FIDESlib::U64)
        {
            status = launch_copy_for_limb<uint64_t>(
                out,
                src,
                src_indices,
                dst_indices,
                limb_id,
                static_cast<size_t>(N));
        }
        else if (limb_impl.index() == FIDESlib::U32)
        {
            status = launch_copy_for_limb<uint32_t>(
                out,
                src,
                src_indices,
                dst_indices,
                limb_id,
                static_cast<size_t>(N));
        }
        else
        {
            return set_error("unsupported limb type in gpu_matrix_copy_block");
        }
        if (status != 0)
        {
            return status;
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        const size_t src_idx = src_indices[i];
        const size_t dst_idx = dst_indices[i];
        if (!out->polys[dst_idx] || !src->polys[src_idx])
        {
            return set_error("null polynomial after copy in gpu_matrix_copy_block");
        }
        out->polys[dst_idx]->level = src->polys[src_idx]->level;
        out->polys[dst_idx]->format = src->polys[src_idx]->format;
    }

    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_fill_gadget(
    GpuMatrix *out,
    uint32_t base_bits)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_fill_gadget arguments");
    }
    if (base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid base_bits in gpu_matrix_fill_gadget");
    }

    const size_t rows = out->rows;
    const size_t cols = out->cols;
    const size_t count = rows * cols;
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_fill_gadget");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (out->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_fill_gadget");
    }
    auto &limb_map = out->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_fill_gadget");
    }

    uint32_t crt_bits = 0;
    for (size_t i = 0; i < crt_depth; ++i)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(out->ctx->moduli[i]));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_fill_gadget");
    }
    const uint32_t digits_per_tower = static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_fill_gadget");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (cols != rows * log_base_q)
    {
        return set_error("output size mismatch in gpu_matrix_fill_gadget");
    }

    struct FillBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> moduli;
        std::vector<uint32_t> limb_indices;
    };
    std::vector<FillBatch> batches;
    auto get_batch = [&](int device) -> FillBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(FillBatch{device, nullptr, {}, {}, {}});
        return batches.back();
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> limb_ptrs;
        limb_ptrs.reserve(count);
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out->polys[idx];
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                return set_error("invalid output poly in gpu_matrix_fill_gadget");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_matrix_fill_gadget");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_matrix_fill_gadget");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                return set_error("unsupported limb type in gpu_matrix_fill_gadget");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            if (limb_device == -1)
            {
                limb_device = partition.device;
                limb_stream = limb_u64.stream.ptr;
            }
            else if (limb_device != partition.device)
            {
                return set_error("inconsistent limb device in gpu_matrix_fill_gadget");
            }
            limb_ptrs.push_back(limb_u64.v.data);
        }

        if (limb_device < 0)
        {
            return set_error("invalid limb device in gpu_matrix_fill_gadget");
        }
        auto &batch = get_batch(limb_device);
        if (!batch.stream)
        {
            batch.stream = limb_stream;
        }
        batch.moduli.push_back(out->ctx->moduli[static_cast<size_t>(limb)]);
        batch.limb_indices.push_back(static_cast<uint32_t>(limb));
        batch.dst_ptrs.insert(batch.dst_ptrs.end(), limb_ptrs.begin(), limb_ptrs.end());
    }

    for (auto &batch : batches)
    {
        if (!batch.stream)
        {
            return set_error("null stream in gpu_matrix_fill_gadget");
        }
        cudaError_t err = cudaSetDevice(batch.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int status = launch_fill_gadget_multi_limb_kernel(
            batch.dst_ptrs,
            count,
            static_cast<size_t>(out->ctx->N),
            batch.moduli,
            batch.limb_indices,
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            base_bits,
            batch.stream);
        if (status != 0)
        {
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        poly->format = PolyFormat::Coeff;
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;
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

    GpuMatrix *tmp_inputs_matrix = nullptr;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        for (size_t i = 0; i < count; ++i)
        {
            int status = sync_poly_partition_streams(
                src->polys[i],
                "failed to synchronize source partition stream before clone in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = sync_poly_limb_streams(
                src->polys[i],
                "failed to synchronize source limb stream before clone in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
        }

        const int matrix_format =
            src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        int status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = gpu_matrix_copy(tmp_inputs_matrix, src);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }

        for (auto *clone : tmp_inputs_matrix->polys)
        {
            status = sync_poly_partition_streams(
                clone,
                "failed to synchronize clone partition stream before gpu_poly_intt in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = sync_poly_limb_streams(
                clone,
                "failed to synchronize clone limb stream before gpu_poly_intt in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
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

    // Ensure all pending source-side INTT work is finished before one-shot kernels.
    for (int device : src->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(out->polys.size() * crt_depth);

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output poly in gpu_matrix_decompose_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb GPU partition in gpu_matrix_decompose_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb index in gpu_matrix_decompose_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                cleanup_tmp_inputs();
                return set_error("unsupported limb type in gpu_matrix_decompose_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }

            bool seen_stream = false;
            for (const auto &entry : out_zero_streams)
            {
                if (entry.first == partition.device && entry.second == limb_u64.stream.ptr)
                {
                    seen_stream = true;
                    break;
                }
            }
            if (!seen_stream)
            {
                out_zero_streams.emplace_back(partition.device, limb_u64.stream.ptr);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    // Output limbs are zeroed asynchronously on per-limb streams.
    // Synchronize those streams once before decomposition kernels start.
    for (const auto &entry : out_zero_streams)
    {
        cudaError_t err = cudaSetDevice(entry.first);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.second);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    struct DecomposeBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<const uint64_t *> src_ptrs;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint32_t> shifts;
        std::vector<uint64_t> masks;
        std::vector<uint64_t> out_moduli;
    };
    std::vector<DecomposeBatch> batches;
    auto get_batch = [&](int device) -> DecomposeBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(DecomposeBatch{device, nullptr, {}, {}, {}, {}, {}});
        return batches.back();
    };

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb GPU partition in gpu_matrix_decompose_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb index in gpu_matrix_decompose_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            cleanup_tmp_inputs();
            return set_error("unsupported source limb type in gpu_matrix_decompose_base");
        }

        const uint32_t src_bits = bit_width_u64(src->ctx->moduli[static_cast<size_t>(src_limb)]);

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const uint32_t shift = digit_idx * base_bits;
            uint64_t mask = 0;
            if (shift < src_bits)
            {
                const uint32_t remaining = src_bits - shift;
                const uint32_t digit_bits = std::min(base_bits, remaining);
                mask = digit_bits >= 64 ? std::numeric_limits<uint64_t>::max()
                                        : ((uint64_t{1} << digit_bits) - 1);
            }

            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    cleanup_tmp_inputs();
                    return set_error("unexpected input source limb index in gpu_matrix_decompose_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    cleanup_tmp_inputs();
                    return set_error("unsupported input source limb type in gpu_matrix_decompose_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;
                int out_device = -1;

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("input/output limb device mismatch in gpu_matrix_decompose_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        cleanup_tmp_inputs();
                        return set_error("unexpected output limb index in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        cleanup_tmp_inputs();
                        return set_error("unsupported output limb type in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                        out_device = out_partition.device;
                    }
                    else if (out_device != out_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("inconsistent output limb device in gpu_matrix_decompose_base");
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                if (out_device < 0 || !out_stream)
                {
                    cleanup_tmp_inputs();
                    return set_error("invalid output stream/device in gpu_matrix_decompose_base");
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
                }
                batch_ref.shifts.push_back(shift);
                batch_ref.masks.push_back(mask);
                batch_ref.out_moduli.push_back(src->ctx->moduli[static_cast<size_t>(out_limb)]);
                batch_ref.src_ptrs.insert(batch_ref.src_ptrs.end(), src_ptrs.begin(), src_ptrs.end());
                batch_ref.dst_ptrs.insert(batch_ref.dst_ptrs.end(), dst_ptrs.begin(), dst_ptrs.end());
            }
        }
    }

    for (auto &batch_ref : batches)
    {
        if (!batch_ref.stream)
        {
            cleanup_tmp_inputs();
            return set_error("null stream in gpu_matrix_decompose_base");
        }
        cudaError_t err = cudaSetDevice(batch_ref.device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        int status = launch_decompose_multi_kernel<uint64_t>(
            batch_ref.src_ptrs,
            batch_ref.dst_ptrs,
            count,
            static_cast<size_t>(src->ctx->N),
            batch_ref.shifts,
            batch_ref.masks,
            batch_ref.out_moduli,
            batch_ref.stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup_tmp_inputs();
    return 0;
}
using gpu_chacha::DeviceChaChaRng;
using gpu_chacha::rng_init;
using gpu_chacha::rng_next_u64;

__device__ __forceinline__ double uniform_open01(DeviceChaChaRng &rng)
{
    constexpr double kScale = 1.0 / 9007199254740992.0; // 2^53
    double u = static_cast<double>(rng_next_u64(rng) >> 11U) * kScale;
    if (u <= 0.0)
    {
        u = kScale;
    }
    else if (u >= 1.0)
    {
        u = 1.0 - kScale;
    }
    return u;
}

__device__ __forceinline__ double sample_standard_normal(DeviceChaChaRng &rng)
{
    double u1 = uniform_open01(rng);
    double u2 = uniform_open01(rng);
    double r = sqrt(-2.0 * log(u1));
    double theta = kTwoPi * u2;
    return r * cos(theta);
}

__device__ __forceinline__ bool karney_algorithm_h(DeviceChaChaRng &rng)
{
    double h_a = uniform_open01(rng);
    if (!(h_a < 0.5))
    {
        return true;
    }
    for (;;)
    {
        double h_b = uniform_open01(rng);
        if (!(h_b < h_a))
        {
            return false;
        }
        h_a = uniform_open01(rng);
        if (!(h_a < h_b))
        {
            return true;
        }
    }
}

__device__ __forceinline__ int32_t karney_algorithm_g(DeviceChaChaRng &rng)
{
    int32_t n = 0;
    while (karney_algorithm_h(rng))
    {
        ++n;
        if (n > 1024)
        {
            break;
        }
    }
    return n;
}

__device__ __forceinline__ bool karney_algorithm_p(DeviceChaChaRng &rng, int32_t n)
{
    while (n-- && karney_algorithm_h(rng))
    {
    }
    return n < 0;
}

__device__ __forceinline__ bool karney_algorithm_b(DeviceChaChaRng &rng, int32_t k, double x)
{
    double y = x;
    int32_t n = 0;
    double m = static_cast<double>(2 * k + 2);
    for (;; ++n)
    {
        double z = uniform_open01(rng);
        if (!(z < y))
        {
            break;
        }
        double r = uniform_open01(rng);
        if (!(r < (2.0 * static_cast<double>(k) + x) / m))
        {
            break;
        }
        y = z;
        if (n > 4096)
        {
            break;
        }
    }
    return (n % 2) == 0;
}

__device__ __forceinline__ int64_t sample_integer_karney(DeviceChaChaRng &rng, double mean, double stddev)
{
    if (!(stddev > 0.0) || !isfinite(mean) || !isfinite(stddev))
    {
        return static_cast<int64_t>(llround(mean));
    }

    int64_t ceil_std = static_cast<int64_t>(ceil(stddev));
    if (ceil_std <= 0)
    {
        return static_cast<int64_t>(llround(mean));
    }

    for (int iter = 0; iter < 1 << 16; ++iter)
    {
        int32_t k = karney_algorithm_g(rng);
        if (!karney_algorithm_p(rng, k * (k - 1)))
        {
            continue;
        }

        int64_t s = (rng_next_u64(rng) & 1ULL) ? 1 : -1;
        double di0 = stddev * static_cast<double>(k) + static_cast<double>(s) * mean;
        int64_t i0 = static_cast<int64_t>(ceil(di0));
        double x0 = (static_cast<double>(i0) - di0) / stddev;
        int64_t j = static_cast<int64_t>(rng_next_u64(rng) % static_cast<uint64_t>(ceil_std));
        double x = x0 + static_cast<double>(j) / stddev;

        if (!(x < 1.0) || (x == 0.0 && s < 0 && k == 0))
        {
            continue;
        }

        int32_t h = k + 1;
        while (h-- > 0 && karney_algorithm_b(rng, k, x))
        {
        }
        if (h >= 0)
        {
            continue;
        }

        return s * (i0 + j);
    }

    // Fallback in case the rejection loop takes too long.
    return static_cast<int64_t>(llround(mean + stddev * sample_standard_normal(rng)));
}

__device__ __forceinline__ void get_base_digits_u64(
    uint64_t value,
    uint64_t base,
    uint32_t digits,
    int64_t *out_digits)
{
    for (uint32_t i = 0; i < digits; ++i)
    {
        out_digits[i] = static_cast<int64_t>(value % base);
        value /= base;
    }
}

__device__ __forceinline__ uint64_t signed_mod_i64(int64_t value, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    if (value >= 0)
    {
        return static_cast<uint64_t>(value) % modulus;
    }
    uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
    uint64_t rem = magnitude % modulus;
    return rem == 0 ? 0 : (modulus - rem);
}

__device__ __forceinline__ uint64_t sample_uniform_mod(DeviceChaChaRng &rng, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    constexpr uint64_t kU64Max = ~uint64_t{0};
    const uint64_t threshold = kU64Max - (kU64Max % modulus);
    for (;;)
    {
        uint64_t x = rng_next_u64(rng);
        if (x < threshold)
        {
            return x % modulus;
        }
    }
}

__device__ __forceinline__ int64_t centered_residue_i64(uint64_t value, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    uint64_t reduced = value % modulus;
    uint64_t half = modulus >> 1;
    if (reduced <= half)
    {
        return static_cast<int64_t>(reduced);
    }
    uint64_t neg = modulus - reduced;
    return -static_cast<int64_t>(neg);
}

__global__ void matrix_sample_distribution_multi_limb_kernel(
    uint64_t **dst,
    size_t poly_count,
    size_t limb_count,
    size_t n,
    const uint64_t *moduli,
    const uint32_t *limb_indices,
    int dist_type,
    double sigma,
    uint64_t seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = limb_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t per_limb = poly_count * n;
    const size_t limb_slot = idx / per_limb;
    const size_t rem = idx - limb_slot * per_limb;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = limb_slot * poly_count + poly_idx;

    const uint64_t modulus = moduli[limb_slot];
    const uint32_t limb_idx = limb_indices[limb_slot];

    uint64_t sample = 0;
    if (dist_type == GPU_MATRIX_DIST_UNIFORM)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            static_cast<uint64_t>(limb_idx + 1),
            0x6f70656e66686531ULL);
        sample = sample_uniform_mod(rng, modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_GAUSS)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686532ULL);
        int64_t z = sample_integer_karney(rng, 0.0, sigma);
        sample = signed_mod_i64(z, modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_BIT)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686533ULL);
        sample = (rng_next_u64(rng) & 1ULL) % modulus;
    }
    else if (dist_type == GPU_MATRIX_DIST_TERNARY)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686534ULL);
        uint64_t pick = rng_next_u64(rng) % 3ULL;
        int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
        sample = signed_mod_i64(z, modulus);
    }

    dst[ptr_idx][coeff_idx] = sample;
}

__device__ __forceinline__ uint64_t pow_mod_u64(uint64_t base, uint32_t exp, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    uint64_t result = 1ULL % modulus;
    uint64_t cur = base % modulus;
    uint32_t e = exp;
    while (e > 0)
    {
        if (e & 1U)
        {
            result = static_cast<uint64_t>((static_cast<unsigned __int128>(result) * cur) % modulus);
        }
        e >>= 1U;
        if (e > 0)
        {
            cur = static_cast<uint64_t>((static_cast<unsigned __int128>(cur) * cur) % modulus);
        }
    }
    return result;
}

__global__ void matrix_fill_gadget_multi_limb_kernel(
    uint64_t **dst,
    size_t poly_count,
    size_t limb_count,
    size_t n,
    const uint64_t *moduli,
    const uint32_t *limb_indices,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = limb_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t per_limb = poly_count * n;
    const size_t limb_slot = idx / per_limb;
    const size_t rem = idx - limb_slot * per_limb;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = limb_slot * poly_count + poly_idx;

    const uint64_t modulus = moduli[limb_slot];
    const uint32_t limb_idx = limb_indices[limb_slot];

    uint64_t value = 0;
    if (coeff_idx == 0 && rows > 0 && cols > 0 && log_base_q > 0)
    {
        size_t row = poly_idx / cols;
        size_t col = poly_idx - row * cols;
        size_t block_start = row * log_base_q;
        if (col >= block_start && col < block_start + log_base_q)
        {
            size_t local = col - block_start;
            uint32_t tower = static_cast<uint32_t>(local / static_cast<size_t>(digits_per_tower));
            uint32_t digit = static_cast<uint32_t>(local % static_cast<size_t>(digits_per_tower));
            if (tower == limb_idx)
            {
                uint64_t base = uint64_t{1} << base_bits;
                value = pow_mod_u64(base, digit, modulus);
            }
        }
    }
    dst[ptr_idx][coeff_idx] = value;
}

__global__ void matrix_sample_p1_full_kernel(
    const uint64_t **a_entries,
    const uint64_t **b_entries,
    const uint64_t **d_entries,
    const uint64_t **tp2_entries,
    uint64_t **out_entries,
    size_t d,
    size_t cols,
    size_t n,
    size_t sample_start,
    size_t sample_count,
    double *cov_workspace,
    double *mean_workspace,
    double *col_workspace,
    int64_t *sampled_workspace,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint32_t limb_idx,
    uint64_t seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= sample_count)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0)
    {
        return;
    }

    const size_t sample_idx = sample_start + idx;
    const size_t col_idx = sample_idx / n;
    const size_t coeff_idx = sample_idx - col_idx * n;
    const size_t cov_stride = m * m;
    const size_t vec_stride = m;
    double *cov = cov_workspace + idx * cov_stride;
    double *mean = mean_workspace + idx * vec_stride;
    double *col_buf = col_workspace + idx * vec_stride;
    int64_t *sampled = sampled_workspace + idx * vec_stride;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        static_cast<uint64_t>(limb_idx + 1),
        0x7065727475726231ULL);

    const double sigma2 = sigma * sigma;
    const double s2 = s * s;
    const double denom = s2 - sigma2;
    if (!(denom > 0.0))
    {
        return;
    }
    const double c_scale = -sigma2 / denom;
    const double fallback_var = dgg_stddev * dgg_stddev;
    const double eps = 1e-9;

    for (size_t i = 0; i < d; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            const size_t ij = matrix_index(i, j, d);
            const size_t ji = matrix_index(j, i, d);
            const double a_ij = static_cast<double>(centered_residue_i64(a_entries[ij][coeff_idx], modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(d_entries[ij][coeff_idx], modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(b_entries[ij][coeff_idx], modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(b_entries[ji][coeff_idx], modulus));

            const double af = -sigma2 * a_ij + (i == j ? s2 : 0.0);
            const double df = -sigma2 * d_ij + (i == j ? s2 : 0.0);
            const double bf = -sigma2 * b_ij;
            const double bt = -sigma2 * b_ji;

            cov[matrix_index(i, j, m)] = af;
            cov[matrix_index(i + d, j + d, m)] = df;
            cov[matrix_index(i, j + d, m)] = bf;
            cov[matrix_index(i + d, j, m)] = bt;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(tp2_entries[tp_idx][coeff_idx], modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        double var = cov[matrix_index(tt, tt, m)];
        if (!(var > eps))
        {
            var = fallback_var;
        }
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt(var));
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            col_buf[static_cast<size_t>(i)] =
                cov[matrix_index(static_cast<size_t>(i), tt, m)];
        }

        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                (col_buf[static_cast<size_t>(i)] / var) * delta;
        }

        for (int i = 0; i < t; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                double updated = cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                                 (col_buf[static_cast<size_t>(i)] * col_buf[static_cast<size_t>(j)] / var);
                cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
            }
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols);
        out_entries[out_idx][coeff_idx] = signed_mod_i64(sampled[row], modulus);
    }
}

__global__ void matrix_gauss_samp_gq_arb_base_multi_kernel(
    const uint64_t **src,
    uint64_t **dst,
    size_t poly_count,
    size_t n,
    size_t job_count,
    const uint64_t *tower_moduli,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    const uint32_t *digit_indices,
    double c,
    const uint32_t *tower_indices,
    uint64_t seed,
    const uint64_t *out_moduli)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = job_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
    {
        return;
    }

    const size_t per_job = poly_count * n;
    const size_t job_idx = idx / per_job;
    const size_t rem = idx - job_idx * per_job;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = job_idx * poly_count + poly_idx;

    const uint64_t tower_modulus = tower_moduli[job_idx];
    const uint64_t out_modulus = out_moduli[job_idx];
    const uint32_t digit_idx = digit_indices[job_idx];
    const uint32_t tower_idx = tower_indices[job_idx];

    uint64_t value = src[ptr_idx][coeff_idx];
    if (tower_modulus != 0)
    {
        value %= tower_modulus;
    }

    uint64_t base = uint64_t{1} << base_bits;
    double base_f = static_cast<double>(base);
    double sigma = c / (base_f + 1.0);

    int64_t m_digits[kGaussMaxDigits];
    int64_t v_digits[kGaussMaxDigits];
    double l[kGaussMaxDigits];
    double h[kGaussMaxDigits];
    double c_vec[kGaussMaxDigits];
    double p[kGaussMaxDigits];
    double a[kGaussMaxDigits];
    double zf[kGaussMaxDigits];
    int64_t z[kGaussMaxDigits];

    get_base_digits_u64(tower_modulus, base, digits_per_tower, m_digits);
    get_base_digits_u64(value, base, digits_per_tower, v_digits);

    const double kf = static_cast<double>(digits_per_tower);
    l[0] = sqrt(base_f * (1.0 + 1.0 / kf) + 1.0);
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        l[i] = sqrt(base_f * (1.0 + 1.0 / (kf - static_cast<double>(i))));
    }

    h[0] = 0.0;
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        h[i] = sqrt(base_f * (1.0 - 1.0 / (kf - static_cast<double>(i - 1))));
    }

    c_vec[0] = static_cast<double>(m_digits[0]) / base_f;
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        c_vec[i] = (c_vec[i - 1] + static_cast<double>(m_digits[i])) / base_f;
    }

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(tower_idx + 1),
        static_cast<uint64_t>(poly_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0x6761646765746731ULL);

    for (uint32_t i = 0; i < digits_per_tower; ++i)
    {
        zf[i] = sigma * sample_standard_normal(rng);
    }
    for (uint32_t i = 0; i + 1 < digits_per_tower; ++i)
    {
        p[i] = l[i] * zf[i] + h[i + 1] * zf[i + 1];
    }
    p[digits_per_tower - 1] = h[digits_per_tower - 1] * zf[digits_per_tower - 1];

    a[0] = (static_cast<double>(v_digits[0]) - p[0]) / base_f;
    for (uint32_t t = 1; t < digits_per_tower; ++t)
    {
        a[t] = (a[t - 1] + static_cast<double>(v_digits[t]) - p[t]) / base_f;
    }

    const uint32_t last = digits_per_tower - 1;
    z[last] = sample_integer_karney(rng, -a[last] / c_vec[last], sigma / c_vec[last]);
    for (uint32_t i = 0; i < digits_per_tower; ++i)
    {
        a[i] += static_cast<double>(z[last]) * c_vec[i];
    }
    for (uint32_t i = 0; i < last; ++i)
    {
        z[i] = sample_integer_karney(rng, -a[i], sigma);
    }

    int64_t out_digit = 0;
    if (digits_per_tower == 1)
    {
        out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[0] + v_digits[0];
    }
    else if (digit_idx == 0)
    {
        out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[last] + v_digits[0];
    }
    else if (digit_idx < last)
    {
        out_digit = static_cast<int64_t>(base) * z[digit_idx] - z[digit_idx - 1] +
                    m_digits[digit_idx] * z[last] + v_digits[digit_idx];
    }
    else
    {
        out_digit = m_digits[last] * z[last] - z[last - 1] + v_digits[last];
    }

    dst[ptr_idx][coeff_idx] = signed_mod_i64(out_digit, out_modulus);
}

int launch_gauss_samp_gq_arb_base_multi_kernel(
    const std::vector<const uint64_t *> &src_ptrs,
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &tower_moduli,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    const std::vector<uint32_t> &digit_indices,
    double c,
    const std::vector<uint32_t> &tower_indices,
    uint64_t seed,
    const std::vector<uint64_t> &out_moduli,
    int device,
    cudaStream_t stream)
{
    const size_t job_count = tower_moduli.size();
    if (job_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (digit_indices.size() != job_count || tower_indices.size() != job_count ||
        out_moduli.size() != job_count)
    {
        return set_error("unexpected job parameter counts in matrix_gauss_samp_gq_arb_base_multi_kernel");
    }
    const size_t ptr_count = job_count * poly_count;
    if (src_ptrs.size() != ptr_count || dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_gauss_samp_gq_arb_base_multi_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const uint64_t **d_src = nullptr;
    uint64_t **d_dst = nullptr;
    uint64_t *d_tower_moduli = nullptr;
    uint32_t *d_digit_indices = nullptr;
    uint32_t *d_tower_indices = nullptr;
    uint64_t *d_out_moduli = nullptr;

    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = job_count * sizeof(uint64_t);
    const size_t u32_bytes = job_count * sizeof(uint32_t);

    err = cudaMallocAsync(&d_src, ptr_bytes, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMallocAsync(&d_dst, ptr_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_tower_moduli, u64_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_digit_indices, u32_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_tower_indices, u32_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_out_moduli, u64_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_src, src_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_moduli, tower_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_digit_indices, digit_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_indices, tower_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_out_moduli, out_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_gauss_samp_gq_arb_base_multi_kernel<<<blocks, threads, 0, stream>>>(
        d_src,
        d_dst,
        poly_count,
        n,
        job_count,
        d_tower_moduli,
        base_bits,
        digits_per_tower,
        d_digit_indices,
        c,
        d_tower_indices,
        seed,
        d_out_moduli);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    err = cudaFreeAsync(d_src, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_dst, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_tower_moduli, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_digit_indices, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_tower_indices, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_out_moduli, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int launch_sample_distribution_multi_limb_kernel(
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &moduli,
    const std::vector<uint32_t> &limb_indices,
    int dist_type,
    double sigma,
    uint64_t seed,
    cudaStream_t stream)
{
    const size_t limb_count = moduli.size();
    if (limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (limb_indices.size() != limb_count)
    {
        return set_error("unexpected limb parameter counts in matrix_sample_distribution_multi_limb_kernel");
    }
    const size_t ptr_count = limb_count * poly_count;
    if (dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_sample_distribution_multi_limb_kernel");
    }

    uint64_t **d_dst = nullptr;
    uint64_t *d_moduli = nullptr;
    uint32_t *d_limb_indices = nullptr;
    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = limb_count * sizeof(uint64_t);
    const size_t u32_bytes = limb_count * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&d_dst, ptr_bytes);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMalloc(&d_moduli, u64_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        return set_error(err);
    }
    err = cudaMalloc(&d_limb_indices, u32_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_moduli, moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_limb_indices, limb_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_sample_distribution_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        d_dst,
        poly_count,
        limb_count,
        n,
        d_moduli,
        d_limb_indices,
        dist_type,
        sigma,
        seed);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    cudaFree(d_dst);
    cudaFree(d_moduli);
    cudaFree(d_limb_indices);
    return 0;
}

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
    cudaStream_t stream)
{
    const size_t limb_count = moduli.size();
    if (limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (limb_indices.size() != limb_count)
    {
        return set_error("unexpected limb parameter counts in matrix_fill_gadget_multi_limb_kernel");
    }
    const size_t ptr_count = limb_count * poly_count;
    if (dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_fill_gadget_multi_limb_kernel");
    }

    uint64_t **d_dst = nullptr;
    uint64_t *d_moduli = nullptr;
    uint32_t *d_limb_indices = nullptr;
    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = limb_count * sizeof(uint64_t);
    const size_t u32_bytes = limb_count * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&d_dst, ptr_bytes);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMalloc(&d_moduli, u64_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        return set_error(err);
    }
    err = cudaMalloc(&d_limb_indices, u32_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_moduli, moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_limb_indices, limb_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_fill_gadget_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        d_dst,
        poly_count,
        limb_count,
        n,
        d_moduli,
        d_limb_indices,
        rows,
        cols,
        log_base_q,
        digits_per_tower,
        base_bits);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    cudaFree(d_dst);
    cudaFree(d_moduli);
    cudaFree(d_limb_indices);
    return 0;
}

int launch_sample_p1_full_kernel(
    const std::vector<const uint64_t *> &a_entries,
    const std::vector<const uint64_t *> &b_entries,
    const std::vector<const uint64_t *> &d_entries,
    const std::vector<const uint64_t *> &tp2_entries,
    const std::vector<uint64_t *> &out_entries,
    size_t d,
    size_t cols,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint32_t limb_idx,
    uint64_t seed,
    cudaStream_t stream,
    int device_id)
{
    if (d == 0 || cols == 0 || n == 0)
    {
        return 0;
    }
    const size_t mat_entries = d * d;
    const size_t vec_entries = 2 * d * cols;
    if (a_entries.size() != mat_entries || b_entries.size() != mat_entries ||
        d_entries.size() != mat_entries || tp2_entries.size() != vec_entries ||
        out_entries.size() != vec_entries)
    {
        return set_error("unexpected pointer counts in matrix_sample_p1_full_kernel");
    }

    if (device_id < 0)
    {
        return set_error("invalid device in matrix_sample_p1_full_kernel");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t m = 2 * d;
    if (m == 0)
    {
        return set_error("invalid dimension in matrix_sample_p1_full_kernel");
    }
    if (m > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    const size_t cov_elems_per_sample = m * m;
    if (cov_elems_per_sample > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    const size_t cov_bytes_per_sample = cov_elems_per_sample * sizeof(double);
    if (m > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    const size_t vec_bytes_per_sample = m * sizeof(double);
    if (m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
    if (cov_bytes_per_sample > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    size_t bytes_per_sample_total = cov_bytes_per_sample + vec_bytes_per_sample;
    if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    bytes_per_sample_total += vec_bytes_per_sample;
    if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_full_kernel");
    }
    bytes_per_sample_total += sampled_bytes_per_sample;

    const size_t total_samples = cols * n;
    size_t chunk_samples = total_samples;

    const uint64_t **d_a_entries = nullptr;
    const uint64_t **d_b_entries = nullptr;
    const uint64_t **d_d_entries = nullptr;
    const uint64_t **d_tp2_entries = nullptr;
    uint64_t **d_out_entries = nullptr;
    const size_t mat_bytes = mat_entries * sizeof(uint64_t *);
    const size_t vec_bytes = vec_entries * sizeof(uint64_t *);

    auto free_all = [&]()
    {
        if (d_a_entries)
            cudaFree(d_a_entries);
        if (d_b_entries)
            cudaFree(d_b_entries);
        if (d_d_entries)
            cudaFree(d_d_entries);
        if (d_tp2_entries)
            cudaFree(d_tp2_entries);
        if (d_out_entries)
            cudaFree(d_out_entries);
    };

    err = cudaMalloc(&d_a_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_b_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_d_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_tp2_entries, vec_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_out_entries, vec_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_a_entries, a_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_b_entries, b_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_d_entries, d_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tp2_entries, tp2_entries.data(), vec_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_out_entries, out_entries.data(), vec_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    double *cov_workspace = nullptr;
    double *mean_workspace = nullptr;
    double *col_workspace = nullptr;
    int64_t *sampled_workspace = nullptr;
    auto free_workspace = [&]()
    {
        if (cov_workspace)
            cudaFree(cov_workspace);
        if (mean_workspace)
            cudaFree(mean_workspace);
        if (col_workspace)
            cudaFree(col_workspace);
        if (sampled_workspace)
            cudaFree(sampled_workspace);
        cov_workspace = nullptr;
        mean_workspace = nullptr;
        col_workspace = nullptr;
        sampled_workspace = nullptr;
    };

    auto alloc_workspace = [&](size_t samples) -> bool
    {
        if (samples == 0)
        {
            return false;
        }
        if (samples > std::numeric_limits<size_t>::max() / cov_bytes_per_sample ||
            samples > std::numeric_limits<size_t>::max() / vec_bytes_per_sample ||
            samples > std::numeric_limits<size_t>::max() / sampled_bytes_per_sample)
        {
            return false;
        }
        cudaError_t local_err = cudaMalloc(&cov_workspace, samples * cov_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&mean_workspace, samples * vec_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&col_workspace, samples * vec_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&sampled_workspace, samples * sampled_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        return true;
    };

    while (!alloc_workspace(chunk_samples))
    {
        if (chunk_samples <= 1)
        {
            free_all();
            return set_error("failed to allocate workspace in matrix_sample_p1_full_kernel");
        }
        chunk_samples = (chunk_samples + 1) / 2;
    }

    const int threads = 256;
    for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
    {
        size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
        const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
        matrix_sample_p1_full_kernel<<<blocks, threads, 0, stream>>>(
            d_a_entries,
            d_b_entries,
            d_d_entries,
            d_tp2_entries,
            d_out_entries,
            d,
            cols,
            n,
            sample_start,
            sample_count,
            cov_workspace,
            mean_workspace,
            col_workspace,
            sampled_workspace,
            modulus,
            sigma,
            s,
            dgg_stddev,
            limb_idx,
            seed);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            free_workspace();
            free_all();
            return set_error(err);
        }
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_workspace();
        free_all();
        return set_error(err);
    }

    free_workspace();
    free_all();
    return 0;
}

extern "C" int gpu_matrix_sample_distribution(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_sample_distribution arguments");
    }
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
    {
        return set_error("invalid dist_type in gpu_matrix_sample_distribution");
    }
    if (dist_type == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0))
    {
        return set_error("sigma must be positive in gpu_matrix_sample_distribution");
    }

    const size_t count = out->rows * out->cols;
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_distribution");
    }
    if (out->ctx->moduli.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_distribution");
    }

    auto &limb_map = out->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_distribution");
    }

    struct DistBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> moduli;
        std::vector<uint32_t> limb_indices;
    };
    std::vector<DistBatch> batches;
    auto get_batch = [&](int device) -> DistBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(DistBatch{device, nullptr, {}, {}, {}});
        return batches.back();
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> limb_ptrs;
        limb_ptrs.reserve(count);
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out->polys[idx];
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                return set_error("invalid output poly in gpu_matrix_sample_distribution");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_matrix_sample_distribution");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_matrix_sample_distribution");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                return set_error("unsupported limb type in gpu_matrix_sample_distribution");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            if (limb_device == -1)
            {
                limb_device = partition.device;
                limb_stream = limb_u64.stream.ptr;
            }
            else if (limb_device != partition.device)
            {
                return set_error("inconsistent limb device in gpu_matrix_sample_distribution");
            }
            limb_ptrs.push_back(limb_u64.v.data);
        }

        if (limb_device < 0)
        {
            return set_error("invalid limb device in gpu_matrix_sample_distribution");
        }
        auto &batch = get_batch(limb_device);
        if (!batch.stream)
        {
            batch.stream = limb_stream;
        }
        batch.moduli.push_back(out->ctx->moduli[static_cast<size_t>(limb)]);
        batch.limb_indices.push_back(static_cast<uint32_t>(limb));
        batch.dst_ptrs.insert(batch.dst_ptrs.end(), limb_ptrs.begin(), limb_ptrs.end());
    }

    for (auto &batch : batches)
    {
        if (!batch.stream)
        {
            return set_error("null stream in gpu_matrix_sample_distribution");
        }
        cudaError_t err = cudaSetDevice(batch.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int status = launch_sample_distribution_multi_limb_kernel(
            batch.dst_ptrs,
            count,
            static_cast<size_t>(out->ctx->N),
            batch.moduli,
            batch.limb_indices,
            dist_type,
            sigma,
            seed,
            batch.stream);
        if (status != 0)
        {
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        poly->format = PolyFormat::Coeff;
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;
    return 0;
}
extern "C" int gpu_matrix_gauss_samp_gq_arb_base(
    const GpuMatrix *src,
    uint32_t base_bits,
    double c,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    (void)dgg_stddev;
    if (!src || !out)
    {
        return set_error("invalid gpu_matrix_gauss_samp_gq_arb_base arguments");
    }
    if (base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid base_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (!(c > 0.0))
    {
        return set_error("c must be positive in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    uint32_t crt_bits = 0;
    for (const auto &modulus : src->ctx->moduli)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const uint32_t digits_per_tower = static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    GpuMatrix *tmp_inputs_matrix = nullptr;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        for (size_t i = 0; i < count; ++i)
        {
            int sync_status = sync_poly_partition_streams(
                src->polys[i],
                "failed to synchronize source partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                src->polys[i],
                "failed to synchronize source limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
        }

        const int matrix_format =
            src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        int status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = gpu_matrix_copy(tmp_inputs_matrix, src);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }

        for (auto *clone : tmp_inputs_matrix->polys)
        {
            int sync_status = sync_poly_partition_streams(
                clone,
                "failed to synchronize clone partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                clone,
                "failed to synchronize clone limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
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

    for (int device : src->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(out->polys.size() * crt_depth);

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output poly in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb index in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                cleanup_tmp_inputs();
                return set_error("unsupported limb type in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }

            bool seen_stream = false;
            for (const auto &entry : out_zero_streams)
            {
                if (entry.first == partition.device && entry.second == limb_u64.stream.ptr)
                {
                    seen_stream = true;
                    break;
                }
            }
            if (!seen_stream)
            {
                out_zero_streams.emplace_back(partition.device, limb_u64.stream.ptr);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    for (const auto &entry : out_zero_streams)
    {
        cudaError_t err = cudaSetDevice(entry.first);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.second);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_gauss_samp_gq_arb_base");
    }

    struct GaussBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<const uint64_t *> src_ptrs;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> tower_moduli;
        std::vector<uint32_t> digit_indices;
        std::vector<uint32_t> tower_indices;
        std::vector<uint64_t> out_moduli;
    };
    std::vector<GaussBatch> batches;
    auto get_batch = [&](int device) -> GaussBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(GaussBatch{device, nullptr, {}, {}, {}, {}, {}, {}});
        return batches.back();
    };

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            cleanup_tmp_inputs();
            return set_error("unsupported source limb type in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    cleanup_tmp_inputs();
                    return set_error("unexpected input source limb index in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    cleanup_tmp_inputs();
                    return set_error("unsupported input source limb type in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;
                int out_device = -1;

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("input/output limb device mismatch in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        cleanup_tmp_inputs();
                        return set_error("unexpected output limb index in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        cleanup_tmp_inputs();
                        return set_error("unsupported output limb type in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                        out_device = out_partition.device;
                    }
                    else if (out_device != out_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("inconsistent output limb device in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                if (out_device < 0 || !out_stream)
                {
                    cleanup_tmp_inputs();
                    return set_error("invalid output stream/device in gpu_matrix_gauss_samp_gq_arb_base");
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
                }
                batch_ref.tower_moduli.push_back(src->ctx->moduli[static_cast<size_t>(src_limb)]);
                batch_ref.digit_indices.push_back(digit_idx);
                batch_ref.tower_indices.push_back(static_cast<uint32_t>(src_limb));
                batch_ref.out_moduli.push_back(src->ctx->moduli[static_cast<size_t>(out_limb)]);
                batch_ref.src_ptrs.insert(batch_ref.src_ptrs.end(), src_ptrs.begin(), src_ptrs.end());
                batch_ref.dst_ptrs.insert(batch_ref.dst_ptrs.end(), dst_ptrs.begin(), dst_ptrs.end());
            }
        }
    }

    for (auto &batch_ref : batches)
    {
        if (!batch_ref.stream)
        {
            cleanup_tmp_inputs();
            return set_error("null stream in gpu_matrix_gauss_samp_gq_arb_base");
        }
        int status = launch_gauss_samp_gq_arb_base_multi_kernel(
            batch_ref.src_ptrs,
            batch_ref.dst_ptrs,
            count,
            static_cast<size_t>(src->ctx->N),
            batch_ref.tower_moduli,
            base_bits,
            digits_per_tower,
            batch_ref.digit_indices,
            c,
            batch_ref.tower_indices,
            seed,
            batch_ref.out_moduli,
            batch_ref.device,
            batch_ref.stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup_tmp_inputs();
    return 0;
}

extern "C" int gpu_matrix_sample_p1_full(
    const GpuMatrix *a_mat,
    const GpuMatrix *b_mat,
    const GpuMatrix *d_mat,
    const GpuMatrix *tp2,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    if (!a_mat || !b_mat || !d_mat || !tp2 || !out)
    {
        return set_error("invalid gpu_matrix_sample_p1_full arguments");
    }
    if (!(sigma > 0.0) || !(s > sigma))
    {
        return set_error("invalid sigma/s in gpu_matrix_sample_p1_full");
    }
    if (!(dgg_stddev > 0.0))
    {
        return set_error("dgg_stddev must be positive in gpu_matrix_sample_p1_full");
    }
    if (a_mat->ctx != b_mat->ctx || a_mat->ctx != d_mat->ctx || a_mat->ctx != tp2->ctx || a_mat->ctx != out->ctx)
    {
        return set_error("context mismatch in gpu_matrix_sample_p1_full");
    }
    if (a_mat->level != b_mat->level || a_mat->level != d_mat->level ||
        a_mat->level != tp2->level || a_mat->level != out->level)
    {
        return set_error("level mismatch in gpu_matrix_sample_p1_full");
    }

    const size_t d_rows = a_mat->rows;
    if (a_mat->cols != d_rows || b_mat->rows != d_rows || b_mat->cols != d_rows ||
        d_mat->rows != d_rows || d_mat->cols != d_rows)
    {
        return set_error("A/B/D must be dxd in gpu_matrix_sample_p1_full");
    }
    const size_t cols = tp2->cols;
    if (tp2->rows != 2 * d_rows || out->rows != 2 * d_rows || out->cols != cols)
    {
        return set_error("tp2/out shape mismatch in gpu_matrix_sample_p1_full");
    }
    if (cols == 0 || d_rows == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = a_mat->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_p1_full");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (a_mat->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_p1_full");
    }
    auto &limb_map = a_mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_p1_full");
    }

    GpuMatrix *tmp_a = nullptr;
    GpuMatrix *tmp_b = nullptr;
    GpuMatrix *tmp_d = nullptr;
    GpuMatrix *tmp_tp2 = nullptr;
    std::vector<const GpuPoly *> a_inputs;
    std::vector<const GpuPoly *> b_inputs;
    std::vector<const GpuPoly *> d_inputs;
    std::vector<const GpuPoly *> tp2_inputs;

    auto cleanup = [&]()
    {
        if (tmp_a)
        {
            gpu_matrix_destroy(tmp_a);
            tmp_a = nullptr;
        }
        if (tmp_b)
        {
            gpu_matrix_destroy(tmp_b);
            tmp_b = nullptr;
        }
        if (tmp_d)
        {
            gpu_matrix_destroy(tmp_d);
            tmp_d = nullptr;
        }
        if (tmp_tp2)
        {
            gpu_matrix_destroy(tmp_tp2);
            tmp_tp2 = nullptr;
        }
    };

    auto collect_coeff_inputs = [&](const GpuMatrix *src, GpuMatrix **owned, std::vector<const GpuPoly *> &inputs) -> int
    {
        const size_t count = src->rows * src->cols;
        inputs.clear();
        inputs.reserve(count);
        const int batch = default_batch(src->ctx);
        *owned = nullptr;
        if (src->format == PolyFormat::Eval)
        {
            const int matrix_format =
                src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            int status = gpu_matrix_create(
                src->ctx,
                src->level,
                src->rows,
                src->cols,
                matrix_format,
                owned);
            if (status != 0)
            {
                return status;
            }
            status = gpu_matrix_copy(*owned, src);
            if (status != 0)
            {
                gpu_matrix_destroy(*owned);
                *owned = nullptr;
                return status;
            }
            for (auto *clone : (*owned)->polys)
            {
                status = gpu_poly_intt(clone, batch);
                if (status != 0)
                {
                    gpu_matrix_destroy(*owned);
                    *owned = nullptr;
                    return status;
                }
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
        return 0;
    };

    int status = collect_coeff_inputs(a_mat, &tmp_a, a_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(b_mat, &tmp_b, b_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(d_mat, &tmp_d, d_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(tp2, &tmp_tp2, tp2_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    // Ensure all pending INTT work has completed before cross-stream reads.
    for (int device : a_mat->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != a_mat->ctx || poly->level != level)
        {
            cleanup();
            return set_error("invalid output poly in gpu_matrix_sample_p1_full");
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }
    out->format = PolyFormat::Coeff;

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];

        std::vector<const uint64_t *> a_entry_ptrs;
        std::vector<const uint64_t *> b_entry_ptrs;
        std::vector<const uint64_t *> d_entry_ptrs;
        std::vector<const uint64_t *> tp2_entry_ptrs;
        std::vector<uint64_t *> out_entry_ptrs;
        a_entry_ptrs.reserve(d_rows * d_rows);
        b_entry_ptrs.reserve(d_rows * d_rows);
        d_entry_ptrs.reserve(d_rows * d_rows);
        tp2_entry_ptrs.reserve(2 * d_rows * cols);
        out_entry_ptrs.reserve(2 * d_rows * cols);

        cudaStream_t out_stream = nullptr;
        int out_device = -1;
        for (size_t i = 0; i < d_rows; ++i)
        {
            for (size_t j = 0; j < d_rows; ++j)
            {
                const size_t idx = matrix_index(i, j, d_rows);
                if (limb_id.x >= a_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= b_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= d_inputs[idx]->poly->GPU.size())
                {
                    cleanup();
                    return set_error("unexpected A/B/D limb GPU partition in gpu_matrix_sample_p1_full");
                }
                const auto &a_part = a_inputs[idx]->poly->GPU[limb_id.x];
                const auto &b_part = b_inputs[idx]->poly->GPU[limb_id.x];
                const auto &d_part = d_inputs[idx]->poly->GPU[limb_id.x];
                if (limb_id.y >= a_part.limb.size() ||
                    limb_id.y >= b_part.limb.size() ||
                    limb_id.y >= d_part.limb.size())
                {
                    cleanup();
                    return set_error("unexpected A/B/D limb index in gpu_matrix_sample_p1_full");
                }
                const auto &a_impl = a_part.limb[limb_id.y];
                const auto &b_impl = b_part.limb[limb_id.y];
                const auto &d_impl = d_part.limb[limb_id.y];
                if (a_impl.index() != FIDESlib::U64 ||
                    b_impl.index() != FIDESlib::U64 ||
                    d_impl.index() != FIDESlib::U64)
                {
                    cleanup();
                    return set_error("unsupported A/B/D limb type in gpu_matrix_sample_p1_full");
                }
                a_entry_ptrs.push_back(std::get<FIDESlib::U64>(a_impl).v.data);
                b_entry_ptrs.push_back(std::get<FIDESlib::U64>(b_impl).v.data);
                d_entry_ptrs.push_back(std::get<FIDESlib::U64>(d_impl).v.data);
            }
        }
        for (size_t row = 0; row < 2 * d_rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                const size_t idx = matrix_index(row, col, cols);
                if (limb_id.x >= tp2_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= out->polys[idx]->poly->GPU.size())
                {
                    cleanup();
                    return set_error("unexpected tp2/output limb GPU partition in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_part = tp2_inputs[idx]->poly->GPU[limb_id.x];
                auto &out_part = out->polys[idx]->poly->GPU[limb_id.x];
                if (tp2_part.device != out_part.device)
                {
                    cleanup();
                    return set_error("input/output limb device mismatch in gpu_matrix_sample_p1_full");
                }
                if (out_device < 0)
                {
                    out_device = out_part.device;
                }
                else if (out_device != out_part.device)
                {
                    cleanup();
                    return set_error("mixed output devices in gpu_matrix_sample_p1_full");
                }
                if (limb_id.y >= tp2_part.limb.size() || limb_id.y >= out_part.limb.size())
                {
                    cleanup();
                    return set_error("unexpected tp2/output limb index in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_impl = tp2_part.limb[limb_id.y];
                auto &out_impl = out_part.limb[limb_id.y];
                if (tp2_impl.index() != FIDESlib::U64 || out_impl.index() != FIDESlib::U64)
                {
                    cleanup();
                    return set_error("unsupported tp2/output limb type in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_u64 = std::get<FIDESlib::U64>(tp2_impl);
                auto &out_u64 = std::get<FIDESlib::U64>(out_impl);
                if (!out_stream)
                {
                    out_stream = out_u64.stream.ptr;
                }
                tp2_entry_ptrs.push_back(tp2_u64.v.data);
                out_entry_ptrs.push_back(out_u64.v.data);
            }
        }

        status = launch_sample_p1_full_kernel(
            a_entry_ptrs,
            b_entry_ptrs,
            d_entry_ptrs,
            tp2_entry_ptrs,
            out_entry_ptrs,
            d_rows,
            cols,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            sigma,
            s,
            dgg_stddev,
            static_cast<uint32_t>(limb),
            seed,
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup();
    return 0;
}
