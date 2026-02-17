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
            return set_error("unexpected pointer counts in launch_block_matmul_kernel");
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

    template <typename T>
    int launch_copy_for_limb(
        GpuMatrix *out,
        const GpuMatrix *src,
        const std::vector<size_t> &src_indices,
        const std::vector<size_t> &dst_indices,
        const dim3 &limb_id,
        size_t n)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported limb type in launch_copy_for_limb");
        }
        const size_t count = src_indices.size();
        if (count == 0)
        {
            return 0;
        }
        if (dst_indices.size() != count)
        {
            return set_error("mismatched index counts in launch_copy_for_limb");
        }

        int src_device = -1;
        int status = matrix_limb_device(src, limb_id, &src_device);
        if (status != 0)
        {
            return status;
        }
        int dst_device = -1;
        status = matrix_limb_device(out, limb_id, &dst_device);
        if (status != 0)
        {
            return status;
        }
        if (src_device != dst_device)
        {
            return set_error("source/destination device mismatch in launch_copy_for_limb");
        }

        cudaStream_t exec_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &exec_stream);
        if (status != 0)
        {
            return status;
        }
        if (!exec_stream)
        {
            return set_error("null execution stream in launch_copy_for_limb");
        }

        cudaError_t err = cudaSetDevice(dst_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        std::vector<const T *> src_ptrs;
        std::vector<T *> dst_ptrs;
        src_ptrs.reserve(count);
        dst_ptrs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            const uint64_t *src_ptr = matrix_limb_ptr_by_id(src, src_indices[i], limb_id);
            uint64_t *dst_ptr = matrix_limb_ptr_by_id(out, dst_indices[i], limb_id);
            if (!src_ptr || !dst_ptr)
            {
                return set_error("null limb pointer in launch_copy_for_limb");
            }
            src_ptrs.push_back(reinterpret_cast<const T *>(src_ptr));
            dst_ptrs.push_back(reinterpret_cast<T *>(dst_ptr));
        }

        status = launch_copy_kernel(src_ptrs, dst_ptrs, n, exec_stream);
        if (status != 0)
        {
            return status;
        }

        return 0;
    }

    template <typename T>
    __global__ void block_equal_kernel(
        const T **lhs,
        const T **rhs,
        size_t poly_count,
        size_t n,
        int *out_equal)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx - poly_idx * n;
        if (lhs[poly_idx][coeff_idx] != rhs[poly_idx][coeff_idx])
        {
            atomicExch(out_equal, 0);
        }
    }

    template <typename T>
    int launch_block_equal_kernel(
        const std::vector<const T *> &lhs_ptrs,
        const std::vector<const T *> &rhs_ptrs,
        size_t n,
        cudaStream_t stream,
        bool &is_equal)
    {
        const size_t count = lhs_ptrs.size();
        if (count == 0 || n == 0)
        {
            is_equal = true;
            return 0;
        }
        if (rhs_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in block_equal_kernel");
        }

        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        int *d_equal = nullptr;
        const size_t ptr_bytes = count * sizeof(T *);
        cudaError_t err = cudaMalloc(&d_lhs, ptr_bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_rhs, ptr_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            return set_error(err);
        }
        err = cudaMalloc(&d_equal, sizeof(int));
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }

        int h_equal = 1;
        err = cudaMemcpyAsync(d_equal, &h_equal, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        block_equal_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, count, n, d_equal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }

        err = cudaMemcpyAsync(&h_equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            cudaFree(d_equal);
            return set_error(err);
        }

        cudaFree(d_lhs);
        cudaFree(d_rhs);
        cudaFree(d_equal);
        is_equal = (h_equal != 0);
        return 0;
    }

    int arith_wait_stream_on_stream(int device, cudaStream_t consumer, cudaStream_t producer)
    {
        if (!consumer || !producer || consumer == producer)
        {
            return 0;
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        cudaEvent_t ev = nullptr;
        err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaEventRecord(ev, producer);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(ev);
            return set_error(err);
        }
        err = cudaStreamWaitEvent(consumer, ev, 0);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(ev);
            return set_error(err);
        }
        err = cudaEventDestroy(ev);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int get_scalar_limb_u64(
        const GpuMatrix *scalar,
        const dim3 &limb_id,
        const uint64_t **out_ptr,
        int *out_device,
        cudaStream_t *out_stream)
    {
        if (!scalar || !out_ptr || !out_device || !out_stream)
        {
            return set_error("invalid scalar arguments in get_scalar_limb_u64");
        }
        if (scalar->rows != 1 || scalar->cols != 1)
        {
            return set_error("scalar matrix must be 1x1 in get_scalar_limb_u64");
        }
        const uint64_t *ptr = matrix_limb_ptr_by_id(scalar, 0, limb_id);
        if (!ptr)
        {
            return set_error("null scalar limb pointer in get_scalar_limb_u64");
        }
        int device = -1;
        int status = matrix_limb_device(scalar, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(scalar, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        *out_ptr = ptr;
        *out_device = device;
        *out_stream = stream;
        return 0;
    }

    template <typename T>
    int launch_matrix_elementwise_for_limb(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        size_t count,
        size_t n,
        int limb,
        const dim3 &limb_id,
        BlockOp op)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_elementwise_for_limb");
        }
        if (count == 0 || n == 0)
        {
            return 0;
        }

        int lhs_device = -1;
        int rhs_device = -1;
        int out_device = -1;
        int status = matrix_limb_device(lhs, limb_id, &lhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(rhs, limb_id, &rhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        if (lhs_device != rhs_device || lhs_device != out_device)
        {
            return set_error("device mismatch in launch_matrix_elementwise_for_limb");
        }

        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(count);
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint64_t *lhs_ptr = matrix_limb_ptr_by_id(lhs, idx, limb_id);
            const uint64_t *rhs_ptr = matrix_limb_ptr_by_id(rhs, idx, limb_id);
            uint64_t *out_ptr = matrix_limb_ptr_by_id(out, idx, limb_id);
            if (!lhs_ptr || !rhs_ptr || !out_ptr)
            {
                return set_error("null matrix limb pointer in launch_matrix_elementwise_for_limb");
            }
            lhs_ptrs.push_back(reinterpret_cast<const T *>(lhs_ptr));
            rhs_ptrs.push_back(reinterpret_cast<const T *>(rhs_ptr));
            out_ptrs.push_back(reinterpret_cast<T *>(out_ptr));
        }

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_elementwise_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, n, modulus, op, stream);
    }

    template <typename T>
    int launch_matrix_matmul_for_limb(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        int limb,
        const dim3 &limb_id,
        double *out_kernel_ms)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_matmul_for_limb");
        }
        const size_t out_count = rows * cols;
        const size_t lhs_count = rows * inner;
        const size_t rhs_count = inner * cols;
        if (out_count == 0 || n == 0)
        {
            return 0;
        }

        int lhs_device = -1;
        int rhs_device = -1;
        int out_device = -1;
        int status = matrix_limb_device(lhs, limb_id, &lhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(rhs, limb_id, &rhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        if (lhs_device != rhs_device || lhs_device != out_device)
        {
            return set_error("device mismatch in launch_matrix_matmul_for_limb");
        }

        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(out_count);
        lhs_ptrs.reserve(lhs_count);
        rhs_ptrs.reserve(rhs_count);

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t k = 0; k < inner; ++k)
            {
                const size_t idx = matrix_index(r, k, lhs->cols);
                const uint64_t *lhs_ptr = matrix_limb_ptr_by_id(lhs, idx, limb_id);
                if (!lhs_ptr)
                {
                    return set_error("null lhs pointer in launch_matrix_matmul_for_limb");
                }
                lhs_ptrs.push_back(reinterpret_cast<const T *>(lhs_ptr));
            }
        }
        for (size_t k = 0; k < inner; ++k)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                const size_t idx = matrix_index(k, c, rhs->cols);
                const uint64_t *rhs_ptr = matrix_limb_ptr_by_id(rhs, idx, limb_id);
                if (!rhs_ptr)
                {
                    return set_error("null rhs pointer in launch_matrix_matmul_for_limb");
                }
                rhs_ptrs.push_back(reinterpret_cast<const T *>(rhs_ptr));
            }
        }
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                const size_t idx = matrix_index(r, c, out->cols);
                uint64_t *out_ptr = matrix_limb_ptr_by_id(out, idx, limb_id);
                if (!out_ptr)
                {
                    return set_error("null out pointer in launch_matrix_matmul_for_limb");
                }
                out_ptrs.push_back(reinterpret_cast<T *>(out_ptr));
            }
        }

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_matmul_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_matmul_kernel(
            out_ptrs,
            lhs_ptrs,
            rhs_ptrs,
            rows,
            inner,
            cols,
            n,
            modulus,
            stream,
            out_kernel_ms);
    }

    template <typename T>
    int launch_matrix_scalar_mul_for_limb(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar,
        size_t count,
        size_t n,
        int limb,
        const dim3 &limb_id)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_scalar_mul_for_limb");
        }
        if (count == 0 || n == 0)
        {
            return 0;
        }

        int lhs_device = -1;
        int out_device = -1;
        int status = matrix_limb_device(lhs, limb_id, &lhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        if (lhs_device != out_device)
        {
            return set_error("device mismatch in launch_matrix_scalar_mul_for_limb");
        }

        const uint64_t *scalar_ptr = nullptr;
        int scalar_device = -1;
        cudaStream_t scalar_stream = nullptr;
        status = get_scalar_limb_u64(scalar, limb_id, &scalar_ptr, &scalar_device, &scalar_stream);
        if (status != 0)
        {
            return status;
        }
        if (scalar_device != out_device)
        {
            return set_error("scalar device mismatch in launch_matrix_scalar_mul_for_limb");
        }

        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = arith_wait_stream_on_stream(out_device, stream, scalar_stream);
        if (status != 0)
        {
            return status;
        }

        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(count);
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);
        const T *scalar_ptr_t = reinterpret_cast<const T *>(scalar_ptr);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint64_t *lhs_ptr = matrix_limb_ptr_by_id(lhs, idx, limb_id);
            uint64_t *out_ptr = matrix_limb_ptr_by_id(out, idx, limb_id);
            if (!lhs_ptr || !out_ptr)
            {
                return set_error("null matrix limb pointer in launch_matrix_scalar_mul_for_limb");
            }
            lhs_ptrs.push_back(reinterpret_cast<const T *>(lhs_ptr));
            rhs_ptrs.push_back(scalar_ptr_t);
            out_ptrs.push_back(reinterpret_cast<T *>(out_ptr));
        }

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_scalar_mul_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, n, modulus, BlockOp::Mul, stream);
    }

    template <typename T>
    int launch_matrix_equal_for_limb(
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        size_t count,
        size_t n,
        const dim3 &limb_id,
        bool &is_equal)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_equal_for_limb");
        }
        if (count == 0 || n == 0)
        {
            is_equal = true;
            return 0;
        }

        int lhs_device = -1;
        int rhs_device = -1;
        int status = matrix_limb_device(lhs, limb_id, &lhs_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_device(rhs, limb_id, &rhs_device);
        if (status != 0)
        {
            return status;
        }
        if (lhs_device != rhs_device)
        {
            return set_error("device mismatch in launch_matrix_equal_for_limb");
        }

        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(lhs, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        cudaError_t err = cudaSetDevice(lhs_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint64_t *lhs_ptr = matrix_limb_ptr_by_id(lhs, idx, limb_id);
            const uint64_t *rhs_ptr = matrix_limb_ptr_by_id(rhs, idx, limb_id);
            if (!lhs_ptr || !rhs_ptr)
            {
                return set_error("null matrix limb pointer in launch_matrix_equal_for_limb");
            }
            lhs_ptrs.push_back(reinterpret_cast<const T *>(lhs_ptr));
            rhs_ptrs.push_back(reinterpret_cast<const T *>(rhs_ptr));
        }

        return launch_block_equal_kernel(lhs_ptrs, rhs_ptrs, n, stream, is_equal);
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
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_add");
    }

    const size_t count = lhs->rows * lhs->cols;
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_add");
    }

    const int N = lhs->ctx->N;
    if (N <= 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_add");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_elementwise_for_limb<uint64_t>(
            out,
            lhs,
            rhs,
            count,
            static_cast<size_t>(N),
            limb,
            limb_id,
            BlockOp::Add);
        if (status != 0)
        {
            return status;
        }
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
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_sub");
    }

    const size_t count = lhs->rows * lhs->cols;
    if (count == 0)
    {
        out->format = lhs->format;
        return 0;
    }

    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sub");
    }

    const int N = lhs->ctx->N;
    if (N <= 0)
    {
        out->format = lhs->format;
        return 0;
    }

    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sub");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_elementwise_for_limb<uint64_t>(
            out,
            lhs,
            rhs,
            count,
            static_cast<size_t>(N),
            limb,
            limb_id,
            BlockOp::Sub);
        if (status != 0)
        {
            return status;
        }
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
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_mul");
    }
    if (lhs->format != PolyFormat::Eval || rhs->format != PolyFormat::Eval)
    {
        return set_error("gpu_matrix_mul requires Eval format");
    }

    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_mul");
    }
    const int N = lhs->ctx->N;
    if (N <= 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_mul");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_matmul_for_limb<uint64_t>(
            out,
            lhs,
            rhs,
            lhs->rows,
            lhs->cols,
            rhs->cols,
            static_cast<size_t>(N),
            limb,
            limb_id,
            nullptr);
        if (status != 0)
        {
            return status;
        }
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
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_equal");
    }
    if (lhs->format != rhs->format)
    {
        return 0;
    }
    const size_t count = lhs->rows * lhs->cols;
    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_equal");
    }
    const int N = lhs->ctx->N;
    if (N <= 0 || count == 0)
    {
        *out_equal = 1;
        return 0;
    }
    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_equal");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        bool limb_equal = false;
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_equal_for_limb<uint64_t>(
            lhs,
            rhs,
            count,
            static_cast<size_t>(N),
            limb_id,
            limb_equal);
        if (status != 0)
        {
            return status;
        }
        if (!limb_equal)
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
    *out_kernel_ms = 0.0;
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
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_mul_timed");
    }
    if (lhs->format != PolyFormat::Eval || rhs->format != PolyFormat::Eval)
    {
        return set_error("gpu_matrix_mul_timed requires Eval format");
    }

    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_mul_timed");
    }
    const int N = lhs->ctx->N;
    if (N <= 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_mul_timed");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_matmul_for_limb<uint64_t>(
            out,
            lhs,
            rhs,
            lhs->rows,
            lhs->cols,
            rhs->cols,
            static_cast<size_t>(N),
            limb,
            limb_id,
            out_kernel_ms);
        if (status != 0)
        {
            return status;
        }
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
    if (scalar->rows != 1 || scalar->cols != 1)
    {
        return set_error("gpu_matrix_mul_scalar requires 1x1 scalar matrix");
    }
    if (scalar->ctx != lhs->ctx || scalar->level != lhs->level)
    {
        return set_error("scalar context mismatch in gpu_matrix_mul_scalar");
    }
    if (!lhs->ctx || !lhs->ctx->ctx)
    {
        return set_error("null context in gpu_matrix_mul_scalar");
    }
    if (lhs->format != PolyFormat::Eval || scalar->format != PolyFormat::Eval)
    {
        return set_error("gpu_matrix_mul_scalar requires Eval format");
    }

    const size_t count = lhs->rows * lhs->cols;
    if (count == 0)
    {
        out->format = lhs->format;
        return 0;
    }
    const int level = lhs->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_mul_scalar");
    }
    const int N = lhs->ctx->N;
    if (N <= 0)
    {
        out->format = lhs->format;
        return 0;
    }

    auto &limb_map = lhs->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_mul_scalar");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int status = launch_matrix_scalar_mul_for_limb<uint64_t>(
            out,
            lhs,
            scalar,
            count,
            static_cast<size_t>(N),
            limb,
            limb_id);
        if (status != 0)
        {
            return status;
        }
    }

    out->format = lhs->format;
    return 0;
}
