namespace
{
    enum class BlockOp
    {
        Add,
        Sub,
        Mul,
    };

#ifndef GPU_MATMUL_TILE_M
#define GPU_MATMUL_TILE_M 8
#endif
#ifndef GPU_MATMUL_TILE_N
#define GPU_MATMUL_TILE_N 32
#endif
#ifndef GPU_MATMUL_TILE_K
#define GPU_MATMUL_TILE_K 16
#endif

    constexpr int kMatmulTileM = GPU_MATMUL_TILE_M;
    constexpr int kMatmulTileN = GPU_MATMUL_TILE_N;
    constexpr int kMatmulTileK = GPU_MATMUL_TILE_K;
    static_assert(kMatmulTileM > 0 && kMatmulTileN > 0 && kMatmulTileK > 0, "invalid matmul tile size");
    static_assert(kMatmulTileM * kMatmulTileN <= 1024, "matmul tile thread count exceeds CUDA limit");
    template <typename T>
    __global__ void block_add_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t poly_count,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
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
        T a = lhs_base[poly_idx * lhs_stride + coeff_idx];
        T b = rhs_base[poly_idx * rhs_stride + coeff_idx];
        T sum = a + b;
        sum = sum >= modulus ? (sum - modulus) : sum;
        out_base[poly_idx * out_stride + coeff_idx] = sum;
    }

    template <typename T>
    __global__ void block_sub_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t poly_count,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
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
        T a = lhs_base[poly_idx * lhs_stride + coeff_idx];
        T b = rhs_base[poly_idx * rhs_stride + coeff_idx];
        T diff = a >= b ? (a - b) : (modulus - (b - a));
        out_base[poly_idx * out_stride + coeff_idx] = diff;
    }

    template <typename T>
    __global__ void matrix_decompose_multi_kernel(
        const T *src_base,
        T *dst_base,
        size_t poly_count,
        size_t n,
        size_t src_stride,
        size_t dst_stride,
        size_t src_cols,
        size_t out_cols,
        size_t log_base_q,
        size_t digit_offset,
        uint32_t shift,
        T mask,
        T out_modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx - poly_idx * n;
        T residue = src_base[poly_idx * src_stride + coeff_idx];
        T digit = shift >= static_cast<uint32_t>(sizeof(T) * 8) ? 0 : ((residue >> shift) & mask);
        if (out_modulus != 0 && digit >= out_modulus)
        {
            digit %= out_modulus;
        }
        const size_t row = src_cols == 0 ? 0 : (poly_idx / src_cols);
        const size_t col = src_cols == 0 ? 0 : (poly_idx - row * src_cols);
        const size_t out_row = row * log_base_q + digit_offset;
        const size_t out_poly_idx = out_row * out_cols + col;
        dst_base[out_poly_idx * dst_stride + coeff_idx] = digit;
    }

    template <typename T>
    __global__ void block_mul_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t poly_count,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
        T modulus,
        int rhs_is_scalar)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        size_t rhs_poly_idx = rhs_is_scalar ? 0 : poly_idx;
        T a = lhs_base[poly_idx * lhs_stride + coeff_idx];
        T b = rhs_base[rhs_poly_idx * rhs_stride + coeff_idx];
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            out_base[poly_idx * out_stride + coeff_idx] = mul_mod_u64(a, b, modulus);
        }
        else
        {
            out_base[poly_idx * out_stride + coeff_idx] = mul_mod_u32(a, b, modulus);
        }
    }

    template <typename T>
    __global__ void block_copy_rect_kernel(
        const T *src_base,
        T *dst_base,
        size_t copy_rows,
        size_t copy_cols,
        size_t n,
        size_t src_stride,
        size_t dst_stride,
        size_t src_cols,
        size_t dst_cols,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_poly = copy_rows * copy_cols;
        const size_t total = total_poly * n;
        if (idx >= total)
        {
            return;
        }
        const size_t poly_offset = idx / n;
        const size_t coeff_idx = idx - poly_offset * n;
        const size_t local_row = poly_offset / copy_cols;
        const size_t local_col = poly_offset - local_row * copy_cols;
        const size_t src_poly_idx = (src_row + local_row) * src_cols + (src_col + local_col);
        const size_t dst_poly_idx = (dst_row + local_row) * dst_cols + (dst_col + local_col);
        dst_base[dst_poly_idx * dst_stride + coeff_idx] =
            src_base[src_poly_idx * src_stride + coeff_idx];
    }

    template <typename T>
    __global__ void block_matmul_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
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
                    const size_t lhs_poly_idx = lhs_row * inner + lhs_k;
                    val = lhs_base[lhs_poly_idx * lhs_stride + coeff_idx];
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
                    const size_t rhs_poly_idx = rhs_k * cols + rhs_col;
                    val = rhs_base[rhs_poly_idx * rhs_stride + coeff_idx];
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
            const size_t out_poly_idx = row * cols + col;
            out_base[out_poly_idx * out_stride + coeff_idx] = acc;
        }
    }

    size_t align_up_size(size_t value, size_t alignment)
    {
        if (alignment == 0)
        {
            return value;
        }
        return (value + alignment - 1) & ~(alignment - 1);
    }

    int acquire_matrix_aux_workspace(
        const GpuMatrix *aux_owner,
        const dim3 *aux_limb_id,
        size_t bytes,
        void **out_ptr,
        bool *out_shared,
        cudaStream_t stream)
    {
        if (!out_ptr || !out_shared)
        {
            return set_error("invalid acquire_matrix_aux_workspace arguments");
        }
        *out_ptr = nullptr;
        *out_shared = false;
        if (bytes == 0)
        {
            return 0;
        }
        if (aux_owner && aux_limb_id && matrix_aux_slice_for_limb(aux_owner, *aux_limb_id, bytes, out_ptr))
        {
            *out_shared = true;
            return 0;
        }
        if (!stream)
        {
            return set_error("null stream in acquire_matrix_aux_workspace");
        }
        cudaError_t err = cudaMallocAsync(out_ptr, bytes, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int release_matrix_aux_workspace(void *ptr, bool from_shared, cudaStream_t stream)
    {
        if (!ptr || from_shared)
        {
            return 0;
        }
        if (!stream)
        {
            return set_error("null stream in release_matrix_aux_workspace");
        }
        cudaError_t err = cudaFreeAsync(ptr, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_block_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t poly_count,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
        T modulus,
        BlockOp op,
        cudaStream_t stream,
        int rhs_is_scalar)
    {
        if (!lhs_base || !rhs_base || !out_base)
        {
            return set_error("null base pointer in launch_block_kernel");
        }
        if (poly_count == 0 || n == 0)
        {
            return 0;
        }

        const int threads = 256;
        const size_t total = poly_count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        switch (op)
        {
        case BlockOp::Add:
            block_add_kernel<<<blocks, threads, 0, stream>>>(
                lhs_base,
                rhs_base,
                out_base,
                poly_count,
                n,
                lhs_stride,
                rhs_stride,
                out_stride,
                modulus);
            break;
        case BlockOp::Sub:
            block_sub_kernel<<<blocks, threads, 0, stream>>>(
                lhs_base,
                rhs_base,
                out_base,
                poly_count,
                n,
                lhs_stride,
                rhs_stride,
                out_stride,
                modulus);
            break;
        case BlockOp::Mul:
            block_mul_kernel<<<blocks, threads, 0, stream>>>(
                lhs_base,
                rhs_base,
                out_base,
                poly_count,
                n,
                lhs_stride,
                rhs_stride,
                out_stride,
                modulus,
                rhs_is_scalar);
            break;
        }

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_block_matmul_kernel(
        const T *lhs_base,
        const T *rhs_base,
        T *out_base,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        size_t lhs_stride,
        size_t rhs_stride,
        size_t out_stride,
        T modulus,
        cudaStream_t stream,
        const GpuMatrix *,
        const dim3 *)
    {
        if (!lhs_base || !rhs_base || !out_base)
        {
            return set_error("null base pointer in launch_block_matmul_kernel");
        }
        if (rows == 0 || inner == 0 || cols == 0 || n == 0)
        {
            return 0;
        }

        const dim3 threads(kMatmulTileN, kMatmulTileM);
        const dim3 blocks(
            static_cast<unsigned int>((cols + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(n));

        block_matmul_kernel<<<blocks, threads, 0, stream>>>(
            lhs_base,
            rhs_base,
            out_base,
            rows,
            inner,
            cols,
            n,
            lhs_stride,
            rhs_stride,
            out_stride,
            modulus);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_decompose_multi_kernel(
        const T *src_base,
        T *dst_base,
        size_t poly_count,
        size_t n,
        size_t src_stride,
        size_t dst_stride,
        size_t src_cols,
        size_t out_cols,
        size_t log_base_q,
        size_t digit_offset,
        uint32_t shift,
        T mask,
        T out_modulus,
        cudaStream_t stream,
        const GpuMatrix *,
        const dim3 *)
    {
        if (!src_base || !dst_base)
        {
            return set_error("null base pointer in launch_decompose_multi_kernel");
        }
        if (poly_count == 0 || n == 0)
        {
            return 0;
        }
        if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
        {
            return set_error("invalid matrix shape in launch_decompose_multi_kernel");
        }

        const int threads = 256;
        const size_t total = poly_count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        cudaError_t err;
        matrix_decompose_multi_kernel<<<blocks, threads, 0, stream>>>(
            src_base,
            dst_base,
            poly_count,
            n,
            src_stride,
            dst_stride,
            src_cols,
            out_cols,
            log_base_q,
            digit_offset,
            shift,
            mask,
            out_modulus);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_copy_kernel(
        const T *src_base,
        T *dst_base,
        size_t n,
        size_t src_stride,
        size_t dst_stride,
        size_t src_cols,
        size_t dst_cols,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col,
        size_t copy_rows,
        size_t copy_cols,
        cudaStream_t stream)
    {
        if (!src_base || !dst_base)
        {
            return set_error("null base pointer in block_copy_rect_kernel");
        }
        if (copy_rows == 0 || copy_cols == 0 || n == 0)
        {
            return 0;
        }
        if (src_cols == 0 || dst_cols == 0)
        {
            return set_error("invalid matrix shape in block_copy_rect_kernel");
        }

        const int threads = 256;
        const size_t total = copy_rows * copy_cols * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        block_copy_rect_kernel<<<blocks, threads, 0, stream>>>(
            src_base,
            dst_base,
            copy_rows,
            copy_cols,
            n,
            src_stride,
            dst_stride,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    template <typename T>
    int launch_copy_for_limb(
        GpuMatrix *out,
        const GpuMatrix *src,
        const dim3 &limb_id,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col,
        size_t copy_rows,
        size_t copy_cols,
        size_t src_cols,
        size_t dst_cols,
        size_t n)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported limb type in launch_copy_for_limb");
        }
        if (copy_rows == 0 || copy_cols == 0)
        {
            return 0;
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
        status = matrix_wait_limb_stream(src, limb_id, dst_device, exec_stream);
        if (status != 0)
        {
            return status;
        }

        const T *src_base = reinterpret_cast<const T *>(matrix_limb_ptr_by_id(src, 0, limb_id));
        T *dst_base = reinterpret_cast<T *>(matrix_limb_ptr_by_id(out, 0, limb_id));
        if (!src_base || !dst_base)
        {
            return set_error("null limb base pointer in launch_copy_for_limb");
        }

        if (limb_id.x >= src->shared_limb_buffers.size() ||
            limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in launch_copy_for_limb");
        }
        const size_t src_stride = src->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t dst_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;

        status = launch_copy_kernel(
            src_base,
            dst_base,
            n,
            src_stride,
            dst_stride,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col,
            copy_rows,
            copy_cols,
            exec_stream);
        if (status != 0)
        {
            return status;
        }

        status = matrix_track_limb_consumer(src, limb_id, dst_device, exec_stream);
        if (status != 0)
        {
            return status;
        }
        return matrix_record_limb_write(out, limb_id, exec_stream);
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

    int get_scalar_limb_u64(
        const GpuMatrix *scalar,
        const dim3 &limb_id,
        const uint64_t **out_ptr,
        int *out_device)
    {
        if (!scalar || !out_ptr || !out_device)
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
        *out_ptr = ptr;
        *out_device = device;
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
        status = matrix_wait_limb_stream(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }

        const uint64_t *lhs_base_u64 = matrix_limb_ptr_by_id(lhs, 0, limb_id);
        const uint64_t *rhs_base_u64 = matrix_limb_ptr_by_id(rhs, 0, limb_id);
        uint64_t *out_base_u64 = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!lhs_base_u64 || !rhs_base_u64 || !out_base_u64)
        {
            return set_error("null matrix limb base pointer in launch_matrix_elementwise_for_limb");
        }
        if (limb_id.x >= lhs->shared_limb_buffers.size() ||
            limb_id.x >= rhs->shared_limb_buffers.size() ||
            limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in launch_matrix_elementwise_for_limb");
        }
        const size_t lhs_stride = lhs->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t rhs_stride = rhs->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t out_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_elementwise_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        status = launch_block_kernel(
            reinterpret_cast<const T *>(lhs_base_u64),
            reinterpret_cast<const T *>(rhs_base_u64),
            reinterpret_cast<T *>(out_base_u64),
            count,
            n,
            lhs_stride,
            rhs_stride,
            out_stride,
            modulus,
            op,
            stream,
            0);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        return matrix_record_limb_write(out, limb_id, stream);
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
        const dim3 &limb_id)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_matmul_for_limb");
        }
        if (rows == 0 || inner == 0 || cols == 0 || n == 0)
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
        status = matrix_wait_limb_stream(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }

        const uint64_t *lhs_base_u64 = matrix_limb_ptr_by_id(lhs, 0, limb_id);
        const uint64_t *rhs_base_u64 = matrix_limb_ptr_by_id(rhs, 0, limb_id);
        uint64_t *out_base_u64 = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!lhs_base_u64 || !rhs_base_u64 || !out_base_u64)
        {
            return set_error("null matrix limb base pointer in launch_matrix_matmul_for_limb");
        }
        if (limb_id.x >= lhs->shared_limb_buffers.size() ||
            limb_id.x >= rhs->shared_limb_buffers.size() ||
            limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in launch_matrix_matmul_for_limb");
        }
        const size_t lhs_stride = lhs->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t rhs_stride = rhs->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t out_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_matmul_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        status = launch_block_matmul_kernel(
            reinterpret_cast<const T *>(lhs_base_u64),
            reinterpret_cast<const T *>(rhs_base_u64),
            reinterpret_cast<T *>(out_base_u64),
            rows,
            inner,
            cols,
            n,
            lhs_stride,
            rhs_stride,
            out_stride,
            modulus,
            stream,
            out,
            &limb_id);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        return matrix_record_limb_write(out, limb_id, stream);
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
        status = get_scalar_limb_u64(scalar, limb_id, &scalar_ptr, &scalar_device);
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
        status = matrix_wait_limb_stream(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(scalar, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }

        const uint64_t *lhs_base_u64 = matrix_limb_ptr_by_id(lhs, 0, limb_id);
        uint64_t *out_base_u64 = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!lhs_base_u64 || !out_base_u64)
        {
            return set_error("null matrix limb base pointer in launch_matrix_scalar_mul_for_limb");
        }
        if (limb_id.x >= lhs->shared_limb_buffers.size() || limb_id.x >= out->shared_limb_buffers.size() ||
            limb_id.x >= scalar->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in launch_matrix_scalar_mul_for_limb");
        }
        const size_t lhs_stride = lhs->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t out_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;
        const size_t scalar_stride = scalar->shared_limb_buffers[limb_id.x].words_per_poly;

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_scalar_mul_for_limb");
        }
        const uint64_t modulus64 = lhs->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        status = launch_block_kernel(
            reinterpret_cast<const T *>(lhs_base_u64),
            reinterpret_cast<const T *>(scalar_ptr),
            reinterpret_cast<T *>(out_base_u64),
            count,
            n,
            lhs_stride,
            scalar_stride,
            out_stride,
            modulus,
            BlockOp::Mul,
            stream,
            1);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(lhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(scalar, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        return matrix_record_limb_write(out, limb_id, stream);
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
        status = matrix_wait_limb_stream(rhs, limb_id, lhs_device, stream);
        if (status != 0)
        {
            return status;
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
    int64_t **sampled_out_device,
    cudaEvent_t sampled_ready_event,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id);

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    const std::vector<uint64_t *> &out_entries,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id);

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
    if (!lhs->ctx)
    {
        return set_error("null context in gpu_matrix_add");
    }

    const size_t count = lhs->rows * lhs->cols;
    if (count == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
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
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    auto &limb_map = lhs->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_add");
    }

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        status = launch_matrix_elementwise_for_limb<uint64_t>(
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

    out->format = GPU_POLY_FORMAT_EVAL;
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
    if (!lhs->ctx)
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
    auto &limb_map = lhs->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sub");
    }

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        status = launch_matrix_elementwise_for_limb<uint64_t>(
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
    if (!lhs->ctx)
    {
        return set_error("null context in gpu_matrix_mul");
    }
    if (lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL)
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
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    auto &limb_map = lhs->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_mul");
    }

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        status = launch_matrix_matmul_for_limb<uint64_t>(
            out,
            lhs,
            rhs,
            lhs->rows,
            lhs->cols,
            rhs->cols,
            static_cast<size_t>(N),
            limb,
            limb_id);
        if (status != 0)
        {
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_EVAL;
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
    if (!lhs->ctx)
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
    auto &limb_map = lhs->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_equal");
    }

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        bool limb_equal = false;
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        status = launch_matrix_equal_for_limb<uint64_t>(
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
    if (!lhs->ctx)
    {
        return set_error("null context in gpu_matrix_mul_scalar");
    }
    if (lhs->format != GPU_POLY_FORMAT_EVAL || scalar->format != GPU_POLY_FORMAT_EVAL)
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
    auto &limb_map = lhs->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_mul_scalar");
    }

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        status = launch_matrix_scalar_mul_for_limb<uint64_t>(
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
