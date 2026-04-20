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
    constexpr size_t kMatmulMaxGridZ = 65535;
    static_assert(kMatmulTileM > 0 && kMatmulTileN > 0 && kMatmulTileK > 0, "invalid matmul tile size");
    static_assert(kMatmulTileM * kMatmulTileN <= 1024, "matmul tile thread count exceeds CUDA limit");
    __global__ void block_elementwise_all_limbs_kernel(
        const uint8_t *const *lhs_bases,
        const uint8_t *const *rhs_bases,
        uint8_t *const *out_bases,
        const size_t *lhs_stride_bytes,
        const size_t *rhs_stride_bytes,
        const size_t *out_stride_bytes,
        const uint8_t *lhs_coeff_bytes,
        const uint8_t *rhs_coeff_bytes,
        const uint8_t *out_coeff_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t poly_count,
        size_t n,
        int op,
        int rhs_is_scalar)
    {
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }

        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx - poly_idx * n;
        const size_t rhs_poly_idx = rhs_is_scalar ? 0 : poly_idx;
        const size_t lhs_stride = lhs_stride_bytes[limb_idx];
        const size_t rhs_stride = rhs_stride_bytes[limb_idx];
        const size_t out_stride = out_stride_bytes[limb_idx];
        const uint8_t *lhs_base = lhs_bases[limb_idx];
        const uint8_t *rhs_base = rhs_bases[limb_idx];
        uint8_t *out_base = out_bases[limb_idx];
        const uint8_t lhs_bytes = lhs_coeff_bytes[limb_idx];
        const uint8_t rhs_bytes = rhs_coeff_bytes[limb_idx];
        const uint8_t out_bytes = out_coeff_bytes[limb_idx];
        const uint64_t modulus = moduli[limb_idx];

        const uint64_t a = matrix_load_limb_u64(lhs_base, poly_idx, coeff_idx, lhs_stride, lhs_bytes);
        const uint64_t b = matrix_load_limb_u64(rhs_base, rhs_poly_idx, coeff_idx, rhs_stride, rhs_bytes);
        uint64_t result = 0;
        if (op == static_cast<int>(BlockOp::Add))
        {
            uint64_t sum = a + b;
            result = sum >= modulus ? (sum - modulus) : sum;
        }
        else if (op == static_cast<int>(BlockOp::Sub))
        {
            result = a >= b ? (a - b) : (modulus - (b - a));
        }
        else
        {
            result = mul_mod_u64(a, b, modulus);
        }
        matrix_store_limb_u64(out_base, poly_idx, coeff_idx, out_stride, out_bytes, result);
    }

    __global__ void block_copy_rect_all_limbs_kernel(
        const uint8_t *const *src_bases,
        uint8_t *const *dst_bases,
        const size_t *src_stride_bytes,
        const size_t *dst_stride_bytes,
        const uint8_t *src_coeff_bytes,
        const uint8_t *dst_coeff_bytes,
        size_t limb_count,
        size_t copy_rows,
        size_t copy_cols,
        size_t n,
        size_t src_cols,
        size_t dst_cols,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col)
    {
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
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

        const size_t src_stride = src_stride_bytes[limb_idx];
        const size_t dst_stride = dst_stride_bytes[limb_idx];
        const uint8_t src_bytes = src_coeff_bytes[limb_idx];
        const uint8_t dst_bytes = dst_coeff_bytes[limb_idx];
        const uint8_t *src_base = src_bases[limb_idx];
        uint8_t *dst_base = dst_bases[limb_idx];
        const uint64_t value =
            matrix_load_limb_u64(src_base, src_poly_idx, coeff_idx, src_stride, src_bytes);
        matrix_store_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes, value);
    }

    __global__ void block_add_rect_all_limbs_kernel(
        const uint8_t *const *src_bases,
        uint8_t *const *dst_bases,
        const size_t *src_stride_bytes,
        const size_t *dst_stride_bytes,
        const uint8_t *src_coeff_bytes,
        const uint8_t *dst_coeff_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t add_rows,
        size_t add_cols,
        size_t n,
        size_t src_cols,
        size_t dst_cols,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col)
    {
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_poly = add_rows * add_cols;
        const size_t total = total_poly * n;
        if (idx >= total)
        {
            return;
        }

        const size_t poly_offset = idx / n;
        const size_t coeff_idx = idx - poly_offset * n;
        const size_t local_row = poly_offset / add_cols;
        const size_t local_col = poly_offset - local_row * add_cols;
        const size_t src_poly_idx = (src_row + local_row) * src_cols + (src_col + local_col);
        const size_t dst_poly_idx = (dst_row + local_row) * dst_cols + (dst_col + local_col);

        const size_t src_stride = src_stride_bytes[limb_idx];
        const size_t dst_stride = dst_stride_bytes[limb_idx];
        const uint8_t src_bytes = src_coeff_bytes[limb_idx];
        const uint8_t dst_bytes = dst_coeff_bytes[limb_idx];
        const uint8_t *src_base = src_bases[limb_idx];
        uint8_t *dst_base = dst_bases[limb_idx];
        const uint64_t modulus = moduli[limb_idx];
        const uint64_t src_value =
            matrix_load_limb_u64(src_base, src_poly_idx, coeff_idx, src_stride, src_bytes);
        const uint64_t dst_value =
            matrix_load_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes);
        const uint64_t sum = add_mod_u64(dst_value, src_value, modulus);
        matrix_store_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes, sum);
    }

    __global__ void block_matmul_kernel(
        const uint8_t *lhs_base,
        const uint8_t *rhs_base,
        uint8_t *out_base,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        size_t lhs_stride_bytes,
        size_t rhs_stride_bytes,
        size_t out_stride_bytes,
        uint8_t lhs_coeff_bytes,
        uint8_t rhs_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        __shared__ uint64_t lhs_tile[kMatmulTileM][kMatmulTileK];
        __shared__ uint64_t rhs_tile[kMatmulTileK][kMatmulTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        for (size_t coeff_idx = static_cast<size_t>(blockIdx.z);
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z))
        {
            uint64_t acc = 0;
            for (size_t k0 = 0; k0 < inner; k0 += kMatmulTileK)
            {
                for (int i = tid; i < kMatmulTileM * kMatmulTileK; i += threads)
                {
                    const int r = i / kMatmulTileK;
                    const int k = i - r * kMatmulTileK;
                    const size_t lhs_row = row_base + static_cast<size_t>(r);
                    const size_t lhs_k = k0 + static_cast<size_t>(k);
                    uint64_t val = 0;
                    if (lhs_row < rows && lhs_k < inner)
                    {
                        const size_t lhs_poly_idx = lhs_row * inner + lhs_k;
                        val = matrix_load_limb_u64(
                            lhs_base,
                            lhs_poly_idx,
                            coeff_idx,
                            lhs_stride_bytes,
                            lhs_coeff_bytes);
                    }
                    lhs_tile[r][k] = val;
                }
                for (int i = tid; i < kMatmulTileK * kMatmulTileN; i += threads)
                {
                    const int k = i / kMatmulTileN;
                    const int c = i - k * kMatmulTileN;
                    const size_t rhs_k = k0 + static_cast<size_t>(k);
                    const size_t rhs_col = col_base + static_cast<size_t>(c);
                    uint64_t val = 0;
                    if (rhs_k < inner && rhs_col < cols)
                    {
                        const size_t rhs_poly_idx = rhs_k * cols + rhs_col;
                        val = matrix_load_limb_u64(
                            rhs_base,
                            rhs_poly_idx,
                            coeff_idx,
                            rhs_stride_bytes,
                            rhs_coeff_bytes);
                    }
                    rhs_tile[k][c] = val;
                }
                __syncthreads();

                if (row < rows && col < cols)
                {
                    for (int kk = 0; kk < kMatmulTileK; ++kk)
                    {
                        const uint64_t prod =
                            mul_mod_u64(lhs_tile[threadIdx.y][kk], rhs_tile[kk][threadIdx.x], modulus);
                        acc = add_mod_u64(acc, prod, modulus);
                    }
                }
                __syncthreads();
            }

            if (row < rows && col < cols)
            {
                const size_t out_poly_idx = row * cols + col;
                matrix_store_limb_u64(
                    out_base,
                    out_poly_idx,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes,
                    acc);
            }
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

    int launch_block_kernel_all_limbs(
        const uint8_t *const *lhs_bases,
        const uint8_t *const *rhs_bases,
        uint8_t *const *out_bases,
        const size_t *lhs_stride_bytes,
        const size_t *rhs_stride_bytes,
        const size_t *out_stride_bytes,
        const uint8_t *lhs_coeff_bytes,
        const uint8_t *rhs_coeff_bytes,
        const uint8_t *out_coeff_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t poly_count,
        size_t n,
        BlockOp op,
        cudaStream_t stream,
        int rhs_is_scalar)
    {
        if (!lhs_bases || !rhs_bases || !out_bases ||
            !lhs_stride_bytes || !rhs_stride_bytes || !out_stride_bytes ||
            !lhs_coeff_bytes || !rhs_coeff_bytes || !out_coeff_bytes || !moduli)
        {
            return set_error("null metadata pointer in launch_block_kernel_all_limbs");
        }
        if (limb_count == 0 || poly_count == 0 || n == 0)
        {
            return 0;
        }

        const int threads = 256;
        const size_t total = poly_count * n;
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_elementwise_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            lhs_bases,
            rhs_bases,
            out_bases,
            lhs_stride_bytes,
            rhs_stride_bytes,
            out_stride_bytes,
            lhs_coeff_bytes,
            rhs_coeff_bytes,
            out_coeff_bytes,
            moduli,
            limb_count,
            poly_count,
            n,
            static_cast<int>(op),
            rhs_is_scalar);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int launch_block_matmul_kernel(
        const uint8_t *lhs_base,
        const uint8_t *rhs_base,
        uint8_t *out_base,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        size_t lhs_stride_bytes,
        size_t rhs_stride_bytes,
        size_t out_stride_bytes,
        uint8_t lhs_coeff_bytes,
        uint8_t rhs_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus,
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
        const size_t grid_z = n < kMatmulMaxGridZ ? n : kMatmulMaxGridZ;
        const dim3 blocks(
            static_cast<unsigned int>((cols + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(grid_z));

        block_matmul_kernel<<<blocks, threads, 0, stream>>>(
            lhs_base,
            rhs_base,
            out_base,
            rows,
            inner,
            cols,
            n,
            lhs_stride_bytes,
            rhs_stride_bytes,
            out_stride_bytes,
            lhs_coeff_bytes,
            rhs_coeff_bytes,
            out_coeff_bytes,
            modulus);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int launch_copy_kernel_all_limbs(
        const uint8_t *const *src_bases,
        uint8_t *const *dst_bases,
        const size_t *src_stride_bytes,
        const size_t *dst_stride_bytes,
        const uint8_t *src_coeff_bytes,
        const uint8_t *dst_coeff_bytes,
        size_t limb_count,
        size_t n,
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
        if (!src_bases || !dst_bases || !src_stride_bytes || !dst_stride_bytes ||
            !src_coeff_bytes || !dst_coeff_bytes)
        {
            return set_error("null metadata pointer in launch_copy_kernel_all_limbs");
        }
        if (limb_count == 0 || copy_rows == 0 || copy_cols == 0 || n == 0)
        {
            return 0;
        }
        if (src_cols == 0 || dst_cols == 0)
        {
            return set_error("invalid matrix shape in launch_copy_kernel_all_limbs");
        }

        const int threads = 256;
        const size_t total = copy_rows * copy_cols * n;
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_copy_rect_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            src_bases,
            dst_bases,
            src_stride_bytes,
            dst_stride_bytes,
            src_coeff_bytes,
            dst_coeff_bytes,
            limb_count,
            copy_rows,
            copy_cols,
            n,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int launch_add_block_kernel_all_limbs(
        const uint8_t *const *src_bases,
        uint8_t *const *dst_bases,
        const size_t *src_stride_bytes,
        const size_t *dst_stride_bytes,
        const uint8_t *src_coeff_bytes,
        const uint8_t *dst_coeff_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t n,
        size_t src_cols,
        size_t dst_cols,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col,
        size_t add_rows,
        size_t add_cols,
        cudaStream_t stream)
    {
        if (!src_bases || !dst_bases || !src_stride_bytes || !dst_stride_bytes ||
            !src_coeff_bytes || !dst_coeff_bytes || !moduli)
        {
            return set_error("null metadata pointer in launch_add_block_kernel_all_limbs");
        }
        if (limb_count == 0 || add_rows == 0 || add_cols == 0 || n == 0)
        {
            return 0;
        }
        if (src_cols == 0 || dst_cols == 0)
        {
            return set_error("invalid matrix shape in launch_add_block_kernel_all_limbs");
        }

        const int threads = 256;
        const size_t total = add_rows * add_cols * n;
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_add_rect_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            src_bases,
            dst_bases,
            src_stride_bytes,
            dst_stride_bytes,
            src_coeff_bytes,
            dst_coeff_bytes,
            moduli,
            limb_count,
            add_rows,
            add_cols,
            n,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    __global__ void block_equal_kernel(
        const uint8_t *const *lhs,
        const uint8_t *const *rhs,
        uint8_t lhs_coeff_bytes,
        uint8_t rhs_coeff_bytes,
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
        const uint64_t lhs_value =
            matrix_load_packed_u64_at(lhs[poly_idx] + coeff_idx * static_cast<size_t>(lhs_coeff_bytes), lhs_coeff_bytes);
        const uint64_t rhs_value =
            matrix_load_packed_u64_at(rhs[poly_idx] + coeff_idx * static_cast<size_t>(rhs_coeff_bytes), rhs_coeff_bytes);
        if (lhs_value != rhs_value)
        {
            atomicExch(out_equal, 0);
        }
    }

    int launch_block_equal_kernel(
        const std::vector<const uint8_t *> &lhs_ptrs,
        const std::vector<const uint8_t *> &rhs_ptrs,
        uint8_t lhs_coeff_bytes,
        uint8_t rhs_coeff_bytes,
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

        const uint8_t **d_lhs = nullptr;
        const uint8_t **d_rhs = nullptr;
        int *d_equal = nullptr;
        auto release = [&]() {
            if (d_equal)
            {
                cudaFreeAsync(d_equal, stream);
                d_equal = nullptr;
            }
            if (d_rhs)
            {
                cudaFreeAsync(const_cast<uint8_t **>(d_rhs), stream);
                d_rhs = nullptr;
            }
            if (d_lhs)
            {
                cudaFreeAsync(const_cast<uint8_t **>(d_lhs), stream);
                d_lhs = nullptr;
            }
        };
        const size_t ptr_bytes = count * sizeof(uint8_t *);
        cudaError_t err =
            cudaMallocAsync(reinterpret_cast<void **>(&d_lhs), ptr_bytes, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&d_rhs), ptr_bytes, stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&d_equal), sizeof(int), stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }

        int h_equal = 1;
        err = cudaMemcpyAsync(d_equal, &h_equal, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        block_equal_kernel<<<blocks, threads, 0, stream>>>(
            d_lhs,
            d_rhs,
            lhs_coeff_bytes,
            rhs_coeff_bytes,
            count,
            n,
            d_equal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }

        err = cudaMemcpyAsync(&h_equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }

        release();
        is_equal = (h_equal != 0);
        return 0;
    }

    int get_scalar_limb_u64(
        const GpuMatrix *scalar,
        const dim3 &limb_id,
        const uint8_t **out_ptr,
        size_t *out_stride_bytes,
        uint8_t *out_coeff_bytes,
        int *out_device)
    {
        if (!scalar || !out_ptr || !out_stride_bytes || !out_coeff_bytes || !out_device)
        {
            return set_error("invalid scalar arguments in get_scalar_limb_u64");
        }
        if (scalar->rows != 1 || scalar->cols != 1)
        {
            return set_error("scalar matrix must be 1x1 in get_scalar_limb_u64");
        }
        const uint8_t *ptr = matrix_limb_ptr_by_id(scalar, 0, limb_id);
        if (!ptr)
        {
            return set_error("null scalar limb pointer in get_scalar_limb_u64");
        }
        size_t stride_bytes = 0;
        uint8_t coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(scalar, limb_id, &stride_bytes, &coeff_bytes))
        {
            return set_error("invalid scalar limb metadata in get_scalar_limb_u64");
        }
        int device = -1;
        int status = matrix_limb_device(scalar, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        *out_ptr = ptr;
        *out_stride_bytes = stride_bytes;
        *out_coeff_bytes = coeff_bytes;
        *out_device = device;
        return 0;
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

        const uint8_t *lhs_base = matrix_limb_ptr_by_id(lhs, 0, limb_id);
        const uint8_t *rhs_base = matrix_limb_ptr_by_id(rhs, 0, limb_id);
        uint8_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!lhs_base || !rhs_base || !out_base)
        {
            return set_error("null matrix limb base pointer in launch_matrix_matmul_for_limb");
        }
        size_t lhs_stride_bytes = 0;
        size_t rhs_stride_bytes = 0;
        size_t out_stride_bytes = 0;
        uint8_t lhs_coeff_bytes = 0;
        uint8_t rhs_coeff_bytes = 0;
        uint8_t out_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride_bytes, &lhs_coeff_bytes) ||
            !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride_bytes, &rhs_coeff_bytes) ||
            !matrix_limb_metadata_by_id(out, limb_id, &out_stride_bytes, &out_coeff_bytes))
        {
            return set_error("invalid matrix limb metadata in launch_matrix_matmul_for_limb");
        }
        if (lhs_coeff_bytes != rhs_coeff_bytes || lhs_coeff_bytes != out_coeff_bytes)
        {
            return set_error("inconsistent limb byte-width in launch_matrix_matmul_for_limb");
        }

        if (static_cast<size_t>(limb) >= lhs->ctx->moduli.size())
        {
            return set_error("unexpected modulus index in launch_matrix_matmul_for_limb");
        }
        const uint64_t modulus = lhs->ctx->moduli[static_cast<size_t>(limb)];
        status = launch_block_matmul_kernel(
            lhs_base,
            rhs_base,
            out_base,
            rows,
            inner,
            cols,
            n,
            lhs_stride_bytes,
            rhs_stride_bytes,
            out_stride_bytes,
            lhs_coeff_bytes,
            rhs_coeff_bytes,
            out_coeff_bytes,
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
    int launch_matrix_elementwise_all_limbs(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        size_t count,
        size_t n,
        int level,
        BlockOp op)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_elementwise_all_limbs");
        }
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (level < 0)
        {
            return set_error("invalid level in launch_matrix_elementwise_all_limbs");
        }
        if (!lhs || !rhs || !out || !lhs->ctx)
        {
            return set_error("invalid matrix arguments in launch_matrix_elementwise_all_limbs");
        }

        const size_t limb_count = static_cast<size_t>(level + 1);
        auto &limb_map = lhs->ctx->limb_gpu_ids;
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in launch_matrix_elementwise_all_limbs");
        }
        if (lhs->ctx->moduli.size() < limb_count)
        {
            return set_error("unexpected modulus count in launch_matrix_elementwise_all_limbs");
        }

        std::vector<dim3> active_limb_ids(limb_count);
        std::vector<const uint8_t *> lhs_bases(limb_count, nullptr);
        std::vector<const uint8_t *> rhs_bases(limb_count, nullptr);
        std::vector<uint8_t *> out_bases(limb_count, nullptr);
        std::vector<size_t> lhs_stride_bytes(limb_count, 0);
        std::vector<size_t> rhs_stride_bytes(limb_count, 0);
        std::vector<size_t> out_stride_bytes(limb_count, 0);
        std::vector<uint8_t> lhs_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> rhs_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> out_coeff_bytes(limb_count, 0);
        std::vector<uint64_t> moduli(limb_count, 0);

        int dispatch_device = -1;
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;
        for (int limb = 0; limb <= level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
            active_limb_ids[limb_idx] = limb_id;

            int lhs_device = -1;
            int rhs_device = -1;
            int out_device = -1;
            status = matrix_limb_device(lhs, limb_id, &lhs_device);
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
                return set_error("device mismatch in launch_matrix_elementwise_all_limbs");
            }

            if (limb == 0)
            {
                dispatch_device = out_device;
                status = matrix_limb_stream(out, limb_id, &dispatch_stream);
                if (status != 0)
                {
                    return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in launch_matrix_elementwise_all_limbs");
                }
            }
            else if (out_device != dispatch_device)
            {
                return set_error(
                    "single-device path requires all limbs on one device in launch_matrix_elementwise_all_limbs");
            }

            const uint8_t *lhs_base = matrix_limb_ptr_by_id(lhs, 0, limb_id);
            const uint8_t *rhs_base = matrix_limb_ptr_by_id(rhs, 0, limb_id);
            uint8_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
            if (!lhs_base || !rhs_base || !out_base)
            {
                return set_error("null matrix limb base pointer in launch_matrix_elementwise_all_limbs");
            }
            if (!matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride_bytes[limb_idx], &lhs_coeff_bytes[limb_idx]) ||
                !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride_bytes[limb_idx], &rhs_coeff_bytes[limb_idx]) ||
                !matrix_limb_metadata_by_id(out, limb_id, &out_stride_bytes[limb_idx], &out_coeff_bytes[limb_idx]))
            {
                return set_error("invalid matrix limb metadata in launch_matrix_elementwise_all_limbs");
            }
            if (lhs_coeff_bytes[limb_idx] != rhs_coeff_bytes[limb_idx] ||
                lhs_coeff_bytes[limb_idx] != out_coeff_bytes[limb_idx])
            {
                return set_error("inconsistent limb byte-width in launch_matrix_elementwise_all_limbs");
            }

            lhs_bases[limb_idx] = lhs_base;
            rhs_bases[limb_idx] = rhs_base;
            out_bases[limb_idx] = out_base;
            moduli[limb_idx] = lhs->ctx->moduli[limb_idx];
        }
        if (dispatch_device < 0 || !dispatch_stream)
        {
            return set_error("invalid dispatch metadata in launch_matrix_elementwise_all_limbs");
        }

        cudaError_t err = cudaSetDevice(dispatch_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_wait_limb_stream(lhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(rhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(out, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
        }

        const size_t ptr_bytes = limb_count * sizeof(uint8_t *);
        const size_t stride_bytes = limb_count * sizeof(size_t);
        const size_t coeff_bytes_bytes = limb_count * sizeof(uint8_t);
        const size_t modulus_bytes = limb_count * sizeof(uint64_t);
        const uint8_t **lhs_bases_device = nullptr;
        const uint8_t **rhs_bases_device = nullptr;
        uint8_t **out_bases_device = nullptr;
        size_t *lhs_stride_bytes_device = nullptr;
        size_t *rhs_stride_bytes_device = nullptr;
        size_t *out_stride_bytes_device = nullptr;
        uint8_t *lhs_coeff_bytes_device = nullptr;
        uint8_t *rhs_coeff_bytes_device = nullptr;
        uint8_t *out_coeff_bytes_device = nullptr;
        uint64_t *moduli_device = nullptr;
        auto cleanup = [&]()
        {
            if (dispatch_device >= 0)
            {
                cudaSetDevice(dispatch_device);
            }
            if (moduli_device)
            {
                cudaFreeAsync(moduli_device, dispatch_stream);
                moduli_device = nullptr;
            }
            if (out_coeff_bytes_device)
            {
                cudaFreeAsync(out_coeff_bytes_device, dispatch_stream);
                out_coeff_bytes_device = nullptr;
            }
            if (rhs_coeff_bytes_device)
            {
                cudaFreeAsync(rhs_coeff_bytes_device, dispatch_stream);
                rhs_coeff_bytes_device = nullptr;
            }
            if (lhs_coeff_bytes_device)
            {
                cudaFreeAsync(lhs_coeff_bytes_device, dispatch_stream);
                lhs_coeff_bytes_device = nullptr;
            }
            if (out_stride_bytes_device)
            {
                cudaFreeAsync(out_stride_bytes_device, dispatch_stream);
                out_stride_bytes_device = nullptr;
            }
            if (rhs_stride_bytes_device)
            {
                cudaFreeAsync(rhs_stride_bytes_device, dispatch_stream);
                rhs_stride_bytes_device = nullptr;
            }
            if (lhs_stride_bytes_device)
            {
                cudaFreeAsync(lhs_stride_bytes_device, dispatch_stream);
                lhs_stride_bytes_device = nullptr;
            }
            if (out_bases_device)
            {
                cudaFreeAsync(out_bases_device, dispatch_stream);
                out_bases_device = nullptr;
            }
            if (rhs_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(rhs_bases_device), dispatch_stream);
                rhs_bases_device = nullptr;
            }
            if (lhs_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(lhs_bases_device), dispatch_stream);
                lhs_bases_device = nullptr;
            }
        };

        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&rhs_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&rhs_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&rhs_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&moduli_device), modulus_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        err = cudaMemcpyAsync(
            lhs_bases_device,
            lhs_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            rhs_bases_device,
            rhs_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_bases_device,
            out_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            lhs_stride_bytes_device,
            lhs_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            rhs_stride_bytes_device,
            rhs_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_stride_bytes_device,
            out_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            lhs_coeff_bytes_device,
            lhs_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            rhs_coeff_bytes_device,
            rhs_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_coeff_bytes_device,
            out_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            moduli_device,
            moduli.data(),
            modulus_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        status = launch_block_kernel_all_limbs(
            lhs_bases_device,
            rhs_bases_device,
            out_bases_device,
            lhs_stride_bytes_device,
            rhs_stride_bytes_device,
            out_stride_bytes_device,
            lhs_coeff_bytes_device,
            rhs_coeff_bytes_device,
            out_coeff_bytes_device,
            moduli_device,
            limb_count,
            count,
            n,
            op,
            dispatch_stream,
            0);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_track_limb_consumer(lhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_track_limb_consumer(rhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_record_limb_write(out, limb_id, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        cleanup();
        return 0;
    }

    template <typename T>
    int launch_matrix_scalar_mul_all_limbs(
        GpuMatrix *out,
        const GpuMatrix *lhs,
        const GpuMatrix *scalar,
        size_t count,
        size_t n,
        int level)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported matrix limb type in launch_matrix_scalar_mul_all_limbs");
        }
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (level < 0)
        {
            return set_error("invalid level in launch_matrix_scalar_mul_all_limbs");
        }
        if (!lhs || !scalar || !out || !lhs->ctx)
        {
            return set_error("invalid matrix arguments in launch_matrix_scalar_mul_all_limbs");
        }

        const size_t limb_count = static_cast<size_t>(level + 1);
        auto &limb_map = lhs->ctx->limb_gpu_ids;
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in launch_matrix_scalar_mul_all_limbs");
        }
        if (lhs->ctx->moduli.size() < limb_count)
        {
            return set_error("unexpected modulus count in launch_matrix_scalar_mul_all_limbs");
        }

        std::vector<dim3> active_limb_ids(limb_count);
        std::vector<const uint8_t *> lhs_bases(limb_count, nullptr);
        std::vector<const uint8_t *> scalar_bases(limb_count, nullptr);
        std::vector<uint8_t *> out_bases(limb_count, nullptr);
        std::vector<size_t> lhs_stride_bytes(limb_count, 0);
        std::vector<size_t> scalar_stride_bytes(limb_count, 0);
        std::vector<size_t> out_stride_bytes(limb_count, 0);
        std::vector<uint8_t> lhs_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> scalar_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> out_coeff_bytes(limb_count, 0);
        std::vector<uint64_t> moduli(limb_count, 0);

        int dispatch_device = -1;
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;
        for (int limb = 0; limb <= level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
            active_limb_ids[limb_idx] = limb_id;

            int lhs_device = -1;
            int out_device = -1;
            status = matrix_limb_device(lhs, limb_id, &lhs_device);
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
                return set_error("device mismatch in launch_matrix_scalar_mul_all_limbs");
            }

            const uint8_t *scalar_ptr = nullptr;
            size_t scalar_stride = 0;
            uint8_t scalar_bytes = 0;
            int scalar_device = -1;
            status = get_scalar_limb_u64(
                scalar,
                limb_id,
                &scalar_ptr,
                &scalar_stride,
                &scalar_bytes,
                &scalar_device);
            if (status != 0)
            {
                return status;
            }
            if (scalar_device != out_device)
            {
                return set_error("scalar device mismatch in launch_matrix_scalar_mul_all_limbs");
            }

            if (limb == 0)
            {
                dispatch_device = out_device;
                status = matrix_limb_stream(out, limb_id, &dispatch_stream);
                if (status != 0)
                {
                    return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in launch_matrix_scalar_mul_all_limbs");
                }
            }
            else if (out_device != dispatch_device)
            {
                return set_error(
                    "single-device path requires all limbs on one device in launch_matrix_scalar_mul_all_limbs");
            }

            const uint8_t *lhs_base = matrix_limb_ptr_by_id(lhs, 0, limb_id);
            uint8_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
            if (!lhs_base || !out_base)
            {
                return set_error("null matrix limb base pointer in launch_matrix_scalar_mul_all_limbs");
            }
            if (!matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride_bytes[limb_idx], &lhs_coeff_bytes[limb_idx]) ||
                !matrix_limb_metadata_by_id(out, limb_id, &out_stride_bytes[limb_idx], &out_coeff_bytes[limb_idx]))
            {
                return set_error("invalid matrix limb metadata in launch_matrix_scalar_mul_all_limbs");
            }
            scalar_stride_bytes[limb_idx] = scalar_stride;
            scalar_coeff_bytes[limb_idx] = scalar_bytes;
            if (lhs_coeff_bytes[limb_idx] != scalar_coeff_bytes[limb_idx] ||
                lhs_coeff_bytes[limb_idx] != out_coeff_bytes[limb_idx])
            {
                return set_error("inconsistent limb byte-width in launch_matrix_scalar_mul_all_limbs");
            }

            lhs_bases[limb_idx] = lhs_base;
            scalar_bases[limb_idx] = scalar_ptr;
            out_bases[limb_idx] = out_base;
            moduli[limb_idx] = lhs->ctx->moduli[limb_idx];
        }
        if (dispatch_device < 0 || !dispatch_stream)
        {
            return set_error("invalid dispatch metadata in launch_matrix_scalar_mul_all_limbs");
        }

        cudaError_t err = cudaSetDevice(dispatch_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_wait_limb_stream(lhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(scalar, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(out, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
        }

        const size_t ptr_bytes = limb_count * sizeof(uint8_t *);
        const size_t stride_bytes = limb_count * sizeof(size_t);
        const size_t coeff_bytes_bytes = limb_count * sizeof(uint8_t);
        const size_t modulus_bytes = limb_count * sizeof(uint64_t);
        const uint8_t **lhs_bases_device = nullptr;
        const uint8_t **scalar_bases_device = nullptr;
        uint8_t **out_bases_device = nullptr;
        size_t *lhs_stride_bytes_device = nullptr;
        size_t *scalar_stride_bytes_device = nullptr;
        size_t *out_stride_bytes_device = nullptr;
        uint8_t *lhs_coeff_bytes_device = nullptr;
        uint8_t *scalar_coeff_bytes_device = nullptr;
        uint8_t *out_coeff_bytes_device = nullptr;
        uint64_t *moduli_device = nullptr;
        auto cleanup = [&]()
        {
            if (dispatch_device >= 0)
            {
                cudaSetDevice(dispatch_device);
            }
            if (moduli_device)
            {
                cudaFreeAsync(moduli_device, dispatch_stream);
                moduli_device = nullptr;
            }
            if (out_coeff_bytes_device)
            {
                cudaFreeAsync(out_coeff_bytes_device, dispatch_stream);
                out_coeff_bytes_device = nullptr;
            }
            if (scalar_coeff_bytes_device)
            {
                cudaFreeAsync(scalar_coeff_bytes_device, dispatch_stream);
                scalar_coeff_bytes_device = nullptr;
            }
            if (lhs_coeff_bytes_device)
            {
                cudaFreeAsync(lhs_coeff_bytes_device, dispatch_stream);
                lhs_coeff_bytes_device = nullptr;
            }
            if (out_stride_bytes_device)
            {
                cudaFreeAsync(out_stride_bytes_device, dispatch_stream);
                out_stride_bytes_device = nullptr;
            }
            if (scalar_stride_bytes_device)
            {
                cudaFreeAsync(scalar_stride_bytes_device, dispatch_stream);
                scalar_stride_bytes_device = nullptr;
            }
            if (lhs_stride_bytes_device)
            {
                cudaFreeAsync(lhs_stride_bytes_device, dispatch_stream);
                lhs_stride_bytes_device = nullptr;
            }
            if (out_bases_device)
            {
                cudaFreeAsync(out_bases_device, dispatch_stream);
                out_bases_device = nullptr;
            }
            if (scalar_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(scalar_bases_device), dispatch_stream);
                scalar_bases_device = nullptr;
            }
            if (lhs_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(lhs_bases_device), dispatch_stream);
                lhs_bases_device = nullptr;
            }
        };

        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&scalar_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&scalar_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&lhs_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&scalar_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&out_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&moduli_device), modulus_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        err = cudaMemcpyAsync(
            lhs_bases_device,
            lhs_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            scalar_bases_device,
            scalar_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_bases_device,
            out_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            lhs_stride_bytes_device,
            lhs_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            scalar_stride_bytes_device,
            scalar_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_stride_bytes_device,
            out_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            lhs_coeff_bytes_device,
            lhs_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            scalar_coeff_bytes_device,
            scalar_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            out_coeff_bytes_device,
            out_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            moduli_device,
            moduli.data(),
            modulus_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        status = launch_block_kernel_all_limbs(
            lhs_bases_device,
            scalar_bases_device,
            out_bases_device,
            lhs_stride_bytes_device,
            scalar_stride_bytes_device,
            out_stride_bytes_device,
            lhs_coeff_bytes_device,
            scalar_coeff_bytes_device,
            out_coeff_bytes_device,
            moduli_device,
            limb_count,
            count,
            n,
            BlockOp::Mul,
            dispatch_stream,
            1);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_track_limb_consumer(lhs, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_track_limb_consumer(scalar, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_record_limb_write(out, limb_id, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        cleanup();
        return 0;
    }

    template <typename T>
    int launch_copy_for_all_limbs(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col,
        size_t copy_rows,
        size_t copy_cols,
        size_t src_cols,
        size_t dst_cols,
        size_t n,
        int level)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported limb type in launch_copy_for_all_limbs");
        }
        if (copy_rows == 0 || copy_cols == 0 || n == 0)
        {
            return 0;
        }
        if (level < 0)
        {
            return set_error("invalid level in launch_copy_for_all_limbs");
        }
        if (!src || !out || !src->ctx)
        {
            return set_error("invalid matrix arguments in launch_copy_for_all_limbs");
        }

        const size_t limb_count = static_cast<size_t>(level + 1);
        auto &limb_map = src->ctx->limb_gpu_ids;
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in launch_copy_for_all_limbs");
        }

        std::vector<dim3> active_limb_ids(limb_count);
        std::vector<const uint8_t *> src_bases(limb_count, nullptr);
        std::vector<uint8_t *> dst_bases(limb_count, nullptr);
        std::vector<size_t> src_stride_bytes(limb_count, 0);
        std::vector<size_t> dst_stride_bytes(limb_count, 0);
        std::vector<uint8_t> src_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> dst_coeff_bytes(limb_count, 0);

        int dispatch_device = -1;
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;
        for (int limb = 0; limb <= level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
            active_limb_ids[limb_idx] = limb_id;

            int src_device = -1;
            int dst_device = -1;
            status = matrix_limb_device(src, limb_id, &src_device);
            if (status != 0)
            {
                return status;
            }
            status = matrix_limb_device(out, limb_id, &dst_device);
            if (status != 0)
            {
                return status;
            }
            if (src_device != dst_device)
            {
                return set_error("source/destination device mismatch in launch_copy_for_all_limbs");
            }

            if (limb == 0)
            {
                dispatch_device = dst_device;
                status = matrix_limb_stream(out, limb_id, &dispatch_stream);
                if (status != 0)
                {
                    return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in launch_copy_for_all_limbs");
                }
            }
            else if (dst_device != dispatch_device)
            {
                return set_error(
                    "single-device path requires all limbs on one device in launch_copy_for_all_limbs");
            }

            const uint8_t *src_base = matrix_limb_ptr_by_id(src, 0, limb_id);
            uint8_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
            if (!src_base || !dst_base)
            {
                return set_error("null limb base pointer in launch_copy_for_all_limbs");
            }
            if (!matrix_limb_metadata_by_id(src, limb_id, &src_stride_bytes[limb_idx], &src_coeff_bytes[limb_idx]) ||
                !matrix_limb_metadata_by_id(out, limb_id, &dst_stride_bytes[limb_idx], &dst_coeff_bytes[limb_idx]))
            {
                return set_error("invalid matrix limb metadata in launch_copy_for_all_limbs");
            }
            if (src_coeff_bytes[limb_idx] != dst_coeff_bytes[limb_idx])
            {
                return set_error("inconsistent limb byte-width in launch_copy_for_all_limbs");
            }

            src_bases[limb_idx] = src_base;
            dst_bases[limb_idx] = dst_base;
        }
        if (dispatch_device < 0 || !dispatch_stream)
        {
            return set_error("invalid dispatch metadata in launch_copy_for_all_limbs");
        }

        cudaError_t err = cudaSetDevice(dispatch_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_wait_limb_stream(src, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(out, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
        }

        const size_t ptr_bytes = limb_count * sizeof(uint8_t *);
        const size_t stride_bytes = limb_count * sizeof(size_t);
        const size_t coeff_bytes_bytes = limb_count * sizeof(uint8_t);
        const uint8_t **src_bases_device = nullptr;
        uint8_t **dst_bases_device = nullptr;
        size_t *src_stride_bytes_device = nullptr;
        size_t *dst_stride_bytes_device = nullptr;
        uint8_t *src_coeff_bytes_device = nullptr;
        uint8_t *dst_coeff_bytes_device = nullptr;
        auto cleanup = [&]()
        {
            if (dispatch_device >= 0)
            {
                cudaSetDevice(dispatch_device);
            }
            if (dst_coeff_bytes_device)
            {
                cudaFreeAsync(dst_coeff_bytes_device, dispatch_stream);
                dst_coeff_bytes_device = nullptr;
            }
            if (src_coeff_bytes_device)
            {
                cudaFreeAsync(src_coeff_bytes_device, dispatch_stream);
                src_coeff_bytes_device = nullptr;
            }
            if (dst_stride_bytes_device)
            {
                cudaFreeAsync(dst_stride_bytes_device, dispatch_stream);
                dst_stride_bytes_device = nullptr;
            }
            if (src_stride_bytes_device)
            {
                cudaFreeAsync(src_stride_bytes_device, dispatch_stream);
                src_stride_bytes_device = nullptr;
            }
            if (dst_bases_device)
            {
                cudaFreeAsync(dst_bases_device, dispatch_stream);
                dst_bases_device = nullptr;
            }
            if (src_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(src_bases_device), dispatch_stream);
                src_bases_device = nullptr;
            }
        };

        err = cudaMallocAsync(reinterpret_cast<void **>(&src_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&src_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&src_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        err = cudaMemcpyAsync(
            src_bases_device,
            src_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            dst_bases_device,
            dst_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            src_stride_bytes_device,
            src_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            dst_stride_bytes_device,
            dst_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            src_coeff_bytes_device,
            src_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            dst_coeff_bytes_device,
            dst_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        status = launch_copy_kernel_all_limbs(
            src_bases_device,
            dst_bases_device,
            src_stride_bytes_device,
            dst_stride_bytes_device,
            src_coeff_bytes_device,
            dst_coeff_bytes_device,
            limb_count,
            n,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col,
            copy_rows,
            copy_cols,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_track_limb_consumer(src, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_record_limb_write(out, limb_id, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        cleanup();
        return 0;
    }

    template <typename T>
    int launch_add_block_for_all_limbs(
        GpuMatrix *out,
        const GpuMatrix *src,
        size_t src_row,
        size_t src_col,
        size_t dst_row,
        size_t dst_col,
        size_t add_rows,
        size_t add_cols,
        size_t src_cols,
        size_t dst_cols,
        size_t n,
        int level)
    {
        if constexpr (!std::is_same_v<T, uint64_t>)
        {
            return set_error("unsupported limb type in launch_add_block_for_all_limbs");
        }
        if (add_rows == 0 || add_cols == 0 || n == 0)
        {
            return 0;
        }
        if (level < 0)
        {
            return set_error("invalid level in launch_add_block_for_all_limbs");
        }
        if (!src || !out || !src->ctx)
        {
            return set_error("invalid matrix arguments in launch_add_block_for_all_limbs");
        }

        const size_t limb_count = static_cast<size_t>(level + 1);
        auto &limb_map = src->ctx->limb_gpu_ids;
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in launch_add_block_for_all_limbs");
        }
        if (src->ctx->moduli.size() < limb_count)
        {
            return set_error("unexpected modulus count in launch_add_block_for_all_limbs");
        }

        std::vector<dim3> active_limb_ids(limb_count);
        std::vector<const uint8_t *> src_bases(limb_count, nullptr);
        std::vector<uint8_t *> dst_bases(limb_count, nullptr);
        std::vector<size_t> src_stride_bytes(limb_count, 0);
        std::vector<size_t> dst_stride_bytes(limb_count, 0);
        std::vector<uint8_t> src_coeff_bytes(limb_count, 0);
        std::vector<uint8_t> dst_coeff_bytes(limb_count, 0);
        std::vector<uint64_t> moduli(limb_count, 0);

        int dispatch_device = -1;
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;
        for (int limb = 0; limb <= level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
            active_limb_ids[limb_idx] = limb_id;

            int src_device = -1;
            int dst_device = -1;
            status = matrix_limb_device(src, limb_id, &src_device);
            if (status != 0)
            {
                return status;
            }
            status = matrix_limb_device(out, limb_id, &dst_device);
            if (status != 0)
            {
                return status;
            }
            if (src_device != dst_device)
            {
                return set_error("source/destination device mismatch in launch_add_block_for_all_limbs");
            }

            if (limb == 0)
            {
                dispatch_device = dst_device;
                status = matrix_limb_stream(out, limb_id, &dispatch_stream);
                if (status != 0)
                {
                    return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in launch_add_block_for_all_limbs");
                }
            }
            else if (dst_device != dispatch_device)
            {
                return set_error(
                    "single-device path requires all limbs on one device in launch_add_block_for_all_limbs");
            }

            const uint8_t *src_base = matrix_limb_ptr_by_id(src, 0, limb_id);
            uint8_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
            if (!src_base || !dst_base)
            {
                return set_error("null limb base pointer in launch_add_block_for_all_limbs");
            }
            if (!matrix_limb_metadata_by_id(src, limb_id, &src_stride_bytes[limb_idx], &src_coeff_bytes[limb_idx]) ||
                !matrix_limb_metadata_by_id(out, limb_id, &dst_stride_bytes[limb_idx], &dst_coeff_bytes[limb_idx]))
            {
                return set_error("invalid matrix limb metadata in launch_add_block_for_all_limbs");
            }
            if (src_coeff_bytes[limb_idx] != dst_coeff_bytes[limb_idx])
            {
                return set_error("inconsistent limb byte-width in launch_add_block_for_all_limbs");
            }

            src_bases[limb_idx] = src_base;
            dst_bases[limb_idx] = dst_base;
            moduli[limb_idx] = src->ctx->moduli[limb_idx];
        }
        if (dispatch_device < 0 || !dispatch_stream)
        {
            return set_error("invalid dispatch metadata in launch_add_block_for_all_limbs");
        }

        cudaError_t err = cudaSetDevice(dispatch_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_wait_limb_stream(src, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
            status = matrix_wait_limb_stream(out, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
        }

        const size_t ptr_bytes = limb_count * sizeof(uint8_t *);
        const size_t stride_bytes = limb_count * sizeof(size_t);
        const size_t coeff_bytes_bytes = limb_count * sizeof(uint8_t);
        const size_t modulus_bytes = limb_count * sizeof(uint64_t);
        const uint8_t **src_bases_device = nullptr;
        uint8_t **dst_bases_device = nullptr;
        size_t *src_stride_bytes_device = nullptr;
        size_t *dst_stride_bytes_device = nullptr;
        uint8_t *src_coeff_bytes_device = nullptr;
        uint8_t *dst_coeff_bytes_device = nullptr;
        uint64_t *moduli_device = nullptr;
        auto cleanup = [&]()
        {
            if (dispatch_device >= 0)
            {
                cudaSetDevice(dispatch_device);
            }
            if (moduli_device)
            {
                cudaFreeAsync(moduli_device, dispatch_stream);
                moduli_device = nullptr;
            }
            if (dst_coeff_bytes_device)
            {
                cudaFreeAsync(dst_coeff_bytes_device, dispatch_stream);
                dst_coeff_bytes_device = nullptr;
            }
            if (src_coeff_bytes_device)
            {
                cudaFreeAsync(src_coeff_bytes_device, dispatch_stream);
                src_coeff_bytes_device = nullptr;
            }
            if (dst_stride_bytes_device)
            {
                cudaFreeAsync(dst_stride_bytes_device, dispatch_stream);
                dst_stride_bytes_device = nullptr;
            }
            if (src_stride_bytes_device)
            {
                cudaFreeAsync(src_stride_bytes_device, dispatch_stream);
                src_stride_bytes_device = nullptr;
            }
            if (dst_bases_device)
            {
                cudaFreeAsync(dst_bases_device, dispatch_stream);
                dst_bases_device = nullptr;
            }
            if (src_bases_device)
            {
                cudaFreeAsync(const_cast<uint8_t **>(src_bases_device), dispatch_stream);
                src_bases_device = nullptr;
            }
        };

        err = cudaMallocAsync(reinterpret_cast<void **>(&src_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&src_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_stride_bytes_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&src_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&dst_coeff_bytes_device), coeff_bytes_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&moduli_device), modulus_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        err = cudaMemcpyAsync(src_bases_device, src_bases.data(), ptr_bytes, cudaMemcpyHostToDevice, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(dst_bases_device, dst_bases.data(), ptr_bytes, cudaMemcpyHostToDevice, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            src_stride_bytes_device,
            src_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            dst_stride_bytes_device,
            dst_stride_bytes.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            src_coeff_bytes_device,
            src_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            dst_coeff_bytes_device,
            dst_coeff_bytes.data(),
            coeff_bytes_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(moduli_device, moduli.data(), modulus_bytes, cudaMemcpyHostToDevice, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        status = launch_add_block_kernel_all_limbs(
            src_bases_device,
            dst_bases_device,
            src_stride_bytes_device,
            dst_stride_bytes_device,
            src_coeff_bytes_device,
            dst_coeff_bytes_device,
            moduli_device,
            limb_count,
            n,
            src_cols,
            dst_cols,
            src_row,
            src_col,
            dst_row,
            dst_col,
            add_rows,
            add_cols,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
        {
            const dim3 limb_id = active_limb_ids[limb_idx];
            status = matrix_track_limb_consumer(src, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = matrix_record_limb_write(out, limb_id, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        cleanup();
        return 0;
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

        size_t lhs_stride_bytes = 0;
        size_t rhs_stride_bytes = 0;
        uint8_t lhs_coeff_bytes = 0;
        uint8_t rhs_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride_bytes, &lhs_coeff_bytes) ||
            !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride_bytes, &rhs_coeff_bytes))
        {
            return set_error("invalid matrix limb metadata in launch_matrix_equal_for_limb");
        }
        if (lhs_coeff_bytes != rhs_coeff_bytes)
        {
            return set_error("inconsistent limb byte-width in launch_matrix_equal_for_limb");
        }

        std::vector<const uint8_t *> lhs_ptrs;
        std::vector<const uint8_t *> rhs_ptrs;
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint8_t *lhs_ptr = matrix_limb_ptr_by_id(lhs, idx, limb_id);
            const uint8_t *rhs_ptr = matrix_limb_ptr_by_id(rhs, idx, limb_id);
            if (!lhs_ptr || !rhs_ptr)
            {
                return set_error("null matrix limb pointer in launch_matrix_equal_for_limb");
            }
            lhs_ptrs.push_back(lhs_ptr);
            rhs_ptrs.push_back(rhs_ptr);
        }

        return launch_block_equal_kernel(lhs_ptrs, rhs_ptrs, lhs_coeff_bytes, rhs_coeff_bytes, n, stream, is_equal);
    }

} // namespace

int launch_sample_p1_integer_kernel(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    const uint8_t *tp2_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    size_t tp2_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    uint8_t tp2_coeff_bytes,
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
    cudaEvent_t sampled_ready_event);

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    uint8_t *out_base,
    size_t out_stride_bytes,
    uint8_t out_coeff_bytes,
    size_t entry_count,
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

    int status = launch_matrix_elementwise_all_limbs<uint64_t>(
        out,
        lhs,
        rhs,
        count,
        static_cast<size_t>(N),
        level,
        BlockOp::Add);
    if (status != 0)
    {
        return status;
    }

    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_add_block(
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
        return set_error("invalid gpu_matrix_add_block arguments");
    }
    if (src_row + rows > src->rows || src_col + cols > src->cols)
    {
        return set_error("source bounds exceeded in gpu_matrix_add_block");
    }
    if (dst_row + rows > out->rows || dst_col + cols > out->cols)
    {
        return set_error("dest bounds exceeded in gpu_matrix_add_block");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_add_block");
    }
    if (!src->ctx)
    {
        return set_error("null context in gpu_matrix_add_block");
    }

    if (rows == 0 || cols == 0)
    {
        out->format = src->format;
        return 0;
    }

    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_add_block");
    }
    const int N = src->ctx->N;
    if (N <= 0)
    {
        out->format = src->format;
        return 0;
    }
    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_add_block");
    }

    int status = launch_add_block_for_all_limbs<uint64_t>(
        out,
        src,
        src_row,
        src_col,
        dst_row,
        dst_col,
        rows,
        cols,
        src->cols,
        out->cols,
        static_cast<size_t>(N),
        level);
    if (status != 0)
    {
        return status;
    }

    out->format = src->format;
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

    int status = launch_matrix_elementwise_all_limbs<uint64_t>(
        out,
        lhs,
        rhs,
        count,
        static_cast<size_t>(N),
        level,
        BlockOp::Sub);
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

    int status = launch_matrix_scalar_mul_all_limbs<uint64_t>(
        out,
        lhs,
        scalar,
        count,
        static_cast<size_t>(N),
        level);
    if (status != 0)
    {
        return status;
    }

    out->format = lhs->format;
    return 0;
}
