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
    constexpr int kThinMatmulWarpSize = 32;
    constexpr int kThinMatmulColumnsPerBlock = 4;
    constexpr size_t kMatmulMaxGridZ = 65535;
    static_assert(kMatmulTileM > 0 && kMatmulTileN > 0 && kMatmulTileK > 0, "invalid matmul tile size");
    static_assert(kMatmulTileM * kMatmulTileN <= 1024, "matmul tile thread count exceeds CUDA limit");
    // CUDA copies value arguments into launch-owned parameter storage before
    // returning. Small metadata therefore needs neither pinned staging nor a
    // device allocation/reclaimer job. Context construction enforces the limb limit.
    constexpr size_t kArithMetadataLimbs = GPU_RUNTIME_MAX_LIMBS;
    struct BlockElementwiseMetadata
    {
        const uint8_t * lhs_bases[kArithMetadataLimbs];
        const uint8_t * rhs_bases[kArithMetadataLimbs];
        uint8_t * out_bases[kArithMetadataLimbs];
        size_t lhs_stride_bytes[kArithMetadataLimbs];
        size_t rhs_stride_bytes[kArithMetadataLimbs];
        size_t out_stride_bytes[kArithMetadataLimbs];
        uint8_t lhs_coeff_bytes[kArithMetadataLimbs];
        uint8_t rhs_coeff_bytes[kArithMetadataLimbs];
        uint8_t out_coeff_bytes[kArithMetadataLimbs];
        uint64_t moduli[kArithMetadataLimbs];
    };
    static_assert(sizeof(BlockElementwiseMetadata) + 3 * sizeof(size_t) + 2 * sizeof(int) <= 4096,
                  "arithmetic launch exceeds portable CUDA parameter budget");
    struct BlockCopyMetadata
    {
        const uint8_t * src_bases[kArithMetadataLimbs];
        uint8_t * dst_bases[kArithMetadataLimbs];
        size_t src_stride_bytes[kArithMetadataLimbs];
        size_t dst_stride_bytes[kArithMetadataLimbs];
        uint8_t src_coeff_bytes[kArithMetadataLimbs];
        uint8_t dst_coeff_bytes[kArithMetadataLimbs];
    };
    static_assert(sizeof(BlockCopyMetadata) + 10 * sizeof(size_t) + 0 * sizeof(int) <= 4096,
                  "arithmetic launch exceeds portable CUDA parameter budget");
    struct BlockAddMetadata
    {
        const uint8_t * src_bases[kArithMetadataLimbs];
        uint8_t * dst_bases[kArithMetadataLimbs];
        size_t src_stride_bytes[kArithMetadataLimbs];
        size_t dst_stride_bytes[kArithMetadataLimbs];
        uint8_t src_coeff_bytes[kArithMetadataLimbs];
        uint8_t dst_coeff_bytes[kArithMetadataLimbs];
        uint64_t moduli[kArithMetadataLimbs];
    };
    static_assert(sizeof(BlockAddMetadata) + 10 * sizeof(size_t) + 0 * sizeof(int) <= 4096,
                  "arithmetic launch exceeds portable CUDA parameter budget");

    // Small output matrices are coefficient-parallel dots, not spatial GEMMs.
    // Device-owned descriptors preserve compact limb widths and polynomial strides.
    struct DescriptorProductMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *lhs;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *rhs;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out;
        size_t indices[kArithMetadataLimbs];
        uint64_t moduli[kArithMetadataLimbs];
    };
    static_assert(sizeof(DescriptorProductMetadata) + 5 * sizeof(size_t) <= 4096,
                  "descriptor product launch exceeds portable CUDA parameter budget");

    struct TransposeMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out;
        size_t indices[kArithMetadataLimbs];
    };
    static_assert(sizeof(TransposeMetadata) + 4 * sizeof(size_t) <= 4096,
                  "transpose exceeds portable CUDA parameter budget");

    __global__ void transpose_all_limbs_kernel(
        TransposeMetadata metadata, size_t source_rows, size_t source_cols, size_t count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const auto source = metadata.source[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t coefficient = index % n;
            const size_t source_poly = (poly % source_rows) * source_cols + poly / source_rows;
            const uint64_t value = matrix_load_limb_u64(
                source.base, source_poly, coefficient, source.stride, source.width);
            matrix_store_limb_u64(out.base, poly, coefficient, out.stride, out.width, value);
        }
    }

    struct RowSumMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out;
        size_t indices[kArithMetadataLimbs];
        uint64_t moduli[kArithMetadataLimbs];
        size_t rows[32];
        size_t offsets[17];
    };
    static_assert(sizeof(RowSumMetadata) + 3 * sizeof(size_t) <= 4096,
                  "row sum exceeds portable CUDA parameter budget");

    __global__ void sum_rows_all_limbs_kernel(RowSumMetadata metadata, size_t cols, size_t count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const auto source = metadata.source[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t row = poly / cols;
            const size_t col = poly % cols;
            const size_t coefficient = index % n;
            uint64_t sum = 0;
            for (size_t term = metadata.offsets[row]; term < metadata.offsets[row + 1]; ++term)
                sum = add_mod_u64(sum, matrix_load_limb_u64(source.base,
                    metadata.rows[term] * cols + col, coefficient, source.stride, source.width),
                    metadata.moduli[limb]);
            matrix_store_limb_u64(out.base, poly, coefficient, out.stride, out.width, sum);
        }
    }

    constexpr size_t kRowAddMaxBlocks = 16;
    struct RowBlockAddMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *blocks[kRowAddMaxBlocks];
        size_t rows[kRowAddMaxBlocks];
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *rhs;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out;
        size_t indices[kArithMetadataLimbs];
        uint64_t moduli[kArithMetadataLimbs];
    };
    static_assert(sizeof(RowBlockAddMetadata) + 3 * sizeof(size_t) <= 4096,
                  "row block add exceeds portable CUDA parameter budget");

    __global__ void add_row_blocks_all_limbs_kernel(
        RowBlockAddMetadata metadata, size_t cols, size_t count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const size_t descriptor = metadata.indices[limb];
        const auto rhs = metadata.rhs[descriptor];
        const auto out = metadata.out[descriptor];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t coefficient = index % n;
            size_t row = poly / cols;
            size_t block = 0;
            while (row >= metadata.rows[block]) row -= metadata.rows[block++];
            const auto lhs = metadata.blocks[block][descriptor];
            const uint64_t a = matrix_load_limb_u64(lhs.base, row * cols + poly % cols,
                                                  coefficient, lhs.stride, lhs.width);
            const uint64_t b = matrix_load_limb_u64(rhs.base, poly, coefficient, rhs.stride, rhs.width);
            matrix_store_limb_u64(out.base, poly, coefficient, out.stride, out.width,
                                 add_mod_u64(a, b, metadata.moduli[limb]));
        }
    }

    __global__ void small_dot_all_limbs_kernel(
        DescriptorProductMetadata metadata, size_t rows, size_t inner, size_t cols, size_t n)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= n) return;
        const size_t limb = blockIdx.z;
        const auto lhs = metadata.lhs[metadata.indices[limb]];
        const auto rhs = metadata.rhs[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        const uint64_t modulus = metadata.moduli[limb];
        const size_t column = blockIdx.y;
        for (size_t row = 0; row < rows; ++row)
        {
            unsigned __int128 sum = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                const uint64_t a = matrix_load_limb_u64(lhs.base, row * inner + k,
                    coefficient, lhs.stride, lhs.width);
                const uint64_t b = matrix_load_limb_u64(rhs.base, k * cols + column,
                    coefficient, rhs.stride, rhs.width);
                unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
                // Usually the entire short dot fits in 128 bits (including the
                // 36/54-bit CRT bases). Preserve exactness for arbitrary u64 inputs.
                if (~static_cast<unsigned __int128>(0) - sum < product)
                {
                    sum %= modulus;
                    product %= modulus;
                }
                sum += product;
            }
            matrix_store_limb_u64(out.base, row * cols + column, coefficient,
                out.stride, out.width, static_cast<uint64_t>(sum % modulus));
        }
    }

    __global__ void tensor_all_limbs_kernel(
        DescriptorProductMetadata metadata, size_t lhs_cols, size_t rhs_rows,
        size_t rhs_cols, size_t output_count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const auto lhs = metadata.lhs[metadata.indices[limb]];
        const auto rhs = metadata.rhs[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        const size_t output_cols = lhs_cols * rhs_cols;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < output_count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t coefficient = index % n;
            const size_t row = poly / output_cols;
            const size_t col = poly % output_cols;
            const size_t left_poly = (row / rhs_rows) * lhs_cols + col / rhs_cols;
            const size_t right_poly = (row % rhs_rows) * rhs_cols + col % rhs_cols;
            const uint64_t a = matrix_load_limb_u64(lhs.base, left_poly, coefficient,
                                                  lhs.stride, lhs.width);
            const uint64_t b = matrix_load_limb_u64(rhs.base, right_poly, coefficient,
                                                  rhs.stride, rhs.width);
            matrix_store_limb_u64(out.base, poly, coefficient, out.stride, out.width,
                                 mul_mod_u64(a, b, metadata.moduli[limb]));
        }
    }

    int launch_descriptor_product(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, bool tensor)
    {
        const size_t limb_count = static_cast<size_t>(lhs->level) + 1;
        if (limb_count > kArithMetadataLimbs || lhs->ctx->moduli.size() < limb_count)
            return set_error("invalid descriptor product modulus count");
        DescriptorProductMetadata metadata{};
        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
        if (status != 0) return status;
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = lhs->ctx->limb_gpu_ids[limb];
            if (id.x >= lhs->shared_limb_buffers.size() ||
                id.x >= rhs->shared_limb_buffers.size() || id.x >= out->shared_limb_buffers.size())
                return set_error("invalid descriptor product limb partition");
            const auto &left = lhs->shared_limb_buffers[id.x];
            const auto &right = rhs->shared_limb_buffers[id.x];
            const auto &output = out->shared_limb_buffers[id.x];
            if (!left.device_descriptors || !right.device_descriptors || !output.device_descriptors ||
                id.y >= left.limb_count || id.y >= right.limb_count || id.y >= output.limb_count ||
                left.device != right.device || left.device != output.device)
                return set_error("invalid descriptor product descriptors");
            if (limb == 0)
            {
                device = output.device;
                metadata.lhs = left.device_descriptors;
                metadata.rhs = right.device_descriptors;
                metadata.out = output.device_descriptors;
                const cudaError_t error = cudaSetDevice(device);
                if (error != cudaSuccess) return set_error(error);
            }
            else if (device != output.device || metadata.lhs != left.device_descriptors ||
                     metadata.rhs != right.device_descriptors || metadata.out != output.device_descriptors)
                return set_error("product descriptors span multiple partitions");
            metadata.indices[limb] = id.y;
            metadata.moduli[limb] = lhs->ctx->moduli[limb];
        }
        status = matrix_wait_all_limb_streams(lhs, device, stream);
        if (status != 0) return status;
        status = matrix_wait_all_limb_streams(rhs, device, stream);
        if (status != 0) return status;
        status = matrix_wait_all_limb_streams(out, device, stream);
        if (status != 0) return status;
        const size_t n = static_cast<size_t>(lhs->ctx->N);
        if (tensor)
        {
            const size_t count = out->rows * out->cols * n;
            const size_t blocks = count / 256 + (count % 256 != 0);
            const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                            1, static_cast<unsigned int>(limb_count));
            tensor_all_limbs_kernel<<<grid, 256, 0, stream>>>(
                metadata, lhs->cols, rhs->rows, rhs->cols, count, n);
        }
        else
        {
            const dim3 grid(static_cast<unsigned int>((n + 255) / 256),
                            static_cast<unsigned int>(rhs->cols), static_cast<unsigned int>(limb_count));
            small_dot_all_limbs_kernel<<<grid, 256, 0, stream>>>(
                metadata, lhs->rows, lhs->cols, rhs->cols, n);
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        status = matrix_track_all_limb_consumers(lhs, device, stream);
        if (status != 0) return status;
        status = matrix_track_all_limb_consumers(rhs, device, stream);
        if (status != 0) return status;
        status = matrix_record_all_limb_writes(out, stream);
        if (status != 0) return status;
        return 0;
    }

    __global__ void block_elementwise_all_limbs_kernel(
        const BlockElementwiseMetadata metadata,
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
        const size_t lhs_stride = metadata.lhs_stride_bytes[limb_idx];
        const size_t rhs_stride = metadata.rhs_stride_bytes[limb_idx];
        const size_t out_stride = metadata.out_stride_bytes[limb_idx];
        const uint8_t *lhs_base = metadata.lhs_bases[limb_idx];
        const uint8_t *rhs_base = metadata.rhs_bases[limb_idx];
        uint8_t *out_base = metadata.out_bases[limb_idx];
        const uint8_t lhs_bytes = metadata.lhs_coeff_bytes[limb_idx];
        const uint8_t rhs_bytes = metadata.rhs_coeff_bytes[limb_idx];
        const uint8_t out_bytes = metadata.out_coeff_bytes[limb_idx];
        const uint64_t modulus = metadata.moduli[limb_idx];

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
        const BlockCopyMetadata metadata,
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

        const size_t src_stride = metadata.src_stride_bytes[limb_idx];
        const size_t dst_stride = metadata.dst_stride_bytes[limb_idx];
        const uint8_t src_bytes = metadata.src_coeff_bytes[limb_idx];
        const uint8_t dst_bytes = metadata.dst_coeff_bytes[limb_idx];
        const uint8_t *src_base = metadata.src_bases[limb_idx];
        uint8_t *dst_base = metadata.dst_bases[limb_idx];
        const uint64_t value =
            matrix_load_limb_u64(src_base, src_poly_idx, coeff_idx, src_stride, src_bytes);
        matrix_store_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes, value);
    }

    __global__ void block_add_rect_all_limbs_kernel(
        const BlockAddMetadata metadata,
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

        const size_t src_stride = metadata.src_stride_bytes[limb_idx];
        const size_t dst_stride = metadata.dst_stride_bytes[limb_idx];
        const uint8_t src_bytes = metadata.src_coeff_bytes[limb_idx];
        const uint8_t dst_bytes = metadata.dst_coeff_bytes[limb_idx];
        const uint8_t *src_base = metadata.src_bases[limb_idx];
        uint8_t *dst_base = metadata.dst_bases[limb_idx];
        const uint64_t modulus = metadata.moduli[limb_idx];
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

    __global__ void block_thin_row_matmul_kernel(
        const uint8_t *lhs_base,
        const uint8_t *rhs_base,
        uint8_t *out_base,
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
        uint64_t reciprocal,
        bool lazy_reduction)
    {
        const size_t col =
            static_cast<size_t>(blockIdx.x) * kThinMatmulColumnsPerBlock + threadIdx.y;
        if (col >= cols)
        {
            return;
        }

        const size_t coefficient_group = static_cast<size_t>(blockIdx.z);
        for (size_t coeff_idx = coefficient_group * kThinMatmulWarpSize + threadIdx.x;
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z) * kThinMatmulWarpSize)
        {
            uint64_t acc = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                const uint64_t lhs = matrix_load_limb_u64(
                    lhs_base,
                    k,
                    coeff_idx,
                    lhs_stride_bytes,
                    lhs_coeff_bytes);
                const uint64_t rhs = matrix_load_limb_u64(
                    rhs_base,
                    k * cols + col,
                    coeff_idx,
                    rhs_stride_bytes,
                    rhs_coeff_bytes);
                if (lazy_reduction)
                {
                    acc += lhs * rhs;
                }
                else
                {
                    acc = add_mod_u64(
                        acc,
                        mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal),
                        modulus);
                }
            }
            acc = lazy_reduction ? reduce_barrett_u32(acc, modulus, reciprocal) : acc;
            matrix_store_limb_u64(
                out_base,
                col,
                coeff_idx,
                out_stride_bytes,
                out_coeff_bytes,
                acc);
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
        BlockElementwiseMetadata metadata{};
        std::copy_n(lhs_bases, limb_count, metadata.lhs_bases);
        std::copy_n(rhs_bases, limb_count, metadata.rhs_bases);
        std::copy_n(out_bases, limb_count, metadata.out_bases);
        std::copy_n(lhs_stride_bytes, limb_count, metadata.lhs_stride_bytes);
        std::copy_n(rhs_stride_bytes, limb_count, metadata.rhs_stride_bytes);
        std::copy_n(out_stride_bytes, limb_count, metadata.out_stride_bytes);
        std::copy_n(lhs_coeff_bytes, limb_count, metadata.lhs_coeff_bytes);
        std::copy_n(rhs_coeff_bytes, limb_count, metadata.rhs_coeff_bytes);
        std::copy_n(out_coeff_bytes, limb_count, metadata.out_coeff_bytes);
        std::copy_n(moduli, limb_count, metadata.moduli);
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_elementwise_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            metadata,
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

        uint64_t reciprocal = 0;
        const bool use_thin_row_kernel =
            rows == 1 && matrix_barrett_u32_reciprocal(modulus, &reciprocal);
        const size_t coefficient_groups = use_thin_row_kernel
                                              ? (n + kThinMatmulWarpSize - 1) /
                                                    kThinMatmulWarpSize
                                              : n;
        const size_t grid_z = coefficient_groups < kMatmulMaxGridZ
                                  ? coefficient_groups
                                  : kMatmulMaxGridZ;
        if (use_thin_row_kernel)
        {
            const dim3 threads(kThinMatmulWarpSize, kThinMatmulColumnsPerBlock);
            const dim3 blocks(
                static_cast<unsigned int>(
                    (cols + kThinMatmulColumnsPerBlock - 1) /
                    kThinMatmulColumnsPerBlock),
                1,
                static_cast<unsigned int>(grid_z));
            block_thin_row_matmul_kernel<<<blocks, threads, 0, stream>>>(
                lhs_base,
                rhs_base,
                out_base,
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
                reciprocal,
                matrix_lazy_dot_u64(inner, modulus));
        }
        else
        {
            const dim3 threads(kMatmulTileN, kMatmulTileM);
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
        }

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
        BlockCopyMetadata metadata{};
        std::copy_n(src_bases, limb_count, metadata.src_bases);
        std::copy_n(dst_bases, limb_count, metadata.dst_bases);
        std::copy_n(src_stride_bytes, limb_count, metadata.src_stride_bytes);
        std::copy_n(dst_stride_bytes, limb_count, metadata.dst_stride_bytes);
        std::copy_n(src_coeff_bytes, limb_count, metadata.src_coeff_bytes);
        std::copy_n(dst_coeff_bytes, limb_count, metadata.dst_coeff_bytes);
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_copy_rect_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            metadata,
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
        BlockAddMetadata metadata{};
        std::copy_n(src_bases, limb_count, metadata.src_bases);
        std::copy_n(dst_bases, limb_count, metadata.dst_bases);
        std::copy_n(src_stride_bytes, limb_count, metadata.src_stride_bytes);
        std::copy_n(dst_stride_bytes, limb_count, metadata.dst_stride_bytes);
        std::copy_n(src_coeff_bytes, limb_count, metadata.src_coeff_bytes);
        std::copy_n(dst_coeff_bytes, limb_count, metadata.dst_coeff_bytes);
        std::copy_n(moduli, limb_count, metadata.moduli);
        const dim3 blocks(
            static_cast<unsigned int>((total + static_cast<size_t>(threads) - 1) /
                                      static_cast<size_t>(threads)),
            1u,
            static_cast<unsigned int>(limb_count));
        block_add_rect_all_limbs_kernel<<<blocks, threads, 0, stream>>>(
            metadata,
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
        status = matrix_wait_all_limb_streams(lhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(rhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(out, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

        status = launch_block_kernel_all_limbs(
            lhs_bases.data(),
            rhs_bases.data(),
            out_bases.data(),
            lhs_stride_bytes.data(),
            rhs_stride_bytes.data(),
            out_stride_bytes.data(),
            lhs_coeff_bytes.data(),
            rhs_coeff_bytes.data(),
            out_coeff_bytes.data(),
            moduli.data(),
            limb_count,
            count,
            n,
            op,
            dispatch_stream,
            0);
        if (status != 0)
        {
            return status;
        }

        status = matrix_track_all_limb_consumers(lhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_all_limb_consumers(rhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_all_limb_writes(out, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

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
        status = matrix_wait_all_limb_streams(lhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(scalar, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(out, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

        status = launch_block_kernel_all_limbs(
            lhs_bases.data(),
            scalar_bases.data(),
            out_bases.data(),
            lhs_stride_bytes.data(),
            scalar_stride_bytes.data(),
            out_stride_bytes.data(),
            lhs_coeff_bytes.data(),
            scalar_coeff_bytes.data(),
            out_coeff_bytes.data(),
            moduli.data(),
            limb_count,
            count,
            n,
            BlockOp::Mul,
            dispatch_stream,
            1);
        if (status != 0)
        {
            return status;
        }

        status = matrix_track_all_limb_consumers(lhs, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_all_limb_consumers(scalar, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_all_limb_writes(out, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

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
        status = matrix_wait_all_limb_streams(src, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(out, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

        status = launch_copy_kernel_all_limbs(
            src_bases.data(),
            dst_bases.data(),
            src_stride_bytes.data(),
            dst_stride_bytes.data(),
            src_coeff_bytes.data(),
            dst_coeff_bytes.data(),
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
            return status;
        }

        status = matrix_track_all_limb_consumers(src, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_all_limb_writes(out, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

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
        status = matrix_wait_all_limb_streams(src, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(out, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

        status = launch_add_block_kernel_all_limbs(
            src_bases.data(),
            dst_bases.data(),
            src_stride_bytes.data(),
            dst_stride_bytes.data(),
            src_coeff_bytes.data(),
            dst_coeff_bytes.data(),
            moduli.data(),
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
            return status;
        }

        status = matrix_track_all_limb_consumers(src, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_all_limb_writes(out, dispatch_stream);
        if (status != 0)
        {
            return status;
        }

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
        // A write to lhs may have been dispatched on a temporary work stream
        // (for example compact deserialization), so its ordinary limb stream
        // is not necessarily ordered after the latest write event.
        status = matrix_wait_limb_stream(lhs, limb_id, lhs_device, stream);
        if (status != 0)
        {
            return status;
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
    gpu_chacha::GpuRngSeed seed,
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
        out->format = lhs->format;
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
        out->format = lhs->format;
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

    out->format = lhs->format;
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

extern "C" int gpu_matrix_transpose(GpuMatrix *out, const GpuMatrix *source)
{
    if (!out || !source || out == source || !source->ctx || out->ctx != source->ctx ||
        source->level < 0 || out->level != source->level || out->format != source->format ||
        out->rows != source->cols || out->cols != source->rows)
        return set_error("invalid gpu_matrix_transpose arguments");
    if (source->rows == 0 || source->cols == 0 || source->ctx->N <= 0) return 0;
    const size_t n = static_cast<size_t>(source->ctx->N);
    const size_t limb_count = static_cast<size_t>(source->level) + 1;
    if (limb_count > kArithMetadataLimbs || source->ctx->limb_gpu_ids.size() < limb_count ||
        source->rows > std::numeric_limits<size_t>::max() / source->cols ||
        source->rows * source->cols > std::numeric_limits<size_t>::max() / n)
        return set_error("transpose shape overflow or invalid basis");
    TransposeMetadata metadata{};
    int device = -1;
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x >= source->shared_limb_buffers.size() || id.x >= out->shared_limb_buffers.size())
            return set_error("invalid transpose partition");
        const auto &input = source->shared_limb_buffers[id.x];
        const auto &output = out->shared_limb_buffers[id.x];
        if (!input.device_descriptors || !output.device_descriptors || id.y >= input.limb_count ||
            id.y >= output.limb_count || input.device != output.device)
            return set_error("invalid transpose descriptors");
        if (limb == 0)
        {
            device = output.device;
            metadata.source = input.device_descriptors;
            metadata.out = output.device_descriptors;
            const cudaError_t error = cudaSetDevice(device);
            if (error != cudaSuccess) return set_error(error);
        }
        else if (device != output.device || metadata.source != input.device_descriptors ||
                 metadata.out != output.device_descriptors)
            return set_error("transpose descriptors span partitions");
        metadata.indices[limb] = id.y;
    }
    status = matrix_wait_all_limb_streams(source, device, stream);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, device, stream);
    if (status != 0) return status;
    const size_t count = source->rows * source->cols * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    transpose_all_limbs_kernel<<<grid, 256, 0, stream>>>(
        metadata, source->rows, source->cols, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = matrix_track_all_limb_consumers(source, device, stream);
    if (status != 0) return status;
    return matrix_record_all_limb_writes(out, stream);
}

extern "C" int gpu_matrix_sum_rows(
    GpuMatrix *out, const GpuMatrix *source, const size_t *rows, const size_t *offsets,
    size_t group_count, size_t term_count)
{
    if (!out || !source || !source->ctx || out->ctx != source->ctx || source->level < 0 ||
        out->level != source->level || out->format != source->format ||
        out->rows != group_count || out->cols != source->cols || !rows || !offsets ||
        group_count == 0 || group_count > 16 || term_count == 0 || term_count > 32 ||
        offsets[0] != 0 || offsets[group_count] != term_count)
        return set_error("invalid gpu_matrix_sum_rows arguments");
    RowSumMetadata metadata{};
    for (size_t group = 0; group < group_count; ++group)
    {
        if (offsets[group] >= offsets[group + 1] || offsets[group + 1] > term_count)
            return set_error("invalid row sum group offsets");
        metadata.offsets[group] = offsets[group];
    }
    metadata.offsets[group_count] = term_count;
    for (size_t term = 0; term < term_count; ++term)
    {
        if (rows[term] >= source->rows) return set_error("row sum input index out of bounds");
        metadata.rows[term] = rows[term];
    }
    if (source->cols == 0 || source->ctx->N <= 0) return 0;
    const size_t n = static_cast<size_t>(source->ctx->N);
    const size_t limb_count = static_cast<size_t>(source->level) + 1;
    if (limb_count > kArithMetadataLimbs || source->ctx->limb_gpu_ids.size() < limb_count ||
        source->ctx->moduli.size() < limb_count || out->rows > std::numeric_limits<size_t>::max() / out->cols ||
        out->rows * out->cols > std::numeric_limits<size_t>::max() / n)
        return set_error("row sum shape overflow or invalid basis");
    int device = -1;
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x >= source->shared_limb_buffers.size() || id.x >= out->shared_limb_buffers.size())
            return set_error("invalid row sum partition");
        const auto &input = source->shared_limb_buffers[id.x];
        const auto &output = out->shared_limb_buffers[id.x];
        if (!input.device_descriptors || !output.device_descriptors || id.y >= input.limb_count ||
            id.y >= output.limb_count || input.device != output.device)
            return set_error("invalid row sum descriptors");
        if (limb == 0)
        {
            device = output.device;
            metadata.source = input.device_descriptors;
            metadata.out = output.device_descriptors;
            const cudaError_t error = cudaSetDevice(device);
            if (error != cudaSuccess) return set_error(error);
        }
        else if (device != output.device || metadata.source != input.device_descriptors ||
                 metadata.out != output.device_descriptors)
            return set_error("row sum descriptors span partitions");
        metadata.indices[limb] = id.y;
        metadata.moduli[limb] = source->ctx->moduli[limb];
    }
    status = matrix_wait_all_limb_streams(source, device, stream);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, device, stream);
    if (status != 0) return status;
    const size_t count = out->rows * out->cols * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    sum_rows_all_limbs_kernel<<<grid, 256, 0, stream>>>(metadata, source->cols, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = matrix_track_all_limb_consumers(source, device, stream);
    if (status != 0) return status;
    return matrix_record_all_limb_writes(out, stream);
}

extern "C" int gpu_matrix_add_row_blocks(
    GpuMatrix *out, const GpuMatrix *const *lhs_blocks, size_t block_count, const GpuMatrix *rhs)
{
    if (!out || !rhs || !rhs->ctx || !lhs_blocks || block_count == 0 || block_count > kRowAddMaxBlocks ||
        out->ctx != rhs->ctx || out->level != rhs->level || rhs->level < 0 ||
        out->rows != rhs->rows || out->cols != rhs->cols || rhs->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid gpu_matrix_add_row_blocks arguments");
    RowBlockAddMetadata metadata{};
    size_t total_rows = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *matrix = lhs_blocks[block];
        if (!matrix || matrix->ctx != rhs->ctx || matrix->level != rhs->level ||
            matrix->format != GPU_POLY_FORMAT_EVAL || matrix->cols != rhs->cols ||
            matrix->rows > std::numeric_limits<size_t>::max() - total_rows)
            return set_error("invalid row block input");
        metadata.rows[block] = matrix->rows;
        total_rows += matrix->rows;
    }
    if (total_rows != rhs->rows) return set_error("row block sum differs from output rows");
    if (rhs->rows == 0 || rhs->cols == 0 || rhs->ctx->N <= 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    const size_t n = static_cast<size_t>(rhs->ctx->N);
    const size_t limb_count = static_cast<size_t>(rhs->level) + 1;
    if (limb_count > kArithMetadataLimbs || rhs->ctx->limb_gpu_ids.size() < limb_count ||
        rhs->ctx->moduli.size() < limb_count || rhs->rows > std::numeric_limits<size_t>::max() / rhs->cols ||
        rhs->rows * rhs->cols > std::numeric_limits<size_t>::max() / n)
        return set_error("invalid row block add shape or basis");
    int device = -1;
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    // Gather one descriptor pointer per matrix. Limb pointers/strides are
    // resolved by the GPU; each matrix retains its readiness dependency.
    auto prepare = [&](const GpuMatrix *matrix,
                       const GpuMatrix::SharedLimbBuffer::DeviceDescriptor **descriptors) -> int {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = rhs->ctx->limb_gpu_ids[limb];
            if (id.x >= matrix->shared_limb_buffers.size()) return set_error("invalid row block partition");
            const auto &buffer = matrix->shared_limb_buffers[id.x];
            if (!buffer.device_descriptors || id.y >= buffer.limb_count)
                return set_error("missing row block descriptors");
            if (device < 0)
            {
                device = buffer.device;
                const cudaError_t error = cudaSetDevice(device);
                if (error != cudaSuccess) return set_error(error);
            }
            if (device != buffer.device) return set_error("row blocks must share a device");
            if (limb == 0) *descriptors = buffer.device_descriptors;
            else if (*descriptors != buffer.device_descriptors)
                return set_error("row block descriptors span partitions");
        }
        return matrix_wait_all_limb_streams(matrix, device, stream);
    };
    status = prepare(out, &metadata.out);
    if (status != 0) return status;
    status = prepare(rhs, &metadata.rhs);
    if (status != 0) return status;
    for (size_t block = 0; block < block_count; ++block)
    {
        if (metadata.rows[block] == 0) continue;
        status = prepare(lhs_blocks[block], &metadata.blocks[block]);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        metadata.indices[limb] = rhs->ctx->limb_gpu_ids[limb].y;
        metadata.moduli[limb] = rhs->ctx->moduli[limb];
    }
    const size_t count = rhs->rows * rhs->cols * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    add_row_blocks_all_limbs_kernel<<<grid, 256, 0, stream>>>(metadata, rhs->cols, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    for (size_t block = 0; block < block_count; ++block)
    {
        if (metadata.rows[block] == 0) continue;
        status = matrix_track_all_limb_consumers(lhs_blocks[block], device, stream);
        if (status != 0) return status;
    }
    status = matrix_track_all_limb_consumers(rhs, device, stream);
    if (status != 0) return status;
    status = matrix_record_all_limb_writes(out, stream);
    if (status != 0) return status;
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_tensor(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs || !lhs->ctx || out->ctx != lhs->ctx || rhs->ctx != lhs->ctx ||
        lhs->level < 0 || out->level != lhs->level || rhs->level != lhs->level ||
        lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid gpu_matrix_tensor arguments");
    const size_t maximum = std::numeric_limits<size_t>::max();
    if ((rhs->rows != 0 && lhs->rows > maximum / rhs->rows) ||
        (rhs->cols != 0 && lhs->cols > maximum / rhs->cols) ||
        out->rows != lhs->rows * rhs->rows || out->cols != lhs->cols * rhs->cols)
        return set_error("invalid gpu_matrix_tensor output shape");
    if (out->rows == 0 || out->cols == 0 || lhs->ctx->N <= 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    const size_t n = static_cast<size_t>(lhs->ctx->N);
    if (out->rows > maximum / out->cols || out->rows * out->cols > maximum / n ||
        lhs->ctx->limb_gpu_ids.size() <= static_cast<size_t>(lhs->level))
        return set_error("gpu_matrix_tensor shape overflow or invalid limb mapping");
    const int status = launch_descriptor_product(out, lhs, rhs, true);
    if (status == 0) out->format = GPU_POLY_FORMAT_EVAL;
    return status;
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

    if (lhs->rows > 0 && lhs->rows <= 4 && rhs->cols > 0 && rhs->cols <= 4 &&
        lhs->cols > 0 && lhs->cols <= 16)
    {
        const int status = launch_descriptor_product(out, lhs, rhs, false);
        if (status == 0) out->format = GPU_POLY_FORMAT_EVAL;
        return status;
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
