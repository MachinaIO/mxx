#include <cudaTypedefs.h>
#include <array>
#include <string>

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
        GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out;
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
        size_t source_offset, source_stride, output_offset, output_stride;
    };
    static_assert(sizeof(TransposeMetadata) + 4 * sizeof(size_t) <= 4096,
                  "transpose exceeds portable CUDA parameter budget");

    __global__ void transpose_all_limbs_kernel(
        TransposeMetadata metadata, size_t source_rows, size_t count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const auto source = metadata.source[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t coefficient = index % n;
            const size_t source_poly = metadata.source_offset +
                (poly % source_rows) * metadata.source_stride + poly / source_rows;
            const size_t output_poly = metadata.output_offset +
                (poly / source_rows) * metadata.output_stride + poly % source_rows;
            const uint64_t value = matrix_load_limb_u64(
                source.base, source_poly, coefficient, source.stride, source.width);
            matrix_store_limb_u64(out.base, output_poly, coefficient, out.stride, out.width, value);
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
        size_t source_offset, source_stride, output_offset, output_stride;
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
                    metadata.source_offset + metadata.rows[term] * metadata.source_stride + col,
                    coefficient, source.stride, source.width),
                    metadata.moduli[limb]);
            matrix_store_limb_u64(out.base, metadata.output_offset + row * metadata.output_stride + col,
                coefficient, out.stride, out.width, sum);
        }
    }

    constexpr size_t kRowAddMaxBlocks = 16;
    struct RowBlockAddMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *blocks[kRowAddMaxBlocks];
        size_t rows[kRowAddMaxBlocks];
        size_t offsets[kRowAddMaxBlocks];
        size_t strides[kRowAddMaxBlocks];
        size_t rhs_offset, rhs_stride, out_offset, out_stride;
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
            const uint64_t a = matrix_load_limb_u64(lhs.base, metadata.offsets[block] + row * metadata.strides[block] + poly % cols,
                                                  coefficient, lhs.stride, lhs.width);
            const uint64_t b = matrix_load_limb_u64(rhs.base, metadata.rhs_offset + (poly / cols) * metadata.rhs_stride + poly % cols, coefficient, rhs.stride, rhs.width);
            matrix_store_limb_u64(out.base, metadata.out_offset + (poly / cols) * metadata.out_stride + poly % cols, coefficient, out.stride, out.width,
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

    struct TensorGeometry
    {
        size_t lhs_start, lhs_stride, rhs_start, rhs_stride;
        size_t output_start, output_stride, column_start, columns;
    };

    int validate_tensor_ranges(const GpuMatrix *out, const GpuMatrix *lhs,
        const GpuMatrix *rhs, const GpuMatrixBatchView *view, size_t column_start,
        bool grouped, size_t group_count)
    {
        if (!view) return column_start == 0 ? 0 : set_error("tensor offset requires ranges");
        const auto valid = [](const GpuMatrix *m, const GpuMatrixRange &r) {
            return r.row_start <= r.row_end && r.row_end <= m->rows &&
                r.column_start <= r.column_end && r.column_end <= m->cols &&
                (m->cols == 0 || m->rows <= SIZE_MAX / m->cols);
        };
        if (out == lhs || out == rhs || out->format != GPU_POLY_FORMAT_EVAL ||
            !valid(lhs, view->left) || !valid(rhs, view->right) || !valid(out, view->output))
            return set_error("invalid tensor owner or rectangle");
        const size_t lr = view->left.row_end - view->left.row_start;
        const size_t lc = view->left.column_end - view->left.column_start;
        const size_t rr = view->right.row_end - view->right.row_start;
        const size_t rc = view->right.column_end - view->right.column_start;
        if ((rr && lr > SIZE_MAX / rr) || (rc && lc > SIZE_MAX / rc))
            return set_error("tensor rectangle product overflow");
        const size_t columns = view->output.column_end - view->output.column_start;
        if (view->output.row_end - view->output.row_start != (grouped ? group_count : lr * rr) ||
            column_start > lc * rc || columns > lc * rc - column_start)
            return set_error("tensor output interval is outside its exact product");
        return 0;
    }

    __global__ void tensor_all_limbs_kernel(
        DescriptorProductMetadata metadata, size_t lhs_cols, size_t rhs_rows,
        size_t rhs_cols, size_t output_count, size_t n, TensorGeometry geometry)
    {
        const size_t limb = blockIdx.z;
        const auto lhs = metadata.lhs[metadata.indices[limb]];
        const auto rhs = metadata.rhs[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        const size_t output_cols = geometry.columns;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < output_count; index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const size_t poly = index / n;
            const size_t coefficient = index % n;
            const size_t row = poly / output_cols;
            const size_t col = geometry.column_start + poly % output_cols;
            const size_t left_poly = geometry.lhs_start + (row / rhs_rows) * geometry.lhs_stride + col / rhs_cols;
            const size_t right_poly = geometry.rhs_start + (row % rhs_rows) * geometry.rhs_stride + col % rhs_cols;
            const size_t output_poly = geometry.output_start + row * geometry.output_stride + poly % output_cols;
            const uint64_t a = matrix_load_limb_u64(lhs.base, left_poly, coefficient,
                                                  lhs.stride, lhs.width);
            const uint64_t b = matrix_load_limb_u64(rhs.base, right_poly, coefficient,
                                                  rhs.stride, rhs.width);
            matrix_store_limb_u64(out.base, output_poly, coefficient, out.stride, out.width,
                                 mul_mod_u64(a, b, metadata.moduli[limb]));
        }
    }

    struct TensorRowSumMetadata
    {
        DescriptorProductMetadata product;
        TensorGeometry geometry;
        size_t rows[32];
        size_t offsets[17];
        uint8_t *output_base;
        size_t output_stride;
        size_t output_offsets[kArithMetadataLimbs];
        uint8_t output_widths[kArithMetadataLimbs];
        bool initialize_output_descriptors;
        GpuBarrettReciprocal reciprocals[kArithMetadataLimbs];
    };
    static_assert(sizeof(TensorRowSumMetadata) + 5 * sizeof(size_t) <= 4096,
                  "tensor row sum exceeds portable CUDA parameter budget");

    template <bool SeparatePolynomials>
    __global__ void tensor_sum_rows_all_limbs_kernel(
        TensorRowSumMetadata metadata, size_t lhs_cols, size_t rhs_rows,
        size_t rhs_cols, size_t poly_count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const size_t descriptor = metadata.product.indices[limb];
        const auto lhs = metadata.product.lhs[descriptor];
        const auto rhs = metadata.product.rhs[descriptor];
        // Deferred output descriptors cannot be read by this launch: another
        // block may not have written them yet. Arithmetic uses by-value layout.
        const auto out = metadata.initialize_output_descriptors
            ? GpuMatrix::SharedLimbBuffer::DeviceDescriptor{
                metadata.output_base + metadata.output_offsets[limb],
                metadata.output_stride, metadata.output_widths[limb]}
            : metadata.product.out[descriptor];
        if (metadata.initialize_output_descriptors && blockIdx.x == 0 &&
            blockIdx.y == 0 && threadIdx.x == 0)
            metadata.product.out[descriptor] = out;
        const uint64_t modulus = metadata.product.moduli[limb];
        const auto reciprocal = metadata.reciprocals[limb];
        const auto geometry = metadata.geometry;
        const size_t cols = geometry.columns;
        const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if constexpr (SeparatePolynomials)
            if (thread >= n) return;
        const size_t end = SeparatePolynomials ? poly_count : poly_count * n;
        const size_t step = SeparatePolynomials ? gridDim.y :
            static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t work = SeparatePolynomials ? blockIdx.y : thread; work < end; work += step)
        {
            const size_t poly = SeparatePolynomials ? work : work / n;
            const size_t group = poly / cols;
            const size_t column = geometry.column_start + poly % cols;
            const size_t coefficient = SeparatePolynomials ? thread : work % n;
            unsigned __int128 sum = 0;
            for (size_t term = metadata.offsets[group]; term < metadata.offsets[group + 1]; ++term)
            {
                const size_t row = metadata.rows[term];
                const uint64_t a = matrix_load_limb_u64(lhs.base,
                    geometry.lhs_start + (row / rhs_rows) * geometry.lhs_stride + column / rhs_cols,
                    coefficient, lhs.stride, lhs.width);
                const uint64_t b = matrix_load_limb_u64(rhs.base,
                    geometry.rhs_start + (row % rhs_rows) * geometry.rhs_stride + column % rhs_cols,
                    coefficient, rhs.stride, rhs.width);
                unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
                // The common short dot needs only one final division. Flush
                // before a 128-bit carry, as in the generic small-dot kernel.
                if (~static_cast<unsigned __int128>(0) - sum < product)
                {
                    sum = matrix_reduce_barrett_u128(sum, modulus, reciprocal.lo, reciprocal.hi);
                    product = matrix_reduce_barrett_u128(product, modulus, reciprocal.lo, reciprocal.hi);
                }
                sum += product;
            }
            matrix_store_limb_u64(out.base, geometry.output_start + group * geometry.output_stride + poly % cols, coefficient, out.stride, out.width,
                                 matrix_reduce_barrett_u128(sum, modulus, reciprocal.lo, reciprocal.hi));
        }
    }

    template <bool SeparatePolynomials>
    int launch_tensor_row_sum_driver(
        const GpuKernelPartition &kernels, TensorRowSumMetadata &metadata,
        size_t lhs_cols, size_t rhs_rows, size_t rhs_cols, size_t poly_count,
        size_t n, dim3 grid, cudaStream_t stream)
    {
        const auto function = kernels.tensor_row_sum[SeparatePolynomials ? 1 : 0];
        if (!kernels.launch_entry || !function)
            return set_error("tensor row-sum kernel was not provisioned at context setup");
        void *arguments[] = {&metadata, &lhs_cols, &rhs_rows, &rhs_cols, &poly_count, &n};
        const CUresult result = reinterpret_cast<PFN_cuLaunchKernel_v4000>(kernels.launch_entry)(
            reinterpret_cast<CUfunction>(function), grid.x, grid.y, grid.z,
            256, 1, 1, 0, reinterpret_cast<CUstream>(stream), arguments, nullptr);
        if (result != CUDA_SUCCESS)
        {
            // The caller retires the output's producer stream on failure,
            // including asynchronous errors reported by the driver launch.
            return set_error(("cuLaunchKernel failed: " + std::to_string(result)).c_str());
        }
        return 0;
    }

    int launch_descriptor_product(
        GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, bool tensor,
        const size_t *rows = nullptr, const size_t *offsets = nullptr,
        size_t group_count = 0, size_t term_count = 0,
        const GpuMatrixBatchView *view = nullptr, size_t column_start = 0)
    {
        const size_t limb_count = static_cast<size_t>(lhs->level) + 1;
        if (limb_count > kArithMetadataLimbs || lhs->ctx->moduli.size() < limb_count)
            return set_error("invalid descriptor product modulus count");
        const GpuMatrixRange left = view ? view->left : GpuMatrixRange{0, lhs->rows, 0, lhs->cols};
        const GpuMatrixRange right = view ? view->right : GpuMatrixRange{0, rhs->rows, 0, rhs->cols};
        const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
        const size_t left_columns = left.column_end - left.column_start;
        const size_t right_rows = right.row_end - right.row_start;
        const size_t right_columns = right.column_end - right.column_start;
        const TensorGeometry geometry{left.row_start * lhs->cols + left.column_start, lhs->cols,
            right.row_start * rhs->cols + right.column_start, rhs->cols,
            output.row_start * out->cols + output.column_start, out->cols, column_start,
            output.column_end - output.column_start};
        const size_t output_rows = output.row_end - output.row_start;
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
        // The descriptor partition checks above establish one selected device
        // for the entire submission. Nested helpers retain all event joins.
        status = matrix_wait_all_limb_streams(lhs, device, stream, true, true);
        if (status != 0) return status;
        status = matrix_wait_all_limb_streams(rhs, device, stream, true, true);
        if (status != 0) return status;
        status = matrix_wait_all_limb_streams(out, device, stream, true);
        if (status != 0) return status;
        const size_t n = static_cast<size_t>(lhs->ctx->N);
        if (tensor)
        {
            const size_t count = output_rows * geometry.columns * n;
            const size_t blocks = count / 256 + (count % 256 != 0);
            const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                            1, static_cast<unsigned int>(limb_count));
            if (rows)
            {
                TensorRowSumMetadata grouped{};
                grouped.product = metadata;
                grouped.geometry = geometry;
                std::copy_n(lhs->ctx->barrett_reciprocals.data(), limb_count, grouped.reciprocals);
                grouped.initialize_output_descriptors = !out->descriptors_initialized;
                if (grouped.initialize_output_descriptors)
                {
                    const auto &buffer = out->shared_limb_buffers[out->ctx->limb_gpu_ids[0].x];
                    grouped.output_base = buffer.ptr;
                    grouped.output_stride = buffer.bytes_per_poly;
                    for (size_t limb = 0; limb < limb_count; ++limb)
                    {
                        const size_t local = metadata.indices[limb];
                        grouped.output_offsets[limb] = buffer.limb_offsets_bytes[local];
                        grouped.output_widths[limb] = buffer.limb_coeff_bytes[local];
                    }
                }
                std::copy_n(rows, term_count, grouped.rows);
                std::copy_n(offsets, group_count + 1, grouped.offsets);
                const size_t poly_count = output_rows * geometry.columns;
                const size_t partition = out->ctx->limb_gpu_ids[0].x;
                if (partition >= out->ctx->execution->kernel_partitions.size())
                    return set_error("missing tensor row-sum kernel partition");
                const auto &kernels = out->ctx->execution->kernel_partitions[partition];
                if (n >= 256)
                {
                    const dim3 grouped_grid(static_cast<unsigned int>((n + 255) / 256),
                        static_cast<unsigned int>(std::min(poly_count, size_t{65535})),
                        static_cast<unsigned int>(limb_count));
                    status = launch_tensor_row_sum_driver<true>(
                        kernels, grouped, left_columns, right_rows, right_columns, poly_count, n, grouped_grid, stream);
                }
                else
                {
                    // Pack tiny polynomials together to retain full thread blocks.
                    status = launch_tensor_row_sum_driver<false>(
                        kernels, grouped, left_columns, right_rows, right_columns, poly_count, n, grid, stream);
                }
                if (status != 0)
                {
                    gpu_matrix_retire_submitted_work(out);
                    return status;
                }
            }
            else
            {
                tensor_all_limbs_kernel<<<grid, 256, 0, stream>>>(
                    metadata, left_columns, right_rows, right_columns, count, n, geometry);
            }
        }
        else
        {
            const dim3 grid(static_cast<unsigned int>((n + 255) / 256),
                            static_cast<unsigned int>(rhs->cols), static_cast<unsigned int>(limb_count));
            small_dot_all_limbs_kernel<<<grid, 256, 0, stream>>>(
                metadata, lhs->rows, lhs->cols, rhs->cols, n);
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            gpu_matrix_retire_submitted_work(out);
            return set_error(error);
        }
        // Descriptor writes and arithmetic share the output completion below.
        if (rows) out->descriptors_initialized = true;
        status = matrix_record_all_limb_writes(out, stream, true);
        if (status != 0)
        {
            // No input lifetime join exists yet when output recording fails.
            gpu_matrix_retire_submitted_work(out);
            return status;
        }
        // The output event already covers this kernel. Reuse it for both
        // producer-stream joins instead of recording two temporary events.
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const auto &states = out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        status = matrix_track_all_limb_consumers(lhs, device, stream, completion, true, view != nullptr);
        if (status != 0)
        {
            gpu_matrix_retire_submitted_work(out);
            return status;
        }
        status = matrix_track_all_limb_consumers(rhs, device, stream, completion, true, view != nullptr);
        if (status != 0)
        {
            gpu_matrix_retire_submitted_work(out);
            return status;
        }
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

    __global__ void prepared_add_rect_all_limbs_kernel(
        const BlockElementwiseMetadata metadata, size_t limb_count,
        size_t rows, size_t columns, size_t n,
        size_t lhs_cols, size_t rhs_cols, size_t out_cols,
        size_t lhs_row, size_t lhs_column, size_t rhs_row, size_t rhs_column,
        size_t out_row, size_t out_column)
    {
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (limb >= limb_count) return;
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = rows * columns * n;
        if (index >= total) return;
        const size_t poly = index / n;
        const size_t coefficient = index % n;
        const size_t row = poly / columns;
        const size_t column = poly % columns;
        const size_t lhs_poly = (lhs_row + row) * lhs_cols + lhs_column + column;
        const size_t rhs_poly = (rhs_row + row) * rhs_cols + rhs_column + column;
        const size_t out_poly = (out_row + row) * out_cols + out_column + column;
        const uint64_t lhs = matrix_load_limb_u64(
            metadata.lhs_bases[limb], lhs_poly, coefficient,
            metadata.lhs_stride_bytes[limb], metadata.lhs_coeff_bytes[limb]);
        const uint64_t rhs = matrix_load_limb_u64(
            metadata.rhs_bases[limb], rhs_poly, coefficient,
            metadata.rhs_stride_bytes[limb], metadata.rhs_coeff_bytes[limb]);
        matrix_store_limb_u64(
            metadata.out_bases[limb], out_poly, coefficient,
            metadata.out_stride_bytes[limb], metadata.out_coeff_bytes[limb],
            add_mod_u64(lhs, rhs, metadata.moduli[limb]));
    }

    __device__ __forceinline__ uint64_t prepared_sub_mod_u64(uint64_t lhs, uint64_t rhs, uint64_t modulus)
    {
        return lhs >= rhs ? lhs - rhs : modulus - (rhs - lhs);
    }

    __global__ void prepared_sub_rect_all_limbs_kernel(
        const BlockElementwiseMetadata metadata, size_t limb_count,
        size_t rows, size_t columns, size_t n,
        size_t lhs_cols, size_t rhs_cols, size_t out_cols,
        size_t lhs_row, size_t lhs_column, size_t rhs_row, size_t rhs_column,
        size_t out_row, size_t out_column)
    {
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (limb >= limb_count) return;
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = rows * columns * n;
        if (index >= total) return;
        const size_t poly = index / n;
        const size_t coefficient = index % n;
        const size_t row = poly / columns;
        const size_t column = poly % columns;
        const size_t lhs_poly = (lhs_row + row) * lhs_cols + lhs_column + column;
        const size_t rhs_poly = (rhs_row + row) * rhs_cols + rhs_column + column;
        const size_t out_poly = (out_row + row) * out_cols + out_column + column;
        const uint64_t lhs = matrix_load_limb_u64(
            metadata.lhs_bases[limb], lhs_poly, coefficient,
            metadata.lhs_stride_bytes[limb], metadata.lhs_coeff_bytes[limb]);
        const uint64_t rhs = matrix_load_limb_u64(
            metadata.rhs_bases[limb], rhs_poly, coefficient,
            metadata.rhs_stride_bytes[limb], metadata.rhs_coeff_bytes[limb]);
        matrix_store_limb_u64(
            metadata.out_bases[limb], out_poly, coefficient,
            metadata.out_stride_bytes[limb], metadata.out_coeff_bytes[limb],
            prepared_sub_mod_u64(lhs, rhs, metadata.moduli[limb]));
    }

    __global__ void prepared_unary_rect_all_limbs_kernel(
        const BlockElementwiseMetadata metadata, const uint64_t *scalars,
        size_t limb_count, size_t rows, size_t columns, size_t n,
        size_t lhs_cols, size_t out_cols, size_t lhs_row, size_t lhs_column,
        size_t out_row, size_t out_column, int operation)
    {
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (limb >= limb_count) return;
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = rows * columns * n;
        if (index >= total) return;
        const size_t poly = index / n;
        const size_t coefficient = index % n;
        const size_t row = poly / columns;
        const size_t column = poly % columns;
        const size_t lhs_poly = (lhs_row + row) * lhs_cols + lhs_column + column;
        const size_t out_poly = (out_row + row) * out_cols + out_column + column;
        const uint64_t value = matrix_load_limb_u64(
            metadata.lhs_bases[limb], lhs_poly, coefficient,
            metadata.lhs_stride_bytes[limb], metadata.lhs_coeff_bytes[limb]);
        const uint64_t output = operation == 0
            ? (value == 0 ? 0 : metadata.moduli[limb] - value)
            : mul_mod_u64(value, scalars[limb], metadata.moduli[limb]);
        matrix_store_limb_u64(
            metadata.out_bases[limb], out_poly, coefficient,
            metadata.out_stride_bytes[limb], metadata.out_coeff_bytes[limb], output);
    }

    __global__ void prepared_automorphism_rect_all_limbs_kernel(
        const BlockElementwiseMetadata metadata, size_t limb_count,
        size_t rows, size_t columns, size_t n, size_t lhs_cols, size_t out_cols,
        size_t lhs_row, size_t lhs_column, size_t out_row, size_t out_column,
        size_t index, bool evaluation, unsigned log_n)
    {
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (limb >= limb_count) return;
        const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = rows * columns * n;
        if (linear >= total) return;
        const size_t poly = linear / n;
        const size_t position = linear % n;
        const size_t row = poly / columns;
        const size_t column = poly % columns;
        const size_t lhs_poly = (lhs_row + row) * lhs_cols + lhs_column + column;
        const size_t out_poly = (out_row + row) * out_cols + out_column + column;
        size_t source = position;
        size_t target = position;
        uint64_t value;
        if (evaluation)
        {
            const size_t k = log_n == 0 ? 0 : __brev(static_cast<unsigned>(position)) >> (32 - log_n);
            const size_t exponent = ((2 * k + 1) * index) % (2 * n);
            const unsigned natural = static_cast<unsigned>((exponent - 1) / 2);
            source = log_n == 0 ? 0 : __brev(natural) >> (32 - log_n);
            value = matrix_load_limb_u64(metadata.lhs_bases[limb], lhs_poly, source,
                metadata.lhs_stride_bytes[limb], metadata.lhs_coeff_bytes[limb]);
            target = position;
        }
        else
        {
            const size_t exponent = (position * index) % (2 * n);
            target = exponent < n ? exponent : exponent - n;
            value = matrix_load_limb_u64(metadata.lhs_bases[limb], lhs_poly, source,
                metadata.lhs_stride_bytes[limb], metadata.lhs_coeff_bytes[limb]);
            if (exponent >= n && value != 0) value = metadata.moduli[limb] - value;
        }
        matrix_store_limb_u64(metadata.out_bases[limb], out_poly, target,
            metadata.out_stride_bytes[limb], metadata.out_coeff_bytes[limb], value);
    }

    __global__ void prepared_small_dot_rect_kernel(
        DescriptorProductMetadata metadata, size_t rows, size_t inner, size_t columns,
        size_t n, size_t lhs_cols, size_t rhs_cols, size_t out_cols,
        size_t lhs_row, size_t lhs_column, size_t rhs_row, size_t rhs_column,
        size_t out_row, size_t out_column)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= n) return;
        const size_t limb = blockIdx.z;
        const auto lhs = metadata.lhs[metadata.indices[limb]];
        const auto rhs = metadata.rhs[metadata.indices[limb]];
        const auto out = metadata.out[metadata.indices[limb]];
        const uint64_t modulus = metadata.moduli[limb];
        const size_t column = blockIdx.y;
        if (column >= columns) return;
        for (size_t row = 0; row < rows; ++row)
        {
            unsigned __int128 sum = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                const uint64_t a = matrix_load_limb_u64(
                    lhs.base, (lhs_row + row) * lhs_cols + lhs_column + k,
                    coefficient, lhs.stride, lhs.width);
                const uint64_t b = matrix_load_limb_u64(
                    rhs.base, (rhs_row + k) * rhs_cols + rhs_column + column,
                    coefficient, rhs.stride, rhs.width);
                const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
                if (~static_cast<unsigned __int128>(0) - sum < product)
                {
                    sum %= modulus;
                }
                sum += product;
            }
            matrix_store_limb_u64(
                out.base, (out_row + row) * out_cols + out_column + column,
                coefficient, out.stride, out.width, static_cast<uint64_t>(sum % modulus));
        }
    }

    __global__ void prepared_block_matmul_rect_kernel(
        const uint8_t *lhs_base, const uint8_t *rhs_base, uint8_t *out_base,
        size_t rows, size_t inner, size_t columns, size_t n,
        size_t lhs_pitch, size_t rhs_pitch, size_t out_pitch,
        size_t lhs_row, size_t lhs_column, size_t rhs_row, size_t rhs_column,
        size_t out_row, size_t out_column,
        size_t lhs_stride, size_t rhs_stride, size_t out_stride,
        uint8_t lhs_width, uint8_t rhs_width, uint8_t out_width, uint64_t modulus)
    {
        __shared__ uint64_t lhs_tile[kMatmulTileM][kMatmulTileK];
        __shared__ uint64_t rhs_tile[kMatmulTileK][kMatmulTileN];
        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t column = col_base + threadIdx.x;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        for (size_t coefficient = static_cast<size_t>(blockIdx.z);
             coefficient < n; coefficient += static_cast<size_t>(gridDim.z))
        {
            uint64_t sum = 0;
            for (size_t k0 = 0; k0 < inner; k0 += kMatmulTileK)
            {
                for (int i = tid; i < kMatmulTileM * kMatmulTileK; i += threads)
                {
                    const int tile_row = i / kMatmulTileK;
                    const int tile_k = i - tile_row * kMatmulTileK;
                    const size_t source_row = row_base + static_cast<size_t>(tile_row);
                    const size_t source_column = k0 + static_cast<size_t>(tile_k);
                    uint64_t value = 0;
                    if (source_row < rows && source_column < inner)
                        value = matrix_load_limb_u64(
                            lhs_base, (lhs_row + source_row) * lhs_pitch + lhs_column + source_column,
                            coefficient, lhs_stride, lhs_width);
                    lhs_tile[tile_row][tile_k] = value;
                }
                for (int i = tid; i < kMatmulTileK * kMatmulTileN; i += threads)
                {
                    const int tile_k = i / kMatmulTileN;
                    const int tile_column = i - tile_k * kMatmulTileN;
                    const size_t source_row = k0 + static_cast<size_t>(tile_k);
                    const size_t source_column = col_base + static_cast<size_t>(tile_column);
                    uint64_t value = 0;
                    if (source_row < inner && source_column < columns)
                        value = matrix_load_limb_u64(
                            rhs_base, (rhs_row + source_row) * rhs_pitch + rhs_column + source_column,
                            coefficient, rhs_stride, rhs_width);
                    rhs_tile[tile_k][tile_column] = value;
                }
                __syncthreads();
                if (row < rows && column < columns)
                    for (int k = 0; k < kMatmulTileK; ++k)
                        sum = add_mod_u64(sum, mul_mod_u64(
                            lhs_tile[threadIdx.y][k], rhs_tile[k][threadIdx.x], modulus), modulus);
                __syncthreads();
            }
            if (row < rows && column < columns)
                matrix_store_limb_u64(
                    out_base, (out_row + row) * out_pitch + out_column + column,
                    coefficient, out_stride, out_width, sum);
        }
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

    // Each comparison owns one result span per physical partition. Every
    // preceding limb readback has completed before the next stream reuses it.
    struct EqualityWorkspace
    {
        GpuDeviceWorkspace storage;
        cudaStream_t last_stream = nullptr;
        ~EqualityWorkspace() { storage.release(last_stream); }
    };

    __global__ void block_equal_kernel(
        const uint8_t *lhs,
        const uint8_t *rhs,
        size_t lhs_stride_bytes,
        size_t rhs_stride_bytes,
        uint8_t lhs_coeff_bytes,
        uint8_t rhs_coeff_bytes,
        size_t poly_count,
        size_t n,
        int *out_equal)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= poly_count * n) return;
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx - poly_idx * n;
        const uint64_t lhs_value = matrix_load_packed_u64_at(
            lhs + poly_idx * lhs_stride_bytes + coeff_idx * lhs_coeff_bytes, lhs_coeff_bytes);
        const uint64_t rhs_value = matrix_load_packed_u64_at(
            rhs + poly_idx * rhs_stride_bytes + coeff_idx * rhs_coeff_bytes, rhs_coeff_bytes);
        if (lhs_value != rhs_value) atomicExch(out_equal, 0);
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
        status = matrix_wait_limb_stream(lhs, limb_id, out_device, stream, false, true);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(rhs, limb_id, out_device, stream, false, true);
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
        status = matrix_wait_all_limb_streams(lhs, dispatch_device, dispatch_stream, false, true);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(rhs, dispatch_device, dispatch_stream, false, true);
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
        status = matrix_wait_all_limb_streams(lhs, dispatch_device, dispatch_stream, false, true);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_all_limb_streams(scalar, dispatch_device, dispatch_stream, false, true);
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
        status = matrix_wait_all_limb_streams(src, dispatch_device, dispatch_stream, false, true);
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

        status = matrix_record_all_limb_writes(out, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        // The output's completion already dominates the copy. Reuse its
        // recorded state to retire source reads without a temporary TLS event
        // or rewriting the immutable input's writer event.
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const cudaEvent_t copied = out->exec_limb_states[first.x][first.y].write_done;
        return matrix_track_all_limb_consumers(
            src, dispatch_device, dispatch_stream, copied, true, true);
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
        status = matrix_wait_all_limb_streams(src, dispatch_device, dispatch_stream, false, true);
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

    int launch_matrix_equal_for_limb(
        const GpuMatrix *lhs,
        const GpuMatrix *rhs,
        size_t count,
        size_t n,
        const dim3 &limb_id,
        EqualityWorkspace &workspace,
        bool &is_equal)
    {
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
        status = matrix_wait_limb_stream(lhs, limb_id, lhs_device, stream, false, true);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(rhs, limb_id, lhs_device, stream, false, true);
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

        const auto *lhs_base = matrix_limb_ptr_by_id(lhs, 0, limb_id);
        const auto *rhs_base = matrix_limb_ptr_by_id(rhs, 0, limb_id);
        // Validate the last polynomial too, without preparing pointer arrays.
        if (!lhs_base || !rhs_base ||
            !matrix_limb_ptr_by_id(lhs, count - 1, limb_id) ||
            !matrix_limb_ptr_by_id(rhs, count - 1, limb_id))
            return set_error("invalid matrix extent in launch_matrix_equal_for_limb");
        if (count > SIZE_MAX / n || count * n > static_cast<size_t>(INT_MAX) * 256)
            return set_error("equality launch size overflow");
        if (!workspace.storage.data)
        {
            status = workspace.storage.acquire(
                lhs->ctx, lhs_device, GPU_PREPARED_TRANSFER_WORKSPACE,
                sizeof(int), alignof(int), stream);
            if (status != 0) return status;
        }
        workspace.last_stream = stream;
        auto *device_equal = reinterpret_cast<int *>(workspace.storage.data);
        int host_equal = 1;
        err = cudaMemcpyAsync(device_equal, &host_equal, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) return set_error(err);
        const size_t total = count * n;
        block_equal_kernel<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(
            lhs_base, rhs_base, lhs_stride_bytes, rhs_stride_bytes,
            lhs_coeff_bytes, rhs_coeff_bytes, count, n, device_equal);
        err = cudaGetLastError();
        if (err != cudaSuccess) return set_error(err);
        err = cudaMemcpyAsync(&host_equal, device_equal, sizeof(int), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) return set_error(err);
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(lhs->ctx->execution.get());
            return set_error(err);
        }
        is_equal = host_equal != 0;
        return 0;
    }

    enum class PreparedArithmeticKind
    {
        Copy = 0,
        Add = 1,
        Tensor = 2,
        TensorSumRows = 3,
        Multiply = 4,
        Subtract = 5,
        Negate = 6,
        Scale = 7,
        Automorphism = 8,
    };

    int query_arithmetic_layout(
        size_t ring_dimension, size_t limb_count, size_t left_rows,
        size_t left_columns, size_t right_rows, size_t right_columns,
        size_t output_rows, size_t output_columns, size_t column_start,
        size_t group_count, size_t term_count, int kind, int device,
        int evaluation_format, int thin, int lazy_reduction,
        GpuPreparedArithmeticLayout &out)
    {
        const auto checked_ceil_div_u32 = [](size_t value, size_t divisor,
                                               unsigned int *result) {
            if (!result || divisor == 0 ||
                value > std::numeric_limits<size_t>::max() - (divisor - 1))
                return false;
            const size_t quotient = (value + divisor - 1) / divisor;
            if (quotient > std::numeric_limits<unsigned int>::max()) return false;
            *result = static_cast<unsigned int>(quotient);
            return true;
        };
        if (ring_dimension < 2 || (ring_dimension & (ring_dimension - 1)) != 0 ||
            limb_count == 0 || limb_count > kArithMetadataLimbs || kind < 0 || kind > 8 ||
            output_rows == 0 || output_columns == 0)
            return set_error("invalid prepared arithmetic layout request");
        if (left_rows == 0 || left_columns == 0 || right_rows == 0 || right_columns == 0)
            return set_error("prepared arithmetic layout has an empty operand");
        if ((kind == static_cast<int>(PreparedArithmeticKind::Tensor) ||
             kind == static_cast<int>(PreparedArithmeticKind::TensorSumRows) ||
             kind == static_cast<int>(PreparedArithmeticKind::Multiply)) && !evaluation_format)
            return set_error("prepared arithmetic product requires evaluation format");
        const size_t cells = output_rows > std::numeric_limits<size_t>::max() / output_columns
            ? 0 : output_rows * output_columns;
        if (cells == 0 || cells > std::numeric_limits<size_t>::max() / ring_dimension)
            return set_error("prepared arithmetic layout size overflow");
        const size_t coefficients = cells * ring_dimension;
        GpuPreparedArithmeticLayout layout{};
        layout.kind = kind;
        layout.device = device;
        layout.ring_dimension = ring_dimension;
        layout.limb_count = limb_count;
        layout.left_rows = left_rows;
        layout.left_columns = left_columns;
        layout.right_rows = right_rows;
        layout.right_columns = right_columns;
        layout.output_rows = output_rows;
        layout.output_columns = output_columns;
        layout.column_start = column_start;
        layout.group_count = group_count;
        layout.term_count = term_count;
        layout.workspace_bytes = 0;
        layout.alignment = alignof(uint64_t);
        layout.event_count = limb_count;
        layout.block_x = 256;
        layout.block_y = 1;
        layout.block_z = 1;
        layout.grid_z = static_cast<unsigned int>(limb_count);
        layout.thin = thin != 0;
        layout.lazy_reduction = lazy_reduction != 0;
        if (kind == static_cast<int>(PreparedArithmeticKind::Multiply))
        {
            if (left_rows <= 4 && right_columns <= 4 && left_columns <= 16)
            {
                if (!checked_ceil_div_u32(ring_dimension, 256, &layout.grid_x) ||
                    right_columns > std::numeric_limits<unsigned int>::max())
                    return set_error("prepared arithmetic grid dimensions overflow");
                layout.grid_y = static_cast<unsigned int>(right_columns);
            }
            else if (thin)
            {
                layout.block_x = kThinMatmulWarpSize;
                layout.block_y = kThinMatmulColumnsPerBlock;
                if (!checked_ceil_div_u32(right_columns, kThinMatmulColumnsPerBlock, &layout.grid_x))
                    return set_error("prepared arithmetic grid dimensions overflow");
                layout.grid_y = 1;
                if (ring_dimension > std::numeric_limits<size_t>::max() -
                        (kThinMatmulWarpSize - 1))
                    return set_error("prepared arithmetic grid dimensions overflow");
                const size_t grid_z = (ring_dimension + kThinMatmulWarpSize - 1) /
                    kThinMatmulWarpSize;
                layout.grid_z = static_cast<unsigned int>(std::min(
                    grid_z, static_cast<size_t>(kMatmulMaxGridZ)));
            }
            else
            {
                layout.block_x = kMatmulTileN;
                layout.block_y = kMatmulTileM;
                if (!checked_ceil_div_u32(right_columns, kMatmulTileN, &layout.grid_x) ||
                    !checked_ceil_div_u32(left_rows, kMatmulTileM, &layout.grid_y))
                    return set_error("prepared arithmetic grid dimensions overflow");
                layout.grid_z = static_cast<unsigned int>(std::min(ring_dimension, static_cast<size_t>(kMatmulMaxGridZ)));
            }
        }
        else if (kind == static_cast<int>(PreparedArithmeticKind::TensorSumRows) && ring_dimension >= 256)
        {
            if (!checked_ceil_div_u32(ring_dimension, 256, &layout.grid_x))
                return set_error("prepared arithmetic grid dimensions overflow");
            layout.grid_y = static_cast<unsigned int>(std::min(cells, size_t{65535}));
        }
        else
        {
            if (!checked_ceil_div_u32(coefficients, 256, &layout.grid_x))
                return set_error("prepared arithmetic grid dimensions overflow");
            layout.grid_y = 1;
        }
        out = layout;
        return 0;
    }

extern "C" int gpu_matrix_query_arithmetic_layout(
    size_t ring_dimension, size_t limb_count, size_t left_rows,
    size_t left_columns, size_t right_rows, size_t right_columns,
    size_t output_rows, size_t output_columns, size_t column_start,
    size_t group_count, size_t term_count, int kind, int device,
    int evaluation_format, int thin, int lazy_reduction,
    GpuPreparedArithmeticLayout *out)
{
    if (!out) return set_error("null prepared arithmetic layout output");
    return query_arithmetic_layout(
        ring_dimension, limb_count, left_rows, left_columns, right_rows,
        right_columns, output_rows, output_columns, column_start,
        group_count, term_count, kind, device, evaluation_format, thin,
        lazy_reduction, *out);
}

static int query_rect_layout(
    size_t ring_dimension, size_t limb_count, size_t rows, size_t columns,
    int device, int stage_role, GpuPreparedRectLayout &out)
{
    if (ring_dimension == 0 || limb_count == 0 || rows == 0 || columns == 0)
        return set_error("invalid prepared rectangular layout request");
    if (rows > std::numeric_limits<size_t>::max() / columns ||
        rows * columns > std::numeric_limits<size_t>::max() / ring_dimension)
        return set_error("prepared rectangular layout size overflow");
    const size_t coefficients = rows * columns * ring_dimension;
    if (coefficients > std::numeric_limits<size_t>::max() - 255)
        return set_error("prepared rectangular layout grid overflow");
    const size_t quotient = (coefficients + 255) / 256;
    const size_t grid_x = stage_role == 2 ? std::min(quotient, size_t{65535}) : quotient;
    if (grid_x > std::numeric_limits<unsigned int>::max())
        return set_error("prepared rectangular layout grid overflow");
    out = GpuPreparedRectLayout{
        rows, columns, ring_dimension, limb_count, 0, alignof(uint64_t), limb_count,
        static_cast<unsigned int>(grid_x), 1,
        static_cast<unsigned int>(limb_count), 256, 1, 1, stage_role, device,
    };
    return 0;
}

extern "C" int gpu_matrix_query_input_copy_layout(
    size_t ring_dimension, size_t limb_count, size_t rows, size_t columns,
    int device, GpuPreparedRectLayout *out)
{
    if (!out) return set_error("null prepared input-copy layout output");
    return query_rect_layout(ring_dimension, limb_count, rows, columns, device, 1, *out);
}

extern "C" int gpu_matrix_query_transpose_layout(
    size_t ring_dimension, size_t limb_count, size_t rows, size_t columns,
    int device, GpuPreparedRectLayout *out)
{
    if (!out) return set_error("null prepared transpose layout output");
    return query_rect_layout(ring_dimension, limb_count, rows, columns, device, 2, *out);
}

    struct PreparedMatmulLaunch
    {
        const uint8_t *lhs_base;
        const uint8_t *rhs_base;
        uint8_t *out_base;
        size_t lhs_stride;
        size_t rhs_stride;
        size_t out_stride;
        uint8_t lhs_width;
        uint8_t rhs_width;
        uint8_t out_width;
        uint64_t modulus;
        uint64_t reciprocal;
        size_t rows;
        size_t inner;
        size_t columns;
        size_t n;
        size_t lhs_pitch;
        size_t rhs_pitch;
        size_t out_pitch;
        size_t lhs_row;
        size_t lhs_column;
        size_t rhs_row;
        size_t rhs_column;
        size_t out_row;
        size_t out_column;
        bool thin;
        bool lazy_reduction;
        dim3 grid;
        dim3 block;
    };

    struct GpuPreparedArithmeticState
    {
        PreparedArithmeticKind kind;
        GpuMatrix *out;
        const GpuMatrix *lhs;
        const GpuMatrix *rhs;
        GpuMatrixRange left_range;
        GpuMatrixRange right_range;
        GpuMatrixRange output_range;
        size_t column_start;
        size_t left_columns;
        size_t right_rows;
        size_t right_columns;
        size_t output_count;
        size_t n;
        size_t limb_count;
        int device;
        cudaStream_t stream;
        dim3 grid;
        BlockCopyMetadata copy;
        BlockElementwiseMetadata elementwise;
        DescriptorProductMetadata product;
        TensorGeometry geometry;
        TensorRowSumMetadata row_sum;
        bool row_sum_separate_polynomials;
        PreparedMatmulLaunch matmul[kArithMetadataLimbs];
        size_t matmul_count;
        uint64_t scalars[kArithMetadataLimbs];
        size_t automorphism_index;
        bool evaluation;
        unsigned log_n;
        // Prepared resources are acquired in descriptor order: the optional
        // workspace first, then one completion event per active limb.
        GpuDeviceWorkspace workspace;
        std::array<std::unique_ptr<GpuCudaResource>, kArithMetadataLimbs> completion;

        ~GpuPreparedArithmeticState()
        {
            (void)workspace.release(stream);
            for (auto &event : completion)
                if (event) (void)event->release();
        }
    };

    struct GpuPreparedInputCopyState
    {
        GpuMatrix *out;
        GpuMatrixRange input_range;
        GpuMatrixRange output_range;
        BlockCopyMetadata source_layout;
        size_t source_rows;
        size_t source_columns;
        size_t level;
        size_t n;
        int format;
        int device;
        cudaStream_t stream;
        dim3 grid;
        std::array<std::unique_ptr<GpuCudaResource>, kArithMetadataLimbs> completion;

        ~GpuPreparedInputCopyState()
        {
            for (auto &event : completion)
                if (event) (void)event->release();
        }
    };

    int prepared_matrix_metadata(
        const GpuMatrix *matrix, const dim3 &id, const uint8_t **base,
        size_t *stride, uint8_t *width, int *device);

    int prepare_copy_layout(
        GpuMatrix *out, const GpuMatrix *source,
        const GpuMatrixBatchView *view, GpuPreparedInputCopyState &prepared,
        const GpuPreparedLaunchLayout *saved_launch = nullptr)
    {
        if (!out || !source || !out->ctx || source->ctx != out->ctx ||
            source->level < 0 || out->level != source->level)
            return set_error("prepared input copy owners are incompatible");
        prepared.input_range = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
        prepared.output_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
        const auto valid = [](const GpuMatrix *matrix, const GpuMatrixRange &range) {
            return range.row_start <= range.row_end && range.row_end <= matrix->rows &&
                range.column_start <= range.column_end && range.column_end <= matrix->cols;
        };
        if (!valid(source, prepared.input_range) ||
            (view && (!valid(source, view->right) || std::memcmp(&view->left, &view->right,
                sizeof(GpuMatrixRange)) != 0)) ||
            !valid(out, prepared.output_range) ||
            prepared.input_range.row_end - prepared.input_range.row_start !=
                prepared.output_range.row_end - prepared.output_range.row_start ||
            prepared.input_range.column_end - prepared.input_range.column_start !=
                prepared.output_range.column_end - prepared.output_range.column_start)
            return set_error("prepared input copy ranges are incompatible");
        prepared.out = out;
        prepared.source_rows = source->rows;
        prepared.source_columns = source->cols;
        prepared.level = static_cast<size_t>(source->level);
        prepared.n = static_cast<size_t>(source->ctx->N);
        prepared.format = source->format;
        prepared.device = -1;
        int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &prepared.stream);
        if (status != 0) return status;
        status = matrix_limb_device(out, out->ctx->limb_gpu_ids[0], &prepared.device);
        if (status != 0) return status;
        prepared.source_layout = {};
        for (size_t limb = 0; limb <= prepared.level; ++limb)
        {
            const dim3 id = out->ctx->limb_gpu_ids[limb];
            int source_device = -1, output_device = -1;
            const uint8_t *source_base = nullptr, *output_base = nullptr;
            size_t source_stride = 0, output_stride = 0;
            uint8_t source_width = 0, output_width = 0;
            status = prepared_matrix_metadata(source, id, &source_base, &source_stride, &source_width, &source_device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(out, id, &output_base, &output_stride, &output_width, &output_device);
            if (status != 0) return status;
            if (source_device != prepared.device || output_device != prepared.device)
                return set_error("prepared input copy owners use different devices");
            prepared.source_layout.src_bases[limb] = source_base;
            prepared.source_layout.src_stride_bytes[limb] = source_stride;
            prepared.source_layout.src_coeff_bytes[limb] = source_width;
            prepared.source_layout.dst_bases[limb] = const_cast<uint8_t *>(output_base);
            prepared.source_layout.dst_stride_bytes[limb] = output_stride;
            prepared.source_layout.dst_coeff_bytes[limb] = output_width;
        }
        const size_t rows = prepared.output_range.row_end - prepared.output_range.row_start;
        const size_t columns = prepared.output_range.column_end - prepared.output_range.column_start;
        if (saved_launch)
        {
            prepared.grid = saved_launch->grid;
        }
        else
        {
            GpuPreparedRectLayout layout{};
            status = query_rect_layout(prepared.n, prepared.level + 1, rows, columns,
                prepared.device, 1, layout);
            if (status != 0) return status;
            prepared.grid = dim3(layout.grid_x, layout.grid_y, layout.grid_z);
        }
        return 0;
    }

    std::atomic<size_t> input_copy_prepared_acquisitions{0};

    int validate_saved_input_copy_descriptor(
        GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
        const GpuPreparedPlanDescriptor *layout, GpuPreparedInputCopyState &prepared)
    {
        if (!layout || !out || !source || !out->ctx || source->level < 0 ||
            source->ctx != out->ctx ||
            (source->format != GPU_POLY_FORMAT_COEFF && source->format != GPU_POLY_FORMAT_EVAL) ||
            (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL))
            return set_error("invalid saved input copy descriptor owners");
        if (layout->allocation_count != static_cast<size_t>(source->level + 1) ||
            layout->stream_count != 1 || layout->launch_count != 1)
            return set_error("saved input copy descriptor shape is invalid");
        const auto &launch = layout->launches[0];
        const size_t n = static_cast<size_t>(out->ctx->N);
        const size_t limbs = static_cast<size_t>(source->level + 1);
        if (!out->ctx->execution || out->level != source->level ||
            out->ctx->limb_gpu_ids.size() < limbs)
            return set_error("saved input copy descriptor owner metadata is invalid");
        const GpuMatrixRange input = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
        const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
        if (!n || !limbs || limbs > kArithMetadataLimbs ||
            input.row_end < input.row_start || input.column_end < input.column_start ||
            output.row_end < output.row_start || output.column_end < output.column_start ||
            input.row_end - input.row_start != output.row_end - output.row_start ||
            input.column_end - input.column_start != output.column_end - output.column_start)
            return set_error("saved input copy descriptor geometry is invalid");
        const size_t rows = output.row_end - output.row_start;
        const size_t columns = output.column_end - output.column_start;
        if (!rows || !columns || rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / n)
            return set_error("saved input copy descriptor dimensions overflow");
        const size_t coefficients = rows * columns * n;
        if (coefficients > SIZE_MAX - 255)
            return set_error("saved input copy descriptor grid overflow");
        const size_t expected_grid_x = (coefficients + 255) / 256;
        if (expected_grid_x > std::numeric_limits<unsigned int>::max() ||
            launch.phase != 0 || launch.len != n || launch.limb_offset != 0 ||
            launch.limb_count != limbs || launch.narrow != 0 ||
            launch.grid.x != expected_grid_x || launch.grid.y != 1 ||
            launch.grid.z != limbs || launch.block.x != 256 ||
            launch.block.y != 1 || launch.block.z != 1)
            return set_error("saved input copy launch geometry differs from descriptor");
        const int status = prepare_copy_layout(out, source, view, prepared, &launch);
        if (status != 0) return status;
        dim3 first{};
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            dim3 id{};
            GpuPreparedResourceKey key{};
            if (gpu_prepared_limb_key(out->ctx, source->level, limb,
                    GPU_PREPARED_STAGE_TRANSFORM, &id, &key) != 0 ||
                gpu_prepared_require_allocation(layout, limb,
                    GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0)
                return set_error("saved input copy completion event differs from descriptor");
            const auto &entry = layout->allocations[limb];
            if (entry.rows != 0 || entry.columns != 0 || entry.level != -1 || entry.format != -1)
                return set_error("saved input copy completion event metadata is invalid");
            if (limb == 0) first = id;
        }
        const auto &stream_entry = layout->streams[0];
        GpuPreparedResourceKey stream_key{};
        if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
            gpu_prepared_limb_key(out->ctx, source->level, 0, GPU_PREPARED_STAGE_TRANSFORM,
                &first, &stream_key) != 0 || std::memcmp(&stream_entry.key, &stream_key, sizeof(stream_key)) != 0 ||
            gpu_prepared_require_stream_slot(out->ctx, first.x, prepared.stream,
                stream_entry.pool_slot) != 0)
        {
            return set_error("saved input copy stream differs from descriptor");
        }
        return 0;
    }

    int construct_saved_input_copy_resources(
        GpuPreparedInputCopyState &prepared, const GpuPreparedPlanDescriptor *layout)
    {
        for (size_t limb = 0; limb <= prepared.level; ++limb)
        {
            const auto &entry = layout->allocations[limb];
            auto event = std::make_unique<GpuCudaResource>();
            const int status = event->acquire(
                prepared.out->ctx, entry.key.device, GPU_PREPARED_COMPLETION_EVENT);
            if (status != 0) return status;
            prepared.completion[limb] = std::move(event);
            input_copy_prepared_acquisitions.fetch_add(1, std::memory_order_relaxed);
        }
        return 0;
    }

    int prepared_matrix_partition(
        const GpuMatrix *matrix, const dim3 &id, int *device,
        GpuMatrix::SharedLimbBuffer::DeviceDescriptor **descriptors)
    {
        if (id.x >= matrix->shared_limb_buffers.size())
            return set_error("prepared arithmetic partition is unavailable");
        const auto &buffer = matrix->shared_limb_buffers[id.x];
        if (!buffer.device_descriptors || id.y >= buffer.limb_count)
            return set_error("prepared arithmetic descriptors are unavailable");
        if (*device < 0) *device = buffer.device;
        if (*device != buffer.device)
            return set_error("prepared arithmetic requires one device");
        if (*descriptors && *descriptors != buffer.device_descriptors)
            return set_error("prepared arithmetic descriptors span partitions");
        *descriptors = buffer.device_descriptors;
        return 0;
    }

    int prepared_matrix_metadata(
        const GpuMatrix *matrix, const dim3 &id, const uint8_t **base,
        size_t *stride, uint8_t *width, int *device)
    {
        *base = matrix_limb_ptr_by_id(matrix, 0, id);
        if (!*base || !matrix_limb_metadata_by_id(matrix, id, stride, width))
            return set_error("prepared arithmetic matrix metadata is unavailable");
        return matrix_limb_device(matrix, id, device);
    }

    int prepare_product_metadata(
        GpuPreparedArithmeticState &prepared, const GpuMatrix *lhs,
        const GpuMatrix *rhs, GpuMatrix *out)
    {
        prepared.product = {};
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            const dim3 id = lhs->ctx->limb_gpu_ids[limb];
            int device = -1;
            GpuMatrix::SharedLimbBuffer::DeviceDescriptor *lhs_descriptors = nullptr;
            GpuMatrix::SharedLimbBuffer::DeviceDescriptor *rhs_descriptors = nullptr;
            GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out_descriptors = nullptr;
            int status = prepared_matrix_partition(lhs, id, &device, &lhs_descriptors);
            if (status != 0) return status;
            status = prepared_matrix_partition(rhs, id, &device, &rhs_descriptors);
            if (status != 0) return status;
            status = prepared_matrix_partition(out, id, &device, &out_descriptors);
            if (status != 0) return status;
            if (limb == 0)
            {
                prepared.device = device;
                prepared.product.lhs = lhs_descriptors;
                prepared.product.rhs = rhs_descriptors;
                prepared.product.out = out_descriptors;
            }
            else if (prepared.product.lhs != lhs_descriptors ||
                     prepared.product.rhs != rhs_descriptors ||
                     prepared.product.out != out_descriptors)
                return set_error("prepared arithmetic descriptors changed by limb");
            prepared.product.indices[limb] = id.y;
            prepared.product.moduli[limb] = lhs->ctx->moduli[limb];
        }
        return 0;
    }

    int prepare_block_metadata(
        GpuPreparedArithmeticState &prepared, const GpuMatrix *lhs,
        const GpuMatrix *rhs, GpuMatrix *out)
    {
        prepared.elementwise = {};
        prepared.copy = {};
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            const dim3 id = lhs->ctx->limb_gpu_ids[limb];
            const uint8_t *lhs_base = nullptr;
            const uint8_t *rhs_base = nullptr;
            const uint8_t *src_base = nullptr;
            const uint8_t *dst_base = nullptr;
            uint8_t *out_base = nullptr;
            size_t lhs_stride = 0, rhs_stride = 0, out_stride = 0;
            size_t src_stride = 0, dst_stride = 0;
            uint8_t lhs_width = 0, rhs_width = 0, out_width = 0;
            uint8_t src_width = 0, dst_width = 0;
            int lhs_device = -1, rhs_device = -1, out_device = -1;
            int status = prepared_matrix_metadata(lhs, id, &lhs_base, &lhs_stride, &lhs_width, &lhs_device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(rhs, id, &rhs_base, &rhs_stride, &rhs_width, &rhs_device);
            if (status != 0) return status;
            const uint8_t *out_const = nullptr;
            status = prepared_matrix_metadata(out, id, &out_const, &out_stride, &out_width, &out_device);
            if (status != 0) return status;
            out_base = const_cast<uint8_t *>(out_const);
            if (lhs_device != rhs_device || lhs_device != out_device)
                return set_error("prepared arithmetic owners use different devices");
            if (limb == 0) prepared.device = out_device;
            if (prepared.device != out_device) return set_error("prepared arithmetic uses multiple devices");
            prepared.elementwise.lhs_bases[limb] = lhs_base;
            prepared.elementwise.rhs_bases[limb] = rhs_base;
            prepared.elementwise.out_bases[limb] = out_base;
            prepared.elementwise.lhs_stride_bytes[limb] = lhs_stride;
            prepared.elementwise.rhs_stride_bytes[limb] = rhs_stride;
            prepared.elementwise.out_stride_bytes[limb] = out_stride;
            prepared.elementwise.lhs_coeff_bytes[limb] = lhs_width;
            prepared.elementwise.rhs_coeff_bytes[limb] = rhs_width;
            prepared.elementwise.out_coeff_bytes[limb] = out_width;
            prepared.elementwise.moduli[limb] = lhs->ctx->moduli[limb];
            status = prepared_matrix_metadata(lhs, id, &src_base, &src_stride, &src_width, &lhs_device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(out, id, &dst_base, &dst_stride, &dst_width, &out_device);
            if (status != 0) return status;
            prepared.copy.src_bases[limb] = src_base;
            prepared.copy.dst_bases[limb] = const_cast<uint8_t *>(dst_base);
            prepared.copy.src_stride_bytes[limb] = src_stride;
            prepared.copy.dst_stride_bytes[limb] = dst_stride;
            prepared.copy.src_coeff_bytes[limb] = src_width;
            prepared.copy.dst_coeff_bytes[limb] = dst_width;
        }
        return 0;
    }

    int prepare_unary_metadata(
        GpuPreparedArithmeticState &prepared, const GpuMatrix *lhs, GpuMatrix *out)
    {
        prepared.elementwise = {};
        prepared.copy = {};
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            const dim3 id = lhs->ctx->limb_gpu_ids[limb];
            const uint8_t *lhs_base = nullptr;
            const uint8_t *out_base = nullptr;
            size_t lhs_stride = 0, out_stride = 0;
            uint8_t lhs_width = 0, out_width = 0;
            int lhs_device = -1, out_device = -1;
            int status = prepared_matrix_metadata(
                lhs, id, &lhs_base, &lhs_stride, &lhs_width, &lhs_device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(
                out, id, &out_base, &out_stride, &out_width, &out_device);
            if (status != 0) return status;
            if (lhs_device != out_device)
                return set_error("prepared unary owners use different devices");
            if (limb == 0) prepared.device = out_device;
            if (prepared.device != out_device)
                return set_error("prepared unary uses multiple devices");
            prepared.elementwise.lhs_bases[limb] = lhs_base;
            prepared.elementwise.out_bases[limb] = const_cast<uint8_t *>(out_base);
            prepared.elementwise.lhs_stride_bytes[limb] = lhs_stride;
            prepared.elementwise.out_stride_bytes[limb] = out_stride;
            prepared.elementwise.lhs_coeff_bytes[limb] = lhs_width;
            prepared.elementwise.out_coeff_bytes[limb] = out_width;
            prepared.elementwise.moduli[limb] = lhs->ctx->moduli[limb];
        }
        return 0;
    }

    int configure_arithmetic_plan(
        GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
        int kind, const size_t *rows, const size_t *offsets,
        size_t group_count, size_t term_count, const GpuMatrixBatchView *view,
        size_t column_start, const uint64_t *scalar_residues, size_t scalar_count,
        size_t automorphism_index, GpuPreparedArithmeticState &prepared,
        const GpuPreparedArithmeticLayout *saved_geometry = nullptr)
    {
        if (!out || !lhs || !lhs->ctx || !out->ctx || out->ctx != lhs->ctx ||
            out->level != lhs->level || lhs->level < 0 || kind < 0 || kind > 8)
            return set_error("invalid prepared arithmetic owners");
        const bool unary = kind == static_cast<int>(PreparedArithmeticKind::Negate) ||
            kind == static_cast<int>(PreparedArithmeticKind::Scale) ||
            kind == static_cast<int>(PreparedArithmeticKind::Automorphism);
        if (!unary && kind != static_cast<int>(PreparedArithmeticKind::Copy) && !rhs)
            return set_error("prepared arithmetic requires a right owner");
        if (rhs && (rhs->ctx != lhs->ctx || rhs->level != lhs->level))
            return set_error("prepared arithmetic context mismatch");
        prepared.kind = static_cast<PreparedArithmeticKind>(kind);
        prepared.out = out;
        prepared.lhs = lhs;
        prepared.rhs = rhs;
        prepared.limb_count = static_cast<size_t>(lhs->level) + 1;
        prepared.n = static_cast<size_t>(lhs->ctx->N);
        if (!prepared.n || prepared.limb_count > kArithMetadataLimbs)
            return set_error("prepared arithmetic basis is unavailable");
        prepared.left_range = view ? view->left : GpuMatrixRange{0, lhs->rows, 0, lhs->cols};
        prepared.right_range = view && rhs ? view->right : GpuMatrixRange{0, rhs ? rhs->rows : lhs->rows, 0, rhs ? rhs->cols : lhs->cols};
        prepared.output_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
        prepared.column_start = column_start;
        const auto valid = [](const GpuMatrix *matrix, const GpuMatrixRange &range) {
            return range.row_start <= range.row_end && range.row_end <= matrix->rows &&
                range.column_start <= range.column_end && range.column_end <= matrix->cols;
        };
        if (!valid(lhs, prepared.left_range) || !valid(out, prepared.output_range) ||
            (rhs && !valid(rhs, prepared.right_range)))
            return set_error("prepared arithmetic range is invalid");
        prepared.left_columns = prepared.left_range.column_end - prepared.left_range.column_start;
        prepared.right_rows = prepared.right_range.row_end - prepared.right_range.row_start;
        prepared.right_columns = prepared.right_range.column_end - prepared.right_range.column_start;
        const size_t output_rows = prepared.output_range.row_end - prepared.output_range.row_start;
        const size_t output_columns = prepared.output_range.column_end - prepared.output_range.column_start;
        prepared.output_count = output_rows * output_columns;
        int status = matrix_limb_stream(out, lhs->ctx->limb_gpu_ids[0], &prepared.stream);
        if (status != 0) return status;
        prepared.device = -1;
        status = matrix_limb_device(out, lhs->ctx->limb_gpu_ids[0], &prepared.device);
        if (status != 0) return status;
        GpuPreparedArithmeticLayout structural{};
        if (saved_geometry)
            structural = *saved_geometry;
        else
        {
            status = query_arithmetic_layout(
                prepared.n, prepared.limb_count,
                prepared.left_range.row_end - prepared.left_range.row_start,
                prepared.left_columns, prepared.right_rows, prepared.right_columns,
                output_rows, output_columns, column_start, group_count, term_count,
                kind, prepared.device, lhs->format == GPU_POLY_FORMAT_EVAL, 0, 0,
                structural);
            if (status != 0) return status;
        }
        if (kind == static_cast<int>(PreparedArithmeticKind::Copy))
        {
            if (lhs->format != out->format || output_rows != prepared.left_range.row_end - prepared.left_range.row_start ||
                output_columns != prepared.left_columns)
                return set_error("prepared copy shape or format mismatch");
            status = prepare_block_metadata(prepared, lhs, lhs, out);
            if (status != 0) return status;
            prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            return 0;
        }
        if (kind == static_cast<int>(PreparedArithmeticKind::Add))
        {
            if (prepared.left_range.row_end - prepared.left_range.row_start !=
                    prepared.right_range.row_end - prepared.right_range.row_start ||
                prepared.left_columns != prepared.right_columns ||
                output_rows != prepared.left_range.row_end - prepared.left_range.row_start ||
                output_columns != prepared.left_columns ||
                lhs->format != rhs->format || lhs->format != out->format)
                return set_error("prepared add shape or format mismatch");
            status = prepare_block_metadata(prepared, lhs, rhs, out);
            if (status != 0) return status;
            prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            return 0;
        }
        if (kind == static_cast<int>(PreparedArithmeticKind::Subtract))
        {
            if (prepared.left_range.row_end - prepared.left_range.row_start !=
                    prepared.right_range.row_end - prepared.right_range.row_start ||
                prepared.left_columns != prepared.right_columns ||
                output_rows != prepared.left_range.row_end - prepared.left_range.row_start ||
                output_columns != prepared.left_columns ||
                lhs->format != rhs->format || lhs->format != out->format)
                return set_error("prepared subtract shape or format mismatch");
            status = prepare_block_metadata(prepared, lhs, rhs, out);
            if (status != 0) return status;
            prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            return 0;
        }
        if (unary)
        {
            if (output_rows != prepared.left_range.row_end - prepared.left_range.row_start ||
                output_columns != prepared.left_columns || lhs->format != out->format)
                return set_error("prepared unary shape or format mismatch");
            if (kind == static_cast<int>(PreparedArithmeticKind::Scale))
            {
                if (!scalar_residues || scalar_count != prepared.limb_count)
                    return set_error("prepared scale requires one residue per limb");
                std::copy_n(scalar_residues, scalar_count, prepared.scalars);
            }
            if (kind == static_cast<int>(PreparedArithmeticKind::Automorphism))
            {
                if (automorphism_index == 0 || automorphism_index >= 2 * prepared.n ||
                    (automorphism_index & 1) == 0 || (prepared.n & (prepared.n - 1)) != 0)
                    return set_error("prepared automorphism index is invalid");
                prepared.automorphism_index = automorphism_index;
                prepared.evaluation = lhs->format == GPU_POLY_FORMAT_EVAL;
                prepared.log_n = static_cast<unsigned>(__builtin_ctzll(prepared.n));
            }
            status = prepare_unary_metadata(prepared, lhs, out);
            if (status != 0) return status;
            prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            return 0;
        }
        if (lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL ||
            out->format != GPU_POLY_FORMAT_EVAL)
            return set_error("prepared product requires Eval format");
        status = prepare_product_metadata(prepared, lhs, rhs, out);
        if (status != 0) return status;
        prepared.geometry = TensorGeometry{
            prepared.left_range.row_start * lhs->cols + prepared.left_range.column_start, lhs->cols,
            prepared.right_range.row_start * rhs->cols + prepared.right_range.column_start, rhs->cols,
            prepared.output_range.row_start * out->cols + prepared.output_range.column_start, out->cols,
            column_start, output_columns};
        if (kind == static_cast<int>(PreparedArithmeticKind::Tensor) ||
            kind == static_cast<int>(PreparedArithmeticKind::TensorSumRows))
        {
            const bool grouped = kind == static_cast<int>(PreparedArithmeticKind::TensorSumRows);
            if (prepared.right_rows == 0 || prepared.left_columns == 0 || prepared.right_columns == 0 ||
                (grouped && (!rows || !offsets || group_count == 0 || group_count > 16 || term_count == 0 || term_count > 32)))
                return set_error("prepared tensor shape or grouping is invalid");
            const size_t product_rows = (prepared.left_range.row_end - prepared.left_range.row_start) * prepared.right_rows;
            if ((!grouped && output_rows != product_rows) || output_columns > prepared.left_columns * prepared.right_columns ||
                column_start > prepared.left_columns * prepared.right_columns - output_columns)
                return set_error("prepared tensor output shape is invalid");
            if (grouped)
            {
                if (output_rows != group_count || offsets[0] != 0 || offsets[group_count] != term_count)
                    return set_error("prepared tensor row groups are invalid");
                prepared.row_sum = {};
                prepared.row_sum.product = prepared.product;
                prepared.row_sum.geometry = prepared.geometry;
                std::copy_n(rows, term_count, prepared.row_sum.rows);
                std::copy_n(offsets, group_count + 1, prepared.row_sum.offsets);
                std::copy_n(lhs->ctx->barrett_reciprocals.data(), prepared.limb_count, prepared.row_sum.reciprocals);
                for (size_t term = 0; term < term_count; ++term)
                    if (rows[term] >= product_rows) return set_error("prepared tensor row index is invalid");
                prepared.row_sum_separate_polynomials = prepared.n >= 256;
                prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            }
            else
            {
                prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            }
            return 0;
        }
        if (kind != static_cast<int>(PreparedArithmeticKind::Multiply))
            return set_error("unknown prepared arithmetic kind");
        const size_t rows_count = prepared.left_range.row_end - prepared.left_range.row_start;
        const size_t inner = prepared.left_columns;
        const size_t cols_count = prepared.right_columns;
        if (prepared.right_rows != inner || output_rows != rows_count || output_columns != cols_count || !inner)
            return set_error("prepared multiply shape is invalid");
        if (rows_count <= 4 && cols_count <= 4 && inner <= 16)
        {
            prepared.grid = dim3(structural.grid_x, structural.grid_y, structural.grid_z);
            return 0;
        }
        prepared.matmul_count = prepared.limb_count;
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            auto &launch = prepared.matmul[limb];
            const dim3 id = lhs->ctx->limb_gpu_ids[limb];
            int device = prepared.device;
            const uint8_t *lhs_base = nullptr, *rhs_base = nullptr, *out_base_const = nullptr;
            status = prepared_matrix_metadata(lhs, id, &lhs_base, &launch.lhs_stride, &launch.lhs_width, &device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(rhs, id, &rhs_base, &launch.rhs_stride, &launch.rhs_width, &device);
            if (status != 0) return status;
            status = prepared_matrix_metadata(out, id, &out_base_const, &launch.out_stride, &launch.out_width, &device);
            if (status != 0) return status;
            launch.lhs_base = lhs_base;
            launch.rhs_base = rhs_base;
            launch.out_base = const_cast<uint8_t *>(out_base_const);
            launch.rows = rows_count;
            launch.inner = inner;
            launch.columns = cols_count;
            launch.n = prepared.n;
            launch.lhs_pitch = lhs->cols;
            launch.rhs_pitch = rhs->cols;
            launch.out_pitch = out->cols;
            launch.lhs_row = prepared.left_range.row_start;
            launch.lhs_column = prepared.left_range.column_start;
            launch.rhs_row = prepared.right_range.row_start;
            launch.rhs_column = prepared.right_range.column_start;
            launch.out_row = prepared.output_range.row_start;
            launch.out_column = prepared.output_range.column_start;
            launch.thin = view == nullptr && rows_count == 1 &&
                matrix_barrett_u32_reciprocal(lhs->ctx->moduli[limb], &launch.reciprocal);
            launch.lazy_reduction = launch.thin && matrix_lazy_dot_u64(inner, lhs->ctx->moduli[limb]);
            launch.modulus = lhs->ctx->moduli[limb];
            GpuPreparedArithmeticLayout matmul_layout{};
            if (saved_geometry)
                matmul_layout = structural;
            else
            {
                status = query_arithmetic_layout(
                    prepared.n, prepared.limb_count, rows_count, inner,
                    prepared.right_rows, cols_count, output_rows, output_columns,
                    column_start, group_count, term_count, kind, prepared.device,
                    true, launch.thin ? 1 : 0, launch.lazy_reduction ? 1 : 0,
                    matmul_layout);
                if (status != 0) return status;
            }
            launch.block = dim3(matmul_layout.block_x, matmul_layout.block_y, matmul_layout.block_z);
            launch.grid = dim3(matmul_layout.grid_x, matmul_layout.grid_y, matmul_layout.grid_z);
        }
        return 0;
    }

    // Build only the launch record needed by a saved bind.  This deliberately
    // does not call query_arithmetic_layout: that query is the legacy planning
    // path and a saved descriptor must be the source of every prepared
    // resource and geometry decision.  Older descriptors have no launch table
    // for arithmetic, so the compact arithmetic geometry is constructed
    // directly from the already-validated structural arguments.
    int saved_arithmetic_geometry(
        const GpuMatrix *lhs, const GpuMatrix *rhs, const GpuMatrix *out,
        int kind, const GpuMatrixBatchView *view, size_t column_start,
        const GpuPreparedPlanDescriptor *layout, GpuPreparedArithmeticLayout &geometry)
    {
        const GpuMatrix *effective_rhs = rhs ? rhs : lhs;
        if (!lhs || !effective_rhs || !out || !lhs->ctx || lhs->level < 0 ||
            out->ctx != lhs->ctx || effective_rhs->ctx != lhs->ctx || out->level != lhs->level)
            return set_error("invalid saved arithmetic geometry owners");
        const auto valid = [](const GpuMatrix *matrix, const GpuMatrixRange &range) {
            return range.row_start <= range.row_end && range.row_end <= matrix->rows &&
                range.column_start <= range.column_end && range.column_end <= matrix->cols;
        };
        const GpuMatrixRange left = view ? view->left : GpuMatrixRange{0, lhs->rows, 0, lhs->cols};
        const GpuMatrixRange right = view ? view->right : GpuMatrixRange{0, effective_rhs->rows, 0, effective_rhs->cols};
        const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
        if (!valid(lhs, left) || !valid(effective_rhs, right) || !valid(out, output))
            return set_error("saved arithmetic geometry range is invalid");
        const size_t left_rows = left.row_end - left.row_start;
        const size_t left_columns = left.column_end - left.column_start;
        const size_t right_rows = right.row_end - right.row_start;
        const size_t right_columns = right.column_end - right.column_start;
        const size_t output_rows = output.row_end - output.row_start;
        const size_t output_columns = output.column_end - output.column_start;
        const size_t n = static_cast<size_t>(lhs->ctx->N);
        const size_t limbs = static_cast<size_t>(lhs->level) + 1;
        if (n < 2 || !n || !limbs || limbs > kArithMetadataLimbs ||
            lhs->ctx->moduli.size() < limbs ||
            !output_rows || !output_columns || !left_rows || !left_columns ||
            !right_rows || !right_columns || kind < 0 || kind > 8)
            return set_error("saved arithmetic geometry is empty or invalid");
        if (output_rows > SIZE_MAX / output_columns ||
            output_rows * output_columns > SIZE_MAX / n)
            return set_error("saved arithmetic geometry size overflow");
        const size_t cells = output_rows * output_columns;
        const size_t coefficients = cells * n;
        if (right_columns > SIZE_MAX / left_columns)
            return set_error("saved arithmetic geometry column count overflow");
        const size_t product_columns = left_columns * right_columns;
        if (column_start > product_columns || output_columns > product_columns - column_start)
            return set_error("saved arithmetic geometry column interval is invalid");
        geometry = {};
        geometry.kind = kind;
        geometry.device = -1;
        geometry.ring_dimension = n;
        geometry.limb_count = limbs;
        geometry.left_rows = left_rows;
        geometry.left_columns = left_columns;
        geometry.right_rows = right_rows;
        geometry.right_columns = right_columns;
        geometry.output_rows = output_rows;
        geometry.output_columns = output_columns;
        geometry.column_start = column_start;
        geometry.alignment = alignof(uint64_t);
        geometry.event_count = limbs;
        geometry.block_x = 256;
        geometry.block_y = 1;
        geometry.block_z = 1;
        geometry.grid_y = 1;
        geometry.grid_z = static_cast<unsigned int>(limbs);
        if (limbs > std::numeric_limits<unsigned int>::max())
            return set_error("saved arithmetic geometry limb count overflow");

        // A descriptor may carry an explicit launch record.  Consume it
        // verbatim after checking the fields that are meaningful for this
        // operation; zero launch records are accepted for legacy arithmetic
        // descriptors whose geometry is represented by the operation itself.
        if (layout && layout->launch_count > 1)
            return set_error("saved arithmetic geometry has multiple launches");
        if (layout && layout->launch_count == 1)
        {
            const auto &launch = layout->launches[0];
            if (!launch.grid.x || !launch.grid.y || !launch.grid.z ||
                !launch.block.x || !launch.block.y || !launch.block.z)
                return set_error("saved arithmetic launch geometry is empty");
            geometry.grid_x = launch.grid.x;
            geometry.grid_y = launch.grid.y;
            geometry.grid_z = launch.grid.z;
            geometry.block_x = launch.block.x;
            geometry.block_y = launch.block.y;
            geometry.block_z = launch.block.z;
            return 0;
        }
        const auto ceil_div = [](size_t value, size_t divisor, unsigned int *result) {
            if (!result || !divisor || value > SIZE_MAX - (divisor - 1)) return false;
            const size_t quotient = (value + divisor - 1) / divisor;
            if (quotient > std::numeric_limits<unsigned int>::max()) return false;
            *result = static_cast<unsigned int>(quotient);
            return true;
        };
        if (kind == static_cast<int>(PreparedArithmeticKind::Multiply))
        {
            const bool small = left_rows <= 4 && right_columns <= 4 && left_columns <= 16;
            bool thin = !view && left_rows == 1;
            if (thin)
                for (size_t limb = 0; limb < limbs; ++limb)
                    thin = thin && lhs->ctx->moduli[limb] > 1 &&
                        lhs->ctx->moduli[limb] <= std::numeric_limits<uint32_t>::max();
            if (small)
            {
                if (!ceil_div(n, 256, &geometry.grid_x) ||
                    right_columns > std::numeric_limits<unsigned int>::max())
                    return set_error("saved arithmetic grid dimensions overflow");
                geometry.grid_y = static_cast<unsigned int>(right_columns);
            }
            else if (thin)
            {
                geometry.block_x = kThinMatmulWarpSize;
                geometry.block_y = kThinMatmulColumnsPerBlock;
                if (!ceil_div(right_columns, kThinMatmulColumnsPerBlock, &geometry.grid_x) ||
                    n > SIZE_MAX - (kThinMatmulWarpSize - 1))
                    return set_error("saved arithmetic grid dimensions overflow");
                geometry.grid_y = 1;
                geometry.grid_z = static_cast<unsigned int>(std::min(
                    (n + kThinMatmulWarpSize - 1) / kThinMatmulWarpSize,
                    kMatmulMaxGridZ));
            }
            else
            {
                geometry.block_x = kMatmulTileN;
                geometry.block_y = kMatmulTileM;
                if (!ceil_div(right_columns, kMatmulTileN, &geometry.grid_x) ||
                    !ceil_div(left_rows, kMatmulTileM, &geometry.grid_y))
                    return set_error("saved arithmetic grid dimensions overflow");
                geometry.grid_z = static_cast<unsigned int>(std::min(n, kMatmulMaxGridZ));
            }
        }
        else if (kind == static_cast<int>(PreparedArithmeticKind::TensorSumRows) && n >= 256)
        {
            if (!ceil_div(n, 256, &geometry.grid_x))
                return set_error("saved arithmetic grid dimensions overflow");
            geometry.grid_y = static_cast<unsigned int>(std::min(cells, size_t{65535}));
        }
        else if (!ceil_div(coefficients, 256, &geometry.grid_x))
            return set_error("saved arithmetic grid dimensions overflow");
        return 0;
    }

    int validate_saved_arithmetic_descriptor(
        const GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
        int kind, const GpuMatrixBatchView *view, size_t column_start,
        const GpuPreparedPlanDescriptor *layout, GpuPreparedArithmeticLayout &geometry,
        size_t &workspace_bytes, size_t &workspace_alignment)
    {
        const GpuMatrix *effective_rhs = rhs ? rhs : lhs;
        if (!layout || !out || !lhs || !effective_rhs || !lhs->ctx || !out->ctx ||
            out->ctx != lhs->ctx || effective_rhs->ctx != lhs->ctx || out->level != lhs->level ||
            lhs->level < 0)
            return set_error("invalid saved arithmetic descriptor owners");
        const size_t limbs = static_cast<size_t>(lhs->level) + 1;
        if (layout->allocation_count != limbs && layout->allocation_count != limbs + 1)
            return set_error("saved arithmetic descriptor allocation count mismatch");
        // The planner writes one owner-stream footprint per active limb.  A
        // shared-stream owner may repeat the same physical slot, but its key
        // still carries that limb and must be checked independently.
        if (layout->stream_count != limbs)
            return set_error("saved arithmetic descriptor stream count mismatch");
        if (out->ctx->limb_gpu_ids.size() < limbs)
            return set_error("saved arithmetic owner has incomplete limb metadata");
        int device = -1;
        cudaStream_t stream = nullptr;
        if (matrix_limb_device(out, out->ctx->limb_gpu_ids[0], &device) != 0 ||
            matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream) != 0 || !stream)
            return set_error("saved arithmetic descriptor owner stream is unavailable");
        int status = saved_arithmetic_geometry(
            lhs, effective_rhs, out, kind, view, column_start, layout, geometry);
        if (status != 0) return status;
        geometry.device = device;
        const size_t workspace_count = layout->allocation_count - limbs;
        workspace_bytes = 0;
        workspace_alignment = alignof(uint64_t);
        GpuPreparedResourceKey first_key{};
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            dim3 id{};
            GpuPreparedResourceKey key{};
            status = gpu_prepared_limb_key(out->ctx, lhs->level, limb,
                GPU_PREPARED_STAGE_ARITHMETIC, &id, &key);
            if (status != 0) return status;
            if (limb == 0) first_key = key;
            if (gpu_prepared_require_allocation(layout, workspace_count + limb,
                    GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0)
                return set_error("saved arithmetic completion event differs from descriptor");
            const auto &event_entry = layout->allocations[workspace_count + limb];
            if (event_entry.rows != 0 || event_entry.columns != 0 ||
                event_entry.level != -1 || event_entry.format != -1)
                return set_error("saved arithmetic completion event metadata is invalid");
        }
        if (workspace_count)
        {
            const auto &entry = layout->allocations[0];
            if (entry.kind != GPU_PREPARED_BATCH_WORKSPACE || entry.bytes == 0 ||
                entry.alignment == 0 || entry.rows != geometry.output_rows ||
                entry.columns != geometry.output_columns || entry.level != lhs->level ||
                entry.format != lhs->format ||
                entry.alignment > 256 || (entry.alignment & (entry.alignment - 1)) != 0)
                return set_error("saved arithmetic workspace entry is invalid");
            if (gpu_prepared_require_allocation(layout, 0, entry.kind, &first_key,
                    entry.bytes, entry.alignment) != 0)
                return set_error("saved arithmetic workspace differs from descriptor");
            workspace_bytes = entry.bytes;
            workspace_alignment = entry.alignment;
        }
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            const auto &stream_entry = layout->streams[limb];
            dim3 id{};
            GpuPreparedResourceKey key{};
            status = gpu_prepared_limb_key(out->ctx, lhs->level, limb,
                GPU_PREPARED_STAGE_ARITHMETIC, &id, &key);
            if (status != 0) return status;
            cudaStream_t owner_stream = nullptr;
            status = matrix_limb_stream(out, id, &owner_stream);
            if (status != 0 || !owner_stream)
                return set_error("saved arithmetic owner stream is unavailable");
            if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
                gpu_prepared_require_stream_slot(out->ctx, id.x, owner_stream,
                    stream_entry.pool_slot) != 0 ||
                std::memcmp(&stream_entry.key, &key, sizeof(key)) != 0)
                return set_error("saved arithmetic stream differs from descriptor");
        }
        return 0;
    }

    int construct_saved_arithmetic_resources(
        GpuPreparedArithmeticState &prepared, size_t workspace_bytes,
        size_t workspace_alignment)
    {
        // Keep this order identical to the descriptor writer: workspace first,
        // followed by one completion event for each active limb.
        if (workspace_bytes != 0)
        {
            const int status = prepared.workspace.acquire(
                prepared.out->ctx, prepared.device, GPU_PREPARED_BATCH_WORKSPACE,
                workspace_bytes, workspace_alignment, prepared.stream);
            if (status != 0) return status;
        }
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            auto event = std::make_unique<GpuCudaResource>();
            const int status = event->acquire(
                prepared.out->ctx, prepared.device, GPU_PREPARED_COMPLETION_EVENT);
            if (status != 0) return status;
            prepared.completion[limb] = std::move(event);
        }
        return 0;
    }

    int prepared_arithmetic_wait(const GpuPreparedArithmeticState &prepared)
    {
        int status = matrix_wait_all_limb_streams(prepared.lhs, prepared.device, prepared.stream, true, true);
        if (status != 0) return status;
        if (prepared.rhs && prepared.rhs != prepared.lhs)
        {
            status = matrix_wait_all_limb_streams(prepared.rhs, prepared.device, prepared.stream, true, true);
            if (status != 0) return status;
        }
        return matrix_wait_all_limb_streams(prepared.out, prepared.device, prepared.stream, true);
    }

    int prepared_arithmetic_finish(const GpuPreparedArithmeticState &prepared)
    {
        for (size_t limb = 0; limb < prepared.limb_count; ++limb)
        {
            if (!prepared.completion[limb]) continue;
            const cudaError_t error = cudaEventRecord(
                prepared.completion[limb]->event, prepared.stream);
            if (error != cudaSuccess) return set_error(error);
        }
        int status = matrix_record_all_limb_writes(prepared.out, prepared.stream, true);
        if (status != 0) return status;
        const dim3 first = prepared.out->ctx->limb_gpu_ids[0];
        const auto &states = prepared.out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        status = matrix_track_all_limb_consumers(prepared.lhs, prepared.device, prepared.stream, completion, true, true);
        if (status != 0) return status;
        if (prepared.rhs && prepared.rhs != prepared.lhs)
            status = matrix_track_all_limb_consumers(prepared.rhs, prepared.device, prepared.stream, completion, true, true);
        return status;
    }

} // namespace

extern "C" int gpu_matrix_prepare_arithmetic_with_layout(
    GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, int kind,
    const size_t *rows, const size_t *offsets, size_t group_count,
    size_t term_count, const GpuMatrixBatchView *view, size_t column_start,
    const uint64_t *scalar_residues, size_t scalar_count, size_t automorphism_index,
    const GpuPreparedPlanDescriptor *layout, GpuPreparedArithmetic **plan)
{
    if (!layout || !plan || gpu_prepared_validate_descriptor(layout) != 0)
        return set_error("prepared arithmetic layout is missing");
    *plan = nullptr;
    try
    {
        GpuPreparedArithmeticLayout geometry{};
        size_t workspace_bytes = 0;
        size_t workspace_alignment = alignof(uint64_t);
        const int validation = validate_saved_arithmetic_descriptor(
            out, lhs, rhs, kind, view, column_start, layout, geometry,
            workspace_bytes, workspace_alignment);
        if (validation != 0) return validation;

        auto prepared = std::make_unique<GpuPreparedArithmeticState>();
        const int status = configure_arithmetic_plan(
            out, lhs, rhs, kind, rows, offsets, group_count, term_count, view,
            column_start, scalar_residues, scalar_count, automorphism_index,
            *prepared, &geometry);
        if (status != 0) return status;
        const int resource_status = construct_saved_arithmetic_resources(
            *prepared, workspace_bytes, workspace_alignment);
        if (resource_status != 0) return resource_status;
        *plan = reinterpret_cast<GpuPreparedArithmetic *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

// Keep the legacy entry point after the descriptor-driven entry point.  This
// makes the one-way call graph apparent in source as well as at runtime.
// Standalone non-prepared arithmetic API. The saved prepared runtime consumes
// gpu_matrix_prepare_arithmetic_with_layout directly.
extern "C" int gpu_matrix_prepare_arithmetic(
    GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs, int kind,
    const size_t *rows, const size_t *offsets, size_t group_count,
    size_t term_count, const GpuMatrixBatchView *view, size_t column_start,
    const uint64_t *scalar_residues, size_t scalar_count, size_t automorphism_index,
    GpuPreparedArithmetic **plan)
{
    if (!plan) return set_error("null prepared arithmetic output");
    *plan = nullptr;
    try
    {
        auto prepared = std::make_unique<GpuPreparedArithmeticState>();
        const int status = configure_arithmetic_plan(
            out, lhs, rhs, kind, rows, offsets, group_count, term_count,
            view, column_start, scalar_residues, scalar_count, automorphism_index, *prepared);
        if (status != 0) return status;
        *plan = reinterpret_cast<GpuPreparedArithmetic *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" int gpu_matrix_submit_arithmetic(const GpuPreparedArithmetic *opaque)
{
    const auto *prepared = reinterpret_cast<const GpuPreparedArithmeticState *>(opaque);
    if (!prepared || !prepared->out || !prepared->lhs || !prepared->stream)
        return set_error("invalid prepared arithmetic plan");
    int status = cudaSetDevice(prepared->device);
    if (status != cudaSuccess) return set_error(static_cast<cudaError_t>(status));
    status = prepared_arithmetic_wait(*prepared);
    if (status != 0) return status;
    switch (prepared->kind)
    {
        case PreparedArithmeticKind::Copy:
            block_copy_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->copy, prepared->limb_count,
                prepared->output_range.row_end - prepared->output_range.row_start,
                prepared->output_range.column_end - prepared->output_range.column_start,
                prepared->n, prepared->lhs->cols, prepared->out->cols,
                prepared->left_range.row_start, prepared->left_range.column_start,
                prepared->output_range.row_start, prepared->output_range.column_start);
            break;
        case PreparedArithmeticKind::Add:
            prepared_add_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->elementwise, prepared->limb_count,
                prepared->output_range.row_end - prepared->output_range.row_start,
                prepared->output_range.column_end - prepared->output_range.column_start,
                prepared->n, prepared->lhs->cols, prepared->rhs->cols, prepared->out->cols,
                prepared->left_range.row_start, prepared->left_range.column_start,
                prepared->right_range.row_start, prepared->right_range.column_start,
                prepared->output_range.row_start, prepared->output_range.column_start);
            break;
        case PreparedArithmeticKind::Subtract:
            prepared_sub_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->elementwise, prepared->limb_count,
                prepared->output_range.row_end - prepared->output_range.row_start,
                prepared->output_range.column_end - prepared->output_range.column_start,
                prepared->n, prepared->lhs->cols, prepared->rhs->cols, prepared->out->cols,
                prepared->left_range.row_start, prepared->left_range.column_start,
                prepared->right_range.row_start, prepared->right_range.column_start,
                prepared->output_range.row_start, prepared->output_range.column_start);
            break;
        case PreparedArithmeticKind::Negate:
        case PreparedArithmeticKind::Scale:
            prepared_unary_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->elementwise, prepared->scalars, prepared->limb_count,
                prepared->output_range.row_end - prepared->output_range.row_start,
                prepared->output_range.column_end - prepared->output_range.column_start,
                prepared->n, prepared->lhs->cols, prepared->out->cols,
                prepared->left_range.row_start, prepared->left_range.column_start,
                prepared->output_range.row_start, prepared->output_range.column_start,
                prepared->kind == PreparedArithmeticKind::Negate ? 0 : 1);
            break;
        case PreparedArithmeticKind::Automorphism:
            prepared_automorphism_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->elementwise, prepared->limb_count,
                prepared->output_range.row_end - prepared->output_range.row_start,
                prepared->output_range.column_end - prepared->output_range.column_start,
                prepared->n, prepared->lhs->cols, prepared->out->cols,
                prepared->left_range.row_start, prepared->left_range.column_start,
                prepared->output_range.row_start, prepared->output_range.column_start,
                prepared->automorphism_index, prepared->evaluation, prepared->log_n);
            break;
        case PreparedArithmeticKind::Tensor:
            tensor_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                prepared->product, prepared->left_columns, prepared->right_rows,
                prepared->right_columns, prepared->output_count * prepared->n,
                prepared->n, prepared->geometry);
            break;
        case PreparedArithmeticKind::TensorSumRows:
            if (prepared->row_sum_separate_polynomials)
                tensor_sum_rows_all_limbs_kernel<true><<<prepared->grid, 256, 0, prepared->stream>>>(
                    prepared->row_sum, prepared->left_columns, prepared->right_rows,
                    prepared->right_columns, prepared->output_count, prepared->n);
            else
                tensor_sum_rows_all_limbs_kernel<false><<<prepared->grid, 256, 0, prepared->stream>>>(
                    prepared->row_sum, prepared->left_columns, prepared->right_rows,
                    prepared->right_columns, prepared->output_count, prepared->n);
            break;
        case PreparedArithmeticKind::Multiply:
            if (prepared->matmul_count == 0)
                prepared_small_dot_rect_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
                    prepared->product,
                    prepared->output_range.row_end - prepared->output_range.row_start,
                    prepared->left_columns, prepared->right_columns, prepared->n,
                    prepared->lhs->cols, prepared->rhs->cols, prepared->out->cols,
                    prepared->left_range.row_start, prepared->left_range.column_start,
                    prepared->right_range.row_start, prepared->right_range.column_start,
                    prepared->output_range.row_start, prepared->output_range.column_start);
            else
            {
                for (size_t limb = 0; limb < prepared->matmul_count; ++limb)
                {
                    const auto &launch = prepared->matmul[limb];
                    if (launch.thin)
                        block_thin_row_matmul_kernel<<<launch.grid, launch.block, 0, prepared->stream>>>(
                            launch.lhs_base, launch.rhs_base, launch.out_base,
                            launch.inner, launch.columns, launch.n, launch.lhs_stride,
                            launch.rhs_stride, launch.out_stride, launch.lhs_width,
                            launch.rhs_width, launch.out_width, launch.modulus,
                            launch.reciprocal, launch.lazy_reduction);
                    else
                        prepared_block_matmul_rect_kernel<<<launch.grid, launch.block, 0, prepared->stream>>>(
                            launch.lhs_base, launch.rhs_base, launch.out_base,
                            launch.rows, launch.inner, launch.columns, launch.n,
                            launch.lhs_pitch, launch.rhs_pitch, launch.out_pitch,
                            launch.lhs_row, launch.lhs_column,
                            launch.rhs_row, launch.rhs_column,
                            launch.out_row, launch.out_column,
                            launch.lhs_stride, launch.rhs_stride, launch.out_stride,
                            launch.lhs_width, launch.rhs_width,
                            launch.out_width, launch.modulus);
                }
            }
            break;
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        gpu_matrix_retire_submitted_work(prepared->out);
        return set_error(error);
    }
    status = prepared_arithmetic_finish(*prepared);
    if (status != 0)
    {
        gpu_matrix_retire_submitted_work(prepared->out);
        return status;
    }
    prepared->out->format = prepared->lhs->format;
    return 0;
}

extern "C" void gpu_matrix_destroy_arithmetic_plan(GpuPreparedArithmetic *opaque)
{
    delete reinterpret_cast<GpuPreparedArithmeticState *>(opaque);
}

// Standalone non-prepared copy API; descriptor-driven binds use the entry
// immediately below and do not route through this planner.
extern "C" int gpu_matrix_prepare_input_copy(
    GpuMatrix *out, const GpuMatrix *source_template,
    const GpuMatrixBatchView *view, GpuPreparedInputCopy **plan)
{
    if (!plan) return set_error("null prepared input copy output");
    *plan = nullptr;
    try
    {
        auto prepared = std::make_unique<GpuPreparedInputCopyState>();
        const int status = prepare_copy_layout(out, source_template, view, *prepared);
        if (status != 0) return status;
        *plan = reinterpret_cast<GpuPreparedInputCopy *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" int gpu_matrix_prepare_input_copy_with_layout(
    GpuMatrix *out, const GpuMatrix *source_template,
    const GpuMatrixBatchView *view, const GpuPreparedPlanDescriptor *layout,
    GpuPreparedInputCopy **plan)
{
    if (!layout || !plan || gpu_prepared_validate_descriptor(layout) != 0)
        return set_error("prepared input copy layout is missing");
    *plan = nullptr;
    try
    {
        auto prepared = std::make_unique<GpuPreparedInputCopyState>();
        const int validation = validate_saved_input_copy_descriptor(
            out, source_template, view, layout, *prepared);
        if (validation != 0) return validation;
        const int resource_status = construct_saved_input_copy_resources(*prepared, layout);
        if (resource_status != 0) return resource_status;
        *plan = reinterpret_cast<GpuPreparedInputCopy *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" void gpu_matrix_test_reset_input_copy_bind_counters()
{
    input_copy_prepared_acquisitions.store(0, std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_input_copy_prepared_acquisitions()
{
    return input_copy_prepared_acquisitions.load(std::memory_order_relaxed);
}

extern "C" int gpu_matrix_submit_input_copy(
    const GpuPreparedInputCopy *opaque, const GpuMatrix *source)
{
    const auto *prepared = reinterpret_cast<const GpuPreparedInputCopyState *>(opaque);
    if (!prepared || !prepared->out || !source)
        return set_error("invalid prepared input copy");
    if (source->ctx != prepared->out->ctx || source->level != static_cast<int>(prepared->level) ||
        source->rows != prepared->source_rows || source->cols != prepared->source_columns ||
        source->format != prepared->format)
        return set_error("prepared input copy source contract mismatch");
    BlockCopyMetadata metadata = prepared->source_layout;
    for (size_t limb = 0; limb <= prepared->level; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        int device = -1;
        const uint8_t *base = nullptr;
        size_t stride = 0;
        uint8_t width = 0;
        int status = prepared_matrix_metadata(source, id, &base, &stride, &width, &device);
        if (status != 0) return status;
        if (device != prepared->device ||
            stride != prepared->source_layout.src_stride_bytes[limb] ||
            width != prepared->source_layout.src_coeff_bytes[limb])
            return set_error("prepared input copy source byte extent mismatch");
        metadata.src_bases[limb] = base;
    }
    int status = cudaSetDevice(prepared->device);
    if (status != cudaSuccess) return set_error(static_cast<cudaError_t>(status));
    status = matrix_wait_all_limb_streams(source, prepared->device, prepared->stream, true, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(prepared->out, prepared->device, prepared->stream, true);
    if (status != 0) return status;
    const size_t rows = prepared->output_range.row_end - prepared->output_range.row_start;
    const size_t columns = prepared->output_range.column_end - prepared->output_range.column_start;
    block_copy_rect_all_limbs_kernel<<<prepared->grid, 256, 0, prepared->stream>>>(
        metadata, prepared->level + 1, rows, columns, prepared->n,
        source->cols, prepared->out->cols,
        prepared->input_range.row_start, prepared->input_range.column_start,
        prepared->output_range.row_start, prepared->output_range.column_start);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb <= prepared->level; ++limb)
    {
        if (!prepared->completion[limb]) continue;
        error = cudaEventRecord(prepared->completion[limb]->event, prepared->stream);
        if (error != cudaSuccess) return set_error(error);
    }
    status = matrix_record_all_limb_writes(prepared->out, prepared->stream, true);
    if (status != 0) return status;
    const dim3 first = prepared->out->ctx->limb_gpu_ids[0];
    const auto &states = prepared->out->exec_limb_states[first.x];
    const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
    return matrix_track_all_limb_consumers(source, prepared->device, prepared->stream, completion, true, true);
}

extern "C" void gpu_matrix_destroy_input_copy(GpuPreparedInputCopy *opaque)
{
    delete reinterpret_cast<GpuPreparedInputCopyState *>(opaque);
}

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
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_add");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
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
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_add_block");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
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
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_sub");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
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

struct GpuPreparedTransposeState
{
    GpuMatrix *out;
    const GpuMatrix *source;
    GpuMatrixBatchView view;
    TransposeMetadata metadata;
    size_t rows;
    size_t count;
    size_t limb_count;
    dim3 grid;
    int device;
    cudaStream_t stream;
    bool has_view;
    std::array<std::unique_ptr<GpuCudaResource>, kArithMetadataLimbs> completion;

    ~GpuPreparedTransposeState()
    {
        for (auto &event : completion)
            if (event) (void)event->release();
    }
};

static std::atomic<size_t> transpose_prepared_acquisitions{0};

static int prepare_transpose_plan(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
    GpuPreparedTransposeState &prepared,
    const GpuPreparedLaunchLayout *saved_launch = nullptr)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in prepared transpose");
    if (!source || out == source || !source->ctx || out->ctx != source->ctx ||
        source->level < 0 || out->level != source->level || out->format != source->format ||
        (!view && (out->rows != source->cols || out->cols != source->rows)))
        return set_error("invalid prepared transpose arguments");
    prepared.has_view = view != nullptr;
    prepared.view = view ? *view : GpuMatrixBatchView{
        GpuMatrixRange{0, source->rows, 0, source->cols},
        GpuMatrixRange{0, source->rows, 0, source->cols},
        GpuMatrixRange{0, out->rows, 0, out->cols}};
    const GpuMatrixRange &input = prepared.view.left;
    const GpuMatrixRange &output = prepared.view.output;
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input, source) || !valid(output, out) ||
        (view && std::memcmp(&view->left, &view->right, sizeof(GpuMatrixRange)) != 0) ||
        input.row_end - input.row_start != output.column_end - output.column_start ||
        input.column_end - input.column_start != output.row_end - output.row_start)
        return set_error("invalid prepared transpose view");
    prepared.rows = input.row_end - input.row_start;
    const size_t columns = input.column_end - input.column_start;
    if (prepared.rows == 0 || columns == 0 || source->ctx->N <= 0)
        return set_error("empty prepared transpose is not replayable");
    const size_t n = static_cast<size_t>(source->ctx->N);
    prepared.limb_count = static_cast<size_t>(source->level) + 1;
    if (prepared.limb_count > kArithMetadataLimbs ||
        source->ctx->limb_gpu_ids.size() < prepared.limb_count ||
        source->rows > std::numeric_limits<size_t>::max() / source->cols ||
        source->rows * source->cols > std::numeric_limits<size_t>::max() / n)
        return set_error("prepared transpose shape overflow or invalid basis");
    prepared.metadata = {};
    prepared.metadata.source_offset = input.row_start * source->cols + input.column_start;
    prepared.metadata.source_stride = source->cols;
    prepared.metadata.output_offset = output.row_start * out->cols + output.column_start;
    prepared.metadata.output_stride = out->cols;
    prepared.device = -1;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &prepared.stream);
    if (status != 0) return status;
    for (size_t limb = 0; limb < prepared.limb_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x >= source->shared_limb_buffers.size() || id.x >= out->shared_limb_buffers.size())
            return set_error("invalid prepared transpose partition");
        const auto &input_buffer = source->shared_limb_buffers[id.x];
        const auto &output_buffer = out->shared_limb_buffers[id.x];
        if (!input_buffer.device_descriptors || !output_buffer.device_descriptors ||
            id.y >= input_buffer.limb_count || id.y >= output_buffer.limb_count ||
            input_buffer.device != output_buffer.device)
            return set_error("invalid prepared transpose descriptors");
        if (limb == 0)
        {
            prepared.device = output_buffer.device;
            prepared.metadata.source = input_buffer.device_descriptors;
            prepared.metadata.out = output_buffer.device_descriptors;
        }
        else if (prepared.device != output_buffer.device ||
                 prepared.metadata.source != input_buffer.device_descriptors ||
                 prepared.metadata.out != output_buffer.device_descriptors)
            return set_error("prepared transpose descriptors span partitions");
        prepared.metadata.indices[limb] = id.y;
    }
    prepared.count = prepared.rows * columns * n;
    if (prepared.count > std::numeric_limits<size_t>::max() / 256)
        return set_error("prepared transpose grid overflow");
    if (saved_launch)
    {
        prepared.grid = saved_launch->grid;
    }
    else
    {
        GpuPreparedRectLayout layout{};
        status = query_rect_layout(n, prepared.limb_count, prepared.rows, columns,
            prepared.device, 2, layout);
        if (status != 0) return status;
        prepared.grid = dim3(layout.grid_x, layout.grid_y, layout.grid_z);
    }
    prepared.out = out;
    prepared.source = source;
    return 0;
}

static int validate_saved_transpose_descriptor(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
    const GpuPreparedPlanDescriptor *layout, GpuPreparedTransposeState &prepared)
{
    if (!layout || !out || !source || !out->ctx || out->ctx->N <= 0 ||
        source->ctx != out->ctx || source->level < 0 ||
        out->level != source->level || out->format != source->format ||
        out->ctx->limb_gpu_ids.size() < static_cast<size_t>(source->level + 1) ||
        !out->ctx->execution)
        return set_error("invalid saved transpose descriptor owners");
    const size_t limbs = static_cast<size_t>(source->level + 1);
    if (!limbs || limbs > kArithMetadataLimbs || layout->allocation_count != limbs ||
        layout->stream_count != 1 || layout->launch_count != 1)
        return set_error("saved transpose descriptor shape is invalid");
    const GpuMatrixRange input = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (input.row_end < input.row_start || input.column_end < input.column_start ||
        output.row_end < output.row_start || output.column_end < output.column_start)
        return set_error("saved transpose descriptor range is invalid");
    const size_t rows = input.row_end - input.row_start;
    const size_t columns = input.column_end - input.column_start;
    if (!rows || !columns || rows > SIZE_MAX / columns ||
        rows * columns > SIZE_MAX / static_cast<size_t>(out->ctx->N))
        return set_error("saved transpose descriptor dimensions overflow");
    const size_t coefficients = rows * columns * static_cast<size_t>(out->ctx->N);
    if (coefficients > SIZE_MAX - 255)
        return set_error("saved transpose descriptor grid overflow");
    const size_t quotient = (coefficients + 255) / 256;
    const size_t expected_grid_x = std::min(quotient, size_t{65535});
    const auto &launch = layout->launches[0];
    if (launch.phase != 0 || launch.len != static_cast<size_t>(out->ctx->N) ||
        launch.limb_offset != 0 || launch.limb_count != limbs || launch.narrow != 0 ||
        launch.grid.x != expected_grid_x || launch.grid.y != 1 ||
        launch.grid.z != limbs || launch.block.x != 256 || launch.block.y != 1 ||
        launch.block.z != 1)
        return set_error("saved transpose launch geometry differs from descriptor");
    const int status = prepare_transpose_plan(out, source, view, prepared, &launch);
    if (status != 0) return status;
    dim3 first{};
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        dim3 id{};
        GpuPreparedResourceKey key{};
        if (gpu_prepared_limb_key(out->ctx, source->level, limb,
                GPU_PREPARED_STAGE_TRANSFORM, &id, &key) != 0 ||
            gpu_prepared_require_allocation(layout, limb,
                GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0)
            return set_error("saved transpose completion event differs from descriptor");
        const auto &entry = layout->allocations[limb];
        if (entry.rows != 0 || entry.columns != 0 || entry.level != -1 || entry.format != -1)
            return set_error("saved transpose completion event metadata is invalid");
        if (limb == 0) first = id;
    }
    const auto &stream_entry = layout->streams[0];
    GpuPreparedResourceKey stream_key{};
    if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        gpu_prepared_limb_key(out->ctx, source->level, 0, GPU_PREPARED_STAGE_TRANSFORM,
            &first, &stream_key) != 0 || std::memcmp(&stream_entry.key, &stream_key, sizeof(stream_key)) != 0 ||
        gpu_prepared_require_stream_slot(out->ctx, first.x, prepared.stream,
            stream_entry.pool_slot) != 0)
        return set_error("saved transpose stream differs from descriptor");
    return 0;
}

static int construct_saved_transpose_resources(
    GpuPreparedTransposeState &prepared, const GpuPreparedPlanDescriptor *layout)
{
    for (size_t limb = 0; limb < prepared.limb_count; ++limb)
    {
        const auto &entry = layout->allocations[limb];
        auto event = std::make_unique<GpuCudaResource>();
        const int status = event->acquire(
            prepared.out->ctx, entry.key.device, GPU_PREPARED_COMPLETION_EVENT);
        if (status != 0) return status;
        prepared.completion[limb] = std::move(event);
        transpose_prepared_acquisitions.fetch_add(1, std::memory_order_relaxed);
    }
    return 0;
}

static int submit_transpose_plan(const GpuPreparedTransposeState &prepared)
{
    cudaError_t error = cudaSetDevice(prepared.device);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(prepared.source, prepared.device, prepared.stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(prepared.out, prepared.device, prepared.stream);
    if (status != 0) return status;
    transpose_all_limbs_kernel<<<prepared.grid, 256, 0, prepared.stream>>>(
        prepared.metadata, prepared.rows, prepared.count, static_cast<size_t>(prepared.source->ctx->N));
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < prepared.limb_count; ++limb)
    {
        if (!prepared.completion[limb]) continue;
        error = cudaEventRecord(prepared.completion[limb]->event, prepared.stream);
        if (error != cudaSuccess) return set_error(error);
    }
    if (prepared.has_view)
    {
        status = matrix_record_all_limb_writes(prepared.out, prepared.stream, true);
        if (status != 0) return status;
        const dim3 first = prepared.out->ctx->limb_gpu_ids[0];
        const cudaEvent_t completion = prepared.out->exec_limb_states[first.x][first.y].write_done;
        return matrix_track_all_limb_consumers(
            prepared.source, prepared.device, prepared.stream, completion, true, true);
    }
    status = matrix_track_all_limb_consumers(prepared.source, prepared.device, prepared.stream);
    if (status != 0) return status;
    return matrix_record_all_limb_writes(prepared.out, prepared.stream);
}

// Standalone non-prepared transpose API; descriptor-driven binds use the
// saved-layout entry and do not route through this planner.
extern "C" int gpu_matrix_prepare_transpose(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
    GpuPreparedTranspose **plan)
{
    if (!plan) return set_error("null prepared transpose output");
    *plan = nullptr;
    try
    {
        auto prepared = std::make_unique<GpuPreparedTransposeState>();
        const int status = prepare_transpose_plan(out, source, view, *prepared);
        if (status != 0) return status;
        *plan = reinterpret_cast<GpuPreparedTranspose *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" int gpu_matrix_prepare_transpose_with_layout(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view,
    const GpuPreparedPlanDescriptor *layout, GpuPreparedTranspose **plan)
{
    if (!layout || !plan || gpu_prepared_validate_descriptor(layout) != 0)
        return set_error("prepared transpose layout is missing");
    *plan = nullptr;
    try
    {
        auto prepared = std::make_unique<GpuPreparedTransposeState>();
        const int validation = validate_saved_transpose_descriptor(
            out, source, view, layout, *prepared);
        if (validation != 0) return validation;
        const int resource_status = construct_saved_transpose_resources(*prepared, layout);
        if (resource_status != 0) return resource_status;
        *plan = reinterpret_cast<GpuPreparedTranspose *>(prepared.release());
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" void gpu_matrix_test_reset_transpose_bind_counters()
{
    transpose_prepared_acquisitions.store(0, std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_transpose_prepared_acquisitions()
{
    return transpose_prepared_acquisitions.load(std::memory_order_relaxed);
}

extern "C" int gpu_matrix_submit_transpose(const GpuPreparedTranspose *plan)
{
    const auto *prepared = reinterpret_cast<const GpuPreparedTransposeState *>(plan);
    if (!prepared || !prepared->out || !prepared->source || !prepared->stream)
        return set_error("invalid prepared transpose plan");
    return submit_transpose_plan(*prepared);
}

extern "C" void gpu_matrix_destroy_transpose(GpuPreparedTranspose *plan)
{
    delete reinterpret_cast<GpuPreparedTransposeState *>(plan);
}

extern "C" int gpu_matrix_transpose(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_transpose");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !source || out == source || !source->ctx || out->ctx != source->ctx ||
        source->level < 0 || out->level != source->level || out->format != source->format ||
        (!view && (out->rows != source->cols || out->cols != source->rows)))
        return set_error("invalid gpu_matrix_transpose arguments");
    const GpuMatrixRange input = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input, source) || !valid(output, out) ||
        input.row_end - input.row_start != output.column_end - output.column_start ||
        input.column_end - input.column_start != output.row_end - output.row_start)
        return set_error("invalid transpose view");
    const size_t rows = input.row_end - input.row_start;
    const size_t cols = input.column_end - input.column_start;
    if (rows == 0 || cols == 0 || source->ctx->N <= 0) return 0;
    const size_t n = static_cast<size_t>(source->ctx->N);
    const size_t limb_count = static_cast<size_t>(source->level) + 1;
    if (limb_count > kArithMetadataLimbs || source->ctx->limb_gpu_ids.size() < limb_count ||
        source->rows > std::numeric_limits<size_t>::max() / source->cols ||
        source->rows * source->cols > std::numeric_limits<size_t>::max() / n)
        return set_error("transpose shape overflow or invalid basis");
    TransposeMetadata metadata{};
    metadata.source_offset = input.row_start * source->cols + input.column_start;
    metadata.source_stride = source->cols;
    metadata.output_offset = output.row_start * out->cols + output.column_start;
    metadata.output_stride = out->cols;
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
    status = matrix_wait_all_limb_streams(source, device, stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, device, stream);
    if (status != 0) return status;
    const size_t count = rows * cols * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    transpose_all_limbs_kernel<<<grid, 256, 0, stream>>>(
        metadata, rows, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    if (view) {
        // The retained destination event covers this reader, as in matrix batch
        // views. Publish it before joining source reuse/release; no lazy event
        // allocation or source writer-event mutation is needed.
        status = matrix_record_all_limb_writes(out, stream, true);
        if (status != 0) return status;
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const cudaEvent_t completion = out->exec_limb_states[first.x][first.y].write_done;
        return matrix_track_all_limb_consumers(source, device, stream, completion, true, true);
    }
    status = matrix_track_all_limb_consumers(source, device, stream);
    if (status != 0) return status;
    return matrix_record_all_limb_writes(out, stream);
}

extern "C" int gpu_matrix_sum_rows(
    GpuMatrix *out, const GpuMatrix *source, const size_t *rows, const size_t *offsets,
    size_t group_count, size_t term_count, const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_sum_rows");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !source || out == source || !source->ctx || out->ctx != source->ctx || source->level < 0 ||
        out->level != source->level || out->format != source->format ||
        (!view && (out->rows != group_count || out->cols != source->cols)) || !rows || !offsets ||
        group_count == 0 || group_count > 16 || term_count == 0 || term_count > 32 ||
        offsets[0] != 0 || offsets[group_count] != term_count)
        return set_error("invalid gpu_matrix_sum_rows arguments");
    const GpuMatrixRange input = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input, source) || !valid(output, out) ||
        output.row_end - output.row_start != group_count ||
        input.column_end - input.column_start != output.column_end - output.column_start)
        return set_error("invalid row sum view");
    const size_t columns = input.column_end - input.column_start;
    RowSumMetadata metadata{};
    metadata.source_offset = input.row_start * source->cols + input.column_start;
    metadata.source_stride = source->cols;
    metadata.output_offset = output.row_start * out->cols + output.column_start;
    metadata.output_stride = out->cols;
    for (size_t group = 0; group < group_count; ++group)
    {
        if (offsets[group] >= offsets[group + 1] || offsets[group + 1] > term_count)
            return set_error("invalid row sum group offsets");
        metadata.offsets[group] = offsets[group];
    }
    metadata.offsets[group_count] = term_count;
    for (size_t term = 0; term < term_count; ++term)
    {
        if (rows[term] >= input.row_end - input.row_start) return set_error("row sum input index out of bounds");
        metadata.rows[term] = rows[term];
    }
    if (columns == 0 || source->ctx->N <= 0) return 0;
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
    status = matrix_wait_all_limb_streams(source, device, stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, device, stream);
    if (status != 0) return status;
    const size_t count = group_count * columns * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    sum_rows_all_limbs_kernel<<<grid, 256, 0, stream>>>(metadata, columns, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    if (view) {
        // The retained destination event covers this reader, as in matrix batch
        // views. Publish it before joining source reuse/release; no lazy event
        // allocation or source writer-event mutation is needed.
        status = matrix_record_all_limb_writes(out, stream, true);
        if (status != 0) return status;
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const cudaEvent_t completion = out->exec_limb_states[first.x][first.y].write_done;
        return matrix_track_all_limb_consumers(source, device, stream, completion, true, true);
    }
    status = matrix_track_all_limb_consumers(source, device, stream);
    if (status != 0) return status;
    return matrix_record_all_limb_writes(out, stream);
}

extern "C" int gpu_matrix_add_row_blocks(
    GpuMatrix *out, const GpuMatrix *const *lhs_blocks, size_t block_count, const GpuMatrix *rhs,
    const GpuMatrixRange *block_views, const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_add_row_blocks");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!rhs || !rhs->ctx || !lhs_blocks || block_count == 0 || block_count > kRowAddMaxBlocks ||
        out == rhs || out->ctx != rhs->ctx || out->level != rhs->level || rhs->level < 0 ||
        out->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL ||
        ((block_views == nullptr) != (view == nullptr)))
        return set_error("invalid gpu_matrix_add_row_blocks arguments");
    const size_t maximum = std::numeric_limits<size_t>::max();
    auto valid_range = [maximum](const GpuMatrix *matrix, const GpuMatrixRange &range) {
        return range.row_start <= range.row_end && range.row_end <= matrix->rows &&
            range.column_start <= range.column_end && range.column_end <= matrix->cols &&
            (matrix->cols == 0 || matrix->rows <= maximum / matrix->cols);
    };
    const GpuMatrixRange rhs_range = view ? view->right : GpuMatrixRange{0, rhs->rows, 0, rhs->cols};
    const GpuMatrixRange out_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (!valid_range(rhs, rhs_range) || !valid_range(out, out_range) ||
        rhs_range.row_end - rhs_range.row_start != out_range.row_end - out_range.row_start ||
        rhs_range.column_end - rhs_range.column_start != out_range.column_end - out_range.column_start)
        return set_error("invalid row block output range");
    const size_t rows = rhs_range.row_end - rhs_range.row_start;
    const size_t columns = rhs_range.column_end - rhs_range.column_start;
    RowBlockAddMetadata metadata{};
    metadata.rhs_offset = rhs_range.row_start * rhs->cols + rhs_range.column_start;
    metadata.rhs_stride = rhs->cols;
    metadata.out_offset = out_range.row_start * out->cols + out_range.column_start;
    metadata.out_stride = out->cols;
    size_t total_rows = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *matrix = lhs_blocks[block];
        if (!matrix || matrix == out || matrix->ctx != rhs->ctx || matrix->level != rhs->level ||
            matrix->format != GPU_POLY_FORMAT_EVAL)
            return set_error("invalid row block input");
        const GpuMatrixRange range = block_views ? block_views[block] : GpuMatrixRange{0, matrix->rows, 0, matrix->cols};
        if (!valid_range(matrix, range) || range.column_end - range.column_start != columns ||
            range.row_end - range.row_start > maximum - total_rows)
            return set_error("invalid row block input range");
        metadata.rows[block] = range.row_end - range.row_start;
        metadata.offsets[block] = range.row_start * matrix->cols + range.column_start;
        metadata.strides[block] = matrix->cols;
        total_rows += metadata.rows[block];
    }
    if (total_rows != rows) return set_error("row block sum differs from output rows");
    if (rows == 0 || columns == 0 || rhs->ctx->N <= 0) return 0;
    const size_t n = static_cast<size_t>(rhs->ctx->N);
    const size_t limb_count = static_cast<size_t>(rhs->level) + 1;
    if (limb_count > kArithMetadataLimbs || rhs->ctx->limb_gpu_ids.size() < limb_count ||
        rhs->ctx->moduli.size() < limb_count || rows > maximum / columns || rows * columns > maximum / n)
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
        return matrix_wait_all_limb_streams(matrix, device, stream, false, matrix != out);
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
    const size_t count = rows * columns * n;
    const size_t blocks = count / 256 + (count % 256 != 0);
    const dim3 grid(static_cast<unsigned int>(std::min(blocks, size_t{65535})),
                    1, static_cast<unsigned int>(limb_count));
    add_row_blocks_all_limbs_kernel<<<grid, 256, 0, stream>>>(metadata, columns, count, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    if (view) {
        status = matrix_record_all_limb_writes(out, stream, true);
        if (status != 0) return status;
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const cudaEvent_t completion = out->exec_limb_states[first.x][first.y].write_done;
        for (size_t block = 0; block < block_count; ++block) {
            if (metadata.rows[block] == 0) continue;
            status = matrix_track_all_limb_consumers(lhs_blocks[block], device, stream, completion, true, true);
            if (status != 0) return status;
        }
        return matrix_track_all_limb_consumers(rhs, device, stream, completion, true, true);
    }
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

extern "C" int gpu_matrix_tensor_sum_rows(
    GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
    const size_t *rows, const size_t *offsets, size_t group_count, size_t term_count,
    const GpuMatrixBatchView *view, size_t column_start)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_tensor_sum_rows");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !lhs || !rhs || !lhs->ctx || out->ctx != lhs->ctx || rhs->ctx != lhs->ctx ||
        lhs->level < 0 || out->level != lhs->level || rhs->level != lhs->level ||
        lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL ||
        out->format != GPU_POLY_FORMAT_EVAL || !rows || !offsets ||
        group_count == 0 || group_count > 16 || term_count == 0 || term_count > 32 ||
        offsets[0] != 0 || offsets[group_count] != term_count)
        return set_error("invalid gpu_matrix_tensor_sum_rows arguments");
    const int view_status = validate_tensor_ranges(out, lhs, rhs, view, column_start, true, group_count);
    if (view_status != 0) return view_status;
    if (view && (view->output.row_start == view->output.row_end ||
        view->output.column_start == view->output.column_end)) return 0;
    const size_t product_rows = view ? (view->left.row_end - view->left.row_start) *
        (view->right.row_end - view->right.row_start) : lhs->rows * rhs->rows;
    const size_t maximum = std::numeric_limits<size_t>::max();
    if ((rhs->rows != 0 && lhs->rows > maximum / rhs->rows) ||
        (rhs->cols != 0 && lhs->cols > maximum / rhs->cols) ||
        (!view && (out->rows != group_count || out->cols != lhs->cols * rhs->cols)))
        return set_error("invalid tensor row sum shape");
    for (size_t group = 0; group < group_count; ++group)
        if (offsets[group] >= offsets[group + 1] || offsets[group + 1] > term_count)
            return set_error("invalid tensor row sum offsets");
    for (size_t term = 0; term < term_count; ++term)
        if (rows[term] >= product_rows)
            return set_error("tensor row sum index out of bounds");
    if (out->cols == 0 || lhs->ctx->N <= 0) return 0;
    const size_t n = static_cast<size_t>(lhs->ctx->N);
    if (out->rows > maximum / out->cols || out->rows * out->cols > maximum / n ||
        lhs->ctx->limb_gpu_ids.size() <= static_cast<size_t>(lhs->level))
        return set_error("tensor row sum size overflow or invalid limb mapping");
    return launch_descriptor_product(out, lhs, rhs, true, rows, offsets, group_count, term_count, view, column_start);
}

extern "C" int gpu_matrix_tensor(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs,
    const GpuMatrixBatchView *view, size_t column_start)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_tensor");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !lhs || !rhs || !lhs->ctx || out->ctx != lhs->ctx || rhs->ctx != lhs->ctx ||
        lhs->level < 0 || out->level != lhs->level || rhs->level != lhs->level ||
        lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid gpu_matrix_tensor arguments");
    const int view_status = validate_tensor_ranges(out, lhs, rhs, view, column_start, false, 0);
    if (view_status != 0) return view_status;
    if (view && (view->output.row_start == view->output.row_end ||
        view->output.column_start == view->output.column_end)) return 0;
    const size_t maximum = std::numeric_limits<size_t>::max();
    if ((rhs->rows != 0 && lhs->rows > maximum / rhs->rows) ||
        (rhs->cols != 0 && lhs->cols > maximum / rhs->cols) ||
        (!view && (out->rows != lhs->rows * rhs->rows || out->cols != lhs->cols * rhs->cols)))
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
    const int status = launch_descriptor_product(out, lhs, rhs, true, nullptr, nullptr, 0, 0, view, column_start);
    if (status == 0) out->format = GPU_POLY_FORMAT_EVAL;
    return status;
}

extern "C" int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_mul");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
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

extern "C" int gpu_matrix_query_equality_workspace(GpuPreparedWorkspaceLayout *out)
{
    if (!out) return set_error("null equality workspace output");
    *out = {sizeof(int), alignof(int), GPU_PREPARED_TRANSFER_WORKSPACE};
    return 0;
}

extern "C" int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal)
{
    if (!lhs || !lhs->ctx || !lhs->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_equal");
    GpuAllocationActivity activity(lhs->ctx->execution.get(), -1);
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
    if (lhs->rows != 0 && lhs->cols > SIZE_MAX / lhs->rows)
        return set_error("matrix equality shape overflow");
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

    std::vector<std::unique_ptr<EqualityWorkspace>> workspaces(lhs->ctx->gpu_ids.size());
    const auto release_workspaces = [&]() {
        int first_error = 0;
        for (auto &workspace : workspaces)
        {
            if (!workspace) continue;
            const int error = workspace->storage.release(workspace->last_stream);
            if (first_error == 0) first_error = error;
        }
        return first_error;
    };
    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        bool limb_equal = false;
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        if (limb_id.x >= workspaces.size())
            return set_error("invalid equality device partition");
        if (!workspaces[limb_id.x])
            workspaces[limb_id.x] = std::make_unique<EqualityWorkspace>();
        status = launch_matrix_equal_for_limb(
            lhs,
            rhs,
            count,
            static_cast<size_t>(N),
            limb_id,
            *workspaces[limb_id.x],
            limb_equal);
        if (status != 0)
        {
            return status;
        }
        if (!limb_equal)
        {
            return release_workspaces();
        }
    }

    status = release_workspaces();
    if (status != 0) return status;
    *out_equal = 1;
    return 0;
}

extern "C" int gpu_matrix_mul_scalar(
    GpuMatrix *out,
    const GpuMatrix *lhs,
    const GpuMatrix *scalar)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_mul_scalar");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
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
