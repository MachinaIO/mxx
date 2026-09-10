namespace
{
    using MatrixDescriptor = GpuMatrix::SharedLimbBuffer::DeviceDescriptor;

    // Allocation-owned descriptors are initialized once with the matrix. A
    // submission names each independent allocation once; kernels resolve CRT
    // limbs through these persistent descriptors rather than host-built tables.
    __device__ __forceinline__ uint8_t *batch_limb(
        const MatrixDescriptor *const *matrices, size_t index, size_t limbs)
    {
        const auto *descriptors = matrices[index / limbs];
        return descriptors ? descriptors[index % limbs].base : nullptr;
    }

    __global__ void matrix_binary_batch_kernel(
        const MatrixDescriptor *const *left,
        const MatrixDescriptor *const *right,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n,
        int operation)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = left[0][limb].stride;
        const uint8_t bytes = left[0][limb].width;
        const uint64_t lhs = matrix_load_limb_u64(
            batch_limb(left, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes);
        const uint64_t rhs = matrix_load_limb_u64(
            batch_limb(right, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes);
        const uint64_t value = operation == 0
                                   ? add_mod_u64(lhs, rhs, moduli[limb])
                                   : sub_mod_u64(lhs, rhs, moduli[limb]);
        matrix_store_limb_u64(
            batch_limb(outputs, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes, value);
    }

    __global__ void matrix_negate_batch_kernel(
        const MatrixDescriptor *const *inputs,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = inputs[0][limb].stride;
        const uint8_t bytes = inputs[0][limb].width;
        const uint64_t value = matrix_load_limb_u64(
            batch_limb(inputs, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes);
        const uint64_t negated = value == 0 ? 0 : moduli[limb] - value;
        matrix_store_limb_u64(
            batch_limb(outputs, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes, negated);
    }

    __global__ void matrix_ring_automorphism_batch_kernel(
        const MatrixDescriptor *const *inputs,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        const size_t *indices,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients) return;
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t matrix_idx = matrix_limb / limb_count;
        const size_t limb = matrix_limb % limb_count;
        const size_t local = idx % coefficients_per_limb;
        const size_t polynomial = local / n;
        const size_t source = local % n;
        const size_t exponent = (source * indices[matrix_idx]) % (2 * n);
        const size_t target = exponent < n ? exponent : exponent - n;
        const size_t stride = inputs[0][limb].stride;
        const uint8_t bytes = inputs[0][limb].width;
        const uint64_t modulus = moduli[limb];
        uint64_t value = matrix_load_limb_u64(batch_limb(inputs, matrix_limb, limb_count), polynomial, source, stride, bytes);
        if (exponent >= n && value != 0) value = modulus - value;
        matrix_store_limb_u64(batch_limb(outputs, matrix_limb, limb_count), polynomial, target, stride, bytes, value);
    }

    __global__ void matrix_scalar_mul_batch_kernel(
        const MatrixDescriptor *const *matrices,
        const MatrixDescriptor *const *scalars,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients) return;
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = matrices[0][limb].stride;
        const uint8_t bytes = matrices[0][limb].width;
        const uint64_t value = matrix_load_limb_u64(
            batch_limb(matrices, matrix_limb, limb_count), poly_idx, coefficient_idx, stride, bytes);
        const uint64_t scalar = matrix_load_limb_u64(
            batch_limb(scalars, matrix_limb, limb_count), 0, coefficient_idx, stride, bytes);
        matrix_store_limb_u64(
            batch_limb(outputs, matrix_limb, limb_count),
            poly_idx,
            coefficient_idx,
            stride,
            bytes,
            mul_mod_u64(value, scalar, moduli[limb]));
    }

    __global__ void matrix_matmul_batch_kernel(
        const MatrixDescriptor *const *left,
        const MatrixDescriptor *const *right,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        size_t limb_count,
        size_t rows,
        size_t inner,
        size_t columns,
        size_t n,
        size_t coefficient_groups)
    {
        __shared__ uint64_t left_tile[kMatmulTileM][kMatmulTileK];
        __shared__ uint64_t right_tile[kMatmulTileK][kMatmulTileN];
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t limb = matrix_limb % limb_count;
        const uint8_t *left_base = batch_limb(left, matrix_limb, limb_count);
        const uint8_t *right_base = batch_limb(right, matrix_limb, limb_count);
        uint8_t *output_base = batch_limb(outputs, matrix_limb, limb_count);
        const size_t stride = left[0][limb].stride;
        const uint8_t bytes = left[0][limb].width;
        const uint64_t modulus = moduli[limb];
        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t column_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t column = column_base + threadIdx.x;
        const int thread = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;
        for (size_t coefficient_idx = first_coefficient;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups)
        {
            uint64_t accumulator = 0;
            for (size_t inner_base = 0; inner_base < inner; inner_base += kMatmulTileK)
            {
                for (int index = thread; index < kMatmulTileM * kMatmulTileK; index += thread_count)
                {
                    const int tile_row = index / kMatmulTileK;
                    const int tile_inner = index % kMatmulTileK;
                    const size_t source_row = row_base + static_cast<size_t>(tile_row);
                    const size_t source_inner = inner_base + static_cast<size_t>(tile_inner);
                    left_tile[tile_row][tile_inner] =
                        source_row < rows && source_inner < inner
                            ? matrix_load_limb_u64(
                                  left_base,
                                  source_row * inner + source_inner,
                                  coefficient_idx,
                                  stride,
                                  bytes)
                            : 0;
                }
                for (int index = thread; index < kMatmulTileK * kMatmulTileN; index += thread_count)
                {
                    const int tile_inner = index / kMatmulTileN;
                    const int tile_column = index % kMatmulTileN;
                    const size_t source_inner = inner_base + static_cast<size_t>(tile_inner);
                    const size_t source_column = column_base + static_cast<size_t>(tile_column);
                    right_tile[tile_inner][tile_column] =
                        source_inner < inner && source_column < columns
                            ? matrix_load_limb_u64(
                                  right_base,
                                  source_inner * columns + source_column,
                                  coefficient_idx,
                                  stride,
                                  bytes)
                            : 0;
                }
                __syncthreads();
                if (row < rows && column < columns)
                {
                    for (int k = 0; k < kMatmulTileK; ++k)
                    {
                        accumulator = add_mod_u64(
                            accumulator,
                            mul_mod_u64(
                                left_tile[threadIdx.y][k],
                                right_tile[k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }
            if (row < rows && column < columns)
            {
                matrix_store_limb_u64(
                    output_base,
                    row * columns + column,
                    coefficient_idx,
                    stride,
                    bytes,
                    accumulator);
            }
        }
    }

    __global__ void matrix_thin_row_matmul_batch_kernel(
        const MatrixDescriptor *const *left,
        const MatrixDescriptor *const *right,
        const MatrixDescriptor *const *outputs,
        const uint64_t *moduli,
        const uint64_t *reciprocals,
        size_t limb_count,
        size_t inner,
        size_t columns,
        size_t n,
        size_t coefficient_groups,
        bool lazy_reduction)
    {
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t limb = matrix_limb % limb_count;
        const size_t column =
            static_cast<size_t>(blockIdx.x) * kThinMatmulColumnsPerBlock + threadIdx.y;
        if (column >= columns)
        {
            return;
        }

        const uint8_t *left_base = batch_limb(left, matrix_limb, limb_count);
        const uint8_t *right_base = batch_limb(right, matrix_limb, limb_count);
        uint8_t *output_base = batch_limb(outputs, matrix_limb, limb_count);
        const size_t stride = left[0][limb].stride;
        const uint8_t bytes = left[0][limb].width;
        const uint64_t modulus = moduli[limb];
        const uint64_t reciprocal = reciprocals[limb];
        for (size_t coefficient_idx = first_coefficient * kThinMatmulWarpSize + threadIdx.x;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups * kThinMatmulWarpSize)
        {
            uint64_t accumulator = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                const uint64_t lhs = matrix_load_limb_u64(
                    left_base,
                    k,
                    coefficient_idx,
                    stride,
                    bytes);
                const uint64_t rhs = matrix_load_limb_u64(
                    right_base,
                    k * columns + column,
                    coefficient_idx,
                    stride,
                    bytes);
                if (lazy_reduction)
                {
                    accumulator += lhs * rhs;
                }
                else
                {
                    accumulator = add_mod_u64(
                        accumulator,
                        mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal),
                        modulus);
                }
            }
            accumulator =
                lazy_reduction ? reduce_barrett_u32(accumulator, modulus, reciprocal) : accumulator;
            matrix_store_limb_u64(
                output_base,
                column,
                coefficient_idx,
                stride,
                bytes,
                accumulator);
        }
    }

    // For canonical a,b and a k-bit modulus (33 <= k <= 62), mu is
    // floor(2^(2k)/modulus). Truncating the product at k-1 bits underestimates
    // the quotient by at most two, so 0 <= remainder < 3*modulus < 2^64.
    // The low-word subtraction is exact modulo 2^64 and needs two corrections.
    __device__ __forceinline__ uint64_t matrix_mul_accumulate_residue(
        uint64_t a, uint64_t b, uint64_t modulus, uint64_t reciprocal)
    {
        if (modulus <= UINT32_MAX)
            return mul_mod_barrett_u32(a, b, modulus, reciprocal);
        if (reciprocal == 0) return mul_mod_u64(a, b, modulus);
        const unsigned int bits = 64U - __clzll(modulus);
        const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
        const uint64_t high = static_cast<uint64_t>(product >> (bits - 1));
        const uint64_t quotient = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(high) * reciprocal) >> (bits + 1));
        uint64_t remainder = static_cast<uint64_t>(product) - quotient * modulus;
        if (remainder >= modulus) remainder -= modulus;
        if (remainder >= modulus) remainder -= modulus;
        return remainder;
    }

    __global__ void matrix_thin_row_mul_accumulate_batch_kernel(
        const MatrixDescriptor *const *left,
        const MatrixDescriptor *const *right,
        const MatrixDescriptor *const *coefficients,
        const MatrixDescriptor *const *biases,
        const MatrixDescriptor *const *outputs,
        const size_t *inner_dimensions,
        const uint64_t *moduli,
        const uint64_t *reciprocals,
        size_t limb_count,
        size_t product_count,
        size_t columns,
        size_t n,
        size_t coefficient_groups,
        bool lazy_reduction)
    {
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t matrix = matrix_limb / limb_count;
        const size_t limb = matrix_limb % limb_count;
        const size_t column =
            static_cast<size_t>(blockIdx.x) * kThinMatmulColumnsPerBlock + threadIdx.y;
        if (column >= columns) return;
        const size_t stride = left[0][limb].stride;
        const uint8_t bytes = left[0][limb].width;
        const uint64_t modulus = moduli[limb];
        const uint64_t reciprocal = reciprocals[limb];
        for (size_t coefficient_idx = first_coefficient * kThinMatmulWarpSize + threadIdx.x;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups * kThinMatmulWarpSize)
        {
            uint64_t total = 0;
            for (size_t product = 0; product < product_count; ++product)
            {
                const size_t product_index = matrix * product_count + product;
                const size_t pointer_index = product_index * limb_count + limb;
                const size_t inner = inner_dimensions[product_index];
                uint64_t dot = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    const uint64_t lhs = matrix_load_limb_u64(
                        batch_limb(left, pointer_index, limb_count), k, coefficient_idx, stride, bytes);
                    const uint64_t rhs = matrix_load_limb_u64(
                        batch_limb(right, pointer_index, limb_count), k * columns + column, coefficient_idx, stride, bytes);
                    if (lazy_reduction)
                    {
                        dot += lhs * rhs;
                    }
                    else
                    {
                        dot = add_mod_u64(
                            dot, matrix_mul_accumulate_residue(lhs, rhs, modulus, reciprocal), modulus);
                    }
                }
                if (lazy_reduction) dot = reduce_barrett_u32(dot, modulus, reciprocal);
                if (batch_limb(coefficients, pointer_index, limb_count))
                {
                    const uint64_t scalar = matrix_load_limb_u64(
                        batch_limb(coefficients, pointer_index, limb_count), 0, coefficient_idx, stride, bytes);
                    dot = matrix_mul_accumulate_residue(dot, scalar, modulus, reciprocal);
                }
                total = add_mod_u64(total, dot, modulus);
            }
            const uint8_t *bias = batch_limb(biases, matrix_limb, limb_count);
            if (bias)
            {
                total = add_mod_u64(
                    total,
                    matrix_load_limb_u64(bias, column, coefficient_idx, stride, bytes),
                    modulus);
            }
            matrix_store_limb_u64(
                batch_limb(outputs, matrix_limb, limb_count), column, coefficient_idx, stride, bytes, total);
        }
    }

    struct MatrixBatchMetadata
    {
        GpuContext *context = nullptr;
        size_t matrix_count = 0;
        size_t limb_count = 0;
        size_t rows = 0;
        size_t inner = 0;
        size_t columns = 0;
        size_t n = 0;
        int level = -1;
        int device = -1;
        cudaStream_t stream = nullptr;
        std::vector<dim3> limb_ids;
        std::vector<size_t> strides;
        std::vector<uint8_t> coefficient_bytes;
        std::vector<const MatrixDescriptor *> left;
        std::vector<const MatrixDescriptor *> right;
        std::vector<const MatrixDescriptor *> outputs;
    };

    struct MatrixBatchUpload
    {
        struct Segment
        {
            void **device;
            const void *host;
            size_t bytes;
        };
        void *device = nullptr;
        cudaStream_t stream = nullptr;

        ~MatrixBatchUpload()
        {
            if (device) cudaFreeAsync(device, stream);
        }

        int initialize(const MatrixBatchMetadata &metadata,
                       std::initializer_list<Segment> segments)
        {
            stream = metadata.stream;
            size_t bytes = 0;
            for (const auto &segment : segments)
            {
                if (segment.bytes > SIZE_MAX - bytes)
                    return set_error("matrix batch submission size overflow");
                bytes += segment.bytes;
            }
            void *host = gpu_pinned_alloc(bytes);
            if (!host) return set_error("matrix batch submission staging allocation failed");
            cudaError_t error = cudaMallocAsync(&device, bytes, stream);
            if (error != cudaSuccess)
            {
                gpu_pinned_free(host);
                return set_error(error);
            }
            size_t offset = 0;
            for (const auto &segment : segments)
            {
                if (segment.bytes != 0)
                {
                    std::memcpy(static_cast<uint8_t *>(host) + offset, segment.host, segment.bytes);
                    *segment.device = static_cast<uint8_t *>(device) + offset;
                    offset += segment.bytes;
                }
            }
            // One logical submission packet, never a matrix-by-limb pointer
            // table. Pinned storage remains owned until this stream completes
            // the upload, including error paths; the caller does not wait.
            error = cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, stream);
            const int reclaim_status = gpu_defer_pinned_frees(
                metadata.context, metadata.device, stream, &host, 1);
            if (error != cudaSuccess) return set_error(error);
            return reclaim_status;
        }
    };

    int prepare_matrix_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count,
        int operation,
        MatrixBatchMetadata *metadata)
    {
        if (!outputs || !left || matrix_count == 0 || !metadata)
        {
            return set_error("invalid matrix batch arguments");
        }
        const GpuMatrix *first_left = left[0];
        const GpuMatrix *first_right = right ? right[0] : nullptr;
        GpuMatrix *first_output = outputs[0];
        if (!first_left || !first_output || !first_left->ctx ||
            (right && !first_right))
        {
            return set_error("null matrix in batch");
        }
        metadata->context = first_left->ctx;
        metadata->matrix_count = matrix_count;
        metadata->limb_count = static_cast<size_t>(first_left->level + 1);
        metadata->rows = first_left->rows;
        const bool multiplication = operation == 1;
        const bool scalar_multiplication = operation == 2;
        metadata->inner = multiplication ? first_left->cols : first_left->rows * first_left->cols;
        metadata->columns = multiplication ? first_right->cols : 1;
        metadata->n = static_cast<size_t>(first_left->ctx->N);
        metadata->level = first_left->level;
        if ((multiplication || scalar_multiplication) &&
            (first_left->format != GPU_POLY_FORMAT_EVAL ||
             first_right->format != GPU_POLY_FORMAT_EVAL))
        {
            return set_error("matrix multiplication batch requires Eval format");
        }
        if (metadata->limb_count == 0 ||
            metadata->limb_count > first_left->ctx->limb_gpu_ids.size())
        {
            return set_error("invalid matrix batch level");
        }
        metadata->limb_ids.assign(
            first_left->ctx->limb_gpu_ids.begin(),
            first_left->ctx->limb_gpu_ids.begin() + metadata->limb_count);
        const size_t constants_slot = metadata->limb_ids[0].x;
        if (constants_slot >= metadata->context->ntt_device_constants.size() ||
            !metadata->context->ntt_device_constants[constants_slot].moduli ||
            metadata->context->ntt_device_constants[constants_slot].limb_count < metadata->limb_count)
            return set_error("missing matrix batch device constants");
        metadata->strides.resize(metadata->limb_count);
        metadata->coefficient_bytes.resize(metadata->limb_count);
        metadata->left.resize(matrix_count);
        if (right) metadata->right.resize(matrix_count);
        metadata->outputs.resize(matrix_count);
        for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
        {
            const GpuMatrix *lhs = left[matrix_idx];
            const GpuMatrix *rhs = right ? right[matrix_idx] : nullptr;
            GpuMatrix *out = outputs[matrix_idx];
            if (!lhs || !out || lhs->ctx != metadata->context || out->ctx != metadata->context ||
                lhs->level != metadata->level || out->level != metadata->level ||
                lhs->format != first_left->format ||
                lhs->rows != metadata->rows ||
                (!multiplication && (lhs->cols != first_left->cols || out->rows != lhs->rows || out->cols != lhs->cols)) ||
                (multiplication && (!rhs || rhs->ctx != metadata->context || rhs->level != metadata->level ||
                                    rhs->format != first_right->format ||
                                    lhs->cols != rhs->rows || rhs->cols != metadata->columns ||
                                    out->rows != metadata->rows || out->cols != metadata->columns)) ||
                (scalar_multiplication && (!rhs || rhs->rows != 1 || rhs->cols != 1)) ||
                (!multiplication && rhs && (rhs->ctx != metadata->context || rhs->level != metadata->level ||
                                             rhs->format != first_right->format ||
                                             (!scalar_multiplication &&
                                              (rhs->rows != lhs->rows || rhs->cols != lhs->cols)))))
            {
                return set_error("matrix batch is not homogeneous");
            }
            const size_t slot = metadata->limb_ids[0].x;
            if (slot >= lhs->shared_limb_buffers.size() || slot >= out->shared_limb_buffers.size() ||
                (rhs && slot >= rhs->shared_limb_buffers.size()))
                return set_error("missing matrix batch descriptors");
            metadata->left[matrix_idx] = lhs->shared_limb_buffers[slot].device_descriptors;
            metadata->outputs[matrix_idx] = out->shared_limb_buffers[slot].device_descriptors;
            if (rhs) metadata->right[matrix_idx] = rhs->shared_limb_buffers[slot].device_descriptors;
            if (!metadata->left[matrix_idx] || !metadata->outputs[matrix_idx] ||
                lhs->shared_limb_buffers[slot].limb_count < metadata->limb_count ||
                out->shared_limb_buffers[slot].limb_count < metadata->limb_count ||
                (rhs && (!metadata->right[matrix_idx] ||
                         rhs->shared_limb_buffers[slot].limb_count < metadata->limb_count)))
                return set_error("missing matrix batch descriptors");
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
            {
                const dim3 limb_id = metadata->limb_ids[limb];
                if (limb_id.x != slot || limb_id.y != limb)
                    return set_error("matrix batch requires ordered limbs in one allocation");
                int lhs_device = -1;
                int out_device = -1;
                cudaStream_t out_stream = nullptr;
                size_t lhs_stride = 0;
                size_t out_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t out_bytes = 0;
                if (matrix_limb_device(lhs, limb_id, &lhs_device) != 0 ||
                    matrix_limb_device(out, limb_id, &out_device) != 0 ||
                    matrix_limb_stream(out, limb_id, &out_stream) != 0 ||
                    !matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride, &lhs_bytes) ||
                    !matrix_limb_metadata_by_id(out, limb_id, &out_stride, &out_bytes) ||
                    lhs_device != out_device || lhs_stride != out_stride || lhs_bytes != out_bytes)
                {
                    return set_error("invalid matrix batch limb");
                }
                if (metadata->device < 0)
                {
                    metadata->device = out_device;
                    metadata->stream = out_stream;
                }
                else if (metadata->device != out_device)
                {
                    return set_error("matrix batch requires one placement");
                }
                if (matrix_idx == 0)
                {
                    metadata->strides[limb] = lhs_stride;
                    metadata->coefficient_bytes[limb] = lhs_bytes;
                }
                else if (metadata->strides[limb] != lhs_stride ||
                         metadata->coefficient_bytes[limb] != lhs_bytes)
                {
                    return set_error("matrix batch limb layout differs");
                }
                if (rhs)
                {
                    int rhs_device = -1;
                    size_t rhs_stride = 0;
                    uint8_t rhs_bytes = 0;
                    if (matrix_limb_device(rhs, limb_id, &rhs_device) != 0 ||
                        !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride, &rhs_bytes) ||
                        rhs_device != metadata->device || rhs_stride != lhs_stride || rhs_bytes != lhs_bytes)
                    {
                        return set_error("invalid right matrix batch limb");
                    }

                }
            }
        }
        cudaError_t error = cudaSetDevice(metadata->device);
        if (error != cudaSuccess || !metadata->stream)
        {
            return error != cudaSuccess ? set_error(error) : set_error("null matrix batch stream");
        }
        for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
        {
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
            {
                const dim3 limb_id = metadata->limb_ids[limb];
                int status = matrix_wait_limb_stream(left[matrix_idx], limb_id, metadata->device, metadata->stream);
                if (status == 0 && right)
                {
                    status = matrix_wait_limb_stream(right[matrix_idx], limb_id, metadata->device, metadata->stream);
                }
                if (status == 0)
                {
                    // Other outputs can initialize their persistent descriptors
                    // on independent allocation streams. Order those writes
                    // before this batch reads the descriptors and writes data.
                    status = matrix_wait_limb_stream(outputs[matrix_idx], limb_id, metadata->device, metadata->stream);
                }
                if (status != 0)
                {
                    return status;
                }
            }
        }
        return 0;
    }

    int finish_matrix_batch(
        const MatrixBatchMetadata &metadata,
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right)
    {
        for (size_t matrix_idx = 0; matrix_idx < metadata.matrix_count; ++matrix_idx)
        {
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const dim3 limb_id = metadata.limb_ids[limb];
                int status = matrix_track_limb_consumer(left[matrix_idx], limb_id, metadata.device, metadata.stream);
                if (status == 0 && right)
                {
                    status = matrix_track_limb_consumer(right[matrix_idx], limb_id, metadata.device, metadata.stream);
                }
                if (status == 0)
                {
                    status = matrix_record_limb_write(outputs[matrix_idx], limb_id, metadata.stream);
                }
                if (status != 0)
                {
                    return status;
                }
            }
        }
        return 0;
    }
}

extern "C" int gpu_matrix_binary_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    size_t matrix_count,
    int operation)
{
    if (operation != 0 && operation != 1)
    {
        return set_error("invalid matrix binary batch operation");
    }
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, left, right, matrix_count, 0, &metadata);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const MatrixDescriptor **d_left = nullptr;
    const MatrixDescriptor **d_right = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_left), metadata.left.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_right), metadata.right.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_binary_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_left, d_right, d_outputs, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n, operation);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = left[matrix_idx]->format;
    return status;
}

extern "C" int gpu_matrix_negate_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t matrix_count)
{
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, inputs, nullptr, matrix_count, 0, &metadata);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const MatrixDescriptor **d_inputs = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_inputs), metadata.left.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_negate_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = inputs[matrix_idx]->format;
    return status;
}

extern "C" int gpu_matrix_validate_ring_automorphism(
    size_t ring_dimension,
    const size_t *indices,
    size_t matrix_count)
{
    if (ring_dimension == 0 || (ring_dimension & (ring_dimension - 1)) != 0)
        return set_error("ring automorphism requires a power-of-two ring dimension");
    if (matrix_count != 0 && !indices)
        return set_error("null ring automorphism indices");
    if (ring_dimension > SIZE_MAX / 2)
        return set_error("ring automorphism dimension overflow");
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        if (indices[matrix] == 0 || indices[matrix] >= 2 * ring_dimension || indices[matrix] % 2 == 0)
            return set_error("invalid ring automorphism index");
    }
    return 0;
}

extern "C" int gpu_matrix_ring_automorphism_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    const size_t *indices,
    size_t matrix_count)
{
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, inputs, nullptr, matrix_count, 0, &metadata);
    if (status != 0) return status;
    if (inputs[0]->format != GPU_POLY_FORMAT_COEFF)
        return set_error("ring automorphism requires coefficient format");
    status = gpu_matrix_validate_ring_automorphism(metadata.n, indices, matrix_count);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const MatrixDescriptor **d_inputs = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    size_t *d_indices = nullptr;
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_inputs), metadata.left.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_indices), indices, matrix_count * sizeof(size_t)},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_ring_automorphism_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_moduli, d_indices,
        metadata.limb_count, coefficients_per_limb, total_coefficients, metadata.n);
    error = cudaGetLastError();
    if (error != cudaSuccess) { return set_error(error); }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) outputs[matrix]->format = GPU_POLY_FORMAT_COEFF;
    return status;
}

extern "C" int gpu_matrix_mul_scalar_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *matrices,
    const GpuMatrix *const *scalars,
    size_t matrix_count)
{
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, matrices, scalars, matrix_count, 2, &metadata);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    if (total_coefficients == 0) return 0;
    const MatrixDescriptor **d_matrices = nullptr;
    const MatrixDescriptor **d_scalars = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_matrices), metadata.left.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_scalars), metadata.right.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_scalar_mul_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_matrices,
        d_scalars,
        d_outputs,
        d_moduli,
        metadata.limb_count,
        coefficients_per_limb,
        total_coefficients,
        metadata.n);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, matrices, scalars);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        outputs[matrix_idx]->format = matrices[matrix_idx]->format;
    }
    return status;
}

extern "C" int gpu_matrix_mul_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    size_t matrix_count)
{
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, left, right, matrix_count, 1, &metadata);
    if (status != 0) return status;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    if (pointer_count > 65535)
    {
        return set_error("matrix multiplication batch exceeds CUDA grid depth");
    }
    const MatrixDescriptor **d_left = nullptr;
    const MatrixDescriptor **d_right = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    uint64_t *d_reciprocals = nullptr;
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool use_thin_row_kernel = metadata.rows == 1;
    bool lazy_reduction = use_thin_row_kernel;
    if (use_thin_row_kernel)
    {
        for (size_t limb = 0; limb < metadata.limb_count; ++limb)
        {
            if (!matrix_barrett_u32_reciprocal(metadata.context->moduli[limb], &reciprocals[limb]))
            {
                use_thin_row_kernel = false;
                break;
            }
            lazy_reduction =
                lazy_reduction && matrix_lazy_dot_u64(metadata.inner, metadata.context->moduli[limb]);
        }
    }
    const size_t groups_per_matrix = use_thin_row_kernel
                                         ? (metadata.n + kThinMatmulWarpSize - 1) /
                                               kThinMatmulWarpSize
                                         : metadata.n;
    const size_t coefficient_groups =
        std::min(groups_per_matrix, static_cast<size_t>(65535) / pointer_count);
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_left), metadata.left.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_right), metadata.right.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_reciprocals), reciprocals.data(), use_thin_row_kernel ? metadata.limb_count * sizeof(uint64_t) : 0},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    if (use_thin_row_kernel)
    {
        const dim3 block(kThinMatmulWarpSize, kThinMatmulColumnsPerBlock, 1);
        const dim3 grid(
            static_cast<unsigned int>(
                (metadata.columns + kThinMatmulColumnsPerBlock - 1) /
                kThinMatmulColumnsPerBlock),
            1,
            static_cast<unsigned int>(pointer_count * coefficient_groups));
        matrix_thin_row_matmul_batch_kernel<<<grid, block, 0, metadata.stream>>>(
            d_left, d_right, d_outputs, d_moduli, d_reciprocals,
            metadata.limb_count, metadata.inner, metadata.columns, metadata.n,
            coefficient_groups, lazy_reduction);
    }
    else
    {
        const dim3 block(kMatmulTileN, kMatmulTileM, 1);
        const dim3 grid(
            static_cast<unsigned int>((metadata.columns + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((metadata.rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(pointer_count * coefficient_groups));
        matrix_matmul_batch_kernel<<<grid, block, 0, metadata.stream>>>(
            d_left, d_right, d_outputs, d_moduli, metadata.limb_count,
            metadata.rows, metadata.inner, metadata.columns, metadata.n, coefficient_groups);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = GPU_POLY_FORMAT_EVAL;
    return status;
}

extern "C" int gpu_matrix_mul_accumulate_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    const GpuMatrix *const *coefficients,
    const GpuMatrix *const *biases,
    const size_t *inner_dimensions,
    size_t matrix_count,
    size_t product_count)
{
    if (!outputs || !left || !right || !coefficients || !biases || !inner_dimensions ||
        matrix_count == 0 || product_count == 0)
    {
        return set_error("invalid multiply-accumulate batch arguments");
    }
    std::vector<const GpuMatrix *> first_left(matrix_count);
    std::vector<const GpuMatrix *> first_right(matrix_count);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        first_left[matrix] = left[matrix * product_count];
        first_right[matrix] = right[matrix * product_count];
    }
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(
        outputs, first_left.data(), first_right.data(), matrix_count, 1, &metadata);
    if (status != 0) return status;
    if (metadata.rows != 1) return set_error("multiply-accumulate requires row matrices");
    const size_t product_total = matrix_count * product_count;
    const size_t output_pointer_count = matrix_count * metadata.limb_count;
    if (output_pointer_count > 65535)
    {
        return set_error("multiply-accumulate batch exceeds CUDA grid depth");
    }
    std::vector<const MatrixDescriptor *> host_left(product_total);
    std::vector<const MatrixDescriptor *> host_right(product_total);
    std::vector<const MatrixDescriptor *> host_coefficients(product_total, nullptr);
    std::vector<const MatrixDescriptor *> host_biases(matrix_count, nullptr);
    size_t maximum_inner = 0;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        for (size_t product = 0; product < product_count; ++product)
        {
            const size_t product_index = matrix * product_count + product;
            const GpuMatrix *lhs = left[product_index];
            const GpuMatrix *rhs = right[product_index];
            const GpuMatrix *coefficient = coefficients[product_index];
            const size_t inner = inner_dimensions[product_index];
            maximum_inner = std::max(maximum_inner, inner);
            if (!lhs || !rhs || lhs->ctx != metadata.context || rhs->ctx != metadata.context ||
                lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL ||
                lhs->level != metadata.level || rhs->level != metadata.level ||
                lhs->rows != 1 || lhs->cols != inner || rhs->rows != inner ||
                rhs->cols != metadata.columns ||
                (coefficient && (coefficient->ctx != metadata.context ||
                                 coefficient->level != metadata.level ||
                                 coefficient->format != GPU_POLY_FORMAT_EVAL ||
                                 coefficient->rows != 1 || coefficient->cols != 1)))
            {
                return set_error("invalid multiply-accumulate product layout");
            }
            const size_t slot = metadata.limb_ids[0].x;
            if (slot >= lhs->shared_limb_buffers.size() || slot >= rhs->shared_limb_buffers.size() ||
                (coefficient && slot >= coefficient->shared_limb_buffers.size()))
                return set_error("missing multiply-accumulate descriptors");
            host_left[product_index] = lhs->shared_limb_buffers[slot].device_descriptors;
            host_right[product_index] = rhs->shared_limb_buffers[slot].device_descriptors;
            if (coefficient)
                host_coefficients[product_index] = coefficient->shared_limb_buffers[slot].device_descriptors;
            if (lhs->shared_limb_buffers[slot].limb_count < metadata.limb_count ||
                rhs->shared_limb_buffers[slot].limb_count < metadata.limb_count ||
                (coefficient && coefficient->shared_limb_buffers[slot].limb_count < metadata.limb_count))
                return set_error("incomplete multiply-accumulate descriptors");
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const dim3 limb_id = metadata.limb_ids[limb];
                int lhs_device = -1;
                int rhs_device = -1;
                size_t lhs_stride = 0;
                size_t rhs_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t rhs_bytes = 0;
                if (!host_left[product_index] || !host_right[product_index] ||
                    (coefficient && !host_coefficients[product_index]) ||
                    matrix_limb_device(lhs, limb_id, &lhs_device) != 0 ||
                    matrix_limb_device(rhs, limb_id, &rhs_device) != 0 ||
                    !matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride, &lhs_bytes) ||
                    !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride, &rhs_bytes) ||
                    lhs_device != metadata.device || rhs_device != metadata.device ||
                    lhs_stride != metadata.strides[limb] ||
                    rhs_stride != metadata.strides[limb] ||
                    lhs_bytes != metadata.coefficient_bytes[limb] ||
                    rhs_bytes != metadata.coefficient_bytes[limb])
                    return set_error("invalid multiply-accumulate limb pointer");
                status = matrix_wait_limb_stream(lhs, limb_id, metadata.device, metadata.stream);
                if (status == 0)
                    status = matrix_wait_limb_stream(rhs, limb_id, metadata.device, metadata.stream);
                if (status == 0 && coefficient)
                    status = matrix_wait_limb_stream(
                        coefficient, limb_id, metadata.device, metadata.stream);
                if (status != 0) return status;
            }
        }
        const GpuMatrix *bias = biases[matrix];
        if (bias && (bias->ctx != metadata.context || bias->level != metadata.level ||
                     bias->format != GPU_POLY_FORMAT_EVAL || bias->rows != 1 ||
                     bias->cols != metadata.columns))
            return set_error("invalid multiply-accumulate bias layout");
        if (bias)
        {
            const size_t slot = metadata.limb_ids[0].x;
            if (slot >= bias->shared_limb_buffers.size())
                return set_error("missing multiply-accumulate bias descriptors");
            host_biases[matrix] = bias->shared_limb_buffers[slot].device_descriptors;
            if (!host_biases[matrix] || bias->shared_limb_buffers[slot].limb_count < metadata.limb_count)
                return set_error("missing multiply-accumulate bias descriptors");
        }
        for (size_t limb = 0; limb < metadata.limb_count && bias; ++limb)
        {
            const dim3 limb_id = metadata.limb_ids[limb];
            status = matrix_wait_limb_stream(bias, limb_id, metadata.device, metadata.stream);
            if (status != 0) return status;
        }
    }
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool lazy_reduction = true;
    for (size_t limb = 0; limb < metadata.limb_count; ++limb)
    {
        const bool small_modulus =
            matrix_barrett_u32_reciprocal(metadata.context->moduli[limb], &reciprocals[limb]);
        if (!small_modulus)
        {
            const unsigned int bits = 64U - __builtin_clzll(metadata.context->moduli[limb]);
            reciprocals[limb] = bits <= 62
                ? static_cast<uint64_t>((static_cast<unsigned __int128>(1) << (2 * bits)) /
                                        metadata.context->moduli[limb])
                : 0;
        }
        lazy_reduction = lazy_reduction && small_modulus &&
            matrix_lazy_dot_u64(maximum_inner, metadata.context->moduli[limb]);
    }
    const MatrixDescriptor **d_left = nullptr;
    const MatrixDescriptor **d_right = nullptr;
    const MatrixDescriptor **d_coefficients = nullptr;
    const MatrixDescriptor **d_biases = nullptr;
    const MatrixDescriptor **d_outputs = nullptr;
    size_t *d_inner = nullptr;
    const uint64_t *d_moduli = metadata.context->ntt_device_constants[metadata.limb_ids[0].x].moduli;
    uint64_t *d_reciprocals = nullptr;
    MatrixBatchUpload upload;
    status = upload.initialize(metadata, {
        {reinterpret_cast<void **>(&d_left), host_left.data(), product_total * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_right), host_right.data(), product_total * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_coefficients), host_coefficients.data(), product_total * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_biases), host_biases.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_outputs), metadata.outputs.data(), matrix_count * sizeof(MatrixDescriptor *)},
        {reinterpret_cast<void **>(&d_inner), inner_dimensions, product_total * sizeof(size_t)},
        {reinterpret_cast<void **>(&d_reciprocals), reciprocals.data(), metadata.limb_count * sizeof(uint64_t)},
    });
    if (status != 0) return status;
    cudaError_t error = cudaSuccess;
    const size_t groups_per_matrix = (metadata.n + kThinMatmulWarpSize - 1) / kThinMatmulWarpSize;
    const size_t coefficient_groups =
        std::min(groups_per_matrix, static_cast<size_t>(65535) / output_pointer_count);
    const dim3 block(kThinMatmulWarpSize, kThinMatmulColumnsPerBlock, 1);
    const dim3 grid(
        static_cast<unsigned int>((metadata.columns + kThinMatmulColumnsPerBlock - 1) /
                                  kThinMatmulColumnsPerBlock),
        1,
        static_cast<unsigned int>(output_pointer_count * coefficient_groups));
    matrix_thin_row_mul_accumulate_batch_kernel<<<grid, block, 0, metadata.stream>>>(
        d_left, d_right, d_coefficients, d_biases, d_outputs, d_inner,
        d_moduli, d_reciprocals, metadata.limb_count, product_count, metadata.columns,
        metadata.n, coefficient_groups, lazy_reduction);
    error = cudaGetLastError();
    if (error != cudaSuccess) { return set_error(error); }
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        for (size_t limb = 0; limb < metadata.limb_count; ++limb)
        {
            const dim3 limb_id = metadata.limb_ids[limb];
            for (size_t product = 0; product < product_count; ++product)
            {
                const size_t index = matrix * product_count + product;
                status = matrix_track_limb_consumer(left[index], limb_id, metadata.device, metadata.stream);
                if (status == 0) status = matrix_track_limb_consumer(right[index], limb_id, metadata.device, metadata.stream);
                if (status == 0 && coefficients[index]) status = matrix_track_limb_consumer(coefficients[index], limb_id, metadata.device, metadata.stream);
                if (status != 0) { return status; }
            }
            if (biases[matrix])
            {
                status = matrix_track_limb_consumer(biases[matrix], limb_id, metadata.device, metadata.stream);
                if (status != 0) { return status; }
            }
            status = matrix_record_limb_write(outputs[matrix], limb_id, metadata.stream);
            if (status != 0) { return status; }
        }
        outputs[matrix]->format = GPU_POLY_FORMAT_EVAL;
    }
    return 0;
}
