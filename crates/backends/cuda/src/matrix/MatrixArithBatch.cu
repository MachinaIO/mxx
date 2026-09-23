namespace
{
    template <class T>
    cudaError_t matrix_batch_upload(GpuContext *, void *destination, const T *source, size_t bytes, cudaStream_t stream)
    {
        return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
    }

    template <typename Metadata, size_t PointerCount>
    cudaError_t matrix_lane_metadata_upload(
        GpuContext *ctx,
        Metadata *destination,
        const std::vector<Metadata> &source,
        size_t lane_count,
        size_t limb_count,
        uint64_t active_lane_mask,
        const size_t (&pointer_offsets)[PointerCount],
        const uint32_t *binding_indices,
        size_t binding_count,
        cudaStream_t stream)
    {
        return cudaMemcpyAsync(destination, source.data(),
            source.size() * sizeof(Metadata), cudaMemcpyHostToDevice, stream);
    }
}

extern "C" int gpu_matrix_family_descriptor_table_bind_live_sources(
    GpuContext *ctx,
    void *destination,
    const uint64_t *source_descriptors,
    size_t source_count,
    const uint32_t *source_binding_indices,
    size_t source_binding_count,
    cudaStream_t stream)
{
    if (!ctx || !destination || !source_descriptors || !stream || source_count == 0 ||
        source_count > static_cast<size_t>(-1) / sizeof(uint64_t))
        return set_error("invalid family descriptor live-source binding arguments");
    for (size_t index = 0; index < source_count; ++index)
        if (source_descriptors[index] == 0)
            return set_error("family descriptor live-source address is null");
    const cudaError_t error = cudaMemcpyAsync(destination, source_descriptors,
        source_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess ? 0 : set_error(error);
}

namespace
{
    __global__ void matrix_binary_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t lhs = matrix_load_limb_u64(
            left[matrix_limb], poly_idx, coefficient_idx, stride, bytes);
        const uint64_t rhs = matrix_load_limb_u64(
            right[matrix_limb], poly_idx, coefficient_idx, stride, bytes);
        const uint64_t value = operation == 0
                                   ? add_mod_u64(lhs, rhs, moduli[limb])
                                   : sub_mod_u64(lhs, rhs, moduli[limb]);
        matrix_store_limb_u64(
            outputs[matrix_limb], poly_idx, coefficient_idx, stride, bytes, value);
    }

    __global__ void matrix_negate_batch_kernel(
        const uint8_t *const *inputs,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t value = matrix_load_limb_u64(
            inputs[matrix_limb], poly_idx, coefficient_idx, stride, bytes);
        const uint64_t negated = value == 0 ? 0 : moduli[limb] - value;
        matrix_store_limb_u64(
            outputs[matrix_limb], poly_idx, coefficient_idx, stride, bytes, negated);
    }

    __global__ void matrix_ring_automorphism_batch_kernel(
        const uint8_t *const *inputs,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t modulus = moduli[limb];
        uint64_t value = matrix_load_limb_u64(inputs[matrix_limb], polynomial, source, stride, bytes);
        if (exponent >= n && value != 0) value = modulus - value;
        matrix_store_limb_u64(outputs[matrix_limb], polynomial, target, stride, bytes, value);
    }

    __global__ void matrix_scalar_mul_batch_kernel(
        const uint8_t *const *matrices,
        const uint8_t *const *scalars,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t value = matrix_load_limb_u64(
            matrices[matrix_limb], poly_idx, coefficient_idx, stride, bytes);
        const uint64_t scalar = matrix_load_limb_u64(
            scalars[matrix_limb], 0, coefficient_idx, stride, bytes);
        matrix_store_limb_u64(
            outputs[matrix_limb],
            poly_idx,
            coefficient_idx,
            stride,
            bytes,
            mul_mod_u64(value, scalar, moduli[limb]));
    }

    __global__ void matrix_matmul_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const uint8_t *left_base = left[matrix_limb];
        const uint8_t *right_base = right[matrix_limb];
        uint8_t *output_base = outputs[matrix_limb];
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
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
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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

        const uint8_t *left_base = left[matrix_limb];
        const uint8_t *right_base = right[matrix_limb];
        uint8_t *output_base = outputs[matrix_limb];
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
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

    __global__ void matrix_thin_row_mul_accumulate_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        const uint8_t *const *coefficients,
        const uint8_t *const *biases,
        uint8_t *const *outputs,
        const size_t *inner_dimensions,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
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
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
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
                        left[pointer_index], k, coefficient_idx, stride, bytes);
                    const uint64_t rhs = matrix_load_limb_u64(
                        right[pointer_index], k * columns + column, coefficient_idx, stride, bytes);
                    if (lazy_reduction)
                    {
                        dot += lhs * rhs;
                    }
                    else
                    {
                        dot = add_mod_u64(
                            dot, mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal), modulus);
                    }
                }
                if (lazy_reduction) dot = reduce_barrett_u32(dot, modulus, reciprocal);
                if (coefficients[pointer_index])
                {
                    const uint64_t scalar = matrix_load_limb_u64(
                        coefficients[pointer_index], 0, coefficient_idx, stride, bytes);
                    dot = mul_mod_u64(dot, scalar, modulus);
                }
                total = add_mod_u64(total, dot, modulus);
            }
            const uint8_t *bias = biases[matrix_limb];
            if (bias)
            {
                total = add_mod_u64(
                    total,
                    matrix_load_limb_u64(bias, column, coefficient_idx, stride, bytes),
                    modulus);
            }
            matrix_store_limb_u64(
                outputs[matrix_limb], column, coefficient_idx, stride, bytes, total);
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
        std::vector<uint64_t> moduli;
        std::vector<const uint8_t *> left;
        std::vector<const uint8_t *> right;
        std::vector<uint8_t *> outputs;
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
        metadata->strides.resize(metadata->limb_count);
        metadata->coefficient_bytes.resize(metadata->limb_count);
        metadata->moduli.assign(
            first_left->ctx->moduli.begin(),
            first_left->ctx->moduli.begin() + metadata->limb_count);
        metadata->left.resize(matrix_count * metadata->limb_count);
        if (right) metadata->right.resize(matrix_count * metadata->limb_count);
        metadata->outputs.resize(matrix_count * metadata->limb_count);
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
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
            {
                const dim3 limb_id = metadata->limb_ids[limb];
                int lhs_device = -1;
                int out_device = -1;
                cudaStream_t out_stream = nullptr;
                size_t lhs_stride = 0;
                size_t out_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t out_bytes = 0;
                const uint8_t *lhs_pointer = matrix_limb_ptr_by_id(lhs, 0, limb_id);
                uint8_t *out_pointer = matrix_limb_ptr_by_id(out, 0, limb_id);
                if (!lhs_pointer || !out_pointer ||
                    matrix_limb_device(lhs, limb_id, &lhs_device) != 0 ||
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
                const size_t flat = matrix_idx * metadata->limb_count + limb;
                metadata->left[flat] = lhs_pointer;
                metadata->outputs[flat] = out_pointer;
                if (rhs)
                {
                    int rhs_device = -1;
                    size_t rhs_stride = 0;
                    uint8_t rhs_bytes = 0;
                    const uint8_t *rhs_pointer = matrix_limb_ptr_by_id(rhs, 0, limb_id);
                    if (!rhs_pointer || matrix_limb_device(rhs, limb_id, &rhs_device) != 0 ||
                        !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride, &rhs_bytes) ||
                        rhs_device != metadata->device || rhs_stride != lhs_stride || rhs_bytes != lhs_bytes)
                    {
                        return set_error("invalid right matrix batch limb");
                    }
                    metadata->right[flat] = rhs_pointer;
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
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    auto release = [&]() {
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_right) cudaFreeAsync(d_right, metadata.stream);
        if (d_left) cudaFreeAsync(d_left, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&d_left), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_right), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_outputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_binary_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n, operation);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = left[matrix_idx]->format;
    release();
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
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_inputs = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    auto release = [&]() {
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_inputs) cudaFreeAsync(d_inputs, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&d_inputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_outputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_inputs, metadata.left.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_negate_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n);
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
    {
        release();
        return set_error(launch_error);
    }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = inputs[matrix_idx]->format;
    release();
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
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_inputs = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    size_t *d_indices = nullptr;
    auto release = [&]() {
        if (d_indices) cudaFreeAsync(d_indices, metadata.stream);
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_inputs) cudaFreeAsync(d_inputs, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&d_inputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_outputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_indices), matrix_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_inputs, metadata.left.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_indices, indices, matrix_count * sizeof(size_t), metadata.stream);
    if (error != cudaSuccess) { release(); return set_error(error); }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_ring_automorphism_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_strides, d_bytes, d_moduli, d_indices,
        metadata.limb_count, coefficients_per_limb, total_coefficients, metadata.n);
    error = cudaGetLastError();
    if (error != cudaSuccess) { release(); return set_error(error); }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) outputs[matrix]->format = GPU_POLY_FORMAT_COEFF;
    release();
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
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_matrices = nullptr;
    const uint8_t **d_scalars = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    auto release = [&]() {
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_scalars) cudaFreeAsync(d_scalars, metadata.stream);
        if (d_matrices) cudaFreeAsync(d_matrices, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_matrices), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_scalars), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_outputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_matrices, metadata.left.data(), pointer_count * sizeof(uint8_t *),
        metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_scalars, metadata.right.data(), pointer_count * sizeof(uint8_t *),
        metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *),
        metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t),
        metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count,
        metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx,
        d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t),
        metadata.stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_scalar_mul_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_matrices,
        d_scalars,
        d_outputs,
        d_strides,
        d_bytes,
        d_moduli,
        metadata.limb_count,
        coefficients_per_limb,
        total_coefficients,
        metadata.n);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, matrices, scalars);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        outputs[matrix_idx]->format = matrices[matrix_idx]->format;
    }
    release();
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
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_reciprocals = nullptr;
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool use_thin_row_kernel = metadata.rows == 1;
    bool lazy_reduction = use_thin_row_kernel;
    if (use_thin_row_kernel)
    {
        for (size_t limb = 0; limb < metadata.limb_count; ++limb)
        {
            if (!matrix_barrett_u32_reciprocal(metadata.moduli[limb], &reciprocals[limb]))
            {
                use_thin_row_kernel = false;
                break;
            }
            lazy_reduction =
                lazy_reduction && matrix_lazy_dot_u64(metadata.inner, metadata.moduli[limb]);
        }
    }
    const size_t groups_per_matrix = use_thin_row_kernel
                                         ? (metadata.n + kThinMatmulWarpSize - 1) /
                                               kThinMatmulWarpSize
                                         : metadata.n;
    const size_t coefficient_groups =
        std::min(groups_per_matrix, static_cast<size_t>(65535) / pointer_count);
    auto release = [&]() {
        if (d_reciprocals) cudaFreeAsync(d_reciprocals, metadata.stream);
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_right) cudaFreeAsync(d_right, metadata.stream);
        if (d_left) cudaFreeAsync(d_left, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&d_left), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_right), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_outputs), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess && use_thin_row_kernel) error = cudaMallocAsync(reinterpret_cast<void **>(&d_reciprocals), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess && use_thin_row_kernel) error = matrix_batch_upload(outputs[0]->ctx, d_reciprocals, reciprocals.data(), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
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
            d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, d_reciprocals,
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
            d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
            metadata.rows, metadata.inner, metadata.columns, metadata.n, coefficient_groups);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = GPU_POLY_FORMAT_EVAL;
    release();
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
    const size_t product_pointer_count = product_total * metadata.limb_count;
    const size_t output_pointer_count = matrix_count * metadata.limb_count;
    if (output_pointer_count > 65535)
    {
        return set_error("multiply-accumulate batch exceeds CUDA grid depth");
    }
    std::vector<const uint8_t *> host_left(product_pointer_count);
    std::vector<const uint8_t *> host_right(product_pointer_count);
    std::vector<const uint8_t *> host_coefficients(product_pointer_count, nullptr);
    std::vector<const uint8_t *> host_biases(output_pointer_count, nullptr);
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
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const dim3 limb_id = metadata.limb_ids[limb];
                const size_t pointer_index = product_index * metadata.limb_count + limb;
                host_left[pointer_index] = matrix_limb_ptr_by_id(lhs, 0, limb_id);
                host_right[pointer_index] = matrix_limb_ptr_by_id(rhs, 0, limb_id);
                if (coefficient)
                    host_coefficients[pointer_index] = matrix_limb_ptr_by_id(coefficient, 0, limb_id);
                int lhs_device = -1;
                int rhs_device = -1;
                size_t lhs_stride = 0;
                size_t rhs_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t rhs_bytes = 0;
                if (!host_left[pointer_index] || !host_right[pointer_index] ||
                    (coefficient && !host_coefficients[pointer_index]) ||
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
        for (size_t limb = 0; limb < metadata.limb_count && bias; ++limb)
        {
            const dim3 limb_id = metadata.limb_ids[limb];
            host_biases[matrix * metadata.limb_count + limb] =
                matrix_limb_ptr_by_id(bias, 0, limb_id);
            if (!host_biases[matrix * metadata.limb_count + limb])
                return set_error("invalid multiply-accumulate bias pointer");
            status = matrix_wait_limb_stream(bias, limb_id, metadata.device, metadata.stream);
            if (status != 0) return status;
        }
    }
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool lazy_reduction = true;
    for (size_t limb = 0; limb < metadata.limb_count; ++limb)
    {
        if (!matrix_barrett_u32_reciprocal(metadata.moduli[limb], &reciprocals[limb]))
            return set_error("multiply-accumulate requires 32-bit CRT moduli");
        lazy_reduction = lazy_reduction &&
            matrix_lazy_dot_u64(maximum_inner, metadata.moduli[limb]);
    }
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    const uint8_t **d_coefficients = nullptr;
    const uint8_t **d_biases = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_inner = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_reciprocals = nullptr;
    auto release = [&]() {
        if (d_reciprocals) cudaFreeAsync(d_reciprocals, metadata.stream);
        if (d_moduli) cudaFreeAsync(d_moduli, metadata.stream);
        if (d_bytes) cudaFreeAsync(d_bytes, metadata.stream);
        if (d_strides) cudaFreeAsync(d_strides, metadata.stream);
        if (d_inner) cudaFreeAsync(d_inner, metadata.stream);
        if (d_outputs) cudaFreeAsync(d_outputs, metadata.stream);
        if (d_biases) cudaFreeAsync(d_biases, metadata.stream);
        if (d_coefficients) cudaFreeAsync(d_coefficients, metadata.stream);
        if (d_right) cudaFreeAsync(d_right, metadata.stream);
        if (d_left) cudaFreeAsync(d_left, metadata.stream);
    };
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&d_left), product_pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_right), product_pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_coefficients), product_pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_biases), output_pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_outputs), output_pointer_count * sizeof(uint8_t *), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_inner), product_total * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_strides), metadata.limb_count * sizeof(size_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_bytes), metadata.limb_count, metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_moduli), metadata.limb_count * sizeof(uint64_t), metadata.stream);
    if (error == cudaSuccess) error = cudaMallocAsync(reinterpret_cast<void **>(&d_reciprocals), metadata.limb_count * sizeof(uint64_t), metadata.stream);
#define COPY_ASYNC(dst, src, bytes) if (error == cudaSuccess) error = matrix_batch_upload(outputs[0]->ctx, dst, src, bytes, metadata.stream)
    COPY_ASYNC(d_left, host_left.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_right, host_right.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_coefficients, host_coefficients.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_biases, host_biases.data(), output_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_outputs, metadata.outputs.data(), output_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_inner, inner_dimensions, product_total * sizeof(size_t));
    COPY_ASYNC(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t));
    COPY_ASYNC(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count);
    COPY_ASYNC(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t));
    COPY_ASYNC(d_reciprocals, reciprocals.data(), metadata.limb_count * sizeof(uint64_t));
#undef COPY_ASYNC
    if (error != cudaSuccess) { release(); return set_error(error); }
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
        d_left, d_right, d_coefficients, d_biases, d_outputs, d_inner, d_strides, d_bytes,
        d_moduli, d_reciprocals, metadata.limb_count, product_count, metadata.columns,
        metadata.n, coefficient_groups, lazy_reduction);
    error = cudaGetLastError();
    if (error != cudaSuccess) { release(); return set_error(error); }
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
                if (status != 0) { release(); return status; }
            }
            if (biases[matrix])
            {
                status = matrix_track_limb_consumer(biases[matrix], limb_id, metadata.device, metadata.stream);
                if (status != 0) { release(); return status; }
            }
            status = matrix_record_limb_write(outputs[matrix], limb_id, metadata.stream);
            if (status != 0) { release(); return status; }
        }
        outputs[matrix]->format = GPU_POLY_FORMAT_EVAL;
    }
    release();
    return 0;
}

namespace
{
    // One record describes one lane and one CRT limb.  Keeping the physical
    // descriptor alongside the data pointer makes the binding identity
    // explicit even though the elementwise kernel only needs the data view.
    struct MatrixLaneBinaryMetadata
    {
        const uint8_t *lhs_data;
        const uint8_t *rhs_data;
        uint8_t *out_data;
        const void *lhs_descriptors;
        const void *rhs_descriptors;
        const void *out_descriptors;
        size_t lhs_stride_bytes;
        size_t rhs_stride_bytes;
        size_t out_stride_bytes;
        size_t rows;
        size_t columns;
        uint8_t lhs_coefficient_bytes;
        uint8_t rhs_coefficient_bytes;
        uint8_t out_coefficient_bytes;
        uint8_t rhs_is_scalar;
        uint8_t reserved[4];
    };

    __global__ void matrix_binary_lane_batch_kernel(
        const MatrixLaneBinaryMetadata *metadata,
        const uint64_t *moduli,
        size_t lane_count,
        size_t limb_count,
        size_t n,
        int operation,
        uint64_t active_lane_mask)
    {
        const size_t lane = static_cast<size_t>(blockIdx.y);
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (lane >= lane_count || limb >= limb_count ||
            ((active_lane_mask >> lane) & UINT64_C(1)) == 0)
        {
            return;
        }
        const MatrixLaneBinaryMetadata &entry = metadata[lane * limb_count + limb];
        const size_t polynomial_count = entry.rows * entry.columns;
        const size_t total = polynomial_count * n;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < total;
             index += stride)
        {
            const size_t polynomial = index / n;
            const size_t coefficient = index - polynomial * n;
            const uint64_t lhs = matrix_load_limb_u64(
                entry.lhs_data,
                polynomial,
                coefficient,
                entry.lhs_stride_bytes,
                entry.lhs_coefficient_bytes);
            const uint64_t rhs = matrix_load_limb_u64(
                entry.rhs_data,
                entry.rhs_is_scalar ? 0 : polynomial,
                coefficient,
                entry.rhs_stride_bytes,
                entry.rhs_coefficient_bytes);
            const uint64_t modulus = moduli[limb];
            const uint64_t value = operation == 0
                                       ? add_mod_u64(lhs, rhs, modulus)
                                       : sub_mod_u64(lhs, rhs, modulus);
            matrix_store_limb_u64(
                entry.out_data,
                polynomial,
                coefficient,
                entry.out_stride_bytes,
                entry.out_coefficient_bytes,
                value);
        }
    }
}

namespace
{
    bool lane_mask_is_valid(size_t lane_count, size_t active_count, uint64_t mask)
    {
        if (lane_count == 0 || lane_count > 64 || active_count > lane_count)
        {
            return false;
        }
        const uint64_t lane_bits = lane_count == 64
                                       ? UINT64_MAX
                                       : ((UINT64_C(1) << lane_count) - UINT64_C(1));
        return (mask & ~lane_bits) == 0 &&
               static_cast<size_t>(__builtin_popcountll(mask)) == active_count;
    }

    bool lane_layout_matches_owner(
        const GpuMatrixLanePhysicalLayout &layout,
        size_t expected_limbs,
        size_t expected_rows,
        size_t expected_columns,
        const char *role)
    {
        if (!layout.owner || !layout.limbs || layout.limb_count != expected_limbs ||
            layout.rows != expected_rows || layout.columns != expected_columns ||
            layout.owner->rows != layout.rows || layout.owner->cols != layout.columns)
        {
            (void)role;
            set_error("invalid lane shape/layout");
            return false;
        }
        if (layout.owner->level < 0 ||
            static_cast<size_t>(layout.owner->level + 1) != expected_limbs)
        {
            (void)role;
            set_error("lane level does not match physical layout");
            return false;
        }
        return true;
    }

    bool fill_lane_limb_metadata(
        MatrixLaneBinaryMetadata *destination,
        const GpuMatrixLanePhysicalLayout &lhs,
        const GpuMatrixLanePhysicalLayout &rhs,
        const GpuMatrixLanePhysicalLayout &out,
        const std::vector<dim3> &limb_ids,
        size_t lane,
        size_t limb,
        int expected_device,
        bool scalar_rhs)
    {
        const dim3 id = limb_ids[limb];
        const auto &lhs_view = lhs.limbs[limb];
        const auto &rhs_view = rhs.limbs[limb];
        const auto &out_view = out.limbs[limb];
        const uint8_t *lhs_data = matrix_limb_ptr_by_id(lhs.owner, 0, id);
        const uint8_t *rhs_data = matrix_limb_ptr_by_id(rhs.owner, 0, id);
        uint8_t *out_data = matrix_limb_ptr_by_id(
            const_cast<GpuMatrix *>(out.owner), 0, id);
        size_t lhs_stride = 0;
        size_t rhs_stride = 0;
        size_t out_stride = 0;
        uint8_t lhs_bytes = 0;
        uint8_t rhs_bytes = 0;
        uint8_t out_bytes = 0;
        if (!lhs_data || !rhs_data || !out_data ||
            !matrix_limb_metadata_by_id(lhs.owner, id, &lhs_stride, &lhs_bytes) ||
            !matrix_limb_metadata_by_id(rhs.owner, id, &rhs_stride, &rhs_bytes) ||
            !matrix_limb_metadata_by_id(out.owner, id, &out_stride, &out_bytes) ||
            lhs_view.data != lhs_data || rhs_view.data != rhs_data ||
            out_view.data != out_data || lhs_view.stride_bytes != lhs_stride ||
            rhs_view.stride_bytes != rhs_stride || out_view.stride_bytes != out_stride ||
            lhs_view.coefficient_bytes != lhs_bytes ||
            rhs_view.coefficient_bytes != rhs_bytes ||
            out_view.coefficient_bytes != out_bytes ||
            !lhs.owner->shared_limb_buffers[id.x].device_descriptors ||
            !rhs.owner->shared_limb_buffers[id.x].device_descriptors ||
            !out.owner->shared_limb_buffers[id.x].device_descriptors ||
            lhs_view.descriptors != lhs.owner->shared_limb_buffers[id.x].device_descriptors ||
            rhs_view.descriptors != rhs.owner->shared_limb_buffers[id.x].device_descriptors ||
            out_view.descriptors != out.owner->shared_limb_buffers[id.x].device_descriptors)
        {
            set_error("lane physical descriptor/data binding identity mismatch");
            return false;
        }
        int lhs_device = -1;
        int rhs_device = -1;
        int out_device = -1;
        if (matrix_limb_device(lhs.owner, id, &lhs_device) != 0 ||
            matrix_limb_device(rhs.owner, id, &rhs_device) != 0 ||
            matrix_limb_device(out.owner, id, &out_device) != 0 ||
            lhs_device != expected_device || rhs_device != expected_device ||
            out_device != expected_device)
        {
            set_error("lane physical layout spans multiple devices");
            return false;
        }
        destination->lhs_data = lhs_data;
        destination->rhs_data = rhs_data;
        destination->out_data = out_data;
        destination->lhs_descriptors = lhs_view.descriptors;
        destination->rhs_descriptors = rhs_view.descriptors;
        destination->out_descriptors = out_view.descriptors;
        destination->lhs_stride_bytes = lhs_stride;
        destination->rhs_stride_bytes = rhs_stride;
        destination->out_stride_bytes = out_stride;
        destination->rows = out.rows;
        destination->columns = out.columns;
        destination->lhs_coefficient_bytes = lhs_bytes;
        destination->rhs_coefficient_bytes = rhs_bytes;
        destination->out_coefficient_bytes = out_bytes;
        destination->rhs_is_scalar = scalar_rhs ? 1 : 0;
        std::fill(std::begin(destination->reserved), std::end(destination->reserved), 0);
        (void)lane;
        (void)scalar_rhs;
        return true;
    }
}

extern "C" int gpu_matrix_lane_physical_layout(
    const GpuMatrix *owner,
    GpuMatrixLaneLimbLayout *limbs,
    size_t limb_capacity,
    size_t *out_limb_count,
    size_t *out_rows,
    size_t *out_columns)
{
    if (!owner || !owner->ctx || !limbs || !out_limb_count || !out_rows || !out_columns ||
        owner->level < 0)
    {
        return set_error("invalid lane physical layout query");
    }
    const size_t limb_count = static_cast<size_t>(owner->level + 1);
    if (limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS || limb_capacity < limb_count ||
        owner->ctx->limb_gpu_ids.size() < limb_count)
    {
        return set_error("lane physical layout capacity mismatch");
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = owner->ctx->limb_gpu_ids[limb];
        size_t stride = 0;
        uint8_t coefficient_bytes = 0;
        const uint8_t *data = matrix_limb_ptr_by_id(owner, 0, id);
        if (!data || !matrix_limb_metadata_by_id(owner, id, &stride, &coefficient_bytes) ||
            id.x >= owner->shared_limb_buffers.size() ||
            !owner->shared_limb_buffers[id.x].device_descriptors)
        {
            return set_error("lane physical layout has no resident limb");
        }
        limbs[limb] = {
            data,
            owner->shared_limb_buffers[id.x].device_descriptors,
            stride,
            coefficient_bytes,
        };
    }
    *out_limb_count = limb_count;
    *out_rows = owner->rows;
    *out_columns = owner->cols;
    return 0;
}

extern "C" int gpu_matrix_binary_lane_batch_layout(
    const GpuMatrixLanePhysicalLayout *outputs,
    const GpuMatrixLanePhysicalLayout *left_layouts,
    const GpuMatrixLanePhysicalLayout *right_layouts,
    size_t lane_count,
    size_t active_count,
    uint64_t active_lane_mask,
    int operation,
    int rhs_broadcast,
    const uint32_t *metadata_binding_indices,
    size_t metadata_binding_count)
{
    if (!outputs || !left_layouts || !right_layouts ||
        !lane_mask_is_valid(lane_count, active_count, active_lane_mask) ||
        (operation != 0 && operation != 1) || (rhs_broadcast != 0 && rhs_broadcast != 1))
    {
        return set_error("invalid lane binary batch arguments");
    }
    if (active_count == 0)
    {
        return 0;
    }

    const size_t first_lane = static_cast<size_t>(__builtin_ctzll(active_lane_mask));
    const auto &first_output = outputs[first_lane];
    const auto &first_left = left_layouts[first_lane];
    const auto &first_right = rhs_broadcast ? right_layouts[0] : right_layouts[first_lane];
    if (!first_output.owner || !first_left.owner || !first_right.owner ||
        first_left.owner->ctx != first_output.owner->ctx ||
        first_right.owner->ctx != first_output.owner->ctx ||
        first_left.owner->level != first_output.owner->level ||
        first_right.owner->level != first_output.owner->level ||
        first_left.owner->format != first_output.owner->format ||
        first_right.owner->format != first_output.owner->format)
    {
        return set_error("lane binary batch context, level, or format mismatch");
    }
    GpuContext *ctx = first_output.owner->ctx;
    const size_t limb_count = static_cast<size_t>(first_output.owner->level + 1);
    if (!ctx || limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS ||
        ctx->N <= 0 || ctx->limb_gpu_ids.size() < limb_count ||
        ctx->moduli.size() < limb_count)
    {
        return set_error("invalid lane binary batch CRT configuration");
    }
    const size_t n = static_cast<size_t>(ctx->N);
    std::vector<MatrixLaneBinaryMetadata> host_metadata(lane_count * limb_count);
    std::vector<uint8_t> active(lane_count, 0);
    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    size_t max_polynomials = 0;

    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (((active_lane_mask >> lane) & UINT64_C(1)) == 0)
        {
            continue;
        }
        active[lane] = 1;
        const auto &output = outputs[lane];
        const auto &left = left_layouts[lane];
        const auto &right = rhs_broadcast ? right_layouts[0] : right_layouts[lane];
        if (!lane_layout_matches_owner(output, limb_count, output.rows, output.columns, "output") ||
            !lane_layout_matches_owner(left, limb_count, left.rows, left.columns, "left") ||
            !lane_layout_matches_owner(right, limb_count, right.rows, right.columns, "right") ||
            left.owner->ctx != ctx || right.owner->ctx != ctx ||
            left.owner->level != first_left.owner->level ||
            right.owner->level != first_left.owner->level ||
            output.owner->format != left.owner->format ||
            right.owner->format != left.owner->format ||
            output.rows != left.rows || output.columns != left.columns)
        {
            return set_error("lane binary batch shape or owner mismatch");
        }
        const bool scalar_rhs = right.rows == 1 && right.columns == 1 &&
                                (left.rows != 1 || left.columns != 1);
        if (!scalar_rhs && (right.rows != left.rows || right.columns != left.columns))
        {
            return set_error("lane binary batch RHS shape mismatch");
        }
        if (output.rows != 0 && output.columns > static_cast<size_t>(-1) / output.rows)
        {
            return set_error("lane binary batch shape overflow");
        }
        const size_t polynomials = output.rows * output.columns;
        max_polynomials = std::max(max_polynomials, polynomials);
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = ctx->limb_gpu_ids[limb];
            int device = -1;
            cudaStream_t stream = nullptr;
            if (matrix_limb_device(output.owner, id, &device) != 0 ||
                matrix_limb_stream(output.owner, id, &stream) != 0)
            {
                return 1;
            }
            if (dispatch_device < 0)
            {
                dispatch_device = device;
                dispatch_stream = stream;
            }
            else if (dispatch_device != device || !stream || !dispatch_stream)
            {
                return set_error("lane binary batch requires one physical placement");
            }
            if (!fill_lane_limb_metadata(
                    &host_metadata[lane * limb_count + limb],
                    left,
                    right,
                    output,
                    ctx->limb_gpu_ids,
                    lane,
                    limb,
                    dispatch_device,
                    scalar_rhs))
            {
                return 1;
            }
        }
    }
    if (!dispatch_stream || dispatch_device < 0 || max_polynomials == 0)
    {
        return 0;
    }
    cudaError_t error = cudaSetDevice(dispatch_device);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }

    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (!active[lane]) continue;
        const auto &left = left_layouts[lane];
        const auto &right = rhs_broadcast ? right_layouts[0] : right_layouts[lane];
        auto &output = outputs[lane];
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = ctx->limb_gpu_ids[limb];
            int status = matrix_wait_limb_stream(left.owner, id, dispatch_device, dispatch_stream);
            if (status == 0) status = matrix_wait_limb_stream(right.owner, id, dispatch_device, dispatch_stream);
            if (status == 0) status = matrix_wait_limb_stream(output.owner, id, dispatch_device, dispatch_stream);
            if (status != 0) return status;
        }
    }

    MatrixLaneBinaryMetadata *device_metadata = nullptr;
    uint64_t *device_moduli = nullptr;
    const size_t metadata_bytes = host_metadata.size() * sizeof(MatrixLaneBinaryMetadata);
    auto release = [&]() {
        if (device_moduli) cudaFreeAsync(device_moduli, dispatch_stream);
        if (device_metadata) cudaFreeAsync(device_metadata, dispatch_stream);
    };
    error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_metadata), metadata_bytes, dispatch_stream);
    if (error == cudaSuccess)
    {
        error = cudaMallocAsync(
            reinterpret_cast<void **>(&device_moduli), limb_count * sizeof(uint64_t), dispatch_stream);
    }
    if (error == cudaSuccess)
    {
        error = cudaMemcpyAsync(device_metadata, host_metadata.data(),
            metadata_bytes, cudaMemcpyHostToDevice, dispatch_stream);
    }
    if (error == cudaSuccess)
    {
        error = matrix_batch_upload(
            ctx,
            device_moduli,
            ctx->moduli.data(),
            limb_count * sizeof(uint64_t),
            dispatch_stream);
    }
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }

    constexpr int threads = 256;
    if (max_polynomials != 0 && n > static_cast<size_t>(-1) / max_polynomials)
    {
        release();
        return set_error("lane binary batch coefficient count overflow");
    }
    const size_t work = max_polynomials * n;
    const size_t block_count = std::max<size_t>(
        1,
        std::min<size_t>((work + static_cast<size_t>(threads) - 1) /
                             static_cast<size_t>(threads),
                         65535));
    const dim3 grid(
        static_cast<unsigned int>(block_count),
        static_cast<unsigned int>(lane_count),
        static_cast<unsigned int>(limb_count));
    matrix_binary_lane_batch_kernel<<<grid, threads, 0, dispatch_stream>>>(
        device_metadata,
        device_moduli,
        lane_count,
        limb_count,
        n,
        operation,
        active_lane_mask);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }

    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (!active[lane]) continue;
        const auto &left = left_layouts[lane];
        const auto &right = rhs_broadcast ? right_layouts[0] : right_layouts[lane];
        auto &output = outputs[lane];
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = ctx->limb_gpu_ids[limb];
            int status = matrix_track_limb_consumer(left.owner, id, dispatch_device, dispatch_stream);
            if (status == 0) status = matrix_track_limb_consumer(right.owner, id, dispatch_device, dispatch_stream);
            if (status == 0) status = matrix_record_limb_write(
                const_cast<GpuMatrix *>(output.owner), id, dispatch_stream);
            if (status != 0)
            {
                release();
                return status;
            }
        }
        const_cast<GpuMatrix *>(output.owner)->format = left.owner->format;
    }
    release();
    return 0;
}

namespace
{
    struct MatrixLaneMulMetadata
    {
        const uint8_t *lhs_data;
        const uint8_t *rhs_data;
        uint8_t *out_data;
        size_t lhs_stride_bytes;
        size_t rhs_stride_bytes;
        size_t out_stride_bytes;
        size_t rows;
        size_t inner;
        size_t columns;
        uint8_t lhs_coefficient_bytes;
        uint8_t rhs_coefficient_bytes;
        uint8_t out_coefficient_bytes;
        uint8_t reserved[5];
    };

    struct MatrixLaneUnaryMetadata
    {
        const uint8_t *input_data;
        const uint8_t *scalar_data;
        uint8_t *out_data;
        size_t input_stride_bytes;
        size_t scalar_stride_bytes;
        size_t out_stride_bytes;
        size_t rows;
        size_t columns;
        uint8_t input_coefficient_bytes;
        uint8_t scalar_coefficient_bytes;
        uint8_t out_coefficient_bytes;
        uint8_t scalar_is_broadcast;
        uint8_t reserved[4];
    };

    struct MatrixLaneTransposeMetadata
    {
        const uint8_t *input_data;
        uint8_t *out_data;
        size_t input_stride_bytes;
        size_t out_stride_bytes;
        size_t input_rows;
        size_t input_columns;
        uint8_t input_coefficient_bytes;
        uint8_t out_coefficient_bytes;
        uint8_t reserved[6];
    };

    __global__ void matrix_mul_lane_batch_kernel(
        const MatrixLaneMulMetadata *metadata,
        const uint64_t *moduli,
        size_t lane_count,
        size_t limb_count,
        size_t n,
        uint64_t active_lane_mask)
    {
        const size_t lane = static_cast<size_t>(blockIdx.y);
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (lane >= lane_count || limb >= limb_count ||
            ((active_lane_mask >> lane) & UINT64_C(1)) == 0)
            return;
        const auto &entry = metadata[lane * limb_count + limb];
        const size_t total = entry.rows * entry.columns * n;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        const uint64_t modulus = moduli[limb];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < total;
             index += stride)
        {
            const size_t coefficient = index % n;
            const size_t polynomial = index / n;
            const size_t row = polynomial / entry.columns;
            const size_t column = polynomial - row * entry.columns;
            uint64_t value = 0;
            for (size_t inner = 0; inner < entry.inner; ++inner)
            {
                const uint64_t lhs = matrix_load_limb_u64(
                    entry.lhs_data,
                    row * entry.inner + inner,
                    coefficient,
                    entry.lhs_stride_bytes,
                    entry.lhs_coefficient_bytes);
                const uint64_t rhs = matrix_load_limb_u64(
                    entry.rhs_data,
                    inner * entry.columns + column,
                    coefficient,
                    entry.rhs_stride_bytes,
                    entry.rhs_coefficient_bytes);
                value = add_mod_u64(value, mul_mod_u64(lhs, rhs, modulus), modulus);
            }
            matrix_store_limb_u64(
                entry.out_data,
                polynomial,
                coefficient,
                entry.out_stride_bytes,
                entry.out_coefficient_bytes,
                value);
        }
    }

    __global__ void matrix_unary_lane_batch_kernel(
        const MatrixLaneUnaryMetadata *metadata,
        const uint64_t *moduli,
        size_t lane_count,
        size_t limb_count,
        size_t n,
        int operation,
        uint64_t active_lane_mask)
    {
        const size_t lane = static_cast<size_t>(blockIdx.y);
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (lane >= lane_count || limb >= limb_count ||
            ((active_lane_mask >> lane) & UINT64_C(1)) == 0)
            return;
        const auto &entry = metadata[lane * limb_count + limb];
        const size_t total = entry.rows * entry.columns * n;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        const uint64_t modulus = moduli[limb];
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < total;
             index += stride)
        {
            const size_t coefficient = index % n;
            const size_t polynomial = index / n;
            const uint64_t input = matrix_load_limb_u64(
                entry.input_data,
                polynomial,
                coefficient,
                entry.input_stride_bytes,
                entry.input_coefficient_bytes);
            uint64_t value = 0;
            if (operation == 0)
            {
                value = input == 0 ? 0 : modulus - input;
            }
            else
            {
                const uint64_t scalar = matrix_load_limb_u64(
                    entry.scalar_data,
                    entry.scalar_is_broadcast ? 0 : polynomial,
                    coefficient,
                    entry.scalar_stride_bytes,
                    entry.scalar_coefficient_bytes);
                value = mul_mod_u64(input, scalar, modulus);
            }
            matrix_store_limb_u64(
                entry.out_data,
                polynomial,
                coefficient,
                entry.out_stride_bytes,
                entry.out_coefficient_bytes,
                value);
        }
    }

    __global__ void matrix_transpose_lane_batch_kernel(
        const MatrixLaneTransposeMetadata *metadata,
        size_t lane_count,
        size_t limb_count,
        size_t n,
        uint64_t active_lane_mask)
    {
        const size_t lane = static_cast<size_t>(blockIdx.y);
        const size_t limb = static_cast<size_t>(blockIdx.z);
        if (lane >= lane_count || limb >= limb_count ||
            ((active_lane_mask >> lane) & UINT64_C(1)) == 0)
            return;
        const auto &entry = metadata[lane * limb_count + limb];
        const size_t total = entry.input_rows * entry.input_columns * n;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < total;
             index += stride)
        {
            const size_t coefficient = index % n;
            const size_t output_polynomial = index / n;
            const size_t row = output_polynomial / entry.input_rows;
            const size_t column = output_polynomial - row * entry.input_rows;
            const size_t source_polynomial = column * entry.input_columns + row;
            const uint64_t value = matrix_load_limb_u64(
                entry.input_data,
                source_polynomial,
                coefficient,
                entry.input_stride_bytes,
                entry.input_coefficient_bytes);
            matrix_store_limb_u64(
                entry.out_data,
                output_polynomial,
                coefficient,
                entry.out_stride_bytes,
                entry.out_coefficient_bytes,
                value);
        }
    }

    bool lane_layout_limb_binding(
        const GpuMatrixLanePhysicalLayout &layout,
        const dim3 &id,
        const uint8_t *expected_data,
        size_t expected_stride,
        uint8_t expected_bytes)
    {
        return layout.limbs && layout.limbs[id.y].data == expected_data &&
               layout.limbs[id.y].stride_bytes == expected_stride &&
               layout.limbs[id.y].coefficient_bytes == expected_bytes &&
               layout.limbs[id.y].descriptors ==
                   layout.owner->shared_limb_buffers[id.x].device_descriptors;
    }

    bool lane_context_and_mask(
        const GpuMatrixLanePhysicalLayout *outputs,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        GpuContext **out_context,
        size_t *out_limb_count,
        int *out_device,
        cudaStream_t *out_stream)
    {
        if (!outputs || !out_context || !out_limb_count || !out_device || !out_stream ||
            !lane_mask_is_valid(lane_count, active_count, active_lane_mask) || active_count == 0)
            return false;
        const size_t first = static_cast<size_t>(__builtin_ctzll(active_lane_mask));
        if (!outputs[first].owner || !outputs[first].owner->ctx ||
            outputs[first].owner->level < 0)
            return false;
        GpuContext *context = outputs[first].owner->ctx;
        const size_t limb_count = static_cast<size_t>(outputs[first].owner->level + 1);
        if (limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS ||
            context->limb_gpu_ids.size() < limb_count || context->moduli.size() < limb_count ||
            context->N <= 0)
            return false;
        int device = -1;
        cudaStream_t stream = nullptr;
        const dim3 id = context->limb_gpu_ids[0];
        if (matrix_limb_device(outputs[first].owner, id, &device) != 0 ||
            matrix_limb_stream(outputs[first].owner, id, &stream) != 0 || !stream)
            return false;
        *out_context = context;
        *out_limb_count = limb_count;
        *out_device = device;
        *out_stream = stream;
        return true;
    }

    bool lane_metadata_common_valid(
        const GpuMatrixLanePhysicalLayout &output,
        const GpuMatrixLanePhysicalLayout &input,
        size_t limb_count,
        GpuContext *context)
    {
        return lane_layout_matches_owner(output, limb_count, output.rows, output.columns, "output") &&
               lane_layout_matches_owner(input, limb_count, input.rows, input.columns, "input") &&
               output.owner->ctx == context && input.owner->ctx == context &&
               output.owner->level == input.owner->level &&
               output.owner->format == input.owner->format;
    }

    size_t lane_grid_blocks(size_t polynomials, size_t n)
    {
        if (polynomials == 0 || n == 0) return 1;
        if (n > static_cast<size_t>(-1) / polynomials) return 0;
        const size_t work = polynomials * n;
        const size_t block_count = work / 256 + (work % 256 != 0 ? 1 : 0);
        return std::max<size_t>(1, std::min<size_t>(
            block_count, 65535));
    }
}

extern "C" int gpu_matrix_mul_lane_batch_layout(
    const GpuMatrixLanePhysicalLayout *outputs,
    const GpuMatrixLanePhysicalLayout *left_layouts,
    const GpuMatrixLanePhysicalLayout *right_layouts,
    size_t lane_count,
    size_t active_count,
    uint64_t active_lane_mask,
    const uint32_t *metadata_binding_indices,
    size_t metadata_binding_count)
{
    if (active_count == 0) return 0;
    GpuContext *context = nullptr;
    size_t limb_count = 0;
    int device = -1;
    cudaStream_t stream = nullptr;
    if (!left_layouts || !right_layouts || !lane_context_and_mask(
            outputs, lane_count, active_count, active_lane_mask,
            &context, &limb_count, &device, &stream))
        return set_error("invalid lane matrix multiply arguments");
    std::vector<MatrixLaneMulMetadata> host(lane_count * limb_count);
    size_t max_polynomials = 0;
    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
        const auto &out = outputs[lane];
        const auto &lhs = left_layouts[lane];
        const auto &rhs = right_layouts[lane];
        if (!lane_layout_matches_owner(out, limb_count, out.rows, out.columns, "output") ||
            !lane_layout_matches_owner(lhs, limb_count, lhs.rows, lhs.columns, "left") ||
            !lane_layout_matches_owner(rhs, limb_count, rhs.rows, rhs.columns, "right") ||
            out.owner->ctx != context ||
            lhs.owner->ctx != context || rhs.owner->ctx != context ||
            out.owner->level != lhs.owner->level || out.owner->level != rhs.owner->level ||
            lhs.owner->format != GPU_POLY_FORMAT_EVAL ||
            rhs.owner->format != GPU_POLY_FORMAT_EVAL ||
            out.owner->format != GPU_POLY_FORMAT_EVAL || lhs.columns != rhs.rows ||
            out.rows != lhs.rows || out.columns != rhs.columns ||
            (out.rows != 0 && out.columns > static_cast<size_t>(-1) / out.rows))
            return set_error("lane matrix multiply shape/domain mismatch");
        max_polynomials = std::max(max_polynomials, out.rows * out.columns);
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = context->limb_gpu_ids[limb];
            size_t lhs_stride = 0, rhs_stride = 0, out_stride = 0;
            uint8_t lhs_bytes = 0, rhs_bytes = 0, out_bytes = 0;
            const uint8_t *lhs_data = matrix_limb_ptr_by_id(lhs.owner, 0, id);
            const uint8_t *rhs_data = matrix_limb_ptr_by_id(rhs.owner, 0, id);
            uint8_t *out_data = matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(out.owner), 0, id);
            if (!lhs_data || !rhs_data || !out_data ||
                !matrix_limb_metadata_by_id(lhs.owner, id, &lhs_stride, &lhs_bytes) ||
                !matrix_limb_metadata_by_id(rhs.owner, id, &rhs_stride, &rhs_bytes) ||
                !matrix_limb_metadata_by_id(out.owner, id, &out_stride, &out_bytes) ||
                !lane_layout_limb_binding(lhs, id, lhs_data, lhs_stride, lhs_bytes) ||
                !lane_layout_limb_binding(rhs, id, rhs_data, rhs_stride, rhs_bytes) ||
                !lane_layout_limb_binding(out, id, out_data, out_stride, out_bytes))
                return set_error("lane matrix multiply binding identity mismatch");
            int lhs_device = -1, rhs_device = -1, out_device = -1;
            if (matrix_limb_device(lhs.owner, id, &lhs_device) != 0 ||
                matrix_limb_device(rhs.owner, id, &rhs_device) != 0 ||
                matrix_limb_device(out.owner, id, &out_device) != 0 ||
                lhs_device != device || rhs_device != device || out_device != device)
                return set_error("lane matrix multiply placement mismatch");
            host[lane * limb_count + limb] = {
                lhs_data, rhs_data, out_data, lhs_stride, rhs_stride, out_stride,
                lhs.rows, lhs.columns, rhs.columns, lhs_bytes, rhs_bytes, out_bytes, {0}};
        }
    }
    if (max_polynomials == 0) return 0;
    const size_t blocks = lane_grid_blocks(max_polynomials, static_cast<size_t>(context->N));
    if (blocks == 0) return set_error("lane matrix multiply shape overflow");
    cudaSetDevice(device);
    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = context->limb_gpu_ids[limb];
            int status = matrix_wait_limb_stream(left_layouts[lane].owner, id, device, stream);
            if (status == 0) status = matrix_wait_limb_stream(right_layouts[lane].owner, id, device, stream);
            if (status == 0) status = matrix_wait_limb_stream(outputs[lane].owner, id, device, stream);
            if (status != 0) return status;
        }
    }
    MatrixLaneMulMetadata *device_metadata = nullptr;
    uint64_t *device_moduli = nullptr;
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&device_metadata),
        host.size() * sizeof(MatrixLaneMulMetadata), stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_moduli), limb_count * sizeof(uint64_t), stream);
    constexpr size_t pointer_offsets[] = {
        offsetof(MatrixLaneMulMetadata, lhs_data),
        offsetof(MatrixLaneMulMetadata, rhs_data),
        offsetof(MatrixLaneMulMetadata, out_data),
    };
    if (error == cudaSuccess) error = matrix_lane_metadata_upload(
        context,
        device_metadata,
        host,
        lane_count,
        limb_count,
        active_lane_mask,
        pointer_offsets,
        metadata_binding_indices,
        metadata_binding_count,
        stream);
    if (error == cudaSuccess) error = matrix_batch_upload(
        context, device_moduli, context->moduli.data(), limb_count * sizeof(uint64_t), stream);
    if (error != cudaSuccess)
    {
        if (device_moduli) cudaFreeAsync(device_moduli, stream);
        if (device_metadata) cudaFreeAsync(device_metadata, stream);
        return set_error(error);
    }
    matrix_mul_lane_batch_kernel<<<dim3(static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(lane_count), static_cast<unsigned int>(limb_count)), 256, 0, stream>>>(
            device_metadata, device_moduli, lane_count, limb_count,
            static_cast<size_t>(context->N), active_lane_mask);
    error = cudaGetLastError();
    if (error == cudaSuccess)
    {
        for (size_t lane = 0; lane < lane_count && error == cudaSuccess; ++lane)
        {
            if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const dim3 id = context->limb_gpu_ids[limb];
                int status = matrix_track_limb_consumer(left_layouts[lane].owner, id, device, stream);
                if (status == 0) status = matrix_track_limb_consumer(right_layouts[lane].owner, id, device, stream);
                if (status == 0) status = matrix_record_limb_write(
                    const_cast<GpuMatrix *>(outputs[lane].owner), id, stream);
                if (status != 0) error = cudaErrorUnknown;
            }
        }
    }
    cudaFreeAsync(device_moduli, stream);
    cudaFreeAsync(device_metadata, stream);
    if (error != cudaSuccess) return set_error(error);
    return 0;
}

namespace
{
    int launch_lane_unary_batch(
        const GpuMatrixLanePhysicalLayout *outputs,
        const GpuMatrixLanePhysicalLayout *inputs,
        const GpuMatrixLanePhysicalLayout *scalars,
        size_t lane_count,
        size_t active_count,
        uint64_t active_lane_mask,
        int operation,
        int scalar_broadcast,
        const uint32_t *metadata_binding_indices,
        size_t metadata_binding_count)
    {
        if (active_count == 0) return 0;
        GpuContext *context = nullptr;
        size_t limb_count = 0;
        int device = -1;
        cudaStream_t stream = nullptr;
        if (!inputs || !lane_context_and_mask(
                outputs, lane_count, active_count, active_lane_mask,
                &context, &limb_count, &device, &stream))
            return set_error("invalid lane unary arguments");
        if (operation != 0 && !scalars)
            return set_error("missing lane scalar layouts");
        std::vector<MatrixLaneUnaryMetadata> host(lane_count * limb_count);
        size_t max_polynomials = 0;
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
            const auto &out = outputs[lane];
            const auto &input = inputs[lane];
            const auto *scalar = operation == 0
                ? nullptr
                : &(scalar_broadcast ? scalars[0] : scalars[lane]);
            if (!lane_metadata_common_valid(out, input, limb_count, context) ||
                out.rows != input.rows || out.columns != input.columns ||
                (out.rows != 0 && out.columns > static_cast<size_t>(-1) / out.rows) ||
                (operation != 0 &&
                 (!lane_layout_matches_owner(*scalar, limb_count, scalar->rows, scalar->columns, "scalar") ||
                  scalar->owner->ctx != context || scalar->owner->level != input.owner->level ||
                  scalar->owner->format != input.owner->format ||
                  !((scalar->rows == 1 && scalar->columns == 1) ||
                    (scalar->rows == input.rows && scalar->columns == input.columns)))))
                return set_error("lane unary shape/domain mismatch");
            const bool scalar_is_broadcast = operation != 0 && scalar->rows == 1 && scalar->columns == 1 &&
                                             (input.rows != 1 || input.columns != 1);
            max_polynomials = std::max(max_polynomials, out.rows * out.columns);
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const dim3 id = context->limb_gpu_ids[limb];
                size_t input_stride = 0, scalar_stride = 0, out_stride = 0;
                uint8_t input_bytes = 0, scalar_bytes = 0, out_bytes = 0;
                const uint8_t *input_data = matrix_limb_ptr_by_id(input.owner, 0, id);
                const uint8_t *scalar_data = operation == 0
                    ? input_data
                    : matrix_limb_ptr_by_id(scalar->owner, 0, id);
                uint8_t *out_data = matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(out.owner), 0, id);
                if (!input_data || !out_data ||
                    !matrix_limb_metadata_by_id(input.owner, id, &input_stride, &input_bytes) ||
                    !matrix_limb_metadata_by_id(out.owner, id, &out_stride, &out_bytes) ||
                    !lane_layout_limb_binding(input, id, input_data, input_stride, input_bytes) ||
                    !lane_layout_limb_binding(out, id, out_data, out_stride, out_bytes))
                    return set_error("lane unary binding identity mismatch");
                if (operation == 0)
                {
                    scalar_stride = input_stride;
                    scalar_bytes = input_bytes;
                }
                else if (!scalar_data ||
                         !matrix_limb_metadata_by_id(scalar->owner, id, &scalar_stride, &scalar_bytes) ||
                         !lane_layout_limb_binding(*scalar, id, scalar_data, scalar_stride, scalar_bytes))
                    return set_error("lane scalar binding identity mismatch");
                int input_device = -1, scalar_device = device, out_device = -1;
                if (matrix_limb_device(input.owner, id, &input_device) != 0 ||
                    matrix_limb_device(out.owner, id, &out_device) != 0 ||
                    input_device != device || out_device != device ||
                    (operation != 0 && (matrix_limb_device(scalar->owner, id, &scalar_device) != 0 ||
                                        scalar_device != device)))
                    return set_error("lane unary placement mismatch");
                host[lane * limb_count + limb] = {
                    input_data, scalar_data, out_data, input_stride, scalar_stride, out_stride,
                    out.rows, out.columns, input_bytes, scalar_bytes, out_bytes,
                    static_cast<uint8_t>(scalar_is_broadcast), {0}};
            }
        }
        if (max_polynomials == 0) return 0;
        const size_t blocks = lane_grid_blocks(max_polynomials, static_cast<size_t>(context->N));
        if (blocks == 0) return set_error("lane unary shape overflow");
        cudaSetDevice(device);
        for (size_t lane = 0; lane < lane_count; ++lane)
        {
            if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const dim3 id = context->limb_gpu_ids[limb];
                int status = matrix_wait_limb_stream(inputs[lane].owner, id, device, stream);
                if (status == 0 && operation != 0)
                    status = matrix_wait_limb_stream(
                        (scalar_broadcast ? scalars[0] : scalars[lane]).owner,
                        id, device, stream);
                if (status == 0) status = matrix_wait_limb_stream(outputs[lane].owner, id, device, stream);
                if (status != 0) return status;
            }
        }
        MatrixLaneUnaryMetadata *device_metadata = nullptr;
        uint64_t *device_moduli = nullptr;
        cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&device_metadata),
            host.size() * sizeof(MatrixLaneUnaryMetadata), stream);
        if (error == cudaSuccess) error = cudaMallocAsync(
            reinterpret_cast<void **>(&device_moduli), limb_count * sizeof(uint64_t), stream);
        constexpr size_t pointer_offsets[] = {
            offsetof(MatrixLaneUnaryMetadata, input_data),
            offsetof(MatrixLaneUnaryMetadata, scalar_data),
            offsetof(MatrixLaneUnaryMetadata, out_data),
        };
        if (error == cudaSuccess) error = matrix_lane_metadata_upload(
            context,
            device_metadata,
            host,
            lane_count,
            limb_count,
            active_lane_mask,
            pointer_offsets,
            metadata_binding_indices,
            metadata_binding_count,
            stream);
        if (error == cudaSuccess) error = matrix_batch_upload(
            context, device_moduli, context->moduli.data(), limb_count * sizeof(uint64_t), stream);
        if (error != cudaSuccess)
        {
            if (device_moduli) cudaFreeAsync(device_moduli, stream);
            if (device_metadata) cudaFreeAsync(device_metadata, stream);
            return set_error(error);
        }
        matrix_unary_lane_batch_kernel<<<dim3(static_cast<unsigned int>(blocks),
            static_cast<unsigned int>(lane_count), static_cast<unsigned int>(limb_count)), 256, 0, stream>>>(
                device_metadata, device_moduli, lane_count, limb_count,
                static_cast<size_t>(context->N), operation, active_lane_mask);
        error = cudaGetLastError();
        if (error == cudaSuccess)
        {
            for (size_t lane = 0; lane < lane_count && error == cudaSuccess; ++lane)
            {
                if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
                for (size_t limb = 0; limb < limb_count; ++limb)
                {
                    const dim3 id = context->limb_gpu_ids[limb];
                    int status = matrix_track_limb_consumer(inputs[lane].owner, id, device, stream);
                    if (status == 0 && operation != 0)
                        status = matrix_track_limb_consumer(
                            (scalar_broadcast ? scalars[0] : scalars[lane]).owner,
                            id, device, stream);
                    if (status == 0) status = matrix_record_limb_write(
                        const_cast<GpuMatrix *>(outputs[lane].owner), id, stream);
                    if (status != 0) error = cudaErrorUnknown;
                }
            }
        }
        cudaFreeAsync(device_moduli, stream);
        cudaFreeAsync(device_metadata, stream);
        return error == cudaSuccess ? 0 : set_error(error);
    }
}

extern "C" int gpu_matrix_negate_lane_batch_layout(
    const GpuMatrixLanePhysicalLayout *outputs,
    const GpuMatrixLanePhysicalLayout *inputs,
    size_t lane_count,
    size_t active_count,
    uint64_t active_lane_mask,
    const uint32_t *metadata_binding_indices,
    size_t metadata_binding_count)
{
    return launch_lane_unary_batch(
        outputs, inputs, nullptr, lane_count, active_count, active_lane_mask, 0, 0,
        metadata_binding_indices, metadata_binding_count);
}

extern "C" int gpu_matrix_scalar_mul_lane_batch_layout(
    const GpuMatrixLanePhysicalLayout *outputs,
    const GpuMatrixLanePhysicalLayout *inputs,
    const GpuMatrixLanePhysicalLayout *scalars,
    size_t lane_count,
    size_t active_count,
    uint64_t active_lane_mask,
    int scalar_broadcast,
    const uint32_t *metadata_binding_indices,
    size_t metadata_binding_count)
{
    if (scalar_broadcast != 0 && scalar_broadcast != 1)
        return set_error("invalid scalar broadcast flag");
    return launch_lane_unary_batch(
        outputs, inputs, scalars, lane_count, active_count, active_lane_mask, 1, scalar_broadcast,
        metadata_binding_indices, metadata_binding_count);
}

extern "C" int gpu_matrix_transpose_lane_batch_layout(
    const GpuMatrixLanePhysicalLayout *outputs,
    const GpuMatrixLanePhysicalLayout *inputs,
    size_t lane_count,
    size_t active_count,
    uint64_t active_lane_mask,
    const uint32_t *metadata_binding_indices,
    size_t metadata_binding_count)
{
    if (active_count == 0) return 0;
    GpuContext *context = nullptr;
    size_t limb_count = 0;
    int device = -1;
    cudaStream_t stream = nullptr;
    if (!inputs || !lane_context_and_mask(
            outputs, lane_count, active_count, active_lane_mask,
            &context, &limb_count, &device, &stream))
        return set_error("invalid lane transpose arguments");
    std::vector<MatrixLaneTransposeMetadata> host(lane_count * limb_count);
    size_t max_polynomials = 0;
    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
        const auto &out = outputs[lane];
        const auto &input = inputs[lane];
        if (!lane_layout_matches_owner(out, limb_count, out.rows, out.columns, "output") ||
            !lane_layout_matches_owner(input, limb_count, input.rows, input.columns, "input") ||
            out.owner->ctx != context || input.owner->ctx != context ||
            out.owner->level != input.owner->level || out.owner->format != input.owner->format ||
            out.rows != input.columns || out.columns != input.rows ||
            (input.rows != 0 && input.columns > static_cast<size_t>(-1) / input.rows))
            return set_error("lane transpose shape/domain mismatch");
        max_polynomials = std::max(max_polynomials, out.rows * out.columns);
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = context->limb_gpu_ids[limb];
            size_t input_stride = 0, out_stride = 0;
            uint8_t input_bytes = 0, out_bytes = 0;
            const uint8_t *input_data = matrix_limb_ptr_by_id(input.owner, 0, id);
            uint8_t *out_data = matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(out.owner), 0, id);
            if (!input_data || !out_data ||
                !matrix_limb_metadata_by_id(input.owner, id, &input_stride, &input_bytes) ||
                !matrix_limb_metadata_by_id(out.owner, id, &out_stride, &out_bytes) ||
                !lane_layout_limb_binding(input, id, input_data, input_stride, input_bytes) ||
                !lane_layout_limb_binding(out, id, out_data, out_stride, out_bytes))
                return set_error("lane transpose binding identity mismatch");
            host[lane * limb_count + limb] = {
                input_data, out_data, input_stride, out_stride,
                input.rows, input.columns, input_bytes, out_bytes, {0}};
        }
    }
    if (max_polynomials == 0) return 0;
    const size_t blocks = lane_grid_blocks(max_polynomials, static_cast<size_t>(context->N));
    if (blocks == 0) return set_error("lane transpose shape overflow");
    cudaSetDevice(device);
    for (size_t lane = 0; lane < lane_count; ++lane)
    {
        if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = context->limb_gpu_ids[limb];
            int status = matrix_wait_limb_stream(inputs[lane].owner, id, device, stream);
            if (status == 0) status = matrix_wait_limb_stream(outputs[lane].owner, id, device, stream);
            if (status != 0) return status;
        }
    }
    MatrixLaneTransposeMetadata *device_metadata = nullptr;
    cudaError_t error = cudaMallocAsync(reinterpret_cast<void **>(&device_metadata),
        host.size() * sizeof(MatrixLaneTransposeMetadata), stream);
    constexpr size_t pointer_offsets[] = {
        offsetof(MatrixLaneTransposeMetadata, input_data),
        offsetof(MatrixLaneTransposeMetadata, out_data),
    };
    if (error == cudaSuccess) error = matrix_lane_metadata_upload(
        context,
        device_metadata,
        host,
        lane_count,
        limb_count,
        active_lane_mask,
        pointer_offsets,
        metadata_binding_indices,
        metadata_binding_count,
        stream);
    if (error != cudaSuccess)
    {
        if (device_metadata) cudaFreeAsync(device_metadata, stream);
        return set_error(error);
    }
    matrix_transpose_lane_batch_kernel<<<dim3(static_cast<unsigned int>(blocks),
        static_cast<unsigned int>(lane_count), static_cast<unsigned int>(limb_count)), 256, 0, stream>>>(
            device_metadata, lane_count, limb_count, static_cast<size_t>(context->N), active_lane_mask);
    error = cudaGetLastError();
    if (error == cudaSuccess)
    {
        for (size_t lane = 0; lane < lane_count && error == cudaSuccess; ++lane)
        {
            if (((active_lane_mask >> lane) & UINT64_C(1)) == 0) continue;
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const dim3 id = context->limb_gpu_ids[limb];
                int status = matrix_track_limb_consumer(inputs[lane].owner, id, device, stream);
                if (status == 0) status = matrix_record_limb_write(
                    const_cast<GpuMatrix *>(outputs[lane].owner), id, stream);
                if (status != 0) error = cudaErrorUnknown;
            }
        }
    }
    cudaFreeAsync(device_metadata, stream);
    return error == cudaSuccess ? 0 : set_error(error);
}
