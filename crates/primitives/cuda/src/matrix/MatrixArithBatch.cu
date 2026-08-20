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
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
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
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_inputs, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
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
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_matrices, metadata.left.data(), pointer_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_scalars, metadata.right.data(), pointer_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count,
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t),
        cudaMemcpyHostToDevice, metadata.stream);
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
    const size_t coefficient_groups = std::min(metadata.n, static_cast<size_t>(65535) / pointer_count);
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
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    const dim3 block(kMatmulTileN, kMatmulTileM, 1);
    const dim3 grid(
        static_cast<unsigned int>((metadata.columns + kMatmulTileN - 1) / kMatmulTileN),
        static_cast<unsigned int>((metadata.rows + kMatmulTileM - 1) / kMatmulTileM),
        static_cast<unsigned int>(pointer_count * coefficient_groups));
    matrix_matmul_batch_kernel<<<grid, block, 0, metadata.stream>>>(
        d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
        metadata.rows, metadata.inner, metadata.columns, metadata.n, coefficient_groups);
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
