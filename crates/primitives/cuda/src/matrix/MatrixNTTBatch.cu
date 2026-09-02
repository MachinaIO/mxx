namespace
{
    __global__ void batch_ntt_twist_kernel(
        uint8_t *const *bases,
        const size_t *strides,
        const uint8_t *widths,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        size_t poly_count)
    {
        const uint32_t coefficient = blockIdx.x * blockDim.x + threadIdx.x;
        if (coefficient >= n || blockIdx.y >= poly_count) return;
        const size_t matrix_limb = blockIdx.z;
        const size_t limb = matrix_limb % limb_count;
        uint8_t *base = bases[matrix_limb];
        const uint64_t modulus = moduli[limb];
        const uint64_t value = matrix_load_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb]);
        const size_t twiddle_index = limb * static_cast<size_t>(n) + coefficient;
        const uint64_t twist = twiddles[twiddle_index];
        matrix_store_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb],
            mul_mod_shoup_u64(
                value,
                twist,
                twiddle_shoup[twiddle_index],
                modulus));
    }

    __global__ void batch_ntt_bit_reverse_kernel(
        uint8_t *const *bases,
        const size_t *strides,
        const uint8_t *widths,
        size_t limb_count,
        uint32_t n,
        uint32_t log_n,
        size_t poly_count)
    {
        const uint32_t coefficient = blockIdx.x * blockDim.x + threadIdx.x;
        if (coefficient >= n || blockIdx.y >= poly_count) return;
        const uint32_t reversed = __brev(coefficient) >> (32 - log_n);
        if (coefficient >= reversed) return;
        const size_t matrix_limb = blockIdx.z;
        const size_t limb = matrix_limb % limb_count;
        uint8_t *base = bases[matrix_limb];
        const uint64_t left = matrix_load_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb]);
        const uint64_t right = matrix_load_limb_u64(
            base, blockIdx.y, reversed, strides[limb], widths[limb]);
        matrix_store_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb], right);
        matrix_store_limb_u64(
            base, blockIdx.y, reversed, strides[limb], widths[limb], left);
    }

    __global__ void batch_ntt_stage_kernel(
        uint8_t *const *bases,
        const size_t *strides,
        const uint8_t *widths,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        uint32_t len,
        size_t poly_count)
    {
        const uint32_t butterfly = blockIdx.x * blockDim.x + threadIdx.x;
        if (butterfly >= n / 2 || blockIdx.y >= poly_count) return;
        const size_t matrix_limb = blockIdx.z;
        const size_t limb = matrix_limb % limb_count;
        const uint32_t half = len / 2;
        const uint32_t group = butterfly / half;
        const uint32_t j = butterfly % half;
        const uint32_t index = group * len + j;
        uint8_t *base = bases[matrix_limb];
        const uint64_t modulus = moduli[limb];
        const uint32_t twiddle_exponent = 2U * (n / len) * j;
        const size_t twiddle_index =
            limb * static_cast<size_t>(n) + twiddle_exponent;
        const uint64_t twiddle = twiddles[twiddle_index];
        const uint64_t lower = matrix_load_limb_u64(
            base, blockIdx.y, index, strides[limb], widths[limb]);
        const uint64_t upper = matrix_load_limb_u64(
            base, blockIdx.y, index + half, strides[limb], widths[limb]);
        const uint64_t product = mul_mod_shoup_u64(
            upper,
            twiddle,
            twiddle_shoup[twiddle_index],
            modulus);
        matrix_store_limb_u64(
            base, blockIdx.y, index, strides[limb], widths[limb],
            add_mod_u64(lower, product, modulus));
        matrix_store_limb_u64(
            base, blockIdx.y, index + half, strides[limb], widths[limb],
            sub_mod_u64(lower, product, modulus));
    }

    __global__ void batch_ntt_first_stage_out_of_place_kernel(
        const uint8_t *const *inputs,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *widths,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        size_t poly_count)
    {
        const uint32_t butterfly = blockIdx.x * blockDim.x + threadIdx.x;
        if (butterfly >= n / 2 || blockIdx.y >= poly_count) return;
        const size_t matrix_limb = blockIdx.z;
        const size_t limb = matrix_limb % limb_count;
        const uint32_t index = butterfly * 2;
        const uint64_t modulus = moduli[limb];
        const uint64_t lower = matrix_load_limb_u64(
            inputs[matrix_limb], blockIdx.y, index, strides[limb], widths[limb]);
        const uint64_t upper = matrix_load_limb_u64(
            inputs[matrix_limb], blockIdx.y, index + 1, strides[limb], widths[limb]);
        const uint64_t product = upper;
        matrix_store_limb_u64(
            outputs[matrix_limb], blockIdx.y, index, strides[limb], widths[limb],
            add_mod_u64(lower, product, modulus));
        matrix_store_limb_u64(
            outputs[matrix_limb], blockIdx.y, index + 1, strides[limb], widths[limb],
            sub_mod_u64(lower, product, modulus));
    }

    __global__ void batch_ntt_scale_twist_kernel(
        uint8_t *const *bases,
        const size_t *strides,
        const uint8_t *widths,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        const uint64_t *n_inv,
        const uint64_t *n_inv_shoup,
        size_t limb_count,
        uint32_t n,
        size_t poly_count)
    {
        const uint32_t coefficient = blockIdx.x * blockDim.x + threadIdx.x;
        if (coefficient >= n || blockIdx.y >= poly_count) return;
        const size_t matrix_limb = blockIdx.z;
        const size_t limb = matrix_limb % limb_count;
        uint8_t *base = bases[matrix_limb];
        const uint64_t modulus = moduli[limb];
        const uint64_t value = matrix_load_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb]);
        const uint64_t scaled = mul_mod_shoup_u64(
            value,
            n_inv[limb],
            n_inv_shoup[limb],
            modulus);
        const size_t twiddle_index = limb * static_cast<size_t>(n) + coefficient;
        const uint64_t twist = twiddles[twiddle_index];
        matrix_store_limb_u64(
            base, blockIdx.y, coefficient, strides[limb], widths[limb],
            mul_mod_shoup_u64(
                scaled,
                twist,
                twiddle_shoup[twiddle_index],
                modulus));
    }
}

namespace
{
int run_matrix_transform_batch(
    GpuMatrix *const *matrices,
    const GpuMatrix *const *sources,
    size_t matrix_count,
    bool forward)
{
    if (!matrices || matrix_count == 0 || !matrices[0] || !matrices[0]->ctx)
        return set_error("invalid run_matrix_transform_batch arguments");
    GpuMatrix *first = matrices[0];
    const GpuPolyFormat input_format =
        forward ? GPU_POLY_FORMAT_COEFF : GPU_POLY_FORMAT_EVAL;
    const GpuPolyFormat output_format =
        forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
    if (!sources && first->format == output_format)
    {
        for (size_t index = 1; index < matrix_count; ++index)
            if (!matrices[index] || matrices[index]->format != output_format)
                return set_error("mixed formats in run_matrix_transform_batch");
        return 0;
    }
    const uint32_t n = static_cast<uint32_t>(first->ctx->N);
    const size_t limb_count = static_cast<size_t>(first->level + 1);
    const size_t poly_count = matrix_poly_count(first);
    if (!is_power_of_two_u32(n) || n < 2 || limb_count == 0 || poly_count == 0)
        return set_error("invalid matrix shape in gpu_matrix_transform_batch");
    uint32_t log_n = 0;
    for (uint32_t value = n; value > 1; value >>= 1) ++log_n;
    if (matrix_count * limb_count > 65535 || poly_count > 65535)
        return set_error("gpu_matrix_transform_batch exceeds CUDA grid dimensions");
    const auto &limb_ids = first->ctx->limb_gpu_ids;
    if (limb_ids.size() < limb_count) return set_error("missing batch INTT limb mapping");
    std::vector<uint8_t *> bases(matrix_count * limb_count);
    std::vector<const uint8_t *> source_bases;
    if (sources) source_bases.resize(matrix_count * limb_count);
    std::vector<size_t> strides(limb_count);
    std::vector<uint8_t> widths(limb_count);
    int device = -1;
    cudaStream_t stream = nullptr;
    for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
    {
        GpuMatrix *matrix = matrices[matrix_index];
        const GpuMatrix *source = sources ? sources[matrix_index] : matrix;
        if (!matrix || !source || matrix->ctx != first->ctx || source->ctx != first->ctx ||
            matrix->rows != first->rows || matrix->cols != first->cols ||
            source->rows != first->rows || source->cols != first->cols ||
            matrix->level != first->level || source->level != first->level ||
            source->format != input_format || (sources && forward))
            return set_error("run_matrix_transform_batch requires homogeneous matrices");
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 limb_id = limb_ids[limb];
            int limb_device = -1;
            int source_device = -1;
            size_t stride = 0;
            size_t source_stride = 0;
            uint8_t width = 0;
            uint8_t source_width = 0;
            uint8_t *base = matrix_limb_ptr_by_id(matrix, 0, limb_id);
            const uint8_t *source_base = matrix_limb_ptr_by_id(source, 0, limb_id);
            if (!base || !source_base || matrix_limb_device(matrix, limb_id, &limb_device) != 0 ||
                matrix_limb_device(source, limb_id, &source_device) != 0 ||
                !matrix_limb_metadata_by_id(matrix, limb_id, &stride, &width) ||
                !matrix_limb_metadata_by_id(source, limb_id, &source_stride, &source_width) ||
                source_device != limb_device || source_stride != stride || source_width != width)
                return set_error("invalid limb in gpu_matrix_transform_batch");
            if (device < 0)
            {
                device = limb_device;
                if (matrix_limb_stream(matrix, limb_id, &stream) != 0 || !stream)
                    return set_error("missing batch INTT stream");
            }
            else if (limb_device != device)
                return set_error("gpu_matrix_transform_batch requires one placement");
            if (matrix_index == 0)
            {
                strides[limb] = stride;
                widths[limb] = width;
            }
            else if (strides[limb] != stride || widths[limb] != width)
                return set_error("incompatible batch INTT limb layout");
            bases[matrix_index * limb_count + limb] = base;
            if (sources) source_bases[matrix_index * limb_count + limb] = source_base;
            const int status = matrix_wait_limb_stream(source, limb_id, device, stream);
            if (status != 0) return status;
        }
    }
    const size_t partition = static_cast<size_t>(limb_ids[0].x);
    if (partition >= first->ctx->ntt_device_constants.size())
        return set_error("missing batch INTT constants");
    const auto &constants = first->ctx->ntt_device_constants[partition];
    if (constants.device != device || constants.ring_dimension != n ||
        constants.limb_count < limb_count || !constants.twiddle_inverse ||
        !constants.twiddle_forward || !constants.twiddle_shoup_inverse ||
        !constants.twiddle_shoup_forward || !constants.moduli ||
        !constants.n_inv || !constants.n_inv_shoup)
        return set_error("incompatible batch INTT constants");
    const uint64_t *twiddles =
        forward ? constants.twiddle_forward : constants.twiddle_inverse;
    const uint64_t *twiddle_shoup = forward ?
        constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse;

    cudaError_t error = cudaSetDevice(device);
    uint8_t **device_bases = nullptr;
    const uint8_t **device_sources = nullptr;
    size_t *device_strides = nullptr;
    uint8_t *device_widths = nullptr;
    auto release = [&]() {
        if (device_widths) cudaFreeAsync(device_widths, stream);
        if (device_strides) cudaFreeAsync(device_strides, stream);
        if (device_sources) cudaFreeAsync(device_sources, stream);
        if (device_bases) cudaFreeAsync(device_bases, stream);
    };
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_bases), bases.size() * sizeof(uint8_t *), stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_strides), strides.size() * sizeof(size_t), stream);
    if (error == cudaSuccess) error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_widths), widths.size(), stream);
    if (error == cudaSuccess && sources) error = cudaMallocAsync(
        reinterpret_cast<void **>(&device_sources),
        source_bases.size() * sizeof(uint8_t *), stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        device_bases, bases.data(), bases.size() * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        device_strides, strides.data(), strides.size() * sizeof(size_t),
        cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        device_widths, widths.data(), widths.size(), cudaMemcpyHostToDevice, stream);
    if (error == cudaSuccess && sources) error = cudaMemcpyAsync(
        device_sources, source_bases.data(), source_bases.size() * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    const dim3 coefficient_grid(
        (n + kTransformThreads - 1) / kTransformThreads,
        static_cast<uint32_t>(poly_count),
        static_cast<uint32_t>(bases.size()));
    if (forward)
    {
        batch_ntt_twist_kernel<<<coefficient_grid, kTransformThreads, 0, stream>>>(
            device_bases,
            device_strides,
            device_widths,
            twiddles,
            twiddle_shoup,
            constants.moduli,
            limb_count,
            n,
            poly_count);
        error = cudaGetLastError();
        if (error == cudaSuccess)
        {
            batch_ntt_bit_reverse_kernel<<<coefficient_grid, kTransformThreads, 0, stream>>>(
                device_bases, device_strides, device_widths,
                limb_count, n, log_n, poly_count);
            error = cudaGetLastError();
        }
        if (error != cudaSuccess)
        {
            release();
            return set_error(error);
        }
    }
    const dim3 stage_grid(
        (n / 2 + kTransformThreads - 1) / kTransformThreads,
        static_cast<uint32_t>(poly_count),
        static_cast<uint32_t>(bases.size()));
    uint32_t first_length = 2;
    if (sources)
    {
        batch_ntt_first_stage_out_of_place_kernel<<<
            stage_grid, kTransformThreads, 0, stream>>>(
            device_sources,
            device_bases,
            device_strides,
            device_widths,
            constants.moduli,
            limb_count,
            n,
            poly_count);
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            release();
            return set_error(error);
        }
        first_length = 4;
    }
    for (uint32_t len = first_length; len <= n; len <<= 1)
    {
        batch_ntt_stage_kernel<<<stage_grid, kTransformThreads, 0, stream>>>(
            device_bases,
            device_strides,
            device_widths,
            twiddles,
            twiddle_shoup,
            constants.moduli,
            limb_count, n, len, poly_count);
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            release();
            return set_error(error);
        }
    }
    if (forward)
    {
        batch_ntt_bit_reverse_kernel<<<coefficient_grid, kTransformThreads, 0, stream>>>(
            device_bases, device_strides, device_widths,
            limb_count, n, log_n, poly_count);
    }
    else
    {
        batch_ntt_scale_twist_kernel<<<coefficient_grid, kTransformThreads, 0, stream>>>(
            device_bases,
            device_strides,
            device_widths,
            twiddles,
            twiddle_shoup,
            constants.moduli,
            constants.n_inv,
            constants.n_inv_shoup,
            limb_count,
            n,
            poly_count);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        release();
        return set_error(error);
    }
    for (size_t matrix_index = 0; matrix_index < matrix_count; ++matrix_index)
    {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            int status = 0;
            if (sources)
            {
                status = matrix_track_limb_consumer(
                    sources[matrix_index], limb_ids[limb], device, stream);
            }
            if (status == 0)
            {
                status = matrix_record_limb_write(
                matrices[matrix_index], limb_ids[limb], stream);
            }
            if (status != 0)
            {
                release();
                return status;
            }
        }
        matrices[matrix_index]->format = output_format;
    }
    release();
    return 0;
}
}

extern "C" int gpu_matrix_intt_out_of_place_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t matrix_count)
{
    return run_matrix_transform_batch(outputs, inputs, matrix_count, false);
}

extern "C" int gpu_matrix_ntt_in_place_batch(
    GpuMatrix *const *matrices,
    size_t matrix_count)
{
    return run_matrix_transform_batch(matrices, nullptr, matrix_count, true);
}
