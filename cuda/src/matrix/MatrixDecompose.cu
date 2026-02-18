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
    uint64_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride,
    uint64_t modulus,
    uint32_t limb_idx,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;

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
    dst_base[poly_idx * dst_stride + coeff_idx] = value;
}

int launch_fill_gadget_multi_limb_kernel(
    uint64_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride,
    uint64_t modulus,
    uint32_t limb_idx,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits,
    cudaStream_t stream,
    const GpuMatrix *,
    const dim3 *)
{
    if (!dst_base)
    {
        return set_error("null output base pointer in matrix_fill_gadget_multi_limb_kernel");
    }
    if (poly_count == 0 || n == 0)
    {
        return 0;
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_fill_gadget_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        dst_base,
        poly_count,
        n,
        dst_stride,
        modulus,
        limb_idx,
        rows,
        cols,
        log_base_q,
        digits_per_tower,
        base_bits);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
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
        out->format = GPU_POLY_FORMAT_EVAL;
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
    auto &limb_map = out->ctx->limb_gpu_ids;
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
    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;
        status = matrix_limb_device(out, limb_id, &limb_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(out, limb_id, &limb_stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_device < 0 || !limb_stream)
        {
            return set_error("invalid limb metadata in gpu_matrix_fill_gadget");
        }
        uint64_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            return set_error("null output limb base pointer in gpu_matrix_fill_gadget");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid output partition index in gpu_matrix_fill_gadget");
        }
        const size_t dst_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;

        cudaError_t err = cudaSetDevice(limb_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = launch_fill_gadget_multi_limb_kernel(
            dst_base,
            count,
            static_cast<size_t>(out->ctx->N),
            dst_stride,
            out->ctx->moduli[static_cast<size_t>(limb)],
            static_cast<uint32_t>(limb),
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            base_bits,
            limb_stream,
            out,
            &limb_id);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, limb_stream);
        if (status != 0)
        {
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    status = gpu_matrix_ntt_all(out, default_batch(out->ctx));
    if (status != 0)
    {
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;
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
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_decompose_base");
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
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    GpuMatrix *tmp_inputs_matrix = nullptr;
    const GpuMatrix *inputs_matrix = src;
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };

    int status = 0;
    const int batch = default_batch(src->ctx);
    if (src->format == GPU_POLY_FORMAT_EVAL)
    {
        const int matrix_format =
            src->format == GPU_POLY_FORMAT_EVAL ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
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
        status = gpu_matrix_intt_all(tmp_inputs_matrix, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        inputs_matrix = tmp_inputs_matrix;
    }

    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    const size_t out_count = out->rows * out->cols;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        uint64_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb pointer in gpu_matrix_decompose_base");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            cleanup_tmp_inputs();
            return set_error("invalid partition index in gpu_matrix_decompose_base");
        }
        const auto &buffer = out->shared_limb_buffers[limb_id.x];
        const size_t dst_pitch = buffer.words_per_poly * sizeof(uint64_t);
        const size_t coeff_bytes = static_cast<size_t>(src->ctx->N) * sizeof(uint64_t);
        if (out_count > 0)
        {
            err = cudaMemset2DAsync(dst, dst_pitch, 0, coeff_bytes, out_count, out_stream);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
        }
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        int src_device = -1;
        status = matrix_limb_device(inputs_matrix, src_limb_id, &src_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        const uint32_t src_bits = bit_width_u64(src->ctx->moduli[static_cast<size_t>(src_limb)]);

        const uint64_t *src_base = matrix_limb_ptr_by_id(inputs_matrix, 0, src_limb_id);
        if (!src_base)
        {
            cleanup_tmp_inputs();
            return set_error("null source limb base pointer in gpu_matrix_decompose_base");
        }
        if (src_limb_id.x >= inputs_matrix->shared_limb_buffers.size())
        {
            cleanup_tmp_inputs();
            return set_error("invalid source partition index in gpu_matrix_decompose_base");
        }
        const size_t src_stride = inputs_matrix->shared_limb_buffers[src_limb_id.x].words_per_poly;

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

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                int out_device = -1;
                status = matrix_limb_device(out, out_limb_id, &out_device);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                if (out_device != src_device)
                {
                    cleanup_tmp_inputs();
                    return set_error("input/output limb device mismatch in gpu_matrix_decompose_base");
                }
                cudaStream_t out_stream = nullptr;
                status = matrix_limb_stream(out, out_limb_id, &out_stream);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                status = matrix_wait_limb_stream(inputs_matrix, src_limb_id, out_device, out_stream);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                uint64_t *dst_base = matrix_limb_ptr_by_id(out, 0, out_limb_id);
                if (!dst_base)
                {
                    cleanup_tmp_inputs();
                    return set_error("null output limb base pointer in gpu_matrix_decompose_base");
                }
                if (out_limb_id.x >= out->shared_limb_buffers.size())
                {
                    cleanup_tmp_inputs();
                    return set_error("invalid output partition index in gpu_matrix_decompose_base");
                }
                const size_t dst_stride = out->shared_limb_buffers[out_limb_id.x].words_per_poly;

                status = launch_decompose_multi_kernel<uint64_t>(
                    src_base,
                    dst_base,
                    count,
                    static_cast<size_t>(src->ctx->N),
                    src_stride,
                    dst_stride,
                    cols,
                    out->cols,
                    log_base_q,
                    digit_offset,
                    shift,
                    mask,
                    src->ctx->moduli[static_cast<size_t>(out_limb)],
                    out_stream,
                    out,
                    &out_limb_id);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                status = matrix_track_limb_consumer(
                    inputs_matrix,
                    src_limb_id,
                    out_device,
                    out_stream);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                status = matrix_record_limb_write(out, out_limb_id, out_stream);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
            }
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    if (requested_out_format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_ntt_all(out, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        out->format = GPU_POLY_FORMAT_EVAL;
    }

    cleanup_tmp_inputs();
    return 0;
}
