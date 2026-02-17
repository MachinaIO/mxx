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
    uint64_t **dst,
    size_t poly_count,
    size_t limb_count,
    size_t n,
    const uint64_t *moduli,
    const uint32_t *limb_indices,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = limb_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t per_limb = poly_count * n;
    const size_t limb_slot = idx / per_limb;
    const size_t rem = idx - limb_slot * per_limb;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = limb_slot * poly_count + poly_idx;

    const uint64_t modulus = moduli[limb_slot];
    const uint32_t limb_idx = limb_indices[limb_slot];

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
    dst[ptr_idx][coeff_idx] = value;
}

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
    cudaStream_t stream,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id)
{
    const size_t limb_count = moduli.size();
    if (limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (limb_indices.size() != limb_count)
    {
        return set_error("unexpected limb parameter counts in matrix_fill_gadget_multi_limb_kernel");
    }
    const size_t ptr_count = limb_count * poly_count;
    if (dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_fill_gadget_multi_limb_kernel");
    }

    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = limb_count * sizeof(uint64_t);
    const size_t u32_bytes = limb_count * sizeof(uint32_t);
    const size_t moduli_offset = matrix_align_up_size(ptr_bytes, alignof(uint64_t));
    const size_t limb_indices_offset =
        matrix_align_up_size(moduli_offset + u64_bytes, alignof(uint32_t));
    const size_t workspace_bytes = limb_indices_offset + u32_bytes;

    void *workspace = nullptr;
    bool from_shared = false;
    int status = matrix_acquire_aux_workspace(
        aux_owner,
        aux_limb_id,
        workspace_bytes,
        &workspace,
        &from_shared);
    if (status != 0)
    {
        return status;
    }
    auto cleanup_workspace = [&]() -> int { return matrix_release_aux_workspace(workspace, from_shared); };

    auto *workspace_base = reinterpret_cast<uint8_t *>(workspace);
    uint64_t **d_dst = reinterpret_cast<uint64_t **>(workspace_base);
    uint64_t *d_moduli = reinterpret_cast<uint64_t *>(workspace_base + moduli_offset);
    uint32_t *d_limb_indices = reinterpret_cast<uint32_t *>(workspace_base + limb_indices_offset);

    cudaError_t err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_moduli, moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_limb_indices, limb_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_fill_gadget_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        d_dst,
        poly_count,
        limb_count,
        n,
        d_moduli,
        d_limb_indices,
        rows,
        cols,
        log_base_q,
        digits_per_tower,
        base_bits);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    status = cleanup_workspace();
    if (status != 0)
    {
        return status;
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
    auto &limb_map = out->ctx->ctx->limbGPUid;
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

    struct FillBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> moduli;
        std::vector<uint32_t> limb_indices;
        dim3 aux_limb_id;
        bool has_aux_limb_id;
    };
    std::vector<FillBatch> batches;
    auto get_batch = [&](int device) -> FillBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(FillBatch{device, nullptr, {}, {}, {}, dim3{0, 0, 0}, false});
        return batches.back();
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> limb_ptrs;
        limb_ptrs.reserve(count);
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;
        int status = matrix_limb_device(out, limb_id, &limb_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(out, limb_id, &limb_stream);
        if (status != 0)
        {
            return status;
        }

        for (size_t idx = 0; idx < count; ++idx)
        {
            uint64_t *dst = matrix_limb_ptr_by_id(out, idx, limb_id);
            if (!dst)
            {
                return set_error("null output limb pointer in gpu_matrix_fill_gadget");
            }
            limb_ptrs.push_back(dst);
        }

        if (limb_device < 0 || !limb_stream)
        {
            return set_error("invalid limb metadata in gpu_matrix_fill_gadget");
        }
        auto &batch = get_batch(limb_device);
        if (!batch.stream)
        {
            batch.stream = limb_stream;
        }
        if (!batch.has_aux_limb_id)
        {
            batch.aux_limb_id = limb_id;
            batch.has_aux_limb_id = true;
        }
        batch.moduli.push_back(out->ctx->moduli[static_cast<size_t>(limb)]);
        batch.limb_indices.push_back(static_cast<uint32_t>(limb));
        batch.dst_ptrs.insert(batch.dst_ptrs.end(), limb_ptrs.begin(), limb_ptrs.end());
    }

    for (auto &batch : batches)
    {
        if (!batch.stream)
        {
            return set_error("null stream in gpu_matrix_fill_gadget");
        }
        cudaError_t err = cudaSetDevice(batch.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int status = launch_fill_gadget_multi_limb_kernel(
            batch.dst_ptrs,
            count,
            static_cast<size_t>(out->ctx->N),
            batch.moduli,
            batch.limb_indices,
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            base_bits,
            batch.stream,
            out,
            batch.has_aux_limb_id ? &batch.aux_limb_id : nullptr);
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

    int status = sync_matrix_limb_streams(
        src,
        "failed to synchronize source limb stream before clone in gpu_matrix_decompose_base");
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }

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

    status = sync_matrix_limb_streams(
        inputs_matrix,
        "failed to synchronize input limb stream before decompose kernel in gpu_matrix_decompose_base");
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    const size_t out_count = out->rows * out->cols;
    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(crt_depth);
    auto add_out_stream = [&](int device, cudaStream_t stream) {
        for (const auto &entry : out_zero_streams)
        {
            if (entry.first == device && entry.second == stream)
            {
                return;
            }
        }
        out_zero_streams.emplace_back(device, stream);
    };

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
        add_out_stream(out_device, out_stream);
    }

    for (const auto &entry : out_zero_streams)
    {
        cudaError_t err = cudaSetDevice(entry.first);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.second);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    struct DecomposeBatch
    {
        int device;
        cudaStream_t stream;
        bool has_aux_limb;
        dim3 aux_limb_id;
        std::vector<const uint64_t *> src_ptrs;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint32_t> shifts;
        std::vector<uint64_t> masks;
        std::vector<uint64_t> out_moduli;
    };
    std::vector<DecomposeBatch> batches;
    auto get_batch = [&](int device) -> DecomposeBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(DecomposeBatch{device, nullptr, false, dim3{}, {}, {}, {}, {}, {}});
        return batches.back();
    };

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

        std::vector<const uint64_t *> src_ptrs;
        src_ptrs.reserve(count);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint64_t *src_ptr = matrix_limb_ptr_by_id(inputs_matrix, idx, src_limb_id);
            if (!src_ptr)
            {
                cleanup_tmp_inputs();
                return set_error("null source limb pointer in gpu_matrix_decompose_base");
            }
            src_ptrs.push_back(src_ptr);
        }

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

                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                for (size_t idx = 0; idx < count; ++idx)
                {
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    uint64_t *dst_ptr = matrix_limb_ptr_by_id(out, out_idx, out_limb_id);
                    if (!dst_ptr)
                    {
                        cleanup_tmp_inputs();
                        return set_error("null output limb pointer in gpu_matrix_decompose_base");
                    }
                    dst_ptrs.push_back(dst_ptr);
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
                }
                if (!batch_ref.has_aux_limb)
                {
                    batch_ref.aux_limb_id = out_limb_id;
                    batch_ref.has_aux_limb = true;
                }
                batch_ref.shifts.push_back(shift);
                batch_ref.masks.push_back(mask);
                batch_ref.out_moduli.push_back(src->ctx->moduli[static_cast<size_t>(out_limb)]);
                batch_ref.src_ptrs.insert(batch_ref.src_ptrs.end(), src_ptrs.begin(), src_ptrs.end());
                batch_ref.dst_ptrs.insert(batch_ref.dst_ptrs.end(), dst_ptrs.begin(), dst_ptrs.end());
            }
        }
    }

    for (auto &batch_ref : batches)
    {
        if (!batch_ref.stream)
        {
            cleanup_tmp_inputs();
            return set_error("null stream in gpu_matrix_decompose_base");
        }
        cudaError_t err = cudaSetDevice(batch_ref.device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        status = launch_decompose_multi_kernel<uint64_t>(
            batch_ref.src_ptrs,
            batch_ref.dst_ptrs,
            count,
            static_cast<size_t>(src->ctx->N),
            batch_ref.shifts,
            batch_ref.masks,
            batch_ref.out_moduli,
            batch_ref.stream,
            out,
            batch_ref.has_aux_limb ? &batch_ref.aux_limb_id : nullptr);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
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
