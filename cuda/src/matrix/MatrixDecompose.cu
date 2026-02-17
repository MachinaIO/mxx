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
    cudaStream_t stream)
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

    uint64_t **d_dst = nullptr;
    uint64_t *d_moduli = nullptr;
    uint32_t *d_limb_indices = nullptr;
    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = limb_count * sizeof(uint64_t);
    const size_t u32_bytes = limb_count * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&d_dst, ptr_bytes);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMalloc(&d_moduli, u64_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        return set_error(err);
    }
    err = cudaMalloc(&d_limb_indices, u32_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_moduli, moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_limb_indices, limb_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
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
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    cudaFree(d_dst);
    cudaFree(d_moduli);
    cudaFree(d_limb_indices);
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
        out->format = PolyFormat::Eval;
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

    struct FillBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> moduli;
        std::vector<uint32_t> limb_indices;
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
        batches.push_back(FillBatch{device, nullptr, {}, {}, {}});
        return batches.back();
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> limb_ptrs;
        limb_ptrs.reserve(count);
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out->polys[idx].get();
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                return set_error("invalid output poly in gpu_matrix_fill_gadget");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_matrix_fill_gadget");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_matrix_fill_gadget");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                return set_error("unsupported limb type in gpu_matrix_fill_gadget");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            if (limb_device == -1)
            {
                limb_device = partition.device;
                limb_stream = limb_u64.stream.ptr;
            }
            else if (limb_device != partition.device)
            {
                return set_error("inconsistent limb device in gpu_matrix_fill_gadget");
            }
            limb_ptrs.push_back(limb_u64.v.data);
        }

        if (limb_device < 0)
        {
            return set_error("invalid limb device in gpu_matrix_fill_gadget");
        }
        auto &batch = get_batch(limb_device);
        if (!batch.stream)
        {
            batch.stream = limb_stream;
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
            batch.stream);
        if (status != 0)
        {
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto &poly_holder : out->polys)
    {
        GpuPoly *poly = poly_holder.get();
        poly->format = PolyFormat::Coeff;
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;
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
        out->format = PolyFormat::Eval;
        return 0;
    }

    GpuMatrix *tmp_inputs_matrix = nullptr;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        for (size_t i = 0; i < count; ++i)
        {
            int status = sync_poly_partition_streams(
                src->polys[i].get(),
                "failed to synchronize source partition stream before clone in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = sync_poly_limb_streams(
                src->polys[i].get(),
                "failed to synchronize source limb stream before clone in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
        }

        const int matrix_format =
            src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        int status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
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

        for (auto &clone_holder : tmp_inputs_matrix->polys)
        {
            GpuPoly *clone = clone_holder.get();
            status = sync_poly_partition_streams(
                clone,
                "failed to synchronize clone partition stream before gpu_poly_intt in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = sync_poly_limb_streams(
                clone,
                "failed to synchronize clone limb stream before gpu_poly_intt in gpu_matrix_decompose_base");
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            inputs.push_back(clone);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            inputs.push_back(src->polys[i].get());
        }
    }

    // Ensure all pending source-side INTT work is finished before one-shot kernels.
    for (int device : src->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(out->polys.size() * crt_depth);

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx].get();
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output poly in gpu_matrix_decompose_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb GPU partition in gpu_matrix_decompose_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb index in gpu_matrix_decompose_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                cleanup_tmp_inputs();
                return set_error("unsupported limb type in gpu_matrix_decompose_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }

            bool seen_stream = false;
            for (const auto &entry : out_zero_streams)
            {
                if (entry.first == partition.device && entry.second == limb_u64.stream.ptr)
                {
                    seen_stream = true;
                    break;
                }
            }
            if (!seen_stream)
            {
                out_zero_streams.emplace_back(partition.device, limb_u64.stream.ptr);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    // Output limbs are zeroed asynchronously on per-limb streams.
    // Synchronize those streams once before decomposition kernels start.
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
        batches.push_back(DecomposeBatch{device, nullptr, {}, {}, {}, {}, {}});
        return batches.back();
    };

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb GPU partition in gpu_matrix_decompose_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb index in gpu_matrix_decompose_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            cleanup_tmp_inputs();
            return set_error("unsupported source limb type in gpu_matrix_decompose_base");
        }

        const uint32_t src_bits = bit_width_u64(src->ctx->moduli[static_cast<size_t>(src_limb)]);

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
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    cleanup_tmp_inputs();
                    return set_error("unexpected input source limb index in gpu_matrix_decompose_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    cleanup_tmp_inputs();
                    return set_error("unsupported input source limb type in gpu_matrix_decompose_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;
                int out_device = -1;

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("input/output limb device mismatch in gpu_matrix_decompose_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        cleanup_tmp_inputs();
                        return set_error("unexpected output limb index in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        cleanup_tmp_inputs();
                        return set_error("unsupported output limb type in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                        out_device = out_partition.device;
                    }
                    else if (out_device != out_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("inconsistent output limb device in gpu_matrix_decompose_base");
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                if (out_device < 0 || !out_stream)
                {
                    cleanup_tmp_inputs();
                    return set_error("invalid output stream/device in gpu_matrix_decompose_base");
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
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
        int status = launch_decompose_multi_kernel<uint64_t>(
            batch_ref.src_ptrs,
            batch_ref.dst_ptrs,
            count,
            static_cast<size_t>(src->ctx->N),
            batch_ref.shifts,
            batch_ref.masks,
            batch_ref.out_moduli,
            batch_ref.stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
    }

    for (auto &poly_holder : out->polys)
    {
        GpuPoly *poly = poly_holder.get();
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup_tmp_inputs();
    return 0;
}
