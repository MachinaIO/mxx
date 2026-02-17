namespace
{
    bool checked_mul_size(size_t a, size_t b, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (a != 0 && b > static_cast<size_t>(-1) / a)
        {
            return false;
        }
        *out = a * b;
        return true;
    }

    size_t limb_count_for_level(const std::vector<FIDESlib::LimbRecord> &meta, int level)
    {
        size_t count = 0;
        for (const auto &record : meta)
        {
            if (record.id > level)
            {
                break;
            }
            ++count;
        }
        return count;
    }

    void free_matrix_shared_buffers(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        for (const auto &entry : mat->shared_limb_buffers)
        {
            if (!entry.ptr)
            {
                continue;
            }
            cudaSetDevice(entry.device);
            cudaFree(entry.ptr);
        }
        mat->shared_limb_buffers.clear();
        for (const auto &entry : mat->shared_aux_buffers)
        {
            if (!entry.ptr)
            {
                continue;
            }
            cudaSetDevice(entry.device);
            cudaFree(entry.ptr);
        }
        mat->shared_aux_buffers.clear();
    }

    void detach_matrix_shared_aux_buffers(GpuMatrix *mat)
    {
        if (!mat || mat->shared_aux_buffers.empty())
        {
            return;
        }
        for (auto *poly : mat->polys)
        {
            if (!poly || !poly->poly)
            {
                continue;
            }
            for (auto &partition : poly->poly->GPU)
            {
                partition.bufferAUXptrs = nullptr;
            }
        }
    }

    void destroy_matrix_contents(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        detach_matrix_shared_aux_buffers(mat);
        for (auto *poly : mat->polys)
        {
            gpu_poly_destroy(poly);
        }
        mat->polys.clear();
        free_matrix_shared_buffers(mat);
    }
}

extern "C" int gpu_matrix_create(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
    GpuMatrix **out)
{
    if (!ctx || !out)
    {
        return set_error("invalid gpu_matrix_create arguments");
    }
    *out = nullptr;
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_create");
    }
    if (level < -1 || level > ctx->ctx->L)
    {
        return set_error("invalid level in gpu_matrix_create");
    }

    size_t count = 0;
    if (!checked_mul_size(rows, cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_create");
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}, {}, {}};
    mat->polys.reserve(count);
    const size_t partition_count = ctx->ctx->meta.size();
    if (partition_count != ctx->ctx->GPUid.size())
    {
        destroy_matrix_contents(mat);
        delete mat;
        return set_error("unexpected context partition mapping in gpu_matrix_create");
    }

    std::vector<size_t> limb_counts(partition_count, 0);
    std::vector<size_t> words_per_poly(partition_count, 0);
    std::vector<size_t> words_total(partition_count, 0);
    std::vector<uint64_t *> device_bases(partition_count, nullptr);
    std::vector<void **> aux_device_bases(partition_count, nullptr);
    std::vector<size_t> aux_slots_per_poly(partition_count, 0);
    std::vector<size_t> aux_slots_total(partition_count, 0);
    std::vector<FIDESlib::Stream *> alloc_streams(partition_count, nullptr);
    const size_t n = static_cast<size_t>(ctx->N);

    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
    {
        auto &meta = ctx->ctx->meta[partition_idx];
        const size_t limbs = limb_count_for_level(meta, level);
        limb_counts[partition_idx] = limbs;
        if (limbs == 0 || count == 0)
        {
            continue;
        }

        size_t limb_words = 0;
        size_t poly_words = 0;
        size_t total_words = 0;
        size_t total_bytes = 0;
        if (!checked_mul_size(n, limbs, &limb_words) ||
            !checked_mul_size(limb_words, static_cast<size_t>(2), &poly_words) ||
            !checked_mul_size(poly_words, count, &total_words) ||
            !checked_mul_size(total_words, sizeof(uint64_t), &total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix limb allocation overflow in gpu_matrix_create");
        }

        cudaError_t err = cudaSetDevice(ctx->ctx->GPUid[partition_idx]);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        FIDESlib::Stream *alloc_stream = nullptr;
        for (size_t limb_idx = 0; limb_idx < meta.size(); ++limb_idx)
        {
            if (meta[limb_idx].id > level)
            {
                break;
            }
            alloc_stream = &meta[limb_idx].stream;
            break;
        }
        if (!alloc_stream || !alloc_stream->ptr)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("missing allocation stream in gpu_matrix_create");
        }

        uint64_t *base = nullptr;
        err = cudaMallocAsync(&base, total_bytes, alloc_stream->ptr);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        size_t aux_slots = 0;
        size_t aux_total_slots = 0;
        size_t aux_total_bytes = 0;
        const size_t decomp_count = ctx->ctx->decompMeta[partition_idx].size();
        if (!checked_mul_size(static_cast<size_t>(FIDESlib::MAXP), static_cast<size_t>(4 + 4 * decomp_count), &aux_slots) ||
            !checked_mul_size(aux_slots, count, &aux_total_slots) ||
            !checked_mul_size(aux_total_slots, sizeof(void *), &aux_total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix aux allocation overflow in gpu_matrix_create");
        }

        void **aux_base = nullptr;
        err = cudaMallocAsync(&aux_base, aux_total_bytes, alloc_stream->ptr);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        for (size_t limb_idx = 0; limb_idx < meta.size(); ++limb_idx)
        {
            if (meta[limb_idx].id > level)
            {
                break;
            }
            FIDESlib::Stream *limb_stream = &meta[limb_idx].stream;
            if (limb_stream != alloc_stream)
            {
                limb_stream->wait(*alloc_stream);
            }
        }

        device_bases[partition_idx] = base;
        aux_device_bases[partition_idx] = aux_base;
        aux_slots_per_poly[partition_idx] = aux_slots;
        aux_slots_total[partition_idx] = aux_total_slots;
        alloc_streams[partition_idx] = alloc_stream;
        words_per_poly[partition_idx] = poly_words;
        words_total[partition_idx] = total_words;
        mat->shared_limb_buffers.push_back(GpuMatrix::SharedLimbBuffer{
            ctx->ctx->GPUid[partition_idx],
            base});
        mat->shared_aux_buffers.push_back(GpuMatrix::SharedAuxBuffer{
            ctx->ctx->GPUid[partition_idx],
            aux_base});
    }

    std::vector<size_t> words_used(partition_count, 0);
    std::vector<size_t> aux_slots_used(partition_count, 0);
    for (size_t poly_idx = 0; poly_idx < count; ++poly_idx)
    {
        auto *poly_impl = new CKKS::RNSPoly(*ctx->ctx, -1, false);
        poly_impl->setLevel(level);
        auto *poly = new GpuPoly{poly_impl, ctx, level, fmt};

        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
        {
            if (partition_idx >= poly_impl->GPU.size())
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected poly partition size in gpu_matrix_create");
            }

            auto &partition = poly_impl->GPU[partition_idx];
            if (partition.device != ctx->ctx->GPUid[partition_idx])
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected partition device in gpu_matrix_create");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
            if (alloc_streams[partition_idx])
            {
                partition.s.wait(*alloc_streams[partition_idx]);
            }

            void **aux_base = aux_device_bases[partition_idx];
            if (!aux_base)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("missing shared aux buffer in gpu_matrix_create");
            }
            const size_t aux_poly_slots = aux_slots_per_poly[partition_idx];
            const size_t aux_offset_slots = aux_slots_used[partition_idx];
            if (aux_offset_slots > aux_slots_total[partition_idx] ||
                aux_poly_slots > aux_slots_total[partition_idx] - aux_offset_slots)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("shared aux buffer overflow in gpu_matrix_create");
            }
            if (partition.bufferAUXptrs)
            {
                err = cudaFree(partition.bufferAUXptrs);
                if (err != cudaSuccess)
                {
                    gpu_poly_destroy(poly);
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error(err);
                }
            }
            void **poly_aux_base = aux_base + aux_offset_slots;
            partition.bufferAUXptrs = poly_aux_base;
            const size_t aux_stride = static_cast<size_t>(FIDESlib::MAXP);
            const size_t decomp_ptr_count = partition.DECOMPlimbptr.size();
            if (partition.DECOMPauxptr.size() != decomp_ptr_count ||
                partition.DIGITlimbptr.size() != decomp_ptr_count ||
                partition.DIGITauxptr.size() != decomp_ptr_count)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected aux pointer layout in gpu_matrix_create");
            }
            partition.limbptr.data = poly_aux_base;
            partition.auxptr.data = poly_aux_base + aux_stride;
            partition.SPECIALlimbptr.data = poly_aux_base + 2 * aux_stride;
            partition.SPECIALauxptr.data = poly_aux_base + 3 * aux_stride;
            for (size_t decomp_idx = 0; decomp_idx < decomp_ptr_count; ++decomp_idx)
            {
                partition.DECOMPlimbptr[decomp_idx].data =
                    poly_aux_base + (4 + decomp_idx) * aux_stride;
                partition.DECOMPauxptr[decomp_idx].data =
                    poly_aux_base + (4 + decomp_ptr_count + decomp_idx) * aux_stride;
                partition.DIGITlimbptr[decomp_idx].data =
                    poly_aux_base + (4 + 2 * decomp_ptr_count + decomp_idx) * aux_stride;
                partition.DIGITauxptr[decomp_idx].data =
                    poly_aux_base + (4 + 3 * decomp_ptr_count + decomp_idx) * aux_stride;
            }
            aux_slots_used[partition_idx] = aux_offset_slots + aux_poly_slots;

            const size_t limbs = limb_counts[partition_idx];
            if (limbs == 0)
            {
                continue;
            }

            uint64_t *base = device_bases[partition_idx];
            if (!base)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("missing shared limb buffer in gpu_matrix_create");
            }

            const size_t poly_words = words_per_poly[partition_idx];
            const size_t limb_words = poly_words / 2;
            const size_t offset_words = words_used[partition_idx];
            if (offset_words > words_total[partition_idx] ||
                poly_words > words_total[partition_idx] - offset_words)
            {
                gpu_poly_destroy(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("shared limb buffer overflow in gpu_matrix_create");
            }

            partition.generate(
                partition.meta,
                partition.limb,
                partition.limbptr,
                static_cast<int>(limbs) - 1,
                &partition.auxptr,
                base,
                offset_words,
                base,
                offset_words + limb_words);
            words_used[partition_idx] = offset_words + poly_words;
        }

        mat->polys.push_back(poly);
    }
    *out = mat;
    return 0;
}

extern "C" void gpu_matrix_destroy(GpuMatrix *mat)
{
    if (!mat)
    {
        return;
    }

    destroy_matrix_contents(mat);
    delete mat;
}

extern "C" int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src)
{
    if (!dst || !src)
    {
        return set_error("invalid gpu_matrix_copy arguments");
    }
    if (dst->rows != src->rows || dst->cols != src->cols)
    {
        return set_error("size mismatch in gpu_matrix_copy");
    }
    if (dst->level != src->level || dst->ctx != src->ctx)
    {
        return set_error("context mismatch in gpu_matrix_copy");
    }
    const size_t count = src->rows * src->cols;
    for (size_t i = 0; i < count; ++i)
    {
        int status = gpu_poly_copy(dst->polys[i], src->polys[i]);
        if (status != 0)
        {
            return status;
        }
    }
    dst->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_entry_clone(
    const GpuMatrix *mat,
    size_t row,
    size_t col,
    GpuPoly **out_poly,
    GpuEventSet **out_events)
{
    if (!mat || !out_poly || !out_events)
    {
        return set_error("invalid gpu_matrix_entry_clone arguments");
    }
    *out_events = nullptr;
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_entry_clone");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_clone_async(mat->polys[idx], out_poly, out_events);
}

extern "C" int gpu_matrix_copy_entry(
    GpuMatrix *mat,
    size_t row,
    size_t col,
    const GpuPoly *src)
{
    if (!mat || !src)
    {
        return set_error("invalid gpu_matrix_copy_entry arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_copy_entry");
    }
    if (src->ctx != mat->ctx || src->level != mat->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_entry");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_copy(mat->polys[idx], src);
}

extern "C" int gpu_matrix_copy_block(
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
        return set_error("invalid gpu_matrix_copy_block arguments");
    }
    if (src_row + rows > src->rows || src_col + cols > src->cols)
    {
        return set_error("source bounds exceeded in gpu_matrix_copy_block");
    }
    if (dst_row + rows > out->rows || dst_col + cols > out->cols)
    {
        return set_error("dest bounds exceeded in gpu_matrix_copy_block");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_block");
    }

    const size_t count = rows * cols;
    if (count == 0)
    {
        out->format = src->format;
        return 0;
    }

    std::vector<size_t> src_indices;
    std::vector<size_t> dst_indices;
    src_indices.reserve(count);
    dst_indices.reserve(count);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            src_indices.push_back(matrix_index(src_row + i, src_col + j, src->cols));
            dst_indices.push_back(matrix_index(dst_row + i, dst_col + j, out->cols));
        }
    }

    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_copy_block");
    }
    const int N = src->ctx->N;
    if (N <= 0)
    {
        out->format = src->format;
        return 0;
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_copy_block");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        const GpuPoly *src_poly0 = src->polys[src_indices[0]];
        if (!src_poly0)
        {
            return set_error("null source polynomial in gpu_matrix_copy_block");
        }
        if (limb_id.x >= src_poly0->poly->GPU.size())
        {
            return set_error("unexpected limb GPU partition in gpu_matrix_copy_block");
        }
        const auto &src_partition0 = src_poly0->poly->GPU[limb_id.x];
        if (limb_id.y >= src_partition0.limb.size())
        {
            return set_error("unexpected limb index in gpu_matrix_copy_block");
        }

        const auto &limb_impl = src_partition0.limb[limb_id.y];
        int status = 0;
        if (limb_impl.index() == FIDESlib::U64)
        {
            status = launch_copy_for_limb<uint64_t>(
                out,
                src,
                src_indices,
                dst_indices,
                limb_id,
                static_cast<size_t>(N));
        }
        else if (limb_impl.index() == FIDESlib::U32)
        {
            status = launch_copy_for_limb<uint32_t>(
                out,
                src,
                src_indices,
                dst_indices,
                limb_id,
                static_cast<size_t>(N));
        }
        else
        {
            return set_error("unsupported limb type in gpu_matrix_copy_block");
        }
        if (status != 0)
        {
            return status;
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        const size_t src_idx = src_indices[i];
        const size_t dst_idx = dst_indices[i];
        if (!out->polys[dst_idx] || !src->polys[src_idx])
        {
            return set_error("null polynomial after copy in gpu_matrix_copy_block");
        }
        out->polys[dst_idx]->level = src->polys[src_idx]->level;
        out->polys[dst_idx]->format = src->polys[src_idx]->format;
    }

    out->format = src->format;
    return 0;
}
