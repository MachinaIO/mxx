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

    void destroy_matrix_contents(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
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
    GpuPolyFormat fmt;
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

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}, {}};
    const size_t partition_count = ctx->ctx->meta.size();
    if (partition_count != ctx->ctx->GPUid.size())
    {
        destroy_matrix_contents(mat);
        delete mat;
        return set_error("unexpected context partition mapping in gpu_matrix_create");
    }
    mat->shared_limb_buffers.resize(partition_count);
    mat->shared_aux_buffers.resize(partition_count);

    const size_t n = static_cast<size_t>(ctx->N);

    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
    {
        auto &meta = ctx->ctx->meta[partition_idx];
        const size_t limbs = limb_count_for_level(meta, level);
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

        mat->shared_limb_buffers[partition_idx] = GpuMatrix::SharedLimbBuffer{
            ctx->ctx->GPUid[partition_idx],
            base,
            limbs,
            poly_words,
            total_words,
            n};
        mat->shared_aux_buffers[partition_idx] = GpuMatrix::SharedAuxBuffer{
            ctx->ctx->GPUid[partition_idx],
            aux_base,
            aux_slots,
            aux_total_slots};
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
    return gpu_matrix_copy_block(dst, src, 0, 0, 0, 0, src->rows, src->cols);
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
        int status = launch_copy_for_limb<uint64_t>(
            out,
            src,
            src_indices,
            dst_indices,
            limb_id,
            static_cast<size_t>(N));
        if (status != 0)
        {
            return status;
        }
    }

    out->format = src->format;
    return 0;
}
