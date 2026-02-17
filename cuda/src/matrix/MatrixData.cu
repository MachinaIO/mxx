namespace
{
    struct StreamRef
    {
        int device;
        cudaStream_t stream;
    };

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

    void append_unique_stream(std::vector<StreamRef> &streams, int device, cudaStream_t stream)
    {
        if (!stream)
        {
            return;
        }
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                return;
            }
        }
        streams.push_back(StreamRef{device, stream});
    }

    int wait_stream_on_stream(int device, cudaStream_t consumer, cudaStream_t producer)
    {
        if (!consumer || !producer || consumer == producer)
        {
            return 0;
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        cudaEvent_t ev = nullptr;
        err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaEventRecord(ev, producer);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(ev);
            return set_error(err);
        }
        err = cudaStreamWaitEvent(consumer, ev, 0);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(ev);
            return set_error(err);
        }
        err = cudaEventDestroy(ev);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int build_event_set_from_streams(const std::vector<StreamRef> &streams, GpuEventSet **out_events)
    {
        if (!out_events)
        {
            return set_error("invalid out_events in build_event_set_from_streams");
        }
        *out_events = nullptr;
        if (streams.empty())
        {
            return 0;
        }

        auto *event_set = new GpuEventSet();
        event_set->entries.reserve(streams.size());

        for (const auto &entry : streams)
        {
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                gpu_event_set_destroy(event_set);
                return set_error(err);
            }

            cudaEvent_t ev = nullptr;
            err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                gpu_event_set_destroy(event_set);
                return set_error(err);
            }
            err = cudaEventRecord(ev, entry.stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(ev);
                gpu_event_set_destroy(event_set);
                return set_error(err);
            }
            event_set->entries.push_back(GpuEventSet::Entry{ev, entry.device});
        }

        *out_events = event_set;
        return 0;
    }

    int get_poly_limb_u64_const(
        const GpuPoly *poly,
        const dim3 &limb_id,
        const uint64_t **out_ptr,
        cudaStream_t *out_stream,
        int *out_device)
    {
        if (!poly || !poly->poly || !out_ptr || !out_stream || !out_device)
        {
            return set_error("invalid get_poly_limb_u64_const arguments");
        }
        if (limb_id.x >= poly->poly->GPU.size())
        {
            return set_error("invalid partition index in get_poly_limb_u64_const");
        }
        const auto &partition = poly->poly->GPU[limb_id.x];
        if (limb_id.y >= partition.limb.size())
        {
            return set_error("invalid limb index in get_poly_limb_u64_const");
        }
        const auto &limb_impl = partition.limb[limb_id.y];
        if (limb_impl.index() != FIDESlib::U64)
        {
            return set_error("unsupported limb type in get_poly_limb_u64_const");
        }
        const auto &u64 = std::get<FIDESlib::U64>(limb_impl);
        if (!u64.v.data || !u64.stream.ptr)
        {
            return set_error("null limb storage in get_poly_limb_u64_const");
        }
        *out_ptr = u64.v.data;
        *out_stream = u64.stream.ptr;
        *out_device = partition.device;
        return 0;
    }

    int get_poly_limb_u64_mut(
        GpuPoly *poly,
        const dim3 &limb_id,
        uint64_t **out_ptr,
        cudaStream_t *out_stream,
        int *out_device)
    {
        if (!poly || !poly->poly || !out_ptr || !out_stream || !out_device)
        {
            return set_error("invalid get_poly_limb_u64_mut arguments");
        }
        if (limb_id.x >= poly->poly->GPU.size())
        {
            return set_error("invalid partition index in get_poly_limb_u64_mut");
        }
        auto &partition = poly->poly->GPU[limb_id.x];
        if (limb_id.y >= partition.limb.size())
        {
            return set_error("invalid limb index in get_poly_limb_u64_mut");
        }
        auto &limb_impl = partition.limb[limb_id.y];
        if (limb_impl.index() != FIDESlib::U64)
        {
            return set_error("unsupported limb type in get_poly_limb_u64_mut");
        }
        auto &u64 = std::get<FIDESlib::U64>(limb_impl);
        if (!u64.v.data || !u64.stream.ptr)
        {
            return set_error("null limb storage in get_poly_limb_u64_mut");
        }
        *out_ptr = u64.v.data;
        *out_stream = u64.stream.ptr;
        *out_device = partition.device;
        return 0;
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
    if (!mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid context in gpu_matrix_entry_clone");
    }
    *out_poly = nullptr;
    const size_t idx = matrix_index(row, col, mat->cols);
    int status = gpu_poly_create(mat->ctx, mat->level, out_poly);
    if (status != 0)
    {
        return status;
    }
    (*out_poly)->format = mat->format;

    const int N = mat->ctx->N;
    if (N <= 0 || mat->level < 0)
    {
        return 0;
    }

    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(mat->level + 1))
    {
        gpu_poly_destroy(*out_poly);
        *out_poly = nullptr;
        return set_error("unexpected limb mapping size in gpu_matrix_entry_clone");
    }

    std::vector<StreamRef> streams;
    streams.reserve(static_cast<size_t>(mat->level + 1));
    for (int limb = 0; limb <= mat->level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];

        const uint64_t *src_ptr = matrix_limb_ptr_by_id(mat, idx, limb_id);
        if (!src_ptr)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return set_error("null matrix limb pointer in gpu_matrix_entry_clone");
        }

        int src_device = -1;
        cudaStream_t src_stream = nullptr;
        status = matrix_limb_device(mat, limb_id, &src_device);
        if (status != 0)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &src_stream);
        if (status != 0)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return status;
        }

        uint64_t *dst_ptr = nullptr;
        int dst_device = -1;
        cudaStream_t dst_stream = nullptr;
        status = get_poly_limb_u64_mut(*out_poly, limb_id, &dst_ptr, &dst_stream, &dst_device);
        if (status != 0)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return status;
        }
        if (src_device != dst_device)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return set_error("device mismatch in gpu_matrix_entry_clone");
        }

        cudaError_t err = cudaSetDevice(dst_device);
        if (err != cudaSuccess)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return set_error(err);
        }

        status = wait_stream_on_stream(dst_device, dst_stream, src_stream);
        if (status != 0)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return status;
        }

        err = cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            static_cast<size_t>(N) * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            dst_stream);
        if (err != cudaSuccess)
        {
            gpu_poly_destroy(*out_poly);
            *out_poly = nullptr;
            return set_error(err);
        }
        append_unique_stream(streams, dst_device, dst_stream);
    }

    status = build_event_set_from_streams(streams, out_events);
    if (status != 0)
    {
        gpu_poly_destroy(*out_poly);
        *out_poly = nullptr;
        return status;
    }
    return status;
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
    if (!mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid context in gpu_matrix_copy_entry");
    }
    const size_t idx = matrix_index(row, col, mat->cols);

    const int N = mat->ctx->N;
    if (N <= 0 || mat->level < 0)
    {
        mat->format = src->format;
        return 0;
    }

    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(mat->level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_copy_entry");
    }

    std::vector<StreamRef> streams;
    streams.reserve(static_cast<size_t>(mat->level + 1));
    for (int limb = 0; limb <= mat->level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];

        const uint64_t *src_ptr = nullptr;
        int src_device = -1;
        cudaStream_t src_stream = nullptr;
        int status = get_poly_limb_u64_const(src, limb_id, &src_ptr, &src_stream, &src_device);
        if (status != 0)
        {
            return status;
        }

        uint64_t *dst_ptr = matrix_limb_ptr_by_id(mat, idx, limb_id);
        if (!dst_ptr)
        {
            return set_error("null matrix limb pointer in gpu_matrix_copy_entry");
        }

        int dst_device = -1;
        cudaStream_t dst_stream = nullptr;
        status = matrix_limb_device(mat, limb_id, &dst_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &dst_stream);
        if (status != 0)
        {
            return status;
        }
        if (src_device != dst_device)
        {
            return set_error("device mismatch in gpu_matrix_copy_entry");
        }

        cudaError_t err = cudaSetDevice(dst_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        status = wait_stream_on_stream(dst_device, dst_stream, src_stream);
        if (status != 0)
        {
            return status;
        }

        err = cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            static_cast<size_t>(N) * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice,
            dst_stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        append_unique_stream(streams, dst_device, dst_stream);
    }

    for (const auto &entry : streams)
    {
        cudaError_t err = cudaSetDevice(entry.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }

    mat->format = src->format;
    return 0;
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
