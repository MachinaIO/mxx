constexpr uint32_t kGaussMaxDigits = 64;
constexpr double kTwoPi = 6.283185307179586476925286766559;

int set_error(const char *msg)
{
    return gpu_set_last_error(msg);
}

int set_error(cudaError_t err)
{
    return gpu_set_last_error(cudaGetErrorString(err));
}

int default_batch(const GpuContext *ctx)
{
    return static_cast<int>(ctx && ctx->batch != 0 ? ctx->batch : 1);
}

bool parse_format(int format, PolyFormat &out)
{
    switch (format)
    {
    case GPU_POLY_FORMAT_COEFF:
        out = PolyFormat::Coeff;
        return true;
    case GPU_POLY_FORMAT_EVAL:
        out = PolyFormat::Eval;
        return true;
    default:
        return false;
    }
}

int collect_limb_storage_ptrs(
    const std::vector<CKKS::LimbImpl> &limbs,
    bool collect_aux,
    std::vector<void *> &out_ptrs)
{
    out_ptrs.clear();
    out_ptrs.reserve(limbs.size());
    for (const auto &limb : limbs)
    {
        if (limb.index() == FIDESlib::U64)
        {
            const auto &u64 = std::get<FIDESlib::U64>(limb);
            out_ptrs.push_back(collect_aux ? static_cast<void *>(u64.aux.data)
                                           : static_cast<void *>(u64.v.data));
            continue;
        }
        if (limb.index() == FIDESlib::U32)
        {
            const auto &u32 = std::get<FIDESlib::U32>(limb);
            out_ptrs.push_back(collect_aux ? static_cast<void *>(u32.aux.data)
                                           : static_cast<void *>(u32.v.data));
            continue;
        }
        return set_error("unsupported limb type in collect_limb_storage_ptrs");
    }
    return 0;
}

int upload_ptr_table(
    const std::vector<void *> &host_ptrs,
    FIDESlib::VectorGPU<void *> &device_table,
    cudaStream_t stream,
    const char *context)
{
    if (host_ptrs.empty())
    {
        return 0;
    }
    if (!device_table.data || device_table.size < static_cast<int>(host_ptrs.size()))
    {
        return set_error(context);
    }
    cudaError_t err = cudaMemcpyAsync(
        device_table.data,
        host_ptrs.data(),
        host_ptrs.size() * sizeof(void *),
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

size_t matrix_poly_count(const GpuMatrix *mat)
{
    if (!mat)
    {
        return 0;
    }
    return mat->rows * mat->cols;
}

void detach_poly_aux_buffers(GpuPoly *poly)
{
    if (!poly || !poly->poly)
    {
        return;
    }
    for (auto &partition : poly->poly->GPU)
    {
        partition.bufferAUXptrs = nullptr;
    }
}

void release_poly_views(std::vector<GpuPoly *> &polys)
{
    struct StreamRef
    {
        int device;
        cudaStream_t stream;
    };
    std::vector<StreamRef> streams;
    auto add_stream = [&](int device, cudaStream_t stream) {
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
    };
    auto add_limb_streams = [&](int device, const std::vector<CKKS::LimbImpl> &limbs) {
        for (const auto &limb : limbs)
        {
            if (limb.index() == FIDESlib::U64)
            {
                add_stream(device, std::get<FIDESlib::U64>(limb).stream.ptr);
            }
            else if (limb.index() == FIDESlib::U32)
            {
                add_stream(device, std::get<FIDESlib::U32>(limb).stream.ptr);
            }
        }
    };

    for (auto *poly : polys)
    {
        if (!poly || !poly->poly)
        {
            continue;
        }
        for (const auto &partition : poly->poly->GPU)
        {
            add_stream(partition.device, partition.s.ptr);
            add_limb_streams(partition.device, partition.limb);
            add_limb_streams(partition.device, partition.SPECIALlimb);
            for (const auto &decomp_group : partition.DECOMPlimb)
            {
                add_limb_streams(partition.device, decomp_group);
            }
            for (const auto &digit_group : partition.DIGITlimb)
            {
                add_limb_streams(partition.device, digit_group);
            }
        }
    }

    for (const auto &entry : streams)
    {
        cudaError_t err = cudaSetDevice(entry.device);
        if (err != cudaSuccess)
        {
            continue;
        }
        cudaStreamSynchronize(entry.stream);
    }

    for (auto *poly : polys)
    {
        detach_poly_aux_buffers(poly);
        gpu_poly_destroy(poly);
    }
    polys.clear();
}

int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device)
{
    if (!mat || !mat->ctx || !out_device)
    {
        return set_error("invalid matrix_limb_device arguments");
    }
    if (limb_id.x >= mat->shared_limb_buffers.size())
    {
        return set_error("invalid partition index in matrix_limb_device");
    }
    const auto &buffer = mat->shared_limb_buffers[limb_id.x];
    if (!buffer.ptr || limb_id.y >= buffer.limb_count)
    {
        return set_error("invalid limb index in matrix_limb_device");
    }
    *out_device = buffer.device;
    return 0;
}

int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, cudaStream_t *out_stream)
{
    if (!mat || !mat->ctx || !out_stream)
    {
        return set_error("invalid matrix_limb_stream arguments");
    }
    if (limb_id.x >= mat->ctx->ctx->meta.size())
    {
        return set_error("invalid partition index in matrix_limb_stream");
    }
    const auto &meta = mat->ctx->ctx->meta[limb_id.x];
    if (limb_id.y >= meta.size())
    {
        return set_error("invalid limb index in matrix_limb_stream");
    }
    *out_stream = meta[limb_id.y].stream.ptr;
    return *out_stream ? 0 : set_error("null stream in matrix_limb_stream");
}

uint64_t *matrix_limb_ptr_by_id(GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
{
    if (!mat || limb_id.x >= mat->shared_limb_buffers.size())
    {
        return nullptr;
    }
    const size_t count = matrix_poly_count(mat);
    if (poly_idx >= count)
    {
        return nullptr;
    }
    auto &buffer = mat->shared_limb_buffers[limb_id.x];
    if (!buffer.ptr || limb_id.y >= buffer.limb_count)
    {
        return nullptr;
    }
    const size_t offset_words = poly_idx * buffer.words_per_poly + limb_id.y * buffer.n;
    if (offset_words >= buffer.words_total)
    {
        return nullptr;
    }
    return buffer.ptr + offset_words;
}

const uint64_t *matrix_limb_ptr_by_id(const GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
{
    return matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(mat), poly_idx, limb_id);
}

int materialize_matrix_poly_view(const GpuMatrix *mat, size_t poly_idx, GpuPoly **out_poly)
{
    if (!mat || !mat->ctx || !out_poly)
    {
        return set_error("invalid materialize_matrix_poly_view arguments");
    }
    *out_poly = nullptr;
    const size_t count = matrix_poly_count(mat);
    if (poly_idx >= count)
    {
        return set_error("poly index out of bounds in materialize_matrix_poly_view");
    }
    const size_t partition_count = mat->ctx->ctx->meta.size();
    if (mat->shared_limb_buffers.size() != partition_count ||
        mat->shared_aux_buffers.size() != partition_count)
    {
        return set_error("unexpected partition mapping in materialize_matrix_poly_view");
    }

    auto *poly_impl = new CKKS::RNSPoly(*mat->ctx->ctx, -1, false);
    poly_impl->setLevel(mat->level);
    auto *poly = new GpuPoly{poly_impl, mat->ctx, mat->level, mat->format};

    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
    {
        if (partition_idx >= poly_impl->GPU.size())
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return set_error("unexpected partition size in materialize_matrix_poly_view");
        }
        auto &partition = poly_impl->GPU[partition_idx];
        const auto &limb_buffer = mat->shared_limb_buffers[partition_idx];
        const auto &aux_buffer = mat->shared_aux_buffers[partition_idx];
        if (!limb_buffer.ptr || limb_buffer.limb_count == 0)
        {
            continue;
        }
            if (partition.device != limb_buffer.device)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return set_error("partition device mismatch in materialize_matrix_poly_view");
            }
            FIDESlib::Stream *alloc_stream = nullptr;
            const auto &meta = mat->ctx->ctx->meta[partition_idx];
            for (size_t limb_idx = 0; limb_idx < meta.size(); ++limb_idx)
            {
                if (meta[limb_idx].id > mat->level)
                {
                    break;
                }
                alloc_stream = const_cast<FIDESlib::Stream *>(&meta[limb_idx].stream);
                break;
            }
            if (!alloc_stream || !alloc_stream->ptr)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return set_error("missing allocation stream in materialize_matrix_poly_view");
            }
            partition.s.wait(*alloc_stream);
            if (limb_buffer.n == 0 || limb_buffer.words_per_poly == 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return set_error("invalid limb buffer metadata in materialize_matrix_poly_view");
        }
        const size_t limb_words = limb_buffer.limb_count * limb_buffer.n;
        const size_t offset_words = poly_idx * limb_buffer.words_per_poly;
        if (offset_words > limb_buffer.words_total ||
            limb_buffer.words_per_poly > limb_buffer.words_total - offset_words ||
            limb_words > limb_buffer.words_per_poly)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return set_error("limb buffer overflow in materialize_matrix_poly_view");
        }

        partition.generate(
            partition.meta,
            partition.limb,
            partition.limbptr,
            static_cast<int>(limb_buffer.limb_count) - 1,
            &partition.auxptr,
            limb_buffer.ptr,
            offset_words,
            limb_buffer.ptr,
            offset_words + limb_words);

        if (!aux_buffer.ptr || aux_buffer.slots_per_poly == 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return set_error("missing aux buffer in materialize_matrix_poly_view");
        }
        const size_t aux_offset = poly_idx * aux_buffer.slots_per_poly;
        if (aux_offset > aux_buffer.slots_total ||
            aux_buffer.slots_per_poly > aux_buffer.slots_total - aux_offset)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return set_error("aux buffer overflow in materialize_matrix_poly_view");
        }
        if (partition.bufferAUXptrs)
        {
            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return set_error(err);
            }
            err = cudaFreeAsync(partition.bufferAUXptrs, partition.s.ptr);
            if (err != cudaSuccess)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return set_error(err);
            }
        }
        void **poly_aux_base = aux_buffer.ptr + aux_offset;
        partition.bufferAUXptrs = poly_aux_base;
        const size_t aux_stride = static_cast<size_t>(FIDESlib::MAXP);
        const size_t decomp_ptr_count = partition.DECOMPlimbptr.size();
        if (partition.DECOMPauxptr.size() != decomp_ptr_count ||
            partition.DIGITlimbptr.size() != decomp_ptr_count ||
            partition.DIGITauxptr.size() != decomp_ptr_count)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return set_error("unexpected aux pointer layout in materialize_matrix_poly_view");
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

        std::vector<void *> ptrs;
        int status = collect_limb_storage_ptrs(partition.limb, false, ptrs);
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }
        status = upload_ptr_table(
            ptrs,
            partition.limbptr,
            partition.s.ptr,
            "invalid limb pointer table in materialize_matrix_poly_view");
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }

        status = collect_limb_storage_ptrs(partition.limb, true, ptrs);
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }
        status = upload_ptr_table(
            ptrs,
            partition.auxptr,
            partition.s.ptr,
            "invalid aux pointer table in materialize_matrix_poly_view");
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }

        status = collect_limb_storage_ptrs(partition.SPECIALlimb, false, ptrs);
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }
        status = upload_ptr_table(
            ptrs,
            partition.SPECIALlimbptr,
            partition.s.ptr,
            "invalid special limb pointer table in materialize_matrix_poly_view");
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }

        status = collect_limb_storage_ptrs(partition.SPECIALlimb, true, ptrs);
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }
        status = upload_ptr_table(
            ptrs,
            partition.SPECIALauxptr,
            partition.s.ptr,
            "invalid special aux pointer table in materialize_matrix_poly_view");
        if (status != 0)
        {
            detach_poly_aux_buffers(poly);
            gpu_poly_destroy(poly);
            return status;
        }

        for (size_t decomp_idx = 0; decomp_idx < decomp_ptr_count; ++decomp_idx)
        {
            status = collect_limb_storage_ptrs(partition.DECOMPlimb[decomp_idx], false, ptrs);
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }
            status = upload_ptr_table(
                ptrs,
                partition.DECOMPlimbptr[decomp_idx],
                partition.s.ptr,
                "invalid decomp limb pointer table in materialize_matrix_poly_view");
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }

            status = collect_limb_storage_ptrs(partition.DECOMPlimb[decomp_idx], true, ptrs);
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }
            status = upload_ptr_table(
                ptrs,
                partition.DECOMPauxptr[decomp_idx],
                partition.s.ptr,
                "invalid decomp aux pointer table in materialize_matrix_poly_view");
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }

            status = collect_limb_storage_ptrs(partition.DIGITlimb[decomp_idx], false, ptrs);
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }
            status = upload_ptr_table(
                ptrs,
                partition.DIGITlimbptr[decomp_idx],
                partition.s.ptr,
                "invalid digit limb pointer table in materialize_matrix_poly_view");
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }

            status = collect_limb_storage_ptrs(partition.DIGITlimb[decomp_idx], true, ptrs);
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }
            status = upload_ptr_table(
                ptrs,
                partition.DIGITauxptr[decomp_idx],
                partition.s.ptr,
                "invalid digit aux pointer table in materialize_matrix_poly_view");
            if (status != 0)
            {
                detach_poly_aux_buffers(poly);
                gpu_poly_destroy(poly);
                return status;
            }
        }
    }

    *out_poly = poly;
    return 0;
}

int materialize_matrix_poly_views(const GpuMatrix *mat, std::vector<GpuPoly *> &out_polys)
{
    out_polys.clear();
    if (!mat)
    {
        return set_error("invalid matrix in materialize_matrix_poly_views");
    }
    const size_t count = matrix_poly_count(mat);
    out_polys.reserve(count);
    for (size_t poly_idx = 0; poly_idx < count; ++poly_idx)
    {
        GpuPoly *poly = nullptr;
        int status = materialize_matrix_poly_view(mat, poly_idx, &poly);
        if (status != 0)
        {
            release_poly_views(out_polys);
            return status;
        }
        out_polys.push_back(poly);
    }
    return 0;
}

std::vector<const GpuPoly *> collect_poly_const_ptrs(const std::vector<GpuPoly *> &polys)
{
    std::vector<const GpuPoly *> out;
    out.reserve(polys.size());
    for (auto *poly : polys)
    {
        out.push_back(poly);
    }
    return out;
}

int sync_poly_limb_streams(const GpuPoly *poly, const char *context)
{
    if (!poly || !poly->ctx || !poly->poly)
    {
        return set_error(context);
    }
    const int level = poly->level;
    if (level < 0)
    {
        return set_error(context);
    }
    auto &limb_map = poly->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error(context);
    }
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        if (limb_id.x >= poly->poly->GPU.size())
        {
            return set_error(context);
        }
        const auto &partition = poly->poly->GPU[limb_id.x];
        if (limb_id.y >= partition.limb.size())
        {
            return set_error(context);
        }

        cudaError_t err = cudaSetDevice(partition.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const auto &limb_impl = partition.limb[limb_id.y];
        cudaStream_t stream = nullptr;
        if (limb_impl.index() == FIDESlib::U64)
        {
            stream = std::get<FIDESlib::U64>(limb_impl).stream.ptr;
        }
        else if (limb_impl.index() == FIDESlib::U32)
        {
            stream = std::get<FIDESlib::U32>(limb_impl).stream.ptr;
        }
        else
        {
            return set_error(context);
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }
    return 0;
}

int sync_poly_partition_streams(const GpuPoly *poly, const char *context)
{
    if (!poly || !poly->poly)
    {
        return set_error(context);
    }
    for (const auto &partition : poly->poly->GPU)
    {
        cudaError_t err = cudaSetDevice(partition.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaStreamSynchronize(partition.s.ptr);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }
    return 0;
}

uint32_t bit_width_u64(uint64_t v)
{
    if (v == 0)
    {
        return 0;
    }
    return static_cast<uint32_t>(64 - __builtin_clzll(v));
}

__host__ __device__ __forceinline__ size_t matrix_index(size_t row, size_t col, size_t cols)
{
    return row * cols + col;
}

__device__ __forceinline__ uint32_t mul_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
{
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(prod % mod);
}

__device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % mod);
}

__device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
{
    uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (sum >= mod)
    {
        sum -= mod;
    }
    return static_cast<uint32_t>(sum);
}

__device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
    if (sum >= mod)
    {
        sum -= mod;
    }
    return static_cast<uint64_t>(sum);
}
