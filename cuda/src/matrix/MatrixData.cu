#include "matrix/MatrixData.cuh"

namespace
{
    template <typename T>
    __global__ void matrix_copy_kernel(
        const T **src,
        T **dst,
        size_t poly_count,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx - poly_idx * n;
        dst[poly_idx][coeff_idx] = src[poly_idx][coeff_idx];
    }

    template <typename T>
    struct CopyPointerScratch
    {
        int device;
        const T **d_src;
        T **d_dst;
        size_t src_capacity;
        size_t dst_capacity;
    };

    template <typename T>
    CopyPointerScratch<T> &copy_pointer_scratch_for_device(int device)
    {
        thread_local std::vector<CopyPointerScratch<T>> scratches;
        for (auto &scratch : scratches)
        {
            if (scratch.device == device)
            {
                return scratch;
            }
        }
        scratches.push_back(CopyPointerScratch<T>{
            device,
            nullptr,
            nullptr,
            0,
            0});
        return scratches.back();
    }

    template <typename PtrType>
    int ensure_copy_pointer_capacity(
        int device,
        PtrType **ptr,
        size_t *capacity,
        size_t required_count)
    {
        if (!ptr || !capacity)
        {
            return set_error("invalid pointer capacity arguments in ensure_copy_pointer_capacity");
        }
        if (required_count <= *capacity)
        {
            return 0;
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        if (*ptr)
        {
            err = cudaFree(*ptr);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            *ptr = nullptr;
            *capacity = 0;
        }
        if (required_count == 0)
        {
            return 0;
        }

        err = cudaMalloc(ptr, required_count * sizeof(PtrType));
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        *capacity = required_count;
        return 0;
    }

    template <typename T>
    int ensure_copy_pointer_scratch(
        int device,
        size_t count,
        const T ***src_table,
        T ***dst_table)
    {
        if (!src_table || !dst_table)
        {
            return set_error("invalid output tables in ensure_copy_pointer_scratch");
        }
        auto &scratch = copy_pointer_scratch_for_device<T>(device);
        int status = ensure_copy_pointer_capacity<const T *>(
            device,
            &scratch.d_src,
            &scratch.src_capacity,
            count);
        if (status != 0)
        {
            return status;
        }
        status = ensure_copy_pointer_capacity<T *>(
            device,
            &scratch.d_dst,
            &scratch.dst_capacity,
            count);
        if (status != 0)
        {
            return status;
        }

        *src_table = scratch.d_src;
        *dst_table = scratch.d_dst;
        return 0;
    }

    template <typename T>
    int launch_copy_kernel(
        const std::vector<const T *> &src_ptrs,
        const std::vector<T *> &dst_ptrs,
        size_t n,
        cudaStream_t stream)
    {
        const size_t count = src_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (dst_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in launch_copy_kernel");
        }

        int device = -1;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const T **d_src = nullptr;
        T **d_dst = nullptr;
        int status = ensure_copy_pointer_scratch<T>(
            device,
            count,
            &d_src,
            &d_dst);
        if (status != 0)
        {
            return status;
        }

        const size_t ptr_bytes = count * sizeof(T *);
        err = cudaMemcpyAsync(d_src, src_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        matrix_copy_kernel<<<blocks, threads, 0, stream>>>(d_src, d_dst, count, n);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

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

    void destroy_gpu_poly(GpuPoly *poly)
    {
        if (!poly)
        {
            return;
        }
        delete poly->poly;
        delete poly;
    }

    int clone_gpu_poly(const GpuPoly *src, GpuPoly **out_poly)
    {
        try
        {
            if (!src || !out_poly)
            {
                return set_error("invalid clone_gpu_poly arguments");
            }
            auto *poly = new CKKS::RNSPoly(src->poly->clone());
            *out_poly = new GpuPoly{poly, src->ctx, src->level, src->format};
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e.what());
        }
        catch (...)
        {
            return set_error("unknown exception in clone_gpu_poly");
        }
    }

    int copy_gpu_poly(GpuPoly *dst, const GpuPoly *src)
    {
        try
        {
            if (!dst || !src)
            {
                return set_error("invalid copy_gpu_poly arguments");
            }
            dst->poly->copy(*src->poly);
            dst->level = src->level;
            dst->format = src->format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e.what());
        }
        catch (...)
        {
            return set_error("unknown exception in copy_gpu_poly");
        }
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

    void free_matrix_u64_limb_table_device_ptrs(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        for (auto &table : mat->limb_u64_tables)
        {
            if (!table.device_entry_ptrs)
            {
                continue;
            }
            cudaSetDevice(table.device);
            cudaFree(table.device_entry_ptrs);
            table.device_entry_ptrs = nullptr;
        }
    }

    int rebuild_matrix_u64_limb_tables(GpuMatrix *mat)
    {
        if (!mat || !mat->ctx)
        {
            return set_error("invalid matrix in rebuild_matrix_u64_limb_tables");
        }
        free_matrix_u64_limb_table_device_ptrs(mat);
        mat->limb_u64_tables.clear();

        const int level = mat->level;
        if (level < 0)
        {
            return 0;
        }

        auto &limb_map = mat->ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error("unexpected limb mapping size in rebuild_matrix_u64_limb_tables");
        }

        const size_t count = mat->rows * mat->cols;
        mat->limb_u64_tables.reserve(static_cast<size_t>(level + 1));
        for (int limb = 0; limb <= level; ++limb)
        {
            GpuMatrix::LimbU64Table table{
                limb,
                false,
                -1,
                nullptr,
                {},
                nullptr};
            table.entry_ptrs.reserve(count);

            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            bool unsupported_limb_type = false;
            for (size_t idx = 0; idx < count; ++idx)
            {
                GpuPoly *poly = mat->polys[idx];
                if (!poly || !poly->poly)
                {
                    return set_error("null matrix polynomial in rebuild_matrix_u64_limb_tables");
                }
                if (limb_id.x >= poly->poly->GPU.size())
                {
                    return set_error("unexpected limb GPU partition in rebuild_matrix_u64_limb_tables");
                }
                auto &partition = poly->poly->GPU[limb_id.x];
                if (limb_id.y >= partition.limb.size())
                {
                    return set_error("unexpected limb index in rebuild_matrix_u64_limb_tables");
                }

                auto &limb_impl = partition.limb[limb_id.y];
                if (limb_impl.index() != FIDESlib::U64)
                {
                    unsupported_limb_type = true;
                    break;
                }
                auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
                if (table.device < 0)
                {
                    table.device = partition.device;
                    table.stream = limb_u64.stream.ptr;
                }
                else if (table.device != partition.device)
                {
                    return set_error("inconsistent limb device in rebuild_matrix_u64_limb_tables");
                }
                else if (table.stream != limb_u64.stream.ptr)
                {
                    return set_error("inconsistent limb stream in rebuild_matrix_u64_limb_tables");
                }
                table.entry_ptrs.push_back(limb_u64.v.data);
            }

            if (unsupported_limb_type)
            {
                table.available = false;
                table.device = -1;
                table.stream = nullptr;
                table.entry_ptrs.clear();
                table.device_entry_ptrs = nullptr;
            }
            else
            {
                table.available = true;
                if (!table.entry_ptrs.empty())
                {
                    cudaError_t err = cudaSetDevice(table.device);
                    if (err != cudaSuccess)
                    {
                        return set_error(err);
                    }
                    const size_t ptr_bytes = table.entry_ptrs.size() * sizeof(uint64_t *);
                    err = cudaMalloc(&table.device_entry_ptrs, ptr_bytes);
                    if (err != cudaSuccess)
                    {
                        return set_error(err);
                    }
                    err = cudaMemcpyAsync(
                        table.device_entry_ptrs,
                        table.entry_ptrs.data(),
                        ptr_bytes,
                        cudaMemcpyHostToDevice,
                        table.stream);
                    if (err != cudaSuccess)
                    {
                        cudaFree(table.device_entry_ptrs);
                        table.device_entry_ptrs = nullptr;
                        return set_error(err);
                    }
                    err = cudaStreamSynchronize(table.stream);
                    if (err != cudaSuccess)
                    {
                        cudaFree(table.device_entry_ptrs);
                        table.device_entry_ptrs = nullptr;
                        return set_error(err);
                    }
                }
            }
            mat->limb_u64_tables.push_back(std::move(table));
        }
        return 0;
    }

    int get_matrix_u64_limb_table(
        const GpuMatrix *mat,
        int limb,
        const GpuMatrix::LimbU64Table **out_table,
        const char *unsupported_msg)
    {
        if (!mat || !out_table)
        {
            return set_error("invalid get_matrix_u64_limb_table arguments");
        }
        *out_table = nullptr;
        if (limb < 0)
        {
            return set_error("invalid limb in get_matrix_u64_limb_table");
        }
        if (mat->limb_u64_tables.size() <= static_cast<size_t>(limb))
        {
            return set_error("missing matrix limb table");
        }
        const auto &table = mat->limb_u64_tables[static_cast<size_t>(limb)];
        if (!table.available)
        {
            return set_error(unsupported_msg ? unsupported_msg : "requested limb table is unavailable");
        }
        const size_t expected_count = mat->rows * mat->cols;
        if (table.entry_ptrs.size() != expected_count)
        {
            return set_error("matrix limb table size mismatch");
        }
        *out_table = &table;
        return 0;
    }

    int try_get_matrix_u64_limb_table(
        const GpuMatrix *mat,
        int limb,
        const GpuMatrix::LimbU64Table **out_table,
        bool *out_available)
    {
        if (!mat || !out_table || !out_available)
        {
            return set_error("invalid try_get_matrix_u64_limb_table arguments");
        }
        *out_table = nullptr;
        *out_available = false;
        if (limb < 0)
        {
            return set_error("invalid limb in try_get_matrix_u64_limb_table");
        }
        if (mat->limb_u64_tables.size() <= static_cast<size_t>(limb))
        {
            return set_error("missing matrix limb table");
        }

        const auto &table = mat->limb_u64_tables[static_cast<size_t>(limb)];
        if (!table.available)
        {
            return 0;
        }

        const size_t expected_count = mat->rows * mat->cols;
        if (table.entry_ptrs.size() != expected_count)
        {
            return set_error("matrix limb table size mismatch");
        }
        *out_table = &table;
        *out_available = true;
        return 0;
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
            destroy_gpu_poly(poly);
        }
        mat->polys.clear();
        free_matrix_u64_limb_table_device_ptrs(mat);
        mat->limb_u64_tables.clear();
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

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}, {}, {}, {}};
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
                destroy_gpu_poly(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected poly partition size in gpu_matrix_create");
            }

            auto &partition = poly_impl->GPU[partition_idx];
            if (partition.device != ctx->ctx->GPUid[partition_idx])
            {
                destroy_gpu_poly(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected partition device in gpu_matrix_create");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                destroy_gpu_poly(poly);
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
                destroy_gpu_poly(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("missing shared aux buffer in gpu_matrix_create");
            }
            const size_t aux_poly_slots = aux_slots_per_poly[partition_idx];
            const size_t aux_offset_slots = aux_slots_used[partition_idx];
            if (aux_offset_slots > aux_slots_total[partition_idx] ||
                aux_poly_slots > aux_slots_total[partition_idx] - aux_offset_slots)
            {
                destroy_gpu_poly(poly);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("shared aux buffer overflow in gpu_matrix_create");
            }
            if (partition.bufferAUXptrs)
            {
                err = cudaFree(partition.bufferAUXptrs);
                if (err != cudaSuccess)
                {
                    destroy_gpu_poly(poly);
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
                destroy_gpu_poly(poly);
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
                destroy_gpu_poly(poly);
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
                destroy_gpu_poly(poly);
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

    int table_status = rebuild_matrix_u64_limb_tables(mat);
    if (table_status != 0)
    {
        destroy_matrix_contents(mat);
        delete mat;
        return table_status;
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
    return gpu_matrix_copy_block(
        dst,
        src,
        0,
        0,
        0,
        0,
        src->rows,
        src->cols);
}

extern "C" int gpu_matrix_entry_clone(
    const GpuMatrix *mat,
    size_t row,
    size_t col,
    GpuPoly **out_poly)
{
    if (!mat || !out_poly)
    {
        return set_error("invalid gpu_matrix_entry_clone arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_entry_clone");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return clone_gpu_poly(mat->polys[idx], out_poly);
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
    return copy_gpu_poly(mat->polys[idx], src);
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

    auto wait_stream = [](cudaStream_t waiter, cudaStream_t signaler) -> int
    {
        if (!waiter || !signaler || waiter == signaler)
        {
            return 0;
        }
        cudaEvent_t event = nullptr;
        cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaEventRecord(event, signaler);
        if (err == cudaSuccess)
        {
            err = cudaStreamWaitEvent(waiter, event, 0);
        }
        cudaError_t destroy_err = cudaEventDestroy(event);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        if (destroy_err != cudaSuccess)
        {
            return set_error(destroy_err);
        }
        return 0;
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const GpuMatrix::LimbU64Table *src_table = nullptr;
        const GpuMatrix::LimbU64Table *dst_table = nullptr;
        bool src_has_u64_table = false;
        bool dst_has_u64_table = false;
        int status = try_get_matrix_u64_limb_table(src, limb, &src_table, &src_has_u64_table);
        if (status != 0)
        {
            return status;
        }
        status = try_get_matrix_u64_limb_table(out, limb, &dst_table, &dst_has_u64_table);
        if (status != 0)
        {
            return status;
        }

        if (!src_has_u64_table || !dst_has_u64_table)
        {
            return set_error("non-u64 limb is unsupported in gpu_matrix_copy_block");
        }

        if (src_table->device != dst_table->device)
        {
            return set_error("source/destination device mismatch in gpu_matrix_copy_block");
        }
        if (!src_table->stream || !dst_table->stream)
        {
            return set_error("null stream in gpu_matrix_copy_block");
        }

        std::vector<const uint64_t *> src_ptrs;
        std::vector<uint64_t *> dst_ptrs;
        src_ptrs.reserve(count);
        dst_ptrs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            src_ptrs.push_back(src_table->entry_ptrs[src_indices[i]]);
            dst_ptrs.push_back(dst_table->entry_ptrs[dst_indices[i]]);
        }

        cudaError_t err = cudaSetDevice(dst_table->device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = wait_stream(dst_table->stream, src_table->stream);
        if (status != 0)
        {
            return status;
        }
        status = launch_copy_kernel<uint64_t>(
            src_ptrs,
            dst_ptrs,
            static_cast<size_t>(N),
            dst_table->stream);
        if (status != 0)
        {
            return status;
        }
        status = wait_stream(src_table->stream, dst_table->stream);
        if (status != 0)
        {
            return status;
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        const size_t dst_idx = dst_indices[i];
        if (dst_idx < out->polys.size() && out->polys[dst_idx])
        {
            out->polys[dst_idx]->level = src->level;
            out->polys[dst_idx]->format = src->format;
        }
    }

    out->format = src->format;
    return 0;
}
