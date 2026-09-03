namespace
{
    struct DeviceDescriptorInit
    {
        uint8_t *base;
        size_t stride;
        size_t count;
        uint8_t widths[GPU_RUNTIME_MAX_LIMBS];
        size_t offsets[GPU_RUNTIME_MAX_LIMBS];
    };

    __global__ void initialize_device_descriptors(
        GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
        DeviceDescriptorInit init)
    {
        const size_t limb = blockIdx.x * blockDim.x + threadIdx.x;
        if (limb >= init.count)
        {
            return;
        }
        descriptors[limb] = GpuMatrix::SharedLimbBuffer::DeviceDescriptor{
            init.base + init.offsets[limb], init.stride, init.widths[limb], 0};
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

    bool checked_add_size(size_t a, size_t b, size_t *out)
    {
        if (!out || a > static_cast<size_t>(-1) - b)
        {
            return false;
        }
        *out = a + b;
        return true;
    }

    int calculate_matrix_allocation_bytes(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        GpuMatrixAllocationBytes *out)
    {
        if (!ctx || !out)
        {
            return set_error("invalid matrix allocation accounting arguments");
        }
        GpuPolyFormat parsed_format;
        if (!parse_format(format, parsed_format) || level < -1 || level > ctx->level ||
            ctx->N < 0 || ctx->gpu_ids.empty())
        {
            return set_error("invalid matrix allocation accounting parameters");
        }
        (void)parsed_format;
        size_t count = 0;
        if (!checked_mul_size(rows, cols, &count))
        {
            return set_error("matrix size overflow in allocation accounting");
        }
        const size_t active_limbs = level < 0 ? 0 : static_cast<size_t>(level + 1);
        if (active_limbs > GPU_RUNTIME_MAX_LIMBS)
        {
            return set_error("matrix limb count exceeds CUDA runtime limit");
        }
        if (ctx->limb_gpu_ids.size() < active_limbs ||
            ctx->limb_coeff_bytes.size() < active_limbs ||
            ctx->decomp_counts_by_partition.size() < ctx->gpu_ids.size())
        {
            return set_error("missing matrix allocation metadata");
        }
        *out = GpuMatrixAllocationBytes{};
        const size_t n = static_cast<size_t>(ctx->N);
        for (size_t partition = 0; partition < ctx->gpu_ids.size(); ++partition)
        {
            size_t local_limbs = 0;
            for (size_t limb = 0; limb < active_limbs; ++limb)
            {
                const dim3 id = ctx->limb_gpu_ids[limb];
                if (id.x >= ctx->gpu_ids.size())
                {
                    return set_error("invalid limb partition in allocation accounting");
                }
                if (id.x == partition)
                {
                    local_limbs = std::max(local_limbs, static_cast<size_t>(id.y) + 1);
                }
            }
            if (local_limbs == 0 || count == 0)
            {
                continue;
            }
            size_t coeff_bytes_per_poly = 0;
            for (size_t local = 0; local < local_limbs; ++local)
            {
                size_t global_limb = active_limbs;
                for (size_t limb = 0; limb < active_limbs; ++limb)
                {
                    const dim3 id = ctx->limb_gpu_ids[limb];
                    if (id.x == partition && id.y == local)
                    {
                        global_limb = limb;
                        break;
                    }
                }
                if (global_limb == active_limbs || ctx->limb_coeff_bytes[global_limb] == 0)
                {
                    return set_error("missing limb width in allocation accounting");
                }
                size_t limb_bytes = 0;
                if (!checked_mul_size(
                        n,
                        static_cast<size_t>(ctx->limb_coeff_bytes[global_limb]),
                        &limb_bytes) ||
                    !checked_add_size(coeff_bytes_per_poly, limb_bytes, &coeff_bytes_per_poly))
                {
                    return set_error("limb size overflow in allocation accounting");
                }
            }
            size_t bytes_per_poly = 0;
            size_t data_bytes = 0;
            if (!checked_mul_size(coeff_bytes_per_poly, 2, &bytes_per_poly) ||
                !checked_mul_size(bytes_per_poly, count, &data_bytes))
            {
                return set_error("data size overflow in allocation accounting");
            }
            size_t decomp_slots = 0;
            size_t aux_slots = 0;
            size_t aux_total_slots = 0;
            size_t aux_bytes = 0;
            if (!checked_mul_size(
                    static_cast<size_t>(4),
                    ctx->decomp_counts_by_partition[partition],
                    &decomp_slots) ||
                !checked_add_size(static_cast<size_t>(4), decomp_slots, &decomp_slots) ||
                !checked_mul_size(ctx->max_aux_limbs, decomp_slots, &aux_slots) ||
                !checked_mul_size(aux_slots, count, &aux_total_slots) ||
                !checked_mul_size(aux_total_slots, sizeof(void *), &aux_bytes))
            {
                return set_error("auxiliary size overflow in allocation accounting");
            }
            size_t descriptor_bytes = 0;
            if (!checked_mul_size(
                    local_limbs,
                    sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor),
                    &descriptor_bytes) ||
                !checked_add_size(aux_bytes, descriptor_bytes, &aux_bytes))
            {
                return set_error("descriptor size overflow in allocation accounting");
            }
            size_t event_count = 0;
            size_t event_bytes = 0;
            if (!checked_add_size(local_limbs, 1, &event_count) ||
                !checked_mul_size(event_count, sizeof(cudaEvent_t), &event_bytes) ||
                !checked_add_size(out->data_bytes, data_bytes, &out->data_bytes) ||
                !checked_add_size(out->aux_bytes, aux_bytes, &out->aux_bytes) ||
                !checked_add_size(out->event_bytes, event_bytes, &out->event_bytes))
            {
                return set_error("total size overflow in allocation accounting");
            }
        }
        if (!checked_add_size(out->data_bytes, out->aux_bytes, &out->total_bytes) ||
            !checked_add_size(out->total_bytes, out->event_bytes, &out->total_bytes))
        {
            return set_error("total size overflow in allocation accounting");
        }
        return 0;
    }

    bool queue_shared_allocation_free_locked(
        const std::shared_ptr<GpuMatrix::SharedAllocation> &allocation)
    {
        if (!allocation)
        {
            return true;
        }
        if (allocation->free_queued)
        {
            return true;
        }
        if (allocation->release_blocked)
        {
            return false;
        }
        if (!allocation->limb_base && !allocation->aux_base)
        {
            if (allocation->allocation_ready)
            {
                cudaSetDevice(allocation->device);
                cudaEventDestroy(allocation->allocation_ready);
                allocation->allocation_ready = nullptr;
            }
            allocation->free_queued = true;
            return true;
        }
        cudaError_t error = cudaSetDevice(allocation->device);
        if (error != cudaSuccess)
        {
            return false;
        }
        cudaStream_t stream = allocation->release_stream;
        if (!stream)
        {
            error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (error != cudaSuccess)
            {
                return false;
            }
            allocation->release_stream = stream;
            allocation->release_stream_owned = true;
        }
        if (allocation->allocation_ready &&
            cudaStreamWaitEvent(stream, allocation->allocation_ready, 0) != cudaSuccess)
        {
            return false;
        }
        for (const auto &[event_device, event] : allocation->pending_write_events)
        {
            if (!event || event_device != allocation->device)
            {
                return false;
            }
            if (cudaStreamWaitEvent(stream, event, 0) != cudaSuccess)
            {
                return false;
            }
        }
        if (allocation->limb_base)
        {
            error = cudaFreeAsync(allocation->limb_base, stream);
            if (error != cudaSuccess)
            {
                return false;
            }
            // cudaFreeAsync has successfully taken ownership of this free
            // operation.  Clear the pointer before attempting the auxiliary
            // free so a later retry cannot enqueue the same free twice.
            allocation->limb_base = nullptr;
        }
        if (allocation->aux_base)
        {
            error = cudaFreeAsync(allocation->aux_base, stream);
            if (error != cudaSuccess)
            {
                return false;
            }
            allocation->aux_base = nullptr;
        }
        for (const auto &[event_device, event] : allocation->pending_write_events)
        {
            if (event)
            {
                cudaSetDevice(event_device);
                cudaEventDestroy(event);
            }
        }
        allocation->pending_write_events.clear();
        if (allocation->allocation_ready)
        {
            cudaSetDevice(allocation->device);
            cudaEventDestroy(allocation->allocation_ready);
            allocation->allocation_ready = nullptr;
        }
        allocation->free_queued = true;
        return true;
    }

    void release_view_allocation(GpuMatrix *mat, size_t partition_idx)
    {
        if (!mat)
        {
            return;
        }
        std::shared_ptr<GpuMatrix::SharedAllocation> allocation;
        if (partition_idx < mat->shared_limb_buffers.size())
        {
            allocation = mat->shared_limb_buffers[partition_idx].allocation;
        }
        if (!allocation && partition_idx < mat->shared_aux_buffers.size())
        {
            allocation = mat->shared_aux_buffers[partition_idx].allocation;
        }
        if (allocation)
        {
            std::lock_guard<std::mutex> lock(allocation->mutex);
            // Preserve events from views that are destroyed before the last
            // view.  Their work may still be using the packed storage.
            auto &states = mat->exec_limb_states[partition_idx];
            for (auto &state : states)
            {
                if (state.write_done)
                {
                    allocation->pending_write_events.emplace_back(
                        state.device, state.write_done);
                    state.write_done = nullptr;
                    state.write_done_valid = false;
                }
            }
            if (allocation->live_views > 0)
            {
                --allocation->live_views;
            }
            if (allocation->live_views == 0)
            {
                // Do not free until every sibling has contributed its most
                // recent write event.  If CUDA rejects an ordering operation,
                // retain the allocation and retry from the shared owner
                // destructor rather than risking a use-after-free.
                queue_shared_allocation_free_locked(allocation);
            }
        }
        if (partition_idx < mat->shared_limb_buffers.size())
        {
            mat->shared_limb_buffers[partition_idx].ptr = nullptr;
            mat->shared_limb_buffers[partition_idx].device_descriptors = nullptr;
            mat->shared_limb_buffers[partition_idx].allocation.reset();
        }
        if (partition_idx < mat->shared_aux_buffers.size())
        {
            mat->shared_aux_buffers[partition_idx].ptr = nullptr;
            mat->shared_aux_buffers[partition_idx].allocation.reset();
        }
    }

    void free_matrix_shared_buffers(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        const size_t partition_count =
            std::max(mat->shared_limb_buffers.size(), mat->shared_aux_buffers.size());
        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
        {
            release_view_allocation(mat, partition_idx);
        }
        mat->shared_limb_buffers.clear();
        mat->shared_aux_buffers.clear();
    }

    void free_matrix_exec_states(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        for (size_t partition_idx = 0; partition_idx < mat->exec_limb_states.size(); ++partition_idx)
        {
            auto &states = mat->exec_limb_states[partition_idx];
            int device = -1;
            for (const auto &state : states)
            {
                if (state.device >= 0)
                {
                    device = state.device;
                    break;
                }
            }
            if (device >= 0)
            {
                cudaSetDevice(device);
            }
            for (auto &state : states)
            {
                if (state.write_done)
                {
                    cudaEventDestroy(state.write_done);
                    state.write_done = nullptr;
                }
                state.stream = nullptr;
                state.write_done_valid = false;
            }
        }
        mat->exec_limb_states.clear();
    }

    void destroy_matrix_contents(GpuMatrix *mat)
    {
        if (!mat)
        {
            return;
        }
        free_matrix_shared_buffers(mat);
        free_matrix_exec_states(mat);
    }

}

GpuMatrix::SharedAllocation::~SharedAllocation()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!free_queued)
    {
        // This also covers construction failures before a matrix descriptor
        // was published.  It is intentionally stream ordered; a failed
        // dependency setup leaves the allocation live instead of freeing it
        // while an asynchronous operation could still reference it.
        queue_shared_allocation_free_locked(
            std::shared_ptr<GpuMatrix::SharedAllocation>(this, [](auto *) {}));
    }
    if (free_queued && release_stream_owned && release_stream)
    {
        cudaSetDevice(device);
        cudaStreamDestroy(release_stream);
    }
}

extern "C" int gpu_matrix_query_allocation_bytes(
    const GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
    GpuMatrixAllocationBytes *out)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_query_allocation_bytes output");
    }
    return calculate_matrix_allocation_bytes(ctx, level, rows, cols, format, out);
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
    if (level < -1 || level > ctx->level)
    {
        return set_error("invalid level in gpu_matrix_create");
    }

    size_t count = 0;
    if (!checked_mul_size(rows, cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_create");
    }
    GpuMatrixAllocationBytes checked_allocation{};
    const int allocation_status =
        calculate_matrix_allocation_bytes(ctx, level, rows, cols, format, &checked_allocation);
    if (allocation_status != 0)
    {
        return allocation_status;
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}, {}, {}};
    const size_t partition_count = ctx->gpu_ids.size();
    if (partition_count == 0)
    {
        destroy_matrix_contents(mat);
        delete mat;
        return set_error("unexpected empty gpu_ids in gpu_matrix_create");
    }
    mat->shared_limb_buffers.resize(partition_count);
    mat->shared_aux_buffers.resize(partition_count);
    mat->exec_limb_states.resize(partition_count);

    const size_t n = static_cast<size_t>(ctx->N);

    for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
    {
        size_t limbs = 0;
        std::vector<uint8_t> local_limb_bytes;
        std::vector<size_t> local_limb_offsets;
        size_t coeff_bytes_per_poly = 0;
        if (level >= 0)
        {
            const size_t active_limbs = static_cast<size_t>(level + 1);
            if (ctx->limb_gpu_ids.size() < active_limbs)
            {
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected limb mapping size in gpu_matrix_create");
            }
            if (ctx->limb_coeff_bytes.size() < active_limbs)
            {
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("unexpected limb byte-width metadata in gpu_matrix_create");
            }
            for (size_t limb = 0; limb < active_limbs; ++limb)
            {
                const dim3 limb_id = ctx->limb_gpu_ids[limb];
                if (limb_id.x == partition_idx)
                {
                    limbs = std::max(limbs, static_cast<size_t>(limb_id.y) + 1);
                }
            }
            local_limb_bytes.assign(limbs, 0);
            local_limb_offsets.assign(limbs, 0);
            for (size_t limb = 0; limb < active_limbs; ++limb)
            {
                const dim3 limb_id = ctx->limb_gpu_ids[limb];
                if (limb_id.x != partition_idx)
                {
                    continue;
                }
                if (limb_id.y >= local_limb_bytes.size())
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error("invalid local limb index in gpu_matrix_create");
                }
                local_limb_bytes[limb_id.y] = ctx->limb_coeff_bytes[limb];
            }
            for (size_t limb_idx = 0; limb_idx < limbs; ++limb_idx)
            {
                const uint8_t coeff_bytes = local_limb_bytes[limb_idx];
                if (coeff_bytes == 0)
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error("missing local limb byte-width in gpu_matrix_create");
                }
                size_t limb_region_bytes = 0;
                if (!checked_mul_size(n, static_cast<size_t>(coeff_bytes), &limb_region_bytes))
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error("matrix limb region overflow in gpu_matrix_create");
                }
                local_limb_offsets[limb_idx] = coeff_bytes_per_poly;
                if (coeff_bytes_per_poly > static_cast<size_t>(-1) - limb_region_bytes)
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error("matrix limb offset overflow in gpu_matrix_create");
                }
                coeff_bytes_per_poly += limb_region_bytes;
            }
        }
        if (limbs == 0 || count == 0)
        {
            continue;
        }
        if (limbs > GPU_RUNTIME_MAX_LIMBS)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix descriptor count exceeds CUDA runtime limit");
        }

        mat->exec_limb_states[partition_idx].resize(limbs);

        size_t bytes_per_poly = 0;
        size_t total_bytes = 0;
        if (!checked_mul_size(coeff_bytes_per_poly, static_cast<size_t>(2), &bytes_per_poly) ||
            !checked_mul_size(bytes_per_poly, count, &total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix limb allocation overflow in gpu_matrix_create");
        }

        cudaError_t err = cudaSetDevice(ctx->gpu_ids[partition_idx]);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        auto &exec_states = mat->exec_limb_states[partition_idx];
        for (size_t limb_idx = 0; limb_idx < limbs; ++limb_idx)
        {
            auto &state = exec_states[limb_idx];
            state.device = ctx->gpu_ids[partition_idx];
            state.stream = nullptr;
            state.write_done = nullptr;
            state.write_done_valid = false;
            auto &stream_pool = ctx->compute_streams_by_partition[partition_idx];
            if (stream_pool.empty())
            {
                destroy_matrix_contents(mat);
                delete mat;
                return set_error("empty compute stream pool in gpu_matrix_create");
            }
            const size_t stream_slot =
                ctx->next_compute_stream.fetch_add(1, std::memory_order_relaxed) %
                stream_pool.size();
            state.stream = stream_pool[stream_slot];
            err = cudaEventCreateWithFlags(&state.write_done, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
            err = cudaEventRecord(state.write_done, state.stream);
            if (err != cudaSuccess)
            {
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
            state.write_done_valid = true;
        }

        cudaStream_t alloc_stream = exec_states[0].stream;
        if (!alloc_stream)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("missing allocation stream in gpu_matrix_create");
        }

        auto allocation = std::make_shared<GpuMatrix::SharedAllocation>();
        allocation->device = ctx->gpu_ids[partition_idx];
        allocation->release_stream =
            partition_idx < ctx->release_streams_by_partition.size()
                ? ctx->release_streams_by_partition[partition_idx]
                : nullptr;

        err = cudaEventCreateWithFlags(&allocation->allocation_ready, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        uint8_t *base = nullptr;
        err = cudaMallocAsync(reinterpret_cast<void **>(&base), total_bytes, alloc_stream);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }
        allocation->limb_base = base;
        allocation->limb_bytes = total_bytes;
        mat->shared_limb_buffers[partition_idx].allocation = allocation;
        mat->shared_limb_buffers[partition_idx].ptr = base;
        err = cudaEventRecord(allocation->allocation_ready, alloc_stream);
        if (err != cudaSuccess)
        {
            // The allocation command is already queued on alloc_stream, but
            // allocation_ready is not a usable dependency when recording it
            // failed.  Fail closed instead of asking a release stream to
            // wait on an unrecorded event; the shared allocation destructor
            // will intentionally retain the device storage.
            {
                std::lock_guard<std::mutex> lock(allocation->mutex);
                allocation->release_blocked = true;
            }
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        size_t aux_slots = 0;
        size_t aux_total_slots = 0;
        size_t aux_total_bytes = 0;
        if (partition_idx >= ctx->decomp_counts_by_partition.size())
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("unexpected decomp metadata size in gpu_matrix_create");
        }
        const size_t decomp_count = ctx->decomp_counts_by_partition[partition_idx];
        size_t decomp_slots = 0;
        if (!checked_mul_size(static_cast<size_t>(4), decomp_count, &decomp_slots) ||
            !checked_add_size(static_cast<size_t>(4), decomp_slots, &decomp_slots) ||
            !checked_mul_size(ctx->max_aux_limbs, decomp_slots, &aux_slots) ||
            !checked_mul_size(aux_slots, count, &aux_total_slots) ||
            !checked_mul_size(aux_total_slots, sizeof(void *), &aux_total_bytes))
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix aux allocation overflow in gpu_matrix_create");
        }
        size_t descriptor_bytes = 0;
        if (!checked_mul_size(
                limbs,
                sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor),
                &descriptor_bytes) ||
            aux_total_bytes > static_cast<size_t>(-1) - descriptor_bytes)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("matrix device descriptor allocation overflow in gpu_matrix_create");
        }
        aux_total_bytes += descriptor_bytes;

        void **aux_base = nullptr;
        err = cudaMallocAsync(&aux_base, aux_total_bytes, alloc_stream);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }
        allocation->aux_base = aux_base;
        allocation->aux_bytes = aux_total_bytes;
        mat->shared_aux_buffers[partition_idx].allocation = allocation;
        mat->shared_aux_buffers[partition_idx].ptr = aux_base;

        auto *device_descriptors = reinterpret_cast<GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(
            reinterpret_cast<uint8_t *>(aux_base) + aux_total_slots * sizeof(void *));
        DeviceDescriptorInit descriptor_init{};
        descriptor_init.base = base;
        descriptor_init.stride = bytes_per_poly;
        descriptor_init.count = limbs;
        for (size_t limb_idx = 0; limb_idx < limbs; ++limb_idx)
        {
            descriptor_init.offsets[limb_idx] = local_limb_offsets[limb_idx];
            descriptor_init.widths[limb_idx] = local_limb_bytes[limb_idx];
        }
        initialize_device_descriptors<<<1, static_cast<unsigned int>(limbs), 0, alloc_stream>>>(
            device_descriptors, descriptor_init);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            {
                std::lock_guard<std::mutex> lock(allocation->mutex);
                allocation->release_blocked = true;
            }
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        err = cudaEventRecord(allocation->allocation_ready, alloc_stream);
        if (err != cudaSuccess)
        {
            // Both asynchronous allocations have been submitted.  Without a
            // successfully recorded readiness event, no other stream can be
            // proven to observe their completion, so leak rather than free
            // through an unrecorded event during failure cleanup.
            {
                std::lock_guard<std::mutex> lock(allocation->mutex);
                allocation->release_blocked = true;
            }
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        cudaEvent_t alloc_ready = nullptr;
        err = cudaEventCreateWithFlags(&alloc_ready, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }
        err = cudaEventRecord(alloc_ready, alloc_stream);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(alloc_ready);
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        for (size_t limb_idx = 0; limb_idx < limbs; ++limb_idx)
        {
            auto &state = exec_states[limb_idx];
            if (!state.stream || state.stream == alloc_stream)
            {
                continue;
            }
            err = cudaStreamWaitEvent(state.stream, alloc_ready, 0);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(alloc_ready);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
        }
        // The initial write events above predate the asynchronous allocations
        // and therefore cannot be used as readiness events.  Re-record every
        // event after the allocation dependency, including the allocation
        // stream itself, so all future consumers and destruction paths wait
        // for both limb and auxiliary storage to exist.
        for (size_t limb_idx = 0; limb_idx < limbs; ++limb_idx)
        {
            auto &state = exec_states[limb_idx];
            err = cudaEventRecord(state.write_done, state.stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(alloc_ready);
                destroy_matrix_contents(mat);
                delete mat;
                return set_error(err);
            }
            state.write_done_valid = true;
        }
        cudaEventDestroy(alloc_ready);

        mat->shared_limb_buffers[partition_idx] = GpuMatrix::SharedLimbBuffer{
            ctx->gpu_ids[partition_idx],
            base,
            device_descriptors,
            0,
            limbs,
            bytes_per_poly,
            total_bytes,
            n,
            std::move(local_limb_bytes),
            std::move(local_limb_offsets),
            allocation};
        mat->shared_aux_buffers[partition_idx] = GpuMatrix::SharedAuxBuffer{
            ctx->gpu_ids[partition_idx],
            aux_base,
            aux_slots,
            aux_total_slots,
            allocation};
    }

    *out = mat;
    return 0;
}

extern "C" int gpu_matrix_create_batch(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
    size_t output_count,
    GpuMatrix **outputs)
{
    if (!ctx || !outputs || output_count == 0)
    {
        return output_count == 0 ? 0 : set_error("invalid gpu_matrix_create_batch arguments");
    }
    for (size_t i = 0; i < output_count; ++i)
    {
        outputs[i] = nullptr;
    }
    size_t packed_rows = 0;
    if (!checked_mul_size(rows, output_count, &packed_rows))
    {
        return set_error("packed matrix size overflow in gpu_matrix_create_batch");
    }

    // Build one ordinary matrix whose row-major storage is exactly the
    // concatenation of all requested outputs.  The handles below are views
    // into that allocation and therefore remain compatible with every
    // existing matrix operation.
    GpuMatrix *owner = nullptr;
    int status = gpu_matrix_create(ctx, level, packed_rows, cols, format, &owner);
    if (status != 0)
    {
        return status;
    }

    const size_t partition_count = ctx->gpu_ids.size();
    std::vector<GpuMatrix *> views;
    views.reserve(output_count);
    auto fail = [&](const char *message) {
        for (GpuMatrix *view : views)
        {
            destroy_matrix_contents(view);
            delete view;
        }
        // Detach the owner buffers only after the views have released their
        // references.  The final shared owner queues the actual free.
        destroy_matrix_contents(owner);
        delete owner;
        for (size_t i = 0; i < output_count; ++i) outputs[i] = nullptr;
        return set_error(message);
    };

    for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
    {
        auto *view = new GpuMatrix{ctx, rows, cols, level, static_cast<GpuPolyFormat>(format), {}, {}, {}};
        view->shared_limb_buffers.resize(partition_count);
        view->shared_aux_buffers.resize(partition_count);
        view->exec_limb_states.resize(partition_count);
        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
        {
            if (partition_idx >= owner->shared_limb_buffers.size() ||
                partition_idx >= owner->shared_aux_buffers.size())
            {
                destroy_matrix_contents(view);
                delete view;
                return fail("missing owner partition in gpu_matrix_create_batch");
            }
            const auto &owner_limb = owner->shared_limb_buffers[partition_idx];
            const auto &owner_aux = owner->shared_aux_buffers[partition_idx];
            if (!owner_limb.allocation || !owner_aux.allocation)
            {
                // Empty matrices have no device allocation and need no view
                // metadata.  Preserve the ordinary empty-matrix behavior.
                continue;
            }
            size_t view_bytes = 0;
            size_t packed_offset = 0;
            size_t view_poly_count = 0;
            if (!checked_mul_size(rows, cols, &view_poly_count) ||
                !checked_mul_size(owner_limb.bytes_per_poly, view_poly_count, &view_bytes) ||
                !checked_mul_size(view_bytes, output_idx, &packed_offset) ||
                packed_offset > owner_limb.bytes_total ||
                owner_limb.bytes_total - packed_offset < view_bytes)
            {
                destroy_matrix_contents(view);
                delete view;
                return fail("packed limb view overflow in gpu_matrix_create_batch");
            }
            size_t view_aux_slots = 0;
            size_t view_aux_bytes = 0;
            size_t aux_offset = 0;
            size_t owner_aux_bytes = 0;
            if (!checked_mul_size(owner_aux.slots_per_poly, view_poly_count, &view_aux_slots) ||
                !checked_mul_size(view_aux_slots, sizeof(void *), &view_aux_bytes) ||
                !checked_mul_size(view_aux_bytes, output_idx, &aux_offset) ||
                !checked_mul_size(owner_aux.slots_total, sizeof(void *), &owner_aux_bytes) ||
                aux_offset > owner_aux_bytes || owner_aux_bytes - aux_offset < view_aux_bytes)
            {
                destroy_matrix_contents(view);
                delete view;
                return fail("packed auxiliary view overflow in gpu_matrix_create_batch");
            }

            const size_t limb_offset = packed_offset;
            view->shared_limb_buffers[partition_idx] = GpuMatrix::SharedLimbBuffer{
                owner_limb.device,
                owner_limb.ptr ? owner_limb.ptr + limb_offset : nullptr,
                owner_limb.device_descriptors,
                limb_offset,
                owner_limb.limb_count,
                owner_limb.bytes_per_poly,
                view_bytes,
                owner_limb.n,
                owner_limb.limb_coeff_bytes,
                owner_limb.limb_offsets_bytes,
                owner_limb.allocation};
            view->shared_aux_buffers[partition_idx] = GpuMatrix::SharedAuxBuffer{
                owner_aux.device,
                owner_aux.ptr ? reinterpret_cast<void **>(
                                     reinterpret_cast<uint8_t *>(owner_aux.ptr) + aux_offset)
                               : nullptr,
                owner_aux.slots_per_poly,
                view_aux_slots,
                owner_aux.allocation};
            {
                std::lock_guard<std::mutex> lock(owner_limb.allocation->mutex);
                ++owner_limb.allocation->live_views;
            }

            const size_t limb_count = owner->exec_limb_states[partition_idx].size();
            view->exec_limb_states[partition_idx].resize(limb_count);
            for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
            {
                auto &state = view->exec_limb_states[partition_idx][limb_idx];
                const auto &owner_state = owner->exec_limb_states[partition_idx][limb_idx];
                state.device = owner_state.device;
                auto &pool = ctx->compute_streams_by_partition[partition_idx];
                if (pool.empty())
                {
                    destroy_matrix_contents(view);
                    delete view;
                    return fail("empty compute stream pool in gpu_matrix_create_batch");
                }
                state.stream = pool[ctx->next_compute_stream.fetch_add(1, std::memory_order_relaxed) %
                                    pool.size()];
                state.write_done = nullptr;
                state.write_done_valid = false;
                cudaSetDevice(state.device);
                cudaError_t error = cudaEventCreateWithFlags(&state.write_done, cudaEventDisableTiming);
                if (error != cudaSuccess)
                {
                    destroy_matrix_contents(view);
                    delete view;
                    return fail("failed to create batch view event");
                }
                if (owner_state.write_done && state.stream != owner_state.stream)
                {
                    error = cudaStreamWaitEvent(state.stream, owner_state.write_done, 0);
                    if (error != cudaSuccess)
                    {
                        destroy_matrix_contents(view);
                        delete view;
                        return fail("failed to link batch view allocation event");
                    }
                }
                error = cudaEventRecord(state.write_done, state.stream);
                if (error != cudaSuccess)
                {
                    destroy_matrix_contents(view);
                    delete view;
                    return fail("failed to record batch view event");
                }
                state.write_done_valid = true;
            }
        }
        views.push_back(view);
    }

    // The temporary owner has no ownership after its buffer descriptors are
    // detached; all allocations are now referenced by the returned views.
    for (auto &buffer : owner->shared_limb_buffers)
    {
        if (!buffer.allocation)
        {
            continue;
        }
        std::lock_guard<std::mutex> lock(buffer.allocation->mutex);
        if (buffer.allocation->live_views > 0)
        {
            --buffer.allocation->live_views;
        }
    }
    free_matrix_exec_states(owner);
    owner->shared_limb_buffers.clear();
    owner->shared_aux_buffers.clear();
    delete owner;
    for (size_t i = 0; i < output_count; ++i)
    {
        outputs[i] = views[i];
    }
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

extern "C" int gpu_matrix_wait(const GpuMatrix *mat)
{
    if (!mat || !mat->ctx)
    {
        return set_error("invalid gpu_matrix_wait arguments");
    }
    for (const auto &partition : mat->exec_limb_states)
    {
        for (const auto &state : partition)
        {
            if (!state.write_done || !state.write_done_valid)
            {
                continue;
            }
            cudaError_t err = cudaSetDevice(state.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaEventSynchronize(state.write_done);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
    }
    return 0;
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

    if (rows == 0 || cols == 0)
    {
        out->format = src->format;
        return 0;
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
    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_copy_block");
    }

    int status = launch_copy_for_all_limbs<uint64_t>(
        out,
        src,
        src_row,
        src_col,
        dst_row,
        dst_col,
        rows,
        cols,
        src->cols,
        out->cols,
        static_cast<size_t>(N),
        level);
    if (status != 0)
    {
        return status;
    }

    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_copy_peer(GpuMatrix *dst, const GpuMatrix *src, int *out_copied)
{
    if (!dst || !src || !out_copied || !dst->ctx || !src->ctx)
    {
        return set_error("invalid gpu_matrix_copy_peer arguments");
    }
    *out_copied = 0;
    if (dst->rows != src->rows || dst->cols != src->cols || dst->level != src->level ||
        dst->format != src->format || dst->ctx->N != src->ctx->N)
    {
        return set_error("incompatible matrices in gpu_matrix_copy_peer");
    }
    const size_t active_limbs = static_cast<size_t>(dst->level + 1);
    if (dst->ctx->moduli.size() < active_limbs || src->ctx->moduli.size() < active_limbs)
    {
        return set_error("missing active CRT moduli in gpu_matrix_copy_peer");
    }
    if (!std::equal(
            dst->ctx->moduli.begin(),
            dst->ctx->moduli.begin() + active_limbs,
            src->ctx->moduli.begin()) ||
        dst->shared_limb_buffers.size() != 1 || src->shared_limb_buffers.size() != 1)
    {
        return 0;
    }
    auto &destination_buffer = dst->shared_limb_buffers[0];
    const auto &source_buffer = src->shared_limb_buffers[0];
    if (!destination_buffer.ptr || !source_buffer.ptr ||
        destination_buffer.bytes_total != source_buffer.bytes_total ||
        destination_buffer.limb_count != source_buffer.limb_count ||
        destination_buffer.bytes_per_poly != source_buffer.bytes_per_poly ||
        destination_buffer.limb_coeff_bytes != source_buffer.limb_coeff_bytes ||
        destination_buffer.limb_offsets_bytes != source_buffer.limb_offsets_bytes)
    {
        return 0;
    }
    const int destination_device = destination_buffer.device;
    const int source_device = source_buffer.device;
    cudaError_t error = cudaSetDevice(destination_device);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    if (destination_device != source_device)
    {
        int can_access = 0;
        error = cudaDeviceCanAccessPeer(&can_access, destination_device, source_device);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (!can_access)
        {
            return 0;
        }
        error = cudaDeviceEnablePeerAccess(source_device, 0);
        if (error == cudaErrorPeerAccessAlreadyEnabled)
        {
            cudaGetLastError();
            error = cudaSuccess;
        }
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
    }
    if (dst->exec_limb_states.empty() || dst->exec_limb_states[0].empty() ||
        src->exec_limb_states.empty())
    {
        return set_error("missing matrix execution state in gpu_matrix_copy_peer");
    }
    cudaStream_t destination_stream = dst->exec_limb_states[0][0].stream;
    if (!destination_stream)
    {
        return set_error("missing destination stream in gpu_matrix_copy_peer");
    }
    for (const auto &states : src->exec_limb_states)
    {
        for (const auto &state : states)
        {
            if (state.write_done && state.write_done_valid)
            {
                error = cudaStreamWaitEvent(destination_stream, state.write_done, 0);
                if (error != cudaSuccess)
                {
                    return set_error(error);
                }
            }
        }
    }
    if (destination_device == source_device)
    {
        error = cudaMemcpyAsync(
            destination_buffer.ptr,
            source_buffer.ptr,
            source_buffer.bytes_total,
            cudaMemcpyDeviceToDevice,
            destination_stream);
    }
    else
    {
        error = cudaMemcpyPeerAsync(
            destination_buffer.ptr,
            destination_device,
            source_buffer.ptr,
            source_device,
            source_buffer.bytes_total,
            destination_stream);
    }
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    cudaEvent_t peer_copy_done = nullptr;
    error = cudaEventCreateWithFlags(&peer_copy_done, cudaEventDisableTiming);
    if (error == cudaSuccess)
    {
        error = cudaEventRecord(peer_copy_done, destination_stream);
    }
    if (error != cudaSuccess)
    {
        if (peer_copy_done) cudaEventDestroy(peer_copy_done);
        return set_error(error);
    }
    if (src->ctx->release_streams_by_partition.empty() ||
        !src->ctx->release_streams_by_partition[0])
    {
        cudaEventDestroy(peer_copy_done);
        return set_error("missing source release stream in gpu_matrix_copy_peer");
    }
    error = cudaSetDevice(source_device);
    if (error == cudaSuccess)
    {
        error = cudaStreamWaitEvent(
            src->ctx->release_streams_by_partition[0], peer_copy_done, 0);
    }
    if (error != cudaSuccess)
    {
        cudaSetDevice(destination_device);
        cudaEventDestroy(peer_copy_done);
        return set_error(error);
    }
    error = cudaSetDevice(destination_device);
    if (error != cudaSuccess)
    {
        cudaEventDestroy(peer_copy_done);
        return set_error(error);
    }
    for (size_t limb = 0; limb < active_limbs; ++limb)
    {
        const int status = matrix_record_limb_write(
            dst,
            dst->ctx->limb_gpu_ids[limb],
            destination_stream);
        if (status != 0)
        {
            cudaEventDestroy(peer_copy_done);
            return status;
        }
    }
    cudaEventDestroy(peer_copy_done);
    *out_copied = 1;
    return 0;
}
