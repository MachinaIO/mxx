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
            init.base + init.offsets[limb], init.stride, init.widths[limb]};
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

    struct MatrixPartitionAllocationPlan
    {
        size_t partition;
        size_t local_limb_count;
        size_t bytes_per_poly;
        size_t data_bytes;
        size_t aux_slots_per_poly;
        size_t aux_slots_total;
        size_t aux_bytes;
        std::vector<uint8_t> limb_coeff_bytes;
        std::vector<size_t> limb_offsets_bytes;
    };

    struct MatrixAllocationPlan
    {
        size_t count;
        GpuPolyFormat format;
        std::vector<MatrixPartitionAllocationPlan> partitions;
        GpuMatrixAllocationBytes totals;
    };

    int build_matrix_allocation_plan(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        int format,
        MatrixAllocationPlan *out)
    {
        if (!ctx || !out)
        {
            return set_error("invalid matrix allocation plan arguments");
        }
        GpuPolyFormat parsed_format;
        if (!parse_format(format, parsed_format))
        {
            return set_error("invalid format in matrix allocation plan");
        }
        if (level < -1 || level > ctx->level)
        {
            return set_error("invalid level in matrix allocation plan");
        }
        if (ctx->N < 0 || ctx->gpu_ids.empty())
        {
            return set_error("invalid context in matrix allocation plan");
        }

        MatrixAllocationPlan plan{};
        plan.format = parsed_format;
        if (!checked_mul_size(rows, cols, &plan.count))
        {
            return set_error("matrix size overflow in matrix allocation plan");
        }
        plan.partitions.reserve(ctx->gpu_ids.size());

        const size_t n = static_cast<size_t>(ctx->N);
        const size_t active_limbs = level < 0 ? 0 : static_cast<size_t>(level + 1);
        if (ctx->limb_gpu_ids.size() < active_limbs ||
            ctx->limb_coeff_bytes.size() < active_limbs)
        {
            return set_error("unexpected limb metadata in matrix allocation plan");
        }
        if (ctx->decomp_counts_by_partition.size() < ctx->gpu_ids.size())
        {
            return set_error("unexpected decomposition metadata in matrix allocation plan");
        }

        for (size_t partition_idx = 0; partition_idx < ctx->gpu_ids.size(); ++partition_idx)
        {
            MatrixPartitionAllocationPlan partition{};
            partition.partition = partition_idx;
            for (size_t limb = 0; limb < active_limbs; ++limb)
            {
                const dim3 limb_id = ctx->limb_gpu_ids[limb];
                if (limb_id.x == partition_idx)
                {
                    partition.local_limb_count = std::max(
                        partition.local_limb_count,
                        static_cast<size_t>(limb_id.y) + 1);
                }
                else if (limb_id.x >= ctx->gpu_ids.size())
                {
                    return set_error("invalid limb partition in matrix allocation plan");
                }
            }

            partition.limb_coeff_bytes.assign(partition.local_limb_count, 0);
            partition.limb_offsets_bytes.assign(partition.local_limb_count, 0);
            for (size_t limb = 0; limb < active_limbs; ++limb)
            {
                const dim3 limb_id = ctx->limb_gpu_ids[limb];
                if (limb_id.x != partition_idx)
                {
                    continue;
                }
                if (limb_id.y >= partition.local_limb_count)
                {
                    return set_error("invalid local limb index in matrix allocation plan");
                }
                partition.limb_coeff_bytes[limb_id.y] = ctx->limb_coeff_bytes[limb];
            }

            size_t coefficient_bytes_per_poly = 0;
            for (size_t local_limb = 0; local_limb < partition.local_limb_count; ++local_limb)
            {
                const uint8_t coefficient_bytes = partition.limb_coeff_bytes[local_limb];
                if (coefficient_bytes == 0)
                {
                    return set_error("missing local limb width in matrix allocation plan");
                }
                size_t limb_region_bytes = 0;
                if (!checked_mul_size(n, static_cast<size_t>(coefficient_bytes), &limb_region_bytes))
                {
                    return set_error("matrix limb region overflow in matrix allocation plan");
                }
                partition.limb_offsets_bytes[local_limb] = coefficient_bytes_per_poly;
                if (!checked_add_size(
                        coefficient_bytes_per_poly,
                        limb_region_bytes,
                        &coefficient_bytes_per_poly))
                {
                    return set_error("matrix limb offset overflow in matrix allocation plan");
                }
            }

            if (partition.local_limb_count != 0 && plan.count != 0)
            {
                if (!checked_mul_size(
                        coefficient_bytes_per_poly,
                        static_cast<size_t>(2),
                        &partition.bytes_per_poly) ||
                    !checked_mul_size(
                        partition.bytes_per_poly,
                        plan.count,
                        &partition.data_bytes))
                {
                    return set_error("matrix data allocation overflow in matrix allocation plan");
                }
                const size_t decomp_count =
                    ctx->decomp_counts_by_partition[partition_idx];
                size_t decomp_slots = 0;
                if (!checked_mul_size(static_cast<size_t>(4), decomp_count, &decomp_slots) ||
                    !checked_add_size(static_cast<size_t>(4), decomp_slots, &decomp_slots) ||
                    !checked_mul_size(
                        ctx->max_aux_limbs,
                        decomp_slots,
                        &partition.aux_slots_per_poly) ||
                    !checked_mul_size(
                        partition.aux_slots_per_poly,
                        plan.count,
                        &partition.aux_slots_total) ||
                    !checked_mul_size(
                        partition.aux_slots_total,
                        sizeof(void *),
                        &partition.aux_bytes))
                {
                    return set_error("matrix aux allocation overflow in matrix allocation plan");
                }
                size_t descriptor_bytes = 0;
                if (!checked_mul_size(
                        partition.local_limb_count,
                        sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor),
                        &descriptor_bytes) ||
                    !checked_add_size(partition.aux_bytes, descriptor_bytes, &partition.aux_bytes))
                {
                    return set_error("matrix device descriptor allocation overflow");
                }

                size_t partition_event_count = 0;
                size_t partition_event_bytes = 0;
                if (!checked_add_size(
                        partition.local_limb_count,
                        static_cast<size_t>(1),
                        &partition_event_count) ||
                    !checked_mul_size(
                        partition_event_count,
                        sizeof(cudaEvent_t),
                        &partition_event_bytes) ||
                    !checked_add_size(
                        plan.totals.event_bytes,
                        partition_event_bytes,
                        &plan.totals.event_bytes))
                {
                    return set_error("matrix event accounting overflow in matrix allocation plan");
                }
            }

            if (!checked_add_size(
                    plan.totals.data_bytes,
                    partition.data_bytes,
                    &plan.totals.data_bytes) ||
                !checked_add_size(
                    plan.totals.aux_bytes,
                    partition.aux_bytes,
                    &plan.totals.aux_bytes))
            {
                return set_error("matrix allocation total overflow");
            }
            plan.partitions.push_back(std::move(partition));
        }

        if (!checked_add_size(
                plan.totals.data_bytes,
                plan.totals.aux_bytes,
                &plan.totals.total_bytes) ||
            !checked_add_size(
                plan.totals.total_bytes,
                plan.totals.event_bytes,
                &plan.totals.total_bytes))
        {
            return set_error("matrix allocation total overflow");
        }
        *out = std::move(plan);
        return 0;
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
            uint8_t *limb_ptr = nullptr;
            int limb_device = -1;
            if (partition_idx < mat->shared_limb_buffers.size())
            {
                limb_ptr = mat->shared_limb_buffers[partition_idx].ptr;
                limb_device = mat->shared_limb_buffers[partition_idx].device;
            }
            void **aux_ptr = nullptr;
            int aux_device = -1;
            if (partition_idx < mat->shared_aux_buffers.size())
            {
                aux_ptr = mat->shared_aux_buffers[partition_idx].ptr;
                aux_device = mat->shared_aux_buffers[partition_idx].device;
            }
            int device = limb_device >= 0 ? limb_device : aux_device;
            if (device < 0 || (!limb_ptr && !aux_ptr))
            {
                continue;
            }
            cudaSetDevice(device);

            cudaStream_t free_stream =
                partition_idx < mat->ctx->release_streams_by_partition.size()
                    ? mat->ctx->release_streams_by_partition[partition_idx]
                    : nullptr;

            if (free_stream)
            {
                auto &states = mat->exec_limb_states[partition_idx];
                bool dependency_ok = true;
                bool async_free_queued = false;
                cudaError_t err = cudaSuccess;
                for (auto &state : states)
                {
                    if (!state.stream)
                    {
                        continue;
                    }
                    if (state.device != device || !state.write_done)
                    {
                        dependency_ok = false;
                        break;
                    }
                    if (state.write_done_valid)
                    {
                        err = cudaStreamWaitEvent(free_stream, state.write_done, 0);
                        if (err != cudaSuccess)
                        {
                            dependency_ok = false;
                            break;
                        }
                    }
                }
                if (dependency_ok)
                {
                    if (limb_ptr)
                    {
                        cudaFreeAsync(limb_ptr, free_stream);
                    }
                    if (aux_ptr)
                    {
                        cudaFreeAsync(aux_ptr, free_stream);
                    }
                    async_free_queued = true;
                }
                if (!dependency_ok && !async_free_queued)
                {
                    if (limb_ptr)
                    {
                        cudaFree(limb_ptr);
                        limb_ptr = nullptr;
                    }
                    if (aux_ptr)
                    {
                        cudaFree(aux_ptr);
                        aux_ptr = nullptr;
                    }
                }
            }
            else
            {
                if (limb_ptr)
                {
                    cudaFree(limb_ptr);
                }
                if (aux_ptr)
                {
                    cudaFree(aux_ptr);
                }
            }
            if (partition_idx < mat->shared_limb_buffers.size())
            {
                mat->shared_limb_buffers[partition_idx].ptr = nullptr;
                mat->shared_limb_buffers[partition_idx].device_descriptors = nullptr;
            }
            if (partition_idx < mat->shared_aux_buffers.size())
            {
                mat->shared_aux_buffers[partition_idx].ptr = nullptr;
            }
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
    *out = GpuMatrixAllocationBytes{};
    MatrixAllocationPlan plan{};
    const int status = build_matrix_allocation_plan(ctx, level, rows, cols, format, &plan);
    if (status != 0)
    {
        return status;
    }
    *out = plan.totals;
    return 0;
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
    MatrixAllocationPlan plan{};
    const int plan_status =
        build_matrix_allocation_plan(ctx, level, rows, cols, format, &plan);
    if (plan_status != 0)
    {
        return plan_status;
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, plan.format, {}, {}, {}};
    const size_t partition_count = plan.partitions.size();
    mat->shared_limb_buffers.resize(partition_count);
    mat->shared_aux_buffers.resize(partition_count);
    mat->exec_limb_states.resize(partition_count);

    const size_t n = static_cast<size_t>(ctx->N);
    for (auto &partition : plan.partitions)
    {
        const size_t partition_idx = partition.partition;
        if (partition.local_limb_count == 0 || plan.count == 0)
        {
            continue;
        }

        mat->exec_limb_states[partition_idx].resize(partition.local_limb_count);

        cudaError_t err = cudaSetDevice(ctx->gpu_ids[partition_idx]);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        auto &exec_states = mat->exec_limb_states[partition_idx];
        for (size_t limb_idx = 0; limb_idx < partition.local_limb_count; ++limb_idx)
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

        uint8_t *base = nullptr;
        err = cudaMallocAsync(
            reinterpret_cast<void **>(&base),
            partition.data_bytes,
            alloc_stream);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        void **aux_base = nullptr;
        err = cudaMallocAsync(&aux_base, partition.aux_bytes, alloc_stream);
        if (err != cudaSuccess)
        {
            const cudaError_t allocation_error = err;
            cudaFreeAsync(base, alloc_stream);
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(allocation_error);
        }

        auto *device_descriptors = reinterpret_cast<GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(
            reinterpret_cast<uint8_t *>(aux_base) +
            partition.aux_slots_total * sizeof(void *));
        DeviceDescriptorInit descriptor_init{};
        descriptor_init.base = base;
        descriptor_init.stride = partition.bytes_per_poly;
        descriptor_init.count = partition.local_limb_count;
        for (size_t limb_idx = 0; limb_idx < partition.local_limb_count; ++limb_idx)
        {
            descriptor_init.offsets[limb_idx] = partition.limb_offsets_bytes[limb_idx];
            descriptor_init.widths[limb_idx] = partition.limb_coeff_bytes[limb_idx];
        }
        mat->shared_limb_buffers[partition_idx] = GpuMatrix::SharedLimbBuffer{
            ctx->gpu_ids[partition_idx],
            base,
            device_descriptors,
            partition.local_limb_count,
            partition.bytes_per_poly,
            partition.data_bytes,
            n,
            std::move(partition.limb_coeff_bytes),
            std::move(partition.limb_offsets_bytes)};
        mat->shared_aux_buffers[partition_idx] = GpuMatrix::SharedAuxBuffer{
            ctx->gpu_ids[partition_idx],
            aux_base,
            partition.aux_slots_per_poly,
            partition.aux_slots_total};
        auto fail_after_descriptor_enqueue = [&](cudaError_t failure) -> int {
            // Only an error path may block here. Until the post-allocation
            // events are installed, the older limb events do not protect the
            // descriptor initialization from asynchronous owner cleanup.
            cudaStreamSynchronize(alloc_stream);
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(failure);
        };
        initialize_device_descriptors<<<1, static_cast<unsigned int>(partition.local_limb_count), 0, alloc_stream>>>(
            device_descriptors, descriptor_init);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return fail_after_descriptor_enqueue(err);
        }

        cudaEvent_t alloc_ready = nullptr;
        err = cudaEventCreateWithFlags(&alloc_ready, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            return fail_after_descriptor_enqueue(err);
        }
        err = cudaEventRecord(alloc_ready, alloc_stream);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(alloc_ready);
            return fail_after_descriptor_enqueue(err);
        }

        for (size_t limb_idx = 0; limb_idx < partition.local_limb_count; ++limb_idx)
        {
            auto &state = exec_states[limb_idx];
            if (!state.stream)
            {
                continue;
            }
            if (state.stream != alloc_stream)
            {
                err = cudaStreamWaitEvent(state.stream, alloc_ready, 0);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(alloc_ready);
                    return fail_after_descriptor_enqueue(err);
                }
            }
            err = cudaEventRecord(state.write_done, state.stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(alloc_ready);
                return fail_after_descriptor_enqueue(err);
            }
        }
        cudaEventDestroy(alloc_ready);
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
