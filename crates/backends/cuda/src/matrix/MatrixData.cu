#include "Control.cuh"

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
        size_t aux_offset;
        size_t allocation_bytes;
        std::vector<uint8_t> limb_coeff_bytes;
        std::vector<size_t> limb_offsets_bytes;
    };

    struct MatrixAllocationPlan
    {
        size_t count;
        std::vector<MatrixPartitionAllocationPlan> partitions;
        GpuMatrixAllocationBytes totals;
    };

    int build_matrix_allocation_plan(
        const GpuContext *ctx,
        int level,
        size_t rows,
        size_t cols,
        MatrixAllocationPlan *out)
    {
        if (!ctx || !out)
        {
            return set_error("invalid matrix allocation plan arguments");
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

                // Aux pointer slots and device descriptors share the data allocation.
                // Keep data_bytes logical: peer copies must never copy aux pointers.
                constexpr size_t aux_alignment = std::max(
                    alignof(void *), alignof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor));
                const size_t padding =
                    (aux_alignment - partition.data_bytes % aux_alignment) % aux_alignment;
                if (!checked_add_size(partition.data_bytes, padding, &partition.aux_offset) ||
                    !checked_add_size(partition.aux_offset, partition.aux_bytes,
                                      &partition.allocation_bytes) ||
                    !checked_add_size(partition.aux_bytes, padding, &partition.aux_bytes))
                    return set_error("combined matrix allocation overflow");

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
        const size_t partition_count = mat->shared_limb_buffers.size();
        for (size_t partition_idx = 0; partition_idx < partition_count; ++partition_idx)
        {
            // The limb buffer owns the allocation; aux is an aligned interior view.
            uint8_t *limb_ptr = mat->shared_limb_buffers[partition_idx].ptr;
            const int device = mat->shared_limb_buffers[partition_idx].device;
            if (device < 0 || !limb_ptr)
            {
                continue;
            }
            cudaSetDevice(device);

            cudaStream_t free_stream =
                partition_idx < mat->ctx->execution->release_streams_by_partition.size()
                    ? mat->ctx->execution->release_streams_by_partition[partition_idx]
                    : nullptr;
            if (free_stream)
            {
                auto &states = mat->exec_limb_states[partition_idx];
                bool dependency_ok = true;
                bool async_free_queued = false;
                cudaError_t err = cudaSuccess;
                uint64_t seen_completions = 0;
                for (auto &state : states)
                {
                    if (!state.stream)
                    {
                        continue;
                    }
                    const auto &completion = states[state.completion_owner];
                    const uint64_t bit = uint64_t{1} << state.completion_owner;
                    if (seen_completions & bit) continue;
                    seen_completions |= bit;
                    if (completion.device != device || !completion.write_done)
                    {
                        dependency_ok = false;
                        break;
                    }
                    if (completion.write_done_valid)
                    {
                        err = cudaStreamWaitEvent(free_stream, completion.write_done, 0);
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
                        err = cudaFreeAsync(limb_ptr, free_stream);
                        if (err != cudaSuccess) set_error(err);
                    }
                    async_free_queued = err == cudaSuccess;
                    dependency_ok = async_free_queued;
                }
                if (!dependency_ok && !async_free_queued)
                {
                    if (limb_ptr)
                    {
                        cudaFree(limb_ptr);
                        limb_ptr = nullptr;
                    }
                }
            }
            else
            {
                if (limb_ptr)
                {
                    cudaFree(limb_ptr);
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
                state.completion_owner = 0;
                state.last_write_stream = nullptr;
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
    GpuMatrixAllocationBytes *out)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_query_allocation_bytes output");
    }
    *out = GpuMatrixAllocationBytes{};
    MatrixAllocationPlan plan{};
    const int status = build_matrix_allocation_plan(ctx, level, rows, cols, &plan);
    if (status != 0)
    {
        return status;
    }
    *out = plan.totals;
    return 0;
}

extern "C" int gpu_matrix_binding_component_count(
    const GpuMatrix *mat,
    size_t *out_count)
{
    if (!mat || !out_count || mat->shared_limb_buffers.size() != mat->shared_aux_buffers.size())
    {
        return set_error("invalid gpu_matrix_binding_component_count arguments");
    }
    *out_count = mat->shared_limb_buffers.size();
    return 0;
}

extern "C" int gpu_matrix_binding_component(
    const GpuMatrix *mat,
    size_t component_index,
    GpuMatrixBindingComponent *out)
{
    if (!mat || !out || mat->shared_limb_buffers.size() != mat->shared_aux_buffers.size() ||
        component_index >= mat->shared_limb_buffers.size())
    {
        return set_error("invalid gpu_matrix_binding_component arguments");
    }
    const auto &buffer = mat->shared_limb_buffers[component_index];
    const auto &aux = mat->shared_aux_buffers[component_index];
    *out = GpuMatrixBindingComponent{
        buffer.device,
        buffer.limb_count,
        buffer.bytes_per_poly,
        buffer.bytes_total,
        buffer.n,
        sizeof(GpuMatrix::SharedLimbBuffer::DeviceDescriptor),
        buffer.ptr,
        buffer.device_descriptors,
        aux.ptr,
        aux.slots_per_poly,
        aux.slots_total};
    return 0;
}

extern "C" int gpu_matrix_binding_limb_count(
    const GpuMatrix *mat, size_t *out_count)
{
    if (!mat || !mat->ctx || !out_count || mat->level < 0 ||
        static_cast<size_t>(mat->level) >= mat->ctx->limb_gpu_ids.size())
        return set_error("invalid gpu_matrix_binding_limb_count arguments");
    *out_count = static_cast<size_t>(mat->level) + 1;
    return 0;
}

extern "C" int gpu_matrix_binding_limb(
    const GpuMatrix *mat, size_t crt_limb_index,
    GpuMatrixBindingLimb *out)
{
    if (!mat || !mat->ctx || !out || mat->level < 0 ||
        crt_limb_index > static_cast<size_t>(mat->level) ||
        crt_limb_index >= mat->ctx->limb_gpu_ids.size())
        return set_error("invalid gpu_matrix_binding_limb arguments");
    const dim3 id = mat->ctx->limb_gpu_ids[crt_limb_index];
    if (id.x >= mat->shared_limb_buffers.size())
        return set_error("matrix binding limb partition is missing");
    const auto &buffer = mat->shared_limb_buffers[id.x];
    if (id.y >= buffer.limb_count ||
        id.y >= buffer.limb_offsets_bytes.size() ||
        id.y >= buffer.limb_coeff_bytes.size() ||
        !buffer.ptr || buffer.n == 0 || buffer.bytes_per_poly == 0 ||
        buffer.bytes_per_poly % 2 != 0)
        return set_error("matrix binding limb allocation is invalid");
    const size_t width = buffer.limb_coeff_bytes[id.y];
    const size_t offset = buffer.limb_offsets_bytes[id.y];
    const size_t coefficient_half = buffer.bytes_per_poly / 2;
    if ((width != 4 && width != 8) ||
        offset > coefficient_half ||
        width > (coefficient_half - offset) / buffer.n ||
        buffer.bytes_total < buffer.bytes_per_poly ||
        mat->cols > SIZE_MAX / buffer.bytes_per_poly ||
        crt_limb_index >= mat->ctx->moduli.size())
        return set_error("matrix binding limb width/offset exceeds owner allocation");
    *out = GpuMatrixBindingLimb{
        buffer.device,
        crt_limb_index,
        static_cast<size_t>(id.x),
        static_cast<size_t>(id.y),
        offset,
        width,
        buffer.bytes_per_poly,
        mat->cols * buffer.bytes_per_poly,
        coefficient_half,
        buffer.bytes_total,
        mat->ctx->moduli[crt_limb_index],
        buffer.ptr + offset};
    return 0;
}

extern "C" int gpu_matrix_create(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    GpuMatrix **out,
    bool initialize_descriptors)
{
    if (!ctx || !out)
    {
        return set_error("invalid gpu_matrix_create arguments");
    }
    *out = nullptr;
    MatrixAllocationPlan plan{};
    const int plan_status =
        build_matrix_allocation_plan(ctx, level, rows, cols, &plan);
    if (plan_status != 0)
    {
        return plan_status;
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, {}, {}, {}};
    mat->descriptors_initialized = initialize_descriptors || plan.count == 0;
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

        // Small matrices use all-limb kernels on one stream. Preserve per-limb
        // stream parallelism for larger matrices and their tiled GEMM kernels.
        // Independent matrices remain distributed across the context's pool.
        // Uniform-stream allocation owns one dominating completion per partition.
        // Individual limb events are created only when a per-limb record needs one.
        auto &stream_pool = ctx->execution->compute_streams_by_partition[partition_idx];
        if (stream_pool.empty())
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error("empty compute stream pool in gpu_matrix_create");
        }
        const bool shared_stream = rows <= 4 && cols <= 4;
        cudaStream_t partition_stream = nullptr;
        if (shared_stream)
        {
            const size_t stream_slot =
                ctx->execution->next_compute_stream.fetch_add(1, std::memory_order_relaxed) %
                stream_pool.size();
            partition_stream = stream_pool[stream_slot];
        }
        auto &exec_states = mat->exec_limb_states[partition_idx];
        for (size_t limb_idx = 0; limb_idx < partition.local_limb_count; ++limb_idx)
        {
            auto &state = exec_states[limb_idx];
            state.device = ctx->gpu_ids[partition_idx];
            state.stream = nullptr;
            state.write_done = nullptr;
            state.completion_owner = static_cast<uint32_t>(limb_idx);
            state.last_write_stream = nullptr;
            state.write_done_valid = false;
            if (shared_stream)
                state.stream = partition_stream;
            else
            {
                const size_t stream_slot =
                    ctx->execution->next_compute_stream.fetch_add(1, std::memory_order_relaxed) %
                    stream_pool.size();
                state.stream = stream_pool[stream_slot];
            }
            if (!shared_stream || limb_idx == 0)
            {
                err = cudaEventCreateWithFlags(&state.write_done, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error(err);
                }
            }
            if (!shared_stream)
            {
                err = cudaEventRecord(state.write_done, state.stream);
                if (err != cudaSuccess)
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error(err);
                }
                state.last_write_stream = state.stream;
                state.write_done_valid = true;
            }
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
            partition.allocation_bytes,
            alloc_stream);
        if (err != cudaSuccess)
        {
            destroy_matrix_contents(mat);
            delete mat;
            return set_error(err);
        }

        auto **aux_base = reinterpret_cast<void **>(base + partition.aux_offset);

        auto *device_descriptors = reinterpret_cast<GpuMatrix::SharedLimbBuffer::DeviceDescriptor *>(
            reinterpret_cast<uint8_t *>(aux_base) +
            partition.aux_slots_total * sizeof(void *));
        DeviceDescriptorInit descriptor_init{};
        descriptor_init.base = base;
        descriptor_init.stride = partition.bytes_per_poly;
        descriptor_init.count = partition.local_limb_count;
        for (size_t limb_idx = 0; initialize_descriptors && limb_idx < partition.local_limb_count; ++limb_idx)
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
        if (initialize_descriptors)
        {
            initialize_device_descriptors<<<1, static_cast<unsigned int>(partition.local_limb_count), 0, alloc_stream>>>(
                device_descriptors, descriptor_init);
            err = cudaGetLastError();
            if (err != cudaSuccess) return fail_after_descriptor_enqueue(err);
        }
        // Even deferred descriptors require the allocation completion below:
        // unwinding before the filling kernel must still free after allocation.

        if (shared_stream)
        {
            // Allocation and descriptor initialization are already ordered on
            // this partition's sole producer stream. One owned completion
            // protects every limb. Later per-limb records allocate their owned
            // event before detaching from this completion alias.
            auto &completion = exec_states[0];
            err = cudaEventRecord(completion.write_done, alloc_stream);
            if (err != cudaSuccess) return fail_after_descriptor_enqueue(err);
            completion.last_write_stream = alloc_stream;
            completion.write_done_valid = true;
            for (auto &state : exec_states) state.completion_owner = 0;
            continue;
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
            state.last_write_stream = state.stream;
            // The allocation and (when requested) descriptor initialization
            // are ordered before this completion event.
            state.write_done_valid = true;
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
        uint64_t seen_completions = 0;
        for (const auto &state : partition)
        {
            const auto &completion = partition[state.completion_owner];
            const uint64_t bit = uint64_t{1} << state.completion_owner;
            if (seen_completions & bit) continue;
            seen_completions |= bit;
            if (!completion.write_done || !completion.write_done_valid)
            {
                continue;
            }
            cudaError_t err = cudaSetDevice(completion.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaEventSynchronize(completion.write_done);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
    }
    mat->host_observed_writer_ready.store(true, std::memory_order_release);
    return 0;
}

extern "C" int gpu_matrix_wait_compiled_inputs(
    const GpuMatrix *mat,
    int consumer_device,
    void *consumer_stream_raw,
    bool read_only)
{
    if (!mat || !consumer_stream_raw)
    {
        return set_error("invalid gpu_matrix_wait_compiled_inputs arguments");
    }
    return matrix_wait_all_limb_streams(
        mat,
        consumer_device,
        reinterpret_cast<cudaStream_t>(consumer_stream_raw),
        false,
        read_only);
}

extern "C" int gpu_matrix_record_compiled_write(GpuMatrix *mat, void *stream_raw)
{
    if (!mat || !stream_raw)
    {
        return set_error("invalid gpu_matrix_record_compiled_write arguments");
    }
    return matrix_record_all_limb_writes(mat, reinterpret_cast<cudaStream_t>(stream_raw), false);
}

