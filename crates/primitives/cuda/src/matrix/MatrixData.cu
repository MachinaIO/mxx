#include "Control.cuh"

namespace
{
    constexpr int kGatherThreads = 256;
    constexpr uint32_t kGatherInvalidIndex = MXX_GPU_CONTROL_INVALID_INDEX;

    using MatrixDeviceDescriptor = GpuMatrix::SharedLimbBuffer::DeviceDescriptor;

    __device__ __forceinline__ void gather_invalid_index(uint32_t *status)
    {
        if (status != nullptr)
        {
            // Preserve an earlier device-side failure.  In particular, a
            // valid lane must never turn a previously latched error into OK.
            atomicCAS(status, 0U, kGatherInvalidIndex);
        }
    }

    __global__ void matrix_family_gather_kernel(
        const MatrixDeviceDescriptor *const *family_descriptors,
        MatrixDeviceDescriptor *destination_descriptors,
        const int64_t *indices,
        uint32_t *status,
        size_t family_count,
        size_t index_count,
        int64_t wave_base,
        int64_t lane_offset,
        size_t lane_count,
        size_t source_columns,
        size_t destination_columns,
        size_t limb_count,
        size_t rows,
        size_t n)
    {
        const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_per_lane = limb_count * rows * source_columns * n;
        const size_t total = lane_count * total_per_lane;
        if (flat >= total || total_per_lane == 0)
        {
            return;
        }

        size_t remainder = flat;
        const size_t lane = remainder / total_per_lane;
        remainder -= lane * total_per_lane;
        const size_t limb = remainder / (rows * source_columns * n);
        remainder -= limb * rows * source_columns * n;
        const size_t row = remainder / (source_columns * n);
        remainder -= row * source_columns * n;
        const size_t column = remainder / n;
        const size_t coefficient = remainder - column * n;

        bool valid = family_descriptors != nullptr && indices != nullptr;
        int64_t selected = 0;
        size_t index_position = 0;
        if (valid)
        {
            const __int128 position = static_cast<__int128>(wave_base) +
                static_cast<__int128>(lane_offset) + static_cast<__int128>(lane);
            if (position < 0 || position > static_cast<__int128>(SIZE_MAX))
            {
                valid = false;
            }
            else
            {
                index_position = static_cast<size_t>(position);
                if (index_position >= index_count)
                {
                    valid = false;
                }
                else
                {
                    selected = indices[index_position];
                    const __int128 selected_wide = static_cast<__int128>(selected);
                    if (selected_wide < 0 || selected_wide >= static_cast<__int128>(family_count))
                    {
                        valid = false;
                    }
                }
            }
        }

        const MatrixDeviceDescriptor destination = destination_descriptors[limb];
        const size_t destination_poly = row * destination_columns + lane * source_columns + column;
        if (!valid)
        {
            gather_invalid_index(status);
            matrix_store_limb_u64(
                destination.base,
                destination_poly,
                coefficient,
                destination.stride,
                destination.width,
                0);
            return;
        }

        const MatrixDeviceDescriptor *source_descriptors = family_descriptors[selected];
        const MatrixDeviceDescriptor *source =
            source_descriptors == nullptr ? nullptr : &source_descriptors[limb];
        if (source == nullptr || source->base == nullptr)
        {
            gather_invalid_index(status);
            matrix_store_limb_u64(
                destination.base,
                destination_poly,
                coefficient,
                destination.stride,
                destination.width,
                0);
            return;
        }
        const uint64_t value = matrix_load_limb_u64(
            source->base, row * source_columns + column, coefficient, source->stride, source->width);
        matrix_store_limb_u64(
            destination.base,
            destination_poly,
            coefficient,
            destination.stride,
            destination.width,
            value);
    }

    struct MatrixFamilyGatherLaneArguments
    {
        const MatrixDeviceDescriptor *const *family_descriptors;
        const int64_t *indices;
        uint32_t *status;
        MatrixDeviceDescriptor *destination_descriptors[64];
        size_t family_count;
        size_t index_count;
        size_t active_count;
        size_t destination_count;
        uint64_t active_lane_mask;
        int64_t semantic_offset;
        size_t rows;
        size_t columns;
        size_t limb_count;
        size_t n;
    };

    __global__ void matrix_family_gather_lanes_kernel(MatrixFamilyGatherLaneArguments args)
    {
        const size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total_per_lane = args.limb_count * args.rows * args.columns * args.n;
        const size_t total = args.destination_count * total_per_lane;
        if (flat >= total || total_per_lane == 0)
        {
            return;
        }

        size_t remainder = flat;
        const size_t lane = remainder / total_per_lane;
        remainder -= lane * total_per_lane;
        if ((args.active_lane_mask & (UINT64_C(1) << lane)) == 0)
        {
            return;
        }
        const uint64_t prior_lane_mask = lane == 0
            ? 0
            : args.active_lane_mask & ((UINT64_C(1) << lane) - UINT64_C(1));
        const size_t index_position = static_cast<size_t>(__popcll(prior_lane_mask));
        const size_t limb = remainder / (args.rows * args.columns * args.n);
        remainder -= limb * args.rows * args.columns * args.n;
        const size_t row = remainder / (args.columns * args.n);
        remainder -= row * args.columns * args.n;
        const size_t column = remainder / args.n;
        const size_t coefficient = remainder - column * args.n;

        MatrixDeviceDescriptor *destination_descriptors =
            args.destination_descriptors[lane];
        MatrixDeviceDescriptor destination{};
        bool destination_valid = destination_descriptors != nullptr;
        if (destination_valid)
        {
            destination = destination_descriptors[limb];
            destination_valid = destination.base != nullptr;
        }

        bool valid = destination_valid && args.family_descriptors != nullptr &&
                     args.indices != nullptr && index_position < args.active_count;
        int64_t selected = 0;
        if (valid)
        {
            if (index_position >= args.index_count)
            {
                valid = false;
            }
            else
            {
                const __int128 shifted = static_cast<__int128>(args.indices[index_position]) +
                    static_cast<__int128>(args.semantic_offset);
                if (shifted < 0 || shifted >= static_cast<__int128>(args.family_count))
                {
                    valid = false;
                }
                else
                {
                    selected = static_cast<int64_t>(shifted);
                }
            }
        }

        const size_t destination_poly = row * args.columns + column;
        if (!valid)
        {
            gather_invalid_index(args.status);
            if (destination_valid)
            {
                matrix_store_limb_u64(
                    destination.base,
                    destination_poly,
                    coefficient,
                    destination.stride,
                    destination.width,
                    0);
            }
            return;
        }

        const MatrixDeviceDescriptor *source_descriptors = args.family_descriptors[selected];
        const MatrixDeviceDescriptor *source =
            source_descriptors == nullptr ? nullptr : &source_descriptors[limb];
        if (source == nullptr || source->base == nullptr)
        {
            gather_invalid_index(args.status);
            matrix_store_limb_u64(
                destination.base,
                destination_poly,
                coefficient,
                destination.stride,
                destination.width,
                0);
            return;
        }
        const uint64_t value = matrix_load_limb_u64(
            source->base, row * args.columns + column, coefficient, source->stride, source->width);
        matrix_store_limb_u64(
            destination.base,
            destination_poly,
            coefficient,
            destination.stride,
            destination.width,
            value);
    }

    int gather_grid_blocks(size_t total, size_t width, unsigned int *out_blocks)
    {
        if (!out_blocks || width == 0)
        {
            return set_error("invalid gather launch width");
        }
        const size_t needed = total == 0 ? 1 : (total - 1) / kGatherThreads + 1;
        const size_t bounded = std::min<size_t>(std::max<size_t>(needed, 1), width);
        *out_blocks = static_cast<unsigned int>(std::min<size_t>(bounded, 65535));
        return 0;
    }

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
            free_stream = matrix_capture_stream_for_device(mat->ctx, device, free_stream);

            if (free_stream)
            {
                auto &states = mat->exec_limb_states[partition_idx];
                const bool capture_active = matrix_stream_is_capturing(free_stream);
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
                    if (completion.device != device ||
                        (capture_active ? !completion.capture_write_done : !completion.write_done))
                    {
                        dependency_ok = false;
                        break;
                    }
                    if (capture_active || completion.write_done_valid)
                    {
                        const cudaEvent_t completion_event = capture_active
                            ? completion.capture_write_done->event : completion.write_done;
                        err = cudaStreamWaitEvent(free_stream, completion_event, 0);
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
                    // A synchronous free is never a valid fallback from an
                    // active capture. The capture owner keeps the allocation
                    // alive until abort/finish cleanup can run outside the
                    // capture. Returning here would invalidate the capture;
                    // leave the pointer fail-closed instead.
                    if (capture_active)
                    {
                        continue;
                    }
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
                    if (matrix_stream_is_capturing(mat->exec_limb_states[partition_idx][0].stream))
                    {
                        continue;
                    }
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
                if (state.capture_write_done)
                {
                    mxx_gpu_capture_event_release(state.capture_write_done);
                    state.capture_write_done = nullptr;
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

extern "C" int gpu_matrix_create(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
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
        build_matrix_allocation_plan(ctx, level, rows, cols, format, &plan);
    if (plan_status != 0)
    {
        return plan_status;
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, plan.format, {}, {}, {}};
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
            state.capture_write_done = nullptr;
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
            state.stream = matrix_capture_stream_for_device(ctx, state.device, state.stream);
            if (!shared_stream || limb_idx == 0)
            {
                if (matrix_stream_is_capturing(state.stream))
                {
                    if (!matrix_capture_event_for_owner(
                            ctx, state.device, &state.capture_write_done))
                    {
                        destroy_matrix_contents(mat);
                        delete mat;
                        return 1;
                    }
                }
                else
                {
                    err = cudaEventCreateWithFlags(&state.write_done, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        destroy_matrix_contents(mat);
                        delete mat;
                        return set_error(err);
                    }
                }
            }
            if (!shared_stream)
            {
                if (matrix_stream_is_capturing(state.stream))
                {
                    err = cudaEventRecord(state.capture_write_done->event, state.stream);
                }
                else
                {
                    err = cudaEventRecord(state.write_done, state.stream);
                }
                if (err != cudaSuccess)
                {
                    destroy_matrix_contents(mat);
                    delete mat;
                    return set_error(err);
                }
                state.last_write_stream = state.stream;
                state.write_done_valid = !matrix_stream_is_capturing(state.stream);
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
            if (!matrix_stream_is_capturing(alloc_stream))
            {
                cudaStreamSynchronize(alloc_stream);
            }
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
            if (matrix_stream_is_capturing(alloc_stream))
            {
                if (!matrix_capture_event_for_owner(
                        ctx, completion.device, &completion.capture_write_done))
                {
                    return fail_after_descriptor_enqueue(cudaErrorUnknown);
                }
                err = cudaEventRecord(completion.capture_write_done->event, alloc_stream);
            }
            else
            {
                err = cudaEventRecord(completion.write_done, alloc_stream);
            }
            if (err != cudaSuccess) return fail_after_descriptor_enqueue(err);
            completion.last_write_stream = alloc_stream;
            completion.write_done_valid = !matrix_stream_is_capturing(alloc_stream);
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
            if (matrix_stream_is_capturing(state.stream))
            {
                if (!matrix_capture_event_for_owner(
                        ctx, state.device, &state.capture_write_done))
                {
                    cudaEventDestroy(alloc_ready);
                    return fail_after_descriptor_enqueue(cudaErrorUnknown);
                }
                err = cudaEventRecord(state.capture_write_done->event, state.stream);
            }
            else
            {
                err = cudaEventRecord(state.write_done, state.stream);
            }
            if (err != cudaSuccess)
            {
                cudaEventDestroy(alloc_ready);
                return fail_after_descriptor_enqueue(err);
            }
            state.last_write_stream = state.stream;
            // The allocation and (when requested) descriptor initialization
            // are ordered before this completion event.  Mark the event as
            // usable so external capture preparation can join the stream
            // ordered allocation instead of launching against an unfinished
            // cudaMallocAsync.  The shared-stream path sets this explicitly
            // above; keep the per-limb path equivalent.
            state.write_done_valid = !matrix_stream_is_capturing(state.stream);
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
    if (mxx_graph_retain_released_resource(mat->ctx, mat, [](void *resource) {
        auto *matrix = static_cast<GpuMatrix *>(resource);
        destroy_matrix_contents(matrix);
        delete matrix;
    })) return;
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

extern "C" int gpu_matrix_track_compiled_consumer(
    const GpuMatrix *mat,
    int consumer_device,
    void *consumer_stream_raw,
    void *completion_event_raw,
    bool read_only)
{
    if (!mat || !consumer_stream_raw || !completion_event_raw)
    {
        return set_error("invalid gpu_matrix_track_compiled_consumer arguments");
    }
    return matrix_track_all_limb_consumers(
        mat,
        consumer_device,
        reinterpret_cast<cudaStream_t>(consumer_stream_raw),
        reinterpret_cast<cudaEvent_t>(completion_event_raw),
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

extern "C" int gpu_matrix_prepare_external_for_capture(GpuMatrix *mat)
{
    if (!mat || !mat->ctx)
    {
        return set_error("invalid gpu_matrix_prepare_external_for_capture arguments");
    }
    for (const auto &partition : mat->exec_limb_states)
    {
        for (const auto &state : partition)
        {
            if (matrix_stream_is_capturing(state.stream))
            {
                return set_error("external capture preparation must precede CUDA capture");
            }
        }
    }
    // This helper is intentionally preparation-only: it must run before the
    // origin stream enters capture. Resolve every currently valid writer with
    // the ordinary readiness path, then invalidate only the old dependency.
    if (gpu_matrix_wait(mat) != 0)
    {
        return 1;
    }
    for (auto &partition : mat->exec_limb_states)
    {
        for (auto &state : partition)
        {
            state.write_done_valid = false;
        }
    }
    mat->host_observed_writer_ready.store(true, std::memory_order_release);
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

    // Matrix copies can be reached from a capture-time materialization (for
    // example, cloning a borrowed input while assembling a concat).  Keep
    // the copy ABI explicit in that path as well: local binding 0 is the
    // destination allocation and the source allocation starts immediately
    // after the destination's CRT limbs.  The active capture map translates
    // these operation-local identities to the replay owners; outside capture
    // the registration helper is a no-op and the ordinary stream/event path
    // is unchanged.
    const uint32_t limb_count = static_cast<uint32_t>(level + 1);
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
        level,
        nullptr,
        limb_count,
        0);
    if (status != 0) return status;

    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_copy_block_on_stream(
    GpuMatrix *out,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t src_row,
    size_t src_col,
    size_t rows,
    size_t cols,
    cudaStream_t stream)
{
    if (!out || !src || !stream)
        return set_error("invalid gpu_matrix_copy_block_on_stream arguments");
    if (src_row + rows > src->rows || src_col + cols > src->cols ||
        dst_row + rows > out->rows || dst_col + cols > out->cols ||
        src->ctx != out->ctx || src->level != out->level)
        return set_error("matrix bounds/context mismatch in gpu_matrix_copy_block_on_stream");
    if (rows == 0 || cols == 0)
    {
        out->format = src->format;
        return 0;
    }
    const int level = src->level;
    if (level < 0 || src->ctx->N <= 0)
        return set_error("invalid matrix in gpu_matrix_copy_block_on_stream");
    const int status = launch_copy_for_all_limbs<uint64_t>(
        out, src, src_row, src_col, dst_row, dst_col, rows, cols,
        src->cols, out->cols, static_cast<size_t>(src->ctx->N), level, stream);
    if (status != 0) return status;
    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_copy_block_on_capture_stream(
    GpuMatrix *out,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t src_row,
    size_t src_col,
    size_t rows,
    size_t cols,
    cudaStream_t stream,
    uint32_t source_binding_base,
    uint32_t destination_binding_base)
{
    if (!out || !src || !stream)
        return set_error("invalid matrix capture copy arguments");
    if (src_row + rows > src->rows || src_col + cols > src->cols ||
        dst_row + rows > out->rows || dst_col + cols > out->cols ||
        src->ctx != out->ctx || src->level != out->level)
        return set_error("matrix bounds/context mismatch in capture copy");
    if (rows == 0 || cols == 0)
    {
        out->format = src->format;
        return 0;
    }
    const int level = src->level;
    if (level < 0 || src->ctx->N <= 0)
        return set_error("invalid matrix in capture copy");
    const int status = launch_copy_for_all_limbs<uint64_t>(
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
        static_cast<size_t>(src->ctx->N),
        level,
        stream,
        source_binding_base,
        destination_binding_base);
    if (status != 0)
        return status;
    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_gather_family(
    GpuMatrix *destination,
    const void *family_descriptors_raw,
    size_t family_count,
    const int64_t *indices,
    size_t index_count,
    int64_t wave_base,
    int64_t lane_offset,
    size_t lane_count,
    size_t source_columns,
    size_t width,
    cudaStream_t stream,
    uint32_t *status,
    size_t destination_partition,
    uint32_t destination_binding_index,
    uint32_t indices_binding_index,
    uint32_t status_binding_index)
{
    if (!destination || !destination->ctx || !family_descriptors_raw || !stream)
    {
        return set_error("invalid gpu_matrix_gather_family arguments");
    }
    if (destination_partition >= destination->shared_limb_buffers.size())
    {
        return set_error("invalid destination partition in gpu_matrix_gather_family");
    }
    if (source_columns == 0 || lane_count == 0 || width == 0)
    {
        return set_error("invalid gather dimensions");
    }
    // Empty families/index ranges are valid launch inputs: every lane is an
    // invalid selection and must still be zero-filled while latching status.
    if (destination->rows == 0 || destination->cols == 0 || destination->level < 0)
    {
        return set_error("invalid gather destination");
    }
    if (lane_count > SIZE_MAX / source_columns ||
        destination->cols != lane_count * source_columns)
    {
        return set_error("gather destination columns do not match lane width");
    }

    auto &destination_buffer = destination->shared_limb_buffers[destination_partition];
    if (destination_partition >= destination->ctx->gpu_ids.size())
    {
        return set_error("destination partition has no physical device");
    }
    if (destination_buffer.device_descriptors == nullptr || destination_buffer.limb_count == 0 ||
        destination_buffer.n == 0 || destination_buffer.device != destination->ctx->gpu_ids[destination_partition])
    {
        return set_error("invalid gather destination descriptors");
    }
    const size_t limb_count = destination_buffer.limb_count;
    const size_t n = destination_buffer.n;
    size_t total_per_lane = 0;
    size_t total = 0;
    if (!checked_mul_size(limb_count, destination->rows, &total_per_lane) ||
        !checked_mul_size(total_per_lane, source_columns, &total_per_lane) ||
        !checked_mul_size(total_per_lane, n, &total_per_lane) ||
        !checked_mul_size(total_per_lane, lane_count, &total))
    {
        return set_error("gather launch size overflow");
    }

    const int device = destination_buffer.device;
    cudaStream_t dispatch_stream = matrix_capture_stream_for_device(destination->ctx, device, stream);
    if (!dispatch_stream)
    {
        return set_error("invalid gather dispatch stream");
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    // External graph callers resolve owner events before capture. Ordinary
    // calls still get the normal stream-ordered destination dependency.
    if (!matrix_stream_is_capturing(dispatch_stream))
    {
        const int wait = matrix_wait_all_limb_streams(
            destination, device, dispatch_stream, false, false);
        if (wait != 0)
        {
            return wait;
        }
    }

    unsigned int blocks = 0;
    if (gather_grid_blocks(total, width, &blocks) != 0)
    {
        return 1;
    }
    auto *family_descriptors = reinterpret_cast<const MatrixDeviceDescriptor *const *>(
        const_cast<void *>(family_descriptors_raw));
    matrix_family_gather_kernel<<<blocks, kGatherThreads, 0, dispatch_stream>>>(
        family_descriptors,
        destination_buffer.device_descriptors,
        indices,
        status,
        family_count,
        index_count,
        wave_base,
        lane_offset,
        lane_count,
        source_columns,
        destination->cols,
        limb_count,
        destination->rows,
        n);

    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(size_t), sizeof(size_t), sizeof(int64_t), sizeof(int64_t),
        sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t),
        sizeof(size_t), sizeof(size_t),
    };
    MxxGraphPatch patches[3]{};
    size_t patch_count = 0;
    if (destination_binding_index != UINT32_MAX)
    {
        patches[patch_count++] = graph_kernel_pointer_patch(1, 0, destination_binding_index);
    }
    if (indices_binding_index != UINT32_MAX)
    {
        patches[patch_count++] = graph_kernel_pointer_patch(2, 0, indices_binding_index);
    }
    if (status_binding_index != UINT32_MAX)
    {
        patches[patch_count++] = graph_kernel_pointer_patch(3, 0, status_binding_index);
    }
    if (patch_count != 0)
    {
        const int registration = register_kernel_pointer_fields(
            destination->ctx,
            dispatch_stream,
            argument_sizes,
            std::size(argument_sizes),
            patches,
            patch_count);
        if (registration != 0)
        {
            return registration;
        }
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    destination->host_observed_writer_ready.store(false, std::memory_order_release);
    return 0;
}

extern "C" int gpu_matrix_gather_family_lanes(
    const void *family_descriptors_raw,
    size_t family_count,
    const GpuMatrix *const *destinations,
    size_t destination_count,
    const int64_t *indices,
    size_t index_count,
    size_t active_count,
    uint64_t active_lane_mask,
    int64_t semantic_offset,
    size_t width,
    int physical_device,
    cudaStream_t stream,
    uint32_t *status,
    uint32_t indices_binding_index,
    uint32_t status_binding_index,
    const uint32_t *destination_binding_indices)
{
    if (!family_descriptors_raw || !destinations || !destination_binding_indices || !stream)
    {
        return set_error("invalid gpu_matrix_gather_family_lanes arguments");
    }
    const uint64_t lane_bits = destination_count >= 64
        ? UINT64_MAX
        : (destination_count == 0 ? 0 : ((UINT64_C(1) << destination_count) - 1));
    if (destination_count == 0 || destination_count > 64 || active_count > destination_count ||
        width == 0 || (active_lane_mask & ~lane_bits) != 0 ||
        static_cast<size_t>(__builtin_popcountll(active_lane_mask)) != active_count)
    {
        return set_error("invalid lane gather count or mask");
    }
    if (active_count == 0)
    {
        return 0;
    }
    const GpuMatrix *first = destinations[0];
    if (!first || !first->ctx || first->level < 0 || first->rows == 0 || first->cols == 0)
    {
        return set_error("invalid lane gather destination owner");
    }
    GpuContext *ctx = first->ctx;
    size_t destination_partition = SIZE_MAX;
    for (size_t partition = 0; partition < ctx->gpu_ids.size(); ++partition)
    {
        if (ctx->gpu_ids[partition] == physical_device)
        {
            destination_partition = partition;
            break;
        }
    }
    if (destination_partition == SIZE_MAX || destination_partition >= first->shared_limb_buffers.size())
    {
        return set_error("lane gather destination partition is unavailable");
    }
    const auto &first_buffer = first->shared_limb_buffers[destination_partition];
    if (!first_buffer.device_descriptors || first_buffer.limb_count == 0 || first_buffer.n == 0)
    {
        return set_error("lane gather destination descriptors are unavailable");
    }
    const size_t limb_count = first_buffer.limb_count;
    const size_t n = first_buffer.n;
    for (size_t lane = 0; lane < destination_count; ++lane)
    {
        const GpuMatrix *destination = destinations[lane];
        if (!destination || destination->ctx != ctx || destination->level != first->level ||
            destination->format != first->format || destination->rows != first->rows ||
            destination->cols != first->cols || destination_partition >= destination->shared_limb_buffers.size())
        {
            return set_error("lane gather destination owner shape or context mismatch");
        }
        const auto &buffer = destination->shared_limb_buffers[destination_partition];
        if (!buffer.device_descriptors || buffer.limb_count != limb_count || buffer.n != n)
        {
            return set_error("lane gather destination limb layout mismatch");
        }
    }

    size_t total_per_lane = 0;
    size_t total = 0;
    if (!checked_mul_size(limb_count, first->rows, &total_per_lane) ||
        !checked_mul_size(total_per_lane, first->cols, &total_per_lane) ||
        !checked_mul_size(total_per_lane, n, &total_per_lane) ||
        !checked_mul_size(total_per_lane, destination_count, &total))
    {
        return set_error("lane gather launch size overflow");
    }
    const int device = first_buffer.device;
    cudaStream_t dispatch_stream = matrix_capture_stream_for_device(ctx, device, stream);
    if (!dispatch_stream)
    {
        return set_error("invalid lane gather dispatch stream");
    }
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    for (size_t lane = 0; lane < destination_count; ++lane)
    {
        if ((active_lane_mask & (UINT64_C(1) << lane)) == 0)
        {
            continue;
        }
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 id = ctx->limb_gpu_ids[limb];
            const int wait = matrix_wait_limb_stream(destinations[lane], id, device, dispatch_stream);
            if (wait != 0)
            {
                return wait;
            }
        }
    }

    unsigned int blocks = 0;
    if (gather_grid_blocks(total, width, &blocks) != 0)
    {
        return 1;
    }
    MatrixFamilyGatherLaneArguments args{};
    args.family_descriptors = reinterpret_cast<const MatrixDeviceDescriptor *const *>(
        const_cast<void *>(family_descriptors_raw));
    args.indices = indices;
    args.status = status;
    args.family_count = family_count;
    args.index_count = index_count;
    args.active_count = active_count;
    args.destination_count = destination_count;
    args.active_lane_mask = active_lane_mask;
    args.semantic_offset = semantic_offset;
    args.rows = first->rows;
    args.columns = first->cols;
    args.limb_count = limb_count;
    args.n = n;
    for (size_t lane = 0; lane < destination_count; ++lane)
    {
        args.destination_descriptors[lane] =
            destinations[lane]->shared_limb_buffers[destination_partition].device_descriptors;
    }
    matrix_family_gather_lanes_kernel<<<blocks, kGatherThreads, 0, dispatch_stream>>>(args);

    const size_t argument_sizes[] = {sizeof(MatrixFamilyGatherLaneArguments)};
    std::vector<MxxGraphPatch> patches;
    patches.reserve(destination_count + 2);
    if (indices_binding_index != UINT32_MAX)
    {
        patches.push_back(graph_kernel_pointer_patch(
            0, offsetof(MatrixFamilyGatherLaneArguments, indices), indices_binding_index));
    }
    if (status_binding_index != UINT32_MAX)
    {
        patches.push_back(graph_kernel_pointer_patch(
            0, offsetof(MatrixFamilyGatherLaneArguments, status), status_binding_index));
    }
    for (size_t lane = 0; lane < destination_count; ++lane)
    {
        const uint32_t binding = destination_binding_indices[lane];
        if ((active_lane_mask & (UINT64_C(1) << lane)) != 0 && binding == UINT32_MAX)
        {
            return set_error("active lane gather destination has no binding identity");
        }
        if (binding != UINT32_MAX)
        {
            patches.push_back(graph_kernel_pointer_patch(
                0,
                offsetof(MatrixFamilyGatherLaneArguments, destination_descriptors) +
                    lane * sizeof(void *),
                binding));
        }
    }
    if (!patches.empty())
    {
        const int registration = register_kernel_pointer_fields(
            ctx,
            dispatch_stream,
            argument_sizes,
            std::size(argument_sizes),
            patches.data(),
            patches.size());
        if (registration != 0)
        {
            return registration;
        }
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    for (size_t lane = 0; lane < destination_count; ++lane)
    {
        if ((active_lane_mask & (UINT64_C(1) << lane)) == 0)
        {
            continue;
        }
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const int record = matrix_record_limb_write(
                const_cast<GpuMatrix *>(destinations[lane]), ctx->limb_gpu_ids[limb], dispatch_stream);
            if (record != 0)
            {
                return record;
            }
        }
    }
    return 0;
}


extern "C" int gpu_matrix_copy_peer(GpuMatrix *dst, const GpuMatrix *src, int *out_copied)
{
    if (dst) dst->host_observed_writer_ready.store(false, std::memory_order_release);
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
        uint64_t seen_completions = 0;
        for (const auto &state : states)
        {
            const auto &completion = states[state.completion_owner];
            const uint64_t bit = uint64_t{1} << state.completion_owner;
            if (seen_completions & bit) continue;
            seen_completions |= bit;
            if (completion.write_done && completion.write_done_valid)
            {
                error = cudaStreamWaitEvent(destination_stream, completion.write_done, 0);
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
    if (src->ctx->execution->release_streams_by_partition.empty() ||
        !src->ctx->execution->release_streams_by_partition[0])
    {
        cudaEventDestroy(peer_copy_done);
        return set_error("missing source release stream in gpu_matrix_copy_peer");
    }
    error = cudaSetDevice(source_device);
    if (error == cudaSuccess)
    {
        error = cudaStreamWaitEvent(
            src->ctx->execution->release_streams_by_partition[0], peer_copy_done, 0);
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

extern "C" int gpu_matrix_copy_peer_query(
    const GpuMatrix *src,
    const GpuContext *dst_ctx,
    int *out_compatible)
{
    if (!src || !dst_ctx || !out_compatible || !src->ctx)
    {
        return set_error("invalid gpu_matrix_copy_peer_query arguments");
    }
    *out_compatible = 0;

    // Mirror the destination object produced by gpu_matrix_create.  Keep
    // this path side-effect free: unlike the copy itself it must not call
    // cudaSetDevice/cudaDeviceEnablePeerAccess, allocate a matrix, or enqueue
    // any work.
    if (src->level < 0 || src->level > dst_ctx->level || src->ctx->N != dst_ctx->N)
    {
        return 0;
    }
    const size_t active_limbs = static_cast<size_t>(src->level + 1);
    if (src->ctx->moduli.size() < active_limbs || dst_ctx->moduli.size() < active_limbs ||
        !std::equal(
            src->ctx->moduli.begin(),
            src->ctx->moduli.begin() + active_limbs,
            dst_ctx->moduli.begin()))
    {
        return 0;
    }
    if (src->shared_limb_buffers.size() != 1 || src->exec_limb_states.size() != 1 ||
        src->exec_limb_states[0].empty() || !src->ctx->execution ||
        src->ctx->execution->release_streams_by_partition.empty() ||
        !src->ctx->execution->release_streams_by_partition[0] || !dst_ctx->execution)
    {
        return 0;
    }

    MatrixAllocationPlan destination_plan{};
    const int plan_status = build_matrix_allocation_plan(
        dst_ctx,
        src->level,
        src->rows,
        src->cols,
        static_cast<int>(src->format),
        &destination_plan);
    if (plan_status != 0)
    {
        // The route is incompatible with this context.  Do not turn a
        // negative admission result into a stale native error for callers.
        return 0;
    }
    if (destination_plan.partitions.size() != 1 ||
        destination_plan.partitions[0].local_limb_count == 0 ||
        dst_ctx->execution->compute_streams_by_partition.size() != 1 ||
        dst_ctx->execution->compute_streams_by_partition[0].empty())
    {
        return 0;
    }

    const auto &source_buffer = src->shared_limb_buffers[0];
    const auto &destination_partition = destination_plan.partitions[0];
    if (!source_buffer.ptr || source_buffer.bytes_total != destination_partition.data_bytes ||
        source_buffer.limb_count != destination_partition.local_limb_count ||
        source_buffer.bytes_per_poly != destination_partition.bytes_per_poly ||
        source_buffer.limb_coeff_bytes != destination_partition.limb_coeff_bytes ||
        source_buffer.limb_offsets_bytes != destination_partition.limb_offsets_bytes)
    {
        return 0;
    }

    const int source_device = source_buffer.device;
    const int destination_device = destination_partition.partition < dst_ctx->gpu_ids.size()
        ? dst_ctx->gpu_ids[destination_partition.partition]
        : -1;
    if (source_device < 0 || destination_device < 0)
    {
        return 0;
    }
    if (source_device != destination_device)
    {
        int can_access = 0;
        const cudaError_t error = cudaDeviceCanAccessPeer(
            &can_access,
            destination_device,
            source_device);
        if (error != cudaSuccess || !can_access)
        {
            return 0;
        }
    }
    *out_compatible = 1;
    return 0;
}
