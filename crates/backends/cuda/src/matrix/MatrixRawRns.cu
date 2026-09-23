// Plan-time CRT metadata and explicit-Graph physical descriptor adapters for
// RNS ModUp/ModDown and BlockModSwitch. Arithmetic uses MatrixCrt kernels.
namespace
{
    using RawCrtDescriptor = GpuMatrix::SharedLimbBuffer::DeviceDescriptor;

    uint64_t raw_crt_mul_mod(uint64_t left, uint64_t right, uint64_t modulus)
    {
        return static_cast<uint64_t>(
            (static_cast<unsigned __int128>(left) * right) % modulus);
    }

    __global__ void raw_crt_descriptor_kernel(MxxRawMatrixLimb limb,
        RawCrtDescriptor *destination)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            destination->base = reinterpret_cast<uint8_t *>(limb.address);
            destination->stride = static_cast<size_t>(limb.column_stride_bytes);
            destination->width = static_cast<uint8_t>(limb.word_bytes);
        }
    }

    bool raw_crt_basis_matches(const MxxRawMatrixView *view,
        const uint64_t *moduli, size_t count, const GpuModulusConversionPlan *plan)
    {
        if (!view || !view->limbs || view->limb_count != count ||
            view->physical_device != plan->device ||
            view->degree != plan->ring_dimension ||
            !view->rows || !view->columns ||
            view->rows > SIZE_MAX / view->columns)
            return false;
        for (size_t index = 0; index < count; ++index)
        {
            const auto &limb = view->limbs[index];
            if (!limb.address || limb.crt_limb_index != index ||
                limb.modulus != moduli[index] ||
                (limb.word_bytes != 4 && limb.word_bytes != 8) ||
                (limb.word_bytes == 4 && limb.modulus > UINT32_MAX) ||
                limb.coefficient_stride_bytes != limb.word_bytes ||
                view->degree > UINT64_MAX / limb.word_bytes ||
                limb.column_stride_bytes < view->degree * limb.word_bytes ||
                view->columns > UINT64_MAX / limb.column_stride_bytes ||
                limb.row_stride_bytes != view->columns * limb.column_stride_bytes)
                return false;
        }
        return true;
    }

    int raw_crt_emit_descriptors(GpuModulusConversionPlan *plan,
        GpuContext *ctx, cudaStream_t stream,
        const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
        uint32_t source_binding_base, uint32_t destination_binding_base)
    {
        if (source_binding_base > UINT32_MAX - source->limb_count ||
            destination_binding_base > UINT32_MAX - destination->limb_count)
            return set_error("raw CRT binding range overflow");
        for (size_t index = 0; index < source->limb_count; ++index)
        {
            const MxxGraphPatch patch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), source_binding_base + static_cast<uint32_t>(index), 0};
            const int result = mxx_gpu_launch_kernel(ctx, stream,
                raw_crt_descriptor_kernel, dim3(1), dim3(1), 0,
                &patch, 1, source->limbs[index],
                plan->raw_source_descriptors + index);
            if (result != 0) return result;
        }
        for (size_t index = 0; index < destination->limb_count; ++index)
        {
            const MxxGraphPatch patch{nullptr,
                MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), destination_binding_base + static_cast<uint32_t>(index), 0};
            const int result = mxx_gpu_launch_kernel(ctx, stream,
                raw_crt_descriptor_kernel, dim3(1), dim3(1), 0,
                &patch, 1, destination->limbs[index],
                plan->raw_target_descriptors + index);
            if (result != 0) return result;
        }
        return 0;
    }

    int raw_crt_plan_allocate(GpuModulusConversionPlan *plan,
        size_t static_metadata_bytes)
    {
        const size_t descriptor_count = plan->source_count + plan->target_count;
        if (!descriptor_count || descriptor_count > SIZE_MAX / sizeof(RawCrtDescriptor) ||
            static_metadata_bytes > SIZE_MAX - descriptor_count * sizeof(RawCrtDescriptor))
            return set_error("raw CRT metadata allocation overflow");
        plan->metadata_bytes = static_metadata_bytes + descriptor_count * sizeof(RawCrtDescriptor);
        cudaError_t error = cudaSetDevice(plan->device);
        if (error == cudaSuccess)
            error = cudaMallocAsync(&plan->device_metadata,
                plan->metadata_bytes, plan->stream);
        if (error == cudaSuccess)
            error = cudaHostAlloc(&plan->pinned_metadata,
                static_metadata_bytes, cudaHostAllocPortable);
        if (error == cudaSuccess)
            error = cudaEventCreateWithFlags(&plan->raw_ready,
                cudaEventDisableTiming);
        if (error != cudaSuccess) return set_error(error);
        auto *descriptor_base = reinterpret_cast<RawCrtDescriptor *>(
            static_cast<uint8_t *>(plan->device_metadata) + static_metadata_bytes);
        plan->raw_source_descriptors = descriptor_base;
        plan->raw_target_descriptors = descriptor_base + plan->source_count;
        return 0;
    }

    int raw_crt_plan_upload_static(GpuModulusConversionPlan *plan,
        const void *metadata, size_t bytes, bool device_copy)
    {
        std::memcpy(plan->pinned_metadata, metadata, bytes);
        cudaError_t error = cudaSuccess;
        if (device_copy)
            error = cudaMemcpyAsync(plan->device_metadata, plan->pinned_metadata,
                bytes, cudaMemcpyHostToDevice, plan->stream);
        if (error == cudaSuccess)
            error = cudaEventRecord(plan->raw_ready, plan->stream);
        return error == cudaSuccess ? 0 : set_error(error);
    }

    uint64_t raw_crt_words_mod(const uint64_t *words, size_t count, uint64_t modulus)
    {
        uint64_t value = 0;
        const uint64_t radix = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(1) << 64) % modulus);
        for (size_t word = count; word-- > 0;)
            value = static_cast<uint64_t>((static_cast<unsigned __int128>(value) *
                radix + words[word] % modulus) % modulus);
        return value;
    }
}

extern "C" int gpu_raw_rns_conversion_prepare(
    GpuContext *ctx, int32_t device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    size_t digit_size, int normalize,
    const uint64_t *plaintext_words, size_t plaintext_word_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !out_plan || !source_moduli || !target_moduli ||
        !source_count || !target_count || source_count > kCrtMaxLimbs ||
        target_count > kCrtMaxLimbs || !digit_size ||
        (normalize != 0 && normalize != 1) ||
        (plaintext_word_count != 0 && (!plaintext_words ||
            plaintext_words[plaintext_word_count - 1] == 0 ||
            (plaintext_word_count == 1 && plaintext_words[0] <= 1))))
        return set_error("invalid raw RNS conversion plan");
    *out_plan = nullptr;
    const bool down = plaintext_word_count != 0;
    if (down && target_count >= source_count)
        return set_error("raw RNS ModDown needs strict target subset");
    bool retained[kCrtMaxLimbs]{};
    RnsConversionMetadata metadata{};
    auto &launch = metadata.plan;
    launch.source_count = source_count;
    launch.target_count = target_count;
    launch.digit_size = digit_size;
    launch.group_count = down ? 1 : (source_count - 1) / digit_size + 1;
    // The scalar is only a mode marker; the kernel uses per-prime residues.
    launch.plaintext_modulus = down ? 1 : 0;
    for (size_t input = 0; input < source_count; ++input)
    {
        const uint64_t q = source_moduli[input];
        if (q <= 1 || !(q & 1)) return set_error("invalid raw RNS source prime");
        for (size_t previous = 0; previous < input; ++previous)
            if (source_moduli[previous] == q)
                return set_error("duplicate raw RNS source prime");
        launch.source_moduli[input] = q;
    }
    for (size_t target = 0; target < target_count; ++target)
    {
        const uint64_t q = target_moduli[target];
        if (q <= 1 || !(q & 1)) return set_error("invalid raw RNS target prime");
        for (size_t previous = 0; previous < target; ++previous)
            if (target_moduli[previous] == q)
                return set_error("duplicate raw RNS target prime");
        launch.target_moduli[target] = q;
        if (down)
            launch.plaintext_residues[target] = raw_crt_words_mod(
                plaintext_words, plaintext_word_count, q);
        size_t source = 0;
        while (source < source_count && source_moduli[source] != q) ++source;
        if (down)
        {
            if (source == source_count || retained[source])
                return set_error("raw RNS ModDown target is not a unique source subset");
            retained[source] = true;
            launch.retained[target] = source;
        }
    }
    if (!down)
        for (size_t input = 0; input < source_count; ++input)
        {
            bool found = false;
            for (size_t target = 0; target < target_count; ++target)
                found = found || source_moduli[input] == target_moduli[target];
            if (!found) return set_error("raw RNS ModUp target misses source prime");
        }
    for (size_t input = 0; input < source_count; ++input)
    {
        const uint64_t q = source_moduli[input];
        if (down && retained[input]) { launch.scales[input] = 0; continue; }
        uint64_t complement = 1;
        uint64_t outside = 1;
        for (size_t other = 0; other < source_count; ++other)
        {
            if (other == input) continue;
            const bool same_group = down ? !retained[other] :
                other / digit_size == input / digit_size;
            uint64_t &product = same_group ? complement : outside;
            product = raw_crt_mul_mod(product, source_moduli[other] % q, q);
        }
        uint64_t inverse = 0;
        if (!mod_inverse_u64(complement, q, inverse))
            return set_error("raw RNS complement is not invertible");
        uint64_t factor = 1;
        if (down)
        {
            if (!mod_inverse_u64(raw_crt_words_mod(
                plaintext_words, plaintext_word_count, q), q, factor))
                return set_error("raw RNS plaintext modulus is not invertible");
            factor = q - factor;
        }
        else if (normalize && !mod_inverse_u64(outside, q, factor))
            return set_error("raw RNS normalization factor is not invertible");
        launch.scales[input] = raw_crt_mul_mod(inverse, factor, q);
    }
    for (size_t target = 0; target < target_count; ++target)
    {
        const uint64_t q = target_moduli[target];
        uint64_t product = 1;
        if (down)
            for (size_t source = 0; source < source_count; ++source)
                if (!retained[source])
                    product = raw_crt_mul_mod(product, source_moduli[source] % q, q);
        if (!mod_inverse_u64(product, q, launch.inverses[target]))
            return set_error("raw RNS auxiliary modulus is not invertible");
    }
    for (size_t input = 0; input < source_count; ++input)
        for (size_t target = 0; target < target_count; ++target)
        {
            const uint64_t q = target_moduli[target];
            uint64_t weight = down && retained[input] ? 0 : 1;
            const size_t begin = down ? 0 : input / digit_size * digit_size;
            const size_t end = down ? source_count :
                std::min(source_count, begin + digit_size);
            for (size_t other = begin; other < end && weight != 0; ++other)
                if (other != input && (!down || !retained[other]))
                    weight = raw_crt_mul_mod(weight, source_moduli[other] % q, q);
            metadata.weights[input * kCrtMaxLimbs + target] = weight;
        }
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw RNS plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->target_count = target_count;
    plan->raw_rns = true;
    int result = raw_crt_plan_allocate(plan, sizeof(metadata));
    if (result == 0) result = raw_crt_plan_upload_static(plan, &metadata, sizeof(metadata), true);
    if (result != 0) { gpu_matrix_modulus_conversion_plan_destroy(plan); return result; }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_block_mod_switch_prepare(
    GpuContext *ctx, int32_t device, void *stream_raw,
    const uint64_t *source_moduli, size_t source_count,
    const uint64_t *target_moduli, size_t target_count,
    const uint64_t *plaintext_words, size_t word_count,
    GpuModulusConversionPlan **out_plan)
{
    if (!ctx || !stream_raw || !out_plan || !source_moduli || !target_moduli ||
        !plaintext_words || !word_count || !source_count || !target_count ||
        source_count > kCrtMaxLimbs || target_count >= source_count)
        return set_error("invalid raw BlockModSwitch plan");
    *out_plan = nullptr;
    BlockModSwitchMetadata metadata{};
    metadata.source_count = source_count;
    metadata.output.limb_count = target_count;
    bool retained[kCrtMaxLimbs]{};
    for (size_t input = 0; input < source_count; ++input)
    {
        if (source_moduli[input] <= 1 || !(source_moduli[input] & 1))
            return set_error("invalid raw BlockModSwitch source basis");
        for (size_t previous = 0; previous < input; ++previous)
            if (source_moduli[previous] == source_moduli[input])
                return set_error("duplicate raw BlockModSwitch source prime");
        metadata.source_moduli[input] = source_moduli[input];
    }
    for (size_t target = 0; target < target_count; ++target)
    {
        const uint64_t q = target_moduli[target];
        size_t input = 0;
        while (input < source_count && source_moduli[input] != q) ++input;
        if (q <= 1 || !(q & 1) || input == source_count || retained[input])
            return set_error("invalid raw BlockModSwitch target basis");
        retained[input] = true;
        metadata.target_source[target] = static_cast<int>(input);
        metadata.output.moduli[target] = q;
        metadata.target_t[target] = raw_crt_words_mod(plaintext_words, word_count, q);
    }
    for (size_t input = 0; input < source_count; ++input)
        if (!retained[input]) metadata.dropped_indices[metadata.dropped_count++] = input;
    if (!metadata.dropped_count)
        return set_error("raw BlockModSwitch requires dropped basis");
    for (size_t current = 0; current < metadata.dropped_count; ++current)
    {
        const uint64_t p = source_moduli[metadata.dropped_indices[current]];
        if (!mod_inverse_u64(raw_crt_words_mod(plaintext_words, word_count, p),
                p, metadata.dropped_t_inverses[current]))
            return set_error("raw BlockModSwitch t is not invertible");
        uint64_t product = 1;
        for (size_t previous = 0; previous < current; ++previous)
            product = raw_crt_mul_mod(product,
                source_moduli[metadata.dropped_indices[previous]] % p, p);
        if (!mod_inverse_u64(product, p, metadata.prefix_inverses[current]))
            return set_error("raw BlockModSwitch dropped basis is not invertible");
    }
    for (size_t target = 0; target < target_count; ++target)
    {
        const uint64_t q = target_moduli[target];
        uint64_t product = 1;
        for (size_t current = 0; current < metadata.dropped_count; ++current)
            product = raw_crt_mul_mod(product,
                source_moduli[metadata.dropped_indices[current]] % q, q);
        if (!mod_inverse_u64(product, q, metadata.target_p_inverses[target]))
            return set_error("raw BlockModSwitch dropped product is not invertible");
    }
    auto *plan = new (std::nothrow) GpuModulusConversionPlan();
    if (!plan) return set_error("failed to allocate raw BlockModSwitch plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->ring_dimension = static_cast<size_t>(ctx->N);
    plan->source_count = source_count;
    plan->target_count = target_count;
    plan->raw_block = true;
    int result = raw_crt_plan_allocate(plan, sizeof(metadata));
    if (result == 0)
    {
        metadata.source = plan->raw_source_descriptors;
        metadata.output.descriptors = plan->raw_target_descriptors;
        result = raw_crt_plan_upload_static(plan, &metadata, sizeof(metadata), false);
    }
    if (result != 0) { gpu_matrix_modulus_conversion_plan_destroy(plan); return result; }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_conversion_plan_wait(
    const GpuModulusConversionPlan *plan, void *stream_raw)
{
    if (!plan || !stream_raw)
        return set_error("invalid raw conversion plan wait");
    if (!plan->raw_ready) return 0;
    cudaError_t error = cudaSetDevice(plan->device);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_raw),
            plan->raw_ready, 0);
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" int gpu_raw_rns_conversion_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (!plan || !plan->raw_rns || !ctx || plan->context != ctx ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream_raw))
        return set_error("invalid raw RNS graph operation");
    const auto *metadata = static_cast<const RnsConversionMetadata *>(plan->pinned_metadata);
    const auto &launch = metadata->plan;
    if (!raw_crt_basis_matches(source, launch.source_moduli, plan->source_count, plan) ||
        !raw_crt_basis_matches(destination, launch.target_moduli, plan->target_count, plan) ||
        source->columns != destination->columns ||
        source->rows > SIZE_MAX / launch.group_count ||
        destination->rows != source->rows * launch.group_count)
        return set_error("invalid raw RNS physical views");
    if (cudaSetDevice(plan->device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    int result = raw_crt_emit_descriptors(plan, ctx, stream, source, destination,
        source_binding_base, destination_binding_base);
    if (result != 0) return result;
    if (source->rows * source->columns > SIZE_MAX / source->degree)
        return set_error("raw RNS coefficient count overflow");
    const size_t count = source->rows * source->columns * source->degree;
    if (count > SIZE_MAX / plan->target_count / launch.group_count)
        return set_error("raw RNS grid overflow");
    const size_t total = count * plan->target_count * launch.group_count;
    const size_t blocks = (total - 1) / 128 + 1;
    if (blocks > INT_MAX) return set_error("raw RNS grid exceeds CUDA limit");
    return mxx_gpu_launch_kernel(ctx, stream, rns_conversion_kernel,
        dim3(static_cast<unsigned>(blocks)), dim3(128), 0, nullptr, 0,
        plan->raw_source_descriptors, plan->raw_target_descriptors,
        static_cast<const RnsConversionMetadata *>(plan->device_metadata),
        count, static_cast<size_t>(source->degree), launch.plaintext_modulus != 0);
}

extern "C" int gpu_raw_block_mod_switch_emit(
    GpuModulusConversionPlan *plan, GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (!plan || !plan->raw_block || !ctx || plan->context != ctx ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream_raw))
        return set_error("invalid raw BlockModSwitch graph operation");
    const auto metadata = *static_cast<const BlockModSwitchMetadata *>(plan->pinned_metadata);
    if (!raw_crt_basis_matches(source, metadata.source_moduli, plan->source_count, plan) ||
        !raw_crt_basis_matches(destination, metadata.output.moduli, plan->target_count, plan) ||
        source->rows != destination->rows || source->columns != destination->columns)
        return set_error("invalid raw BlockModSwitch physical views");
    if (cudaSetDevice(plan->device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    int result = raw_crt_emit_descriptors(plan, ctx, stream, source, destination,
        source_binding_base, destination_binding_base);
    if (result != 0) return result;
    if (source->rows * source->columns > SIZE_MAX / source->degree)
        return set_error("raw BlockModSwitch coefficient count overflow");
    const size_t count = source->rows * source->columns * source->degree;
    if (count > SIZE_MAX / plan->target_count)
        return set_error("raw BlockModSwitch grid overflow");
    const size_t blocks = (count * plan->target_count - 1) / 128 + 1;
    if (blocks > INT_MAX) return set_error("raw BlockModSwitch grid exceeds CUDA limit");
    return mxx_gpu_launch_kernel(ctx, stream, block_mod_switch_kernel,
        dim3(static_cast<unsigned>(blocks)), dim3(128), 0, nullptr, 0,
        metadata, count, static_cast<size_t>(source->degree));
}
