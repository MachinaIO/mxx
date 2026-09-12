namespace
{
    constexpr int kCrtMaxLimbs = 64;
    constexpr int kCrtMaxWords = 64;

    struct ModulusConversionPlan
    {
        size_t source_count;
        size_t target_count;
        size_t discarded_count;
        uint8_t retained[kCrtMaxLimbs];
        uint8_t discarded[kCrtMaxLimbs];
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t target_moduli[kCrtMaxLimbs];
        uint64_t division_inverses[kCrtMaxLimbs];
        uint64_t input_scales[kCrtMaxLimbs];
        uint64_t plaintext_modulus;
        uint64_t divisor_residues[kCrtMaxLimbs];
        const uint64_t *garner;
        size_t garner_stride;
    };
    static_assert(sizeof(ModulusConversionPlan) + 10 * sizeof(size_t) < 4096,
        "retained CRT metadata must fit baseline CUDA kernel arguments");

    struct ModulusConversionMetadata {
        ModulusConversionPlan plan;
        uint64_t garner[kCrtMaxLimbs * kCrtMaxLimbs];
    };

    __device__ __forceinline__ void convert_modulus_coefficient(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const ModulusConversionPlan *metadata,
        size_t coefficient_count,
        size_t ring_dimension,
        int conversion, size_t columns, size_t input_offset, size_t input_pitch,
        size_t output_offset, size_t output_pitch)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        const size_t input_poly = input_offset + (poly / columns) * input_pitch + poly % columns;
        const size_t output_poly = output_offset + (poly / columns) * output_pitch + poly % columns;
        uint64_t digits[kCrtMaxLimbs];
        bool negative = false;
        if (conversion != 0)
        {
            for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            {
                const size_t source_index = metadata->discarded[limb];
                const auto descriptor = source[source_index];
                const uint64_t modulus = metadata->source_moduli[source_index];
                uint64_t digit = matrix_load_limb_u64(descriptor.base, input_poly, coefficient,
                    descriptor.stride, descriptor.width) % modulus;
                digit = mul_mod_u64(digit, metadata->input_scales[source_index], modulus);
                for (size_t previous = 0; previous < limb; ++previous)
                {
                    const uint64_t residue = digits[previous] % modulus;
                    const uint64_t difference = digit >= residue ? digit - residue :
                        modulus - (residue - digit);
                    digit = mul_mod_u64(difference,
                        metadata->garner[metadata->discarded[previous] * metadata->garner_stride + source_index], modulus);
                }
                digits[limb] = digit;
            }
            // For odd radices, (J-1)/2 has mixed-radix digits (q_i-1)/2.
            // Comparing from the most significant digit chooses the exact
            // centered representative of x mod J without a large-integer divide.
            for (size_t limb = 0; limb < metadata->discarded_count; ++limb)
            {
                const uint64_t half = metadata->source_moduli[metadata->discarded[limb]] / 2;
                negative = (digits[limb] > half) | ((digits[limb] == half) & negative);
            }
        }
        for (size_t limb = 0; limb < metadata->target_count; ++limb)
        {
            const uint64_t modulus = metadata->target_moduli[limb];
            uint64_t value = 0;
            if (conversion != 2)
            {
                const auto input = source[metadata->retained[limb]];
                value = matrix_load_limb_u64(input.base, input_poly, coefficient,
                    input.stride, input.width) % modulus;
            }
            if (conversion != 0)
            {
                uint64_t residual = 0;
                for (size_t digit = metadata->discarded_count; digit-- > 0;)
                {
                    residual = add_mod_u64(mul_mod_u64(residual,
                        metadata->source_moduli[metadata->discarded[digit]] % modulus,
                        modulus), digits[digit] % modulus, modulus);
                }
                const uint64_t divisor = metadata->divisor_residues[limb] &
                    (uint64_t{0} - static_cast<uint64_t>(negative));
                residual = residual >= divisor ? residual - divisor : modulus - (divisor - residual);
                if (conversion == 2) value = residual;
                else
                {
                    if (conversion == 3)
                        residual = mul_mod_u64(residual, metadata->plaintext_modulus % modulus, modulus);
                    value = value >= residual ? value - residual : modulus - (residual - value);
                    value = mul_mod_u64(value, metadata->division_inverses[limb], modulus);
                }
            }
            const auto output = target[limb];
            matrix_store_limb_u64(output.base, output_poly, coefficient,
                output.stride, output.width, value);
        }
    }

    __global__ void convert_modulus_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const ModulusConversionPlan *metadata, size_t coefficient_count,
        size_t ring_dimension, int conversion, size_t columns,
        size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        convert_modulus_coefficient(source, target, metadata, coefficient_count,
            ring_dimension, conversion, columns, input_offset, input_pitch, output_offset, output_pitch);
    }

    __global__ void convert_modulus_range_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        ModulusConversionPlan metadata, size_t coefficient_count,
        size_t ring_dimension, int conversion, size_t columns,
        size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        convert_modulus_coefficient(source, target, &metadata, coefficient_count,
            ring_dimension, conversion, columns, input_offset, input_pitch, output_offset, output_pitch);
    }

    __device__ __forceinline__ int crt_compare_words(
        const uint64_t *lhs,
        const uint64_t *rhs,
        int word_count)
    {
        for (int word = word_count - 1; word >= 0; --word)
        {
            if (lhs[word] != rhs[word])
            {
                return lhs[word] > rhs[word] ? 1 : -1;
            }
        }
        return 0;
    }

    __device__ uint64_t crt_rounded_scale(
        uint64_t *value_words,
        const uint64_t *modulus_words,
        int word_count,
        uint64_t plaintext_modulus)
    {
        // Form plaintext_modulus * value + floor(Q / 2) exactly. The extra
        // word contains the multiplication/addition carry.
        uint64_t carry = 0;
        for (int word = 0; word < word_count; ++word)
        {
            const unsigned __int128 product =
                static_cast<unsigned __int128>(value_words[word]) * plaintext_modulus + carry;
            value_words[word] = static_cast<uint64_t>(product);
            carry = static_cast<uint64_t>(product >> 64);
        }
        value_words[word_count] = carry;

        uint64_t add_carry = 0;
        for (int word = 0; word < word_count; ++word)
        {
            const uint64_t current = modulus_words[word];
            const uint64_t next = word + 1 < word_count ? modulus_words[word + 1] : 0;
            const uint64_t half_word = (current >> 1) | ((next & 1) << 63);
            const unsigned __int128 sum =
                static_cast<unsigned __int128>(value_words[word]) + half_word + add_carry;
            value_words[word] = static_cast<uint64_t>(sum);
            add_carry = static_cast<uint64_t>(sum >> 64);
        }
        value_words[word_count] += add_carry;

        // The quotient is at most plaintext_modulus. Binary search avoids a
        // general multi-word division while retaining exact integer rounding.
        uint64_t low = 0;
        uint64_t high = plaintext_modulus;
        uint64_t multiple[kCrtMaxWords + 1];
        while (low < high)
        {
            const uint64_t midpoint = low + (high - low) / 2 + (high - low) % 2;
            uint64_t mul_carry = 0;
            for (int word = 0; word < word_count; ++word)
            {
                const unsigned __int128 product =
                    static_cast<unsigned __int128>(modulus_words[word]) * midpoint + mul_carry;
                multiple[word] = static_cast<uint64_t>(product);
                mul_carry = static_cast<uint64_t>(product >> 64);
            }
            multiple[word_count] = mul_carry;
            if (crt_compare_words(value_words, multiple, word_count + 1) >= 0)
            {
                low = midpoint;
            }
            else
            {
                high = midpoint - 1;
            }
        }
        return low % plaintext_modulus;
    }

    struct CrtLevelMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        size_t limb_count;
        int word_count;
        uint64_t plaintext_modulus;
        uint64_t moduli[kCrtMaxLimbs];
        const uint64_t *garner;
        size_t garner_stride;
        size_t input_offset;
        uint64_t modulus_words[kCrtMaxWords];
        uint64_t reconstruction[kCrtMaxLimbs];
    };

    struct CrtOutputMetadata
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        size_t limb_count;
        uint64_t moduli[kCrtMaxLimbs];
    };

    __global__ void centered_rebase_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        CrtOutputMetadata output, uint64_t source_modulus,
        size_t coefficient_count, size_t ring_dimension, size_t columns,
        size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * output.limb_count) return;
        const size_t limb = index / coefficient_count;
        const size_t position = index % coefficient_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        const auto input = source[0];
        const uint64_t residue = matrix_load_limb_u64(input.base,
            input_offset + (poly / columns) * input_pitch + poly % columns, coefficient,
            input.stride, input.width) % source_modulus;
        const uint64_t modulus = output.moduli[limb];
        // Keep the magnitude unsigned, including for source moduli above INT64_MAX.
        const bool negative = residue > source_modulus / 2;
        const uint64_t magnitude = (negative ? source_modulus - residue : residue) % modulus;
        const uint64_t value = negative && magnitude != 0 ? modulus - magnitude : magnitude;
        const auto target = output.descriptors[limb];
        matrix_store_limb_u64(target.base,
            output_offset + (poly / columns) * output_pitch + poly % columns, coefficient,
            target.stride, target.width, value);
    }

    __device__ __forceinline__ void crt_recompose_coefficient(
        const CrtLevelMetadata *levels, const CrtOutputMetadata *output,
        size_t level_count, size_t coefficient_count, size_t ring_dimension, size_t output_offset)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const size_t poly = index / ring_dimension;
        const size_t coefficient = index % ring_dimension;
        uint64_t accumulated[kCrtMaxLimbs]{};
        for (size_t level = 0; level < level_count; ++level)
        {
            const auto &metadata = levels[level];
            uint64_t mixed_digits[kCrtMaxLimbs];
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const auto descriptor = metadata.descriptors[limb];
                const uint64_t modulus = metadata.moduli[limb];
                uint64_t digit = matrix_load_limb_u64(descriptor.base, metadata.input_offset + poly, coefficient,
                    descriptor.stride, descriptor.width) % modulus;
                for (size_t previous = 0; previous < limb; ++previous)
                {
                    const uint64_t residue = mixed_digits[previous] % modulus;
                    const uint64_t difference = digit >= residue ? digit - residue :
                        modulus - (residue - digit);
                    digit = mul_mod_u64(difference,
                        metadata.garner[previous * metadata.garner_stride + limb], modulus);
                }
                mixed_digits[limb] = digit;
            }
            uint64_t value_words[kCrtMaxWords + 1]{};
            for (size_t limb = metadata.limb_count; limb-- > 0;)
            {
                uint64_t carry = mixed_digits[limb];
                for (int word = 0; word < metadata.word_count; ++word)
                {
                    const unsigned __int128 term =
                        static_cast<unsigned __int128>(value_words[word]) * metadata.moduli[limb] + carry;
                    value_words[word] = static_cast<uint64_t>(term);
                    carry = static_cast<uint64_t>(term >> 64);
                }
            }
            const uint64_t rounded = crt_rounded_scale(value_words, metadata.modulus_words,
                metadata.word_count, metadata.plaintext_modulus);
            for (size_t limb = 0; limb < output->limb_count; ++limb)
            {
                const uint64_t modulus = output->moduli[limb];
                accumulated[limb] = add_mod_u64(accumulated[limb],
                    mul_mod_u64(rounded % modulus, metadata.reconstruction[limb], modulus), modulus);
            }
        }
        for (size_t limb = 0; limb < output->limb_count; ++limb)
        {
            const auto descriptor = output->descriptors[limb];
            matrix_store_limb_u64(descriptor.base, output_offset + poly, coefficient,
                descriptor.stride, descriptor.width, accumulated[limb]);
        }
    }
    struct CrtLaunchMetadata {
        CrtLevelMetadata levels[2];
        CrtOutputMetadata output;
    };
    static_assert(sizeof(CrtLaunchMetadata) + 6 * sizeof(size_t) <= 4096,
        "CRT range metadata must fit portable CUDA kernel arguments");
    static_assert(sizeof(CrtLevelMetadata) % sizeof(uint64_t) == 0 && sizeof(CrtOutputMetadata) % sizeof(uint64_t) == 0,
        "CRT setup metadata must consist of aligned whole words");

    __global__ void crt_recompose_kernel(const CrtLevelMetadata *levels, const CrtOutputMetadata *output,
        size_t level_count, size_t count, size_t dimension, size_t output_offset)
    {
        crt_recompose_coefficient(levels, output, level_count, count, dimension, output_offset);
    }

    __global__ void crt_recompose_range_kernel(CrtLaunchMetadata metadata,
        size_t level_count, size_t count, size_t dimension, size_t output_offset)
    {
        crt_recompose_coefficient(metadata.levels, &metadata.output, level_count, count, dimension, output_offset);
    }

    // Copy two public launch records at a time without a reusable host DMA buffer.
    // All input coefficient arrays and immutable Garner tables remain device-resident.
    __global__ void crt_recompose_setup_kernel(CrtLevelMetadata *levels, CrtOutputMetadata *output,
        CrtLaunchMetadata metadata, size_t start, size_t count)
    {
        const auto *source = reinterpret_cast<const uint64_t *>(metadata.levels);
        auto *destination = reinterpret_cast<uint64_t *>(levels + start);
        for (size_t word = threadIdx.x; word < count * sizeof(CrtLevelMetadata) / sizeof(uint64_t); word += blockDim.x)
            destination[word] = source[word];
        if (start == 0) {
            source = reinterpret_cast<const uint64_t *>(&metadata.output);
            destination = reinterpret_cast<uint64_t *>(output);
            for (size_t word = threadIdx.x; word < sizeof(CrtOutputMetadata) / sizeof(uint64_t); word += blockDim.x)
                destination[word] = source[word];
        }
    }

}

extern "C" int gpu_matrix_convert_modulus(
    GpuMatrix *out,
    const GpuMatrix *source,
    int conversion,
    const uint64_t *division_inverses,
    size_t inverse_count,
    uint64_t plaintext_modulus,
    const uint64_t *input_scales,
    const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_convert_modulus");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!source || out == source || !source->ctx ||
        out->ctx->execution != source->ctx->execution ||
        out->ctx->N != source->ctx->N || out->ctx->N <= 0 ||
        (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL) ||
        (conversion == 0 && out->format != source->format) ||
        (conversion != 0 && source->format != GPU_POLY_FORMAT_COEFF) ||
        (conversion < 0 || conversion > 3) || source->level < 0 || out->level < 0 ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("invalid coefficient modulus conversion layout");
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs || (conversion != 2 && target_count > source_count) ||
        source_count > source->ctx->moduli.size() || target_count > out->ctx->moduli.size() ||
        !division_inverses || inverse_count != target_count)
        return set_error("invalid coefficient modulus conversion basis");
    if (conversion == 3 && (plaintext_modulus == 0 || !input_scales || target_count >= source_count))
        return set_error("invalid exact BGV block switch parameters");
    const GpuMatrixRange input_range = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input_range, source) || !valid(output_range, out) ||
        input_range.row_end - input_range.row_start != output_range.row_end - output_range.row_start ||
        input_range.column_end - input_range.column_start != output_range.column_end - output_range.column_start)
        return set_error("invalid modulus conversion rectangle");
    const size_t rows = input_range.row_end - input_range.row_start;
    const size_t columns = input_range.column_end - input_range.column_start;
    GpuMatrixTransformWorkspaceBytes requirements{};
    int status = 0;
    ModulusConversionPlan metadata{};
    std::vector<ModulusConversionMetadata> host;
    metadata.plaintext_modulus = plaintext_modulus;
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    bool retained[kCrtMaxLimbs]{};
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        metadata.source_moduli[limb] = source->ctx->moduli[limb];
        metadata.input_scales[limb] = conversion == 3 ? input_scales[limb] : 1;
        if (metadata.source_moduli[limb] <= 1 || !(metadata.source_moduli[limb] & 1))
            return set_error("nearest conversion requires odd CRT moduli");
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = out->ctx->moduli[limb];
        size_t selected = 0;
        while (selected < source_count && metadata.source_moduli[selected] != modulus) ++selected;
        if (conversion != 2 && (selected == source_count || retained[selected]))
            return set_error("destination CRT basis is not an exact subset");
        if (selected < source_count) retained[selected] = true;
        metadata.retained[limb] = selected;
        metadata.target_moduli[limb] = modulus;
        metadata.division_inverses[limb] = division_inverses[limb];
    }
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        if (conversion == 2 && !retained[limb])
            return set_error("centered extension requires a containing CRT basis");
        if (conversion == 3 && (static_cast<unsigned __int128>(plaintext_modulus) *
            metadata.input_scales[limb]) % metadata.source_moduli[limb] != 1)
            return set_error("invalid BGV plaintext inverse");
        if (conversion == 2 || !retained[limb]) metadata.discarded[metadata.discarded_count++] = limb;
    }
    if (!view) {
        host.resize(1);
        metadata.garner_stride = kCrtMaxLimbs;
        for (size_t row = 0; row < source_count; ++row)
            for (size_t column = 0; column < source_count; ++column)
                host[0].garner[row * kCrtMaxLimbs + column] = source->ctx->garner_inverse_table[
                    row * source->ctx->moduli.size() + column];
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const uint64_t modulus = metadata.target_moduli[limb];
        uint64_t divisor = 1;
        for (size_t discarded = 0; discarded < metadata.discarded_count; ++discarded)
            divisor = static_cast<uint64_t>((static_cast<unsigned __int128>(divisor) *
                metadata.source_moduli[metadata.discarded[discarded]]) % modulus);
        metadata.divisor_residues[limb] = divisor;
        if ((conversion == 1 || conversion == 3) && (static_cast<unsigned __int128>(divisor) * division_inverses[limb]) % modulus != 1)
            return set_error("invalid exact modulus division inverse");
    }
    if (!rows || !columns) return 0;
    const size_t n = static_cast<size_t>(out->ctx->N);
    if (out->rows > SIZE_MAX / out->cols || source->rows > SIZE_MAX / source->cols ||
        out->rows * out->cols > SIZE_MAX / n || source->rows * source->cols > SIZE_MAX / n)
        return set_error("modulus conversion shape overflow");
    status = gpu_matrix_query_crt_workspace_bytes(out->ctx, out->level, out->rows, out->cols,
        GPU_MATRIX_CRT_CONVERT_MODULUS, source_count, 1, view != nullptr, &requirements);
    if (status != 0) return status;
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("coefficient conversion requires colocated device descriptors");
    cudaStream_t stream = nullptr;
    status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported source CRT limb placement");
        status = matrix_wait_limb_stream(source, id, output.device, stream, false, true);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb)
            return set_error("unsupported destination CRT limb placement");
    }
    const size_t coefficient_count = rows * columns * n;
    const size_t blocks = coefficient_count / 128 + (coefficient_count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("coefficient conversion exceeds CUDA grid capacity");
    MatrixTransformWorkspace workspace;
    if (view) {
        if (source->ctx->ring_device_constants.empty() ||
            source->ctx->ring_device_constants[0].device != output.device ||
            !source->ctx->ring_device_constants[0].garner_inverses)
            return set_error("missing setup CRT inverse table");
        metadata.garner = source->ctx->ring_device_constants[0].garner_inverses;
        metadata.garner_stride = source->ctx->moduli.size();
        status = matrix_wait_all_limb_streams(out, output.device, stream);
        if (status != 0) return status;
        // CUDA owns by-value launch arguments before this call returns. No host
        // staging allocation can remain in flight when the next wave is submitted.
        convert_modulus_range_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            input.device_descriptors, output.device_descriptors, metadata, coefficient_count, n, conversion, columns,
            input_range.row_start * source->cols + input_range.column_start, source->cols,
            output_range.row_start * out->cols + output_range.column_start, out->cols);
        error = cudaGetLastError();
        status = error == cudaSuccess ? matrix_record_all_limb_writes(out, stream, true) : set_error(error);
    } else {
        status = workspace.acquire(out, output.device, stream, requirements);
        if (status != 0) return status;
        auto *device_metadata = reinterpret_cast<ModulusConversionMetadata *>(workspace.base);
        metadata.garner = device_metadata->garner;
        host[0].plan = metadata;
        std::memcpy(workspace.pinned, host.data(), sizeof(ModulusConversionMetadata));
        status = workspace.upload();
        if (status == 0) {
            convert_modulus_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
                input.device_descriptors, output.device_descriptors, &device_metadata->plan,
                coefficient_count, n, conversion, columns,
                input_range.row_start * source->cols + input_range.column_start, source->cols,
                output_range.row_start * out->cols + output_range.column_start, out->cols);
            error = cudaGetLastError();
            if (error != cudaSuccess) status = set_error(error);
        }
        if (status == 0) status = workspace.complete();
    }
    if (status == 0) {
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const auto &states = out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        status = matrix_track_all_limb_consumers(source, output.device, stream, completion, true, true);
    }
    if (status == 0 && conversion != 0 && out->format == GPU_POLY_FORMAT_EVAL)
        status = run_matrix_transform_u64<true>(out, &output_range);
    if (status != 0) {
        workspace.retire();
        const int retired = gpu_matrix_retire_submitted_work(out);
        if (retired != 0) return retired;
    }
    return status;
}

extern "C" int gpu_matrix_centered_rebase(
    GpuMatrix *out, const GpuMatrix *source, const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_centered_rebase");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!source || out == source || !source->ctx ||
        out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
        source->level != 0 || source->ctx->moduli.size() != 1 || out->level < 0 ||
        source->format != GPU_POLY_FORMAT_COEFF ||
        (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL))
        return set_error("centered rebase requires single-limb coefficients and matching execution");
    const GpuMatrixRange input_range = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input_range, source) || !valid(output_range, out) ||
        input_range.row_end - input_range.row_start != output_range.row_end - output_range.row_start ||
        input_range.column_end - input_range.column_start != output_range.column_end - output_range.column_start)
        return set_error("invalid centered rebase rectangle");
    const size_t rows = input_range.row_end - input_range.row_start;
    const size_t columns = input_range.column_end - input_range.column_start;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (out->ctx->N <= 0 || target_count > kCrtMaxLimbs || target_count > out->ctx->moduli.size())
        return set_error("invalid centered rebase destination basis");
    if (!rows || !columns) return 0;
    const size_t n = static_cast<size_t>(out->ctx->N);
    if (source->rows > SIZE_MAX / source->cols || out->rows > SIZE_MAX / out->cols ||
        source->rows * source->cols > SIZE_MAX / n || out->rows * out->cols > SIZE_MAX / n)
        return set_error("centered rebase shape overflow");
    const size_t coefficient_count = rows * columns * n;
    if (coefficient_count > SIZE_MAX / target_count)
        return set_error("centered rebase coefficient count overflow");
    const size_t count = coefficient_count * target_count;
    const size_t blocks = count / 128 + (count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("centered rebase exceeds CUDA grid capacity");
    if (source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("centered rebase requires colocated partitions");
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count != 1 || output.limb_count < target_count)
        return set_error("centered rebase requires colocated device descriptors");
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(source, output.device, stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, output.device, stream);
    if (status != 0) return status;
    CrtOutputMetadata metadata{};
    metadata.descriptors = output.device_descriptors;
    metadata.limb_count = target_count;
    std::copy_n(out->ctx->moduli.begin(), target_count, metadata.moduli);
    centered_rebase_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
        input.device_descriptors, metadata, source->ctx->moduli[0], coefficient_count, n, columns,
        input_range.row_start * source->cols + input_range.column_start, source->cols,
        output_range.row_start * out->cols + output_range.column_start, out->cols);
    error = cudaGetLastError();
    status = error == cudaSuccess ? matrix_record_all_limb_writes(out, stream, true) : set_error(error);
    if (status == 0) {
        // Reuse the retained output completion; an immediately dropped source
        // joins this reader without creating another event or mutating its writer.
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const auto &states = out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        status = matrix_track_all_limb_consumers(source, output.device, stream, completion, true, true);
    }
    if (status == 0 && out->format == GPU_POLY_FORMAT_EVAL)
        status = run_matrix_transform_u64<true>(out, &output_range);
    if (status != 0) {
        const int retired = gpu_matrix_retire_submitted_work(out);
        if (retired != 0) return retired;
    }
    return status;
}

extern "C" int gpu_matrix_crt_recompose(
    GpuMatrix *out, const GpuMatrix *const *levels, size_t level_count,
    const uint64_t *plaintext_moduli, const uint64_t *reconstruction_residues,
    size_t reconstruction_stride, const GpuMatrixRange *input_views, const GpuMatrixRange *output_view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_crt_recompose");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !out->ctx || !levels || !plaintext_moduli || !reconstruction_residues ||
        !level_count || out->level < 0 || out->ctx->N <= 0 ||
        (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL) ||
        ((input_views == nullptr) != (output_view == nullptr)) || out->shared_limb_buffers.size() != 1)
        return set_error("invalid CRT recomposition output");
    const GpuMatrixRange output_range = output_view ? *output_view : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols && r.row_end - r.row_start == 1;
    };
    if (!valid(output_range, out)) return set_error("invalid CRT output rectangle");
    const size_t columns = output_range.column_end - output_range.column_start;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    if (target_count > kCrtMaxLimbs || target_count > out->ctx->moduli.size() ||
        reconstruction_stride != target_count)
        return set_error("invalid CRT recomposition output basis");
    const auto &target = out->shared_limb_buffers[0];
    if (level_count > (SIZE_MAX - sizeof(CrtOutputMetadata)) / sizeof(CrtLevelMetadata))
        return set_error("CRT metadata count overflow");
    int status = 0;
    std::vector<CrtLevelMetadata> metadata(level_count);
    std::vector<CrtOutputMetadata> output_metadata(1);
    output_metadata[0].descriptors = target.device_descriptors;
    output_metadata[0].limb_count = target_count;
    std::copy_n(out->ctx->moduli.begin(), target_count, output_metadata[0].moduli);

    for (size_t level = 0; level < level_count; ++level)
    {
        const auto *source = levels[level];
        if (!source || !source->ctx || source->ctx->execution != out->ctx->execution ||
            source == out || source->ctx->N != out->ctx->N ||
            source->level < 0 || source->format != GPU_POLY_FORMAT_COEFF ||
            source->shared_limb_buffers.size() != 1 || !plaintext_moduli[level])
            return set_error("invalid mixed-modulus CRT input");
        const GpuMatrixRange range = input_views ? input_views[level] : GpuMatrixRange{0, source->rows, 0, source->cols};
        if (!valid(range, source) || range.column_end - range.column_start != columns)
            return set_error("invalid CRT input rectangle");
        if (source->cols != 0 && (source->rows > SIZE_MAX / source->cols ||
            source->rows * source->cols > SIZE_MAX / static_cast<size_t>(source->ctx->N)))
            return set_error("CRT input owner size overflow");
        const size_t count = static_cast<size_t>(source->level) + 1;
        const auto &buffer = source->shared_limb_buffers[0];
        if (count > kCrtMaxLimbs || count > source->ctx->moduli.size() ||
            buffer.device != target.device || (columns && (!buffer.device_descriptors || buffer.limb_count < count)))
            return set_error("invalid mixed-modulus CRT basis or placement");
        auto &entry = metadata[level];
        entry.descriptors = buffer.device_descriptors;
        entry.input_offset = range.row_start * source->cols + range.column_start;
        entry.limb_count = count;
        entry.plaintext_modulus = plaintext_moduli[level];
        std::copy_n(source->ctx->moduli.begin(), count, entry.moduli);
        std::copy_n(reconstruction_residues + level * target_count, target_count, entry.reconstruction);
        std::vector<uint64_t> words;
        const std::vector<uint64_t> active(source->ctx->moduli.begin(), source->ctx->moduli.begin() + count);
        if (!serde_compute_modulus_words_le(active, &words) || words.empty() || words.size() > kCrtMaxWords)
            return set_error("unsupported CRT source modulus size");
        entry.word_count = static_cast<int>(words.size());
        std::copy(words.begin(), words.end(), entry.modulus_words);
        if (source->ctx->ring_device_constants.empty() || source->ctx->ring_device_constants[0].device != target.device)
            return set_error("missing colocated CRT constants");
        const auto &constants = source->ctx->ring_device_constants[0];
        if (columns && !constants.garner_inverses) return set_error("missing setup-owned CRT Garner table");
        entry.garner = constants.garner_inverses;
        entry.garner_stride = source->ctx->moduli.size();
    }
    if (!columns) return 0;
    if (!target.device_descriptors || target.limb_count < target_count)
        return set_error("missing CRT output descriptors");
    if (out->rows > SIZE_MAX / out->cols || out->rows * out->cols > SIZE_MAX / static_cast<size_t>(out->ctx->N))
        return set_error("CRT output owner size overflow");
    GpuMatrixTransformWorkspaceBytes requirements{};
    status = gpu_matrix_query_crt_workspace_bytes(out->ctx, out->level, out->rows, out->cols,
        GPU_MATRIX_CRT_RECOMPOSE, 1, level_count, output_view != nullptr, &requirements);
    if (status != 0) return status;
    cudaStream_t stream = nullptr;
    status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    auto error = cudaSetDevice(target.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t level = 0; level < level_count; ++level) {
        status = matrix_wait_all_limb_streams(levels[level], target.device, stream, false, true);
        if (status != 0) return status;
    }
    status = matrix_wait_all_limb_streams(out, target.device, stream);
    if (status != 0) return status;
    if (out->cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(out->ctx->N))
        return set_error("CRT coefficient shape overflow");
    const size_t count = columns * static_cast<size_t>(out->ctx->N);
    const size_t blocks = count / 128 + (count % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("CRT recomposition exceeds CUDA grid capacity");

    MatrixTransformWorkspace workspace;
    const size_t output_offset = output_range.row_start * out->cols + output_range.column_start;
    if (output_view && level_count <= 2) {
        CrtLaunchMetadata launch{};
        std::copy_n(metadata.begin(), level_count, launch.levels);
        launch.output = output_metadata[0];
        crt_recompose_range_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            launch, level_count, count, static_cast<size_t>(out->ctx->N), output_offset);
        error = cudaGetLastError();
        status = error == cudaSuccess ? matrix_record_all_limb_writes(out, stream) : set_error(error);
    } else {
        status = workspace.acquire(out, target.device, stream, requirements);
        if (status != 0) return status;
        const size_t levels_bytes = level_count * sizeof(CrtLevelMetadata);
        auto *device_levels = reinterpret_cast<CrtLevelMetadata *>(workspace.base);
        auto *device_output = reinterpret_cast<CrtOutputMetadata *>(workspace.base + levels_bytes);
        if (output_view) {
            for (size_t start = 0; start < level_count && status == 0; start += 2) {
                CrtLaunchMetadata launch{};
                const size_t batch = std::min(size_t{2}, level_count - start);
                std::copy_n(metadata.begin() + start, batch, launch.levels);
                launch.output = output_metadata[0];
                crt_recompose_setup_kernel<<<1, 128, 0, stream>>>(device_levels, device_output, launch, start, batch);
                error = cudaGetLastError();
                if (error != cudaSuccess) status = set_error(error);
            }
        } else {
            std::memcpy(workspace.pinned, metadata.data(), levels_bytes);
            std::memcpy(workspace.pinned + levels_bytes, output_metadata.data(), sizeof(CrtOutputMetadata));
            status = workspace.upload();
        }
        if (status == 0) {
            crt_recompose_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
                device_levels, device_output, level_count, count, static_cast<size_t>(out->ctx->N), output_offset);
            error = cudaGetLastError();
            if (error != cudaSuccess) status = set_error(error);
        }
        if (status == 0) status = workspace.complete();
    }
    if (status == 0) {
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const auto &states = out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        for (size_t level = 0; level < level_count && status == 0; ++level)
            status = matrix_track_all_limb_consumers(levels[level], target.device, stream, completion, true, true);
    }
    if (status == 0 && out->format == GPU_POLY_FORMAT_EVAL)
        status = run_matrix_transform_u64<true>(out, &output_range);
    if (status != 0) {
        workspace.retire();
        const int retired = gpu_matrix_retire_submitted_work(out);
        if (retired != 0) return retired;
    }
    return status;
}

namespace
{
    // By-value launch parameters avoid pinned allocation and host metadata
    // transfers. CUDA copies these arguments before the launch returns.
    struct RnsLaunchMetadata
    {
        size_t source_count;
        size_t target_count;
        size_t digit_size;
        size_t group_count;
        size_t retained[kCrtMaxLimbs];
        uint64_t source_moduli[kCrtMaxLimbs];
        uint64_t target_moduli[kCrtMaxLimbs];
        uint64_t scales[kCrtMaxLimbs];
        uint64_t inverses[kCrtMaxLimbs];
        uint64_t plaintext_modulus;
    };
    static_assert(sizeof(RnsLaunchMetadata) + sizeof(void *) < 4096,
        "RNS setup arguments must fit the baseline CUDA kernel argument limit");

    struct RnsConversionMetadata
    {
        RnsLaunchMetadata plan;
        uint64_t weights[kCrtMaxLimbs * kCrtMaxLimbs];
    };

    __global__ void rns_setup_kernel(RnsConversionMetadata *metadata, RnsLaunchMetadata plan)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index == 0) metadata->plan = plan;
        if (index >= plan.source_count * plan.target_count) return;
        const size_t input = index / plan.target_count;
        const size_t target = index % plan.target_count;
        const bool down = plan.plaintext_modulus != 0;
        const uint64_t modulus = plan.target_moduli[target];
        uint64_t weight = down && plan.scales[input] == 0 ? 0 : 1;
        const size_t begin = down ? 0 : (input / plan.digit_size) * plan.digit_size;
        const size_t end = down ? plan.source_count : min(plan.source_count, begin + plan.digit_size);
        for (size_t limb = begin; limb < end && weight != 0; ++limb)
        {
            // Zero scales mark Q limbs retained by ModDown; P is their complement.
            if (limb != input && (!down || plan.scales[limb] != 0))
                weight = mul_mod_u64(weight, plan.source_moduli[limb] % modulus, modulus);
        }
        metadata->weights[input * kCrtMaxLimbs + target] = weight;
    }

    __device__ __forceinline__ void rns_convert_coefficient(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const RnsLaunchMetadata &plan, const uint64_t *weights, size_t weight_stride,
        size_t coefficient_count, size_t ring_dimension, bool mod_down,
        size_t columns, size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count * plan.target_count * plan.group_count) return;
        const size_t position = index % coefficient_count;
        const size_t limb = (index / coefficient_count) % plan.target_count;
        const size_t group = index / coefficient_count / plan.target_count;
        const size_t poly = position / ring_dimension;
        const size_t coefficient = position % ring_dimension;
        const size_t input_poly = input_offset + (poly / columns) * input_pitch + poly % columns;
        const uint64_t modulus = plan.target_moduli[limb];
        const size_t begin = mod_down ? 0 : group * plan.digit_size;
        const size_t end = mod_down ? plan.source_count :
            min(plan.source_count, begin + plan.digit_size);
        uint64_t sum = 0;
        for (size_t input_limb = begin; input_limb < end; ++input_limb)
        {
            const uint64_t weight = weights[input_limb * weight_stride + limb];
            if (weight == 0) continue;
            const auto input = source[input_limb];
            const uint64_t source_modulus = plan.source_moduli[input_limb];
            const uint64_t residue = mul_mod_u64(matrix_load_limb_u64(input.base, input_poly,
                coefficient, input.stride, input.width) % source_modulus,
                plan.scales[input_limb], source_modulus);
            const bool negative = residue > source_modulus / 2;
            const uint64_t magnitude = (negative ? source_modulus - residue : residue) % modulus;
            const uint64_t centered = negative && magnitude != 0 ? modulus - magnitude : magnitude;
            sum = add_mod_u64(sum, mul_mod_u64(centered, weight, modulus), modulus);
        }
        if (mod_down)
        {
            const auto input = source[plan.retained[limb]];
            const uint64_t residue = matrix_load_limb_u64(input.base, input_poly, coefficient,
                input.stride, input.width) % modulus;
            sum = mul_mod_u64(add_mod_u64(residue,
                mul_mod_u64(plan.plaintext_modulus % modulus, sum, modulus), modulus),
                plan.inverses[limb], modulus);
        }
        const auto output = target[limb];
        const size_t logical_output = group * (coefficient_count / ring_dimension) + poly;
        const size_t output_poly = output_offset + (logical_output / columns) * output_pitch + logical_output % columns;
        matrix_store_limb_u64(output.base, output_poly, coefficient,
            output.stride, output.width, sum);
    }
    struct RnsCompactMetadata
    {
        RnsLaunchMetadata plan;
        uint64_t weights[64];
    };
    static_assert(sizeof(RnsCompactMetadata) + 2 * sizeof(void *) + 7 * sizeof(size_t) + sizeof(bool) <= 4096,
                  "compact RNS launch exceeds portable CUDA parameter budget");

    __global__ void rns_compact_conversion_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        RnsCompactMetadata metadata, size_t count, size_t dimension, bool down,
        size_t columns, size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        rns_convert_coefficient(source, target, metadata.plan, metadata.weights,
                                metadata.plan.target_count, count, dimension, down, columns, input_offset, input_pitch, output_offset, output_pitch);
    }

    __global__ void rns_conversion_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *source,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *target,
        const RnsConversionMetadata *metadata, size_t count, size_t dimension, bool down,
        size_t columns, size_t input_offset, size_t input_pitch, size_t output_offset, size_t output_pitch)
    {
        rns_convert_coefficient(source, target, metadata->plan, metadata->weights,
                                kCrtMaxLimbs, count, dimension, down, columns, input_offset, input_pitch, output_offset, output_pitch);
    }

}

extern "C" int gpu_matrix_rns_conversion(
    GpuMatrix *out, const GpuMatrix *source, size_t digit_size,
    uint64_t plaintext_modulus, const uint64_t *scales,
    const uint64_t *inverses, const uint64_t *weights, const GpuMatrixBatchView *view)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_rns_conversion");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    const bool mod_down = plaintext_modulus != 0;
    if (plaintext_modulus == 1) return set_error("RNS ModDown plaintext modulus must be at least two");
    if (!source || out == source || !source->ctx ||
        out->ctx->execution != source->ctx->execution || out->ctx->N != source->ctx->N ||
        (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL) ||
        source->format != GPU_POLY_FORMAT_COEFF || out->ctx->N <= 0 ||
        out->level < 0 || source->level < 0 || !scales || !inverses ||
        digit_size == 0 ||
        source->shared_limb_buffers.size() != 1 || out->shared_limb_buffers.size() != 1)
        return set_error("invalid fused RNS conversion layout");
    const GpuMatrixRange input_range = view ? view->left : GpuMatrixRange{0, source->rows, 0, source->cols};
    const GpuMatrixRange output_range = view ? view->output : GpuMatrixRange{0, out->rows, 0, out->cols};
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    if (!valid(input_range, source) || !valid(output_range, out) ||
        input_range.column_end - input_range.column_start != output_range.column_end - output_range.column_start)
        return set_error("invalid RNS conversion rectangle");
    const size_t rows = input_range.row_end - input_range.row_start;
    const size_t columns = input_range.column_end - input_range.column_start;
    const size_t source_count = static_cast<size_t>(source->level) + 1;
    const size_t target_count = static_cast<size_t>(out->level) + 1;
    const size_t groups = mod_down ? 1 : source_count / digit_size + (source_count % digit_size != 0);
    if (source_count > kCrtMaxLimbs || target_count > kCrtMaxLimbs ||
        source_count != source->ctx->moduli.size() || target_count != out->ctx->moduli.size() ||
        rows > SIZE_MAX / groups || output_range.row_end - output_range.row_start != rows * groups)
        return set_error("invalid fused RNS conversion basis or shape");
    RnsLaunchMetadata metadata{};
    metadata.source_count = source_count;
    metadata.target_count = target_count;
    metadata.digit_size = digit_size;
    metadata.group_count = groups;
    metadata.plaintext_modulus = plaintext_modulus;
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        metadata.source_moduli[limb] = source->ctx->moduli[limb];
        metadata.scales[limb] = scales[limb];
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        metadata.target_moduli[limb] = out->ctx->moduli[limb];
        metadata.inverses[limb] = inverses[limb];
        if (mod_down)
        {
            size_t retained = 0;
            while (retained < source_count && source->ctx->moduli[retained] != out->ctx->moduli[limb]) ++retained;
            if (retained == source_count) return set_error("ModDown target must be a source subset");
            metadata.retained[limb] = retained;
        }
    }
    if (!mod_down) {
        for (size_t limb = 0; limb < source_count; ++limb)
            if (std::find(out->ctx->moduli.begin(), out->ctx->moduli.end(), source->ctx->moduli[limb]) == out->ctx->moduli.end())
                return set_error("ModUp target must contain the source basis");
    } else if (target_count >= source_count) return set_error("ModDown target must be strictly smaller");
    if (!rows || !columns) return 0;
    const auto &input = source->shared_limb_buffers[0];
    const auto &output = out->shared_limb_buffers[0];
    if (input.device != output.device || !input.device_descriptors || !output.device_descriptors ||
        input.limb_count < source_count || output.limb_count < target_count)
        return set_error("fused RNS conversion requires colocated descriptors");
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, out->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    cudaError_t error = cudaSetDevice(output.device);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < source_count; ++limb)
    {
        const dim3 id = source->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb) return set_error("unsupported source RNS limb placement");
    }
    for (size_t limb = 0; limb < target_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x != 0 || id.y != limb) return set_error("unsupported target RNS limb placement");
    }
    status = matrix_wait_all_limb_streams(source, output.device, stream, false, true);
    if (status != 0) return status;
    status = matrix_wait_all_limb_streams(out, output.device, stream);
    if (status != 0) return status;
    const size_t dimension = static_cast<size_t>(source->ctx->N);
    if (source->rows == 0 || source->cols == 0 ||
        source->rows > std::numeric_limits<size_t>::max() / source->cols ||
        source->rows * source->cols > SIZE_MAX / dimension ||
        out->rows > SIZE_MAX / out->cols || out->rows * out->cols > SIZE_MAX / dimension)
        return set_error("fused RNS conversion shape overflow");
    const size_t count = rows * columns * dimension;
    if (count > std::numeric_limits<size_t>::max() / target_count / groups)
        return set_error("fused RNS conversion grid overflow");
    const size_t total = count * target_count * groups;
    const size_t blocks = total / 128 + (total % 128 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("fused RNS conversion exceeds CUDA grid capacity");
    GpuMatrixTransformWorkspaceBytes requirements{};
    status = gpu_matrix_query_crt_workspace_bytes(
        out->ctx, out->level, out->rows, out->cols, GPU_MATRIX_CRT_RNS_CONVERSION,
        source_count, 1, view != nullptr, &requirements);
    if (status != 0) return status;
    MatrixTransformWorkspace workspace;
    status = workspace.acquire(out, output.device, stream, requirements);
    if (status != 0) return status;
    auto *device_metadata = reinterpret_cast<RnsConversionMetadata *>(workspace.base);
    if (source_count * target_count <= 64)
    {
        if (!weights) return set_error("missing compact RNS weights");
        RnsCompactMetadata compact{};
        compact.plan = metadata;
        std::copy_n(weights, source_count * target_count, compact.weights);
        rns_compact_conversion_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
            input.device_descriptors, output.device_descriptors, compact, count, dimension, mod_down, columns,
            input_range.row_start * source->cols + input_range.column_start, source->cols,
            output_range.row_start * out->cols + output_range.column_start, out->cols);
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
    }
    else
    {
        const size_t setup_count = source_count * target_count;
        const size_t setup_blocks = setup_count / 128 + (setup_count % 128 != 0);
        rns_setup_kernel<<<static_cast<int>(setup_blocks), 128, 0, stream>>>(device_metadata, metadata);
        error = cudaGetLastError();
        if (error != cudaSuccess) status = set_error(error);
        if (status == 0)
        {
            rns_conversion_kernel<<<static_cast<int>(blocks), 128, 0, stream>>>(
                input.device_descriptors, output.device_descriptors, device_metadata, count, dimension, mod_down, columns,
            input_range.row_start * source->cols + input_range.column_start, source->cols,
            output_range.row_start * out->cols + output_range.column_start, out->cols);
            error = cudaGetLastError();
            if (error != cudaSuccess) status = set_error(error);
        }
    }
    if (status == 0) status = workspace.complete();
    if (status == 0) {
        const dim3 first = out->ctx->limb_gpu_ids[0];
        const auto &states = out->exec_limb_states[first.x];
        const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
        status = matrix_track_all_limb_consumers(source, output.device, stream, completion, true, true);
    }
    if (status == 0 && out->format == GPU_POLY_FORMAT_EVAL)
        status = run_matrix_transform_u64<true>(out, &output_range);
    if (status != 0) {
        workspace.retire();
        const int retired = gpu_matrix_retire_submitted_work(out);
        if (retired != 0) return retired;
    }
    return status;
}

extern "C" int gpu_matrix_query_crt_workspace_bytes(
    const GpuContext *ctx, int level, size_t rows, size_t cols,
    GpuMatrixCrtOperation operation, size_t source_limb_count, size_t level_count,
    bool with_views, GpuMatrixTransformWorkspaceBytes *out)
{
    if (!out || level < 0 || !source_limb_count || source_limb_count > kCrtMaxLimbs ||
        !level_count || (operation != GPU_MATRIX_CRT_RECOMPOSE && level_count != 1) ||
        (with_views && operation != GPU_MATRIX_CRT_CONVERT_MODULUS &&
            operation != GPU_MATRIX_CRT_RNS_CONVERSION && operation != GPU_MATRIX_CRT_RECOMPOSE))
        return set_error("invalid CRT workspace query");
    GpuMatrixAllocationBytes allocation{};
    const int status = gpu_matrix_query_allocation_bytes(
        ctx, level, rows, cols, GPU_POLY_FORMAT_COEFF, &allocation);
    if (status != 0) return status;
    const size_t target_count = static_cast<size_t>(level) + 1;
    if (target_count > kCrtMaxLimbs) return set_error("unsupported CRT workspace limb count");
    size_t bytes = 0;
    bool staged = false;
    switch (operation)
    {
    case GPU_MATRIX_CRT_CONVERT_MODULUS:
        bytes = with_views ? 0 : sizeof(ModulusConversionMetadata);
        staged = !with_views;
        break;
    case GPU_MATRIX_CRT_CENTERED_REBASE:
        if (source_limb_count != 1) return set_error("centered rebase requires one source limb");
        break;
    case GPU_MATRIX_CRT_RECOMPOSE:
        if (level_count > (SIZE_MAX - sizeof(CrtOutputMetadata)) / sizeof(CrtLevelMetadata))
            return set_error("CRT recomposition workspace size overflow");
        bytes = with_views && level_count <= 2 ? 0 : level_count * sizeof(CrtLevelMetadata) + sizeof(CrtOutputMetadata);
        staged = !with_views;
        break;
    case GPU_MATRIX_CRT_RNS_CONVERSION:
        if (source_limb_count * target_count > 64) bytes = sizeof(RnsConversionMetadata);
        break;
    default:
        return set_error("unknown CRT workspace operation");
    }
    // All records use uint64_t/pointer/size_t alignment; array boundaries are
    // naturally aligned. This is an allocator request, not its pool granularity.
    static_assert(sizeof(CrtLevelMetadata) % alignof(CrtOutputMetadata) == 0,
        "CRT output metadata must follow an aligned level array");
    if (rows == 0 || cols == 0) return set_error("CRT conversion requires nonempty matrices");
    // A retained RNS range uses a separate GPU arena regardless of owner width.
    // Its setup kernel writes metadata; no pinned host buffer is reused.
    *out = {bytes, !with_views && bytes <= allocation.aux_workspace_bytes ? 0 : bytes,
            staged ? bytes : 0, alignof(uint64_t)};
    return 0;
}
