// Direct, allocation-free preimage retry operations. Included after the
// matrix arithmetic, trapdoor, serde, and compact-RHS kernels in Matrix.cu.
namespace
{
    struct RawPreimageLimbSet
    {
        MxxRawMatrixLimb limbs[kMaxRnsLimbs];
    };

    MxxGraphPatch raw_preimage_patch(
        uint32_t argument, uint32_t offset, uint32_t binding)
    {
        return {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            argument, offset, static_cast<uint32_t>(sizeof(void *)), binding, 0};
    }

    bool raw_preimage_packed(const MxxRawMatrixView *view)
    {
        if (!view || !view->rows || !view->columns || !view->degree ||
            !view->limbs || !view->limb_count || view->limb_count > kMaxRnsLimbs)
            return false;
        for (size_t index = 0; index < view->limb_count; ++index)
        {
            const auto &limb = view->limbs[index];
            if (!limb.address || (limb.word_bytes != 4 && limb.word_bytes != 8) ||
                limb.coefficient_stride_bytes != limb.word_bytes ||
                view->degree > UINT64_MAX / limb.word_bytes ||
                limb.column_stride_bytes < view->degree * limb.word_bytes ||
                view->columns > UINT64_MAX / limb.column_stride_bytes ||
                limb.row_stride_bytes != view->columns * limb.column_stride_bytes)
                return false;
        }
        return true;
    }

    bool raw_preimage_equal_basis(
        const MxxRawMatrixView *left, const MxxRawMatrixView *right)
    {
        if (left->limb_count != right->limb_count ||
            left->degree != right->degree ||
            left->physical_device != right->physical_device)
            return false;
        for (size_t limb = 0; limb < left->limb_count; ++limb)
            if (left->limbs[limb].crt_limb_index != right->limbs[limb].crt_limb_index ||
                left->limbs[limb].modulus != right->limbs[limb].modulus)
                return false;
        return true;
    }

    bool raw_preimage_coefficient_count(
        uint64_t rows, uint64_t columns, uint32_t degree, size_t *result)
    {
        if (!result || !rows || !columns || !degree ||
            rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / degree)
            return false;
        *result = static_cast<size_t>(rows * columns * degree);
        return true;
    }

    __global__ void raw_preimage_add_bottom_kernel(
        MxxRawMatrixLimb z, MxxRawMatrixLimb output,
        size_t top_rows, size_t bottom_rows, size_t columns,
        size_t degree, size_t count)
    {
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += step)
        {
            const size_t coefficient = index % degree;
            const size_t poly = index / degree;
            const size_t row = poly / columns;
            if (row >= bottom_rows) continue;
            const uint64_t result = add_mod_u64(
                raw_matrix_load(output, (top_rows + row) * columns + poly % columns,
                    coefficient, columns),
                raw_matrix_load(z, poly, coefficient, columns), output.modulus);
            raw_matrix_store(output,
                (top_rows + row) * columns + poly % columns,
                coefficient, columns, result);
        }
    }

    __global__ void raw_preimage_publish_kernel(
        uint8_t *destination, const uint8_t *staging,
        const MxxPreimageStatus *status, size_t byte_count,
        size_t columns, size_t storage_columns, size_t column_offset,
        size_t degree, size_t cell_width)
    {
        if (status->accepted != 1U ||
            status->error_code != MXX_PREIMAGE_SUCCESS) return;
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < byte_count; index += step)
        {
            const size_t cell = index / cell_width;
            const size_t byte = index % cell_width;
            const size_t coefficient = cell % degree;
            const size_t polynomial = cell / degree;
            const size_t row = polynomial / columns;
            const size_t column = polynomial % columns;
            const size_t destination_cell =
                ((row * storage_columns + column_offset + column) * degree + coefficient);
            destination[destination_cell * cell_width + byte] = staging[index];
        }
    }
}

struct GpuRawPreimageCutoffPlan
{
    GpuContext *ctx = nullptr;
    int32_t physical_device = -1;
    cudaStream_t stream = nullptr;
    cudaEvent_t upload_ready = nullptr;
    size_t limb_count = 0;
    int subset_count = 0;
    int words_per_coeff = 0;
    uint32_t magnitude_bytes = 0;
    uint64_t *moduli = nullptr;
    uint64_t *garner_inverses = nullptr;
    int *subset_indices = nullptr;
    uint64_t *modulus_words = nullptr;
    uint64_t *half_words = nullptr;
    uint64_t *bound_words = nullptr;
};

extern "C" int gpu_raw_preimage_cutoff_metadata_ranges(
    const GpuRawPreimageCutoffPlan *plan, uint64_t *addresses,
    size_t *bytes, size_t capacity, size_t *out_count)
{
    if (!plan || !addresses || !bytes || !out_count || capacity < 6)
        return set_error("invalid raw preimage metadata range query");
    const void *pointers[6] = {plan->moduli, plan->garner_inverses,
        plan->subset_indices, plan->modulus_words, plan->half_words,
        plan->bound_words};
    const size_t lengths[6] = {
        plan->limb_count * sizeof(uint64_t),
        plan->limb_count * plan->limb_count * sizeof(uint64_t),
        static_cast<size_t>(plan->subset_count) * sizeof(int),
        static_cast<size_t>(plan->words_per_coeff) * sizeof(uint64_t),
        static_cast<size_t>(plan->words_per_coeff) * sizeof(uint64_t),
        static_cast<size_t>(plan->words_per_coeff) * sizeof(uint64_t),
    };
    for (size_t index = 0; index < 6; ++index)
    {
        addresses[index] = reinterpret_cast<uint64_t>(pointers[index]);
        bytes[index] = lengths[index];
    }
    *out_count = 6;
    return 0;
}

extern "C" void gpu_raw_preimage_cutoff_destroy(GpuRawPreimageCutoffPlan *plan)
{
    if (!plan) return;
    cudaSetDevice(plan->physical_device);
    if (plan->moduli) cudaFreeAsync(plan->moduli, plan->stream);
    if (plan->garner_inverses) cudaFreeAsync(plan->garner_inverses, plan->stream);
    if (plan->subset_indices) cudaFreeAsync(plan->subset_indices, plan->stream);
    if (plan->modulus_words) cudaFreeAsync(plan->modulus_words, plan->stream);
    if (plan->half_words) cudaFreeAsync(plan->half_words, plan->stream);
    if (plan->bound_words) cudaFreeAsync(plan->bound_words, plan->stream);
    if (plan->upload_ready) cudaEventDestroy(plan->upload_ready);
    delete plan;
}

extern "C" int gpu_raw_preimage_cutoff_plan_wait(
    const GpuRawPreimageCutoffPlan *plan, void *consumer_stream)
{
    if (!plan || !plan->upload_ready || !consumer_stream)
        return set_error("invalid raw preimage cutoff upload wait");
    cudaError_t error = cudaSetDevice(plan->physical_device);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(
            reinterpret_cast<cudaStream_t>(consumer_stream), plan->upload_ready, 0);
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" int gpu_raw_preimage_cutoff_prepare(
    GpuContext *ctx, int32_t physical_device, void *stream_raw,
    const uint64_t *bound_words, size_t bound_word_count,
    uint32_t magnitude_bytes, GpuRawPreimageCutoffPlan **out_plan)
{
    if (!ctx || !stream_raw || !bound_words || !bound_word_count || !out_plan ||
        magnitude_bytes == 0 || magnitude_bytes > 64 || physical_device < 0)
        return set_error("invalid raw preimage cutoff preparation");
    *out_plan = nullptr;
    cudaError_t error = cudaSetDevice(physical_device);
    if (error != cudaSuccess) return set_error(error);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    error = cudaStreamIsCapturing(
        reinterpret_cast<cudaStream_t>(stream_raw), &capture);
    if (error != cudaSuccess) return set_error(error);
    if (capture != cudaStreamCaptureStatusNone)
        return set_error("raw preimage cutoff metadata must be prepared before Graph construction");
    const size_t limb_count = ctx->moduli.size();
    if (!limb_count || limb_count > kMaxRnsLimbs ||
        ctx->garner_inverse_table.size() != limb_count * limb_count)
        return set_error("invalid raw preimage cutoff CRT basis");
    std::vector<uint64_t> bound(bound_words, bound_words + bound_word_count);
    std::vector<uint64_t> doubled_bound;
    compact_double_words(bound.data(), bound.size(), &doubled_bound);
    std::vector<int> subset_indices;
    std::vector<uint64_t> subset_moduli;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        if (compact_compare_words(
                std::vector<uint64_t>{ctx->moduli[limb]}, doubled_bound) > 0)
        {
            subset_indices.push_back(static_cast<int>(limb));
            subset_moduli.push_back(ctx->moduli[limb]);
            break;
        }
    }
    std::vector<uint64_t> modulus_words;
    if (subset_indices.empty())
    {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            subset_indices.push_back(static_cast<int>(limb));
            subset_moduli.push_back(ctx->moduli[limb]);
            if (!serde_compute_modulus_words_le(subset_moduli, &modulus_words))
                return set_error("failed to compute raw preimage partial CRT modulus");
            if (compact_compare_words(modulus_words, doubled_bound) > 0) break;
        }
    }
    if (modulus_words.empty() &&
        !serde_compute_modulus_words_le(subset_moduli, &modulus_words))
        return set_error("failed to compute raw preimage anchor modulus");
    if (compact_compare_words(modulus_words, doubled_bound) <= 0)
        return set_error("raw preimage CRT modulus must exceed twice the cutoff");
    const size_t words_per_coeff = std::max(modulus_words.size(), bound.size());
    if (words_per_coeff > kMaxCoeffWords)
        return set_error("raw preimage cutoff word count exceeds supported maximum");
    for (size_t word = (magnitude_bytes + 7) / 8; word < bound.size(); ++word)
        if (bound[word] != 0)
            return set_error("raw preimage cutoff exceeds compact magnitude width");
    if (magnitude_bytes % 8 != 0 && bound.size() > magnitude_bytes / 8 &&
        (bound[magnitude_bytes / 8] >> (8 * (magnitude_bytes % 8))) != 0)
        return set_error("raw preimage cutoff exceeds compact magnitude width");
    std::vector<uint64_t> half_words = modulus_words;
    serde_shift_words_right_one_le(&half_words);
    modulus_words.resize(words_per_coeff, 0);
    half_words.resize(words_per_coeff, 0);
    bound.resize(words_per_coeff, 0);
    auto *plan = new GpuRawPreimageCutoffPlan();
    plan->ctx = ctx;
    plan->physical_device = physical_device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->limb_count = limb_count;
    plan->subset_count = static_cast<int>(subset_indices.size());
    plan->words_per_coeff = static_cast<int>(words_per_coeff);
    plan->magnitude_bytes = magnitude_bytes;
    error = cudaEventCreateWithFlags(&plan->upload_ready, cudaEventDisableTiming);
    if (error != cudaSuccess)
    {
        gpu_raw_preimage_cutoff_destroy(plan);
        return set_error(error);
    }
    std::vector<void *> pinned_uploads;
    auto upload = [&](auto **destination, const auto *source, size_t count) -> cudaError_t {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*destination)>>;
        const size_t bytes = count * sizeof(T);
        void *pinned = nullptr;
        cudaError_t result = cudaHostAlloc(&pinned, bytes, cudaHostAllocPortable);
        if (result != cudaSuccess) return result;
        std::memcpy(pinned, source, bytes);
        result = cudaMallocAsync(reinterpret_cast<void **>(destination),
            bytes, plan->stream);
        if (result == cudaSuccess)
            result = cudaMemcpyAsync(*destination, pinned, bytes,
                cudaMemcpyHostToDevice, plan->stream);
        pinned_uploads.push_back(pinned);
        return result;
    };
    error = upload(&plan->moduli, ctx->moduli.data(), limb_count);
    if (error == cudaSuccess)
        error = upload(&plan->garner_inverses,
            ctx->garner_inverse_table.data(), limb_count * limb_count);
    if (error == cudaSuccess)
        error = upload(&plan->subset_indices,
            subset_indices.data(), subset_indices.size());
    if (error == cudaSuccess)
        error = upload(&plan->modulus_words, modulus_words.data(), words_per_coeff);
    if (error == cudaSuccess)
        error = upload(&plan->half_words, half_words.data(), words_per_coeff);
    if (error == cudaSuccess)
        error = upload(&plan->bound_words, bound.data(), words_per_coeff);
    if (error == cudaSuccess)
        error = cudaEventRecord(plan->upload_ready, plan->stream);
    if (!pinned_uploads.empty() &&
        gpu_defer_pinned_frees(ctx, physical_device, plan->stream,
            pinned_uploads.data(), pinned_uploads.size()) != 0)
    {
        gpu_raw_preimage_cutoff_destroy(plan);
        return 1;
    }
    if (error != cudaSuccess)
    {
        gpu_raw_preimage_cutoff_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_preimage_add_correction(
    GpuContext *ctx, void *stream_raw, const MxxRawMatrixView *candidate,
    const MxxRawMatrixView *r, const MxxRawMatrixView *e,
    const MxxRawMatrixView *z, uint32_t candidate_binding_base,
    uint32_t r_binding_base, uint32_t e_binding_base,
    uint32_t z_binding_base)
{
    if (!ctx || !stream_raw || !raw_preimage_packed(candidate) ||
        !raw_preimage_packed(r) || !raw_preimage_packed(e) ||
        !raw_preimage_packed(z) ||
        validate_raw_view(ctx, candidate, stream_raw) != 0 ||
        validate_raw_view(ctx, r, stream_raw) != 0 ||
        validate_raw_view(ctx, e, stream_raw) != 0 ||
        validate_raw_view(ctx, z, stream_raw) != 0 ||
        !raw_preimage_equal_basis(candidate, r) ||
        !raw_preimage_equal_basis(candidate, e) ||
        !raw_preimage_equal_basis(candidate, z) ||
        candidate->row_origin != 0 || r->row_origin != 0 ||
        e->row_origin != 0 || z->row_origin != 0 ||
        r->column_origin != 0 || e->column_origin != 0 ||
        candidate->column_origin != z->column_origin ||
        r->rows != e->rows || r->columns != e->columns ||
        z->rows != r->columns ||
        r->rows > (UINT64_MAX - z->rows) / 2 ||
        candidate->rows != 2 * r->rows + z->rows ||
        candidate->columns != z->columns ||
        candidate_binding_base > UINT32_MAX - candidate->limb_count ||
        r_binding_base > UINT32_MAX - r->limb_count ||
        e_binding_base > UINT32_MAX - e->limb_count ||
        z_binding_base > UINT32_MAX - z->limb_count)
        return set_error("invalid packed raw preimage correction views");
    if (cudaSetDevice(candidate->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    size_t bottom_count = 0;
    if (!raw_preimage_coefficient_count(z->rows, z->columns, z->degree, &bottom_count))
        return set_error("raw preimage correction coefficient count overflows");
    const uint64_t top_rows = 2 * r->rows;
    const uint64_t top_grid_x = (candidate->columns - 1) / kPreimageTileN + 1;
    const uint64_t top_grid_y = (top_rows - 1) / kPreimageTileM + 1;
    const uint64_t bottom_grid = std::min<size_t>((bottom_count - 1) / 256 + 1, 65535);
    if (top_grid_x > UINT32_MAX || top_grid_y > UINT32_MAX ||
        !top_grid_y)
        return set_error("raw preimage correction tile exceeds native launch limits");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t limb = 0; limb < candidate->limb_count; ++limb)
    {
        const auto &out_limb = candidate->limbs[limb];
        const auto &r_limb = r->limbs[limb];
        const auto &e_limb = e->limbs[limb];
        const auto &z_limb = z->limbs[limb];
        const MxxGraphPatch top_patches[] = {
            raw_preimage_patch(0, 0, r_binding_base + limb),
            raw_preimage_patch(1, 0, e_binding_base + limb),
            raw_preimage_patch(2, 0, z_binding_base + limb),
            raw_preimage_patch(3, 0, candidate_binding_base + limb),
        };
        const dim3 top_grid(
            static_cast<unsigned>(top_grid_x), static_cast<unsigned>(top_grid_y),
            std::min<uint32_t>(candidate->degree, 65535));
        int status = mxx_gpu_launch_kernel(ctx, stream,
            matrix_preimage_add_correction_top_kernel,
            top_grid, dim3(kPreimageTileN, kPreimageTileM), 0,
            top_patches, std::size(top_patches),
            reinterpret_cast<const uint8_t *>(r_limb.address),
            reinterpret_cast<const uint8_t *>(e_limb.address),
            reinterpret_cast<const uint8_t *>(z_limb.address),
            reinterpret_cast<uint8_t *>(out_limb.address),
            static_cast<size_t>(r->rows), static_cast<size_t>(r->columns),
            static_cast<size_t>(z->columns), static_cast<size_t>(candidate->columns),
            static_cast<size_t>(candidate->degree),
            static_cast<size_t>(r_limb.column_stride_bytes),
            static_cast<size_t>(e_limb.column_stride_bytes),
            static_cast<size_t>(z_limb.column_stride_bytes),
            static_cast<size_t>(out_limb.column_stride_bytes),
            static_cast<uint8_t>(r_limb.word_bytes),
            static_cast<uint8_t>(e_limb.word_bytes),
            static_cast<uint8_t>(z_limb.word_bytes),
            static_cast<uint8_t>(out_limb.word_bytes), out_limb.modulus);
        if (status != 0) return status;
        const MxxGraphPatch bottom_patches[] = {
            raw_preimage_patch(0, offsetof(MxxRawMatrixLimb, address), z_binding_base + limb),
            raw_preimage_patch(1, offsetof(MxxRawMatrixLimb, address),
                candidate_binding_base + limb),
        };
        status = mxx_gpu_launch_kernel(ctx, stream,
            raw_preimage_add_bottom_kernel,
            dim3(static_cast<unsigned>(bottom_grid)), dim3(256), 0,
            bottom_patches, std::size(bottom_patches),
            z_limb, out_limb,
            static_cast<size_t>(top_rows), static_cast<size_t>(z->rows),
            static_cast<size_t>(candidate->columns),
            static_cast<size_t>(candidate->degree), bottom_count);
        if (status != 0) return status;
    }
    return 0;
}

namespace
{
    __global__ void raw_preimage_status_init_attempt_kernel(
        MxxPreimageStatus *status, const uint64_t *attempt_word)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        const uint64_t attempt = *attempt_word;
        if (attempt >= UINT32_MAX)
        {
            status->attempts = 0;
            status->accepted = 0;
            status->error_code = MXX_PREIMAGE_INVALID_ATTEMPT;
            status->reserved = 1;
            return;
        }
        if (attempt == 0) status->reserved = 0;
        if (status->reserved == 0)
        {
            status->attempts = static_cast<uint32_t>(attempt) + 1U;
            status->accepted = 1U;
        }
        status->error_code = MXX_PREIMAGE_SUCCESS;
    }

    __global__ void raw_preimage_check_pack_kernel(
        RawPreimageLimbSet source,
        const uint64_t *moduli, const uint64_t *garner_inverses,
        int inverse_stride, const int *subset_indices,
        int subset_count, int limb_count, size_t coefficient_count,
        size_t degree, size_t columns, int words_per_coeff,
        const uint64_t *subset_modulus_words,
        const uint64_t *subset_half_words,
        const uint64_t *bound_words, size_t magnitude_bytes,
        MxxPreimageStatus *status, uint8_t *staging)
    {
        if (status->reserved != 0U) return;
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < coefficient_count; index += step)
        {
            const size_t poly = index / degree;
            const size_t coefficient = index % degree;
            uint64_t mixed_digits[kMaxRnsLimbs];
            uint64_t magnitude[kMaxCoeffWords];
            for (int i = 0; i < subset_count; ++i)
            {
                const int limb = subset_indices[i];
                mixed_digits[i] = raw_matrix_load(
                    source.limbs[limb], poly, coefficient, columns) % moduli[limb];
            }
            for (int i = 1; i < subset_count; ++i)
            {
                const int limb_i = subset_indices[i];
                const uint64_t qi = moduli[limb_i];
                uint64_t digit = mixed_digits[i];
                for (int j = 0; j < i; ++j)
                {
                    const int limb_j = subset_indices[j];
                    const uint64_t previous = mixed_digits[j] % qi;
                    const uint64_t difference = digit >= previous ? digit - previous :
                        static_cast<uint64_t>(
                            static_cast<unsigned __int128>(digit) + qi - previous);
                    digit = serde_mul_mod_u64_device(
                        difference,
                        garner_inverses[static_cast<size_t>(limb_j) * inverse_stride + limb_i],
                        qi);
                }
                mixed_digits[i] = digit;
            }
            for (int word = 0; word < words_per_coeff; ++word) magnitude[word] = 0;
            for (int i = subset_count; i-- > 0;)
            {
                const uint64_t modulus = moduli[subset_indices[i]];
                uint64_t carry = mixed_digits[i];
                for (int word = 0; word < words_per_coeff; ++word)
                {
                    const unsigned __int128 term =
                        static_cast<unsigned __int128>(magnitude[word]) * modulus + carry;
                    magnitude[word] = static_cast<uint64_t>(term);
                    carry = static_cast<uint64_t>(term >> 64);
                }
            }
            const bool negative = serde_compare_words_desc_device(
                magnitude, subset_half_words, words_per_coeff) > 0;
            if (negative)
            {
                uint64_t borrow = 0;
                for (int word = 0; word < words_per_coeff; ++word)
                {
                    const unsigned __int128 minuend = subset_modulus_words[word];
                    const unsigned __int128 subtrahend =
                        static_cast<unsigned __int128>(magnitude[word]) + borrow;
                    if (minuend >= subtrahend)
                    {
                        magnitude[word] = static_cast<uint64_t>(minuend - subtrahend);
                        borrow = 0;
                    }
                    else
                    {
                        magnitude[word] = static_cast<uint64_t>(
                            minuend + (static_cast<unsigned __int128>(1) << 64) - subtrahend);
                        borrow = 1;
                    }
                }
            }
            bool valid = serde_compare_words_desc_device(
                magnitude, bound_words, words_per_coeff) <= 0;
            for (int limb = 0; limb < limb_count && valid; ++limb)
            {
                const uint64_t modulus = moduli[limb];
                uint64_t expected = compact_words_mod(
                    magnitude, words_per_coeff, modulus);
                if (negative && expected != 0) expected = modulus - expected;
                const uint64_t actual = raw_matrix_load(
                    source.limbs[limb], poly, coefficient, columns) % modulus;
                valid = actual == expected;
            }
            if (!valid)
            {
                atomicExch(&status->accepted, 0U);
                continue;
            }
            const size_t width = magnitude_bytes + 1;
            uint8_t *destination = staging + index * width;
            bool zero = true;
            for (int word = 0; word < words_per_coeff; ++word)
                zero = zero && magnitude[word] == 0;
            destination[0] = zero ? 0 : (negative ? 2 : 1);
            for (size_t byte = 0; byte < magnitude_bytes; ++byte)
            {
                const size_t word = byte / sizeof(uint64_t);
                destination[1 + byte] = word < static_cast<size_t>(words_per_coeff) ?
                    static_cast<uint8_t>(magnitude[word] >>
                        (8 * (byte % sizeof(uint64_t)))) : 0;
            }
        }
    }

}

extern "C" int gpu_raw_preimage_hard_cutoff(
    GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *candidate,
    const GpuRawPreimageCutoffPlan *plan,
    void *staging, size_t staging_bytes,
    const uint64_t *attempt,
    MxxPreimageStatus *status,
    uint32_t candidate_binding_base, uint32_t staging_binding,
    uint32_t control_binding, uint32_t status_binding)
{
    if (!ctx || !stream_raw || !plan || plan->ctx != ctx ||
        !raw_preimage_packed(candidate) ||
        validate_raw_view(ctx, candidate, stream_raw) != 0 ||
        candidate->row_origin != 0 ||
        candidate->physical_device != plan->physical_device ||
        candidate->limb_count != plan->limb_count ||
        !staging || !attempt || !status ||
        candidate_binding_base > UINT32_MAX - candidate->limb_count)
        return set_error("invalid raw preimage cutoff inputs");
    for (size_t limb = 0; limb < candidate->limb_count; ++limb)
        if (candidate->limbs[limb].crt_limb_index != limb ||
            candidate->limbs[limb].modulus != ctx->moduli[limb])
            return set_error("raw preimage cutoff requires the full ordered CRT basis");
    size_t count = 0;
    if (!raw_preimage_coefficient_count(
            candidate->rows, candidate->columns, candidate->degree, &count))
        return set_error("raw preimage cutoff coefficient count overflows");
    if (count > SIZE_MAX / (1 + plan->magnitude_bytes) ||
        staging_bytes != count * (1 + plan->magnitude_bytes))
        return set_error("raw preimage cutoff staging length disagrees with candidate");
    if (cudaSetDevice(plan->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const MxxGraphPatch init_patches[] = {
        raw_preimage_patch(0, 0, status_binding),
        raw_preimage_patch(1, 0, control_binding),
    };
    int result = mxx_gpu_launch_kernel(ctx, stream,
        raw_preimage_status_init_attempt_kernel, dim3(1), dim3(1), 0,
        init_patches, std::size(init_patches), status, attempt);
    if (result != 0) return result;
    RawPreimageLimbSet source{};
    MxxGraphPatch patches[kMaxRnsLimbs + 2];
    size_t patch_count = 0;
    for (size_t limb = 0; limb < candidate->limb_count; ++limb)
    {
        source.limbs[limb] = candidate->limbs[limb];
        patches[patch_count++] = raw_preimage_patch(0,
            static_cast<uint32_t>(offsetof(RawPreimageLimbSet, limbs) +
                limb * sizeof(MxxRawMatrixLimb) + offsetof(MxxRawMatrixLimb, address)),
            candidate_binding_base + static_cast<uint32_t>(limb));
    }
    patches[patch_count++] = raw_preimage_patch(15, 0, status_binding);
    patches[patch_count++] = raw_preimage_patch(16, 0, staging_binding);
    const uint32_t grid = static_cast<uint32_t>(
        std::min<size_t>((count + kSmallThreads - 1) / kSmallThreads, 65535));
    result = mxx_gpu_launch_kernel(ctx, stream,
        raw_preimage_check_pack_kernel, dim3(grid), dim3(kSmallThreads), 0,
        patches, patch_count,
        source, plan->moduli, plan->garner_inverses,
        static_cast<int>(plan->limb_count), plan->subset_indices,
        plan->subset_count, static_cast<int>(plan->limb_count), count,
        static_cast<size_t>(candidate->degree),
        static_cast<size_t>(candidate->columns), plan->words_per_coeff,
        plan->modulus_words, plan->half_words, plan->bound_words,
        static_cast<size_t>(plan->magnitude_bytes), status,
        static_cast<uint8_t *>(staging));
    if (result != 0) return result;
    const MxxGraphPatch latch_patch = raw_preimage_patch(0, 0, status_binding);
    return mxx_gpu_launch_kernel(ctx, stream,
        preimage_status_latch_accept_kernel, dim3(1), dim3(1), 0,
        &latch_patch, 1, status);
}

extern "C" int gpu_raw_preimage_publish_accepted(
    GpuContext *ctx, void *stream_raw,
    const MxxRawSmallMatrixView *destination,
    const GpuRawPreimageCutoffPlan *plan,
    const void *staging, size_t staging_bytes,
    const MxxPreimageStatus *status,
    uint32_t destination_binding, uint32_t staging_binding,
    uint32_t status_binding)
{
    if (!ctx || !stream_raw || !destination || !plan || plan->ctx != ctx ||
        destination->physical_device != plan->physical_device ||
        !destination->payload_address || !destination->rows ||
        !destination->columns || !destination->degree ||
        destination->storage_columns < destination->columns ||
        destination->column_offset >
            destination->storage_columns - destination->columns ||
        destination->magnitude_bytes != plan->magnitude_bytes ||
        destination->bound_domain != 0 || destination->crt_depth != 1 ||
        !staging || !status)
        return set_error("invalid raw preimage accepted-output view");
    size_t count = 0;
    if (!raw_preimage_coefficient_count(
            destination->rows, destination->columns, destination->degree, &count))
        return set_error("raw preimage publication coefficient count overflows");
    if (count > SIZE_MAX / (1 + plan->magnitude_bytes) ||
        staging_bytes != count * (1 + plan->magnitude_bytes))
        return set_error("raw preimage publication staging length disagrees with destination");
    const size_t total_bytes = staging_bytes;
    const size_t cell_width = 1 + plan->magnitude_bytes;
    if (destination->rows > SIZE_MAX / destination->storage_columns ||
        destination->rows * destination->storage_columns > SIZE_MAX / destination->degree ||
        destination->rows * destination->storage_columns * destination->degree >
            SIZE_MAX / cell_width)
        return set_error("raw preimage publication owner extent overflows");
    const uint64_t grid = std::min<size_t>(
        (total_bytes - 1) / kSmallThreads + 1, 65535);
    if (cudaSetDevice(plan->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const MxxGraphPatch patches[] = {
        raw_preimage_patch(0, 0, destination_binding),
        raw_preimage_patch(1, 0, staging_binding),
        raw_preimage_patch(2, 0, status_binding),
    };
    return mxx_gpu_launch_kernel(ctx,
        reinterpret_cast<cudaStream_t>(stream_raw),
        raw_preimage_publish_kernel,
        dim3(static_cast<unsigned>(grid)), dim3(kSmallThreads), 0,
        patches, std::size(patches),
        reinterpret_cast<uint8_t *>(destination->payload_address),
        static_cast<const uint8_t *>(staging), status, total_bytes,
        static_cast<size_t>(destination->columns),
        static_cast<size_t>(destination->storage_columns),
        static_cast<size_t>(destination->column_offset),
        static_cast<size_t>(destination->degree), cell_width);
}
