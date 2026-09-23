// Exact CPU HashSample transcript and full-CRT rejection, emitted as direct
// CUDA Graph nodes. Static metadata and replayable tag operands belong to the
// plan; no host read is needed when a tag expression depends on device state.
#include <climits>
#include <new>

struct MxxRawHashTagSegment
{
    uint32_t kind; // 0 exact static bytes, 1 integer, 2 decimal, 3 u64-le
    uint32_t operand_index;
    uint64_t static_offset;
    uint64_t static_length;
};

struct MxxRawHashTagOperand
{
    uint64_t address;
    int32_t encoding; // 0 SignedI64, 1 CanonicalU64, 2 + magnitude words
    uint32_t reserved;
};

struct GpuRawHashPlan
{
    GpuContext *context = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    void *allocation = nullptr;
    size_t allocation_bytes = 0;
    void *pinned_static = nullptr;
    MxxRawHashTagOperand *pinned_operands = nullptr;
    cudaEvent_t static_ready = nullptr;
    uint64_t *device_q = nullptr;
    uint64_t *device_moduli = nullptr;
    MxxRawHashTagSegment *device_segments = nullptr;
    uint8_t *device_static = nullptr;
    MxxRawHashTagOperand *device_operands = nullptr;
    uint64_t *device_tag_length = nullptr;
    uint8_t *device_tag = nullptr;
    uint64_t *device_decimal_scratch = nullptr;
    MxxRawMatrixLimb *device_descriptors = nullptr;
    std::vector<uint64_t> moduli;
    std::vector<int32_t> operand_encodings;
    size_t q_word_count = 0;
    size_t q_bits = 0;
    size_t segment_count = 0;
    size_t static_bytes = 0;
    size_t max_tag_bytes = 0;
    size_t decimal_words = 0;
};

namespace
{
    bool hash_align_add(size_t &cursor, size_t alignment, size_t bytes, size_t &offset)
    {
        if (!alignment || cursor > SIZE_MAX - (alignment - 1)) return false;
        const size_t aligned = (cursor + alignment - 1) & ~(alignment - 1);
        if (bytes > SIZE_MAX - aligned) return false;
        offset = aligned;
        cursor = aligned + bytes;
        return true;
    }

    bool hash_product_words(const uint64_t *moduli, size_t count,
        std::vector<uint64_t> &words)
    {
        words.assign(1, 1);
        for (size_t limb = 0; limb < count; ++limb)
        {
            const uint64_t modulus = moduli[limb];
            if (modulus <= 1 || !(modulus & 1) || modulus >= (uint64_t(1) << 60))
                return false;
            for (size_t earlier = 0; earlier < limb; ++earlier)
                if (moduli[earlier] == modulus) return false;
            uint64_t carry = 0;
            for (uint64_t &word : words)
            {
                const unsigned __int128 product =
                    static_cast<unsigned __int128>(word) * modulus + carry;
                word = static_cast<uint64_t>(product);
                carry = static_cast<uint64_t>(product >> 64);
            }
            if (carry) words.push_back(carry);
        }
        return true;
    }

    __device__ __forceinline__ void hash_absorb_byte(uint64_t state[25],
        size_t &position, uint8_t value)
    {
        state[position / 8] ^= uint64_t(value) << (8 * (position % 8));
        if (++position == 136)
        {
            preimage_seed_keccak_f(state);
            position = 0;
        }
    }

    __device__ __forceinline__ void hash_absorb_u64(uint64_t state[25],
        size_t &position, uint64_t value)
    {
        for (int byte = 0; byte < 8; ++byte)
            hash_absorb_byte(state, position, static_cast<uint8_t>(value >> (8 * byte)));
    }

    __device__ __forceinline__ void hash_digest(
        const uint8_t *key, const uint8_t *tag, uint64_t tag_length,
        uint64_t row, uint64_t column, uint64_t coefficient,
        uint64_t attempt, uint64_t block, uint8_t digest[32])
    {
        uint64_t state[25]{};
        size_t position = 0;
        for (size_t byte = 0; byte < 32; ++byte)
            hash_absorb_byte(state, position, key[byte]);
        for (uint64_t byte = 0; byte < tag_length; ++byte)
            hash_absorb_byte(state, position, tag[byte]);
        hash_absorb_u64(state, position, row);
        hash_absorb_u64(state, position, column);
        hash_absorb_u64(state, position, coefficient);
        hash_absorb_u64(state, position, attempt);
        hash_absorb_u64(state, position, block);
        state[position / 8] ^= uint64_t(1) << (8 * (position % 8));
        state[135 / 8] ^= uint64_t(0x80) << (8 * (135 % 8));
        preimage_seed_keccak_f(state);
        for (size_t byte = 0; byte < 32; ++byte)
            digest[byte] = static_cast<uint8_t>(
                state[byte / 8] >> (8 * (byte % 8)));
    }

    __device__ __forceinline__ void hash_write_be64(
        uint8_t *destination, uint64_t value)
    {
        for (int byte = 0; byte < 8; ++byte)
            destination[byte] = static_cast<uint8_t>(value >> (8 * (7 - byte)));
    }

    __device__ bool hash_operand_magnitude(
        const MxxRawHashTagOperand &operand, uint64_t *scratch,
        size_t scratch_words, size_t &word_count, bool &negative)
    {
        const auto *source = reinterpret_cast<const uint64_t *>(operand.address);
        if (!source || !scratch_words) return false;
        negative = false;
        if (operand.encoding == 0)
        {
            const int64_t value = static_cast<int64_t>(source[0]);
            negative = value < 0;
            scratch[0] = negative ? uint64_t(-(value + 1)) + 1 : uint64_t(value);
            word_count = scratch[0] ? 1 : 0;
            return true;
        }
        if (operand.encoding == 1)
        {
            scratch[0] = source[0];
            word_count = scratch[0] ? 1 : 0;
            return true;
        }
        if (operand.encoding <= 2) return false;
        const size_t count = static_cast<size_t>(operand.encoding - 2);
        if (count > scratch_words || source[0] > 1) return false;
        negative = source[0] == 1;
        for (size_t word = 0; word < count; ++word)
            scratch[word] = source[word + 1];
        word_count = count;
        while (word_count && scratch[word_count - 1] == 0) --word_count;
        if (!word_count) negative = false;
        return true;
    }

    __global__ void raw_hash_tag_build_kernel(
        const MxxRawHashTagSegment *segments, size_t segment_count,
        const uint8_t *static_bytes, const MxxRawHashTagOperand *operands,
        size_t operand_count, uint64_t *tag_length, uint8_t *tag,
        size_t tag_capacity, uint64_t *scratch, size_t scratch_words,
        uint32_t *status)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        *tag_length = 0;
        size_t cursor = 0;
        for (size_t part = 0; part < segment_count; ++part)
        {
            const auto segment = segments[part];
            if (segment.kind == 0)
            {
                if (segment.static_length > tag_capacity - cursor) goto invalid;
                for (uint64_t byte = 0; byte < segment.static_length; ++byte)
                    tag[cursor++] = static_bytes[segment.static_offset + byte];
                continue;
            }
            if (segment.operand_index >= operand_count) goto invalid;
            const auto operand = operands[segment.operand_index];
            size_t words = 0;
            bool negative = false;
            if (!hash_operand_magnitude(operand, scratch, scratch_words, words, negative))
                goto invalid;
            if (segment.kind == 3)
            {
                if (negative || words > 1 || tag_capacity - cursor < 9) goto invalid;
                tag[cursor++] = 3;
                const uint64_t value = words ? scratch[0] : 0;
                for (int byte = 0; byte < 8; ++byte)
                    tag[cursor++] = static_cast<uint8_t>(value >> (8 * byte));
                continue;
            }
            if (segment.kind == 1)
            {
                const uint64_t highest = words ? scratch[words - 1] : 0;
                size_t high_bytes = 0;
                for (uint64_t remaining = highest; remaining; remaining >>= 8) ++high_bytes;
                const size_t magnitude_bytes = words ? (words - 1) * 8 + high_bytes : 0;
                if (magnitude_bytes > tag_capacity - cursor ||
                    tag_capacity - cursor - magnitude_bytes < 10) goto invalid;
                tag[cursor++] = 1;
                tag[cursor++] = negative ? 1 : 0;
                hash_write_be64(tag + cursor, magnitude_bytes);
                cursor += 8;
                for (size_t byte = magnitude_bytes; byte-- > 0;)
                    tag[cursor++] = static_cast<uint8_t>(
                        scratch[byte / 8] >> (8 * (byte % 8)));
                continue;
            }
            if (segment.kind != 2 || tag_capacity - cursor < 10) goto invalid;
            tag[cursor++] = 2;
            const size_t length_position = cursor;
            cursor += 8;
            const size_t text_start = cursor;
            if (negative)
            {
                if (cursor == tag_capacity) goto invalid;
                tag[cursor++] = '-';
            }
            if (!words)
            {
                if (cursor == tag_capacity) goto invalid;
                tag[cursor++] = '0';
            }
            else
            {
                const size_t digits_start = cursor;
                while (words)
                {
                    unsigned __int128 remainder = 0;
                    for (size_t word = words; word-- > 0;)
                    {
                        const unsigned __int128 numerator =
                            (remainder << 64) | scratch[word];
                        scratch[word] = static_cast<uint64_t>(numerator / 10);
                        remainder = numerator % 10;
                    }
                    if (cursor == tag_capacity) goto invalid;
                    tag[cursor++] = static_cast<uint8_t>('0' + remainder);
                    while (words && !scratch[words - 1]) --words;
                }
                for (size_t left = digits_start, right = cursor - 1; left < right; ++left, --right)
                {
                    const uint8_t swap = tag[left];
                    tag[left] = tag[right];
                    tag[right] = swap;
                }
            }
            hash_write_be64(tag + length_position, cursor - text_start);
        }
        *tag_length = cursor;
        return;
    invalid:
        atomicCAS(status, 0U, 2U);
    }

    __global__ void raw_hash_sample_kernel(
        const uint8_t *key, const uint64_t *tag_length, const uint8_t *tag,
        size_t tag_capacity, const uint64_t *q_words, size_t q_word_count,
        size_t q_bits, const uint64_t *moduli, size_t limb_count,
        const MxxRawMatrixLimb *destinations,
        uint64_t row_origin, uint64_t column_origin,
        uint64_t rows, uint64_t columns, uint32_t degree, uint32_t *status)
    {
        if (*status != 0 || *tag_length > tag_capacity) return;
        const uint64_t total = rows * columns * degree;
        const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
        for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < total; index += stride)
        {
            const uint64_t poly = index / degree;
            const uint64_t coefficient = index % degree;
            const uint64_t row = row_origin + poly / columns;
            const uint64_t column = column_origin + poly % columns;
            uint64_t candidate[64]{};
            uint64_t attempt = 0;
            const size_t coefficient_bytes = (q_bits + 7) / 8;
            for (;; ++attempt)
            {
                for (size_t word = 0; word < q_word_count; ++word) candidate[word] = 0;
                for (size_t block = 0; block < (coefficient_bytes + 31) / 32; ++block)
                {
                    uint8_t digest[32];
                    hash_digest(key, tag, *tag_length, row, column, coefficient,
                        attempt, block, digest);
                    for (size_t byte = 0; byte < 32 && block * 32 + byte < coefficient_bytes; ++byte)
                    {
                        const size_t output = block * 32 + byte;
                        uint8_t value = digest[byte];
                        if (output + 1 == coefficient_bytes && (q_bits & 7))
                            value &= static_cast<uint8_t>((1U << (q_bits & 7)) - 1);
                        candidate[output / 8] |= uint64_t(value) << (8 * (output % 8));
                    }
                }
                bool less = false;
                bool greater = false;
                for (size_t word = q_word_count; word-- > 0;)
                {
                    if (candidate[word] < q_words[word]) { less = true; break; }
                    if (candidate[word] > q_words[word]) { greater = true; break; }
                }
                if (less && !greater) break;
            }
            for (size_t limb = 0; limb < limb_count; ++limb)
            {
                const uint64_t modulus = moduli[limb];
                const uint64_t radix = static_cast<uint64_t>(
                    (static_cast<unsigned __int128>(1) << 64) % modulus);
                uint64_t residue = 0;
                for (size_t word = q_word_count; word-- > 0;)
                    residue = static_cast<uint64_t>(
                        (static_cast<unsigned __int128>(residue) * radix +
                            candidate[word]) % modulus);
                raw_matrix_store(destinations[limb], poly, coefficient,
                    columns, residue);
            }
        }
    }

    __global__ void raw_hash_descriptor_kernel(MxxRawMatrixLimb limb,
        MxxRawMatrixLimb *output)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *output = limb;
        }
    }
}

extern "C" void gpu_raw_hash_plan_destroy(GpuRawHashPlan *plan)
{
    if (!plan) return;
    if (plan->allocation)
    {
        mxx_set_device(plan->device);
        cudaFreeAsync(plan->allocation, plan->stream);
    }
    void *pinned[2] = {plan->pinned_static, plan->pinned_operands};
    if (plan->context)
    {
        void *live[2]{};
        size_t count = 0;
        for (void *item : pinned) if (item) live[count++] = item;
        if (count) gpu_defer_pinned_frees(plan->context, plan->device,
            plan->stream, live, count);
    }
    if (plan->static_ready) cudaEventDestroy(plan->static_ready);
    delete plan;
}

extern "C" int gpu_raw_hash_plan_create(
    GpuContext *ctx, int32_t device, void *stream_raw,
    const uint64_t *moduli, size_t limb_count,
    const MxxRawHashTagSegment *segments, size_t segment_count,
    const uint8_t *static_bytes, size_t static_byte_count,
    const int32_t *operand_encodings, size_t operand_count,
    GpuRawHashPlan **out_plan)
{
    if (!ctx || !stream_raw || !out_plan || !moduli || !limb_count ||
        limb_count > 64 || (segment_count && !segments) ||
        (static_byte_count && !static_bytes) ||
        (operand_count && !operand_encodings) ||
        ctx->moduli.size() < limb_count)
        return set_error("invalid raw hash sample plan");
    *out_plan = nullptr;
    std::vector<uint64_t> q_words;
    if (!hash_product_words(moduli, limb_count, q_words) || q_words.size() > 64)
        return set_error("invalid raw hash CRT product");
    for (size_t limb = 0; limb < limb_count; ++limb)
        if (ctx->moduli[limb] != moduli[limb])
            return set_error("raw hash plan ordered CRT basis mismatch");
    size_t max_tag_bytes = 0;
    size_t max_words = 1;
    for (size_t part = 0; part < segment_count; ++part)
    {
        const auto segment = segments[part];
        size_t extra = 0;
        if (segment.kind == 0)
        {
            if (segment.static_offset > static_byte_count ||
                segment.static_length > static_byte_count - segment.static_offset)
                return set_error("raw hash static tag segment out of range");
            extra = static_cast<size_t>(segment.static_length);
        }
        else
        {
            if (segment.kind > 3 || segment.operand_index >= operand_count)
                return set_error("invalid raw hash dynamic tag segment");
            const int32_t encoding = operand_encodings[segment.operand_index];
            if (encoding < 0 || encoding == 2)
                return set_error("invalid raw hash signed integer encoding");
            const size_t words = encoding <= 1 ? 1 : static_cast<size_t>(encoding - 2);
            if (words == 0 || words > SIZE_MAX / 20)
                return set_error("raw hash operand size overflow");
            max_words = std::max(max_words, words);
            if (segment.kind == 1) extra = 10 + words * 8;
            if (segment.kind == 2) extra = 10 + words * 20;
            if (segment.kind == 3) extra = 9;
        }
        if (extra > SIZE_MAX - max_tag_bytes)
            return set_error("raw hash tag capacity overflow");
        max_tag_bytes += extra;
    }
    if (segment_count > SIZE_MAX / sizeof(MxxRawHashTagSegment) ||
        operand_count > SIZE_MAX / sizeof(MxxRawHashTagOperand) ||
        max_words > SIZE_MAX / sizeof(uint64_t))
        return set_error("raw hash plan allocation overflow");
    size_t cursor = 0;
    size_t q_offset = 0, moduli_offset = 0, segments_offset = 0;
    size_t static_offset = 0, operands_offset = 0, tag_length_offset = 0;
    size_t tag_offset = 0, scratch_offset = 0, descriptors_offset = 0;
    const bool sizes_ok =
        hash_align_add(cursor, 8, q_words.size() * 8, q_offset) &&
        hash_align_add(cursor, 8, limb_count * 8, moduli_offset) &&
        hash_align_add(cursor, 8, segment_count * sizeof(MxxRawHashTagSegment), segments_offset) &&
        hash_align_add(cursor, 8, static_byte_count, static_offset);
    if (!sizes_ok) return set_error("raw hash static allocation overflow");
    const size_t static_region_bytes = cursor;
    if (!hash_align_add(cursor, 8, operand_count * sizeof(MxxRawHashTagOperand), operands_offset) ||
        !hash_align_add(cursor, 8, 8, tag_length_offset) ||
        !hash_align_add(cursor, 8, max_tag_bytes, tag_offset) ||
        !hash_align_add(cursor, 8, max_words * 8, scratch_offset) ||
        !hash_align_add(cursor, 8, limb_count * sizeof(MxxRawMatrixLimb),
            descriptors_offset))
        return set_error("raw hash dynamic allocation overflow");
    auto *plan = new (std::nothrow) GpuRawHashPlan();
    if (!plan) return set_error("failed to allocate raw hash plan");
    plan->context = ctx;
    plan->device = device;
    plan->stream = reinterpret_cast<cudaStream_t>(stream_raw);
    plan->allocation_bytes = cursor;
    plan->moduli.assign(moduli, moduli + limb_count);
    if (operand_count)
        plan->operand_encodings.assign(operand_encodings, operand_encodings + operand_count);
    plan->q_word_count = q_words.size();
    plan->q_bits = (q_words.size() - 1) * 64 +
        (64 - __builtin_clzll(q_words.back()));
    plan->segment_count = segment_count;
    plan->static_bytes = static_byte_count;
    plan->max_tag_bytes = max_tag_bytes;
    plan->decimal_words = max_words;
    cudaError_t error = mxx_set_device(device);
    if (error == cudaSuccess)
        error = cudaMallocAsync(&plan->allocation, cursor, plan->stream);
    if (error == cudaSuccess)
        error = cudaHostAlloc(&plan->pinned_static, static_region_bytes,
            cudaHostAllocPortable);
    if (error == cudaSuccess && operand_count)
        error = cudaHostAlloc(reinterpret_cast<void **>(&plan->pinned_operands),
            operand_count * sizeof(MxxRawHashTagOperand), cudaHostAllocPortable);
    if (error == cudaSuccess)
        error = cudaEventCreateWithFlags(&plan->static_ready, cudaEventDisableTiming);
    if (error != cudaSuccess)
    {
        gpu_raw_hash_plan_destroy(plan);
        return set_error(error);
    }
    auto *base = static_cast<uint8_t *>(plan->allocation);
    plan->device_q = reinterpret_cast<uint64_t *>(base + q_offset);
    plan->device_moduli = reinterpret_cast<uint64_t *>(base + moduli_offset);
    plan->device_segments = reinterpret_cast<MxxRawHashTagSegment *>(base + segments_offset);
    plan->device_static = base + static_offset;
    plan->device_operands = reinterpret_cast<MxxRawHashTagOperand *>(base + operands_offset);
    plan->device_tag_length = reinterpret_cast<uint64_t *>(base + tag_length_offset);
    plan->device_tag = base + tag_offset;
    plan->device_decimal_scratch = reinterpret_cast<uint64_t *>(base + scratch_offset);
    plan->device_descriptors = reinterpret_cast<MxxRawMatrixLimb *>(
        base + descriptors_offset);
    std::memset(plan->pinned_static, 0, static_region_bytes);
    auto *host = static_cast<uint8_t *>(plan->pinned_static);
    std::memcpy(host + q_offset, q_words.data(), q_words.size() * 8);
    std::memcpy(host + moduli_offset, moduli, limb_count * 8);
    if (segment_count)
        std::memcpy(host + segments_offset, segments,
            segment_count * sizeof(MxxRawHashTagSegment));
    if (static_byte_count)
        std::memcpy(host + static_offset, static_bytes, static_byte_count);
    error = cudaMemcpyAsync(plan->allocation, plan->pinned_static,
        static_region_bytes, cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess)
        error = cudaEventRecord(plan->static_ready, plan->stream);
    if (error != cudaSuccess)
    {
        gpu_raw_hash_plan_destroy(plan);
        return set_error(error);
    }
    *out_plan = plan;
    return 0;
}

extern "C" int gpu_raw_hash_plan_prepare_graph_launch(
    GpuRawHashPlan *plan, void *stream_raw,
    const uint64_t *operand_addresses, const int32_t *operand_encodings,
    size_t operand_count)
{
    if (!plan || !stream_raw || operand_count != plan->operand_encodings.size() ||
        (operand_count && (!operand_addresses || !operand_encodings)))
        return set_error("invalid raw hash replay operand table");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = mxx_set_device(plan->device);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(stream, plan->static_ready, 0);
    for (size_t operand = 0; operand < operand_count; ++operand)
    {
        if (!operand_addresses[operand] ||
            operand_encodings[operand] != plan->operand_encodings[operand])
            return set_error("raw hash replay operand encoding/address changed");
        plan->pinned_operands[operand] = MxxRawHashTagOperand{
            operand_addresses[operand], operand_encodings[operand], 0};
    }
    if (error == cudaSuccess && operand_count)
        error = cudaMemcpyAsync(plan->device_operands, plan->pinned_operands,
            operand_count * sizeof(MxxRawHashTagOperand),
            cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" int gpu_raw_hash_plan_allocation_range(
    const GpuRawHashPlan *plan, uint64_t *address, size_t *bytes)
{
    if (!plan || !address || !bytes || !plan->allocation)
        return set_error("invalid raw hash plan allocation query");
    *address = reinterpret_cast<uint64_t>(plan->allocation);
    *bytes = plan->allocation_bytes;
    return 0;
}

extern "C" int gpu_raw_hash_sample_emit(
    GpuRawHashPlan *plan, GpuContext *ctx, void *stream_raw,
    const uint8_t *key, const MxxRawMatrixView *destination,
    uint32_t *status, uint32_t key_binding,
    uint32_t destination_binding_base, uint32_t status_binding)
{
    if (!plan || plan->context != ctx || !stream_raw || !key || !status ||
        !destination || !destination->limbs || !destination->rows ||
        !destination->columns || !destination->degree ||
        destination->physical_device != plan->device ||
        destination->degree != static_cast<uint32_t>(ctx->N) ||
        destination->limb_count != plan->moduli.size() ||
        destination_binding_base > UINT32_MAX - destination->limb_count ||
        destination->rows > UINT64_MAX / destination->columns ||
        destination->rows * destination->columns > UINT64_MAX / destination->degree ||
        destination->row_origin > UINT64_MAX - destination->rows ||
        destination->column_origin > UINT64_MAX - destination->columns ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream_raw))
        return set_error("invalid raw hash sample physical view");
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const auto &view = destination->limbs[limb];
        if (!view.address || view.crt_limb_index != limb ||
            view.modulus != plan->moduli[limb] ||
            (view.word_bytes != 4 && view.word_bytes != 8) ||
            (view.word_bytes == 4 && view.modulus > UINT32_MAX) ||
            view.coefficient_stride_bytes != view.word_bytes ||
            destination->degree > UINT64_MAX / view.word_bytes ||
            view.column_stride_bytes < destination->degree * view.word_bytes ||
            destination->columns > UINT64_MAX / view.column_stride_bytes ||
            view.row_stride_bytes < destination->columns * view.column_stride_bytes)
            return set_error("invalid raw hash sample owner limb layout");
    }
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const MxxGraphPatch tag_status_patch{nullptr,
        MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 10, 0,
        sizeof(void *), status_binding, 0};
    int result = mxx_gpu_launch_kernel(ctx, stream,
        raw_hash_tag_build_kernel, dim3(1), dim3(1), 0,
        &tag_status_patch, 1,
        plan->device_segments, plan->segment_count,
        plan->device_static, plan->device_operands,
        plan->operand_encodings.size(), plan->device_tag_length,
        plan->device_tag, plan->max_tag_bytes,
        plan->device_decimal_scratch, plan->decimal_words, status);
    if (result != 0) return result;
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const MxxGraphPatch patch{nullptr,
            MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
            static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
            sizeof(void *), destination_binding_base + static_cast<uint32_t>(limb), 0};
        result = mxx_gpu_launch_kernel(ctx, stream,
            raw_hash_descriptor_kernel, dim3(1), dim3(1), 0,
            &patch, 1, destination->limbs[limb], plan->device_descriptors + limb);
        if (result != 0) return result;
    }
    const uint64_t total = destination->rows * destination->columns * destination->degree;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(
        65535, (total + 127) / 128));
    const MxxGraphPatch sample_patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0,
            sizeof(void *), key_binding, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 15, 0,
            sizeof(void *), status_binding, 0},
    };
    return mxx_gpu_launch_kernel(ctx, stream,
        raw_hash_sample_kernel, dim3(blocks), dim3(128), 0,
        sample_patches, 2, key, plan->device_tag_length, plan->device_tag,
        plan->max_tag_bytes, plan->device_q, plan->q_word_count,
        plan->q_bits, plan->device_moduli, plan->moduli.size(),
        plan->device_descriptors, destination->row_origin,
        destination->column_origin, destination->rows, destination->columns,
        destination->degree, status);
}
