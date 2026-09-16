#include "matrix/MatrixScalar.cuh"
#include <math_constants.h>

namespace {
bool scalar_checked_mul_size(size_t left, size_t right, size_t *out)
{
    if (!out || (right && left > SIZE_MAX / right)) return false;
    *out = left * right;
    return true;
}

struct PreparedScalarView {
    const uint64_t *words;
    size_t word_count;
    uint64_t *status;
};

static int validate_scalar_stream(const GpuPreparedPlanDescriptor *descriptor,
    const GpuContext *ctx, int level, int role, cudaStream_t stream, size_t index)
{
    if (!descriptor || !ctx || !stream || index >= descriptor->stream_count)
        return set_error("invalid saved scalar stream descriptor");
    dim3 limb{};
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(ctx, level, 0, role, &limb, &key) != 0)
        return set_error("invalid saved scalar stream key");
    const auto &footprint = descriptor->streams[index];
    if (footprint.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        std::memcmp(&footprint.key, &key, sizeof(key)) != 0 ||
        gpu_prepared_require_stream_slot(ctx, limb.x, stream, footprint.pool_slot) != 0)
        return set_error("saved scalar stream differs from descriptor");
    return 0;
}

static int require_scalar_allocation(const GpuPreparedPlanDescriptor *descriptor,
    size_t index, int kind, const GpuPreparedResourceKey &key, size_t bytes,
    size_t alignment)
{
    return gpu_prepared_require_allocation(descriptor, index, kind, &key, bytes, alignment);
}

static int validate_scalar_buffer_layout(const GpuMatrix *anchor, size_t count,
    size_t words, const GpuPreparedPlanDescriptor *saved_plan)
{
    if (!anchor || !anchor->ctx || !saved_plan || gpu_prepared_validate_descriptor(saved_plan) != 0 ||
        !count || !words || words == SIZE_MAX || count > SIZE_MAX / (words + 1) / 8 ||
        anchor->level < 0 || anchor->shared_limb_buffers.size() != 1 ||
        anchor->ctx->limb_gpu_ids.empty() || anchor->exec_limb_states.empty() ||
        anchor->exec_limb_states[0].empty())
        return set_error("invalid scalar buffer binding");
    dim3 limb = anchor->ctx->limb_gpu_ids[0];
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(anchor->ctx, 0, 0, GPU_PREPARED_STAGE_SCALAR_BUFFER, &limb, &key) != 0 ||
        saved_plan->allocation_count != 4 || saved_plan->stream_count != 1 ||
        require_scalar_allocation(saved_plan, 0, GPU_PREPARED_PINNED_HOST, key,
            count * (words + 1) * 8, alignof(uint64_t)) != 0 ||
        require_scalar_allocation(saved_plan, 1, GPU_PREPARED_BATCH_WORKSPACE, key,
            count * (words + 1) * 8, alignof(uint64_t)) != 0 ||
        require_scalar_allocation(saved_plan, 2, GPU_PREPARED_COMPLETION_EVENT, key, 0, 1) != 0 ||
        require_scalar_allocation(saved_plan, 3, GPU_PREPARED_COMPLETION_EVENT, key, 0, 1) != 0 ||
        saved_plan->allocations[0].rows != 0 || saved_plan->allocations[0].columns != 0 ||
        saved_plan->allocations[0].level != -1 || saved_plan->allocations[0].format != -1 ||
        saved_plan->allocations[1].rows != 0 || saved_plan->allocations[1].columns != 0 ||
        saved_plan->allocations[1].level != 0 || saved_plan->allocations[1].format != GPU_POLY_FORMAT_COEFF)
        return set_error("scalar buffer owner differs from its saved descriptor");
    for (size_t index = 2; index < 4; ++index)
        if (saved_plan->allocations[index].rows != 0 || saved_plan->allocations[index].columns != 0 ||
            saved_plan->allocations[index].level != -1 || saved_plan->allocations[index].format != -1)
            return set_error("scalar buffer completion descriptor differs from its saved descriptor");
    return 0;
}

__device__ void prepared_threshold_coefficient(CrtLevelMetadata metadata,
    size_t coefficient, uint64_t *value)
{
    uint64_t digits[kCrtMaxLimbs];
    for (size_t limb = 0; limb < metadata.limb_count; ++limb) {
        const auto descriptor = metadata.descriptors[limb];
        const uint64_t modulus = metadata.moduli[limb];
        uint64_t digit = matrix_load_limb_u64(descriptor.base, 0, coefficient,
            descriptor.stride, descriptor.width) % modulus;
        for (size_t previous = 0; previous < limb; ++previous) {
            const uint64_t residue = digits[previous] % modulus;
            const uint64_t difference = digit >= residue ? digit - residue : modulus - (residue - digit);
            digit = mul_mod_u64(difference,
                metadata.garner[previous * metadata.garner_stride + limb], modulus);
        }
        digits[limb] = digit;
    }
    for (size_t limb = metadata.limb_count; limb-- > 0;) {
        uint64_t carry = digits[limb];
        for (int word = 0; word < metadata.word_count; ++word) {
            const unsigned __int128 term = static_cast<unsigned __int128>(value[word]) * metadata.moduli[limb] + carry;
            value[word] = static_cast<uint64_t>(term);
            carry = static_cast<uint64_t>(term >> 64);
        }
    }
}

__global__ void prepared_threshold_kernel(CrtLevelMetadata metadata,
    const uint64_t *plaintext, size_t plaintext_words, bool output_bool,
    size_t count, size_t output_words, uint64_t *output, uint64_t *status, uint64_t *scratch)
{
    const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coefficient >= count) return;
    status[coefficient] = 0;
    output += coefficient * output_words;
    for (size_t word = 0; word < output_words; ++word) output[word] = 0;
    uint64_t value[kCrtMaxWords + 1]{};
    prepared_threshold_coefficient(metadata, coefficient, value);
    if (plaintext_words == 1) {
        const uint64_t rounded = crt_rounded_scale(value, metadata.modulus_words,
            metadata.word_count, metadata.plaintext_modulus);
        output[0] = output_bool ? static_cast<uint64_t>(rounded != 0) : rounded;
        return;
    }
    const size_t numerator_words = static_cast<size_t>(metadata.word_count) + plaintext_words + 1;
    uint64_t *numerator = scratch + coefficient * (numerator_words + plaintext_words);
    uint64_t *quotient = numerator + numerator_words;
    for (size_t word = 0; word < numerator_words + plaintext_words; ++word) numerator[word] = 0;
    // Form p*x + floor(Q/2) with no bound on the warmup-known plaintext width.
    for (size_t p = 0; p < plaintext_words; ++p) {
        uint64_t carry = 0;
        for (int q = 0; q < metadata.word_count; ++q) {
            const size_t word = p + static_cast<size_t>(q);
            const unsigned __int128 product = static_cast<unsigned __int128>(plaintext[p]) * value[q] + numerator[word] + carry;
            numerator[word] = static_cast<uint64_t>(product);
            carry = static_cast<uint64_t>(product >> 64);
        }
        size_t word = p + static_cast<size_t>(metadata.word_count);
        while (carry != 0) {
            const unsigned __int128 sum = static_cast<unsigned __int128>(numerator[word]) + carry;
            numerator[word++] = static_cast<uint64_t>(sum);
            carry = static_cast<uint64_t>(sum >> 64);
        }
    }
    uint64_t carry = 0;
    for (int word = 0; word < metadata.word_count; ++word) {
        const uint64_t next = word + 1 < metadata.word_count ? metadata.modulus_words[word + 1] : 0;
        const uint64_t half = (metadata.modulus_words[word] >> 1) | ((next & 1) << 63);
        const unsigned __int128 sum = static_cast<unsigned __int128>(numerator[word]) + half + carry;
        numerator[word] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    for (size_t word = metadata.word_count; carry != 0; ++word) {
        const unsigned __int128 sum = static_cast<unsigned __int128>(numerator[word]) + carry;
        numerator[word] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
    }
    uint64_t remainder[kCrtMaxWords + 1]{};
    // Restoring long division. Q has at most the accepted CRT basis width;
    // the quotient occupies the preparation-sized plaintext payload.
    for (size_t bit = numerator_words * 64; bit-- > 0;) {
        uint64_t shifted = (numerator[bit / 64] >> (bit % 64)) & 1;
        for (int word = 0; word <= metadata.word_count; ++word) {
            const uint64_t next = remainder[word] >> 63;
            remainder[word] = (remainder[word] << 1) | shifted;
            shifted = next;
        }
        if (remainder[metadata.word_count] != 0 || crt_compare_words(remainder, metadata.modulus_words, metadata.word_count) >= 0) {
            uint64_t borrow = 0;
            for (int word = 0; word < metadata.word_count; ++word) {
                const uint64_t before = remainder[word];
                const uint64_t sub = metadata.modulus_words[word];
                remainder[word] = before - sub - borrow;
                borrow = before < sub || (borrow != 0 && before == sub);
            }
            remainder[metadata.word_count] -= borrow;
            if (bit / 64 < plaintext_words) quotient[bit / 64] |= uint64_t{1} << (bit % 64);
        }
    }
    bool equal = true;
    bool nonzero = false;
    for (size_t word = 0; word < plaintext_words; ++word) {
        equal = equal && quotient[word] == plaintext[word];
        nonzero = nonzero || quotient[word] != 0;
    }
    // The rounded quotient is in [0,p], so canonical reduction is one equality.
    if (output_bool) output[0] = !equal && nonzero;
    else for (size_t word = 0; word < plaintext_words; ++word)
        output[word] = equal ? 0 : quotient[word];
}

__global__ void prepared_scalar_pack_kernel(const PreparedScalarView *values,
    CrtOutputMetadata output, size_t ring_dimension, size_t coefficient_bits)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= ring_dimension * output.limb_count) return;
    const size_t limb = index / ring_dimension;
    const size_t coefficient = index % ring_dimension;
    const uint64_t modulus = output.moduli[limb];
    uint64_t residue = 0;
    if (coefficient_bits != 0) {
        for (size_t bit = coefficient_bits; bit-- > 0;) {
            const auto source = values[coefficient * coefficient_bits + bit];
            bool nonzero = false;
            for (size_t word = 0; word < source.word_count; ++word) nonzero |= source.words[word] != 0;
            residue = add_mod_u64(add_mod_u64(residue, residue, modulus), nonzero, modulus);
        }
    } else {
        const auto source = values[coefficient];
        for (size_t word = source.word_count; word-- > 0;) {
            for (unsigned bit = 64; bit-- > 0;)
                residue = add_mod_u64(add_mod_u64(residue, residue, modulus), (source.words[word] >> bit) & 1, modulus);
        }
        if ((source.words[source.word_count - 1] >> 63) != 0) {
            uint64_t power = 1;
            for (size_t bit = 0; bit < source.word_count * 64; ++bit) power = add_mod_u64(power, power, modulus);
            residue = residue >= power ? residue - power : modulus - (power - residue);
        }
    }
    const auto target = output.descriptors[limb];
    matrix_store_limb_u64(target.base, 0, coefficient, target.stride, target.width, residue);
}
}

struct GpuPreparedScalarBuffer {
    const GpuMatrix *anchor = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    size_t count = 0, words = 0;
    uint64_t *host = nullptr;
    // Explicit scalar capacity growth is admitted and ledgered by Rust before
    // this pointer is published. It is distinct from operation scratch and
    // never grows implicitly during a prepared submit.
    uint64_t *dynamic_data = nullptr;
    GpuDeviceWorkspace workspace;
    std::shared_ptr<GpuCudaResource> completion, readback;
    bool finalized = false;
    uint64_t *data() const { return dynamic_data ? dynamic_data : reinterpret_cast<uint64_t *>(workspace.data); }
    uint64_t *status() const { return data() + count * words; }
    size_t bytes() const { return count * (words + 1) * 8; }
    ~GpuPreparedScalarBuffer() {
        if (!finalized && dynamic_data) {
            (void)cudaSetDevice(device);
            (void)cudaFreeAsync(dynamic_data, stream);
        }
        if (!finalized) (void)workspace.release(stream);
    }
};

extern "C" int gpu_matrix_prepare_scalar_buffer(const GpuMatrix *anchor, size_t count,
    size_t words, uint64_t *host, const GpuPreparedPlanDescriptor *saved_plan, GpuPreparedScalarBuffer **out)
try {
    if (!out || !host || validate_scalar_buffer_layout(anchor, count, words, saved_plan) != 0)
        return set_error("invalid scalar buffer binding");
    *out = nullptr;
    auto buffer = std::make_unique<GpuPreparedScalarBuffer>();
    buffer->anchor = anchor;
    buffer->device = anchor->shared_limb_buffers[0].device;
    buffer->stream = anchor->exec_limb_states[0][0].stream;
    buffer->count = count; buffer->words = words; buffer->host = host;
    if (validate_scalar_stream(saved_plan, anchor->ctx, 0,
            GPU_PREPARED_STAGE_SCALAR_BUFFER, buffer->stream, 0) != 0)
        return set_error("scalar buffer stream differs from descriptor");
    int status = buffer->workspace.acquire(anchor->ctx, buffer->device, GPU_PREPARED_BATCH_WORKSPACE, buffer->bytes(), 8, buffer->stream);
    if (status == 0) {
        buffer->completion = std::make_shared<GpuCudaResource>();
        status = buffer->completion->acquire(anchor->ctx, buffer->device, GPU_PREPARED_COMPLETION_EVENT);
    }
    if (status == 0) {
        buffer->readback = std::make_shared<GpuCudaResource>();
        status = buffer->readback->acquire(anchor->ctx, buffer->device, GPU_PREPARED_COMPLETION_EVENT);
    }
    if (status != 0) return status;
    auto error = cudaMemsetAsync(buffer->data(), 0, buffer->bytes(), buffer->stream);
    if (error == cudaSuccess) error = cudaEventRecord(buffer->completion->event, buffer->stream);
    if (error != cudaSuccess) return set_error(error);
    *out = buffer.release();
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_matrix_validate_scalar_buffer(const GpuMatrix *anchor, size_t count,
    size_t words, const GpuPreparedPlanDescriptor *saved_plan)
{
    return validate_scalar_buffer_layout(anchor, count, words, saved_plan);
}

extern "C" int gpu_matrix_upload_scalar_buffer(const GpuPreparedScalarBuffer *buffer)
{
    auto error = cudaSetDevice(buffer->device);
    if (error == cudaSuccess) error = cudaMemcpyAsync(buffer->data(), buffer->host, buffer->bytes(), cudaMemcpyHostToDevice, buffer->stream);
    if (error == cudaSuccess) error = cudaEventRecord(buffer->completion->event, buffer->stream);
    if (error != cudaSuccess) {
        buffer->anchor->ctx->execution->unretired_work.store(true, std::memory_order_release);
        buffer->anchor->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
    }
    return error == cudaSuccess ? 0 : set_error(error);
}
extern "C" int gpu_matrix_resize_scalar_buffer(GpuPreparedScalarBuffer *buffer, size_t words, uint64_t *host)
try {
    if (!buffer || !host || !words || words == SIZE_MAX || words <= buffer->words ||
        buffer->count > SIZE_MAX / (words + 1) / 8)
        return set_error("invalid scalar buffer growth");
    auto error = cudaSetDevice(buffer->device);
    size_t words_with_status = 0, elements = 0, bytes = 0;
    if (!scalar_checked_mul_size(words, 1, &words_with_status) || words_with_status == SIZE_MAX ||
        !scalar_checked_mul_size(buffer->count, words_with_status + 1, &elements) ||
        !scalar_checked_mul_size(elements, sizeof(uint64_t), &bytes))
        return set_error("scalar buffer growth overflow");
    uint64_t *replacement = nullptr;
    if (error == cudaSuccess) {
        gpu_test_record_cuda_allocation();
        error = cudaMallocAsync(reinterpret_cast<void **>(&replacement), bytes, buffer->stream);
    }
    if (error == cudaSuccess) error = cudaMemsetAsync(replacement, 0, bytes, buffer->stream);
    if (error != cudaSuccess) {
        if (replacement) (void)cudaFreeAsync(replacement, buffer->stream);
        return set_error(error);
    }
    if (buffer->dynamic_data) {
        error = cudaFreeAsync(buffer->dynamic_data, buffer->stream);
        if (error != cudaSuccess) {
            (void)cudaFreeAsync(replacement, buffer->stream);
            return set_error(error);
        }
    }
    if (error == cudaSuccess) error = cudaEventRecord(buffer->completion->event, buffer->stream);
    if (error != cudaSuccess) {
        (void)cudaFreeAsync(replacement, buffer->stream);
        return set_error(error);
    }
    buffer->dynamic_data = replacement;
    buffer->words = words;
    buffer->host = host;
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }
extern "C" int gpu_matrix_query_scalar_buffer_completion(
    const GpuPreparedScalarBuffer *buffer, int *out_ready)
{
    if (!buffer || !out_ready) return set_error("invalid scalar completion query");
    auto error = cudaSetDevice(buffer->device);
    if (error == cudaSuccess) {
        error = cudaEventQuery(buffer->completion->event);
        if (error == cudaErrorNotReady) {
            (void)cudaGetLastError();
            *out_ready = 0;
            return 0;
        }
    }
    if (error != cudaSuccess) return set_error(error);
    *out_ready = 1;
    return 0;
}
extern "C" int gpu_matrix_read_scalar_buffer(const GpuPreparedScalarBuffer *buffer)
{
    auto error = cudaSetDevice(buffer->device);
    if (error == cudaSuccess) error = cudaMemcpyAsync(buffer->host, buffer->data(), buffer->bytes(), cudaMemcpyDeviceToHost, buffer->stream);
    if (error == cudaSuccess) error = cudaEventRecord(buffer->readback->event, buffer->stream);
    if (error == cudaSuccess) error = cudaEventSynchronize(buffer->readback->event);
    if (error != cudaSuccess) {
        buffer->anchor->ctx->execution->unretired_work.store(true, std::memory_order_release);
        buffer->anchor->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
    }
    return error == cudaSuccess ? 0 : set_error(error);
}
extern "C" int gpu_matrix_wait_scalar_buffer(const GpuPreparedScalarBuffer *buffer)
{
    auto error = cudaSetDevice(buffer->device);
    if (error == cudaSuccess) error = cudaEventSynchronize(buffer->completion->event);
    return error == cudaSuccess ? 0 : set_error(error);
}
extern "C" int gpu_matrix_finalize_scalar_buffer(
    GpuPreparedScalarBuffer *buffer, GpuEventSet **out_completion)
try {
    if (!buffer || !out_completion || buffer->finalized || !buffer->completion ||
        !buffer->readback || !buffer->anchor || !buffer->anchor->ctx ||
        !buffer->anchor->ctx->execution)
        return set_error("invalid scalar buffer finalization");
    *out_completion = nullptr;
    // Build the event-set owner before queuing any release. The completion
    // event is then handed to the returned owner, while the host-release event
    // remains available for the pinned reclaimer.
    auto completion = std::make_unique<GpuEventSet>();
    completion->execution = buffer->anchor->ctx->execution;
    cudaError_t error = cudaSetDevice(buffer->device);
    // Host staging is ordered by the current stream. Recording this preclaimed
    // event before the asynchronous device frees proves that every preceding
    // H2D copy has stopped using the pending host allocations.
    if (error == cudaSuccess) error = cudaEventRecord(buffer->readback->event, buffer->stream);
    if (error == cudaSuccess && buffer->dynamic_data) {
        error = cudaFreeAsync(buffer->dynamic_data, buffer->stream);
        if (error == cudaSuccess) buffer->dynamic_data = nullptr;
    }
    if (error == cudaSuccess) {
        const int status = buffer->workspace.release(buffer->stream);
        if (status != 0) error = cudaErrorUnknown;
    }
    // The ledger lease must not observe completion until the final dynamic
    // free (and the prepared workspace release) has been queued.
    if (error == cudaSuccess) error = cudaEventRecord(buffer->completion->event, buffer->stream);
    if (error != cudaSuccess) {
        buffer->anchor->ctx->execution->unretired_work.store(true, std::memory_order_release);
        buffer->anchor->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
        return set_error(error);
    }
    completion->entries.push_back(
        {buffer->completion->event, buffer->device, std::move(buffer->completion)});
    buffer->finalized = true;
    *out_completion = completion.release();
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_matrix_defer_scalar_buffer_pinned_free(
    GpuPreparedScalarBuffer *buffer, void *const *pointers, size_t count)
{
    if (!buffer || !buffer->anchor || !buffer->anchor->ctx ||
        !buffer->anchor->ctx->execution || !pointers || !count || !buffer->stream ||
        !buffer->readback)
        return set_error("invalid scalar buffer pinned free");
    if (!buffer->finalized) {
        cudaError_t error = cudaSetDevice(buffer->device);
        if (error == cudaSuccess)
            error = cudaEventRecord(buffer->readback->event, buffer->stream);
        if (error != cudaSuccess) return set_error(error);
    }
    auto events = std::make_unique<GpuEventSet>();
    events->execution = buffer->anchor->ctx->execution;
    events->entries.push_back(
        {buffer->readback->event, buffer->device, std::move(buffer->readback)});
    return gpu_event_set_defer_pinned_frees(
        buffer->anchor->ctx, events.release(), pointers, count);
}
extern "C" void gpu_matrix_destroy_scalar_buffer(GpuPreparedScalarBuffer *buffer) { delete buffer; }

namespace {
__device__ uint64_t scalar_word(PreparedScalarView value, size_t word)
{
    return word < value.word_count ? value.words[word] : uint64_t{0} - (value.words[value.word_count - 1] >> 63);
}
__device__ bool scalar_negative(PreparedScalarView value) { return (value.words[value.word_count - 1] >> 63) != 0; }
__device__ void scalar_negate(uint64_t *words, size_t count)
{
    uint64_t carry = 1;
    for (size_t word = 0; word < count; ++word) {
        const uint64_t inverted = ~words[word];
        words[word] = inverted + carry;
        carry = carry && words[word] == 0;
    }
}
__device__ int scalar_compare(const uint64_t *left, const uint64_t *right, size_t words)
{
    for (size_t word = words; word-- > 0;)
        if (left[word] != right[word]) return left[word] < right[word] ? -1 : 1;
    return 0;
}
__device__ void scalar_subtract(uint64_t *left, const uint64_t *right, size_t words)
{
    uint64_t borrow = 0;
    for (size_t word = 0; word < words; ++word) {
        const uint64_t before = left[word];
        left[word] = before - right[word] - borrow;
        borrow = before < right[word] || (borrow && before == right[word]);
    }
}
struct PreparedScalarOpRecord {
    PreparedScalarView left{}, right{}, output{};
    uint64_t *scratch = nullptr;
    size_t scratch_words = 0, bit = 0;
    const PreparedScalarView *candidates = nullptr;
    size_t candidate_count = 0;
    int opcode = 0;
};
__global__ void prepared_scalar_op_kernel(PreparedScalarOpRecord plan)
{
    const auto left = plan.left, right = plan.right, output = plan.output;
    auto *result = const_cast<uint64_t *>(output.words);
    for (size_t word = 0; word < output.word_count; ++word) result[word] = 0;
    *output.status = *left.status | (right.words ? *right.status : 0);
    if (*output.status != 0) return;
    if (plan.opcode == 17) {
        size_t selected = left.words[0];
        bool invalid = selected >= plan.candidate_count;
        for (size_t word = 1; word < left.word_count; ++word) invalid |= left.words[word] != 0;
        if (invalid) { *output.status = 2; return; }
        const auto source = plan.candidates[selected];
        *output.status |= *source.status;
        for (size_t word = 0; word < output.word_count; ++word) result[word] = scalar_word(source, word);
        return;
    }
    if (plan.opcode <= 2) {
        if (plan.opcode < 2) {
            uint64_t carry = plan.opcode == 1;
            for (size_t word = 0; word < output.word_count; ++word) {
                const uint64_t b = scalar_word(right, word) ^ (uint64_t{0} - static_cast<uint64_t>(plan.opcode == 1));
                const unsigned __int128 sum = static_cast<unsigned __int128>(scalar_word(left, word)) + b + carry;
                result[word] = static_cast<uint64_t>(sum);
                carry = static_cast<uint64_t>(sum >> 64);
            }
        } else {
            // Sign-extend operands to the mathematically sufficient result
            // width, then multiply modulo that exact two's-complement width.
            for (size_t a = 0; a < output.word_count; ++a) {
                uint64_t carry = 0;
                for (size_t b = 0; a + b < output.word_count; ++b) {
                    const unsigned __int128 product = static_cast<unsigned __int128>(scalar_word(left, a)) * scalar_word(right, b) + result[a + b] + carry;
                    result[a + b] = static_cast<uint64_t>(product);
                    carry = static_cast<uint64_t>(product >> 64);
                }
            }
        }
        return;
    }
    if (plan.opcode == 3 || plan.opcode == 4 || plan.opcode == 9) {
        const size_t words = plan.scratch_words;
        uint64_t *a = plan.scratch, *b = a + words, *remainder = b + words;
        for (size_t word = 0; word < words; ++word) { a[word] = scalar_word(left, word); b[word] = right.words ? scalar_word(right, word) : 0; remainder[word] = 0; }
        const bool negative_left = scalar_negative(left);
        const bool negative_right = right.words && scalar_negative(right);
        if (negative_left) scalar_negate(a, words);
        if (negative_right) scalar_negate(b, words);
        if (plan.opcode == 9) {
            size_t high = words;
            while (high && a[high - 1] == 0) --high;
            if (!high) { result[0] = __double_as_longlong(0.0); return; }
            const size_t bits = (high - 1) * 64 + 64 - __clzll(a[high - 1]);
            const size_t shift = bits > 64 ? bits - 64 : 0;
            const size_t offset = shift / 64;
            const unsigned part = shift % 64;
            uint64_t top = a[offset] >> part;
            if (part && offset + 1 < words) top |= a[offset + 1] << (64 - part);
            bool sticky = part && (a[offset] << (64 - part)) != 0;
            for (size_t word = 0; word < offset; ++word) sticky |= a[word] != 0;
            top |= static_cast<uint64_t>(sticky);
            const double value = shift > 1024 ? CUDART_INF : ldexp(__ull2double_rn(top), static_cast<int>(shift));
            result[0] = __double_as_longlong(negative_left ? -value : value);
            return;
        }
        bool divisor_zero = true;
        for (size_t word = 0; word < words; ++word) divisor_zero &= b[word] == 0;
        if (divisor_zero) { *output.status = 1; return; }
        for (size_t bit = words * 64; bit-- > 0;) {
            uint64_t carry = (a[bit / 64] >> (bit % 64)) & 1;
            for (size_t word = 0; word < words; ++word) {
                const uint64_t next = remainder[word] >> 63;
                remainder[word] = (remainder[word] << 1) | carry; carry = next;
            }
            if (scalar_compare(remainder, b, words) >= 0) {
                scalar_subtract(remainder, b, words);
                if (plan.opcode == 3 && bit / 64 < output.word_count) result[bit / 64] |= uint64_t{1} << (bit % 64);
            }
        }
        if (plan.opcode == 4) for (size_t word = 0; word < output.word_count; ++word) result[word] = remainder[word];
        if (plan.opcode == 3 ? negative_left != negative_right : negative_left) scalar_negate(result, output.word_count);
        return;
    }
    if (plan.opcode >= 5 && plan.opcode <= 7) {
        int comparison = 0;
        if (scalar_negative(left) != scalar_negative(right)) comparison = scalar_negative(left) ? -1 : 1;
        else for (size_t word = max(left.word_count, right.word_count); word-- > 0;) {
            const uint64_t a = scalar_word(left, word), b = scalar_word(right, word);
            if (a != b) { comparison = a < b ? -1 : 1; break; }
        }
        result[0] = plan.opcode == 5 ? comparison == 0 : plan.opcode == 6 ? comparison < 0 : comparison <= 0;
    } else if (plan.opcode == 16) {
        for (size_t word = 0; word < output.word_count; ++word) result[word] = scalar_word(left, word);
    } else if (plan.opcode == 8) result[0] = (scalar_word(left, plan.bit / 64) >> (plan.bit % 64)) & 1;
    else if (plan.opcode == 10) result[0] = left.words[0] != 0;
    else {
        const double a = __longlong_as_double(left.words[0]);
        const double b = right.words ? __longlong_as_double(right.words[0]) : 0;
        double value = 0;
        switch (plan.opcode) {
        case 11: value = __dadd_rn(a, b); break;
        case 12: value = __dsub_rn(a, b); break;
        case 13: value = __dmul_rn(a, b); break;
        case 14: value = __ddiv_rn(a, b); break;
        case 15: value = __dsqrt_rn(a); break;
        }
        result[0] = __double_as_longlong(value);
    }
}
}

struct GpuPreparedScalarOp {
    PreparedScalarOpRecord record;
    GpuPreparedScalarRef left_ref{}, right_ref{}, output_ref{};
    std::vector<GpuPreparedScalarRef> candidate_refs;
    std::vector<PreparedScalarView> candidate_views;
    const GpuPreparedScalarBuffer *output = nullptr;
    std::vector<const GpuPreparedScalarBuffer *> sources;
    GpuDeviceWorkspace workspace;
    size_t workspace_bytes = 0;
    uint8_t *dynamic_workspace = nullptr;
    ~GpuPreparedScalarOp() {
        if (dynamic_workspace && output && output->anchor && output->anchor->ctx) {
            (void)cudaSetDevice(output->device);
            (void)cudaFreeAsync(dynamic_workspace, output->stream);
        }
        (void)workspace.release(output ? output->stream : nullptr);
    }
};
extern "C" size_t gpu_matrix_scalar_op_workspace_bytes(size_t left, size_t right, size_t output, size_t candidate_count)
{ return (std::max({left, right, output}) + 1) * 3 * 8 + candidate_count * sizeof(PreparedScalarView); }
extern "C" int gpu_matrix_prepare_scalar_op(int opcode, GpuPreparedScalarRef left,
    GpuPreparedScalarRef right, GpuPreparedScalarRef output, size_t bit,
    const GpuPreparedScalarRef *candidates, size_t candidate_count,
    const GpuPreparedPlanDescriptor *saved_plan, GpuPreparedScalarOp **out)
try {
    if (!out || !saved_plan || gpu_prepared_validate_descriptor(saved_plan) != 0 || !left.owner || !output.owner ||
        !left.owner->anchor || !output.owner->anchor || !left.owner->anchor->ctx || !output.owner->anchor->ctx ||
        left.owner->anchor->ctx->limb_gpu_ids.empty() ||
        opcode < 0 || opcode > 17 || left.index >= left.owner->count || output.index >= output.owner->count ||
        (right.owner && right.index >= right.owner->count)) return set_error("invalid scalar operation binding");
    if ((right.owner && (!right.owner->anchor || !right.owner->anchor->ctx)) ||
        (opcode == 8 && (left.owner->words > SIZE_MAX / 64 || bit >= left.owner->words * 64)))
        return set_error("invalid scalar operation geometry");
    if (opcode == 17 && (!candidates || !candidate_count)) return set_error("scalar selection has no candidates");
    if (((opcode <= 7 || (opcode >= 11 && opcode <= 14)) && !right.owner) ||
        (left.owner == output.owner && left.index == output.index) ||
        (right.owner == output.owner && right.index == output.index)) return set_error("scalar operation requires distinct output and bound operands");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedScalarOp>();
    plan->left_ref = left;
    plan->right_ref = right;
    plan->output_ref = output;
    plan->output = output.owner;
    plan->sources.push_back(left.owner);
    if (right.owner) plan->sources.push_back(right.owner);
    const auto view = [](GpuPreparedScalarRef ref) {
        return ref.owner ? PreparedScalarView{ref.owner->data() + ref.index * ref.owner->words, ref.owner->words, ref.owner->status() + ref.index} : PreparedScalarView{};
    };
    plan->record.left = view(left); plan->record.right = view(right); plan->record.output = view(output);
    plan->record.bit = bit; plan->record.opcode = opcode;
    plan->record.scratch_words = std::max({left.owner->words, right.owner ? right.owner->words : 0, output.owner->words}) + 1;
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto candidate = candidates[index];
        if (!candidate.owner || candidate.index >= candidate.owner->count ||
            (candidate.owner == output.owner && candidate.index == output.index)) return set_error("invalid scalar selection candidate");
        plan->candidate_views.push_back(view(candidate));
        plan->candidate_refs.push_back(candidate);
        if (std::find(plan->sources.begin(), plan->sources.end(), candidate.owner) == plan->sources.end()) plan->sources.push_back(candidate.owner);
    }
    for (auto *source : plan->sources) if (
        source->device != output.owner->device || !source->anchor ||
        source->anchor->ctx != output.owner->anchor->ctx ||
        source->anchor->ctx->execution != output.owner->anchor->ctx->execution)
        return set_error("scalar operation context mismatch");
    const size_t workspace_bytes = gpu_matrix_scalar_op_workspace_bytes(
        left.owner->words, right.owner ? right.owner->words : 0, output.owner->words, candidate_count);
    dim3 limb = output.owner->anchor->ctx->limb_gpu_ids[0];
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(output.owner->anchor->ctx, 0, 0, GPU_PREPARED_STAGE_SCALAR_OP, &limb, &key) != 0 ||
        saved_plan->allocation_count != 1 || saved_plan->stream_count != 1 ||
        gpu_prepared_require_allocation(saved_plan, 0, GPU_PREPARED_BATCH_WORKSPACE, &key, workspace_bytes, 8) != 0 ||
        saved_plan->allocations[0].rows != 0 || saved_plan->allocations[0].columns != 0 ||
        saved_plan->allocations[0].level != 0 || saved_plan->allocations[0].format != GPU_POLY_FORMAT_COEFF ||
        validate_scalar_stream(saved_plan, output.owner->anchor->ctx, 0,
            GPU_PREPARED_STAGE_SCALAR_OP, output.owner->stream, 0) != 0)
        return set_error("scalar operation owner differs from its saved descriptor");
    plan->workspace_bytes = saved_plan->allocations[0].bytes;
    const int status = plan->workspace.acquire(output.owner->anchor->ctx, output.owner->device, GPU_PREPARED_BATCH_WORKSPACE,
        workspace_bytes, 8, output.owner->stream);
    if (status != 0) return status;
    plan->record.scratch = reinterpret_cast<uint64_t *>(plan->workspace.data);
    if (candidate_count) {
        auto *destination = reinterpret_cast<PreparedScalarView *>(plan->record.scratch + plan->record.scratch_words * 3);
        auto error = cudaMemcpyAsync(destination, plan->candidate_views.data(), candidate_count * sizeof(PreparedScalarView), cudaMemcpyHostToDevice, output.owner->stream);
        if (error == cudaSuccess) error = cudaEventRecord(output.owner->completion->event, output.owner->stream);
        if (error != cudaSuccess) return set_error(error);
        plan->record.candidates = destination;
        plan->record.candidate_count = candidate_count;
    }
    *out = plan.release(); return 0;
} catch (const std::exception &error) { return set_error(error.what()); }
extern "C" int gpu_matrix_resize_scalar_op_workspace(GpuPreparedScalarOp *plan, size_t words)
try {
    if (!plan || !plan->output || !words || words == SIZE_MAX)
        return set_error("invalid scalar operation workspace growth");
    const size_t required_words = words + 1;
    size_t scratch_elements = 0, scratch_bytes = 0, candidate_bytes = 0, required_bytes = 0;
    if (!scalar_checked_mul_size(required_words, 3, &scratch_elements) ||
        !scalar_checked_mul_size(scratch_elements, sizeof(uint64_t), &scratch_bytes) ||
        !scalar_checked_mul_size(plan->candidate_refs.size(), sizeof(PreparedScalarView), &candidate_bytes) ||
        scratch_bytes > SIZE_MAX - candidate_bytes)
        return set_error("scalar operation workspace overflow");
    required_bytes = scratch_bytes + candidate_bytes;
    if (required_bytes <= plan->workspace_bytes) return 0;
    cudaError_t error = cudaSetDevice(plan->output->device);
    uint8_t *replacement = nullptr;
    if (error == cudaSuccess) {
        gpu_test_record_cuda_allocation();
        error = cudaMallocAsync(reinterpret_cast<void **>(&replacement), required_bytes, plan->output->stream);
    }
    if (error == cudaSuccess) error = cudaMemsetAsync(replacement, 0, required_bytes, plan->output->stream);
    if (error != cudaSuccess) {
        if (replacement) (void)cudaFreeAsync(replacement, plan->output->stream);
        return set_error(error);
    }
    if (plan->dynamic_workspace) {
        error = cudaFreeAsync(plan->dynamic_workspace, plan->output->stream);
        if (error != cudaSuccess) {
            (void)cudaFreeAsync(replacement, plan->output->stream);
            return set_error(error);
        }
    }
    if (error == cudaSuccess) error = cudaEventRecord(plan->output->completion->event, plan->output->stream);
    if (error != cudaSuccess) {
        (void)cudaFreeAsync(replacement, plan->output->stream);
        return set_error(error);
    }
    plan->dynamic_workspace = replacement;
    plan->workspace_bytes = required_bytes;
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }
extern "C" int gpu_matrix_submit_scalar_op(GpuPreparedScalarOp *plan)
{
    auto error = cudaSetDevice(plan->output->device);
    if (error != cudaSuccess) return set_error(error);
    const auto view = [](GpuPreparedScalarRef ref) {
        return ref.owner ? PreparedScalarView{ref.owner->data() + ref.index * ref.owner->words,
            ref.owner->words, ref.owner->status() + ref.index} : PreparedScalarView{};
    };
    const size_t maximum_words = std::max({plan->left_ref.owner->words,
        plan->right_ref.owner ? plan->right_ref.owner->words : size_t{0},
        plan->output_ref.owner->words});
    if (maximum_words == SIZE_MAX) return set_error("scalar operation word count overflow");
    const size_t required_words = maximum_words + 1;
    size_t candidate_bytes = 0, scratch_elements = 0, scratch_bytes = 0, required_bytes = 0;
    if (!scalar_checked_mul_size(plan->candidate_refs.size(), sizeof(PreparedScalarView), &candidate_bytes) ||
        !scalar_checked_mul_size(required_words, 3, &scratch_elements) ||
        !scalar_checked_mul_size(scratch_elements, sizeof(uint64_t), &scratch_bytes) ||
        scratch_bytes > SIZE_MAX - candidate_bytes)
        return set_error("scalar operation workspace overflow");
    required_bytes = scratch_bytes + candidate_bytes;
    if (required_bytes > plan->workspace_bytes)
        return set_error("scalar operation exceeds saved workspace reservation");
    plan->record.left = view(plan->left_ref);
    plan->record.right = view(plan->right_ref);
    plan->record.output = view(plan->output_ref);
    plan->record.scratch_words = required_words;
    plan->record.scratch = reinterpret_cast<uint64_t *>(
        plan->dynamic_workspace ? plan->dynamic_workspace : plan->workspace.data);
    if (!plan->candidate_refs.empty()) {
        for (size_t index = 0; index < plan->candidate_refs.size(); ++index)
            plan->candidate_views[index] = view(plan->candidate_refs[index]);
        auto *destination = reinterpret_cast<PreparedScalarView *>(plan->record.scratch + required_words * 3);
        error = cudaMemcpyAsync(destination, plan->candidate_views.data(), candidate_bytes,
            cudaMemcpyHostToDevice, plan->output->stream);
        if (error != cudaSuccess) return set_error(error);
        plan->record.candidates = destination;
    }
    for (auto *source : plan->sources) {
        error = cudaStreamWaitEvent(plan->output->stream, source->completion->event, 0);
        if (error != cudaSuccess) return set_error(error);
    }
    gpu_test_record_kernel_launch();
    prepared_scalar_op_kernel<<<1, 1, 0, plan->output->stream>>>(plan->record);
    error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaEventRecord(plan->output->completion->event, plan->output->stream);
    if (error != cudaSuccess) return set_error(error);
    for (auto *source : plan->sources) {
        error = cudaStreamWaitEvent(source->stream, plan->output->completion->event, 0);
        if (error != cudaSuccess) return set_error(error);
    }
    return 0;
}
extern "C" void gpu_matrix_destroy_scalar_op(GpuPreparedScalarOp *plan) { delete plan; }

struct PreparedScalarMatrixCopy {
    BlockCopyMetadata metadata;
    GpuMatrixRange input, output;
    size_t source_columns, output_columns, n;
};
struct GpuPreparedScalarMatrixSelect {
    GpuMatrix *output = nullptr;
    GpuPreparedScalarRef selector{};
    std::vector<const GpuMatrix *> sources;
    std::vector<PreparedScalarMatrixCopy> records;
    GpuDeviceWorkspace workspace;
    int device = -1;
    cudaStream_t stream = nullptr;
    dim3 grid;
    ~GpuPreparedScalarMatrixSelect() { (void)workspace.release(stream); }
};
__global__ void prepared_scalar_matrix_select_kernel(const PreparedScalarMatrixCopy *candidates,
    size_t count, PreparedScalarView selector)
{
    size_t selected = selector.words[0];
    bool invalid = selected >= count;
    for (size_t word = 1; word < selector.word_count; ++word) invalid |= selector.words[word] != 0;
    if (invalid && blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0) atomicOr(reinterpret_cast<unsigned long long *>(selector.status), 2ull);
    const auto candidate = candidates[invalid ? 0 : selected];
    const size_t columns = candidate.output.column_end - candidate.output.column_start;
    const size_t rows = candidate.output.row_end - candidate.output.row_start;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * columns * candidate.n) return;
    const size_t polynomial = index / candidate.n;
    const size_t coefficient = index % candidate.n;
    const size_t row = polynomial / columns, column = polynomial % columns;
    const size_t source = (candidate.input.row_start + row) * candidate.source_columns + candidate.input.column_start + column;
    const size_t target = (candidate.output.row_start + row) * candidate.output_columns + candidate.output.column_start + column;
    const size_t limb = blockIdx.z;
    const auto &metadata = candidate.metadata;
    const uint64_t value = invalid ? 0 : matrix_load_limb_u64(metadata.src_bases[limb], source, coefficient, metadata.src_stride_bytes[limb], metadata.src_coeff_bytes[limb]);
    matrix_store_limb_u64(metadata.dst_bases[limb], target, coefficient, metadata.dst_stride_bytes[limb], metadata.dst_coeff_bytes[limb], value);
}
extern "C" size_t gpu_matrix_scalar_matrix_select_workspace_bytes(size_t count)
{ return count * sizeof(PreparedScalarMatrixCopy); }
extern "C" int gpu_matrix_prepare_scalar_matrix_select(GpuMatrix *output, GpuPreparedScalarRef selector,
    const GpuMatrix *const *sources, const GpuMatrixBatchView *views, size_t count,
    const GpuPreparedPlanDescriptor *saved_plan, GpuPreparedScalarMatrixSelect **out)
try {
    if (!output || !out || !saved_plan || gpu_prepared_validate_descriptor(saved_plan) != 0 || !sources || !views || !count ||
        !output->ctx || output->level < 0 || static_cast<size_t>(output->level) >= output->ctx->moduli.size() ||
        static_cast<size_t>(output->level) + 1 > GPU_RUNTIME_MAX_LIMBS ||
        !selector.owner || !selector.owner->anchor || selector.index >= selector.owner->count)
        return set_error("invalid matrix scalar selection");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedScalarMatrixSelect>();
    plan->output = output; plan->selector = selector;
    for (size_t index = 0; index < count; ++index) {
        if (!sources[index] || !sources[index]->ctx || sources[index]->ctx->execution != output->ctx->execution)
            return set_error("matrix scalar selection source context mismatch");
        GpuPreparedInputCopyState copy;
        int status = prepare_copy_layout(output, sources[index], &views[index], copy);
        if (status != 0) return status;
        if (sources[index] == output || sources[index]->format != output->format)
            return set_error("matrix scalar selection requires distinct aligned owners");
        if (index != 0 && (copy.device != plan->device || copy.stream != plan->stream ||
                std::memcmp(&copy.grid, &plan->grid, sizeof(dim3)) != 0))
            return set_error("matrix scalar selection sources use different launch streams");
        plan->device = copy.device; plan->stream = copy.stream; plan->grid = copy.grid;
        plan->records.push_back({copy.source_layout, copy.input_range, copy.output_range, copy.source_columns, output->cols, copy.n});
        plan->sources.push_back(sources[index]);
    }
    if (selector.owner->device != plan->device || selector.owner->anchor->ctx->execution != output->ctx->execution)
        return set_error("matrix scalar selector context mismatch");
    size_t bytes = 0;
    if (!scalar_checked_mul_size(count, sizeof(PreparedScalarMatrixCopy), &bytes))
        return set_error("matrix scalar selection workspace overflow");
    dim3 limb = output->ctx->limb_gpu_ids[0];
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(output->ctx, output->level, 0, GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT, &limb, &key) != 0 ||
        saved_plan->allocation_count != 1 || saved_plan->stream_count != 1 ||
        gpu_prepared_require_allocation(saved_plan, 0, GPU_PREPARED_BATCH_WORKSPACE, &key, bytes, 8) != 0 ||
        saved_plan->allocations[0].level != output->level || saved_plan->allocations[0].format != GPU_POLY_FORMAT_COEFF ||
        saved_plan->allocations[0].rows != plan->grid.x || saved_plan->allocations[0].columns != plan->grid.z)
        return set_error("matrix scalar selection owner differs from its saved descriptor");
    if (validate_scalar_stream(saved_plan, output->ctx, output->level,
            GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT, plan->stream, 0) != 0)
        return set_error("matrix scalar selection stream differs from descriptor");
    int status = plan->workspace.acquire(output->ctx, plan->device, GPU_PREPARED_BATCH_WORKSPACE, bytes, 8, plan->stream);
    if (status != 0) return status;
    auto error = cudaMemcpyAsync(plan->workspace.data, plan->records.data(), bytes, cudaMemcpyHostToDevice, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    status = matrix_record_all_limb_writes(output, plan->stream, true);
    if (status != 0) return status;
    *out = plan.release(); return 0;
} catch (const std::exception &error) { return set_error(error.what()); }
extern "C" int gpu_matrix_submit_scalar_matrix_select(const GpuPreparedScalarMatrixSelect *plan)
{
    auto error = cudaSetDevice(plan->device);
    if (error == cudaSuccess) error = cudaStreamWaitEvent(plan->stream, plan->selector.owner->completion->event, 0);
    if (error != cudaSuccess) return set_error(error);
    for (const auto *source : plan->sources) {
        const int status = matrix_wait_all_limb_streams(source, plan->device, plan->stream, true, true);
        if (status != 0) return status;
    }
    int status = matrix_wait_all_limb_streams(plan->output, plan->device, plan->stream, true);
    if (status != 0) return status;
    const auto selector = plan->selector;
    gpu_test_record_kernel_launch();
    prepared_scalar_matrix_select_kernel<<<plan->grid, 256, 0, plan->stream>>>(reinterpret_cast<const PreparedScalarMatrixCopy *>(plan->workspace.data), plan->sources.size(),
        {selector.owner->data() + selector.index * selector.owner->words, selector.owner->words, selector.owner->status() + selector.index});
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = matrix_record_all_limb_writes(plan->output, plan->stream, true);
    if (status != 0) return status;
    const auto &states = plan->output->exec_limb_states[0];
    const auto completion = states[states[0].completion_owner].write_done;
    for (const auto *source : plan->sources) {
        status = matrix_track_all_limb_consumers(source, plan->device, plan->stream, completion, true, true);
        if (status != 0) return status;
    }
    error = cudaStreamWaitEvent(selector.owner->stream, completion, 0);
    return error == cudaSuccess ? 0 : set_error(error);
}
extern "C" void gpu_matrix_destroy_scalar_matrix_select(GpuPreparedScalarMatrixSelect *plan) { delete plan; }

struct GpuPreparedThreshold {
    const GpuMatrix *source = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    CrtLevelMetadata metadata{};
    size_t count = 0, plaintext_words = 0;
    bool output_bool = false;
    GpuDeviceWorkspace workspace;
    GpuPreparedScalarBuffer *output = nullptr;
    uint64_t *plaintext = nullptr, *scratch = nullptr;
    std::vector<uint64_t> plaintext_host;
    ~GpuPreparedThreshold() { (void)workspace.release(stream); }
};

struct GpuPreparedScalarPack {
    GpuMatrix *output = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    CrtOutputMetadata metadata{};
    size_t coefficient_bits = 0, count = 0;
    std::vector<const GpuPreparedScalarBuffer *> sources;
    std::vector<GpuPreparedScalarRef> value_refs;
    mutable std::vector<PreparedScalarView> value_views;
    GpuDeviceWorkspace workspace;
    GpuCudaResource completion;
    GpuMatrixTransformPlan *transform = nullptr;
    ~GpuPreparedScalarPack() {
        if (transform) gpu_matrix_destroy_ntt_plan(transform);
        (void)workspace.release(stream);
    }
};

extern "C" int gpu_matrix_threshold_workspace_bytes(const GpuMatrix *source, size_t count,
    size_t plaintext_words, size_t *bytes)
{
    if (!source || !bytes || !plaintext_words) return set_error("invalid threshold layout");
    std::vector<uint64_t> modulus;
    const std::vector<uint64_t> active(source->ctx->moduli.begin(), source->ctx->moduli.begin() + source->level + 1);
    if (!serde_compute_modulus_words_le(active, &modulus)) return set_error("threshold modulus layout");
    const size_t scratch_words = plaintext_words == 1 ? 0 : modulus.size() + 2 * plaintext_words + 1;
    size_t words;
    if (!scalar_checked_mul_size(count, scratch_words, &words) || words > SIZE_MAX - plaintext_words ||
        !scalar_checked_mul_size(words + plaintext_words, sizeof(uint64_t), bytes)) return set_error("threshold layout overflow");
    return 0;
}

extern "C" int gpu_matrix_prepare_threshold(const GpuMatrix *source, size_t count,
    const uint64_t *plaintext, size_t plaintext_words, bool output_bool,
    GpuPreparedScalarBuffer *output, const GpuPreparedPlanDescriptor *saved_plan, GpuPreparedThreshold **out)
try {
    if (!out || !saved_plan || gpu_prepared_validate_descriptor(saved_plan) != 0 || !source || !source->ctx || !plaintext || !output ||
        !output->anchor || source->rows != 1 || source->cols != 1 ||
        source->format != GPU_POLY_FORMAT_COEFF || source->shared_limb_buffers.size() != 1 ||
        source->level < 0 || source->ctx->moduli.empty() ||
        static_cast<size_t>(source->level) != source->ctx->moduli.size() - 1 ||
        source->ctx->moduli.size() > kCrtMaxLimbs ||
        count == 0 || count > static_cast<size_t>(source->ctx->N) || plaintext_words == 0)
        return set_error("invalid prepared threshold binding");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedThreshold>();
    plan->source = source;
    plan->count = count;
    plan->plaintext_words = plaintext_words;
    if (output->count < count || output->words < (output_bool ? 1 : plaintext_words + 1) ||
        output->anchor->ctx != source->ctx || output->anchor->ctx->execution != source->ctx->execution ||
        output->device != source->shared_limb_buffers[0].device)
        return set_error("threshold scalar output contract mismatch");
    plan->output_bool = output_bool;
    plan->output = output;
    plan->device = source->shared_limb_buffers[0].device;
    plan->stream = output->stream;
    auto &metadata = plan->metadata;
    metadata.descriptors = source->shared_limb_buffers[0].device_descriptors;
    metadata.limb_count = source->level + 1;
    std::copy_n(source->ctx->moduli.begin(), metadata.limb_count, metadata.moduli);
    metadata.garner = source->ctx->ring_device_constants[0].garner_inverses;
    metadata.garner_stride = source->ctx->moduli.size();
    metadata.plaintext_modulus = plaintext[0];
    std::vector<uint64_t> modulus;
    const std::vector<uint64_t> active(source->ctx->moduli.begin(), source->ctx->moduli.begin() + metadata.limb_count);
    if (!serde_compute_modulus_words_le(active, &modulus) || modulus.size() > kCrtMaxWords) return set_error("threshold CRT basis too wide");
    metadata.word_count = static_cast<int>(modulus.size());
    std::copy(modulus.begin(), modulus.end(), metadata.modulus_words);
    size_t bytes = 0;
    int status = gpu_matrix_threshold_workspace_bytes(source, count, plaintext_words, &bytes);
    if (status != 0) return status;
    dim3 limb = source->ctx->limb_gpu_ids[0];
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(source->ctx, static_cast<int>(source->ctx->moduli.size() - 1), 0,
            GPU_PREPARED_STAGE_THRESHOLD, &limb, &key) != 0 ||
        saved_plan->allocation_count != 1 || saved_plan->stream_count != 1 ||
        gpu_prepared_require_allocation(saved_plan, 0, GPU_PREPARED_BATCH_WORKSPACE, &key, bytes, 8) != 0 ||
        saved_plan->allocations[0].rows != 0 || saved_plan->allocations[0].columns != 0 ||
        saved_plan->allocations[0].level != static_cast<int>(source->ctx->moduli.size() - 1) ||
        saved_plan->allocations[0].format != GPU_POLY_FORMAT_COEFF ||
        validate_scalar_stream(saved_plan, source->ctx,
            static_cast<int>(source->ctx->moduli.size() - 1), GPU_PREPARED_STAGE_THRESHOLD,
            plan->stream, 0) != 0)
        return set_error("threshold owner differs from its saved descriptor");
    status = plan->workspace.acquire(source->ctx, plan->device, GPU_PREPARED_BATCH_WORKSPACE, bytes, 8, plan->stream);
    if (status != 0) return status;
    plan->plaintext = reinterpret_cast<uint64_t *>(plan->workspace.data);
    plan->scratch = plan->plaintext + plaintext_words;
    plan->plaintext_host.assign(plaintext, plaintext + plaintext_words);
    auto error = cudaMemcpyAsync(plan->plaintext, plan->plaintext_host.data(), plaintext_words * 8,
        cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess) error = cudaEventRecord(output->completion->event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    *out = plan.release();
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_matrix_submit_threshold(const GpuPreparedThreshold *plan)
{
    auto error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(plan->source, plan->device, plan->stream, true, true);
    if (status != 0) return status;
    gpu_test_record_kernel_launch();
    prepared_threshold_kernel<<<(plan->count + 127) / 128, 128, 0, plan->stream>>>(plan->metadata,
        plan->plaintext, plan->plaintext_words, plan->output_bool, plan->count, plan->output->words, plan->output->data(), plan->output->status(), plan->scratch);
    error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaEventRecord(plan->output->completion->event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    return matrix_track_all_limb_consumers(plan->source, plan->device, plan->stream, plan->output->completion->event, true, true);
}
extern "C" void gpu_matrix_destroy_threshold(GpuPreparedThreshold *plan) { delete plan; }

extern "C" size_t gpu_matrix_scalar_pack_workspace_bytes(size_t count) { return count * sizeof(PreparedScalarView); }
extern "C" int gpu_matrix_prepare_scalar_pack(GpuMatrix *output, const GpuPreparedScalarRef *values,
    size_t count, size_t coefficient_bits, const GpuPreparedPlanDescriptor *saved_plan,
    GpuPreparedScalarPack **out)
try {
    if (!out || !saved_plan || gpu_prepared_validate_descriptor(saved_plan) != 0 || !output || !output->ctx ||
        !values || output->rows != 1 || output->cols != 1 || output->level < 0 ||
        static_cast<size_t>(output->level) >= output->ctx->moduli.size() ||
        static_cast<size_t>(output->level) + 1 > kCrtMaxLimbs || output->ctx->limb_gpu_ids.empty() ||
        output->shared_limb_buffers.size() != 1 || output->exec_limb_states.empty() ||
        output->exec_limb_states[0].empty() ||
        (output->format != GPU_POLY_FORMAT_COEFF && output->format != GPU_POLY_FORMAT_EVAL) ||
        count != static_cast<size_t>(output->ctx->N) * (coefficient_bits ? coefficient_bits : 1))
        return set_error("invalid prepared scalar pack binding");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedScalarPack>();
    plan->output = output;
    plan->device = output->shared_limb_buffers[0].device;
    plan->stream = output->exec_limb_states[0][0].stream;
    plan->coefficient_bits = coefficient_bits;
    plan->count = count;
    plan->metadata.descriptors = output->shared_limb_buffers[0].device_descriptors;
    plan->metadata.limb_count = output->level + 1;
    std::copy_n(output->ctx->moduli.begin(), plan->metadata.limb_count, plan->metadata.moduli);
    for (size_t index = 0; index < count; ++index) {
        const auto &value = values[index];
        if (!value.owner || !value.owner->anchor || value.index >= value.owner->count ||
            value.owner->device != plan->device || value.owner->anchor->ctx != output->ctx ||
            value.owner->anchor->ctx->execution != output->ctx->execution)
            return set_error("invalid prepared scalar source");
        plan->value_views.push_back({value.owner->data() + value.index * value.owner->words,
            value.owner->words, value.owner->status() + value.index});
        plan->value_refs.push_back(value);
        if (std::find(plan->sources.begin(), plan->sources.end(), value.owner) == plan->sources.end()) plan->sources.push_back(value.owner);
    }
    size_t workspace_bytes = 0;
    if (!scalar_checked_mul_size(count, sizeof(PreparedScalarView), &workspace_bytes))
        return set_error("scalar pack workspace overflow");
    dim3 limb = output->ctx->limb_gpu_ids[0];
    GpuPreparedResourceKey key{};
    if (gpu_prepared_limb_key(output->ctx, output->level, 0, GPU_PREPARED_STAGE_SCALAR_PACK, &limb, &key) != 0 ||
        saved_plan->allocation_count != (output->format == GPU_POLY_FORMAT_EVAL ? 3u : 2u) ||
        saved_plan->stream_count != (output->format == GPU_POLY_FORMAT_EVAL ? 2u : 1u) ||
        (output->format == GPU_POLY_FORMAT_COEFF && saved_plan->launch_count != 0) ||
        (output->format == GPU_POLY_FORMAT_EVAL && saved_plan->launch_count == 0) ||
        gpu_prepared_require_allocation(saved_plan, 0, GPU_PREPARED_BATCH_WORKSPACE, &key,
            workspace_bytes, 8) != 0 ||
        saved_plan->allocations[0].rows != 0 || saved_plan->allocations[0].columns != 0 ||
        saved_plan->allocations[0].level != output->level || saved_plan->allocations[0].format != output->format ||
        gpu_prepared_require_allocation(saved_plan, 1, GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0 ||
        saved_plan->allocations[1].rows != 0 || saved_plan->allocations[1].columns != 0 ||
        saved_plan->allocations[1].level != -1 || saved_plan->allocations[1].format != -1 ||
        validate_scalar_stream(saved_plan, output->ctx, output->level,
            GPU_PREPARED_STAGE_SCALAR_PACK, plan->stream, 0) != 0)
        return set_error("scalar pack owner differs from its saved descriptor");
    if (output->format == GPU_POLY_FORMAT_EVAL) {
        GpuPreparedPlanDescriptor ntt_layout{};
        ntt_layout.allocation_count = 1;
        ntt_layout.allocations[0] = saved_plan->allocations[2];
        ntt_layout.stream_count = 1;
        ntt_layout.streams[0] = saved_plan->streams[1];
        ntt_layout.launch_count = saved_plan->launch_count;
        std::copy_n(saved_plan->launches, saved_plan->launch_count, ntt_layout.launches);
        const GpuMatrixRange range{0, 1, 0, 1};
        const int ntt_status = gpu_matrix_prepare_ntt_plan_with_layout(
            output, &range, true, &ntt_layout, &plan->transform);
        if (ntt_status != 0) return ntt_status;
    }
    int status = plan->workspace.acquire(output->ctx, plan->device, GPU_PREPARED_BATCH_WORKSPACE,
        workspace_bytes, 8, plan->stream);
    if (status != 0) return status;
    status = plan->completion.acquire(output->ctx, plan->device, GPU_PREPARED_COMPLETION_EVENT);
    if (status != 0) return status;
    auto error = cudaMemcpyAsync(plan->workspace.data, plan->value_views.data(), workspace_bytes, cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess) error = cudaEventRecord(plan->completion.event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    *out = plan.release();
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_matrix_submit_scalar_pack(const GpuPreparedScalarPack *plan)
{
    auto error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    for (const auto *source : plan->sources) {
        error = cudaStreamWaitEvent(plan->stream, source->completion->event, 0);
        if (error != cudaSuccess) return set_error(error);
    }
    for (size_t index = 0; index < plan->value_refs.size(); ++index) {
        const auto ref = plan->value_refs[index];
        plan->value_views[index] = {ref.owner->data() + ref.index * ref.owner->words,
            ref.owner->words, ref.owner->status() + ref.index};
    }
    error = cudaMemcpyAsync(plan->workspace.data, plan->value_views.data(), plan->value_views.size() * sizeof(PreparedScalarView),
        cudaMemcpyHostToDevice, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(plan->output, plan->device, plan->stream, true);
    if (status != 0) return status;
    gpu_test_record_kernel_launch();
    prepared_scalar_pack_kernel<<<(plan->output->ctx->N * plan->metadata.limb_count + 127) / 128, 128, 0, plan->stream>>>(
        reinterpret_cast<const PreparedScalarView *>(plan->workspace.data), plan->metadata, plan->output->ctx->N, plan->coefficient_bits);
    error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaEventRecord(plan->completion.event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    for (const auto *source : plan->sources) {
        error = cudaStreamWaitEvent(source->stream, plan->completion.event, 0);
        if (error != cudaSuccess) return set_error(error);
    }
    status = matrix_record_all_limb_writes(plan->output, plan->stream, true);
    if (status != 0) return status;
    return plan->transform ? gpu_matrix_submit_ntt_plan(plan->transform, plan->output) : 0;
}
extern "C" void gpu_matrix_destroy_scalar_pack(GpuPreparedScalarPack *plan) { delete plan; }
