#include "matrix/MatrixSmallRhs.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

/*
 * Compact RHS storage deliberately has no relationship to GpuMatrix.  The
 * only device allocation owned by this object is the canonical sign and
 * magnitude byte stream.  The host-side bound is metadata supplied by the
 * already validated Rust schema; it is never inferred from the stream.
 */
// Immutable CRT reconstruction metadata is shared with captured graphs.  The
// payload, acceptance status and staging bytes remain per-invocation owners.
struct SmallHardCutoffMetadata
{
    int device = -1;
    cudaStream_t stream = nullptr;
    uint64_t *garner_inverses = nullptr;
    int *subset_indices = nullptr;
    uint64_t *modulus_words = nullptr;
    uint64_t *half_modulus_words = nullptr;
    uint64_t *bound_words = nullptr;

    ~SmallHardCutoffMetadata()
    {
        cudaSetDevice(device);
        if (bound_words) cudaFreeAsync(bound_words, stream);
        if (half_modulus_words) cudaFreeAsync(half_modulus_words, stream);
        if (modulus_words) cudaFreeAsync(modulus_words, stream);
        if (subset_indices) cudaFreeAsync(subset_indices, stream);
        if (garner_inverses) cudaFreeAsync(garner_inverses, stream);
    }
};

struct GpuSmallMatrix
{
    GpuContext *ctx = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t storage_cols = 0;
    size_t column_offset = 0;
    size_t n = 0;
    size_t magnitude_bytes = 0;
    size_t payload_bytes = 0;
    size_t resident_payload_bytes = 0;
    bool owns_payload = true;
    bool owns_write_event = true;
    int device = -1;
    cudaStream_t stream = nullptr;
    uint8_t *payload = nullptr;
    std::vector<uint64_t> bound_words;
    size_t hard_cutoff_limb_count = 0;
    int hard_cutoff_subset_count = 0;
    int hard_cutoff_words_per_coeff = 0;
    uint64_t *hard_cutoff_garner_inverses = nullptr;
    int *hard_cutoff_subset_indices = nullptr;
    uint64_t *hard_cutoff_modulus_words = nullptr;
    uint64_t *hard_cutoff_half_modulus_words = nullptr;
    uint64_t *hard_cutoff_bound_words = nullptr;
    std::shared_ptr<SmallHardCutoffMetadata> hard_cutoff_metadata;
    // One fixed status record is reused by every retry invocation.  It is
    // never allocated, copied, or synchronised from inside the attempt body.
    MxxPreimageStatus *hard_cutoff_device_status = nullptr;
    MxxPreimageStatus *hard_cutoff_host_status = nullptr;
    uint8_t *hard_cutoff_staging = nullptr;
    size_t hard_cutoff_staging_bytes = 0;
    cudaEvent_t hard_cutoff_decision_ready = nullptr;
    bool hard_cutoff_decision_recorded = false;
    cudaEvent_t write_done = nullptr;
    MxxGpuCaptureEvent *capture_write_done = nullptr;
    bool write_done_valid = false;
};

namespace
{
constexpr int kSmallThreads = 256;
constexpr size_t kMaxSmallLimbCount = GPU_RUNTIME_MAX_LIMBS;
constexpr uint32_t kCompactNttSuffixSize = 4096;

MxxGraphPatch small_kernel_pointer_patch(
    uint32_t argument_index,
    size_t byte_offset,
    uint32_t binding_index,
    size_t byte_count = sizeof(void *))
{
    MxxGraphPatch patch{};
    patch.node = nullptr;
    patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
    patch.argument_index = argument_index;
    patch.byte_offset = static_cast<uint32_t>(byte_offset);
    patch.byte_count = static_cast<uint32_t>(byte_count);
    patch.binding_index = binding_index;
    patch.address_addend = 0;
    return patch;
}

int register_compact_rhs_update(
    GpuContext *ctx,
    cudaStream_t stream,
    uint32_t rhs_binding_index)
{
    const size_t argument_sizes[] = {
        sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *), sizeof(void *),
        sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t),
        sizeof(uint32_t), sizeof(size_t), sizeof(size_t),
    };
    // Only the compact payload is an externally rebound pointer.  The
    // twiddle/modulus/workspace addresses belong to this captured executable
    // and remain graph-local across replay.  Keeping them out of the binding
    // schema also leaves the low binding indices available for the row-block
    // matrix descriptors used by the accumulate phase.
    const MxxGraphPatch patches[] = {
        small_kernel_pointer_patch(0, 0, rhs_binding_index),
    };
    return mxx_graph_register_kernel_update_for_stream(
        ctx, reinterpret_cast<void *>(stream), argument_sizes, std::size(argument_sizes),
        patches, std::size(patches));
}

bool small_mul_size(size_t a, size_t b, size_t *out)
{
    if (!out || (a != 0 && b > std::numeric_limits<size_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool small_add_size(size_t a, size_t b, size_t *out)
{
    if (!out || b > std::numeric_limits<size_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int small_set_device(const GpuSmallMatrix *mat)
{
    if (!mat || mat->device < 0) return set_error("invalid compact matrix device");
    const cudaError_t err = cudaSetDevice(mat->device);
    return err == cudaSuccess ? 0 : set_error(err);
}

cudaStream_t small_capture_stream(const GpuSmallMatrix *mat)
{
    if (!mat) return nullptr;
    return matrix_capture_stream_for_device(mat->ctx, mat->device, mat->stream);
}

int small_wait(const GpuSmallMatrix *mat, cudaStream_t stream)
{
    if (!mat || !stream) return set_error("invalid compact matrix wait arguments");
    if (matrix_stream_is_capturing(stream)) return 0;
    if (!mat->write_done_valid) return 0;
    const cudaError_t err = cudaStreamWaitEvent(stream, mat->write_done, 0);
    return err == cudaSuccess ? 0 : set_error(err);
}

int small_record(GpuSmallMatrix *mat, cudaStream_t stream)
{
    if (!mat || !stream || !mat->write_done)
        return set_error("invalid compact matrix event arguments");
    if (matrix_stream_is_capturing(stream))
    {
        MxxGpuCaptureEvent *capture_event = matrix_capture_event_for_owner(
            mat->ctx, mat->device, &mat->capture_write_done);
        if (!capture_event)
        {
            return 1;
        }
        const cudaError_t capture_error = cudaEventRecord(capture_event->event, stream);
        return capture_error == cudaSuccess ? 0 : set_error(capture_error);
    }
    const cudaError_t err = cudaEventRecord(mat->write_done, stream);
    if (err != cudaSuccess) return set_error(err);
    mat->write_done_valid = true;
    return 0;
}

cudaError_t small_fence_stream_with_event(cudaStream_t stream)
{
    if (!stream) return cudaErrorInvalidResourceHandle;
    cudaEvent_t completion = nullptr;
    cudaError_t err = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
    if (err == cudaSuccess) err = cudaEventRecord(completion, stream);
    if (err != cudaSuccess)
    {
        if (completion) cudaEventDestroy(completion);
        // No event can establish a completion dependency when creation or
        // recording itself fails. This is the sole stream-wide fallback and
        // is confined to an already failing path.
        if (matrix_stream_is_capturing(stream))
        {
            return err;
        }
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        return sync_err == cudaSuccess ? err : sync_err;
    }
    if (matrix_stream_is_capturing(stream))
    {
        // Host observation is not part of a captured region. Leave the
        // event for abort-time cleanup and fail closed without synchronizing.
        return cudaErrorStreamCaptureUnsupported;
    }
    const cudaError_t sync_err = cudaEventSynchronize(completion);
    if (sync_err == cudaSuccess)
        cudaEventDestroy(completion);
    // On an asynchronous device error the event is deliberately leaked: its
    // completion state is uncertain, so destroying it would weaken safety.
    return sync_err;
}

int small_track_consumer(const GpuSmallMatrix *mat, cudaStream_t consumer_stream, cudaEvent_t completion)
{
    if (!mat || !mat->ctx || !consumer_stream || mat->device < 0)
        return set_error("invalid compact matrix consumer arguments");
    const cudaStream_t producer_stream = small_capture_stream(mat);
    cudaStream_t release_stream = producer_stream;
    if (!mat->ctx->execution->release_streams_by_partition.empty() &&
        mat->ctx->execution->release_streams_by_partition.front())
    {
        release_stream = matrix_capture_stream_for_device(
            mat->ctx,
            mat->device,
            mat->ctx->execution->release_streams_by_partition.front());
    }
    if (!release_stream) return set_error("missing compact matrix release stream");

    cudaError_t err = cudaSetDevice(mat->device);
    if (err != cudaSuccess) return set_error(err);
    if (matrix_stream_is_capturing(consumer_stream))
    {
        // Capture stream order covers the compact consumer. Its replay owner
        // receives a fresh non-captured completion event after launch.
        return 0;
    }
    // The caller retains this dominating completion until the release wait
    // is enqueued. Reuse it without creating or destroying another event.
    if (!completion) return set_error("missing compact consumer completion");
    err = cudaStreamWaitEvent(release_stream, completion, 0);
    if (err == cudaSuccess && producer_stream != consumer_stream && producer_stream != release_stream)
        err = cudaStreamWaitEvent(producer_stream, completion, 0);
    if (err != cudaSuccess)
        (void)small_fence_stream_with_event(consumer_stream);
    return err == cudaSuccess ? 0 : set_error(err);
}

int small_payload_size(
    size_t rows,
    size_t cols,
    size_t n,
    size_t magnitude_bytes,
    size_t *out)
{
    size_t count = 0;
    size_t width = 0;
    if (!small_mul_size(rows, cols, &count) || !small_mul_size(count, n, &count) ||
        !small_add_size(magnitude_bytes, 1, &width) || !small_mul_size(count, width, out))
        return set_error("compact matrix payload size overflow");
    return 0;
}

__device__ __forceinline__ uint64_t compact_mod_magnitude(
    const uint8_t *magnitude,
    size_t width,
    uint64_t modulus)
{
    uint64_t value = 0;
    for (size_t i = width; i-- > 0;)
    {
        value = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(value) * 256u + magnitude[i]) % modulus);
    }
    return value;
}

__device__ __forceinline__ void compact_store_signed(
    uint8_t *dst,
    size_t width,
    int64_t value)
{
    if (value == 0)
    {
        dst[0] = 0;
        for (size_t i = 0; i < width; ++i) dst[1 + i] = 0;
        return;
    }
    const bool negative = value < 0;
    dst[0] = negative ? 2 : 1;
    uint64_t magnitude = negative
        ? static_cast<uint64_t>(-(value + 1)) + 1
        : static_cast<uint64_t>(value);
    for (size_t i = 0; i < width; ++i)
    {
        dst[1 + i] = static_cast<uint8_t>(magnitude & 0xffu);
        magnitude >>= 8;
    }
}

// One descriptor per owned row block, bounded independently of matrix size.
constexpr size_t kCompactRowBlocks = 32;
constexpr size_t kCompactSourceFragments = 128;
struct CompactRowBlocks
{
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *inputs[kCompactSourceFragments];
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *outputs[kCompactRowBlocks];
    size_t ends[kCompactRowBlocks];
    size_t fragment_ends[kCompactRowBlocks];
    size_t column_ends[kCompactSourceFragments];
    __device__ size_t locate(size_t &row) const
    {
        size_t block = 0;
        while (row >= ends[block]) ++block;
        if (block) row -= ends[block - 1];
        return block;
    }
};
static_assert(sizeof(CompactRowBlocks) + 9 * sizeof(size_t) < 4096, "bounded compact block metadata");

int register_compact_accumulate_update(
    GpuContext *ctx,
    cudaStream_t stream,
    size_t block_count,
    size_t source_count,
    const uint32_t *source_binding_indices,
    const uint32_t *destination_binding_indices)
{
    const size_t argument_sizes[] = {
        sizeof(CompactRowBlocks), sizeof(void *), sizeof(size_t), sizeof(void *),
        sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(bool),
    };
    std::vector<MxxGraphPatch> patches;
    patches.reserve(source_count + block_count);
    for (size_t block = 0; block < source_count; ++block)
    {
        const uint32_t source_binding = source_binding_indices
            ? source_binding_indices[block]
            : static_cast<uint32_t>(block);
        patches.push_back(small_kernel_pointer_patch(
            0, offsetof(CompactRowBlocks, inputs) + block * sizeof(void *), source_binding));
    }
    for (size_t block = 0; block < block_count; ++block)
    {
        if (destination_binding_indices)
        {
            patches.push_back(small_kernel_pointer_patch(
                0,
                offsetof(CompactRowBlocks, outputs) + block * sizeof(void *),
                destination_binding_indices[block]));
        }
    }
    if (block_count == 0 || block_count > kCompactRowBlocks) return set_error("invalid compact accumulate block count");
    return mxx_graph_register_kernel_update_for_stream(
        ctx, reinterpret_cast<void *>(stream), argument_sizes, std::size(argument_sizes),
        patches.data(), patches.size());
}

struct CompactDecomposeFragments
{
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *inputs[kCompactRowBlocks];
    size_t ends[kCompactRowBlocks], source_columns[kCompactRowBlocks];
    MxxDecomposeFragmentRange ranges[kCompactRowBlocks];
};

__global__ void compact_decompose_kernel(
    CompactDecomposeFragments blocks,
    const uint64_t *src_moduli,
    uint8_t *dst,
    size_t poly_count,
    size_t out_cols,
    size_t slots,
    size_t n,
    size_t digits,
    size_t magnitude_bytes,
    uint32_t base_bits,
    bool balanced,
    bool small)
{
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t poly = static_cast<size_t>(blockIdx.y);
    const size_t slot = static_cast<size_t>(blockIdx.z);
    if (coeff >= n || poly >= poly_count || slot >= slots) return;
    const size_t source_limb = small ? 0 : slot / digits;
    const size_t digit_idx = slot % digits;
    const uint64_t modulus = src_moduli[source_limb];
    size_t block = 0;
    while (poly >= blocks.ends[block]) ++block;
    const size_t local_poly = poly - (block ? blocks.ends[block - 1] : 0);
    const auto range = blocks.ranges[block];
    const size_t local_row = local_poly / range.columns;
    const size_t local_col = local_poly % range.columns;
    const auto descriptor = blocks.inputs[block][source_limb];
    const uint64_t residue = matrix_load_limb_u64(
        descriptor.base, local_row * blocks.source_columns[block] + range.source_column + local_col,
        coeff, descriptor.stride, descriptor.width);
    int64_t digit = 0;
    if (balanced)
    {
        int64_t value = centered_lift_u64(residue, modulus);
        const int64_t base = int64_t{1} << base_bits;
        for (size_t i = 0; i <= digit_idx; ++i)
        {
            int64_t next = 0;
            const int64_t current = balanced_digit_step(value, base, &next);
            if (i == digit_idx) digit = current;
            value = next;
        }
    }
    else
    {
        const uint32_t shift = static_cast<uint32_t>(digit_idx * base_bits);
        const uint32_t bits = shift >= 64 ? 0 : min(base_bits, 64u - shift);
        const uint64_t mask = bits == 64 ? ~uint64_t{0} : (bits == 0 ? 0 : ((uint64_t{1} << bits) - 1));
        digit = static_cast<int64_t>((residue >> shift) & mask);
    }
    const size_t out_row = (range.destination_row + local_row) * slots + slot;
    const size_t out_poly = out_row * out_cols + range.destination_column + local_col;
    const size_t out_idx = (out_poly * n + coeff) * (1 + magnitude_bytes);
    compact_store_signed(dst + out_idx, magnitude_bytes, digit);
}

template <class Word>
__global__ void compact_rhs_dif_first_kernel(
    const uint8_t *payload,
    Word *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t cols,
    size_t source_cols,
    uint32_t n,
    size_t magnitude_bytes,
    size_t source_column_offset)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t half = n >> 1;
    const size_t total = limb_count * poly_count * half;
    if (idx >= total) return;
    const size_t j = idx % half;
    const size_t q = idx / half;
    const size_t poly = q % poly_count;
    const size_t local_limb = q / poly_count;
    const size_t limb = limb_offset + local_limb;
    const uint64_t modulus = moduli[limb];
    const size_t k = poly / cols;
    const size_t c = poly % cols;
    const size_t rhs_col = source_column_offset + c;
    const size_t width = 1 + magnitude_bytes;
    const size_t lower_source_index =
        ((k * source_cols + rhs_col) * static_cast<size_t>(n) + j) * width;
    const size_t upper_source_index = lower_source_index + half * width;
    const uint8_t *lower_source = payload + lower_source_index;
    const uint8_t *upper_source = payload + upper_source_index;
    uint64_t lower = compact_mod_magnitude(lower_source + 1, magnitude_bytes, modulus);
    uint64_t upper = compact_mod_magnitude(upper_source + 1, magnitude_bytes, modulus);
    if (lower_source[0] == 2 && lower != 0) lower = modulus - lower;
    if (upper_source[0] == 2 && upper != 0) upper = modulus - upper;
    const size_t twiddle_base = limb * static_cast<size_t>(n);
    lower = mul_mod_shoup_u64(
        lower, twiddles[twiddle_base + j], twiddle_shoup[twiddle_base + j], modulus);
    upper = mul_mod_shoup_u64(
        upper, twiddles[twiddle_base + j + half],
        twiddle_shoup[twiddle_base + j + half], modulus);
    const size_t workspace_base =
        (local_limb * poly_count + poly) * static_cast<size_t>(n);
    workspace[workspace_base + j] = static_cast<Word>(add_mod_u64(lower, upper, modulus));
    const uint64_t difference = sub_mod_u64(lower, upper, modulus);
    const size_t stage_twiddle = twiddle_base + 2 * j;
    workspace[workspace_base + j + half] = static_cast<Word>(mul_mod_shoup_u64(
        difference, twiddles[stage_twiddle], twiddle_shoup[stage_twiddle], modulus));
}

template <class Word>
__global__ void compact_ntt_dif_stage_kernel(
    Word *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t n,
    uint32_t len)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t butterflies_per_poly = n / 2;
    const size_t total = limb_count * poly_count * butterflies_per_poly;
    if (idx >= total) return;
    const size_t butterfly = idx % butterflies_per_poly;
    const size_t q = idx / butterflies_per_poly;
    const size_t poly = q % poly_count;
    const size_t local_limb = q / poly_count;
    const uint32_t half = len >> 1;
    const uint32_t group = static_cast<uint32_t>(butterfly) / half;
    const uint32_t j = static_cast<uint32_t>(butterfly) % half;
    const uint32_t i = group * len + j;
    const size_t limb = limb_offset + local_limb;
    const uint64_t modulus = moduli[limb];
    const size_t base = (local_limb * poly_count + poly) * n;
    const uint64_t lower = workspace[base + i];
    const uint64_t upper = workspace[base + i + half];
    workspace[base + i] = static_cast<Word>(add_mod_u64(lower, upper, modulus));
    const uint64_t difference = sub_mod_u64(lower, upper, modulus);
    const size_t twiddle_index =
        limb * n + 2u * (n / len) * j;
    workspace[base + i + half] = static_cast<Word>(mul_mod_shoup_u64(
        difference, twiddles[twiddle_index], twiddle_shoup[twiddle_index], modulus));
}

template <class Word>
__global__ void compact_rhs_dif_all_shared_kernel(
    const uint8_t *payload,
    Word *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t cols,
    size_t source_cols,
    uint32_t n,
    size_t magnitude_bytes,
    size_t source_column_offset)
{
    __shared__ Word values[kCompactNttSuffixSize];
    const size_t poly = static_cast<size_t>(blockIdx.x);
    const size_t local_limb = static_cast<size_t>(blockIdx.y);
    if (poly >= poly_count || local_limb >= limb_count) return;
    const size_t limb = limb_offset + local_limb;
    const uint64_t modulus = moduli[limb];
    const size_t k = poly / cols;
    const size_t c = poly % cols;
    const size_t rhs_col = source_column_offset + c;
    const size_t width = 1 + magnitude_bytes;
    for (uint32_t coefficient = threadIdx.x; coefficient < n; coefficient += blockDim.x)
    {
        const uint8_t *source =
            payload + ((k * source_cols + rhs_col) * static_cast<size_t>(n) + coefficient) * width;
        uint64_t value = compact_mod_magnitude(source + 1, magnitude_bytes, modulus);
        if (source[0] == 2 && value != 0) value = modulus - value;
        const size_t twiddle_index = limb * static_cast<size_t>(n) + coefficient;
        values[coefficient] = static_cast<Word>(mul_mod_shoup_u64(
            value, twiddles[twiddle_index], twiddle_shoup[twiddle_index], modulus));
    }
    __syncthreads();
    for (uint32_t len = n; len >= 2; len >>= 1)
    {
        const uint32_t half = len >> 1;
        for (uint32_t butterfly = threadIdx.x; butterfly < (n >> 1); butterfly += blockDim.x)
        {
            const uint32_t group = butterfly / half;
            const uint32_t j = butterfly % half;
            const uint32_t i = group * len + j;
            const uint64_t lower = values[i];
            const uint64_t upper = values[i + half];
            values[i] = static_cast<Word>(add_mod_u64(lower, upper, modulus));
            const uint64_t difference = sub_mod_u64(lower, upper, modulus);
            const size_t twiddle_index =
                limb * static_cast<size_t>(n) + 2u * (n / len) * j;
            values[i + half] = static_cast<Word>(mul_mod_shoup_u64(
                difference, twiddles[twiddle_index],
                twiddle_shoup[twiddle_index], modulus));
        }
        __syncthreads();
    }
    const size_t base = (local_limb * poly_count + poly) * static_cast<size_t>(n);
    for (uint32_t coefficient = threadIdx.x; coefficient < n; coefficient += blockDim.x)
        workspace[base + coefficient] = values[coefficient];
}

template <class Word>
__global__ void compact_rhs_dif_suffix_kernel(
    Word *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    uint32_t n)
{
    __shared__ Word values[kCompactNttSuffixSize];
    const size_t segments_per_poly = n / kCompactNttSuffixSize;
    const size_t flat_segment = static_cast<size_t>(blockIdx.x);
    const size_t poly = flat_segment / segments_per_poly;
    const size_t segment = flat_segment % segments_per_poly;
    const size_t local_limb = static_cast<size_t>(blockIdx.y);
    if (poly >= poly_count || local_limb >= limb_count) return;
    const size_t base = (local_limb * poly_count + poly) * static_cast<size_t>(n) +
                        segment * kCompactNttSuffixSize;
    for (uint32_t local = threadIdx.x; local < kCompactNttSuffixSize; local += blockDim.x)
        values[local] = workspace[base + local];
    __syncthreads();
    const size_t limb = limb_offset + local_limb;
    const uint64_t modulus = moduli[limb];
    for (uint32_t len = kCompactNttSuffixSize; len >= 2; len >>= 1)
    {
        const uint32_t half = len >> 1;
        for (uint32_t butterfly = threadIdx.x;
             butterfly < (kCompactNttSuffixSize >> 1);
             butterfly += blockDim.x)
        {
            const uint32_t group = butterfly / half;
            const uint32_t j = butterfly % half;
            const uint32_t i = group * len + j;
            const uint64_t lower = values[i];
            const uint64_t upper = values[i + half];
            values[i] = static_cast<Word>(add_mod_u64(lower, upper, modulus));
            const uint64_t difference = sub_mod_u64(lower, upper, modulus);
            const size_t twiddle_index =
                limb * static_cast<size_t>(n) + 2u * (n / len) * j;
            values[i + half] = static_cast<Word>(mul_mod_shoup_u64(
                difference, twiddles[twiddle_index],
                twiddle_shoup[twiddle_index], modulus));
        }
        __syncthreads();
    }
    for (uint32_t local = threadIdx.x; local < kCompactNttSuffixSize; local += blockDim.x)
        workspace[base + local] = values[local];
}

template <class Word>
__global__ void compact_accumulate_kernel(
    CompactRowBlocks blocks,
    const uint64_t *moduli,
    size_t limb_offset,
    const Word *workspace,
    size_t limb_count,
    size_t rows,
    size_t inner,
    size_t out_cols,
    size_t n,
    bool lazy_reduce)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = limb_count * rows * out_cols * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t q = idx / n;
    const size_t c = q % out_cols;
    size_t row = (q / out_cols) % rows;
    const size_t local = q / (rows * out_cols);
    const size_t global = limb_offset + local;
    const uint64_t modulus = moduli[global];
    const size_t block = blocks.locate(row);
    const size_t first_fragment = block ? blocks.fragment_ends[block - 1] : 0;
    size_t fragment = first_fragment;
    const auto out_descriptor = blocks.outputs[block][global];
    uint64_t acc = 0;
    if (lazy_reduce)
    {
        unsigned __int128 wide_acc = acc;
        for (size_t k = 0; k < inner; ++k)
        {
            while (k >= blocks.column_ends[fragment]) ++fragment;
            const size_t start = fragment == first_fragment ? 0 : blocks.column_ends[fragment - 1];
            const auto lhs_descriptor = blocks.inputs[fragment][global];
            const uint64_t lhs = matrix_load_limb_u64(
                lhs_descriptor.base,
                row * (blocks.column_ends[fragment] - start) + k - start,
                coeff,
                lhs_descriptor.stride,
                lhs_descriptor.width);
            const size_t rhs_index =
                (local * (inner * out_cols) + k * out_cols + c) * n + coeff;
            wide_acc += static_cast<unsigned __int128>(lhs) * workspace[rhs_index];
        }
        acc = static_cast<uint64_t>(wide_acc % modulus);
    }
    else
    {
        for (size_t k = 0; k < inner; ++k)
        {
            while (k >= blocks.column_ends[fragment]) ++fragment;
            const size_t start = fragment == first_fragment ? 0 : blocks.column_ends[fragment - 1];
            const auto lhs_descriptor = blocks.inputs[fragment][global];
            const uint64_t lhs = matrix_load_limb_u64(
                lhs_descriptor.base,
                row * (blocks.column_ends[fragment] - start) + k - start,
                coeff,
                lhs_descriptor.stride,
                lhs_descriptor.width);
            const size_t rhs_index =
                (local * (inner * out_cols) + k * out_cols + c) * n + coeff;
            acc = add_mod_u64(
                acc, mul_mod_u64(lhs, workspace[rhs_index], modulus), modulus);
        }
    }
    matrix_store_limb_u64(
        out_descriptor.base,
        row * out_cols + c,
        coeff,
        out_descriptor.stride,
        out_descriptor.width,
        acc);
}

bool compact_lazy_dot_is_safe(
    const std::vector<uint64_t> &moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t terms)
{
    if (terms == 0 || limb_offset > moduli.size() || limb_count > moduli.size() - limb_offset)
        return false;
    constexpr unsigned __int128 kMaxU128 = ~static_cast<unsigned __int128>(0);
    for (size_t local = 0; local < limb_count; ++local)
    {
        const unsigned __int128 max_residue = moduli[limb_offset + local] - 1;
        const unsigned __int128 max_product = max_residue * max_residue;
        if (max_product > (kMaxU128 - max_residue) / terms) return false;
    }
    return true;
}

__device__ __forceinline__ uint64_t compact_words_mod(
    const uint64_t *words,
    int word_count,
    uint64_t modulus)
{
    uint64_t residue = 0;
    for (int word = word_count; word-- > 0;)
    {
        const unsigned __int128 value =
            (static_cast<unsigned __int128>(residue) << 64) | words[word];
        residue = static_cast<uint64_t>(value % modulus);
    }
    return residue;
}

__global__ void compact_check_pack_preimage_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *limb_descriptors,
    const uint64_t *moduli,
    const uint64_t *garner_inverses,
    int inverse_stride,
    const int *subset_indices,
    int subset_count,
    int limb_count,
    size_t coefficient_count,
    size_t n,
    int words_per_coeff,
    const uint64_t *subset_modulus_words,
    const uint64_t *subset_half_words,
    const uint64_t *bound_words,
    size_t magnitude_bytes,
    MxxPreimageStatus *status,
    uint8_t *staging)
{
    if (status->reserved != 0U) return;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= coefficient_count) return;
    const size_t poly = idx / n;
    const size_t coeff = idx % n;

    uint64_t mixed_digits[kMaxRnsLimbs];
    uint64_t magnitude[kMaxCoeffWords];
    for (int i = 0; i < subset_count; ++i)
    {
        const int limb = subset_indices[i];
        const auto descriptor = limb_descriptors[limb];
        mixed_digits[i] = matrix_load_limb_u64(
                              descriptor.base, poly, coeff,
                              descriptor.stride, descriptor.width) %
                          moduli[limb];
    }
    const size_t inverse_stride_size = static_cast<size_t>(inverse_stride);
    for (int i = 1; i < subset_count; ++i)
    {
        const int limb_i = subset_indices[i];
        const uint64_t qi = moduli[limb_i];
        uint64_t digit = mixed_digits[i];
        for (int j = 0; j < i; ++j)
        {
            const int limb_j = subset_indices[j];
            const uint64_t previous = mixed_digits[j] % qi;
            const uint64_t difference =
                digit >= previous
                    ? digit - previous
                    : static_cast<uint64_t>(
                          static_cast<unsigned __int128>(digit) + qi - previous);
            digit = serde_mul_mod_u64_device(
                difference,
                garner_inverses[static_cast<size_t>(limb_j) * inverse_stride_size +
                                static_cast<size_t>(limb_i)],
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
        uint64_t expected = compact_words_mod(magnitude, words_per_coeff, modulus);
        if (negative && expected != 0) expected = modulus - expected;
        const auto descriptor = limb_descriptors[limb];
        const uint64_t actual = matrix_load_limb_u64(
                                    descriptor.base, poly, coeff,
                                    descriptor.stride, descriptor.width) %
                                modulus;
        valid = actual == expected;
    }
    if (!valid)
    {
        atomicExch(&status->accepted, 0U);
        return;
    }

    const size_t width = magnitude_bytes + 1;
    uint8_t *dst = staging + idx * width;
    bool zero = true;
    for (int word = 0; word < words_per_coeff; ++word) zero = zero && magnitude[word] == 0;
    dst[0] = zero ? 0 : (negative ? 2 : 1);
    for (size_t byte = 0; byte < magnitude_bytes; ++byte)
    {
        const size_t word = byte / sizeof(uint64_t);
        dst[1 + byte] = word < static_cast<size_t>(words_per_coeff)
                            ? static_cast<uint8_t>(magnitude[word] >> (8 * (byte % sizeof(uint64_t))))
                            : 0;
    }
}

// The Rust sampler defines the retry transcript as
//
//   Keccak256("mxx-preimage-sampler/v2" || nonce || instance || column ||
//             len("candidate") || "candidate" || attempt)
//
// with the integer fields encoded little-endian.  Keep this implementation
// local to the allocation-free retry primitive so a conditional body can
// regenerate a candidate seed from device control on every replay.  This is
// Keccak-256 (domain byte 0x01), not SHA3-256 (domain byte 0x06).
__device__ __forceinline__ uint64_t preimage_seed_rotl64(uint64_t value, int count)
{
    return count == 0 ? value : (value << count) | (value >> (64 - count));
}

__device__ __forceinline__ void preimage_seed_keccak_f(uint64_t state[25])
{
    constexpr uint64_t round_constants[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL,
    };
    constexpr int rotation[25] = {
        0, 1, 62, 28, 27,
        36, 44, 6, 55, 20,
        3, 10, 43, 25, 39,
        41, 45, 15, 21, 8,
        18, 2, 61, 56, 14,
    };

    for (int round = 0; round < 24; ++round)
    {
        uint64_t column_parity[5];
        for (int x = 0; x < 5; ++x)
        {
            column_parity[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^
                               state[x + 15] ^ state[x + 20];
        }
        uint64_t theta[5];
        for (int x = 0; x < 5; ++x)
        {
            theta[x] = column_parity[(x + 4) % 5] ^
                       preimage_seed_rotl64(column_parity[(x + 1) % 5], 1);
        }
        for (int x = 0; x < 5; ++x)
        {
            for (int y = 0; y < 5; ++y) state[x + 5 * y] ^= theta[x];
        }

        uint64_t rho_pi[25];
        for (int x = 0; x < 5; ++x)
        {
            for (int y = 0; y < 5; ++y)
            {
                const int destination_x = y;
                const int destination_y = (2 * x + 3 * y) % 5;
                rho_pi[destination_x + 5 * destination_y] =
                    preimage_seed_rotl64(state[x + 5 * y], rotation[x + 5 * y]);
            }
        }
        for (int x = 0; x < 5; ++x)
        {
            for (int y = 0; y < 5; ++y)
            {
                state[x + 5 * y] = rho_pi[x + 5 * y] ^
                    ((~rho_pi[(x + 1) % 5 + 5 * y]) & rho_pi[(x + 2) % 5 + 5 * y]);
            }
        }
        state[0] ^= round_constants[round];
    }
}

__device__ __forceinline__ uint64_t preimage_seed_load_le64(const uint8_t *bytes)
{
    uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte)
        value |= static_cast<uint64_t>(bytes[byte]) << (8 * byte);
    return value;
}

__device__ __forceinline__ void preimage_seed_from_control_stage_body(
    const MxxPreimageLaunchControl *control,
    gpu_chacha::GpuRngSeed *seed,
    uint32_t stage_id)
{
    if (blockIdx.x != 0 || threadIdx.x != 0 || !control || !seed) return;

    // The transcript is 92 bytes, so one 136-byte Keccak rate block suffices.
    uint8_t message[136] = {};
    size_t offset = 0;
    const bool host_stage_abi = stage_id <= 2U;
    constexpr char v1_prefix[] = "mxx-preimage-sampler/v1";
    constexpr char v2_prefix[] = "mxx-preimage-sampler/v2";
    const char *prefix = host_stage_abi ? v1_prefix : v2_prefix;
    const size_t prefix_length = host_stage_abi ? sizeof(v1_prefix) - 1 : sizeof(v2_prefix) - 1;
    for (size_t i = 0; i < prefix_length; ++i) message[offset++] = static_cast<uint8_t>(prefix[i]);
    for (size_t i = 0; i < sizeof(control->execution_nonce); ++i)
        message[offset++] = control->execution_nonce[i];
    const char *stage = nullptr;
    size_t stage_length = 0;
    if (stage_id == 0U) { stage = "p2"; stage_length = 2; }
    else if (stage_id == 1U) { stage = "p1"; stage_length = 2; }
    else if (stage_id == 2U) { stage = "z"; stage_length = 1; }
    else { stage = "candidate"; stage_length = 9; }
    if (!host_stage_abi)
    {
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(control->logical_instance >> (8 * byte));
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(control->global_column_start >> (8 * byte));
    }
    for (int byte = 0; byte < 8; ++byte)
        message[offset++] = static_cast<uint8_t>(stage_length >> (8 * byte));
    for (size_t i = 0; i < stage_length; ++i) message[offset++] = static_cast<uint8_t>(stage[i]);
    if (host_stage_abi)
    {
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(control->global_column_start >> (8 * byte));
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(control->attempt >> (8 * byte));
    }
    else
    {
        for (int byte = 0; byte < 4; ++byte)
            message[offset++] = static_cast<uint8_t>(control->attempt >> (8 * byte));
    }

    // Keccak's pad10*1 with the Keccak domain suffix 0x01.
    message[offset] ^= 0x01U;
    message[135] ^= 0x80U;
    uint64_t state[25] = {};
    constexpr size_t rate_lanes = 17;
    for (size_t lane = 0; lane < rate_lanes; ++lane)
        state[lane] ^= preimage_seed_load_le64(message + lane * sizeof(uint64_t));
    preimage_seed_keccak_f(state);
    for (int word = 0; word < 4; ++word) seed->words[word] = state[word];
}

__global__ void preimage_seed_from_control_stage_kernel(
    const MxxPreimageLaunchControl *control,
    gpu_chacha::GpuRngSeed *seed,
    uint32_t stage_id)
{
    preimage_seed_from_control_stage_body(control, seed, stage_id);
}

__global__ void preimage_status_init_kernel(MxxPreimageStatus *status, uint32_t attempt)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        if (attempt == 0U) status->reserved = 0U;
        if (status->reserved == 0U)
        {
            status->attempts = attempt + 1U;
            status->accepted = 1U;
        }
        status->error_code = MXX_PREIMAGE_SUCCESS;
    }
}

__global__ void preimage_status_init_control_kernel(
    MxxPreimageStatus *status,
    const MxxPreimageLaunchControl *control)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && control)
    {
        const uint32_t attempt = control->attempt;
        if (attempt == 0U) status->reserved = 0U;
        if (status->reserved == 0U)
        {
            status->attempts = attempt + 1U;
            status->accepted = 1U;
        }
        status->error_code = MXX_PREIMAGE_SUCCESS;
    }
}

__global__ void preimage_status_latch_accept_kernel(MxxPreimageStatus *status)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && status->accepted == 1U)
        status->reserved = 1U;
}

__global__ void preimage_status_mark_exhausted_kernel(MxxPreimageStatus *status)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && status->accepted != 1U)
        status->error_code = MXX_PREIMAGE_EXHAUSTED;
}

__global__ void compact_commit_preimage_tile_if_accepted_kernel(
    uint8_t *payload,
    const uint8_t *staging,
    const MxxPreimageStatus *status,
    size_t n,
    size_t rows,
    size_t tile_cols,
    size_t dst_cols,
    size_t dst_row,
    size_t dst_col,
    size_t width)
{
    if (status->accepted != 1U || status->error_code != MXX_PREIMAGE_SUCCESS) return;
    const size_t idx = static_cast<size_t>(blockIdx.x) * kSmallThreads + threadIdx.x;
    const size_t total = rows * tile_cols * n * width;
    if (idx >= total) return;
    const size_t byte = idx % width;
    const size_t coefficient = idx / width;
    const size_t coeff = coefficient % n;
    const size_t column = (coefficient / n) % tile_cols;
    const size_t row = coefficient / (n * tile_cols);
    const size_t dst_index = (((dst_row + row) * dst_cols + dst_col + column) * n + coeff) * width + byte;
    payload[dst_index] = staging[idx];
}

__global__ void compact_commit_preimage_tile_kernel(
    uint8_t *payload,
    const uint8_t *staging,
    size_t n,
    size_t rows,
    size_t tile_cols,
    size_t dst_cols,
    size_t dst_row,
    size_t dst_col,
    size_t width)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * tile_cols * n * width;
    if (idx >= total) return;
    const size_t byte = idx % width;
    const size_t coefficient = idx / width;
    const size_t coeff = coefficient % n;
    const size_t poly = coefficient / n;
    const size_t row = poly / tile_cols;
    const size_t col = poly % tile_cols;
    const size_t dst_index =
        (((dst_row + row) * dst_cols + dst_col + col) * n + coeff) * width + byte;
    payload[dst_index] = staging[idx];
}

void compact_trim_words(std::vector<uint64_t> *words)
{
    while (words->size() > 1 && words->back() == 0) words->pop_back();
}

int compact_compare_words(
    const std::vector<uint64_t> &lhs,
    const std::vector<uint64_t> &rhs)
{
    size_t lhs_size = lhs.size();
    while (lhs_size > 1 && lhs[lhs_size - 1] == 0) --lhs_size;
    size_t rhs_size = rhs.size();
    while (rhs_size > 1 && rhs[rhs_size - 1] == 0) --rhs_size;
    if (lhs_size != rhs_size) return lhs_size < rhs_size ? -1 : 1;
    for (size_t word = lhs_size; word-- > 0;)
        if (lhs[word] != rhs[word]) return lhs[word] < rhs[word] ? -1 : 1;
    return 0;
}

bool compact_double_words(
    const uint64_t *words,
    size_t word_count,
    std::vector<uint64_t> *out)
{
    out->assign(words, words + word_count);
    uint64_t carry = 0;
    for (uint64_t &word : *out)
    {
        const uint64_t next = word >> 63;
        word = (word << 1) | carry;
        carry = next;
    }
    if (carry) out->push_back(carry);
    compact_trim_words(out);
    return true;
}

void small_release_hard_cutoff_plan(GpuSmallMatrix *mat, cudaStream_t stream)
{
    if (!mat) return;
    if (mat->hard_cutoff_decision_ready && mat->hard_cutoff_decision_recorded && stream)
    {
        // The final D2H status copy is ordered by decision_ready.  Preserve
        // that dependency before handing the pinned page to the reclaimer.
        if (cudaStreamWaitEvent(stream, mat->hard_cutoff_decision_ready, 0) != cudaSuccess)
            set_error(cudaGetLastError());
    }
    if (mat->hard_cutoff_device_status)
        cudaFreeAsync(mat->hard_cutoff_device_status, stream);
    if (mat->hard_cutoff_staging)
        cudaFreeAsync(mat->hard_cutoff_staging, stream);
    if (mat->hard_cutoff_metadata)
    {
        mat->hard_cutoff_metadata.reset();
    }
    else
    {
        // Failed partial initialization has not transferred metadata ownership.
        if (mat->hard_cutoff_bound_words) cudaFreeAsync(mat->hard_cutoff_bound_words, stream);
        if (mat->hard_cutoff_half_modulus_words)
            cudaFreeAsync(mat->hard_cutoff_half_modulus_words, stream);
        if (mat->hard_cutoff_modulus_words)
            cudaFreeAsync(mat->hard_cutoff_modulus_words, stream);
        if (mat->hard_cutoff_subset_indices)
            cudaFreeAsync(mat->hard_cutoff_subset_indices, stream);
        if (mat->hard_cutoff_garner_inverses)
            cudaFreeAsync(mat->hard_cutoff_garner_inverses, stream);
    }
    if (mat->hard_cutoff_decision_ready)
        cudaEventDestroy(mat->hard_cutoff_decision_ready);
    if (mat->hard_cutoff_host_status)
    {
        void *host_status = mat->hard_cutoff_host_status;
        if (mat->ctx && stream && mat->device >= 0)
        {
            // The reclaimer records a completion event on the dependency
            // ordered release stream and only then calls cudaFreeHost.
            if (gpu_defer_pinned_frees(mat->ctx, mat->device, stream, &host_status, 1) != 0)
                set_error("failed to defer compact preimage status page release");
        }
        else
        {
            // Partial construction has not submitted a status copy.
            cudaFreeHost(host_status);
        }
    }
    mat->hard_cutoff_device_status = nullptr;
    mat->hard_cutoff_staging = nullptr;
    mat->hard_cutoff_staging_bytes = 0;
    mat->hard_cutoff_bound_words = nullptr;
    mat->hard_cutoff_half_modulus_words = nullptr;
    mat->hard_cutoff_modulus_words = nullptr;
    mat->hard_cutoff_subset_indices = nullptr;
    mat->hard_cutoff_garner_inverses = nullptr;
    mat->hard_cutoff_decision_ready = nullptr;
    mat->hard_cutoff_decision_recorded = false;
    mat->hard_cutoff_host_status = nullptr;
}

int small_initialize_hard_cutoff_plan(
    GpuSmallMatrix *mat,
    size_t scratch_rows,
    size_t scratch_cols)
{
    if (!mat || !mat->ctx || !mat->stream || mat->device < 0 || mat->bound_words.empty())
        return set_error("invalid compact hard-cutoff plan owner");
    const size_t limb_count = mat->ctx->moduli.size();
    if (limb_count == 0 || limb_count > static_cast<size_t>(kMaxRnsLimbs) ||
        mat->ctx->garner_inverse_table.size() != limb_count * limb_count)
        return set_error("invalid compact hard-cutoff CRT basis");

    std::vector<uint64_t> doubled_bound;
    compact_double_words(mat->bound_words.data(), mat->bound_words.size(), &doubled_bound);
    std::vector<int> subset_indices;
    std::vector<uint64_t> subset_moduli;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        if (compact_compare_words(
                std::vector<uint64_t>{mat->ctx->moduli[limb]}, doubled_bound) > 0)
        {
            subset_indices.push_back(static_cast<int>(limb));
            subset_moduli.push_back(mat->ctx->moduli[limb]);
            break;
        }
    }
    std::vector<uint64_t> modulus_words;
    if (subset_indices.empty())
    {
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            subset_indices.push_back(static_cast<int>(limb));
            subset_moduli.push_back(mat->ctx->moduli[limb]);
            if (!serde_compute_modulus_words_le(subset_moduli, &modulus_words))
                return set_error("failed to compute compact partial CRT modulus");
            if (compact_compare_words(modulus_words, doubled_bound) > 0) break;
        }
    }
    if (modulus_words.empty() &&
        !serde_compute_modulus_words_le(subset_moduli, &modulus_words))
        return set_error("failed to compute compact anchor modulus");
    if (compact_compare_words(modulus_words, doubled_bound) <= 0)
        return set_error("compact CRT modulus must exceed twice the hard cutoff");
    const size_t words_per_coeff = std::max(modulus_words.size(), mat->bound_words.size());
    if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
        return set_error("compact partial CRT width exceeds supported maximum");
    for (size_t word = (mat->magnitude_bytes + 7) / 8;
         word < mat->bound_words.size(); ++word)
        if (mat->bound_words[word] != 0)
            return set_error("compact hard cutoff exceeds magnitude width");
    if (mat->magnitude_bytes % 8 != 0 &&
        mat->bound_words.size() > mat->magnitude_bytes / 8 &&
        (mat->bound_words[mat->magnitude_bytes / 8] >>
         (8 * (mat->magnitude_bytes % 8))) != 0)
        return set_error("compact hard cutoff exceeds magnitude width");

    std::vector<uint64_t> half_modulus_words = modulus_words;
    serde_shift_words_right_one_le(&half_modulus_words);
    modulus_words.resize(words_per_coeff, 0);
    half_modulus_words.resize(words_per_coeff, 0);
    std::vector<uint64_t> padded_bound(words_per_coeff, 0);
    std::copy(mat->bound_words.begin(), mat->bound_words.end(), padded_bound.begin());

    std::vector<void *> pinned_uploads;
    auto upload = [&](auto **device_out, const auto *source, size_t count) -> cudaError_t {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*device_out)>>;
        const size_t bytes = count * sizeof(T);
        void *pinned = nullptr;
        cudaError_t err = cudaHostAlloc(&pinned, bytes, cudaHostAllocPortable);
        if (err != cudaSuccess) return err;
        std::memcpy(pinned, source, bytes);
        err = cudaMallocAsync(reinterpret_cast<void **>(device_out), bytes, mat->stream);
        if (err == cudaSuccess)
            err = cudaMemcpyAsync(*device_out, pinned, bytes, cudaMemcpyHostToDevice, mat->stream);
        if (err != cudaSuccess)
        {
            cudaFreeHost(pinned);
            return err;
        }
        pinned_uploads.push_back(pinned);
        return cudaSuccess;
    };

    cudaError_t err = upload(
        &mat->hard_cutoff_garner_inverses,
        mat->ctx->garner_inverse_table.data(),
        mat->ctx->garner_inverse_table.size());
    if (err == cudaSuccess)
        err = upload(&mat->hard_cutoff_subset_indices,
                     subset_indices.data(), subset_indices.size());
    if (err == cudaSuccess)
        err = upload(&mat->hard_cutoff_modulus_words,
                     modulus_words.data(), modulus_words.size());
    if (err == cudaSuccess)
        err = upload(&mat->hard_cutoff_half_modulus_words,
                     half_modulus_words.data(), half_modulus_words.size());
    if (err == cudaSuccess)
        err = upload(&mat->hard_cutoff_bound_words,
                     padded_bound.data(), padded_bound.size());
    if (!pinned_uploads.empty() &&
        gpu_defer_pinned_frees(
            mat->ctx, mat->device, mat->stream,
            pinned_uploads.data(), pinned_uploads.size()) != 0)
        return 1;
    if (err != cudaSuccess) return set_error(err);
    mat->hard_cutoff_metadata = std::make_shared<SmallHardCutoffMetadata>();
    mat->hard_cutoff_metadata->device = mat->device;
    mat->hard_cutoff_metadata->stream = mat->stream;
    mat->hard_cutoff_metadata->garner_inverses = mat->hard_cutoff_garner_inverses;
    mat->hard_cutoff_metadata->subset_indices = mat->hard_cutoff_subset_indices;
    mat->hard_cutoff_metadata->modulus_words = mat->hard_cutoff_modulus_words;
    mat->hard_cutoff_metadata->half_modulus_words = mat->hard_cutoff_half_modulus_words;
    mat->hard_cutoff_metadata->bound_words = mat->hard_cutoff_bound_words;
    size_t staging_coefficients = 0;
    size_t staging_bytes = 0;
    if (!small_mul_size(scratch_rows, scratch_cols, &staging_coefficients) ||
        !small_mul_size(staging_coefficients, mat->n, &staging_coefficients) ||
        !small_mul_size(staging_coefficients, 1 + mat->magnitude_bytes, &staging_bytes) ||
        staging_bytes == 0)
        return set_error("compact preimage scratch size overflow");
    err = cudaMallocAsync(
        reinterpret_cast<void **>(&mat->hard_cutoff_staging), staging_bytes, mat->stream);
    if (err == cudaSuccess)
        err = cudaHostAlloc(
            reinterpret_cast<void **>(&mat->hard_cutoff_host_status),
            sizeof(MxxPreimageStatus), cudaHostAllocPortable);
    if (err == cudaSuccess)
        err = cudaMallocAsync(
            reinterpret_cast<void **>(&mat->hard_cutoff_device_status),
            sizeof(MxxPreimageStatus), mat->stream);
    if (err == cudaSuccess)
        err = cudaEventCreateWithFlags(
            &mat->hard_cutoff_decision_ready, cudaEventDisableTiming);
    if (err != cudaSuccess) return set_error(err);
    mat->hard_cutoff_staging_bytes = staging_bytes;
    mat->hard_cutoff_limb_count = limb_count;
    mat->hard_cutoff_subset_count = static_cast<int>(subset_indices.size());
    mat->hard_cutoff_words_per_coeff = static_cast<int>(words_per_coeff);
    err = cudaMemsetAsync(
        mat->hard_cutoff_device_status, 0, sizeof(MxxPreimageStatus), mat->stream);
    return err == cudaSuccess ? 0 : set_error(err);
}

}

extern "C" int gpu_preimage_seed_from_control_stage_async(
    GpuContext *ctx,
    const void *device_control,
    void *device_seed,
    uint32_t stage_id,
    void *stream_raw,
    uint32_t control_binding_index,
    uint32_t seed_binding_index)
{
    if (!ctx || !device_control || !device_seed || !stream_raw || stage_id > 2U)
        return set_error("invalid preimage stage seed derivation arguments");
    if (control_binding_index == seed_binding_index)
        return set_error("preimage stage seed bindings must be distinct");
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    preimage_seed_from_control_stage_kernel<<<1, 1, 0, stream>>>(
        static_cast<const MxxPreimageLaunchControl *>(device_control),
        static_cast<gpu_chacha::GpuRngSeed *>(device_seed), stage_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return set_error(err);
    const size_t argument_sizes[] = {sizeof(void *), sizeof(void *), sizeof(uint32_t)};
    const MxxGraphPatch patches[] = {
        small_kernel_pointer_patch(0, 0, control_binding_index),
        small_kernel_pointer_patch(1, 0, seed_binding_index),
    };
    return mxx_graph_register_kernel_update_for_stream(
        ctx, stream_raw, argument_sizes, std::size(argument_sizes), patches, std::size(patches));
}

extern "C" int gpu_small_matrix_create(
    GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    const uint64_t *bound_words,
    size_t bound_word_count,
    GpuSmallMatrix **out)
{
    if (!ctx || !out || !bound_words || bound_word_count == 0 || magnitude_bytes == 0)
        return set_error("invalid gpu_small_matrix_create arguments");
    *out = nullptr;
    if (ctx->gpu_ids.empty() || ctx->N <= 0 || magnitude_bytes > 255)
        return set_error("invalid compact matrix context or width");
    auto *mat = new GpuSmallMatrix();
    mat->ctx = ctx;
    mat->rows = rows;
    mat->cols = cols;
    mat->storage_cols = cols;
    mat->n = static_cast<size_t>(ctx->N);
    mat->magnitude_bytes = magnitude_bytes;
    mat->bound_words.assign(bound_words, bound_words + bound_word_count);
    if (small_payload_size(rows, cols, mat->n, magnitude_bytes, &mat->payload_bytes) != 0)
    {
        delete mat;
        return 1;
    }
    mat->resident_payload_bytes = mat->payload_bytes;
    mat->device = ctx->gpu_ids.front();
    if (ctx->execution->compute_streams_by_partition.empty() || ctx->execution->compute_streams_by_partition.front().empty())
    {
        delete mat;
        return set_error("missing compact matrix stream");
    }
    mat->stream = ctx->execution->compute_streams_by_partition.front().front();
    cudaError_t err = cudaSetDevice(mat->device);
    if (err == cudaSuccess)
        err = cudaEventCreateWithFlags(&mat->write_done, cudaEventDisableTiming);
    if (err == cudaSuccess)
        err = cudaMallocAsync(reinterpret_cast<void **>(&mat->payload), mat->payload_bytes, mat->stream);
    if (err == cudaSuccess)
        err = cudaEventRecord(mat->write_done, mat->stream);
    if (err != cudaSuccess)
    {
        if (mat->payload) cudaFreeAsync(mat->payload, mat->stream);
        if (mat->write_done) cudaEventDestroy(mat->write_done);
        delete mat;
        return set_error(err);
    }
    mat->write_done_valid = true;
    *out = mat;
    return 0;
}

extern "C" int gpu_small_matrix_query_allocation_bytes(
    const GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    GpuMatrixAllocationBytes *out)
{
    if (!ctx || !out || rows == 0 || cols == 0 || magnitude_bytes == 0)
        return set_error("invalid compact matrix allocation query arguments");
    if (ctx->N <= 0 || magnitude_bytes > 255)
        return set_error("invalid compact matrix allocation query context or width");
    size_t payload_bytes = 0;
    if (small_payload_size(
            rows, cols, static_cast<size_t>(ctx->N), magnitude_bytes, &payload_bytes) != 0)
        return 1;
    // The compact owner has one device payload.  Its stream event and bound
    // words are host-side lifetime state, matching GpuSmallMatrix::allocation_bytes.
    *out = GpuMatrixAllocationBytes{};
    out->data_bytes = payload_bytes;
    out->total_bytes = payload_bytes;
    return 0;
}

extern "C" int gpu_small_matrix_binding_descriptor(
    const GpuSmallMatrix *mat,
    GpuSmallMatrixBindingDescriptor *out)
{
    if (!mat || !out || !mat->ctx)
    {
        return set_error("invalid gpu_small_matrix_binding_descriptor arguments");
    }
    // A column view borrows the owner's allocation but its payload begins at
    // the view's first column.  Return the address of that logical payload,
    // not the owner's base pointer, so graph bindings patch the exact byte
    // range consumed by the compact launch.
    const size_t coefficient_bytes = 1 + mat->magnitude_bytes;
    const size_t payload_offset = mat->column_offset * mat->n * coefficient_bytes;
    void *payload = mat->payload ? mat->payload + payload_offset : nullptr;
    *out = GpuSmallMatrixBindingDescriptor{
        mat->device,
        mat->rows,
        mat->cols,
        mat->n,
        mat->magnitude_bytes,
        mat->payload_bytes,
        payload,
        mat->hard_cutoff_device_status,
        mat->hard_cutoff_host_status,
        mat->hard_cutoff_staging,
        mat->hard_cutoff_staging_bytes};
    return 0;
}

extern "C" void gpu_small_matrix_destroy(GpuSmallMatrix *mat)
{
    if (!mat) return;
    if (mxx_graph_retain_released_resource(mat->ctx, mat, [](void *resource) {
        gpu_small_matrix_destroy(static_cast<GpuSmallMatrix *>(resource));
    })) return;
    if (mat->device >= 0 && cudaSetDevice(mat->device) == cudaSuccess)
    {
        cudaStream_t release_stream = small_capture_stream(mat);
        const size_t partition = 0;
        if (mat->ctx && partition < mat->ctx->execution->release_streams_by_partition.size() &&
            mat->ctx->execution->release_streams_by_partition[partition])
        {
            release_stream = matrix_capture_stream_for_device(
                mat->ctx,
                mat->device,
                mat->ctx->execution->release_streams_by_partition[partition]);
            if (mat->write_done_valid) cudaStreamWaitEvent(release_stream, mat->write_done, 0);
        }
        if (mat->owns_payload)
        {
            small_release_hard_cutoff_plan(mat, release_stream);
            if (mat->payload && release_stream) cudaFreeAsync(mat->payload, release_stream);
        }
        if (mat->owns_write_event && mat->write_done) cudaEventDestroy(mat->write_done);
        if (mat->capture_write_done)
            mxx_gpu_capture_event_release(mat->capture_write_done);
    }
    delete mat;
}

extern "C" int gpu_small_matrix_wait(const GpuSmallMatrix *mat)
{
    if (!mat || !mat->write_done)
        return set_error("invalid compact matrix wait arguments");
    if (small_set_device(mat) != 0) return 1;
    if (!mat->write_done_valid) return 0;
    const cudaError_t err = cudaEventSynchronize(mat->write_done);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_wait_compiled_inputs(
    const GpuSmallMatrix *mat,
    int consumer_device,
    void *consumer_stream_raw)
{
    if (!mat || consumer_device < 0 || !consumer_stream_raw)
    {
        return set_error("invalid gpu_small_matrix_wait_compiled_inputs arguments");
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(consumer_stream_raw);
    if (matrix_stream_is_capturing(stream)) return 0;
    const cudaError_t device_status = cudaSetDevice(consumer_device);
    if (device_status != cudaSuccess)
    {
        return set_error(device_status);
    }
    return small_wait(mat, stream);
}

extern "C" int gpu_small_matrix_track_compiled_consumer(
    const GpuSmallMatrix *mat,
    void *consumer_stream_raw,
    void *completion_event_raw)
{
    if (!mat || !consumer_stream_raw || !completion_event_raw)
    {
        return set_error("invalid gpu_small_matrix_track_compiled_consumer arguments");
    }
    return small_track_consumer(
        mat,
        reinterpret_cast<cudaStream_t>(consumer_stream_raw),
        reinterpret_cast<cudaEvent_t>(completion_event_raw));
}

extern "C" int gpu_small_matrix_record_compiled_write(
    GpuSmallMatrix *mat,
    void *stream_raw)
{
    if (!mat || !stream_raw)
    {
        return set_error("invalid gpu_small_matrix_record_compiled_write arguments");
    }
    if (small_set_device(mat) != 0)
    {
        return 1;
    }
    return small_record(mat, reinterpret_cast<cudaStream_t>(stream_raw));
}

extern "C" int gpu_small_matrix_prepare_external_for_capture(GpuSmallMatrix *mat)
{
    if (!mat || !mat->write_done)
    {
        return set_error("invalid compact capture preparation arguments");
    }
    if (matrix_stream_is_capturing(mat->stream))
    {
        return set_error("external capture preparation must precede CUDA capture");
    }
    if (gpu_small_matrix_wait(mat) != 0)
    {
        return 1;
    }
    // Preserve the event object for writes recorded during capture; only the
    // pre-capture writer dependency is cleared.
    mat->write_done_valid = false;
    return 0;
}

extern "C" int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src)
{
    if (!out || !src || out->ctx != src->ctx || out->rows != src->rows || out->cols != src->cols ||
        out->n != src->n || out->magnitude_bytes != src->magnitude_bytes || out->payload_bytes != src->payload_bytes)
        return set_error("incompatible compact matrix copy");
    const cudaStream_t stream = small_capture_stream(out);
    if (small_set_device(out) != 0 || small_wait(src, stream) != 0) return 1;
    const size_t row_bytes = out->cols * out->n * (1 + out->magnitude_bytes);
    const size_t out_pitch = out->storage_cols * out->n * (1 + out->magnitude_bytes);
    const size_t src_pitch = src->storage_cols * src->n * (1 + src->magnitude_bytes);
    auto *destination = out->payload + out->column_offset * out->n * (1 + out->magnitude_bytes);
    const auto *source = src->payload + src->column_offset * src->n * (1 + src->magnitude_bytes);
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, out_pitch, source, src_pitch, row_bytes, out->rows,
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, stream) != 0) return 1;
    return small_track_consumer(src, stream, out->write_done);
}

extern "C" int gpu_small_matrix_copy_cross_context(GpuSmallMatrix *out, const GpuSmallMatrix *src)
{
    if (!out || !src || out->device != src->device || out->rows != src->rows ||
        out->cols != src->cols || out->n != src->n ||
        out->magnitude_bytes != src->magnitude_bytes || out->payload_bytes != src->payload_bytes)
        return set_error("incompatible cross-context compact matrix copy");
    const cudaStream_t stream = small_capture_stream(out);
    if (small_set_device(out) != 0 || small_wait(src, stream) != 0) return 1;
    const size_t row_bytes = out->cols * out->n * (1 + out->magnitude_bytes);
    const size_t out_pitch = out->storage_cols * out->n * (1 + out->magnitude_bytes);
    const size_t src_pitch = src->storage_cols * src->n * (1 + src->magnitude_bytes);
    auto *destination = out->payload + out->column_offset * out->n * (1 + out->magnitude_bytes);
    const auto *source = src->payload + src->column_offset * src->n * (1 + src->magnitude_bytes);
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, out_pitch, source, src_pitch, row_bytes, out->rows,
        cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, stream) != 0) return 1;
    return small_track_consumer(src, stream, out->write_done);
}

extern "C" int gpu_small_matrix_copy_columns(
    GpuSmallMatrix *out,
    const GpuSmallMatrix *src,
    size_t source_column_start)
{
    if (!out || !src || out->ctx != src->ctx || out->rows != src->rows ||
        out->n != src->n || out->magnitude_bytes != src->magnitude_bytes ||
        source_column_start > src->cols || out->cols > src->cols - source_column_start)
        return set_error("incompatible compact matrix column slice");
    const cudaStream_t stream = small_capture_stream(out);
    if (small_set_device(out) != 0 || small_wait(src, stream) != 0) return 1;
    const size_t coefficient_bytes = 1 + out->magnitude_bytes;
    const size_t column_bytes = out->n * coefficient_bytes;
    const size_t destination_pitch = out->storage_cols * column_bytes;
    const size_t source_pitch = src->storage_cols * column_bytes;
    auto *destination = out->payload + out->column_offset * column_bytes;
    const auto *source = src->payload + (src->column_offset + source_column_start) * column_bytes;
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, destination_pitch, source, source_pitch,
        out->cols * column_bytes, out->rows, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, stream) != 0) return 1;
    return small_track_consumer(src, stream, out->write_done);
}

extern "C" int gpu_small_matrix_view_columns(
    const GpuSmallMatrix *src,
    size_t source_column_start,
    size_t columns,
    GpuSmallMatrix **out)
{
    if (!src || !out || columns == 0 || source_column_start > src->cols ||
        columns > src->cols - source_column_start)
        return set_error("invalid compact matrix column view");
    *out = nullptr;
    auto *view = new GpuSmallMatrix();
    view->ctx = src->ctx;
    view->device = src->device;
    view->rows = src->rows;
    view->cols = columns;
    view->storage_cols = src->storage_cols;
    view->column_offset = src->column_offset + source_column_start;
    view->n = src->n;
    view->magnitude_bytes = src->magnitude_bytes;
    view->payload = src->payload;
    view->bound_words = src->bound_words;
    view->resident_payload_bytes = src->resident_payload_bytes;
    view->owns_payload = false;
    view->owns_write_event = false;
    view->stream = src->stream;
    view->write_done = src->write_done;
    view->write_done_valid = src->write_done_valid;
    if (small_payload_size(view->rows, view->cols, view->n, view->magnitude_bytes,
                           &view->payload_bytes) != 0)
    {
        delete view;
        return 1;
    }
    *out = view;
    return 0;
}

extern "C" int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat, const uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes || !mat->owns_payload)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0) return 1;
    if (payload_len == 0) return 0;
    const cudaStream_t stream = small_capture_stream(mat);
    uint8_t *staging = nullptr;
    cudaError_t err = cudaHostAlloc(
        reinterpret_cast<void **>(&staging), payload_len, cudaHostAllocPortable);
    if (err != cudaSuccess) return set_error(err);
    std::memcpy(staging, payload, payload_len);
    err = cudaMemcpyAsync(
        mat->payload, staging, payload_len, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeHost(staging);
        return set_error(err);
    }
    const int record_status = small_record(mat, stream);
    void *deferred[] = {staging};
    const int defer_status = gpu_defer_pinned_frees(
        mat->ctx, mat->device, stream, deferred, 1);
    if (record_status != 0)
    {
        (void)small_fence_stream_with_event(stream);
        return record_status;
    }
    return defer_status;
}

extern "C" int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat, uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    const cudaStream_t stream = small_capture_stream(mat);
    if (small_set_device(mat) != 0 || small_wait(mat, stream) != 0) return 1;
    if (payload_len == 0) return 0;
    const size_t coefficient_bytes = 1 + mat->magnitude_bytes;
    const size_t row_bytes = mat->cols * mat->n * coefficient_bytes;
    const size_t source_pitch = mat->storage_cols * mat->n * coefficient_bytes;
    const auto *source = mat->payload + mat->column_offset * mat->n * coefficient_bytes;
    cudaError_t err = cudaMemcpy2DAsync(
        payload, row_bytes, source, source_pitch, row_bytes, mat->rows,
        cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) err = small_fence_stream_with_event(stream);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_decompose_base(
    const GpuMatrix *const *sources,
    size_t block_count,
    uint32_t base_bits,
    int small_mode,
    const uint64_t *max_coefficient_bound,
    size_t bound_word_count,
    GpuSmallMatrix *out,
    size_t dropped_moduli,
    const MxxDecomposeFragmentRange *ranges)
{
    if (!sources || block_count == 0 || block_count > kCompactRowBlocks)
        return set_error("invalid compact decomposition blocks");
    const GpuMatrix *src = sources[0];
    if (!src || !out || !src->ctx || src->ctx != out->ctx || !max_coefficient_bound ||
        bound_word_count == 0 || base_bits == 0 || base_bits >= 63 ||
        (small_mode != 0 && small_mode != 1) || src->format != GPU_POLY_FORMAT_COEFF)
        return set_error("invalid compact decomposition arguments");
    const size_t limbs = static_cast<size_t>(src->level + 1);
    if (src->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || src->ctx->limb_gpu_ids.size() < limbs)
        return set_error("invalid compact decomposition level");
    CompactDecomposeFragments blocks{};
    size_t rows = 0;
    size_t poly_count = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *source = sources[block];
        if (!source || source->ctx != src->ctx || source->level != src->level ||
            source->format != GPU_POLY_FORMAT_COEFF)
            return set_error("incompatible compact decomposition block");
        const MxxDecomposeFragmentRange range = ranges
            ? ranges[block] : MxxDecomposeFragmentRange{0, rows, 0, source->cols};
        size_t row_end = 0, polynomials = 0;
        if (range.columns == 0 || range.source_column > source->cols ||
            range.columns > source->cols - range.source_column ||
            range.destination_column > out->cols || range.columns > out->cols - range.destination_column ||
            !small_add_size(range.destination_row, source->rows, &row_end) ||
            !small_mul_size(source->rows, range.columns, &polynomials) ||
            !small_add_size(poly_count, polynomials, &poly_count))
            return set_error("invalid compact decomposition fragment range");
        for (size_t previous = 0; previous < block; ++previous)
        {
            const auto other = blocks.ranges[previous];
            if (range.destination_row < other.destination_row + sources[previous]->rows &&
                other.destination_row < row_end &&
                range.destination_column < other.destination_column + other.columns &&
                other.destination_column < range.destination_column + range.columns)
                return set_error("overlapping compact decomposition fragments");
        }
        rows = std::max(rows, row_end);
        blocks.ends[block] = poly_count;
        blocks.source_columns[block] = source->cols;
        blocks.ranges[block] = range;
    }
    uint32_t crt_bits = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
        crt_bits = std::max(crt_bits, bit_width_u64(src->ctx->moduli[limb]));
    const size_t digits = (crt_bits + base_bits - 1) / base_bits;
    const bool small = small_mode != 0;
    if (dropped_moduli >= limbs) return set_error("invalid dropped_moduli");
    size_t expected_rows = 0;
    if (!small_mul_size(rows, digits, &expected_rows) ||
        (!small && !small_mul_size(expected_rows, limbs - dropped_moduli, &expected_rows)))
        return set_error("compact decomposition shape overflow");
    const uint64_t base = uint64_t{1} << base_bits;
    const uint64_t expected_bound = small ? base - 1 : (base + 1) / 2;
    if (out->rows < expected_rows || (!ranges && (out->rows != expected_rows || out->cols != src->cols)) || out->bound_words.size() != 1 ||
        out->bound_words[0] != expected_bound)
        return set_error("compact decomposition shape or bound mismatch");
    if (small_set_device(out) != 0) return 1;
    std::vector<uint64_t> requested_bound(max_coefficient_bound,
                                          max_coefficient_bound + bound_word_count);
    if (requested_bound != out->bound_words)
        return set_error("compact decomposition bound metadata mismatch");
    // The compact output is created on a pool stream, but decomposition may
    // run inside the runtime's active graph-capture stream.  Route every
    // input wait and the launch through the capture stream; using out->stream
    // here makes the launch uncaptured while matrix_wait_all_limb_streams has
    // already attempted to establish dependencies for the captured region.
    cudaStream_t stream = small_capture_stream(out);
    if (!stream) return set_error("missing compact decomposition stream");
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *source = sources[block];
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            const dim3 id = source->ctx->limb_gpu_ids[limb];
            int limb_device = -1;
            if (matrix_limb_device(source, id, &limb_device) != 0 || limb_device != out->device ||
                id.x >= source->shared_limb_buffers.size() ||
                !source->shared_limb_buffers[id.x].device_descriptors ||
                id.y >= source->shared_limb_buffers[id.x].limb_count)
                return set_error("compact decomposition requires one device");
            if (limb == 0) dispatch_slot = static_cast<size_t>(id.x);
            else if (id.x != dispatch_slot)
                return set_error("compact decomposition requires one device");
        }
        if (matrix_wait_all_limb_streams(source, out->device, stream, true, true) != 0) return 1;
        blocks.inputs[block] = source->shared_limb_buffers[dispatch_slot].device_descriptors;
    }
    if (dispatch_slot >= src->ctx->ntt_device_constants.size())
        return set_error("missing compact decomposition constants");
    const auto &constants = src->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != out->device || constants.limb_count < limbs || !constants.moduli)
        return set_error("invalid compact decomposition constants");
    const size_t slots = digits * (small ? 1 : limbs - dropped_moduli);
    const dim3 grid((out->n + kSmallThreads - 1) / kSmallThreads,
                    static_cast<uint32_t>(poly_count), static_cast<uint32_t>(slots));
    compact_decompose_kernel<<<grid, kSmallThreads, 0, stream>>>(
        blocks, constants.moduli, out->payload,
        poly_count, out->cols, slots, out->n, digits, out->magnitude_bytes, base_bits, !small, small);
    {
        const size_t argument_sizes[] = {
            sizeof(CompactDecomposeFragments), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(size_t),
            sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(uint32_t), sizeof(bool), sizeof(bool),
        };
        std::vector<MxxGraphPatch> patches;
        patches.reserve(block_count + 1);
        for (size_t block = 0; block < block_count; ++block)
        {
            patches.push_back(small_kernel_pointer_patch(
                0, offsetof(CompactDecomposeFragments, inputs) + block * sizeof(blocks.inputs[0]),
                static_cast<uint32_t>(block)));
        }
        patches.push_back(small_kernel_pointer_patch(2, 0, static_cast<uint32_t>(block_count)));
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches.data(), patches.size());
        if (registration_status != 0)
        {
            (void)small_fence_stream_with_event(stream);
            return registration_status;
        }
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess || small_record(out, stream) != 0)
    {
        (void)small_fence_stream_with_event(stream);
        return err != cudaSuccess ? set_error(err) : 1;
    }
    for (size_t block = 0; block < block_count; ++block)
        if (matrix_track_all_limb_consumers(sources[block], out->device, stream, out->write_done, true, true) != 0)
        {
            (void)small_fence_stream_with_event(stream);
            return 1;
        }
    return 0;
}

extern "C" int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat)
{
    if (!mat) return set_error("invalid compact preimage hard-cutoff owner");
    if (mat->hard_cutoff_subset_count > 0) return 0;
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    const int status = small_initialize_hard_cutoff_plan(mat, mat->rows, mat->cols);
    if (status != 0)
    {
        small_release_hard_cutoff_plan(mat, mat->stream);
        return status;
    }
    return small_record(mat, mat->stream);
}

extern "C" int gpu_small_matrix_prepare_preimage_hard_cutoff_for_tile(
    GpuSmallMatrix *mat,
    size_t rows,
    size_t cols)
{
    if (!mat || rows == 0 || cols == 0 || rows > mat->rows || cols > mat->cols)
        return set_error("invalid compact preimage tile scratch shape");
    if (mat->hard_cutoff_subset_count > 0)
    {
        size_t required_coefficients = 0;
        size_t required_bytes = 0;
        if (!small_mul_size(rows, cols, &required_coefficients) ||
            !small_mul_size(required_coefficients, mat->n, &required_coefficients) ||
            !small_mul_size(required_coefficients, 1 + mat->magnitude_bytes, &required_bytes) ||
            required_bytes > mat->hard_cutoff_staging_bytes)
            return set_error("compact preimage tile scratch exceeds prepared workspace");
        return 0;
    }
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    const int status = small_initialize_hard_cutoff_plan(mat, rows, cols);
    if (status != 0)
    {
        small_release_hard_cutoff_plan(mat, mat->stream);
        return status;
    }
    return small_record(mat, mat->stream);
}

int small_submit_preimage_hard_cutoff_tile(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    uint32_t attempt,
    void *stream_raw,
    void *control_raw)
{
    if (!dst || !src || !bound_words || bound_word_count == 0 ||
        src->ctx != dst->ctx || src->format != GPU_POLY_FORMAT_COEFF ||
        rows == 0 || cols == 0 || src->rows != rows || src->cols != cols ||
        dst_row > dst->rows || rows > dst->rows - dst_row ||
        dst_col > dst->cols || cols > dst->cols - dst_col ||
        bound_word_count != dst->bound_words.size() ||
        !std::equal(bound_words, bound_words + bound_word_count, dst->bound_words.begin()))
        return set_error("invalid compact tile arguments");
    if (small_set_device(dst) != 0) return 1;
    const cudaStream_t stream = stream_raw
        ? reinterpret_cast<cudaStream_t>(stream_raw)
        : small_capture_stream(dst);
    if (!stream || small_wait(dst, stream) != 0) return 1;
    if (!dst->hard_cutoff_metadata)
        return set_error("missing compact preimage reconstruction metadata owner");
    auto *metadata_owner =
        new std::shared_ptr<SmallHardCutoffMetadata>(dst->hard_cutoff_metadata);
    if (!mxx_graph_retain_released_resource(dst->ctx, metadata_owner, [](void *resource) {
            delete static_cast<std::shared_ptr<SmallHardCutoffMetadata> *>(resource);
        }))
        delete metadata_owner;
    if (src->level < 0) return set_error("invalid compact tile source level");
    const size_t limb_count = static_cast<size_t>(src->level + 1);
    if (limb_count == 0 || limb_count > static_cast<size_t>(kMaxRnsLimbs) ||
        src->ctx->limb_gpu_ids.size() < limb_count ||
        limb_count != dst->hard_cutoff_limb_count ||
        dst->hard_cutoff_subset_count <= 0 || dst->hard_cutoff_words_per_coeff <= 0 ||
        !dst->hard_cutoff_garner_inverses || !dst->hard_cutoff_subset_indices ||
        !dst->hard_cutoff_modulus_words || !dst->hard_cutoff_half_modulus_words ||
        !dst->hard_cutoff_bound_words || !dst->hard_cutoff_device_status ||
        !dst->hard_cutoff_staging || !dst->hard_cutoff_decision_ready)
        return set_error("invalid compact tile active CRT basis");
    size_t poly_count = 0;
    size_t total_coefficients = 0;
    if (!small_mul_size(rows, cols, &poly_count) ||
        !small_mul_size(poly_count, dst->n, &total_coefficients))
        return set_error("compact tile coefficient count overflow");

    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int device = -1;
        if (matrix_limb_device(src, id, &device) != 0 || device != dst->device)
            return set_error("compact tile requires one-device active CRT placement");
        if (limb == 0) dispatch_slot = static_cast<size_t>(id.x);
        else if (static_cast<size_t>(id.x) != dispatch_slot)
            return set_error("compact tile active CRT limbs span devices");
        if (id.y != limb || matrix_wait_limb_stream(src, id, dst->device, stream, true, true) != 0)
            return set_error("invalid compact tile active CRT limb");
    }
    if (dispatch_slot >= src->shared_limb_buffers.size() ||
        dispatch_slot >= src->ctx->ntt_device_constants.size() ||
        src->shared_limb_buffers[dispatch_slot].limb_count < limb_count ||
        !src->shared_limb_buffers[dispatch_slot].device_descriptors)
        return set_error("missing compact tile device descriptors");
    const auto &constants = src->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != dst->device || constants.limb_count < limb_count ||
        !constants.moduli)
        return set_error("missing compact tile device moduli");

    cudaError_t err = cudaSuccess;
    size_t staging_bytes = 0;
    if (!small_mul_size(total_coefficients, 1 + dst->magnitude_bytes, &staging_bytes))
        return set_error("compact tile staging size overflow");
    if (staging_bytes > dst->hard_cutoff_staging_bytes)
        return set_error("compact tile staging exceeds fixed retry workspace");
    if (control_raw)
    {
        preimage_status_init_control_kernel<<<1, 1, 0, stream>>>(
            dst->hard_cutoff_device_status,
            static_cast<const MxxPreimageLaunchControl *>(control_raw));
    }
    else
    {
        preimage_status_init_kernel<<<1, 1, 0, stream>>>(
            dst->hard_cutoff_device_status, attempt);
    }
    if (control_raw)
    {
        const size_t argument_sizes[] = {sizeof(void *), sizeof(void *)};
        const MxxGraphPatch patches[] = {
            small_kernel_pointer_patch(0, 0, 7), small_kernel_pointer_patch(1, 0, 10),
        };
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
    }
    else
    {
        const size_t argument_sizes[] = {sizeof(void *), sizeof(uint32_t)};
        const MxxGraphPatch patches[] = {small_kernel_pointer_patch(0, 0, 7)};
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
    }
    err = cudaGetLastError();
    if (err == cudaSuccess)
        compact_check_pack_preimage_kernel<<<
            (total_coefficients + kSmallThreads - 1) / kSmallThreads,
            kSmallThreads, 0, stream>>>(
                src->shared_limb_buffers[dispatch_slot].device_descriptors,
                constants.moduli,
                dst->hard_cutoff_garner_inverses,
                static_cast<int>(src->ctx->moduli.size()),
                dst->hard_cutoff_subset_indices,
                dst->hard_cutoff_subset_count,
                static_cast<int>(limb_count),
                total_coefficients,
                dst->n,
                dst->hard_cutoff_words_per_coeff,
                dst->hard_cutoff_modulus_words,
                dst->hard_cutoff_half_modulus_words,
                dst->hard_cutoff_bound_words,
                dst->magnitude_bytes,
                dst->hard_cutoff_device_status,
                dst->hard_cutoff_staging);
    if (err == cudaSuccess)
    {
        const size_t argument_sizes[] = {
            sizeof(void *), sizeof(void *), sizeof(void *), sizeof(int), sizeof(void *),
            sizeof(int), sizeof(int), sizeof(size_t), sizeof(size_t), sizeof(int),
            sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(void *),
            sizeof(void *),
        };
        const MxxGraphPatch patches[] = {
            small_kernel_pointer_patch(0, 0, 0), small_kernel_pointer_patch(1, 0, 1),
            small_kernel_pointer_patch(2, 0, 2), small_kernel_pointer_patch(4, 0, 3),
            small_kernel_pointer_patch(10, 0, 4), small_kernel_pointer_patch(11, 0, 5),
            small_kernel_pointer_patch(12, 0, 6), small_kernel_pointer_patch(14, 0, 7),
            small_kernel_pointer_patch(15, 0, 8),
        };
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
    }
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        preimage_status_latch_accept_kernel<<<1, 1, 0, stream>>>(dst->hard_cutoff_device_status);
        const size_t argument_sizes[] = {sizeof(void *)};
        const MxxGraphPatch patches[] = {small_kernel_pointer_patch(0, 0, 7)};
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
        err = cudaGetLastError();
    }
    for (size_t limb = 0; limb < limb_count && err == cudaSuccess; ++limb)
    {
        if (matrix_track_limb_consumer_readonly(
                src, src->ctx->limb_gpu_ids[limb], dst->device, stream) != 0)
            err = cudaErrorInvalidResourceHandle;
    }
    if (err == cudaSuccess)
    {
        const dim3 commit_grid((staging_bytes + kSmallThreads - 1) / kSmallThreads);
        compact_commit_preimage_tile_if_accepted_kernel<<<commit_grid, kSmallThreads, 0, stream>>>(
            dst->payload,
            dst->hard_cutoff_staging,
            dst->hard_cutoff_device_status,
            dst->n,
            rows,
            cols,
            dst->cols,
            dst_row,
            dst_col,
            1 + dst->magnitude_bytes);
        const size_t argument_sizes[] = {
            sizeof(void *), sizeof(void *), sizeof(void *), sizeof(size_t), sizeof(size_t),
            sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t), sizeof(size_t),
        };
        const MxxGraphPatch patches[] = {
            small_kernel_pointer_patch(0, 0, 9), small_kernel_pointer_patch(1, 0, 8),
            small_kernel_pointer_patch(2, 0, 7),
        };
        const int registration_status = mxx_graph_register_kernel_update_for_stream(
            src->ctx, reinterpret_cast<void *>(stream), argument_sizes,
            std::size(argument_sizes), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
        err = cudaGetLastError();
    }
    if (err == cudaSuccess && small_record(dst, stream) != 0)
        err = cudaErrorInvalidResourceHandle;
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_submit_preimage_hard_cutoff_tile_on_stream(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    uint32_t attempt,
    void *stream)
{
    return small_submit_preimage_hard_cutoff_tile(
        dst,
        src,
        dst_row,
        dst_col,
        rows,
        cols,
        bound_words,
        bound_word_count,
        attempt,
        stream,
        nullptr);
}

extern "C" int gpu_small_matrix_submit_preimage_hard_cutoff_tile_control(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    void *device_control,
    void *stream)
{
    return small_submit_preimage_hard_cutoff_tile(
        dst,
        src,
        dst_row,
        dst_col,
        rows,
        cols,
        bound_words,
        bound_word_count,
        0,
        stream,
        device_control);
}

extern "C" int gpu_small_matrix_submit_preimage_hard_cutoff_tile(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    uint32_t attempt)
{
    return small_submit_preimage_hard_cutoff_tile(
        dst,
        src,
        dst_row,
        dst_col,
        rows,
        cols,
        bound_words,
        bound_word_count,
        attempt,
        nullptr,
        nullptr);
}

static int gpu_small_matrix_copy_preimage_status_async_impl(
    GpuSmallMatrix *mat,
    uint32_t status_binding_index,
    uint32_t host_status_binding_index)
{
    if (!mat || !mat->hard_cutoff_device_status || !mat->hard_cutoff_host_status ||
        !mat->hard_cutoff_decision_ready)
        return set_error("invalid compact preimage status owner");
    if (small_set_device(mat) != 0) return 1;
    const cudaStream_t stream = small_capture_stream(mat);
    if (!stream) return set_error("missing compact preimage status stream");
    cudaError_t err = cudaMemcpyAsync(
        mat->hard_cutoff_host_status,
        mat->hard_cutoff_device_status,
        sizeof(MxxPreimageStatus),
        cudaMemcpyDeviceToHost,
        stream);
    if (err == cudaSuccess)
    {
        MxxGraphPatch patches[2]{};
        patches[0].target = MXX_GRAPH_PATCH_MEMCPY_1D_DST;
        patches[0].byte_count = sizeof(uint64_t);
        patches[0].binding_index = host_status_binding_index;
        patches[1].target = MXX_GRAPH_PATCH_MEMCPY_1D_SRC;
        patches[1].byte_count = sizeof(uint64_t);
        patches[1].binding_index = status_binding_index;
        const int registration_status = mxx_graph_register_memcpy1d_update_for_stream(
            mat->ctx, reinterpret_cast<void *>(stream), sizeof(MxxPreimageStatus),
            static_cast<int>(cudaMemcpyDeviceToHost), patches, std::size(patches));
        if (registration_status != 0) return registration_status;
    }
    if (err == cudaSuccess)
    {
        err = cudaEventRecord(mat->hard_cutoff_decision_ready, stream);
        if (err == cudaSuccess) mat->hard_cutoff_decision_recorded = true;
    }
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_copy_preimage_status_async(GpuSmallMatrix *mat)
{
    // Legacy ordinary execution schema.  Captured production retry regions
    // call the explicit variant below with the MxxPreimageRetrySpec indices.
    return gpu_small_matrix_copy_preimage_status_async_impl(mat, 7, 10);
}

extern "C" int gpu_small_matrix_copy_preimage_status_async_with_bindings(
    GpuSmallMatrix *mat,
    uint32_t status_binding_index,
    uint32_t host_status_binding_index)
{
    if (status_binding_index == UINT32_MAX || host_status_binding_index == UINT32_MAX ||
        status_binding_index == host_status_binding_index)
        return set_error("invalid compact preimage status binding schema");
    return gpu_small_matrix_copy_preimage_status_async_impl(
        mat, status_binding_index, host_status_binding_index);
}

extern "C" int gpu_small_matrix_mark_preimage_exhausted(GpuSmallMatrix *mat)
{
    if (!mat || !mat->hard_cutoff_device_status)
        return set_error("invalid compact preimage exhaustion status owner");
    const cudaStream_t stream = small_capture_stream(mat);
    if (!stream) return set_error("missing compact preimage exhaustion stream");
    preimage_status_mark_exhausted_kernel<<<1, 1, 0, stream>>>(mat->hard_cutoff_device_status);
    const size_t argument_sizes[] = {sizeof(void *)};
    const MxxGraphPatch patches[] = {small_kernel_pointer_patch(0, 0, 7)};
    const int registration_status = mxx_graph_register_kernel_update_for_stream(
        mat->ctx, reinterpret_cast<void *>(stream), argument_sizes,
        std::size(argument_sizes), patches, std::size(patches));
    if (registration_status != 0) return registration_status;
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_wait_preimage_status(
    GpuSmallMatrix *mat,
    MxxPreimageStatus *out_status)
{
    if (!mat || !out_status || !mat->hard_cutoff_host_status ||
        !mat->hard_cutoff_decision_ready)
        return set_error("invalid compact preimage status wait arguments");
    cudaError_t err = cudaEventSynchronize(mat->hard_cutoff_decision_ready);
    if (err != cudaSuccess) return set_error(err);
    *out_status = *mat->hard_cutoff_host_status;
    return 0;
}

// Compatibility entry point for legacy callers. Production retry graphs use
// submit + one final copy/wait; this wrapper retains the old synchronous
// result shape without allocating an attempt-local staging buffer.
extern "C" int gpu_small_matrix_try_pack_preimage_hard_cutoff_tile(
    GpuSmallMatrix *dst,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t rows,
    size_t cols,
    const uint64_t *bound_words,
    size_t bound_word_count,
    int32_t *accepted_out)
{
    if (!accepted_out) return set_error("null compact preimage acceptance output");
    int status = gpu_small_matrix_submit_preimage_hard_cutoff_tile(
        dst,
        src,
        dst_row,
        dst_col,
        rows,
        cols,
        bound_words,
        bound_word_count,
        0);
    if (status != 0) return status;
    status = gpu_small_matrix_copy_preimage_status_async(dst);
    if (status != 0) return status;
    MxxPreimageStatus result{};
    status = gpu_small_matrix_wait_preimage_status(dst, &result);
    if (status == 0) *accepted_out = result.accepted == 1U ? 1 : 0;
    return status;
}

int gpu_matrix_mul_small_rhs_impl(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report,
    cudaStream_t requested_stream,
    const uint32_t *source_binding_indices,
    const uint32_t *destination_binding_indices,
    uint32_t rhs_payload_binding_index,
    const size_t *block_fragment_ends)
{
    if (!outputs || !inputs || block_count == 0 || block_count > kCompactRowBlocks)
        return set_error("invalid compact multiplication blocks");
    const size_t source_count = block_fragment_ends ? block_fragment_ends[block_count - 1] : block_count;
    if (source_count == 0 || source_count > kCompactSourceFragments)
        return set_error("compact multiplication source fragment limit exceeded");
    const bool bound = requested_stream != nullptr;
    if (bound && (!source_binding_indices || !destination_binding_indices ||
                  rhs_payload_binding_index == UINT32_MAX))
        return set_error("invalid compact multiplication binding schema");
    if (bound)
    {
        for (size_t block = 0; block < source_count; ++block)
        {
            if (source_binding_indices[block] == UINT32_MAX)
                return set_error("invalid compact multiplication binding index");
        }
        for (size_t block = 0; block < block_count; ++block)
            if (destination_binding_indices[block] == UINT32_MAX)
                return set_error("invalid compact multiplication destination binding index");
    }
    GpuMatrix *out = outputs[0];
    const GpuMatrix *lhs_eval = inputs[0];
    if (!out || !lhs_eval || !rhs_small || !out->ctx || out->ctx != lhs_eval->ctx || out->ctx != rhs_small->ctx ||
        lhs_eval->format != GPU_POLY_FORMAT_EVAL || out->format != GPU_POLY_FORMAT_EVAL ||
        out->rows != lhs_eval->rows || out->cols != rhs_small->cols ||
        !allocation_report)
        return set_error("invalid compact RHS multiplication arguments");
    const size_t limbs = static_cast<size_t>(lhs_eval->level + 1);
    if (lhs_eval->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || out->level != lhs_eval->level ||
        out->ctx->limb_gpu_ids.size() < limbs || out->ctx->N < 2 ||
        !is_power_of_two_u32(static_cast<uint32_t>(out->ctx->N)))
        return set_error("invalid compact RHS multiplication level");
    CompactRowBlocks blocks{};
    size_t rows = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *output = outputs[block];
        if (!output || output->ctx != out->ctx || output->level != out->level ||
            output->format != GPU_POLY_FORMAT_EVAL || output->cols != out->cols ||
            !small_add_size(rows, output->rows, &rows))
            return set_error("incompatible compact multiplication block");
        blocks.ends[block] = rows;
        const size_t start = block ? blocks.fragment_ends[block - 1] : 0;
        const size_t end = block_fragment_ends ? block_fragment_ends[block] : block + 1;
        if (end <= start || end > source_count) return set_error("invalid compact source fragment offsets");
        size_t columns = 0;
        for (size_t fragment = start; fragment < end; ++fragment)
        {
            const auto *left = inputs[fragment];
            if (!left || left->ctx != lhs_eval->ctx || left->level != lhs_eval->level ||
                left->format != GPU_POLY_FORMAT_EVAL || left->rows != output->rows || left->cols == 0 ||
                !small_add_size(columns, left->cols, &columns))
                return set_error("incompatible compact multiplication source fragment");
            blocks.column_ends[fragment] = columns;
        }
        if (columns != rhs_small->rows) return set_error("compact fragments do not cover the inner dimension");
        blocks.fragment_ends[block] = end;
    }
    if (small_set_device(rhs_small) != 0) return 1;
    cudaStream_t stream = requested_stream;
    int dispatch_device = -1;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    size_t output_block = 0;
    for (size_t block = 0; block < source_count; ++block)
    {
        while (block >= blocks.fragment_ends[output_block]) ++output_block;
        const auto *lhs_eval = inputs[block];
        auto *out = outputs[output_block];
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            const dim3 id = out->ctx->limb_gpu_ids[limb];
            if (id.x >= out->shared_limb_buffers.size())
                return set_error("invalid compact multiplication limb partition");
            int lhs_device = -1;
            int out_device = -1;
            size_t lhs_stride = 0;
            size_t out_stride = 0;
            uint8_t lhs_width = 0;
            uint8_t out_width = 0;
            if (matrix_limb_device(lhs_eval, id, &lhs_device) != 0 || matrix_limb_device(out, id, &out_device) != 0 ||
                lhs_device != rhs_small->device || out_device != rhs_small->device ||
                id.y >= lhs_eval->shared_limb_buffers[id.x].limb_count ||
                id.y >= out->shared_limb_buffers[id.x].limb_count ||
                !lhs_eval->shared_limb_buffers[id.x].device_descriptors ||
                !out->shared_limb_buffers[id.x].device_descriptors ||
                !matrix_limb_metadata_by_id(lhs_eval, id, &lhs_stride, &lhs_width) ||
                !matrix_limb_metadata_by_id(out, id, &out_stride, &out_width))
                return set_error("compact RHS multiplication requires one placement");
            if (block == 0 && limb == 0)
            {
                dispatch_device = out_device;
                dispatch_slot = static_cast<size_t>(id.x);
                if (!stream && matrix_limb_stream(out, id, &stream) != 0) return 1;
                // The owner stream is the normal asynchronous launch stream,
                // but a fused compact product can be captured on the
                // execution context's capture stream.  Route the complete
                // dependency chain (waits, workspace, and kernels) through
                // that stream so CUDA never sees an uncaptured wait created
                // by this operation.
                if (!requested_stream)
                    stream = matrix_capture_stream_for_device(out->ctx, out_device, stream);
            }
            else if (out_device != dispatch_device)
                return set_error("compact RHS multiplication requires one device");
        }
        if (matrix_wait_all_limb_streams(lhs_eval, rhs_small->device, stream, true, true) != 0 ||
            matrix_wait_all_limb_streams(out, rhs_small->device, stream, true) != 0) return 1;
        blocks.inputs[block] = lhs_eval->shared_limb_buffers[dispatch_slot].device_descriptors;
        blocks.outputs[output_block] = out->shared_limb_buffers[dispatch_slot].device_descriptors;
    }
    if (!stream) return set_error("missing compact multiplication stream");
    if (small_wait(rhs_small, stream) != 0) return 1;
    const size_t n = rhs_small->n;
    const size_t inner = rhs_small->rows;
    const size_t cols = rhs_small->cols;
    if (dispatch_slot >= out->ctx->ntt_device_constants.size())
        return set_error("missing compact multiplication NTT partition");
    const auto &constants = out->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != dispatch_device || constants.ring_dimension != n ||
        constants.limb_count < limbs || !constants.twiddle_forward ||
        !constants.twiddle_shoup_forward || !constants.moduli)
        return set_error("missing compact multiplication NTT constants");

    struct LimbGroup
    {
        size_t limb_offset;
        size_t limb_count;
        size_t typed_limb_offset;
        bool narrow;
    };
    std::vector<LimbGroup> limb_groups;
    std::vector<uint64_t> active_moduli;
    active_moduli.reserve(limbs);
    size_t u32_limb_count = 0;
    size_t u64_limb_count = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        if (limb >= out->ctx->limb_prime_ids.size())
            return set_error("missing compact multiplication limb prime metadata");
        const int prime_id = out->ctx->limb_prime_ids[limb];
        if (prime_id < 0 || static_cast<size_t>(prime_id) >= out->ctx->moduli.size())
            return set_error("invalid compact multiplication limb prime metadata");
        const uint64_t modulus = out->ctx->moduli[static_cast<size_t>(prime_id)];
        active_moduli.push_back(modulus);
        const bool narrow = modulus <= UINT32_MAX;
        size_t &typed_count = narrow ? u32_limb_count : u64_limb_count;
        if (limb_groups.empty() || limb_groups.back().narrow != narrow)
            limb_groups.push_back(LimbGroup{limb, 1, typed_count, narrow});
        else
            ++limb_groups.back().limb_count;
        ++typed_count;
    }

    uint32_t *workspace_u32 = nullptr;
    uint64_t *workspace_u64 = nullptr;
    auto release = [&]() -> cudaError_t {
        cudaError_t cleanup_err = cudaSuccess;
        auto free_async = [&](void *ptr) {
            if (!ptr) return;
            const cudaError_t free_err = cudaFreeAsync(ptr, stream);
            if (cleanup_err == cudaSuccess && free_err != cudaSuccess) cleanup_err = free_err;
        };
        free_async(workspace_u32);
        free_async(workspace_u64);
        workspace_u32 = nullptr;
        workspace_u64 = nullptr;
        return cleanup_err;
    };
    auto fail = [&](cudaError_t failure) -> int {
        // Failure paths may leave output work queued without the normal owner
        // events. Fence only this stream through a temporary event. A
        // stream-wide fallback occurs inside the helper only when an event
        // cannot be created/recorded; the success path remains asynchronous.
        const cudaError_t fence_err = small_fence_stream_with_event(stream);
        const cudaError_t cleanup_err = release();
        if (failure == cudaSuccess) failure = fence_err;
        if (failure == cudaSuccess) failure = cleanup_err;
        return set_error(failure);
    };
    size_t lhs_eval_bytes = 0;
    size_t full_output_bytes = 0;
    for (size_t block = 0; block < source_count; ++block)
    {
        GpuMatrixAllocationBytes lhs_allocation{};
        const auto *left = inputs[block];
        if (gpu_matrix_query_allocation_bytes(left->ctx, left->level, left->rows, left->cols,
                left->format, &lhs_allocation) != 0 ||
            !small_add_size(lhs_eval_bytes, lhs_allocation.total_bytes, &lhs_eval_bytes))
            return set_error("compact source allocation overflow");
    }
    for (size_t block = 0; block < block_count; ++block)
    {
        GpuMatrixAllocationBytes output_allocation{};
        const auto *output = outputs[block];
        if (
            gpu_matrix_query_allocation_bytes(output->ctx, output->level, output->rows, output->cols,
                output->format, &output_allocation) != 0 ||
            !small_add_size(full_output_bytes, output_allocation.total_bytes, &full_output_bytes))
            return set_error("compact block allocation overflow");
    }
    size_t workspace_words_per_limb = 0;
    size_t workspace_u32_words = 0;
    size_t workspace_u64_words = 0;
    size_t workspace_u32_bytes = 0;
    size_t workspace_u64_bytes = 0;
    size_t expanded_rhs_workspace_bytes = 0;
    if (!small_mul_size(inner, cols, &workspace_words_per_limb) ||
        !small_mul_size(workspace_words_per_limb, n, &workspace_words_per_limb) ||
        !small_mul_size(u32_limb_count, workspace_words_per_limb, &workspace_u32_words) ||
        !small_mul_size(u64_limb_count, workspace_words_per_limb, &workspace_u64_words) ||
        !small_mul_size(workspace_u32_words, sizeof(uint32_t), &workspace_u32_bytes) ||
        !small_mul_size(workspace_u64_words, sizeof(uint64_t), &workspace_u64_bytes) ||
        !small_add_size(workspace_u32_bytes, workspace_u64_bytes, &expanded_rhs_workspace_bytes))
        return set_error("compact RHS workspace size overflow");
    size_t event_overhead_bytes = 0;
    if (!small_mul_size(limbs + 1, sizeof(cudaEvent_t), &event_overhead_bytes))
        return set_error("compact RHS event accounting overflow");
    size_t high_water_bytes = 0;
    if (!small_add_size(lhs_eval_bytes, rhs_small->resident_payload_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, full_output_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, expanded_rhs_workspace_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, event_overhead_bytes, &high_water_bytes))
        return set_error("compact RHS residency size overflow");
    allocation_report->lhs_eval_bytes = lhs_eval_bytes;
    allocation_report->compact_rhs_bytes = rhs_small->resident_payload_bytes;
    allocation_report->full_output_bytes = full_output_bytes;
    allocation_report->expanded_rhs_workspace_bytes = expanded_rhs_workspace_bytes;
    allocation_report->event_overhead_bytes = event_overhead_bytes;
    allocation_report->high_water_bytes = high_water_bytes;
    allocation_report->full_expanded_rhs_bytes = expanded_rhs_workspace_bytes;
    allocation_report->workspace_word_bytes =
        u32_limb_count == limbs ? sizeof(uint32_t) :
        (u64_limb_count == limbs ? sizeof(uint64_t) : 0);
    allocation_report->u32_workspace_limb_count = u32_limb_count;
    allocation_report->u64_workspace_limb_count = u64_limb_count;
    const uint32_t n_u32 = static_cast<uint32_t>(n);
    size_t launches_per_group = 1;
    if (n_u32 > kCompactNttSuffixSize)
    {
        ++launches_per_group;
        for (uint32_t len = n_u32 >> 1; len > kCompactNttSuffixSize; len >>= 1)
            ++launches_per_group;
    }
    if (!small_mul_size(
            launches_per_group, limb_groups.size(),
            &allocation_report->ntt_preparation_launches))
        return set_error("compact RHS NTT launch count overflow");
    if (high_water_bytes > residency_budget_bytes)
        return 2;
    cudaError_t err = cudaSuccess;
    if (workspace_u32_bytes != 0)
        err = cudaMallocAsync(
            reinterpret_cast<void **>(&workspace_u32), workspace_u32_bytes, stream);
    if (err == cudaSuccess && workspace_u64_bytes != 0)
        err = cudaMallocAsync(
            reinterpret_cast<void **>(&workspace_u64), workspace_u64_bytes, stream);
    if (err != cudaSuccess) return fail(err);

    const size_t poly_count = inner * cols;
    for (const LimbGroup &group : limb_groups)
    {
        if (n_u32 <= kCompactNttSuffixSize)
        {
            if (poly_count > std::numeric_limits<uint32_t>::max())
                return fail(cudaErrorInvalidConfiguration);
            const dim3 grid(
                static_cast<uint32_t>(poly_count),
                static_cast<uint32_t>(group.limb_count));
            if (group.narrow)
            {
                uint32_t *group_workspace =
                    workspace_u32 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_all_shared_kernel<<<grid, kSmallThreads, 0, stream>>>(
                    rhs_small->payload, group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, cols,
                    rhs_small->storage_cols, n_u32, rhs_small->magnitude_bytes,
                    rhs_small->column_offset);
            }
            else
            {
                uint64_t *group_workspace =
                    workspace_u64 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_all_shared_kernel<<<grid, kSmallThreads, 0, stream>>>(
                    rhs_small->payload, group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, cols,
                    rhs_small->storage_cols, n_u32, rhs_small->magnitude_bytes,
                    rhs_small->column_offset);
            }
            if (register_compact_rhs_update(
                    rhs_small->ctx,
                    stream,
                    bound ? rhs_payload_binding_index : static_cast<uint32_t>(block_count)) != 0)
                return fail(cudaErrorInvalidResourceHandle);
            err = cudaGetLastError();
        }
        else
        {
            size_t first_butterflies = 0;
            if (!small_mul_size(group.limb_count, poly_count, &first_butterflies) ||
                !small_mul_size(first_butterflies, n / 2, &first_butterflies))
                return fail(cudaErrorInvalidConfiguration);
            const size_t first_blocks =
                (first_butterflies + kSmallThreads - 1) / kSmallThreads;
            if (first_blocks > std::numeric_limits<uint32_t>::max())
                return fail(cudaErrorInvalidConfiguration);
            const dim3 first_grid(static_cast<uint32_t>(first_blocks));
            if (group.narrow)
            {
                uint32_t *group_workspace =
                    workspace_u32 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_first_kernel<<<first_grid, kSmallThreads, 0, stream>>>(
                    rhs_small->payload, group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, cols,
                    rhs_small->storage_cols, n_u32, rhs_small->magnitude_bytes,
                    rhs_small->column_offset);
            }
            else
            {
                uint64_t *group_workspace =
                    workspace_u64 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_first_kernel<<<first_grid, kSmallThreads, 0, stream>>>(
                    rhs_small->payload, group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, cols,
                    rhs_small->storage_cols, n_u32, rhs_small->magnitude_bytes,
                    rhs_small->column_offset);
            }
            if (register_compact_rhs_update(
                    rhs_small->ctx,
                    stream,
                    bound ? rhs_payload_binding_index : static_cast<uint32_t>(block_count)) != 0)
                return fail(cudaErrorInvalidResourceHandle);
            err = cudaGetLastError();
            for (uint32_t len = n_u32 >> 1;
                 err == cudaSuccess && len > kCompactNttSuffixSize;
                 len >>= 1)
            {
                size_t butterflies = 0;
                if (!small_mul_size(group.limb_count, poly_count, &butterflies) ||
                    !small_mul_size(butterflies, n / 2, &butterflies))
                    return fail(cudaErrorInvalidConfiguration);
                const size_t grid_blocks =
                    (butterflies + kSmallThreads - 1) / kSmallThreads;
                if (grid_blocks > std::numeric_limits<uint32_t>::max())
                    return fail(cudaErrorInvalidConfiguration);
                const dim3 grid(static_cast<uint32_t>(grid_blocks));
                if (group.narrow)
                {
                    uint32_t *group_workspace =
                        workspace_u32 + group.typed_limb_offset * workspace_words_per_limb;
                    compact_ntt_dif_stage_kernel<<<grid, kSmallThreads, 0, stream>>>(
                        group_workspace, constants.twiddle_forward,
                        constants.twiddle_shoup_forward, constants.moduli,
                        group.limb_offset, group.limb_count, poly_count, n, len);
                }
                else
                {
                    uint64_t *group_workspace =
                        workspace_u64 + group.typed_limb_offset * workspace_words_per_limb;
                    compact_ntt_dif_stage_kernel<<<grid, kSmallThreads, 0, stream>>>(
                        group_workspace, constants.twiddle_forward,
                        constants.twiddle_shoup_forward, constants.moduli,
                        group.limb_offset, group.limb_count, poly_count, n, len);
                }
                err = cudaGetLastError();
            }
            if (err != cudaSuccess) break;
            size_t suffix_blocks = 0;
            if (!small_mul_size(
                    poly_count, n / kCompactNttSuffixSize, &suffix_blocks) ||
                suffix_blocks > std::numeric_limits<uint32_t>::max())
                return fail(cudaErrorInvalidConfiguration);
            const dim3 suffix_grid(
                static_cast<uint32_t>(suffix_blocks),
                static_cast<uint32_t>(group.limb_count));
            if (group.narrow)
            {
                uint32_t *group_workspace =
                    workspace_u32 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_suffix_kernel<<<suffix_grid, kSmallThreads, 0, stream>>>(
                    group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, n_u32);
            }
            else
            {
                uint64_t *group_workspace =
                    workspace_u64 + group.typed_limb_offset * workspace_words_per_limb;
                compact_rhs_dif_suffix_kernel<<<suffix_grid, kSmallThreads, 0, stream>>>(
                    group_workspace, constants.twiddle_forward,
                    constants.twiddle_shoup_forward, constants.moduli,
                    group.limb_offset, group.limb_count, poly_count, n_u32);
            }
            err = cudaGetLastError();
        }
        if (err != cudaSuccess) break;
    }
    if (err == cudaSuccess)
    {
        for (const LimbGroup &group : limb_groups)
        {
            const bool lazy_reduce = compact_lazy_dot_is_safe(
                active_moduli, group.limb_offset, group.limb_count, inner);
            size_t output_words = 0;
            if (!small_mul_size(group.limb_count, rows, &output_words) ||
                !small_mul_size(output_words, cols, &output_words) ||
                !small_mul_size(output_words, n, &output_words))
                return fail(cudaErrorInvalidConfiguration);
            const size_t grid_blocks = (output_words + kSmallThreads - 1) / kSmallThreads;
            if (grid_blocks > std::numeric_limits<uint32_t>::max())
                return fail(cudaErrorInvalidConfiguration);
            const dim3 grid(static_cast<uint32_t>(grid_blocks));
            if (group.narrow)
            {
                const uint32_t *group_workspace =
                    workspace_u32 + group.typed_limb_offset * workspace_words_per_limb;
                compact_accumulate_kernel<<<grid, kSmallThreads, 0, stream>>>(
                    blocks, constants.moduli,
                    group.limb_offset, group_workspace, group.limb_count,
                    rows, inner, cols, n, lazy_reduce);
            }
            else
            {
                const uint64_t *group_workspace =
                    workspace_u64 + group.typed_limb_offset * workspace_words_per_limb;
                compact_accumulate_kernel<<<grid, kSmallThreads, 0, stream>>>(
                    blocks, constants.moduli,
                    group.limb_offset, group_workspace, group.limb_count,
                    rows, inner, cols, n, lazy_reduce);
            }
            if (register_compact_accumulate_update(
                    rhs_small->ctx,
                    stream,
                    block_count,
                    source_count,
                    bound ? source_binding_indices : nullptr,
                    bound ? destination_binding_indices : nullptr) != 0)
                return fail(cudaErrorInvalidResourceHandle);
            err = cudaGetLastError();
            if (err != cudaSuccess) break;
        }
    }
    if (err != cudaSuccess) return fail(err);
    for (size_t block = 0; block < block_count; ++block)
        if (matrix_record_all_limb_writes(outputs[block], stream, true) != 0)
            return fail(cudaErrorInvalidResourceHandle);
    const dim3 first = out->ctx->limb_gpu_ids[0];
    const auto &states = out->exec_limb_states[first.x];
    const cudaEvent_t completion = states[states[first.y].completion_owner].write_done;
    for (size_t block = 0; block < source_count; ++block)
        if (matrix_track_all_limb_consumers(inputs[block], rhs_small->device, stream, completion, true, true) != 0)
            return fail(cudaErrorInvalidResourceHandle);
    if (small_track_consumer(rhs_small, stream, completion) != 0)
        return fail(cudaErrorInvalidResourceHandle);
    const cudaError_t cleanup_err = release();
    if (cleanup_err != cudaSuccess) return set_error(cleanup_err);
    return 0;
}

extern "C" int gpu_matrix_mul_small_rhs(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report)
{
    return gpu_matrix_mul_small_rhs_impl(
        outputs,
        inputs,
        block_count,
        rhs_small,
        residency_budget_bytes,
        allocation_report,
        nullptr,
        nullptr,
        nullptr,
        UINT32_MAX,
        nullptr);
}

extern "C" int gpu_matrix_mul_small_rhs_into_bound(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report,
    void *stream,
    const uint32_t *source_binding_indices,
    const uint32_t *destination_binding_indices,
    uint32_t rhs_payload_binding_index,
    const size_t *block_fragment_ends)
{
    if (!stream) return set_error("missing compact multiplication capture stream");
    return gpu_matrix_mul_small_rhs_impl(
        outputs,
        inputs,
        block_count,
        rhs_small,
        residency_budget_bytes,
        allocation_report,
        reinterpret_cast<cudaStream_t>(stream),
        source_binding_indices,
        destination_binding_indices,
        rhs_payload_binding_index,
        block_fragment_ends);
}
