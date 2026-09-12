#include <array>
#include "matrix/MatrixSmallRhs.cuh"
#include "gpu_admission.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

/*
 * Compact RHS storage deliberately has no relationship to GpuMatrix.  The
 * only device allocation owned by this object is the canonical sign and
 * magnitude byte stream.  The host-side bound is metadata supplied by the
 * already validated Rust schema; it is never inferred from the stream.
 */
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
    GpuDeviceWorkspace payload_owner;
    // Empty payloads still need a typed completion resource in admitted mode.
    GpuCudaResource empty_payload_completion;
    std::array<GpuDeviceWorkspace, 6> hard_cutoff_owners;
    std::vector<uint64_t> bound_words;
    size_t hard_cutoff_limb_count = 0;
    int hard_cutoff_subset_count = 0;
    int hard_cutoff_words_per_coeff = 0;
    uint64_t *hard_cutoff_garner_inverses = nullptr;
    int *hard_cutoff_subset_indices = nullptr;
    uint64_t *hard_cutoff_modulus_words = nullptr;
    uint64_t *hard_cutoff_half_modulus_words = nullptr;
    uint64_t *hard_cutoff_bound_words = nullptr;
    int *hard_cutoff_device_accepted = nullptr;
    int *hard_cutoff_host_accepted = nullptr;
    cudaEvent_t hard_cutoff_decision_ready = nullptr;
    bool owns_hard_cutoff_decision_event = true;
    cudaEvent_t write_done = nullptr;
    bool write_done_valid = false;
};

namespace
{
constexpr int kSmallThreads = 256;
constexpr size_t kMaxSmallLimbCount = GPU_RUNTIME_MAX_LIMBS;
constexpr uint32_t kCompactNttSuffixSize = 4096;

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

int small_wait(const GpuSmallMatrix *mat, cudaStream_t stream)
{
    if (!mat || !stream) return set_error("invalid compact matrix wait arguments");
    if (!mat->write_done_valid) return 0;
    const cudaError_t err = cudaStreamWaitEvent(stream, mat->write_done, 0);
    return err == cudaSuccess ? 0 : set_error(err);
}

int small_record(GpuSmallMatrix *mat, cudaStream_t stream)
{
    if (!mat || !stream || !mat->write_done)
        return set_error("invalid compact matrix event arguments");
    const cudaError_t err = cudaEventRecord(mat->write_done, stream);
    if (err != cudaSuccess) return set_error(err);
    mat->write_done_valid = true;
    return 0;
}


int small_track_consumer(const GpuSmallMatrix *mat, cudaStream_t consumer_stream, cudaEvent_t completion)
{
    if (!mat || !mat->ctx || !consumer_stream || mat->device < 0)
        return set_error("invalid compact matrix consumer arguments");
    cudaStream_t release_stream = mat->stream;
    if (!mat->ctx->execution->release_streams_by_partition.empty() &&
        mat->ctx->execution->release_streams_by_partition.front())
    {
        release_stream = mat->ctx->execution->release_streams_by_partition.front();
    }
    if (!release_stream) return set_error("missing compact matrix release stream");

    cudaError_t err = cudaSetDevice(mat->device);
    if (err != cudaSuccess) return set_error(err);
    // The caller retains this dominating completion until the release wait
    // is enqueued. Reuse it without creating or destroying another event.
    if (!completion) return set_error("missing compact consumer completion");
    err = cudaStreamWaitEvent(release_stream, completion, 0);
    if (err == cudaSuccess && mat->stream != consumer_stream && mat->stream != release_stream)
        err = cudaStreamWaitEvent(mat->stream, completion, 0);
    if (err != cudaSuccess)
        (void)gpu_context_retire_stream(mat->ctx, mat->device, consumer_stream);
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
    if (width <= sizeof(uint64_t))
    {
        // Typical gadget digits fit in one word and are already smaller than
        // each CRT prime. Avoid a 128-bit remainder for every input byte.
        for (size_t i = 0; i < width; ++i)
            value |= static_cast<uint64_t>(magnitude[i]) << (8 * i);
        return value < modulus ? value : value % modulus;
    }
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
struct CompactRowBlocks
{
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *inputs[kCompactRowBlocks];
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *outputs[kCompactRowBlocks];
    size_t ends[kCompactRowBlocks];
    size_t input_offsets[kCompactRowBlocks];
    size_t input_pitches[kCompactRowBlocks];
    size_t output_offsets[kCompactRowBlocks];
    size_t output_pitches[kCompactRowBlocks];
    __device__ size_t locate(size_t &row) const
    {
        size_t block = 0;
        while (row >= ends[block]) ++block;
        if (block) row -= ends[block - 1];
        return block;
    }
};
static_assert(sizeof(CompactRowBlocks) + 16 * sizeof(size_t) < 4096,
    "compact block ranges must fit portable CUDA launch arguments");

__global__ void compact_decompose_kernel(
    CompactRowBlocks blocks,
    const uint64_t *src_moduli,
    uint8_t *dst,
    size_t src_rows,
    size_t src_cols,
    size_t slots,
    size_t destination_offset,
    size_t destination_pitch,
    size_t n,
    size_t digits,
    size_t magnitude_bytes,
    uint32_t base_bits,
    bool balanced,
    bool small)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= src_rows * slots * src_cols * n) return;
    const size_t coeff = index % n;
    const size_t output_row = index / n / src_cols;
    const size_t column = index / n % src_cols;
    const size_t slot = output_row % slots;
    const size_t source_limb = small ? 0 : slot / digits;
    const size_t digit_idx = slot % digits;
    const uint64_t modulus = src_moduli[source_limb];
    size_t local_row = output_row / slots;
    const size_t block = blocks.locate(local_row);
    const auto descriptor = blocks.inputs[block][source_limb];
    const uint64_t residue = matrix_load_limb_u64(
        descriptor.base, blocks.input_offsets[block] + local_row * blocks.input_pitches[block] + column,
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
    const size_t out_poly = destination_offset + output_row * destination_pitch + column;
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
    const auto lhs_descriptor = blocks.inputs[block][global];
    const auto out_descriptor = blocks.outputs[block][global];
    uint64_t acc = 0;
    if (lazy_reduce)
    {
        unsigned __int128 wide_acc = acc;
        for (size_t k = 0; k < inner; ++k)
        {
            const uint64_t lhs = matrix_load_limb_u64(
                lhs_descriptor.base,
                blocks.input_offsets[block] + row * blocks.input_pitches[block] + k,
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
            const uint64_t lhs = matrix_load_limb_u64(
                lhs_descriptor.base,
                blocks.input_offsets[block] + row * blocks.input_pitches[block] + k,
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
        blocks.output_offsets[block] + row * blocks.output_pitches[block] + c,
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
    int *accepted,
    uint8_t *staging)
{
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
        atomicExch(accepted, 0);
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
    if (mat->hard_cutoff_host_accepted)
    {
        // The host decision word is read only after its decision event has been
        // synchronized, so once that event completes no DMA can touch it. Wait
        // for it here and return the pinned word immediately: destruction runs
        // outside any dispatch, where a deferred free could not claim its event.
        if (mat->hard_cutoff_decision_ready &&
            cudaEventSynchronize(mat->hard_cutoff_decision_ready) != cudaSuccess)
            mat->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
        else if (gpu_pinned_free(mat->hard_cutoff_host_accepted) != 0)
            mat->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
    }
    for (auto &owner : mat->hard_cutoff_owners)
        if (owner.release(stream) != 0)
            mat->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
    if (mat->owns_hard_cutoff_decision_event && mat->hard_cutoff_decision_ready &&
        cudaEventDestroy(mat->hard_cutoff_decision_ready) != cudaSuccess)
        mat->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
    mat->hard_cutoff_device_accepted = nullptr;
    mat->hard_cutoff_bound_words = nullptr;
    mat->hard_cutoff_half_modulus_words = nullptr;
    mat->hard_cutoff_modulus_words = nullptr;
    mat->hard_cutoff_subset_indices = nullptr;
    mat->hard_cutoff_garner_inverses = nullptr;
    mat->hard_cutoff_decision_ready = nullptr;
    mat->hard_cutoff_host_accepted = nullptr;
}

int small_initialize_hard_cutoff_plan(GpuSmallMatrix *mat)
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
    pinned_uploads.reserve(5);
    size_t workspace_index = 0;
    // A failed claim keeps its own precise error; the CUDA code is a sentinel.
    bool claim_failed = false;
    auto upload = [&](auto **device_out, const auto *source, size_t count) -> cudaError_t {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(*device_out)>>;
        const size_t bytes = count * sizeof(T);
        void *pinned = gpu_pinned_alloc(mat->ctx, bytes, alignof(T));
        if (!pinned) { claim_failed = true; return cudaErrorMemoryAllocation; }
        // Keep ownership before submission, including partially failed copies.
        pinned_uploads.push_back(pinned);
        std::memcpy(pinned, source, bytes);
        auto &owner = mat->hard_cutoff_owners[workspace_index++];
        if (owner.acquire(mat->ctx, mat->device, GPU_PREPARED_COMPACT_WORKSPACE,
                bytes, alignof(T), mat->stream) != 0) { claim_failed = true; return cudaErrorMemoryAllocation; }
        *device_out = reinterpret_cast<T *>(owner.data);
        cudaError_t err = cudaMemcpyAsync(*device_out, pinned, bytes, cudaMemcpyHostToDevice, mat->stream);
        return err;
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
    if (claim_failed) return 1;
    if (err != cudaSuccess) return set_error(err);
    auto &decision_owner = mat->hard_cutoff_owners.back();
    if (decision_owner.acquire(mat->ctx, mat->device, GPU_PREPARED_COMPACT_WORKSPACE,
            sizeof(int), alignof(int), mat->stream) != 0) return 1;
    mat->hard_cutoff_device_accepted = reinterpret_cast<int *>(decision_owner.data);
    if (err == cudaSuccess) {
        mat->hard_cutoff_host_accepted = static_cast<int *>(
            gpu_pinned_alloc(mat->ctx, sizeof(int), alignof(int)));
        if (!mat->hard_cutoff_host_accepted) return 1;
    }
    if (err == cudaSuccess) {
        mat->hard_cutoff_decision_ready = decision_owner.completion_event();
        mat->owns_hard_cutoff_decision_event = !mat->hard_cutoff_decision_ready;
        if (mat->owns_hard_cutoff_decision_event)
            err = cudaEventCreateWithFlags(&mat->hard_cutoff_decision_ready, cudaEventDisableTiming);
    }
    if (err != cudaSuccess) return set_error(err);
    mat->hard_cutoff_limb_count = limb_count;
    mat->hard_cutoff_subset_count = static_cast<int>(subset_indices.size());
    mat->hard_cutoff_words_per_coeff = static_cast<int>(words_per_coeff);
    return 0;
}

}

extern "C" int gpu_small_matrix_create(
    GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    const uint64_t *bound_words,
    size_t bound_word_count,
    bool initialize_zero,
    GpuSmallMatrix **out)
{
    if (!ctx || !ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_create");
    GpuAllocationActivity activity(ctx->execution.get(), -1);

    if (!ctx || !out || !bound_words || bound_word_count == 0 || magnitude_bytes == 0)
        return set_error("invalid gpu_small_matrix_create arguments");
    *out = nullptr;
    if (ctx->execution->unretired_work.load(std::memory_order_acquire))
        return set_error("GPU execution has unretired work; compact allocation rejected");
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
    if (err == cudaSuccess) {
        const int status = mat->payload_owner.acquire(ctx, mat->device,
            GPU_PREPARED_COMPACT_PAYLOAD, mat->payload_bytes, 256, mat->stream);
        if (status != 0) {
            if (mat->owns_write_event && mat->write_done && cudaEventDestroy(mat->write_done) != cudaSuccess)
                gpu_execution_mark_allocation_unknown(ctx->execution.get());
            delete mat;
            return status;
        }
        mat->payload = mat->payload_owner.data;
        // A prepared payload's reuse event also provides its writer completion.
        // Admission already captured the prior release before this new record.
        mat->write_done = mat->payload_owner.completion_event();
        mat->owns_write_event = !mat->write_done;
        if (mat->owns_write_event) {
            if (ctx->execution->resource_admission_required.load(std::memory_order_acquire)) {
                const int status = mat->empty_payload_completion.acquire(
                    ctx, mat->device, GPU_PREPARED_COMPLETION_EVENT);
                if (status != 0) { delete mat; return status; }
                mat->write_done = mat->empty_payload_completion.event;
                mat->owns_write_event = false;
            } else {
                // Prepared payload slots carry their completion event; only an
                // open-domain payload creates one, so this is not a claim.
                err = cudaEventCreateWithFlags(&mat->write_done, cudaEventDisableTiming);
            }
        }
    }
    if (err == cudaSuccess && initialize_zero && mat->payload_bytes != 0)
        err = cudaMemsetAsync(mat->payload, 0, mat->payload_bytes, mat->stream);
    if (err == cudaSuccess)
        err = cudaEventRecord(mat->write_done, mat->stream);
    if (err != cudaSuccess)
    {
        if (mat->payload_owner.release() != 0)
            gpu_execution_mark_allocation_unknown(ctx->execution.get());
        if (mat->owns_write_event && mat->write_done && cudaEventDestroy(mat->write_done) != cudaSuccess)
            gpu_execution_mark_allocation_unknown(ctx->execution.get());
        delete mat;
        return set_error(err);
    }
    mat->write_done_valid = true;
    *out = mat;
    return 0;
}

extern "C" void gpu_small_matrix_destroy(GpuSmallMatrix *mat)
{
    if (!mat) return;
    if (!mat->ctx || !mat->ctx->execution) { delete mat; return; }
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);
    auto &execution = *mat->ctx->execution;
    if (execution.unretired_work.load(std::memory_order_acquire)) return;
    cudaError_t error = mat->device < 0 ? cudaErrorInvalidDevice : cudaSetDevice(mat->device);
    cudaStream_t release_stream = execution.release_streams_by_partition.empty()
        ? nullptr : execution.release_streams_by_partition[0];
    if (error == cudaSuccess && !release_stream) error = cudaErrorInvalidResourceHandle;
    if (error == cudaSuccess && mat->write_done_valid)
        error = cudaStreamWaitEvent(release_stream, mat->write_done, 0);
    if (error != cudaSuccess)
    {
        // The context contains device tables used by pending readers as well
        // as this payload. Retain that owner when release ordering is unknown.
        execution.memory_release_failed.store(true, std::memory_order_release);
        execution.unretired_work.store(true, std::memory_order_release);
        return;
    }
    if (error == cudaSuccess && mat->owns_payload)
    {
        small_release_hard_cutoff_plan(mat, release_stream);
        if (execution.memory_release_failed.load(std::memory_order_acquire))
        {
            execution.unretired_work.store(true, std::memory_order_release);
            return;
        }
        if (mat->payload_owner.release(release_stream) != 0) error = cudaErrorUnknown;
    }
    if (error != cudaSuccess)
    {
        execution.memory_release_failed.store(true, std::memory_order_release);
        execution.unretired_work.store(true, std::memory_order_release);
        return;
    }
    if (mat->owns_write_event && mat->write_done &&
        cudaEventDestroy(mat->write_done) != cudaSuccess)
        execution.memory_release_failed.store(true, std::memory_order_release);
    delete mat;
}

extern "C" int gpu_small_matrix_wait(const GpuSmallMatrix *mat)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_wait");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat || !mat->write_done)
        return set_error("invalid compact matrix wait arguments");
    if (small_set_device(mat) != 0) return 1;
    if (!mat->write_done_valid) return 0;
    const cudaError_t err = cudaEventSynchronize(mat->write_done);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_copy");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!src || !src->ctx || !src->ctx->execution)
        return set_error("invalid compact copy source owner");
    GpuAllocationActivity source_activity(src->ctx->execution.get(), -1);

    // Compact coefficients are signed integers independent of the CRT basis.
    // A containing-basis conversion may therefore cross parameter contexts on
    // the same CUDA device. The destination owns its allocation and stream;
    // source readiness and consumption are connected by the existing events.
    if (!out || !src || out->device != src->device || out->rows != src->rows || out->cols != src->cols ||
        out->n != src->n || out->magnitude_bytes != src->magnitude_bytes || out->payload_bytes != src->payload_bytes)
        return set_error("incompatible compact matrix copy");
    if (small_set_device(out) != 0 || small_wait(src, out->stream) != 0) return 1;
    const size_t row_bytes = out->cols * out->n * (1 + out->magnitude_bytes);
    const size_t out_pitch = out->storage_cols * out->n * (1 + out->magnitude_bytes);
    const size_t src_pitch = src->storage_cols * src->n * (1 + src->magnitude_bytes);
    auto *destination = out->payload + out->column_offset * out->n * (1 + out->magnitude_bytes);
    const auto *source = src->payload + src->column_offset * src->n * (1 + src->magnitude_bytes);
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, out_pitch, source, src_pitch, row_bytes, out->rows,
        cudaMemcpyDeviceToDevice, out->stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, out->stream) != 0) return 1;
    return small_track_consumer(src, out->stream, out->write_done);
}

extern "C" int gpu_small_matrix_copy_columns(
    GpuSmallMatrix *out,
    const GpuSmallMatrix *src,
    size_t source_column_start)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_copy_columns");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (!out || !src || out->ctx != src->ctx || out->rows != src->rows ||
        out->n != src->n || out->magnitude_bytes != src->magnitude_bytes ||
        source_column_start > src->cols || out->cols > src->cols - source_column_start)
        return set_error("incompatible compact matrix column slice");
    if (small_set_device(out) != 0 || small_wait(src, out->stream) != 0) return 1;
    const size_t coefficient_bytes = 1 + out->magnitude_bytes;
    const size_t column_bytes = out->n * coefficient_bytes;
    const size_t destination_pitch = out->storage_cols * column_bytes;
    const size_t source_pitch = src->storage_cols * column_bytes;
    auto *destination = out->payload + out->column_offset * column_bytes;
    const auto *source = src->payload + (src->column_offset + source_column_start) * column_bytes;
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, destination_pitch, source, source_pitch,
        out->cols * column_bytes, out->rows, cudaMemcpyDeviceToDevice, out->stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, out->stream) != 0) return 1;
    return small_track_consumer(src, out->stream, out->write_done);
}

extern "C" int gpu_small_matrix_copy_range(
    GpuSmallMatrix *out,
    size_t destination_column_start,
    const GpuSmallMatrix *src,
    size_t source_column_start,
    size_t columns)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_copy_range");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!src || !src->ctx || !src->ctx->execution)
        return set_error("invalid compact range copy source owner");
    GpuAllocationActivity source_activity(src->ctx->execution.get(), -1);

    // Compact payloads are signed coefficients independent of the CRT basis, so
    // a retained destination in a containing basis may live in another context
    // on the same device. The destination owner records the write; the source
    // stays alive until this consumer completes.
    if (out->device != src->device || out->rows != src->rows || out->n != src->n ||
        out->magnitude_bytes != src->magnitude_bytes || columns == 0 ||
        destination_column_start > out->cols || columns > out->cols - destination_column_start ||
        source_column_start > src->cols || columns > src->cols - source_column_start)
        return set_error("incompatible compact matrix range copy");
    if (small_set_device(out) != 0 || small_wait(src, out->stream) != 0) return 1;
    const size_t coefficient_bytes = 1 + out->magnitude_bytes;
    const size_t column_bytes = out->n * coefficient_bytes;
    const size_t destination_pitch = out->storage_cols * column_bytes;
    const size_t source_pitch = src->storage_cols * column_bytes;
    auto *destination =
        out->payload + (out->column_offset + destination_column_start) * column_bytes;
    const auto *source = src->payload + (src->column_offset + source_column_start) * column_bytes;
    const cudaError_t err = cudaMemcpy2DAsync(
        destination, destination_pitch, source, source_pitch,
        columns * column_bytes, out->rows, cudaMemcpyDeviceToDevice, out->stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, out->stream) != 0) return 1;
    return small_track_consumer(src, out->stream, out->write_done);
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
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_load_coefficients");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat || !payload || payload_len != mat->payload_bytes || !mat->owns_payload)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0) return 1;
    if (payload_len == 0) return 0;
    auto *staging = static_cast<uint8_t *>(gpu_pinned_alloc(mat->ctx, payload_len, alignof(uint8_t)));
    if (!staging) return 1;
    std::memcpy(staging, payload, payload_len);
    cudaError_t err = cudaMemcpyAsync(
        mat->payload, staging, payload_len, cudaMemcpyHostToDevice, mat->stream);
    if (err != cudaSuccess)
    {
        void *pointer = staging;
        if (gpu_defer_pinned_frees(mat->ctx, mat->device, mat->stream, &pointer, 1) != 0)
            gpu_execution_mark_allocation_unknown(mat->ctx->execution.get());
        return set_error(err);
    }
    const int record_status = small_record(mat, mat->stream);
    void *deferred[] = {staging};
    const int defer_status = gpu_defer_pinned_frees(
        mat->ctx, mat->device, mat->stream, deferred, 1);
    if (record_status != 0)
    {
        (void)gpu_context_retire_stream(mat->ctx, mat->device, mat->stream);
        return record_status;
    }
    return defer_status;
}

extern "C" int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat, uint8_t *payload, size_t payload_len)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_store_coefficients");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    if (payload_len == 0) return 0;
    const size_t coefficient_bytes = 1 + mat->magnitude_bytes;
    const size_t row_bytes = mat->cols * mat->n * coefficient_bytes;
    const size_t source_pitch = mat->storage_cols * mat->n * coefficient_bytes;
    const auto *source = mat->payload + mat->column_offset * mat->n * coefficient_bytes;
    GpuCudaResource completion;
    const int acquired = completion.acquire(mat->ctx, mat->device, GPU_PREPARED_COMPLETION_EVENT);
    if (acquired != 0) return acquired;
    cudaError_t err = cudaMemcpy2DAsync(
        payload, row_bytes, source, source_pitch, row_bytes, mat->rows,
        cudaMemcpyDeviceToHost, mat->stream);
    if (err != cudaSuccess) return set_error(err);
    err = cudaEventRecord(completion.event, mat->stream);
    if (err != cudaSuccess) {
        completion.quarantine();
        // Preserve the existing error-only completion fallback for the caller's
        // borrowed host destination. Missing permits fail before the copy above.
        const cudaError_t sync = cudaStreamSynchronize(mat->stream);
        return set_error(sync == cudaSuccess ? err : sync);
    }
    err = cudaEventSynchronize(completion.event);
    if (err != cudaSuccess) {
        completion.quarantine();
        return set_error(err);
    }
    return completion.release();
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
    const GpuMatrixRange *source_views,
    const GpuMatrixRange *destination_view)
{
    if (!sources || block_count == 0 || block_count > kCompactRowBlocks)
        return set_error("invalid compact decomposition blocks");
    const GpuMatrix *src = sources[0];
    if (!src || !out || !src->ctx || src->ctx != out->ctx || !max_coefficient_bound ||
        bound_word_count == 0 || base_bits == 0 || base_bits >= 63 ||
        (small_mode != 0 && small_mode != 1) || src->format != GPU_POLY_FORMAT_COEFF)
        return set_error("invalid compact decomposition arguments");
    GpuAllocationActivity activity(src->ctx->execution.get(), -1);
    const size_t limbs = static_cast<size_t>(src->level + 1);
    if (src->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || src->ctx->limb_gpu_ids.size() < limbs)
        return set_error("invalid compact decomposition level");
    CompactRowBlocks blocks{};
    const size_t columns = source_views ? source_views[0].column_end - source_views[0].column_start : src->cols;
    size_t rows = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *source = sources[block];
        if (!source || source->ctx != src->ctx || source->level != src->level ||
            source->format != GPU_POLY_FORMAT_COEFF)
            return set_error("incompatible compact decomposition block");
        const GpuMatrixRange range = source_views ? source_views[block] :
            GpuMatrixRange{0, source->rows, 0, source->cols};
        if (range.row_start >= range.row_end || range.row_end > source->rows ||
            range.column_start >= range.column_end || range.column_end > source->cols ||
            range.column_end - range.column_start != columns ||
            (source->cols && source->rows > SIZE_MAX / source->cols) ||
            !small_add_size(rows, range.row_end - range.row_start, &rows))
            return set_error("invalid compact decomposition rectangle");
        blocks.input_offsets[block] = range.row_start * source->cols + range.column_start;
        blocks.input_pitches[block] = source->cols;
        blocks.ends[block] = rows;
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
    const GpuMatrixRange destination = destination_view ? *destination_view :
        GpuMatrixRange{0, out->rows, 0, out->cols};
    if (!out->owns_payload || destination.row_start >= destination.row_end ||
        destination.row_end > out->rows || destination.row_end - destination.row_start != expected_rows ||
        destination.column_start >= destination.column_end || destination.column_end > out->cols ||
        destination.column_end - destination.column_start != columns || out->bound_words.size() != 1 ||
        out->bound_words[0] != expected_bound)
        return set_error("compact decomposition shape or bound mismatch");
    size_t coefficients = 0;
    if (!small_mul_size(expected_rows, columns, &coefficients) ||
        !small_mul_size(coefficients, out->n, &coefficients))
        return set_error("compact decomposition coefficient count overflow");
    const size_t launch_blocks = coefficients / kSmallThreads + (coefficients % kSmallThreads != 0);
    if (launch_blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("compact decomposition exceeds CUDA grid capacity");
    if (small_set_device(out) != 0 || small_wait(out, out->stream) != 0) return 1;
    std::vector<uint64_t> requested_bound(max_coefficient_bound,
                                          max_coefficient_bound + bound_word_count);
    if (requested_bound != out->bound_words)
        return set_error("compact decomposition bound metadata mismatch");
    cudaStream_t stream = out->stream;
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
    if (dispatch_slot >= src->ctx->ring_device_constants.size())
        return set_error("missing compact decomposition constants");
    const auto &constants = src->ctx->ring_device_constants[dispatch_slot];
    if (constants.device != out->device || constants.limb_count < limbs || !constants.moduli)
        return set_error("invalid compact decomposition constants");
    const size_t slots = digits * (small ? 1 : limbs - dropped_moduli);
    compact_decompose_kernel<<<static_cast<unsigned int>(launch_blocks), kSmallThreads, 0, stream>>>(
        blocks, constants.moduli, out->payload, rows, columns, slots,
        destination.row_start * out->storage_cols + destination.column_start,
        out->storage_cols, out->n, digits, out->magnitude_bytes, base_bits, !small, small);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess || small_record(out, stream) != 0)
    {
        (void)gpu_context_retire_stream(out->ctx, out->device, stream);
        return err != cudaSuccess ? set_error(err) : 1;
    }
    for (size_t block = 0; block < block_count; ++block)
        if (matrix_track_all_limb_consumers(sources[block], out->device, stream, out->write_done, true, true) != 0)
        {
            (void)gpu_context_retire_stream(out->ctx, out->device, stream);
            return 1;
        }
    return 0;
}

extern "C" int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_prepare_preimage_hard_cutoff");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat) return set_error("invalid compact preimage hard-cutoff owner");
    if (mat->hard_cutoff_subset_count > 0) return 0;
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    const int status = small_initialize_hard_cutoff_plan(mat);
    if (status != 0)
    {
        small_release_hard_cutoff_plan(mat, mat->stream);
        return status;
    }
    return small_record(mat, mat->stream);
}

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
    if (!dst || !dst->ctx || !dst->ctx->execution)
        return set_error("invalid allocation owner in gpu_small_matrix_try_pack_preimage_hard_cutoff_tile");
    GpuAllocationActivity activity(dst->ctx->execution.get(), -1);

    if (!dst || !src || !bound_words || bound_word_count == 0 || !accepted_out ||
        src->ctx != dst->ctx || src->format != GPU_POLY_FORMAT_COEFF ||
        rows == 0 || cols == 0 || src->rows != rows || src->cols != cols ||
        dst_row > dst->rows || rows > dst->rows - dst_row ||
        dst_col > dst->cols || cols > dst->cols - dst_col ||
        bound_word_count != dst->bound_words.size() ||
        !std::equal(bound_words, bound_words + bound_word_count, dst->bound_words.begin()))
        return set_error("invalid compact tile arguments");
    if (small_set_device(dst) != 0 || small_wait(dst, dst->stream) != 0) return 1;
    if (src->level < 0) return set_error("invalid compact tile source level");
    const size_t limb_count = static_cast<size_t>(src->level + 1);
    if (limb_count == 0 || limb_count > static_cast<size_t>(kMaxRnsLimbs) ||
        src->ctx->limb_gpu_ids.size() < limb_count ||
        limb_count != dst->hard_cutoff_limb_count ||
        dst->hard_cutoff_subset_count <= 0 || dst->hard_cutoff_words_per_coeff <= 0 ||
        !dst->hard_cutoff_garner_inverses || !dst->hard_cutoff_subset_indices ||
        !dst->hard_cutoff_modulus_words || !dst->hard_cutoff_half_modulus_words ||
        !dst->hard_cutoff_bound_words || !dst->hard_cutoff_device_accepted ||
        !dst->hard_cutoff_host_accepted || !dst->hard_cutoff_decision_ready)
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
        if (id.y != limb || matrix_wait_limb_stream(src, id, dst->device, dst->stream, true, true) != 0)
            return set_error("invalid compact tile active CRT limb");
    }
    if (dispatch_slot >= src->shared_limb_buffers.size() ||
        dispatch_slot >= src->ctx->ring_device_constants.size() ||
        src->shared_limb_buffers[dispatch_slot].limb_count < limb_count ||
        !src->shared_limb_buffers[dispatch_slot].device_descriptors)
        return set_error("missing compact tile device descriptors");
    const auto &constants = src->ctx->ring_device_constants[dispatch_slot];
    if (constants.device != dst->device || constants.limb_count < limb_count ||
        !constants.moduli)
        return set_error("missing compact tile device moduli");

    uint8_t *d_staging = nullptr;
    GpuDeviceWorkspace staging_owner;
    cudaError_t err = cudaSuccess;
    size_t staging_bytes = 0;
    if (!small_mul_size(total_coefficients, 1 + dst->magnitude_bytes, &staging_bytes))
        return set_error("compact tile staging size overflow");
    if (staging_owner.acquire(dst->ctx, dst->device, GPU_PREPARED_COMPACT_WORKSPACE,
            staging_bytes, alignof(uint8_t), dst->stream) != 0) return 1;
    d_staging = staging_owner.data;
    if (err == cudaSuccess)
        err = cudaMemsetAsync(dst->hard_cutoff_device_accepted, 1, sizeof(int), dst->stream);
    if (err == cudaSuccess)
        compact_check_pack_preimage_kernel<<<
            (total_coefficients + kSmallThreads - 1) / kSmallThreads,
            kSmallThreads, 0, dst->stream>>>(
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
                dst->hard_cutoff_device_accepted,
                d_staging);
    if (err == cudaSuccess) err = cudaGetLastError();
    for (size_t limb = 0; limb < limb_count && err == cudaSuccess; ++limb)
    {
        if (matrix_track_limb_consumer_readonly(
                src, src->ctx->limb_gpu_ids[limb], dst->device, dst->stream) != 0)
            err = cudaErrorInvalidResourceHandle;
    }
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(
            dst->hard_cutoff_host_accepted, dst->hard_cutoff_device_accepted,
            sizeof(int), cudaMemcpyDeviceToHost, dst->stream);
    if (err == cudaSuccess)
        err = cudaEventRecord(dst->hard_cutoff_decision_ready, dst->stream);
    if (err == cudaSuccess)
        err = cudaEventSynchronize(dst->hard_cutoff_decision_ready);
    if (err != cudaSuccess)
    {
        if (staging_owner.release() != 0)
            gpu_execution_mark_allocation_unknown(dst->ctx->execution.get());
        return set_error(err);
    }
    *accepted_out = *dst->hard_cutoff_host_accepted;
    if (*accepted_out != 0)
    {
        const dim3 commit_grid((staging_bytes + kSmallThreads - 1) / kSmallThreads);
        compact_commit_preimage_tile_kernel<<<commit_grid, kSmallThreads, 0, dst->stream>>>(
            dst->payload, d_staging, dst->n, rows, cols, dst->cols, dst_row, dst_col,
            1 + dst->magnitude_bytes);
        err = cudaGetLastError();
        if (err == cudaSuccess && small_record(dst, dst->stream) != 0)
            err = cudaErrorInvalidResourceHandle;
    }
    if (staging_owner.release() != 0)
        gpu_execution_mark_allocation_unknown(dst->ctx->execution.get());
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_matrix_query_small_rhs_workspace_bytes(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t *narrow_bytes, size_t *wide_bytes)
{
    if (!ctx || !narrow_bytes || !wide_bytes || level < 0 || ctx->N < 2 ||
        !is_power_of_two_u32(static_cast<uint32_t>(ctx->N)))
        return set_error("invalid compact RHS workspace query");
    const size_t limbs = static_cast<size_t>(level) + 1;
    if (limbs > kMaxSmallLimbCount || limbs > ctx->limb_prime_ids.size())
        return set_error("invalid compact RHS workspace level");
    size_t narrow = 0;
    for (size_t limb = 0; limb < limbs; ++limb) {
        const int index = ctx->limb_prime_ids[limb];
        if (index < 0 || static_cast<size_t>(index) >= ctx->moduli.size())
            return set_error("invalid compact RHS workspace modulus");
        narrow += ctx->moduli[index] <= UINT32_MAX;
    }
    size_t words = 0;
    if (!small_mul_size(inner, columns, &words) || !small_mul_size(words, ctx->N, &words) ||
        !small_mul_size(words, narrow, narrow_bytes) || !small_mul_size(*narrow_bytes, sizeof(uint32_t), narrow_bytes) ||
        !small_mul_size(words, limbs - narrow, wide_bytes) || !small_mul_size(*wide_bytes, sizeof(uint64_t), wide_bytes))
        return set_error("compact RHS workspace overflow");
    return 0;
}

extern "C" int gpu_matrix_mul_small_rhs(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    size_t block_count,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report, const GpuMatrixBatchView *views)
{
    if (!outputs || !inputs || block_count == 0 || block_count > kCompactRowBlocks)
        return set_error("invalid compact multiplication blocks");
    GpuMatrix *out = outputs[0];
    const GpuMatrix *lhs_eval = inputs[0];
    if (!out || !lhs_eval || !rhs_small || !out->ctx || out->ctx != lhs_eval->ctx || out->ctx != rhs_small->ctx ||
        lhs_eval->format != GPU_POLY_FORMAT_EVAL || out->format != GPU_POLY_FORMAT_EVAL ||
        !allocation_report)
        return set_error("invalid compact RHS multiplication arguments");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    const size_t limbs = static_cast<size_t>(lhs_eval->level + 1);
    if (lhs_eval->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || out->level != lhs_eval->level ||
        out->ctx->limb_gpu_ids.size() < limbs || out->ctx->N < 2 ||
        !is_power_of_two_u32(static_cast<uint32_t>(out->ctx->N)))
        return set_error("invalid compact RHS multiplication level");
    const auto valid = [](const GpuMatrixRange &r, const GpuMatrix *m) {
        return r.row_start <= r.row_end && r.row_end <= m->rows &&
            r.column_start <= r.column_end && r.column_end <= m->cols;
    };
    CompactRowBlocks blocks{};
    size_t rows = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *left = inputs[block];
        const auto *output = outputs[block];
        if (!left || !output || left->ctx != lhs_eval->ctx || output->ctx != out->ctx ||
            left->level != lhs_eval->level || output->level != out->level ||
            left->format != GPU_POLY_FORMAT_EVAL || output->format != GPU_POLY_FORMAT_EVAL ||
            output->ctx->N != lhs_eval->ctx->N)
            return set_error("incompatible compact multiplication block");
        const GpuMatrixRange input_range = views ? views[block].left : GpuMatrixRange{0,left->rows,0,left->cols};
        const GpuMatrixRange output_range = views ? views[block].output : GpuMatrixRange{0,output->rows,0,output->cols};
        if (!valid(input_range,left) || !valid(output_range,output) ||
            input_range.row_end-input_range.row_start != output_range.row_end-output_range.row_start ||
            input_range.column_end-input_range.column_start != rhs_small->rows ||
            output_range.column_end-output_range.column_start != rhs_small->cols ||
            !small_add_size(rows,input_range.row_end-input_range.row_start,&rows))
            return set_error("invalid compact multiplication rectangle");
        if ((left->cols && left->rows > SIZE_MAX / left->cols) || (output->cols && output->rows > SIZE_MAX / output->cols))
            return set_error("compact multiplication owner size overflow");
        for (size_t other=0; other<block_count; ++other)
            if (inputs[other]==output || (other<block && outputs[other]==output))
                return set_error("compact multiplication requires independent output owners");
        blocks.input_offsets[block]=input_range.row_start*left->cols+input_range.column_start;
        blocks.input_pitches[block]=left->cols;
        blocks.output_offsets[block]=output_range.row_start*output->cols+output_range.column_start;
        blocks.output_pitches[block]=output->cols;
        blocks.ends[block] = rows;
    }
    if (small_set_device(rhs_small) != 0) return 1;
    cudaStream_t stream = nullptr;
    int dispatch_device = -1;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    for (size_t block = 0; block < block_count; ++block)
    {
        const auto *lhs_eval = inputs[block];
        auto *out = outputs[block];
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
                if (matrix_limb_stream(out, id, &stream) != 0) return 1;
            }
            else if (out_device != dispatch_device)
                return set_error("compact RHS multiplication requires one device");
        }
        if (matrix_wait_all_limb_streams(lhs_eval, rhs_small->device, stream, true, true) != 0 ||
            matrix_wait_all_limb_streams(out, rhs_small->device, stream, true) != 0) return 1;
        blocks.inputs[block] = lhs_eval->shared_limb_buffers[dispatch_slot].device_descriptors;
        blocks.outputs[block] = out->shared_limb_buffers[dispatch_slot].device_descriptors;
    }
    if (!stream) return set_error("missing compact multiplication stream");
    if (small_wait(rhs_small, stream) != 0) return 1;
    const size_t n = rhs_small->n;
    const size_t inner = rhs_small->rows;
    const size_t cols = rhs_small->cols;
    if (dispatch_slot >= out->ctx->ring_device_constants.size())
        return set_error("missing compact multiplication NTT partition");
    const auto &constants = out->ctx->ring_device_constants[dispatch_slot];
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
    GpuDeviceWorkspace workspace_u32_owner, workspace_u64_owner;
    auto release = [&]() -> cudaError_t {
        const int narrow = workspace_u32_owner.release();
        const int wide = workspace_u64_owner.release();
        workspace_u32 = nullptr;
        workspace_u64 = nullptr;
        return narrow == 0 && wide == 0 ? cudaSuccess : cudaErrorUnknown;
    };
    auto fail = [&](cudaError_t failure) -> int {
        // Join already submitted readers to the owner before any wrapper can
        // release its inputs. Failed retirement quarantines that owner instead
        // of waiting on this host thread or freeing uncertain scratch.
        if (gpu_context_retire_stream(out->ctx, rhs_small->device, stream) != 0)
            return 1;
        const cudaError_t cleanup_err = release();
        if (failure == cudaSuccess) failure = cleanup_err;
        return set_error(failure);
    };
    size_t lhs_eval_bytes = 0;
    size_t full_output_bytes = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        GpuMatrixAllocationBytes lhs_allocation{}, output_allocation{};
        const auto *left = inputs[block];
        const auto *output = outputs[block];
        if (gpu_matrix_query_allocation_bytes(left->ctx, left->level, left->rows, left->cols,
                left->format, &lhs_allocation) != 0 ||
            gpu_matrix_query_allocation_bytes(output->ctx, output->level, output->rows, output->cols,
                output->format, &output_allocation) != 0 ||
            !small_add_size(lhs_eval_bytes, lhs_allocation.total_bytes, &lhs_eval_bytes) ||
            !small_add_size(full_output_bytes, output_allocation.total_bytes, &full_output_bytes))
            return set_error("compact block allocation overflow");
    }
    size_t workspace_words_per_limb = 0;
    size_t workspace_u32_bytes = 0;
    size_t workspace_u64_bytes = 0;
    size_t expanded_rhs_workspace_bytes = 0;
    if (!small_mul_size(inner, cols, &workspace_words_per_limb) ||
        !small_mul_size(workspace_words_per_limb, n, &workspace_words_per_limb) ||
        gpu_matrix_query_small_rhs_workspace_bytes(out->ctx, out->level, inner, cols,
            &workspace_u32_bytes, &workspace_u64_bytes) != 0 ||
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
    if (workspace_u32_owner.acquire(out->ctx, dispatch_device, GPU_PREPARED_COMPACT_WORKSPACE,
            workspace_u32_bytes, alignof(uint32_t), stream) != 0)
        return fail(cudaErrorMemoryAllocation);
    if (workspace_u64_owner.acquire(out->ctx, dispatch_device, GPU_PREPARED_COMPACT_WORKSPACE,
            workspace_u64_bytes, alignof(uint64_t), stream) != 0)
        return fail(cudaErrorMemoryAllocation);
    workspace_u32 = reinterpret_cast<uint32_t *>(workspace_u32_owner.data);
    workspace_u64 = reinterpret_cast<uint64_t *>(workspace_u64_owner.data);

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
    for (size_t block = 0; block < block_count; ++block)
        if (matrix_track_all_limb_consumers(inputs[block], rhs_small->device, stream, completion, true, true) != 0)
            return fail(cudaErrorInvalidResourceHandle);
    if (small_track_consumer(rhs_small, stream, completion) != 0)
        return fail(cudaErrorInvalidResourceHandle);
    const cudaError_t cleanup_err = release();
    if (cleanup_err != cudaSuccess) return set_error(cleanup_err);
    return 0;
}
