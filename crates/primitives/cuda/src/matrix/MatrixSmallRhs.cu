#include "matrix/MatrixSmallRhs.cuh"

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
    cudaEvent_t write_done = nullptr;
    bool write_done_valid = false;
};

namespace
{
constexpr int kSmallThreads = 256;
constexpr size_t kMaxSmallLimbCount = GPU_RUNTIME_MAX_LIMBS;

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
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        return sync_err == cudaSuccess ? err : sync_err;
    }
    const cudaError_t sync_err = cudaEventSynchronize(completion);
    if (sync_err == cudaSuccess)
        cudaEventDestroy(completion);
    // On an asynchronous device error the event is deliberately leaked: its
    // completion state is uncertain, so destroying it would weaken safety.
    return sync_err;
}

int small_track_consumer(const GpuSmallMatrix *mat, cudaStream_t consumer_stream)
{
    if (!mat || !mat->ctx || !consumer_stream || mat->device < 0)
        return set_error("invalid compact matrix consumer arguments");
    cudaStream_t release_stream = mat->stream;
    if (!mat->ctx->release_streams_by_partition.empty() &&
        mat->ctx->release_streams_by_partition.front())
    {
        release_stream = mat->ctx->release_streams_by_partition.front();
    }
    if (!release_stream) return set_error("missing compact matrix release stream");

    cudaError_t err = cudaSetDevice(mat->device);
    if (err != cudaSuccess) return set_error(err);
    cudaEvent_t consumer_done = nullptr;
    err = cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        (void)small_fence_stream_with_event(consumer_stream);
        return set_error(err);
    }
    err = cudaEventRecord(consumer_done, consumer_stream);
    if (err != cudaSuccess)
    {
        cudaEventDestroy(consumer_done);
        (void)small_fence_stream_with_event(consumer_stream);
        return set_error(err);
    }
    err = cudaStreamWaitEvent(release_stream, consumer_done, 0);
    if (err != cudaSuccess)
    {
        const cudaError_t fence_err = cudaEventSynchronize(consumer_done);
        if (fence_err == cudaSuccess)
            cudaEventDestroy(consumer_done);
        else
            (void)small_fence_stream_with_event(consumer_stream);
        return set_error(err);
    }
    err = cudaEventDestroy(consumer_done);
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

__global__ void compact_decompose_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *src_descriptors,
    const uint64_t *src_moduli,
    uint8_t *dst,
    size_t src_rows,
    size_t src_cols,
    size_t out_rows,
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
    if (coeff >= n || poly >= src_rows * src_cols || slot >= out_rows / src_rows) return;
    const size_t source_limb = small ? 0 : slot / digits;
    const size_t digit_idx = slot % digits;
    const uint64_t modulus = src_moduli[source_limb];
    const auto descriptor = src_descriptors[source_limb];
    const uint64_t residue = matrix_load_limb_u64(
        descriptor.base, poly, coeff, descriptor.stride, descriptor.width);
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
    const size_t row = poly / src_cols;
    const size_t col = poly % src_cols;
    const size_t out_row = row * (out_rows / src_rows) + slot;
    const size_t out_poly = out_row * src_cols + col;
    const size_t out_idx = (out_poly * n + coeff) * (1 + magnitude_bytes);
    compact_store_signed(dst + out_idx, magnitude_bytes, digit);
}

__global__ void compact_unpack_twist_kernel(
    const uint8_t *payload,
    uint64_t *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t inner,
    size_t cols,
    size_t source_cols,
    size_t n,
    size_t magnitude_bytes,
    size_t source_column_offset)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = limb_count * inner * cols * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t q = idx / n;
    const size_t c = q % cols;
    const size_t k = (q / cols) % inner;
    const size_t local_limb = q / (inner * cols);
    const size_t rhs_col = source_column_offset + c;
    const size_t width = 1 + magnitude_bytes;
    const uint8_t *src = payload + ((k * source_cols + rhs_col) * n + coeff) * width;
    const uint64_t modulus = moduli[limb_offset + local_limb];
    uint64_t value = compact_mod_magnitude(src + 1, magnitude_bytes, modulus);
    if (src[0] == 2 && value != 0) value = modulus - value;
    const size_t twiddle_index = (limb_offset + local_limb) * n + coeff;
    workspace[idx] = mul_mod_shoup_u64(
        value, twiddles[twiddle_index], twiddle_shoup[twiddle_index], modulus);
}

__global__ void compact_ntt_bit_reverse_kernel(
    uint64_t *workspace,
    size_t limb_count,
    size_t poly_count,
    size_t n,
    uint32_t log_n)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = limb_count * poly_count * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t q = idx / n;
    const size_t poly = q % poly_count;
    const size_t local = q / poly_count;
    const uint32_t reverse = __brev(static_cast<uint32_t>(coeff)) >> (32 - log_n);
    if (coeff >= reverse) return;
    const size_t left = (local * poly_count + poly) * n + coeff;
    const size_t right = (local * poly_count + poly) * n + reverse;
    const uint64_t tmp = workspace[left];
    workspace[left] = workspace[right];
    workspace[right] = tmp;
}

__global__ void compact_ntt_stage_kernel(
    uint64_t *workspace,
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
    const size_t local = q / poly_count;
    const uint32_t half = len / 2;
    const uint32_t group = static_cast<uint32_t>(butterfly) / half;
    const uint32_t j = static_cast<uint32_t>(butterfly) % half;
    const uint32_t i = group * len + j;
    const size_t global = limb_offset + local;
    const uint64_t modulus = moduli[global];
    const size_t base = (local * poly_count + poly) * n;
    const uint64_t u = workspace[base + i];
    const size_t twiddle_index = global * n + 2u * (n / len) * j;
    const uint64_t v = mul_mod_shoup_u64(
        workspace[base + i + half], twiddles[twiddle_index], twiddle_shoup[twiddle_index], modulus);
    workspace[base + i] = add_mod_u64(u, v, modulus);
    workspace[base + i + half] = sub_mod_u64(u, v, modulus);
}

__global__ void compact_accumulate_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *lhs_descriptors,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out_descriptors,
    const uint64_t *moduli,
    size_t limb_offset,
    const uint64_t *workspace,
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
    const size_t row = (q / out_cols) % rows;
    const size_t local = q / (rows * out_cols);
    const size_t global = limb_offset + local;
    const uint64_t modulus = moduli[global];
    const auto lhs_descriptor = lhs_descriptors[local];
    const auto out_descriptor = out_descriptors[local];
    uint64_t acc = 0;
    if (lazy_reduce)
    {
        unsigned __int128 wide_acc = acc;
        for (size_t k = 0; k < inner; ++k)
        {
            const uint64_t lhs = matrix_load_limb_u64(
                lhs_descriptor.base,
                row * inner + k,
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
                row * inner + k,
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
    if (mat->hard_cutoff_device_accepted)
        cudaFreeAsync(mat->hard_cutoff_device_accepted, stream);
    if (mat->hard_cutoff_bound_words) cudaFreeAsync(mat->hard_cutoff_bound_words, stream);
    if (mat->hard_cutoff_half_modulus_words)
        cudaFreeAsync(mat->hard_cutoff_half_modulus_words, stream);
    if (mat->hard_cutoff_modulus_words)
        cudaFreeAsync(mat->hard_cutoff_modulus_words, stream);
    if (mat->hard_cutoff_subset_indices)
        cudaFreeAsync(mat->hard_cutoff_subset_indices, stream);
    if (mat->hard_cutoff_garner_inverses)
        cudaFreeAsync(mat->hard_cutoff_garner_inverses, stream);
    if (mat->hard_cutoff_decision_ready)
        cudaEventDestroy(mat->hard_cutoff_decision_ready);
    if (mat->hard_cutoff_host_accepted)
        cudaFreeHost(mat->hard_cutoff_host_accepted);
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
    err = cudaMallocAsync(
        reinterpret_cast<void **>(&mat->hard_cutoff_device_accepted),
        sizeof(int), mat->stream);
    if (err == cudaSuccess)
        err = cudaHostAlloc(
            reinterpret_cast<void **>(&mat->hard_cutoff_host_accepted),
            sizeof(int), cudaHostAllocPortable);
    if (err == cudaSuccess)
        err = cudaEventCreateWithFlags(
            &mat->hard_cutoff_decision_ready, cudaEventDisableTiming);
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
    if (ctx->compute_streams_by_partition.empty() || ctx->compute_streams_by_partition.front().empty())
    {
        delete mat;
        return set_error("missing compact matrix stream");
    }
    mat->stream = ctx->compute_streams_by_partition.front().front();
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

extern "C" void gpu_small_matrix_destroy(GpuSmallMatrix *mat)
{
    if (!mat) return;
    if (mat->device >= 0 && cudaSetDevice(mat->device) == cudaSuccess)
    {
        cudaStream_t release_stream = mat->stream;
        const size_t partition = 0;
        if (mat->ctx && partition < mat->ctx->release_streams_by_partition.size() &&
            mat->ctx->release_streams_by_partition[partition])
        {
            release_stream = mat->ctx->release_streams_by_partition[partition];
            if (mat->write_done_valid) cudaStreamWaitEvent(release_stream, mat->write_done, 0);
        }
        if (mat->owns_payload)
        {
            small_release_hard_cutoff_plan(mat, release_stream);
            if (mat->payload && release_stream) cudaFreeAsync(mat->payload, release_stream);
        }
        if (mat->owns_write_event && mat->write_done) cudaEventDestroy(mat->write_done);
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

extern "C" int gpu_small_matrix_copy(GpuSmallMatrix *out, const GpuSmallMatrix *src)
{
    if (!out || !src || out->ctx != src->ctx || out->rows != src->rows || out->cols != src->cols ||
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
    return small_track_consumer(src, out->stream);
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
    return small_track_consumer(src, out->stream);
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
    uint8_t *staging = nullptr;
    cudaError_t err = cudaHostAlloc(
        reinterpret_cast<void **>(&staging), payload_len, cudaHostAllocPortable);
    if (err != cudaSuccess) return set_error(err);
    std::memcpy(staging, payload, payload_len);
    err = cudaMemcpyAsync(
        mat->payload, staging, payload_len, cudaMemcpyHostToDevice, mat->stream);
    if (err != cudaSuccess)
    {
        cudaFreeHost(staging);
        return set_error(err);
    }
    const int record_status = small_record(mat, mat->stream);
    void *deferred[] = {staging};
    const int defer_status = gpu_defer_pinned_frees(
        mat->ctx, mat->device, mat->stream, deferred, 1);
    if (record_status != 0)
    {
        (void)small_fence_stream_with_event(mat->stream);
        return record_status;
    }
    return defer_status;
}

extern "C" int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat, uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    if (payload_len == 0) return 0;
    const size_t coefficient_bytes = 1 + mat->magnitude_bytes;
    const size_t row_bytes = mat->cols * mat->n * coefficient_bytes;
    const size_t source_pitch = mat->storage_cols * mat->n * coefficient_bytes;
    const auto *source = mat->payload + mat->column_offset * mat->n * coefficient_bytes;
    cudaError_t err = cudaMemcpy2DAsync(
        payload, row_bytes, source, source_pitch, row_bytes, mat->rows,
        cudaMemcpyDeviceToHost, mat->stream);
    if (err == cudaSuccess) err = small_fence_stream_with_event(mat->stream);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_decompose_base(
    const GpuMatrix *src,
    uint32_t base_bits,
    int small_mode,
    const uint64_t *max_coefficient_bound,
    size_t bound_word_count,
    GpuSmallMatrix *out)
{
    if (!src || !out || !src->ctx || src->ctx != out->ctx || !max_coefficient_bound ||
        bound_word_count == 0 || base_bits == 0 || base_bits >= 63 ||
        (small_mode != 0 && small_mode != 1) || src->format != GPU_POLY_FORMAT_COEFF)
        return set_error("invalid compact decomposition arguments");
    const size_t limbs = static_cast<size_t>(src->level + 1);
    if (src->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || src->ctx->limb_gpu_ids.size() < limbs)
        return set_error("invalid compact decomposition level");
    uint32_t crt_bits = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
        crt_bits = std::max(crt_bits, bit_width_u64(src->ctx->moduli[limb]));
    const size_t digits = (crt_bits + base_bits - 1) / base_bits;
    const bool small = small_mode != 0;
    size_t expected_rows = 0;
    if (!small_mul_size(src->rows, digits, &expected_rows) ||
        (!small && !small_mul_size(expected_rows, limbs, &expected_rows)))
        return set_error("compact decomposition shape overflow");
    const uint64_t base = uint64_t{1} << base_bits;
    const uint64_t expected_bound = small ? base - 1 : (base + 1) / 2;
    if (out->rows != expected_rows || out->cols != src->cols || out->bound_words.size() != 1 ||
        out->bound_words[0] != expected_bound)
        return set_error("compact decomposition shape or bound mismatch");
    if (small_set_device(out) != 0) return 1;
    std::vector<uint64_t> requested_bound(max_coefficient_bound,
                                          max_coefficient_bound + bound_word_count);
    if (requested_bound != out->bound_words)
        return set_error("compact decomposition bound metadata mismatch");
    cudaStream_t stream = out->stream;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        if (matrix_limb_device(src, id, &limb_device) != 0 || limb_device != out->device ||
            id.x >= src->shared_limb_buffers.size() ||
            !src->shared_limb_buffers[id.x].device_descriptors ||
            id.y >= src->shared_limb_buffers[id.x].limb_count)
            return set_error("compact decomposition requires one device");
        if (limb == 0) dispatch_slot = static_cast<size_t>(id.x);
        else if (id.x != dispatch_slot)
            return set_error("compact decomposition requires one device");
        if (matrix_wait_limb_stream(src, id, out->device, stream) != 0) return 1;
    }
    if (dispatch_slot >= src->ctx->ntt_device_constants.size())
        return set_error("missing compact decomposition constants");
    const auto &constants = src->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != out->device || constants.limb_count < limbs || !constants.moduli)
        return set_error("invalid compact decomposition constants");
    size_t poly_count = 0;
    if (!small_mul_size(src->rows, src->cols, &poly_count))
        return set_error("compact decomposition polynomial count overflow");
    const size_t slots = digits * (small ? 1 : limbs);
    const dim3 grid((out->n + kSmallThreads - 1) / kSmallThreads,
                    static_cast<uint32_t>(poly_count), static_cast<uint32_t>(slots));
    compact_decompose_kernel<<<grid, kSmallThreads, 0, stream>>>(
        src->shared_limb_buffers[dispatch_slot].device_descriptors, constants.moduli, out->payload,
        src->rows, src->cols, out->rows, out->n, digits, out->magnitude_bytes, base_bits, !small, small);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            if (matrix_track_limb_consumer_readonly(src, src->ctx->limb_gpu_ids[limb], out->device, stream) != 0)
            {
                err = cudaErrorInvalidResourceHandle;
                break;
            }
        }
    }
    if (err != cudaSuccess) return set_error(err);
    return small_record(out, stream);
}

extern "C" int gpu_small_matrix_prepare_preimage_hard_cutoff(GpuSmallMatrix *mat)
{
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
        if (id.y != limb || matrix_wait_limb_stream(src, id, dst->device, dst->stream) != 0)
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

    uint8_t *d_staging = nullptr;
    cudaError_t err = cudaSuccess;
    size_t staging_bytes = 0;
    if (!small_mul_size(total_coefficients, 1 + dst->magnitude_bytes, &staging_bytes))
        return set_error("compact tile staging size overflow");
    err = cudaMallocAsync(reinterpret_cast<void **>(&d_staging), staging_bytes, dst->stream);
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
        if (d_staging) cudaFreeAsync(d_staging, dst->stream);
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
    if (d_staging) cudaFreeAsync(d_staging, dst->stream);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_matrix_mul_small_rhs(
    GpuMatrix *out,
    const GpuMatrix *lhs_eval,
    const GpuSmallMatrix *rhs_small,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report)
{
    if (!out || !lhs_eval || !rhs_small || !out->ctx || out->ctx != lhs_eval->ctx || out->ctx != rhs_small->ctx ||
        lhs_eval->format != GPU_POLY_FORMAT_EVAL || out->format != GPU_POLY_FORMAT_EVAL ||
        lhs_eval->cols != rhs_small->rows || out->rows != lhs_eval->rows || out->cols != rhs_small->cols ||
        !allocation_report)
        return set_error("invalid compact RHS multiplication arguments");
    const size_t limbs = static_cast<size_t>(lhs_eval->level + 1);
    if (lhs_eval->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || out->level != lhs_eval->level ||
        out->ctx->limb_gpu_ids.size() < limbs || !is_power_of_two_u32(static_cast<uint32_t>(out->ctx->N)))
        return set_error("invalid compact RHS multiplication level");
    if (small_set_device(rhs_small) != 0) return 1;
    cudaStream_t stream = nullptr;
    int dispatch_device = -1;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
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
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_slot = static_cast<size_t>(id.x);
            if (matrix_limb_stream(out, id, &stream) != 0) return 1;
        }
        else if (out_device != dispatch_device)
            return set_error("compact RHS multiplication requires one device");
        if (matrix_wait_limb_stream(lhs_eval, id, rhs_small->device, stream) != 0 ||
            matrix_wait_limb_stream(out, id, rhs_small->device, stream) != 0)
            return 1;
    }
    if (!stream) return set_error("missing compact multiplication stream");
    if (small_wait(rhs_small, stream) != 0) return 1;
    const size_t n = rhs_small->n;
    const size_t rows = lhs_eval->rows;
    const size_t inner = lhs_eval->cols;
    const size_t cols = rhs_small->cols;
    size_t workspace_product = 0;
    if (!small_mul_size(limbs, inner, &workspace_product) ||
        !small_mul_size(workspace_product, cols, &workspace_product))
        return set_error("compact RHS workspace shape overflow");
    uint32_t log_n = 0;
    for (size_t x = n; x > 1; x >>= 1) ++log_n;
    if (dispatch_slot >= out->ctx->ntt_device_constants.size())
        return set_error("missing compact multiplication NTT partition");
    const auto &constants = out->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != dispatch_device || constants.ring_dimension != n ||
        constants.limb_count < limbs || !constants.twiddle_forward ||
        !constants.twiddle_shoup_forward || !constants.moduli)
        return set_error("missing compact multiplication NTT constants");
    const auto *lhs_descriptors = lhs_eval->shared_limb_buffers[dispatch_slot].device_descriptors;
    const auto *out_descriptors = out->shared_limb_buffers[dispatch_slot].device_descriptors;
    if (!lhs_descriptors || !out_descriptors)
        return set_error("missing compact multiplication matrix descriptors");

    uint64_t *workspace = nullptr;
    auto release = [&]() -> cudaError_t {
        cudaError_t cleanup_err = cudaSuccess;
        auto free_async = [&](void *ptr) {
            if (!ptr) return;
            const cudaError_t free_err = cudaFreeAsync(ptr, stream);
            if (cleanup_err == cudaSuccess && free_err != cudaSuccess) cleanup_err = free_err;
        };
        free_async(workspace);
        workspace = nullptr;
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
    GpuMatrixAllocationBytes lhs_allocation{};
    GpuMatrixAllocationBytes output_allocation{};
    if (gpu_matrix_query_allocation_bytes(
            lhs_eval->ctx, lhs_eval->level, lhs_eval->rows, lhs_eval->cols,
            lhs_eval->format, &lhs_allocation) != 0 ||
        gpu_matrix_query_allocation_bytes(
            out->ctx, out->level, out->rows, out->cols,
            out->format, &output_allocation) != 0)
        return 1;
    const size_t lhs_eval_bytes = lhs_allocation.total_bytes;
    const size_t full_output_bytes = output_allocation.total_bytes;
    size_t workspace_words = 0;
    size_t expanded_rhs_workspace_bytes = 0;
    if (!small_mul_size(workspace_product, n, &workspace_words) ||
        !small_mul_size(workspace_words, sizeof(uint64_t), &expanded_rhs_workspace_bytes))
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
    if (high_water_bytes > residency_budget_bytes)
        return 2;
    cudaError_t err = cudaMallocAsync(
        reinterpret_cast<void **>(&workspace),
        workspace_words * sizeof(uint64_t),
        stream);
    if (err != cudaSuccess) return fail(err);

    const size_t poly_count = inner * cols;
    const size_t current_workspace_words = limbs * poly_count * n;
    const dim3 unpack_grid((current_workspace_words + kSmallThreads - 1) / kSmallThreads);
    compact_unpack_twist_kernel<<<unpack_grid, kSmallThreads, 0, stream>>>(
        rhs_small->payload, workspace, constants.twiddle_forward,
        constants.twiddle_shoup_forward, constants.moduli, 0, limbs,
        inner, cols, rhs_small->storage_cols, n, rhs_small->magnitude_bytes,
        rhs_small->column_offset);
    err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        const dim3 grid((current_workspace_words + kSmallThreads - 1) / kSmallThreads);
        compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, stream>>>(
            workspace, limbs, poly_count, n, log_n);
        err = cudaGetLastError();
    }
    for (uint32_t len = 2; err == cudaSuccess && len <= n; len <<= 1)
    {
        const size_t butterflies = limbs * poly_count * (n / 2);
        const dim3 grid((butterflies + kSmallThreads - 1) / kSmallThreads);
        compact_ntt_stage_kernel<<<grid, kSmallThreads, 0, stream>>>(
            workspace, constants.twiddle_forward, constants.twiddle_shoup_forward,
            constants.moduli, 0, limbs, poly_count, n, len);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess)
    {
        const dim3 grid((current_workspace_words + kSmallThreads - 1) / kSmallThreads);
        compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, stream>>>(
            workspace, limbs, poly_count, n, log_n);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess)
    {
        const bool lazy_reduce = compact_lazy_dot_is_safe(
            rhs_small->ctx->moduli, 0, limbs, inner);
        const size_t output_words = limbs * rows * cols * n;
        const dim3 grid((output_words + kSmallThreads - 1) / kSmallThreads);
        compact_accumulate_kernel<<<grid, kSmallThreads, 0, stream>>>(
            lhs_descriptors, out_descriptors, constants.moduli, 0, workspace, limbs,
            rows, inner, cols, n, lazy_reduce);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return fail(err);
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        if (matrix_track_limb_consumer_readonly(lhs_eval, out->ctx->limb_gpu_ids[limb], rhs_small->device, stream) != 0 ||
            matrix_record_limb_write(out, out->ctx->limb_gpu_ids[limb], stream) != 0)
            return fail(cudaErrorInvalidResourceHandle);
    }
    if (small_track_consumer(rhs_small, stream) != 0) return fail(cudaErrorInvalidResourceHandle);
    const cudaError_t cleanup_err = release();
    if (cleanup_err != cudaSuccess) return set_error(cleanup_err);
    return 0;
}
