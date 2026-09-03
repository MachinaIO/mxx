#include "matrix/MatrixSmallRhs.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

/*
 * Compact RHS storage deliberately has no relationship to GpuMatrix.  The
 * only device allocation owned by this object is the canonical sign and
 * magnitude byte stream.  The host-side bound is metadata supplied by the
 * already validated Rust schema; it is never inferred from the stream.
 */
struct GpuSmallMatrix
{
    struct DevicePayload
    {
        size_t partition = 0;
        int device = -1;
        cudaStream_t stream = nullptr;
        uint8_t *payload = nullptr;
        cudaEvent_t ready = nullptr;
        bool ready_valid = false;
    };

    GpuContext *ctx = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t n = 0;
    size_t magnitude_bytes = 0;
    size_t payload_bytes = 0;
    int device = -1;
    cudaStream_t stream = nullptr;
    uint8_t *payload = nullptr;
    std::vector<uint64_t> bound_words;
    cudaEvent_t write_done = nullptr;
    bool write_done_valid = false;
    std::vector<DevicePayload> device_payloads;
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

int small_track_partition_consumer(
    const GpuSmallMatrix *mat, size_t partition, int consumer_device, cudaStream_t consumer_stream)
{
    if (!mat || !mat->ctx || !consumer_stream || mat->device < 0 || consumer_device < 0)
        return set_error("invalid compact matrix consumer arguments");
    int owner_device = mat->device;
    cudaStream_t release_stream = mat->stream;
    if (partition != 0)
    {
        bool found = false;
        for (const auto &entry : mat->device_payloads)
        {
            if (entry.partition == partition)
            {
                owner_device = entry.device;
                found = true;
                break;
            }
        }
        if (!found) return set_error("missing compact matrix partition payload");
    }
    if (partition < mat->ctx->release_streams_by_partition.size() &&
        mat->ctx->release_streams_by_partition[partition])
        release_stream = mat->ctx->release_streams_by_partition[partition];
    if (!release_stream) return set_error("missing compact matrix release stream");

    cudaError_t err = cudaSetDevice(consumer_device);
    if (err != cudaSuccess) return set_error(err);
    cudaEvent_t consumer_done = nullptr;
    err = cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming);
    if (err == cudaSuccess) err = cudaEventRecord(consumer_done, consumer_stream);
    if (err == cudaSuccess) err = cudaSetDevice(owner_device);
    if (err == cudaSuccess) err = cudaStreamWaitEvent(release_stream, consumer_done, 0);
    const cudaError_t destroy_err = consumer_done ? cudaEventDestroy(consumer_done) : cudaSuccess;
    if (err == cudaSuccess) err = destroy_err;
    if (err != cudaSuccess)
    {
        // The owner can be dropped immediately after this error.  Fence the
        // consumer before returning, while leaving the producer event intact.
        cudaSetDevice(consumer_device);
        cudaStreamSynchronize(consumer_stream);
        return set_error(err);
    }
    return 0;
}

int small_track_consumer(const GpuSmallMatrix *mat, int consumer_device, cudaStream_t consumer_stream)
{
    return small_track_partition_consumer(mat, 0, consumer_device, consumer_stream);
}

GpuSmallMatrix::DevicePayload *small_payload_for_partition(
    GpuSmallMatrix *mat,
    size_t partition,
    int device,
    cudaStream_t stream)
{
    if (!mat || !mat->ctx || !stream || device < 0) return nullptr;
    if (partition == 0 && device == mat->device)
    {
        return nullptr;
    }
    for (auto &entry : mat->device_payloads)
    {
        if (entry.partition == partition)
        {
            if (entry.device != device) return nullptr;
            return &entry;
        }
    }
    if (cudaSetDevice(device) != cudaSuccess) return nullptr;
    if (device != mat->device)
    {
        int can_access = 0;
        if (cudaDeviceCanAccessPeer(&can_access, device, mat->device) != cudaSuccess || !can_access)
            return nullptr;
        cudaError_t peer_error = cudaDeviceEnablePeerAccess(mat->device, 0);
        if (peer_error == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else if (peer_error != cudaSuccess) return nullptr;
    }
    GpuSmallMatrix::DevicePayload entry;
    entry.partition = partition;
    entry.device = device;
    entry.stream = stream;
    cudaError_t error = cudaEventCreateWithFlags(&entry.ready, cudaEventDisableTiming);
    if (error == cudaSuccess)
        error = cudaMallocAsync(reinterpret_cast<void **>(&entry.payload), mat->payload_bytes, stream);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(stream, mat->write_done, 0);
    if (error == cudaSuccess)
        error = cudaMemcpyPeerAsync(
            entry.payload, device, mat->payload, mat->device, mat->payload_bytes, stream);
    if (error == cudaSuccess) error = cudaEventRecord(entry.ready, stream);
    if (error != cudaSuccess)
    {
        if (entry.payload) cudaFreeAsync(entry.payload, stream);
        if (entry.ready) cudaEventDestroy(entry.ready);
        return nullptr;
    }
    entry.ready_valid = true;
    mat->device_payloads.push_back(entry);
    return &mat->device_payloads.back();
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

__global__ void compact_unpack_kernel(
    const uint8_t *payload,
    uint64_t *workspace,
    const uint64_t *moduli,
    const uint32_t *global_limb_ids,
    size_t limb_offset,
    size_t limb_count,
    size_t k_tile,
    size_t c_tile,
    size_t rhs_cols,
    size_t n,
    size_t magnitude_bytes,
    size_t k_offset,
    size_t c_offset)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = limb_count * k_tile * c_tile * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t q = idx / n;
    const size_t c = q % c_tile;
    const size_t k = (q / c_tile) % k_tile;
    const size_t local_limb = q / (k_tile * c_tile);
    const size_t rhs_row = k_offset + k;
    const size_t rhs_col = c_offset + c;
    const size_t width = 1 + magnitude_bytes;
    const uint8_t *src = payload + ((rhs_row * rhs_cols + rhs_col) * n + coeff) * width;
    const size_t global_limb = global_limb_ids
        ? static_cast<size_t>(global_limb_ids[local_limb])
        : limb_offset + local_limb;
    const uint64_t modulus = moduli[global_limb];
    uint64_t value = compact_mod_magnitude(src + 1, magnitude_bytes, modulus);
    if (src[0] == 2 && value != 0) value = modulus - value;
    workspace[idx] = value;
}

__global__ void compact_ntt_twist_kernel(
    uint64_t *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    const uint32_t *global_limb_ids,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t n)
{
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t poly = static_cast<size_t>(blockIdx.y);
    const size_t local = static_cast<size_t>(blockIdx.z);
    if (coeff >= n || poly >= poly_count || local >= limb_count) return;
    const size_t global = global_limb_ids
        ? static_cast<size_t>(global_limb_ids[local])
        : limb_offset + local;
    const uint64_t modulus = moduli[global];
    const size_t index = (local * poly_count + poly) * n + coeff;
    const size_t twiddle_index = global * n + coeff;
    workspace[index] = mul_mod_shoup_u64(
        workspace[index], twiddles[twiddle_index], twiddle_shoup[twiddle_index], modulus);
}

__global__ void compact_ntt_bit_reverse_kernel(
    uint64_t *workspace,
    size_t limb_count,
    size_t poly_count,
    size_t n,
    uint32_t log_n)
{
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t poly = static_cast<size_t>(blockIdx.y);
    const size_t local = static_cast<size_t>(blockIdx.z);
    if (coeff >= n || poly >= poly_count || local >= limb_count) return;
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
    const uint32_t *global_limb_ids,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t n,
    uint32_t len)
{
    const size_t butterfly = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t local = static_cast<size_t>(blockIdx.z);
    if (butterfly >= n / 2 || local >= limb_count || blockIdx.y >= poly_count) return;
    const uint32_t half = len / 2;
    const uint32_t group = static_cast<uint32_t>(butterfly) / half;
    const uint32_t j = static_cast<uint32_t>(butterfly) % half;
    const uint32_t i = group * len + j;
    const size_t global = global_limb_ids
        ? static_cast<size_t>(global_limb_ids[local])
        : limb_offset + local;
    const uint64_t modulus = moduli[global];
    const size_t base = (local * poly_count + static_cast<size_t>(blockIdx.y)) * n;
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
    const uint32_t *global_limb_ids,
    size_t limb_offset,
    const uint64_t *workspace,
    size_t limb_count,
    size_t rows,
    size_t inner,
    size_t out_cols,
    size_t k_tile,
    size_t c_tile,
    size_t n,
    size_t k_offset,
    size_t c_offset)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = limb_count * rows * c_tile * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t q = idx / n;
    const size_t c = q % c_tile;
    const size_t row = (q / c_tile) % rows;
    const size_t local = q / (rows * c_tile);
    const size_t global = global_limb_ids
        ? static_cast<size_t>(global_limb_ids[local])
        : limb_offset + local;
    const uint64_t modulus = moduli[global];
    const auto lhs_descriptor = lhs_descriptors[local];
    const auto out_descriptor = out_descriptors[local];
    uint64_t acc = 0;
    if (k_offset != 0)
    {
        acc = matrix_load_limb_u64(
            out_descriptor.base,
            row * out_cols + c_offset + c,
            coeff,
            out_descriptor.stride,
            out_descriptor.width);
    }
    for (size_t k = 0; k < k_tile && k_offset + k < inner; ++k)
    {
        const uint64_t lhs = matrix_load_limb_u64(
            lhs_descriptor.base,
            row * inner + k_offset + k,
            coeff,
            lhs_descriptor.stride,
            lhs_descriptor.width);
        const size_t rhs_index = (local * (k_tile * c_tile) + k * c_tile + c) * n + coeff;
        acc = add_mod_u64(acc, mul_mod_u64(lhs, workspace[rhs_index], modulus), modulus);
    }
    matrix_store_limb_u64(
        out_descriptor.base,
        row * out_cols + c_offset + c,
        coeff,
        out_descriptor.stride,
        out_descriptor.width,
        acc);
}

// The bound-check kernel intentionally does not own compact serialization.
// This companion kernel writes the canonical sign/magnitude payload after the
// same CRT reconstruction, so rejected candidates still leave a complete
// (but private) payload that the next attempt overwrites.
__global__ void compact_pack_payload_kernel(
    const uint8_t *const *matrix_limb_ptrs,
    const size_t *limb_strides,
    const uint8_t *limb_coeff_bytes,
    const uint64_t *moduli,
    const uint64_t *garner_inverses,
    int inverse_stride,
    int limb_count,
    size_t rows,
    size_t cols,
    size_t n,
    int words_per_coeff,
    const uint64_t *modulus_words,
    const uint64_t *half_modulus_words,
    uint8_t *payload,
    size_t payload_cols,
    size_t magnitude_bytes,
    size_t dst_row,
    size_t dst_col)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = rows * cols * n;
    if (idx >= total) return;
    const size_t coeff = idx % n;
    const size_t poly = idx / n;
    const size_t row = poly / cols;
    const size_t col = poly % cols;
    uint64_t mixed_digits[kMaxRnsLimbs] = {};
    uint64_t coeff_words[kMaxCoeffWords] = {};
    for (int i = 0; i < limb_count; ++i)
    {
        mixed_digits[i] = matrix_load_limb_u64(
            matrix_limb_ptrs[i], poly, coeff, limb_strides[i], limb_coeff_bytes[i]) % moduli[i];
    }
    for (int i = 1; i < limb_count; ++i)
    {
        const uint64_t qi = moduli[i];
        uint64_t t = mixed_digits[i];
        for (int j = 0; j < i; ++j)
        {
            const uint64_t xj = mixed_digits[j] % qi;
            const uint64_t diff = t >= xj ? t - xj : t + qi - xj;
            t = serde_mul_mod_u64_device(
                diff, garner_inverses[static_cast<size_t>(j) * inverse_stride + i], qi);
        }
        mixed_digits[i] = t;
    }
    for (int i = limb_count - 1; i >= 0; --i)
    {
        uint64_t carry = mixed_digits[i];
        for (int w = 0; w < words_per_coeff; ++w)
        {
            const unsigned __int128 term =
                static_cast<unsigned __int128>(coeff_words[w]) * moduli[i] + carry;
            coeff_words[w] = static_cast<uint64_t>(term);
            carry = static_cast<uint64_t>(term >> 64);
        }
    }
    bool negative = serde_compare_words_desc_device(
        coeff_words, half_modulus_words, words_per_coeff) > 0;
    if (negative)
    {
        uint64_t borrow = 0;
        for (int w = 0; w < words_per_coeff; ++w)
        {
            const unsigned __int128 sub =
                static_cast<unsigned __int128>(coeff_words[w]) + borrow;
            const unsigned __int128 minuend = modulus_words[w];
            coeff_words[w] = static_cast<uint64_t>(minuend - sub);
            borrow = minuend < sub ? 1 : 0;
        }
    }
    const size_t width = 1 + magnitude_bytes;
    uint8_t *out = payload + (((dst_row + row) * payload_cols + dst_col + col) * n + coeff) * width;
    out[0] = negative ? 2 : 1;
    bool zero = true;
    for (int w = 0; w < words_per_coeff; ++w) zero = zero && coeff_words[w] == 0;
    if (zero) out[0] = 0;
    for (size_t byte = 0; byte < magnitude_bytes; ++byte)
    {
        const size_t word = byte / sizeof(uint64_t);
        const size_t shift = (byte % sizeof(uint64_t)) * 8;
        out[1 + byte] = word < static_cast<size_t>(words_per_coeff)
            ? static_cast<uint8_t>(coeff_words[word] >> shift)
            : 0;
    }
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
    mat->n = static_cast<size_t>(ctx->N);
    mat->magnitude_bytes = magnitude_bytes;
    mat->bound_words.assign(bound_words, bound_words + bound_word_count);
    if (small_payload_size(rows, cols, mat->n, magnitude_bytes, &mat->payload_bytes) != 0)
    {
        delete mat;
        return 1;
    }
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
    for (auto &entry : mat->device_payloads)
    {
        if (entry.device < 0 || cudaSetDevice(entry.device) != cudaSuccess) continue;
        cudaStream_t release_stream = entry.stream;
        if (entry.partition < mat->ctx->release_streams_by_partition.size() &&
            mat->ctx->release_streams_by_partition[entry.partition])
        {
            release_stream = mat->ctx->release_streams_by_partition[entry.partition];
            if (entry.ready_valid) cudaStreamWaitEvent(release_stream, entry.ready, 0);
        }
        if (entry.payload && release_stream) cudaFreeAsync(entry.payload, release_stream);
        if (entry.ready) cudaEventDestroy(entry.ready);
    }
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
        if (mat->payload && release_stream) cudaFreeAsync(mat->payload, release_stream);
        if (mat->write_done) cudaEventDestroy(mat->write_done);
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
    const cudaError_t err = cudaMemcpyAsync(out->payload, src->payload, out->payload_bytes,
                                            cudaMemcpyDeviceToDevice, out->stream);
    if (err != cudaSuccess) return set_error(err);
    if (small_record(out, out->stream) != 0) return 1;
    return small_track_consumer(src, out->device, out->stream);
}

extern "C" int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat, const uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0) return 1;
    const cudaError_t err = cudaMemcpyAsync(mat->payload, payload, payload_len,
                                            cudaMemcpyHostToDevice, mat->stream);
    if (err != cudaSuccess) return set_error(err);
    return small_record(mat, mat->stream);
}

extern "C" int gpu_small_matrix_store_coefficients(
    const GpuSmallMatrix *mat, uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0 || small_wait(mat, mat->stream) != 0) return 1;
    cudaError_t err = cudaMemcpyAsync(payload, mat->payload, payload_len,
                                      cudaMemcpyDeviceToHost, mat->stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(mat->stream);
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
    if (!stream) return set_error("missing compact decomposition stream");
    std::vector<uint8_t *> remote_bases(src->ctx->gpu_ids.size(), nullptr);
    auto release_remote = [&]() {
        for (uint8_t *base : remote_bases)
            if (base) cudaFreeAsync(base, stream);
        std::fill(remote_bases.begin(), remote_bases.end(), nullptr);
    };
    std::vector<GpuMatrix::SharedLimbBuffer::DeviceDescriptor> descriptors(limbs);
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        size_t stride = 0;
        uint8_t width = 0;
        if (matrix_limb_device(src, id, &limb_device) != 0 ||
            id.x >= src->shared_limb_buffers.size() ||
            !src->shared_limb_buffers[id.x].device_descriptors ||
            id.y >= src->shared_limb_buffers[id.x].limb_count ||
            !matrix_limb_metadata_by_id(src, id, &stride, &width) ||
            matrix_wait_limb_stream(src, id, out->device, stream) != 0)
        {
            release_remote();
            return set_error("invalid compact decomposition active limb");
        }
        const uint8_t *source_base = matrix_limb_ptr_by_id(src, 0, id);
        uint8_t *base = const_cast<uint8_t *>(source_base);
        if (limb_device != out->device)
        {
            int can_access = 0;
            if (cudaSetDevice(out->device) != cudaSuccess ||
                cudaDeviceCanAccessPeer(&can_access, out->device, limb_device) != cudaSuccess ||
                !can_access)
            {
                release_remote();
                return set_error("compact decomposition requires peer access");
            }
            cudaError_t peer = cudaDeviceEnablePeerAccess(limb_device, 0);
            if (peer == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else if (peer != cudaSuccess)
            {
                release_remote();
                return set_error(peer);
            }
            const auto &buffer = src->shared_limb_buffers[id.x];
            uint8_t *remote = remote_bases[id.x];
            if (!remote)
                peer = cudaMallocAsync(reinterpret_cast<void **>(&remote), buffer.bytes_total, stream);
            if (peer == cudaSuccess && !remote_bases[id.x])
                peer = cudaMemcpyPeerAsync(remote, out->device, buffer.ptr, limb_device,
                                           buffer.bytes_total, stream);
            if (peer != cudaSuccess)
            {
                if (remote) cudaFreeAsync(remote, stream);
                release_remote();
                return set_error(peer);
            }
            remote_bases[id.x] = remote;
            base = remote + buffer.limb_offsets_bytes[id.y];
        }
        if (!base)
        {
            release_remote();
            return set_error("invalid compact decomposition limb pointer");
        }
        descriptors[limb] = {base, stride, width, 0};
    }
    const size_t dispatch_slot = 0;
    if (dispatch_slot >= src->ctx->ntt_device_constants.size())
    {
        release_remote();
        return set_error("missing compact decomposition constants");
    }
    const auto &constants = src->ctx->ntt_device_constants[dispatch_slot];
    if (constants.device != out->device || constants.limb_count < limbs || !constants.moduli)
    {
        release_remote();
        return set_error("invalid compact decomposition constants");
    }
    size_t poly_count = 0;
    if (!small_mul_size(src->rows, src->cols, &poly_count))
    {
        release_remote();
        return set_error("compact decomposition polynomial count overflow");
    }
    const size_t slots = digits * (small ? 1 : limbs);
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor *device_descriptors = nullptr;
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&device_descriptors),
                                      descriptors.size() * sizeof(descriptors[0]), stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(device_descriptors, descriptors.data(),
                              descriptors.size() * sizeof(descriptors[0]),
                              cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        if (device_descriptors) cudaFreeAsync(device_descriptors, stream);
        release_remote();
        return set_error(err);
    }
    const dim3 grid((out->n + kSmallThreads - 1) / kSmallThreads,
                    static_cast<uint32_t>(poly_count), static_cast<uint32_t>(slots));
    compact_decompose_kernel<<<grid, kSmallThreads, 0, stream>>>(
        device_descriptors, constants.moduli, out->payload,
        src->rows, src->cols, out->rows, out->n, digits, out->magnitude_bytes, base_bits, !small, small);
    err = cudaGetLastError();
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
    if (err == cudaSuccess && small_record(out, stream) != 0)
        err = cudaErrorInvalidResourceHandle;
    if (device_descriptors) cudaFreeAsync(device_descriptors, stream);
    release_remote();
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_small_matrix_pack_checked_tile(
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
        src->ctx->limb_gpu_ids.size() < limb_count)
        return set_error("invalid compact tile active CRT basis");
    size_t total_bits = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const size_t bits = static_cast<size_t>(bit_width_u64(src->ctx->moduli[limb]));
        if (!small_add_size(total_bits, bits, &total_bits))
            return set_error("compact tile coefficient width overflow");
    }
    const size_t words_per_coeff = std::max<size_t>(1, (total_bits + 63) / 64);
    if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords) ||
        bound_word_count > words_per_coeff)
        return set_error("compact tile bound width exceeds active CRT width");
    size_t poly_count = 0;
    size_t total_coefficients = 0;
    if (!small_mul_size(rows, cols, &poly_count) ||
        !small_mul_size(poly_count, dst->n, &total_coefficients))
        return set_error("compact tile coefficient count overflow");

    std::vector<const uint8_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    std::vector<uint8_t> limb_widths(limb_count);
    std::vector<uint8_t *> remote_bases(src->ctx->gpu_ids.size(), nullptr);
    auto release_remote = [&]() {
        for (uint8_t *base : remote_bases)
            if (base) cudaFreeAsync(base, dst->stream);
        std::fill(remote_bases.begin(), remote_bases.end(), nullptr);
    };
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int device = -1;
        if (matrix_limb_device(src, id, &device) != 0 ||
            !matrix_limb_metadata_by_id(src, id, &limb_strides[limb], &limb_widths[limb]) ||
            matrix_wait_limb_stream(src, id, dst->device, dst->stream) != 0)
        {
            release_remote();
            return set_error("invalid compact tile active CRT limb");
        }
        if (device == dst->device)
            limb_ptrs[limb] = matrix_limb_ptr_by_id(src, 0, id);
        else
        {
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(&can_access, dst->device, device) != cudaSuccess || !can_access)
            {
                release_remote();
                return set_error("compact tile requires peer access");
            }
            cudaError_t peer = cudaDeviceEnablePeerAccess(device, 0);
            if (peer == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else if (peer != cudaSuccess)
            {
                release_remote();
                return set_error(peer);
            }
            auto &remote = remote_bases[id.x];
            if (!remote)
            {
                const auto &buffer = src->shared_limb_buffers[id.x];
                peer = cudaMallocAsync(reinterpret_cast<void **>(&remote), buffer.bytes_total, dst->stream);
                if (peer == cudaSuccess)
                    peer = cudaMemcpyPeerAsync(remote, dst->device, buffer.ptr, device,
                                               buffer.bytes_total, dst->stream);
                if (peer != cudaSuccess)
                {
                    if (remote) cudaFreeAsync(remote, dst->stream);
                    remote = nullptr;
                    release_remote();
                    return set_error(peer);
                }
            }
            const auto &buffer = src->shared_limb_buffers[id.x];
            limb_ptrs[limb] = remote + buffer.limb_offsets_bytes[id.y];
        }
        if (!limb_ptrs[limb])
        {
            release_remote();
            return set_error("invalid compact tile active CRT limb pointer");
        }
    }
    const std::vector<uint64_t> moduli(
        src->ctx->moduli.begin(), src->ctx->moduli.begin() + limb_count);
    std::vector<uint64_t> modulus_words;
    if (!serde_compute_modulus_words_le(moduli, &modulus_words))
    {
        release_remote();
        return set_error("failed to compute compact tile CRT modulus");
    }
    std::vector<uint64_t> half_modulus_words = modulus_words;
    serde_shift_words_right_one_le(&half_modulus_words);
    modulus_words.resize(words_per_coeff, 0);
    half_modulus_words.resize(words_per_coeff, 0);
    std::vector<uint64_t> padded_bound(words_per_coeff, 0);
    std::copy(bound_words, bound_words + bound_word_count, padded_bound.begin());

    const uint8_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint8_t *d_limb_widths = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_garner_inverses = nullptr;
    uint64_t *d_modulus_words = nullptr;
    uint64_t *d_half_modulus_words = nullptr;
    uint64_t *d_bound_words = nullptr;
    int *d_accepted = nullptr;
    int *h_accepted = nullptr;
    cudaEvent_t decision_ready = nullptr;
    auto release = [&]() {
        if (d_accepted) cudaFreeAsync(d_accepted, dst->stream);
        if (d_bound_words) cudaFreeAsync(d_bound_words, dst->stream);
        if (d_half_modulus_words) cudaFreeAsync(d_half_modulus_words, dst->stream);
        if (d_modulus_words) cudaFreeAsync(d_modulus_words, dst->stream);
        if (d_garner_inverses) cudaFreeAsync(d_garner_inverses, dst->stream);
        if (d_moduli) cudaFreeAsync(d_moduli, dst->stream);
        if (d_limb_widths) cudaFreeAsync(d_limb_widths, dst->stream);
        if (d_limb_strides) cudaFreeAsync(d_limb_strides, dst->stream);
        if (d_limb_ptrs) cudaFreeAsync(d_limb_ptrs, dst->stream);
        release_remote();
    };
    cudaError_t err = cudaSuccess;
#define MXX_COMPACT_PACK_ALLOC(dst_ptr, bytes)                                         \
    do                                                                                 \
    {                                                                                  \
        err = cudaMallocAsync(reinterpret_cast<void **>(&(dst_ptr)), (bytes), dst->stream); \
        if (err != cudaSuccess)                                                        \
        {                                                                              \
            release();                                                                 \
            return set_error(err);                                                     \
        }                                                                              \
    } while (false)
    MXX_COMPACT_PACK_ALLOC(d_limb_ptrs, limb_ptrs.size() * sizeof(uint8_t *));
    MXX_COMPACT_PACK_ALLOC(d_limb_strides, limb_strides.size() * sizeof(size_t));
    MXX_COMPACT_PACK_ALLOC(d_limb_widths, limb_widths.size() * sizeof(uint8_t));
    MXX_COMPACT_PACK_ALLOC(d_moduli, moduli.size() * sizeof(uint64_t));
    MXX_COMPACT_PACK_ALLOC(
        d_garner_inverses,
        src->ctx->garner_inverse_table.size() * sizeof(uint64_t));
    MXX_COMPACT_PACK_ALLOC(d_modulus_words, words_per_coeff * sizeof(uint64_t));
    MXX_COMPACT_PACK_ALLOC(d_half_modulus_words, words_per_coeff * sizeof(uint64_t));
    MXX_COMPACT_PACK_ALLOC(d_bound_words, words_per_coeff * sizeof(uint64_t));
    MXX_COMPACT_PACK_ALLOC(d_accepted, sizeof(int));
#undef MXX_COMPACT_PACK_ALLOC
    err = cudaHostAlloc(
        reinterpret_cast<void **>(&h_accepted), sizeof(int), cudaHostAllocPortable);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaEventCreateWithFlags(&decision_ready, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        cudaFreeHost(h_accepted);
        release();
        return set_error(err);
    }
    auto copy_to_device = [&](void *out, const void *in, size_t bytes) {
        return cudaMemcpyAsync(out, in, bytes, cudaMemcpyHostToDevice, dst->stream);
    };
    err = copy_to_device(d_limb_ptrs, limb_ptrs.data(), limb_ptrs.size() * sizeof(uint8_t *));
    if (err == cudaSuccess) err = copy_to_device(d_limb_strides, limb_strides.data(), limb_strides.size() * sizeof(size_t));
    if (err == cudaSuccess) err = copy_to_device(d_limb_widths, limb_widths.data(), limb_widths.size() * sizeof(uint8_t));
    if (err == cudaSuccess) err = copy_to_device(d_moduli, moduli.data(), moduli.size() * sizeof(uint64_t));
    if (err == cudaSuccess) err = copy_to_device(d_garner_inverses, src->ctx->garner_inverse_table.data(), src->ctx->garner_inverse_table.size() * sizeof(uint64_t));
    if (err == cudaSuccess) err = copy_to_device(d_modulus_words, modulus_words.data(), words_per_coeff * sizeof(uint64_t));
    if (err == cudaSuccess) err = copy_to_device(d_half_modulus_words, half_modulus_words.data(), words_per_coeff * sizeof(uint64_t));
    if (err == cudaSuccess) err = copy_to_device(d_bound_words, padded_bound.data(), words_per_coeff * sizeof(uint64_t));
    int initial_accepted = 1;
    if (err == cudaSuccess) err = cudaMemcpyAsync(d_accepted, &initial_accepted, sizeof(int), cudaMemcpyHostToDevice, dst->stream);
    if (err == cudaSuccess)
        serde_check_centered_bound_batch_kernel<<<
            (total_coefficients + kSmallThreads - 1) / kSmallThreads,
            kSmallThreads, 0, dst->stream>>>(
                d_limb_ptrs,
                d_limb_strides,
                d_limb_widths,
                d_moduli,
                d_garner_inverses,
                static_cast<int>(src->ctx->moduli.size()),
                static_cast<int>(limb_count),
                total_coefficients,
                total_coefficients,
                dst->n,
                static_cast<int>(words_per_coeff),
                d_modulus_words,
                d_half_modulus_words,
                d_bound_words,
                d_accepted
                );
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        compact_pack_payload_kernel<<<
            (total_coefficients + kSmallThreads - 1) / kSmallThreads,
            kSmallThreads,
            0,
            dst->stream>>>(
            d_limb_ptrs,
            d_limb_strides,
            d_limb_widths,
            d_moduli,
            d_garner_inverses,
            static_cast<int>(src->ctx->moduli.size()),
            static_cast<int>(limb_count),
            rows,
            cols,
            dst->n,
            static_cast<int>(words_per_coeff),
            d_modulus_words,
            d_half_modulus_words,
            dst->payload,
            dst->cols,
            dst->magnitude_bytes,
            dst_row,
            dst_col);
        err = cudaGetLastError();
    }
    for (size_t limb = 0; limb < limb_count && err == cudaSuccess; ++limb)
    {
        if (matrix_track_limb_consumer_readonly(
                src, src->ctx->limb_gpu_ids[limb], dst->device, dst->stream) != 0)
            err = cudaErrorInvalidResourceHandle;
    }
    if (err == cudaSuccess && small_record(dst, dst->stream) != 0)
        err = cudaErrorInvalidResourceHandle;
    if (err == cudaSuccess) err = cudaMemcpyAsync(h_accepted, d_accepted, sizeof(int), cudaMemcpyDeviceToHost, dst->stream);
    release();
    if (err == cudaSuccess) err = cudaEventRecord(decision_ready, dst->stream);
    if (err == cudaSuccess) err = cudaEventSynchronize(decision_ready);
    if (err != cudaSuccess)
    {
        cudaStreamSynchronize(dst->stream);
        if (decision_ready) cudaEventDestroy(decision_ready);
        if (h_accepted) cudaFreeHost(h_accepted);
        return set_error(err);
    }
    *accepted_out = *h_accepted;
    if (decision_ready) cudaEventDestroy(decision_ready);
    if (h_accepted) cudaFreeHost(h_accepted);
    return err == cudaSuccess ? 0 : set_error(err);
}

extern "C" int gpu_matrix_mul_small_rhs(
    GpuMatrix *out, const GpuMatrix *lhs_eval, const GpuSmallMatrix *rhs_small,
    size_t ct, size_t kt, size_t ell, size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report)
{
    if (!out || !lhs_eval || !rhs_small || !out->ctx || out->ctx != lhs_eval->ctx ||
        out->ctx != rhs_small->ctx || lhs_eval->format != GPU_POLY_FORMAT_EVAL ||
        out->format != GPU_POLY_FORMAT_EVAL || lhs_eval->cols != rhs_small->rows ||
        out->rows != lhs_eval->rows || out->cols != rhs_small->cols || ct == 0 || kt == 0 ||
        ell == 0 || !allocation_report)
        return set_error("invalid compact RHS multiplication arguments");
    const size_t limbs = static_cast<size_t>(lhs_eval->level + 1);
    if (lhs_eval->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount ||
        out->level != lhs_eval->level || out->ctx->limb_gpu_ids.size() < limbs ||
        !is_power_of_two_u32(static_cast<uint32_t>(out->ctx->N)) ||
        ct > rhs_small->cols || kt > lhs_eval->cols || ell > limbs)
        return set_error("invalid compact RHS multiplication level or tile");

    struct PartitionWork
    {
        size_t partition = 0;
        int device = -1;
        cudaStream_t stream = nullptr;
        std::vector<uint32_t> global_limb_ids;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *lhs_descriptors = nullptr;
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *out_descriptors = nullptr;
        const uint8_t *rhs_payload = nullptr;
        uint32_t *device_global_limb_ids = nullptr;
        uint64_t *workspace = nullptr;
    };
    std::vector<PartitionWork> works;
    works.reserve(out->ctx->gpu_ids.size());
    const size_t n = rhs_small->n;
    const size_t rows = lhs_eval->rows;
    const size_t inner = lhs_eval->cols;
    const size_t cols = rhs_small->cols;
    for (size_t partition = 0; partition < out->ctx->gpu_ids.size(); ++partition)
    {
        PartitionWork work;
        work.partition = partition;
        for (size_t limb = 0; limb < limbs; ++limb)
        {
            const dim3 id = out->ctx->limb_gpu_ids[limb];
            if (id.x == partition) work.global_limb_ids.push_back(static_cast<uint32_t>(limb));
        }
        if (work.global_limb_ids.empty()) continue;
        const dim3 first_id = out->ctx->limb_gpu_ids[work.global_limb_ids.front()];
        if (matrix_limb_device(lhs_eval, first_id, &work.device) != 0 ||
            matrix_limb_stream(out, first_id, &work.stream) != 0 || !work.stream ||
            work.device != out->ctx->gpu_ids[partition] ||
            partition >= lhs_eval->shared_limb_buffers.size() ||
            partition >= out->shared_limb_buffers.size() ||
            !lhs_eval->shared_limb_buffers[partition].device_descriptors ||
            !out->shared_limb_buffers[partition].device_descriptors)
            return set_error("invalid compact multiplication partition owner");
        work.lhs_descriptors = lhs_eval->shared_limb_buffers[partition].device_descriptors;
        work.out_descriptors = out->shared_limb_buffers[partition].device_descriptors;
        for (uint32_t global_limb : work.global_limb_ids)
        {
            const dim3 id = out->ctx->limb_gpu_ids[global_limb];
            int lhs_device = -1;
            int out_device = -1;
            size_t lhs_stride = 0;
            size_t out_stride = 0;
            uint8_t lhs_width = 0;
            uint8_t out_width = 0;
            if (matrix_limb_device(lhs_eval, id, &lhs_device) != 0 ||
                matrix_limb_device(out, id, &out_device) != 0 || lhs_device != work.device ||
                out_device != work.device || id.y >= lhs_eval->shared_limb_buffers[partition].limb_count ||
                id.y >= out->shared_limb_buffers[partition].limb_count ||
                !matrix_limb_metadata_by_id(lhs_eval, id, &lhs_stride, &lhs_width) ||
                !matrix_limb_metadata_by_id(out, id, &out_stride, &out_width) ||
                lhs_stride != lhs_eval->shared_limb_buffers[partition].bytes_per_poly ||
                out_stride != out->shared_limb_buffers[partition].bytes_per_poly)
                return set_error("invalid compact multiplication local limb");
            if (matrix_wait_limb_stream(lhs_eval, id, work.device, work.stream) != 0 ||
                matrix_wait_limb_stream(out, id, work.device, work.stream) != 0)
                return 1;
        }
        work.rhs_payload = rhs_small->payload;
        if (partition != 0 || work.device != rhs_small->device)
        {
            auto *replica = small_payload_for_partition(
                const_cast<GpuSmallMatrix *>(rhs_small), partition, work.device, work.stream);
            if (!replica) return set_error("failed to make compact RHS partition replica");
            work.rhs_payload = replica->payload;
            if (replica->ready_valid &&
                cudaStreamWaitEvent(work.stream, replica->ready, 0) != cudaSuccess)
                return set_error("failed to order compact RHS partition replica");
        }
        else if (small_wait(rhs_small, work.stream) != 0)
            return 1;
        if (partition >= out->ctx->ntt_device_constants.size())
            return set_error("missing compact multiplication NTT partition");
        const auto &constants = out->ctx->ntt_device_constants[partition];
        if (constants.device != work.device || constants.ring_dimension != n ||
            constants.limb_count < limbs || !constants.twiddle_forward ||
            !constants.twiddle_shoup_forward || !constants.moduli)
            return set_error("missing compact multiplication NTT constants");
        works.push_back(std::move(work));
    }
    if (works.empty()) return set_error("compact multiplication has no active partition");

    size_t lhs_eval_bytes = 0;
    size_t full_output_bytes = 0;
    size_t compact_rhs_bytes = 0;
    size_t expanded_rhs_workspace_bytes = 0;
    for (const auto &work : works)
    {
        const auto &lhs_buffer = lhs_eval->shared_limb_buffers[work.partition];
        const auto &out_buffer = out->shared_limb_buffers[work.partition];
        size_t lhs_owner_bytes = 0;
        size_t output_owner_bytes = 0;
        const size_t local_limbs = work.global_limb_ids.size();
        // bytes_total is the complete physical owner allocation reported by
        // MatrixData, including its combined local-limb data and auxiliary
        // storage. Do not add aux bytes or per-limb events a second time.
        lhs_owner_bytes = lhs_buffer.bytes_total;
        output_owner_bytes = out_buffer.bytes_total;
        if (!small_add_size(lhs_eval_bytes, lhs_owner_bytes, &lhs_eval_bytes) ||
            !small_add_size(full_output_bytes, output_owner_bytes, &full_output_bytes) ||
            !small_add_size(compact_rhs_bytes, rhs_small->payload_bytes, &compact_rhs_bytes))
            return set_error("compact RHS resident size overflow");
        size_t map_bytes = 0;
        size_t map_words = 0;
        size_t map_storage_bytes = 0;
        size_t workspace_words = 0;
        if (!small_mul_size(local_limbs, sizeof(uint32_t), &map_bytes) ||
            !small_add_size(map_bytes, sizeof(uint64_t) - 1, &map_words) ||
            !small_mul_size(map_words / sizeof(uint64_t), sizeof(uint64_t), &map_storage_bytes) ||
            !small_mul_size(std::min(ell, local_limbs), kt, &workspace_words) ||
            !small_mul_size(workspace_words, ct, &workspace_words) ||
            !small_mul_size(workspace_words, n, &workspace_words) ||
            !small_mul_size(workspace_words, sizeof(uint64_t), &workspace_words) ||
            !small_add_size(map_storage_bytes, workspace_words, &workspace_words) ||
            !small_add_size(expanded_rhs_workspace_bytes, workspace_words, &expanded_rhs_workspace_bytes))
            return set_error("compact RHS workspace size overflow");
    }
    size_t event_overhead_bytes = 0;
    // Each active device has one compact payload-ready event and one
    // short-lived consumer fence. CUDA's opaque event/allocator driver
    // footprint is not represented by sizeof(cudaEvent_t); physical
    // high-water usage must be measured separately on the target device.
    if (!small_mul_size(works.size(), 2 * sizeof(cudaEvent_t), &event_overhead_bytes))
        return set_error("compact RHS event accounting overflow");
    size_t high_water_bytes = 0;
    if (!small_add_size(lhs_eval_bytes, compact_rhs_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, full_output_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, expanded_rhs_workspace_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, event_overhead_bytes, &high_water_bytes))
        return set_error("compact RHS residency size overflow");
    *allocation_report = GpuSmallMatrixAllocationReport{
        lhs_eval_bytes, compact_rhs_bytes, full_output_bytes, expanded_rhs_workspace_bytes,
        event_overhead_bytes, high_water_bytes, 0};
    if (high_water_bytes > residency_budget_bytes) return 2;

    uint32_t log_n = 0;
    for (size_t value = n; value > 1; value >>= 1) ++log_n;
    cudaError_t error = cudaSuccess;
    auto release = [&]() {
        cudaError_t cleanup = cudaSuccess;
        for (auto &work : works)
        {
            if (cudaSetDevice(work.device) != cudaSuccess) { cleanup = cudaErrorInvalidDevice; continue; }
            if (work.device_global_limb_ids)
            {
                const cudaError_t current = cudaFreeAsync(work.device_global_limb_ids, work.stream);
                if (cleanup == cudaSuccess && current != cudaSuccess) cleanup = current;
                work.device_global_limb_ids = nullptr;
            }
            if (work.workspace)
            {
                const cudaError_t current = cudaFreeAsync(work.workspace, work.stream);
                if (cleanup == cudaSuccess && current != cudaSuccess) cleanup = current;
                work.workspace = nullptr;
            }
        }
        return cleanup;
    };
    auto fail = [&](cudaError_t failure) {
        for (auto &work : works)
        {
            cudaSetDevice(work.device);
            cudaStreamSynchronize(work.stream);
        }
        const cudaError_t cleanup = release();
        if (failure == cudaSuccess) failure = cleanup;
        return set_error(failure);
    };
    for (auto &work : works)
    {
        if (cudaSetDevice(work.device) != cudaSuccess) return fail(cudaErrorInvalidDevice);
        const size_t local_limbs = work.global_limb_ids.size();
        size_t map_bytes = 0;
        size_t map_words = 0;
        size_t workspace_words = 0;
        if (!small_mul_size(local_limbs, sizeof(uint32_t), &map_bytes) ||
            !small_add_size(map_bytes, sizeof(uint64_t) - 1, &map_words) ||
            !small_mul_size(map_words / sizeof(uint64_t), sizeof(uint64_t), &map_bytes) ||
            !small_mul_size(std::min(ell, local_limbs), kt, &workspace_words) ||
            !small_mul_size(workspace_words, ct, &workspace_words) ||
            !small_mul_size(workspace_words, n, &workspace_words) ||
            !small_add_size(workspace_words, map_words / sizeof(uint64_t), &workspace_words))
            return fail(cudaErrorInvalidValue);
        if (cudaMallocAsync(reinterpret_cast<void **>(&work.device_global_limb_ids), map_bytes, work.stream) != cudaSuccess ||
            cudaMemcpyAsync(work.device_global_limb_ids, work.global_limb_ids.data(), map_bytes,
                            cudaMemcpyHostToDevice, work.stream) != cudaSuccess ||
            cudaMallocAsync(reinterpret_cast<void **>(&work.workspace), workspace_words * sizeof(uint64_t), work.stream) != cudaSuccess)
            return fail(cudaErrorMemoryAllocation);
        for (size_t c0 = 0; c0 < cols; c0 += ct)
        {
            const size_t current_ct = std::min(ct, cols - c0);
            for (size_t k0 = 0; k0 < inner; k0 += kt)
            {
                const size_t current_kt = std::min(kt, inner - k0);
                for (size_t l0 = 0; l0 < local_limbs; l0 += ell)
                {
                    const size_t current_ell = std::min(ell, local_limbs - l0);
                    const size_t current_words = current_ell * current_kt * current_ct * n;
                    uint64_t *rhs_workspace = work.workspace + map_words / sizeof(uint64_t);
                    const uint32_t *limb_map = work.device_global_limb_ids + l0;
                    compact_unpack_kernel<<<(current_words + kSmallThreads - 1) / kSmallThreads, kSmallThreads, 0, work.stream>>>(
                        work.rhs_payload, rhs_workspace, out->ctx->ntt_device_constants[work.partition].moduli,
                        limb_map, 0, current_ell, current_kt, current_ct, rhs_small->cols, n,
                        rhs_small->magnitude_bytes, k0, c0);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) return fail(error);
                    const dim3 grid((n + kSmallThreads - 1) / kSmallThreads,
                                    static_cast<uint32_t>(current_kt * current_ct), static_cast<uint32_t>(current_ell));
                    const auto &constants = out->ctx->ntt_device_constants[work.partition];
                    compact_ntt_twist_kernel<<<grid, kSmallThreads, 0, work.stream>>>(
                        rhs_workspace, constants.twiddle_forward, constants.twiddle_shoup_forward,
                        constants.moduli, limb_map, 0, current_ell, current_kt * current_ct, n);
                    error = cudaGetLastError();
                    if (error == cudaSuccess)
                        compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, work.stream>>>(
                            rhs_workspace, current_ell, current_kt * current_ct, n, log_n);
                    error = error == cudaSuccess ? cudaGetLastError() : error;
                    for (uint32_t len = 2; error == cudaSuccess && len <= n; len <<= 1)
                    {
                        compact_ntt_stage_kernel<<<grid, kSmallThreads, 0, work.stream>>>(
                            rhs_workspace, constants.twiddle_forward, constants.twiddle_shoup_forward,
                            constants.moduli, limb_map, 0, current_ell, current_kt * current_ct, n, len);
                        error = cudaGetLastError();
                    }
                    if (error == cudaSuccess)
                        compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, work.stream>>>(
                            rhs_workspace, current_ell, current_kt * current_ct, n, log_n);
                    error = error == cudaSuccess ? cudaGetLastError() : error;
                    if (error == cudaSuccess)
                    {
                        const dim3 accumulate_grid((current_ell * rows * current_ct * n + kSmallThreads - 1) / kSmallThreads);
                        compact_accumulate_kernel<<<accumulate_grid, kSmallThreads, 0, work.stream>>>(
                            work.lhs_descriptors + l0, work.out_descriptors + l0, constants.moduli,
                            limb_map, 0, rhs_workspace, current_ell, rows, inner, cols,
                            current_kt, current_ct, n, k0, c0);
                        error = cudaGetLastError();
                    }
                    if (error != cudaSuccess) return fail(error);
                }
            }
        }
    }
    for (auto &work : works)
    {
        for (uint32_t global_limb : work.global_limb_ids)
        {
            const dim3 id = out->ctx->limb_gpu_ids[global_limb];
            if (matrix_track_limb_consumer_readonly(lhs_eval, id, work.device, work.stream) != 0 ||
                matrix_record_limb_write(out, id, work.stream) != 0)
                return fail(cudaErrorInvalidResourceHandle);
        }
        if (small_track_partition_consumer(rhs_small, work.partition, work.device, work.stream) != 0)
            return fail(cudaErrorInvalidResourceHandle);
    }
    const cudaError_t cleanup_error = release();
    return cleanup_error == cudaSuccess ? 0 : set_error(cleanup_error);
}
