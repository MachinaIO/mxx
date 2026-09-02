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
    if (err == cudaSuccess) err = cudaEventRecord(consumer_done, consumer_stream);
    if (err == cudaSuccess) err = cudaStreamWaitEvent(release_stream, consumer_done, 0);
    const cudaError_t destroy_err = consumer_done ? cudaEventDestroy(consumer_done) : cudaSuccess;
    if (err == cudaSuccess) err = destroy_err;
    if (err != cudaSuccess)
    {
        // The owner can be dropped immediately after this error.  Fence the
        // consumer before returning, while leaving the producer event intact.
        cudaStreamSynchronize(consumer_stream);
        return set_error(err);
    }
    return 0;
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
    const uint64_t modulus = moduli[limb_offset + local_limb];
    uint64_t value = compact_mod_magnitude(src + 1, magnitude_bytes, modulus);
    if (src[0] == 2 && value != 0) value = modulus - value;
    workspace[idx] = value;
}

__global__ void compact_ntt_twist_kernel(
    uint64_t *workspace,
    const uint64_t *twiddles,
    const uint64_t *twiddle_shoup,
    const uint64_t *moduli,
    size_t limb_offset,
    size_t limb_count,
    size_t poly_count,
    size_t n)
{
    const size_t coeff = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t poly = static_cast<size_t>(blockIdx.y);
    const size_t local = static_cast<size_t>(blockIdx.z);
    if (coeff >= n || poly >= poly_count || local >= limb_count) return;
    const size_t global = limb_offset + local;
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
    const size_t global = limb_offset + local;
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
    const size_t global = limb_offset + local;
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
    return small_track_consumer(src, out->stream);
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
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int device = -1;
        if (matrix_limb_device(src, id, &device) != 0 || device != dst->device ||
            !matrix_limb_metadata_by_id(src, id, &limb_strides[limb], &limb_widths[limb]))
            return set_error("compact tile requires one-device active CRT placement");
        limb_ptrs[limb] = matrix_limb_ptr_by_id(src, 0, id);
        if (!limb_ptrs[limb] ||
            matrix_wait_limb_stream(src, id, dst->device, dst->stream) != 0)
            return set_error("invalid compact tile active CRT limb");
    }
    const std::vector<uint64_t> moduli(
        src->ctx->moduli.begin(), src->ctx->moduli.begin() + limb_count);
    std::vector<uint64_t> modulus_words;
    if (!serde_compute_modulus_words_le(moduli, &modulus_words))
        return set_error("failed to compute compact tile CRT modulus");
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
                d_accepted,
                dst->payload,
                dst->cols,
                dst->magnitude_bytes,
                dst_row,
                dst_col,
                cols);
    if (err == cudaSuccess) err = cudaGetLastError();
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
    GpuMatrix *out,
    const GpuMatrix *lhs_eval,
    const GpuSmallMatrix *rhs_small,
    size_t ct,
    size_t kt,
    size_t ell,
    size_t residency_budget_bytes,
    GpuSmallMatrixAllocationReport *allocation_report)
{
    if (!out || !lhs_eval || !rhs_small || !out->ctx || out->ctx != lhs_eval->ctx || out->ctx != rhs_small->ctx ||
        lhs_eval->format != GPU_POLY_FORMAT_EVAL || out->format != GPU_POLY_FORMAT_EVAL ||
        lhs_eval->cols != rhs_small->rows || out->rows != lhs_eval->rows || out->cols != rhs_small->cols ||
        ct == 0 || kt == 0 || ell == 0 || !allocation_report)
        return set_error("invalid compact RHS multiplication arguments");
    const size_t limbs = static_cast<size_t>(lhs_eval->level + 1);
    if (lhs_eval->level < 0 || limbs == 0 || limbs > kMaxSmallLimbCount || out->level != lhs_eval->level ||
        out->ctx->limb_gpu_ids.size() < limbs || !is_power_of_two_u32(static_cast<uint32_t>(out->ctx->N)))
        return set_error("invalid compact RHS multiplication level");
    if (small_set_device(rhs_small) != 0) return 1;
    cudaStream_t stream = nullptr;
    std::vector<size_t> lhs_strides(limbs), out_strides(limbs);
    int dispatch_device = -1;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        if (id.x >= out->shared_limb_buffers.size())
            return set_error("invalid compact multiplication limb partition");
        int lhs_device = -1;
        int out_device = -1;
        uint8_t lhs_width = 0;
        uint8_t out_width = 0;
        if (matrix_limb_device(lhs_eval, id, &lhs_device) != 0 || matrix_limb_device(out, id, &out_device) != 0 ||
            lhs_device != rhs_small->device || out_device != rhs_small->device ||
            id.y >= lhs_eval->shared_limb_buffers[id.x].limb_count ||
            id.y >= out->shared_limb_buffers[id.x].limb_count ||
            !lhs_eval->shared_limb_buffers[id.x].device_descriptors ||
            !out->shared_limb_buffers[id.x].device_descriptors ||
            !matrix_limb_metadata_by_id(lhs_eval, id, &lhs_strides[limb], &lhs_width) ||
            !matrix_limb_metadata_by_id(out, id, &out_strides[limb], &out_width))
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
    if (ct > cols || kt > inner || ell > limbs)
        return set_error("resolved compact RHS tile exceeds operation dimensions");
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
        // Failure paths may leave output work queued without a completion
        // event.  Fence before releasing temporary metadata or returning an
        // owner to Rust; the normal success path remains fully asynchronous.
        cudaStreamSynchronize(stream);
        const cudaError_t cleanup_err = release();
        if (failure == cudaSuccess) failure = cleanup_err;
        return set_error(failure);
    };
    constexpr size_t metadata_bytes = 0;
    size_t lhs_eval_bytes = 0;
    size_t full_output_bytes = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        size_t bytes = 0;
        if (!small_mul_size(lhs_strides[limb], lhs_eval->rows, &bytes) ||
            !small_add_size(lhs_eval_bytes, bytes, &lhs_eval_bytes) ||
            !small_mul_size(out_strides[limb], out->rows, &bytes) ||
            !small_add_size(full_output_bytes, bytes, &full_output_bytes))
            return set_error("compact RHS resident size overflow");
    }
    size_t workspace_words = 0;
    size_t expanded_rhs_workspace_bytes = 0;
    if (!small_mul_size(ell, kt, &workspace_words) || !small_mul_size(workspace_words, ct, &workspace_words) ||
        !small_mul_size(workspace_words, n, &workspace_words) ||
        !small_mul_size(workspace_words, sizeof(uint64_t), &expanded_rhs_workspace_bytes))
        return set_error("compact RHS workspace size overflow");
    size_t event_overhead_bytes = 0;
    if (!small_mul_size(limbs + 1, sizeof(cudaEvent_t), &event_overhead_bytes))
        return set_error("compact RHS event accounting overflow");
    size_t high_water_bytes = 0;
    if (!small_add_size(lhs_eval_bytes, rhs_small->payload_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, full_output_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, metadata_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, expanded_rhs_workspace_bytes, &high_water_bytes) ||
        !small_add_size(high_water_bytes, event_overhead_bytes, &high_water_bytes))
        return set_error("compact RHS residency size overflow");
    allocation_report->lhs_eval_bytes = lhs_eval_bytes;
    allocation_report->compact_rhs_bytes = rhs_small->payload_bytes;
    allocation_report->full_output_bytes = full_output_bytes;
    allocation_report->expanded_rhs_workspace_bytes = expanded_rhs_workspace_bytes;
    allocation_report->event_overhead_bytes = event_overhead_bytes;
    allocation_report->high_water_bytes = high_water_bytes;
    allocation_report->full_expanded_rhs_bytes = 0;
    if (high_water_bytes > residency_budget_bytes)
        return 2;
    cudaError_t err = cudaMallocAsync(
        reinterpret_cast<void **>(&workspace),
        workspace_words * sizeof(uint64_t),
        stream);
    if (err != cudaSuccess) return fail(err);

    for (size_t c0 = 0; c0 < cols; c0 += ct)
    {
        const size_t current_ct = std::min(ct, cols - c0);
        for (size_t k0 = 0; k0 < inner; k0 += kt)
        {
            const size_t current_kt = std::min(kt, inner - k0);
            for (size_t l0 = 0; l0 < limbs; l0 += ell)
            {
                const size_t current_ell = std::min(ell, limbs - l0);
                const size_t current_workspace_words = current_ell * current_kt * current_ct * n;
                const dim3 unpack_grid((current_workspace_words + kSmallThreads - 1) / kSmallThreads);
                compact_unpack_kernel<<<unpack_grid, kSmallThreads, 0, stream>>>(
                    rhs_small->payload, workspace, constants.moduli, l0, current_ell, current_kt, current_ct, rhs_small->cols, n,
                    rhs_small->magnitude_bytes, k0, c0);
                err = cudaGetLastError();
                if (err == cudaSuccess)
                {
                    const dim3 grid((n + kSmallThreads - 1) / kSmallThreads, static_cast<uint32_t>(current_kt * current_ct), static_cast<uint32_t>(current_ell));
                    compact_ntt_twist_kernel<<<grid, kSmallThreads, 0, stream>>>(workspace, constants.twiddle_forward,
                        constants.twiddle_shoup_forward, constants.moduli, l0, current_ell, current_kt * current_ct, n);
                    err = cudaGetLastError();
                    if (err == cudaSuccess)
                        compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, stream>>>(workspace, current_ell, current_kt * current_ct, n, log_n);
                    err = err == cudaSuccess ? cudaGetLastError() : err;
                }
                for (uint32_t len = 2; err == cudaSuccess && len <= n; len <<= 1)
                {
                    const dim3 grid((n / 2 + kSmallThreads - 1) / kSmallThreads, static_cast<uint32_t>(current_kt * current_ct), static_cast<uint32_t>(current_ell));
                    compact_ntt_stage_kernel<<<grid, kSmallThreads, 0, stream>>>(workspace, constants.twiddle_forward,
                        constants.twiddle_shoup_forward, constants.moduli, l0, current_ell, current_kt * current_ct, n, len);
                    err = cudaGetLastError();
                }
                if (err == cudaSuccess)
                {
                    const dim3 grid((n + kSmallThreads - 1) / kSmallThreads,
                                    static_cast<uint32_t>(current_kt * current_ct), static_cast<uint32_t>(current_ell));
                    compact_ntt_bit_reverse_kernel<<<grid, kSmallThreads, 0, stream>>>(
                        workspace, current_ell, current_kt * current_ct, n, log_n);
                    err = cudaGetLastError();
                }
                if (err == cudaSuccess)
                {
                    const dim3 grid((current_ell * rows * current_ct * n + kSmallThreads - 1) / kSmallThreads);
                    compact_accumulate_kernel<<<grid, kSmallThreads, 0, stream>>>(
                        lhs_descriptors + l0, out_descriptors + l0, constants.moduli, l0, workspace, current_ell,
                        rows, inner, cols, current_kt, current_ct, n, k0, c0);
                    err = cudaGetLastError();
                }
                if (err != cudaSuccess) return fail(err);
            }
        }
    }
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
