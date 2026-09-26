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
struct GpuSmallMatrix
{
    GpuContext *ctx = nullptr;
    size_t rows = 0;
    size_t cols = 0;
    size_t n = 0;
    size_t magnitude_bytes = 0;
    uint32_t bound_domain = 0;
    size_t crt_depth = 1;
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
    const cudaError_t err = mxx_set_device(mat->device);
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

int small_payload_size(
    size_t rows,
    size_t cols,
    size_t n,
    size_t magnitude_bytes,
    size_t crt_depth,
    size_t *out)
{
    size_t count = 0;
    size_t width = 0;
    if (!small_mul_size(rows, cols, &count) || !small_mul_size(count, n, &count) ||
        !small_mul_size(count, crt_depth, &count) ||
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
    if (width <= 7)
    {
        // A magnitude of at most 56 bits is one word, usually already
        // reduced (a gadget digit is far below the modulus).
        for (size_t i = 0; i < width; ++i) value |= static_cast<uint64_t>(magnitude[i]) << (8 * i);
        return value < modulus ? value : value % modulus;
    }
    for (size_t i = width; i-- > 0;)
    {
        value = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(value) * 256u + magnitude[i]) % modulus);
    }
    return value;
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

__global__ void preimage_status_latch_accept_kernel(MxxPreimageStatus *status)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && status->accepted == 1U)
        status->reserved = 1U;
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

}

extern "C" int gpu_small_matrix_create(
    GpuContext *ctx,
    size_t rows,
    size_t cols,
    size_t magnitude_bytes,
    uint32_t bound_domain,
    const uint64_t *bound_words,
    size_t bound_word_count,
    GpuSmallMatrix **out)
{
    if (!ctx || !out || !bound_words || bound_word_count == 0 || magnitude_bytes == 0 ||
        bound_domain > 1)
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
    mat->bound_domain = bound_domain;
    mat->crt_depth = bound_domain == 1 ? ctx->moduli.size() : 1;
    if (mat->crt_depth == 0) { delete mat; return set_error("compact CRT depth is zero"); }
    mat->bound_words.assign(bound_words, bound_words + bound_word_count);
    if (small_payload_size(rows, cols, mat->n, magnitude_bytes, mat->crt_depth,
                           &mat->payload_bytes) != 0)
    {
        delete mat;
        return 1;
    }
    mat->device = ctx->gpu_ids.front();
    if (ctx->execution->compute_streams_by_partition.empty() || ctx->execution->compute_streams_by_partition.front().empty())
    {
        delete mat;
        return set_error("missing compact matrix stream");
    }
    mat->stream = ctx->execution->compute_streams_by_partition.front().front();
    cudaError_t err = mxx_set_device(mat->device);
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
    uint32_t bound_domain,
    GpuMatrixAllocationBytes *out)
{
    if (!ctx || !out || rows == 0 || cols == 0 || magnitude_bytes == 0 || bound_domain > 1)
        return set_error("invalid compact matrix allocation query arguments");
    if (ctx->N <= 0 || magnitude_bytes > 255)
        return set_error("invalid compact matrix allocation query context or width");
    size_t payload_bytes = 0;
    if (small_payload_size(
            rows, cols, static_cast<size_t>(ctx->N), magnitude_bytes,
            bound_domain == 1 ? ctx->moduli.size() : 1, &payload_bytes) != 0)
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
    const size_t coefficient_bytes = 1 + mat->magnitude_bytes;
    *out = GpuSmallMatrixBindingDescriptor{
        mat->device,
        mat->rows,
        mat->cols,
        mat->n,
        mat->magnitude_bytes,
        mat->bound_domain,
        mat->crt_depth,
        mat->payload_bytes,
        mat->cols * mat->n * mat->crt_depth * coefficient_bytes,
        mat->n * mat->crt_depth * coefficient_bytes,
        mat->crt_depth * coefficient_bytes,
        coefficient_bytes,
        mat->payload};
    return 0;
}

extern "C" void gpu_small_matrix_destroy(GpuSmallMatrix *mat)
{
    if (!mat) return;
    if (mat->device >= 0 && mxx_set_device(mat->device) == cudaSuccess)
    {
        cudaStream_t release_stream = mat->stream;
        const size_t partition = 0;
        if (mat->ctx && partition < mat->ctx->execution->release_streams_by_partition.size() &&
            mat->ctx->execution->release_streams_by_partition[partition])
        {
            release_stream = mat->ctx->execution->release_streams_by_partition[partition];
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

extern "C" int gpu_small_matrix_load_coefficients(
    GpuSmallMatrix *mat, const uint8_t *payload, size_t payload_len)
{
    if (!mat || !payload || payload_len != mat->payload_bytes)
        return set_error("compact matrix payload length mismatch");
    if (small_set_device(mat) != 0) return 1;
    if (payload_len == 0) return 0;
    const cudaStream_t stream = mat->stream;
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

namespace
{
    // The signed compact coefficient of polynomial `poly` of the view as a
    // residue of `modulus`; a per-CRT-limb view stores limb `compact_limb`.
    __device__ __forceinline__ uint64_t compact_residue(
        const MxxRawSmallMatrixView &source, size_t poly, size_t coefficient,
        size_t compact_limb, uint64_t modulus)
    {
        const size_t row = poly / source.columns;
        const size_t column = poly - row * source.columns;
        const size_t width = 1 + source.magnitude_bytes;
        const size_t depth = source.bound_domain == 1 ? source.crt_depth : 1;
        const size_t source_poly = row * source.storage_columns +
            source.column_offset + column;
        const auto *encoded = reinterpret_cast<const uint8_t *>(source.payload_address) +
            ((source_poly * source.degree + coefficient) * depth +
                (source.bound_domain == 1 ? compact_limb : 0)) * width;
        uint64_t residue = compact_mod_magnitude(encoded + 1, source.magnitude_bytes, modulus);
        if (encoded[0] == 2 && residue != 0) residue = modulus - residue;
        return residue;
    }

    // One launch covers up to kRawNttLimbs destination limbs; blockIdx.y
    // picks the limb `first_limb + blockIdx.y`.
    __global__ void raw_small_rhs_expand_kernel(
        MxxRawSmallMatrixView source, RawLimbSet destinations,
        size_t first_limb, size_t coefficient_count, size_t coefficient_offset)
    {
        const size_t index = coefficient_offset +
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (index >= coefficient_count) return;
        const MxxRawMatrixLimb &destination = destinations.limb[blockIdx.y];
        const size_t compact_limb = first_limb + blockIdx.y;
        const size_t coefficient = index % source.degree;
        const size_t poly = index / source.degree;
        raw_matrix_store(destination, poly, coefficient, source.columns,
            compact_residue(source, poly, coefficient, compact_limb, destination.modulus));
    }
}

namespace
{
    // Reads the compact right operand as residues while the NTT loads its
    // tile, so the operand is transformed without a decoded workspace pass.
    struct RawNttCompactSource
    {
        MxxRawSmallMatrixView view;
        uint64_t first_limb;

        __device__ __forceinline__ uint64_t load(const RawNttBatch &batch, uint32_t limb,
            size_t poly, size_t coefficient, size_t) const
        {
            return compact_residue(view, poly, coefficient, first_limb + limb,
                batch.destination[limb].modulus);
        }
    };
}

static bool valid_compact_rhs_view(const MxxRawSmallMatrixView *source, size_t limb_count)
{
    return source && source->payload_address &&
        source->column_offset <= source->storage_columns &&
        source->columns <= source->storage_columns - source->column_offset &&
        source->magnitude_bytes != 0 && source->magnitude_bytes <= 64 &&
        source->bound_domain <= 1 && source->crt_depth != 0 &&
        (source->bound_domain == 1 || source->crt_depth == 1) &&
        (source->bound_domain == 0 || source->crt_depth == limb_count);
}

// left (L x K, evaluation domain) times the compact bounded right operand
// (K x C) into destination (L x C, evaluation domain). The right operand is
// transformed one workspace-wide column chunk at a time: the compact chunk is
// decoded into the workspace, the shared NTT transforms it in place, and the
// shared product kernel writes the chunk's output columns. The right operand
// is never expanded in full. A non-null `addend` (L x C, evaluation domain,
// possibly the destination itself) is added to the product in the same pass.
extern "C" int gpu_raw_matrix_mul_small_rhs(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawSmallMatrixView *right,
    const MxxRawMatrixView *workspace, const MxxRawMatrixView *destination,
    const MxxRawMatrixView *addend,
    uint32_t left_binding_base, uint32_t right_binding,
    uint32_t workspace_binding_base, uint32_t destination_binding_base,
    uint32_t addend_binding_base)
{
    if (addend &&
        (validate_raw_view(ctx, addend, stream_raw) != 0 || !same_raw_extent(addend, destination) ||
            addend_binding_base > UINT32_MAX - addend->limb_count))
        return set_error("invalid raw small-RHS product addend");
    if (validate_raw_view(ctx, left, stream_raw) != 0 ||
        validate_raw_view(ctx, workspace, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !valid_compact_rhs_view(right, left->limb_count) ||
        left->physical_device != right->physical_device ||
        left->physical_device != workspace->physical_device ||
        left->physical_device != destination->physical_device ||
        left->degree != right->degree || left->degree != workspace->degree ||
        left->degree != destination->degree ||
        left->limb_count != workspace->limb_count ||
        left->limb_count != destination->limb_count ||
        left->columns != right->rows || workspace->rows != right->rows ||
        workspace->columns == 0 || left->rows != destination->rows ||
        right->columns != destination->columns ||
        left_binding_base > UINT32_MAX - left->limb_count ||
        workspace_binding_base > UINT32_MAX - workspace->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw small-RHS product views");
    for (size_t limb = 0; limb < left->limb_count; ++limb)
        if (left->limbs[limb].crt_limb_index != workspace->limbs[limb].crt_limb_index ||
            left->limbs[limb].crt_limb_index != destination->limbs[limb].crt_limb_index ||
            left->limbs[limb].modulus != workspace->limbs[limb].modulus ||
            left->limbs[limb].modulus != destination->limbs[limb].modulus ||
            workspace->limbs[limb].address == left->limbs[limb].address ||
            workspace->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw small-RHS product limb mismatch or alias");
    if (mxx_set_device(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t degree = left->degree;
    for (size_t start = 0; start < right->columns; start += workspace->columns)
    {
        const size_t width = std::min<size_t>(workspace->columns, right->columns - start);
        MxxRawSmallMatrixView chunk = *right;
        chunk.columns = width;
        chunk.column_offset += start;
        // The chunk occupies the first `width` workspace columns.
        MxxRawMatrixView transformed = *workspace;
        transformed.columns = width;
        const size_t chunk_coefficients = right->rows * width * degree;
        constexpr size_t maximum_decode = 65535ULL * 256ULL;
        // A whole-ring tile decodes the compact chunk while the NTT loads it.
        const bool decode_in_ntt = raw_ntt_whole_ring(degree);
        if (decode_in_ntt)
        {
            const std::vector<MxxGraphPatch> loader_patches{
                {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 5,
                    static_cast<uint32_t>(offsetof(RawNttCompactSource, view) +
                        offsetof(MxxRawSmallMatrixView, payload_address)),
                    sizeof(void *), right_binding, 0}};
            if (raw_matrix_ntt(ctx, stream_raw, &transformed, &transformed, 0,
                    workspace_binding_base, workspace_binding_base,
                    RawNttCompactSource{chunk, 0}, loader_patches) != 0)
                return 1;
        }
        for (size_t first = 0; !decode_in_ntt && first < left->limb_count; first += kRawNttLimbs)
        {
            const size_t limbs = std::min(kRawNttLimbs, left->limb_count - first);
            RawLimbSet destinations{};
            std::vector<MxxGraphPatch> decode_patches{
                {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                    offsetof(MxxRawSmallMatrixView, payload_address), sizeof(void *),
                    right_binding, 0}};
            raw_limb_set(workspace, first, limbs, 1, workspace_binding_base, destinations,
                decode_patches);
            for (size_t offset = 0; offset < chunk_coefficients; offset += maximum_decode)
            {
                const size_t count = std::min(maximum_decode, chunk_coefficients - offset);
                const int status = mxx_gpu_launch_kernel(ctx, stream,
                    raw_small_rhs_expand_kernel,
                    dim3(static_cast<uint32_t>((count + 255) / 256), static_cast<uint32_t>(limbs)),
                    dim3(256), 0, decode_patches.data(), decode_patches.size(), chunk,
                    destinations, first, chunk_coefficients, offset);
                if (status != 0) return status;
            }
        }
        if (!decode_in_ntt &&
            gpu_raw_matrix_ntt(ctx, stream_raw, &transformed, &transformed, 0,
                workspace_binding_base, workspace_binding_base) != 0)
            return 1;
        // The shared product kernel writes the chunk's output columns, which
        // start `start` columns into the destination.
        const size_t output_polys = destination->rows * width;
        for (size_t first = 0; first < left->limb_count; first += kRawNttLimbs)
        {
            const size_t limbs = std::min(kRawNttLimbs, left->limb_count - first);
            RawLimbSet lefts{}, rights{}, outputs{}, addends{};
            std::vector<MxxGraphPatch> patches;
            raw_limb_set(left, first, limbs, 0, left_binding_base, lefts, patches);
            raw_limb_set(&transformed, first, limbs, 1, workspace_binding_base, rights, patches);
            raw_limb_set(destination, first, limbs, 2, destination_binding_base, outputs, patches,
                start);
            if (addend)
                raw_limb_set(addend, first, limbs, 3, addend_binding_base, addends, patches, start);
            for (size_t offset = 0; offset < output_polys; offset += kMaxGridY)
            {
                const size_t polys = std::min(kMaxGridY, output_polys - offset);
                const dim3 grid((degree + kMulCoefficients - 1) / kMulCoefficients,
                    static_cast<uint32_t>(polys), static_cast<uint32_t>(limbs));
                const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_mul_kernel,
                    grid, dim3(kMulCoefficients, kMulSlices), 0, patches.data(), patches.size(),
                    lefts, rights, outputs, addends, static_cast<size_t>(left->columns), width,
                    width, static_cast<size_t>(destination->rows), static_cast<size_t>(degree),
                    offset, addend != nullptr, false);
                if (status != 0) return status;
            }
        }
    }
    return 0;
}

extern "C" int gpu_raw_small_rhs_expand(GpuContext *ctx, void *stream_raw,
    const MxxRawSmallMatrixView *source,
    const MxxRawMatrixView *destination,
    uint32_t source_binding, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !valid_compact_rhs_view(source, destination->limb_count) ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->rows != destination->rows ||
        source->columns != destination->columns ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw compact RHS expansion views");
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    if (source->rows > SIZE_MAX / source->columns ||
        source->rows * source->columns > SIZE_MAX / source->degree)
        return set_error("raw compact RHS coefficient count overflow");
    const size_t count = source->rows * source->columns * source->degree;
    constexpr size_t maximum_chunk = 65535ULL * 256ULL;
    for (size_t first = 0; first < destination->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, destination->limb_count - first);
        RawLimbSet destinations{};
        std::vector<MxxGraphPatch> patches{
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                offsetof(MxxRawSmallMatrixView, payload_address), sizeof(void *),
                source_binding, 0}};
        raw_limb_set(destination, first, limbs, 1, destination_binding_base, destinations,
            patches);
        for (size_t offset = 0; offset < count; offset += maximum_chunk)
        {
            const size_t chunk = std::min(maximum_chunk, count - offset);
            const dim3 grid(static_cast<uint32_t>((chunk + 255) / 256),
                static_cast<uint32_t>(limbs));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_small_rhs_expand_kernel, grid, dim3(256), 0,
                patches.data(), patches.size(), *source, destinations, first, count, offset);
            if (status != 0) return status;
        }
    }
    return 0;
}
