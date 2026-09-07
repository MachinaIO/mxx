constexpr uint32_t kGaussMaxDigits = 64;
constexpr double kTwoPi = 6.283185307179586476925286766559;

int set_error(const char *msg)
{
    return gpu_set_last_error(msg);
}

int set_error(cudaError_t err)
{
    return gpu_set_last_error(cudaGetErrorString(err));
}

bool parse_format(int format, GpuPolyFormat &out)
{
    switch (format)
    {
    case GPU_POLY_FORMAT_COEFF:
        out = GPU_POLY_FORMAT_COEFF;
        return true;
    case GPU_POLY_FORMAT_EVAL:
        out = GPU_POLY_FORMAT_EVAL;
        return true;
    default:
        return false;
    }
}

size_t matrix_poly_count(const GpuMatrix *mat)
{
    if (!mat)
    {
        return 0;
    }
    return mat->rows * mat->cols;
}

namespace
{
    struct ThreadLocalConsumerEventState
    {
        int device = -1;
        cudaEvent_t event = nullptr;

        ~ThreadLocalConsumerEventState()
        {
            if (!event || device < 0)
            {
                return;
            }
            cudaError_t err = cudaSetDevice(device);
            if (err == cudaSuccess)
            {
                cudaEventDestroy(event);
            }
            event = nullptr;
            device = -1;
        }
    };

    thread_local ThreadLocalConsumerEventState g_thread_local_consumer_event;

    int matrix_get_thread_local_consumer_event(int device, cudaEvent_t *out_event)
    {
        if (device < 0 || !out_event)
        {
            return set_error("invalid matrix_get_thread_local_consumer_event arguments");
        }

        auto &tls = g_thread_local_consumer_event;
        if (tls.event && tls.device == device)
        {
            *out_event = tls.event;
            return 0;
        }

        if (tls.event)
        {
            cudaError_t err = cudaSetDevice(tls.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaEventDestroy(tls.event);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            tls.event = nullptr;
            tls.device = -1;
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaEventCreateWithFlags(&tls.event, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            tls.event = nullptr;
            tls.device = -1;
            return set_error(err);
        }
        tls.device = device;
        *out_event = tls.event;
        return 0;
    }

    const GpuMatrix::LimbExecState *matrix_limb_state_const(
        const GpuMatrix *mat,
        const dim3 &limb_id,
        const char *context)
    {
        if (!mat || !mat->ctx)
        {
            set_error("invalid matrix context");
            return nullptr;
        }
        if (limb_id.x >= mat->exec_limb_states.size())
        {
            set_error(context);
            return nullptr;
        }
        const auto &states = mat->exec_limb_states[limb_id.x];
        if (limb_id.y >= states.size())
        {
            set_error(context);
            return nullptr;
        }
        return &states[limb_id.y];
    }

    GpuMatrix::LimbExecState *matrix_limb_state(
        GpuMatrix *mat,
        const dim3 &limb_id,
        const char *context)
    {
        return const_cast<GpuMatrix::LimbExecState *>(
            matrix_limb_state_const(mat, limb_id, context));
    }
}


int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device)
{
    if (!mat || !mat->ctx || !out_device)
    {
        return set_error("invalid matrix_limb_device arguments");
    }
    const auto *state =
        matrix_limb_state_const(mat, limb_id, "invalid limb index in matrix_limb_device");
    if (!state)
    {
        return 1;
    }
    *out_device = state->device;
    return 0;
}

int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, cudaStream_t *out_stream)
{
    if (!mat || !mat->ctx || !out_stream)
    {
        return set_error("invalid matrix_limb_stream arguments");
    }
    const auto *state =
        matrix_limb_state_const(mat, limb_id, "invalid limb index in matrix_limb_stream");
    if (!state)
    {
        return 1;
    }
    *out_stream = state->stream;
    return *out_stream ? 0 : set_error("null stream in matrix_limb_stream");
}

uint8_t *matrix_limb_ptr_by_id(GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
{
    if (!mat || limb_id.x >= mat->shared_limb_buffers.size())
    {
        return nullptr;
    }
    const size_t count = matrix_poly_count(mat);
    if (poly_idx >= count)
    {
        return nullptr;
    }
    auto &buffer = mat->shared_limb_buffers[limb_id.x];
    if (!buffer.ptr || limb_id.y >= buffer.limb_count)
    {
        return nullptr;
    }
    if (limb_id.y >= buffer.limb_offsets_bytes.size() || limb_id.y >= buffer.limb_coeff_bytes.size())
    {
        return nullptr;
    }
    const size_t coeff_bytes = static_cast<size_t>(buffer.limb_coeff_bytes[limb_id.y]);
    if (coeff_bytes == 0)
    {
        return nullptr;
    }
    size_t coeff_region_bytes = 0;
    if (buffer.n != 0 && coeff_bytes > static_cast<size_t>(-1) / buffer.n)
    {
        return nullptr;
    }
    coeff_region_bytes = buffer.n * coeff_bytes;
    const size_t base_offset = buffer.limb_offsets_bytes[limb_id.y];
    size_t poly_offset = 0;
    if (poly_idx != 0 && buffer.bytes_per_poly > static_cast<size_t>(-1) / poly_idx)
    {
        return nullptr;
    }
    poly_offset = poly_idx * buffer.bytes_per_poly;
    if (base_offset > static_cast<size_t>(-1) - poly_offset)
    {
        return nullptr;
    }
    const size_t offset_bytes = poly_offset + base_offset;
    if (offset_bytes > buffer.bytes_total || buffer.bytes_total - offset_bytes < coeff_region_bytes)
    {
        return nullptr;
    }
    return buffer.ptr + offset_bytes;
}

const uint8_t *matrix_limb_ptr_by_id(const GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
{
    return matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(mat), poly_idx, limb_id);
}

bool matrix_limb_metadata_by_id(
    const GpuMatrix *mat,
    const dim3 &limb_id,
    size_t *out_stride_bytes,
    uint8_t *out_coeff_bytes)
{
    if (!mat || !out_stride_bytes || !out_coeff_bytes || limb_id.x >= mat->shared_limb_buffers.size())
    {
        return false;
    }
    const auto &buffer = mat->shared_limb_buffers[limb_id.x];
    if (limb_id.y >= buffer.limb_count ||
        limb_id.y >= buffer.limb_coeff_bytes.size() ||
        limb_id.y >= buffer.limb_offsets_bytes.size())
    {
        return false;
    }
    const uint8_t coeff_bytes = buffer.limb_coeff_bytes[limb_id.y];
    if (coeff_bytes == 0)
    {
        return false;
    }
    *out_stride_bytes = buffer.bytes_per_poly;
    *out_coeff_bytes = coeff_bytes;
    return true;
}

int matrix_wait_limb_stream(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream)
{
    if (!src || !src->ctx || !consumer_stream || consumer_device < 0)
    {
        return set_error("invalid matrix_wait_limb_stream arguments");
    }
    const auto *state =
        matrix_limb_state_const(src, limb_id, "invalid limb index in matrix_wait_limb_stream");
    if (!state)
    {
        return 1;
    }
    const auto &completion = src->exec_limb_states[limb_id.x][state->completion_owner];
    if (!completion.write_done || !completion.write_done_valid)
    {
        return 0;
    }
    if (state->device != consumer_device)
    {
        return set_error("device mismatch in matrix_wait_limb_stream");
    }
    cudaError_t err = cudaSetDevice(consumer_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    // Stream order already covers this event when its latest record was on
    // the consumer itself. The owning state->stream alone cannot establish it.
    if (completion.last_write_stream == consumer_stream)
    {
        return 0;
    }
    err = cudaStreamWaitEvent(consumer_stream, completion.write_done, 0);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int matrix_track_limb_consumer(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream)
{
    if (!src || !src->ctx || !consumer_stream || consumer_device < 0)
    {
        return set_error("invalid matrix_track_limb_consumer arguments");
    }
    auto *state = matrix_limb_state(
        const_cast<GpuMatrix *>(src),
        limb_id,
        "invalid limb index in matrix_track_limb_consumer");
    if (!state)
    {
        return 1;
    }
    if (state->device != consumer_device)
    {
        return set_error("device mismatch in matrix_track_limb_consumer");
    }
    if (!state->stream)
    {
        return set_error("null producer stream in matrix_track_limb_consumer");
    }
    cudaError_t err = cudaSetDevice(consumer_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    // A shared-stream matrix may still alias its partition's initial event.
    // Acquire this limb's owned event before recording a separate completion.
    if (!state->write_done)
    {
        err = cudaEventCreateWithFlags(&state->write_done, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            // The consumer kernel is already queued. Its old completion alias
            // cannot protect source cleanup when lazy event allocation fails.
            cudaStreamSynchronize(consumer_stream);
            return set_error(err);
        }
    }

    // Fast path: consumer already runs on the producer stream.
    if (state->stream == consumer_stream)
    {
        err = cudaEventRecord(state->write_done, consumer_stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        state->completion_owner = limb_id.y;
        state->last_write_stream = consumer_stream;
        state->write_done_valid = true;
        return 0;
    }

    cudaEvent_t consumer_done = nullptr;
    int status = matrix_get_thread_local_consumer_event(consumer_device, &consumer_done);
    if (status != 0)
    {
        return status;
    }
    err = cudaEventRecord(consumer_done, consumer_stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaStreamWaitEvent(state->stream, consumer_done, 0);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    // Fold consumer completion into write_done so later waits/free use one event.
    err = cudaEventRecord(state->write_done, state->stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    state->completion_owner = limb_id.y;
    state->last_write_stream = state->stream;
    state->write_done_valid = true;
    return 0;
}

int matrix_track_limb_consumer_readonly(
    const GpuMatrix *src,
    const dim3 &limb_id,
    int consumer_device,
    cudaStream_t consumer_stream)
{
    if (!src || !src->ctx || !consumer_stream || consumer_device < 0)
    {
        return set_error("invalid matrix_track_limb_consumer_readonly arguments");
    }
    const auto *state = matrix_limb_state_const(
        src,
        limb_id,
        "invalid limb index in matrix_track_limb_consumer_readonly");
    if (!state)
    {
        return 1;
    }
    if (state->device != consumer_device || !state->stream)
    {
        return set_error("invalid source placement in matrix_track_limb_consumer_readonly");
    }
    if (limb_id.x >= src->ctx->execution->release_streams_by_partition.size() ||
        !src->ctx->execution->release_streams_by_partition[limb_id.x])
    {
        return set_error("missing source release stream in matrix_track_limb_consumer_readonly");
    }

    cudaError_t err = cudaSetDevice(consumer_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    cudaEvent_t consumer_done = nullptr;
    err = cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming);
    if (err == cudaSuccess)
    {
        err = cudaEventRecord(consumer_done, consumer_stream);
    }
    if (err == cudaSuccess)
    {
        err = cudaStreamWaitEvent(
            src->ctx->execution->release_streams_by_partition[limb_id.x],
            consumer_done,
            0);
    }
    const cudaError_t destroy_err = consumer_done ? cudaEventDestroy(consumer_done) : cudaSuccess;
    if (err == cudaSuccess)
    {
        err = destroy_err;
    }
    if (err != cudaSuccess)
    {
        // The caller may release the source immediately after an error.  A
        // synchronous error-path fence keeps that release safe without
        // changing the producer's write_done event.
        cudaStreamSynchronize(consumer_stream);
        return set_error(err);
    }
    return 0;
}

int matrix_record_limb_write(GpuMatrix *dst, const dim3 &limb_id, cudaStream_t stream)
{
    if (!dst || !dst->ctx)
    {
        return set_error("invalid matrix_record_limb_write arguments");
    }
    auto *state =
        matrix_limb_state(dst, limb_id, "invalid limb index in matrix_record_limb_write");
    if (!state)
    {
        return 1;
    }
    if (!stream)
    {
        stream = state->stream;
    }
    if (!stream)
    {
        return set_error("invalid stream in matrix_record_limb_write");
    }
    cudaError_t err = cudaSetDevice(state->device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    if (!state->write_done)
    {
        err = cudaEventCreateWithFlags(&state->write_done, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            // The write is already queued; protect immediate owner cleanup
            // when its new completion event cannot be allocated.
            cudaStreamSynchronize(stream);
            return set_error(err);
        }
    }
    err = cudaEventRecord(state->write_done, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    state->completion_owner = limb_id.y;
    state->last_write_stream = stream;
    state->write_done_valid = true;
    return 0;
}

namespace
{
    // Whole-matrix aliasing is confined to one partition with one producer
    // stream. General layouts retain the established per-limb dependencies.
    bool matrix_has_uniform_limb_producer(const GpuMatrix *matrix)
    {
        const auto &ids = matrix->ctx->limb_gpu_ids;
        const auto *first = matrix_limb_state_const(matrix, ids[0], "missing first limb state");
        if (!first || !first->stream) return false;
        for (int limb = 1; limb <= matrix->level; ++limb)
        {
            const auto id = ids[static_cast<size_t>(limb)];
            const auto *state = matrix_limb_state_const(matrix, id, "missing limb state");
            if (id.x != ids[0].x || !state || state->stream != first->stream || state->device != first->device)
                return false;
        }
        return true;
    }

    bool matrix_has_active_limb_states(const GpuMatrix *matrix)
    {
        return matrix && matrix->ctx && matrix->level >= 0 &&
            matrix->ctx->limb_gpu_ids.size() > static_cast<size_t>(matrix->level);
    }

    void matrix_alias_active_completions(GpuMatrix *matrix)
    {
        const auto &ids = matrix->ctx->limb_gpu_ids;
        auto &states = matrix->exec_limb_states[ids[0].x];
        for (int limb = 0; limb <= matrix->level; ++limb)
            states[ids[static_cast<size_t>(limb)].y].completion_owner = ids[0].y;
    }
}

int matrix_wait_all_limb_streams(
    const GpuMatrix *src, int consumer_device, cudaStream_t consumer_stream)
{
    if (!matrix_has_active_limb_states(src) || consumer_device < 0 || !consumer_stream)
        return set_error("invalid matrix_wait_all_limb_streams arguments");
    const auto &ids = src->ctx->limb_gpu_ids;
    const bool uniform = matrix_has_uniform_limb_producer(src);
    uint64_t seen_completions = 0;
    for (int limb = 0; limb <= src->level; ++limb)
    {
        const auto id = ids[static_cast<size_t>(limb)];
        if (uniform)
        {
            const auto &state = src->exec_limb_states[id.x][id.y];
            const uint64_t bit = uint64_t{1} << state.completion_owner;
            if (seen_completions & bit) continue;
            seen_completions |= bit;
        }
        const int status = matrix_wait_limb_stream(src, id, consumer_device, consumer_stream);
        if (status != 0) return status;
    }
    return 0;
}

int matrix_track_all_limb_consumers(
    const GpuMatrix *src, int consumer_device, cudaStream_t consumer_stream)
{
    if (!matrix_has_active_limb_states(src) || consumer_device < 0 || !consumer_stream)
        return set_error("invalid matrix_track_all_limb_consumers arguments");
    const auto &ids = src->ctx->limb_gpu_ids;
    if (!matrix_has_uniform_limb_producer(src))
    {
        for (int limb = 0; limb <= src->level; ++limb)
        {
            const int status = matrix_track_limb_consumer(
                src, ids[static_cast<size_t>(limb)], consumer_device, consumer_stream);
            if (status != 0) return status;
        }
        return 0;
    }
    // The ordinary helper retains the producer-stream bridge when needed.
    // Its recorded event dominates this consumer's work across all limbs.
    const int status = matrix_track_limb_consumer(src, ids[0], consumer_device, consumer_stream);
    if (status != 0) return status;
    matrix_alias_active_completions(const_cast<GpuMatrix *>(src));
    return 0;
}

int matrix_record_all_limb_writes(GpuMatrix *dst, cudaStream_t stream)
{
    if (!matrix_has_active_limb_states(dst))
        return set_error("invalid matrix_record_all_limb_writes arguments");
    const auto &ids = dst->ctx->limb_gpu_ids;
    if (!matrix_has_uniform_limb_producer(dst))
    {
        for (int limb = 0; limb <= dst->level; ++limb)
        {
            const int status = matrix_record_limb_write(dst, ids[static_cast<size_t>(limb)], stream);
            if (status != 0) return status;
        }
        return 0;
    }
    const int status = matrix_record_limb_write(dst, ids[0], stream);
    if (status != 0) return status;
    matrix_alias_active_completions(dst);
    return 0;
}

bool matrix_aux_slice_for_limb(const GpuMatrix *mat, const dim3 &limb_id, size_t bytes, void **out_ptr)
{
    if (!out_ptr)
    {
        return false;
    }
    *out_ptr = nullptr;
    if (!mat || limb_id.x >= mat->shared_aux_buffers.size() ||
        limb_id.x >= mat->shared_limb_buffers.size())
    {
        return false;
    }

    const auto &aux_buffer = mat->shared_aux_buffers[limb_id.x];
    const auto &limb_buffer = mat->shared_limb_buffers[limb_id.x];
    if (!aux_buffer.ptr || limb_buffer.limb_count == 0 || limb_id.y >= limb_buffer.limb_count)
    {
        return false;
    }

    size_t total_bytes = 0;
    if (aux_buffer.slots_total > static_cast<size_t>(-1) / sizeof(void *))
    {
        return false;
    }
    total_bytes = aux_buffer.slots_total * sizeof(void *);
    if (total_bytes == 0)
    {
        return false;
    }

    const size_t limbs = limb_buffer.limb_count;
    const size_t alignment = alignof(void *);
    size_t bytes_per_limb = total_bytes / limbs;
    bytes_per_limb -= bytes_per_limb % alignment;
    if (bytes_per_limb == 0 || bytes > bytes_per_limb)
    {
        return false;
    }

    const size_t limb_offset = static_cast<size_t>(limb_id.y) * bytes_per_limb;
    if (limb_offset > total_bytes || total_bytes - limb_offset < bytes)
    {
        return false;
    }

    auto *base = reinterpret_cast<uint8_t *>(aux_buffer.ptr);
    *out_ptr = static_cast<void *>(base + limb_offset);
    return true;
}

size_t matrix_align_up_size(size_t value, size_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

int matrix_acquire_aux_workspace(
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id,
    size_t bytes,
    void **out_ptr,
    bool *out_shared,
    cudaStream_t stream)
{
    if (!out_ptr || !out_shared)
    {
        return set_error("invalid matrix_acquire_aux_workspace arguments");
    }
    *out_ptr = nullptr;
    *out_shared = false;
    if (bytes == 0)
    {
        return 0;
    }
    if (aux_owner && aux_limb_id)
    {
        if (matrix_aux_slice_for_limb(aux_owner, *aux_limb_id, bytes, out_ptr))
        {
            *out_shared = true;
            return 0;
        }
        return set_error("preallocated matrix auxiliary workspace is insufficient");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_acquire_aux_workspace");
    }
    cudaError_t err = cudaMallocAsync(out_ptr, bytes, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int matrix_release_aux_workspace(void *ptr, bool from_shared, cudaStream_t stream)
{
    if (!ptr || from_shared)
    {
        return 0;
    }
    if (!stream)
    {
        return set_error("null stream in matrix_release_aux_workspace");
    }
    cudaError_t err = cudaFreeAsync(ptr, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


uint32_t bit_width_u64(uint64_t v)
{
    if (v == 0)
    {
        return 0;
    }
    return static_cast<uint32_t>(64 - __builtin_clzll(v));
}

__host__ __device__ __forceinline__ size_t matrix_index(size_t row, size_t col, size_t cols)
{
    return row * cols + col;
}

__device__ __forceinline__ uint32_t mul_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
{
    uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(prod % mod);
}

__device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % mod);
}

// Returns floor(2^64 / modulus), which fits in uint64_t for modulus > 1. This
// reciprocal is used only by the narrow-matrix multiplication fast path. Keep
// the general 64-bit modular multiplication above for CRT moduli wider than 32
// bits and for callers that do not guarantee canonical inputs.
bool matrix_barrett_u32_reciprocal(uint64_t modulus, uint64_t *out_reciprocal)
{
    if (!out_reciprocal || modulus <= 1 || modulus > UINT32_MAX)
    {
        return false;
    }
    const unsigned __int128 two_to_64 = static_cast<unsigned __int128>(1) << 64U;
    *out_reciprocal = static_cast<uint64_t>(two_to_64 / modulus);
    return true;
}

bool matrix_lazy_dot_u64(size_t inner, uint64_t modulus)
{
    if (modulus <= 1 || modulus > UINT32_MAX)
    {
        return false;
    }
    const unsigned __int128 maximum = static_cast<unsigned __int128>(modulus - 1) *
                                      static_cast<unsigned __int128>(modulus - 1) *
                                      static_cast<unsigned __int128>(inner);
    return maximum <= UINT64_MAX;
}

__device__ __forceinline__ uint64_t reduce_barrett_u32(
    uint64_t value,
    uint64_t modulus,
    uint64_t reciprocal)
{
    const uint64_t quotient = __umul64hi(value, reciprocal);
    const uint64_t remainder = value - quotient * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

// Preconditions: 1 < modulus <= UINT32_MAX, a < modulus, b < modulus, and
// reciprocal = floor(2^64 / modulus). Then product = a*b fits in uint64_t.
// The reciprocal quotient underestimates floor(product/modulus) by at most one,
// so a single conditional subtraction produces the canonical residue.
__device__ __forceinline__ uint64_t mul_mod_barrett_u32(
    uint64_t a,
    uint64_t b,
    uint64_t modulus,
    uint64_t reciprocal)
{
    const uint64_t product = a * b;
    const uint64_t quotient = __umul64hi(product, reciprocal);
    const uint64_t remainder = product - quotient * modulus;
    return remainder >= modulus ? remainder - modulus : remainder;
}

__device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
{
    uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
    if (sum >= mod)
    {
        sum -= mod;
    }
    return static_cast<uint32_t>(sum);
}

__device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
    if (sum >= mod)
    {
        sum -= mod;
    }
    return static_cast<uint64_t>(sum);
}
