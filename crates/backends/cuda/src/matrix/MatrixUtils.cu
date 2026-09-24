constexpr uint32_t kGaussMaxDigits = 64;
constexpr double kTwoPi = 6.283185307179586476925286766559;

int set_error(const char *msg)
{
    return gpu_set_last_error(msg);
}

int set_error(cudaError_t err)
{
    return gpu_set_last_error_cuda(static_cast<int>(err));
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
            cudaError_t err = mxx_set_device(device);
            if (err == cudaSuccess)
            {
                cudaEventDestroy(event);
            }
            event = nullptr;
            device = -1;
        }
    };

    thread_local ThreadLocalConsumerEventState g_thread_local_consumer_event;

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
    cudaStream_t consumer_stream,
    bool device_already_selected, bool read_only)
{
    if (src && !read_only)
        src->host_observed_writer_ready.store(false, std::memory_order_release);
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
    if (state->device != consumer_device)
    {
        return set_error("device mismatch in matrix_wait_limb_stream");
    }
    cudaError_t err = device_already_selected ? cudaSuccess : mxx_set_device(consumer_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    if (read_only && src->host_observed_writer_ready.load(std::memory_order_acquire))
        return 0;
    if (!completion.write_done || !completion.write_done_valid)
    {
        return 0;
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

int matrix_record_limb_write(
    GpuMatrix *dst, const dim3 &limb_id, cudaStream_t stream, bool device_already_selected)
{
    if (dst) dst->host_observed_writer_ready.store(false, std::memory_order_release);
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
    cudaError_t err = device_already_selected ? cudaSuccess : mxx_set_device(state->device);
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
    const GpuMatrix *src, int consumer_device, cudaStream_t consumer_stream,
    bool device_already_selected, bool read_only)
{
    if (src && !read_only)
        src->host_observed_writer_ready.store(false, std::memory_order_release);
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
        const int status = matrix_wait_limb_stream(
            src, id, consumer_device, consumer_stream, device_already_selected, read_only);
        if (status != 0) return status;
    }
    return 0;
}

int matrix_record_all_limb_writes(
    GpuMatrix *dst, cudaStream_t stream, bool device_already_selected)
{
    if (dst) dst->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!matrix_has_active_limb_states(dst))
        return set_error("invalid matrix_record_all_limb_writes arguments");
    const auto &ids = dst->ctx->limb_gpu_ids;
    if (!matrix_has_uniform_limb_producer(dst))
    {
        for (int limb = 0; limb <= dst->level; ++limb)
        {
            const int status = matrix_record_limb_write(
                dst, ids[static_cast<size_t>(limb)], stream, device_already_selected);
            if (status != 0) return status;
        }
        return 0;
    }
    const int status = matrix_record_limb_write(dst, ids[0], stream, device_already_selected);
    if (status != 0) return status;
    matrix_alias_active_completions(dst);
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

__device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    return static_cast<uint64_t>(prod % mod);
}

__device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
{
    // For a, b < mod, a wrapped sum still exceeds mod, and s - mod wraps back.
    const uint64_t sum = a + b;
    return sum < a || sum >= mod ? sum - mod : sum;
}

