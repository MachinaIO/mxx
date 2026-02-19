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

int default_batch(const GpuContext *ctx)
{
    return static_cast<int>(ctx && ctx->batch != 0 ? ctx->batch : 1);
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

uint64_t *matrix_limb_ptr_by_id(GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
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
    const size_t offset_words = poly_idx * buffer.words_per_poly + limb_id.y * buffer.n;
    if (offset_words >= buffer.words_total)
    {
        return nullptr;
    }
    return buffer.ptr + offset_words;
}

const uint64_t *matrix_limb_ptr_by_id(const GpuMatrix *mat, size_t poly_idx, const dim3 &limb_id)
{
    return matrix_limb_ptr_by_id(const_cast<GpuMatrix *>(mat), poly_idx, limb_id);
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
    if (!state->write_done || !state->write_done_valid)
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
    err = cudaStreamWaitEvent(consumer_stream, state->write_done, 0);
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
    cudaEvent_t consumer_done = nullptr;
    err = cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaEventRecord(consumer_done, consumer_stream);
    if (err != cudaSuccess)
    {
        cudaEventDestroy(consumer_done);
        return set_error(err);
    }
    err = cudaStreamWaitEvent(state->stream, consumer_done, 0);
    cudaError_t destroy_err = cudaEventDestroy(consumer_done);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    if (destroy_err != cudaSuccess)
    {
        return set_error(destroy_err);
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
    if (!stream || !state->write_done)
    {
        return set_error("invalid stream or event in matrix_record_limb_write");
    }
    cudaError_t err = cudaSetDevice(state->device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaEventRecord(state->write_done, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    state->write_done_valid = true;
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
    if (aux_owner && aux_limb_id && matrix_aux_slice_for_limb(aux_owner, *aux_limb_id, bytes, out_ptr))
    {
        *out_shared = true;
        return 0;
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
