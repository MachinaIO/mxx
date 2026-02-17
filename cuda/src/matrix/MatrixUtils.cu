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


int matrix_limb_device(const GpuMatrix *mat, const dim3 &limb_id, int *out_device)
{
    if (!mat || !mat->ctx || !out_device)
    {
        return set_error("invalid matrix_limb_device arguments");
    }
    if (limb_id.x >= mat->shared_limb_buffers.size())
    {
        return set_error("invalid partition index in matrix_limb_device");
    }
    const auto &buffer = mat->shared_limb_buffers[limb_id.x];
    if (!buffer.ptr || limb_id.y >= buffer.limb_count)
    {
        return set_error("invalid limb index in matrix_limb_device");
    }
    *out_device = buffer.device;
    return 0;
}

int matrix_limb_stream(const GpuMatrix *mat, const dim3 &limb_id, cudaStream_t *out_stream)
{
    if (!mat || !mat->ctx || !out_stream)
    {
        return set_error("invalid matrix_limb_stream arguments");
    }
    if (limb_id.x >= mat->ctx->ctx->meta.size())
    {
        return set_error("invalid partition index in matrix_limb_stream");
    }
    const auto &meta = mat->ctx->ctx->meta[limb_id.x];
    if (limb_id.y >= meta.size())
    {
        return set_error("invalid limb index in matrix_limb_stream");
    }
    *out_stream = meta[limb_id.y].stream.ptr;
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


int sync_matrix_limb_streams(const GpuMatrix *mat, const char *context)
{
    if (!mat || !mat->ctx || !mat->ctx->ctx || !context)
    {
        return set_error("invalid sync_matrix_limb_streams arguments");
    }
    if (mat->level < 0)
    {
        return set_error(context);
    }
    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(mat->level + 1))
    {
        return set_error(context);
    }

    struct StreamRef
    {
        int device;
        cudaStream_t stream;
    };
    std::vector<StreamRef> streams;
    streams.reserve(static_cast<size_t>(mat->level + 1));
    auto add_stream = [&](int device, cudaStream_t stream) {
        if (!stream)
        {
            return;
        }
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                return;
            }
        }
        streams.push_back(StreamRef{device, stream});
    };

    for (int limb = 0; limb <= mat->level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int device = -1;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return set_error(context);
        }
        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return set_error(context);
        }
        add_stream(device, stream);
    }

    for (const auto &entry : streams)
    {
        cudaError_t err = cudaSetDevice(entry.device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
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
