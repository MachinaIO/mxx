using gpu_chacha::DeviceChaChaRng;
using gpu_chacha::GpuRngSeed;
using gpu_chacha::rng_init;
using gpu_chacha::rng_next_u64;

__device__ __forceinline__ double uniform_open01(DeviceChaChaRng &rng)
{
    constexpr double kScale = 1.0 / 9007199254740992.0; // 2^53
    double u = static_cast<double>(rng_next_u64(rng) >> 11U) * kScale;
    if (u <= 0.0)
    {
        u = kScale;
    }
    else if (u >= 1.0)
    {
        u = 1.0 - kScale;
    }
    return u;
}

__device__ __forceinline__ double sample_standard_normal(DeviceChaChaRng &rng)
{
    double u1 = uniform_open01(rng);
    double u2 = uniform_open01(rng);
    double r = sqrt(-2.0 * log(u1));
    double theta = kTwoPi * u2;
    return r * cos(theta);
}

__device__ __forceinline__ bool karney_algorithm_h(DeviceChaChaRng &rng)
{
    double h_a = uniform_open01(rng);
    if (!(h_a < 0.5))
    {
        return true;
    }
    for (;;)
    {
        double h_b = uniform_open01(rng);
        if (!(h_b < h_a))
        {
            return false;
        }
        h_a = uniform_open01(rng);
        if (!(h_a < h_b))
        {
            return true;
        }
    }
}

__device__ __forceinline__ int32_t karney_algorithm_g(DeviceChaChaRng &rng)
{
    int32_t n = 0;
    while (karney_algorithm_h(rng))
    {
        ++n;
        if (n > 1024)
        {
            break;
        }
    }
    return n;
}

__device__ __forceinline__ bool karney_algorithm_p(DeviceChaChaRng &rng, int32_t n)
{
    while (n-- && karney_algorithm_h(rng))
    {
    }
    return n < 0;
}

__device__ __forceinline__ bool karney_algorithm_b(DeviceChaChaRng &rng, int32_t k, double x)
{
    double y = x;
    int32_t n = 0;
    double m = static_cast<double>(2 * k + 2);
    for (;; ++n)
    {
        double z = uniform_open01(rng);
        if (!(z < y))
        {
            break;
        }
        double r = uniform_open01(rng);
        if (!(r < (2.0 * static_cast<double>(k) + x) / m))
        {
            break;
        }
        y = z;
        if (n > 4096)
        {
            break;
        }
    }
    return (n % 2) == 0;
}

__device__ __forceinline__ int64_t sample_integer_karney(DeviceChaChaRng &rng, double mean, double stddev)
{
    if (!(stddev > 0.0) || !isfinite(mean) || !isfinite(stddev))
    {
        return static_cast<int64_t>(llround(mean));
    }

    int64_t ceil_std = static_cast<int64_t>(ceil(stddev));
    if (ceil_std <= 0)
    {
        return static_cast<int64_t>(llround(mean));
    }

    for (int iter = 0; iter < 1 << 16; ++iter)
    {
        int32_t k = karney_algorithm_g(rng);
        if (!karney_algorithm_p(rng, k * (k - 1)))
        {
            continue;
        }

        int64_t s = (rng_next_u64(rng) & 1ULL) ? 1 : -1;
        double di0 = stddev * static_cast<double>(k) + static_cast<double>(s) * mean;
        int64_t i0 = static_cast<int64_t>(ceil(di0));
        double x0 = (static_cast<double>(i0) - di0) / stddev;
        int64_t j = static_cast<int64_t>(rng_next_u64(rng) % static_cast<uint64_t>(ceil_std));
        double x = x0 + static_cast<double>(j) / stddev;

        if (!(x < 1.0) || (x == 0.0 && s < 0 && k == 0))
        {
            continue;
        }

        int32_t h = k + 1;
        while (h-- > 0 && karney_algorithm_b(rng, k, x))
        {
        }
        if (h >= 0)
        {
            continue;
        }

        return s * (i0 + j);
    }

    // Fallback in case the rejection loop takes too long.
    return static_cast<int64_t>(llround(mean + stddev * sample_standard_normal(rng)));
}

__device__ __forceinline__ void get_base_digits_u64(
    uint64_t value,
    uint64_t base,
    uint32_t digits,
    int64_t *out_digits)
{
    for (uint32_t i = 0; i < digits; ++i)
    {
        out_digits[i] = static_cast<int64_t>(value % base);
        value /= base;
    }
}

__device__ __forceinline__ uint64_t signed_mod_i64(int64_t value, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    if (value >= 0)
    {
        const uint64_t magnitude = static_cast<uint64_t>(value);
        return magnitude < modulus ? magnitude : magnitude % modulus;
    }
    uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
    uint64_t rem = magnitude < modulus ? magnitude : magnitude % modulus;
    return rem == 0 ? 0 : (modulus - rem);
}

__device__ __forceinline__ uint64_t abs_i64(int64_t value)
{
    if (value >= 0)
    {
        return static_cast<uint64_t>(value);
    }
    return static_cast<uint64_t>(-(value + 1)) + 1;
}

__device__ __forceinline__ uint64_t centered_sample_abs_i64(
    int64_t value,
    uint64_t coefficient_modulus)
{
    if (coefficient_modulus == 0)
    {
        // A zero sentinel means the full CRT modulus does not fit in u64. Since the sampler
        // produces an i64, reduction cannot change its centered representative in that case.
        return abs_i64(value);
    }
    const uint64_t residue = signed_mod_i64(value, coefficient_modulus);
    const uint64_t negative_magnitude = coefficient_modulus - residue;
    return residue < negative_magnitude ? residue : negative_magnitude;
}

__device__ __forceinline__ uint64_t sample_uniform_mod(
    DeviceChaChaRng &rng,
    uint64_t modulus,
    uint64_t rejection_threshold)
{
    if (modulus == 0)
    {
        return 0;
    }
    for (;;)
    {
        const uint64_t random = rng_next_u64(rng);
        const uint64_t low = random * modulus;
        if (low >= rejection_threshold)
        {
            return __umul64hi(random, modulus);
        }
    }
}

__device__ __forceinline__ int64_t centered_residue_i64(uint64_t value, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    uint64_t reduced = value % modulus;
    uint64_t half = modulus >> 1;
    if (reduced <= half)
    {
        return static_cast<int64_t>(reduced);
    }
    uint64_t neg = modulus - reduced;
    return -static_cast<int64_t>(neg);
}

// Sampling coordinates describe the logical full matrix, independently of
// the retained owner's physical rectangle. Bounded metadata stays in arguments.
struct MatrixSampleDescriptors {
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
    uint32_t indices[GPU_RUNTIME_MAX_LIMBS];
    uint64_t moduli[GPU_RUNTIME_MAX_LIMBS];
    size_t offset, pitch;
};
static_assert(sizeof(MatrixSampleDescriptors) + 128 < 4096, "bounded sampling arguments");

__global__ void matrix_sample_distribution_multi_limb_kernel(
    MatrixSampleDescriptors layout,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    const size_t limb_idx = blockIdx.z;
    const auto descriptor = layout.descriptors[layout.indices[limb_idx]];
    const uint64_t modulus = layout.moduli[limb_idx];
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    const size_t total_chunks = poly_count * chunks_per_poly;
    for (size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         chunk_idx < total_chunks;
         chunk_idx += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const size_t local_poly_idx = chunk_idx / chunks_per_poly;
        const size_t coeff_start =
            (chunk_idx - local_poly_idx * chunks_per_poly) * kSamplesPerThread;
        const size_t row_idx = local_poly_idx / local_ncol;
        const size_t local_col_idx = local_poly_idx - row_idx * local_ncol;
        const size_t global_poly_idx = row_idx * full_ncol + (col_offset + local_col_idx);

        const uint64_t domain = dist_type == GPU_MATRIX_DIST_UNIFORM ?
            0x6f70656e66686531ULL :
            (dist_type == GPU_MATRIX_DIST_GAUSS ? 0x6f70656e66686532ULL :
             (dist_type == GPU_MATRIX_DIST_BIT ? 0x6f70656e66686533ULL :
                                                0x6f70656e66686534ULL));
        const uint64_t limb_domain =
            dist_type == GPU_MATRIX_DIST_UNIFORM ? static_cast<uint64_t>(limb_idx + 1) : 0;
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(global_poly_idx + 1),
            static_cast<uint64_t>(coeff_start + 1),
            limb_domain,
            domain);
        const uint64_t uniform_rejection_threshold =
            dist_type == GPU_MATRIX_DIST_UNIFORM && modulus != 0 ?
            static_cast<uint64_t>(-modulus) % modulus : 0;

        for (size_t lane = 0; lane < kSamplesPerThread; ++lane)
        {
            const size_t coeff_idx = coeff_start + lane;
            if (coeff_idx >= n) break;
            uint64_t sample = 0;
            if (dist_type == GPU_MATRIX_DIST_UNIFORM)
            {
                sample = sample_uniform_mod(rng, modulus, uniform_rejection_threshold);
            }
            else if (dist_type == GPU_MATRIX_DIST_GAUSS)
            {
                int64_t z;
                do
                {
                    z = sample_integer_karney(rng, 0.0, sigma);
                } while (centered_sample_abs_i64(z, coefficient_modulus) > max_coefficient_bound);
                sample = signed_mod_i64(z, modulus);
            }
            else if (dist_type == GPU_MATRIX_DIST_BIT)
            {
                sample = rng_next_u64(rng) & 1ULL;
            }
            else if (dist_type == GPU_MATRIX_DIST_TERNARY)
            {
                const uint64_t pick = rng_next_u64(rng) % 3ULL;
                const int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
                sample = signed_mod_i64(z, modulus);
            }

            matrix_store_limb_u64(
                descriptor.base,
                layout.offset + row_idx * layout.pitch + local_col_idx,
                coeff_idx,
                descriptor.stride,
                descriptor.width,
                sample);
        }
    }
}

static int gpu_matrix_sample_distribution_impl(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range)
{
    if (!out || !out->ctx || out->ctx->N < 2 || out->level < 0 ||
        static_cast<size_t>(out->level) >= GPU_RUNTIME_MAX_LIMBS)
        return set_error("invalid matrix in gpu_matrix_sample_distribution");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("invalid dist_type in gpu_matrix_sample_distribution");
    if (dist_type == GPU_MATRIX_DIST_GAUSS && (!(sigma > 0.0) || !std::isfinite(sigma)))
        return set_error("sigma must be finite and positive in gpu_matrix_sample_distribution");
    const GpuMatrixRange rectangle = range ? *range : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (rectangle.row_start > rectangle.row_end || rectangle.row_end > out->rows ||
        rectangle.column_start > rectangle.column_end || rectangle.column_end > out->cols)
        return set_error("invalid destination rectangle in gpu_matrix_sample_distribution");
    const size_t rows = rectangle.row_end - rectangle.row_start;
    const size_t columns = rectangle.column_end - rectangle.column_start;
    const size_t n = static_cast<size_t>(out->ctx->N);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (col_offset > full_ncol || columns > full_ncol - col_offset ||
        (out->rows && out->cols > limit / out->rows) ||
        (rows && full_ncol > limit / rows) || (rows && columns > limit / rows / n))
        return set_error("invalid logical shape in gpu_matrix_sample_distribution");
    if (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid output format in gpu_matrix_sample_distribution");
    const size_t count = rows * columns;
    if (!count) return 0;
    const size_t limb_count = static_cast<size_t>(out->level) + 1;
    if (out->ctx->limb_gpu_ids.size() < limb_count || out->ctx->moduli.size() < limb_count)
        return set_error("invalid limb count in gpu_matrix_sample_distribution");
    MatrixSampleDescriptors layout{};
    layout.offset = rectangle.row_start * out->cols + rectangle.column_start;
    layout.pitch = out->cols;
    int device = -1;
    cudaStream_t stream = nullptr;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        int status = matrix_limb_device(out, id, &limb_device);
        if (status != 0) return status;
        if (id.x >= out->shared_limb_buffers.size())
            return set_error("invalid sampling partition");
        const auto &buffer = out->shared_limb_buffers[id.x];
        size_t stride = 0;
        uint8_t width = 0;
        if (!buffer.device_descriptors || id.y >= buffer.limb_count ||
            !matrix_limb_ptr_by_id(out, 0, id) ||
            !matrix_limb_metadata_by_id(out, id, &stride, &width) ||
            (width == 0 || width > 8) || stride < n * width || !out->ctx->moduli[limb])
            return set_error("invalid sampling descriptor");
        if (limb == 0) {
            device = limb_device;
            layout.descriptors = buffer.device_descriptors;
            status = matrix_limb_stream(out, id, &stream);
            if (status != 0) return status;
        } else if (limb_device != device || layout.descriptors != buffer.device_descriptors) {
            return set_error("sampling requires one device and descriptor partition");
        }
        layout.indices[limb] = id.y;
        layout.moduli[limb] = out->ctx->moduli[limb];
    }
    if (device < 0 || !stream) return set_error("invalid sampling stream");
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(out, device, stream, true);
    if (status != 0) return status;
    constexpr size_t threads = 256;
    const size_t chunks = count * ((n + 3) / 4);
    const size_t blocks = std::min<size_t>(65535, chunks / threads + (chunks % threads != 0));
    matrix_sample_distribution_multi_limb_kernel<<<dim3(blocks, 1, limb_count), threads, 0, stream>>>(
        layout, count, columns, full_ncol, col_offset, n,
        dist_type, sigma, max_coefficient_bound, coefficient_modulus, seed);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = matrix_record_all_limb_writes(out, stream);
    if (status != 0) return status;
    // Existing evaluation entries outside the rectangle must not be transformed.
    // The same optimized NTT kernels consume only the newly sampled coefficients.
    if (out->format == GPU_POLY_FORMAT_EVAL)
        return run_matrix_transform_u64<true>(out, range);
    return 0;
}


extern "C" int gpu_matrix_sample_distribution(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    return gpu_matrix_sample_distribution_impl(
        out,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed,
        out ? out->cols : 0,
        0,
        nullptr);
}

extern "C" int gpu_matrix_sample_distribution_columns(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range)
{
    return gpu_matrix_sample_distribution_impl(
        out,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed,
        full_ncol,
        col_offset,
        range);
}
