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

__device__ __forceinline__ void matrix_sample_distribution_multi_limb_kernel_body(
    uint8_t *dst_base,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    const size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_chunks = poly_count * chunks_per_poly;
    if (chunk_idx >= total_chunks)
    {
        return;
    }
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
            dst_base,
            local_poly_idx,
            coeff_idx,
            dst_stride_bytes,
            dst_coeff_bytes,
            sample);
    }
}

__global__ void matrix_sample_distribution_multi_limb_kernel(
    uint8_t *dst_base,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    matrix_sample_distribution_multi_limb_kernel_body(
        dst_base, poly_count, local_ncol, full_ncol, col_offset, n,
        dst_stride_bytes, dst_coeff_bytes, modulus, limb_idx, dist_type,
        sigma, max_coefficient_bound, coefficient_modulus, seed);
}

__global__ void matrix_sample_distribution_multi_limb_device_seed_kernel(
    uint8_t *dst_base,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    const GpuRngSeed *device_seed)
{
    if (device_seed == nullptr)
        return;
    matrix_sample_distribution_multi_limb_kernel_body(
        dst_base, poly_count, local_ncol, full_ncol, col_offset, n,
        dst_stride_bytes, dst_coeff_bytes, modulus, limb_idx, dist_type,
        sigma, max_coefficient_bound, coefficient_modulus, *device_seed);
}

int launch_sample_distribution_multi_limb_kernel(
    GpuContext *,
    uint8_t *dst_base,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    uint32_t,
    cudaStream_t stream)
{
    if (!dst_base)
    {
        return set_error("null output base pointer in matrix_sample_distribution_multi_limb_kernel");
    }
    if (poly_count == 0 || n == 0)
    {
        return 0;
    }

    const int threads = 256;
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    const size_t total_chunks = poly_count * chunks_per_poly;
    const int blocks = static_cast<int>((total_chunks + threads - 1) / threads);
    matrix_sample_distribution_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        dst_base,
        poly_count,
        local_ncol,
        full_ncol,
        col_offset,
        n,
        dst_stride_bytes,
        dst_coeff_bytes,
        modulus,
        limb_idx,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    // The sampled output is allocated inside the captured fixed operation and
    // remains graph-owned.  The replay destination owner is populated by the
    // capture adapter's post-operation copy, so its pointer is not a resident
    // graph binding here.
    return 0;
}

static int gpu_matrix_sample_distribution_impl(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset)
{
    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!out)
    {
        return set_error("invalid gpu_matrix_sample_distribution arguments");
    }
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
    {
        return set_error("invalid dist_type in gpu_matrix_sample_distribution");
    }
    if (dist_type == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0))
    {
        return set_error("sigma must be positive in gpu_matrix_sample_distribution");
    }
    if (col_offset > full_ncol || out->cols > full_ncol - col_offset)
    {
        return set_error("column range out of bounds in gpu_matrix_sample_distribution");
    }
    const GpuPolyFormat requested_format = out->format;
    if (requested_format != GPU_POLY_FORMAT_COEFF && requested_format != GPU_POLY_FORMAT_EVAL)
    {
        return set_error("invalid output format in gpu_matrix_sample_distribution");
    }

    const size_t count = out->rows * out->cols;
    if (count == 0)
    {
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_distribution");
    }
    if (out->ctx->moduli.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_distribution");
    }

    auto &limb_map = out->ctx->limb_gpu_ids;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_distribution");
    }
    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;
        status = matrix_limb_device(out, limb_id, &limb_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(out, limb_id, &limb_stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_device < 0 || !limb_stream)
        {
            return set_error("invalid limb metadata in gpu_matrix_sample_distribution");
        }
        uint8_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            return set_error("null output limb base pointer in gpu_matrix_sample_distribution");
        }
        size_t dst_stride_bytes = 0;
        uint8_t dst_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(out, limb_id, &dst_stride_bytes, &dst_coeff_bytes))
        {
            return set_error("invalid output limb metadata in gpu_matrix_sample_distribution");
        }
        cudaError_t err = cudaSetDevice(limb_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = launch_sample_distribution_multi_limb_kernel(
            out->ctx,
            dst_base,
            count,
            out->cols,
            full_ncol,
            col_offset,
            static_cast<size_t>(out->ctx->N),
            dst_stride_bytes,
            dst_coeff_bytes,
            out->ctx->moduli[static_cast<size_t>(limb)],
            static_cast<uint32_t>(limb),
            dist_type,
            sigma,
            max_coefficient_bound,
            coefficient_modulus,
            seed,
            static_cast<uint32_t>(limb),
            limb_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, limb_stream);
        if (status != 0)
        {
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    if (requested_format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_ntt_all(out);
        if (status != 0)
        {
            return status;
        }
    }
    out->format = requested_format;
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
        0);
}

extern "C" int gpu_matrix_sample_distribution_columns(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset)
{
    return gpu_matrix_sample_distribution_impl(
        out,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed,
        full_ncol,
        col_offset);
}

extern "C" int gpu_matrix_sample_distribution_columns_device_seed(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    const GpuRngSeed *device_seed,
    size_t full_ncol,
    size_t col_offset,
    cudaStream_t body_stream)
{
    if (!out || !device_seed || !body_stream)
        return set_error("invalid device-seed sampling arguments");
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("invalid dist_type in device-seed sampling");
    if (col_offset > full_ncol || out->cols > full_ncol - col_offset)
        return set_error("column range out of bounds in device-seed sampling");
    if (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid output format in device-seed sampling");
    if (out->level < 0 || out->ctx->moduli.size() < static_cast<size_t>(out->level + 1) ||
        out->ctx->limb_gpu_ids.size() < static_cast<size_t>(out->level + 1))
        return set_error("invalid device-seed sampling level");
    const GpuPolyFormat requested_format = out->format;
    out->format = GPU_POLY_FORMAT_COEFF;
    const size_t count = out->rows * out->cols;
    for (int limb = 0; limb <= out->level; ++limb)
    {
        const dim3 limb_id = out->ctx->limb_gpu_ids[static_cast<size_t>(limb)];
        int device = -1;
        int status = matrix_limb_device(out, limb_id, &device);
        if (status != 0) return status;
        if (device < 0) return set_error("invalid device-seed limb metadata");
        uint8_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        size_t stride = 0;
        uint8_t coeff_bytes = 0;
        if (!dst || !matrix_limb_metadata_by_id(out, limb_id, &stride, &coeff_bytes))
            return set_error("invalid device-seed output metadata");
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) return set_error(err);
        const size_t chunks = (out->ctx->N + 3) / 4;
        const size_t total = count * chunks;
        if (total != 0)
        {
            matrix_sample_distribution_multi_limb_device_seed_kernel<<<
                static_cast<int>((total + 255) / 256), 256, 0, body_stream>>>(
                dst, count, out->cols, full_ncol, col_offset, out->ctx->N,
                stride, coeff_bytes, out->ctx->moduli[static_cast<size_t>(limb)],
                static_cast<uint32_t>(limb), dist_type, sigma, max_coefficient_bound,
                coefficient_modulus, device_seed);
            err = cudaGetLastError();
            if (err != cudaSuccess) return set_error(err);
        }
        status = matrix_record_limb_write(out, limb_id, body_stream);
        if (status != 0) return status;
    }
    if (requested_format == GPU_POLY_FORMAT_EVAL)
        return gpu_matrix_ntt_all(out);
    return 0;
}

namespace
{
    __global__ void raw_matrix_sample_kernel(
        MxxRawMatrixLimb destination, const GpuRngSeed *device_seed,
        uint64_t rows, uint64_t columns, uint64_t row_origin,
        uint64_t column_origin, uint64_t full_columns,
        uint32_t degree, int distribution, double sigma,
        uint64_t max_coefficient_bound, uint64_t coefficient_modulus,
        uint64_t chunk_offset, uint64_t sample_domain)
    {
        if (!device_seed) return;
        constexpr uint64_t samples_per_thread = 4;
        const uint64_t chunks_per_poly = (degree + samples_per_thread - 1) / samples_per_thread;
        const uint64_t chunk = chunk_offset +
            static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const uint64_t poly = chunk / chunks_per_poly;
        if (poly >= rows * columns) return;
        const uint64_t coefficient_start =
            (chunk - poly * chunks_per_poly) * samples_per_thread;
        const uint64_t row = poly / columns;
        const uint64_t column = poly - row * columns;
        const uint64_t global_poly =
            (row_origin + row) * full_columns + column_origin + column;
        const uint64_t domain = distribution == GPU_MATRIX_DIST_UNIFORM ?
            0x6f70656e66686531ULL :
            (distribution == GPU_MATRIX_DIST_GAUSS ? 0x6f70656e66686532ULL :
             (distribution == GPU_MATRIX_DIST_BIT ? 0x6f70656e66686533ULL :
                                                0x6f70656e66686534ULL));
        const uint64_t limb_domain = distribution == GPU_MATRIX_DIST_UNIFORM ?
            static_cast<uint64_t>(destination.crt_limb_index + 1) : 0;
        DeviceChaChaRng rng;
        rng_init(rng, *device_seed, global_poly + 1,
            coefficient_start + 1, limb_domain, domain ^ sample_domain);
        const uint64_t modulus = destination.modulus;
        const uint64_t rejection_threshold = distribution == GPU_MATRIX_DIST_UNIFORM ?
            static_cast<uint64_t>(-modulus) % modulus : 0;
        for (uint64_t lane = 0; lane < samples_per_thread; ++lane)
        {
            const uint64_t coefficient = coefficient_start + lane;
            if (coefficient >= degree) break;
            uint64_t sample = 0;
            if (distribution == GPU_MATRIX_DIST_UNIFORM)
                sample = sample_uniform_mod(rng, modulus, rejection_threshold);
            else if (distribution == GPU_MATRIX_DIST_GAUSS)
            {
                int64_t signed_sample;
                do
                {
                    signed_sample = sample_integer_karney(rng, 0.0, sigma);
                } while (centered_sample_abs_i64(signed_sample, coefficient_modulus) >
                    max_coefficient_bound);
                sample = signed_mod_i64(signed_sample, modulus);
            }
            else if (distribution == GPU_MATRIX_DIST_BIT)
                sample = rng_next_u64(rng) & 1ULL;
            else
            {
                const uint64_t pick = rng_next_u64(rng) % 3ULL;
                sample = signed_mod_i64(pick == 0 ? 0 : (pick == 1 ? 1 : -1), modulus);
            }
            raw_matrix_store(destination, poly, coefficient, columns, sample);
        }
    }
}

extern "C" int gpu_raw_matrix_sample(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *destination, int distribution, double sigma,
    uint64_t max_coefficient_bound, uint64_t coefficient_modulus,
    const void *device_seed, uint64_t full_columns, uint64_t sample_domain,
    uint32_t destination_binding_base, uint32_t seed_binding)
{
    if (validate_raw_view(ctx, destination, stream_raw) != 0 || !device_seed ||
        distribution < GPU_MATRIX_DIST_UNIFORM ||
        distribution > GPU_MATRIX_DIST_TERNARY ||
        (distribution == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0)) ||
        destination->column_origin > full_columns ||
        destination->columns > full_columns - destination->column_origin ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix sampler arguments");
    if (cudaSetDevice(destination->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint64_t chunks_per_poly = (destination->degree + 3) / 4;
    if (destination->row_origin > UINT64_MAX - (destination->rows - 1) ||
        destination->column_origin > UINT64_MAX - (destination->columns - 1) ||
        destination->row_origin + destination->rows - 1 >
            (UINT64_MAX - (destination->column_origin + destination->columns - 1)) /
                full_columns ||
        destination->rows > UINT64_MAX / destination->columns ||
        destination->rows * destination->columns > UINT64_MAX / chunks_per_poly)
        return set_error("raw sampler chunk count overflow");
    const uint64_t chunks = destination->rows * destination->columns * chunks_per_poly;
    constexpr uint64_t max_chunks = 65535ULL * 256ULL;
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                seed_binding, 0},
        };
        for (uint64_t offset = 0; offset < chunks; offset += max_chunks)
        {
            const uint64_t count = std::min(max_chunks, chunks - offset);
            const dim3 grid(static_cast<uint32_t>((count + 255) / 256));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_sample_kernel, grid, dim3(256), 0, patches, 2,
                destination->limbs[limb], static_cast<const GpuRngSeed *>(device_seed),
                destination->rows, destination->columns,
                destination->row_origin, destination->column_origin,
                full_columns, destination->degree, distribution, sigma,
                max_coefficient_bound, coefficient_modulus, offset, sample_domain);
            if (status != 0) return status;
        }
    }
    return 0;
}
