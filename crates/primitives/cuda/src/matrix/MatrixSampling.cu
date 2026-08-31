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

// The batch kernel deliberately uses the same RNG coordinates as the scalar
// kernel.  The output ordinal selects only the seed; it is not part of the
// ChaCha stream/counter domain, so batching cannot change any output's bytes.
__global__ void matrix_sample_distribution_multi_limb_batch_kernel(
    uint8_t *packed_dst_base,
    size_t packed_output_stride_bytes,
    const GpuRngSeed *seeds,
    size_t output_count,
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
    uint64_t coefficient_modulus)
{
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    const size_t output_idx = static_cast<size_t>(blockIdx.z);
    if (output_idx >= output_count)
    {
        return;
    }
    uint8_t *dst_base = packed_dst_base + output_idx * packed_output_stride_bytes;
    const size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (chunk_idx >= chunks_per_poly || blockIdx.y >= poly_count)
    {
        return;
    }
    const size_t local_poly_idx = static_cast<size_t>(blockIdx.y);
    const size_t coeff_start = chunk_idx * kSamplesPerThread;
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
        seeds[output_idx],
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

int launch_sample_distribution_multi_limb_kernel(
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
    cudaStream_t stream,
    const GpuMatrix *,
    const dim3 *)
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
    return 0;
}

static int launch_sample_distribution_multi_limb_batch_kernel_device(
    uint8_t *packed_dst_base,
    size_t packed_output_stride_bytes,
    const GpuRngSeed *device_seeds,
    size_t output_count,
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
    cudaStream_t stream)
{
    if (!packed_dst_base || !device_seeds || output_count == 0 || poly_count == 0 || n == 0)
    {
        return set_error("invalid empty or mismatched sampling batch");
    }
    if (limb_idx >= GPU_RUNTIME_MAX_LIMBS)
    {
        return set_error("invalid limb index in matrix sampling batch");
    }
    if (output_count > 65535)
    {
        return set_error("sampling batch exceeds CUDA grid dimensions");
    }
    if (poly_count > 65535)
    {
        return set_error("sampling batch exceeds CUDA grid dimensions");
    }
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    if (poly_count > static_cast<size_t>(-1) / chunks_per_poly)
    {
        return set_error("sampling batch size overflow");
    }
    const int threads = 256;
    matrix_sample_distribution_multi_limb_batch_kernel<<<
        dim3(
            static_cast<unsigned int>((chunks_per_poly + threads - 1) / threads),
            static_cast<unsigned int>(poly_count),
            static_cast<unsigned int>(output_count)),
        threads,
        0,
        stream>>>(
        packed_dst_base,
        packed_output_stride_bytes,
        device_seeds,
        output_count,
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
        coefficient_modulus);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
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

    const size_t count = out->rows * out->cols;
    if (count == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
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
            limb_stream,
            out,
            &limb_id);
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
    status = gpu_matrix_ntt_all(out);
    if (status != 0)
    {
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_sample_distribution_batch(
    GpuMatrix *const *outputs,
    size_t output_count,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    const GpuRngSeed *seeds)
{
    if (!outputs || !seeds || output_count == 0)
    {
        return output_count == 0 ? 0 : set_error("invalid sampling batch arguments");
    }
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
    {
        return set_error("invalid dist_type in gpu_matrix_sample_distribution_batch");
    }
    if (dist_type == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0))
    {
        return set_error("sigma must be positive in gpu_matrix_sample_distribution_batch");
    }

    GpuMatrix *first = outputs[0];
    if (!first || !first->ctx || first->level < 0)
    {
        return set_error("invalid first output in gpu_matrix_sample_distribution_batch");
    }
    const size_t limb_count = static_cast<size_t>(first->level + 1);
    if (limb_count == 0 || first->ctx->moduli.size() < limb_count ||
        first->ctx->limb_gpu_ids.size() < limb_count)
    {
        return set_error("invalid context in gpu_matrix_sample_distribution_batch");
    }
    const size_t poly_count = matrix_poly_count(first);
    if (poly_count > 65535 || output_count > 65535)
    {
        return set_error("sampling batch exceeds CUDA grid dimensions");
    }
    const size_t ring_dimension = static_cast<size_t>(first->ctx->N);
    if (ring_dimension == 0 || first->rows != 0 && poly_count / first->rows != first->cols)
    {
        return set_error("invalid output shape in gpu_matrix_sample_distribution_batch");
    }
    for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
    {
        GpuMatrix *output = outputs[output_idx];
        if (!output || output->ctx != first->ctx || output->rows != first->rows ||
            output->cols != first->cols || output->level != first->level ||
            output->format != GPU_POLY_FORMAT_EVAL)
        {
            return set_error("gpu_matrix_sample_distribution_batch requires homogeneous outputs");
        }
    }
    if (poly_count == 0)
    {
        for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
        {
            outputs[output_idx]->format = GPU_POLY_FORMAT_EVAL;
        }
        return 0;
    }

    // Upload the immutable seed table once for each device.  Each device's
    // limb streams wait on the upload event, and the device table is released
    // only after a completion event from every sampling launch on that device.
    struct SeedUpload
    {
        int device = -1;
        GpuRngSeed *host_seeds = nullptr;
        GpuRngSeed *device_seeds = nullptr;
        cudaEvent_t ready = nullptr;
        cudaStream_t release_stream = nullptr;
        std::vector<cudaEvent_t> cleanup_events;
        bool device_allocation_started = false;
        bool ready_recorded = false;
        bool cleanup_safe = true;
    };
    const size_t seed_bytes = output_count * sizeof(GpuRngSeed);
    std::vector<SeedUpload> uploads;
    auto cleanup_uploads = [&]() {
        for (auto &upload : uploads)
        {
            bool dependencies_queued = upload.cleanup_safe && upload.release_stream != nullptr &&
                (!upload.device_allocation_started || upload.ready_recorded);
            if (dependencies_queued)
            {
                if (cudaSetDevice(upload.device) != cudaSuccess)
                {
                    dependencies_queued = false;
                }
            }
            if (dependencies_queued && upload.ready)
            {
                dependencies_queued = cudaStreamWaitEvent(
                    upload.release_stream, upload.ready, 0) == cudaSuccess;
            }
            if (dependencies_queued)
            {
                for (const cudaEvent_t cleanup_event : upload.cleanup_events)
                {
                    if (cudaStreamWaitEvent(upload.release_stream, cleanup_event, 0) != cudaSuccess)
                    {
                        dependencies_queued = false;
                        break;
                    }
                }
            }
            if (dependencies_queued && upload.device_seeds)
            {
                const cudaError_t device_status = cudaFreeAsync(
                    upload.device_seeds, upload.release_stream);
                dependencies_queued = device_status == cudaSuccess;
                if (dependencies_queued)
                {
                    // The free is now owned by the release stream.  Clearing
                    // the host-side handle makes cleanup idempotent even when
                    // the subsequent pinned-host callback fails.
                    upload.device_seeds = nullptr;
                }
            }
            if (dependencies_queued && upload.host_seeds)
            {
                void *host_seed_pointer = upload.host_seeds;
                const int host_status = gpu_defer_pinned_frees(
                    first->ctx,
                    upload.device,
                    upload.release_stream,
                    &host_seed_pointer,
                    1);
                // The reclaimer owns the pointer on both success and
                // fail-closed failure (the latter intentionally leaks it).
                upload.host_seeds = nullptr;
                if (host_status != 0)
                {
                    dependencies_queued = false;
                }
            }
            if (upload.ready)
            {
                cudaEventDestroy(upload.ready);
                upload.ready = nullptr;
            }
            for (cudaEvent_t &cleanup_event : upload.cleanup_events)
            {
                if (cleanup_event)
                {
                    cudaEventDestroy(cleanup_event);
                    cleanup_event = nullptr;
                }
            }
            // Never synchronously free a buffer that may have been touched by
            // an asynchronous operation.  An impossible cleanup failure is
            // retained for the process lifetime rather than causing UAF.
            if (dependencies_queued)
            {
                upload.device_seeds = nullptr;
            }
        }
    };

    int status = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = first->ctx->limb_gpu_ids[limb];
        int device = -1;
        cudaStream_t stream = nullptr;
        status = matrix_limb_device(first, limb_id, &device);
        if (status == 0) status = matrix_limb_stream(first, limb_id, &stream);
        if (status != 0 || device < 0 || !stream)
        {
            const int error = status != 0 ? status : set_error("invalid batch limb metadata");
            cleanup_uploads();
            return error;
        }
        SeedUpload *upload = nullptr;
        for (auto &candidate : uploads)
        {
            if (candidate.device == device)
            {
                upload = &candidate;
                break;
            }
        }
        if (!upload)
        {
            uploads.push_back(SeedUpload{});
            upload = &uploads.back();
            upload->device = device;
            if (limb_id.x >= first->ctx->release_streams_by_partition.size())
            {
                cleanup_uploads();
                return set_error("missing batch seed release stream");
            }
            upload->release_stream = first->ctx->release_streams_by_partition[limb_id.x];
            if (!upload->release_stream)
            {
                cleanup_uploads();
                return set_error("null batch seed release stream");
            }
            cudaError_t error = cudaSetDevice(device);
            if (error == cudaSuccess)
            {
                error = cudaHostAlloc(
                    reinterpret_cast<void **>(&upload->host_seeds),
                    seed_bytes,
                    cudaHostAllocPortable);
            }
            if (error == cudaSuccess)
            {
                std::memcpy(upload->host_seeds, seeds, seed_bytes);
            }
            if (error == cudaSuccess)
            {
                error = cudaMallocAsync(
                    reinterpret_cast<void **>(&upload->device_seeds), seed_bytes, stream);
                if (upload->device_seeds)
                {
                    upload->device_allocation_started = true;
                }
            }
            if (error == cudaSuccess)
            {
                error = cudaMemcpyAsync(
                    upload->device_seeds,
                    upload->host_seeds,
                    seed_bytes,
                    cudaMemcpyHostToDevice,
                    stream);
            }
            if (error == cudaSuccess)
            {
                error = cudaEventCreateWithFlags(&upload->ready, cudaEventDisableTiming);
            }
            if (error == cudaSuccess)
            {
                error = cudaEventRecord(upload->ready, stream);
                if (error == cudaSuccess)
                {
                    upload->ready_recorded = true;
                }
            }
            if (error != cudaSuccess)
            {
                // Once the device allocation or H2D transfer has started,
                // an unrecorded readiness event cannot safely order a free on
                // the release stream.  Leak the buffers on this exceptional
                // path instead of risking an early free.
                if (upload->device_allocation_started && !upload->ready_recorded)
                {
                    upload->cleanup_safe = false;
                }
                cleanup_uploads();
                return set_error(error);
            }
        }
        else if (upload->ready)
        {
            cudaError_t error = cudaSetDevice(device);
            if (error == cudaSuccess)
            {
                error = cudaStreamWaitEvent(stream, upload->ready, 0);
            }
            if (error != cudaSuccess)
            {
                cleanup_uploads();
                return set_error(error);
            }
        }
        size_t stride = 0;
        uint8_t coeff_bytes = 0;
        for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
        {
            GpuMatrix *output = outputs[output_idx];
            int output_device = -1;
            size_t output_stride = 0;
            uint8_t output_coeff_bytes = 0;
            if (!matrix_limb_ptr_by_id(output, 0, limb_id) ||
                matrix_limb_device(output, limb_id, &output_device) != 0 ||
                !matrix_limb_metadata_by_id(
                    output, limb_id, &output_stride, &output_coeff_bytes) ||
                output_device != device)
            {
                cleanup_uploads();
                return set_error("invalid output limb in gpu_matrix_sample_distribution_batch");
            }
            if (output_idx == 0)
            {
                stride = output_stride;
                coeff_bytes = output_coeff_bytes;
            }
            else if (output_stride != stride || output_coeff_bytes != coeff_bytes)
            {
                cleanup_uploads();
                return set_error("incompatible output limb layout in gpu_matrix_sample_distribution_batch");
            }
            status = matrix_wait_limb_stream(output, limb_id, device, stream);
            if (status != 0)
            {
                cleanup_uploads();
                return status;
            }
        }
        cudaError_t error = cudaSetDevice(device);
        if (error != cudaSuccess)
        {
            cleanup_uploads();
            return set_error(error);
        }
        const auto &buffer = first->shared_limb_buffers[limb_id.x];
        // `buffer.ptr` is the beginning of the whole partition.  The batch
        // kernel, however, writes one exact CRT limb at a time.  Use the
        // limb-aware pointer so a non-zero local limb is not written at the
        // partition base by mistake.
        uint8_t *packed_limb_base = matrix_limb_ptr_by_id(first, 0, limb_id);
        if (!buffer.ptr || !packed_limb_base || !buffer.allocation || buffer.bytes_total == 0 ||
            !buffer.allocation->limb_base || limb_id.y >= buffer.limb_offsets_bytes.size() ||
            limb_id.y >= buffer.limb_coeff_bytes.size())
        {
            cleanup_uploads();
            return set_error("invalid packed output allocation in gpu_matrix_sample_distribution_batch");
        }
        const size_t packed_stride = buffer.bytes_total;
        const size_t packed_limb_offset = buffer.limb_offsets_bytes[limb_id.y];
        const uint8_t packed_coeff_bytes = buffer.limb_coeff_bytes[limb_id.y];
        if (packed_coeff_bytes == 0 || packed_limb_base != buffer.ptr + packed_limb_offset)
        {
            cleanup_uploads();
            return set_error("invalid packed output limb layout in gpu_matrix_sample_distribution_batch");
        }
        for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
        {
            const auto &candidate = outputs[output_idx]->shared_limb_buffers[limb_id.x];
            size_t expected_offset = 0;
            if (output_idx != 0 && packed_stride > static_cast<size_t>(-1) / output_idx)
            {
                cleanup_uploads();
                return set_error("packed output stride overflow in gpu_matrix_sample_distribution_batch");
            }
            expected_offset = output_idx * packed_stride;
            if (limb_id.y >= candidate.limb_offsets_bytes.size() ||
                limb_id.y >= candidate.limb_coeff_bytes.size())
            {
                cleanup_uploads();
                return set_error("missing output limb metadata in gpu_matrix_sample_distribution_batch");
            }
            const uint8_t *candidate_limb_base =
                matrix_limb_ptr_by_id(outputs[output_idx], 0, limb_id);
            if (candidate.allocation != buffer.allocation || candidate.bytes_total != packed_stride ||
                expected_offset > buffer.allocation->limb_bytes ||
                buffer.allocation->limb_bytes - expected_offset < packed_stride ||
                candidate.ptr != buffer.allocation->limb_base + expected_offset ||
                candidate.limb_offsets_bytes[limb_id.y] != packed_limb_offset ||
                candidate.limb_coeff_bytes[limb_id.y] != packed_coeff_bytes ||
                !candidate_limb_base ||
                candidate_limb_base != packed_limb_base + expected_offset)
            {
                cleanup_uploads();
                return set_error("batch outputs are not views of one packed allocation");
            }
        }

        // Both events are created before submitting the sampling kernel.  The
        // cleanup event protects the seed table; the guard event is retained
        // for the exceptional case where a matrix write event cannot be
        // recorded after the kernel has already been submitted.  In
        // particular, event creation must not be the first operation after a
        // launch: a creation failure there would leave cleanup without a
        // dependency that proves when the kernel stopped reading the seeds.
        cudaEvent_t cleanup_event = nullptr;
        cudaError_t event_status =
            cudaEventCreateWithFlags(&cleanup_event, cudaEventDisableTiming);
        cudaEvent_t output_guard_event = nullptr;
        if (event_status == cudaSuccess)
        {
            event_status =
                cudaEventCreateWithFlags(&output_guard_event, cudaEventDisableTiming);
        }
        if (event_status != cudaSuccess)
        {
            if (output_guard_event)
            {
                cudaEventDestroy(output_guard_event);
            }
            if (cleanup_event)
            {
                cudaEventDestroy(cleanup_event);
            }
            cleanup_uploads();
            return set_error(event_status);
        }

        // Every output was checked above to be a view of this same packed
        // allocation.  Locking this object makes the fail-closed transition
        // visible to whichever view is later destroyed and attempts release.
        auto block_packed_release = [&]() {
            if (buffer.allocation)
            {
                std::lock_guard<std::mutex> lock(buffer.allocation->mutex);
                buffer.allocation->release_blocked = true;
            }
        };
        status = launch_sample_distribution_multi_limb_batch_kernel_device(
            packed_limb_base,
            packed_stride,
            upload->device_seeds,
            output_count,
            poly_count,
            first->cols,
            first->cols,
            0,
            ring_dimension,
            stride,
            coeff_bytes,
            first->ctx->moduli[limb],
            static_cast<uint32_t>(limb),
            dist_type,
            sigma,
            max_coefficient_bound,
            coefficient_modulus,
            stream);
        if (status != 0)
        {
            // The launch wrapper reports cudaGetLastError(), which may expose
            // an asynchronous failure from an earlier launch rather than
            // proving that this kernel was never submitted.  Treat the
            // output and seed buffers as potentially in use: retain both
            // allocations instead of allowing cleanup to enqueue an
            // unproven free, and make the shared output allocation
            // fail-closed for every view that owns it.
            block_packed_release();
            upload->cleanup_safe = false;
            cudaEventDestroy(output_guard_event);
            cudaEventDestroy(cleanup_event);
            cleanup_uploads();
            return status;
        }
        // Record the seed-table completion on the exact launch stream before
        // touching output write metadata.  If this recording fails, no
        // dependency can prove that the kernel has stopped reading the seed
        // table, so retain the packed output allocation as well as the seed
        // buffers in the fail-closed state.
        event_status = cudaEventRecord(cleanup_event, stream);
        if (event_status != cudaSuccess)
        {
            block_packed_release();
            upload->cleanup_safe = false;
            cudaEventDestroy(output_guard_event);
            cudaEventDestroy(cleanup_event);
            cleanup_uploads();
            return set_error(event_status);
        }
        // Keep the seed cleanup event in the upload record even if a later
        // output metadata operation fails.  The output guard event is a
        // separate handle because the upload reclaimer owns and destroys its
        // cleanup events, while a matrix write state may outlive this call.
        upload->cleanup_events.push_back(cleanup_event);
        for (size_t output_idx = 0; output_idx < output_count; ++output_idx)
        {
            status = matrix_record_limb_write(outputs[output_idx], limb_id, stream);
            if (status != 0)
            {
                // The kernel has run (or is still running), but this output's
                // ordinary write event was not recorded.  First try to make
                // the independently recorded guard event the allocation's
                // completion dependency.  If that hand-off itself fails,
                // stale write events must never be allowed to free the packed
                // storage, so block release under the shared mutex.
                event_status = cudaEventRecord(output_guard_event, stream);
                if (event_status == cudaSuccess)
                {
                    const int dependency_status = matrix_set_limb_completion_event(
                        outputs[0], limb_id, output_guard_event);
                    if (dependency_status == 0)
                    {
                        output_guard_event = nullptr;
                    }
                    else
                    {
                        block_packed_release();
                    }
                }
                else
                {
                    block_packed_release();
                }
                if (output_guard_event)
                {
                    cudaEventDestroy(output_guard_event);
                }
                cleanup_uploads();
                return status;
            }
            outputs[output_idx]->format = GPU_POLY_FORMAT_COEFF;
        }
        cudaEventDestroy(output_guard_event);
    }
    cleanup_uploads();
    status = gpu_matrix_ntt_contiguous_batch(outputs, output_count);
    if (status != 0) return status;
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
