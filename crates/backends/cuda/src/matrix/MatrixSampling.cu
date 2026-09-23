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

namespace
{
    __global__ void raw_matrix_sample_kernel(
        MxxRawMatrixLimb destination, const GpuRngSeed *device_seed,
        uint64_t rows, uint64_t columns, uint64_t row_origin,
        uint64_t column_origin, uint64_t full_columns,
        uint32_t degree, int distribution, double sigma,
        uint64_t max_coefficient_bound, uint64_t coefficient_modulus,
        int64_t interval_minimum, uint64_t interval_span,
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
                                                 0x6f70656e66686535ULL);
        // Non-uniform draws share one integer stream across CRT limbs so every
        // limb reduces the same signed coefficient.
        const uint64_t limb_domain = distribution == GPU_MATRIX_DIST_UNIFORM ?
            static_cast<uint64_t>(destination.crt_limb_index + 1) : 0;
        DeviceChaChaRng rng;
        rng_init(rng, *device_seed, global_poly + 1,
            coefficient_start + 1, limb_domain, domain ^ sample_domain);
        const uint64_t modulus = destination.modulus;
        const uint64_t rejection_threshold = distribution == GPU_MATRIX_DIST_UNIFORM ?
            static_cast<uint64_t>(-modulus) % modulus :
            (distribution == GPU_MATRIX_DIST_INTERVAL && interval_span != 0 ?
                static_cast<uint64_t>(-interval_span) % interval_span : 0);
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
            else
            {
                // interval_span == 0 encodes the full 2^64 span of [i64::MIN, i64::MAX].
                const uint64_t offset = interval_span == 0 ? rng_next_u64(rng) :
                    sample_uniform_mod(rng, interval_span, rejection_threshold);
                sample = signed_mod_i64(static_cast<int64_t>(
                    static_cast<uint64_t>(interval_minimum) + offset), modulus);
            }
            raw_matrix_store(destination, poly, coefficient, columns, sample);
        }
    }
}

extern "C" int gpu_raw_matrix_sample(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *destination, int distribution, double sigma,
    uint64_t max_coefficient_bound, uint64_t coefficient_modulus,
    int64_t interval_minimum, int64_t interval_maximum,
    const void *device_seed, uint64_t full_columns, uint64_t sample_domain,
    uint32_t destination_binding_base, uint32_t seed_binding)
{
    if (validate_raw_view(ctx, destination, stream_raw) != 0 || !device_seed ||
        (distribution != GPU_MATRIX_DIST_UNIFORM && distribution != GPU_MATRIX_DIST_GAUSS &&
            distribution != GPU_MATRIX_DIST_INTERVAL) ||
        (distribution == GPU_MATRIX_DIST_INTERVAL && interval_minimum > interval_maximum) ||
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
                max_coefficient_bound, coefficient_modulus, interval_minimum,
                static_cast<uint64_t>(interval_maximum) - static_cast<uint64_t>(interval_minimum) + 1,
                offset, sample_domain);
            if (status != 0) return status;
        }
    }
    return 0;
}
