using gpu_chacha::DeviceChaChaRng;
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
        return static_cast<uint64_t>(value) % modulus;
    }
    uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
    uint64_t rem = magnitude % modulus;
    return rem == 0 ? 0 : (modulus - rem);
}

__device__ __forceinline__ uint64_t sample_uniform_mod(DeviceChaChaRng &rng, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    constexpr uint64_t kU64Max = ~uint64_t{0};
    const uint64_t threshold = kU64Max - (kU64Max % modulus);
    for (;;)
    {
        uint64_t x = rng_next_u64(rng);
        if (x < threshold)
        {
            return x % modulus;
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
    uint64_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;

    uint64_t sample = 0;
    if (dist_type == GPU_MATRIX_DIST_UNIFORM)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            static_cast<uint64_t>(limb_idx + 1),
            0x6f70656e66686531ULL);
        sample = sample_uniform_mod(rng, modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_GAUSS)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686532ULL);
        int64_t z = sample_integer_karney(rng, 0.0, sigma);
        sample = signed_mod_i64(z, modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_BIT)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686533ULL);
        sample = (rng_next_u64(rng) & 1ULL) % modulus;
    }
    else if (dist_type == GPU_MATRIX_DIST_TERNARY)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686534ULL);
        uint64_t pick = rng_next_u64(rng) % 3ULL;
        int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
        sample = signed_mod_i64(z, modulus);
    }

    dst_base[poly_idx * dst_stride + coeff_idx] = sample;
}

int launch_sample_distribution_multi_limb_kernel(
    uint64_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride,
    uint64_t modulus,
    uint32_t limb_idx,
    int dist_type,
    double sigma,
    uint64_t seed,
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
    const size_t total = poly_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_sample_distribution_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        dst_base,
        poly_count,
        n,
        dst_stride,
        modulus,
        limb_idx,
        dist_type,
        sigma,
        seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


extern "C" int gpu_matrix_sample_distribution(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed)
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
        uint64_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            return set_error("null output limb base pointer in gpu_matrix_sample_distribution");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid output partition index in gpu_matrix_sample_distribution");
        }
        const size_t dst_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;
        cudaError_t err = cudaSetDevice(limb_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = launch_sample_distribution_multi_limb_kernel(
            dst_base,
            count,
            static_cast<size_t>(out->ctx->N),
            dst_stride,
            out->ctx->moduli[static_cast<size_t>(limb)],
            static_cast<uint32_t>(limb),
            dist_type,
            sigma,
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
    status = gpu_matrix_ntt_all(out, default_batch(out->ctx));
    if (status != 0)
    {
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

namespace
{
    constexpr uint32_t kSampleDecomposeThreads = 256;
    constexpr size_t kSampleDecomposeMaxGridY = 65535;
    constexpr size_t kSampleDecomposeMaxGridZ = 65535;
}

__global__ void matrix_sample_distribution_decompose_all_slots_kernel(
    uint64_t *const *dst_bases,
    const size_t *dst_strides,
    const uint64_t *dst_moduli,
    size_t out_limb_count,
    size_t slot_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint32_t src_bits,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    size_t src_digit_offset_base,
    uint32_t src_limb_idx,
    uint64_t src_modulus,
    int dist_type,
    double sigma,
    uint64_t seed,
    size_t poly_offset,
    size_t slot_offset)
{
    if (!dst_bases || !dst_strides || !dst_moduli)
    {
        return;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0 || digits_per_tower == 0)
    {
        return;
    }
    const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= n)
    {
        return;
    }
    const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
    if (poly_idx >= poly_count)
    {
        return;
    }
    const size_t slot_idx = slot_offset + static_cast<size_t>(blockIdx.z);
    if (slot_idx >= slot_count)
    {
        return;
    }
    const size_t out_limb = slot_idx / static_cast<size_t>(digits_per_tower);
    if (out_limb >= out_limb_count)
    {
        return;
    }
    const uint32_t digit_idx = static_cast<uint32_t>(slot_idx % static_cast<size_t>(digits_per_tower));

    uint64_t *const dst_base = dst_bases[out_limb];
    const size_t dst_stride = dst_strides[out_limb];
    const uint64_t out_modulus = dst_moduli[out_limb];
    if (!dst_base || dst_stride < n)
    {
        return;
    }

    uint64_t residue = 0;
    if (dist_type == GPU_MATRIX_DIST_UNIFORM)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            static_cast<uint64_t>(src_limb_idx + 1),
            0x6f70656e66686531ULL);
        residue = sample_uniform_mod(rng, src_modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_GAUSS)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686532ULL);
        int64_t z = sample_integer_karney(rng, 0.0, sigma);
        residue = signed_mod_i64(z, src_modulus);
    }
    else if (dist_type == GPU_MATRIX_DIST_BIT)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686533ULL);
        residue = (rng_next_u64(rng) & 1ULL) % src_modulus;
    }
    else if (dist_type == GPU_MATRIX_DIST_TERNARY)
    {
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0,
            0x6f70656e66686534ULL);
        uint64_t pick = rng_next_u64(rng) % 3ULL;
        int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
        residue = signed_mod_i64(z, src_modulus);
    }

    const uint32_t shift = digit_idx * base_bits;
    uint64_t mask = 0;
    if (shift < src_bits)
    {
        const uint32_t remaining = src_bits - shift;
        const uint32_t digit_bits = min(base_bits, remaining);
        mask = digit_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << digit_bits) - 1);
    }
    uint64_t digit = shift >= 64 ? 0 : ((residue >> shift) & mask);
    if (out_modulus != 0 && digit >= out_modulus)
    {
        digit %= out_modulus;
    }

    const size_t row = poly_idx / src_cols;
    const size_t col = poly_idx - row * src_cols;
    const size_t out_row = row * log_base_q + src_digit_offset_base + static_cast<size_t>(digit_idx);
    const size_t out_poly_idx = out_row * out_cols + col;
    dst_base[out_poly_idx * dst_stride + coeff_idx] = digit;
}

int launch_sample_distribution_decompose_all_slots_kernel(
    uint64_t *const *dst_bases,
    const size_t *dst_strides,
    const uint64_t *dst_moduli,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint32_t src_bits,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    size_t src_digit_offset_base,
    uint32_t src_limb_idx,
    uint64_t src_modulus,
    int dist_type,
    double sigma,
    uint64_t seed,
    cudaStream_t stream)
{
    if (!dst_bases || !dst_strides || !dst_moduli)
    {
        return set_error("null pointer in matrix_sample_distribution_decompose_all_slots_kernel");
    }
    if (out_limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return set_error("invalid matrix shape in matrix_sample_distribution_decompose_all_slots_kernel");
    }
    if (digits_per_tower == 0)
    {
        return set_error("invalid digit count in matrix_sample_distribution_decompose_all_slots_kernel");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_sample_distribution_decompose_all_slots_kernel");
    }
    if (out_limb_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(digits_per_tower))
    {
        return set_error("slot count overflow in matrix_sample_distribution_decompose_all_slots_kernel");
    }
    const size_t slot_count = out_limb_count * static_cast<size_t>(digits_per_tower);
    if (slot_count == 0)
    {
        return 0;
    }

    const uint32_t blocks_x =
        static_cast<uint32_t>((n + kSampleDecomposeThreads - 1) / kSampleDecomposeThreads);
    for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kSampleDecomposeMaxGridY)
    {
        const size_t poly_chunk = std::min(kSampleDecomposeMaxGridY, poly_count - poly_offset);
        for (size_t slot_offset = 0; slot_offset < slot_count; slot_offset += kSampleDecomposeMaxGridZ)
        {
            const size_t slot_chunk = std::min(kSampleDecomposeMaxGridZ, slot_count - slot_offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(poly_chunk),
                static_cast<uint32_t>(slot_chunk)};
            matrix_sample_distribution_decompose_all_slots_kernel<<<grid, kSampleDecomposeThreads, 0, stream>>>(
                dst_bases,
                dst_strides,
                dst_moduli,
                out_limb_count,
                slot_count,
                poly_count,
                n,
                src_cols,
                out_cols,
                log_base_q,
                src_bits,
                base_bits,
                digits_per_tower,
                src_digit_offset_base,
                src_limb_idx,
                src_modulus,
                dist_type,
                sigma,
                seed,
                poly_offset,
                slot_offset);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
    }
    return 0;
}

static int gpu_matrix_sample_distribution_decompose_base_impl(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed,
    uint32_t base_bits,
    bool small)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_sample_distribution_decompose_base arguments");
    }
    if (base_bits == 0)
    {
        return set_error("base_bits must be non-zero in gpu_matrix_sample_distribution_decompose_base");
    }
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
    {
        return set_error("invalid dist_type in gpu_matrix_sample_distribution_decompose_base");
    }
    if (dist_type == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0))
    {
        return set_error("sigma must be positive in gpu_matrix_sample_distribution_decompose_base");
    }
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_sample_distribution_decompose_base");
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_distribution_decompose_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (out->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_distribution_decompose_base");
    }
    auto &limb_map = out->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_distribution_decompose_base");
    }

    uint32_t crt_bits = 0;
    for (size_t i = 0; i < crt_depth; ++i)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(out->ctx->moduli[i]));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_sample_distribution_decompose_base");
    }
    const uint32_t digits_per_tower =
        static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_sample_distribution_decompose_base");
    }

    const size_t out_log_base_q =
        small ? static_cast<size_t>(digits_per_tower)
              : static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out_log_base_q == 0)
    {
        return set_error("invalid out_log_base_q in gpu_matrix_sample_distribution_decompose_base");
    }
    if (out->rows % out_log_base_q != 0)
    {
        return set_error("output row size mismatch in gpu_matrix_sample_distribution_decompose_base");
    }
    const size_t src_rows = out->rows / out_log_base_q;
    const size_t src_cols = out->cols;
    if (src_rows != 0 && src_cols > std::numeric_limits<size_t>::max() / src_rows)
    {
        return set_error("source matrix size overflow in gpu_matrix_sample_distribution_decompose_base");
    }
    const size_t poly_count = src_rows * src_cols;
    if (poly_count == 0)
    {
        out->format = requested_out_format;
        return 0;
    }

    std::vector<dim3> active_limb_ids(crt_depth);
    std::vector<uint64_t *> out_limb_bases(crt_depth, nullptr);
    std::vector<size_t> out_limb_strides(crt_depth, 0);
    std::vector<uint64_t> out_limb_moduli(crt_depth, 0);
    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[idx];
        active_limb_ids[idx] = limb_id;

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            return status;
        }
        if (out_device < 0 || !out_stream)
        {
            return set_error("invalid output limb metadata in gpu_matrix_sample_distribution_decompose_base");
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_stream = out_stream;
        }
        else if (out_device != dispatch_device)
        {
            return set_error("single-device mode requires all limbs on one device in gpu_matrix_sample_distribution_decompose_base");
        }

        uint64_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            return set_error("null output limb pointer in gpu_matrix_sample_distribution_decompose_base");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_matrix_sample_distribution_decompose_base");
        }
        out_limb_bases[idx] = dst;
        out_limb_strides[idx] = out->shared_limb_buffers[limb_id.x].words_per_poly;
        out_limb_moduli[idx] = out->ctx->moduli[idx];
    }
    if (dispatch_device < 0 || !dispatch_stream)
    {
        return set_error("invalid dispatch stream in gpu_matrix_sample_distribution_decompose_base");
    }

    const size_t out_ptr_bytes = crt_depth * sizeof(uint64_t *);
    const size_t out_stride_bytes = crt_depth * sizeof(size_t);
    const size_t out_moduli_bytes = crt_depth * sizeof(uint64_t);
    uint64_t **out_limb_bases_device = nullptr;
    size_t *out_limb_strides_device = nullptr;
    uint64_t *out_limb_moduli_device = nullptr;
    auto cleanup = [&]()
    {
        if (dispatch_device >= 0)
        {
            cudaSetDevice(dispatch_device);
        }
        if (out_limb_bases_device)
        {
            cudaFreeAsync(out_limb_bases_device, dispatch_stream);
            out_limb_bases_device = nullptr;
        }
        if (out_limb_strides_device)
        {
            cudaFreeAsync(out_limb_strides_device, dispatch_stream);
            out_limb_strides_device = nullptr;
        }
        if (out_limb_moduli_device)
        {
            cudaFreeAsync(out_limb_moduli_device, dispatch_stream);
            out_limb_moduli_device = nullptr;
        }
    };

    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    const size_t out_count = out->rows * out->cols;
    const size_t coeff_bytes = static_cast<size_t>(out->ctx->N) * sizeof(uint64_t);
    for (int out_limb = 0; out_limb <= level; ++out_limb)
    {
        const size_t out_idx = static_cast<size_t>(out_limb);
        status = matrix_wait_limb_stream(out, active_limb_ids[out_idx], dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        const size_t dst_pitch = out_limb_strides[out_idx] * sizeof(uint64_t);
        err = cudaMemset2DAsync(
            out_limb_bases[out_idx],
            dst_pitch,
            0,
            coeff_bytes,
            out_count,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
    }

    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_bases_device), out_ptr_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_strides_device), out_stride_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_moduli_device), out_moduli_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    err = cudaMemcpyAsync(
        out_limb_bases_device,
        out_limb_bases.data(),
        out_ptr_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_strides_device,
        out_limb_strides.data(),
        out_stride_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_moduli_device,
        out_limb_moduli.data(),
        out_moduli_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    const int src_limb_begin = 0;
    const int src_limb_end = small ? 1 : (level + 1);
    for (int src_limb = src_limb_begin; src_limb < src_limb_end; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const uint32_t src_bits = bit_width_u64(out->ctx->moduli[src_idx]);
        const size_t src_digit_offset_base =
            small ? 0 : (src_idx * static_cast<size_t>(digits_per_tower));
        status = launch_sample_distribution_decompose_all_slots_kernel(
            out_limb_bases_device,
            out_limb_strides_device,
            out_limb_moduli_device,
            crt_depth,
            poly_count,
            static_cast<size_t>(out->ctx->N),
            src_cols,
            out->cols,
            out_log_base_q,
            src_bits,
            base_bits,
            digits_per_tower,
            src_digit_offset_base,
            static_cast<uint32_t>(src_limb),
            out->ctx->moduli[src_idx],
            dist_type,
            sigma,
            seed,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    for (int out_limb = 0; out_limb <= level; ++out_limb)
    {
        status = matrix_record_limb_write(out, active_limb_ids[static_cast<size_t>(out_limb)], dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    if (requested_out_format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_ntt_all(out, default_batch(out->ctx));
        if (status != 0)
        {
            cleanup();
            return status;
        }
        out->format = GPU_POLY_FORMAT_EVAL;
    }

    cleanup();
    return 0;
}

extern "C" int gpu_matrix_sample_distribution_decompose_base(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed,
    uint32_t base_bits)
{
    return gpu_matrix_sample_distribution_decompose_base_impl(
        out,
        dist_type,
        sigma,
        seed,
        base_bits,
        false);
}

extern "C" int gpu_matrix_sample_distribution_decompose_base_small(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed,
    uint32_t base_bits)
{
    return gpu_matrix_sample_distribution_decompose_base_impl(
        out,
        dist_type,
        sigma,
        seed,
        base_bits,
        true);
}
