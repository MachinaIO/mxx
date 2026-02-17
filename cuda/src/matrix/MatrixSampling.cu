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
    uint64_t **dst,
    size_t poly_count,
    size_t limb_count,
    size_t n,
    const uint64_t *moduli,
    const uint32_t *limb_indices,
    int dist_type,
    double sigma,
    uint64_t seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = limb_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t per_limb = poly_count * n;
    const size_t limb_slot = idx / per_limb;
    const size_t rem = idx - limb_slot * per_limb;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = limb_slot * poly_count + poly_idx;

    const uint64_t modulus = moduli[limb_slot];
    const uint32_t limb_idx = limb_indices[limb_slot];

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

    dst[ptr_idx][coeff_idx] = sample;
}

int launch_sample_distribution_multi_limb_kernel(
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &moduli,
    const std::vector<uint32_t> &limb_indices,
    int dist_type,
    double sigma,
    uint64_t seed,
    cudaStream_t stream)
{
    const size_t limb_count = moduli.size();
    if (limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (limb_indices.size() != limb_count)
    {
        return set_error("unexpected limb parameter counts in matrix_sample_distribution_multi_limb_kernel");
    }
    const size_t ptr_count = limb_count * poly_count;
    if (dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_sample_distribution_multi_limb_kernel");
    }

    uint64_t **d_dst = nullptr;
    uint64_t *d_moduli = nullptr;
    uint32_t *d_limb_indices = nullptr;
    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = limb_count * sizeof(uint64_t);
    const size_t u32_bytes = limb_count * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&d_dst, ptr_bytes);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMalloc(&d_moduli, u64_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        return set_error(err);
    }
    err = cudaMalloc(&d_limb_indices, u32_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_moduli, moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_limb_indices, limb_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_sample_distribution_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        d_dst,
        poly_count,
        limb_count,
        n,
        d_moduli,
        d_limb_indices,
        dist_type,
        sigma,
        seed);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFree(d_dst);
        cudaFree(d_moduli);
        cudaFree(d_limb_indices);
        return set_error(err);
    }

    cudaFree(d_dst);
    cudaFree(d_moduli);
    cudaFree(d_limb_indices);
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
        out->format = PolyFormat::Eval;
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

    auto &limb_map = out->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_distribution");
    }
    std::vector<GpuPoly *> out_polys;
    int status = materialize_matrix_poly_views(out, out_polys);
    if (status != 0)
    {
        return status;
    }
    auto cleanup_out_polys = [&]()
    {
        release_poly_views(out_polys);
    };

    struct DistBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> moduli;
        std::vector<uint32_t> limb_indices;
    };
    std::vector<DistBatch> batches;
    auto get_batch = [&](int device) -> DistBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(DistBatch{device, nullptr, {}, {}, {}});
        return batches.back();
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> limb_ptrs;
        limb_ptrs.reserve(count);
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;
        status = matrix_limb_device(out, limb_id, &limb_device);
        if (status != 0)
        {
            cleanup_out_polys();
            return status;
        }
        status = matrix_limb_stream(out, limb_id, &limb_stream);
        if (status != 0)
        {
            cleanup_out_polys();
            return status;
        }

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out_polys[idx];
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                cleanup_out_polys();
                return set_error("invalid output poly in gpu_matrix_sample_distribution");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                cleanup_out_polys();
                return set_error("unexpected limb GPU partition in gpu_matrix_sample_distribution");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                cleanup_out_polys();
                return set_error("unexpected limb index in gpu_matrix_sample_distribution");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                cleanup_out_polys();
                return set_error("unsupported limb type in gpu_matrix_sample_distribution");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            limb_ptrs.push_back(limb_u64.v.data);
        }

        if (limb_device < 0 || !limb_stream)
        {
            cleanup_out_polys();
            return set_error("invalid limb metadata in gpu_matrix_sample_distribution");
        }
        auto &batch = get_batch(limb_device);
        if (!batch.stream)
        {
            batch.stream = limb_stream;
        }
        batch.moduli.push_back(out->ctx->moduli[static_cast<size_t>(limb)]);
        batch.limb_indices.push_back(static_cast<uint32_t>(limb));
        batch.dst_ptrs.insert(batch.dst_ptrs.end(), limb_ptrs.begin(), limb_ptrs.end());
    }

    for (auto &batch : batches)
    {
        if (!batch.stream)
        {
            cleanup_out_polys();
            return set_error("null stream in gpu_matrix_sample_distribution");
        }
        cudaError_t err = cudaSetDevice(batch.device);
        if (err != cudaSuccess)
        {
            cleanup_out_polys();
            return set_error(err);
        }
        status = launch_sample_distribution_multi_limb_kernel(
            batch.dst_ptrs,
            count,
            static_cast<size_t>(out->ctx->N),
            batch.moduli,
            batch.limb_indices,
            dist_type,
            sigma,
            seed,
            batch.stream);
        if (status != 0)
        {
            cleanup_out_polys();
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out_polys)
    {
        poly->format = PolyFormat::Coeff;
        status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup_out_polys();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    cleanup_out_polys();
    out->format = PolyFormat::Eval;
    return 0;
}
