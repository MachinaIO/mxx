// The statistical trapdoor's perturbation sampler (see `sampler::trapdoor::statistical` for the
// derivation). With `G = T* T` never formed, the perturbation `[p_1; p_2]` of covariance
// `s^2 I - c^2 [T; I][T; I]*` is drawn as
//
//   p_2 ~ D_{Z, sqrt(s^2 - c^2)},
//   p_1 = round_r(center_scale T p_2 + sqrt(a) (z - T f(G) T* z) + s_iso N(0, I)),
//
// where `f(G) w` is the Chebyshev expansion of `f(g) = beta / (1 + sqrt(1 - beta g))` on
// `[0, gmax]` evaluated by the Clenshaw recurrence, and every product with `T` is pointwise in the
// negacyclic Fourier domain of the precomputed spectra of `T`. The floating-point evaluation is
// heuristic, as for every sampler.

namespace
{
    constexpr uint64_t kStatisticalDomainP2 = 0x6d78787374703201ULL;
    constexpr uint64_t kStatisticalDomainZ = 0x6d78787374707a01ULL;
    constexpr uint64_t kStatisticalDomainRound = 0x6d78787374727201ULL;
    constexpr uint32_t kStatisticalThreads = 256;
    constexpr uint32_t kStatisticalMaxDigits = 64;
    constexpr uint32_t kStatisticalChebyshevDegree = 24;
    constexpr double kStatisticalRoundingTail = 12.0;

    __device__ __forceinline__ double2 stat_mul(double2 a, double2 b)
    {
        return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    // conj(a) b
    __device__ __forceinline__ double2 stat_conj_mul(double2 a, double2 b)
    {
        return make_double2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
    }

    // exp(i pi l / n)
    __device__ __forceinline__ double2 stat_twist(uint64_t l, uint32_t n)
    {
        double s, c;
        sincospi(static_cast<double>(l) / static_cast<double>(n), &s, &c);
        return make_double2(c, s);
    }

    __device__ __forceinline__ size_t stat_thread()
    {
        return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    }

    __global__ void stat_zero_kernel(double2 *data, size_t count)
    {
        const size_t index = stat_thread();
        if (index < count) data[index] = make_double2(0.0, 0.0);
    }

    __global__ void stat_fft_bitreverse_kernel(double2 *data, size_t vectors, uint32_t n,
        uint32_t log_n)
    {
        const size_t index = stat_thread();
        if (index >= vectors * n) return;
        const size_t vector = index / n;
        const uint32_t i = static_cast<uint32_t>(index - vector * n);
        const uint32_t j = __brev(i) >> (32 - log_n);
        if (i < j)
        {
            double2 *base = data + vector * n;
            const double2 swap = base[i];
            base[i] = base[j];
            base[j] = swap;
        }
    }

    // One radix-2 stage `x_k ± exp(sign 2 pi i k / len) x_{k + len/2}`.
    __global__ void stat_fft_stage_kernel(double2 *data, size_t vectors, uint32_t n,
        uint32_t half, double sign)
    {
        const size_t index = stat_thread();
        const size_t butterflies = n / 2;
        if (index >= vectors * butterflies) return;
        const size_t vector = index / butterflies;
        const uint32_t b = static_cast<uint32_t>(index - vector * butterflies);
        const uint32_t group = b / half;
        const uint32_t k = b - group * half;
        double2 *base = data + vector * n;
        const uint32_t u = group * 2 * half + k;
        const uint32_t v = u + half;
        double s, c;
        sincospi(sign * static_cast<double>(k) / static_cast<double>(half), &s, &c);
        const double2 t = stat_mul(base[v], make_double2(c, s));
        const double2 x = base[u];
        base[u] = make_double2(x.x + t.x, x.y + t.y);
        base[v] = make_double2(x.x - t.x, x.y - t.y);
    }

    // The twisted coefficients of the small entries of one coefficient-domain chunk of `T`
    // (limb 0, centered), written to `spectra` from vector `vector_offset` on.
    __global__ void stat_load_chunk_kernel(MxxRawMatrixLimb chunk, size_t rows, size_t digits,
        uint32_t n, double2 *spectra, size_t vector_offset)
    {
        const size_t index = stat_thread();
        if (index >= rows * digits * n) return;
        const size_t poly = index / n;
        const uint32_t l = static_cast<uint32_t>(index - poly * n);
        const double value = static_cast<double>(centered_residue_i64(
            raw_matrix_load(chunk, poly, l, digits), chunk.modulus));
        const double2 twist = stat_twist(l, n);
        spectra[(vector_offset + poly) * n + l] = make_double2(value * twist.x, value * twist.y);
    }

    __global__ void stat_sample_p2_kernel(RawLimbSet outputs, uint32_t limbs,
        const GpuRngSeed *seed, size_t m_bar, size_t digits, size_t columns, uint32_t n,
        double sigma, int64_t *p2, double2 *centers, double center_scale)
    {
        const size_t index = stat_thread();
        if (index >= digits * columns * n) return;
        const size_t entry = index / n;
        const uint32_t l = static_cast<uint32_t>(index - entry * n);
        DeviceChaChaRng rng;
        rng_init(rng, *seed, entry + 1, l + 1, 0, kStatisticalDomainP2);
        const int64_t value = sigma > kWideGaussianThreshold ? sample_wide_gaussian(rng, sigma) :
                                                               sample_integer_karney(rng, 0.0, sigma);
        p2[index] = value;
        const double2 twist = stat_twist(l, n);
        const double scaled = center_scale * static_cast<double>(value);
        centers[entry * n + l] = make_double2(scaled * twist.x, scaled * twist.y);
        // Entry `(digit, column)` of `p_2` is row `m_bar + digit` of the perturbation.
        const size_t poly = (m_bar + entry / columns) * columns + entry % columns;
        for (uint32_t limb = 0; limb < limbs; ++limb)
            raw_matrix_store(outputs.limb[limb], poly, l, columns,
                signed_mod_i64(value, outputs.limb[limb].modulus));
    }

    // Twisted standard normal rows `z_i` of columns `0..columns` for rows `row_start..`.
    __global__ void stat_load_normal_kernel(const GpuRngSeed *seed, size_t row_start, size_t rows,
        size_t columns, uint32_t n, double2 *chunk)
    {
        const size_t index = stat_thread();
        if (index >= rows * columns * n) return;
        const size_t entry = index / n;
        const uint32_t l = static_cast<uint32_t>(index - entry * n);
        const size_t row = row_start + entry / columns;
        const size_t column = entry % columns;
        DeviceChaChaRng rng;
        rng_init(rng, *seed, row * columns + column + 1, l + 1, 0, kStatisticalDomainZ);
        const double z = sample_standard_normal(rng);
        const double2 twist = stat_twist(l, n);
        chunk[entry * n + l] = make_double2(z * twist.x, z * twist.y);
    }

    // w_(k, c) += sum over the chunk's rows i of conj(T_(i, k)) zhat_(i, c), pointwise.
    __global__ void stat_adjoint_kernel(const double2 *spectra, const double2 *chunk,
        size_t row_start, size_t rows, size_t digits, size_t columns, uint32_t n, double2 *w)
    {
        const size_t index = stat_thread();
        if (index >= digits * columns * n) return;
        const size_t entry = index / n;
        const uint32_t j = static_cast<uint32_t>(index - entry * n);
        const size_t k = entry / columns;
        const size_t column = entry % columns;
        double2 acc = w[index];
        for (size_t i = 0; i < rows; ++i)
        {
            const double2 t = spectra[((row_start + i) * digits + k) * n + j];
            const double2 z = chunk[(i * columns + column) * n + j];
            const double2 term = stat_conj_mul(t, z);
            acc.x += term.x;
            acc.y += term.y;
        }
        w[index] = acc;
    }

    // out = G v = T* (T v), one evaluation point and column per thread.
    __global__ void stat_gram_kernel(const double2 *spectra, const double2 *v, size_t m_bar,
        size_t digits, size_t columns, uint32_t n, double2 *out)
    {
        const size_t index = stat_thread();
        if (index >= columns * n) return;
        const size_t column = index / n;
        const uint32_t j = static_cast<uint32_t>(index - column * n);
        double2 input[kStatisticalMaxDigits];
        double2 acc[kStatisticalMaxDigits];
        for (size_t k = 0; k < digits; ++k)
        {
            input[k] = v[(k * columns + column) * n + j];
            acc[k] = make_double2(0.0, 0.0);
        }
        for (size_t i = 0; i < m_bar; ++i)
        {
            double2 u = make_double2(0.0, 0.0);
            for (size_t k = 0; k < digits; ++k)
            {
                const double2 term = stat_mul(spectra[(i * digits + k) * n + j], input[k]);
                u.x += term.x;
                u.y += term.y;
            }
            for (size_t k = 0; k < digits; ++k)
            {
                const double2 term = stat_conj_mul(spectra[(i * digits + k) * n + j], u);
                acc[k].x += term.x;
                acc[k].y += term.y;
            }
        }
        for (size_t k = 0; k < digits; ++k)
            out[(k * columns + column) * n + j] = acc[k];
    }

    // out = scale w + twice ((2 / gmax) g - b1) - b2, elementwise; a null `g` reads as zero.
    __global__ void stat_combine_kernel(const double2 *w, const double2 *g, const double2 *b1,
        const double2 *b2, size_t count, double scale, double twice, double inverse_half_gmax,
        double2 *out)
    {
        const size_t index = stat_thread();
        if (index >= count) return;
        const double2 wv = w[index];
        double2 x = make_double2(0.0, 0.0);
        if (g)
        {
            const double2 gv = g[index];
            const double2 bv = b1[index];
            x = make_double2(inverse_half_gmax * gv.x - bv.x, inverse_half_gmax * gv.y - bv.y);
        }
        const double2 previous = b2 ? b2[index] : make_double2(0.0, 0.0);
        out[index] = make_double2(scale * wv.x + twice * x.x - previous.x,
            scale * wv.y + twice * x.y - previous.y);
    }

    // q = centers - sqrt(a) h, elementwise.
    __global__ void stat_q_kernel(const double2 *centers, const double2 *h, size_t count,
        double sqrt_a, double2 *q)
    {
        const size_t index = stat_thread();
        if (index >= count) return;
        q[index] = make_double2(centers[index].x - sqrt_a * h[index].x,
            centers[index].y - sqrt_a * h[index].y);
    }

    // chunk_(i, c) = sum_k T_(i, k) q_(k, c) for rows `row_start..`.
    __global__ void stat_rows_kernel(const double2 *spectra, const double2 *q, size_t row_start,
        size_t rows, size_t digits, size_t columns, uint32_t n, double2 *chunk)
    {
        const size_t index = stat_thread();
        if (index >= rows * columns * n) return;
        const size_t entry = index / n;
        const uint32_t j = static_cast<uint32_t>(index - entry * n);
        const size_t i = entry / columns;
        const size_t column = entry % columns;
        double2 acc = make_double2(0.0, 0.0);
        for (size_t k = 0; k < digits; ++k)
        {
            const double2 term = stat_mul(spectra[((row_start + i) * digits + k) * n + j],
                q[(k * columns + column) * n + j]);
            acc.x += term.x;
            acc.y += term.y;
        }
        chunk[index] = acc;
    }

    // One sample of D_{Z, center, width} by rejection from the integers within the tail.
    __device__ __forceinline__ int64_t stat_round(DeviceChaChaRng &rng, double center, double width)
    {
        const int64_t low = static_cast<int64_t>(floor(center - kStatisticalRoundingTail * width));
        const uint64_t span = static_cast<uint64_t>(
            static_cast<int64_t>(ceil(center + kStatisticalRoundingTail * width)) - low + 1);
        for (int iteration = 0; iteration < 1 << 20; ++iteration)
        {
            const int64_t x = low + static_cast<int64_t>(rng_next_u64(rng) % span);
            const double distance = static_cast<double>(x) - center;
            if (uniform_open01(rng) < exp(-distance * distance / (2.0 * width * width)))
                return x;
        }
        return static_cast<int64_t>(llround(center));
    }

    // The rows `row_start..` of `p_1`: the inverse-transformed chunk, `sqrt(a) z`, the isotropic
    // part, and the randomized rounding, written to every limb.
    __global__ void stat_round_rows_kernel(const double2 *chunk, const GpuRngSeed *seed,
        size_t row_start, size_t rows, size_t columns, uint32_t n, double sqrt_a,
        double isotropic, double rounding, RawLimbSet outputs, uint32_t limbs)
    {
        const size_t index = stat_thread();
        if (index >= rows * columns * n) return;
        const size_t entry = index / n;
        const uint32_t l = static_cast<uint32_t>(index - entry * n);
        const size_t row = row_start + entry / columns;
        const size_t column = entry % columns;
        const double2 twist = stat_twist(l, n);
        const double2 value = chunk[entry * n + l];
        const double tq = (value.x * twist.x + value.y * twist.y) / static_cast<double>(n);
        DeviceChaChaRng normal;
        rng_init(normal, *seed, row * columns + column + 1, l + 1, 0, kStatisticalDomainZ);
        const double z = sample_standard_normal(normal);
        DeviceChaChaRng rng;
        rng_init(rng, *seed, row * columns + column + 1, l + 1, 0, kStatisticalDomainRound);
        const double center = tq + sqrt_a * z + isotropic * sample_standard_normal(rng);
        const int64_t sample = stat_round(rng, center, rounding);
        const size_t poly = row * columns + column;
        for (uint32_t limb = 0; limb < limbs; ++limb)
            raw_matrix_store(outputs.limb[limb], poly, l, columns,
                signed_mod_i64(sample, outputs.limb[limb].modulus));
    }

    dim3 stat_grid(size_t threads)
    {
        return dim3(static_cast<uint32_t>((threads + kStatisticalThreads - 1) / kStatisticalThreads));
    }

    // A pointer argument at or into the allocation of `binding`: the graph builder records the
    // pointer's offset from the binding's registered address, so the patch carries no addend.
    MxxGraphPatch stat_patch(uint32_t argument, uint32_t binding)
    {
        return {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, argument, 0, sizeof(void *),
            binding, 0};
    }

    // An in-place negacyclic-ready DFT `sum_l x_l exp(sign 2 pi i j l / n)` of `vectors`
    // vectors starting at `data + offset` (the twist is applied by the loaders and the rounding).
    int stat_fft(GpuContext *ctx, cudaStream_t stream, double2 *data, uint32_t binding,
        size_t offset, size_t vectors, uint32_t n, double sign)
    {
        uint32_t log_n = 0;
        while ((1u << log_n) < n) ++log_n;
        double2 *base = data + offset;
        const MxxGraphPatch patches[] = {stat_patch(0, binding)};
        int status = mxx_gpu_launch_kernel(ctx, stream, stat_fft_bitreverse_kernel,
            stat_grid(vectors * n), dim3(kStatisticalThreads), 0, patches, 1, base, vectors, n,
            log_n);
        if (status != 0) return status;
        for (uint32_t half = 1; half < n; half <<= 1)
        {
            status = mxx_gpu_launch_kernel(ctx, stream, stat_fft_stage_kernel,
                stat_grid(vectors * n / 2), dim3(kStatisticalThreads), 0, patches, 1, base,
                vectors, n, half, sign);
            if (status != 0) return status;
        }
        return 0;
    }

    double stat_correction(double beta, double g)
    {
        return beta / (1.0 + std::sqrt(1.0 - beta * g));
    }
}

// Writes the spectra of one coefficient-domain row chunk of `T` (rows `row_offset..`) into
// `spectra`, which holds `m_bar x digits` vectors of `n` complex values.
extern "C" int gpu_raw_statistical_spectra(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *chunk, double2 *spectra, size_t spectra_vectors, size_t row_offset,
    uint32_t chunk_binding, uint32_t spectra_binding)
{
    if (validate_raw_view(ctx, chunk, stream_raw) != 0 || !spectra ||
        chunk->limbs[0].crt_limb_index != 0 ||
        (row_offset + chunk->rows) * chunk->columns > spectra_vectors)
        return set_error("invalid statistical trapdoor spectra arguments");
    if (mxx_set_device(chunk->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t n = chunk->degree;
    const size_t vector_offset = row_offset * chunk->columns;
    const size_t vectors = chunk->rows * chunk->columns;
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
            static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)), sizeof(void *),
            chunk_binding, 0},
        stat_patch(4, spectra_binding),
    };
    int status = mxx_gpu_launch_kernel(ctx, stream, stat_load_chunk_kernel,
        stat_grid(vectors * n), dim3(kStatisticalThreads), 0, patches, std::size(patches),
        chunk->limbs[0], static_cast<size_t>(chunk->rows), static_cast<size_t>(chunk->columns), n,
        spectra, vector_offset);
    if (status != 0) return status;
    return stat_fft(ctx, stream, spectra, spectra_binding, vector_offset * n, vectors, n, 1.0);
}

// The perturbation `[p_1; p_2]` (`m_bar + digits` rows) of one tile into `output`, every limb.
// `vectors` holds 7 blocks of `digits x columns` vectors (w, three Clenshaw states, G b, the
// centers, q) and `chunk` `row_chunk x columns` vectors; `p2` holds the integers of `p_2`.
extern "C" int gpu_raw_statistical_perturbation(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *output, const void *seed_raw, const double2 *spectra,
    double2 *vectors, size_t vector_bytes, double2 *chunk, size_t chunk_bytes, int64_t *p2,
    size_t p2_bytes, size_t m_bar, size_t row_chunk, double p2_sigma, double center_scale,
    double a, double beta, double gmax, double isotropic, double rounding,
    uint32_t output_binding_base, uint32_t seed_binding, uint32_t spectra_binding,
    uint32_t vectors_binding, uint32_t chunk_binding, uint32_t p2_binding)
{
    const auto *seed = static_cast<const GpuRngSeed *>(seed_raw);
    if (validate_raw_view(ctx, output, stream_raw) != 0 || !seed || !spectra || !vectors ||
        !chunk || !p2 || output->rows <= m_bar || m_bar == 0 || row_chunk == 0 ||
        output->limb_count > kRawNttLimbs ||
        output_binding_base > UINT32_MAX - output->limb_count)
        return set_error("invalid statistical perturbation arguments");
    const size_t digits = output->rows - m_bar;
    const size_t columns = output->columns;
    const uint32_t n = output->degree;
    const size_t block = digits * columns * n;
    if (digits > kStatisticalMaxDigits || 7 * block * sizeof(double2) > vector_bytes ||
        row_chunk * columns * n * sizeof(double2) > chunk_bytes ||
        block * sizeof(int64_t) > p2_bytes || !(rounding > 0.0) || !(gmax > 0.0))
        return set_error("statistical perturbation workspace is too small");
    if (mxx_set_device(output->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    RawLimbSet outputs{};
    std::vector<MxxGraphPatch> output_patches;
    raw_limb_set(output, 0, output->limb_count, 0, output_binding_base, outputs, output_patches);
    const uint32_t limbs = static_cast<uint32_t>(output->limb_count);
    // Blocks of `vectors`.
    double2 *w = vectors;
    double2 *states[3] = {vectors + block, vectors + 2 * block, vectors + 3 * block};
    double2 *gram = vectors + 4 * block;
    double2 *centers = vectors + 5 * block;
    double2 *q = vectors + 6 * block;
    const auto block_patch = [&](uint32_t argument, const double2 *) {
        return stat_patch(argument, vectors_binding);
    };
    int status = 0;
    // p_2 and the twisted centers `center_scale p_2`.
    {
        std::vector<MxxGraphPatch> patches = output_patches;
        patches.push_back(stat_patch(2, seed_binding));
        patches.push_back(stat_patch(8, p2_binding));
        patches.push_back(block_patch(9, centers));
        status = mxx_gpu_launch_kernel(ctx, stream, stat_sample_p2_kernel, stat_grid(block),
            dim3(kStatisticalThreads), 0, patches.data(), patches.size(), outputs, limbs, seed,
            m_bar, digits, columns, n, p2_sigma, p2, centers, center_scale);
        if (status != 0) return status;
        status = stat_fft(ctx, stream, vectors, vectors_binding, 5 * block, digits * columns, n,
            1.0);
        if (status != 0) return status;
    }
    // w = T* z, chunk by chunk.
    {
        const MxxGraphPatch zero_patches[] = {block_patch(0, w)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_zero_kernel, stat_grid(block),
            dim3(kStatisticalThreads), 0, zero_patches, 1, w, block);
        if (status != 0) return status;
    }
    for (size_t row_start = 0; row_start < m_bar; row_start += row_chunk)
    {
        const size_t rows = std::min(row_chunk, m_bar - row_start);
        const MxxGraphPatch load_patches[] = {stat_patch(0, seed_binding), stat_patch(5, chunk_binding)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_load_normal_kernel,
            stat_grid(rows * columns * n), dim3(kStatisticalThreads), 0, load_patches,
            std::size(load_patches), seed, row_start, rows, columns, n, chunk);
        if (status != 0) return status;
        status = stat_fft(ctx, stream, chunk, chunk_binding, 0, rows * columns, n, 1.0);
        if (status != 0) return status;
        const MxxGraphPatch adjoint_patches[] = {stat_patch(0, spectra_binding),
            stat_patch(1, chunk_binding), block_patch(7, w)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_adjoint_kernel, stat_grid(block),
            dim3(kStatisticalThreads), 0, adjoint_patches, std::size(adjoint_patches), spectra,
            chunk, row_start, rows, digits, columns, n, w);
        if (status != 0) return status;
    }
    // h = f(G) w by the Clenshaw recurrence with the Chebyshev coefficients of f on [0, gmax].
    double coefficients[kStatisticalChebyshevDegree + 1];
    {
        const size_t nodes = kStatisticalChebyshevDegree + 1;
        double values[kStatisticalChebyshevDegree + 1];
        const double pi = 3.14159265358979323846;
        for (size_t m = 0; m < nodes; ++m)
        {
            const double x = std::cos(pi * (m + 0.5) / nodes);
            values[m] = stat_correction(beta, gmax * (x + 1.0) / 2.0);
        }
        for (size_t k = 0; k < nodes; ++k)
        {
            double sum = 0.0;
            for (size_t m = 0; m < nodes; ++m)
                sum += values[m] * std::cos(pi * k * (m + 0.5) / nodes);
            coefficients[k] = (k == 0 ? 1.0 : 2.0) * sum / nodes;
        }
    }
    const double inverse_half_gmax = 2.0 / gmax;
    // states[0] = b_(k+1), states[1] = b_(k+2), states[2] = the new b_k.
    double2 *b1 = states[0];
    double2 *b2 = states[1];
    double2 *bk = states[2];
    const size_t count = block;
    for (size_t k = kStatisticalChebyshevDegree; k >= 1; --k)
    {
        const bool first = k == kStatisticalChebyshevDegree;
        if (!first)
        {
            const MxxGraphPatch gram_patches[] = {stat_patch(0, spectra_binding),
                block_patch(1, b1), block_patch(6, gram)};
            status = mxx_gpu_launch_kernel(ctx, stream, stat_gram_kernel,
                stat_grid(columns * n), dim3(kStatisticalThreads), 0, gram_patches,
                std::size(gram_patches), spectra, b1, m_bar, digits, columns,
                n, gram);
            if (status != 0) return status;
        }
        const bool second = k + 1 == kStatisticalChebyshevDegree;
        std::vector<MxxGraphPatch> combine_patches = {block_patch(0, w), block_patch(8, bk)};
        if (!first)
        {
            combine_patches.push_back(block_patch(1, gram));
            combine_patches.push_back(block_patch(2, b1));
        }
        if (!first && !second) combine_patches.push_back(block_patch(3, b2));
        status = mxx_gpu_launch_kernel(ctx, stream, stat_combine_kernel, stat_grid(count),
            dim3(kStatisticalThreads), 0, combine_patches.data(), combine_patches.size(),
            static_cast<const double2 *>(w), first ? nullptr : static_cast<const double2 *>(gram),
            first ? nullptr : static_cast<const double2 *>(b1),
            (first || second) ? nullptr : static_cast<const double2 *>(b2), count,
            coefficients[k], 2.0, inverse_half_gmax, bk);
        if (status != 0) return status;
        double2 *retired = b2;
        b2 = b1;
        b1 = bk;
        bk = retired;
    }
    {
        const MxxGraphPatch gram_patches[] = {stat_patch(0, spectra_binding), block_patch(1, b1),
            block_patch(6, gram)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_gram_kernel, stat_grid(columns * n),
            dim3(kStatisticalThreads), 0, gram_patches, std::size(gram_patches), spectra, b1,
            m_bar, digits, columns, n, gram);
        if (status != 0) return status;
        const MxxGraphPatch combine_patches[] = {block_patch(0, w), block_patch(1, gram),
            block_patch(2, b1), block_patch(3, b2), block_patch(8, bk)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_combine_kernel, stat_grid(count),
            dim3(kStatisticalThreads), 0, combine_patches, std::size(combine_patches),
            static_cast<const double2 *>(w), static_cast<const double2 *>(gram),
            static_cast<const double2 *>(b1), static_cast<const double2 *>(b2), count,
            coefficients[0], 1.0, inverse_half_gmax, bk);
        if (status != 0) return status;
    }
    const double sqrt_a = std::sqrt(a);
    {
        const MxxGraphPatch q_patches[] = {block_patch(0, centers), block_patch(1, bk),
            block_patch(4, q)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_q_kernel, stat_grid(count),
            dim3(kStatisticalThreads), 0, q_patches, std::size(q_patches),
            static_cast<const double2 *>(centers), static_cast<const double2 *>(bk), count, sqrt_a,
            q);
        if (status != 0) return status;
    }
    // p_1, chunk by chunk.
    for (size_t row_start = 0; row_start < m_bar; row_start += row_chunk)
    {
        const size_t rows = std::min(row_chunk, m_bar - row_start);
        const MxxGraphPatch rows_patches[] = {stat_patch(0, spectra_binding), block_patch(1, q),
            stat_patch(7, chunk_binding)};
        status = mxx_gpu_launch_kernel(ctx, stream, stat_rows_kernel,
            stat_grid(rows * columns * n), dim3(kStatisticalThreads), 0, rows_patches,
            std::size(rows_patches), spectra, static_cast<const double2 *>(q), row_start, rows,
            digits, columns, n, chunk);
        if (status != 0) return status;
        status = stat_fft(ctx, stream, chunk, chunk_binding, 0, rows * columns, n, -1.0);
        if (status != 0) return status;
        std::vector<MxxGraphPatch> round_patches;
        round_patches.push_back(stat_patch(0, chunk_binding));
        round_patches.push_back(stat_patch(1, seed_binding));
        std::vector<MxxGraphPatch> limb_patches;
        RawLimbSet rows_outputs{};
        raw_limb_set(output, 0, output->limb_count, 9, output_binding_base, rows_outputs,
            limb_patches);
        round_patches.insert(round_patches.end(), limb_patches.begin(), limb_patches.end());
        status = mxx_gpu_launch_kernel(ctx, stream, stat_round_rows_kernel,
            stat_grid(rows * columns * n), dim3(kStatisticalThreads), 0, round_patches.data(),
            round_patches.size(), static_cast<const double2 *>(chunk), seed, row_start, rows,
            columns, n, sqrt_a, isotropic, rounding, rows_outputs, limbs);
        if (status != 0) return status;
    }
    return 0;
}
