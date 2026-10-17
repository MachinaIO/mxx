// The statistical trapdoor's precomputation and perturbation sampler: Peikert's convolution
// sampler (CRYPTO 2010, Algorithm 2) with the per-point Cholesky factors of DRTV26 (see
// `sampler::trapdoor::statistical` for the derivation and the layout of the precomputation).
//
// TrapGen writes the values of `T` at the points `j < n / 2` chunk by chunk
// (`gpu_raw_statistical_spectra`) and then the Cholesky factors
// `L_j L_j* = (s² - r²) I - α T(ω_j) T(ω_j)*`, one point per thread
// (`gpu_raw_statistical_factor`), rejecting the attempt in the preimage status record when a pivot
// falls below the floor. A perturbation `[p_1; p_2]` of a preimage tile is
//
//   p_2 ~ D_{Z, √(s² - c²)},
//   p_1 = round_r(F^{-1}(T̂ ĉ + L q̂)),   ĉ = F(center_scale p_2),
//
// with `q̂_j` circular complex Gaussian of variance `n` at the points `j < n / 2` and the
// conjugate values at their partners `n - 1 - j`. The floating-point evaluation is heuristic, as
// for every sampler.

namespace
{
    constexpr uint64_t kStatisticalDomainP2 = 0x6d78787374703201ULL;
    constexpr uint64_t kStatisticalDomainQ = 0x6d78787374707102ULL;
    constexpr uint64_t kStatisticalDomainRound = 0x6d78787374727201ULL;
    constexpr uint32_t kStatisticalThreads = 256;
    // One thread per point runs a whole factorization; small blocks spread the points over SMs.
    constexpr uint32_t kStatisticalFactorThreads = 64;
    constexpr uint32_t kStatisticalMaxDigits = 64;
    // `sampler::trapdoor::statistical::POWER_ITERATIONS`.
    constexpr uint32_t kStatisticalPowerIterations = 64;
    // `sampler::trapdoor::statistical::PRECOMPUTATION_HEADER_BYTES`: the width and the measured
    // singular value.
    constexpr size_t kStatisticalHeaderBytes = 16;
    constexpr double kStatisticalRoundingTail = 12.0;

    __device__ __forceinline__ double2 stat_mul(double2 a, double2 b)
    {
        return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    // a conj(b)
    __device__ __forceinline__ double2 stat_mul_conj(double2 a, double2 b)
    {
        return make_double2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
    }

    __device__ __forceinline__ double stat_norm(double2 a)
    {
        return a.x * a.x + a.y * a.y;
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

    // The index of entry `(i, k)`, `k ≤ i`, of a packed lower triangle.
    __device__ __forceinline__ size_t stat_triangle(size_t i, size_t k)
    {
        return i * (i + 1) / 2 + k;
    }

    size_t stat_precomputation_bytes(uint32_t n, size_t m_bar, size_t digits)
    {
        const size_t points = n / 2;
        return kStatisticalHeaderBytes +
            16 * points * (m_bar * digits + m_bar * (m_bar + 1) / 2);
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
    // (limb 0, centered), one vector of `n` per entry.
    __global__ void stat_load_chunk_kernel(MxxRawMatrixLimb chunk, size_t rows, size_t digits,
        uint32_t n, double2 *scratch)
    {
        const size_t index = stat_thread();
        if (index >= rows * digits * n) return;
        const size_t poly = index / n;
        const uint32_t l = static_cast<uint32_t>(index - poly * n);
        const double value = static_cast<double>(centered_residue_i64(
            raw_matrix_load(chunk, poly, l, digits), chunk.modulus));
        const double2 twist = stat_twist(l, n);
        scratch[poly * n + l] = make_double2(value * twist.x, value * twist.y);
    }

    // The points `j < n / 2` of the chunk's spectra into the precomputation's spectra from
    // vector `vector_offset` on.
    __global__ void stat_half_spectra_kernel(const double2 *scratch, size_t vectors, uint32_t n,
        double2 *spectra, size_t vector_offset)
    {
        const size_t points = n / 2;
        const size_t index = stat_thread();
        if (index >= vectors * points) return;
        const size_t vector = index / points;
        const size_t j = index - vector * points;
        spectra[(vector_offset + vector) * points + j] = scratch[vector * n + j];
    }

    // Writes the width and clears the measured singular value of a new attempt.
    __global__ void stat_header_kernel(double *header, double preimage_sigma,
        const MxxPreimageStatus *status)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0 || status->reserved != 0U) return;
        header[0] = preimage_sigma;
        header[1] = 0.0;
    }

    // One point `j` per thread: the largest singular value of `T(ω_j)` by power iteration
    // (maximized into `measured`), then `Σ'_j` and its in-place Cholesky factorization. A pivot
    // below `pivot_floor` rejects the attempt.
    __global__ void stat_factor_kernel(const double2 *spectra, double2 *factor, size_t m_bar,
        size_t digits, size_t points, double variance, double alpha, double pivot_floor,
        MxxPreimageStatus *status, unsigned long long *measured)
    {
        const size_t j = stat_thread();
        if (j >= points || status->reserved != 0U) return;
        const auto t = [&](size_t i, size_t k) { return spectra[(i * digits + k) * points + j]; };
        // ‖G v‖ / ‖v‖ for G = T* T increases to λ_max(G) from the all-ones vector.
        double2 v[kStatisticalMaxDigits];
        double2 next[kStatisticalMaxDigits];
        for (size_t k = 0; k < digits; ++k) v[k] = make_double2(1.0, 0.0);
        double lambda = 0.0;
        for (uint32_t iteration = 0; iteration < kStatisticalPowerIterations; ++iteration)
        {
            double norm = 0.0;
            for (size_t k = 0; k < digits; ++k) norm += stat_norm(v[k]);
            norm = sqrt(norm);
            if (norm == 0.0) break;
            for (size_t k = 0; k < digits; ++k) next[k] = make_double2(0.0, 0.0);
            for (size_t i = 0; i < m_bar; ++i)
            {
                double2 u = make_double2(0.0, 0.0);
                for (size_t k = 0; k < digits; ++k)
                {
                    const double2 term = stat_mul(t(i, k), v[k]);
                    u.x += term.x / norm;
                    u.y += term.y / norm;
                }
                for (size_t k = 0; k < digits; ++k)
                {
                    // conj(T_ik) u
                    const double2 term = stat_mul_conj(u, t(i, k));
                    next[k].x += term.x;
                    next[k].y += term.y;
                }
            }
            lambda = 0.0;
            for (size_t k = 0; k < digits; ++k)
            {
                lambda += stat_norm(next[k]);
                v[k] = next[k];
            }
            lambda = sqrt(lambda);
        }
        atomicMax(measured, static_cast<unsigned long long>(__double_as_longlong(sqrt(lambda))));
        const auto l = [&](size_t i, size_t k) -> double2 & {
            return factor[stat_triangle(i, k) * points + j];
        };
        for (size_t i = 0; i < m_bar; ++i)
            for (size_t p = 0; p <= i; ++p)
            {
                double2 acc = make_double2(0.0, 0.0);
                for (size_t k = 0; k < digits; ++k)
                {
                    const double2 term = stat_mul_conj(t(i, k), t(p, k));
                    acc.x += term.x;
                    acc.y += term.y;
                }
                l(i, p) = make_double2((i == p ? variance : 0.0) - alpha * acc.x, -alpha * acc.y);
            }
        for (size_t k = 0; k < m_bar; ++k)
        {
            double pivot = l(k, k).x;
            for (size_t p = 0; p < k; ++p) pivot -= stat_norm(l(k, p));
            // `!(>=)` also rejects a NaN pivot.
            if (!(pivot >= pivot_floor))
            {
                atomicExch(&status->accepted, 0U);
                return;
            }
            const double diagonal = sqrt(pivot);
            l(k, k) = make_double2(diagonal, 0.0);
            for (size_t i = k + 1; i < m_bar; ++i)
            {
                double2 acc = l(i, k);
                for (size_t p = 0; p < k; ++p)
                {
                    const double2 term = stat_mul_conj(l(i, p), l(k, p));
                    acc.x -= term.x;
                    acc.y -= term.y;
                }
                l(i, k) = make_double2(acc.x / diagonal, acc.y / diagonal);
            }
        }
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

    // q̂ of entry `(row, column)` at the points `j < n / 2`: real and imaginary parts N(0, n / 2).
    __global__ void stat_normal_kernel(const GpuRngSeed *seed, size_t entries, uint32_t n,
        double2 *buffer)
    {
        const size_t points = n / 2;
        const size_t index = stat_thread();
        if (index >= entries * points) return;
        const size_t entry = index / points;
        const size_t j = index - entry * points;
        DeviceChaChaRng rng;
        rng_init(rng, *seed, entry + 1, j + 1, 0, kStatisticalDomainQ);
        const double scale = sqrt(static_cast<double>(n) / 2.0);
        const double re = sample_standard_normal(rng);
        const double im = sample_standard_normal(rng);
        buffer[entry * n + j] = make_double2(scale * re, scale * im);
    }

    // One point `j < n / 2` and column per thread: rows from the last, `T̂_i· ĉ + Σ_{k≤i} L_ik q̂_k`
    // replaces `q̂_i` (no later row reads it), and its conjugate fills the partner `n - 1 - j`.
    __global__ void stat_combine_kernel(const double2 *spectra, const double2 *factor,
        const double2 *centers, size_t m_bar, size_t digits, size_t columns, uint32_t n,
        double2 *buffer)
    {
        const size_t points = n / 2;
        const size_t index = stat_thread();
        if (index >= columns * points) return;
        const size_t column = index / points;
        const size_t j = index - column * points;
        for (size_t row = m_bar; row-- > 0;)
        {
            double2 acc = make_double2(0.0, 0.0);
            for (size_t d = 0; d < digits; ++d)
            {
                const double2 term = stat_mul(spectra[(row * digits + d) * points + j],
                    centers[(d * columns + column) * n + j]);
                acc.x += term.x;
                acc.y += term.y;
            }
            for (size_t k = 0; k <= row; ++k)
            {
                const double2 term = stat_mul(factor[stat_triangle(row, k) * points + j],
                    buffer[(k * columns + column) * n + j]);
                acc.x += term.x;
                acc.y += term.y;
            }
            double2 *out = buffer + (row * columns + column) * n;
            out[j] = acc;
            out[n - 1 - j] = make_double2(acc.x, -acc.y);
        }
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

    // The rows of `p_1`: the inverse-transformed buffer and its randomized rounding, written to
    // every limb.
    __global__ void stat_round_rows_kernel(const double2 *buffer, const GpuRngSeed *seed,
        size_t entries, uint32_t n, double rounding, RawLimbSet outputs, uint32_t limbs,
        size_t columns)
    {
        const size_t index = stat_thread();
        if (index >= entries * n) return;
        const size_t entry = index / n;
        const uint32_t l = static_cast<uint32_t>(index - entry * n);
        const double2 twist = stat_twist(l, n);
        const double2 value = buffer[entry * n + l];
        const double center = (value.x * twist.x + value.y * twist.y) / static_cast<double>(n);
        DeviceChaChaRng rng;
        rng_init(rng, *seed, entry + 1, l + 1, 0, kStatisticalDomainRound);
        const int64_t sample = stat_round(rng, center, rounding);
        for (uint32_t limb = 0; limb < limbs; ++limb)
            raw_matrix_store(outputs.limb[limb], entry, l, columns,
                signed_mod_i64(sample, outputs.limb[limb].modulus));
    }

    dim3 stat_grid(size_t threads, uint32_t block = kStatisticalThreads)
    {
        return dim3(static_cast<uint32_t>((threads + block - 1) / block));
    }

    // A pointer argument at or into the allocation of `binding`: the graph builder records the
    // pointer's offset from the binding's registered address, so the patch carries no addend.
    MxxGraphPatch stat_patch(uint32_t argument, uint32_t binding)
    {
        return {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, argument, 0, sizeof(void *),
            binding, 0};
    }

    // An in-place negacyclic-ready DFT `sum_l x_l exp(sign 2 pi i j l / n)` of `vectors`
    // vectors at `data` (the twist is applied by the loaders and the rounding).
    int stat_fft(GpuContext *ctx, gpuStream_t stream, double2 *data, uint32_t binding,
        size_t vectors, uint32_t n, double sign)
    {
        uint32_t log_n = 0;
        while ((1u << log_n) < n) ++log_n;
        const MxxGraphPatch patches[] = {stat_patch(0, binding)};
        int status = mxx_gpu_launch_kernel(ctx, stream, stat_fft_bitreverse_kernel,
            stat_grid(vectors * n), dim3(kStatisticalThreads), 0, patches, 1, data, vectors, n,
            log_n);
        if (status != 0) return status;
        for (uint32_t half = 1; half < n; half <<= 1)
        {
            status = mxx_gpu_launch_kernel(ctx, stream, stat_fft_stage_kernel,
                stat_grid(vectors * n / 2), dim3(kStatisticalThreads), 0, patches, 1, data,
                vectors, n, half, sign);
            if (status != 0) return status;
        }
        return 0;
    }
}

// Writes the values at the points `j < n / 2` of one coefficient-domain row chunk of `T` (rows
// `row_offset..`) into the spectra of the precomputation, through `scratch` (the chunk's full
// transforms).
extern "C" int gpu_raw_statistical_spectra(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *chunk, double2 *scratch, size_t scratch_bytes,
    uint8_t *precomputation, size_t precomputation_bytes, size_t m_bar, size_t row_offset,
    uint32_t chunk_binding, uint32_t scratch_binding, uint32_t precomputation_binding)
{
    if (validate_raw_view(ctx, chunk, stream_raw) != 0 || !scratch || !precomputation ||
        chunk->limbs[0].crt_limb_index != 0 || row_offset + chunk->rows > m_bar ||
        chunk->rows * chunk->columns * chunk->degree * sizeof(double2) > scratch_bytes ||
        precomputation_bytes != stat_precomputation_bytes(chunk->degree, m_bar, chunk->columns))
        return set_error("invalid statistical trapdoor spectra arguments");
    if (mxx_set_device(chunk->physical_device) != gpuSuccess)
        return set_error(gpuGetLastError());
    const auto stream = reinterpret_cast<gpuStream_t>(stream_raw);
    const uint32_t n = chunk->degree;
    const size_t vectors = chunk->rows * chunk->columns;
    const MxxGraphPatch load_patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
            static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)), sizeof(void *),
            chunk_binding, 0},
        stat_patch(4, scratch_binding),
    };
    int status = mxx_gpu_launch_kernel(ctx, stream, stat_load_chunk_kernel,
        stat_grid(vectors * n), dim3(kStatisticalThreads), 0, load_patches,
        std::size(load_patches), chunk->limbs[0], static_cast<size_t>(chunk->rows),
        static_cast<size_t>(chunk->columns), n, scratch);
    if (status != 0) return status;
    status = stat_fft(ctx, stream, scratch, scratch_binding, vectors, n, 1.0);
    if (status != 0) return status;
    auto *spectra = reinterpret_cast<double2 *>(precomputation + kStatisticalHeaderBytes);
    const MxxGraphPatch half_patches[] = {
        stat_patch(0, scratch_binding), stat_patch(3, precomputation_binding)};
    return mxx_gpu_launch_kernel(ctx, stream, stat_half_spectra_kernel,
        stat_grid(vectors * (n / 2)), dim3(kStatisticalThreads), 0, half_patches,
        std::size(half_patches), static_cast<const double2 *>(scratch), vectors, n, spectra,
        row_offset * chunk->columns);
}

// One TrapGen attempt's factorization: the header, the measured singular value, and the
// Cholesky factors of the precomputation, with the attempt's status (accepted iff every pivot
// is at least `pivot_floor`) and the retry latch.
extern "C" int gpu_raw_statistical_factor(GpuContext *ctx, void *stream_raw, int physical_device,
    uint8_t *precomputation, size_t precomputation_bytes, size_t m_bar, size_t digits,
    uint32_t n, double preimage_sigma, double variance, double alpha, double pivot_floor,
    const uint64_t *attempt, MxxPreimageStatus *status, uint32_t precomputation_binding,
    uint32_t attempt_binding, uint32_t status_binding)
{
    if (!ctx || !stream_raw || !precomputation || !attempt || !status || m_bar == 0 ||
        digits == 0 || digits > kStatisticalMaxDigits || n < 2 || (n & (n - 1)) != 0 ||
        precomputation_bytes != stat_precomputation_bytes(n, m_bar, digits) ||
        !(variance > 0.0) || !(alpha > 0.0) || !(pivot_floor > 0.0))
        return set_error("invalid statistical trapdoor factor arguments");
    if (mxx_set_device(physical_device) != gpuSuccess) return set_error(gpuGetLastError());
    const auto stream = reinterpret_cast<gpuStream_t>(stream_raw);
    const size_t points = n / 2;
    auto *header = reinterpret_cast<double *>(precomputation);
    auto *spectra = reinterpret_cast<const double2 *>(precomputation + kStatisticalHeaderBytes);
    auto *factor = reinterpret_cast<double2 *>(
        precomputation + kStatisticalHeaderBytes + 16 * points * m_bar * digits);
    const MxxGraphPatch init_patches[] = {
        stat_patch(0, status_binding), stat_patch(1, attempt_binding)};
    int result = mxx_gpu_launch_kernel(ctx, stream, raw_preimage_status_init_attempt_kernel,
        dim3(1), dim3(1), 0, init_patches, std::size(init_patches), status, attempt);
    if (result != 0) return result;
    const MxxGraphPatch header_patches[] = {
        stat_patch(0, precomputation_binding), stat_patch(2, status_binding)};
    result = mxx_gpu_launch_kernel(ctx, stream, stat_header_kernel, dim3(1), dim3(1), 0,
        header_patches, std::size(header_patches), header, preimage_sigma,
        static_cast<const MxxPreimageStatus *>(status));
    if (result != 0) return result;
    const MxxGraphPatch factor_patches[] = {stat_patch(0, precomputation_binding),
        stat_patch(1, precomputation_binding), stat_patch(8, status_binding),
        stat_patch(9, precomputation_binding)};
    result = mxx_gpu_launch_kernel(ctx, stream, stat_factor_kernel,
        stat_grid(points, kStatisticalFactorThreads), dim3(kStatisticalFactorThreads), 0,
        factor_patches, std::size(factor_patches), spectra, factor, m_bar, digits, points,
        variance, alpha, pivot_floor, status,
        reinterpret_cast<unsigned long long *>(header + 1));
    if (result != 0) return result;
    const MxxGraphPatch latch_patch = stat_patch(0, status_binding);
    return mxx_gpu_launch_kernel(ctx, stream, preimage_status_latch_accept_kernel, dim3(1),
        dim3(1), 0, &latch_patch, 1, status);
}

// The perturbation `[p_1; p_2]` (`m_bar + digits` rows) of one tile into `output`, every limb.
// `centers` holds `digits x columns` vectors of `n` complex values, `buffer` `m_bar x columns`,
// and `p2` the integers of `p_2`.
extern "C" int gpu_raw_statistical_perturbation(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *output, const void *seed_raw, const uint8_t *precomputation,
    size_t precomputation_bytes, double2 *centers, size_t centers_bytes, double2 *buffer,
    size_t buffer_bytes, int64_t *p2, size_t p2_bytes, size_t m_bar, double p2_sigma,
    double center_scale, double rounding, uint32_t output_binding_base, uint32_t seed_binding,
    uint32_t precomputation_binding, uint32_t centers_binding, uint32_t buffer_binding,
    uint32_t p2_binding)
{
    const auto *seed = static_cast<const GpuRngSeed *>(seed_raw);
    if (validate_raw_view(ctx, output, stream_raw) != 0 || !seed || !precomputation ||
        !centers || !buffer || !p2 || output->rows <= m_bar || m_bar == 0 ||
        output->limb_count > kRawNttLimbs ||
        output_binding_base > UINT32_MAX - output->limb_count)
        return set_error("invalid statistical perturbation arguments");
    const size_t digits = output->rows - m_bar;
    const size_t columns = output->columns;
    const uint32_t n = output->degree;
    const size_t points = n / 2;
    const size_t block = digits * columns * n;
    if (precomputation_bytes != stat_precomputation_bytes(n, m_bar, digits) ||
        block * sizeof(double2) > centers_bytes ||
        m_bar * columns * n * sizeof(double2) > buffer_bytes ||
        block * sizeof(int64_t) > p2_bytes || !(rounding > 0.0))
        return set_error("statistical perturbation workspace is too small");
    if (mxx_set_device(output->physical_device) != gpuSuccess)
        return set_error(gpuGetLastError());
    const auto stream = reinterpret_cast<gpuStream_t>(stream_raw);
    const uint32_t limbs = static_cast<uint32_t>(output->limb_count);
    const auto *spectra =
        reinterpret_cast<const double2 *>(precomputation + kStatisticalHeaderBytes);
    const auto *factor = reinterpret_cast<const double2 *>(
        precomputation + kStatisticalHeaderBytes + 16 * points * m_bar * digits);
    // p_2 and the transformed centers `center_scale p_2`.
    {
        RawLimbSet outputs{};
        std::vector<MxxGraphPatch> patches;
        raw_limb_set(output, 0, output->limb_count, 0, output_binding_base, outputs, patches);
        patches.push_back(stat_patch(2, seed_binding));
        patches.push_back(stat_patch(8, p2_binding));
        patches.push_back(stat_patch(9, centers_binding));
        int status = mxx_gpu_launch_kernel(ctx, stream, stat_sample_p2_kernel, stat_grid(block),
            dim3(kStatisticalThreads), 0, patches.data(), patches.size(), outputs, limbs, seed,
            m_bar, digits, columns, n, p2_sigma, p2, centers, center_scale);
        if (status != 0) return status;
        status = stat_fft(ctx, stream, centers, centers_binding, digits * columns, n, 1.0);
        if (status != 0) return status;
    }
    const size_t entries = m_bar * columns;
    {
        const MxxGraphPatch patches[] = {stat_patch(0, seed_binding), stat_patch(3, buffer_binding)};
        const int status = mxx_gpu_launch_kernel(ctx, stream, stat_normal_kernel,
            stat_grid(entries * points), dim3(kStatisticalThreads), 0, patches,
            std::size(patches), seed, entries, n, buffer);
        if (status != 0) return status;
    }
    {
        const MxxGraphPatch patches[] = {stat_patch(0, precomputation_binding),
            stat_patch(1, precomputation_binding), stat_patch(2, centers_binding),
            stat_patch(7, buffer_binding)};
        const int status = mxx_gpu_launch_kernel(ctx, stream, stat_combine_kernel,
            stat_grid(columns * points), dim3(kStatisticalThreads), 0, patches,
            std::size(patches), spectra, factor, static_cast<const double2 *>(centers), m_bar,
            digits, columns, n, buffer);
        if (status != 0) return status;
    }
    int status = stat_fft(ctx, stream, buffer, buffer_binding, entries, n, -1.0);
    if (status != 0) return status;
    RawLimbSet outputs{};
    std::vector<MxxGraphPatch> patches = {stat_patch(0, buffer_binding),
        stat_patch(1, seed_binding)};
    raw_limb_set(output, 0, output->limb_count, 5, output_binding_base, outputs, patches);
    return mxx_gpu_launch_kernel(ctx, stream, stat_round_rows_kernel,
        stat_grid(entries * n), dim3(kStatisticalThreads), 0, patches.data(), patches.size(),
        static_cast<const double2 *>(buffer), seed, entries, n, rounding, outputs, limbs,
        columns);
}
