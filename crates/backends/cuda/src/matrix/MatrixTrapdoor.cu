constexpr size_t kSampleP1LocalMaxM = 8;

using gpu_chacha::GpuRngSeed;

namespace
{
MxxGraphPatch trapdoor_pointer_patch(uint32_t argument_index, uint32_t binding_index)
{
    MxxGraphPatch patch{};
    patch.node = nullptr;
    patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
    patch.argument_index = argument_index;
    patch.byte_offset = 0;
    patch.byte_count = sizeof(void *);
    patch.binding_index = binding_index;
    patch.address_addend = 0;
    return patch;
}

    struct ThreadLocalOwnerLinkEventState
    {
        int device = -1;
        cudaEvent_t event = nullptr;

        ~ThreadLocalOwnerLinkEventState()
        {
            if (!event || device < 0)
            {
                return;
            }
            cudaError_t err = cudaSetDevice(device);
            if (err == cudaSuccess)
            {
                cudaEventDestroy(event);
            }
            event = nullptr;
            device = -1;
        }
    };

    thread_local ThreadLocalOwnerLinkEventState g_thread_local_owner_link_event;

}

namespace
{
    // These are CUDA block-shape constants, not a preimage chunk limit. Every
    // kernel below receives the runtime column count selected by
    // AUX_SAMPLING_CHUNK_WIDTH and covers any final partial tile.
    constexpr int kPreimageTileM = 4;
    constexpr int kPreimageTileN = 32;
    constexpr int kPreimageTileK = 16;

    __global__ void matrix_preimage_add_correction_top_kernel(
        const uint8_t *r_base,
        const uint8_t *e_base,
        const uint8_t *z_base,
        uint8_t *out_base,
        size_t d,
        size_t inner,
        size_t z_cols,
        size_t out_cols,
        size_t n,
        size_t r_stride_bytes,
        size_t e_stride_bytes,
        size_t z_stride_bytes,
        size_t out_stride_bytes,
        uint8_t r_coeff_bytes,
        uint8_t e_coeff_bytes,
        uint8_t z_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        __shared__ uint64_t trapdoor_tile[kPreimageTileM][kPreimageTileK];
        __shared__ uint64_t z_tile[kPreimageTileK][kPreimageTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kPreimageTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kPreimageTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t top_rows = 2 * d;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;

        for (size_t coeff_idx = static_cast<size_t>(blockIdx.z);
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z))
        {
            uint64_t acc = 0;
            if (row < top_rows && col < out_cols)
            {
                acc = matrix_load_limb_u64(
                    out_base,
                    row * out_cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes);
            }

            for (size_t k0 = 0; k0 < inner; k0 += kPreimageTileK)
            {
                for (int i = tid; i < kPreimageTileM * kPreimageTileK; i += thread_count)
                {
                    const int local_row = i / kPreimageTileK;
                    const int local_k = i - local_row * kPreimageTileK;
                    const size_t input_row = row_base + static_cast<size_t>(local_row);
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    uint64_t value = 0;
                    if (input_row < top_rows && input_k < inner)
                    {
                        if (input_row < d)
                        {
                            value = matrix_load_limb_u64(
                                r_base,
                                input_row * inner + input_k,
                                coeff_idx,
                                r_stride_bytes,
                                r_coeff_bytes);
                        }
                        else
                        {
                            value = matrix_load_limb_u64(
                                e_base,
                                (input_row - d) * inner + input_k,
                                coeff_idx,
                                e_stride_bytes,
                                e_coeff_bytes);
                        }
                    }
                    trapdoor_tile[local_row][local_k] = value;
                }
                for (int i = tid; i < kPreimageTileK * kPreimageTileN; i += thread_count)
                {
                    const int local_k = i / kPreimageTileN;
                    const int local_col = i - local_k * kPreimageTileN;
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    const size_t input_col = col_base + static_cast<size_t>(local_col);
                    uint64_t value = 0;
                    if (input_k < inner && input_col < out_cols)
                    {
                        value = matrix_load_limb_u64(
                            z_base,
                            input_k * z_cols + input_col,
                            coeff_idx,
                            z_stride_bytes,
                            z_coeff_bytes);
                    }
                    z_tile[local_k][local_col] = value;
                }
                __syncthreads();

                if (row < top_rows && col < out_cols)
                {
                    for (int local_k = 0; local_k < kPreimageTileK; ++local_k)
                    {
                        acc = add_mod_u64(
                            acc,
                            mul_mod_u64(
                                trapdoor_tile[threadIdx.y][local_k],
                                z_tile[local_k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }

            if (row < top_rows && col < out_cols)
            {
                matrix_store_limb_u64(
                    out_base,
                    row * out_cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes,
                    acc);
            }
        }
    }

}

__global__ void matrix_precompute_p1_covariance_kernel(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    size_t d,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    double *cov_workspace,
    double *sqrt_var_out,
    double *update_coeff_out)
{
    const size_t coeff_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff_idx >= n)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0)
    {
        return;
    }

    double *cov = cov_workspace + coeff_idx * m * m;
    double *sqrt_var = sqrt_var_out + coeff_idx * m;
    double *update_coeff = update_coeff_out + coeff_idx * m * m;

    const double sigma2 = sigma * sigma;
    const double s2 = s * s;
    const double fallback_var = dgg_stddev * dgg_stddev;
    const double eps = 1e-9;

    for (size_t i = 0; i < d; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            const size_t ij = matrix_index(i, j, d);
            const size_t ji = matrix_index(j, i, d);
            const double a_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(a_base, ij, coeff_idx, a_stride_bytes, a_coeff_bytes),
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(d_base, ij, coeff_idx, d_stride_bytes, d_coeff_bytes),
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ij, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ji, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));

            cov[matrix_index(i, j, m)] = -sigma2 * a_ij + (i == j ? s2 : 0.0);
            cov[matrix_index(i + d, j + d, m)] = -sigma2 * d_ij + (i == j ? s2 : 0.0);
            cov[matrix_index(i, j + d, m)] = -sigma2 * b_ij;
            cov[matrix_index(i + d, j, m)] = -sigma2 * b_ji;
        }
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        double var = cov[matrix_index(tt, tt, m)];
        if (!(var > eps))
        {
            var = fallback_var;
        }
        sqrt_var[tt] = sqrt(var);

        for (int i = 0; i < t; ++i)
        {
            update_coeff[tt * m + static_cast<size_t>(i)] =
                cov[matrix_index(static_cast<size_t>(i), tt, m)] / var;
        }

        if (t == 0)
        {
            break;
        }

        for (int i = 0; i < t; ++i)
        {
            const double coeff_i = update_coeff[tt * m + static_cast<size_t>(i)];
            for (int j = 0; j <= i; ++j)
            {
                const double col_j =
                    update_coeff[tt * m + static_cast<size_t>(j)] * var;
                double updated =
                    cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                    coeff_i * col_j;
                cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
            }
        }
    }
}

__device__ __forceinline__ void matrix_sample_p1_integer_cached_kernel_small_body(
    const uint8_t *tp2_base,
    size_t tp2_stride_bytes,
    uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base,
    const double *update_coeff_base,
    size_t d,
    size_t cols,
    size_t n,
    int64_t *sampled_out,
    uint64_t modulus,
    double c_scale,
    GpuRngSeed seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_samples = cols * n;
    if (idx >= total_samples)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0 || m > kSampleP1LocalMaxM)
    {
        return;
    }

    const size_t col_idx = idx / n;
    const size_t coeff_idx = idx - col_idx * n;
    double mean[kSampleP1LocalMaxM];
    int64_t sampled[kSampleP1LocalMaxM];
    const double *sqrt_var = sqrt_var_base + coeff_idx * m;
    const double *update_coeff = update_coeff_base + coeff_idx * m * m;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt_var[tt]);
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                update_coeff[tt * m + static_cast<size_t>(i)] * delta;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__device__ __forceinline__ void matrix_sample_p1_integer_cached_kernel_large_body(
    const uint8_t *tp2_base,
    size_t tp2_stride_bytes,
    uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base,
    const double *update_coeff_base,
    size_t d,
    size_t cols,
    size_t n,
    size_t sample_start,
    size_t sample_count,
    double *mean_workspace,
    int64_t *sampled_workspace,
    int64_t *sampled_out,
    uint64_t modulus,
    double c_scale,
    GpuRngSeed seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= sample_count)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0)
    {
        return;
    }

    const size_t sample_idx = sample_start + idx;
    const size_t col_idx = sample_idx / n;
    const size_t coeff_idx = sample_idx - col_idx * n;
    double *mean = mean_workspace + idx * m;
    int64_t *sampled = sampled_workspace + idx * m;
    const double *sqrt_var = sqrt_var_base + coeff_idx * m;
    const double *update_coeff = update_coeff_base + coeff_idx * m * m;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt_var[tt]);
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                update_coeff[tt * m + static_cast<size_t>(i)] * delta;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_sample_p1_integer_cached_kernel_small_device_seed(
    const uint8_t *tp2_base, size_t tp2_stride_bytes, uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base, const double *update_coeff_base, size_t d,
    size_t cols, size_t n, int64_t *sampled_out, uint64_t modulus,
    double c_scale, const GpuRngSeed *device_seed)
{
    if (device_seed)
        matrix_sample_p1_integer_cached_kernel_small_body(
            tp2_base, tp2_stride_bytes, tp2_coeff_bytes, sqrt_var_base,
            update_coeff_base, d, cols, n, sampled_out, modulus, c_scale,
            *device_seed);
}

__global__ void matrix_sample_p1_integer_cached_kernel_large_device_seed(
    const uint8_t *tp2_base, size_t tp2_stride_bytes, uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base, const double *update_coeff_base, size_t d,
    size_t cols, size_t n, size_t sample_start, size_t sample_count,
    double *mean_workspace, int64_t *sampled_workspace, int64_t *sampled_out,
    uint64_t modulus, double c_scale, const GpuRngSeed *device_seed)
{
    if (device_seed)
        matrix_sample_p1_integer_cached_kernel_large_body(
            tp2_base, tp2_stride_bytes, tp2_coeff_bytes, sqrt_var_base,
            update_coeff_base, d, cols, n, sample_start, sample_count,
            mean_workspace, sampled_workspace, sampled_out, modulus, c_scale,
            *device_seed);
}

__global__ void matrix_scatter_p1_integer_to_limb_kernel(
    const int64_t *sampled_in,
    uint8_t *out_base,
    size_t out_stride_bytes,
    uint8_t out_coeff_bytes,
    size_t entry_count,
    size_t n,
    uint64_t modulus)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = entry_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t entry_idx = idx / n;
    const size_t coeff_idx = idx - entry_idx * n;
    matrix_store_limb_u64(
        out_base,
        entry_idx,
        coeff_idx,
        out_stride_bytes,
        out_coeff_bytes,
        signed_mod_i64(sampled_in[idx], modulus));
}

__device__ __forceinline__ void matrix_gauss_samp_gq_arb_base_sample_kernel_body(
    const uint8_t *src_base,
    int64_t *sampled_digits,
    size_t poly_count,
    size_t n,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    uint64_t tower_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    double c,
    uint32_t tower_idx,
    GpuRngSeed seed)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
    {
        return;
    }

    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;

    uint64_t value =
        matrix_load_limb_u64(src_base, poly_idx, coeff_idx, src_stride_bytes, src_coeff_bytes);
    if (tower_modulus != 0)
    {
        value %= tower_modulus;
    }

    uint64_t base = uint64_t{1} << base_bits;
    double base_f = static_cast<double>(base);
    double sigma = c / (base_f + 1.0);

    int64_t m_digits[kGaussMaxDigits];
    int64_t v_digits[kGaussMaxDigits];
    double l[kGaussMaxDigits];
    double h[kGaussMaxDigits];
    double c_vec[kGaussMaxDigits];
    double p[kGaussMaxDigits];
    double a[kGaussMaxDigits];
    double zf[kGaussMaxDigits];
    int64_t z[kGaussMaxDigits];

    get_base_digits_u64(tower_modulus, base, digits_per_tower, m_digits);
    get_base_digits_u64(value, base, digits_per_tower, v_digits);

    const double kf = static_cast<double>(digits_per_tower);
    l[0] = sqrt(base_f * (1.0 + 1.0 / kf) + 1.0);
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        l[i] = sqrt(base_f * (1.0 + 1.0 / (kf - static_cast<double>(i))));
    }

    h[0] = 0.0;
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        h[i] = sqrt(base_f * (1.0 - 1.0 / (kf - static_cast<double>(i - 1))));
    }

    c_vec[0] = static_cast<double>(m_digits[0]) / base_f;
    for (uint32_t i = 1; i < digits_per_tower; ++i)
    {
        c_vec[i] = (c_vec[i - 1] + static_cast<double>(m_digits[i])) / base_f;
    }

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(tower_idx + 1),
        static_cast<uint64_t>(poly_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0x6761646765746731ULL);

    for (uint32_t i = 0; i < digits_per_tower; ++i)
    {
        zf[i] = sigma * sample_standard_normal(rng);
    }
    for (uint32_t i = 0; i + 1 < digits_per_tower; ++i)
    {
        p[i] = l[i] * zf[i] + h[i + 1] * zf[i + 1];
    }
    p[digits_per_tower - 1] = h[digits_per_tower - 1] * zf[digits_per_tower - 1];

    a[0] = (static_cast<double>(v_digits[0]) - p[0]) / base_f;
    for (uint32_t t = 1; t < digits_per_tower; ++t)
    {
        a[t] = (a[t - 1] + static_cast<double>(v_digits[t]) - p[t]) / base_f;
    }

    const uint32_t last = digits_per_tower - 1;
    z[last] = sample_integer_karney(rng, -a[last] / c_vec[last], sigma / c_vec[last]);
    for (uint32_t i = 0; i < digits_per_tower; ++i)
    {
        a[i] += static_cast<double>(z[last]) * c_vec[i];
    }
    for (uint32_t i = 0; i < last; ++i)
    {
        z[i] = sample_integer_karney(rng, -a[i], sigma);
    }

    for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
    {
        int64_t out_digit = 0;
        if (digits_per_tower == 1)
        {
            out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[0] + v_digits[0];
        }
        else if (digit_idx == 0)
        {
            out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[last] + v_digits[0];
        }
        else if (digit_idx < last)
        {
            out_digit = static_cast<int64_t>(base) * z[digit_idx] - z[digit_idx - 1] +
                        m_digits[digit_idx] * z[last] + v_digits[digit_idx];
        }
        else
        {
            out_digit = m_digits[last] * z[last] - z[last - 1] + v_digits[last];
        }

        const size_t sample_idx =
            (poly_idx * static_cast<size_t>(digits_per_tower) + static_cast<size_t>(digit_idx)) * n + coeff_idx;
        sampled_digits[sample_idx] = out_digit;
    }
}

__global__ void matrix_gauss_samp_gq_arb_base_sample_kernel_device_seed(
    const uint8_t *src_base, int64_t *sampled_digits, size_t poly_count,
    size_t n, size_t src_stride_bytes, uint8_t src_coeff_bytes,
    uint64_t tower_modulus, uint32_t base_bits, uint32_t digits_per_tower,
    double c, uint32_t tower_idx, const GpuRngSeed *device_seed)
{
    if (device_seed)
        matrix_gauss_samp_gq_arb_base_sample_kernel_body(
            src_base, sampled_digits, poly_count, n, src_stride_bytes,
            src_coeff_bytes, tower_modulus, base_bits, digits_per_tower, c,
            tower_idx, *device_seed);
}

// Fixed preimage covariance storage is allocated by the compiled plan. These
// direct entry points reuse the existing P1 arithmetic kernels and never
// inspect or mutate GpuMatrix::format during graph construction.
static bool raw_p1_contiguous_limb(const MxxRawMatrixView *view,
    size_t limb_index)
{
    if (!view || limb_index >= view->limb_count) return false;
    const auto &limb = view->limbs[limb_index];
    return limb.coefficient_stride_bytes == limb.word_bytes &&
        limb.column_stride_bytes >=
            static_cast<uint64_t>(view->degree) * limb.word_bytes &&
        limb.row_stride_bytes == view->columns * limb.column_stride_bytes;
}

extern "C" int gpu_raw_p1_covariance_refresh(
    GpuContext *ctx, void *stream_raw, const MxxRawMatrixView *a,
    const MxxRawMatrixView *b, const MxxRawMatrixView *d,
    double sigma, double s, double dgg_stddev,
    void *cov_workspace, double *sqrt_var, double *update_coeff,
    uint32_t a_binding, uint32_t b_binding, uint32_t d_binding,
    uint32_t cov_binding, uint32_t sqrt_binding, uint32_t update_binding)
{
    if (validate_raw_view(ctx, a, stream_raw) != 0 ||
        validate_raw_view(ctx, b, stream_raw) != 0 ||
        validate_raw_view(ctx, d, stream_raw) != 0 ||
        a->limb_count != b->limb_count || a->limb_count != d->limb_count ||
        a->physical_device != b->physical_device ||
        a->physical_device != d->physical_device ||
        a->rows != a->columns || b->rows != a->rows ||
        b->columns != a->columns || d->rows != a->rows ||
        d->columns != a->columns || !raw_p1_contiguous_limb(a, 0) ||
        !raw_p1_contiguous_limb(b, 0) || !raw_p1_contiguous_limb(d, 0) ||
        a->limbs[0].crt_limb_index != 0 ||
        b->limbs[0].crt_limb_index != 0 ||
        d->limbs[0].crt_limb_index != 0 ||
        !cov_workspace || !sqrt_var || !update_coeff ||
        !(sigma > 0.0) || !(s > sigma) || !(dgg_stddev > 0.0))
        return set_error("invalid raw P1 covariance refresh");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const auto *a_base = reinterpret_cast<const uint8_t *>(a->limbs[0].address);
    const auto *b_base = reinterpret_cast<const uint8_t *>(b->limbs[0].address);
    const auto *d_base = reinterpret_cast<const uint8_t *>(d->limbs[0].address);
    const size_t n = a->degree;
    const size_t rows = a->rows;
    const uint64_t modulus = a->limbs[0].modulus;
    const MxxGraphPatch patches[] = {
        trapdoor_pointer_patch(0, a_binding),
        trapdoor_pointer_patch(1, b_binding),
        trapdoor_pointer_patch(2, d_binding),
        trapdoor_pointer_patch(15, cov_binding),
        trapdoor_pointer_patch(16, sqrt_binding),
        trapdoor_pointer_patch(17, update_binding),
    };
    return mxx_gpu_launch_kernel(ctx, stream,
        matrix_precompute_p1_covariance_kernel,
        dim3(static_cast<uint32_t>((n + 255) / 256)), dim3(256), 0,
        patches, std::size(patches), a_base, b_base, d_base,
        static_cast<size_t>(a->limbs[0].column_stride_bytes),
        static_cast<size_t>(b->limbs[0].column_stride_bytes),
        static_cast<size_t>(d->limbs[0].column_stride_bytes),
        static_cast<uint8_t>(a->limbs[0].word_bytes),
        static_cast<uint8_t>(b->limbs[0].word_bytes),
        static_cast<uint8_t>(d->limbs[0].word_bytes),
        rows, n, modulus, sigma, s, dgg_stddev,
        static_cast<double *>(cov_workspace), sqrt_var, update_coeff);
}

extern "C" int gpu_raw_p1_sample(
    GpuContext *ctx, void *stream_raw, const MxxRawMatrixView *tp2,
    const MxxRawMatrixView *output, const void *seed_raw,
    int64_t *sampled, size_t sampled_bytes, void *workspace,
    size_t workspace_bytes, const double *sqrt_var,
    const double *update_coeff, double sigma, double s,
    uint32_t tp2_binding, uint32_t output_binding_base,
    uint32_t seed_binding, uint32_t sampled_binding,
    uint32_t workspace_binding, uint32_t sqrt_binding,
    uint32_t update_binding)
{
    const auto *seed = static_cast<const GpuRngSeed *>(seed_raw);
    if (validate_raw_view(ctx, tp2, stream_raw) != 0 ||
        validate_raw_view(ctx, output, stream_raw) != 0 ||
        !seed || !sampled || !sqrt_var || !update_coeff ||
        !(sigma > 0.0) || !(s > sigma) ||
        tp2->rows != output->rows || tp2->columns != output->columns ||
        tp2->limb_count != output->limb_count ||
        tp2->rows == 0 || tp2->rows % 2 != 0 ||
        tp2->limbs[0].crt_limb_index != 0 ||
        !raw_p1_contiguous_limb(tp2, 0) ||
        tp2->physical_device != output->physical_device ||
        output_binding_base > UINT32_MAX - output->limb_count)
        return set_error("invalid raw P1 sample views");
    for (size_t limb = 0; limb < output->limb_count; ++limb)
        if (!raw_p1_contiguous_limb(output, limb) ||
            output->limbs[limb].crt_limb_index !=
                tp2->limbs[limb].crt_limb_index ||
            output->limbs[limb].modulus != tp2->limbs[limb].modulus)
            return set_error("invalid raw P1 output limb");
    const size_t m = tp2->rows;
    const size_t columns = tp2->columns;
    const size_t n = tp2->degree;
    if (columns > SIZE_MAX / n || m > SIZE_MAX / (columns * n) ||
        m * columns * n > sampled_bytes / sizeof(int64_t))
        return set_error("raw P1 sampled buffer is too small");
    const size_t total_samples = columns * n;
    if (total_samples > static_cast<size_t>(UINT32_MAX) * 256 ||
        m * total_samples > static_cast<size_t>(UINT32_MAX) * 256)
        return set_error("raw P1 grid exceeds device range");
    if (m > kSampleP1LocalMaxM &&
        (total_samples > SIZE_MAX / m ||
         total_samples * m > workspace_bytes / (sizeof(double) + sizeof(int64_t)) ||
         !workspace))
        return set_error("raw P1 workspace is too small");
    if (cudaSetDevice(tp2->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const auto *tp2_base = reinterpret_cast<const uint8_t *>(tp2->limbs[0].address);
    const size_t stride = tp2->limbs[0].column_stride_bytes;
    const uint8_t width = static_cast<uint8_t>(tp2->limbs[0].word_bytes);
    const size_t d_rows = m / 2;
    const uint64_t modulus = tp2->limbs[0].modulus;
    const double c_scale = -(sigma * sigma) / (s * s - sigma * sigma);
    const dim3 grid(static_cast<uint32_t>((total_samples + 255) / 256));
    int result = 0;
    if (m <= kSampleP1LocalMaxM)
    {
        const MxxGraphPatch patches[] = {
            trapdoor_pointer_patch(0, tp2_binding),
            trapdoor_pointer_patch(3, sqrt_binding),
            trapdoor_pointer_patch(4, update_binding),
            trapdoor_pointer_patch(8, sampled_binding),
            trapdoor_pointer_patch(11, seed_binding),
        };
        result = mxx_gpu_launch_kernel(ctx, stream,
            matrix_sample_p1_integer_cached_kernel_small_device_seed,
            grid, dim3(256), 0, patches, std::size(patches),
            tp2_base, stride, width, sqrt_var, update_coeff,
            d_rows, columns, n, sampled, modulus, c_scale, seed);
    }
    else
    {
        auto *mean_workspace = static_cast<double *>(workspace);
        auto *sampled_workspace = reinterpret_cast<int64_t *>(
            static_cast<uint8_t *>(workspace) + total_samples * m * sizeof(double));
        const size_t sample_start = 0;
        const MxxGraphPatch patches[] = {
            trapdoor_pointer_patch(0, tp2_binding),
            trapdoor_pointer_patch(3, sqrt_binding),
            trapdoor_pointer_patch(4, update_binding),
            trapdoor_pointer_patch(10, workspace_binding),
            trapdoor_pointer_patch(11, workspace_binding),
            trapdoor_pointer_patch(12, sampled_binding),
            trapdoor_pointer_patch(15, seed_binding),
        };
        result = mxx_gpu_launch_kernel(ctx, stream,
            matrix_sample_p1_integer_cached_kernel_large_device_seed,
            grid, dim3(256), 0, patches, std::size(patches),
            tp2_base, stride, width, sqrt_var, update_coeff,
            d_rows, columns, n, sample_start, total_samples,
            mean_workspace, sampled_workspace, sampled, modulus, c_scale, seed);
    }
    if (result != 0) return result;
    const size_t entry_count = m * columns;
    const size_t total = entry_count * n;
    const dim3 scatter_grid(static_cast<uint32_t>((total + 255) / 256));
    for (size_t limb = 0; limb < output->limb_count; ++limb)
    {
        auto *out_base = reinterpret_cast<uint8_t *>(output->limbs[limb].address);
        const size_t out_stride = output->limbs[limb].column_stride_bytes;
        const uint8_t out_width = static_cast<uint8_t>(output->limbs[limb].word_bytes);
        const uint64_t out_modulus = output->limbs[limb].modulus;
        const MxxGraphPatch patches[] = {
            trapdoor_pointer_patch(0, sampled_binding),
            trapdoor_pointer_patch(1,
                static_cast<uint32_t>(output_binding_base + limb)),
        };
        result = mxx_gpu_launch_kernel(ctx, stream,
            matrix_scatter_p1_integer_to_limb_kernel,
            scatter_grid, dim3(256), 0, patches, std::size(patches),
            sampled, out_base, out_stride, out_width,
            entry_count, n, out_modulus);
        if (result != 0) return result;
    }
    return 0;
}

__global__ void raw_gq_scatter_kernel(
    const int64_t *sampled_digits, MxxRawMatrixLimb destination,
    size_t source_columns, size_t output_columns, size_t poly_count,
    size_t degree, size_t log_base_q, size_t source_digit_offset,
    uint32_t digits_per_tower)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= poly_count * degree) return;
    const size_t poly = index / degree;
    const size_t coefficient = index - poly * degree;
    const size_t row = poly / source_columns;
    const size_t column = poly - row * source_columns;
    for (uint32_t digit = 0; digit < digits_per_tower; ++digit)
    {
        const int64_t value = sampled_digits[
            (poly * digits_per_tower + digit) * degree + coefficient];
        const size_t output_row = row * log_base_q + source_digit_offset + digit;
        const size_t output_poly = output_row * output_columns + column;
        raw_matrix_store(destination, output_poly, coefficient, output_columns,
            signed_mod_i64(value, destination.modulus));
    }
}

extern "C" int gpu_raw_gq_sample(
    GpuContext *ctx, void *stream_raw, const MxxRawMatrixView *source,
    const MxxRawMatrixView *destination, const void *seed_raw,
    int64_t *sampled, size_t sampled_bytes, uint32_t base_bits,
    double c, uint32_t source_binding_base,
    uint32_t destination_binding_base, uint32_t seed_binding,
    uint32_t sampled_binding)
{
    const auto *seed = static_cast<const GpuRngSeed *>(seed_raw);
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !seed || !sampled || !(c > 0.0) || base_bits == 0 ||
        base_bits >= 63 || source->physical_device != destination->physical_device ||
        source->limb_count != ctx->moduli.size() ||
        destination->limb_count != source->limb_count ||
        source->row_origin != 0 || source->column_origin != 0 ||
        destination->row_origin != 0 || destination->column_origin != 0 ||
        source->columns != destination->columns ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw GQ views");
    uint32_t crt_bits = 0;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        if (!raw_p1_contiguous_limb(source, limb) ||
            !raw_p1_contiguous_limb(destination, limb) ||
            source->limbs[limb].crt_limb_index != limb ||
            destination->limbs[limb].crt_limb_index != limb ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus)
            return set_error("raw GQ requires ordered contiguous CRT limbs");
        crt_bits = std::max(crt_bits, bit_width_u64(source->limbs[limb].modulus));
    }
    const uint32_t digits = (crt_bits + base_bits - 1) / base_bits;
    if (!digits || digits > kGaussMaxDigits ||
        source->limb_count > SIZE_MAX / digits)
        return set_error("invalid raw GQ digit count");
    const size_t log_base_q = source->limb_count * digits;
    if (source->rows > SIZE_MAX / log_base_q ||
        destination->rows != source->rows * log_base_q ||
        source->rows > SIZE_MAX / source->columns)
        return set_error("invalid raw GQ output shape");
    const size_t poly_count = source->rows * source->columns;
    if (poly_count > SIZE_MAX / source->degree ||
        poly_count * source->degree > SIZE_MAX / digits ||
        poly_count * source->degree * digits > sampled_bytes / sizeof(int64_t) ||
        poly_count * source->degree > static_cast<size_t>(UINT32_MAX) * 256)
        return set_error("raw GQ sampled workspace too small");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t count = poly_count * source->degree;
    const dim3 grid(static_cast<uint32_t>((count + 255) / 256));
    for (size_t source_limb = 0; source_limb < source->limb_count; ++source_limb)
    {
        const auto &limb = source->limbs[source_limb];
        const auto *source_base = reinterpret_cast<const uint8_t *>(limb.address);
        const size_t source_stride = limb.column_stride_bytes;
        const uint8_t source_width = static_cast<uint8_t>(limb.word_bytes);
        const uint64_t modulus = limb.modulus;
        const uint32_t tower = static_cast<uint32_t>(source_limb);
        const MxxGraphPatch sample_patches[] = {
            trapdoor_pointer_patch(0,
                static_cast<uint32_t>(source_binding_base + source_limb)),
            trapdoor_pointer_patch(1, sampled_binding),
            trapdoor_pointer_patch(11, seed_binding),
        };
        int result = mxx_gpu_launch_kernel(ctx, stream,
            matrix_gauss_samp_gq_arb_base_sample_kernel_device_seed,
            grid, dim3(256), 0, sample_patches, std::size(sample_patches),
            source_base, sampled, poly_count, static_cast<size_t>(source->degree),
            source_stride, source_width, modulus, base_bits, digits,
            c, tower, seed);
        if (result != 0) return result;
        for (size_t output_limb = 0; output_limb < destination->limb_count; ++output_limb)
        {
            const MxxGraphPatch scatter_patches[] = {
                trapdoor_pointer_patch(0, sampled_binding),
                trapdoor_pointer_patch(1,
                    static_cast<uint32_t>(destination_binding_base + output_limb)),
            };
            result = mxx_gpu_launch_kernel(ctx, stream, raw_gq_scatter_kernel,
                grid, dim3(256), 0, scatter_patches, std::size(scatter_patches),
                sampled, destination->limbs[output_limb], source->columns,
                destination->columns, poly_count,
                static_cast<size_t>(source->degree), log_base_q,
                source_limb * static_cast<size_t>(digits), digits);
            if (result != 0) return result;
        }
    }
    return 0;
}

