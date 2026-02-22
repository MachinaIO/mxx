constexpr size_t kSampleP1LocalMaxM = 8;

__global__ void matrix_sample_p1_integer_kernel_small(
    const uint64_t *a_base,
    const uint64_t *b_base,
    const uint64_t *d_base,
    const uint64_t *tp2_base,
    size_t a_stride,
    size_t b_stride,
    size_t d_stride,
    size_t tp2_stride,
    size_t d,
    size_t cols,
    size_t n,
    int64_t *sampled_out,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed)
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
    double cov[kSampleP1LocalMaxM * kSampleP1LocalMaxM];
    double mean[kSampleP1LocalMaxM];
    double col_buf[kSampleP1LocalMaxM];
    int64_t sampled[kSampleP1LocalMaxM];

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    const double sigma2 = sigma * sigma;
    const double s2 = s * s;
    const double denom = s2 - sigma2;
    if (!(denom > 0.0))
    {
        return;
    }
    const double c_scale = -sigma2 / denom;
    const double fallback_var = dgg_stddev * dgg_stddev;
    const double eps = 1e-9;

    for (size_t i = 0; i < d; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            const size_t ij = matrix_index(i, j, d);
            const size_t ji = matrix_index(j, i, d);
            const double a_ij = static_cast<double>(centered_residue_i64(
                a_base[ij * a_stride + coeff_idx],
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                d_base[ij * d_stride + coeff_idx],
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                b_base[ij * b_stride + coeff_idx],
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                b_base[ji * b_stride + coeff_idx],
                modulus));

            const double af = -sigma2 * a_ij + (i == j ? s2 : 0.0);
            const double df = -sigma2 * d_ij + (i == j ? s2 : 0.0);
            const double bf = -sigma2 * b_ij;
            const double bt = -sigma2 * b_ji;

            cov[matrix_index(i, j, m)] = af;
            cov[matrix_index(i + d, j + d, m)] = df;
            cov[matrix_index(i, j + d, m)] = bf;
            cov[matrix_index(i + d, j, m)] = bt;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            tp2_base[tp_idx * tp2_stride + coeff_idx],
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        double var = cov[matrix_index(tt, tt, m)];
        if (!(var > eps))
        {
            var = fallback_var;
        }
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt(var));
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            col_buf[static_cast<size_t>(i)] =
                cov[matrix_index(static_cast<size_t>(i), tt, m)];
        }

        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                (col_buf[static_cast<size_t>(i)] / var) * delta;
        }

        for (int i = 0; i < t; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                double updated = cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                                 (col_buf[static_cast<size_t>(i)] * col_buf[static_cast<size_t>(j)] / var);
                cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
            }
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_sample_p1_integer_kernel_large(
    const uint64_t *a_base,
    const uint64_t *b_base,
    const uint64_t *d_base,
    const uint64_t *tp2_base,
    size_t a_stride,
    size_t b_stride,
    size_t d_stride,
    size_t tp2_stride,
    size_t d,
    size_t cols,
    size_t n,
    size_t sample_start,
    size_t sample_count,
    double *cov_workspace,
    double *mean_workspace,
    double *col_workspace,
    int64_t *sampled_workspace,
    int64_t *sampled_out,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed)
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
    const size_t cov_stride = m * m;
    const size_t vec_stride = m;
    double *cov = cov_workspace + idx * cov_stride;
    double *mean = mean_workspace + idx * vec_stride;
    double *col_buf = col_workspace + idx * vec_stride;
    int64_t *sampled = sampled_workspace + idx * vec_stride;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    const double sigma2 = sigma * sigma;
    const double s2 = s * s;
    const double denom = s2 - sigma2;
    if (!(denom > 0.0))
    {
        return;
    }
    const double c_scale = -sigma2 / denom;
    const double fallback_var = dgg_stddev * dgg_stddev;
    const double eps = 1e-9;

    for (size_t i = 0; i < d; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            const size_t ij = matrix_index(i, j, d);
            const size_t ji = matrix_index(j, i, d);
            const double a_ij = static_cast<double>(centered_residue_i64(
                a_base[ij * a_stride + coeff_idx],
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                d_base[ij * d_stride + coeff_idx],
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                b_base[ij * b_stride + coeff_idx],
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                b_base[ji * b_stride + coeff_idx],
                modulus));

            const double af = -sigma2 * a_ij + (i == j ? s2 : 0.0);
            const double df = -sigma2 * d_ij + (i == j ? s2 : 0.0);
            const double bf = -sigma2 * b_ij;
            const double bt = -sigma2 * b_ji;

            cov[matrix_index(i, j, m)] = af;
            cov[matrix_index(i + d, j + d, m)] = df;
            cov[matrix_index(i, j + d, m)] = bf;
            cov[matrix_index(i + d, j, m)] = bt;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            tp2_base[tp_idx * tp2_stride + coeff_idx],
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        double var = cov[matrix_index(tt, tt, m)];
        if (!(var > eps))
        {
            var = fallback_var;
        }
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt(var));
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            col_buf[static_cast<size_t>(i)] =
                cov[matrix_index(static_cast<size_t>(i), tt, m)];
        }

        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                (col_buf[static_cast<size_t>(i)] / var) * delta;
        }

        for (int i = 0; i < t; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                double updated = cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                                 (col_buf[static_cast<size_t>(i)] * col_buf[static_cast<size_t>(j)] / var);
                cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
            }
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_scatter_p1_integer_to_limb_kernel(
    const int64_t *sampled_in,
    uint64_t *out_base,
    size_t out_stride,
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
    out_base[entry_idx * out_stride + coeff_idx] = signed_mod_i64(sampled_in[idx], modulus);
}

__global__ void matrix_gauss_samp_gq_arb_base_sample_kernel(
    const uint64_t *src_base,
    int64_t *sampled_digits,
    size_t poly_count,
    size_t n,
    size_t src_stride,
    uint64_t tower_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    double c,
    uint32_t tower_idx,
    uint64_t seed)
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

    uint64_t value = src_base[poly_idx * src_stride + coeff_idx];
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

__global__ void matrix_gauss_samp_gq_arb_base_scatter_kernel(
    const int64_t *sampled_digits,
    uint64_t *const *dst_bases,
    const size_t *dst_strides,
    const uint64_t *out_moduli,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    size_t src_digit_offset,
    uint32_t digits_per_tower)
{
    const size_t out_limb = static_cast<size_t>(blockIdx.y);
    if (out_limb >= out_limb_count)
    {
        return;
    }
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    if (!sampled_digits || !dst_bases || !dst_strides || !out_moduli || src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return;
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return;
    }

    uint64_t *dst_base = dst_bases[out_limb];
    const size_t dst_stride = dst_strides[out_limb];
    const uint64_t out_modulus = out_moduli[out_limb];
    if (!dst_base || dst_stride < n)
    {
        return;
    }

    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;
    const size_t row = poly_idx / src_cols;
    const size_t col = poly_idx - row * src_cols;
    for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
    {
        const size_t sample_idx =
            (poly_idx * static_cast<size_t>(digits_per_tower) + static_cast<size_t>(digit_idx)) * n + coeff_idx;
        const int64_t out_digit = sampled_digits[sample_idx];
        const size_t out_row = row * log_base_q + src_digit_offset + static_cast<size_t>(digit_idx);
        const size_t out_poly_idx = out_row * out_cols + col;
        dst_base[out_poly_idx * dst_stride + coeff_idx] = signed_mod_i64(out_digit, out_modulus);
    }
}

int launch_gauss_samp_gq_arb_base_sample_kernel(
    const uint64_t *src_base,
    int64_t *sampled_digits,
    size_t poly_count,
    size_t n,
    size_t src_stride,
    uint64_t tower_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    double c,
    uint32_t tower_idx,
    uint64_t seed,
    int device,
    cudaStream_t stream)
{
    if (!src_base || !sampled_digits)
    {
        return set_error("null base pointer in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }
    if (poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_stride < n)
    {
        return set_error("invalid stride in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid digits/base in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_gauss_samp_gq_arb_base_sample_kernel<<<blocks, threads, 0, stream>>>(
        src_base,
        sampled_digits,
        poly_count,
        n,
        src_stride,
        tower_modulus,
        base_bits,
        digits_per_tower,
        c,
        tower_idx,
        seed);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int launch_gauss_samp_gq_arb_base_scatter_kernel(
    const int64_t *sampled_digits,
    uint64_t *const *dst_bases,
    const size_t *dst_strides,
    const uint64_t *out_moduli,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    size_t src_digit_offset,
    uint32_t digits_per_tower,
    int device,
    cudaStream_t stream)
{
    if (!sampled_digits || !dst_bases || !dst_strides || !out_moduli)
    {
        return set_error("null pointer in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (out_limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return set_error("invalid matrix shape in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return set_error("invalid digits in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (out_limb_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return set_error("too many out limbs in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const uint32_t blocks_x = static_cast<uint32_t>((total + threads - 1) / threads);
    const dim3 grid{blocks_x, static_cast<uint32_t>(out_limb_count), 1U};
    matrix_gauss_samp_gq_arb_base_scatter_kernel<<<grid, threads, 0, stream>>>(
        sampled_digits,
        dst_bases,
        dst_strides,
        out_moduli,
        out_limb_count,
        poly_count,
        n,
        src_cols,
        out_cols,
        log_base_q,
        src_digit_offset,
        digits_per_tower);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


int launch_sample_p1_integer_kernel(
    const uint64_t *a_base,
    const uint64_t *b_base,
    const uint64_t *d_base,
    const uint64_t *tp2_base,
    size_t a_stride,
    size_t b_stride,
    size_t d_stride,
    size_t tp2_stride,
    size_t d,
    size_t cols,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    cudaStream_t stream,
    int device_id,
    int64_t **sampled_out_device,
    cudaEvent_t sampled_ready_event)
{
    if (!sampled_out_device)
    {
        return set_error("null output pointer in matrix_sample_p1_integer_kernel");
    }
    *sampled_out_device = nullptr;
    if (!a_base || !b_base || !d_base || !tp2_base)
    {
        return set_error("null base pointer in matrix_sample_p1_integer_kernel");
    }
    if (a_stride < n || b_stride < n || d_stride < n || tp2_stride < n)
    {
        return set_error("invalid stride in matrix_sample_p1_integer_kernel");
    }
    if (d == 0 || cols == 0 || n == 0)
    {
        return 0;
    }
    const size_t vec_entries = 2 * d * cols;

    if (device_id < 0)
    {
        return set_error("invalid device in matrix_sample_p1_integer_kernel");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_sample_p1_integer_kernel");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t m = d * 2;
    if (m == 0 || m > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("invalid dimension in matrix_sample_p1_integer_kernel");
    }
    const size_t total_samples = cols * n;
    const size_t total_values = vec_entries * n;
    if (total_samples == 0 || total_values == 0)
    {
        return 0;
    }

    int64_t *d_sampled_out = nullptr;

    auto free_all = [&]()
    {
        if (d_sampled_out)
        {
            cudaFreeAsync(d_sampled_out, stream);
            d_sampled_out = nullptr;
        }
    };

    err = cudaMallocAsync(reinterpret_cast<void **>(&d_sampled_out), total_values * sizeof(int64_t), stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    const int threads = 256;
    if (m <= kSampleP1LocalMaxM)
    {
        const int blocks = static_cast<int>((total_samples + threads - 1) / threads);
        matrix_sample_p1_integer_kernel_small<<<blocks, threads, 0, stream>>>(
            a_base,
            b_base,
            d_base,
            tp2_base,
            a_stride,
            b_stride,
            d_stride,
            tp2_stride,
            d,
            cols,
            n,
            d_sampled_out,
            modulus,
            sigma,
            s,
            dgg_stddev,
            seed);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }
    else
    {
        const size_t cov_elems_per_sample = m * m;
        if (cov_elems_per_sample > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t cov_bytes_per_sample = cov_elems_per_sample * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t vec_bytes_per_sample = m * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
        if (cov_bytes_per_sample > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        size_t bytes_per_sample_total = cov_bytes_per_sample + vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        bytes_per_sample_total += vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        bytes_per_sample_total += sampled_bytes_per_sample;

        void *workspace = nullptr;
        double *cov_workspace = nullptr;
        double *mean_workspace = nullptr;
        double *col_workspace = nullptr;
        int64_t *sampled_workspace = nullptr;
        auto free_workspace = [&]()
        {
            if (workspace)
            {
                cudaFreeAsync(workspace, stream);
                workspace = nullptr;
            }
            cov_workspace = nullptr;
            mean_workspace = nullptr;
            col_workspace = nullptr;
            sampled_workspace = nullptr;
        };

        size_t chunk_samples = total_samples;
        auto alloc_workspace = [&](size_t samples) -> bool
        {
            if (samples == 0)
            {
                return false;
            }
            if (samples > std::numeric_limits<size_t>::max() / cov_bytes_per_sample ||
                samples > std::numeric_limits<size_t>::max() / vec_bytes_per_sample ||
                samples > std::numeric_limits<size_t>::max() / sampled_bytes_per_sample)
            {
                return false;
            }
            if (samples > std::numeric_limits<size_t>::max() / bytes_per_sample_total)
            {
                return false;
            }
            const size_t workspace_bytes = samples * bytes_per_sample_total;
            cudaError_t local_err = cudaMallocAsync(&workspace, workspace_bytes, stream);
            if (local_err != cudaSuccess)
            {
                free_workspace();
                return false;
            }
            auto *workspace_base = reinterpret_cast<uint8_t *>(workspace);
            const size_t cov_bytes = samples * cov_bytes_per_sample;
            const size_t mean_bytes = samples * vec_bytes_per_sample;
            const size_t col_bytes = samples * vec_bytes_per_sample;
            cov_workspace = reinterpret_cast<double *>(workspace_base);
            mean_workspace = reinterpret_cast<double *>(workspace_base + cov_bytes);
            col_workspace = reinterpret_cast<double *>(workspace_base + cov_bytes + mean_bytes);
            sampled_workspace = reinterpret_cast<int64_t *>(
                workspace_base + cov_bytes + mean_bytes + col_bytes);
            return true;
        };

        while (!alloc_workspace(chunk_samples))
        {
            if (chunk_samples <= 1)
            {
                free_workspace();
                free_all();
                return set_error("failed to allocate workspace in matrix_sample_p1_integer_kernel");
            }
            chunk_samples = (chunk_samples + 1) / 2;
        }

        for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
        {
            size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
            const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
            matrix_sample_p1_integer_kernel_large<<<blocks, threads, 0, stream>>>(
                a_base,
                b_base,
                d_base,
                tp2_base,
                a_stride,
                b_stride,
                d_stride,
                tp2_stride,
                d,
                cols,
                n,
                sample_start,
                sample_count,
                cov_workspace,
                mean_workspace,
                col_workspace,
                sampled_workspace,
                d_sampled_out,
                modulus,
                sigma,
                s,
                dgg_stddev,
                seed);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                free_workspace();
                free_all();
                return set_error(err);
            }
        }
        free_workspace();
    }

    if (sampled_ready_event)
    {
        err = cudaEventRecord(sampled_ready_event, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }

    *sampled_out_device = d_sampled_out;
    d_sampled_out = nullptr;
    free_all();
    return 0;
}

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    uint64_t *out_base,
    size_t out_stride,
    size_t entry_count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id)
{
    if (entry_count == 0 || n == 0)
    {
        return 0;
    }
    if (!sampled_in_device)
    {
        return set_error("null sampled device buffer in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (!out_base)
    {
        return set_error("null output base pointer in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (out_stride < n)
    {
        return set_error("invalid output stride in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (device_id < 0)
    {
        return set_error("invalid device in matrix_scatter_p1_integer_to_limb_kernel");
    }
    const size_t total = entry_count * n;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_scatter_p1_integer_to_limb_kernel<<<blocks, threads, 0, stream>>>(
        sampled_in_device,
        out_base,
        out_stride,
        entry_count,
        n,
        modulus);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


extern "C" int gpu_matrix_gauss_samp_gq_arb_base(
    GpuMatrix *src,
    uint32_t base_bits,
    double c,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    (void)dgg_stddev;
    if (!src || !out)
    {
        return set_error("invalid gpu_matrix_gauss_samp_gq_arb_base arguments");
    }
    if (base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid base_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (!(c > 0.0))
    {
        return set_error("c must be positive in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_gauss_samp_gq_arb_base");
    }

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    uint32_t crt_bits = 0;
    for (const auto &modulus : src->ctx->moduli)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const uint32_t digits_per_tower = static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (count == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const GpuMatrix *inputs_matrix = src;
    auto cleanup_tmp_inputs = [&]() {};

    int status = 0;
    const int batch = default_batch(src->ctx);
    if (src->format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_intt_all(src, batch);
        if (status != 0)
        {
            return status;
        }
    }

    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_gauss_samp_gq_arb_base");
    }

    std::vector<dim3> active_limb_ids(crt_depth);
    std::vector<uint64_t *> out_limb_bases(crt_depth, nullptr);
    std::vector<size_t> out_limb_strides(crt_depth, 0);
    std::vector<uint64_t> out_limb_moduli(crt_depth, 0);
    std::vector<const uint64_t *> src_limb_bases(crt_depth, nullptr);
    std::vector<size_t> src_limb_strides(crt_depth, 0);

    int dispatch_device = -1;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t limb_idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[limb_idx];
        active_limb_ids[limb_idx] = limb_id;
        out_limb_moduli[limb_idx] = src->ctx->moduli[limb_idx];

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
        }
        else if (out_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-GPU path requires all out limbs on one device");
        }
        uint64_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb base pointer in gpu_matrix_gauss_samp_gq_arb_base");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            cleanup_tmp_inputs();
            return set_error("invalid output partition index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const size_t dst_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;
        if (dst_stride < static_cast<size_t>(src->ctx->N))
        {
            cleanup_tmp_inputs();
            return set_error("invalid output stride in gpu_matrix_gauss_samp_gq_arb_base");
        }
        out_limb_bases[limb_idx] = dst_base;
        out_limb_strides[limb_idx] = dst_stride;
    }
    if (dispatch_device < 0)
    {
        cleanup_tmp_inputs();
        return set_error("invalid output device in gpu_matrix_gauss_samp_gq_arb_base");
    }

    cudaStream_t dispatch_stream = nullptr;
    status = matrix_limb_stream(out, active_limb_ids[0], &dispatch_stream);
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }
    if (!dispatch_stream)
    {
        cleanup_tmp_inputs();
        return set_error("null dispatch stream in gpu_matrix_gauss_samp_gq_arb_base");
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
        int src_device = -1;
        status = matrix_limb_device(inputs_matrix, src_limb_id, &src_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        if (src_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-GPU path requires all src limbs on one device");
        }
        const uint64_t *src_base = matrix_limb_ptr_by_id(inputs_matrix, 0, src_limb_id);
        if (!src_base)
        {
            cleanup_tmp_inputs();
            return set_error("null source limb base pointer in gpu_matrix_gauss_samp_gq_arb_base");
        }
        if (src_limb_id.x >= inputs_matrix->shared_limb_buffers.size())
        {
            cleanup_tmp_inputs();
            return set_error("invalid source partition index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const size_t src_stride = inputs_matrix->shared_limb_buffers[src_limb_id.x].words_per_poly;
        if (src_stride < static_cast<size_t>(src->ctx->N))
        {
            cleanup_tmp_inputs();
            return set_error("invalid source stride in gpu_matrix_gauss_samp_gq_arb_base");
        }
        src_limb_bases[src_idx] = src_base;
        src_limb_strides[src_idx] = src_stride;
    }

    size_t sampled_values = count;
    if (static_cast<size_t>(src->ctx->N) != 0 &&
        sampled_values > std::numeric_limits<size_t>::max() / static_cast<size_t>(src->ctx->N))
    {
        cleanup_tmp_inputs();
        return set_error("sample size overflow in gpu_matrix_gauss_samp_gq_arb_base");
    }
    sampled_values *= static_cast<size_t>(src->ctx->N);
    if (static_cast<size_t>(digits_per_tower) != 0 &&
        sampled_values > std::numeric_limits<size_t>::max() / static_cast<size_t>(digits_per_tower))
    {
        cleanup_tmp_inputs();
        return set_error("sample size overflow in gpu_matrix_gauss_samp_gq_arb_base");
    }
    sampled_values *= static_cast<size_t>(digits_per_tower);
    if (sampled_values > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        cleanup_tmp_inputs();
        return set_error("sample byte overflow in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t sampled_bytes = sampled_values * sizeof(int64_t);
    if (crt_depth > std::numeric_limits<size_t>::max() / sizeof(uint64_t *) ||
        crt_depth > std::numeric_limits<size_t>::max() / sizeof(size_t) ||
        crt_depth > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
    {
        cleanup_tmp_inputs();
        return set_error("limb metadata size overflow in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t out_ptr_bytes = crt_depth * sizeof(uint64_t *);
    const size_t out_stride_bytes = crt_depth * sizeof(size_t);
    const size_t out_moduli_bytes = crt_depth * sizeof(uint64_t);

    int64_t *sampled_digits_device = nullptr;
    uint64_t **out_limb_bases_device = nullptr;
    size_t *out_limb_strides_device = nullptr;
    uint64_t *out_limb_moduli_device = nullptr;
    auto cleanup = [&]()
    {
        if (dispatch_device >= 0)
        {
            cudaSetDevice(dispatch_device);
        }
        if (sampled_digits_device)
        {
            if (dispatch_stream)
            {
                cudaFreeAsync(sampled_digits_device, dispatch_stream);
            }
            else
            {
                cudaFree(sampled_digits_device);
            }
            sampled_digits_device = nullptr;
        }
        if (out_limb_bases_device)
        {
            if (dispatch_stream)
            {
                cudaFreeAsync(out_limb_bases_device, dispatch_stream);
            }
            else
            {
                cudaFree(out_limb_bases_device);
            }
            out_limb_bases_device = nullptr;
        }
        if (out_limb_strides_device)
        {
            if (dispatch_stream)
            {
                cudaFreeAsync(out_limb_strides_device, dispatch_stream);
            }
            else
            {
                cudaFree(out_limb_strides_device);
            }
            out_limb_strides_device = nullptr;
        }
        if (out_limb_moduli_device)
        {
            if (dispatch_stream)
            {
                cudaFreeAsync(out_limb_moduli_device, dispatch_stream);
            }
            else
            {
                cudaFree(out_limb_moduli_device);
            }
            out_limb_moduli_device = nullptr;
        }
        cleanup_tmp_inputs();
    };

    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    const size_t out_count = out->rows * out->cols;
    const size_t coeff_bytes = static_cast<size_t>(src->ctx->N) * sizeof(uint64_t);
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 out_limb_id = active_limb_ids[static_cast<size_t>(limb)];
        status = matrix_wait_limb_stream(out, out_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        if (out_count > 0)
        {
            const size_t dst_pitch = out_limb_strides[static_cast<size_t>(limb)] * sizeof(uint64_t);
            err = cudaMemset2DAsync(
                out_limb_bases[static_cast<size_t>(limb)],
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
    }

    err = cudaMallocAsync(reinterpret_cast<void **>(&sampled_digits_device), sampled_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
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

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
        status = matrix_wait_limb_stream(inputs_matrix, src_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_gauss_samp_gq_arb_base_sample_kernel(
            src_limb_bases[src_idx],
            sampled_digits_device,
            count,
            static_cast<size_t>(src->ctx->N),
            src_limb_strides[src_idx],
            src->ctx->moduli[src_idx],
            base_bits,
            digits_per_tower,
            c,
            static_cast<uint32_t>(src_limb),
            seed,
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        const size_t src_digit_offset =
            static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower);
        status = launch_gauss_samp_gq_arb_base_scatter_kernel(
            sampled_digits_device,
            out_limb_bases_device,
            out_limb_strides_device,
            out_limb_moduli_device,
            crt_depth,
            count,
            static_cast<size_t>(src->ctx->N),
            cols,
            out->cols,
            log_base_q,
            src_digit_offset,
            digits_per_tower,
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = matrix_track_limb_consumer(
            inputs_matrix,
            src_limb_id,
            dispatch_device,
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
        status = gpu_matrix_ntt_all(out, batch);
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

extern "C" int gpu_matrix_sample_p1_full(
    const GpuMatrix *a_mat,
    const GpuMatrix *b_mat,
    const GpuMatrix *d_mat,
    const GpuMatrix *tp2,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    if (!a_mat || !b_mat || !d_mat || !tp2 || !out)
    {
        return set_error("invalid gpu_matrix_sample_p1_full arguments");
    }
    if (!(sigma > 0.0) || !(s > sigma))
    {
        return set_error("invalid sigma/s in gpu_matrix_sample_p1_full");
    }
    if (!(dgg_stddev > 0.0))
    {
        return set_error("dgg_stddev must be positive in gpu_matrix_sample_p1_full");
    }
    if (a_mat->ctx != b_mat->ctx || a_mat->ctx != d_mat->ctx || a_mat->ctx != tp2->ctx || a_mat->ctx != out->ctx)
    {
        return set_error("context mismatch in gpu_matrix_sample_p1_full");
    }
    if (a_mat->level != b_mat->level || a_mat->level != d_mat->level ||
        a_mat->level != tp2->level || a_mat->level != out->level)
    {
        return set_error("level mismatch in gpu_matrix_sample_p1_full");
    }

    const size_t d_rows = a_mat->rows;
    if (a_mat->cols != d_rows || b_mat->rows != d_rows || b_mat->cols != d_rows ||
        d_mat->rows != d_rows || d_mat->cols != d_rows)
    {
        return set_error("A/B/D must be dxd in gpu_matrix_sample_p1_full");
    }
    const size_t cols = tp2->cols;
    if (tp2->rows != 2 * d_rows || out->rows != 2 * d_rows || out->cols != cols)
    {
        return set_error("tp2/out shape mismatch in gpu_matrix_sample_p1_full");
    }
    if (cols == 0 || d_rows == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const int level = a_mat->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_p1_full");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (a_mat->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_p1_full");
    }
    auto &limb_map = a_mat->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_p1_full");
    }

    GpuMatrix *tmp_a = nullptr;
    GpuMatrix *tmp_b = nullptr;
    GpuMatrix *tmp_d = nullptr;
    GpuMatrix *tmp_tp2 = nullptr;
    const GpuMatrix *a_input = a_mat;
    const GpuMatrix *b_input = b_mat;
    const GpuMatrix *d_input = d_mat;
    const GpuMatrix *tp2_input = tp2;
    struct DeviceSampleBuffer
    {
        int device;
        int64_t *ptr;
        cudaStream_t owner_stream;
        cudaEvent_t ready_event;
    };
    std::vector<DeviceSampleBuffer> sampled_device_buffers;
    cudaEvent_t sampled_ready_event = nullptr;
    int sampled_ready_device = -1;

    auto cleanup = [&]()
    {
        if (sampled_ready_event)
        {
            if (sampled_ready_device >= 0)
            {
                cudaSetDevice(sampled_ready_device);
            }
            cudaEventDestroy(sampled_ready_event);
            sampled_ready_event = nullptr;
        }
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device < 0)
            {
                entry.ptr = nullptr;
                if (entry.ready_event)
                {
                    cudaEventDestroy(entry.ready_event);
                    entry.ready_event = nullptr;
                }
                continue;
            }
            cudaSetDevice(entry.device);
            if (entry.ptr)
            {
                if (entry.owner_stream)
                {
                    cudaFreeAsync(entry.ptr, entry.owner_stream);
                }
                else
                {
                    cudaFree(entry.ptr);
                }
                entry.ptr = nullptr;
            }
            if (entry.ready_event)
            {
                cudaEventDestroy(entry.ready_event);
                entry.ready_event = nullptr;
            }
        }
        sampled_device_buffers.clear();
        if (tmp_a)
        {
            gpu_matrix_destroy(tmp_a);
            tmp_a = nullptr;
        }
        if (tmp_b)
        {
            gpu_matrix_destroy(tmp_b);
            tmp_b = nullptr;
        }
        if (tmp_d)
        {
            gpu_matrix_destroy(tmp_d);
            tmp_d = nullptr;
        }
        if (tmp_tp2)
        {
            gpu_matrix_destroy(tmp_tp2);
            tmp_tp2 = nullptr;
        }
    };

    const int batch = default_batch(a_mat->ctx);
    auto collect_coeff_input_matrix = [&](
                                          const GpuMatrix *src,
                                          GpuMatrix **owned,
                                          const GpuMatrix **coeff_input) -> int
    {
        *owned = nullptr;
        *coeff_input = src;
        if (src->format == GPU_POLY_FORMAT_EVAL)
        {
            const int matrix_format =
                src->format == GPU_POLY_FORMAT_EVAL ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            int status = gpu_matrix_create(
                src->ctx,
                src->level,
                src->rows,
                src->cols,
                matrix_format,
                owned);
            if (status != 0)
            {
                return status;
            }
            status = gpu_matrix_copy(*owned, src);
            if (status != 0)
            {
                gpu_matrix_destroy(*owned);
                *owned = nullptr;
                return status;
            }
            status = gpu_matrix_intt_all(*owned, batch);
            if (status != 0)
            {
                gpu_matrix_destroy(*owned);
                *owned = nullptr;
                return status;
            }
            *coeff_input = *owned;
        }
        return 0;
    };

    int status = collect_coeff_input_matrix(
        a_mat,
        &tmp_a,
        &a_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        b_mat,
        &tmp_b,
        &b_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        d_mat,
        &tmp_d,
        &d_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        tp2,
        &tmp_tp2,
        &tp2_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    out->format = GPU_POLY_FORMAT_COEFF;

    const dim3 ref_limb_id = limb_map[0];
    int ref_device = -1;
    status = matrix_limb_device(out, ref_limb_id, &ref_device);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    cudaStream_t ref_stream = nullptr;
    status = matrix_limb_stream(out, ref_limb_id, &ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    int tp2_ref_device = -1;
    status = matrix_limb_device(tp2_input, ref_limb_id, &tp2_ref_device);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    if (tp2_ref_device != ref_device)
    {
        cleanup();
        return set_error("input/output limb device mismatch in gpu_matrix_sample_p1_full");
    }
    const uint64_t *ref_a_base = matrix_limb_ptr_by_id(a_input, 0, ref_limb_id);
    const uint64_t *ref_b_base = matrix_limb_ptr_by_id(b_input, 0, ref_limb_id);
    const uint64_t *ref_d_base = matrix_limb_ptr_by_id(d_input, 0, ref_limb_id);
    const uint64_t *ref_tp2_base = matrix_limb_ptr_by_id(tp2_input, 0, ref_limb_id);
    if (!ref_a_base || !ref_b_base || !ref_d_base || !ref_tp2_base)
    {
        cleanup();
        return set_error("null reference limb base pointer in gpu_matrix_sample_p1_full");
    }
    if (ref_limb_id.x >= a_input->shared_limb_buffers.size() ||
        ref_limb_id.x >= b_input->shared_limb_buffers.size() ||
        ref_limb_id.x >= d_input->shared_limb_buffers.size() ||
        ref_limb_id.x >= tp2_input->shared_limb_buffers.size())
    {
        cleanup();
        return set_error("invalid reference partition index in gpu_matrix_sample_p1_full");
    }
    const size_t ref_a_stride = a_input->shared_limb_buffers[ref_limb_id.x].words_per_poly;
    const size_t ref_b_stride = b_input->shared_limb_buffers[ref_limb_id.x].words_per_poly;
    const size_t ref_d_stride = d_input->shared_limb_buffers[ref_limb_id.x].words_per_poly;
    const size_t ref_tp2_stride = tp2_input->shared_limb_buffers[ref_limb_id.x].words_per_poly;
    if (!ref_stream || ref_device < 0)
    {
        cleanup();
        return set_error("invalid reference stream/device in gpu_matrix_sample_p1_full");
    }
    status = matrix_wait_limb_stream(a_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(b_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(d_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(tp2_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    cudaError_t err = cudaSetDevice(ref_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaEventCreateWithFlags(&sampled_ready_event, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    sampled_ready_device = ref_device;

    int64_t *sampled_ref_device = nullptr;
    status = launch_sample_p1_integer_kernel(
        ref_a_base,
        ref_b_base,
        ref_d_base,
        ref_tp2_base,
        ref_a_stride,
        ref_b_stride,
        ref_d_stride,
        ref_tp2_stride,
        d_rows,
        cols,
        static_cast<size_t>(a_mat->ctx->N),
        a_mat->ctx->moduli[0],
        sigma,
        s,
        dgg_stddev,
        seed,
        ref_stream,
        ref_device,
        &sampled_ref_device,
        sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(a_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(b_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(d_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(tp2_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    if (sampled_ref_device)
    {
        sampled_device_buffers.push_back(
            DeviceSampleBuffer{ref_device, sampled_ref_device, ref_stream, sampled_ready_event});
        sampled_ready_event = nullptr;
        sampled_ready_device = -1;
    }

    size_t sampled_entry_count = 0;
    size_t sampled_value_count = 0;
    size_t sampled_bytes = 0;
    if (d_rows != 0 && cols > std::numeric_limits<size_t>::max() / (2 * d_rows))
    {
        cleanup();
        return set_error("sample count overflow in gpu_matrix_sample_p1_full");
    }
    sampled_entry_count = 2 * d_rows * cols;
    if (static_cast<size_t>(a_mat->ctx->N) != 0 &&
        sampled_entry_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(a_mat->ctx->N))
    {
        cleanup();
        return set_error("sample count overflow in gpu_matrix_sample_p1_full");
    }
    sampled_value_count = sampled_entry_count * static_cast<size_t>(a_mat->ctx->N);
    if (sampled_value_count > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        cleanup();
        return set_error("sample byte overflow in gpu_matrix_sample_p1_full");
    }
    sampled_bytes = sampled_value_count * sizeof(int64_t);

    auto ensure_sample_buffer_on_device = [&](int device, cudaStream_t stream, int64_t **out_ptr) -> int
    {
        if (!out_ptr)
        {
            return set_error("invalid output in ensure_sample_buffer_on_device");
        }
        if (device < 0 || !stream)
        {
            return set_error("invalid device/stream in ensure_sample_buffer_on_device");
        }
        *out_ptr = nullptr;
        auto link_consumer_to_owner =
            [&](DeviceSampleBuffer &entry, cudaStream_t consumer_stream) -> int
        {
            if (!entry.owner_stream)
            {
                return 0;
            }
            cudaError_t link_err = cudaSetDevice(entry.device);
            if (link_err != cudaSuccess)
            {
                return set_error(link_err);
            }
            cudaEvent_t consumer_done = nullptr;
            link_err = cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming);
            if (link_err != cudaSuccess)
            {
                return set_error(link_err);
            }
            link_err = cudaEventRecord(consumer_done, consumer_stream);
            if (link_err != cudaSuccess)
            {
                cudaEventDestroy(consumer_done);
                return set_error(link_err);
            }
            link_err = cudaStreamWaitEvent(entry.owner_stream, consumer_done, 0);
            cudaError_t destroy_err = cudaEventDestroy(consumer_done);
            if (link_err != cudaSuccess)
            {
                return set_error(link_err);
            }
            if (destroy_err != cudaSuccess)
            {
                return set_error(destroy_err);
            }
            return 0;
        };
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device == device && entry.ptr)
            {
                cudaError_t wait_err = cudaSetDevice(device);
                if (wait_err != cudaSuccess)
                {
                    return set_error(wait_err);
                }
                if (entry.ready_event)
                {
                    wait_err = cudaStreamWaitEvent(stream, entry.ready_event, 0);
                    if (wait_err != cudaSuccess)
                    {
                        return set_error(wait_err);
                    }
                }
                int link_status = link_consumer_to_owner(entry, stream);
                if (link_status != 0)
                {
                    return link_status;
                }
                *out_ptr = entry.ptr;
                return 0;
            }
        }
        if (!sampled_ref_device)
        {
            return set_error("missing reference sample buffer in gpu_matrix_sample_p1_full");
        }
        if (sampled_bytes == 0)
        {
            return 0;
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int64_t *device_copy = nullptr;
        err = cudaMallocAsync(reinterpret_cast<void **>(&device_copy), sampled_bytes, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpyPeerAsync(
            device_copy,
            device,
            sampled_ref_device,
            ref_device,
            sampled_bytes,
            stream);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(device_copy, stream);
            return set_error(err);
        }
        cudaEvent_t copy_ready = nullptr;
        err = cudaEventCreateWithFlags(&copy_ready, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(device_copy, stream);
            return set_error(err);
        }
        err = cudaEventRecord(copy_ready, stream);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(copy_ready);
            cudaFreeAsync(device_copy, stream);
            return set_error(err);
        }
        sampled_device_buffers.push_back(DeviceSampleBuffer{device, device_copy, stream, copy_ready});
        *out_ptr = device_copy;
        return 0;
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        uint64_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!out_base)
        {
            cleanup();
            return set_error("null output limb base pointer in gpu_matrix_sample_p1_full");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            cleanup();
            return set_error("invalid output partition index in gpu_matrix_sample_p1_full");
        }
        const size_t out_stride = out->shared_limb_buffers[limb_id.x].words_per_poly;

        int64_t *sampled_for_device = nullptr;
        status = ensure_sample_buffer_on_device(out_device, out_stream, &sampled_for_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_scatter_p1_integer_to_limb_kernel_device(
            sampled_for_device,
            out_base,
            out_stride,
            sampled_entry_count,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    status = gpu_matrix_ntt_all(out, batch);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;

    cleanup();
    return 0;
}
