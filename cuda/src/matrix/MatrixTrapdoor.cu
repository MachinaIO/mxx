__global__ void matrix_sample_p1_integer_kernel(
    const uint64_t **a_entries,
    const uint64_t **b_entries,
    const uint64_t **d_entries,
    const uint64_t **tp2_entries,
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
            const double a_ij = static_cast<double>(centered_residue_i64(a_entries[ij][coeff_idx], modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(d_entries[ij][coeff_idx], modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(b_entries[ij][coeff_idx], modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(b_entries[ji][coeff_idx], modulus));

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
        const double c_centered = static_cast<double>(centered_residue_i64(tp2_entries[tp_idx][coeff_idx], modulus));
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
    uint64_t **out_entries,
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
    out_entries[entry_idx][coeff_idx] = signed_mod_i64(sampled_in[idx], modulus);
}

__global__ void matrix_gauss_samp_gq_arb_base_multi_kernel(
    const uint64_t **src,
    uint64_t **dst,
    size_t poly_count,
    size_t n,
    size_t job_count,
    const uint64_t *tower_moduli,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    const uint32_t *digit_indices,
    double c,
    const uint32_t *tower_indices,
    uint64_t seed,
    const uint64_t *out_moduli)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = job_count * poly_count * n;
    if (idx >= total)
    {
        return;
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
    {
        return;
    }

    const size_t per_job = poly_count * n;
    const size_t job_idx = idx / per_job;
    const size_t rem = idx - job_idx * per_job;
    const size_t poly_idx = rem / n;
    const size_t coeff_idx = rem - poly_idx * n;
    const size_t ptr_idx = job_idx * poly_count + poly_idx;

    const uint64_t tower_modulus = tower_moduli[job_idx];
    const uint64_t out_modulus = out_moduli[job_idx];
    const uint32_t digit_idx = digit_indices[job_idx];
    const uint32_t tower_idx = tower_indices[job_idx];

    uint64_t value = src[ptr_idx][coeff_idx];
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

    dst[ptr_idx][coeff_idx] = signed_mod_i64(out_digit, out_modulus);
}

int launch_gauss_samp_gq_arb_base_multi_kernel(
    const std::vector<const uint64_t *> &src_ptrs,
    const std::vector<uint64_t *> &dst_ptrs,
    size_t poly_count,
    size_t n,
    const std::vector<uint64_t> &tower_moduli,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    const std::vector<uint32_t> &digit_indices,
    double c,
    const std::vector<uint32_t> &tower_indices,
    uint64_t seed,
    const std::vector<uint64_t> &out_moduli,
    int device,
    cudaStream_t stream)
{
    const size_t job_count = tower_moduli.size();
    if (job_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (digit_indices.size() != job_count || tower_indices.size() != job_count ||
        out_moduli.size() != job_count)
    {
        return set_error("unexpected job parameter counts in matrix_gauss_samp_gq_arb_base_multi_kernel");
    }
    const size_t ptr_count = job_count * poly_count;
    if (src_ptrs.size() != ptr_count || dst_ptrs.size() != ptr_count)
    {
        return set_error("unexpected pointer counts in matrix_gauss_samp_gq_arb_base_multi_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const uint64_t **d_src = nullptr;
    uint64_t **d_dst = nullptr;
    uint64_t *d_tower_moduli = nullptr;
    uint32_t *d_digit_indices = nullptr;
    uint32_t *d_tower_indices = nullptr;
    uint64_t *d_out_moduli = nullptr;

    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = job_count * sizeof(uint64_t);
    const size_t u32_bytes = job_count * sizeof(uint32_t);

    err = cudaMallocAsync(&d_src, ptr_bytes, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaMallocAsync(&d_dst, ptr_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_tower_moduli, u64_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_digit_indices, u32_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_tower_indices, u32_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        return set_error(err);
    }
    err = cudaMallocAsync(&d_out_moduli, u64_bytes, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_src, src_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_moduli, tower_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_digit_indices, digit_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_indices, tower_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_out_moduli, out_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = ptr_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_gauss_samp_gq_arb_base_multi_kernel<<<blocks, threads, 0, stream>>>(
        d_src,
        d_dst,
        poly_count,
        n,
        job_count,
        d_tower_moduli,
        base_bits,
        digits_per_tower,
        d_digit_indices,
        c,
        d_tower_indices,
        seed,
        d_out_moduli);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_src, stream);
        cudaFreeAsync(d_dst, stream);
        cudaFreeAsync(d_tower_moduli, stream);
        cudaFreeAsync(d_digit_indices, stream);
        cudaFreeAsync(d_tower_indices, stream);
        cudaFreeAsync(d_out_moduli, stream);
        return set_error(err);
    }

    err = cudaFreeAsync(d_src, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_dst, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_tower_moduli, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_digit_indices, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_tower_indices, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    err = cudaFreeAsync(d_out_moduli, stream);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


int launch_sample_p1_integer_kernel(
    const std::vector<const uint64_t *> &a_entries,
    const std::vector<const uint64_t *> &b_entries,
    const std::vector<const uint64_t *> &d_entries,
    const std::vector<const uint64_t *> &tp2_entries,
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
    std::vector<int64_t> &sampled_out_host)
{
    if (d == 0 || cols == 0 || n == 0)
    {
        sampled_out_host.clear();
        return 0;
    }
    const size_t mat_entries = d * d;
    const size_t vec_entries = 2 * d * cols;
    if (a_entries.size() != mat_entries || b_entries.size() != mat_entries ||
        d_entries.size() != mat_entries || tp2_entries.size() != vec_entries)
    {
        return set_error("unexpected pointer counts in matrix_sample_p1_integer_kernel");
    }

    if (device_id < 0)
    {
        return set_error("invalid device in matrix_sample_p1_integer_kernel");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t m = 2 * d;
    if (m == 0)
    {
        return set_error("invalid dimension in matrix_sample_p1_integer_kernel");
    }
    if (m > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    const size_t cov_elems_per_sample = m * m;
    if (cov_elems_per_sample > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    const size_t cov_bytes_per_sample = cov_elems_per_sample * sizeof(double);
    if (m > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    const size_t vec_bytes_per_sample = m * sizeof(double);
    if (m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
    if (cov_bytes_per_sample > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    size_t bytes_per_sample_total = cov_bytes_per_sample + vec_bytes_per_sample;
    if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    bytes_per_sample_total += vec_bytes_per_sample;
    if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
    {
        return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
    }
    bytes_per_sample_total += sampled_bytes_per_sample;

    const size_t total_samples = cols * n;
    size_t chunk_samples = total_samples;
    const size_t total_values = vec_entries * n;
    sampled_out_host.assign(total_values, 0);

    const uint64_t **d_a_entries = nullptr;
    const uint64_t **d_b_entries = nullptr;
    const uint64_t **d_d_entries = nullptr;
    const uint64_t **d_tp2_entries = nullptr;
    int64_t *d_sampled_out = nullptr;
    const size_t mat_bytes = mat_entries * sizeof(uint64_t *);
    const size_t vec_bytes = vec_entries * sizeof(uint64_t *);

    auto free_all = [&]()
    {
        if (d_a_entries)
            cudaFree(d_a_entries);
        if (d_b_entries)
            cudaFree(d_b_entries);
        if (d_d_entries)
            cudaFree(d_d_entries);
        if (d_tp2_entries)
            cudaFree(d_tp2_entries);
        if (d_sampled_out)
            cudaFree(d_sampled_out);
    };

    err = cudaMalloc(&d_a_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_b_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_d_entries, mat_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_tp2_entries, vec_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_sampled_out, total_values * sizeof(int64_t));
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_a_entries, a_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_b_entries, b_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_d_entries, d_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tp2_entries, tp2_entries.data(), vec_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    double *cov_workspace = nullptr;
    double *mean_workspace = nullptr;
    double *col_workspace = nullptr;
    int64_t *sampled_workspace = nullptr;
    auto free_workspace = [&]()
    {
        if (cov_workspace)
            cudaFree(cov_workspace);
        if (mean_workspace)
            cudaFree(mean_workspace);
        if (col_workspace)
            cudaFree(col_workspace);
        if (sampled_workspace)
            cudaFree(sampled_workspace);
        cov_workspace = nullptr;
        mean_workspace = nullptr;
        col_workspace = nullptr;
        sampled_workspace = nullptr;
    };

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
        cudaError_t local_err = cudaMalloc(&cov_workspace, samples * cov_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&mean_workspace, samples * vec_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&col_workspace, samples * vec_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        local_err = cudaMalloc(&sampled_workspace, samples * sampled_bytes_per_sample);
        if (local_err != cudaSuccess)
        {
            free_workspace();
            return false;
        }
        return true;
    };

    while (!alloc_workspace(chunk_samples))
    {
        if (chunk_samples <= 1)
        {
            free_all();
            return set_error("failed to allocate workspace in matrix_sample_p1_integer_kernel");
        }
        chunk_samples = (chunk_samples + 1) / 2;
    }

    const int threads = 256;
    for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
    {
        size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
        const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
        matrix_sample_p1_integer_kernel<<<blocks, threads, 0, stream>>>(
            d_a_entries,
            d_b_entries,
            d_d_entries,
            d_tp2_entries,
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

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_workspace();
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        sampled_out_host.data(),
        d_sampled_out,
        total_values * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess)
    {
        free_workspace();
        free_all();
        return set_error(err);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_workspace();
        free_all();
        return set_error(err);
    }

    free_workspace();
    free_all();
    return 0;
}

int launch_scatter_p1_integer_to_limb_kernel(
    const std::vector<int64_t> &sampled_in_host,
    const std::vector<uint64_t *> &out_entries,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id)
{
    if (out_entries.empty() || n == 0)
    {
        return 0;
    }
    if (device_id < 0)
    {
        return set_error("invalid device in matrix_scatter_p1_integer_to_limb_kernel");
    }
    const size_t total = out_entries.size() * n;
    if (sampled_in_host.size() != total)
    {
        return set_error("sampled input size mismatch in matrix_scatter_p1_integer_to_limb_kernel");
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    uint64_t **d_out_entries = nullptr;
    int64_t *d_sampled_in = nullptr;
    const size_t out_ptr_bytes = out_entries.size() * sizeof(uint64_t *);
    const size_t sampled_bytes = total * sizeof(int64_t);

    auto free_all = [&]()
    {
        if (d_out_entries)
            cudaFree(d_out_entries);
        if (d_sampled_in)
            cudaFree(d_sampled_in);
    };

    err = cudaMalloc(&d_out_entries, out_ptr_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMalloc(&d_sampled_in, sampled_bytes);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    err = cudaMemcpyAsync(d_out_entries, out_entries.data(), out_ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_sampled_in,
        sampled_in_host.data(),
        sampled_bytes,
        cudaMemcpyHostToDevice,
        stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_scatter_p1_integer_to_limb_kernel<<<blocks, threads, 0, stream>>>(
        d_sampled_in,
        d_out_entries,
        out_entries.size(),
        n,
        modulus);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    free_all();
    return 0;
}


extern "C" int gpu_matrix_gauss_samp_gq_arb_base(
    const GpuMatrix *src,
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
        out->format = PolyFormat::Eval;
        return 0;
    }

    GpuMatrix *tmp_inputs_matrix = nullptr;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        for (size_t i = 0; i < count; ++i)
        {
            int sync_status = sync_poly_partition_streams(
                src->polys[i],
                "failed to synchronize source partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                src->polys[i],
                "failed to synchronize source limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
        }

        const int matrix_format =
            src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        int status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = gpu_matrix_copy(tmp_inputs_matrix, src);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }

        for (auto *clone : tmp_inputs_matrix->polys)
        {
            int sync_status = sync_poly_partition_streams(
                clone,
                "failed to synchronize clone partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                clone,
                "failed to synchronize clone limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                cleanup_tmp_inputs();
                return sync_status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                cleanup_tmp_inputs();
                return status;
            }
            inputs.push_back(clone);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            inputs.push_back(src->polys[i]);
        }
    }

    for (int device : src->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(out->polys.size() * crt_depth);

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output poly in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                cleanup_tmp_inputs();
                return set_error("unexpected limb index in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                cleanup_tmp_inputs();
                return set_error("unsupported limb type in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }

            bool seen_stream = false;
            for (const auto &entry : out_zero_streams)
            {
                if (entry.first == partition.device && entry.second == limb_u64.stream.ptr)
                {
                    seen_stream = true;
                    break;
                }
            }
            if (!seen_stream)
            {
                out_zero_streams.emplace_back(partition.device, limb_u64.stream.ptr);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    for (const auto &entry : out_zero_streams)
    {
        cudaError_t err = cudaSetDevice(entry.first);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        err = cudaStreamSynchronize(entry.second);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_gauss_samp_gq_arb_base");
    }

    struct GaussBatch
    {
        int device;
        cudaStream_t stream;
        std::vector<const uint64_t *> src_ptrs;
        std::vector<uint64_t *> dst_ptrs;
        std::vector<uint64_t> tower_moduli;
        std::vector<uint32_t> digit_indices;
        std::vector<uint32_t> tower_indices;
        std::vector<uint64_t> out_moduli;
    };
    std::vector<GaussBatch> batches;
    auto get_batch = [&](int device) -> GaussBatch &
    {
        for (auto &b : batches)
        {
            if (b.device == device)
            {
                return b;
            }
        }
        batches.push_back(GaussBatch{device, nullptr, {}, {}, {}, {}, {}, {}});
        return batches.back();
    };

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            cleanup_tmp_inputs();
            return set_error("unexpected source limb index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            cleanup_tmp_inputs();
            return set_error("unsupported source limb type in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    cleanup_tmp_inputs();
                    return set_error("unexpected input source limb index in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    cleanup_tmp_inputs();
                    return set_error("unsupported input source limb type in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;
                int out_device = -1;

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("input/output limb device mismatch in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        cleanup_tmp_inputs();
                        return set_error("unexpected output limb index in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        cleanup_tmp_inputs();
                        return set_error("unsupported output limb type in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                        out_device = out_partition.device;
                    }
                    else if (out_device != out_partition.device)
                    {
                        cleanup_tmp_inputs();
                        return set_error("inconsistent output limb device in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                if (out_device < 0 || !out_stream)
                {
                    cleanup_tmp_inputs();
                    return set_error("invalid output stream/device in gpu_matrix_gauss_samp_gq_arb_base");
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
                }
                batch_ref.tower_moduli.push_back(src->ctx->moduli[static_cast<size_t>(src_limb)]);
                batch_ref.digit_indices.push_back(digit_idx);
                batch_ref.tower_indices.push_back(static_cast<uint32_t>(src_limb));
                batch_ref.out_moduli.push_back(src->ctx->moduli[static_cast<size_t>(out_limb)]);
                batch_ref.src_ptrs.insert(batch_ref.src_ptrs.end(), src_ptrs.begin(), src_ptrs.end());
                batch_ref.dst_ptrs.insert(batch_ref.dst_ptrs.end(), dst_ptrs.begin(), dst_ptrs.end());
            }
        }
    }

    for (auto &batch_ref : batches)
    {
        if (!batch_ref.stream)
        {
            cleanup_tmp_inputs();
            return set_error("null stream in gpu_matrix_gauss_samp_gq_arb_base");
        }
        int status = launch_gauss_samp_gq_arb_base_multi_kernel(
            batch_ref.src_ptrs,
            batch_ref.dst_ptrs,
            count,
            static_cast<size_t>(src->ctx->N),
            batch_ref.tower_moduli,
            base_bits,
            digits_per_tower,
            batch_ref.digit_indices,
            c,
            batch_ref.tower_indices,
            seed,
            batch_ref.out_moduli,
            batch_ref.device,
            batch_ref.stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup_tmp_inputs();
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
        out->format = PolyFormat::Eval;
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
    auto &limb_map = a_mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_p1_full");
    }

    GpuMatrix *tmp_a = nullptr;
    GpuMatrix *tmp_b = nullptr;
    GpuMatrix *tmp_d = nullptr;
    GpuMatrix *tmp_tp2 = nullptr;
    std::vector<const GpuPoly *> a_inputs;
    std::vector<const GpuPoly *> b_inputs;
    std::vector<const GpuPoly *> d_inputs;
    std::vector<const GpuPoly *> tp2_inputs;

    auto cleanup = [&]()
    {
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

    auto collect_coeff_inputs = [&](const GpuMatrix *src, GpuMatrix **owned, std::vector<const GpuPoly *> &inputs) -> int
    {
        const size_t count = src->rows * src->cols;
        inputs.clear();
        inputs.reserve(count);
        const int batch = default_batch(src->ctx);
        *owned = nullptr;
        if (src->format == PolyFormat::Eval)
        {
            const int matrix_format =
                src->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
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
            for (auto *clone : (*owned)->polys)
            {
                status = gpu_poly_intt(clone, batch);
                if (status != 0)
                {
                    gpu_matrix_destroy(*owned);
                    *owned = nullptr;
                    return status;
                }
                inputs.push_back(clone);
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                inputs.push_back(src->polys[i]);
            }
        }
        return 0;
    };

    int status = collect_coeff_inputs(a_mat, &tmp_a, a_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(b_mat, &tmp_b, b_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(d_mat, &tmp_d, d_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(tp2, &tmp_tp2, tp2_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    // Ensure all pending INTT work has completed before cross-stream reads.
    for (int device : a_mat->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != a_mat->ctx || poly->level != level)
        {
            cleanup();
            return set_error("invalid output poly in gpu_matrix_sample_p1_full");
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }
    out->format = PolyFormat::Coeff;

    const dim3 ref_limb_id = limb_map[0];
    std::vector<const uint64_t *> ref_a_entry_ptrs;
    std::vector<const uint64_t *> ref_b_entry_ptrs;
    std::vector<const uint64_t *> ref_d_entry_ptrs;
    std::vector<const uint64_t *> ref_tp2_entry_ptrs;
    ref_a_entry_ptrs.reserve(d_rows * d_rows);
    ref_b_entry_ptrs.reserve(d_rows * d_rows);
    ref_d_entry_ptrs.reserve(d_rows * d_rows);
    ref_tp2_entry_ptrs.reserve(2 * d_rows * cols);

    cudaStream_t ref_stream = nullptr;
    int ref_device = -1;
    for (size_t i = 0; i < d_rows; ++i)
    {
        for (size_t j = 0; j < d_rows; ++j)
        {
            const size_t idx = matrix_index(i, j, d_rows);
            if (ref_limb_id.x >= a_inputs[idx]->poly->GPU.size() ||
                ref_limb_id.x >= b_inputs[idx]->poly->GPU.size() ||
                ref_limb_id.x >= d_inputs[idx]->poly->GPU.size())
            {
                cleanup();
                return set_error("unexpected A/B/D limb GPU partition in gpu_matrix_sample_p1_full");
            }
            const auto &a_part = a_inputs[idx]->poly->GPU[ref_limb_id.x];
            const auto &b_part = b_inputs[idx]->poly->GPU[ref_limb_id.x];
            const auto &d_part = d_inputs[idx]->poly->GPU[ref_limb_id.x];
            if (ref_limb_id.y >= a_part.limb.size() ||
                ref_limb_id.y >= b_part.limb.size() ||
                ref_limb_id.y >= d_part.limb.size())
            {
                cleanup();
                return set_error("unexpected A/B/D limb index in gpu_matrix_sample_p1_full");
            }
            const auto &a_impl = a_part.limb[ref_limb_id.y];
            const auto &b_impl = b_part.limb[ref_limb_id.y];
            const auto &d_impl = d_part.limb[ref_limb_id.y];
            if (a_impl.index() != FIDESlib::U64 ||
                b_impl.index() != FIDESlib::U64 ||
                d_impl.index() != FIDESlib::U64)
            {
                cleanup();
                return set_error("unsupported A/B/D limb type in gpu_matrix_sample_p1_full");
            }
            ref_a_entry_ptrs.push_back(std::get<FIDESlib::U64>(a_impl).v.data);
            ref_b_entry_ptrs.push_back(std::get<FIDESlib::U64>(b_impl).v.data);
            ref_d_entry_ptrs.push_back(std::get<FIDESlib::U64>(d_impl).v.data);
        }
    }
    for (size_t row = 0; row < 2 * d_rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            const size_t idx = matrix_index(row, col, cols);
            if (ref_limb_id.x >= tp2_inputs[idx]->poly->GPU.size() ||
                ref_limb_id.x >= out->polys[idx]->poly->GPU.size())
            {
                cleanup();
                return set_error("unexpected tp2/output limb GPU partition in gpu_matrix_sample_p1_full");
            }
            const auto &tp2_part = tp2_inputs[idx]->poly->GPU[ref_limb_id.x];
            auto &out_part = out->polys[idx]->poly->GPU[ref_limb_id.x];
            if (tp2_part.device != out_part.device)
            {
                cleanup();
                return set_error("input/output limb device mismatch in gpu_matrix_sample_p1_full");
            }
            if (ref_device < 0)
            {
                ref_device = out_part.device;
            }
            else if (ref_device != out_part.device)
            {
                cleanup();
                return set_error("mixed reference output devices in gpu_matrix_sample_p1_full");
            }
            if (ref_limb_id.y >= tp2_part.limb.size() || ref_limb_id.y >= out_part.limb.size())
            {
                cleanup();
                return set_error("unexpected tp2/output limb index in gpu_matrix_sample_p1_full");
            }
            const auto &tp2_impl = tp2_part.limb[ref_limb_id.y];
            auto &out_impl = out_part.limb[ref_limb_id.y];
            if (tp2_impl.index() != FIDESlib::U64 || out_impl.index() != FIDESlib::U64)
            {
                cleanup();
                return set_error("unsupported tp2/output limb type in gpu_matrix_sample_p1_full");
            }
            const auto &tp2_u64 = std::get<FIDESlib::U64>(tp2_impl);
            auto &out_u64 = std::get<FIDESlib::U64>(out_impl);
            if (!ref_stream)
            {
                ref_stream = out_u64.stream.ptr;
            }
            ref_tp2_entry_ptrs.push_back(tp2_u64.v.data);
        }
    }
    if (!ref_stream || ref_device < 0)
    {
        cleanup();
        return set_error("invalid reference stream/device in gpu_matrix_sample_p1_full");
    }

    std::vector<int64_t> sampled_out_host;
    status = launch_sample_p1_integer_kernel(
        ref_a_entry_ptrs,
        ref_b_entry_ptrs,
        ref_d_entry_ptrs,
        ref_tp2_entry_ptrs,
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
        sampled_out_host);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> out_entry_ptrs;
        out_entry_ptrs.reserve(2 * d_rows * cols);

        cudaStream_t out_stream = nullptr;
        int out_device = -1;
        for (size_t row = 0; row < 2 * d_rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                const size_t idx = matrix_index(row, col, cols);
                if (limb_id.x >= out->polys[idx]->poly->GPU.size())
                {
                    cleanup();
                    return set_error("unexpected output limb GPU partition in gpu_matrix_sample_p1_full");
                }
                auto &out_part = out->polys[idx]->poly->GPU[limb_id.x];
                if (out_device < 0)
                {
                    out_device = out_part.device;
                }
                else if (out_device != out_part.device)
                {
                    cleanup();
                    return set_error("mixed output devices in gpu_matrix_sample_p1_full");
                }
                if (limb_id.y >= out_part.limb.size())
                {
                    cleanup();
                    return set_error("unexpected output limb index in gpu_matrix_sample_p1_full");
                }
                auto &out_impl = out_part.limb[limb_id.y];
                if (out_impl.index() != FIDESlib::U64)
                {
                    cleanup();
                    return set_error("unsupported output limb type in gpu_matrix_sample_p1_full");
                }
                auto &out_u64 = std::get<FIDESlib::U64>(out_impl);
                if (!out_stream)
                {
                    out_stream = out_u64.stream.ptr;
                }
                out_entry_ptrs.push_back(out_u64.v.data);
            }
        }
        if (!out_stream || out_device < 0)
        {
            cleanup();
            return set_error("invalid output stream/device in gpu_matrix_sample_p1_full");
        }

        status = launch_scatter_p1_integer_to_limb_kernel(
            sampled_out_host,
            out_entry_ptrs,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup();
    return 0;
}
