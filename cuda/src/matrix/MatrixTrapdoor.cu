constexpr size_t kSampleP1LocalMaxM = 8;

__global__ void matrix_sample_p1_integer_kernel_small(
    const uint64_t **a_entries,
    const uint64_t **b_entries,
    const uint64_t **d_entries,
    const uint64_t **tp2_entries,
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

__global__ void matrix_sample_p1_integer_kernel_large(
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
    cudaStream_t stream,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id)
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

    const size_t ptr_bytes = ptr_count * sizeof(uint64_t *);
    const size_t u64_bytes = job_count * sizeof(uint64_t);
    const size_t u32_bytes = job_count * sizeof(uint32_t);
    const size_t dst_offset = matrix_align_up_size(ptr_bytes, alignof(uint64_t *));
    const size_t tower_moduli_offset =
        matrix_align_up_size(dst_offset + ptr_bytes, alignof(uint64_t));
    const size_t digit_indices_offset =
        matrix_align_up_size(tower_moduli_offset + u64_bytes, alignof(uint32_t));
    const size_t tower_indices_offset =
        matrix_align_up_size(digit_indices_offset + u32_bytes, alignof(uint32_t));
    const size_t out_moduli_offset =
        matrix_align_up_size(tower_indices_offset + u32_bytes, alignof(uint64_t));
    const size_t workspace_bytes = out_moduli_offset + u64_bytes;

    void *workspace = nullptr;
    bool from_shared = false;
    int status = matrix_acquire_aux_workspace(
        aux_owner,
        aux_limb_id,
        workspace_bytes,
        &workspace,
        &from_shared);
    if (status != 0)
    {
        return status;
    }
    auto cleanup_workspace = [&]() -> int { return matrix_release_aux_workspace(workspace, from_shared); };

    auto *workspace_base = reinterpret_cast<uint8_t *>(workspace);
    const uint64_t **d_src = reinterpret_cast<const uint64_t **>(workspace_base);
    uint64_t **d_dst = reinterpret_cast<uint64_t **>(workspace_base + dst_offset);
    uint64_t *d_tower_moduli = reinterpret_cast<uint64_t *>(workspace_base + tower_moduli_offset);
    uint32_t *d_digit_indices = reinterpret_cast<uint32_t *>(workspace_base + digit_indices_offset);
    uint32_t *d_tower_indices = reinterpret_cast<uint32_t *>(workspace_base + tower_indices_offset);
    uint64_t *d_out_moduli = reinterpret_cast<uint64_t *>(workspace_base + out_moduli_offset);

    err = cudaMemcpyAsync(d_src, src_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_moduli, tower_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_digit_indices, digit_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_tower_indices, tower_indices.data(), u32_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaMemcpyAsync(d_out_moduli, out_moduli.data(), u64_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
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
        cleanup_workspace();
        return set_error(err);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    status = cleanup_workspace();
    if (status != 0)
    {
        return status;
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
    int64_t **sampled_out_device,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id)
{
    if (!sampled_out_device)
    {
        return set_error("null output pointer in matrix_sample_p1_integer_kernel");
    }
    *sampled_out_device = nullptr;
    if (d == 0 || cols == 0 || n == 0)
    {
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

    const size_t mat_bytes = mat_entries * sizeof(uint64_t *);
    const size_t vec_bytes = vec_entries * sizeof(uint64_t *);
    const size_t b_offset = matrix_align_up_size(mat_bytes, alignof(uint64_t *));
    const size_t d_offset = matrix_align_up_size(b_offset + mat_bytes, alignof(uint64_t *));
    const size_t tp2_offset = matrix_align_up_size(d_offset + mat_bytes, alignof(uint64_t *));
    const size_t metadata_bytes = tp2_offset + vec_bytes;

    void *metadata_workspace = nullptr;
    bool metadata_from_shared = false;
    int status = matrix_acquire_aux_workspace(
        aux_owner,
        aux_limb_id,
        metadata_bytes,
        &metadata_workspace,
        &metadata_from_shared);
    if (status != 0)
    {
        return status;
    }

    auto *metadata_base = reinterpret_cast<uint8_t *>(metadata_workspace);
    const uint64_t **d_a_entries = reinterpret_cast<const uint64_t **>(metadata_base);
    const uint64_t **d_b_entries = reinterpret_cast<const uint64_t **>(metadata_base + b_offset);
    const uint64_t **d_d_entries = reinterpret_cast<const uint64_t **>(metadata_base + d_offset);
    const uint64_t **d_tp2_entries =
        reinterpret_cast<const uint64_t **>(metadata_base + tp2_offset);
    int64_t *d_sampled_out = nullptr;

    auto free_all = [&]()
    {
        if (d_sampled_out)
            cudaFree(d_sampled_out);
        if (metadata_workspace)
        {
            matrix_release_aux_workspace(metadata_workspace, metadata_from_shared);
            metadata_workspace = nullptr;
        }
    };

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

    const int threads = 256;
    if (m <= kSampleP1LocalMaxM)
    {
        const int blocks = static_cast<int>((total_samples + threads - 1) / threads);
        matrix_sample_p1_integer_kernel_small<<<blocks, threads, 0, stream>>>(
            d_a_entries,
            d_b_entries,
            d_d_entries,
            d_tp2_entries,
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
        free_workspace();
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        free_all();
        return set_error(err);
    }

    *sampled_out_device = d_sampled_out;
    d_sampled_out = nullptr;
    free_all();
    return 0;
}

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    const std::vector<uint64_t *> &out_entries,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id,
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id)
{
    if (out_entries.empty() || n == 0)
    {
        return 0;
    }
    if (!sampled_in_device)
    {
        return set_error("null sampled device buffer in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (device_id < 0)
    {
        return set_error("invalid device in matrix_scatter_p1_integer_to_limb_kernel");
    }
    const size_t total = out_entries.size() * n;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t out_ptr_bytes = out_entries.size() * sizeof(uint64_t *);
    void *workspace = nullptr;
    bool from_shared = false;
    int status = matrix_acquire_aux_workspace(
        aux_owner,
        aux_limb_id,
        out_ptr_bytes,
        &workspace,
        &from_shared);
    if (status != 0)
    {
        return status;
    }
    auto cleanup_workspace = [&]() -> int { return matrix_release_aux_workspace(workspace, from_shared); };
    uint64_t **d_out_entries = reinterpret_cast<uint64_t **>(workspace);

    err = cudaMemcpyAsync(d_out_entries, out_entries.data(), out_ptr_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_scatter_p1_integer_to_limb_kernel<<<blocks, threads, 0, stream>>>(
        sampled_in_device,
        d_out_entries,
        out_entries.size(),
        n,
        modulus);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cleanup_workspace();
        return set_error(err);
    }

    status = cleanup_workspace();
    if (status != 0)
    {
        return status;
    }
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

    GpuMatrix *tmp_inputs_matrix = nullptr;
    const GpuMatrix *inputs_matrix = src;
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };

    int status = sync_matrix_limb_streams(
        src,
        "failed to synchronize source limb stream before clone in gpu_matrix_gauss_samp_gq_arb_base");
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }

    const int batch = default_batch(src->ctx);
    if (src->format == GPU_POLY_FORMAT_EVAL)
    {
        const int matrix_format =
            src->format == GPU_POLY_FORMAT_EVAL ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
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
        status = gpu_matrix_intt_all(tmp_inputs_matrix, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        inputs_matrix = tmp_inputs_matrix;
    }

    status = sync_matrix_limb_streams(
        inputs_matrix,
        "failed to synchronize input limb stream before gauss kernel in gpu_matrix_gauss_samp_gq_arb_base");
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    const size_t out_count = out->rows * out->cols;
    std::vector<std::pair<int, cudaStream_t>> out_zero_streams;
    out_zero_streams.reserve(crt_depth);
    auto add_out_stream = [&](int device, cudaStream_t stream) {
        for (const auto &entry : out_zero_streams)
        {
            if (entry.first == device && entry.second == stream)
            {
                return;
            }
        }
        out_zero_streams.emplace_back(device, stream);
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            cleanup_tmp_inputs();
            return set_error(err);
        }
        uint64_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb pointer in gpu_matrix_gauss_samp_gq_arb_base");
        }
        if (limb_id.x >= out->shared_limb_buffers.size())
        {
            cleanup_tmp_inputs();
            return set_error("invalid partition index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &buffer = out->shared_limb_buffers[limb_id.x];
        const size_t dst_pitch = buffer.words_per_poly * sizeof(uint64_t);
        const size_t coeff_bytes = static_cast<size_t>(src->ctx->N) * sizeof(uint64_t);
        if (out_count > 0)
        {
            err = cudaMemset2DAsync(dst, dst_pitch, 0, coeff_bytes, out_count, out_stream);
            if (err != cudaSuccess)
            {
                cleanup_tmp_inputs();
                return set_error(err);
            }
        }
        add_out_stream(out_device, out_stream);
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
        dim3 aux_limb_id;
        bool has_aux_limb_id;
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
        batches.push_back(GaussBatch{device, nullptr, {}, {}, {}, {}, {}, {}, dim3{0, 0, 0}, false});
        return batches.back();
    };

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        int src_device = -1;
        status = matrix_limb_device(inputs_matrix, src_limb_id, &src_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }

        std::vector<const uint64_t *> src_ptrs;
        src_ptrs.reserve(count);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint64_t *src_ptr = matrix_limb_ptr_by_id(inputs_matrix, idx, src_limb_id);
            if (!src_ptr)
            {
                cleanup_tmp_inputs();
                return set_error("null source limb pointer in gpu_matrix_gauss_samp_gq_arb_base");
            }
            src_ptrs.push_back(src_ptr);
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                int out_device = -1;
                status = matrix_limb_device(out, out_limb_id, &out_device);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }
                if (out_device != src_device)
                {
                    cleanup_tmp_inputs();
                    return set_error("input/output limb device mismatch in gpu_matrix_gauss_samp_gq_arb_base");
                }
                cudaStream_t out_stream = nullptr;
                status = matrix_limb_stream(out, out_limb_id, &out_stream);
                if (status != 0)
                {
                    cleanup_tmp_inputs();
                    return status;
                }

                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                for (size_t idx = 0; idx < count; ++idx)
                {
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    uint64_t *dst_ptr = matrix_limb_ptr_by_id(out, out_idx, out_limb_id);
                    if (!dst_ptr)
                    {
                        cleanup_tmp_inputs();
                        return set_error("null output limb pointer in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    dst_ptrs.push_back(dst_ptr);
                }

                auto &batch_ref = get_batch(out_device);
                if (!batch_ref.stream)
                {
                    batch_ref.stream = out_stream;
                }
                if (!batch_ref.has_aux_limb_id)
                {
                    batch_ref.aux_limb_id = out_limb_id;
                    batch_ref.has_aux_limb_id = true;
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
        status = launch_gauss_samp_gq_arb_base_multi_kernel(
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
            batch_ref.stream,
            out,
            batch_ref.has_aux_limb_id ? &batch_ref.aux_limb_id : nullptr);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    if (requested_out_format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_ntt_all(out, batch);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        out->format = GPU_POLY_FORMAT_EVAL;
    }

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
    auto &limb_map = a_mat->ctx->ctx->limbGPUid;
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
    };
    std::vector<DeviceSampleBuffer> sampled_device_buffers;

    auto cleanup = [&]()
    {
        for (auto &entry : sampled_device_buffers)
        {
            if (!entry.ptr || entry.device < 0)
            {
                continue;
            }
            cudaSetDevice(entry.device);
            cudaFree(entry.ptr);
            entry.ptr = nullptr;
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
                                          const GpuMatrix **coeff_input,
                                          const char *sync_context) -> int
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
        return sync_matrix_limb_streams(*coeff_input, sync_context);
    };

    int status = collect_coeff_input_matrix(
        a_mat,
        &tmp_a,
        &a_input,
        "failed to synchronize A limb streams in gpu_matrix_sample_p1_full");
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        b_mat,
        &tmp_b,
        &b_input,
        "failed to synchronize B limb streams in gpu_matrix_sample_p1_full");
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        d_mat,
        &tmp_d,
        &d_input,
        "failed to synchronize D limb streams in gpu_matrix_sample_p1_full");
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        tp2,
        &tmp_tp2,
        &tp2_input,
        "failed to synchronize tp2 limb streams in gpu_matrix_sample_p1_full");
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

    std::vector<const uint64_t *> ref_a_entry_ptrs;
    std::vector<const uint64_t *> ref_b_entry_ptrs;
    std::vector<const uint64_t *> ref_d_entry_ptrs;
    std::vector<const uint64_t *> ref_tp2_entry_ptrs;
    ref_a_entry_ptrs.reserve(d_rows * d_rows);
    ref_b_entry_ptrs.reserve(d_rows * d_rows);
    ref_d_entry_ptrs.reserve(d_rows * d_rows);
    ref_tp2_entry_ptrs.reserve(2 * d_rows * cols);

    for (size_t i = 0; i < d_rows; ++i)
    {
        for (size_t j = 0; j < d_rows; ++j)
        {
            const size_t idx = matrix_index(i, j, d_rows);
            const uint64_t *a_ptr = matrix_limb_ptr_by_id(a_input, idx, ref_limb_id);
            const uint64_t *b_ptr = matrix_limb_ptr_by_id(b_input, idx, ref_limb_id);
            const uint64_t *d_ptr = matrix_limb_ptr_by_id(d_input, idx, ref_limb_id);
            if (!a_ptr || !b_ptr || !d_ptr)
            {
                cleanup();
                return set_error("null A/B/D limb pointer in gpu_matrix_sample_p1_full");
            }
            ref_a_entry_ptrs.push_back(a_ptr);
            ref_b_entry_ptrs.push_back(b_ptr);
            ref_d_entry_ptrs.push_back(d_ptr);
        }
    }
    for (size_t row = 0; row < 2 * d_rows; ++row)
    {
        for (size_t col = 0; col < cols; ++col)
        {
            const size_t idx = matrix_index(row, col, cols);
            const uint64_t *tp2_ptr = matrix_limb_ptr_by_id(tp2_input, idx, ref_limb_id);
            uint64_t *out_ptr = matrix_limb_ptr_by_id(out, idx, ref_limb_id);
            if (!tp2_ptr || !out_ptr)
            {
                cleanup();
                return set_error("null tp2/output limb pointer in gpu_matrix_sample_p1_full");
            }
            ref_tp2_entry_ptrs.push_back(tp2_ptr);
        }
    }
    if (!ref_stream || ref_device < 0)
    {
        cleanup();
        return set_error("invalid reference stream/device in gpu_matrix_sample_p1_full");
    }

    int64_t *sampled_ref_device = nullptr;
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
        &sampled_ref_device,
        out,
        &ref_limb_id);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    if (sampled_ref_device)
    {
        sampled_device_buffers.push_back(DeviceSampleBuffer{ref_device, sampled_ref_device});
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
        *out_ptr = nullptr;
        for (const auto &entry : sampled_device_buffers)
        {
            if (entry.device == device && entry.ptr)
            {
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
        err = cudaMalloc(&device_copy, sampled_bytes);
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
            cudaFree(device_copy);
            return set_error(err);
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(device_copy);
            return set_error(err);
        }
        sampled_device_buffers.push_back(DeviceSampleBuffer{device, device_copy});
        *out_ptr = device_copy;
        return 0;
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> out_entry_ptrs;
        out_entry_ptrs.reserve(2 * d_rows * cols);

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
        for (size_t row = 0; row < 2 * d_rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                const size_t idx = matrix_index(row, col, cols);
                uint64_t *out_ptr = matrix_limb_ptr_by_id(out, idx, limb_id);
                if (!out_ptr)
                {
                    cleanup();
                    return set_error("null output limb pointer in gpu_matrix_sample_p1_full");
                }
                out_entry_ptrs.push_back(out_ptr);
            }
        }

        int64_t *sampled_for_device = nullptr;
        status = ensure_sample_buffer_on_device(out_device, out_stream, &sampled_for_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_scatter_p1_integer_to_limb_kernel_device(
            sampled_for_device,
            out_entry_ptrs,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            out_stream,
            out_device,
            out,
            &limb_id);
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
