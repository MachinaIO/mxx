namespace
{
    constexpr uint32_t kTransformThreads = 256;
    constexpr size_t kMaxGridY = 65535;

    __device__ __forceinline__ uint64_t sub_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
    {
        if (a >= b)
        {
            return a - b;
        }
        return mod - (b - a);
    }

    __device__ __forceinline__ uint64_t pow_mod_u64_device(uint64_t base, uint32_t exp, uint64_t mod)
    {
        uint64_t result = 1 % mod;
        uint64_t cur = base % mod;
        uint32_t e = exp;
        while (e != 0)
        {
            if ((e & 1U) != 0)
            {
                result = mul_mod_u64(result, cur, mod);
            }
            cur = mul_mod_u64(cur, cur, mod);
            e >>= 1;
        }
        return result;
    }

    uint64_t mul_mod_u64_host(uint64_t a, uint64_t b, uint64_t mod)
    {
        const unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % mod);
    }

    uint64_t pow_mod_u64_host(uint64_t base, uint32_t exp, uint64_t mod)
    {
        uint64_t result = 1 % mod;
        uint64_t cur = base % mod;
        uint32_t e = exp;
        while (e != 0)
        {
            if ((e & 1U) != 0)
            {
                result = mul_mod_u64_host(result, cur, mod);
            }
            cur = mul_mod_u64_host(cur, cur, mod);
            e >>= 1;
        }
        return result;
    }

    __global__ void ntt_twist_kernel(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint64_t twiddle_base,
        uint64_t modulus)
    {
        const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (coeff_idx >= n)
        {
            return;
        }
        const size_t poly_idx = static_cast<size_t>(blockIdx.y);
        uint64_t *poly = base + poly_idx * stride;
        const uint64_t tw = pow_mod_u64_device(twiddle_base, coeff_idx, modulus);
        poly[coeff_idx] = mul_mod_u64(poly[coeff_idx], tw, modulus);
    }

    __global__ void ntt_scale_kernel(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint64_t factor,
        uint64_t modulus)
    {
        const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (coeff_idx >= n)
        {
            return;
        }
        const size_t poly_idx = static_cast<size_t>(blockIdx.y);
        uint64_t *poly = base + poly_idx * stride;
        poly[coeff_idx] = mul_mod_u64(poly[coeff_idx], factor, modulus);
    }

    __global__ void ntt_bit_reverse_kernel(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint32_t log_n)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
        {
            return;
        }
        const uint32_t rev = __brev(idx) >> (32 - log_n);
        if (idx >= rev)
        {
            return;
        }
        const size_t poly_idx = static_cast<size_t>(blockIdx.y);
        uint64_t *poly = base + poly_idx * stride;
        const uint64_t tmp = poly[idx];
        poly[idx] = poly[rev];
        poly[rev] = tmp;
    }

    __global__ void ntt_stage_kernel(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint32_t len,
        uint64_t wlen,
        uint64_t modulus)
    {
        const uint32_t bfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t butterflies = n >> 1;
        if (bfly_idx >= butterflies)
        {
            return;
        }

        const uint32_t half = len >> 1;
        const uint32_t group = bfly_idx / half;
        const uint32_t j = bfly_idx - group * half;
        const uint32_t i = group * len + j;

        const size_t poly_idx = static_cast<size_t>(blockIdx.y);
        uint64_t *poly = base + poly_idx * stride;

        const uint64_t w = pow_mod_u64_device(wlen, j, modulus);
        const uint64_t u = poly[i];
        const uint64_t v = mul_mod_u64(poly[i + half], w, modulus);
        poly[i] = add_mod_u64(u, v, modulus);
        poly[i + half] = sub_mod_u64(u, v, modulus);
    }

    bool is_power_of_two_u32(uint32_t v)
    {
        return v != 0 && (v & (v - 1)) == 0;
    }

    int launch_twist_for_all_polys(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        size_t poly_count,
        uint64_t twiddle_base,
        uint64_t modulus,
        cudaStream_t stream)
    {
        if (!base)
        {
            return set_error("null pointer in launch_twist_for_all_polys");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            uint64_t *chunk_base = base + offset * stride;
            const dim3 grid{blocks_x, static_cast<uint32_t>(chunk)};
            ntt_twist_kernel<<<grid, kTransformThreads, 0, stream>>>(
                chunk_base,
                stride,
                n,
                twiddle_base,
                modulus);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_scale_for_all_polys(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        size_t poly_count,
        uint64_t factor,
        uint64_t modulus,
        cudaStream_t stream)
    {
        if (!base)
        {
            return set_error("null pointer in launch_scale_for_all_polys");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            uint64_t *chunk_base = base + offset * stride;
            const dim3 grid{blocks_x, static_cast<uint32_t>(chunk)};
            ntt_scale_kernel<<<grid, kTransformThreads, 0, stream>>>(chunk_base, stride, n, factor, modulus);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_bit_reverse_for_all_polys(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint32_t log_n,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!base)
        {
            return set_error("null pointer in launch_bit_reverse_for_all_polys");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            uint64_t *chunk_base = base + offset * stride;
            const dim3 grid{blocks_x, static_cast<uint32_t>(chunk)};
            ntt_bit_reverse_kernel<<<grid, kTransformThreads, 0, stream>>>(chunk_base, stride, n, log_n);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_stage_for_all_polys(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        size_t poly_count,
        uint32_t len,
        uint64_t wlen,
        uint64_t modulus,
        cudaStream_t stream)
    {
        if (!base)
        {
            return set_error("null pointer in launch_stage_for_all_polys");
        }
        const uint32_t butterflies = n >> 1;
        const uint32_t blocks_x = (butterflies + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            uint64_t *chunk_base = base + offset * stride;
            const dim3 grid{blocks_x, static_cast<uint32_t>(chunk)};
            ntt_stage_kernel<<<grid, kTransformThreads, 0, stream>>>(
                chunk_base,
                stride,
                n,
                len,
                wlen,
                modulus);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int run_limb_ntt_transform(
        uint64_t *base,
        size_t stride,
        uint32_t n,
        uint32_t log_n,
        size_t poly_count,
        uint64_t modulus,
        uint64_t n_inv,
        uint64_t root,
        uint64_t inv_root,
        cudaStream_t stream,
        bool forward)
    {
        if (!base)
        {
            return set_error("null pointer in run_limb_ntt_transform");
        }

        const uint64_t omega = mul_mod_u64_host(root, root, modulus);
        const uint64_t inv_omega = mul_mod_u64_host(inv_root, inv_root, modulus);

        int status = 0;
        if (forward)
        {
            status = launch_twist_for_all_polys(base, stride, n, poly_count, root, modulus, stream);
            if (status != 0)
            {
                return status;
            }
            status = launch_bit_reverse_for_all_polys(base, stride, n, log_n, poly_count, stream);
            if (status != 0)
            {
                return status;
            }
            for (uint32_t len = 2; len <= n; len <<= 1)
            {
                const uint32_t exp = n / len;
                const uint64_t wlen = pow_mod_u64_host(omega, exp, modulus);
                status = launch_stage_for_all_polys(base, stride, n, poly_count, len, wlen, modulus, stream);
                if (status != 0)
                {
                    return status;
                }
            }
        }
        else
        {
            status = launch_bit_reverse_for_all_polys(base, stride, n, log_n, poly_count, stream);
            if (status != 0)
            {
                return status;
            }
            for (uint32_t len = 2; len <= n; len <<= 1)
            {
                const uint32_t exp = n / len;
                const uint64_t wlen = pow_mod_u64_host(inv_omega, exp, modulus);
                status = launch_stage_for_all_polys(base, stride, n, poly_count, len, wlen, modulus, stream);
                if (status != 0)
                {
                    return status;
                }
            }
            status = launch_scale_for_all_polys(base, stride, n, poly_count, n_inv, modulus, stream);
            if (status != 0)
            {
                return status;
            }
            status = launch_twist_for_all_polys(base, stride, n, poly_count, inv_root, modulus, stream);
            if (status != 0)
            {
                return status;
            }
        }

        return 0;
    }

    template <bool Forward>
    int run_matrix_transform_u64(GpuMatrix *mat)
    {
        if (!mat || !mat->ctx || !mat->ctx->ctx)
        {
            return set_error("invalid matrix in run_matrix_transform_u64");
        }
        if (mat->level < 0)
        {
            return set_error("invalid level in run_matrix_transform_u64");
        }
        if (mat->ctx->N <= 0)
        {
            return set_error("invalid ring dimension in run_matrix_transform_u64");
        }

        const uint32_t n = static_cast<uint32_t>(mat->ctx->N);
        const uint32_t log_n = static_cast<uint32_t>(mat->ctx->ctx->logN);
        if (!is_power_of_two_u32(n) || (1U << log_n) != n || log_n == 0)
        {
            return set_error("invalid ring size/logN in run_matrix_transform_u64");
        }

        auto &limb_map = mat->ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(mat->level + 1))
        {
            return set_error("unexpected limb mapping size in run_matrix_transform_u64");
        }

        const size_t poly_count = matrix_poly_count(mat);
        if (poly_count == 0)
        {
            mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            return 0;
        }

        if (poly_count > std::numeric_limits<uint32_t>::max())
        {
            return set_error("poly_count too large in run_matrix_transform_u64");
        }

        for (int limb = 0; limb <= mat->level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= mat->shared_limb_buffers.size())
            {
                return set_error("invalid shared limb partition in run_matrix_transform_u64");
            }

            int device = -1;
            int status = matrix_limb_device(mat, limb_id, &device);
            if (status != 0)
            {
                return status;
            }
            cudaStream_t stream = nullptr;
            status = matrix_limb_stream(mat, limb_id, &stream);
            if (status != 0)
            {
                return status;
            }

            if (limb_id.x >= mat->ctx->ctx->meta.size() ||
                limb_id.y >= mat->ctx->ctx->meta[limb_id.x].size())
            {
                return set_error("invalid limb metadata in run_matrix_transform_u64");
            }
            const auto &record = mat->ctx->ctx->meta[limb_id.x][limb_id.y];
            if (record.type != FIDESlib::U64)
            {
                return set_error("unsupported limb type in run_matrix_transform_u64");
            }
            const int primeid = record.id;
            if (primeid < 0 || primeid >= FIDESlib::MAXP || limb_id.x >= FIDESlib::MAXD)
            {
                return set_error("invalid prime/device index in run_matrix_transform_u64");
            }

            const uint64_t modulus = FIDESlib::host_constants.primes[primeid];
            const uint64_t n_inv = FIDESlib::host_constants.N_inv[primeid];
            const uint64_t root = FIDESlib::host_global.root[primeid];
            const uint64_t inv_root = FIDESlib::host_global.inv_root[primeid];
            if (modulus == 0 || n_inv == 0 || root == 0 || inv_root == 0)
            {
                return set_error("invalid modulus/root constants in run_matrix_transform_u64");
            }

            const auto &buffer = mat->shared_limb_buffers[limb_id.x];
            if (buffer.words_per_poly < static_cast<size_t>(n))
            {
                return set_error("invalid words_per_poly in run_matrix_transform_u64");
            }

            uint64_t *base = matrix_limb_ptr_by_id(mat, 0, limb_id);
            if (!base)
            {
                return set_error("null matrix limb pointer in run_matrix_transform_u64");
            }

            status = run_limb_ntt_transform(
                base,
                buffer.words_per_poly,
                n,
                log_n,
                poly_count,
                modulus,
                n_inv,
                root,
                inv_root,
                stream,
                Forward);
            if (status != 0)
            {
                return status;
            }
        }

        mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        return 0;
    }
}

int gpu_matrix_ntt_all(GpuMatrix *mat, int batch)
{
    (void)batch;
    if (!mat || !mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid gpu_matrix_ntt_all arguments");
    }
    if (mat->format == GPU_POLY_FORMAT_EVAL)
    {
        return 0;
    }
    return run_matrix_transform_u64<true>(mat);
}

int gpu_matrix_intt_all(GpuMatrix *mat, int batch)
{
    (void)batch;
    if (!mat || !mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid gpu_matrix_intt_all arguments");
    }
    if (mat->format == GPU_POLY_FORMAT_COEFF)
    {
        return 0;
    }
    return run_matrix_transform_u64<false>(mat);
}
