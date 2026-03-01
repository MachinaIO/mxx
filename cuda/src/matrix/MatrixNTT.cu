__constant__ uint64_t gpu_ntt_const_moduli[GPU_RUNTIME_MAX_LIMBS];
__constant__ uint64_t gpu_ntt_const_root[GPU_RUNTIME_MAX_LIMBS];
__constant__ uint64_t gpu_ntt_const_inv_root[GPU_RUNTIME_MAX_LIMBS];
__constant__ uint64_t gpu_ntt_const_n_inv[GPU_RUNTIME_MAX_LIMBS];
__constant__ uint32_t gpu_ntt_const_limb_count;

namespace
{
    constexpr uint32_t kTransformThreads = 256;
    constexpr size_t kMaxGridY = 65535;
    constexpr size_t kMaxGridZ = 65535;

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

    __global__ void ntt_twist_all_limbs_kernel(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        size_t poly_offset,
        bool inverse)
    {
        const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (coeff_idx >= n)
        {
            return;
        }
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
        uint64_t *const base = limb_bases[limb_idx];
        const size_t stride = limb_strides[limb_idx];
        if (limb_idx >= static_cast<size_t>(gpu_ntt_const_limb_count))
        {
            return;
        }
        const uint64_t twiddle_base = inverse ? gpu_ntt_const_inv_root[limb_idx] : gpu_ntt_const_root[limb_idx];
        const uint64_t modulus = gpu_ntt_const_moduli[limb_idx];
        uint64_t *const poly = base + poly_idx * stride;
        const uint64_t tw = pow_mod_u64_device(twiddle_base, coeff_idx, modulus);
        poly[coeff_idx] = mul_mod_u64(poly[coeff_idx], tw, modulus);
    }

    __global__ void ntt_scale_all_limbs_kernel(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        size_t poly_offset)
    {
        const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (coeff_idx >= n)
        {
            return;
        }
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
        uint64_t *const base = limb_bases[limb_idx];
        const size_t stride = limb_strides[limb_idx];
        if (limb_idx >= static_cast<size_t>(gpu_ntt_const_limb_count))
        {
            return;
        }
        const uint64_t factor = gpu_ntt_const_n_inv[limb_idx];
        const uint64_t modulus = gpu_ntt_const_moduli[limb_idx];
        uint64_t *const poly = base + poly_idx * stride;
        poly[coeff_idx] = mul_mod_u64(poly[coeff_idx], factor, modulus);
    }

    __global__ void ntt_bit_reverse_all_limbs_kernel(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        uint32_t log_n,
        size_t poly_offset)
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
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
        uint64_t *const base = limb_bases[limb_idx];
        const size_t stride = limb_strides[limb_idx];
        uint64_t *const poly = base + poly_idx * stride;
        const uint64_t tmp = poly[idx];
        poly[idx] = poly[rev];
        poly[rev] = tmp;
    }

    __global__ void ntt_stage_all_limbs_kernel(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        const uint64_t *limb_wlens,
        size_t limb_count,
        uint32_t n,
        uint32_t len,
        size_t poly_offset)
    {
        const uint32_t bfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t butterflies = n >> 1;
        if (bfly_idx >= butterflies)
        {
            return;
        }
        const size_t limb_idx = static_cast<size_t>(blockIdx.z);
        if (limb_idx >= limb_count)
        {
            return;
        }
        const uint32_t half = len >> 1;
        const uint32_t group = bfly_idx / half;
        const uint32_t j = bfly_idx - group * half;
        const uint32_t i = group * len + j;
        const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
        uint64_t *const base = limb_bases[limb_idx];
        const size_t stride = limb_strides[limb_idx];
        if (limb_idx >= static_cast<size_t>(gpu_ntt_const_limb_count))
        {
            return;
        }
        const uint64_t wlen = limb_wlens[limb_idx];
        const uint64_t modulus = gpu_ntt_const_moduli[limb_idx];
        uint64_t *const poly = base + poly_idx * stride;
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

    int launch_twist_for_all_limbs(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream,
        bool inverse)
    {
        if (!limb_bases || !limb_strides)
        {
            return set_error("null metadata in launch_twist_for_all_limbs");
        }
        if (limb_count == 0 || poly_count == 0)
        {
            return 0;
        }
        if (limb_count > kMaxGridZ)
        {
            return set_error("too many limbs in launch_twist_for_all_limbs");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(limb_count)};
            ntt_twist_all_limbs_kernel<<<grid, kTransformThreads, 0, stream>>>(
                limb_bases,
                limb_strides,
                limb_count,
                n,
                offset,
                inverse);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_scale_for_all_limbs(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!limb_bases || !limb_strides)
        {
            return set_error("null metadata in launch_scale_for_all_limbs");
        }
        if (limb_count == 0 || poly_count == 0)
        {
            return 0;
        }
        if (limb_count > kMaxGridZ)
        {
            return set_error("too many limbs in launch_scale_for_all_limbs");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(limb_count)};
            ntt_scale_all_limbs_kernel<<<grid, kTransformThreads, 0, stream>>>(
                limb_bases,
                limb_strides,
                limb_count,
                n,
                offset);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_bit_reverse_for_all_limbs(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        size_t limb_count,
        uint32_t n,
        uint32_t log_n,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!limb_bases || !limb_strides)
        {
            return set_error("null metadata in launch_bit_reverse_for_all_limbs");
        }
        if (limb_count == 0 || poly_count == 0)
        {
            return 0;
        }
        if (limb_count > kMaxGridZ)
        {
            return set_error("too many limbs in launch_bit_reverse_for_all_limbs");
        }
        const uint32_t blocks_x = (n + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(limb_count)};
            ntt_bit_reverse_all_limbs_kernel<<<grid, kTransformThreads, 0, stream>>>(
                limb_bases,
                limb_strides,
                limb_count,
                n,
                log_n,
                offset);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int launch_stage_for_all_limbs(
        uint64_t *const *limb_bases,
        const size_t *limb_strides,
        const uint64_t *limb_wlens,
        size_t limb_count,
        uint32_t n,
        uint32_t len,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!limb_bases || !limb_strides || !limb_wlens)
        {
            return set_error("null metadata in launch_stage_for_all_limbs");
        }
        if (limb_count == 0 || poly_count == 0)
        {
            return 0;
        }
        if (limb_count > kMaxGridZ)
        {
            return set_error("too many limbs in launch_stage_for_all_limbs");
        }
        const uint32_t butterflies = n >> 1;
        const uint32_t blocks_x = (butterflies + kTransformThreads - 1) / kTransformThreads;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(limb_count)};
            ntt_stage_all_limbs_kernel<<<grid, kTransformThreads, 0, stream>>>(
                limb_bases,
                limb_strides,
                limb_wlens,
                limb_count,
                n,
                len,
                offset);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    template <bool Forward>
    int run_matrix_transform_u64(GpuMatrix *mat)
    {
        if (!mat || !mat->ctx)
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
        uint32_t log_n = 0;
        uint32_t tmp_n = n;
        while ((tmp_n & 1U) == 0U && tmp_n > 1U)
        {
            ++log_n;
            tmp_n >>= 1U;
        }
        if (!is_power_of_two_u32(n) || tmp_n != 1U || log_n == 0)
        {
            return set_error("invalid ring size/logN in run_matrix_transform_u64");
        }

        auto &limb_map = mat->ctx->limb_gpu_ids;
        auto &limb_types = mat->ctx->limb_types;
        auto &limb_prime_ids = mat->ctx->limb_prime_ids;
        auto &moduli = mat->ctx->moduli;
        const size_t limb_count = static_cast<size_t>(mat->level + 1);
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in run_matrix_transform_u64");
        }
        if (limb_types.size() < limb_count || limb_prime_ids.size() < limb_count)
        {
            return set_error("unexpected limb metadata size in run_matrix_transform_u64");
        }
        if (limb_count > kMaxGridZ)
        {
            return set_error("too many limbs in run_matrix_transform_u64");
        }

        const size_t poly_count = matrix_poly_count(mat);
        if (poly_count == 0)
        {
            mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            return 0;
        }

        std::vector<dim3> active_limb_ids(limb_count);
        std::vector<uint64_t *> limb_bases(limb_count, nullptr);
        std::vector<size_t> limb_strides(limb_count, 0);

        int dispatch_device = -1;
        size_t dispatch_slot = std::numeric_limits<size_t>::max();
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;

        for (int limb = 0; limb <= mat->level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
            active_limb_ids[limb_idx] = limb_id;
            if (limb_id.x >= mat->shared_limb_buffers.size())
            {
                return set_error("invalid shared limb partition in run_matrix_transform_u64");
            }

            int limb_device = -1;
            status = matrix_limb_device(mat, limb_id, &limb_device);
            if (status != 0)
            {
                return status;
            }
            if (limb == 0)
            {
                dispatch_device = limb_device;
                dispatch_slot = static_cast<size_t>(limb_id.x);
                status = matrix_limb_stream(mat, limb_id, &dispatch_stream);
                if (status != 0)
                {
                    return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in run_matrix_transform_u64");
                }
            }
            else if (limb_device != dispatch_device)
            {
                return set_error("single-device mode requires all limbs on one device in run_matrix_transform_u64");
            }

            if (limb_types[limb_idx] != GPU_LIMB_U64)
            {
                return set_error("unsupported limb type in run_matrix_transform_u64");
            }
            const int primeid = limb_prime_ids[limb_idx];
            if (primeid < 0 ||
                static_cast<size_t>(primeid) >= moduli.size() ||
                limb_id.x >= mat->ctx->gpu_ids.size())
            {
                return set_error("invalid prime/device index in run_matrix_transform_u64");
            }

            const auto &buffer = mat->shared_limb_buffers[limb_id.x];
            if (buffer.words_per_poly < static_cast<size_t>(n))
            {
                return set_error("invalid words_per_poly in run_matrix_transform_u64");
            }
            limb_strides[limb_idx] = buffer.words_per_poly;

            uint64_t *base = matrix_limb_ptr_by_id(mat, 0, limb_id);
            if (!base)
            {
                return set_error("null matrix limb pointer in run_matrix_transform_u64");
            }
            limb_bases[limb_idx] = base;
        }

        if (dispatch_slot >= mat->ctx->ntt_device_constants.size())
        {
            return set_error("missing per-device NTT constants in run_matrix_transform_u64");
        }
        const GpuNttDeviceConstants &device_constants = mat->ctx->ntt_device_constants[dispatch_slot];
        if (device_constants.device != dispatch_device)
        {
            return set_error("NTT constants device mismatch in run_matrix_transform_u64");
        }
        if (device_constants.limb_count < limb_count)
        {
            return set_error("insufficient NTT limb constants in run_matrix_transform_u64");
        }
        if (device_constants.stage_count != log_n)
        {
            return set_error("NTT stage constants mismatch in run_matrix_transform_u64");
        }
        if (!device_constants.limb_wlen_forward ||
            !device_constants.limb_wlen_inverse)
        {
            return set_error("null per-device NTT constants in run_matrix_transform_u64");
        }

        for (const dim3 &limb_id : active_limb_ids)
        {
            status = matrix_wait_limb_stream(mat, limb_id, dispatch_device, dispatch_stream);
            if (status != 0)
            {
                return status;
            }
        }

        const size_t ptr_bytes = limb_count * sizeof(uint64_t *);
        const size_t stride_bytes = limb_count * sizeof(size_t);
        uint64_t **limb_bases_device = nullptr;
        size_t *limb_strides_device = nullptr;
        const uint64_t *limb_wlens_base =
            Forward ? device_constants.limb_wlen_forward : device_constants.limb_wlen_inverse;
        auto cleanup = [&]()
        {
            if (dispatch_device >= 0)
            {
                cudaSetDevice(dispatch_device);
            }
            if (limb_bases_device)
            {
                cudaFreeAsync(limb_bases_device, dispatch_stream);
                limb_bases_device = nullptr;
            }
            if (limb_strides_device)
            {
                cudaFreeAsync(limb_strides_device, dispatch_stream);
                limb_strides_device = nullptr;
            }
        };

        cudaError_t err = cudaSetDevice(dispatch_device);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&limb_bases_device), ptr_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMallocAsync(reinterpret_cast<void **>(&limb_strides_device), stride_bytes, dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        err = cudaMemcpyAsync(
            limb_bases_device,
            limb_bases.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            limb_strides_device,
            limb_strides.data(),
            stride_bytes,
            cudaMemcpyHostToDevice,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        if constexpr (Forward)
        {
            status = launch_twist_for_all_limbs(
                limb_bases_device,
                limb_strides_device,
                limb_count,
                n,
                poly_count,
                dispatch_stream,
                false);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = launch_bit_reverse_for_all_limbs(
                limb_bases_device,
                limb_strides_device,
                limb_count,
                n,
                log_n,
                poly_count,
                dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            for (uint32_t stage_idx = 0, len = 2; len <= n; len <<= 1, ++stage_idx)
            {
                const uint64_t *limb_wlens_device =
                    limb_wlens_base + static_cast<size_t>(stage_idx) * device_constants.limb_count;
                status = launch_stage_for_all_limbs(
                    limb_bases_device,
                    limb_strides_device,
                    limb_wlens_device,
                    limb_count,
                    n,
                    len,
                    poly_count,
                    dispatch_stream);
                if (status != 0)
                {
                    cleanup();
                    return status;
                }
            }
        }
        else
        {
            status = launch_bit_reverse_for_all_limbs(
                limb_bases_device,
                limb_strides_device,
                limb_count,
                n,
                log_n,
                poly_count,
                dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            for (uint32_t stage_idx = 0, len = 2; len <= n; len <<= 1, ++stage_idx)
            {
                const uint64_t *limb_wlens_device =
                    limb_wlens_base + static_cast<size_t>(stage_idx) * device_constants.limb_count;
                status = launch_stage_for_all_limbs(
                    limb_bases_device,
                    limb_strides_device,
                    limb_wlens_device,
                    limb_count,
                    n,
                    len,
                    poly_count,
                    dispatch_stream);
                if (status != 0)
                {
                    cleanup();
                    return status;
                }
            }
            status = launch_scale_for_all_limbs(
                limb_bases_device,
                limb_strides_device,
                limb_count,
                n,
                poly_count,
                dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
            status = launch_twist_for_all_limbs(
                limb_bases_device,
                limb_strides_device,
                limb_count,
                n,
                poly_count,
                dispatch_stream,
                true);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        for (const dim3 &limb_id : active_limb_ids)
        {
            status = matrix_record_limb_write(mat, limb_id, dispatch_stream);
            if (status != 0)
            {
                cleanup();
                return status;
            }
        }

        mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        cleanup();
        return 0;
    }
}

int gpu_matrix_ntt_all(GpuMatrix *mat, int batch)
{
    (void)batch;
    if (!mat || !mat->ctx)
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
    if (!mat || !mat->ctx)
    {
        return set_error("invalid gpu_matrix_intt_all arguments");
    }
    if (mat->format == GPU_POLY_FORMAT_COEFF)
    {
        return 0;
    }
    return run_matrix_transform_u64<false>(mat);
}
