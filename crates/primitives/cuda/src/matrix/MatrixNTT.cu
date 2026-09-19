namespace
{
    // Descriptors are owned by the matrix and initialized on its allocation
    // stream. Existing limb events protect both descriptor and coefficient use.
    struct MatrixNttDescriptorView
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        uint32_t indices[GPU_RUNTIME_MAX_LIMBS];
        __device__ size_t limb(size_t index) const { return index; }
        __device__ auto output(size_t index) const { return descriptors[indices[index]]; }
        __device__ auto input(size_t index) const { return output(index); }
    };
    static_assert(sizeof(MatrixNttDescriptorView) < 1024, "bounded NTT kernel arguments");

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

    __device__ __forceinline__ uint64_t mul_mod_shoup_u64(
        uint64_t value,
        uint64_t multiplier,
        uint64_t multiplier_shoup,
        uint64_t modulus)
    {
        if (modulus > (UINT64_MAX >> 1U))
        {
            return mul_mod_u64(value, multiplier, modulus);
        }
        const uint64_t quotient = __umul64hi(value, multiplier_shoup);
        uint64_t reduced = value * multiplier - quotient * modulus;
        if (reduced >= modulus) reduced -= modulus;
        return reduced;
    }

    __global__ void ntt_twist_all_limbs_kernel(
        MatrixNttDescriptorView layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
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
        const auto descriptor = layout.descriptors[layout.indices[limb_idx]];
        uint8_t *const base = descriptor.base;
        const size_t stride_bytes = descriptor.stride;
        const uint8_t coeff_bytes = descriptor.width;
        const uint64_t modulus = moduli[limb_idx];
        const size_t twiddle_idx = limb_idx * static_cast<size_t>(n) + coeff_idx;
        const uint64_t tw = twiddles[twiddle_idx];
        const uint64_t src = matrix_load_limb_u64(base, poly_idx, coeff_idx, stride_bytes, coeff_bytes);
        matrix_store_limb_u64(
            base,
            poly_idx,
            coeff_idx,
            stride_bytes,
            coeff_bytes,
            mul_mod_shoup_u64(src, tw, twiddle_shoup[twiddle_idx], modulus));
    }

    __global__ void ntt_scale_all_limbs_kernel(
        MatrixNttDescriptorView layout,
        const uint64_t *moduli,
        const uint64_t *n_inv,
        const uint64_t *n_inv_shoup,
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
        const auto descriptor = layout.descriptors[layout.indices[limb_idx]];
        uint8_t *const base = descriptor.base;
        const size_t stride_bytes = descriptor.stride;
        const uint8_t coeff_bytes = descriptor.width;
        const uint64_t factor = n_inv[limb_idx];
        const uint64_t modulus = moduli[limb_idx];
        const uint64_t src = matrix_load_limb_u64(base, poly_idx, coeff_idx, stride_bytes, coeff_bytes);
        matrix_store_limb_u64(
            base,
            poly_idx,
            coeff_idx,
            stride_bytes,
            coeff_bytes,
            mul_mod_shoup_u64(
                src,
                factor,
                n_inv_shoup[limb_idx],
                modulus));
    }

    template <bool ForwardDif>
    __global__ void ntt_stage_all_limbs_kernel(
        MatrixNttDescriptorView layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
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
        const auto descriptor = layout.descriptors[layout.indices[limb_idx]];
        uint8_t *const base = descriptor.base;
        const size_t stride_bytes = descriptor.stride;
        const uint8_t coeff_bytes = descriptor.width;
        const uint64_t modulus = moduli[limb_idx];
        const uint32_t twiddle_exponent = 2U * (n / len) * j;
        const size_t twiddle_idx =
            limb_idx * static_cast<size_t>(n) + twiddle_exponent;
        const uint64_t w = twiddles[twiddle_idx];
        const uint64_t u = matrix_load_limb_u64(base, poly_idx, i, stride_bytes, coeff_bytes);
        const uint64_t upper =
            matrix_load_limb_u64(base, poly_idx, i + half, stride_bytes, coeff_bytes);
        uint64_t lower_result;
        uint64_t upper_result;
        if constexpr (ForwardDif)
        {
            lower_result = add_mod_u64(u, upper, modulus);
            upper_result = mul_mod_shoup_u64(
                sub_mod_u64(u, upper, modulus), w, twiddle_shoup[twiddle_idx], modulus);
        }
        else
        {
            const uint64_t v = mul_mod_shoup_u64(upper, w, twiddle_shoup[twiddle_idx], modulus);
            lower_result = add_mod_u64(u, v, modulus);
            upper_result = sub_mod_u64(u, v, modulus);
        }
        matrix_store_limb_u64(base, poly_idx, i, stride_bytes, coeff_bytes, lower_result);
        matrix_store_limb_u64(base, poly_idx, i + half, stride_bytes, coeff_bytes, upper_result);
    }

    bool is_power_of_two_u32(uint32_t v)
    {
        return v != 0 && (v & (v - 1)) == 0;
    }

    int launch_twist_for_all_limbs(
        MatrixNttDescriptorView layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!layout.descriptors ||
            !twiddles || !twiddle_shoup || !moduli)
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
                layout,
                twiddles,
                twiddle_shoup,
                moduli,
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

    int launch_scale_for_all_limbs(
        MatrixNttDescriptorView layout,
        const uint64_t *moduli,
        const uint64_t *n_inv,
        const uint64_t *n_inv_shoup,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!layout.descriptors ||
            !moduli || !n_inv || !n_inv_shoup)
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
                layout,
                moduli,
                n_inv,
                n_inv_shoup,
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

    template <bool ForwardDif>
    int launch_stage_for_all_limbs(
        MatrixNttDescriptorView layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        uint32_t len,
        size_t poly_count,
        cudaStream_t stream)
    {
        if (!layout.descriptors ||
            !twiddles || !twiddle_shoup || !moduli)
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
            ntt_stage_all_limbs_kernel<ForwardDif><<<grid, kTransformThreads, 0, stream>>>(
                layout,
                twiddles,
                twiddle_shoup,
                moduli,
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

    // Ten local stages share one load/store pass. An 8 KiB tile avoids
    // architecture-specific shared-memory opt-in and leaves room for occupancy.
    constexpr uint32_t kFusedNttCoefficients = 1024;

    template <bool Forward, typename Layout>
    __global__ void ntt_fused_local_stages_kernel(
        Layout layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        const uint64_t *n_inv,
        const uint64_t *n_inv_shoup,
        uint32_t n,
        uint32_t tile_size,
        size_t poly_offset)
    {
        extern __shared__ uint64_t values[];
        const size_t limb = layout.limb(blockIdx.z);
        const auto descriptor = layout.output(blockIdx.z);
        const auto source = layout.input(blockIdx.z);
        const size_t poly = poly_offset + blockIdx.y;
        const uint32_t first = blockIdx.x * tile_size;
        const uint64_t modulus = moduli[limb];
        const size_t twiddle_base = limb * static_cast<size_t>(n);
        for (uint32_t index = threadIdx.x; index < tile_size; index += blockDim.x)
        {
            uint64_t value = matrix_load_limb_u64(
                source.base, poly, first + index, source.stride, source.width);
            if constexpr (Forward)
            {
                // A whole transform fits in this block only when there were no
                // global stages; otherwise the preceding twist already ran.
                if (tile_size == n)
                {
                    const size_t twiddle = twiddle_base + index;
                    value = mul_mod_shoup_u64(
                        value, twiddles[twiddle], twiddle_shoup[twiddle], modulus);
                }
            }
            values[index] = value;
        }
        __syncthreads();
        uint32_t length = Forward ? tile_size : 2;
        while (length >= 2 && length <= tile_size)
        {
            const uint32_t half = length / 2;
            for (uint32_t butterfly = threadIdx.x; butterfly < tile_size / 2;
                 butterfly += blockDim.x)
            {
                const uint32_t j = butterfly % half;
                const uint32_t index = (butterfly / half) * length + j;
                const size_t twiddle = twiddle_base + 2U * (n / length) * j;
                const uint64_t lower = values[index];
                const uint64_t upper = values[index + half];
                if constexpr (Forward)
                {
                    values[index] = add_mod_u64(lower, upper, modulus);
                    values[index + half] = mul_mod_shoup_u64(
                        sub_mod_u64(lower, upper, modulus),
                        twiddles[twiddle], twiddle_shoup[twiddle], modulus);
                }
                else
                {
                    const uint64_t product = mul_mod_shoup_u64(
                        upper, twiddles[twiddle], twiddle_shoup[twiddle], modulus);
                    values[index] = add_mod_u64(lower, product, modulus);
                    values[index + half] = sub_mod_u64(lower, product, modulus);
                }
            }
            __syncthreads();
            if constexpr (Forward)
            {
                length >>= 1;
            }
            else
            {
                length <<= 1;
            }
        }
        for (uint32_t index = threadIdx.x; index < tile_size; index += blockDim.x)
        {
            uint64_t value = values[index];
            if constexpr (!Forward)
            {
                if (tile_size == n)
                {
                    value = mul_mod_shoup_u64(value, n_inv[limb], n_inv_shoup[limb], modulus);
                    const size_t twiddle = twiddle_base + index;
                    value = mul_mod_shoup_u64(
                        value, twiddles[twiddle], twiddle_shoup[twiddle], modulus);
                }
            }
            matrix_store_limb_u64(
                descriptor.base, poly, first + index,
                descriptor.stride, descriptor.width, value);
        }
    }

    template <bool Forward, typename Layout>
    int launch_fused_local_stages(
        Layout layout,
        const GpuNttDeviceConstants &constants,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream)
    {
        const uint32_t tile_size = std::min(n, kFusedNttCoefficients);
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(n / tile_size, static_cast<uint32_t>(chunk), static_cast<uint32_t>(limb_count));
            ntt_fused_local_stages_kernel<Forward><<<
                grid, kTransformThreads, tile_size * sizeof(uint64_t), stream>>>(
                layout,
                Forward ? constants.twiddle_forward : constants.twiddle_inverse,
                Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse,
                constants.moduli, constants.n_inv, constants.n_inv_shoup,
                n, tile_size, offset);
            const cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) return set_error(error);
        }
        return 0;
    }

    // Each warp subgroup owns values separated by one shared-memory tile.
    // XOR shuffles therefore realize precisely the remaining high DIF/DIT
    // butterfly stages, including the transform boundary twist/normalization.
    template <bool Forward, uint32_t Width, typename Layout>
    __global__ void ntt_fused_top_stages_kernel(
        Layout layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        const uint64_t *n_inv,
        const uint64_t *n_inv_shoup,
        size_t poly_offset)
    {
        constexpr uint32_t n = kFusedNttCoefficients * Width;
        const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t lane = thread % Width;
        const uint32_t column = thread / Width;
        const uint32_t coefficient = column + lane * kFusedNttCoefficients;
        const size_t limb = layout.limb(blockIdx.z);
        const size_t poly = poly_offset + blockIdx.y;
        const auto descriptor = layout.output(blockIdx.z);
        const uint64_t modulus = moduli[limb];
        const size_t twiddle_base = limb * static_cast<size_t>(n);
        uint64_t value = matrix_load_limb_u64(
            descriptor.base, poly, coefficient, descriptor.stride, descriptor.width);
        if constexpr (Forward)
        {
            const size_t twist = twiddle_base + coefficient;
            value = mul_mod_shoup_u64(value, twiddles[twist], twiddle_shoup[twist], modulus);
        }
        uint32_t half_lanes = Forward ? Width / 2 : 1;
        while (half_lanes >= 1 && half_lanes < Width)
        {
            // All supported transforms are multiples of 256 coefficients;
            // complete warps participate and Width divides the warp size.
            const uint64_t partner = __shfl_xor_sync(
                0xffffffffU, static_cast<unsigned long long>(value), half_lanes, Width);
            const bool upper_lane = (lane & half_lanes) != 0;
            const uint64_t lower = upper_lane ? partner : value;
            const uint64_t upper = upper_lane ? value : partner;
            const uint32_t j = column + (lane % half_lanes) * kFusedNttCoefficients;
            const size_t twiddle = twiddle_base + (Width / half_lanes) * j;
            if constexpr (Forward)
            {
                value = upper_lane ? mul_mod_shoup_u64(
                    sub_mod_u64(lower, upper, modulus), twiddles[twiddle], twiddle_shoup[twiddle], modulus)
                    : add_mod_u64(lower, upper, modulus);
                half_lanes >>= 1;
            }
            else
            {
                const uint64_t product = mul_mod_shoup_u64(
                    upper, twiddles[twiddle], twiddle_shoup[twiddle], modulus);
                value = upper_lane ? sub_mod_u64(lower, product, modulus)
                    : add_mod_u64(lower, product, modulus);
                half_lanes <<= 1;
            }
        }
        if constexpr (!Forward)
        {
            value = mul_mod_shoup_u64(value, n_inv[limb], n_inv_shoup[limb], modulus);
            const size_t twist = twiddle_base + coefficient;
            value = mul_mod_shoup_u64(value, twiddles[twist], twiddle_shoup[twist], modulus);
        }
        matrix_store_limb_u64(
            descriptor.base, poly, coefficient, descriptor.stride, descriptor.width, value);
    }

    template <bool Forward, typename Layout>
    int launch_fused_top_stages(
        Layout layout,
        const GpuNttDeviceConstants &constants,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream)
    {
        const uint64_t *twiddles = Forward ? constants.twiddle_forward : constants.twiddle_inverse;
        const uint64_t *shoup = Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(n / kTransformThreads, static_cast<uint32_t>(chunk), static_cast<uint32_t>(limb_count));
            switch (n / kFusedNttCoefficients)
            {
#define MXX_LAUNCH_NTT_TOP(width) \
            case width: \
                ntt_fused_top_stages_kernel<Forward, width><<<grid, kTransformThreads, 0, stream>>>( \
                    layout, twiddles, shoup, constants.moduli, constants.n_inv, constants.n_inv_shoup, offset); \
                break
                MXX_LAUNCH_NTT_TOP(2);
                MXX_LAUNCH_NTT_TOP(4);
                MXX_LAUNCH_NTT_TOP(8);
                MXX_LAUNCH_NTT_TOP(16);
#undef MXX_LAUNCH_NTT_TOP
            }
            const cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) return set_error(error);
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
        if (!is_power_of_two_u32(n) || n < 2)
        {
            return set_error("invalid ring size in run_matrix_transform_u64");
        }

        auto &limb_map = mat->ctx->limb_gpu_ids;
        auto &limb_prime_ids = mat->ctx->limb_prime_ids;
        auto &moduli = mat->ctx->moduli;
        const size_t limb_count = static_cast<size_t>(mat->level + 1);
        if (limb_map.size() < limb_count)
        {
            return set_error("unexpected limb mapping size in run_matrix_transform_u64");
        }
        if (limb_prime_ids.size() < limb_count)
        {
            return set_error("unexpected limb metadata size in run_matrix_transform_u64");
        }
        if (limb_count > GPU_RUNTIME_MAX_LIMBS)
        {
            return set_error("too many limbs in run_matrix_transform_u64");
        }

        const size_t poly_count = matrix_poly_count(mat);
        if (poly_count == 0)
        {
            mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            return 0;
        }

        MatrixNttDescriptorView layout{};

        int dispatch_device = -1;
        size_t dispatch_slot = std::numeric_limits<size_t>::max();
        cudaStream_t dispatch_stream = nullptr;
        int status = 0;

        for (int limb = 0; limb <= mat->level; ++limb)
        {
            const size_t limb_idx = static_cast<size_t>(limb);
            const dim3 limb_id = limb_map[limb_idx];
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

            const int primeid = limb_prime_ids[limb_idx];
            if (primeid < 0 ||
                static_cast<size_t>(primeid) >= moduli.size() ||
                limb_id.x >= mat->ctx->gpu_ids.size())
            {
                return set_error("invalid prime/device index in run_matrix_transform_u64");
            }

            size_t stride_bytes = 0;
            uint8_t coeff_bytes = 0;
            if (!matrix_limb_metadata_by_id(mat, limb_id, &stride_bytes, &coeff_bytes))
            {
                return set_error("invalid limb metadata in run_matrix_transform_u64");
            }
            if (coeff_bytes == 0)
            {
                return set_error("invalid coeff byte-width in run_matrix_transform_u64");
            }
            if (stride_bytes < static_cast<size_t>(n) * static_cast<size_t>(coeff_bytes))
            {
                return set_error("invalid bytes_per_poly in run_matrix_transform_u64");
            }


            uint8_t *base = matrix_limb_ptr_by_id(mat, 0, limb_id);
            if (!base)
            {
                return set_error("null matrix limb pointer in run_matrix_transform_u64");
            }
            const auto &buffer = mat->shared_limb_buffers[limb_id.x];
            if (!buffer.device_descriptors || limb_id.y >= buffer.limb_count)
                return set_error("missing matrix-owned NTT descriptor");
            if (limb_idx == 0) layout.descriptors = buffer.device_descriptors;
            else if (layout.descriptors != buffer.device_descriptors)
                return set_error("NTT descriptors span multiple partitions");
            layout.indices[limb_idx] = limb_id.y;
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
        if (device_constants.ring_dimension != n)
        {
            return set_error("NTT twiddle constants mismatch in run_matrix_transform_u64");
        }
        if (!device_constants.twiddle_forward ||
            !device_constants.twiddle_inverse ||
            !device_constants.twiddle_shoup_forward ||
            !device_constants.twiddle_shoup_inverse ||
            !device_constants.moduli || !device_constants.n_inv ||
            !device_constants.n_inv_shoup)
        {
            return set_error("null per-device NTT constants in run_matrix_transform_u64");
        }

        status = matrix_wait_all_limb_streams(mat, dispatch_device, dispatch_stream);
        if (status != 0) return status;

        const uint64_t *twiddles =
            Forward ? device_constants.twiddle_forward : device_constants.twiddle_inverse;
        const uint64_t *twiddle_shoup = Forward ?
            device_constants.twiddle_shoup_forward : device_constants.twiddle_shoup_inverse;

        if constexpr (Forward)
        {
            if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            {
                status = launch_fused_top_stages<Forward>(
                    layout, device_constants, limb_count, n, poly_count, dispatch_stream);
                if (status != 0) return status;
            }
            else if (n > kFusedNttCoefficients)
            {
                status = launch_twist_for_all_limbs(
                    layout, twiddles, twiddle_shoup, device_constants.moduli,
                    limb_count, n, poly_count, dispatch_stream);
                if (status != 0) return status;
                // Global DIF stages precede independent shared-memory tiles.
                for (uint32_t len = n; len > kFusedNttCoefficients; len >>= 1)
                {
                    status = launch_stage_for_all_limbs<true>(
                        layout, twiddles, twiddle_shoup, device_constants.moduli,
                        limb_count, n, len, poly_count, dispatch_stream);
                    if (status != 0) return status;
                }
            }
            status = launch_fused_local_stages<true>(
                layout, device_constants, limb_count, n, poly_count, dispatch_stream);
            if (status != 0) return status;
        }
        else
        {
            status = launch_fused_local_stages<false>(
                layout, device_constants, limb_count, n, poly_count, dispatch_stream);
            if (status != 0) return status;
            if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            {
                status = launch_fused_top_stages<Forward>(
                    layout, device_constants, limb_count, n, poly_count, dispatch_stream);
                if (status != 0) return status;
            }
            else if (n > kFusedNttCoefficients)
            {
                // Inverse DIT merges the independently transformed tiles.
                for (uint32_t len = kFusedNttCoefficients * 2; len <= n; len <<= 1)
                {
                    status = launch_stage_for_all_limbs<false>(
                        layout, twiddles, twiddle_shoup, device_constants.moduli,
                        limb_count, n, len, poly_count, dispatch_stream);
                    if (status != 0) return status;
                }
                status = launch_scale_for_all_limbs(
                    layout, device_constants.moduli, device_constants.n_inv,
                    device_constants.n_inv_shoup, limb_count, n, poly_count, dispatch_stream);
                if (status != 0) return status;
                status = launch_twist_for_all_limbs(
                    layout, twiddles, twiddle_shoup, device_constants.moduli,
                    limb_count, n, poly_count, dispatch_stream);
                if (status != 0) return status;
            }
        }

        status = matrix_record_all_limb_writes(mat, dispatch_stream);
        if (status != 0) return status;

        mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        return 0;
    }
}

int gpu_matrix_ntt_all(GpuMatrix *mat)
{
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

int gpu_matrix_intt_all(GpuMatrix *mat)
{
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
