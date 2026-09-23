#include <type_traits>

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

    template <typename Kernel, typename Layout, typename... Args>
    int launch_ntt_node(GpuContext *context, cudaStream_t stream,
        uint32_t binding_index,
        Kernel kernel, dim3 grid, dim3 block, size_t shared_bytes,
        Layout layout, Args... values)
    {
        MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            0, 0, sizeof(void *), binding_index, 0};
        if (auto *builder = mxx_gpu_graph_builder_for_stream(
            context, reinterpret_cast<void *>(stream)))
        {
            if constexpr (std::is_same_v<Layout, MatrixNttDescriptorView>)
            {
                if (patch.binding_index == UINT32_MAX &&
                    mxx_gpu_graph_builder_find_binding(builder,
                        reinterpret_cast<uint64_t>(layout.descriptors),
                        &patch.binding_index) != 0) return 1;
                return mxx_gpu_launch_kernel(context, stream, kernel, grid, block,
                    shared_bytes, &patch, 1, layout, values...);
            }
            else
            {
                std::vector<MxxGraphPatch> patches;
                const auto add_descriptors = [&](const auto &addresses) -> int {
                    for (const auto &address : addresses)
                    {
                        if (!address) continue;
                        uint32_t owner_binding = UINT32_MAX;
                        if (mxx_gpu_graph_builder_find_binding(builder,
                            reinterpret_cast<uint64_t>(address), &owner_binding) != 0) return 1;
                        const auto offset = reinterpret_cast<const uint8_t *>(&address) -
                            reinterpret_cast<const uint8_t *>(&layout);
                        if (offset > UINT32_MAX) return set_error("NTT descriptor patch offset overflow");
                        patches.push_back(MxxGraphPatch{nullptr,
                            MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                            static_cast<uint32_t>(offset), sizeof(void *), owner_binding, 0});
                    }
                    return 0;
                };
                if (add_descriptors(layout.outputs) != 0) return 1;
                if (layout.out_of_place && add_descriptors(layout.inputs) != 0) return 1;
                return mxx_gpu_launch_kernel(context, stream, kernel, grid, block,
                    shared_bytes, patches.data(), patches.size(), layout, values...);
            }
        }
        return mxx_gpu_launch_kernel(context, stream, kernel, grid, block,
            shared_bytes, nullptr, 0, layout, values...);
    }

    int launch_twist_for_all_limbs(
        MatrixNttDescriptorView layout,
        const uint64_t *twiddles,
        const uint64_t *twiddle_shoup,
        const uint64_t *moduli,
        size_t limb_count,
        uint32_t n,
        size_t poly_count,
        cudaStream_t stream,
        GpuContext *context,
        uint32_t binding_index)
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
            const int status = launch_ntt_node(context, stream, binding_index,
                ntt_twist_all_limbs_kernel, grid, kTransformThreads, 0,
                layout, twiddles, twiddle_shoup, moduli, limb_count, n, offset);
            if (status != 0) return status;
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
        cudaStream_t stream,
        GpuContext *context,
        uint32_t binding_index)
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
            const int status = launch_ntt_node(context, stream, binding_index,
                ntt_scale_all_limbs_kernel, grid, kTransformThreads, 0,
                layout, moduli, n_inv, n_inv_shoup, limb_count, n, offset);
            if (status != 0) return status;
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
        cudaStream_t stream,
        GpuContext *context,
        uint32_t binding_index)
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
            const int status = launch_ntt_node(context, stream, binding_index,
                ntt_stage_all_limbs_kernel<ForwardDif>, grid, kTransformThreads, 0,
                layout, twiddles, twiddle_shoup, moduli, limb_count, n, len, offset);
            if (status != 0) return status;
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
        cudaStream_t stream,
        GpuContext *context,
        uint32_t binding_index)
    {
        const uint32_t tile_size = std::min(n, kFusedNttCoefficients);
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(n / tile_size, static_cast<uint32_t>(chunk), static_cast<uint32_t>(limb_count));
            const int status = launch_ntt_node(context, stream, binding_index,
                ntt_fused_local_stages_kernel<Forward, Layout>, grid, kTransformThreads,
                tile_size * sizeof(uint64_t), layout,
                Forward ? constants.twiddle_forward : constants.twiddle_inverse,
                Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse,
                constants.moduli, constants.n_inv, constants.n_inv_shoup,
                n, tile_size, offset);
            if (status != 0) return status;
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
        cudaStream_t stream,
        GpuContext *context,
        uint32_t binding_index)
    {
        const uint64_t *twiddles = Forward ? constants.twiddle_forward : constants.twiddle_inverse;
        const uint64_t *shoup = Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse;
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(n / kTransformThreads, static_cast<uint32_t>(chunk), static_cast<uint32_t>(limb_count));
            int status = 0;
            switch (n / kFusedNttCoefficients)
            {
#define MXX_LAUNCH_NTT_TOP(width) \
            case width: \
                status = launch_ntt_node(context, stream, binding_index, \
                    ntt_fused_top_stages_kernel<Forward, width, Layout>, grid, kTransformThreads, 0, \
                    layout, twiddles, shoup, constants.moduli, constants.n_inv, constants.n_inv_shoup, offset); \
                break
                MXX_LAUNCH_NTT_TOP(2);
                MXX_LAUNCH_NTT_TOP(4);
                MXX_LAUNCH_NTT_TOP(8);
                MXX_LAUNCH_NTT_TOP(16);
#undef MXX_LAUNCH_NTT_TOP
            }
            if (status != 0) return status;
        }
        return 0;
    }

    template <bool Forward>
    int run_matrix_transform_u64(
        GpuMatrix *mat,
        cudaStream_t override_stream = nullptr,
        uint32_t binding_index = UINT32_MAX)
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
                if (override_stream)
                    dispatch_stream = override_stream;
                else
                {
                    status = matrix_limb_stream(mat, limb_id, &dispatch_stream);
                    if (status != 0) return status;
                }
                if (!dispatch_stream)
                {
                    return set_error("null dispatch stream in run_matrix_transform_u64");
                }
                dispatch_stream = reinterpret_cast<cudaStream_t>(
                    mxx_gpu_graph_builder_dispatch_stream(mat->ctx, dispatch_device,
                        reinterpret_cast<void *>(dispatch_stream)));
                if (!dispatch_stream) return 1;
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

        if (!override_stream && !mxx_gpu_graph_builder_for_stream(mat->ctx,
            reinterpret_cast<void *>(dispatch_stream)))
        {
            status = matrix_wait_all_limb_streams(mat, dispatch_device, dispatch_stream);
            if (status != 0) return status;
        }

        const uint64_t *twiddles =
            Forward ? device_constants.twiddle_forward : device_constants.twiddle_inverse;
        const uint64_t *twiddle_shoup = Forward ?
            device_constants.twiddle_shoup_forward : device_constants.twiddle_shoup_inverse;

        if constexpr (Forward)
        {
            if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            {
                status = launch_fused_top_stages<Forward>(
                    layout, device_constants, limb_count, n, poly_count, dispatch_stream,
                    mat->ctx, binding_index);
                if (status != 0) return status;
            }
            else if (n > kFusedNttCoefficients)
            {
                status = launch_twist_for_all_limbs(
                    layout, twiddles, twiddle_shoup, device_constants.moduli,
                    limb_count, n, poly_count, dispatch_stream, mat->ctx, binding_index);
                if (status != 0) return status;
                // Global DIF stages precede independent shared-memory tiles.
                for (uint32_t len = n; len > kFusedNttCoefficients; len >>= 1)
                {
                    status = launch_stage_for_all_limbs<true>(
                        layout, twiddles, twiddle_shoup, device_constants.moduli,
                        limb_count, n, len, poly_count, dispatch_stream,
                        mat->ctx, binding_index);
                    if (status != 0) return status;
                }
            }
            status = launch_fused_local_stages<true>(
                layout, device_constants, limb_count, n, poly_count, dispatch_stream,
                mat->ctx, binding_index);
            if (status != 0) return status;
        }
        else
        {
            status = launch_fused_local_stages<false>(
                layout, device_constants, limb_count, n, poly_count, dispatch_stream,
                mat->ctx, binding_index);
            if (status != 0) return status;
            if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            {
                status = launch_fused_top_stages<Forward>(
                    layout, device_constants, limb_count, n, poly_count, dispatch_stream,
                    mat->ctx, binding_index);
                if (status != 0) return status;
            }
            else if (n > kFusedNttCoefficients)
            {
                // Inverse DIT merges the independently transformed tiles.
                for (uint32_t len = kFusedNttCoefficients * 2; len <= n; len <<= 1)
                {
                    status = launch_stage_for_all_limbs<false>(
                        layout, twiddles, twiddle_shoup, device_constants.moduli,
                        limb_count, n, len, poly_count, dispatch_stream,
                        mat->ctx, binding_index);
                    if (status != 0) return status;
                }
                status = launch_scale_for_all_limbs(
                    layout, device_constants.moduli, device_constants.n_inv,
                    device_constants.n_inv_shoup, limb_count, n, poly_count, dispatch_stream,
                    mat->ctx, binding_index);
                if (status != 0) return status;
                status = launch_twist_for_all_limbs(
                    layout, twiddles, twiddle_shoup, device_constants.moduli,
                    limb_count, n, poly_count, dispatch_stream, mat->ctx, binding_index);
                if (status != 0) return status;
            }
        }

        if (!override_stream && !mxx_gpu_graph_builder_for_stream(mat->ctx,
            reinterpret_cast<void *>(dispatch_stream)))
        {
            status = matrix_record_all_limb_writes(mat, dispatch_stream);
            if (status != 0) return status;
        }

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

int gpu_matrix_ntt_all_on_stream(GpuMatrix *mat, cudaStream_t stream)
{
    if (!mat || !mat->ctx || !stream)
        return set_error("invalid gpu_matrix_ntt_all_on_stream arguments");
    if (mat->format == GPU_POLY_FORMAT_EVAL)
        return 0;
    return run_matrix_transform_u64<true>(mat, stream);
}

int gpu_matrix_intt_all_on_stream(GpuMatrix *mat, cudaStream_t stream)
{
    if (!mat || !mat->ctx || !stream)
        return set_error("invalid gpu_matrix_intt_all_on_stream arguments");
    if (mat->format == GPU_POLY_FORMAT_COEFF)
        return 0;
    return run_matrix_transform_u64<false>(mat, stream);
}

int gpu_matrix_ntt_all_on_stream_bound(
    GpuMatrix *mat,
    cudaStream_t stream,
    uint32_t binding_index)
{
    if (!mat || !mat->ctx || !stream)
        return set_error("invalid gpu_matrix_ntt_all_on_stream_bound arguments");
    if (mat->format == GPU_POLY_FORMAT_EVAL)
        return 0;
    return run_matrix_transform_u64<true>(mat, stream, binding_index);
}

int gpu_matrix_intt_all_on_stream_bound(
    GpuMatrix *mat,
    cudaStream_t stream,
    uint32_t binding_index)
{
    if (!mat || !mat->ctx || !stream)
        return set_error("invalid gpu_matrix_intt_all_on_stream_bound arguments");
    if (mat->format == GPU_POLY_FORMAT_COEFF)
        return 0;
    return run_matrix_transform_u64<false>(mat, stream, binding_index);
}

int gpu_matrix_ntt_all_bound(GpuMatrix *mat, uint32_t binding_index)
{
    if (!mat || !mat->ctx)
    {
        return set_error("invalid gpu_matrix_ntt_all_bound arguments");
    }
    if (mat->format == GPU_POLY_FORMAT_EVAL)
        return 0;
    return run_matrix_transform_u64<true>(mat, nullptr, binding_index);
}

int gpu_matrix_intt_all_bound(GpuMatrix *mat, uint32_t binding_index)
{
    if (!mat || !mat->ctx)
    {
        return set_error("invalid gpu_matrix_intt_all_bound arguments");
    }
    if (mat->format == GPU_POLY_FORMAT_COEFF)
        return 0;
    return run_matrix_transform_u64<false>(mat, nullptr, binding_index);
}

namespace
{
    __device__ __forceinline__ uint8_t *raw_matrix_cell(
        const MxxRawMatrixLimb &limb, size_t poly, size_t coefficient,
        size_t columns)
    {
        const size_t row = poly / columns;
        const size_t column = poly - row * columns;
        return reinterpret_cast<uint8_t *>(limb.address +
            row * limb.row_stride_bytes + column * limb.column_stride_bytes +
            coefficient * limb.coefficient_stride_bytes);
    }

    __device__ __forceinline__ uint64_t raw_matrix_load(
        const MxxRawMatrixLimb &limb, size_t poly, size_t coefficient,
        size_t columns)
    {
        const uint8_t *address = raw_matrix_cell(limb, poly, coefficient, columns);
        return limb.word_bytes == 4 ? *reinterpret_cast<const uint32_t *>(address) :
            *reinterpret_cast<const uint64_t *>(address);
    }

    __device__ __forceinline__ void raw_matrix_store(
        const MxxRawMatrixLimb &limb, size_t poly, size_t coefficient,
        size_t columns, uint64_t value)
    {
        uint8_t *address = raw_matrix_cell(limb, poly, coefficient, columns);
        if (limb.word_bytes == 4) *reinterpret_cast<uint32_t *>(address) = static_cast<uint32_t>(value);
        else *reinterpret_cast<uint64_t *>(address) = value;
    }

    __global__ void raw_matrix_copy_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        size_t columns, size_t degree, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        raw_matrix_store(destination, poly, coefficient, columns,
            raw_matrix_load(source, poly, coefficient, columns));
    }

    __device__ uint64_t raw_pow_mod(uint64_t base, uint32_t exponent,
        uint64_t modulus)
    {
        uint64_t value = 1;
        base %= modulus;
        while (exponent != 0)
        {
            if ((exponent & 1U) != 0)
                value = mul_mod_u64(value, base, modulus);
            exponent >>= 1;
            if (exponent != 0) base = mul_mod_u64(base, base, modulus);
        }
        return value;
    }

    __global__ void raw_matrix_structured_fill_kernel(
        MxxRawMatrixLimb destination, uint64_t rows, uint64_t columns,
        uint64_t row_origin, uint64_t column_origin,
        uint64_t column_base, uint64_t slots_per_row,
        uint32_t digits_per_tower, uint64_t base_residue,
        size_t degree, size_t poly_offset, bool gadget)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t row = row_origin + poly / columns;
        const uint64_t column = column_origin + poly % columns - column_base;
        const uint64_t row_start = row * slots_per_row;
        const uint64_t slot = column >= row_start ? column - row_start : slots_per_row;
        uint64_t value = 0;
        if (column < rows * slots_per_row && slot < slots_per_row)
        {
            if (!gadget)
                value = 1;
            else if (slot / digits_per_tower == destination.crt_limb_index)
                value = raw_pow_mod(base_residue,
                    static_cast<uint32_t>(slot % digits_per_tower),
                    destination.modulus);
        }
        raw_matrix_store(destination, poly, coefficient, columns, value);
    }

    __global__ void raw_ntt_twist_kernel(
        MxxRawMatrixLimb limb, const uint64_t *twiddles,
        const uint64_t *shoup, uint64_t modulus,
        size_t columns, size_t degree, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        raw_matrix_store(limb, poly, coefficient, columns,
            mul_mod_shoup_u64(raw_matrix_load(limb, poly, coefficient, columns),
                twiddles[coefficient], shoup[coefficient], modulus));
    }

    template <bool Forward>
    __global__ void raw_ntt_stage_kernel(
        MxxRawMatrixLimb limb, const uint64_t *twiddles,
        const uint64_t *shoup, uint64_t modulus,
        size_t columns, uint32_t degree, uint32_t len, size_t poly_offset)
    {
        const uint32_t butterfly = blockIdx.x * blockDim.x + threadIdx.x;
        if (butterfly >= degree / 2) return;
        const uint32_t half = len / 2;
        const uint32_t group = butterfly / half;
        const uint32_t j = butterfly - group * half;
        const uint32_t i = group * len + j;
        const uint32_t twiddle = 2U * (degree / len) * j;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t lower = raw_matrix_load(limb, poly, i, columns);
        const uint64_t upper = raw_matrix_load(limb, poly, i + half, columns);
        uint64_t left, right;
        if constexpr (Forward)
        {
            left = add_mod_u64(lower, upper, modulus);
            right = mul_mod_shoup_u64(sub_mod_u64(lower, upper, modulus),
                twiddles[twiddle], shoup[twiddle], modulus);
        }
        else
        {
            const uint64_t weighted = mul_mod_shoup_u64(
                upper, twiddles[twiddle], shoup[twiddle], modulus);
            left = add_mod_u64(lower, weighted, modulus);
            right = sub_mod_u64(lower, weighted, modulus);
        }
        raw_matrix_store(limb, poly, i, columns, left);
        raw_matrix_store(limb, poly, i + half, columns, right);
    }

    __global__ void raw_ntt_inverse_finish_kernel(
        MxxRawMatrixLimb limb, const uint64_t *twiddles,
        const uint64_t *shoup, const uint64_t *n_inverse,
        const uint64_t *n_inverse_shoup, size_t local_limb,
        uint64_t modulus,
        size_t columns, size_t degree, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t scaled = mul_mod_shoup_u64(
            raw_matrix_load(limb, poly, coefficient, columns),
            n_inverse[local_limb], n_inverse_shoup[local_limb], modulus);
        raw_matrix_store(limb, poly, coefficient, columns,
            mul_mod_shoup_u64(scaled, twiddles[coefficient],
                shoup[coefficient], modulus));
    }

    __global__ void raw_matrix_add_sub_kernel(
        MxxRawMatrixLimb left, MxxRawMatrixLimb right,
        MxxRawMatrixLimb destination, size_t columns,
        size_t degree, size_t poly_offset, bool subtract)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t lhs = raw_matrix_load(left, poly, coefficient, columns);
        const uint64_t rhs = raw_matrix_load(right, poly, coefficient, columns);
        raw_matrix_store(destination, poly, coefficient, columns,
            subtract ? sub_mod_u64(lhs, rhs, destination.modulus) :
                add_mod_u64(lhs, rhs, destination.modulus));
    }

    __global__ void raw_matrix_scale_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        uint64_t scalar_residue, size_t columns, size_t degree,
        size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t value = raw_matrix_load(source, poly, coefficient, columns);
        raw_matrix_store(destination, poly, coefficient, columns,
            mul_mod_u64(value, scalar_residue, destination.modulus));
    }

    __global__ void raw_matrix_scale_dynamic_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        const uint64_t *scalar, int scalar_encoding, uint32_t *status,
        size_t columns, size_t degree, size_t poly_offset)
    {
        __shared__ uint64_t scalar_residue;
        __shared__ bool scalar_valid;
        if (threadIdx.x == 0)
        {
            scalar_valid = scalar_encoding == 0 || scalar_encoding == 1 ||
                (scalar_encoding > 2 && scalar[0] <= 1);
            scalar_residue = 0;
            if (scalar_valid)
            {
                const uint64_t modulus = destination.modulus;
                bool negative = false;
                if (scalar_encoding == 0)
                {
                    const int64_t signed_value = static_cast<int64_t>(scalar[0]);
                    negative = signed_value < 0;
                    const uint64_t magnitude = negative ? 0ULL - scalar[0] : scalar[0];
                    scalar_residue = magnitude % modulus;
                }
                else if (scalar_encoding == 1)
                    scalar_residue = scalar[0] % modulus;
                else
                {
                    negative = scalar[0] != 0;
                    for (int word = scalar_encoding - 2; word > 0; --word)
                        scalar_residue = static_cast<uint64_t>(
                            ((static_cast<unsigned __int128>(scalar_residue) << 64) |
                                scalar[word]) % modulus);
                }
                if (negative && scalar_residue != 0)
                    scalar_residue = modulus - scalar_residue;
            }
            else
                atomicCAS(status, 0U, 2U);
        }
        __syncthreads();
        if (!scalar_valid) return;
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t value = raw_matrix_load(source, poly, coefficient, columns);
        raw_matrix_store(destination, poly, coefficient, columns,
            mul_mod_u64(value, scalar_residue, destination.modulus));
    }

    __global__ void raw_matrix_mul_kernel(
        MxxRawMatrixLimb left, MxxRawMatrixLimb right,
        MxxRawMatrixLimb destination, size_t left_columns,
        size_t right_columns, size_t output_columns, size_t output_rows,
        size_t degree, size_t poly_offset, bool accumulate,
        bool transpose_rhs)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t output_poly = poly_offset + blockIdx.y;
        const size_t row = output_poly / output_columns;
        const size_t column = output_poly - row * output_columns;
        if (row >= output_rows) return;
        const uint64_t modulus = destination.modulus;
        uint64_t sum = accumulate ? raw_matrix_load(
            destination, output_poly, coefficient, output_columns) : 0;
        for (size_t inner = 0; inner < left_columns; ++inner)
        {
            const uint64_t lhs = raw_matrix_load(
                left, row * left_columns + inner, coefficient, left_columns);
            const size_t rhs_poly = transpose_rhs ?
                column * left_columns + inner : inner * right_columns + column;
            const uint64_t rhs = raw_matrix_load(right, rhs_poly, coefficient,
                transpose_rhs ? left_columns : right_columns);
            sum = add_mod_u64(sum, mul_mod_u64(lhs, rhs, modulus), modulus);
        }
        raw_matrix_store(destination, output_poly, coefficient, output_columns, sum);
    }

    __global__ void raw_matrix_transpose_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        size_t source_columns, size_t destination_columns,
        size_t degree, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t output_poly = poly_offset + blockIdx.y;
        const size_t row = output_poly / destination_columns;
        const size_t column = output_poly - row * destination_columns;
        const size_t input_poly = column * source_columns + row;
        raw_matrix_store(destination, output_poly, coefficient, destination_columns,
            raw_matrix_load(source, input_poly, coefficient, source_columns));
    }

    __global__ void raw_matrix_tensor_kernel(
        MxxRawMatrixLimb left, MxxRawMatrixLimb right,
        MxxRawMatrixLimb destination, size_t left_columns,
        size_t right_rows, size_t right_columns, size_t output_columns,
        size_t degree, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t output_poly = poly_offset + blockIdx.y;
        const size_t output_row = output_poly / output_columns;
        const size_t output_column = output_poly - output_row * output_columns;
        const size_t left_poly = (output_row / right_rows) * left_columns +
            output_column / right_columns;
        const size_t right_poly = (output_row % right_rows) * right_columns +
            output_column % right_columns;
        const uint64_t lhs = raw_matrix_load(left, left_poly, coefficient, left_columns);
        const uint64_t rhs = raw_matrix_load(right, right_poly, coefficient, right_columns);
        raw_matrix_store(destination, output_poly, coefficient, output_columns,
            mul_mod_u64(lhs, rhs, destination.modulus));
    }

    int validate_raw_view(GpuContext *ctx, const MxxRawMatrixView *view, void *stream)
    {
        if (!ctx || !view || !stream || !view->limbs || !view->limb_count ||
            view->physical_device < 0 || view->degree != static_cast<uint32_t>(ctx->N) ||
            !is_power_of_two_u32(view->degree) || !view->rows || !view->columns ||
            view->rows > SIZE_MAX / view->columns ||
            view->limb_count > GPU_RUNTIME_MAX_LIMBS)
            return set_error("invalid raw matrix physical view");
        if (!mxx_gpu_graph_builder_for_stream(ctx, stream))
            return set_error("raw matrix emitter requires active explicit graph operation");
        for (size_t limb_index = 0; limb_index < view->limb_count; ++limb_index)
        {
            const auto &limb = view->limbs[limb_index];
            const unsigned __int128 column_span =
                static_cast<unsigned __int128>(view->degree - 1) *
                limb.coefficient_stride_bytes + limb.word_bytes;
            const unsigned __int128 row_span =
                static_cast<unsigned __int128>(view->columns - 1) *
                limb.column_stride_bytes + column_span;
            if (!limb.address || (limb.word_bytes != 4 && limb.word_bytes != 8) ||
                limb.crt_limb_index >= ctx->limb_gpu_ids.size() ||
                limb.crt_limb_index >= ctx->limb_prime_ids.size() ||
                ctx->limb_prime_ids[limb.crt_limb_index] < 0 ||
                static_cast<size_t>(ctx->limb_prime_ids[limb.crt_limb_index]) >= ctx->moduli.size() ||
                ctx->limb_gpu_ids[limb.crt_limb_index].x >= ctx->gpu_ids.size() ||
                ctx->gpu_ids[ctx->limb_gpu_ids[limb.crt_limb_index].x] != view->physical_device ||
                ctx->moduli[ctx->limb_prime_ids[limb.crt_limb_index]] != limb.modulus ||
                (limb.word_bytes == 4 && limb.modulus > UINT32_MAX) ||
                limb.coefficient_stride_bytes < limb.word_bytes ||
                column_span > UINT64_MAX || row_span > UINT64_MAX ||
                limb.column_stride_bytes < column_span ||
                limb.row_stride_bytes < row_span)
                return set_error("invalid raw matrix limb layout");
        }
        return 0;
    }

    bool same_raw_extent(const MxxRawMatrixView *left, const MxxRawMatrixView *right)
    {
        if (left->physical_device != right->physical_device ||
            left->degree != right->degree || left->row_origin != right->row_origin ||
            left->column_origin != right->column_origin ||
            left->rows != right->rows || left->columns != right->columns ||
            left->limb_count != right->limb_count) return false;
        for (size_t limb = 0; limb < left->limb_count; ++limb)
            if (left->limbs[limb].crt_limb_index != right->limbs[limb].crt_limb_index ||
                left->limbs[limb].modulus != right->limbs[limb].modulus) return false;
        return true;
    }
}

extern "C" int gpu_raw_matrix_ntt(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    int inverse, uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !same_raw_extent(source, destination) || (inverse != 0 && inverse != 1) ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("raw NTT views do not match");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t degree = source->degree;
    const size_t poly_count = source->rows * source->columns;
    for (size_t limb_index = 0; limb_index < source->limb_count; ++limb_index)
    {
        const auto source_limb = source->limbs[limb_index];
        const auto output_limb = destination->limbs[limb_index];
        const dim3 partition = ctx->limb_gpu_ids[source_limb.crt_limb_index];
        if (partition.x >= ctx->ntt_device_constants.size())
            return set_error("missing raw NTT device constants");
        const auto &constants = ctx->ntt_device_constants[partition.x];
        if (constants.device != source->physical_device ||
            partition.y >= constants.limb_count ||
            constants.ring_dimension != degree ||
            !constants.twiddle_forward || !constants.twiddle_inverse ||
            !constants.twiddle_shoup_forward || !constants.twiddle_shoup_inverse ||
            !constants.n_inv || !constants.n_inv_shoup)
            return set_error("invalid raw NTT device constants");
        const size_t table_offset = static_cast<size_t>(partition.y) * degree;
        const uint64_t *twiddles = (inverse ? constants.twiddle_inverse :
            constants.twiddle_forward) + table_offset;
        const uint64_t *shoup = (inverse ? constants.twiddle_shoup_inverse :
            constants.twiddle_shoup_forward) + table_offset;
        const MxxGraphPatch source_patch{nullptr,
            MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
            static_cast<uint32_t>(source_binding_base + limb_index), 0};
        const MxxGraphPatch destination_patch{nullptr,
            MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
            static_cast<uint32_t>(destination_binding_base + limb_index), 0};
        const MxxGraphPatch copy_patches[] = {source_patch,
            MxxGraphPatch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0,
                sizeof(void *), destination_patch.binding_index, 0}};
        const uint64_t modulus = output_limb.modulus;
        for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kMaxGridY)
        {
            const size_t poly_chunk = std::min(kMaxGridY, poly_count - poly_offset);
            const dim3 coeff_grid(
                static_cast<uint32_t>((degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(poly_chunk));
            int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_copy_kernel,
                coeff_grid, dim3(kTransformThreads), 0, copy_patches, 2,
                source_limb, output_limb, source->columns, degree, poly_offset);
            if (status != 0) return status;
            if (!inverse)
            {
                status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_twist_kernel,
                    coeff_grid, dim3(kTransformThreads), 0, &destination_patch, 1,
                    output_limb, twiddles, shoup, modulus,
                    source->columns, degree, poly_offset);
                if (status != 0) return status;
            }
            const dim3 butterfly_grid(
                static_cast<uint32_t>((degree / 2 + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(poly_chunk));
            if (!inverse)
            {
                for (uint32_t len = source->degree; len >= 2; len >>= 1)
                {
                    status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_stage_kernel<true>,
                        butterfly_grid, dim3(kTransformThreads), 0,
                        &destination_patch, 1, output_limb, twiddles, shoup,
                        modulus, source->columns, source->degree, len, poly_offset);
                    if (status != 0) return status;
                }
            }
            else
            {
                for (uint32_t len = 2; len <= source->degree; len <<= 1)
                {
                    status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_stage_kernel<false>,
                        butterfly_grid, dim3(kTransformThreads), 0,
                        &destination_patch, 1, output_limb, twiddles, shoup,
                        modulus, source->columns, source->degree, len, poly_offset);
                    if (status != 0) return status;
                }
                status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_inverse_finish_kernel,
                    coeff_grid, dim3(kTransformThreads), 0, &destination_patch, 1,
                    output_limb, twiddles, shoup,
                    constants.n_inv, constants.n_inv_shoup,
                    static_cast<size_t>(partition.y),
                    modulus, source->columns, degree, poly_offset);
                if (status != 0) return status;
            }
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_add_sub(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination, int subtract,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, left, stream_raw) != 0 ||
        validate_raw_view(ctx, right, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !same_raw_extent(left, right) || !same_raw_extent(left, destination) ||
        (subtract != 0 && subtract != 1) ||
        left_binding_base > UINT32_MAX - left->limb_count ||
        right_binding_base > UINT32_MAX - right->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("raw arithmetic views do not match");
    if (cudaSetDevice(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = left->rows * left->columns;
    for (size_t limb_index = 0; limb_index < left->limb_count; ++limb_index)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(left_binding_base + limb_index), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(right_binding_base + limb_index), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb_index), 0},
        };
        for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kMaxGridY)
        {
            const size_t poly_chunk = std::min(kMaxGridY, poly_count - poly_offset);
            const dim3 grid(
                static_cast<uint32_t>((left->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(poly_chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_add_sub_kernel,
                grid, dim3(kTransformThreads), 0, patches, 3,
                left->limbs[limb_index], right->limbs[limb_index],
                destination->limbs[limb_index], left->columns,
                static_cast<size_t>(left->degree), poly_offset, subtract != 0);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_scale(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const uint64_t *scalar_residues, size_t residue_count,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !same_raw_extent(source, destination) || !scalar_residues ||
        residue_count != source->limb_count ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix scale views or residues");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = source->rows * source->columns;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        if (scalar_residues[limb] >= source->limbs[limb].modulus)
            return set_error("raw matrix scalar residue exceeds CRT modulus");
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(source_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((source->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_scale_kernel,
                grid, dim3(kTransformThreads), 0, patches, 2,
                source->limbs[limb], destination->limbs[limb],
                scalar_residues[limb], source->columns,
                static_cast<size_t>(source->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_scale_dynamic(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *scalar, int scalar_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t scalar_binding, uint32_t status_binding)
{
    if (!ctx || !scalar || !status || scalar_encoding < 0 || scalar_encoding == 2 ||
        validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !same_raw_extent(source, destination) ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid dynamic raw matrix scale views or scalar");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = source->rows * source->columns;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                offsetof(MxxRawMatrixLimb, address), sizeof(void *),
                static_cast<uint32_t>(source_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1,
                offsetof(MxxRawMatrixLimb, address), sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0, sizeof(void *),
                scalar_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 4, 0, sizeof(void *),
                status_binding, 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((source->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int result = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_scale_dynamic_kernel, grid, dim3(kTransformThreads), 0,
                patches, std::size(patches), source->limbs[limb], destination->limbs[limb],
                static_cast<const uint64_t *>(scalar), scalar_encoding, status,
                source->columns, static_cast<size_t>(source->degree), offset);
            if (result != 0) return result;
        }
    }
    return 0;
}

static int raw_matrix_mul_impl(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination, int accumulate, bool transpose_rhs,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, left, stream_raw) != 0 ||
        validate_raw_view(ctx, right, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        left->physical_device != right->physical_device ||
        left->physical_device != destination->physical_device ||
        left->degree != right->degree || left->degree != destination->degree ||
        left->limb_count != right->limb_count ||
        left->limb_count != destination->limb_count ||
        left->rows != destination->rows ||
        left->columns != (transpose_rhs ? right->columns : right->rows) ||
        (transpose_rhs ? right->rows : right->columns) != destination->columns ||
        left->row_origin != destination->row_origin ||
        (transpose_rhs ? right->row_origin : right->column_origin) !=
            destination->column_origin ||
        left->column_origin !=
            (transpose_rhs ? right->column_origin : right->row_origin) ||
        (accumulate != 0 && accumulate != 1) ||
        left_binding_base > UINT32_MAX - left->limb_count ||
        right_binding_base > UINT32_MAX - right->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix product views");
    for (size_t limb = 0; limb < left->limb_count; ++limb)
        if (left->limbs[limb].crt_limb_index != right->limbs[limb].crt_limb_index ||
            left->limbs[limb].crt_limb_index != destination->limbs[limb].crt_limb_index ||
            left->limbs[limb].modulus != right->limbs[limb].modulus ||
            left->limbs[limb].modulus != destination->limbs[limb].modulus ||
            left->limbs[limb].address == destination->limbs[limb].address ||
            right->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw matrix product limb mismatch or alias");
    if (cudaSetDevice(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t limb = 0; limb < left->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(left_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(right_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_mul_kernel,
                grid, dim3(kTransformThreads), 0, patches, 3,
                left->limbs[limb], right->limbs[limb], destination->limbs[limb],
                left->columns, right->columns, destination->columns,
                destination->rows,
                static_cast<size_t>(destination->degree), offset,
                accumulate != 0, transpose_rhs);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_mul(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination, int accumulate,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base)
{
    return raw_matrix_mul_impl(ctx, stream_raw, left, right, destination,
        accumulate, false, left_binding_base, right_binding_base,
        destination_binding_base);
}

extern "C" int gpu_raw_matrix_mul_transpose_rhs(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base)
{
    return raw_matrix_mul_impl(ctx, stream_raw, left, right, destination,
        0, true, left_binding_base, right_binding_base,
        destination_binding_base);
}

extern "C" int gpu_raw_matrix_transpose(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        source->physical_device != destination->physical_device ||
        source->limb_count != destination->limb_count ||
        source->degree != destination->degree ||
        source->rows != destination->columns ||
        source->columns != destination->rows ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix transpose views");
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        if (source->limbs[limb].crt_limb_index !=
                destination->limbs[limb].crt_limb_index ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus ||
            source->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw transpose CRT basis mismatch or alias");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(source_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((source->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_transpose_kernel,
                grid, dim3(kTransformThreads), 0, patches, 2,
                source->limbs[limb], destination->limbs[limb],
                source->columns, destination->columns,
                static_cast<size_t>(source->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_tensor(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *left, const MxxRawMatrixView *right,
    const MxxRawMatrixView *destination,
    uint32_t left_binding_base, uint32_t right_binding_base,
    uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, left, stream_raw) != 0 ||
        validate_raw_view(ctx, right, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        left->physical_device != right->physical_device ||
        left->physical_device != destination->physical_device ||
        left->degree != right->degree || left->degree != destination->degree ||
        left->limb_count != right->limb_count ||
        left->limb_count != destination->limb_count ||
        left->rows > SIZE_MAX / right->rows ||
        left->columns > SIZE_MAX / right->columns ||
        destination->rows != left->rows * right->rows ||
        destination->columns != left->columns * right->columns ||
        left_binding_base > UINT32_MAX - left->limb_count ||
        right_binding_base > UINT32_MAX - right->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix tensor views");
    for (size_t limb = 0; limb < left->limb_count; ++limb)
        if (left->limbs[limb].crt_limb_index != right->limbs[limb].crt_limb_index ||
            left->limbs[limb].crt_limb_index != destination->limbs[limb].crt_limb_index ||
            left->limbs[limb].modulus != right->limbs[limb].modulus ||
            left->limbs[limb].modulus != destination->limbs[limb].modulus ||
            left->limbs[limb].address == destination->limbs[limb].address ||
            right->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw tensor CRT basis mismatch or alias");
    if (cudaSetDevice(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t limb = 0; limb < left->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(left_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(right_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_tensor_kernel,
                grid, dim3(kTransformThreads), 0, patches, 3,
                left->limbs[limb], right->limbs[limb], destination->limbs[limb],
                left->columns, right->rows, right->columns, destination->columns,
                static_cast<size_t>(destination->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_copy(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->rows != destination->rows ||
        source->columns != destination->columns ||
        source->limb_count != destination->limb_count ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw matrix copy views");
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        if (source->limbs[limb].crt_limb_index !=
                destination->limbs[limb].crt_limb_index ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus)
            return set_error("raw matrix copy CRT basis mismatch");
    if (cudaSetDevice(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = source->rows * source->columns;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(source_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((source->degree + kTransformThreads - 1) /
                    kTransformThreads), static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_copy_kernel, grid, dim3(kTransformThreads), 0, patches, 2,
                source->limbs[limb], destination->limbs[limb], source->columns,
                static_cast<size_t>(source->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

static int raw_matrix_fill_impl(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *destination, uint64_t rows,
    uint64_t column_base, uint32_t digits_per_tower,
    const uint64_t *base_residues, bool gadget,
    uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !rows || !digits_per_tower ||
        (gadget && !base_residues) ||
        destination->row_origin > rows ||
        destination->rows > rows - destination->row_origin ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw structured fill arguments");
    const uint64_t tower_count = gadget ? ctx->moduli.size() : 1;
    if (!tower_count || digits_per_tower > UINT64_MAX / tower_count)
        return set_error("raw structured fill slots overflow");
    const uint64_t slots_per_row = digits_per_tower * tower_count;
    if (rows > UINT64_MAX / slots_per_row ||
        column_base > UINT64_MAX - rows * slots_per_row ||
        destination->column_origin < column_base ||
        destination->column_origin - column_base > rows * slots_per_row ||
        destination->columns > rows * slots_per_row -
            (destination->column_origin - column_base))
        return set_error("raw structured fill window mismatch");
    if (cudaSetDevice(destination->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const uint64_t base_residue = gadget ? base_residues[limb] : 1;
        if (base_residue >= destination->limbs[limb].modulus)
            return set_error("raw gadget base residue exceeds CRT modulus");
        const MxxGraphPatch patch = {
            nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
            static_cast<uint32_t>(destination_binding_base + limb), 0};
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kTransformThreads - 1) /
                    kTransformThreads), static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_structured_fill_kernel, grid, dim3(kTransformThreads), 0,
                &patch, 1, destination->limbs[limb], rows,
                destination->columns, destination->row_origin,
                destination->column_origin, column_base, slots_per_row,
                digits_per_tower, base_residue,
                static_cast<size_t>(destination->degree), offset, gadget);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_identity_fill(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *destination, uint64_t square_size,
    uint64_t column_base, uint32_t destination_binding_base)
{
    return raw_matrix_fill_impl(ctx, stream_raw, destination, square_size,
        column_base, 1, nullptr, false, destination_binding_base);
}

extern "C" int gpu_raw_matrix_gadget_fill(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *destination, uint64_t rows,
    uint32_t digits_per_tower, const uint64_t *base_residues,
    uint64_t column_base, uint32_t destination_binding_base)
{
    return raw_matrix_fill_impl(ctx, stream_raw, destination, rows,
        column_base, digits_per_tower, base_residues, true,
        destination_binding_base);
}
