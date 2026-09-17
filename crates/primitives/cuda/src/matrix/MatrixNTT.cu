#include "gpu_prepared_plan.cuh"

#include <atomic>
#include <cstring>
#include <limits>

namespace
{
    // Descriptors are owned by the matrix and initialized on its allocation
    // stream. Existing limb events protect both descriptor and coefficient use.
    struct MatrixNttDescriptorView
    {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
        uint32_t indices[GPU_RUNTIME_MAX_LIMBS];
        size_t offset, columns, pitch;
        __device__ size_t polynomial(size_t index) const {
            return columns == 0 || columns == pitch ? offset + index :
                offset + (index / columns) * pitch + index % columns;
        }
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
        const size_t poly_idx = layout.polynomial(poly_offset + static_cast<size_t>(blockIdx.y));
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
        const size_t poly_idx = layout.polynomial(poly_offset + static_cast<size_t>(blockIdx.y));
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
        const size_t poly_idx = layout.polynomial(poly_offset + static_cast<size_t>(blockIdx.y));
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
            gpu_test_record_kernel_launch();
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
            gpu_test_record_kernel_launch();
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
            gpu_test_record_kernel_launch();
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
        const size_t poly = layout.polynomial(poly_offset + blockIdx.y);
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
        const GpuRingDeviceConstants &constants,
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
                gpu_test_record_kernel_launch();
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
        const size_t poly = layout.polynomial(poly_offset + blockIdx.y);
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
        const GpuRingDeviceConstants &constants,
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
                gpu_test_record_kernel_launch(); \
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
    int run_matrix_transform_u64(GpuMatrix *mat, const GpuMatrixRange *range = nullptr)
    {
        gpu_test_record_native_validation();
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

        const GpuMatrixRange rectangle = range ? *range : GpuMatrixRange{0, mat->rows, 0, mat->cols};
        if (rectangle.row_start > rectangle.row_end || rectangle.row_end > mat->rows ||
            rectangle.column_start > rectangle.column_end || rectangle.column_end > mat->cols ||
            (mat->rows && mat->cols > std::numeric_limits<size_t>::max() / mat->rows))
            return set_error("invalid NTT rectangle");
        const size_t columns = rectangle.column_end - rectangle.column_start;
        const size_t poly_count = (rectangle.row_end - rectangle.row_start) * columns;
        if (poly_count == 0)
        {
            if (!range) mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            return 0;
        }

        MatrixNttDescriptorView layout{};
        if (range) {
            layout.offset = rectangle.row_start * mat->cols + rectangle.column_start;
            layout.columns = columns;
            layout.pitch = mat->cols;
        }

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

        if (dispatch_slot >= mat->ctx->ring_device_constants.size())
        {
            return set_error("missing per-device NTT constants in run_matrix_transform_u64");
        }
        const GpuRingDeviceConstants &device_constants = mat->ctx->ring_device_constants[dispatch_slot];
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

        if (!range) mat->format = Forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        return 0;
    }
}

enum class PreparedNttLaunchKind : uint8_t
{
    FusedTop,
    Twist,
    Stage,
    Scale,
    FusedLocal,
};

struct PreparedNttLaunch
{
    PreparedNttLaunchKind kind;
    MatrixNttDescriptorView layout;
    const GpuRingDeviceConstants *constants;
    cudaStream_t stream;
    dim3 grid;
    dim3 block;
    uint32_t n;
    uint32_t len;
    uint32_t width;
    size_t limb_count;
    size_t poly_offset;
    size_t shared_bytes;
    bool forward;
};

struct GpuMatrixTransformPlan
{
    GpuMatrix *matrix;
    std::vector<PreparedNttLaunch> launches;
};


struct PreparedNttBinding
{
    MatrixNttDescriptorView layout;
    const GpuRingDeviceConstants *constants;
    cudaStream_t stream;
    int device;
    uint32_t n;
    size_t limb_count;
    size_t polynomial_count;
};

int validate_saved_ntt_descriptor(
    const GpuMatrix *mat, const GpuMatrixRange *range, bool forward,
    const GpuPreparedPlanDescriptor *descriptor, PreparedNttBinding &binding)
{
    if (!mat || !mat->ctx || mat->level < 0 || mat->ctx->N < 2 ||
        !is_power_of_two_u32(static_cast<uint32_t>(mat->ctx->N)) || !descriptor)
        return set_error("invalid saved prepared NTT owners");
    const size_t limb_count = static_cast<size_t>(mat->level) + 1;
    if (limb_count > GPU_RUNTIME_MAX_LIMBS || mat->ctx->limb_gpu_ids.size() < limb_count ||
        mat->ctx->limb_prime_ids.size() < limb_count)
        return set_error("invalid saved prepared NTT limb metadata");
    const GpuMatrixRange selected = range ? *range :
        GpuMatrixRange{0, mat->rows, 0, mat->cols};
    if (selected.row_start > selected.row_end || selected.row_end > mat->rows ||
        selected.column_start > selected.column_end || selected.column_end > mat->cols ||
        selected.row_end == selected.row_start || selected.column_end == selected.column_start)
        return set_error("invalid saved prepared NTT range");
    const size_t rows = selected.row_end - selected.row_start;
    const size_t columns = selected.column_end - selected.column_start;
    if (rows > std::numeric_limits<size_t>::max() / columns)
        return set_error("saved prepared NTT polynomial count overflow");
    const size_t polynomial_count = rows * columns;
    if (descriptor->allocation_count != 1 || descriptor->stream_count != 1 ||
        descriptor->launch_count == 0)
        return set_error("saved prepared NTT descriptor counts mismatch");

    MatrixNttDescriptorView layout{};
    layout.offset = selected.row_start * mat->cols + selected.column_start;
    layout.columns = columns;
    layout.pitch = mat->cols;
    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    dim3 first_limb{};
    GpuPreparedResourceKey first_key{};
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = mat->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        int status = matrix_limb_device(mat, limb_id, &limb_device);
        if (status != 0) return status;
        if (limb == 0)
        {
            first_limb = limb_id;
            dispatch_device = limb_device;
            status = matrix_limb_stream(mat, limb_id, &dispatch_stream);
            if (status != 0 || !dispatch_stream)
                return status ? status : set_error("null saved prepared NTT stream");
        }
        else if (limb_device != dispatch_device)
            return set_error("saved prepared NTT requires one device");
        const int primeid = mat->ctx->limb_prime_ids[limb];
        if (primeid < 0 || static_cast<size_t>(primeid) >= mat->ctx->moduli.size() ||
            limb_id.x >= mat->ctx->gpu_ids.size() || limb_id.x >= mat->shared_limb_buffers.size())
            return set_error("invalid saved prepared NTT prime metadata");
        size_t stride_bytes = 0;
        uint8_t coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(mat, limb_id, &stride_bytes, &coeff_bytes) ||
            coeff_bytes == 0 || stride_bytes < static_cast<size_t>(mat->ctx->N) * coeff_bytes ||
            !matrix_limb_ptr_by_id(mat, 0, limb_id))
            return set_error("invalid saved prepared NTT limb layout");
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        if (!buffer.device_descriptors || limb_id.y >= buffer.limb_count)
            return set_error("missing saved prepared NTT descriptors");
        if (limb == 0) layout.descriptors = buffer.device_descriptors;
        else if (layout.descriptors != buffer.device_descriptors)
            return set_error("saved prepared NTT descriptors span partitions");
        layout.indices[limb] = limb_id.y;
    }
    int status = gpu_prepared_limb_key(mat->ctx, mat->level, 0,
        GPU_PREPARED_STAGE_NTT, &first_limb, &first_key);
    if (status != 0) return status;
    const auto &allocation = descriptor->allocations[0];
    if (descriptor->launch_count > std::numeric_limits<size_t>::max() /
            sizeof(GpuPreparedNttLaunchLayout))
        return set_error("saved prepared NTT geometry size overflow");
    const size_t geometry_bytes = descriptor->launch_count * sizeof(GpuPreparedNttLaunchLayout);
    if (gpu_prepared_require_allocation(descriptor, 0, GPU_PREPARED_PLAN_HOST_ONLY,
            &first_key, geometry_bytes, alignof(void *)) != 0 ||
        allocation.rows != 0 || allocation.columns != 0 || allocation.level != -1 ||
        allocation.format != -1)
        return set_error("saved prepared NTT geometry allocation differs from descriptor");
    const auto &stream_entry = descriptor->streams[0];
    if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        gpu_prepared_require_stream_slot(mat->ctx, first_limb.x, dispatch_stream,
            stream_entry.pool_slot) != 0 ||
        std::memcmp(&stream_entry.key, &first_key, sizeof(first_key)) != 0)
        return set_error("saved prepared NTT stream differs from descriptor");
    const uint32_t n = static_cast<uint32_t>(mat->ctx->N);
    const uint32_t tile_size = std::min(n, static_cast<uint32_t>(kFusedNttCoefficients));
    const uint32_t wide_blocks = (n + kTransformThreads - 1) / kTransformThreads;
    const uint32_t stage_blocks = ((n >> 1) + kTransformThreads - 1) / kTransformThreads;
    std::vector<std::pair<int, uint32_t>> expected_phases;
    if (forward)
    {
        if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            expected_phases.emplace_back(GPU_PREPARED_NTT_FUSED_TOP, 0);
        else if (n > kFusedNttCoefficients)
        {
            expected_phases.emplace_back(GPU_PREPARED_NTT_TWIST, 0);
            for (uint32_t len = n; len > kFusedNttCoefficients; len >>= 1)
                expected_phases.emplace_back(GPU_PREPARED_NTT_STAGE, len);
        }
        expected_phases.emplace_back(GPU_PREPARED_NTT_FUSED_LOCAL, 0);
    }
    else
    {
        expected_phases.emplace_back(GPU_PREPARED_NTT_FUSED_LOCAL, 0);
        if (n > kFusedNttCoefficients && n <= 16 * kFusedNttCoefficients)
            expected_phases.emplace_back(GPU_PREPARED_NTT_FUSED_TOP, 0);
        else if (n > kFusedNttCoefficients)
        {
            for (uint32_t len = kFusedNttCoefficients * 2; len <= n; len <<= 1)
                expected_phases.emplace_back(GPU_PREPARED_NTT_STAGE, len);
            expected_phases.emplace_back(GPU_PREPARED_NTT_SCALE, 0);
            expected_phases.emplace_back(GPU_PREPARED_NTT_TWIST, 0);
        }
    }
    if (expected_phases.size() != 0 && expected_phases.size() > descriptor->launch_count)
        return set_error("saved prepared NTT launch table is incomplete");
    size_t expected_phase_index = std::numeric_limits<size_t>::max();
    int previous_phase = -1;
    size_t phase_offset = 0;
    for (size_t index = 0; index < descriptor->launch_count; ++index)
    {
        const auto &launch = descriptor->launches[index];
        if (launch.phase != previous_phase)
        {
            if (previous_phase >= 0 && phase_offset != polynomial_count)
                return set_error("saved prepared NTT launch phase does not cover the range");
            ++expected_phase_index;
            if (expected_phase_index >= expected_phases.size() ||
                launch.phase != expected_phases[expected_phase_index].first)
                return set_error("saved prepared NTT launch phase order differs from descriptor");
            previous_phase = launch.phase;
            phase_offset = 0;
        }
        if (launch.block.x != kTransformThreads || launch.block.y != 1 || launch.block.z != 1 ||
            launch.grid.x == 0 || launch.grid.y == 0 || launch.grid.z != limb_count ||
            launch.grid.y > kMaxGridY || launch.limb_count != limb_count ||
            launch.limb_offset != phase_offset || launch.limb_offset >= polynomial_count ||
            launch.narrow != (forward ? 1 : 0))
            return set_error("saved prepared NTT launch geometry differs from descriptor");
        const size_t chunk = std::min(kMaxGridY, polynomial_count - launch.limb_offset);
        if (launch.grid.y != chunk) return set_error("saved prepared NTT launch chunk differs from descriptor");
        phase_offset += chunk;
        switch (launch.phase)
        {
        case GPU_PREPARED_NTT_FUSED_TOP:
            if (n <= kFusedNttCoefficients || n > 16 * kFusedNttCoefficients ||
                n % kFusedNttCoefficients != 0 || launch.len != expected_phases[expected_phase_index].second ||
                launch.grid.x != n / kTransformThreads)
                return set_error("saved prepared NTT fused-top geometry is invalid");
            break;
        case GPU_PREPARED_NTT_TWIST:
            if (launch.len != expected_phases[expected_phase_index].second || launch.grid.x != wide_blocks)
                return set_error("saved prepared NTT twist geometry is invalid");
            break;
        case GPU_PREPARED_NTT_STAGE:
            if (launch.len != expected_phases[expected_phase_index].second ||
                launch.len <= kFusedNttCoefficients || launch.len > n ||
                !is_power_of_two_u32(launch.len) || launch.grid.x != stage_blocks)
                return set_error("saved prepared NTT stage geometry is invalid");
            break;
        case GPU_PREPARED_NTT_SCALE:
            if (launch.len != expected_phases[expected_phase_index].second || launch.grid.x != wide_blocks)
                return set_error("saved prepared NTT scale geometry is invalid");
            break;
        case GPU_PREPARED_NTT_FUSED_LOCAL:
            if (launch.len != expected_phases[expected_phase_index].second || launch.grid.x != n / tile_size)
                return set_error("saved prepared NTT fused-local geometry is invalid");
            break;
        default:
            return set_error("saved prepared NTT launch phase is invalid");
        }
    }
    if (phase_offset != polynomial_count || expected_phase_index + 1 != expected_phases.size())
        return set_error("saved prepared NTT launch table does not cover the range");
    const size_t partition = static_cast<size_t>(first_limb.x);
    if (partition >= mat->ctx->ring_device_constants.size())
        return set_error("missing saved prepared NTT constants");
    const auto &constants = mat->ctx->ring_device_constants[partition];
    if (constants.device != dispatch_device || constants.limb_count < limb_count ||
        constants.ring_dimension != n || !constants.twiddle_forward || !constants.twiddle_inverse ||
        !constants.twiddle_shoup_forward || !constants.twiddle_shoup_inverse || !constants.moduli ||
        !constants.n_inv || !constants.n_inv_shoup)
        return set_error("invalid saved prepared NTT constants");
    binding = PreparedNttBinding{layout, &constants, dispatch_stream, dispatch_device, n,
        limb_count, polynomial_count};
    return 0;
}

int construct_saved_ntt_plan(
    const GpuMatrix *mat, bool forward, const PreparedNttBinding &binding,
    const GpuPreparedPlanDescriptor *descriptor, GpuMatrixTransformPlan **plan)
{
    try
    {
        auto prepared = std::make_unique<GpuMatrixTransformPlan>();
        prepared->matrix = const_cast<GpuMatrix *>(mat);
        prepared->launches.reserve(descriptor->launch_count);
        for (size_t index = 0; index < descriptor->launch_count; ++index)
        {
            const auto &entry = descriptor->launches[index];
            const int width = entry.phase == GPU_PREPARED_NTT_FUSED_TOP
                ? static_cast<int>(binding.n / kFusedNttCoefficients) : 0;
            const size_t shared_bytes = entry.phase == GPU_PREPARED_NTT_FUSED_LOCAL
                ? static_cast<size_t>(std::min(binding.n,
                    static_cast<uint32_t>(kFusedNttCoefficients))) * sizeof(uint64_t) : 0;
            prepared->launches.push_back(PreparedNttLaunch{
                static_cast<PreparedNttLaunchKind>(entry.phase), binding.layout,
                binding.constants, binding.stream, entry.grid, entry.block, binding.n,
                entry.len, static_cast<uint32_t>(width), entry.limb_count,
                entry.limb_offset, shared_bytes, forward,
            });
        }
        *plan = prepared.release();
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

template <bool Forward>
int submit_prepared_ntt_top_impl(const PreparedNttLaunch &launch)
{
    const auto &constants = *launch.constants;
    const uint64_t *twiddles = Forward ? constants.twiddle_forward : constants.twiddle_inverse;
    const uint64_t *shoup = Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse;
    switch (launch.width)
    {
#define MXX_SUBMIT_NTT_TOP(width) \
    case width: \
        ntt_fused_top_stages_kernel<Forward, width><<<launch.grid, launch.block, launch.shared_bytes, launch.stream>>>( \
            launch.layout, twiddles, shoup, constants.moduli, constants.n_inv, \
            constants.n_inv_shoup, launch.poly_offset); \
        break
        MXX_SUBMIT_NTT_TOP(2);
        MXX_SUBMIT_NTT_TOP(4);
        MXX_SUBMIT_NTT_TOP(8);
        MXX_SUBMIT_NTT_TOP(16);
#undef MXX_SUBMIT_NTT_TOP
    default:
        return set_error("invalid prepared NTT fused-top width");
    }
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
}

template <bool Forward>
int submit_prepared_ntt_twist_impl(const PreparedNttLaunch &launch)
{
    const auto &constants = *launch.constants;
    ntt_twist_all_limbs_kernel<<<launch.grid, launch.block, launch.shared_bytes, launch.stream>>>(
        launch.layout,
        Forward ? constants.twiddle_forward : constants.twiddle_inverse,
        Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse,
        constants.moduli,
        launch.limb_count,
        launch.n,
        launch.poly_offset);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
}

template <bool Forward>
int submit_prepared_ntt_stage_impl(const PreparedNttLaunch &launch)
{
    const auto &constants = *launch.constants;
    ntt_stage_all_limbs_kernel<Forward><<<launch.grid, launch.block, launch.shared_bytes, launch.stream>>>(
        launch.layout,
        Forward ? constants.twiddle_forward : constants.twiddle_inverse,
        Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse,
        constants.moduli,
        launch.limb_count,
        launch.n,
        launch.len,
        launch.poly_offset);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
}

template <bool Forward>
int submit_prepared_ntt_local_impl(const PreparedNttLaunch &launch)
{
    const auto &constants = *launch.constants;
    ntt_fused_local_stages_kernel<Forward><<<launch.grid, launch.block, launch.shared_bytes, launch.stream>>>(
        launch.layout,
        Forward ? constants.twiddle_forward : constants.twiddle_inverse,
        Forward ? constants.twiddle_shoup_forward : constants.twiddle_shoup_inverse,
        constants.moduli, constants.n_inv, constants.n_inv_shoup,
        launch.n,
        std::min(launch.n, kFusedNttCoefficients),
        launch.poly_offset);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
}

int submit_prepared_ntt_scale(const PreparedNttLaunch &launch)
{
    const auto &constants = *launch.constants;
    ntt_scale_all_limbs_kernel<<<launch.grid, launch.block, launch.shared_bytes, launch.stream>>>(
        launch.layout, constants.moduli, constants.n_inv, constants.n_inv_shoup,
        launch.limb_count, launch.n, launch.poly_offset);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : set_error(error);
}

int submit_prepared_ntt_launch(const PreparedNttLaunch &launch)
{
    switch (launch.kind)
    {
    case PreparedNttLaunchKind::FusedTop:
        return launch.forward ? submit_prepared_ntt_top_impl<true>(launch) : submit_prepared_ntt_top_impl<false>(launch);
    case PreparedNttLaunchKind::Twist:
        return launch.forward ? submit_prepared_ntt_twist_impl<true>(launch) : submit_prepared_ntt_twist_impl<false>(launch);
    case PreparedNttLaunchKind::Stage:
        return launch.forward ? submit_prepared_ntt_stage_impl<true>(launch) : submit_prepared_ntt_stage_impl<false>(launch);
    case PreparedNttLaunchKind::Scale:
        return submit_prepared_ntt_scale(launch);
    case PreparedNttLaunchKind::FusedLocal:
        return launch.forward ? submit_prepared_ntt_local_impl<true>(launch) : submit_prepared_ntt_local_impl<false>(launch);
    }
    return set_error("invalid prepared NTT launch kind");
}

extern "C" int gpu_matrix_query_ntt_layout(
    size_t ring_dimension, size_t limb_count, size_t polynomial_count,
    int device, bool forward, GpuPreparedNttLayout *out)
{
    if (!out || ring_dimension < 2 || ring_dimension > std::numeric_limits<uint32_t>::max() ||
        !is_power_of_two_u32(static_cast<uint32_t>(ring_dimension)) ||
        limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS || polynomial_count == 0)
        return set_error("invalid prepared NTT layout request");
    size_t records = 0;
    int status = gpu_prepared_ntt_launch_table(
        static_cast<uint32_t>(ring_dimension), limb_count, polynomial_count,
        forward ? 1 : 0, nullptr, 0, &records);
    if (status != 0) return status;
    if (polynomial_count > std::numeric_limits<size_t>::max() - (kMaxGridY - 1))
        return set_error("prepared NTT layout size overflow");
    const size_t chunks = (polynomial_count + kMaxGridY - 1) / kMaxGridY;
    if (chunks == 0 || records % chunks != 0)
        return set_error("prepared NTT launch table has invalid phase count");
    const size_t stages = records / chunks;
    const size_t tile_size = std::min(ring_dimension, static_cast<size_t>(kFusedNttCoefficients));
    size_t max_grid_x = 0;
    std::vector<GpuPreparedNttLaunchLayout> table(records);
    status = gpu_prepared_ntt_launch_table(
        static_cast<uint32_t>(ring_dimension), limb_count, polynomial_count,
        forward ? 1 : 0, table.data(), table.size(), &records);
    if (status != 0) return status;
    for (const auto &launch : table)
        max_grid_x = std::max(max_grid_x, static_cast<size_t>(launch.grid.x));
    *out = GpuPreparedNttLayout{
        ring_dimension, limb_count, polynomial_count, records, stages,
        tile_size * sizeof(uint64_t), alignof(uint64_t), limb_count,
        max_grid_x, std::min(polynomial_count, kMaxGridY), limb_count,
        device, forward ? 1 : 0,
    };
    return 0;
}

static int prepare_ntt_plan_legacy_impl(
    const GpuMatrix *mat, const GpuMatrixRange *range,
    bool forward, GpuMatrixTransformPlan **plan)
{
    if (!mat || !mat->ctx || !plan || mat->level < 0 || mat->ctx->N < 2 ||
        !is_power_of_two_u32(static_cast<uint32_t>(mat->ctx->N)))
        return set_error("invalid prepared NTT matrix");
    const GpuMatrixRange selected = range ? *range :
        GpuMatrixRange{0, mat->rows, 0, mat->cols};
    if (selected.row_start > selected.row_end || selected.row_end > mat->rows ||
        selected.column_start > selected.column_end || selected.column_end > mat->cols ||
        selected.row_end - selected.row_start == 0 ||
        selected.column_end - selected.column_start == 0)
        return set_error("invalid prepared NTT range");
    const uint32_t n = static_cast<uint32_t>(mat->ctx->N);
    const size_t limb_count = static_cast<size_t>(mat->level) + 1;
    if (limb_count > GPU_RUNTIME_MAX_LIMBS || mat->ctx->limb_gpu_ids.size() < limb_count ||
        mat->ctx->limb_prime_ids.size() < limb_count)
        return set_error("invalid prepared NTT limb metadata");
    MatrixNttDescriptorView layout{};
    layout.offset = selected.row_start * mat->cols + selected.column_start;
    layout.columns = selected.column_end - selected.column_start;
    layout.pitch = mat->cols;
    int dispatch_device = -1;
    size_t dispatch_slot = std::numeric_limits<size_t>::max();
    cudaStream_t dispatch_stream = nullptr;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = mat->ctx->limb_gpu_ids[limb];
        if (limb_id.x >= mat->shared_limb_buffers.size())
            return set_error("invalid prepared NTT limb partition");
        int limb_device = -1;
        int status = matrix_limb_device(mat, limb_id, &limb_device);
        if (status != 0) return status;
        if (limb == 0)
        {
            dispatch_device = limb_device;
            dispatch_slot = static_cast<size_t>(limb_id.x);
            status = matrix_limb_stream(mat, limb_id, &dispatch_stream);
            if (status != 0 || !dispatch_stream) return status ? status : set_error("null prepared NTT stream");
        }
        else if (limb_device != dispatch_device)
            return set_error("prepared NTT requires one device");
        const int primeid = mat->ctx->limb_prime_ids[limb];
        if (primeid < 0 || static_cast<size_t>(primeid) >= mat->ctx->moduli.size() ||
            limb_id.x >= mat->ctx->gpu_ids.size())
            return set_error("invalid prepared NTT prime metadata");
        size_t stride_bytes = 0;
        uint8_t coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(mat, limb_id, &stride_bytes, &coeff_bytes) ||
            coeff_bytes == 0 || stride_bytes < static_cast<size_t>(n) * coeff_bytes)
            return set_error("invalid prepared NTT limb layout");
        if (!matrix_limb_ptr_by_id(mat, 0, limb_id))
            return set_error("null prepared NTT limb pointer");
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        if (!buffer.device_descriptors || limb_id.y >= buffer.limb_count)
            return set_error("missing prepared NTT descriptors");
        if (limb == 0) layout.descriptors = buffer.device_descriptors;
        else if (layout.descriptors != buffer.device_descriptors)
            return set_error("prepared NTT descriptors span partitions");
        layout.indices[limb] = limb_id.y;
    }
    if (dispatch_slot >= mat->ctx->ring_device_constants.size())
        return set_error("missing prepared NTT constants");
    const auto &constants = mat->ctx->ring_device_constants[dispatch_slot];
    if (constants.device != dispatch_device || constants.limb_count < limb_count ||
        constants.ring_dimension != n || !constants.twiddle_forward ||
        !constants.twiddle_inverse || !constants.twiddle_shoup_forward ||
        !constants.twiddle_shoup_inverse || !constants.moduli || !constants.n_inv ||
        !constants.n_inv_shoup)
        return set_error("invalid prepared NTT constants");
    const size_t poly_rows = selected.row_end - selected.row_start;
    const size_t poly_columns = selected.column_end - selected.column_start;
    if (poly_rows != 0 && poly_columns > std::numeric_limits<size_t>::max() / poly_rows)
        return set_error("prepared NTT polynomial count overflow");
    const size_t poly_count = poly_rows * poly_columns;
    GpuPreparedNttLayout structural{};
    int status = gpu_matrix_query_ntt_layout(
        n, limb_count, poly_count, dispatch_device, forward, &structural);
    if (status != 0) return status;
    try
    {
        std::vector<PreparedNttLaunch> launches;
        launches.reserve(structural.launch_count);
        std::vector<GpuPreparedNttLaunchLayout> table(structural.launch_count);
        size_t table_count = 0;
        status = gpu_prepared_ntt_launch_table(
            n, limb_count, poly_count, forward ? 1 : 0, table.data(), table.size(), &table_count);
        if (status != 0) return status;
        if (table_count != structural.launch_count)
            return set_error("prepared NTT launch table count drift");
        for (const auto &entry : table)
        {
            launches.push_back(PreparedNttLaunch{
                static_cast<PreparedNttLaunchKind>(entry.kind),
                layout,
                &constants,
                dispatch_stream,
                entry.grid,
                entry.block,
                entry.n,
                entry.len,
                entry.width,
                entry.limb_count,
                entry.poly_offset,
                entry.shared_bytes,
                entry.forward != 0,
            });
        }
        if (launches.size() != structural.launch_count)
            return set_error("prepared NTT launch descriptor mismatch");
        auto prepared = std::make_unique<GpuMatrixTransformPlan>();
        prepared->matrix = const_cast<GpuMatrix *>(mat);
        prepared->launches = std::move(launches);
        *plan = prepared.release();
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" int gpu_matrix_prepare_ntt_plan_with_layout(
    const GpuMatrix *mat, const GpuMatrixRange *range, bool forward,
    const GpuPreparedPlanDescriptor *layout, GpuMatrixTransformPlan **plan)
{
    if (!layout || !plan || gpu_prepared_validate_descriptor(layout) != 0)
        return set_error("prepared NTT layout is missing");
    *plan = nullptr;
    PreparedNttBinding binding{};
    // The legacy entry point obtains this same geometry from
    // gpu_prepared_ntt_launch_table; the saved bind consumes the descriptor's
    // already-materialized launch table instead of invoking that planner.
    // validate_saved_ntt_descriptor performs gpu_prepared_require_allocation
    // and gpu_prepared_require_stream_slot checks before construction.
    const int status = validate_saved_ntt_descriptor(mat, range, forward, layout, binding);
    if (status != 0) return status;
    // Construction consumes only the launch records already validated above.
    // In particular, this path never enters the legacy preparation function or
    // asks the launch planner to recompute a table.
    return construct_saved_ntt_plan(mat, forward, binding, layout, plan);
}

// Standalone non-prepared transform API. Saved prepared binds use the
// descriptor-driven entry above and never route through this planner.
extern "C" int gpu_matrix_prepare_ntt_plan(
    const GpuMatrix *mat, const GpuMatrixRange *range,
    bool forward, GpuMatrixTransformPlan **plan)
{
    // The legacy implementation uses gpu_prepared_ntt_launch_table to build
    // its launch vector; the saved bind below consumes that table verbatim.
    return prepare_ntt_plan_legacy_impl(mat, range, forward, plan);
}

extern "C" int gpu_matrix_submit_ntt_plan(
    const GpuMatrixTransformPlan *plan, GpuMatrix *mat)
{
    if (!plan || !mat || plan->matrix != mat)
        return set_error("prepared NTT matrix binding mismatch");
    if (plan->launches.empty())
        return set_error("prepared NTT has no launch records");
    const auto &first = plan->launches.front();
    int status = matrix_wait_all_limb_streams(mat, first.constants->device, first.stream);
    if (status != 0) return status;
    for (const auto &launch : plan->launches)
    {
        status = submit_prepared_ntt_launch(launch);
        if (status != 0) return status;
    }
    if (status == 0) status = matrix_record_all_limb_writes(mat, first.stream);
    if (status == 0)
        mat->format = first.forward ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
    return status;
}

extern "C" void gpu_matrix_destroy_ntt_plan(GpuMatrixTransformPlan *plan)
{
    delete plan;
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
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);
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
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);
    return run_matrix_transform_u64<false>(mat);
}
