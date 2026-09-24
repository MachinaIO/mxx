#include <type_traits>

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
        if (modulus < (uint64_t(1) << 31) && value <= UINT32_MAX)
        {
            // floor(w 2^32 / q) is the high word of floor(w 2^64 / q), and a
            // 32-bit Shoup product needs no 64-bit multiplies.
            const uint32_t q = static_cast<uint32_t>(modulus);
            const uint32_t quotient = __umulhi(static_cast<uint32_t>(value),
                static_cast<uint32_t>(multiplier_shoup >> 32));
            uint32_t reduced = static_cast<uint32_t>(value) * static_cast<uint32_t>(multiplier) -
                quotient * q;
            if (reduced >= q) reduced -= q;
            return reduced;
        }
        const uint64_t quotient = __umul64hi(value, multiplier_shoup);
        uint64_t reduced = value * multiplier - quotient * modulus;
        if (reduced >= modulus) reduced -= modulus;
        return reduced;
    }

    bool is_power_of_two_u32(uint32_t v)
    {
        return v != 0 && (v & (v - 1)) == 0;
    }

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

    // One fused transform launch covers up to this many CRT limbs; each limb
    // carries its own views and NTT tables through the kernel arguments.
    constexpr size_t kRawNttLimbs = 8;
    constexpr uint32_t kFusedNttCoefficients = 1024;

    // The views of one operand for every limb of a launch; blockIdx.z picks
    // the limb, so one kernel covers all limbs of up to kRawNttLimbs.
    struct RawLimbSet
    {
        MxxRawMatrixLimb limb[kRawNttLimbs];
    };

    struct RawNttBatch
    {
        MxxRawMatrixLimb source[kRawNttLimbs];
        MxxRawMatrixLimb destination[kRawNttLimbs];
        const uint64_t *twiddles[kRawNttLimbs];
        const uint64_t *shoup[kRawNttLimbs];
        const uint64_t *n_inv[kRawNttLimbs];
        const uint64_t *n_inv_shoup[kRawNttLimbs];
    };
    static_assert(sizeof(RawNttBatch) < 4096, "bounded raw NTT kernel arguments");

    // Up to ten butterfly stages of one 1024-coefficient tile in shared
    // memory. A tile holding the whole transform also applies the twist
    // (forward) or the scaling and untwist (inverse). Every stage twiddle of
    // a tile is one of `tile_size / 2` table entries, staged in shared memory
    // once so the stages never wait on global loads.
    template <bool Forward>
    __global__ void raw_ntt_fused_local_kernel(RawNttBatch batch, size_t columns,
        uint32_t n, uint32_t tile_size, size_t poly_offset)
    {
        extern __shared__ uint64_t values[];
        const uint32_t limb = blockIdx.z;
        const MxxRawMatrixLimb &source = batch.source[limb];
        const MxxRawMatrixLimb &destination = batch.destination[limb];
        const uint64_t *twiddles = batch.twiddles[limb];
        const uint64_t *shoup = batch.shoup[limb];
        const uint64_t modulus = destination.modulus;
        const size_t poly = poly_offset + blockIdx.y;
        const uint32_t first = blockIdx.x * tile_size;
        uint64_t *stage_twiddles = values + tile_size;
        uint64_t *stage_shoup = stage_twiddles + tile_size / 2;
        const uint32_t step = 2U * (n / tile_size);
        for (uint32_t index = threadIdx.x; index < tile_size / 2; index += blockDim.x)
        {
            stage_twiddles[index] = twiddles[index * step];
            stage_shoup[index] = shoup[index * step];
        }
        for (uint32_t index = threadIdx.x; index < tile_size; index += blockDim.x)
        {
            uint64_t value = raw_matrix_load(source, poly, first + index, columns);
            if constexpr (Forward)
            {
                if (tile_size == n)
                    value = mul_mod_shoup_u64(value, twiddles[index], shoup[index], modulus);
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
                // Table entry 2 (n / length) j is staged entry j tile / length.
                const uint32_t twiddle = j * (tile_size / length);
                const uint64_t lower = values[index];
                const uint64_t upper = values[index + half];
                if constexpr (Forward)
                {
                    values[index] = add_mod_u64(lower, upper, modulus);
                    values[index + half] = mul_mod_shoup_u64(sub_mod_u64(lower, upper, modulus),
                        stage_twiddles[twiddle], stage_shoup[twiddle], modulus);
                }
                else
                {
                    const uint64_t product = mul_mod_shoup_u64(
                        upper, stage_twiddles[twiddle], stage_shoup[twiddle], modulus);
                    values[index] = add_mod_u64(lower, product, modulus);
                    values[index + half] = sub_mod_u64(lower, product, modulus);
                }
            }
            __syncthreads();
            length = Forward ? length >> 1 : length << 1;
        }
        for (uint32_t index = threadIdx.x; index < tile_size; index += blockDim.x)
        {
            uint64_t value = values[index];
            if constexpr (!Forward)
            {
                if (tile_size == n)
                {
                    value = mul_mod_shoup_u64(value, *batch.n_inv[limb],
                        *batch.n_inv_shoup[limb], modulus);
                    value = mul_mod_shoup_u64(value, twiddles[index], shoup[index], modulus);
                }
            }
            raw_matrix_store(destination, poly, first + index, columns, value);
        }
    }

    // The stages above one tile: the Width lanes of a group hold coefficients
    // one tile apart, and XOR shuffles realize those butterflies, including
    // the transform boundary twist (forward) or scaling and untwist (inverse).
    template <bool Forward, uint32_t Width>
    __global__ void raw_ntt_fused_top_kernel(RawNttBatch batch, size_t columns,
        size_t poly_offset)
    {
        const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t lane = thread % Width;
        const uint32_t column = thread / Width;
        const uint32_t coefficient = column + lane * kFusedNttCoefficients;
        const uint32_t limb = blockIdx.z;
        const MxxRawMatrixLimb &source = batch.source[limb];
        const MxxRawMatrixLimb &destination = batch.destination[limb];
        const uint64_t *twiddles = batch.twiddles[limb];
        const uint64_t *shoup = batch.shoup[limb];
        const uint64_t modulus = destination.modulus;
        const size_t poly = poly_offset + blockIdx.y;
        uint64_t value = raw_matrix_load(source, poly, coefficient, columns);
        if constexpr (Forward)
            value = mul_mod_shoup_u64(value, twiddles[coefficient], shoup[coefficient], modulus);
        uint32_t half_lanes = Forward ? Width / 2 : 1;
        while (half_lanes >= 1 && half_lanes < Width)
        {
            const uint64_t partner = __shfl_xor_sync(
                0xffffffffU, static_cast<unsigned long long>(value), half_lanes, Width);
            const bool upper_lane = (lane & half_lanes) != 0;
            const uint64_t lower = upper_lane ? partner : value;
            const uint64_t upper = upper_lane ? value : partner;
            const uint32_t j = column + (lane % half_lanes) * kFusedNttCoefficients;
            const uint32_t twiddle = (Width / half_lanes) * j;
            if constexpr (Forward)
            {
                value = upper_lane ? mul_mod_shoup_u64(sub_mod_u64(lower, upper, modulus),
                    twiddles[twiddle], shoup[twiddle], modulus)
                    : add_mod_u64(lower, upper, modulus);
                half_lanes >>= 1;
            }
            else
            {
                const uint64_t product =
                    mul_mod_shoup_u64(upper, twiddles[twiddle], shoup[twiddle], modulus);
                value = upper_lane ? sub_mod_u64(lower, product, modulus)
                    : add_mod_u64(lower, product, modulus);
                half_lanes <<= 1;
            }
        }
        if constexpr (!Forward)
        {
            value = mul_mod_shoup_u64(value, *batch.n_inv[limb], *batch.n_inv_shoup[limb], modulus);
            value = mul_mod_shoup_u64(value, twiddles[coefficient], shoup[coefficient], modulus);
        }
        raw_matrix_store(destination, poly, coefficient, columns, value);
    }

    template <bool Forward>
    int launch_raw_ntt_top(GpuContext *ctx, cudaStream_t stream, const RawNttBatch &batch,
        const MxxGraphPatch *patches, size_t patch_count, size_t limbs, uint32_t n,
        size_t columns, size_t poly_offset, size_t poly_chunk)
    {
        const dim3 grid(n / kTransformThreads, static_cast<uint32_t>(poly_chunk),
            static_cast<uint32_t>(limbs));
        switch (n / kFusedNttCoefficients)
        {
#define MXX_RAW_NTT_TOP(width) \
        case width: \
            return mxx_gpu_launch_kernel(ctx, stream, raw_ntt_fused_top_kernel<Forward, width>, \
                grid, dim3(kTransformThreads), 0, patches, patch_count, batch, columns, \
                poly_offset)
            MXX_RAW_NTT_TOP(2);
            MXX_RAW_NTT_TOP(4);
            MXX_RAW_NTT_TOP(8);
            MXX_RAW_NTT_TOP(16);
            MXX_RAW_NTT_TOP(32);
#undef MXX_RAW_NTT_TOP
        }
        return set_error("raw fused NTT has no top-stage width for this ring");
    }

    // Copies of several equal-shape (source, destination) windows, one
    // entry per window and limb; blockIdx.z picks the entry.
    struct RawCopyBatch
    {
        MxxRawMatrixLimb source[kRawNttLimbs];
        MxxRawMatrixLimb destination[kRawNttLimbs];
        uint64_t columns[kRawNttLimbs];
        uint64_t polys[kRawNttLimbs];
    };
    static_assert(sizeof(RawCopyBatch) < 4096, "bounded raw copy kernel arguments");

    __global__ void raw_matrix_copy_batch_kernel(RawCopyBatch batch, size_t degree,
        size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const uint32_t entry = blockIdx.z;
        const size_t poly = poly_offset + blockIdx.y;
        if (coefficient >= degree || poly >= batch.polys[entry]) return;
        raw_matrix_store(batch.destination[entry], poly, coefficient, batch.columns[entry],
            raw_matrix_load(batch.source[entry], poly, coefficient, batch.columns[entry]));
    }

    __global__ void raw_matrix_add_sub_kernel(
        RawLimbSet lefts, RawLimbSet rights,
        RawLimbSet destinations, size_t columns,
        size_t degree, size_t poly_offset, bool subtract)
    {
        const MxxRawMatrixLimb &left = lefts.limb[blockIdx.z];
        const MxxRawMatrixLimb &right = rights.limb[blockIdx.z];
        const MxxRawMatrixLimb &destination = destinations.limb[blockIdx.z];
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

    constexpr unsigned kMulCoefficients = 64;
    constexpr unsigned kMulSlices = 4;

    // Each block covers `kMulCoefficients` coefficients of one output poly;
    // its `kMulSlices` thread rows split the inner dimension. Products are
    // summed in 128 bits and reduced once per batch that cannot overflow.
    __global__ void raw_matrix_mul_kernel(
        RawLimbSet lefts, RawLimbSet rights,
        RawLimbSet destinations, size_t left_columns,
        size_t right_columns, size_t output_columns, size_t output_rows,
        size_t degree, size_t poly_offset, bool accumulate,
        bool transpose_rhs)
    {
        __shared__ uint64_t partial[kMulSlices][kMulCoefficients];
        const MxxRawMatrixLimb &left = lefts.limb[blockIdx.z];
        const MxxRawMatrixLimb &right = rights.limb[blockIdx.z];
        const MxxRawMatrixLimb &destination = destinations.limb[blockIdx.z];
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t output_poly = poly_offset + blockIdx.y;
        const size_t row = output_poly / output_columns;
        const size_t column = output_poly - row * output_columns;
        const bool active = coefficient < degree && row < output_rows;
        const uint64_t modulus = destination.modulus;
        const unsigned bits = 64 - __clzll(modulus);
        const unsigned headroom = bits <= 63 ? 128 - 2 * bits : 0;
        const size_t batch = headroom >= 6 ? 63 : headroom ? (size_t(1) << headroom) - 1 : 0;
        uint64_t sum = 0;
        if (active)
        {
            unsigned __int128 wide = 0;
            size_t pending = 0;
            for (size_t inner = threadIdx.y; inner < left_columns; inner += kMulSlices)
            {
                const uint64_t lhs = raw_matrix_load(
                    left, row * left_columns + inner, coefficient, left_columns);
                const size_t rhs_poly = transpose_rhs ?
                    column * left_columns + inner : inner * right_columns + column;
                const uint64_t rhs = raw_matrix_load(right, rhs_poly, coefficient,
                    transpose_rhs ? left_columns : right_columns);
                if (!batch)
                {
                    sum = add_mod_u64(sum, mul_mod_u64(lhs, rhs, modulus), modulus);
                    continue;
                }
                wide += static_cast<unsigned __int128>(lhs) * rhs;
                if (++pending == batch)
                {
                    wide %= modulus;
                    pending = 0;
                }
            }
            if (batch) sum = static_cast<uint64_t>(wide % modulus);
        }
        partial[threadIdx.y][threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.y != 0 || !active) return;
        for (unsigned slice = 1; slice < kMulSlices; ++slice)
            sum = add_mod_u64(sum, partial[slice][threadIdx.x], modulus);
        if (accumulate)
            sum = add_mod_u64(sum, raw_matrix_load(
                destination, output_poly, coefficient, output_columns), modulus);
        raw_matrix_store(destination, output_poly, coefficient, output_columns, sum);
    }

    // Multiply every polynomial of `matrix` by the single polynomial of
    // `scalar` in the evaluation domain.
    __global__ void raw_matrix_mul_scalar_kernel(
        MxxRawMatrixLimb matrix, MxxRawMatrixLimb scalar,
        MxxRawMatrixLimb destination, size_t columns, size_t degree,
        size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const size_t poly = poly_offset + blockIdx.y;
        raw_matrix_store(destination, poly, coefficient, columns,
            mul_mod_u64(raw_matrix_load(matrix, poly, coefficient, columns),
                raw_matrix_load(scalar, 0, coefficient, 1), destination.modulus));
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
        RawLimbSet lefts, RawLimbSet rights,
        RawLimbSet destinations, size_t left_columns,
        size_t right_rows, size_t right_columns, size_t output_columns,
        size_t degree, size_t poly_offset)
    {
        const MxxRawMatrixLimb &left = lefts.limb[blockIdx.z];
        const MxxRawMatrixLimb &right = rights.limb[blockIdx.z];
        const MxxRawMatrixLimb &destination = destinations.limb[blockIdx.z];
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

// Fill the views of `limbs` limbs from `first` for kernel argument
// `argument`, with one address patch per limb. The views start
// `column_shift` columns after the bound address; the graph builder records
// that offset from the captured address.
static void raw_limb_set(const MxxRawMatrixView *view, size_t first, size_t limbs,
    uint32_t argument, uint32_t binding_base, RawLimbSet &set,
    std::vector<MxxGraphPatch> &patches, size_t column_shift = 0)
{
    for (size_t local = 0; local < limbs; ++local)
    {
        set.limb[local] = view->limbs[first + local];
        set.limb[local].address += column_shift * set.limb[local].column_stride_bytes;
        patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, argument,
            static_cast<uint32_t>(local * sizeof(MxxRawMatrixLimb) +
                offsetof(MxxRawMatrixLimb, address)),
            sizeof(void *), binding_base + static_cast<uint32_t>(first + local), 0});
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
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t n = source->degree;
    const size_t poly_count = source->rows * source->columns;
    if (n > kFusedNttCoefficients * 32 || n % kTransformThreads != 0 && n > kFusedNttCoefficients)
        return set_error("raw fused NTT supports ring dimensions up to 32768");
    const uint32_t tile_size = std::min(n, kFusedNttCoefficients);
    // Every transform is one or two launches over all limbs of a batch: a
    // ring above one tile runs its top stages and its local stages, reading
    // the source in the first launch and updating the destination in place.
    for (size_t first = 0; first < source->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, source->limb_count - first);
        RawNttBatch batch{};
        std::vector<MxxGraphPatch> source_patches;
        std::vector<MxxGraphPatch> destination_patches;
        for (size_t local = 0; local < limbs; ++local)
        {
            const size_t limb_index = first + local;
            const auto &source_limb = source->limbs[limb_index];
            const auto &output_limb = destination->limbs[limb_index];
            const dim3 partition = ctx->limb_gpu_ids[source_limb.crt_limb_index];
            if (partition.x >= ctx->ntt_device_constants.size())
                return set_error("missing raw NTT device constants");
            const auto &constants = ctx->ntt_device_constants[partition.x];
            if (constants.device != source->physical_device ||
                partition.y >= constants.limb_count ||
                constants.ring_dimension != n ||
                !constants.twiddle_forward || !constants.twiddle_inverse ||
                !constants.twiddle_shoup_forward || !constants.twiddle_shoup_inverse ||
                !constants.n_inv || !constants.n_inv_shoup)
                return set_error("invalid raw NTT device constants");
            const size_t table = static_cast<size_t>(partition.y) * n;
            batch.source[local] = source_limb;
            batch.destination[local] = output_limb;
            batch.twiddles[local] =
                (inverse ? constants.twiddle_inverse : constants.twiddle_forward) + table;
            batch.shoup[local] =
                (inverse ? constants.twiddle_shoup_inverse : constants.twiddle_shoup_forward) + table;
            batch.n_inv[local] = constants.n_inv + partition.y;
            batch.n_inv_shoup[local] = constants.n_inv_shoup + partition.y;
            const uint32_t source_binding = source_binding_base + static_cast<uint32_t>(limb_index);
            const uint32_t destination_binding =
                destination_binding_base + static_cast<uint32_t>(limb_index);
            const uint32_t source_offset = static_cast<uint32_t>(
                offsetof(RawNttBatch, source) + local * sizeof(MxxRawMatrixLimb) +
                offsetof(MxxRawMatrixLimb, address));
            const uint32_t destination_offset = static_cast<uint32_t>(
                offsetof(RawNttBatch, destination) + local * sizeof(MxxRawMatrixLimb) +
                offsetof(MxxRawMatrixLimb, address));
            source_patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                source_offset, sizeof(void *), source_binding, 0});
            source_patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                destination_offset, sizeof(void *), destination_binding, 0});
            destination_patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                source_offset, sizeof(void *), destination_binding, 0});
            destination_patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                destination_offset, sizeof(void *), destination_binding, 0});
        }
        RawNttBatch in_place = batch;
        for (size_t local = 0; local < limbs; ++local) in_place.source[local] = batch.destination[local];
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 local_grid(n / tile_size, static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(limbs));
            const size_t shared = 2 * tile_size * sizeof(uint64_t);
            int status = 0;
            if (n <= kFusedNttCoefficients)
            {
                status = inverse ?
                    mxx_gpu_launch_kernel(ctx, stream, raw_ntt_fused_local_kernel<false>,
                        local_grid, dim3(kTransformThreads), shared, source_patches.data(),
                        source_patches.size(), batch, static_cast<size_t>(source->columns), n,
                        tile_size, offset) :
                    mxx_gpu_launch_kernel(ctx, stream, raw_ntt_fused_local_kernel<true>,
                        local_grid, dim3(kTransformThreads), shared, source_patches.data(),
                        source_patches.size(), batch, static_cast<size_t>(source->columns), n,
                        tile_size, offset);
            }
            else if (!inverse)
            {
                status = launch_raw_ntt_top<true>(ctx, stream, batch, source_patches.data(),
                    source_patches.size(), limbs, n, source->columns, offset, chunk);
                if (status == 0)
                    status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_fused_local_kernel<true>,
                        local_grid, dim3(kTransformThreads), shared, destination_patches.data(),
                        destination_patches.size(), in_place,
                        static_cast<size_t>(source->columns), n, tile_size, offset);
            }
            else
            {
                status = mxx_gpu_launch_kernel(ctx, stream, raw_ntt_fused_local_kernel<false>,
                    local_grid, dim3(kTransformThreads), shared, source_patches.data(),
                    source_patches.size(), batch, static_cast<size_t>(source->columns), n,
                    tile_size, offset);
                if (status == 0)
                    status = launch_raw_ntt_top<false>(ctx, stream, in_place,
                        destination_patches.data(), destination_patches.size(), limbs, n,
                        source->columns, offset, chunk);
            }
            if (status != 0) return status;
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
    if (mxx_set_device(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = left->rows * left->columns;
    for (size_t first = 0; first < left->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, left->limb_count - first);
        RawLimbSet lefts{}, rights{}, destinations{};
        std::vector<MxxGraphPatch> patches;
        raw_limb_set(left, first, limbs, 0, left_binding_base, lefts, patches);
        raw_limb_set(right, first, limbs, 1, right_binding_base, rights, patches);
        raw_limb_set(destination, first, limbs, 2, destination_binding_base, destinations, patches);
        for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kMaxGridY)
        {
            const size_t poly_chunk = std::min(kMaxGridY, poly_count - poly_offset);
            const dim3 grid(
                static_cast<uint32_t>((left->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(poly_chunk), static_cast<uint32_t>(limbs));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_add_sub_kernel,
                grid, dim3(kTransformThreads), 0, patches.data(), patches.size(),
                lefts, rights, destinations, left->columns,
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
    if (mxx_set_device(source->physical_device) != cudaSuccess)
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
    if (mxx_set_device(source->physical_device) != cudaSuccess)
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
    if (mxx_set_device(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t first = 0; first < left->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, left->limb_count - first);
        RawLimbSet lefts{}, rights{}, destinations{};
        std::vector<MxxGraphPatch> patches;
        raw_limb_set(left, first, limbs, 0, left_binding_base, lefts, patches);
        raw_limb_set(right, first, limbs, 1, right_binding_base, rights, patches);
        raw_limb_set(destination, first, limbs, 2, destination_binding_base, destinations, patches);
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kMulCoefficients - 1) / kMulCoefficients),
                static_cast<uint32_t>(chunk), static_cast<uint32_t>(limbs));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_mul_kernel,
                grid, dim3(kMulCoefficients, kMulSlices), 0, patches.data(), patches.size(),
                lefts, rights, destinations,
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

extern "C" int gpu_raw_matrix_mul_scalar(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *matrix, const MxxRawMatrixView *scalar,
    const MxxRawMatrixView *destination, uint32_t matrix_binding_base,
    uint32_t scalar_binding_base, uint32_t destination_binding_base)
{
    if (validate_raw_view(ctx, matrix, stream_raw) != 0 ||
        validate_raw_view(ctx, scalar, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !same_raw_extent(matrix, destination) ||
        scalar->rows != 1 || scalar->columns != 1 ||
        matrix->physical_device != scalar->physical_device ||
        matrix->degree != scalar->degree || matrix->limb_count != scalar->limb_count ||
        matrix_binding_base > UINT32_MAX - matrix->limb_count ||
        scalar_binding_base > UINT32_MAX - scalar->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw scalar product views");
    for (size_t limb = 0; limb < matrix->limb_count; ++limb)
        if (matrix->limbs[limb].crt_limb_index != scalar->limbs[limb].crt_limb_index ||
            matrix->limbs[limb].modulus != scalar->limbs[limb].modulus ||
            scalar->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw scalar product limb mismatch or alias");
    if (mxx_set_device(matrix->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t limb = 0; limb < matrix->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                static_cast<uint32_t>(matrix_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0, sizeof(void *),
                static_cast<uint32_t>(scalar_binding_base + limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2, 0, sizeof(void *),
                static_cast<uint32_t>(destination_binding_base + limb), 0},
        };
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_mul_scalar_kernel,
                grid, dim3(kTransformThreads), 0, patches, 3,
                matrix->limbs[limb], scalar->limbs[limb], destination->limbs[limb],
                static_cast<size_t>(destination->columns),
                static_cast<size_t>(destination->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
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
    if (mxx_set_device(source->physical_device) != cudaSuccess)
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
    if (mxx_set_device(left->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = destination->rows * destination->columns;
    for (size_t first = 0; first < left->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, left->limb_count - first);
        RawLimbSet lefts{}, rights{}, destinations{};
        std::vector<MxxGraphPatch> patches;
        raw_limb_set(left, first, limbs, 0, left_binding_base, lefts, patches);
        raw_limb_set(right, first, limbs, 1, right_binding_base, rights, patches);
        raw_limb_set(destination, first, limbs, 2, destination_binding_base, destinations, patches);
        for (size_t offset = 0; offset < poly_count; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, poly_count - offset);
            const dim3 grid(
                static_cast<uint32_t>((destination->degree + kTransformThreads - 1) / kTransformThreads),
                static_cast<uint32_t>(chunk), static_cast<uint32_t>(limbs));
            const int status = mxx_gpu_launch_kernel(ctx, stream, raw_matrix_tensor_kernel,
                grid, dim3(kTransformThreads), 0, patches.data(), patches.size(),
                lefts, rights, destinations,
                left->columns, right->rows, right->columns, destination->columns,
                static_cast<size_t>(destination->degree), offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_copy(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *sources, const MxxRawMatrixView *destinations, size_t count,
    const uint32_t *source_binding_bases, const uint32_t *destination_binding_bases)
{
    if (!sources || !destinations || !count || !source_binding_bases ||
        !destination_binding_bases)
        return set_error("invalid raw matrix copy batch");
    // Every (window, limb) pair is one entry; entries of any window share
    // one launch per batch.
    std::vector<std::pair<size_t, size_t>> entries;
    size_t max_polys = 0;
    for (size_t pair = 0; pair < count; ++pair)
    {
        const auto *source = sources + pair;
        const auto *destination = destinations + pair;
        if (validate_raw_view(ctx, source, stream_raw) != 0 ||
            validate_raw_view(ctx, destination, stream_raw) != 0 ||
            source->physical_device != sources->physical_device ||
            source->physical_device != destination->physical_device ||
            source->degree != destination->degree || source->degree != sources->degree ||
            source->rows != destination->rows ||
            source->columns != destination->columns ||
            source->limb_count != destination->limb_count ||
            source_binding_bases[pair] > UINT32_MAX - source->limb_count ||
            destination_binding_bases[pair] > UINT32_MAX - destination->limb_count)
            return set_error("invalid raw matrix copy views");
        for (size_t limb = 0; limb < source->limb_count; ++limb)
        {
            if (source->limbs[limb].crt_limb_index !=
                    destination->limbs[limb].crt_limb_index ||
                source->limbs[limb].modulus != destination->limbs[limb].modulus)
                return set_error("raw matrix copy CRT basis mismatch");
            entries.emplace_back(pair, limb);
        }
        max_polys = std::max(max_polys, static_cast<size_t>(source->rows * source->columns));
    }
    if (mxx_set_device(sources->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t degree = sources->degree;
    for (size_t first = 0; first < entries.size(); first += kRawNttLimbs)
    {
        const size_t chunk_entries = std::min(kRawNttLimbs, entries.size() - first);
        RawCopyBatch batch{};
        std::vector<MxxGraphPatch> patches;
        for (size_t local = 0; local < chunk_entries; ++local)
        {
            const auto [pair, limb] = entries[first + local];
            batch.source[local] = sources[pair].limbs[limb];
            batch.destination[local] = destinations[pair].limbs[limb];
            batch.columns[local] = sources[pair].columns;
            batch.polys[local] = sources[pair].rows * sources[pair].columns;
            patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(RawCopyBatch, source) +
                    local * sizeof(MxxRawMatrixLimb) + offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), source_binding_bases[pair] + static_cast<uint32_t>(limb), 0});
            patches.push_back({nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0,
                static_cast<uint32_t>(offsetof(RawCopyBatch, destination) +
                    local * sizeof(MxxRawMatrixLimb) + offsetof(MxxRawMatrixLimb, address)),
                sizeof(void *), destination_binding_bases[pair] + static_cast<uint32_t>(limb), 0});
        }
        for (size_t offset = 0; offset < max_polys; offset += kMaxGridY)
        {
            const size_t chunk = std::min(kMaxGridY, max_polys - offset);
            const dim3 grid(static_cast<uint32_t>((degree + kTransformThreads - 1) /
                kTransformThreads), static_cast<uint32_t>(chunk),
                static_cast<uint32_t>(chunk_entries));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_copy_batch_kernel, grid, dim3(kTransformThreads), 0,
                patches.data(), patches.size(), batch, degree, offset);
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
    if (mxx_set_device(destination->physical_device) != cudaSuccess)
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
