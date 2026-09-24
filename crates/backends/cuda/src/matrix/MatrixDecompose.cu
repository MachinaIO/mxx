

namespace
{
    constexpr uint32_t kDecomposeThreads = 256;
    constexpr size_t kDecomposeMaxGridY = 65535;

    MxxGraphPatch decompose_pointer_patch(
        uint32_t argument_index,
        size_t byte_offset,
        uint32_t binding_index,
        size_t byte_count = sizeof(void *))
    {
        MxxGraphPatch patch{};
        patch.node = nullptr;
        patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
        patch.argument_index = argument_index;
        patch.byte_offset = static_cast<uint32_t>(byte_offset);
        patch.byte_count = static_cast<uint32_t>(byte_count);
        patch.binding_index = binding_index;
        patch.address_addend = 0;
        return patch;
    }
}

__device__ __forceinline__ int64_t centered_lift_u64(uint64_t residue, uint64_t modulus)
{
    __int128 value = static_cast<__int128>(residue);
    if (static_cast<unsigned __int128>(residue) * 2U > static_cast<unsigned __int128>(modulus))
    {
        value -= static_cast<__int128>(modulus);
    }
    return static_cast<int64_t>(value);
}

__device__ __forceinline__ int64_t balanced_digit_step(int64_t value, int64_t base, int64_t *next)
{
    int64_t quotient = value / base;
    int64_t remainder = value % base;
    if (remainder < 0)
    {
        remainder += base;
        quotient -= 1;
    }
    const int64_t half = base / 2;
    if (remainder < half)
    {
        *next = quotient;
        return remainder;
    }
    if (remainder > half)
    {
        *next = quotient + 1;
        return remainder - base;
    }
    if ((quotient & 1LL) == 0)
    {
        *next = quotient;
        return half;
    }
    *next = quotient + 1;
    return half - base;
}

__device__ __forceinline__ uint64_t signed_digit_to_residue(int64_t digit, uint64_t modulus)
{
    if (digit >= 0)
    {
        return static_cast<uint64_t>(digit) % modulus;
    }
    const uint64_t magnitude = static_cast<uint64_t>(-digit) % modulus;
    return magnitude == 0 ? 0 : modulus - magnitude;
}

namespace
{
    // One launch covers up to this many (source limb, output limb) pairs;
    // blockIdx.z picks the pair.
    constexpr size_t kDecomposePairs = 8;

    struct RawDecomposeBatch
    {
        MxxRawMatrixLimb source[kDecomposePairs];
        MxxRawMatrixLimb destination[kDecomposePairs];
        uint64_t source_digit_offset[kDecomposePairs];
    };

    // Each thread peels every balanced digit of one coefficient in order and
    // stores digit `d` to output row `source_row * digits_per_row + offset + d`.
    __global__ void raw_matrix_decompose_coeff_kernel(
        RawDecomposeBatch batch, size_t source_columns, size_t destination_columns,
        size_t degree, size_t output_digits_per_row, uint32_t base_bits,
        uint32_t digits_per_tower, size_t poly_offset)
    {
        const size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (coefficient >= degree) return;
        const MxxRawMatrixLimb &source = batch.source[blockIdx.z];
        const MxxRawMatrixLimb &destination = batch.destination[blockIdx.z];
        const size_t poly = poly_offset + blockIdx.y;
        const uint64_t residue = raw_matrix_load(source, poly, coefficient, source_columns);
        int64_t value = centered_lift_u64(residue, source.modulus);
        const int64_t base = int64_t{1} << base_bits;
        const size_t source_row = poly / source_columns;
        const size_t source_column = poly - source_row * source_columns;
        const size_t first_row = source_row * output_digits_per_row +
            batch.source_digit_offset[blockIdx.z];
        for (uint32_t digit_index = 0; digit_index < digits_per_tower; ++digit_index)
        {
            int64_t next = 0;
            const int64_t signed_digit = balanced_digit_step(value, base, &next);
            value = next;
            raw_matrix_store(destination,
                (first_row + digit_index) * destination_columns + source_column,
                coefficient, destination_columns,
                signed_digit_to_residue(signed_digit, destination.modulus));
        }
    }
}

static int raw_matrix_decompose_coeff_impl(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t base_bits, size_t dropped_moduli,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    bool full_basis_small)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->columns != destination->columns ||
        source->column_origin != destination->column_origin ||
        source->limb_count != destination->limb_count ||
        dropped_moduli >= source->limb_count ||
        base_bits == 0 || base_bits >= 63 ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw coefficient decomposition views");
    if (full_basis_small &&
        (dropped_moduli != 0 || source->limb_count != ctx->moduli.size()))
        return set_error("small balanced decomposition requires the full CRT basis");
    // The small gadget has one shared digit row per source row. Each CRT limb
    // independently carries its own balanced digit in that row. The ordinary
    // gadget concatenates digit rows from the retained source limbs.
    const size_t retained = full_basis_small ? 1 : source->limb_count - dropped_moduli;
    uint32_t max_bits = 0;
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        if (source->limbs[limb].crt_limb_index != destination->limbs[limb].crt_limb_index ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus ||
            (full_basis_small &&
                (source->limbs[limb].crt_limb_index != limb ||
                    source->limbs[limb].modulus != ctx->moduli[limb])))
            return set_error("raw coefficient decomposition CRT basis mismatch");
        max_bits = std::max(max_bits, bit_width_u64(source->limbs[limb].modulus));
    }
    const size_t digits = (max_bits + base_bits - 1) / base_bits;
    if (!digits || retained > SIZE_MAX / digits ||
        source->rows > SIZE_MAX / (digits * retained) ||
        source->row_origin > SIZE_MAX / (digits * retained) ||
        source->rows > SIZE_MAX / source->columns ||
        destination->rows != source->rows * digits * retained ||
        destination->row_origin != source->row_origin * digits * retained)
        return set_error("raw coefficient decomposition output shape mismatch");
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t poly_count = source->rows * source->columns;
    struct Pair { size_t source_limb, output_limb, digit_offset; };
    std::vector<Pair> pairs;
    for (size_t input_limb = 0; input_limb < retained; ++input_limb)
        for (size_t output_limb = 0; output_limb < destination->limb_count; ++output_limb)
            pairs.push_back({full_basis_small ? output_limb : input_limb, output_limb,
                full_basis_small ? 0 : input_limb * digits});
    for (size_t first = 0; first < pairs.size(); first += kDecomposePairs)
    {
        const size_t count = std::min(kDecomposePairs, pairs.size() - first);
        RawDecomposeBatch batch{};
        std::vector<MxxGraphPatch> patches;
        for (size_t local = 0; local < count; ++local)
        {
            const Pair &pair = pairs[first + local];
            batch.source[local] = source->limbs[pair.source_limb];
            batch.destination[local] = destination->limbs[pair.output_limb];
            batch.source_digit_offset[local] = pair.digit_offset;
            patches.push_back(decompose_pointer_patch(0,
                offsetof(RawDecomposeBatch, source) + local * sizeof(MxxRawMatrixLimb) +
                    offsetof(MxxRawMatrixLimb, address),
                static_cast<uint32_t>(source_binding_base + pair.source_limb)));
            patches.push_back(decompose_pointer_patch(0,
                offsetof(RawDecomposeBatch, destination) + local * sizeof(MxxRawMatrixLimb) +
                    offsetof(MxxRawMatrixLimb, address),
                static_cast<uint32_t>(destination_binding_base + pair.output_limb)));
        }
        for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kDecomposeMaxGridY)
        {
            const size_t poly_chunk = std::min(kDecomposeMaxGridY, poly_count - poly_offset);
            const dim3 grid(
                static_cast<uint32_t>((source->degree + kDecomposeThreads - 1) / kDecomposeThreads),
                static_cast<uint32_t>(poly_chunk), static_cast<uint32_t>(count));
            const int status = mxx_gpu_launch_kernel(ctx, stream,
                raw_matrix_decompose_coeff_kernel, grid, dim3(kDecomposeThreads), 0,
                patches.data(), patches.size(), batch, source->columns, destination->columns,
                static_cast<size_t>(source->degree), digits * retained, base_bits,
                static_cast<uint32_t>(digits), poly_offset);
            if (status != 0) return status;
        }
    }
    return 0;
}

extern "C" int gpu_raw_matrix_decompose_coeff(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t base_bits, size_t dropped_moduli,
    uint32_t source_binding_base, uint32_t destination_binding_base)
{
    return raw_matrix_decompose_coeff_impl(ctx, stream_raw, source, destination,
        base_bits, dropped_moduli, source_binding_base,
        destination_binding_base, false);
}

extern "C" int gpu_raw_matrix_decompose_small_balanced(
    GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    uint32_t base_bits, uint32_t source_binding_base,
    uint32_t destination_binding_base)
{
    return raw_matrix_decompose_coeff_impl(ctx, stream_raw, source, destination,
        base_bits, 0, source_binding_base, destination_binding_base, true);
}
