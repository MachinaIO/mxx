// Direct physical-view emitters for scalar polynomial operations. This file is
// included after MatrixNTT.cu, which owns the raw matrix load/store helpers.
namespace
{
    constexpr uint32_t kRawScalarInvalid = 2;

    __device__ bool raw_positive_index(const uint64_t *value, int encoding,
        uint64_t exclusive_limit, uint64_t *index)
    {
        if (encoding == 0)
        {
            const int64_t signed_value = static_cast<int64_t>(value[0]);
            if (signed_value <= 0) return false;
            *index = static_cast<uint64_t>(signed_value);
        }
        else if (encoding == 1)
            *index = value[0];
        else if (encoding > 2)
        {
            if (value[0] != 0) return false;
            for (int word = encoding - 2; word > 1; --word)
                if (value[word] != 0) return false;
            *index = value[1];
        }
        else return false;
        return *index != 0 && *index < exclusive_limit && (*index & 1U) != 0;
    }

    __device__ bool raw_integer_modulus_residue(const uint64_t *value,
        int encoding, uint64_t modulus, uint64_t *residue)
    {
        *residue = 0;
        if (encoding == 0)
        {
            const int64_t signed_value = static_cast<int64_t>(value[0]);
            const bool negative = signed_value < 0;
            const uint64_t magnitude = negative ? 0ULL - value[0] : value[0];
            *residue = magnitude % modulus;
            if (negative && *residue != 0) *residue = modulus - *residue;
            return true;
        }
        if (encoding == 1)
        {
            *residue = value[0] % modulus;
            return true;
        }
        if (encoding <= 2 || value[0] > 1) return false;
        const bool negative = value[0] != 0;
        bool zero = true;
        for (int word = encoding - 2; word > 0; --word)
        {
            zero = zero && value[word] == 0;
            *residue = static_cast<uint64_t>(
                ((static_cast<unsigned __int128>(*residue) << 64) | value[word]) % modulus);
        }
        if (negative && zero) return false;
        if (negative && *residue != 0) *residue = modulus - *residue;
        return true;
    }

    __device__ bool raw_nonnegative_extent(const uint64_t *value,
        int encoding, uint64_t limit, uint64_t *result)
    {
        if (encoding == 0)
        {
            const int64_t signed_value = static_cast<int64_t>(value[0]);
            if (signed_value < 0) return false;
            *result = static_cast<uint64_t>(signed_value);
        }
        else if (encoding == 1)
            *result = value[0];
        else if (encoding > 2)
        {
            if (value[0] != 0) return false;
            for (int word = encoding - 2; word > 1; --word)
                if (value[word] != 0) return false;
            *result = value[1];
        }
        else return false;
        return *result <= limit;
    }

    __global__ void raw_dynamic_slice_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        const uint64_t *row_start_value, int row_start_encoding,
        const uint64_t *row_end_value, int row_end_encoding,
        const uint64_t *column_start_value, int column_start_encoding,
        const uint64_t *column_end_value, int column_end_encoding,
        uint32_t *status, size_t source_rows, size_t source_columns,
        size_t destination_rows, size_t destination_columns, size_t degree)
    {
        __shared__ uint64_t row_start;
        __shared__ uint64_t row_end;
        __shared__ uint64_t column_start;
        __shared__ uint64_t column_end;
        __shared__ bool valid;
        if (threadIdx.x == 0)
        {
            valid = raw_nonnegative_extent(row_start_value, row_start_encoding,
                    source_rows, &row_start) &&
                raw_nonnegative_extent(row_end_value, row_end_encoding,
                    source_rows, &row_end) &&
                raw_nonnegative_extent(column_start_value, column_start_encoding,
                    source_columns, &column_start) &&
                raw_nonnegative_extent(column_end_value, column_end_encoding,
                    source_columns, &column_end) &&
                row_end >= row_start && column_end >= column_start &&
                row_end - row_start == destination_rows &&
                column_end - column_start == destination_columns;
            if (!valid) atomicCAS(status, 0U, kRawScalarInvalid);
        }
        __syncthreads();
        if (!valid) return;
        const size_t count = destination_rows * destination_columns * degree;
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count; index += step)
        {
            const size_t coefficient = index % degree;
            const size_t output_poly = index / degree;
            const size_t row = output_poly / destination_columns;
            const size_t column = output_poly % destination_columns;
            const size_t source_poly = (row_start + row) * source_columns +
                column_start + column;
            const uint64_t residue = raw_matrix_load(source, source_poly,
                coefficient, source_columns);
            raw_matrix_store(destination, output_poly, coefficient,
                destination_columns, residue);
        }
    }

    __global__ void raw_ring_automorphism_kernel(
        MxxRawMatrixLimb source, MxxRawMatrixLimb destination,
        const uint64_t *index_value, int index_encoding, uint32_t *status,
        size_t columns, size_t degree, size_t polynomial_count)
    {
        __shared__ uint64_t index;
        __shared__ bool valid;
        if (threadIdx.x == 0)
        {
            valid = raw_positive_index(index_value, index_encoding,
                2ULL * degree, &index);
            if (!valid) atomicCAS(status, 0U, kRawScalarInvalid);
        }
        __syncthreads();
        if (!valid) return;
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             item < polynomial_count * degree; item += step)
        {
            const size_t poly = item / degree;
            const size_t coefficient = item % degree;
            const uint64_t exponent = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(coefficient) * index) % (2ULL * degree));
            const size_t target = exponent < degree ? exponent : exponent - degree;
            uint64_t residue = raw_matrix_load(source, poly, coefficient, columns);
            if (exponent >= degree && residue != 0)
                residue = destination.modulus - residue;
            raw_matrix_store(destination, poly, target, columns, residue);
        }
    }

    // Multiplying by X^k in the evaluation domain scales slot s by X^k's
    // value there. The forward NTT twists by psi^i and leaves slot s in
    // bit-reversed order, so X evaluates to psi^(2 rev(s) + 1) and X^k to
    // psi^(k (2 rev(s) + 1) mod 2n), where psi^(n + t) = -psi^t. One launch
    // covers up to kRawNttLimbs CRT limbs; blockIdx.y picks the limb. With
    // `subtract_source` it writes X^k a - a, the CMUX difference, in the
    // same pass.
    __global__ void raw_monomial_multiply_kernel(RawNttBatch batch,
        const uint64_t *exponent_value, int exponent_encoding, uint32_t *status,
        size_t columns, size_t degree, uint32_t log_degree, size_t polynomial_count,
        bool subtract_source)
    {
        const MxxRawMatrixLimb &source = batch.source[blockIdx.y];
        const MxxRawMatrixLimb &destination = batch.destination[blockIdx.y];
        const uint64_t *twiddles = batch.twiddles[blockIdx.y];
        const uint64_t *shoup = batch.shoup[blockIdx.y];
        __shared__ uint64_t exponent;
        __shared__ bool valid;
        if (threadIdx.x == 0)
        {
            valid = raw_integer_modulus_residue(exponent_value, exponent_encoding,
                2ULL * degree, &exponent);
            if (!valid) atomicCAS(status, 0U, kRawScalarInvalid);
        }
        __syncthreads();
        if (!valid) return;
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             item < polynomial_count * degree; item += step)
        {
            const size_t poly = item / degree;
            const uint32_t slot = static_cast<uint32_t>(item % degree);
            const uint64_t reversed = log_degree ? __brev(slot) >> (32 - log_degree) : 0;
            // The degree is a power of two, so the reduction mod 2n is a mask.
            const uint64_t power = (exponent * (2 * reversed + 1)) & (2ULL * degree - 1);
            const size_t index = power < degree ? power : power - degree;
            const uint64_t input = raw_matrix_load(source, poly, slot, columns);
            uint64_t value =
                mul_mod_shoup_u64(input, twiddles[index], shoup[index], destination.modulus);
            if (power >= degree && value != 0) value = destination.modulus - value;
            if (subtract_source) value = sub_mod_u64(value, input, destination.modulus);
            raw_matrix_store(destination, poly, slot, columns, value);
        }
    }

    __global__ void raw_lift_integer_kernel(const uint64_t *value, int encoding,
        MxxRawMatrixLimb destination, uint32_t *status, uint32_t degree)
    {
        __shared__ uint64_t residue;
        __shared__ bool valid;
        if (threadIdx.x == 0)
        {
            valid = raw_integer_modulus_residue(value, encoding,
                destination.modulus, &residue);
            if (!valid) atomicCAS(status, 0U, kRawScalarInvalid);
        }
        __syncthreads();
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             coefficient < degree; coefficient += step)
            raw_matrix_store(destination, 0, coefficient, 1,
                valid && coefficient == 0 ? residue : 0);
    }

    MxxGraphPatch raw_remaining_patch(uint32_t argument, uint32_t offset,
        uint32_t binding)
    {
        return {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, argument,
            offset, static_cast<uint32_t>(sizeof(void *)), binding, 0};
    }
}

extern "C" int gpu_raw_ring_automorphism(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *index_value, int index_encoding, uint32_t *status,
    uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t index_binding, uint32_t status_binding)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !index_value || !status || index_encoding < 0 || index_encoding == 2 ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->rows != destination->rows ||
        source->columns != destination->columns ||
        source->limb_count != destination->limb_count ||
        source->degree > UINT64_MAX / 2 ||
        source->rows > SIZE_MAX / source->columns ||
        source->rows * source->columns > SIZE_MAX / source->degree ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw ring automorphism views");
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        if (source->limbs[limb].crt_limb_index != limb ||
            destination->limbs[limb].crt_limb_index != limb ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus ||
            source->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw ring automorphism requires distinct ordered CRT views");
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const size_t count = source->rows * source->columns * source->degree;
    const uint32_t grid = static_cast<uint32_t>(std::min<size_t>((count + 255) / 256, 65535));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            raw_remaining_patch(0, offsetof(MxxRawMatrixLimb, address),
                source_binding_base + limb),
            raw_remaining_patch(1, offsetof(MxxRawMatrixLimb, address),
                destination_binding_base + limb),
            raw_remaining_patch(2, 0, index_binding),
            raw_remaining_patch(4, 0, status_binding),
        };
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_ring_automorphism_kernel, dim3(grid), dim3(256), 0,
            patches, std::size(patches), source->limbs[limb],
            destination->limbs[limb], index_value, index_encoding, status,
            static_cast<size_t>(source->columns),
            static_cast<size_t>(source->degree),
            static_cast<size_t>(source->rows * source->columns));
        if (result != 0) return result;
    }
    return 0;
}

extern "C" int gpu_raw_monomial_multiply(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *exponent_value, int exponent_encoding, uint32_t *status,
    int subtract_source, uint32_t source_binding_base, uint32_t destination_binding_base,
    uint32_t exponent_binding, uint32_t status_binding)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !exponent_value || !status || exponent_encoding < 0 || exponent_encoding == 2 ||
        (subtract_source != 0 && subtract_source != 1) ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->rows != destination->rows ||
        source->columns != destination->columns ||
        source->limb_count != destination->limb_count ||
        (source->degree & (source->degree - 1)) != 0 ||
        source->rows > SIZE_MAX / source->columns ||
        source->rows * source->columns > SIZE_MAX / source->degree ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw monomial multiplication views");
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const size_t degree = source->degree;
    const uint32_t log_degree = static_cast<uint32_t>(__builtin_ctzll(degree));
    const size_t count = source->rows * source->columns * degree;
    const uint32_t grid = static_cast<uint32_t>(std::min<size_t>((count + 255) / 256, 65535));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t first = 0; first < source->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, source->limb_count - first);
        RawNttBatch batch{};
        std::vector<MxxGraphPatch> patches;
        for (size_t local = 0; local < limbs; ++local)
        {
            const size_t limb = first + local;
            const auto &source_limb = source->limbs[limb];
            const auto &destination_limb = destination->limbs[limb];
            if (source_limb.crt_limb_index != destination_limb.crt_limb_index ||
                source_limb.modulus != destination_limb.modulus)
                return set_error("raw monomial multiplication requires matching CRT views");
            const dim3 partition = ctx->limb_gpu_ids[source_limb.crt_limb_index];
            if (partition.x >= ctx->ntt_device_constants.size())
                return set_error("missing raw monomial NTT constants");
            const auto &constants = ctx->ntt_device_constants[partition.x];
            if (constants.device != source->physical_device ||
                partition.y >= constants.limb_count || constants.ring_dimension != degree ||
                !constants.twiddle_forward || !constants.twiddle_shoup_forward)
                return set_error("invalid raw monomial NTT constants");
            const size_t table = static_cast<size_t>(partition.y) * degree;
            batch.source[local] = source_limb;
            batch.destination[local] = destination_limb;
            batch.twiddles[local] = constants.twiddle_forward + table;
            batch.shoup[local] = constants.twiddle_shoup_forward + table;
            patches.push_back(raw_remaining_patch(0,
                static_cast<uint32_t>(offsetof(RawNttBatch, source) +
                    local * sizeof(MxxRawMatrixLimb) + offsetof(MxxRawMatrixLimb, address)),
                static_cast<uint32_t>(source_binding_base + limb)));
            patches.push_back(raw_remaining_patch(0,
                static_cast<uint32_t>(offsetof(RawNttBatch, destination) +
                    local * sizeof(MxxRawMatrixLimb) + offsetof(MxxRawMatrixLimb, address)),
                static_cast<uint32_t>(destination_binding_base + limb)));
        }
        patches.push_back(raw_remaining_patch(1, 0, exponent_binding));
        patches.push_back(raw_remaining_patch(3, 0, status_binding));
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_monomial_multiply_kernel, dim3(grid, static_cast<uint32_t>(limbs)), dim3(256), 0,
            patches.data(), patches.size(), batch, exponent_value, exponent_encoding, status,
            static_cast<size_t>(source->columns), degree, log_degree,
            static_cast<size_t>(source->rows * source->columns), subtract_source != 0);
        if (result != 0) return result;
    }
    return 0;
}

extern "C" int gpu_raw_lift_integer_constant(GpuContext *ctx, void *stream_raw,
    const void *value, int value_encoding,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t value_binding, uint32_t destination_binding_base,
    uint32_t status_binding)
{
    if (!value || !status || value_encoding < 0 || value_encoding == 2 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        destination->rows != 1 || destination->columns != 1 ||
        destination->limb_count != ctx->moduli.size() ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw integer-to-polynomial view");
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
        if (destination->limbs[limb].crt_limb_index != limb ||
            destination->limbs[limb].modulus != ctx->moduli[limb])
            return set_error("raw integer-to-polynomial requires ordered CRT basis");
    if (mxx_set_device(destination->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const uint32_t grid = std::min<uint32_t>((destination->degree + 255) / 256, 65535);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            raw_remaining_patch(0, 0, value_binding),
            raw_remaining_patch(2, offsetof(MxxRawMatrixLimb, address),
                destination_binding_base + limb),
            raw_remaining_patch(3, 0, status_binding),
        };
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_lift_integer_kernel, dim3(grid), dim3(256), 0,
            patches, std::size(patches), value, value_encoding,
            destination->limbs[limb], status, destination->degree);
        if (result != 0) return result;
    }
    return 0;
}

extern "C" int gpu_raw_matrix_dynamic_slice(GpuContext *ctx, void *stream_raw,
    const MxxRawMatrixView *source, const MxxRawMatrixView *destination,
    const void *row_start, int row_start_encoding,
    const void *row_end, int row_end_encoding,
    const void *column_start, int column_start_encoding,
    const void *column_end, int column_end_encoding,
    uint32_t *status, uint32_t source_binding_base,
    uint32_t destination_binding_base, uint32_t row_start_binding,
    uint32_t row_end_binding, uint32_t column_start_binding,
    uint32_t column_end_binding, uint32_t status_binding)
{
    if (validate_raw_view(ctx, source, stream_raw) != 0 ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        !row_start || !row_end || !column_start || !column_end || !status ||
        row_start_encoding < 0 || row_start_encoding == 2 ||
        row_end_encoding < 0 || row_end_encoding == 2 ||
        column_start_encoding < 0 || column_start_encoding == 2 ||
        column_end_encoding < 0 || column_end_encoding == 2 ||
        source->physical_device != destination->physical_device ||
        source->degree != destination->degree ||
        source->limb_count != destination->limb_count ||
        source->row_origin != 0 || source->column_origin != 0 ||
        destination->row_origin != 0 || destination->column_origin != 0 ||
        !source->rows || !source->columns ||
        !destination->rows || !destination->columns ||
        source->rows > SIZE_MAX / source->columns ||
        destination->rows > source->rows ||
        destination->columns > source->columns ||
        destination->rows > SIZE_MAX / destination->columns ||
        destination->rows * destination->columns > SIZE_MAX / destination->degree ||
        source_binding_base > UINT32_MAX - source->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw dynamic-slice physical views");
    for (size_t limb = 0; limb < source->limb_count; ++limb)
        if (source->limbs[limb].crt_limb_index != limb ||
            destination->limbs[limb].crt_limb_index != limb ||
            source->limbs[limb].modulus != destination->limbs[limb].modulus ||
            source->limbs[limb].address == destination->limbs[limb].address)
            return set_error("raw dynamic slice requires distinct ordered CRT views");
    if (mxx_set_device(source->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const size_t count = destination->rows * destination->columns * destination->degree;
    const uint32_t grid = static_cast<uint32_t>(
        std::min<size_t>((count + 255) / 256, 65535));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t limb = 0; limb < source->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            raw_remaining_patch(0, offsetof(MxxRawMatrixLimb, address),
                source_binding_base + static_cast<uint32_t>(limb)),
            raw_remaining_patch(1, offsetof(MxxRawMatrixLimb, address),
                destination_binding_base + static_cast<uint32_t>(limb)),
            raw_remaining_patch(2, 0, row_start_binding),
            raw_remaining_patch(4, 0, row_end_binding),
            raw_remaining_patch(6, 0, column_start_binding),
            raw_remaining_patch(8, 0, column_end_binding),
            raw_remaining_patch(10, 0, status_binding),
        };
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_dynamic_slice_kernel, dim3(grid), dim3(256), 0,
            patches, std::size(patches), source->limbs[limb],
            destination->limbs[limb], row_start, row_start_encoding,
            row_end, row_end_encoding, column_start, column_start_encoding,
            column_end, column_end_encoding, status,
            static_cast<size_t>(source->rows),
            static_cast<size_t>(source->columns),
            static_cast<size_t>(destination->rows),
            static_cast<size_t>(destination->columns),
            static_cast<size_t>(source->degree));
        if (result != 0) return result;
    }
    return 0;
}
