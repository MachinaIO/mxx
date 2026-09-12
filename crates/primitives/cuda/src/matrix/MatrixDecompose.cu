MatrixTransformWorkspace::MatrixTransformWorkspace()
    : base(nullptr), pinned(nullptr), owner(nullptr), device(-1), stream(nullptr),
      bytes(0), separate(false), completed(false)
{
}

int MatrixTransformWorkspace::acquire(
    GpuMatrix *output, int selected_device, cudaStream_t selected_stream,
    const GpuMatrixTransformWorkspaceBytes &requirements)
{
    if (owner || !output || !output->ctx || !selected_stream ||
        output->shared_aux_buffers.size() != 1 ||
        output->shared_aux_buffers[0].device != selected_device ||
        (requirements.pinned_bytes != 0 && requirements.pinned_bytes != requirements.workspace_bytes))
        return set_error("invalid matrix transform workspace owner");
    owner = output;
    device = selected_device;
    stream = selected_stream;
    bytes = requirements.workspace_bytes;
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(output, device, stream, true);
    if (status != 0) return status;
    const auto &aux = output->shared_aux_buffers[0];
    if (aux.slots_total > SIZE_MAX / sizeof(void *))
        return set_error("matrix transform auxiliary size overflow");
    separate = requirements.additional_bytes != 0;
    if ((separate && requirements.additional_bytes != bytes) ||
        (!separate && (bytes > aux.slots_total * sizeof(void *) || (bytes != 0 && !aux.ptr))))
        return set_error("matrix transform workspace plan mismatch");
    if (bytes != 0)
    {
        if (separate)
        {
            const int allocated = device_workspace.acquire(
                output->ctx, device, GPU_PREPARED_TRANSFORM_WORKSPACE,
                bytes, requirements.alignment, stream);
            if (allocated != 0) return allocated;
            base = device_workspace.data;
        }
        else base = reinterpret_cast<uint8_t *>(aux.ptr);
    }
    if (requirements.pinned_bytes != 0)
    {
        pinned = static_cast<uint8_t *>(gpu_pinned_alloc(output->ctx, bytes, requirements.alignment));
        if (!pinned) return 1;
        std::memset(pinned, 0, bytes);
    }
    return 0;
}

int MatrixTransformWorkspace::upload()
{
    if (!owner || !pinned || !base || bytes == 0)
        return set_error("missing matrix transform staging storage");
    const cudaError_t error = cudaMemcpyAsync(base, pinned, bytes, cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess ? 0 : set_error(error);
}

int MatrixTransformWorkspace::release()
{
    int status = 0;
    if (pinned)
    {
        void *pointer = pinned;
        pinned = nullptr;
        status = gpu_defer_pinned_frees(owner->ctx, device, stream, &pointer, 1);
        if (status != 0)
            gpu_execution_mark_allocation_unknown(owner->ctx->execution.get());
    }
    if (separate && base)
    {
        const int released = device_workspace.release();
        base = nullptr;
        if (status == 0) status = released;
    }
    return status;
}

int MatrixTransformWorkspace::complete()
{
    if (!owner || completed) return set_error("invalid matrix transform completion");
    const int status = release();
    if (status != 0) return status;
    // Final ownership dominates the metadata free and all auxiliary consumers.
    const int recorded = matrix_record_all_limb_writes(owner, stream);
    if (recorded != 0) return recorded;
    completed = true;
    return 0;
}

int MatrixTransformWorkspace::retire()
{
    if (!owner || completed) return 0;
    cudaError_t error = cudaSetDevice(device);
    int status = error == cudaSuccess ? release() : set_error(error);
    const int retired = gpu_context_retire_stream(owner->ctx, device, stream);
    if (status != 0 || retired != 0)
        gpu_execution_mark_allocation_unknown(owner->ctx->execution.get());
    completed = true;
    return status != 0 ? status : retired;
}

MatrixTransformWorkspace::~MatrixTransformWorkspace()
{
    retire();
}

extern "C" int gpu_matrix_query_gadget_correction_workspace_bytes(
    const GpuContext *ctx, int level, size_t rows, size_t cols,
    size_t dropped_moduli, GpuMatrixTransformWorkspaceBytes *out)
{
    if (!out || level < 0 || dropped_moduli > static_cast<size_t>(level))
        return set_error("invalid gadget correction workspace query");
    GpuMatrixAllocationBytes allocation{};
    // Keep the correction allocation class stable across column waves and
    // short tails. If one column needs separate metadata, every wider range
    // uses that same fixed span, even if its larger auxiliary area could fit it.
    const int status = gpu_matrix_query_allocation_bytes(
        ctx, level, rows, cols == 0 ? 0 : 1, GPU_POLY_FORMAT_COEFF, &allocation);
    if (status != 0) return status;
    const size_t limbs = static_cast<size_t>(level) + 1;
    if (limbs > GPU_RUNTIME_MAX_LIMBS) return set_error("unsupported gadget correction limb count");
    const size_t bytes = rows == 0 || cols == 0 ? 0 :
        dropped_moduli * (limbs - dropped_moduli + 1) * sizeof(uint64_t);
    *out = {bytes, bytes <= allocation.aux_workspace_bytes ? 0 : bytes, 0, alignof(uint64_t)};
    return 0;
}

extern "C" int gpu_matrix_query_decompose_workspace_bytes(
    const GpuContext *ctx, int level, size_t rows, size_t cols, int format,
    uint32_t base_bits, int small, size_t dropped_moduli,
    GpuMatrixDecomposeWorkspaceBytes *out)
{
    if (!ctx || !out || level < 0 || static_cast<size_t>(level) >= ctx->moduli.size() ||
        base_bits == 0 || base_bits > (small ? 64U : 62U) || (small != 0 && small != 1) ||
        (small && dropped_moduli != 0) || dropped_moduli > static_cast<size_t>(level))
        return set_error("invalid decomposition workspace query");
    GpuMatrixAllocationBytes source{};
    int status = gpu_matrix_query_allocation_bytes(ctx, level, rows, cols, format, &source);
    if (status != 0) return status;
    uint32_t crt_bits = 0;
    for (const auto modulus : ctx->moduli) crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    const size_t digits = (crt_bits + base_bits - 1) / base_bits;
    const size_t towers = small ? 1 : static_cast<size_t>(level) + 1 - dropped_moduli;
    if (digits == 0 || digits > SIZE_MAX / towers || rows > SIZE_MAX / (digits * towers))
        return set_error("decomposition output shape overflow");
    GpuMatrixDecomposeWorkspaceBytes plan{};
    plan.output_rows = rows * digits * towers;
    GpuMatrixAllocationBytes output{};
    status = gpu_matrix_query_allocation_bytes(
        ctx, level, plan.output_rows, cols, format, &output);
    if (status != 0) return status;
    if (rows != 0 && cols != 0 && (format == GPU_POLY_FORMAT_EVAL || dropped_moduli != 0))
        plan.coefficient_copy = source;
    status = gpu_matrix_query_gadget_correction_workspace_bytes(
        ctx, level, rows, cols, dropped_moduli, &plan.correction);
    if (status != 0) return status;
    *out = plan;
    return 0;
}

__device__ __forceinline__ uint64_t pow_mod_u64(uint64_t base, uint32_t exp, uint64_t modulus)
{
    if (modulus == 0)
    {
        return 0;
    }
    uint64_t result = 1ULL % modulus;
    uint64_t cur = base % modulus;
    uint32_t e = exp;
    while (e > 0)
    {
        if (e & 1U)
        {
            result = static_cast<uint64_t>((static_cast<unsigned __int128>(result) * cur) % modulus);
        }
        e >>= 1U;
        if (e > 0)
        {
            cur = static_cast<uint64_t>((static_cast<unsigned __int128>(cur) * cur) % modulus);
        }
    }
    return result;
}

namespace
{
    constexpr uint32_t kDecomposeThreads = 256;
    constexpr size_t kDecomposeMaxGridY = 65535;
    constexpr size_t kDecomposeMaxGridZ = 65535;
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

__global__ void matrix_decompose_all_slots_kernel(
    const uint8_t *src_base,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *src_descriptors,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *dst_descriptors,
    const uint64_t *dst_moduli,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    size_t out_limb_count,
    size_t slot_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint64_t src_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    bool balanced,
    size_t src_digit_offset_base,
    size_t poly_offset,
    size_t slot_offset)
{
    if (!src_base || !src_descriptors || !dst_descriptors || !dst_moduli)
    {
        return;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0 || digits_per_tower == 0)
    {
        return;
    }
    const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= n)
    {
        return;
    }
    const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
    if (poly_idx >= poly_count)
    {
        return;
    }
    const size_t slot_idx = slot_offset + static_cast<size_t>(blockIdx.z);
    if (slot_idx >= slot_count)
    {
        return;
    }
    const size_t out_limb = slot_idx / static_cast<size_t>(digits_per_tower);
    if (out_limb >= out_limb_count)
    {
        return;
    }
    const uint32_t digit_idx = static_cast<uint32_t>(slot_idx % static_cast<size_t>(digits_per_tower));

    const auto descriptor = dst_descriptors[out_limb];
    uint8_t *const dst_base = descriptor.base;
    const size_t dst_stride = descriptor.stride;
    const uint8_t dst_bytes = descriptor.width;
    const uint64_t out_modulus = dst_moduli[out_limb];
    if (!dst_base || dst_bytes == 0 || dst_stride < n * static_cast<size_t>(dst_bytes))
    {
        return;
    }

    uint64_t digit = 0;
    if (balanced)
    {
        const uint64_t residue =
            matrix_load_limb_u64(src_base, poly_idx, coeff_idx, src_stride_bytes, src_coeff_bytes);
        int64_t value = centered_lift_u64(residue, src_modulus);
        int64_t signed_digit = 0;
        const int64_t base = int64_t{1} << base_bits;
        for (uint32_t idx = 0; idx <= digit_idx; ++idx)
        {
            int64_t next = 0;
            const int64_t current_digit = balanced_digit_step(value, base, &next);
            if (idx == digit_idx)
            {
                signed_digit = current_digit;
            }
            value = next;
        }
        digit = signed_digit_to_residue(signed_digit, out_modulus);
    }
    else
    {
        // Unsigned small decomposition truncates each actual CRT residue.
        // Negative coefficients have different residues in different towers;
        // broadcasting the first tower's digits changes the represented value.
        const auto source = src_descriptors[out_limb];
        const uint64_t residue = matrix_load_limb_u64(
            source.base, poly_idx, coeff_idx, source.stride, source.width);
        const uint32_t shift = digit_idx * base_bits;
        const uint64_t mask = base_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << base_bits) - 1);
        digit = shift >= 64 ? 0 : ((residue >> shift) & mask);
        if (out_modulus != 0 && digit >= out_modulus)
        {
            digit %= out_modulus;
        }
    }

    const size_t row = poly_idx / src_cols;
    const size_t col = poly_idx - row * src_cols;
    const size_t out_row = row * log_base_q + src_digit_offset_base + static_cast<size_t>(digit_idx);
    const size_t out_poly_idx = out_row * out_cols + col;
    matrix_store_limb_u64(dst_base, out_poly_idx, coeff_idx, dst_stride, dst_bytes, digit);
}

int launch_decompose_all_slots_kernel(
    const uint8_t *src_base,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *src_descriptors,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *dst_descriptors,
    const uint64_t *dst_moduli,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint64_t src_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    bool balanced,
    size_t src_digit_offset_base,
    cudaStream_t stream)
{
    if (!src_base || !src_descriptors || !dst_descriptors || !dst_moduli)
    {
        return set_error("null pointer in matrix_decompose_all_slots_kernel");
    }
    if (out_limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_coeff_bytes == 0 || src_stride_bytes < n * static_cast<size_t>(src_coeff_bytes))
    {
        return set_error("invalid src stride in matrix_decompose_all_slots_kernel");
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return set_error("invalid matrix shape in matrix_decompose_all_slots_kernel");
    }
    if (digits_per_tower == 0)
    {
        return set_error("invalid digit count in matrix_decompose_all_slots_kernel");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_decompose_all_slots_kernel");
    }
    if (out_limb_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(digits_per_tower))
    {
        return set_error("slot count overflow in matrix_decompose_all_slots_kernel");
    }
    const size_t slot_count = out_limb_count * static_cast<size_t>(digits_per_tower);
    if (slot_count == 0)
    {
        return 0;
    }

    const uint32_t blocks_x = static_cast<uint32_t>((n + kDecomposeThreads - 1) / kDecomposeThreads);
    for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kDecomposeMaxGridY)
    {
        const size_t poly_chunk = std::min(kDecomposeMaxGridY, poly_count - poly_offset);
        for (size_t slot_offset = 0; slot_offset < slot_count; slot_offset += kDecomposeMaxGridZ)
        {
            const size_t slot_chunk = std::min(kDecomposeMaxGridZ, slot_count - slot_offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(poly_chunk),
                static_cast<uint32_t>(slot_chunk)};
            matrix_decompose_all_slots_kernel<<<grid, kDecomposeThreads, 0, stream>>>(
                src_base,
                src_descriptors,
                dst_descriptors,
                dst_moduli,
                src_stride_bytes,
                src_coeff_bytes,
                out_limb_count,
                slot_count,
                poly_count,
                n,
                src_cols,
                out_cols,
                log_base_q,
                src_modulus,
                base_bits,
                digits_per_tower,
                balanced,
                src_digit_offset_base,
                poly_offset,
                slot_offset);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
    }
    return 0;
}

namespace {
struct MatrixConstantMetadata {
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *output;
    unsigned indices[GPU_RUNTIME_MAX_LIMBS];
    uint64_t moduli[GPU_RUNTIME_MAX_LIMBS];
    size_t output_offset, output_pitch, columns, global_start, unit_index, columns_per_row;
    unsigned digits_per_tower, base_bits;
    int mode;
    bool small;
};
static_assert(sizeof(MatrixConstantMetadata) + 2 * sizeof(size_t) <= 4096,
              "constant metadata exceeds portable CUDA argument capacity");

__global__ void matrix_fill_constant_columns_kernel(
    const MatrixConstantMetadata metadata, size_t count, size_t n)
{
    const size_t limb = blockIdx.z;
    const auto output = metadata.output[metadata.indices[limb]];
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = index / n / metadata.columns;
        const size_t column = index / n % metadata.columns;
        const size_t global_column = metadata.global_start + column;
        uint64_t value = 0;
        if (metadata.mode == 1) value = row == global_column;
        else if (metadata.mode == 2) value = global_column == metadata.unit_index;
        else if (metadata.mode == 4) value = row == metadata.unit_index;
        else if (metadata.mode == 3) {
            const size_t block_start = row * metadata.columns_per_row;
            if (global_column >= block_start && global_column - block_start < metadata.columns_per_row) {
                const size_t digit = global_column - block_start;
                if (metadata.small || digit / metadata.digits_per_tower == limb) {
                    value = pow_mod_u64(uint64_t{1} << metadata.base_bits,
                        digit % metadata.digits_per_tower, metadata.moduli[limb]);
                }
            }
        }
        matrix_store_limb_u64(output.base,
            metadata.output_offset + row * metadata.output_pitch + column,
            index % n, output.stride, output.width, value);
    }
}
}

extern "C" int gpu_matrix_fill_constant_columns(
    GpuMatrix *out, const GpuMatrixRange *range, size_t global_column_start,
    int mode, size_t total_columns, size_t unit_index,
    uint32_t base_bits, int small, size_t dropped_moduli)
{
    if (!out || !out->ctx || !out->ctx->execution || !range || out->level < 0 ||
        out->ctx->N <= 0 || mode < 0 || mode > 4 || (small != 0 && small != 1) ||
        (mode != 0 && out->format != GPU_POLY_FORMAT_EVAL))
        return set_error("invalid constant destination, format or mode");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    const auto &v = *range;
    if (v.row_start > v.row_end || v.row_end > out->rows ||
        v.column_start > v.column_end || v.column_end > out->cols ||
        (out->cols != 0 && out->rows > SIZE_MAX / out->cols))
        return set_error("invalid constant destination range");
    const size_t rows = v.row_end - v.row_start, columns = v.column_end - v.column_start;
    const size_t limbs = static_cast<size_t>(out->level) + 1;
    auto *ctx = out->ctx;
    if (limbs > GPU_RUNTIME_MAX_LIMBS || ctx->moduli.size() < limbs || ctx->limb_gpu_ids.size() < limbs)
        return set_error("invalid constant limb basis");
    MatrixConstantMetadata metadata{};
    if (mode == 1 && total_columns != rows) return set_error("identity is not square");
    if (mode == 2 && (rows != 1 || unit_index >= total_columns)) return set_error("invalid unit row");
    if (mode == 4 && (total_columns != 1 || unit_index >= rows)) return set_error("invalid unit column");
    if (mode == 3) {
        if (base_bits == 0 || base_bits >= 63 || dropped_moduli >= ctx->moduli.size())
            return set_error("invalid gadget base or dropped moduli");
        unsigned bits = 0;
        // The logical gadget uses the full parameter basis even when an output
        // retains fewer active limbs. Active limbs are its exact projection.
        for (const uint64_t modulus : ctx->moduli) bits = std::max(bits, bit_width_u64(modulus));
        metadata.digits_per_tower = (bits + base_bits - 1) / base_bits;
        metadata.columns_per_row = metadata.digits_per_tower *
            (small ? 1 : ctx->moduli.size() - dropped_moduli);
        if (metadata.columns_per_row == 0 || rows > SIZE_MAX / metadata.columns_per_row ||
            total_columns != rows * metadata.columns_per_row)
            return set_error("invalid gadget full column count");
    }
    if (global_column_start > total_columns || columns > total_columns - global_column_start)
        return set_error("constant global column range is outside its logical shape");
    const size_t n = static_cast<size_t>(ctx->N);
    if ((columns != 0 && rows > SIZE_MAX / columns) || rows * columns > SIZE_MAX / n)
        return set_error("constant coefficient count overflow");
    if (rows == 0 || columns == 0) return 0;
    int device = -1;
    for (size_t limb = 0; limb < limbs; ++limb) {
        const auto id = ctx->limb_gpu_ids[limb];
        if (id.x >= out->shared_limb_buffers.size()) return set_error("invalid constant partition");
        const auto &buffer = out->shared_limb_buffers[id.x];
        if (!buffer.device_descriptors || id.y >= buffer.limb_count)
            return set_error("missing constant output descriptors");
        if (limb == 0) { device = buffer.device; metadata.output = buffer.device_descriptors; }
        else if (device != buffer.device || metadata.output != buffer.device_descriptors)
            return set_error("constant output limbs must share one device and partition");
        metadata.indices[limb] = id.y;
        metadata.moduli[limb] = ctx->moduli[limb];
    }
    cudaStream_t stream = nullptr;
    int status = matrix_limb_stream(out, ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    if (!stream) return set_error("missing constant output stream");
    cudaError_t error = cudaSetDevice(device); if (error != cudaSuccess) return set_error(error);
    // Retained rectangles can already have readers on other streams. Join
    // initialization, prior writers and readers before modifying the owner.
    status = matrix_wait_all_limb_streams(out, device, stream, true);
    if (status != 0) return status;
    metadata.output_offset = v.row_start * out->cols + v.column_start;
    metadata.output_pitch = out->cols; metadata.columns = columns;
    metadata.global_start = global_column_start; metadata.unit_index = unit_index;
    metadata.base_bits = base_bits; metadata.mode = mode; metadata.small = small != 0;
    const size_t count = rows * columns * n;
    const dim3 grid(static_cast<unsigned>(std::min(count / 256 + (count % 256 != 0), size_t{65535})),
                    1, static_cast<unsigned>(limbs));
    matrix_fill_constant_columns_kernel<<<grid, 256, 0, stream>>>(metadata, count, n);
    error = cudaGetLastError(); if (error != cudaSuccess) return set_error(error);
    return matrix_record_all_limb_writes(out, stream, true);
}

__global__ void matrix_fill_small_decomposed_identity_chunk_all_limbs_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *src_descriptors,
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *dst_descriptors,
    size_t limb_count,
    size_t n,
    size_t size,
    size_t chunk_idx,
    size_t chunk_count)
{
    const size_t limb_idx = static_cast<size_t>(blockIdx.z);
    if (limb_idx >= limb_count)
    {
        return;
    }
    const auto source = src_descriptors[limb_idx];
    const auto target = dst_descriptors[limb_idx];
    const uint8_t *src_base = source.base;
    uint8_t *dst_base = target.base;
    const size_t src_stride = source.stride;
    const size_t dst_stride = target.stride;
    const uint8_t src_bytes = source.width;
    const uint8_t dst_bytes = target.width;
    if (!src_base || !dst_base || src_bytes == 0 || dst_bytes == 0 ||
        src_stride < n * static_cast<size_t>(src_bytes) ||
        dst_stride < n * static_cast<size_t>(dst_bytes))
    {
        return;
    }

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size * n;
    if (idx >= total)
    {
        return;
    }
    const size_t local_row = idx / n;
    const size_t coeff_idx = idx - local_row * n;
    const size_t global_row = chunk_idx * size + local_row;
    const size_t src_row = global_row / chunk_count;
    const size_t digit = global_row - src_row * chunk_count;
    if (src_row >= size || digit >= chunk_count)
    {
        return;
    }

    const size_t src_poly_idx = digit;
    const size_t dst_poly_idx = local_row * size + src_row;
    const uint64_t value = matrix_load_limb_u64(src_base, src_poly_idx, coeff_idx, src_stride, src_bytes);
    matrix_store_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes, value);
}

extern "C" int gpu_matrix_fill_small_decomposed_identity_chunk(
    GpuMatrix *out,
    const GpuMatrix *scalar_by_digit,
    size_t chunk_idx)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_fill_small_decomposed_identity_chunk");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (!out || !scalar_by_digit)
    {
        return set_error("invalid gpu_matrix_fill_small_decomposed_identity_chunk arguments");
    }
    if (out->ctx != scalar_by_digit->ctx || out->level != scalar_by_digit->level)
    {
        return set_error("context mismatch in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (out->rows != out->cols)
    {
        return set_error("output must be square in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (scalar_by_digit->rows != 1 || scalar_by_digit->cols == 0)
    {
        return set_error("scalar_by_digit must be 1 x chunk_count in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (out->format != scalar_by_digit->format)
    {
        return set_error("format mismatch in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    const size_t size = out->rows;
    const size_t chunk_count = scalar_by_digit->cols;
    if (chunk_idx >= chunk_count)
    {
        return set_error("chunk_idx out of range in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (size == 0)
    {
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    const size_t limb_count = static_cast<size_t>(level + 1);
    auto &limb_map = out->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_fill_small_decomposed_identity_chunk");
    }

    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    std::vector<dim3> active_limb_ids(limb_count);
    std::vector<uint8_t *> out_limb_bases(limb_count, nullptr);
    std::vector<size_t> out_limb_stride_bytes(limb_count, 0);
    std::vector<size_t> src_limb_stride_bytes(limb_count, 0);
    std::vector<uint8_t> out_limb_coeff_bytes(limb_count, 0);
    std::vector<uint8_t> src_limb_coeff_bytes(limb_count, 0);

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[idx];
        active_limb_ids[idx] = limb_id;
        if (limb_id.x != 0 || limb_id.y != idx)
            return set_error("identity decomposition requires ordered colocated descriptors");

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            return status;
        }
        if (out_device < 0 || !out_stream)
        {
            return set_error("invalid output limb metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_stream = out_stream;
        }
        else if (out_device != dispatch_device)
        {
            return set_error(
                "single-device mode requires all limbs on one device in gpu_matrix_fill_small_decomposed_identity_chunk");
        }

        int src_device = -1;
        status = matrix_limb_device(scalar_by_digit, limb_id, &src_device);
        if (status != 0)
        {
            return status;
        }
        if (src_device != dispatch_device)
        {
            return set_error(
                "single-device mode requires scalar limbs on dispatch device in gpu_matrix_fill_small_decomposed_identity_chunk");
        }

        uint8_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            return set_error("null output limb pointer in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        const uint8_t *src = matrix_limb_ptr_by_id(scalar_by_digit, 0, limb_id);
        if (!src)
        {
            return set_error("null source limb pointer in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_limb_stride_bytes[idx], &out_limb_coeff_bytes[idx]) ||
            !matrix_limb_metadata_by_id(
                scalar_by_digit,
                limb_id,
                &src_limb_stride_bytes[idx],
                &src_limb_coeff_bytes[idx]))
        {
            return set_error("invalid limb metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (out_limb_coeff_bytes[idx] != src_limb_coeff_bytes[idx])
        {
            return set_error("inconsistent limb byte-width in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        out_limb_bases[idx] = dst;
    }

    if (dispatch_device < 0 || !dispatch_stream)
    {
        return set_error("invalid dispatch metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    if (limb_count > kDecomposeMaxGridZ)
    {
        return set_error("too many limbs in gpu_matrix_fill_small_decomposed_identity_chunk");
    }

    if (out->shared_limb_buffers.size() != 1 || scalar_by_digit->shared_limb_buffers.size() != 1 ||
        !out->shared_limb_buffers[0].device_descriptors ||
        !scalar_by_digit->shared_limb_buffers[0].device_descriptors ||
        out->shared_limb_buffers[0].limb_count < limb_count ||
        scalar_by_digit->shared_limb_buffers[0].limb_count < limb_count ||
        size > SIZE_MAX / size || size > SIZE_MAX / static_cast<size_t>(out->ctx->N))
        return set_error("invalid identity decomposition descriptors or shape");
    const size_t total = size * static_cast<size_t>(out->ctx->N);
    if (total / 256 + (total % 256 != 0) > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("identity decomposition exceeds CUDA grid capacity");
    MatrixTransformWorkspace workspace;
    status = workspace.acquire(out, dispatch_device, dispatch_stream, {});
    if (status != 0) return status;
    const size_t out_count = size * size;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = active_limb_ids[idx];
        status = matrix_wait_limb_stream(scalar_by_digit, limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        const size_t dst_pitch = out_limb_stride_bytes[idx];
        const size_t row_bytes = static_cast<size_t>(out->ctx->N) * static_cast<size_t>(out_limb_coeff_bytes[idx]);
        err = cudaMemset2DAsync(
            out_limb_bases[idx],
            dst_pitch,
            0,
            row_bytes,
            out_count,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / threads);
    const dim3 grid{
        static_cast<unsigned int>(blocks),
        1u,
        static_cast<unsigned int>(limb_count)};
    matrix_fill_small_decomposed_identity_chunk_all_limbs_kernel<<<grid, threads, 0, dispatch_stream>>>(
        scalar_by_digit->shared_limb_buffers[0].device_descriptors,
        out->shared_limb_buffers[0].device_descriptors,
        limb_count,
        static_cast<size_t>(out->ctx->N),
        size,
        chunk_idx,
        chunk_count);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        status = matrix_track_limb_consumer(
            scalar_by_digit,
            active_limb_ids[idx],
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            return status;
        }
    }
    out->format = scalar_by_digit->format;
    return workspace.complete();
}

// Constants for Section 3.2 of ePrint 2024/909. No full-modulus integers or
// coefficient transfers are needed: every product is reduced in its own limb.
__global__ void gadget_low_constants_kernel(
    const uint64_t *moduli, uint64_t *constants, size_t retained, size_t dropped)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= dropped * (retained + 1)) return;
    const size_t low = index % dropped;
    const size_t tower = index / dropped;
    const uint64_t modulus = moduli[tower == retained ? retained + low : tower];
    uint64_t weight = 1;
    for (size_t u = 0; u < dropped; ++u)
        if (u != low)
            weight = static_cast<uint64_t>(
                (static_cast<unsigned __int128>(weight) * moduli[retained + u]) % modulus);
    if (tower == retained)
    {
        // Extended Euclid also handles a coprime non-prime CRT basis.
        __int128 t = 0, next_t = 1;
        uint64_t r = modulus, next_r = weight;
        while (next_r != 0)
        {
            const uint64_t quotient = r / next_r;
            const uint64_t remainder = r % next_r;
            const __int128 next = t - static_cast<__int128>(quotient) * next_t;
            r = next_r; next_r = remainder; t = next_t; next_t = next;
        }
        if (t < 0) t += modulus;
        weight = static_cast<uint64_t>(t);
    }
    constants[index] = weight;
}

__global__ void gadget_correct_residues_kernel(
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors,
    const uint64_t *moduli, const uint64_t *constants,
    size_t retained, size_t dropped, size_t n, size_t polys)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t poly = index / n;
    const size_t coefficient = index % n;
    const size_t tower = blockIdx.y;
    if (poly >= polys || tower >= retained) return;
    const uint64_t modulus = moduli[tower];
    const auto dest = descriptors[tower];
    uint64_t value = matrix_load_limb_u64(dest.base, poly, coefficient, dest.stride, dest.width);
    for (size_t low = 0; low < dropped; ++low)
    {
        const auto src = descriptors[retained + low];
        const uint64_t low_modulus = moduli[retained + low];
        const uint64_t residue = matrix_load_limb_u64(
            src.base, poly, coefficient, src.stride, src.width);
        const uint64_t twisted = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(residue) *
             constants[retained * dropped + low]) % low_modulus);
        const uint64_t centered = signed_digit_to_residue(
            centered_lift_u64(twisted, low_modulus), modulus);
        const uint64_t term = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(centered) *
             constants[tower * dropped + low]) % modulus);
        value = value >= term ? value - term : modulus - (term - value);
    }
    // Low limbs are read-only throughout this launch, so no cross-block race.
    matrix_store_limb_u64(dest.base, poly, coefficient, dest.stride, dest.width, value);
}

extern "C" int gpu_matrix_correct_gadget_residues(GpuMatrix *src, size_t dropped)
{
    if (!src || !src->ctx || !src->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_correct_gadget_residues");
    GpuAllocationActivity activity(src->ctx->execution.get(), -1);
    if (!src || !src->ctx || src->format != GPU_POLY_FORMAT_COEFF || src->level < 0)
        return set_error("invalid approximate gadget source");
    const size_t limbs = static_cast<size_t>(src->level + 1);
    if (dropped >= limbs) return set_error("invalid dropped_moduli");
    if (dropped == 0 || src->rows == 0 || src->cols == 0) return 0;
    const size_t retained = limbs - dropped;
    if (src->ctx->limb_gpu_ids.size() < limbs) return set_error("missing approximate gadget limbs");
    const dim3 first = src->ctx->limb_gpu_ids[0];
    int device = -1;
    cudaStream_t stream = nullptr;
    if (matrix_limb_device(src, first, &device) != 0 ||
        matrix_limb_stream(src, first, &stream) != 0) return 1;
    if (first.x >= src->shared_limb_buffers.size() ||
        first.x >= src->ctx->ring_device_constants.size())
        return set_error("missing approximate gadget descriptors");
    const auto &buffers = src->shared_limb_buffers[first.x];
    const auto &constants = src->ctx->ring_device_constants[first.x];
    if (!buffers.device_descriptors || buffers.limb_count < limbs ||
        !constants.moduli || constants.limb_count < limbs)
        return set_error("missing approximate gadget constants");
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const dim3 id = src->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        if (matrix_limb_device(src, id, &limb_device) != 0 || limb_device != device ||
            id.x != first.x || id.y != limb)
            return set_error("approximate gadget requires ordered limbs on one device");
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return set_error(err);
    GpuMatrixTransformWorkspaceBytes requirements{};
    int status = gpu_matrix_query_gadget_correction_workspace_bytes(
        src->ctx, src->level, src->rows, src->cols, dropped, &requirements);
    if (status != 0) return status;
    const size_t coefficient_count = static_cast<size_t>(src->ctx->N) * src->rows * src->cols;
    const size_t blocks = coefficient_count / 256 + (coefficient_count % 256 != 0);
    if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("gadget correction exceeds CUDA grid capacity");
    MatrixTransformWorkspace workspace;
    status = workspace.acquire(src, device, stream, requirements);
    if (status != 0) return status;
    auto *weights = reinterpret_cast<uint64_t *>(workspace.base);
    const size_t entries = dropped * (retained + 1);
    gadget_low_constants_kernel<<<(entries + 255) / 256, 256, 0, stream>>>(
        constants.moduli, weights, retained, dropped);
    err = cudaGetLastError();
    if (err == cudaSuccess)
    {
        const dim3 grid(static_cast<unsigned int>(blocks), static_cast<unsigned int>(retained));
        gadget_correct_residues_kernel<<<grid, 256, 0, stream>>>(
            buffers.device_descriptors, constants.moduli, weights,
            retained, dropped, src->ctx->N, src->rows * src->cols);
        err = cudaGetLastError();
    }
    if (err != cudaSuccess) return set_error(err);
    // All limbs share the exclusive auxiliary allocation. Joining each limb's
    // next use also protects the unmodified, discarded limbs read by this call.
    return workspace.complete();
}

static int gpu_matrix_decompose_base_impl(
    const GpuMatrix *src,
    uint32_t base_bits,
    GpuMatrix *out,
    bool small,
    size_t dropped_moduli)
{
    if (!src || !out)
    {
        return set_error("invalid gpu_matrix_decompose_base arguments");
    }
    if (base_bits == 0)
    {
        return set_error("base_bits must be non-zero in gpu_matrix_decompose_base");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_decompose_base");
    }
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_decompose_base");
    }

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    GpuMatrixDecomposeWorkspaceBytes requirements{};
    int status = gpu_matrix_query_decompose_workspace_bytes(
        src->ctx, src->level, rows, cols, src->format, base_bits, small, dropped_moduli, &requirements);
    if (status != 0) return status;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_decompose_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (dropped_moduli >= crt_depth) return set_error("invalid dropped_moduli");
    uint32_t crt_bits = 0;
    for (const auto &modulus : src->ctx->moduli)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_decompose_base");
    }
    const uint32_t digits_per_tower =
        static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_decompose_base");
    }
    const size_t out_log_base_q =
        small ? static_cast<size_t>(digits_per_tower)
              : static_cast<size_t>(digits_per_tower) * (crt_depth - dropped_moduli);
    if (out->rows != requirements.output_rows || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_decompose_base");
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

    if (src->format == GPU_POLY_FORMAT_EVAL || dropped_moduli > 0)
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
        status = src->format == GPU_POLY_FORMAT_EVAL ? gpu_matrix_intt_all(tmp_inputs_matrix) : 0;
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        inputs_matrix = tmp_inputs_matrix;
        if (dropped_moduli > 0)
        {
            status = gpu_matrix_correct_gadget_residues(tmp_inputs_matrix, dropped_moduli);
            if (status != 0) { cleanup_tmp_inputs(); return status; }
        }
    }

    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    const size_t limb_count = static_cast<size_t>(level + 1);
    std::vector<dim3> active_limb_ids(limb_count);
    std::vector<uint8_t *> out_limb_bases(limb_count, nullptr);
    std::vector<size_t> out_limb_stride_bytes(limb_count, 0);
    std::vector<uint8_t> out_limb_coeff_bytes(limb_count, 0);

    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[idx];
        active_limb_ids[idx] = limb_id;
        if (limb_id.x != 0 || limb_id.y != idx)
        {
            cleanup_tmp_inputs();
            return set_error("decomposition requires ordered colocated descriptors");
        }

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
        if (out_device < 0 || !out_stream)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output limb metadata in gpu_matrix_decompose_base");
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_stream = out_stream;
        }
        else if (out_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-device mode requires all limbs on one device in gpu_matrix_decompose_base");
        }

        uint8_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb pointer in gpu_matrix_decompose_base");
        }
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_limb_stride_bytes[idx], &out_limb_coeff_bytes[idx]))
        {
            cleanup_tmp_inputs();
            return set_error("invalid output limb metadata in gpu_matrix_decompose_base");
        }
        out_limb_bases[idx] = dst;
    }
    if (dispatch_device < 0 || !dispatch_stream)
    {
        cleanup_tmp_inputs();
        return set_error("invalid dispatch stream in gpu_matrix_decompose_base");
    }

    const size_t source_limb_count = small ? limb_count : crt_depth - dropped_moduli;
    for (size_t src_limb = 0; src_limb < source_limb_count; ++src_limb)
    {
        const dim3 src_limb_id = active_limb_ids[static_cast<size_t>(src_limb)];
        int src_device = -1;
        status = matrix_limb_device(inputs_matrix, src_limb_id, &src_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        if (src_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-device mode requires all limbs on one device in gpu_matrix_decompose_base");
        }
    }

    if (out->shared_limb_buffers.size() != 1 || !out->shared_limb_buffers[0].device_descriptors ||
        out->shared_limb_buffers[0].limb_count < limb_count || out->ctx->ring_device_constants.empty() ||
        !out->ctx->ring_device_constants[0].moduli || out->ctx->ring_device_constants[0].limb_count < limb_count ||
        inputs_matrix->shared_limb_buffers.size() != 1 ||
        !inputs_matrix->shared_limb_buffers[0].device_descriptors ||
        inputs_matrix->shared_limb_buffers[0].limb_count < source_limb_count)
    {
        cleanup_tmp_inputs();
        return set_error("missing decomposition device descriptors or moduli");
    }
    MatrixTransformWorkspace workspace;
    status = workspace.acquire(out, dispatch_device, dispatch_stream, {});
    auto cleanup = [&]()
    {
        // Retire before dropping a temporary source whose last reader update
        // might have failed after a successful earlier kernel submission.
        workspace.retire();
        cleanup_tmp_inputs();
    };
    if (status != 0) { cleanup(); return status; }

    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    const size_t out_count = out->rows * out->cols;
    for (int out_limb = 0; out_limb <= level; ++out_limb)
    {
        const size_t out_idx = static_cast<size_t>(out_limb);
        if (out_count > 0)
        {
            const size_t dst_pitch = out_limb_stride_bytes[out_idx];
            const size_t row_bytes =
                static_cast<size_t>(src->ctx->N) * static_cast<size_t>(out_limb_coeff_bytes[out_idx]);
            err = cudaMemset2DAsync(
                out_limb_bases[out_idx],
                dst_pitch,
                0,
                row_bytes,
                out_count,
                dispatch_stream);
            if (err != cudaSuccess)
            {
                cleanup();
                return set_error(err);
            }
        }
    }

    if (small)
    {
        // One small-decomposition launch reads every tower through its resident
        // descriptor. Keep those independent input writers and releases covered.
        status = matrix_wait_all_limb_streams(inputs_matrix, dispatch_device, dispatch_stream, false, true);
        if (status != 0) { cleanup(); return status; }
    }
    const size_t launch_count = small ? 1 : source_limb_count;
    for (size_t src_limb = 0; src_limb < launch_count; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
        if (!small)
        {
            status = matrix_wait_limb_stream(inputs_matrix, src_limb_id, dispatch_device, dispatch_stream);
            if (status != 0) { cleanup(); return status; }
        }

        const uint8_t *src_base = matrix_limb_ptr_by_id(inputs_matrix, 0, src_limb_id);
        if (!src_base)
        {
            cleanup();
            return set_error("null source limb base pointer in gpu_matrix_decompose_base");
        }
        size_t src_stride_bytes = 0;
        uint8_t src_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(
                inputs_matrix,
                src_limb_id,
                &src_stride_bytes,
                &src_coeff_bytes))
        {
            cleanup();
            return set_error("invalid source limb metadata in gpu_matrix_decompose_base");
        }
        const size_t src_digit_offset_base =
            small ? 0 : (src_idx * static_cast<size_t>(digits_per_tower));

        status = launch_decompose_all_slots_kernel(
            src_base,
            inputs_matrix->shared_limb_buffers[0].device_descriptors,
            out->shared_limb_buffers[0].device_descriptors,
            out->ctx->ring_device_constants[0].moduli,
            src_stride_bytes,
            src_coeff_bytes,
            limb_count,
            count,
            static_cast<size_t>(src->ctx->N),
            cols,
            out->cols,
            out_log_base_q,
            src->ctx->moduli[src_idx],
            base_bits,
            digits_per_tower,
            !small,
            src_digit_offset_base,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = small
            ? matrix_track_all_limb_consumers(inputs_matrix, dispatch_device, dispatch_stream,
                                             nullptr, false, true)
            : matrix_track_limb_consumer(inputs_matrix, src_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    status = workspace.complete();
    if (status != 0) { cleanup(); return status; }
    out->format = GPU_POLY_FORMAT_COEFF;
    if (requested_out_format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_ntt_all(out);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        out->format = GPU_POLY_FORMAT_EVAL;
    }

    cleanup();
    return 0;
}

extern "C" int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out, size_t dropped_moduli)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_decompose_base");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    return gpu_matrix_decompose_base_impl(src, base_bits, out, false, dropped_moduli);
}

extern "C" int gpu_matrix_decompose_base_small(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_decompose_base_small");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    return gpu_matrix_decompose_base_impl(src, base_bits, out, true, 0);
}
