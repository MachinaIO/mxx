// Included after the shared serialization shape/overflow helpers.
#include <array>

namespace {
struct CompactWorkspaceLayout {
    std::array<size_t, 14> offsets{};
    std::array<size_t, 14> capacities{};
    size_t bytes = 0;

    bool append(size_t slot, size_t count, size_t element_bytes, size_t alignment) {
        size_t size = 0;
        if (!serde_checked_mul_size(count, element_bytes, &size) ||
            bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) return false;
        const size_t offset = (bytes + alignment - 1) & ~(alignment - 1);
        if (size > std::numeric_limits<size_t>::max() - offset) return false;
        offsets[slot] = offset;
        capacities[slot] = size;
        bytes = offset + size;
        return true;
    }
};

int compact_workspace_layout(
    GpuContext *ctx, int level, size_t rows, size_t columns, size_t matrices,
    int kind, uint16_t max_coeff_bits, CompactWorkspaceLayout &layout)
{
    if (!ctx || ctx->N <= 0 || level < 0 ||
        static_cast<size_t>(level) >= ctx->moduli.size() ||
        (kind != 0 && kind != 1 && kind != 2) || matrices == 0 ||
        (kind != 1 && matrices != 1))
        return set_error("invalid compact transfer workspace layout");
    if (rows == 0 || columns == 0) return 0;
    const size_t limbs = static_cast<size_t>(level) + 1;
    if (limbs > static_cast<size_t>(kMaxRnsLimbs))
        return set_error("compact transfer workspace exceeds limb limit");
    size_t coefficients = 0, total_coefficients = 0, pointers = 0;
    if (!serde_checked_mul_size(rows, columns, &coefficients) ||
        !serde_checked_mul_size(coefficients, static_cast<size_t>(ctx->N), &coefficients) ||
        !serde_checked_mul_size(coefficients, matrices, &total_coefficients) ||
        !serde_checked_mul_size(limbs, matrices, &pointers) ||
        matrices == std::numeric_limits<size_t>::max())
        return set_error("compact transfer workspace shape overflow");
    size_t total_bits = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
        total_bits += static_cast<size_t>(bit_width_u64(ctx->moduli[limb]));
    const size_t words = std::max<size_t>(1, (total_bits + 63) / 64);
    if (words > static_cast<size_t>(kMaxCoeffWords))
        return set_error("compact transfer workspace exceeds coefficient word limit");
    size_t payload = 0;
    // Centering modulo q leaves signed width <= ceil(log2(q)); the sum of
    // individual modulus widths is the production codec's existing upper bound.
    if (!serde_compute_payload_len(coefficients, kind == 2 ? max_coeff_bits : static_cast<uint32_t>(total_bits), &payload))
        return set_error("compact transfer workspace payload overflow");
    bool valid = false;
    if (kind == 2) {
        valid = layout.append(0, payload, 1, 1) &&
            layout.append(1, limbs, sizeof(uint8_t *), alignof(uint8_t *)) &&
            layout.append(2, limbs, sizeof(size_t), alignof(size_t)) &&
            layout.append(3, limbs, sizeof(uint8_t), alignof(uint8_t)) &&
            layout.append(4, limbs, sizeof(uint64_t), alignof(uint64_t));
    } else if (kind == 0) {
        size_t coefficient_words = 0;
        if (!serde_checked_mul_size(coefficients, words, &coefficient_words))
            return set_error("compact transfer workspace coefficient overflow");
        valid = layout.append(0, words, 8, 8) && layout.append(1, words, 8, 8) &&
            layout.append(2, limbs, sizeof(uint8_t *), alignof(uint8_t *)) &&
            layout.append(3, limbs, sizeof(size_t), alignof(size_t)) &&
            layout.append(4, limbs, 1, 1) && layout.append(5, limbs, 8, 8) &&
            layout.append(6, ctx->garner_inverse_table.size(), 8, 8) &&
            layout.append(7, coefficient_words, 8, 8) && layout.append(8, 1, sizeof(int), alignof(int)) &&
            layout.append(9, coefficients, 1, 1) &&
            layout.append(10, 1, sizeof(unsigned int), alignof(unsigned int)) &&
            layout.append(11, payload, 1, 1);
    } else {
        size_t coefficient_words = 0;
        if (!serde_checked_mul_size(total_coefficients, words, &coefficient_words))
            return set_error("compact batch workspace coefficient overflow");
        valid = layout.append(0, pointers, sizeof(uint8_t *), alignof(uint8_t *)) &&
            layout.append(1, pointers, sizeof(size_t), alignof(size_t)) &&
            layout.append(2, pointers, 1, 1) && layout.append(3, limbs, 8, 8) &&
            layout.append(4, ctx->garner_inverse_table.size(), 8, 8) &&
            layout.append(5, words, 8, 8) && layout.append(6, words, 8, 8) &&
            layout.append(7, coefficient_words, 8, 8) &&
            layout.append(8, total_coefficients, 1, 1) &&
            layout.append(9, matrices, sizeof(unsigned int), alignof(unsigned int)) &&
            layout.append(10, 1, sizeof(int), alignof(int)) &&
            layout.append(11, matrices, sizeof(uint32_t), alignof(uint32_t)) &&
            layout.append(12, matrices + 1, sizeof(size_t), alignof(size_t)) &&
            layout.append(13, payload, matrices, 1);
    }
    return valid ? 0 : set_error("compact transfer workspace byte size overflow");
}

struct CompactWorkspace {
    CompactWorkspaceLayout layout;
    GpuDeviceWorkspace storage;

    int acquire(GpuContext *ctx, int device, int level, size_t rows, size_t columns,
        size_t matrices, int kind, uint16_t bits, cudaStream_t stream) {
        const int status = compact_workspace_layout(ctx, level, rows, columns, matrices, kind, bits, layout);
        if (status != 0) return status;
        return storage.acquire(ctx, device, GPU_PREPARED_TRANSFER_WORKSPACE, layout.bytes, 256, stream);
    }
    template <class T> cudaError_t span(size_t index, T **pointer, size_t bytes) {
        if (index >= layout.offsets.size() || bytes > layout.capacities[index] || !storage.data)
            return cudaErrorInvalidValue;
        *pointer = reinterpret_cast<T *>(storage.data + layout.offsets[index]);
        return cudaSuccess;
    }
};
}

extern "C" int gpu_matrix_query_compact_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns, size_t matrices,
    int kind, uint16_t max_coeff_bits, GpuPreparedWorkspaceLayout *out)
{
    if (!out) return set_error("missing compact transfer workspace output");
    CompactWorkspaceLayout layout;
    const int status = compact_workspace_layout(ctx, level, rows, columns, matrices, kind, max_coeff_bits, layout);
    if (status != 0) return status;
    *out = {layout.bytes, 256, GPU_PREPARED_TRANSFER_WORKSPACE};
    return 0;
}
