// Direct Graph emitter for one coefficient-domain polynomial from resident
// sign-and-magnitude integer values. Included after the raw matrix helpers.
namespace
{
    constexpr uint32_t kPolynomialInvalidInteger = 2;

    __global__ void raw_polynomial_from_values_kernel(
        const uint64_t *source, size_t magnitude_words,
        MxxRawMatrixLimb destination, uint32_t degree, uint32_t *status)
    {
        const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
        for (size_t coefficient = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             coefficient < degree; coefficient += step)
        {
            const uint64_t *value = source + coefficient * (magnitude_words + 1);
            const uint64_t sign = value[0];
            uint64_t residue = 0;
            bool zero = true;
            for (size_t word = magnitude_words; word-- > 0;)
            {
                const uint64_t digit = value[word + 1];
                zero = zero && digit == 0;
                residue = static_cast<uint64_t>(
                    ((static_cast<unsigned __int128>(residue) << 64) + digit) %
                    destination.modulus);
            }
            if (sign > 1 || (sign == 1 && zero))
            {
                atomicCAS(status, 0U, kPolynomialInvalidInteger);
                residue = 0;
            }
            else if (sign == 1 && residue != 0)
            {
                residue = destination.modulus - residue;
            }
            raw_matrix_store(destination, 0, coefficient, 1, residue);
        }
    }
}

extern "C" int gpu_raw_polynomial_from_values(
    GpuContext *ctx, void *stream_raw, const uint64_t *source,
    size_t source_count, size_t magnitude_words,
    const MxxRawMatrixView *destination, uint32_t *status,
    uint32_t source_binding, uint32_t destination_binding_base,
    uint32_t status_binding)
{
    if (!ctx || !stream_raw || !source || !status || !destination ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        destination->rows != 1 || destination->columns != 1 ||
        destination->row_origin != 0 || destination->column_origin != 0 ||
        source_count != destination->degree || magnitude_words == 0 ||
        magnitude_words >= SIZE_MAX / sizeof(uint64_t) ||
        source_count > SIZE_MAX / ((magnitude_words + 1) * sizeof(uint64_t)) ||
        destination->limb_count != ctx->moduli.size() ||
        destination_binding_base > UINT32_MAX - destination->limb_count)
        return set_error("invalid raw polynomial-from-values layout");
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
        if (destination->limbs[limb].crt_limb_index != limb ||
            destination->limbs[limb].modulus != ctx->moduli[limb])
            return set_error("raw polynomial-from-values requires ordered CRT basis");
    if (cudaSetDevice(destination->physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const uint32_t grid = (destination->degree - 1) / 256 + 1;
    for (size_t limb = 0; limb < destination->limb_count; ++limb)
    {
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0,
                static_cast<uint32_t>(sizeof(void *)), source_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 2,
                static_cast<uint32_t>(offsetof(MxxRawMatrixLimb, address)),
                static_cast<uint32_t>(sizeof(void *)),
                destination_binding_base + static_cast<uint32_t>(limb), 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 4, 0,
                static_cast<uint32_t>(sizeof(void *)), status_binding, 0},
        };
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            raw_polynomial_from_values_kernel, dim3(grid), dim3(256), 0,
            patches, std::size(patches), source, magnitude_words,
            destination->limbs[limb], destination->degree, status);
        if (result != 0) return result;
    }
    return 0;
}
