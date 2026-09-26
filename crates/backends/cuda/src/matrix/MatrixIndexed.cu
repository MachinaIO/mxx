// Device-indexed selection of independently allocated matrix family members.
// The plan owns one pointer-stable table; replay refreshes its contents before
// launching the explicit graph on the same stream.

struct GpuIndexedMatrixTable
{
    int device = -1;
    size_t family_count = 0;
    size_t limb_count = 0;
    MxxRawMatrixLimb *device_limbs = nullptr;
    MxxRawMatrixLimb *pinned_limbs = nullptr;
    cudaStream_t allocation_stream = nullptr;
    cudaEvent_t allocation_ready = nullptr;
};

namespace
{
    __device__ bool indexed_matrix_index(const void *address, int encoding,
        uint64_t family_count, uint64_t *selected)
    {
        if (encoding == 0)
        {
            const int64_t value = *static_cast<const int64_t *>(address);
            if (value < 0) return false;
            *selected = static_cast<uint64_t>(value);
        }
        else if (encoding == 1)
            *selected = *static_cast<const uint64_t *>(address);
        else if (encoding >= 3)
        {
            const uint64_t *words = static_cast<const uint64_t *>(address);
            const size_t magnitude_words = static_cast<size_t>(encoding - 2);
            const uint64_t sign = words[0];
            if (sign > 1) return false;
            bool high_nonzero = false;
            for (size_t word = 2; word <= magnitude_words; ++word)
                high_nonzero = high_nonzero || words[word] != 0;
            if (sign != 0 || high_nonzero) return false;
            *selected = words[1];
        }
        else return false;
        return *selected < family_count;
    }

    // One launch covers up to kRawNttLimbs destination limbs; blockIdx.y
    // picks the limb `first_limb + blockIdx.y`.
    __global__ void indexed_matrix_copy_kernel(const void *index_address,
        int index_encoding, const MxxRawMatrixLimb *table,
        size_t family_count, size_t limb_count, size_t first_limb,
        RawLimbSet destinations, uint64_t rows, uint64_t columns,
        uint32_t degree, uint32_t *status)
    {
        uint64_t selected = 0;
        if (!indexed_matrix_index(index_address, index_encoding,
            family_count, &selected))
        {
            if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
                atomicCAS(status, 0U, 2U);
            return;
        }
        const MxxRawMatrixLimb &destination = destinations.limb[blockIdx.y];
        const MxxRawMatrixLimb source = table[selected * limb_count + first_limb + blockIdx.y];
        const uint64_t total = rows * columns * degree;
        for (uint64_t flat = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             flat < total; flat += static_cast<uint64_t>(gridDim.x) * blockDim.x)
        {
            const uint64_t poly = flat / degree;
            const uint32_t coefficient = static_cast<uint32_t>(flat % degree);
            raw_matrix_store(destination, poly, coefficient, columns,
                raw_matrix_load(source, poly, coefficient, columns));
        }
    }
}

extern "C" int gpu_indexed_matrix_table_create(GpuContext *ctx, void *stream_raw,
    int32_t physical_device, size_t family_count, size_t limb_count,
    GpuIndexedMatrixTable **out_table)
{
    if (!ctx || !stream_raw || !out_table || physical_device < 0 ||
        !family_count || !limb_count || limb_count > GPU_RUNTIME_MAX_LIMBS ||
        family_count > SIZE_MAX / limb_count ||
        family_count * limb_count > SIZE_MAX / sizeof(MxxRawMatrixLimb))
        return set_error("invalid indexed matrix table allocation");
    *out_table = nullptr;
    if (mxx_set_device(physical_device) != cudaSuccess)
        return set_error(cudaGetLastError());
    auto *table = new GpuIndexedMatrixTable;
    table->device = physical_device;
    table->family_count = family_count;
    table->limb_count = limb_count;
    table->allocation_stream = reinterpret_cast<cudaStream_t>(stream_raw);
    const size_t bytes = family_count * limb_count * sizeof(MxxRawMatrixLimb);
    cudaError_t error = cudaMallocHost(
        reinterpret_cast<void **>(&table->pinned_limbs), bytes);
    if (error == cudaSuccess)
        error = cudaMallocAsync(reinterpret_cast<void **>(&table->device_limbs),
            bytes, table->allocation_stream);
    if (error == cudaSuccess)
        error = cudaEventCreateWithFlags(&table->allocation_ready,
            cudaEventDisableTiming);
    if (error == cudaSuccess)
        error = cudaEventRecord(table->allocation_ready, table->allocation_stream);
    if (error != cudaSuccess)
    {
        if (table->device_limbs)
            cudaFreeAsync(table->device_limbs, table->allocation_stream);
        if (table->pinned_limbs) cudaFreeHost(table->pinned_limbs);
        if (table->allocation_ready) cudaEventDestroy(table->allocation_ready);
        delete table;
        return set_error(error);
    }
    *out_table = table;
    return 0;
}

extern "C" int gpu_indexed_matrix_table_upload(GpuIndexedMatrixTable *table,
    void *stream_raw, const MxxRawMatrixLimb *limbs, size_t limb_count)
{
    if (!table || !stream_raw || !limbs ||
        limb_count != table->family_count * table->limb_count)
        return set_error("invalid indexed matrix table upload");
    if (mxx_set_device(table->device) != cudaSuccess)
        return set_error(cudaGetLastError());
    const size_t bytes = limb_count * sizeof(MxxRawMatrixLimb);
    std::memcpy(table->pinned_limbs, limbs, bytes);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    cudaError_t error = cudaStreamWaitEvent(stream, table->allocation_ready, 0);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(table->device_limbs, table->pinned_limbs,
            bytes, cudaMemcpyHostToDevice, stream);
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" uint64_t gpu_indexed_matrix_table_address(
    const GpuIndexedMatrixTable *table)
{
    return table ? reinterpret_cast<uint64_t>(table->device_limbs) : 0;
}

extern "C" void gpu_indexed_matrix_table_destroy(GpuIndexedMatrixTable *table)
{
    if (!table) return;
    mxx_set_device(table->device);
    if (table->device_limbs)
        cudaFreeAsync(table->device_limbs, table->allocation_stream);
    if (table->allocation_ready) cudaEventDestroy(table->allocation_ready);
    if (table->pinned_limbs) cudaFreeHost(table->pinned_limbs);
    delete table;
}

extern "C" int gpu_raw_matrix_indexed_copy(GpuContext *ctx, void *stream_raw,
    const void *index_address, int index_encoding,
    const GpuIndexedMatrixTable *table, const MxxRawMatrixView *destination,
    uint32_t *status, uint32_t index_binding,
    uint32_t destination_binding_base, uint32_t status_binding)
{
    if (!index_address || !status || !table ||
        validate_raw_view(ctx, destination, stream_raw) != 0 ||
        table->device != destination->physical_device ||
        table->limb_count != destination->limb_count ||
        destination_binding_base > UINT32_MAX - destination->limb_count ||
        index_encoding == 2 || index_encoding < 0 ||
        destination->rows > UINT64_MAX / destination->columns ||
        destination->rows * destination->columns > UINT64_MAX / destination->degree)
        return set_error("invalid indexed matrix copy layout");
    const uint64_t total = destination->rows * destination->columns * destination->degree;
    const uint32_t grid = static_cast<uint32_t>(std::min<uint64_t>((total + 255) / 256, 65535));
    const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
    for (size_t first = 0; first < destination->limb_count; first += kRawNttLimbs)
    {
        const size_t limbs = std::min(kRawNttLimbs, destination->limb_count - first);
        RawLimbSet destinations{};
        std::vector<MxxGraphPatch> patches{
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0,
                sizeof(void *), index_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 10, 0,
                sizeof(void *), status_binding, 0}};
        raw_limb_set(destination, first, limbs, 6, destination_binding_base, destinations,
            patches);
        const int result = mxx_gpu_launch_kernel(ctx, stream,
            indexed_matrix_copy_kernel, dim3(grid, static_cast<uint32_t>(limbs)), dim3(256), 0,
            patches.data(), patches.size(), index_address, index_encoding,
            table->device_limbs, table->family_count, table->limb_count,
            first, destinations, destination->rows,
            destination->columns, destination->degree, status);
        if (result != 0) return result;
    }
    return 0;
}
