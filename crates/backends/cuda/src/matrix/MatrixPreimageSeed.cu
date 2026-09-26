// Per-attempt seed derivation inside an explicit CUDA Graph WHILE body.
// This uses the existing device Keccak implementation used by preimage stages.
namespace
{
    __global__ void raw_preimage_derive_attempt_seed_kernel(
        const uint8_t *base_seed, const uint64_t *attempt,
        uint64_t domain, uint8_t *derived_seed)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        uint8_t message[136] = {};
        constexpr char prefix[] = "mxx-preimage-attempt-seed/v1";
        size_t offset = 0;
        for (size_t byte = 0; byte < sizeof(prefix) - 1; ++byte)
            message[offset++] = static_cast<uint8_t>(prefix[byte]);
        for (size_t byte = 0; byte < 32; ++byte)
            message[offset++] = base_seed[byte];
        const uint64_t attempt_value = *attempt;
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(attempt_value >> (8 * byte));
        for (int byte = 0; byte < 8; ++byte)
            message[offset++] = static_cast<uint8_t>(domain >> (8 * byte));
        message[offset] ^= 0x01U;
        message[135] ^= 0x80U;
        uint64_t state[25] = {};
        for (size_t lane = 0; lane < 17; ++lane)
            state[lane] ^= preimage_seed_load_le64(message + lane * 8);
        preimage_seed_keccak_f(state);
        for (size_t byte = 0; byte < 32; ++byte)
            derived_seed[byte] = static_cast<uint8_t>(
                state[byte / 8] >> (8 * (byte % 8)));
    }
}

extern "C" int gpu_raw_preimage_derive_attempt_seed(
    GpuContext *ctx, void *stream_raw, const uint8_t *base_seed,
    const uint64_t *attempt, uint64_t domain, uint8_t *derived_seed,
    uint32_t base_binding, uint32_t attempt_binding,
    uint32_t derived_binding)
{
    if (!ctx || !stream_raw || !base_seed || !attempt || !derived_seed ||
        base_seed == derived_seed ||
        !mxx_gpu_graph_builder_for_stream(ctx, stream_raw))
        return set_error("invalid raw preimage attempt seed derivation");
    const MxxGraphPatch patches[] = {
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0,
            sizeof(void *), base_binding, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 1, 0,
            sizeof(void *), attempt_binding, 0},
        {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 3, 0,
            sizeof(void *), derived_binding, 0},
    };
    return mxx_gpu_launch_kernel(ctx, reinterpret_cast<cudaStream_t>(stream_raw),
        raw_preimage_derive_attempt_seed_kernel, dim3(1), dim3(1), 0,
        patches, std::size(patches), base_seed, attempt, domain, derived_seed);
}
