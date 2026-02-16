#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

namespace gpu_chacha
{
    struct DeviceChaChaRng
    {
        uint32_t state[16];
        uint32_t block[16];
        uint32_t block_idx;
    };

    __device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n);
    __device__ __forceinline__ uint64_t splitmix64_next(uint64_t &state);
    __device__ __forceinline__ void quarter_round(
        uint32_t &a,
        uint32_t &b,
        uint32_t &c,
        uint32_t &d);
    __device__ __forceinline__ void chacha20_block(
        const uint32_t in_state[16],
        uint32_t out_block[16]);
    __device__ __forceinline__ void rng_init(
        DeviceChaChaRng &rng,
        uint64_t seed,
        uint64_t stream0,
        uint64_t stream1,
        uint64_t stream2,
        uint64_t domain_tag);
    __device__ __forceinline__ void rng_refill(DeviceChaChaRng &rng);
    __device__ __forceinline__ uint64_t rng_next_u64(DeviceChaChaRng &rng);
} // namespace gpu_chacha
