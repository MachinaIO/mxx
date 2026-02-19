#include "ChaCha.cuh"

namespace gpu_chacha
{
    __device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n)
    {
        return (x << n) | (x >> (32U - n));
    }

    __device__ __forceinline__ uint64_t splitmix64_next(uint64_t &state)
    {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

    __device__ __forceinline__ void quarter_round(
        uint32_t &a,
        uint32_t &b,
        uint32_t &c,
        uint32_t &d)
    {
        a += b;
        d ^= a;
        d = rotl32(d, 16U);

        c += d;
        b ^= c;
        b = rotl32(b, 12U);

        a += b;
        d ^= a;
        d = rotl32(d, 8U);

        c += d;
        b ^= c;
        b = rotl32(b, 7U);
    }

    __device__ __forceinline__ void chacha20_block(
        const uint32_t in_state[16],
        uint32_t out_block[16])
    {
        uint32_t x[16];
        for (uint32_t i = 0; i < 16; ++i)
        {
            x[i] = in_state[i];
        }

        for (uint32_t round = 0; round < 10; ++round)
        {
            quarter_round(x[0], x[4], x[8], x[12]);
            quarter_round(x[1], x[5], x[9], x[13]);
            quarter_round(x[2], x[6], x[10], x[14]);
            quarter_round(x[3], x[7], x[11], x[15]);

            quarter_round(x[0], x[5], x[10], x[15]);
            quarter_round(x[1], x[6], x[11], x[12]);
            quarter_round(x[2], x[7], x[8], x[13]);
            quarter_round(x[3], x[4], x[9], x[14]);
        }

        for (uint32_t i = 0; i < 16; ++i)
        {
            out_block[i] = x[i] + in_state[i];
        }
    }

    __device__ __forceinline__ void rng_init(
        DeviceChaChaRng &rng,
        uint64_t seed,
        uint64_t stream0,
        uint64_t stream1,
        uint64_t stream2,
        uint64_t domain_tag)
    {
        rng.state[0] = 0x61707865U;
        rng.state[1] = 0x3320646eU;
        rng.state[2] = 0x79622d32U;
        rng.state[3] = 0x6b206574U;

        uint64_t mix = seed ^ 0x243f6a8885a308d3ULL;
        mix ^= (stream0 + 0x9e3779b97f4a7c15ULL);
        mix ^= (stream1 + 0xbf58476d1ce4e5b9ULL);
        mix ^= (stream2 + 0x94d049bb133111ebULL);
        mix ^= (domain_tag + 0xd6e8feb86659fd93ULL);

        for (uint32_t i = 0; i < 4; ++i)
        {
            const uint64_t v = splitmix64_next(mix);
            rng.state[4 + 2 * i] = static_cast<uint32_t>(v);
            rng.state[5 + 2 * i] = static_cast<uint32_t>(v >> 32U);
        }

        const uint64_t n0 = splitmix64_next(mix);
        const uint64_t n1 = splitmix64_next(mix);
        rng.state[12] = 0U;
        rng.state[13] = static_cast<uint32_t>(n0);
        rng.state[14] = static_cast<uint32_t>(n0 >> 32U);
        rng.state[15] = static_cast<uint32_t>(n1);

        rng.block_idx = 8U;
    }

    __device__ __forceinline__ void rng_refill(DeviceChaChaRng &rng)
    {
        chacha20_block(rng.state, rng.block);
        rng.state[12] += 1U;
        rng.block_idx = 0U;
    }

    __device__ __forceinline__ uint64_t rng_next_u64(DeviceChaChaRng &rng)
    {
        if (rng.block_idx >= 8U)
        {
            rng_refill(rng);
        }
        const uint32_t w0 = rng.block[2U * rng.block_idx];
        const uint32_t w1 = rng.block[2U * rng.block_idx + 1U];
        rng.block_idx += 1U;
        return static_cast<uint64_t>(w0) | (static_cast<uint64_t>(w1) << 32U);
    }
} // namespace gpu_chacha
