#include "ChaCha.cuh"

namespace gpu_chacha
{
    __device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n)
    {
        return (x << n) | (x >> (32U - n));
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

    __device__ __forceinline__ void hchacha20(
        const uint32_t key[8],
        const uint32_t nonce[4],
        uint32_t out_key[8])
    {
        uint32_t x[16];
        x[0] = 0x61707865U;
        x[1] = 0x3320646eU;
        x[2] = 0x79622d32U;
        x[3] = 0x6b206574U;
        for (uint32_t i = 0; i < 8; ++i)
        {
            x[4 + i] = key[i];
        }
        for (uint32_t i = 0; i < 4; ++i)
        {
            x[12 + i] = nonce[i];
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

        out_key[0] = x[0];
        out_key[1] = x[1];
        out_key[2] = x[2];
        out_key[3] = x[3];
        out_key[4] = x[12];
        out_key[5] = x[13];
        out_key[6] = x[14];
        out_key[7] = x[15];
    }

    __device__ __forceinline__ void rng_init(
        DeviceChaChaRng &rng,
        const GpuRngSeed &seed,
        uint64_t stream0,
        uint64_t stream1,
        uint64_t stream2,
        uint64_t domain_tag)
    {
        rng.state[0] = 0x61707865U;
        rng.state[1] = 0x3320646eU;
        rng.state[2] = 0x79622d32U;
        rng.state[3] = 0x6b206574U;

        uint32_t base_key[8];
        for (uint32_t i = 0; i < 4; ++i)
        {
            const uint64_t word = seed.words[i];
            base_key[2 * i] = static_cast<uint32_t>(word);
            base_key[2 * i + 1] = static_cast<uint32_t>(word >> 32U);
        }

        uint32_t hnonce[4] = {
            static_cast<uint32_t>(domain_tag),
            static_cast<uint32_t>(domain_tag >> 32U),
            static_cast<uint32_t>(stream2),
            static_cast<uint32_t>(stream2 >> 32U),
        };
        uint32_t subkey[8];
        hchacha20(base_key, hnonce, subkey);
        for (uint32_t i = 0; i < 8; ++i)
        {
            rng.state[4 + i] = subkey[i];
        }

        rng.state[12] = static_cast<uint32_t>(stream0);
        rng.state[13] = static_cast<uint32_t>(stream0 >> 32U);
        rng.state[14] = static_cast<uint32_t>(stream1);
        rng.state[15] = static_cast<uint32_t>(stream1 >> 32U);

        rng.block_idx = 8U;
    }

    __device__ __forceinline__ void rng_refill(DeviceChaChaRng &rng)
    {
        chacha20_block(rng.state, rng.block);
        rng.state[12] += 1U;
        if (rng.state[12] == 0U)
        {
            rng.state[13] += 1U;
        }
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
