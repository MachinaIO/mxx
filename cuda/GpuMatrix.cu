#include "GpuMatrix.h"
#include "GpuChaCha.cuh"
#include "GpuPolyInternal.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{
    enum class BlockOp
    {
        Add,
        Sub,
        Mul,
    };

    constexpr int kMatmulTileM = 16;
    constexpr int kMatmulTileN = 16;
    constexpr int kMatmulTileK = 8;
    constexpr uint32_t kGaussMaxDigits = 64;
    constexpr double kTwoPi = 6.283185307179586476925286766559;

    int set_error(const char *msg)
    {
        return gpu_set_last_error(msg);
    }

    int set_error(cudaError_t err)
    {
        return gpu_set_last_error(cudaGetErrorString(err));
    }

    int default_batch(const GpuContext *ctx)
    {
        return static_cast<int>(ctx && ctx->batch != 0 ? ctx->batch : 1);
    }

    bool parse_format(int format, PolyFormat &out)
    {
        switch (format)
        {
        case GPU_POLY_FORMAT_COEFF:
            out = PolyFormat::Coeff;
            return true;
        case GPU_POLY_FORMAT_EVAL:
            out = PolyFormat::Eval;
            return true;
        default:
            return false;
        }
    }

    int sync_poly_limb_streams(const GpuPoly *poly, const char *context)
    {
        if (!poly || !poly->ctx || !poly->poly)
        {
            return set_error(context);
        }
        const int level = poly->level;
        if (level < 0)
        {
            return set_error(context);
        }
        auto &limb_map = poly->ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error(context);
        }
        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error(context);
            }
            const auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error(context);
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            const auto &limb_impl = partition.limb[limb_id.y];
            cudaStream_t stream = nullptr;
            if (limb_impl.index() == FIDESlib::U64)
            {
                stream = std::get<FIDESlib::U64>(limb_impl).stream.ptr;
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                stream = std::get<FIDESlib::U32>(limb_impl).stream.ptr;
            }
            else
            {
                return set_error(context);
            }
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int sync_poly_partition_streams(const GpuPoly *poly, const char *context)
    {
        if (!poly || !poly->poly)
        {
            return set_error(context);
        }
        for (const auto &partition : poly->poly->GPU)
        {
            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaStreamSynchronize(partition.s.ptr);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    uint32_t bit_width_u64(uint64_t v)
    {
        if (v == 0)
        {
            return 0;
        }
        return static_cast<uint32_t>(64 - __builtin_clzll(v));
    }

    __host__ __device__ __forceinline__ size_t matrix_index(size_t row, size_t col, size_t cols)
    {
        return row * cols + col;
    }

    template <typename T>
    __global__ void block_add_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        T sum = a + b;
        sum = sum >= modulus ? (sum - modulus) : sum;
        out[poly_idx][coeff_idx] = sum;
    }

    template <typename T>
    __global__ void block_sub_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        T diff = a >= b ? (a - b) : (modulus - (b - a));
        out[poly_idx][coeff_idx] = diff;
    }

    __device__ __forceinline__ uint32_t mul_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
    {
        uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
        return static_cast<uint32_t>(prod % mod);
    }

    __device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
    {
        unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % mod);
    }

    __device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b, uint32_t mod)
    {
        uint64_t sum = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
        if (sum >= mod)
        {
            sum -= mod;
        }
        return static_cast<uint32_t>(sum);
    }

    __device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t mod)
    {
        unsigned __int128 sum = static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b);
        if (sum >= mod)
        {
            sum -= mod;
        }
        return static_cast<uint64_t>(sum);
    }

    template <typename T>
    __global__ void matrix_decompose_kernel(
        const T **src,
        T **dst,
        size_t poly_count,
        size_t n,
        uint32_t shift,
        T mask,
        T out_modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T residue = src[poly_idx][coeff_idx];
        T digit = shift >= static_cast<uint32_t>(sizeof(T) * 8) ? 0 : ((residue >> shift) & mask);
        if (out_modulus != 0 && digit >= out_modulus)
        {
            digit %= out_modulus;
        }
        dst[poly_idx][coeff_idx] = digit;
    }

    using gpu_chacha::DeviceChaChaRng;
    using gpu_chacha::rng_init;
    using gpu_chacha::rng_next_u64;

    __device__ __forceinline__ double uniform_open01(DeviceChaChaRng &rng)
    {
        constexpr double kScale = 1.0 / 9007199254740992.0; // 2^53
        double u = static_cast<double>(rng_next_u64(rng) >> 11U) * kScale;
        if (u <= 0.0)
        {
            u = kScale;
        }
        else if (u >= 1.0)
        {
            u = 1.0 - kScale;
        }
        return u;
    }

    __device__ __forceinline__ double sample_standard_normal(DeviceChaChaRng &rng)
    {
        double u1 = uniform_open01(rng);
        double u2 = uniform_open01(rng);
        double r = sqrt(-2.0 * log(u1));
        double theta = kTwoPi * u2;
        return r * cos(theta);
    }

    __device__ __forceinline__ bool karney_algorithm_h(DeviceChaChaRng &rng)
    {
        double h_a = uniform_open01(rng);
        if (!(h_a < 0.5))
        {
            return true;
        }
        for (;;)
        {
            double h_b = uniform_open01(rng);
            if (!(h_b < h_a))
            {
                return false;
            }
            h_a = uniform_open01(rng);
            if (!(h_a < h_b))
            {
                return true;
            }
        }
    }

    __device__ __forceinline__ int32_t karney_algorithm_g(DeviceChaChaRng &rng)
    {
        int32_t n = 0;
        while (karney_algorithm_h(rng))
        {
            ++n;
            if (n > 1024)
            {
                break;
            }
        }
        return n;
    }

    __device__ __forceinline__ bool karney_algorithm_p(DeviceChaChaRng &rng, int32_t n)
    {
        while (n-- && karney_algorithm_h(rng))
        {
        }
        return n < 0;
    }

    __device__ __forceinline__ bool karney_algorithm_b(DeviceChaChaRng &rng, int32_t k, double x)
    {
        double y = x;
        int32_t n = 0;
        double m = static_cast<double>(2 * k + 2);
        for (;; ++n)
        {
            double z = uniform_open01(rng);
            if (!(z < y))
            {
                break;
            }
            double r = uniform_open01(rng);
            if (!(r < (2.0 * static_cast<double>(k) + x) / m))
            {
                break;
            }
            y = z;
            if (n > 4096)
            {
                break;
            }
        }
        return (n % 2) == 0;
    }

    __device__ __forceinline__ int64_t sample_integer_karney(DeviceChaChaRng &rng, double mean, double stddev)
    {
        if (!(stddev > 0.0) || !isfinite(mean) || !isfinite(stddev))
        {
            return static_cast<int64_t>(llround(mean));
        }

        int64_t ceil_std = static_cast<int64_t>(ceil(stddev));
        if (ceil_std <= 0)
        {
            return static_cast<int64_t>(llround(mean));
        }

        for (int iter = 0; iter < 1 << 16; ++iter)
        {
            int32_t k = karney_algorithm_g(rng);
            if (!karney_algorithm_p(rng, k * (k - 1)))
            {
                continue;
            }

            int64_t s = (rng_next_u64(rng) & 1ULL) ? 1 : -1;
            double di0 = stddev * static_cast<double>(k) + static_cast<double>(s) * mean;
            int64_t i0 = static_cast<int64_t>(ceil(di0));
            double x0 = (static_cast<double>(i0) - di0) / stddev;
            int64_t j = static_cast<int64_t>(rng_next_u64(rng) % static_cast<uint64_t>(ceil_std));
            double x = x0 + static_cast<double>(j) / stddev;

            if (!(x < 1.0) || (x == 0.0 && s < 0 && k == 0))
            {
                continue;
            }

            int32_t h = k + 1;
            while (h-- > 0 && karney_algorithm_b(rng, k, x))
            {
            }
            if (h >= 0)
            {
                continue;
            }

            return s * (i0 + j);
        }

        // Fallback in case the rejection loop takes too long.
        return static_cast<int64_t>(llround(mean + stddev * sample_standard_normal(rng)));
    }

    __device__ __forceinline__ void get_base_digits_u64(
        uint64_t value,
        uint64_t base,
        uint32_t digits,
        int64_t *out_digits)
    {
        for (uint32_t i = 0; i < digits; ++i)
        {
            out_digits[i] = static_cast<int64_t>(value % base);
            value /= base;
        }
    }

    __device__ __forceinline__ uint64_t signed_mod_i64(int64_t value, uint64_t modulus)
    {
        if (modulus == 0)
        {
            return 0;
        }
        if (value >= 0)
        {
            return static_cast<uint64_t>(value) % modulus;
        }
        uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
        uint64_t rem = magnitude % modulus;
        return rem == 0 ? 0 : (modulus - rem);
    }

    __device__ __forceinline__ uint64_t sample_uniform_mod(DeviceChaChaRng &rng, uint64_t modulus)
    {
        if (modulus == 0)
        {
            return 0;
        }
        constexpr uint64_t kU64Max = ~uint64_t{0};
        const uint64_t threshold = kU64Max - (kU64Max % modulus);
        for (;;)
        {
            uint64_t x = rng_next_u64(rng);
            if (x < threshold)
            {
                return x % modulus;
            }
        }
    }

    __device__ __forceinline__ int64_t centered_residue_i64(uint64_t value, uint64_t modulus)
    {
        if (modulus == 0)
        {
            return 0;
        }
        uint64_t reduced = value % modulus;
        uint64_t half = modulus >> 1;
        if (reduced <= half)
        {
            return static_cast<int64_t>(reduced);
        }
        uint64_t neg = modulus - reduced;
        return -static_cast<int64_t>(neg);
    }

    __global__ void matrix_sample_distribution_kernel(
        uint64_t **dst,
        size_t poly_count,
        size_t n,
        uint64_t modulus,
        int dist_type,
        double sigma,
        uint32_t limb_idx,
        uint64_t seed)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;

        uint64_t sample = 0;
        if (dist_type == GPU_MATRIX_DIST_UNIFORM)
        {
            DeviceChaChaRng rng;
            rng_init(
                rng,
                seed,
                static_cast<uint64_t>(poly_idx + 1),
                static_cast<uint64_t>(coeff_idx + 1),
                static_cast<uint64_t>(limb_idx + 1),
                0x6f70656e66686531ULL);
            sample = sample_uniform_mod(rng, modulus);
        }
        else if (dist_type == GPU_MATRIX_DIST_GAUSS)
        {
            DeviceChaChaRng rng;
            rng_init(
                rng,
                seed,
                static_cast<uint64_t>(poly_idx + 1),
                static_cast<uint64_t>(coeff_idx + 1),
                0,
                0x6f70656e66686532ULL);
            int64_t z = sample_integer_karney(rng, 0.0, sigma);
            sample = signed_mod_i64(z, modulus);
        }
        else if (dist_type == GPU_MATRIX_DIST_BIT)
        {
            DeviceChaChaRng rng;
            rng_init(
                rng,
                seed,
                static_cast<uint64_t>(poly_idx + 1),
                static_cast<uint64_t>(coeff_idx + 1),
                0,
                0x6f70656e66686533ULL);
            sample = (rng_next_u64(rng) & 1ULL) % modulus;
        }
        else if (dist_type == GPU_MATRIX_DIST_TERNARY)
        {
            DeviceChaChaRng rng;
            rng_init(
                rng,
                seed,
                static_cast<uint64_t>(poly_idx + 1),
                static_cast<uint64_t>(coeff_idx + 1),
                0,
                0x6f70656e66686534ULL);
            uint64_t pick = rng_next_u64(rng) % 3ULL;
            int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
            sample = signed_mod_i64(z, modulus);
        }

        dst[poly_idx][coeff_idx] = sample;
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

    __global__ void matrix_fill_gadget_kernel(
        uint64_t **dst,
        size_t poly_count,
        size_t n,
        uint64_t modulus,
        size_t rows,
        size_t cols,
        size_t log_base_q,
        uint32_t digits_per_tower,
        uint32_t limb_idx,
        uint32_t base_bits)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;

        uint64_t value = 0;
        if (coeff_idx == 0 && rows > 0 && cols > 0 && log_base_q > 0)
        {
            size_t row = poly_idx / cols;
            size_t col = poly_idx - row * cols;
            size_t block_start = row * log_base_q;
            if (col >= block_start && col < block_start + log_base_q)
            {
                size_t local = col - block_start;
                uint32_t tower = static_cast<uint32_t>(local / static_cast<size_t>(digits_per_tower));
                uint32_t digit = static_cast<uint32_t>(local % static_cast<size_t>(digits_per_tower));
                if (tower == limb_idx)
                {
                    uint64_t base = uint64_t{1} << base_bits;
                    value = pow_mod_u64(base, digit, modulus);
                }
            }
        }
        dst[poly_idx][coeff_idx] = value;
    }

    __global__ void matrix_sample_p1_full_kernel(
        const uint64_t **a_entries,
        const uint64_t **b_entries,
        const uint64_t **d_entries,
        const uint64_t **tp2_entries,
        uint64_t **out_entries,
        size_t d,
        size_t cols,
        size_t n,
        size_t sample_start,
        size_t sample_count,
        double *cov_workspace,
        double *mean_workspace,
        double *col_workspace,
        int64_t *sampled_workspace,
        uint64_t modulus,
        double sigma,
        double s,
        double dgg_stddev,
        uint32_t limb_idx,
        uint64_t seed)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= sample_count)
        {
            return;
        }

        const size_t m = d * 2;
        if (m == 0)
        {
            return;
        }

        const size_t sample_idx = sample_start + idx;
        const size_t col_idx = sample_idx / n;
        const size_t coeff_idx = sample_idx - col_idx * n;
        const size_t cov_stride = m * m;
        const size_t vec_stride = m;
        double *cov = cov_workspace + idx * cov_stride;
        double *mean = mean_workspace + idx * vec_stride;
        double *col_buf = col_workspace + idx * vec_stride;
        int64_t *sampled = sampled_workspace + idx * vec_stride;

        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(col_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            static_cast<uint64_t>(limb_idx + 1),
            0x7065727475726231ULL);

        const double sigma2 = sigma * sigma;
        const double s2 = s * s;
        const double denom = s2 - sigma2;
        if (!(denom > 0.0))
        {
            return;
        }
        const double c_scale = -sigma2 / denom;
        const double fallback_var = dgg_stddev * dgg_stddev;
        const double eps = 1e-9;

        for (size_t i = 0; i < d; ++i)
        {
            for (size_t j = 0; j < d; ++j)
            {
                const size_t ij = matrix_index(i, j, d);
                const size_t ji = matrix_index(j, i, d);
                const double a_ij = static_cast<double>(centered_residue_i64(a_entries[ij][coeff_idx], modulus));
                const double d_ij = static_cast<double>(centered_residue_i64(d_entries[ij][coeff_idx], modulus));
                const double b_ij = static_cast<double>(centered_residue_i64(b_entries[ij][coeff_idx], modulus));
                const double b_ji = static_cast<double>(centered_residue_i64(b_entries[ji][coeff_idx], modulus));

                const double af = -sigma2 * a_ij + (i == j ? s2 : 0.0);
                const double df = -sigma2 * d_ij + (i == j ? s2 : 0.0);
                const double bf = -sigma2 * b_ij;
                const double bt = -sigma2 * b_ji;

                cov[matrix_index(i, j, m)] = af;
                cov[matrix_index(i + d, j + d, m)] = df;
                cov[matrix_index(i, j + d, m)] = bf;
                cov[matrix_index(i + d, j, m)] = bt;
            }
        }

        for (size_t row = 0; row < m; ++row)
        {
            const size_t tp_idx = matrix_index(row, col_idx, cols);
            const double c_centered = static_cast<double>(centered_residue_i64(tp2_entries[tp_idx][coeff_idx], modulus));
            mean[row] = c_scale * c_centered;
        }

        for (int t = static_cast<int>(m) - 1; t >= 0; --t)
        {
            const size_t tt = static_cast<size_t>(t);
            double var = cov[matrix_index(tt, tt, m)];
            if (!(var > eps))
            {
                var = fallback_var;
            }
            const double mu = mean[tt];
            const int64_t z = sample_integer_karney(rng, mu, sqrt(var));
            sampled[tt] = z;

            if (t == 0)
            {
                break;
            }

            const double delta = static_cast<double>(z) - mu;
            for (int i = 0; i < t; ++i)
            {
                col_buf[static_cast<size_t>(i)] =
                    cov[matrix_index(static_cast<size_t>(i), tt, m)];
            }

            for (int i = 0; i < t; ++i)
            {
                mean[static_cast<size_t>(i)] +=
                    (col_buf[static_cast<size_t>(i)] / var) * delta;
            }

            for (int i = 0; i < t; ++i)
            {
                for (int j = 0; j <= i; ++j)
                {
                    double updated = cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                                     (col_buf[static_cast<size_t>(i)] * col_buf[static_cast<size_t>(j)] / var);
                    cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                    cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
                }
            }
        }

        for (size_t row = 0; row < m; ++row)
        {
            const size_t out_idx = matrix_index(row, col_idx, cols);
            out_entries[out_idx][coeff_idx] = signed_mod_i64(sampled[row], modulus);
        }
    }

    __global__ void matrix_gauss_samp_gq_arb_base_kernel(
        const uint64_t **src,
        uint64_t **dst,
        size_t poly_count,
        size_t n,
        uint64_t tower_modulus,
        uint32_t base_bits,
        uint32_t digits_per_tower,
        uint32_t digit_idx,
        double c,
        uint32_t tower_idx,
        uint64_t seed,
        uint64_t out_modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }

        if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
        {
            return;
        }

        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        uint64_t value = src[poly_idx][coeff_idx];
        if (tower_modulus != 0)
        {
            value %= tower_modulus;
        }

        uint64_t base = uint64_t{1} << base_bits;
        double base_f = static_cast<double>(base);
        double sigma = c / (base_f + 1.0);

        int64_t m_digits[kGaussMaxDigits];
        int64_t v_digits[kGaussMaxDigits];
        double l[kGaussMaxDigits];
        double h[kGaussMaxDigits];
        double c_vec[kGaussMaxDigits];
        double p[kGaussMaxDigits];
        double a[kGaussMaxDigits];
        double zf[kGaussMaxDigits];
        int64_t z[kGaussMaxDigits];

        get_base_digits_u64(tower_modulus, base, digits_per_tower, m_digits);
        get_base_digits_u64(value, base, digits_per_tower, v_digits);

        const double kf = static_cast<double>(digits_per_tower);
        l[0] = sqrt(base_f * (1.0 + 1.0 / kf) + 1.0);
        for (uint32_t i = 1; i < digits_per_tower; ++i)
        {
            l[i] = sqrt(base_f * (1.0 + 1.0 / (kf - static_cast<double>(i))));
        }

        h[0] = 0.0;
        for (uint32_t i = 1; i < digits_per_tower; ++i)
        {
            h[i] = sqrt(base_f * (1.0 - 1.0 / (kf - static_cast<double>(i - 1))));
        }

        c_vec[0] = static_cast<double>(m_digits[0]) / base_f;
        for (uint32_t i = 1; i < digits_per_tower; ++i)
        {
            c_vec[i] = (c_vec[i - 1] + static_cast<double>(m_digits[i])) / base_f;
        }

        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(tower_idx + 1),
            static_cast<uint64_t>(poly_idx + 1),
            static_cast<uint64_t>(coeff_idx + 1),
            0x6761646765746731ULL);

        for (uint32_t i = 0; i < digits_per_tower; ++i)
        {
            zf[i] = sigma * sample_standard_normal(rng);
        }
        for (uint32_t i = 0; i + 1 < digits_per_tower; ++i)
        {
            p[i] = l[i] * zf[i] + h[i + 1] * zf[i + 1];
        }
        p[digits_per_tower - 1] = h[digits_per_tower - 1] * zf[digits_per_tower - 1];

        a[0] = (static_cast<double>(v_digits[0]) - p[0]) / base_f;
        for (uint32_t t = 1; t < digits_per_tower; ++t)
        {
            a[t] = (a[t - 1] + static_cast<double>(v_digits[t]) - p[t]) / base_f;
        }

        const uint32_t last = digits_per_tower - 1;
        z[last] = sample_integer_karney(rng, -a[last] / c_vec[last], sigma / c_vec[last]);
        for (uint32_t i = 0; i < digits_per_tower; ++i)
        {
            a[i] += static_cast<double>(z[last]) * c_vec[i];
        }
        for (uint32_t i = 0; i < last; ++i)
        {
            z[i] = sample_integer_karney(rng, -a[i], sigma);
        }

        int64_t out_digit = 0;
        if (digits_per_tower == 1)
        {
            out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[0] + v_digits[0];
        }
        else if (digit_idx == 0)
        {
            out_digit = static_cast<int64_t>(base) * z[0] + m_digits[0] * z[last] + v_digits[0];
        }
        else if (digit_idx < last)
        {
            out_digit = static_cast<int64_t>(base) * z[digit_idx] - z[digit_idx - 1] +
                        m_digits[digit_idx] * z[last] + v_digits[digit_idx];
        }
        else
        {
            out_digit = m_digits[last] * z[last] - z[last - 1] + v_digits[last];
        }

        dst[poly_idx][coeff_idx] = signed_mod_i64(out_digit, out_modulus);
    }

    template <typename T>
    __global__ void block_mul_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t poly_count,
        size_t n,
        T modulus)
    {
        size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        size_t total = poly_count * n;
        if (idx >= total)
        {
            return;
        }
        size_t poly_idx = idx / n;
        size_t coeff_idx = idx - poly_idx * n;
        T a = lhs[poly_idx][coeff_idx];
        T b = rhs[poly_idx][coeff_idx];
        if constexpr (std::is_same_v<T, uint64_t>)
        {
            out[poly_idx][coeff_idx] = mul_mod_u64(a, b, modulus);
        }
        else
        {
            out[poly_idx][coeff_idx] = mul_mod_u32(a, b, modulus);
        }
    }

    template <typename T>
    __global__ void block_matmul_kernel(
        const T **lhs,
        const T **rhs,
        T **out,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        T modulus)
    {
        __shared__ T lhs_tile[kMatmulTileM][kMatmulTileK];
        __shared__ T rhs_tile[kMatmulTileK][kMatmulTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t coeff_idx = static_cast<size_t>(blockIdx.z);
        if (coeff_idx >= n)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;

        T acc = 0;
        for (size_t k0 = 0; k0 < inner; k0 += kMatmulTileK)
        {
            for (int i = tid; i < kMatmulTileM * kMatmulTileK; i += threads)
            {
                const int r = i / kMatmulTileK;
                const int k = i - r * kMatmulTileK;
                const size_t lhs_row = row_base + static_cast<size_t>(r);
                const size_t lhs_k = k0 + static_cast<size_t>(k);
                T val = 0;
                if (lhs_row < rows && lhs_k < inner)
                {
                    const T *lhs_poly = lhs[lhs_row * inner + lhs_k];
                    val = lhs_poly[coeff_idx];
                }
                lhs_tile[r][k] = val;
            }
            for (int i = tid; i < kMatmulTileK * kMatmulTileN; i += threads)
            {
                const int k = i / kMatmulTileN;
                const int c = i - k * kMatmulTileN;
                const size_t rhs_k = k0 + static_cast<size_t>(k);
                const size_t rhs_col = col_base + static_cast<size_t>(c);
                T val = 0;
                if (rhs_k < inner && rhs_col < cols)
                {
                    const T *rhs_poly = rhs[rhs_k * cols + rhs_col];
                    val = rhs_poly[coeff_idx];
                }
                rhs_tile[k][c] = val;
            }
            __syncthreads();

            if (row < rows && col < cols)
            {
                for (int kk = 0; kk < kMatmulTileK; ++kk)
                {
                    T prod;
                    if constexpr (std::is_same_v<T, uint64_t>)
                    {
                        prod = mul_mod_u64(lhs_tile[threadIdx.y][kk], rhs_tile[kk][threadIdx.x], modulus);
                        acc = add_mod_u64(acc, prod, modulus);
                    }
                    else
                    {
                        prod = mul_mod_u32(lhs_tile[threadIdx.y][kk], rhs_tile[kk][threadIdx.x], modulus);
                        acc = add_mod_u32(acc, prod, modulus);
                    }
                }
            }
            __syncthreads();
        }

        if (row < rows && col < cols)
        {
            out[row * cols + col][coeff_idx] = acc;
        }
    }

    template <typename T>
    int launch_block_kernel(
        const std::vector<T *> &out_ptrs,
        const std::vector<const T *> &lhs_ptrs,
        const std::vector<const T *> &rhs_ptrs,
        size_t n,
        T modulus,
        BlockOp op,
        cudaStream_t stream)
    {
        const size_t count = out_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }

        T **d_out = nullptr;
        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        const size_t bytes = count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_out, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_lhs, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            return set_error(err);
        }
        err = cudaMalloc(&d_rhs, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_out, out_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        switch (op)
        {
        case BlockOp::Add:
            block_add_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        case BlockOp::Sub:
            block_sub_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        case BlockOp::Mul:
            block_mul_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, count, n, modulus);
            break;
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        cudaFree(d_out);
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        return 0;
    }

    template <typename T>
    int launch_block_matmul_kernel(
        const std::vector<T *> &out_ptrs,
        const std::vector<const T *> &lhs_ptrs,
        const std::vector<const T *> &rhs_ptrs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        T modulus,
        cudaStream_t stream,
        double *out_kernel_ms)
    {
        const size_t out_count = rows * cols;
        const size_t lhs_count = rows * inner;
        const size_t rhs_count = inner * cols;
        if (out_count == 0 || n == 0)
        {
            return 0;
        }
        if (out_ptrs.size() != out_count || lhs_ptrs.size() != lhs_count || rhs_ptrs.size() != rhs_count)
        {
            return set_error("unexpected pointer counts in gpu_block_mul");
        }

        T **d_out = nullptr;
        const T **d_lhs = nullptr;
        const T **d_rhs = nullptr;
        const size_t out_bytes = out_count * sizeof(T *);
        const size_t lhs_bytes = lhs_count * sizeof(T *);
        const size_t rhs_bytes = rhs_count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_out, out_bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_lhs, lhs_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            return set_error(err);
        }
        err = cudaMalloc(&d_rhs, rhs_bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_out, out_ptrs.data(), out_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_lhs, lhs_ptrs.data(), lhs_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_rhs, rhs_ptrs.data(), rhs_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        const dim3 threads(kMatmulTileN, kMatmulTileM);
        const dim3 blocks(
            static_cast<unsigned int>((cols + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(n));

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (out_kernel_ms)
        {
            err = cudaEventCreate(&start);
            if (err != cudaSuccess)
            {
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventCreate(&stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventRecord(start, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
        }

        block_matmul_kernel<<<blocks, threads, 0, stream>>>(d_lhs, d_rhs, d_out, rows, inner, cols, n, modulus);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            if (start)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            cudaFree(d_out);
            cudaFree(d_lhs);
            cudaFree(d_rhs);
            return set_error(err);
        }

        if (out_kernel_ms)
        {
            err = cudaEventRecord(stop, stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            err = cudaEventSynchronize(stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            float kernel_ms = 0.0f;
            err = cudaEventElapsedTime(&kernel_ms, start, stop);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
            *out_kernel_ms += static_cast<double>(kernel_ms);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        else
        {
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
            {
                cudaFree(d_out);
                cudaFree(d_lhs);
                cudaFree(d_rhs);
                return set_error(err);
            }
        }

        cudaFree(d_out);
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        return 0;
    }

    template <typename T>
    int launch_decompose_kernel(
        const std::vector<const T *> &src_ptrs,
        const std::vector<T *> &dst_ptrs,
        size_t n,
        uint32_t shift,
        T mask,
        T out_modulus,
        cudaStream_t stream)
    {
        const size_t count = src_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (dst_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in matrix_decompose_kernel");
        }

        const T **d_src = nullptr;
        T **d_dst = nullptr;
        const size_t bytes = count * sizeof(T *);

        cudaError_t err = cudaMalloc(&d_src, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMalloc(&d_dst, bytes);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_src, src_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        matrix_decompose_kernel<<<blocks, threads, 0, stream>>>(
            d_src,
            d_dst,
            count,
            n,
            shift,
            mask,
            out_modulus);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_src);
            cudaFree(d_dst);
            return set_error(err);
        }

        cudaFree(d_src);
        cudaFree(d_dst);
        return 0;
    }

    int launch_gauss_samp_gq_arb_base_kernel(
        const std::vector<const uint64_t *> &src_ptrs,
        const std::vector<uint64_t *> &dst_ptrs,
        size_t n,
        uint64_t tower_modulus,
        uint32_t base_bits,
        uint32_t digits_per_tower,
        uint32_t digit_idx,
        double c,
        uint32_t tower_idx,
        uint64_t seed,
        uint64_t out_modulus,
        int device,
        cudaStream_t stream)
    {
        const size_t count = src_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }
        if (dst_ptrs.size() != count)
        {
            return set_error("unexpected pointer counts in matrix_gauss_samp_gq_arb_base_kernel");
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const uint64_t **d_src = nullptr;
        uint64_t **d_dst = nullptr;
        const size_t bytes = count * sizeof(uint64_t *);

        err = cudaMallocAsync(&d_src, bytes, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMallocAsync(&d_dst, bytes, stream);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(d_src, stream);
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_src, src_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(d_src, stream);
            cudaFreeAsync(d_dst, stream);
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFreeAsync(d_src, stream);
            cudaFreeAsync(d_dst, stream);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        matrix_gauss_samp_gq_arb_base_kernel<<<blocks, threads, 0, stream>>>(
            d_src,
            d_dst,
            count,
            n,
            tower_modulus,
            base_bits,
            digits_per_tower,
            digit_idx,
            c,
            tower_idx,
            seed,
            out_modulus);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(d_src, stream);
            cudaFreeAsync(d_dst, stream);
            return set_error(err);
        }

        err = cudaFreeAsync(d_src, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaFreeAsync(d_dst, stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        return 0;
    }

    int launch_sample_distribution_kernel(
        const std::vector<uint64_t *> &dst_ptrs,
        size_t n,
        uint64_t modulus,
        int dist_type,
        double sigma,
        uint32_t limb_idx,
        uint64_t seed,
        cudaStream_t stream)
    {
        const size_t count = dst_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }

        uint64_t **d_dst = nullptr;
        const size_t bytes = count * sizeof(uint64_t *);
        cudaError_t err = cudaMalloc(&d_dst, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        matrix_sample_distribution_kernel<<<blocks, threads, 0, stream>>>(
            d_dst,
            count,
            n,
            modulus,
            dist_type,
            sigma,
            limb_idx,
            seed);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        cudaFree(d_dst);
        return 0;
    }

    int launch_fill_gadget_kernel(
        const std::vector<uint64_t *> &dst_ptrs,
        size_t n,
        uint64_t modulus,
        size_t rows,
        size_t cols,
        size_t log_base_q,
        uint32_t digits_per_tower,
        uint32_t limb_idx,
        uint32_t base_bits,
        cudaStream_t stream)
    {
        const size_t count = dst_ptrs.size();
        if (count == 0 || n == 0)
        {
            return 0;
        }

        uint64_t **d_dst = nullptr;
        const size_t bytes = count * sizeof(uint64_t *);
        cudaError_t err = cudaMalloc(&d_dst, bytes);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_dst, dst_ptrs.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        const int threads = 256;
        const size_t total = count * n;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        matrix_fill_gadget_kernel<<<blocks, threads, 0, stream>>>(
            d_dst,
            count,
            n,
            modulus,
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            limb_idx,
            base_bits);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_dst);
            return set_error(err);
        }

        cudaFree(d_dst);
        return 0;
    }

    int launch_sample_p1_full_kernel(
        const std::vector<const uint64_t *> &a_entries,
        const std::vector<const uint64_t *> &b_entries,
        const std::vector<const uint64_t *> &d_entries,
        const std::vector<const uint64_t *> &tp2_entries,
        const std::vector<uint64_t *> &out_entries,
        size_t d,
        size_t cols,
        size_t n,
        uint64_t modulus,
        double sigma,
        double s,
        double dgg_stddev,
        uint32_t limb_idx,
        uint64_t seed,
        cudaStream_t stream,
        int device_id)
    {
        if (d == 0 || cols == 0 || n == 0)
        {
            return 0;
        }
        const size_t mat_entries = d * d;
        const size_t vec_entries = 2 * d * cols;
        if (a_entries.size() != mat_entries || b_entries.size() != mat_entries ||
            d_entries.size() != mat_entries || tp2_entries.size() != vec_entries ||
            out_entries.size() != vec_entries)
        {
            return set_error("unexpected pointer counts in matrix_sample_p1_full_kernel");
        }

        if (device_id < 0)
        {
            return set_error("invalid device in matrix_sample_p1_full_kernel");
        }
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const size_t m = 2 * d;
        if (m == 0)
        {
            return set_error("invalid dimension in matrix_sample_p1_full_kernel");
        }
        if (m > std::numeric_limits<size_t>::max() / m)
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        const size_t cov_elems_per_sample = m * m;
        if (cov_elems_per_sample > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        const size_t cov_bytes_per_sample = cov_elems_per_sample * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        const size_t vec_bytes_per_sample = m * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
        if (cov_bytes_per_sample > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        size_t bytes_per_sample_total = cov_bytes_per_sample + vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        bytes_per_sample_total += vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
        {
            return set_error("workspace overflow in matrix_sample_p1_full_kernel");
        }
        bytes_per_sample_total += sampled_bytes_per_sample;

        const size_t total_samples = cols * n;
        constexpr size_t kWorkspaceBudgetBytes = 128ULL * 1024ULL * 1024ULL;
        size_t chunk_samples = kWorkspaceBudgetBytes / bytes_per_sample_total;
        if (chunk_samples == 0)
        {
            chunk_samples = 1;
        }
        chunk_samples = std::min(chunk_samples, total_samples);

        const uint64_t **d_a_entries = nullptr;
        const uint64_t **d_b_entries = nullptr;
        const uint64_t **d_d_entries = nullptr;
        const uint64_t **d_tp2_entries = nullptr;
        uint64_t **d_out_entries = nullptr;
        const size_t mat_bytes = mat_entries * sizeof(uint64_t *);
        const size_t vec_bytes = vec_entries * sizeof(uint64_t *);

        auto free_all = [&]() {
            if (d_a_entries)
                cudaFree(d_a_entries);
            if (d_b_entries)
                cudaFree(d_b_entries);
            if (d_d_entries)
                cudaFree(d_d_entries);
            if (d_tp2_entries)
                cudaFree(d_tp2_entries);
            if (d_out_entries)
                cudaFree(d_out_entries);
        };

        err = cudaMalloc(&d_a_entries, mat_bytes);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMalloc(&d_b_entries, mat_bytes);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMalloc(&d_d_entries, mat_bytes);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMalloc(&d_tp2_entries, vec_bytes);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMalloc(&d_out_entries, vec_bytes);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }

        err = cudaMemcpyAsync(d_a_entries, a_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_b_entries, b_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_d_entries, d_entries.data(), mat_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_tp2_entries, tp2_entries.data(), vec_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
        err = cudaMemcpyAsync(d_out_entries, out_entries.data(), vec_bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }

        double *cov_workspace = nullptr;
        double *mean_workspace = nullptr;
        double *col_workspace = nullptr;
        int64_t *sampled_workspace = nullptr;
        auto free_workspace = [&]() {
            if (cov_workspace)
                cudaFree(cov_workspace);
            if (mean_workspace)
                cudaFree(mean_workspace);
            if (col_workspace)
                cudaFree(col_workspace);
            if (sampled_workspace)
                cudaFree(sampled_workspace);
            cov_workspace = nullptr;
            mean_workspace = nullptr;
            col_workspace = nullptr;
            sampled_workspace = nullptr;
        };

        auto alloc_workspace = [&](size_t samples) -> bool {
            if (samples == 0)
            {
                return false;
            }
            if (samples > std::numeric_limits<size_t>::max() / cov_bytes_per_sample ||
                samples > std::numeric_limits<size_t>::max() / vec_bytes_per_sample ||
                samples > std::numeric_limits<size_t>::max() / sampled_bytes_per_sample)
            {
                return false;
            }
            cudaError_t local_err = cudaMalloc(&cov_workspace, samples * cov_bytes_per_sample);
            if (local_err != cudaSuccess)
            {
                free_workspace();
                return false;
            }
            local_err = cudaMalloc(&mean_workspace, samples * vec_bytes_per_sample);
            if (local_err != cudaSuccess)
            {
                free_workspace();
                return false;
            }
            local_err = cudaMalloc(&col_workspace, samples * vec_bytes_per_sample);
            if (local_err != cudaSuccess)
            {
                free_workspace();
                return false;
            }
            local_err = cudaMalloc(&sampled_workspace, samples * sampled_bytes_per_sample);
            if (local_err != cudaSuccess)
            {
                free_workspace();
                return false;
            }
            return true;
        };

        while (!alloc_workspace(chunk_samples))
        {
            if (chunk_samples <= 1)
            {
                free_all();
                return set_error("failed to allocate workspace in matrix_sample_p1_full_kernel");
            }
            chunk_samples = (chunk_samples + 1) / 2;
        }

        const int threads = 256;
        for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
        {
            size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
            const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
            matrix_sample_p1_full_kernel<<<blocks, threads, 0, stream>>>(
                d_a_entries,
                d_b_entries,
                d_d_entries,
                d_tp2_entries,
                d_out_entries,
                d,
                cols,
                n,
                sample_start,
                sample_count,
                cov_workspace,
                mean_workspace,
                col_workspace,
                sampled_workspace,
                modulus,
                sigma,
                s,
                dgg_stddev,
                limb_idx,
                seed);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                free_workspace();
                free_all();
                return set_error(err);
            }
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
        {
            free_workspace();
            free_all();
            return set_error(err);
        }

        free_workspace();
        free_all();
        return 0;
    }

    template <typename T>
    int launch_for_limb(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t count,
        size_t n,
        int limb,
        const dim3 &limb_id,
        BlockOp op)
    {
        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(count);
        lhs_ptrs.reserve(count);
        rhs_ptrs.reserve(count);

        cudaStream_t stream = nullptr;
        for (size_t i = 0; i < count; ++i)
        {
            auto &lhs_partition = lhs[i]->poly->GPU[limb_id.x];
            auto &rhs_partition = rhs[i]->poly->GPU[limb_id.x];
            auto &out_partition = out[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= lhs_partition.limb.size() || limb_id.y >= rhs_partition.limb.size() ||
                limb_id.y >= out_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_op");
            }
            auto &lhs_limb = lhs_partition.limb[limb_id.y];
            auto &rhs_limb = rhs_partition.limb[limb_id.y];
            auto &out_limb = out_partition.limb[limb_id.y];
            if (lhs_limb.index() != rhs_limb.index() || lhs_limb.index() != out_limb.index())
            {
                return set_error("mixed limb types in gpu_block_op");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &lhs_u = std::get<FIDESlib::U64>(lhs_limb);
                auto &rhs_u = std::get<FIDESlib::U64>(rhs_limb);
                auto &out_u = std::get<FIDESlib::U64>(out_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
                rhs_ptrs.push_back(rhs_u.v.data);
                out_ptrs.push_back(out_u.v.data);
            }
            else
            {
                auto &lhs_u = std::get<FIDESlib::U32>(lhs_limb);
                auto &rhs_u = std::get<FIDESlib::U32>(rhs_limb);
                auto &out_u = std::get<FIDESlib::U32>(out_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
                rhs_ptrs.push_back(rhs_u.v.data);
                out_ptrs.push_back(out_u.v.data);
            }
        }

        const uint64_t modulus64 = lhs[0]->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, n, modulus, op, stream);
    }

    template <typename T>
    int launch_for_limb_matmul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        size_t n,
        int limb,
        const dim3 &limb_id,
        double *out_kernel_ms)
    {
        const size_t out_count = rows * cols;
        const size_t lhs_count = rows * inner;
        const size_t rhs_count = inner * cols;

        std::vector<T *> out_ptrs;
        std::vector<const T *> lhs_ptrs;
        std::vector<const T *> rhs_ptrs;
        out_ptrs.reserve(out_count);
        lhs_ptrs.reserve(lhs_count);
        rhs_ptrs.reserve(rhs_count);

        cudaStream_t stream = nullptr;
        int limb_index = -1;

        for (size_t i = 0; i < lhs_count; ++i)
        {
            auto &lhs_partition = lhs[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= lhs_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &lhs_limb = lhs_partition.limb[limb_id.y];
            if constexpr (std::is_same_v<T, uint64_t>)
            {
                if (lhs_limb.index() != FIDESlib::U64)
                {
                    return set_error("mixed limb types in gpu_block_mul");
                }
                auto &lhs_u = std::get<FIDESlib::U64>(lhs_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
            }
            else
            {
                if (lhs_limb.index() != FIDESlib::U32)
                {
                    return set_error("mixed limb types in gpu_block_mul");
                }
                auto &lhs_u = std::get<FIDESlib::U32>(lhs_limb);
                if (!stream)
                {
                    stream = lhs_u.stream.ptr;
                }
                lhs_ptrs.push_back(lhs_u.v.data);
            }
            if (limb_index == -1)
            {
                limb_index = lhs_limb.index();
            }
        }

        for (size_t i = 0; i < rhs_count; ++i)
        {
            auto &rhs_partition = rhs[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= rhs_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &rhs_limb = rhs_partition.limb[limb_id.y];
            if (rhs_limb.index() != limb_index)
            {
                return set_error("mixed limb types in gpu_block_mul");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &rhs_u = std::get<FIDESlib::U64>(rhs_limb);
                rhs_ptrs.push_back(rhs_u.v.data);
            }
            else
            {
                auto &rhs_u = std::get<FIDESlib::U32>(rhs_limb);
                rhs_ptrs.push_back(rhs_u.v.data);
            }
        }

        for (size_t i = 0; i < out_count; ++i)
        {
            auto &out_partition = out[i]->poly->GPU[limb_id.x];
            if (limb_id.y >= out_partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }
            auto &out_limb = out_partition.limb[limb_id.y];
            if (out_limb.index() != limb_index)
            {
                return set_error("mixed limb types in gpu_block_mul");
            }

            if constexpr (std::is_same_v<T, uint64_t>)
            {
                auto &out_u = std::get<FIDESlib::U64>(out_limb);
                out_ptrs.push_back(out_u.v.data);
            }
            else
            {
                auto &out_u = std::get<FIDESlib::U32>(out_limb);
                out_ptrs.push_back(out_u.v.data);
            }
        }

        const uint64_t modulus64 = lhs[0]->ctx->moduli[static_cast<size_t>(limb)];
        const T modulus = static_cast<T>(modulus64);
        return launch_block_matmul_kernel(out_ptrs, lhs_ptrs, rhs_ptrs, rows, inner, cols, n, modulus, stream, out_kernel_ms);
    }

    int gpu_block_op(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count, BlockOp op)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid gpu_block_op arguments");
        }
        if (count == 0)
        {
            return 0;
        }

        const GpuPoly *lhs0 = lhs[0];
        if (!lhs0 || !lhs0->ctx)
        {
            return set_error("null context in gpu_block_op");
        }
        const GpuContext *ctx = lhs0->ctx;
        const int level = lhs0->level;
        const PolyFormat format = lhs0->format;
        if (op == BlockOp::Mul && format != PolyFormat::Eval)
        {
            return set_error("gpu_block_entrywise_mul requires Eval format");
        }

        // for (size_t i = 0; i < count; ++i)
        // {
        //     if (!out[i] || !lhs[i] || !rhs[i])
        //     {
        //         return set_error("null polynomial in gpu_block_op");
        //     }
        //     if (lhs[i]->ctx != ctx || rhs[i]->ctx != ctx || out[i]->ctx != ctx)
        //     {
        //         return set_error("mismatched contexts in gpu_block_op");
        //     }
        //     if (lhs[i]->level != level || rhs[i]->level != level || out[i]->level != level)
        //     {
        //         return set_error("mismatched levels in gpu_block_op");
        //     }
        //     if (lhs[i]->format != format || rhs[i]->format != format)
        //     {
        //         return set_error("mismatched formats in gpu_block_op");
        //     }
        // }

        if (level < 0)
        {
            return set_error("invalid level in gpu_block_op");
        }

        const int N = ctx->N;
        if (N <= 0)
        {
            return 0;
        }

        auto &limb_map = ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error("unexpected limb mapping size in gpu_block_op");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= lhs0->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_block_op");
            }
            const auto &partition = lhs0->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_op");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            const auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() == FIDESlib::U64)
            {
                int status = launch_for_limb<uint64_t>(out, lhs, rhs, count, static_cast<size_t>(N), limb, limb_id, op);
                if (status != 0)
                {
                    return status;
                }
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                int status = launch_for_limb<uint32_t>(out, lhs, rhs, count, static_cast<size_t>(N), limb, limb_id, op);
                if (status != 0)
                {
                    return status;
                }
            }
            else
            {
                return set_error("unsupported limb type in gpu_block_op");
            }
        }

        for (size_t i = 0; i < count; ++i)
        {
            out[i]->level = level;
            out[i]->format = format;
        }
        return 0;
    }

    int gpu_block_matmul(
        GpuPoly *const *out,
        const GpuPoly *const *lhs,
        const GpuPoly *const *rhs,
        size_t rows,
        size_t inner,
        size_t cols,
        double *out_kernel_ms = nullptr)
    {
        if (!out || !lhs || !rhs)
        {
            return set_error("invalid gpu_block_mul arguments");
        }
        if (out_kernel_ms)
        {
            *out_kernel_ms = 0.0;
        }
        const size_t out_count = rows * cols;
        if (rows == 0 || inner == 0 || cols == 0)
        {
            return 0;
        }
        if (out_count == 0)
        {
            return 0;
        }

        const GpuPoly *lhs0 = lhs[0];
        if (!lhs0 || !lhs0->ctx)
        {
            return set_error("null context in gpu_block_mul");
        }
        const GpuContext *ctx = lhs0->ctx;
        const int level = lhs0->level;
        const PolyFormat format = lhs0->format;
        if (format != PolyFormat::Eval)
        {
            return set_error("gpu_block_mul requires Eval format");
        }

        if (level < 0)
        {
            return set_error("invalid level in gpu_block_mul");
        }

        const int N = ctx->N;
        if (N <= 0)
        {
            return 0;
        }

        auto &limb_map = ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(level + 1))
        {
            return set_error("unexpected limb mapping size in gpu_block_mul");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= lhs0->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_block_mul");
            }
            const auto &partition = lhs0->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_block_mul");
            }

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }

            const auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() == FIDESlib::U64)
            {
                int status =
                    launch_for_limb_matmul<uint64_t>(
                        out,
                        lhs,
                        rhs,
                        rows,
                        inner,
                        cols,
                        static_cast<size_t>(N),
                        limb,
                        limb_id,
                        out_kernel_ms);
                if (status != 0)
                {
                    return status;
                }
            }
            else if (limb_impl.index() == FIDESlib::U32)
            {
                int status =
                    launch_for_limb_matmul<uint32_t>(
                        out,
                        lhs,
                        rhs,
                        rows,
                        inner,
                        cols,
                        static_cast<size_t>(N),
                        limb,
                        limb_id,
                        out_kernel_ms);
                if (status != 0)
                {
                    return status;
                }
            }
            else
            {
                return set_error("unsupported limb type in gpu_block_mul");
            }
        }

        for (size_t i = 0; i < out_count; ++i)
        {
            out[i]->level = level;
            out[i]->format = format;
        }
        return 0;
    }
} // namespace

extern "C" int gpu_block_add(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Add);
}

extern "C" int gpu_block_sub(GpuPoly *const *out, const GpuPoly *const *lhs, const GpuPoly *const *rhs, size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Sub);
}

extern "C" int gpu_block_entrywise_mul(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t count)
{
    return gpu_block_op(out, lhs, rhs, count, BlockOp::Mul);
}

extern "C" int gpu_block_mul(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t rows,
    size_t inner,
    size_t cols)
{
    return gpu_block_matmul(out, lhs, rhs, rows, inner, cols);
}

extern "C" int gpu_block_mul_timed(
    GpuPoly *const *out,
    const GpuPoly *const *lhs,
    const GpuPoly *const *rhs,
    size_t rows,
    size_t inner,
    size_t cols,
    double *out_kernel_ms)
{
    if (!out_kernel_ms)
    {
        return set_error("null out_kernel_ms in gpu_block_mul_timed");
    }
    return gpu_block_matmul(out, lhs, rhs, rows, inner, cols, out_kernel_ms);
}

extern "C" int gpu_matrix_create(
    GpuContext *ctx,
    int level,
    size_t rows,
    size_t cols,
    int format,
    GpuMatrix **out)
{
    if (!ctx || !out)
    {
        return set_error("invalid gpu_matrix_create arguments");
    }
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_create");
    }

    auto *mat = new GpuMatrix{ctx, rows, cols, level, fmt, {}};
    const size_t count = rows * cols;
    mat->polys.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        GpuPoly *poly = nullptr;
        int status = gpu_poly_create(ctx, level, &poly);
        if (status != 0)
        {
            for (auto *p : mat->polys)
            {
                gpu_poly_destroy(p);
            }
            delete mat;
            return status;
        }
        poly->format = fmt;
        mat->polys.push_back(poly);
    }
    *out = mat;
    return 0;
}

extern "C" void gpu_matrix_destroy(GpuMatrix *mat)
{
    if (!mat)
    {
        return;
    }
    for (auto *poly : mat->polys)
    {
        gpu_poly_destroy(poly);
    }
    delete mat;
}

extern "C" int gpu_matrix_copy(GpuMatrix *dst, const GpuMatrix *src)
{
    if (!dst || !src)
    {
        return set_error("invalid gpu_matrix_copy arguments");
    }
    if (dst->rows != src->rows || dst->cols != src->cols)
    {
        return set_error("size mismatch in gpu_matrix_copy");
    }
    if (dst->level != src->level || dst->ctx != src->ctx)
    {
        return set_error("context mismatch in gpu_matrix_copy");
    }
    const size_t count = src->rows * src->cols;
    for (size_t i = 0; i < count; ++i)
    {
        int status = gpu_poly_copy(dst->polys[i], src->polys[i]);
        if (status != 0)
        {
            return status;
        }
    }
    dst->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_entry_clone(
    const GpuMatrix *mat,
    size_t row,
    size_t col,
    GpuPoly **out_poly)
{
    if (!mat || !out_poly)
    {
        return set_error("invalid gpu_matrix_entry_clone arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_entry_clone");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_clone(mat->polys[idx], out_poly);
}

extern "C" int gpu_matrix_copy_entry(
    GpuMatrix *mat,
    size_t row,
    size_t col,
    const GpuPoly *src)
{
    if (!mat || !src)
    {
        return set_error("invalid gpu_matrix_copy_entry arguments");
    }
    if (row >= mat->rows || col >= mat->cols)
    {
        return set_error("index out of bounds in gpu_matrix_copy_entry");
    }
    if (src->ctx != mat->ctx || src->level != mat->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_entry");
    }
    const size_t idx = matrix_index(row, col, mat->cols);
    return gpu_poly_copy(mat->polys[idx], src);
}

extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format)
{
    if (!mat)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    const size_t count = mat->rows * mat->cols;
    int status = gpu_poly_load_rns_batch(
        mat->polys.data(),
        count,
        bytes,
        bytes_per_poly,
        format);
    if (status != 0)
    {
        return status;
    }
    PolyFormat fmt;
    if (!parse_format(format, fmt))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }
    mat->format = fmt;
    return 0;
}

extern "C" int gpu_matrix_store_rns_batch(
    const GpuMatrix *mat,
    uint8_t *bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    const size_t count = mat->rows * mat->cols;
    return gpu_poly_store_rns_batch(
        const_cast<GpuPoly *const *>(mat->polys.data()),
        count,
        bytes_out,
        bytes_per_poly,
        format,
        out_events);
}

extern "C" int gpu_matrix_add(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_add arguments");
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return set_error("size mismatch in gpu_matrix_add");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_add");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_add");
    }
    const size_t count = lhs->rows * lhs->cols;
    int status = gpu_block_add(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_sub(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_sub arguments");
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return set_error("size mismatch in gpu_matrix_sub");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_sub");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_sub");
    }
    const size_t count = lhs->rows * lhs->cols;
    int status = gpu_block_sub(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}

extern "C" int gpu_matrix_mul(GpuMatrix *out, const GpuMatrix *lhs, const GpuMatrix *rhs)
{
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_mul arguments");
    }
    if (lhs->cols != rhs->rows)
    {
        return set_error("size mismatch in gpu_matrix_mul");
    }
    if (out->rows != lhs->rows || out->cols != rhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul");
    }
    int status = gpu_block_mul(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        lhs->rows,
        lhs->cols,
        rhs->cols);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_equal(const GpuMatrix *lhs, const GpuMatrix *rhs, int *out_equal)
{
    if (!lhs || !rhs || !out_equal)
    {
        return set_error("invalid gpu_matrix_equal arguments");
    }
    *out_equal = 0;

    if (lhs == rhs)
    {
        *out_equal = 1;
        return 0;
    }
    if (lhs->rows != rhs->rows || lhs->cols != rhs->cols)
    {
        return 0;
    }
    if (lhs->ctx != rhs->ctx || lhs->level != rhs->level)
    {
        return 0;
    }

    const size_t count = lhs->rows * lhs->cols;
    for (size_t i = 0; i < count; ++i)
    {
        int poly_equal = 0;
        int status = gpu_poly_equal(lhs->polys[i], rhs->polys[i], &poly_equal);
        if (status != 0)
        {
            return status;
        }
        if (poly_equal == 0)
        {
            return 0;
        }
    }

    *out_equal = 1;
    return 0;
}

extern "C" int gpu_matrix_mul_timed(
    GpuMatrix *out,
    const GpuMatrix *lhs,
    const GpuMatrix *rhs,
    double *out_kernel_ms)
{
    if (!out_kernel_ms)
    {
        return set_error("null out_kernel_ms in gpu_matrix_mul_timed");
    }
    if (!out || !lhs || !rhs)
    {
        return set_error("invalid gpu_matrix_mul_timed arguments");
    }
    if (lhs->cols != rhs->rows)
    {
        return set_error("size mismatch in gpu_matrix_mul_timed");
    }
    if (out->rows != lhs->rows || out->cols != rhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul_timed");
    }
    if (lhs->ctx != rhs->ctx || lhs->ctx != out->ctx || lhs->level != rhs->level ||
        lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul_timed");
    }
    int status = gpu_block_mul_timed(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        const_cast<const GpuPoly *const *>(rhs->polys.data()),
        lhs->rows,
        lhs->cols,
        rhs->cols,
        out_kernel_ms);
    if (status != 0)
    {
        return status;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_mul_scalar(
    GpuMatrix *out,
    const GpuMatrix *lhs,
    const GpuPoly *scalar)
{
    if (!out || !lhs || !scalar)
    {
        return set_error("invalid gpu_matrix_mul_scalar arguments");
    }
    if (out->rows != lhs->rows || out->cols != lhs->cols)
    {
        return set_error("output size mismatch in gpu_matrix_mul_scalar");
    }
    if (lhs->ctx != out->ctx || lhs->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul_scalar");
    }
    if (scalar->ctx != lhs->ctx || scalar->level != lhs->level)
    {
        return set_error("scalar context mismatch in gpu_matrix_mul_scalar");
    }

    const size_t count = lhs->rows * lhs->cols;
    std::vector<const GpuPoly *> rhs(count, scalar);
    int status = gpu_block_entrywise_mul(
        out->polys.data(),
        const_cast<const GpuPoly *const *>(lhs->polys.data()),
        rhs.data(),
        count);
    if (status != 0)
    {
        return status;
    }
    out->format = lhs->format;
    return 0;
}

extern "C" int gpu_matrix_copy_block(
    GpuMatrix *out,
    const GpuMatrix *src,
    size_t dst_row,
    size_t dst_col,
    size_t src_row,
    size_t src_col,
    size_t rows,
    size_t cols)
{
    if (!out || !src)
    {
        return set_error("invalid gpu_matrix_copy_block arguments");
    }
    if (src_row + rows > src->rows || src_col + cols > src->cols)
    {
        return set_error("source bounds exceeded in gpu_matrix_copy_block");
    }
    if (dst_row + rows > out->rows || dst_col + cols > out->cols)
    {
        return set_error("dest bounds exceeded in gpu_matrix_copy_block");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_copy_block");
    }

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            const size_t src_idx = matrix_index(src_row + i, src_col + j, src->cols);
            const size_t dst_idx = matrix_index(dst_row + i, dst_col + j, out->cols);
            int status = gpu_poly_copy(out->polys[dst_idx], src->polys[src_idx]);
            if (status != 0)
            {
                return status;
            }
        }
    }
    out->format = src->format;
    return 0;
}

extern "C" int gpu_matrix_fill_gadget(
    GpuMatrix *out,
    uint32_t base_bits)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_fill_gadget arguments");
    }
    if (base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid base_bits in gpu_matrix_fill_gadget");
    }

    const size_t rows = out->rows;
    const size_t cols = out->cols;
    const size_t count = rows * cols;
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_fill_gadget");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (out->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_fill_gadget");
    }
    auto &limb_map = out->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_fill_gadget");
    }

    uint32_t crt_bits = 0;
    for (size_t i = 0; i < crt_depth; ++i)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(out->ctx->moduli[i]));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_fill_gadget");
    }
    const uint32_t digits_per_tower = static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_fill_gadget");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (cols != rows * log_base_q)
    {
        return set_error("output size mismatch in gpu_matrix_fill_gadget");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> dst_ptrs;
        dst_ptrs.reserve(count);
        cudaStream_t out_stream = nullptr;

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out->polys[idx];
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                return set_error("invalid output poly in gpu_matrix_fill_gadget");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_matrix_fill_gadget");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_matrix_fill_gadget");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                return set_error("unsupported limb type in gpu_matrix_fill_gadget");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            if (!out_stream)
            {
                out_stream = limb_u64.stream.ptr;
            }
            dst_ptrs.push_back(limb_u64.v.data);
        }

        int status = launch_fill_gadget_kernel(
            dst_ptrs,
            static_cast<size_t>(out->ctx->N),
            out->ctx->moduli[static_cast<size_t>(limb)],
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            static_cast<uint32_t>(limb),
            base_bits,
            out_stream);
        if (status != 0)
        {
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        poly->format = PolyFormat::Coeff;
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_fill_gadget");
        if (status != 0)
        {
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;
    return 0;
}

extern "C" int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out)
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

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_decompose_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
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
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_decompose_base");
    }
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    std::vector<GpuPoly *> tmp_inputs;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        tmp_inputs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            GpuPoly *clone = nullptr;
            int status = gpu_poly_clone(src->polys[i], &clone);
            if (status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                gpu_poly_destroy(clone);
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            tmp_inputs.push_back(clone);
            inputs.push_back(clone);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            inputs.push_back(src->polys[i]);
        }
    }

    // Ensure all pending INTT work has completed before reading source limbs
    // from different streams in the sampling kernels.
    for (int device : src->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error(err);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        for (auto *p : tmp_inputs)
        {
            gpu_poly_destroy(p);
        }
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("invalid output poly in gpu_matrix_decompose_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb GPU partition in gpu_matrix_decompose_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb index in gpu_matrix_decompose_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unsupported limb type in gpu_matrix_decompose_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        for (auto *p : tmp_inputs)
        {
            gpu_poly_destroy(p);
        }
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected source limb GPU partition in gpu_matrix_decompose_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected source limb index in gpu_matrix_decompose_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unsupported source limb type in gpu_matrix_decompose_base");
        }

        const uint32_t src_bits = bit_width_u64(src->ctx->moduli[static_cast<size_t>(src_limb)]);
        cudaError_t err = cudaSetDevice(src_partition0.device);
        if (err != cudaSuccess)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error(err);
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const uint32_t shift = digit_idx * base_bits;
            uint64_t mask = 0;
            if (shift < src_bits)
            {
                const uint32_t remaining = src_bits - shift;
                const uint32_t digit_bits = std::min(base_bits, remaining);
                mask = digit_bits >= 64 ? std::numeric_limits<uint64_t>::max()
                                        : ((uint64_t{1} << digit_bits) - 1);
            }

            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unexpected input source limb index in gpu_matrix_decompose_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unsupported input source limb type in gpu_matrix_decompose_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("input/output limb device mismatch in gpu_matrix_decompose_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("unexpected output limb index in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("unsupported output limb type in gpu_matrix_decompose_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                int status = launch_decompose_kernel<uint64_t>(
                    src_ptrs,
                    dst_ptrs,
                    static_cast<size_t>(src->ctx->N),
                    shift,
                    mask,
                    src->ctx->moduli[static_cast<size_t>(out_limb)],
                    out_stream);
                if (status != 0)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return status;
                }
            }
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return status;
        }
        status = sync_poly_partition_streams(
            poly,
            "failed to synchronize output partition stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return status;
        }
        status = sync_poly_limb_streams(
            poly,
            "failed to synchronize output limb stream after gpu_poly_ntt in gpu_matrix_decompose_base");
        if (status != 0)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    for (auto *p : tmp_inputs)
    {
        gpu_poly_destroy(p);
    }
    return 0;
}

extern "C" int gpu_matrix_gauss_samp_gq_arb_base(
    const GpuMatrix *src,
    uint32_t base_bits,
    double c,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    (void)dgg_stddev;
    if (!src || !out)
    {
        return set_error("invalid gpu_matrix_gauss_samp_gq_arb_base arguments");
    }
    if (base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid base_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (!(c > 0.0))
    {
        return set_error("c must be positive in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (src->ctx != out->ctx || src->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }

    const size_t rows = src->rows;
    const size_t cols = src->cols;
    const size_t count = rows * cols;
    const int level = src->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    uint32_t crt_bits = 0;
    for (const auto &modulus : src->ctx->moduli)
    {
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    }
    if (crt_bits == 0)
    {
        return set_error("invalid crt_bits in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const uint32_t digits_per_tower = static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return set_error("invalid digits_per_tower in gpu_matrix_gauss_samp_gq_arb_base");
    }
    const size_t log_base_q = static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_gauss_samp_gq_arb_base");
    }
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    std::vector<GpuPoly *> tmp_inputs;
    std::vector<const GpuPoly *> inputs;
    inputs.reserve(count);
    const int batch = default_batch(src->ctx);
    if (src->format == PolyFormat::Eval)
    {
        tmp_inputs.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            int sync_status = sync_poly_partition_streams(
                src->polys[i],
                "failed to synchronize source partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                src->polys[i],
                "failed to synchronize source limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return sync_status;
            }

            GpuPoly *clone = nullptr;
            int status = gpu_poly_clone(src->polys[i], &clone);
            if (status != 0)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            sync_status = sync_poly_partition_streams(
                clone,
                "failed to synchronize clone partition stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                gpu_poly_destroy(clone);
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return sync_status;
            }
            sync_status = sync_poly_limb_streams(
                clone,
                "failed to synchronize clone limb stream in gpu_matrix_gauss_samp_gq_arb_base");
            if (sync_status != 0)
            {
                gpu_poly_destroy(clone);
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return sync_status;
            }
            status = gpu_poly_intt(clone, batch);
            if (status != 0)
            {
                gpu_poly_destroy(clone);
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return status;
            }
            tmp_inputs.push_back(clone);
            inputs.push_back(clone);
        }
    }
    else
    {
        for (size_t i = 0; i < count; ++i)
        {
            inputs.push_back(src->polys[i]);
        }
    }

    auto &limb_map = src->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        for (auto *p : tmp_inputs)
        {
            gpu_poly_destroy(p);
        }
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != src->ctx || poly->level != level)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("invalid output poly in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (int limb = 0; limb <= level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            if (limb_id.x >= poly->poly->GPU.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unexpected limb index in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error("unsupported limb type in gpu_matrix_gauss_samp_gq_arb_base");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

            cudaError_t err = cudaSetDevice(partition.device);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
            err = cudaMemsetAsync(
                limb_u64.v.data,
                0,
                static_cast<size_t>(src->ctx->N) * sizeof(uint64_t),
                limb_u64.stream.ptr);
            if (err != cudaSuccess)
            {
                for (auto *p : tmp_inputs)
                {
                    gpu_poly_destroy(p);
                }
                return set_error(err);
            }
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        for (auto *p : tmp_inputs)
        {
            gpu_poly_destroy(p);
        }
        return set_error("unexpected modulus count in gpu_matrix_gauss_samp_gq_arb_base");
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const dim3 src_limb_id = limb_map[static_cast<size_t>(src_limb)];
        if (src_limb_id.x >= inputs[0]->poly->GPU.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected source limb GPU partition in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_partition0 = inputs[0]->poly->GPU[src_limb_id.x];
        if (src_limb_id.y >= src_partition0.limb.size())
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unexpected source limb index in gpu_matrix_gauss_samp_gq_arb_base");
        }
        const auto &src_limb_impl0 = src_partition0.limb[src_limb_id.y];
        if (src_limb_impl0.index() != FIDESlib::U64)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return set_error("unsupported source limb type in gpu_matrix_gauss_samp_gq_arb_base");
        }

        for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
        {
            const size_t digit_offset =
                static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                static_cast<size_t>(digit_idx);
            std::vector<const uint64_t *> src_ptrs;
            src_ptrs.reserve(count);
            std::vector<cudaStream_t> src_streams;
            src_streams.reserve(count);
            for (size_t idx = 0; idx < count; ++idx)
            {
                const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unexpected input source limb index in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error("unsupported input source limb type in gpu_matrix_gauss_samp_gq_arb_base");
                }
                const auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                cudaStream_t src_stream = in_limb_u64.stream.ptr;
                bool seen_src_stream = false;
                for (cudaStream_t s : src_streams)
                {
                    if (s == src_stream)
                    {
                        seen_src_stream = true;
                        break;
                    }
                }
                if (!seen_src_stream)
                {
                    src_streams.push_back(src_stream);
                }
                cudaStream_t src_partition_stream = in_partition.s.ptr;
                seen_src_stream = false;
                for (cudaStream_t s : src_streams)
                {
                    if (s == src_partition_stream)
                    {
                        seen_src_stream = true;
                        break;
                    }
                }
                if (!seen_src_stream)
                {
                    src_streams.push_back(src_partition_stream);
                }
                src_ptrs.push_back(in_limb_u64.v.data);
            }

            for (int out_limb = 0; out_limb <= level; ++out_limb)
            {
                const dim3 out_limb_id = limb_map[static_cast<size_t>(out_limb)];
                std::vector<uint64_t *> dst_ptrs;
                dst_ptrs.reserve(count);
                cudaStream_t out_stream = nullptr;
                int out_device = -1;
                std::vector<cudaStream_t> dst_streams;
                dst_streams.reserve(count);

                for (size_t idx = 0; idx < count; ++idx)
                {
                    const auto &in_partition = inputs[idx]->poly->GPU[src_limb_id.x];
                    const size_t row = idx / cols;
                    const size_t col = idx % cols;
                    const size_t out_row = row * log_base_q + digit_offset;
                    const size_t out_idx = matrix_index(out_row, col, out->cols);
                    auto &out_partition = out->polys[out_idx]->poly->GPU[out_limb_id.x];
                    if (out_partition.device != in_partition.device)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("input/output limb device mismatch in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    if (out_limb_id.y >= out_partition.limb.size())
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("unexpected output limb index in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                    if (out_limb_impl.index() != FIDESlib::U64)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("unsupported output limb type in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    const auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);
                    if (!out_stream)
                    {
                        out_stream = out_limb_u64.stream.ptr;
                        out_device = out_partition.device;
                    }
                    else if (out_device != out_partition.device)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error("inconsistent output limb device in gpu_matrix_gauss_samp_gq_arb_base");
                    }
                    cudaStream_t dst_stream = out_limb_u64.stream.ptr;
                    bool seen_stream = false;
                    for (cudaStream_t s : dst_streams)
                    {
                        if (s == dst_stream)
                        {
                            seen_stream = true;
                            break;
                        }
                    }
                    if (!seen_stream)
                    {
                        dst_streams.push_back(dst_stream);
                    }
                    cudaStream_t dst_partition_stream = out_partition.s.ptr;
                    seen_stream = false;
                    for (cudaStream_t s : dst_streams)
                    {
                        if (s == dst_partition_stream)
                        {
                            seen_stream = true;
                            break;
                        }
                    }
                    if (!seen_stream)
                    {
                        dst_streams.push_back(dst_partition_stream);
                    }
                    dst_ptrs.push_back(out_limb_u64.v.data);
                }

                for (cudaStream_t dst_stream : dst_streams)
                {
                    if (dst_stream == out_stream)
                    {
                        continue;
                    }
                    cudaEvent_t ready = nullptr;
                    cudaError_t err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    err = cudaEventRecord(ready, dst_stream);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(ready);
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    err = cudaStreamWaitEvent(out_stream, ready, 0);
                    cudaError_t destroy_err = cudaEventDestroy(ready);
                    if (err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    if (destroy_err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(destroy_err);
                    }
                }
                for (cudaStream_t src_stream : src_streams)
                {
                    if (src_stream == out_stream)
                    {
                        continue;
                    }
                    cudaEvent_t ready = nullptr;
                    cudaError_t err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
                    if (err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    err = cudaEventRecord(ready, src_stream);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(ready);
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    err = cudaStreamWaitEvent(out_stream, ready, 0);
                    cudaError_t destroy_err = cudaEventDestroy(ready);
                    if (err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                    if (destroy_err != cudaSuccess)
                    {
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(destroy_err);
                    }
                }

                int status = launch_gauss_samp_gq_arb_base_kernel(
                    src_ptrs,
                    dst_ptrs,
                    static_cast<size_t>(src->ctx->N),
                    src->ctx->moduli[static_cast<size_t>(src_limb)],
                    base_bits,
                    digits_per_tower,
                    digit_idx,
                    c,
                    static_cast<uint32_t>(src_limb),
                    seed,
                    src->ctx->moduli[static_cast<size_t>(out_limb)],
                    out_device,
                    out_stream);
                if (status != 0)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return status;
                }
                cudaEvent_t done = nullptr;
                cudaError_t err = cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
                if (err != cudaSuccess)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error(err);
                }
                err = cudaEventRecord(done, out_stream);
                if (err != cudaSuccess)
                {
                    cudaEventDestroy(done);
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error(err);
                }
                for (cudaStream_t dst_stream : dst_streams)
                {
                    if (dst_stream == out_stream)
                    {
                        continue;
                    }
                    err = cudaStreamWaitEvent(dst_stream, done, 0);
                    if (err != cudaSuccess)
                    {
                        cudaEventDestroy(done);
                        for (auto *p : tmp_inputs)
                        {
                            gpu_poly_destroy(p);
                        }
                        return set_error(err);
                    }
                }
                cudaError_t destroy_err = cudaEventDestroy(done);
                if (destroy_err != cudaSuccess)
                {
                    for (auto *p : tmp_inputs)
                    {
                        gpu_poly_destroy(p);
                    }
                    return set_error(destroy_err);
                }
            }
        }
    }

    for (auto *poly : out->polys)
    {
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            for (auto *p : tmp_inputs)
            {
                gpu_poly_destroy(p);
            }
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    for (auto *p : tmp_inputs)
    {
        gpu_poly_destroy(p);
    }
    return 0;
}

extern "C" int gpu_matrix_sample_p1_full(
    const GpuMatrix *a_mat,
    const GpuMatrix *b_mat,
    const GpuMatrix *d_mat,
    const GpuMatrix *tp2,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix *out)
{
    if (!a_mat || !b_mat || !d_mat || !tp2 || !out)
    {
        return set_error("invalid gpu_matrix_sample_p1_full arguments");
    }
    if (!(sigma > 0.0) || !(s > sigma))
    {
        return set_error("invalid sigma/s in gpu_matrix_sample_p1_full");
    }
    if (!(dgg_stddev > 0.0))
    {
        return set_error("dgg_stddev must be positive in gpu_matrix_sample_p1_full");
    }
    if (a_mat->ctx != b_mat->ctx || a_mat->ctx != d_mat->ctx || a_mat->ctx != tp2->ctx || a_mat->ctx != out->ctx)
    {
        return set_error("context mismatch in gpu_matrix_sample_p1_full");
    }
    if (a_mat->level != b_mat->level || a_mat->level != d_mat->level ||
        a_mat->level != tp2->level || a_mat->level != out->level)
    {
        return set_error("level mismatch in gpu_matrix_sample_p1_full");
    }

    const size_t d_rows = a_mat->rows;
    if (a_mat->cols != d_rows || b_mat->rows != d_rows || b_mat->cols != d_rows ||
        d_mat->rows != d_rows || d_mat->cols != d_rows)
    {
        return set_error("A/B/D must be dxd in gpu_matrix_sample_p1_full");
    }
    const size_t cols = tp2->cols;
    if (tp2->rows != 2 * d_rows || out->rows != 2 * d_rows || out->cols != cols)
    {
        return set_error("tp2/out shape mismatch in gpu_matrix_sample_p1_full");
    }
    if (cols == 0 || d_rows == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = a_mat->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_p1_full");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (a_mat->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_p1_full");
    }
    auto &limb_map = a_mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_p1_full");
    }

    std::vector<GpuPoly *> tmp_a;
    std::vector<GpuPoly *> tmp_b;
    std::vector<GpuPoly *> tmp_d;
    std::vector<GpuPoly *> tmp_tp2;
    std::vector<const GpuPoly *> a_inputs;
    std::vector<const GpuPoly *> b_inputs;
    std::vector<const GpuPoly *> d_inputs;
    std::vector<const GpuPoly *> tp2_inputs;

    auto cleanup = [&]() {
        for (auto *p : tmp_a)
        {
            gpu_poly_destroy(p);
        }
        for (auto *p : tmp_b)
        {
            gpu_poly_destroy(p);
        }
        for (auto *p : tmp_d)
        {
            gpu_poly_destroy(p);
        }
        for (auto *p : tmp_tp2)
        {
            gpu_poly_destroy(p);
        }
    };

    auto collect_coeff_inputs = [&](const GpuMatrix *src, std::vector<GpuPoly *> &owned, std::vector<const GpuPoly *> &inputs) -> int {
        const size_t count = src->rows * src->cols;
        inputs.clear();
        inputs.reserve(count);
        const int batch = default_batch(src->ctx);
        if (src->format == PolyFormat::Eval)
        {
            owned.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                GpuPoly *clone = nullptr;
                int status = gpu_poly_clone(src->polys[i], &clone);
                if (status != 0)
                {
                    return status;
                }
                status = gpu_poly_intt(clone, batch);
                if (status != 0)
                {
                    gpu_poly_destroy(clone);
                    return status;
                }
                owned.push_back(clone);
                inputs.push_back(clone);
            }
        }
        else
        {
            for (size_t i = 0; i < count; ++i)
            {
                inputs.push_back(src->polys[i]);
            }
        }
        return 0;
    };

    int status = collect_coeff_inputs(a_mat, tmp_a, a_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(b_mat, tmp_b, b_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(d_mat, tmp_d, d_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_inputs(tp2, tmp_tp2, tp2_inputs);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    // Ensure all pending INTT work has completed before cross-stream reads.
    for (int device : a_mat->ctx->gpu_ids)
    {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
    }

    for (size_t idx = 0; idx < out->polys.size(); ++idx)
    {
        GpuPoly *poly = out->polys[idx];
        if (!poly || poly->ctx != a_mat->ctx || poly->level != level)
        {
            cleanup();
            return set_error("invalid output poly in gpu_matrix_sample_p1_full");
        }
        poly->format = PolyFormat::Coeff;
        poly->level = level;
    }
    out->format = PolyFormat::Coeff;

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];

        std::vector<const uint64_t *> a_entry_ptrs;
        std::vector<const uint64_t *> b_entry_ptrs;
        std::vector<const uint64_t *> d_entry_ptrs;
        std::vector<const uint64_t *> tp2_entry_ptrs;
        std::vector<uint64_t *> out_entry_ptrs;
        a_entry_ptrs.reserve(d_rows * d_rows);
        b_entry_ptrs.reserve(d_rows * d_rows);
        d_entry_ptrs.reserve(d_rows * d_rows);
        tp2_entry_ptrs.reserve(2 * d_rows * cols);
        out_entry_ptrs.reserve(2 * d_rows * cols);

        cudaStream_t out_stream = nullptr;
        int out_device = -1;
        for (size_t i = 0; i < d_rows; ++i)
        {
            for (size_t j = 0; j < d_rows; ++j)
            {
                const size_t idx = matrix_index(i, j, d_rows);
                if (limb_id.x >= a_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= b_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= d_inputs[idx]->poly->GPU.size())
                {
                    cleanup();
                    return set_error("unexpected A/B/D limb GPU partition in gpu_matrix_sample_p1_full");
                }
                const auto &a_part = a_inputs[idx]->poly->GPU[limb_id.x];
                const auto &b_part = b_inputs[idx]->poly->GPU[limb_id.x];
                const auto &d_part = d_inputs[idx]->poly->GPU[limb_id.x];
                if (limb_id.y >= a_part.limb.size() ||
                    limb_id.y >= b_part.limb.size() ||
                    limb_id.y >= d_part.limb.size())
                {
                    cleanup();
                    return set_error("unexpected A/B/D limb index in gpu_matrix_sample_p1_full");
                }
                const auto &a_impl = a_part.limb[limb_id.y];
                const auto &b_impl = b_part.limb[limb_id.y];
                const auto &d_impl = d_part.limb[limb_id.y];
                if (a_impl.index() != FIDESlib::U64 ||
                    b_impl.index() != FIDESlib::U64 ||
                    d_impl.index() != FIDESlib::U64)
                {
                    cleanup();
                    return set_error("unsupported A/B/D limb type in gpu_matrix_sample_p1_full");
                }
                a_entry_ptrs.push_back(std::get<FIDESlib::U64>(a_impl).v.data);
                b_entry_ptrs.push_back(std::get<FIDESlib::U64>(b_impl).v.data);
                d_entry_ptrs.push_back(std::get<FIDESlib::U64>(d_impl).v.data);
            }
        }
        for (size_t row = 0; row < 2 * d_rows; ++row)
        {
            for (size_t col = 0; col < cols; ++col)
            {
                const size_t idx = matrix_index(row, col, cols);
                if (limb_id.x >= tp2_inputs[idx]->poly->GPU.size() ||
                    limb_id.x >= out->polys[idx]->poly->GPU.size())
                {
                    cleanup();
                    return set_error("unexpected tp2/output limb GPU partition in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_part = tp2_inputs[idx]->poly->GPU[limb_id.x];
                auto &out_part = out->polys[idx]->poly->GPU[limb_id.x];
                if (tp2_part.device != out_part.device)
                {
                    cleanup();
                    return set_error("input/output limb device mismatch in gpu_matrix_sample_p1_full");
                }
                if (out_device < 0)
                {
                    out_device = out_part.device;
                }
                else if (out_device != out_part.device)
                {
                    cleanup();
                    return set_error("mixed output devices in gpu_matrix_sample_p1_full");
                }
                if (limb_id.y >= tp2_part.limb.size() || limb_id.y >= out_part.limb.size())
                {
                    cleanup();
                    return set_error("unexpected tp2/output limb index in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_impl = tp2_part.limb[limb_id.y];
                auto &out_impl = out_part.limb[limb_id.y];
                if (tp2_impl.index() != FIDESlib::U64 || out_impl.index() != FIDESlib::U64)
                {
                    cleanup();
                    return set_error("unsupported tp2/output limb type in gpu_matrix_sample_p1_full");
                }
                const auto &tp2_u64 = std::get<FIDESlib::U64>(tp2_impl);
                auto &out_u64 = std::get<FIDESlib::U64>(out_impl);
                if (!out_stream)
                {
                    out_stream = out_u64.stream.ptr;
                }
                tp2_entry_ptrs.push_back(tp2_u64.v.data);
                out_entry_ptrs.push_back(out_u64.v.data);
            }
        }

        status = launch_sample_p1_full_kernel(
            a_entry_ptrs,
            b_entry_ptrs,
            d_entry_ptrs,
            tp2_entry_ptrs,
            out_entry_ptrs,
            d_rows,
            cols,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            sigma,
            s,
            dgg_stddev,
            static_cast<uint32_t>(limb),
            seed,
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;

    cleanup();
    return 0;
}

extern "C" int gpu_matrix_sample_distribution(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t seed)
{
    if (!out)
    {
        return set_error("invalid gpu_matrix_sample_distribution arguments");
    }
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
    {
        return set_error("invalid dist_type in gpu_matrix_sample_distribution");
    }
    if (dist_type == GPU_MATRIX_DIST_GAUSS && !(sigma > 0.0))
    {
        return set_error("sigma must be positive in gpu_matrix_sample_distribution");
    }

    const size_t count = out->rows * out->cols;
    if (count == 0)
    {
        out->format = PolyFormat::Eval;
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_distribution");
    }
    if (out->ctx->moduli.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected modulus count in gpu_matrix_sample_distribution");
    }

    auto &limb_map = out->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(level + 1))
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_distribution");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        std::vector<uint64_t *> dst_ptrs;
        dst_ptrs.reserve(count);
        cudaStream_t out_stream = nullptr;

        for (size_t idx = 0; idx < count; ++idx)
        {
            GpuPoly *poly = out->polys[idx];
            if (!poly || poly->ctx != out->ctx || poly->level != level)
            {
                return set_error("invalid output poly in gpu_matrix_sample_distribution");
            }
            if (limb_id.x >= poly->poly->GPU.size())
            {
                return set_error("unexpected limb GPU partition in gpu_matrix_sample_distribution");
            }
            auto &partition = poly->poly->GPU[limb_id.x];
            if (limb_id.y >= partition.limb.size())
            {
                return set_error("unexpected limb index in gpu_matrix_sample_distribution");
            }
            auto &limb_impl = partition.limb[limb_id.y];
            if (limb_impl.index() != FIDESlib::U64)
            {
                return set_error("unsupported limb type in gpu_matrix_sample_distribution");
            }
            auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);
            if (!out_stream)
            {
                out_stream = limb_u64.stream.ptr;
            }
            dst_ptrs.push_back(limb_u64.v.data);
        }

        int status = launch_sample_distribution_kernel(
            dst_ptrs,
            static_cast<size_t>(out->ctx->N),
            out->ctx->moduli[static_cast<size_t>(limb)],
            dist_type,
            sigma,
            static_cast<uint32_t>(limb),
            seed,
            out_stream);
        if (status != 0)
        {
            return status;
        }
    }

    const int batch = default_batch(out->ctx);
    for (auto *poly : out->polys)
    {
        poly->format = PolyFormat::Coeff;
        int status = gpu_poly_ntt(poly, batch);
        if (status != 0)
        {
            return status;
        }
        poly->format = PolyFormat::Eval;
    }
    out->format = PolyFormat::Eval;
    return 0;
}
