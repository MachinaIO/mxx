#include "gpu_prepared_plan.cuh"

#include <array>
#include <atomic>
#include <cstring>
#include <limits>

using gpu_chacha::DeviceChaChaRng;
using gpu_chacha::GpuRngSeed;
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
        const uint64_t magnitude = static_cast<uint64_t>(value);
        return magnitude < modulus ? magnitude : magnitude % modulus;
    }
    uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
    uint64_t rem = magnitude < modulus ? magnitude : magnitude % modulus;
    return rem == 0 ? 0 : (modulus - rem);
}

__device__ __forceinline__ uint64_t abs_i64(int64_t value)
{
    if (value >= 0)
    {
        return static_cast<uint64_t>(value);
    }
    return static_cast<uint64_t>(-(value + 1)) + 1;
}

__device__ __forceinline__ uint64_t centered_sample_abs_i64(
    int64_t value,
    uint64_t coefficient_modulus)
{
    if (coefficient_modulus == 0)
    {
        // A zero sentinel means the full CRT modulus does not fit in u64. Since the sampler
        // produces an i64, reduction cannot change its centered representative in that case.
        return abs_i64(value);
    }
    const uint64_t residue = signed_mod_i64(value, coefficient_modulus);
    const uint64_t negative_magnitude = coefficient_modulus - residue;
    return residue < negative_magnitude ? residue : negative_magnitude;
}

__device__ __forceinline__ uint64_t sample_uniform_mod(
    DeviceChaChaRng &rng,
    uint64_t modulus,
    uint64_t rejection_threshold)
{
    if (modulus == 0)
    {
        return 0;
    }
    for (;;)
    {
        const uint64_t random = rng_next_u64(rng);
        const uint64_t low = random * modulus;
        if (low >= rejection_threshold)
        {
            return __umul64hi(random, modulus);
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

// Sampling coordinates describe the logical full matrix, independently of
// the retained owner's physical rectangle. Bounded metadata stays in arguments.
struct MatrixSampleDescriptors {
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *descriptors;
    uint32_t indices[GPU_RUNTIME_MAX_LIMBS];
    uint64_t moduli[GPU_RUNTIME_MAX_LIMBS];
    size_t offset, pitch;
    const int32_t *completed = nullptr;
};
static_assert(sizeof(MatrixSampleDescriptors) + 128 < 4096, "bounded sampling arguments");

struct GpuPreparedSampling {
    GpuMatrix *out = nullptr;
    MatrixSampleDescriptors layout{};
    GpuMatrixRange range{};
    GpuMatrixTransformPlan *transform = nullptr;
    cudaStream_t stream = nullptr;
    int device = -1;
    int dist_type = GPU_MATRIX_DIST_UNIFORM;
    double sigma = 0.0;
    uint64_t max_coefficient_bound = 0;
    uint64_t coefficient_modulus = 0;
    size_t poly_count = 0;
    size_t columns = 0;
    size_t full_ncol = 0;
    size_t col_offset = 0;
    size_t n = 0;
    size_t limb_count = 0;
    size_t blocks = 0;
    cudaEvent_t completed_event = nullptr;
    std::array<std::unique_ptr<GpuCudaResource>, GPU_RUNTIME_MAX_LIMBS> completion;

    ~GpuPreparedSampling()
    {
        if (transform)
            gpu_matrix_destroy_ntt_plan(transform);
        for (auto &event : completion)
            if (event) (void)event->release();
    }
};


extern "C" int gpu_matrix_query_sampling_layout(
    size_t ring_dimension, size_t limb_count, size_t rows, size_t columns,
    size_t full_ncol, size_t col_offset, int format, int dist_type,
    int device, GpuPreparedSamplingLayout *out)
{
    if (!out || ring_dimension < 2 || ring_dimension > std::numeric_limits<uint32_t>::max() ||
        !is_power_of_two_u32(static_cast<uint32_t>(ring_dimension)) ||
        limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS ||
        (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL) ||
        dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY ||
        col_offset > full_ncol || columns > full_ncol - col_offset)
        return set_error("invalid prepared sampling layout request");
    if (rows != 0 && columns > std::numeric_limits<size_t>::max() / rows)
        return set_error("prepared sampling layout shape overflow");
    const size_t polynomial_count = rows * columns;
    if (ring_dimension > std::numeric_limits<size_t>::max() - 3)
        return set_error("prepared sampling layout sample overflow");
    const size_t samples_per_poly = (ring_dimension + 3) / 4;
    if (polynomial_count != 0 && samples_per_poly > std::numeric_limits<size_t>::max() / polynomial_count)
        return set_error("prepared sampling layout sample overflow");
    const size_t chunks = polynomial_count * samples_per_poly;
    if (chunks > std::numeric_limits<size_t>::max() - 255)
        return set_error("prepared sampling layout block overflow");
    const size_t blocks = chunks == 0 ? 0 : std::min<size_t>(65535, (chunks + 255) / 256);
    size_t transform_stage_count = 0;
    if (format == GPU_POLY_FORMAT_EVAL && polynomial_count != 0)
    {
        GpuPreparedNttLayout ntt{};
        const int status = gpu_matrix_query_ntt_layout(
            ring_dimension, limb_count, polynomial_count, device, true, &ntt);
        if (status != 0) return status;
        transform_stage_count = ntt.stage_count;
    }
    *out = GpuPreparedSamplingLayout{
        rows, columns, ring_dimension, limb_count, polynomial_count, blocks,
        0, alignof(uint64_t), limb_count, transform_stage_count, device,
        format, dist_type, 3,
    };
    return 0;
}

static int prepare_sampling_layout(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range,
    GpuPreparedSampling &prepared)
{
    if (!out || !out->ctx || out->ctx->N < 2 || out->level < 0 ||
        static_cast<size_t>(out->level) >= GPU_RUNTIME_MAX_LIMBS)
        return set_error("invalid matrix in gpu_matrix_prepare_sampling");
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("invalid dist_type in gpu_matrix_prepare_sampling");
    if (dist_type == GPU_MATRIX_DIST_GAUSS && (!(sigma > 0.0) || !std::isfinite(sigma)))
        return set_error("sigma must be finite and positive in gpu_matrix_prepare_sampling");
    const GpuMatrixRange rectangle = range ? *range : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (rectangle.row_start > rectangle.row_end || rectangle.row_end > out->rows ||
        rectangle.column_start > rectangle.column_end || rectangle.column_end > out->cols)
        return set_error("invalid destination rectangle in gpu_matrix_prepare_sampling");
    const size_t rows = rectangle.row_end - rectangle.row_start;
    const size_t columns = rectangle.column_end - rectangle.column_start;
    const size_t n = static_cast<size_t>(out->ctx->N);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (col_offset > full_ncol || columns > full_ncol - col_offset ||
        (out->rows && out->cols > limit / out->rows) ||
        (rows && full_ncol > limit / rows) || (rows && columns > limit / rows / n))
        return set_error("invalid logical shape in gpu_matrix_prepare_sampling");
    if (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid output format in gpu_matrix_prepare_sampling");
    if (out->ctx->limb_gpu_ids.size() < static_cast<size_t>(out->level) + 1 ||
        out->ctx->moduli.size() < static_cast<size_t>(out->level) + 1)
        return set_error("invalid limb count in gpu_matrix_prepare_sampling");

    prepared.out = out;
    prepared.range = rectangle;
    prepared.layout.offset = rectangle.row_start * out->cols + rectangle.column_start;
    prepared.layout.pitch = out->cols;
    prepared.dist_type = dist_type;
    prepared.sigma = sigma;
    prepared.max_coefficient_bound = max_coefficient_bound;
    prepared.coefficient_modulus = coefficient_modulus;
    prepared.poly_count = rows * columns;
    prepared.columns = columns;
    prepared.full_ncol = full_ncol;
    prepared.col_offset = col_offset;
    prepared.n = n;
    prepared.limb_count = static_cast<size_t>(out->level) + 1;
    for (size_t limb = 0; limb < prepared.limb_count; ++limb) {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        int status = matrix_limb_device(out, id, &limb_device);
        if (status != 0) return status;
        if (id.x >= out->shared_limb_buffers.size())
            return set_error("invalid sampling partition");
        const auto &buffer = out->shared_limb_buffers[id.x];
        size_t stride = 0;
        uint8_t width = 0;
        if (!buffer.device_descriptors || id.y >= buffer.limb_count ||
            !matrix_limb_ptr_by_id(out, 0, id) ||
            !matrix_limb_metadata_by_id(out, id, &stride, &width) ||
            width == 0 || width > 8 || stride < n * width || !out->ctx->moduli[limb])
            return set_error("invalid sampling descriptor");
        if (limb == 0) {
            prepared.device = limb_device;
            prepared.layout.descriptors = buffer.device_descriptors;
            status = matrix_limb_stream(out, id, &prepared.stream);
            if (status != 0) return status;
        } else if (limb_device != prepared.device ||
                   prepared.layout.descriptors != buffer.device_descriptors) {
            return set_error("sampling requires one device and descriptor partition");
        }
        prepared.layout.indices[limb] = id.y;
        prepared.layout.moduli[limb] = out->ctx->moduli[limb];
    }
    if (prepared.device < 0 || !prepared.stream)
        return set_error("invalid sampling stream");
    GpuPreparedSamplingLayout structural{};
    const int status = gpu_matrix_query_sampling_layout(
        n, prepared.limb_count, rows, columns, full_ncol, col_offset,
        out->format, dist_type, prepared.device, &structural);
    if (status != 0) return status;
    prepared.blocks = structural.blocks;
    if (out->format == GPU_POLY_FORMAT_EVAL && prepared.poly_count != 0) {
        int status = gpu_matrix_prepare_ntt_plan(out, &rectangle, true, &prepared.transform);
        if (status != 0) return status;
    }
    return 0;
}

__device__ void matrix_sample_distribution_multi_limb_kernel_body(
    MatrixSampleDescriptors layout,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    const size_t limb_idx = blockIdx.z;
    const auto descriptor = layout.descriptors[layout.indices[limb_idx]];
    const uint64_t modulus = layout.moduli[limb_idx];
    constexpr size_t kSamplesPerThread = 4;
    const size_t chunks_per_poly = (n + kSamplesPerThread - 1) / kSamplesPerThread;
    const size_t total_chunks = poly_count * chunks_per_poly;
    for (size_t chunk_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         chunk_idx < total_chunks;
         chunk_idx += static_cast<size_t>(gridDim.x) * blockDim.x)
    {
        const size_t local_poly_idx = chunk_idx / chunks_per_poly;
        const size_t coeff_start =
            (chunk_idx - local_poly_idx * chunks_per_poly) * kSamplesPerThread;
        const size_t row_idx = local_poly_idx / local_ncol;
        const size_t local_col_idx = local_poly_idx - row_idx * local_ncol;
        const size_t global_poly_idx = row_idx * full_ncol + (col_offset + local_col_idx);

        const uint64_t domain = dist_type == GPU_MATRIX_DIST_UNIFORM ?
            0x6f70656e66686531ULL :
            (dist_type == GPU_MATRIX_DIST_GAUSS ? 0x6f70656e66686532ULL :
             (dist_type == GPU_MATRIX_DIST_BIT ? 0x6f70656e66686533ULL :
                                                0x6f70656e66686534ULL));
        const uint64_t limb_domain =
            dist_type == GPU_MATRIX_DIST_UNIFORM ? static_cast<uint64_t>(limb_idx + 1) : 0;
        DeviceChaChaRng rng;
        rng_init(
            rng,
            seed,
            static_cast<uint64_t>(global_poly_idx + 1),
            static_cast<uint64_t>(coeff_start + 1),
            limb_domain,
            domain);
        const uint64_t uniform_rejection_threshold =
            dist_type == GPU_MATRIX_DIST_UNIFORM && modulus != 0 ?
            static_cast<uint64_t>(-modulus) % modulus : 0;

        for (size_t lane = 0; lane < kSamplesPerThread; ++lane)
        {
            const size_t coeff_idx = coeff_start + lane;
            if (coeff_idx >= n) break;
            uint64_t sample = 0;
            if (dist_type == GPU_MATRIX_DIST_UNIFORM)
            {
                sample = sample_uniform_mod(rng, modulus, uniform_rejection_threshold);
            }
            else if (dist_type == GPU_MATRIX_DIST_GAUSS)
            {
                int64_t z;
                do
                {
                    z = sample_integer_karney(rng, 0.0, sigma);
                } while (centered_sample_abs_i64(z, coefficient_modulus) > max_coefficient_bound);
                sample = signed_mod_i64(z, modulus);
            }
            else if (dist_type == GPU_MATRIX_DIST_BIT)
            {
                sample = rng_next_u64(rng) & 1ULL;
            }
            else if (dist_type == GPU_MATRIX_DIST_TERNARY)
            {
                const uint64_t pick = rng_next_u64(rng) % 3ULL;
                const int64_t z = pick == 0 ? 0 : (pick == 1 ? 1 : -1);
                sample = signed_mod_i64(z, modulus);
            }

            matrix_store_limb_u64(
                descriptor.base,
                layout.offset + row_idx * layout.pitch + local_col_idx,
                coeff_idx,
                descriptor.stride,
                descriptor.width,
                sample);
        }
    }
}

__global__ void matrix_sample_distribution_multi_limb_kernel(
    MatrixSampleDescriptors layout,
    size_t poly_count,
    size_t local_ncol,
    size_t full_ncol,
    size_t col_offset,
    size_t n,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    if (layout.completed && *layout.completed) return;
    matrix_sample_distribution_multi_limb_kernel_body(layout, poly_count, local_ncol, full_ncol, col_offset, n, dist_type, sigma, max_coefficient_bound, coefficient_modulus, seed);
}


constexpr size_t kGaussianBatchMatrices = 64;
struct GaussianBatchDescriptors {
    const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *outputs[kGaussianBatchMatrices];
    GpuRngSeed seeds[kGaussianBatchMatrices];
};
static_assert(sizeof(GaussianBatchDescriptors) + sizeof(MatrixSampleDescriptors) + 128 < 4096,
              "bounded Gaussian sampling kernel arguments");

__global__ void matrix_sample_gaussian_batch_kernel(
    GaussianBatchDescriptors jobs, MatrixSampleDescriptors layout,
    size_t polynomials, size_t columns, size_t n, double sigma)
{
    layout.descriptors = jobs.outputs[blockIdx.y];
    matrix_sample_distribution_multi_limb_kernel_body(
        layout, polynomials, columns, columns, 0, n, GPU_MATRIX_DIST_GAUSS,
        sigma, UINT64_MAX, 0, jobs.seeds[blockIdx.y]);
}

static int gpu_matrix_sample_distribution_impl(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range)
{
    if (!out || !out->ctx || out->ctx->N < 2 || out->level < 0 ||
        static_cast<size_t>(out->level) >= GPU_RUNTIME_MAX_LIMBS)
        return set_error("invalid matrix in gpu_matrix_sample_distribution");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("invalid dist_type in gpu_matrix_sample_distribution");
    if (dist_type == GPU_MATRIX_DIST_GAUSS && (!(sigma > 0.0) || !std::isfinite(sigma)))
        return set_error("sigma must be finite and positive in gpu_matrix_sample_distribution");
    const GpuMatrixRange rectangle = range ? *range : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (rectangle.row_start > rectangle.row_end || rectangle.row_end > out->rows ||
        rectangle.column_start > rectangle.column_end || rectangle.column_end > out->cols)
        return set_error("invalid destination rectangle in gpu_matrix_sample_distribution");
    const size_t rows = rectangle.row_end - rectangle.row_start;
    const size_t columns = rectangle.column_end - rectangle.column_start;
    const size_t n = static_cast<size_t>(out->ctx->N);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (col_offset > full_ncol || columns > full_ncol - col_offset ||
        (out->rows && out->cols > limit / out->rows) ||
        (rows && full_ncol > limit / rows) || (rows && columns > limit / rows / n))
        return set_error("invalid logical shape in gpu_matrix_sample_distribution");
    if (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL)
        return set_error("invalid output format in gpu_matrix_sample_distribution");
    const size_t count = rows * columns;
    if (!count) return 0;
    const size_t limb_count = static_cast<size_t>(out->level) + 1;
    if (out->ctx->limb_gpu_ids.size() < limb_count || out->ctx->moduli.size() < limb_count)
        return set_error("invalid limb count in gpu_matrix_sample_distribution");
    MatrixSampleDescriptors layout{};
    layout.offset = rectangle.row_start * out->cols + rectangle.column_start;
    layout.pitch = out->cols;
    int device = -1;
    cudaStream_t stream = nullptr;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        int limb_device = -1;
        int status = matrix_limb_device(out, id, &limb_device);
        if (status != 0) return status;
        if (id.x >= out->shared_limb_buffers.size())
            return set_error("invalid sampling partition");
        const auto &buffer = out->shared_limb_buffers[id.x];
        size_t stride = 0;
        uint8_t width = 0;
        if (!buffer.device_descriptors || id.y >= buffer.limb_count ||
            !matrix_limb_ptr_by_id(out, 0, id) ||
            !matrix_limb_metadata_by_id(out, id, &stride, &width) ||
            (width == 0 || width > 8) || stride < n * width || !out->ctx->moduli[limb])
            return set_error("invalid sampling descriptor");
        if (limb == 0) {
            device = limb_device;
            layout.descriptors = buffer.device_descriptors;
            status = matrix_limb_stream(out, id, &stream);
            if (status != 0) return status;
        } else if (limb_device != device || layout.descriptors != buffer.device_descriptors) {
            return set_error("sampling requires one device and descriptor partition");
        }
        layout.indices[limb] = id.y;
        layout.moduli[limb] = out->ctx->moduli[limb];
    }
    if (device < 0 || !stream) return set_error("invalid sampling stream");
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    int status = matrix_wait_all_limb_streams(out, device, stream, true);
    if (status != 0) return status;
    constexpr size_t threads = 256;
    const size_t chunks = count * ((n + 3) / 4);
    const size_t blocks = std::min<size_t>(65535, chunks / threads + (chunks % threads != 0));
    matrix_sample_distribution_multi_limb_kernel<<<dim3(blocks, 1, limb_count), threads, 0, stream>>>(
        layout, count, columns, full_ncol, col_offset, n,
        dist_type, sigma, max_coefficient_bound, coefficient_modulus, seed);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = matrix_record_all_limb_writes(out, stream);
    if (status != 0) return status;
    // Existing evaluation entries outside the rectangle must not be transformed.
    // The same optimized NTT kernels consume only the newly sampled coefficients.
    if (out->format == GPU_POLY_FORMAT_EVAL)
        return run_matrix_transform_u64<true>(out, range);
    return 0;
}


extern "C" int gpu_matrix_sample_distribution(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed)
{
    return gpu_matrix_sample_distribution_impl(
        out,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed,
        out ? out->cols : 0,
        0,
        nullptr);
}

extern "C" int gpu_matrix_sample_distribution_columns(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    GpuRngSeed seed,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range)
{
    return gpu_matrix_sample_distribution_impl(
        out,
        dist_type,
        sigma,
        max_coefficient_bound,
        coefficient_modulus,
        seed,
        full_ncol,
        col_offset,
        range);
}

static int validate_saved_sampling_descriptor(
    GpuMatrix *out, int dist_type, double sigma, uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus, size_t full_ncol, size_t col_offset,
    const GpuMatrixRange *range, const GpuPreparedPlanDescriptor *descriptor,
    GpuPreparedSampling &prepared, GpuPreparedPlanDescriptor &ntt_descriptor)
{
    if (!out || !out->ctx || out->ctx->N < 2 || out->level < 0 ||
        static_cast<size_t>(out->level) >= GPU_RUNTIME_MAX_LIMBS)
        return set_error("invalid saved sampling matrix");
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("invalid saved sampling distribution");
    if (dist_type == GPU_MATRIX_DIST_GAUSS && (!(sigma > 0.0) || !std::isfinite(sigma)))
        return set_error("invalid saved sampling sigma");
    const GpuMatrixRange rectangle = range ? *range : GpuMatrixRange{0, out->rows, 0, out->cols};
    if (rectangle.row_start > rectangle.row_end || rectangle.row_end > out->rows ||
        rectangle.column_start > rectangle.column_end || rectangle.column_end > out->cols)
        return set_error("invalid saved sampling range");
    const size_t rows = rectangle.row_end - rectangle.row_start;
    const size_t columns = rectangle.column_end - rectangle.column_start;
    const size_t n = static_cast<size_t>(out->ctx->N);
    const size_t limbs = static_cast<size_t>(out->level) + 1;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (col_offset > full_ncol || columns > full_ncol - col_offset ||
        (out->rows && out->cols > limit / out->rows) ||
        (rows && full_ncol > limit / rows) || (rows && columns > limit / rows / n) ||
        (out->format != GPU_POLY_FORMAT_COEFF && out->format != GPU_POLY_FORMAT_EVAL) ||
        out->ctx->limb_gpu_ids.size() < limbs || out->ctx->moduli.size() < limbs)
        return set_error("invalid saved sampling shape or format");
    const size_t poly_count = rows * columns;
    const size_t chunks_per_poly = (n + 3) / 4;
    if (poly_count != 0 && chunks_per_poly > limit / poly_count)
        return set_error("saved sampling sample count overflow");
    const size_t chunks = poly_count * chunks_per_poly;
    if (chunks > limit - 255)
        return set_error("saved sampling block count overflow");
    const size_t blocks = chunks == 0 ? 0 : std::min<size_t>(65535, (chunks + 255) / 256);
    const bool has_transform = out->format == GPU_POLY_FORMAT_EVAL && poly_count != 0;
    if (descriptor->allocation_count != limbs + has_transform ||
        descriptor->stream_count != 1 ||
        (descriptor->launch_count == 0 && has_transform) ||
        (descriptor->launch_count != 0 && !has_transform))
        return set_error("saved sampling descriptor counts mismatch");

    prepared.out = out;
    prepared.range = rectangle;
    prepared.layout.offset = rectangle.row_start * out->cols + rectangle.column_start;
    prepared.layout.pitch = out->cols;
    prepared.dist_type = dist_type;
    prepared.sigma = sigma;
    prepared.max_coefficient_bound = max_coefficient_bound;
    prepared.coefficient_modulus = coefficient_modulus;
    prepared.poly_count = poly_count;
    prepared.columns = columns;
    prepared.full_ncol = full_ncol;
    prepared.col_offset = col_offset;
    prepared.n = n;
    prepared.limb_count = limbs;
    prepared.blocks = blocks;
    int device = -1;
    cudaStream_t stream = nullptr;
    int status = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const dim3 id = out->ctx->limb_gpu_ids[limb];
        status = matrix_limb_device(out, id, &device);
        if (status != 0) return status;
        if (id.x >= out->shared_limb_buffers.size())
            return set_error("invalid saved sampling partition");
        const auto &buffer = out->shared_limb_buffers[id.x];
        size_t stride = 0;
        uint8_t width = 0;
        if (!buffer.device_descriptors || id.y >= buffer.limb_count ||
            !matrix_limb_ptr_by_id(out, 0, id) ||
            !matrix_limb_metadata_by_id(out, id, &stride, &width) || width == 0 || width > 8 ||
            stride < n * width || !out->ctx->moduli[limb])
            return set_error("invalid saved sampling descriptor");
        if (limb == 0)
        {
            prepared.device = device;
            prepared.layout.descriptors = buffer.device_descriptors;
            status = matrix_limb_stream(out, id, &stream);
            if (status != 0 || !stream)
                return status ? status : set_error("invalid saved sampling stream");
        }
        else if (device != prepared.device || prepared.layout.descriptors != buffer.device_descriptors)
            return set_error("saved sampling requires one device and descriptor partition");
        prepared.layout.indices[limb] = id.y;
        prepared.layout.moduli[limb] = out->ctx->moduli[limb];
        dim3 key_id{};
        GpuPreparedResourceKey key{};
        status = gpu_prepared_limb_key(out->ctx, out->level, limb,
            GPU_PREPARED_STAGE_SAMPLING, &key_id, &key);
        if (status != 0 || gpu_prepared_require_allocation(descriptor, limb,
                GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0)
            return set_error("saved sampling completion differs from descriptor");
        const auto &event_entry = descriptor->allocations[limb];
        if (event_entry.rows != 0 || event_entry.columns != 0 ||
            event_entry.level != -1 || event_entry.format != -1)
            return set_error("saved sampling completion metadata is invalid");
    }
    const auto &sampling_stream = descriptor->streams[0];
    if (sampling_stream.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        gpu_prepared_require_stream_slot(out->ctx, out->ctx->limb_gpu_ids[0].x, stream,
            sampling_stream.pool_slot) != 0)
        return set_error("saved sampling owner stream differs from descriptor");
    dim3 sampling_limb{};
    GpuPreparedResourceKey sampling_key{};
    status = gpu_prepared_limb_key(out->ctx, out->level, 0,
        GPU_PREPARED_STAGE_SAMPLING, &sampling_limb, &sampling_key);
    if (status != 0 || std::memcmp(&sampling_stream.key, &sampling_key, sizeof(sampling_key)) != 0)
        return set_error("saved sampling stream key differs from descriptor");
    if (!has_transform) return 0;

    dim3 first_limb{};
    GpuPreparedResourceKey ntt_key{};
    status = gpu_prepared_limb_key(out->ctx, out->level, 0,
        GPU_PREPARED_STAGE_NTT, &first_limb, &ntt_key);
    if (status != 0) return status;
    const auto &allocation = descriptor->allocations[limbs];
    if (descriptor->launch_count > limit / sizeof(GpuPreparedNttLaunchLayout) ||
        gpu_prepared_require_allocation(descriptor, limbs, GPU_PREPARED_PLAN_HOST_ONLY,
            &ntt_key, descriptor->launch_count * sizeof(GpuPreparedNttLaunchLayout), alignof(void *)) != 0 ||
        allocation.rows != 0 || allocation.columns != 0 || allocation.level != -1 || allocation.format != -1)
        return set_error("saved sampling transform allocation differs from descriptor");
    const auto &stream_entry = descriptor->streams[0];
    if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        gpu_prepared_require_stream_slot(out->ctx, first_limb.x, stream, stream_entry.pool_slot) != 0 ||
        std::memcmp(&stream_entry.key, &sampling_key, sizeof(sampling_key)) != 0)
        return set_error("saved sampling transform stream differs from descriptor");
    ntt_descriptor = GpuPreparedPlanDescriptor{};
    ntt_descriptor.allocation_count = 1;
    ntt_descriptor.allocations[0] = allocation;
    ntt_descriptor.stream_count = 1;
    ntt_descriptor.streams[0] = GpuPreparedStreamFootprint{ntt_key,
        GPU_PREPARED_STREAM_CONTEXT_REUSED, stream_entry.pool_slot};
    ntt_descriptor.launch_count = descriptor->launch_count;
    for (size_t index = 0; index < descriptor->launch_count; ++index)
        ntt_descriptor.launches[index] = descriptor->launches[index];
    return 0;
}

static int construct_saved_sampling_resources(
    GpuPreparedSampling &prepared)
{
    for (size_t limb = 0; limb < prepared.limb_count; ++limb)
    {
        auto event = std::make_unique<GpuCudaResource>();
        const int status = event->acquire(
            prepared.out->ctx, prepared.device, GPU_PREPARED_COMPLETION_EVENT);
        if (status != 0) return status;
        prepared.completion[limb] = std::move(event);
    }
    prepared.completed_event = prepared.completion[0]->event;
    return 0;
}

static int prepare_sampling_legacy_impl(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range,
    GpuPreparedSampling **plan)
{
    if (!plan) return set_error("null prepared sampling output");
    *plan = nullptr;
    try {
        auto prepared = std::make_unique<GpuPreparedSampling>();
        const int status = prepare_sampling_layout(
            out, dist_type, sigma, max_coefficient_bound, coefficient_modulus,
            full_ncol, col_offset, range, *prepared);
        if (status != 0) return status;
        *plan = prepared.release();
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

extern "C" int gpu_matrix_prepare_sampling_with_layout(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range,
    const GpuPreparedPlanDescriptor *layout,
    GpuPreparedSampling **plan)
{
    if (!layout || !plan || gpu_prepared_validate_descriptor(layout) != 0)
        return set_error("prepared sampling layout is missing");
    *plan = nullptr;
    // validate_saved_sampling_descriptor performs the
    // gpu_prepared_require_allocation and gpu_prepared_require_stream_slot
    // checks before any completion event or transform is acquired.
    try
    {
        auto prepared = std::make_unique<GpuPreparedSampling>();
        GpuPreparedPlanDescriptor ntt_descriptor{};
        const int status = validate_saved_sampling_descriptor(
            out, dist_type, sigma, max_coefficient_bound, coefficient_modulus,
            full_ncol, col_offset, range, layout, *prepared, ntt_descriptor);
        if (status != 0) return status;
        const bool has_transform = out->format == GPU_POLY_FORMAT_EVAL && prepared->poly_count != 0;
        if (has_transform)
        {
            // This direct NTT bind validates and materializes the saved launch
            // table before any sampling completion event is acquired.
            const int transform_status = gpu_matrix_prepare_ntt_plan_with_layout(
                out, &prepared->range, true, &ntt_descriptor, &prepared->transform);
            if (transform_status != 0) return transform_status;
        }
        const int resource_status = construct_saved_sampling_resources(*prepared);
        if (resource_status != 0) return resource_status;
        *plan = prepared.release();
        return 0;
    }
    catch (const std::exception &error) { return set_error(error.what()); }
}

// Standalone non-prepared sampling API. Prepared runtime callers use the
// descriptor-driven entry above; this entry remains for direct primitive use.
extern "C" int gpu_matrix_prepare_sampling(
    GpuMatrix *out,
    int dist_type,
    double sigma,
    uint64_t max_coefficient_bound,
    uint64_t coefficient_modulus,
    size_t full_ncol,
    size_t col_offset,
    const GpuMatrixRange *range,
    GpuPreparedSampling **plan)
{
    return prepare_sampling_legacy_impl(
        out, dist_type, sigma, max_coefficient_bound, coefficient_modulus,
        full_ncol, col_offset, range, plan);
}

extern "C" int gpu_matrix_submit_sampling(
    const GpuPreparedSampling *opaque,
    GpuRngSeed seed)
{
    if (!opaque || !opaque->out || !opaque->stream)
        return set_error("invalid prepared sampling plan");
    auto *prepared = const_cast<GpuPreparedSampling *>(opaque);
    int status = cudaSetDevice(prepared->device);
    if (status != cudaSuccess) return set_error(static_cast<cudaError_t>(status));
    if (prepared->completed_event) {
        const auto error = cudaStreamWaitEvent(prepared->stream, prepared->completed_event, 0);
        if (error != cudaSuccess) return set_error(error);
    }
    status = matrix_wait_all_limb_streams(prepared->out, prepared->device, prepared->stream, true);
    if (status != 0) return status;
    if (prepared->poly_count != 0) {
        gpu_test_record_kernel_launch();
        matrix_sample_distribution_multi_limb_kernel<<<
            dim3(prepared->blocks, 1, prepared->limb_count), 256, 0, prepared->stream>>>(
            prepared->layout,
            prepared->poly_count,
            prepared->columns,
            prepared->full_ncol,
            prepared->col_offset,
            prepared->n,
            prepared->dist_type,
            prepared->sigma,
            prepared->max_coefficient_bound,
            prepared->coefficient_modulus,
            seed);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        status = matrix_record_all_limb_writes(prepared->out, prepared->stream);
        if (status != 0) return status;
    }
    if (prepared->transform)
    {
        status = gpu_matrix_submit_ntt_plan(prepared->transform, prepared->out);
        if (status != 0) return status;
    }
    for (size_t limb = 0; limb < prepared->limb_count; ++limb)
    {
        if (!prepared->completion[limb]) continue;
        const cudaError_t error = cudaEventRecord(prepared->completion[limb]->event, prepared->stream);
        if (error != cudaSuccess) return set_error(error);
    }
    return 0;
}

extern "C" void gpu_matrix_destroy_sampling(GpuPreparedSampling *plan)
{
    delete plan;
}


// Fully write homogeneous coefficient owners with unbounded Gaussian samples,
// then convert them to evaluation format. Seeds retain the scalar sampler's
// exact logical coordinates and are independent of the batch ordinal.
extern "C" int gpu_matrix_sample_gaussian_batch(
    GpuMatrix *const *outputs, const GpuRngSeed *seeds, size_t count, double sigma)
{
    if (count == 0) return 0;
    auto *first = outputs[0];
    GpuAllocationActivity activity(first->ctx->execution.get(), -1);
    const size_t polynomials = first->rows * first->cols;
    if (polynomials == 0) {
        for (size_t index = 0; index < count; ++index)
            outputs[index]->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    const size_t n = first->ctx->N;
    const size_t limbs = first->level + 1;
    const size_t partition = first->ctx->limb_gpu_ids[0].x;
    int device = -1;
    cudaStream_t stream = nullptr;
    int status = matrix_limb_device(first, first->ctx->limb_gpu_ids[0], &device);
    if (status == 0) status = matrix_limb_stream(first, first->ctx->limb_gpu_ids[0], &stream);
    if (status != 0) return status;
    auto error = cudaSetDevice(device);
    if (error != cudaSuccess) return set_error(error);
    MatrixSampleDescriptors layout{};
    layout.pitch = first->cols;
    for (size_t limb = 0; limb < limbs; ++limb) {
        layout.indices[limb] = first->ctx->limb_gpu_ids[limb].y;
        layout.moduli[limb] = first->ctx->moduli[limb];
    }
    if (n > std::numeric_limits<size_t>::max() - 3 ||
        polynomials > std::numeric_limits<size_t>::max() / ((n + 3) / 4))
        return set_error("sampling batch dimensions overflow");
    const size_t chunks = polynomials * ((n + 3) / 4);
    if (chunks > std::numeric_limits<size_t>::max() - 255)
        return set_error("sampling batch block overflow");
    const size_t blocks = std::min<size_t>(65535, (chunks + 255) / 256);
    for (size_t offset = 0; offset < count; offset += kGaussianBatchMatrices) {
        const size_t width = std::min(kGaussianBatchMatrices, count - offset);
        GaussianBatchDescriptors jobs{};
        for (size_t local = 0; local < width; ++local) {
            auto *output = outputs[offset + local];
            status = matrix_wait_all_limb_streams(output, device, stream, true);
            if (status != 0) return status;
            jobs.outputs[local] = output->shared_limb_buffers[partition].device_descriptors;
            jobs.seeds[local] = seeds[offset + local];
        }
        matrix_sample_gaussian_batch_kernel<<<dim3(blocks, width, limbs), 256, 0, stream>>>(
            jobs, layout, polynomials, first->cols, n, sigma);
        error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        for (size_t local = 0; local < width; ++local) {
            status = matrix_record_all_limb_writes(outputs[offset + local], stream);
            if (status != 0) return status;
        }
        if (polynomials <= 65535 && width <= 65535 / limbs) {
            status = gpu_matrix_ntt_in_place_batch(outputs + offset, width);
            if (status != 0) return status;
        } else {
            // Native scalar transforms support shapes outside the batch grid.
            for (size_t local = 0; local < width; ++local) {
                status = run_matrix_transform_u64<true>(outputs[offset + local], nullptr);
                if (status != 0) return status;
                outputs[offset + local]->format = GPU_POLY_FORMAT_EVAL;
            }
        }
    }
    return 0;
}
