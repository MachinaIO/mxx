#include "gpu_admission.cuh"
#include <array>
constexpr size_t kSampleP1LocalMaxM = 8;

using gpu_chacha::GpuRngSeed;

namespace
{
    struct ThreadLocalOwnerLinkEventState
    {
        int device = -1;
        cudaEvent_t event = nullptr;

        ~ThreadLocalOwnerLinkEventState()
        {
            if (!event || device < 0)
            {
                return;
            }
            GpuAllocationActivity activity(nullptr, device);
            cudaError_t err = cudaSetDevice(device);
            if (err == cudaSuccess)
            {
                err = cudaEventDestroy(event);
            }
            if (err != cudaSuccess) gpu_device_mark_allocation_unknown(device);
            event = nullptr;
            device = -1;
        }
    };

    thread_local ThreadLocalOwnerLinkEventState g_thread_local_owner_link_event;

    int matrix_get_thread_local_owner_link_event(const GpuContext *ctx, int device, cudaEvent_t *out_event, GpuCudaResource &resource)
    {
        if (device < 0 || !out_event)
        {
            return set_error("invalid matrix_get_thread_local_owner_link_event arguments");
        }

        if (ctx->execution->resource_admission_required.load(std::memory_order_acquire)) {
            const int status = resource.acquire(ctx, device, GPU_PREPARED_COMPLETION_EVENT);
            if (status == 0) *out_event = resource.event;
            return status;
        }
        gpu_claim_trace_record(GPU_PREPARED_COMPLETION_EVENT, 0, 0, -1, -1, 0, 1);
        auto &tls = g_thread_local_owner_link_event;
        if (tls.event && tls.device == device)
        {
            *out_event = tls.event;
            return 0;
        }

        GpuAllocationActivity activity(nullptr, device);
        if (tls.event)
        {
            GpuAllocationActivity previous_activity(nullptr, tls.device);
            cudaError_t err = cudaSetDevice(tls.device);
            if (err != cudaSuccess)
            {
                gpu_device_mark_allocation_unknown(tls.device);
                return set_error(err);
            }
            err = cudaEventDestroy(tls.event);
            if (err != cudaSuccess)
            {
                gpu_device_mark_allocation_unknown(tls.device);
                return set_error(err);
            }
            tls.event = nullptr;
            tls.device = -1;
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaEventCreateWithFlags(&tls.event, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            tls.event = nullptr;
            tls.device = -1;
            return set_error(err);
        }
        tls.device = device;
        *out_event = tls.event;
        return 0;
    }
}

struct GpuP1CovarianceCache
{
    GpuContext *ctx = nullptr;
    // The cache can outlive the final matrix/parameter wrapper. Retain its
    // execution resources directly; destruction must not dereference ctx.
    std::shared_ptr<GpuExecutionOwner> execution;
    int level = -1;
    size_t d_rows = 0;
    size_t n = 0;
    size_t m = 0;
    uint64_t modulus = 0;
    double sigma = 0.0;
    double s = 0.0;
    int device = -1;
    // Claimable owners: under sealed admission the stream/bridge event and the
    // two device spans come from prepared slots; open domains allocate them.
    GpuCudaResource stream_resource;
    GpuDeviceWorkspace sqrt_owner;
    GpuDeviceWorkspace update_owner;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    double *sqrt_var = nullptr;      // [coeff][row]
    double *update_coeff = nullptr;  // [coeff][sampled_row][updated_row]
};

namespace
{
    // These are CUDA block-shape constants, not a preimage chunk limit. Every
    // kernel below receives the runtime column count selected by
    // AUX_SAMPLING_CHUNK_WIDTH and covers any final partial tile.
    constexpr int kPreimageTileM = 4;
    constexpr int kPreimageTileN = 32;
    constexpr int kPreimageTileK = 16;

    __device__ __forceinline__ uint64_t sub_mod_preimage_u64(
        uint64_t lhs,
        uint64_t rhs,
        uint64_t modulus)
    {
        return lhs >= rhs ? lhs - rhs : modulus - (rhs - lhs);
    }

    __global__ void matrix_mul_vertical_pair_kernel(
        const uint8_t *top_base,
        const uint8_t *bottom_base,
        const uint8_t *rhs_base,
        uint8_t *out_base,
        size_t top_rows,
        size_t bottom_rows,
        size_t inner,
        size_t cols,
        size_t n,
        size_t top_stride_bytes,
        size_t bottom_stride_bytes,
        size_t rhs_stride_bytes,
        size_t out_stride_bytes,
        uint8_t top_coeff_bytes,
        uint8_t bottom_coeff_bytes,
        uint8_t rhs_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        __shared__ uint64_t lhs_tile[kPreimageTileM][kPreimageTileK];
        __shared__ uint64_t rhs_tile[kPreimageTileK][kPreimageTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kPreimageTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kPreimageTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t total_rows = top_rows + bottom_rows;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;

        for (size_t coeff_idx = static_cast<size_t>(blockIdx.z);
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z))
        {
            uint64_t acc = 0;
            for (size_t k0 = 0; k0 < inner; k0 += kPreimageTileK)
            {
                for (int i = tid; i < kPreimageTileM * kPreimageTileK; i += thread_count)
                {
                    const int local_row = i / kPreimageTileK;
                    const int local_k = i - local_row * kPreimageTileK;
                    const size_t input_row = row_base + static_cast<size_t>(local_row);
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    uint64_t value = 0;
                    if (input_row < total_rows && input_k < inner)
                    {
                        if (input_row < top_rows)
                        {
                            value = matrix_load_limb_u64(
                                top_base,
                                input_row * inner + input_k,
                                coeff_idx,
                                top_stride_bytes,
                                top_coeff_bytes);
                        }
                        else
                        {
                            value = matrix_load_limb_u64(
                                bottom_base,
                                (input_row - top_rows) * inner + input_k,
                                coeff_idx,
                                bottom_stride_bytes,
                                bottom_coeff_bytes);
                        }
                    }
                    lhs_tile[local_row][local_k] = value;
                }
                for (int i = tid; i < kPreimageTileK * kPreimageTileN; i += thread_count)
                {
                    const int local_k = i / kPreimageTileN;
                    const int local_col = i - local_k * kPreimageTileN;
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    const size_t input_col = col_base + static_cast<size_t>(local_col);
                    uint64_t value = 0;
                    if (input_k < inner && input_col < cols)
                    {
                        value = matrix_load_limb_u64(
                            rhs_base,
                            input_k * cols + input_col,
                            coeff_idx,
                            rhs_stride_bytes,
                            rhs_coeff_bytes);
                    }
                    rhs_tile[local_k][local_col] = value;
                }
                __syncthreads();

                if (row < total_rows && col < cols)
                {
                    for (int local_k = 0; local_k < kPreimageTileK; ++local_k)
                    {
                        acc = add_mod_u64(
                            acc,
                            mul_mod_u64(
                                lhs_tile[threadIdx.y][local_k],
                                rhs_tile[local_k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }

            if (row < total_rows && col < cols)
            {
                matrix_store_limb_u64(
                    out_base,
                    row * cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes,
                    acc);
            }
        }
    }

    __global__ void matrix_preimage_residual_kernel(
        const uint8_t *target_base,
        const uint8_t *public_base,
        const uint8_t *p1_base,
        const uint8_t *p2_base,
        uint8_t *out_base,
        size_t rows,
        size_t p1_rows,
        size_t p2_rows,
        size_t p1_cols,
        size_t p2_cols,
        size_t out_cols,
        size_t n,
        size_t target_stride_bytes,
        size_t public_stride_bytes,
        size_t p1_stride_bytes,
        size_t p2_stride_bytes,
        size_t out_stride_bytes,
        uint8_t target_coeff_bytes,
        uint8_t public_coeff_bytes,
        uint8_t p1_coeff_bytes,
        uint8_t p2_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        __shared__ uint64_t public_tile[kPreimageTileM][kPreimageTileK];
        __shared__ uint64_t perturbation_tile[kPreimageTileK][kPreimageTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kPreimageTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kPreimageTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t inner = p1_rows + p2_rows;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;

        for (size_t coeff_idx = static_cast<size_t>(blockIdx.z);
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z))
        {
            uint64_t acc = 0;
            if (row < rows && col < out_cols)
            {
                acc = matrix_load_limb_u64(
                    target_base,
                    row * out_cols + col,
                    coeff_idx,
                    target_stride_bytes,
                    target_coeff_bytes);
            }

            for (size_t k0 = 0; k0 < inner; k0 += kPreimageTileK)
            {
                for (int i = tid; i < kPreimageTileM * kPreimageTileK; i += thread_count)
                {
                    const int local_row = i / kPreimageTileK;
                    const int local_k = i - local_row * kPreimageTileK;
                    const size_t input_row = row_base + static_cast<size_t>(local_row);
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    uint64_t value = 0;
                    if (input_row < rows && input_k < inner)
                    {
                        value = matrix_load_limb_u64(
                            public_base,
                            input_row * inner + input_k,
                            coeff_idx,
                            public_stride_bytes,
                            public_coeff_bytes);
                    }
                    public_tile[local_row][local_k] = value;
                }
                for (int i = tid; i < kPreimageTileK * kPreimageTileN; i += thread_count)
                {
                    const int local_k = i / kPreimageTileN;
                    const int local_col = i - local_k * kPreimageTileN;
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    const size_t input_col = col_base + static_cast<size_t>(local_col);
                    uint64_t value = 0;
                    if (input_k < inner && input_col < out_cols)
                    {
                        if (input_k < p1_rows)
                        {
                            value = matrix_load_limb_u64(
                                p1_base,
                                input_k * p1_cols + input_col,
                                coeff_idx,
                                p1_stride_bytes,
                                p1_coeff_bytes);
                        }
                        else
                        {
                            value = matrix_load_limb_u64(
                                p2_base,
                                (input_k - p1_rows) * p2_cols + input_col,
                                coeff_idx,
                                p2_stride_bytes,
                                p2_coeff_bytes);
                        }
                    }
                    perturbation_tile[local_k][local_col] = value;
                }
                __syncthreads();

                if (row < rows && col < out_cols)
                {
                    for (int local_k = 0; local_k < kPreimageTileK; ++local_k)
                    {
                        acc = sub_mod_preimage_u64(
                            acc,
                            mul_mod_u64(
                                public_tile[threadIdx.y][local_k],
                                perturbation_tile[local_k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }

            if (row < rows && col < out_cols)
            {
                matrix_store_limb_u64(
                    out_base,
                    row * out_cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes,
                    acc);
            }
        }
    }

    __global__ void matrix_preimage_add_correction_top_kernel(
        const uint8_t *r_base,
        const uint8_t *e_base,
        const uint8_t *z_base,
        uint8_t *out_base,
        size_t d,
        size_t inner,
        size_t z_cols,
        size_t out_cols,
        size_t n,
        size_t r_stride_bytes,
        size_t e_stride_bytes,
        size_t z_stride_bytes,
        size_t out_stride_bytes,
        uint8_t r_coeff_bytes,
        uint8_t e_coeff_bytes,
        uint8_t z_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        __shared__ uint64_t trapdoor_tile[kPreimageTileM][kPreimageTileK];
        __shared__ uint64_t z_tile[kPreimageTileK][kPreimageTileN];

        const size_t row_base = static_cast<size_t>(blockIdx.y) * kPreimageTileM;
        const size_t col_base = static_cast<size_t>(blockIdx.x) * kPreimageTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t col = col_base + threadIdx.x;
        const size_t top_rows = 2 * d;
        const int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;

        for (size_t coeff_idx = static_cast<size_t>(blockIdx.z);
             coeff_idx < n;
             coeff_idx += static_cast<size_t>(gridDim.z))
        {
            uint64_t acc = 0;
            if (row < top_rows && col < out_cols)
            {
                acc = matrix_load_limb_u64(
                    out_base,
                    row * out_cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes);
            }

            for (size_t k0 = 0; k0 < inner; k0 += kPreimageTileK)
            {
                for (int i = tid; i < kPreimageTileM * kPreimageTileK; i += thread_count)
                {
                    const int local_row = i / kPreimageTileK;
                    const int local_k = i - local_row * kPreimageTileK;
                    const size_t input_row = row_base + static_cast<size_t>(local_row);
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    uint64_t value = 0;
                    if (input_row < top_rows && input_k < inner)
                    {
                        if (input_row < d)
                        {
                            value = matrix_load_limb_u64(
                                r_base,
                                input_row * inner + input_k,
                                coeff_idx,
                                r_stride_bytes,
                                r_coeff_bytes);
                        }
                        else
                        {
                            value = matrix_load_limb_u64(
                                e_base,
                                (input_row - d) * inner + input_k,
                                coeff_idx,
                                e_stride_bytes,
                                e_coeff_bytes);
                        }
                    }
                    trapdoor_tile[local_row][local_k] = value;
                }
                for (int i = tid; i < kPreimageTileK * kPreimageTileN; i += thread_count)
                {
                    const int local_k = i / kPreimageTileN;
                    const int local_col = i - local_k * kPreimageTileN;
                    const size_t input_k = k0 + static_cast<size_t>(local_k);
                    const size_t input_col = col_base + static_cast<size_t>(local_col);
                    uint64_t value = 0;
                    if (input_k < inner && input_col < out_cols)
                    {
                        value = matrix_load_limb_u64(
                            z_base,
                            input_k * z_cols + input_col,
                            coeff_idx,
                            z_stride_bytes,
                            z_coeff_bytes);
                    }
                    z_tile[local_k][local_col] = value;
                }
                __syncthreads();

                if (row < top_rows && col < out_cols)
                {
                    for (int local_k = 0; local_k < kPreimageTileK; ++local_k)
                    {
                        acc = add_mod_u64(
                            acc,
                            mul_mod_u64(
                                trapdoor_tile[threadIdx.y][local_k],
                                z_tile[local_k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }

            if (row < top_rows && col < out_cols)
            {
                matrix_store_limb_u64(
                    out_base,
                    row * out_cols + col,
                    coeff_idx,
                    out_stride_bytes,
                    out_coeff_bytes,
                    acc);
            }
        }
    }

    __global__ void matrix_preimage_add_correction_bottom_kernel(
        const uint8_t *z_base,
        uint8_t *out_base,
        size_t top_rows,
        size_t z_rows,
        size_t z_cols,
        size_t out_cols,
        size_t n,
        size_t z_stride_bytes,
        size_t out_stride_bytes,
        uint8_t z_coeff_bytes,
        uint8_t out_coeff_bytes,
        uint64_t modulus)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = z_rows * out_cols * n;
        if (idx >= total)
        {
            return;
        }
        const size_t coeff_idx = idx % n;
        const size_t out_poly_idx = idx / n;
        const size_t row = out_poly_idx / out_cols;
        const size_t col = out_poly_idx % out_cols;
        const size_t out_entry = top_rows * out_cols + out_poly_idx;
        const uint64_t base_value = matrix_load_limb_u64(
            out_base,
            out_entry,
            coeff_idx,
            out_stride_bytes,
            out_coeff_bytes);
        const uint64_t z_value = matrix_load_limb_u64(
            z_base,
            row * z_cols + col,
            coeff_idx,
            z_stride_bytes,
            z_coeff_bytes);
        matrix_store_limb_u64(
            out_base,
            out_entry,
            coeff_idx,
            out_stride_bytes,
            out_coeff_bytes,
            add_mod_u64(base_value, z_value, modulus));
    }

    int preimage_const_limb_view(
        const GpuMatrix *matrix,
        const dim3 &limb_id,
        const uint8_t **base,
        size_t *stride_bytes,
        uint8_t *coeff_bytes,
        int *device)
    {
        if (!matrix || !base || !stride_bytes || !coeff_bytes || !device)
        {
            return set_error("invalid preimage_const_limb_view arguments");
        }
        *base = matrix_limb_ptr_by_id(matrix, 0, limb_id);
        if (!*base)
        {
            return set_error("null matrix limb in preimage_const_limb_view");
        }
        if (!matrix_limb_metadata_by_id(matrix, limb_id, stride_bytes, coeff_bytes))
        {
            return set_error("invalid matrix limb metadata in preimage_const_limb_view");
        }
        return matrix_limb_device(matrix, limb_id, device);
    }

    int preimage_mut_limb_view(
        GpuMatrix *matrix,
        const dim3 &limb_id,
        uint8_t **base,
        size_t *stride_bytes,
        uint8_t *coeff_bytes,
        int *device,
        cudaStream_t *stream)
    {
        if (!matrix || !base || !stride_bytes || !coeff_bytes || !device || !stream)
        {
            return set_error("invalid preimage_mut_limb_view arguments");
        }
        *base = matrix_limb_ptr_by_id(matrix, 0, limb_id);
        if (!*base)
        {
            return set_error("null matrix limb in preimage_mut_limb_view");
        }
        if (!matrix_limb_metadata_by_id(matrix, limb_id, stride_bytes, coeff_bytes))
        {
            return set_error("invalid matrix limb metadata in preimage_mut_limb_view");
        }
        int status = matrix_limb_device(matrix, limb_id, device);
        if (status != 0)
        {
            return status;
        }
        return matrix_limb_stream(matrix, limb_id, stream);
    }
}

__global__ void matrix_precompute_p1_covariance_kernel(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    size_t d,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    double *cov_workspace,
    double *sqrt_var_out,
    double *update_coeff_out)
{
    const size_t coeff_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coeff_idx >= n)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0)
    {
        return;
    }

    double *cov = cov_workspace + coeff_idx * m * m;
    double *sqrt_var = sqrt_var_out + coeff_idx * m;
    double *update_coeff = update_coeff_out + coeff_idx * m * m;

    const double sigma2 = sigma * sigma;
    const double s2 = s * s;
    const double fallback_var = dgg_stddev * dgg_stddev;
    const double eps = 1e-9;

    for (size_t i = 0; i < d; ++i)
    {
        for (size_t j = 0; j < d; ++j)
        {
            const size_t ij = matrix_index(i, j, d);
            const size_t ji = matrix_index(j, i, d);
            const double a_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(a_base, ij, coeff_idx, a_stride_bytes, a_coeff_bytes),
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(d_base, ij, coeff_idx, d_stride_bytes, d_coeff_bytes),
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ij, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ji, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));

            cov[matrix_index(i, j, m)] = -sigma2 * a_ij + (i == j ? s2 : 0.0);
            cov[matrix_index(i + d, j + d, m)] = -sigma2 * d_ij + (i == j ? s2 : 0.0);
            cov[matrix_index(i, j + d, m)] = -sigma2 * b_ij;
            cov[matrix_index(i + d, j, m)] = -sigma2 * b_ji;
        }
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        double var = cov[matrix_index(tt, tt, m)];
        if (!(var > eps))
        {
            var = fallback_var;
        }
        sqrt_var[tt] = sqrt(var);

        for (int i = 0; i < t; ++i)
        {
            update_coeff[tt * m + static_cast<size_t>(i)] =
                cov[matrix_index(static_cast<size_t>(i), tt, m)] / var;
        }

        if (t == 0)
        {
            break;
        }

        for (int i = 0; i < t; ++i)
        {
            const double coeff_i = update_coeff[tt * m + static_cast<size_t>(i)];
            for (int j = 0; j <= i; ++j)
            {
                const double col_j =
                    update_coeff[tt * m + static_cast<size_t>(j)] * var;
                double updated =
                    cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] -
                    coeff_i * col_j;
                cov[matrix_index(static_cast<size_t>(i), static_cast<size_t>(j), m)] = updated;
                cov[matrix_index(static_cast<size_t>(j), static_cast<size_t>(i), m)] = updated;
            }
        }
    }
}

__global__ void matrix_sample_p1_integer_cached_kernel_small(
    const uint8_t *tp2_base,
    size_t tp2_stride_bytes,
    uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base,
    const double *update_coeff_base,
    size_t d,
    size_t cols,
    size_t n,
    int64_t *sampled_out,
    uint64_t modulus,
    double c_scale,
    GpuRngSeed seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_samples = cols * n;
    if (idx >= total_samples)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0 || m > kSampleP1LocalMaxM)
    {
        return;
    }

    const size_t col_idx = idx / n;
    const size_t coeff_idx = idx - col_idx * n;
    double mean[kSampleP1LocalMaxM];
    int64_t sampled[kSampleP1LocalMaxM];
    const double *sqrt_var = sqrt_var_base + coeff_idx * m;
    const double *update_coeff = update_coeff_base + coeff_idx * m * m;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt_var[tt]);
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                update_coeff[tt * m + static_cast<size_t>(i)] * delta;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_sample_p1_integer_cached_kernel_large(
    const uint8_t *tp2_base,
    size_t tp2_stride_bytes,
    uint8_t tp2_coeff_bytes,
    const double *sqrt_var_base,
    const double *update_coeff_base,
    size_t d,
    size_t cols,
    size_t n,
    size_t sample_start,
    size_t sample_count,
    double *mean_workspace,
    int64_t *sampled_workspace,
    int64_t *sampled_out,
    uint64_t modulus,
    double c_scale,
    GpuRngSeed seed)
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
    double *mean = mean_workspace + idx * m;
    int64_t *sampled = sampled_workspace + idx * m;
    const double *sqrt_var = sqrt_var_base + coeff_idx * m;
    const double *update_coeff = update_coeff_base + coeff_idx * m * m;

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
        0x7065727475726231ULL);

    for (size_t row = 0; row < m; ++row)
    {
        const size_t tp_idx = matrix_index(row, col_idx, cols);
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
        mean[row] = c_scale * c_centered;
    }

    for (int t = static_cast<int>(m) - 1; t >= 0; --t)
    {
        const size_t tt = static_cast<size_t>(t);
        const double mu = mean[tt];
        const int64_t z = sample_integer_karney(rng, mu, sqrt_var[tt]);
        sampled[tt] = z;

        if (t == 0)
        {
            break;
        }

        const double delta = static_cast<double>(z) - mu;
        for (int i = 0; i < t; ++i)
        {
            mean[static_cast<size_t>(i)] +=
                update_coeff[tt * m + static_cast<size_t>(i)] * delta;
        }
    }

    for (size_t row = 0; row < m; ++row)
    {
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_sample_p1_integer_kernel_small(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    const uint8_t *tp2_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    size_t tp2_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    uint8_t tp2_coeff_bytes,
    size_t d,
    size_t cols,
    size_t n,
    int64_t *sampled_out,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    GpuRngSeed seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_samples = cols * n;
    if (idx >= total_samples)
    {
        return;
    }

    const size_t m = d * 2;
    if (m == 0 || m > kSampleP1LocalMaxM)
    {
        return;
    }

    const size_t col_idx = idx / n;
    const size_t coeff_idx = idx - col_idx * n;
    double cov[kSampleP1LocalMaxM * kSampleP1LocalMaxM];
    double mean[kSampleP1LocalMaxM];
    double col_buf[kSampleP1LocalMaxM];
    int64_t sampled[kSampleP1LocalMaxM];

    DeviceChaChaRng rng;
    rng_init(
        rng,
        seed,
        static_cast<uint64_t>(col_idx + 1),
        static_cast<uint64_t>(coeff_idx + 1),
        0,
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
            const double a_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(a_base, ij, coeff_idx, a_stride_bytes, a_coeff_bytes),
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(d_base, ij, coeff_idx, d_stride_bytes, d_coeff_bytes),
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ij, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ji, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));

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
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
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
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_sample_p1_integer_kernel_large(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    const uint8_t *tp2_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    size_t tp2_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    uint8_t tp2_coeff_bytes,
    size_t d,
    size_t cols,
    size_t n,
    size_t sample_start,
    size_t sample_count,
    double *cov_workspace,
    double *mean_workspace,
    double *col_workspace,
    int64_t *sampled_workspace,
    int64_t *sampled_out,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    GpuRngSeed seed)
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
        0,
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
            const double a_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(a_base, ij, coeff_idx, a_stride_bytes, a_coeff_bytes),
                modulus));
            const double d_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(d_base, ij, coeff_idx, d_stride_bytes, d_coeff_bytes),
                modulus));
            const double b_ij = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ij, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));
            const double b_ji = static_cast<double>(centered_residue_i64(
                matrix_load_limb_u64(b_base, ji, coeff_idx, b_stride_bytes, b_coeff_bytes),
                modulus));

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
        const double c_centered = static_cast<double>(centered_residue_i64(
            matrix_load_limb_u64(tp2_base, tp_idx, coeff_idx, tp2_stride_bytes, tp2_coeff_bytes),
            modulus));
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
        const size_t out_idx = matrix_index(row, col_idx, cols) * n + coeff_idx;
        sampled_out[out_idx] = sampled[row];
    }
}

__global__ void matrix_scatter_p1_integer_to_limb_kernel(
    const int64_t *sampled_in,
    uint8_t *out_base,
    size_t out_stride_bytes,
    uint8_t out_coeff_bytes,
    size_t entry_count,
    size_t n,
    uint64_t modulus)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = entry_count * n;
    if (idx >= total)
    {
        return;
    }

    const size_t entry_idx = idx / n;
    const size_t coeff_idx = idx - entry_idx * n;
    matrix_store_limb_u64(
        out_base,
        entry_idx,
        coeff_idx,
        out_stride_bytes,
        out_coeff_bytes,
        signed_mod_i64(sampled_in[idx], modulus));
}

__global__ void matrix_gauss_samp_gq_arb_base_sample_kernel(
    const uint8_t *src_base,
    int64_t *sampled_digits,
    size_t poly_count,
    size_t n,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    uint64_t tower_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    double c,
    uint32_t tower_idx,
    GpuRngSeed seed)
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

    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;

    uint64_t value =
        matrix_load_limb_u64(src_base, poly_idx, coeff_idx, src_stride_bytes, src_coeff_bytes);
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

    for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
    {
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

        const size_t sample_idx =
            (poly_idx * static_cast<size_t>(digits_per_tower) + static_cast<size_t>(digit_idx)) * n + coeff_idx;
        sampled_digits[sample_idx] = out_digit;
    }
}

__global__ void matrix_gauss_samp_gq_arb_base_scatter_kernel(
    const int64_t *sampled_digits,
    uint8_t *const *dst_bases,
    const size_t *dst_stride_bytes,
    const uint8_t *dst_coeff_bytes,
    const uint64_t *out_moduli,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    size_t src_digit_offset,
    uint32_t digits_per_tower)
{
    const size_t out_limb = static_cast<size_t>(blockIdx.y);
    if (out_limb >= out_limb_count)
    {
        return;
    }
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    if (!sampled_digits || !dst_bases || !dst_stride_bytes || !dst_coeff_bytes || !out_moduli || src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return;
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return;
    }

    uint8_t *dst_base = dst_bases[out_limb];
    const size_t dst_stride = dst_stride_bytes[out_limb];
    const uint8_t dst_bytes = dst_coeff_bytes[out_limb];
    const uint64_t out_modulus = out_moduli[out_limb];
    if (!dst_base || dst_bytes == 0 || dst_stride < n * static_cast<size_t>(dst_bytes))
    {
        return;
    }

    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;
    const size_t row = poly_idx / src_cols;
    const size_t col = poly_idx - row * src_cols;
    for (uint32_t digit_idx = 0; digit_idx < digits_per_tower; ++digit_idx)
    {
        const size_t sample_idx =
            (poly_idx * static_cast<size_t>(digits_per_tower) + static_cast<size_t>(digit_idx)) * n + coeff_idx;
        const int64_t out_digit = sampled_digits[sample_idx];
        const size_t out_row = row * log_base_q + src_digit_offset + static_cast<size_t>(digit_idx);
        const size_t out_poly_idx = out_row * out_cols + col;
        matrix_store_limb_u64(
            dst_base,
            out_poly_idx,
            coeff_idx,
            dst_stride,
            dst_bytes,
            signed_mod_i64(out_digit, out_modulus));
    }
}

int launch_gauss_samp_gq_arb_base_sample_kernel(
    const uint8_t *src_base,
    int64_t *sampled_digits,
    size_t poly_count,
    size_t n,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    uint64_t tower_modulus,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    double c,
    uint32_t tower_idx,
    GpuRngSeed seed,
    int device,
    cudaStream_t stream)
{
    if (!src_base || !sampled_digits)
    {
        return set_error("null base pointer in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }
    if (poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_coeff_bytes == 0 || src_stride_bytes < n * static_cast<size_t>(src_coeff_bytes))
    {
        return set_error("invalid stride in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits || base_bits == 0 || base_bits >= 63)
    {
        return set_error("invalid digits/base in matrix_gauss_samp_gq_arb_base_sample_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_gauss_samp_gq_arb_base_sample_kernel<<<blocks, threads, 0, stream>>>(
        src_base,
        sampled_digits,
        poly_count,
        n,
        src_stride_bytes,
        src_coeff_bytes,
        tower_modulus,
        base_bits,
        digits_per_tower,
        c,
        tower_idx,
        seed);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int launch_gauss_samp_gq_arb_base_scatter_kernel(
    const int64_t *sampled_digits,
    uint8_t *const *dst_bases,
    const size_t *dst_stride_bytes,
    const uint8_t *dst_coeff_bytes,
    const uint64_t *out_moduli,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    size_t src_digit_offset,
    uint32_t digits_per_tower,
    int device,
    cudaStream_t stream)
{
    if (!sampled_digits || !dst_bases || !dst_stride_bytes || !dst_coeff_bytes || !out_moduli)
    {
        return set_error("null pointer in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (out_limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return set_error("invalid matrix shape in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (digits_per_tower == 0 || digits_per_tower > kGaussMaxDigits)
    {
        return set_error("invalid digits in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }
    if (out_limb_count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
    {
        return set_error("too many out limbs in matrix_gauss_samp_gq_arb_base_scatter_kernel");
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const uint32_t blocks_x = static_cast<uint32_t>((total + threads - 1) / threads);
    const dim3 grid{blocks_x, static_cast<uint32_t>(out_limb_count), 1U};
    matrix_gauss_samp_gq_arb_base_scatter_kernel<<<grid, threads, 0, stream>>>(
        sampled_digits,
        dst_bases,
        dst_stride_bytes,
        dst_coeff_bytes,
        out_moduli,
        out_limb_count,
        poly_count,
        n,
        src_cols,
        out_cols,
        log_base_q,
        src_digit_offset,
        digits_per_tower);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


extern "C" int gpu_matrix_query_p1_workspaces(
    const GpuContext *ctx, size_t rows, size_t columns, int cached,
    GpuPreparedWorkspaceLayout *out)
{
    if (!ctx || !out || ctx->N <= 0 || (cached != 0 && cached != 1) || rows > SIZE_MAX / 2)
        return set_error("invalid P1 workspace query");
    const size_t m = 2 * rows;
    size_t samples = columns;
    if (samples > SIZE_MAX / static_cast<size_t>(ctx->N))
        return set_error("P1 sample count overflow");
    samples *= static_cast<size_t>(ctx->N);
    if (m > SIZE_MAX / sizeof(int64_t) ||
        (m != 0 && samples > SIZE_MAX / (m * sizeof(int64_t))))
        return set_error("P1 sampled output overflow");
    size_t workspace = 0;
    if (m > kSampleP1LocalMaxM && samples != 0) {
        if (m > SIZE_MAX / m) return set_error("P1 covariance dimension overflow");
        const size_t covariance = cached ? 0 : m * m;
        const size_t vectors = cached ? 2 : 3;
        if (m > (SIZE_MAX - covariance) / vectors)
            return set_error("P1 workspace dimension overflow");
        const size_t entries = covariance + vectors * m;
        if (entries > SIZE_MAX / sizeof(double) || samples > SIZE_MAX / (entries * sizeof(double)))
            return set_error("P1 workspace byte overflow");
        workspace = samples * entries * sizeof(double);
    }
    out[0] = {samples * m * sizeof(int64_t), alignof(int64_t), GPU_PREPARED_SAMPLER_WORKSPACE};
    out[1] = {workspace, alignof(double), GPU_PREPARED_SAMPLER_WORKSPACE};
    return 0;
}

int launch_sample_p1_integer_kernel(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    const uint8_t *tp2_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    size_t tp2_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    uint8_t tp2_coeff_bytes,
    size_t d,
    size_t cols,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    GpuRngSeed seed,
    cudaStream_t stream,
    int device_id,
    int64_t **sampled_out_device,
    cudaEvent_t sampled_ready_event,
    GpuContext *ctx, GpuDeviceWorkspace &sampled_owner)
{
    if (!sampled_out_device)
    {
        return set_error("null output pointer in matrix_sample_p1_integer_kernel");
    }
    *sampled_out_device = nullptr;
    if (!a_base || !b_base || !d_base || !tp2_base)
    {
        return set_error("null base pointer in matrix_sample_p1_integer_kernel");
    }
    if (a_coeff_bytes == 0 || b_coeff_bytes == 0 || d_coeff_bytes == 0 || tp2_coeff_bytes == 0 ||
        a_stride_bytes < n * static_cast<size_t>(a_coeff_bytes) ||
        b_stride_bytes < n * static_cast<size_t>(b_coeff_bytes) ||
        d_stride_bytes < n * static_cast<size_t>(d_coeff_bytes) ||
        tp2_stride_bytes < n * static_cast<size_t>(tp2_coeff_bytes))
    {
        return set_error("invalid stride in matrix_sample_p1_integer_kernel");
    }
    if (d == 0 || cols == 0 || n == 0)
    {
        return 0;
    }
    const size_t vec_entries = 2 * d * cols;

    if (device_id < 0)
    {
        return set_error("invalid device in matrix_sample_p1_integer_kernel");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_sample_p1_integer_kernel");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t m = d * 2;
    if (m == 0 || m > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("invalid dimension in matrix_sample_p1_integer_kernel");
    }
    const size_t total_samples = cols * n;
    const size_t total_values = vec_entries * n;
    if (total_samples == 0 || total_values == 0)
    {
        return 0;
    }

    GpuPreparedWorkspaceLayout layouts[2]{};
    const int layout_status = gpu_matrix_query_p1_workspaces(
        ctx, d, cols, 0, layouts);
    if (layout_status != 0) return layout_status;
    int64_t *d_sampled_out = nullptr;

    auto free_all = [&]()
    {
        if (d_sampled_out)
        {
            if (sampled_owner.release() != 0)
                gpu_device_mark_allocation_unknown(device_id);
            d_sampled_out = nullptr;
        }
    };

    const int allocation_status = sampled_owner.acquire(ctx, device_id, layouts[0].kind,
        layouts[0].bytes, layouts[0].alignment, stream);
    if (allocation_status != 0) return allocation_status;
    d_sampled_out = reinterpret_cast<int64_t *>(sampled_owner.data);

    const int threads = 256;
    if (m <= kSampleP1LocalMaxM)
    {
        const int blocks = static_cast<int>((total_samples + threads - 1) / threads);
        matrix_sample_p1_integer_kernel_small<<<blocks, threads, 0, stream>>>(
            a_base,
            b_base,
            d_base,
            tp2_base,
            a_stride_bytes,
            b_stride_bytes,
            d_stride_bytes,
            tp2_stride_bytes,
            a_coeff_bytes,
            b_coeff_bytes,
            d_coeff_bytes,
            tp2_coeff_bytes,
            d,
            cols,
            n,
            d_sampled_out,
            modulus,
            sigma,
            s,
            dgg_stddev,
            seed);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }
    else
    {
        const size_t cov_elems_per_sample = m * m;
        if (cov_elems_per_sample > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t cov_bytes_per_sample = cov_elems_per_sample * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(double))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t vec_bytes_per_sample = m * sizeof(double);
        if (m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
        if (cov_bytes_per_sample > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        size_t bytes_per_sample_total = cov_bytes_per_sample + vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - vec_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        bytes_per_sample_total += vec_bytes_per_sample;
        if (bytes_per_sample_total > std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_kernel");
        }
        bytes_per_sample_total += sampled_bytes_per_sample;

        void *workspace = nullptr;
        GpuDeviceWorkspace workspace_owner;
        double *cov_workspace = nullptr;
        double *mean_workspace = nullptr;
        double *col_workspace = nullptr;
        int64_t *sampled_workspace = nullptr;
        auto free_workspace = [&]()
        {
            if (workspace)
            {
                if (workspace_owner.release() != 0)
                    gpu_device_mark_allocation_unknown(device_id);
                workspace = nullptr;
            }
            cov_workspace = nullptr;
            mean_workspace = nullptr;
            col_workspace = nullptr;
            sampled_workspace = nullptr;
        };

        size_t chunk_samples = total_samples;
        auto alloc_workspace = [&](size_t samples) -> bool
        {
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
            if (samples > std::numeric_limits<size_t>::max() / bytes_per_sample_total)
            {
                return false;
            }
            const size_t workspace_bytes = samples * bytes_per_sample_total;
            if (workspace_bytes != layouts[1].bytes ||
                workspace_owner.acquire(ctx, device_id, layouts[1].kind,
                    layouts[1].bytes, layouts[1].alignment, stream) != 0) return false;
            workspace = workspace_owner.data;
            auto *workspace_base = reinterpret_cast<uint8_t *>(workspace);
            const size_t cov_bytes = samples * cov_bytes_per_sample;
            const size_t mean_bytes = samples * vec_bytes_per_sample;
            const size_t col_bytes = samples * vec_bytes_per_sample;
            cov_workspace = reinterpret_cast<double *>(workspace_base);
            mean_workspace = reinterpret_cast<double *>(workspace_base + cov_bytes);
            col_workspace = reinterpret_cast<double *>(workspace_base + cov_bytes + mean_bytes);
            sampled_workspace = reinterpret_cast<int64_t *>(
                workspace_base + cov_bytes + mean_bytes + col_bytes);
            return true;
        };

        // The submitted column range fixes the complete scratch demand. A
        // failed claim does not silently shrink this operation's execution plan.
        if (!alloc_workspace(chunk_samples)) {
            free_workspace();
            free_all();
            return set_error("P1 workspace does not fit its frozen column range");
        }

        for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
        {
            size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
            const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
            matrix_sample_p1_integer_kernel_large<<<blocks, threads, 0, stream>>>(
                a_base,
                b_base,
                d_base,
                tp2_base,
                a_stride_bytes,
                b_stride_bytes,
                d_stride_bytes,
                tp2_stride_bytes,
                a_coeff_bytes,
                b_coeff_bytes,
                d_coeff_bytes,
                tp2_coeff_bytes,
                d,
                cols,
                n,
                sample_start,
                sample_count,
                cov_workspace,
                mean_workspace,
                col_workspace,
                sampled_workspace,
                d_sampled_out,
                modulus,
                sigma,
                s,
                dgg_stddev,
                seed);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                free_workspace();
                free_all();
                return set_error(err);
            }
        }
        free_workspace();
    }

    if (sampled_ready_event)
    {
        err = cudaEventRecord(sampled_ready_event, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }

    *sampled_out_device = d_sampled_out;
    d_sampled_out = nullptr;
    free_all();
    return 0;
}

int launch_precompute_p1_covariance_kernel(
    const uint8_t *a_base,
    const uint8_t *b_base,
    const uint8_t *d_base,
    size_t a_stride_bytes,
    size_t b_stride_bytes,
    size_t d_stride_bytes,
    uint8_t a_coeff_bytes,
    uint8_t b_coeff_bytes,
    uint8_t d_coeff_bytes,
    size_t d,
    size_t n,
    uint64_t modulus,
    double sigma,
    double s,
    double dgg_stddev,
    int device_id,
    cudaStream_t stream,
    double *cov_workspace,
    double *sqrt_var_out,
    double *update_coeff_out)
{
    if (!a_base || !b_base || !d_base || !cov_workspace || !sqrt_var_out || !update_coeff_out)
    {
        return set_error("null pointer in matrix_precompute_p1_covariance_kernel");
    }
    if (a_coeff_bytes == 0 || b_coeff_bytes == 0 || d_coeff_bytes == 0 ||
        a_stride_bytes < n * static_cast<size_t>(a_coeff_bytes) ||
        b_stride_bytes < n * static_cast<size_t>(b_coeff_bytes) ||
        d_stride_bytes < n * static_cast<size_t>(d_coeff_bytes))
    {
        return set_error("invalid stride in matrix_precompute_p1_covariance_kernel");
    }
    if (d == 0 || n == 0)
    {
        return 0;
    }
    if (!(sigma > 0.0) || !(s > sigma) || !(dgg_stddev > 0.0))
    {
        return set_error("invalid Gaussian parameters in matrix_precompute_p1_covariance_kernel");
    }
    if (device_id < 0 || !stream)
    {
        return set_error("invalid device/stream in matrix_precompute_p1_covariance_kernel");
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    matrix_precompute_p1_covariance_kernel<<<blocks, threads, 0, stream>>>(
        a_base,
        b_base,
        d_base,
        a_stride_bytes,
        b_stride_bytes,
        d_stride_bytes,
        a_coeff_bytes,
        b_coeff_bytes,
        d_coeff_bytes,
        d,
        n,
        modulus,
        sigma,
        s,
        dgg_stddev,
        cov_workspace,
        sqrt_var_out,
        update_coeff_out);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}

int launch_sample_p1_integer_cached_kernel(
    const uint8_t *tp2_base,
    size_t tp2_stride_bytes,
    uint8_t tp2_coeff_bytes,
    const GpuP1CovarianceCache *cache,
    size_t cols,
    int64_t **sampled_out_device,
    GpuRngSeed seed,
    cudaStream_t stream,
    int device_id,
    cudaEvent_t sampled_ready_event,
    GpuContext *ctx, GpuDeviceWorkspace &sampled_owner)
{
    if (!sampled_out_device)
    {
        return set_error("null output pointer in matrix_sample_p1_integer_cached_kernel");
    }
    *sampled_out_device = nullptr;
    if (!tp2_base || !cache || !cache->sqrt_var || !cache->update_coeff)
    {
        return set_error("null pointer in matrix_sample_p1_integer_cached_kernel");
    }
    if (tp2_coeff_bytes == 0 ||
        tp2_stride_bytes < cache->n * static_cast<size_t>(tp2_coeff_bytes))
    {
        return set_error("invalid stride in matrix_sample_p1_integer_cached_kernel");
    }
    if (cache->d_rows == 0 || cols == 0 || cache->n == 0)
    {
        return 0;
    }
    if (device_id < 0 || !stream)
    {
        return set_error("invalid device/stream in matrix_sample_p1_integer_cached_kernel");
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const size_t m = cache->m;
    const size_t entry_count = m * cols;
    if (entry_count > std::numeric_limits<size_t>::max() / cache->n ||
        entry_count * cache->n > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        return set_error("sample byte overflow in matrix_sample_p1_integer_cached_kernel");
    }
    const size_t total_samples = cols * cache->n;
    const double denom = cache->s * cache->s - cache->sigma * cache->sigma;
    if (!(denom > 0.0))
    {
        return set_error("invalid cached Gaussian denominator");
    }
    const double c_scale = -(cache->sigma * cache->sigma) / denom;

    GpuPreparedWorkspaceLayout layouts[2]{};
    const int layout_status = gpu_matrix_query_p1_workspaces(
        ctx, cache->d_rows, cols, 1, layouts);
    if (layout_status != 0) return layout_status;
    int64_t *d_sampled_out = nullptr;
    auto free_all = [&]()
    {
        if (d_sampled_out)
        {
            if (sampled_owner.release() != 0)
                gpu_device_mark_allocation_unknown(device_id);
            d_sampled_out = nullptr;
        }
    };

    const int allocation_status = sampled_owner.acquire(ctx, device_id, layouts[0].kind,
        layouts[0].bytes, layouts[0].alignment, stream);
    if (allocation_status != 0) return allocation_status;
    d_sampled_out = reinterpret_cast<int64_t *>(sampled_owner.data);

    const int threads = 256;
    if (m <= kSampleP1LocalMaxM)
    {
        const int blocks = static_cast<int>((total_samples + threads - 1) / threads);
        matrix_sample_p1_integer_cached_kernel_small<<<blocks, threads, 0, stream>>>(
            tp2_base,
            tp2_stride_bytes,
            tp2_coeff_bytes,
            cache->sqrt_var,
            cache->update_coeff,
            cache->d_rows,
            cols,
            cache->n,
            d_sampled_out,
            cache->modulus,
            c_scale,
            seed);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }
    else
    {
        if (m > std::numeric_limits<size_t>::max() / sizeof(double) ||
            m > std::numeric_limits<size_t>::max() / sizeof(int64_t))
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_cached_kernel");
        }
        const size_t mean_bytes_per_sample = m * sizeof(double);
        const size_t sampled_bytes_per_sample = m * sizeof(int64_t);
        if (mean_bytes_per_sample >
            std::numeric_limits<size_t>::max() - sampled_bytes_per_sample)
        {
            free_all();
            return set_error("workspace overflow in matrix_sample_p1_integer_cached_kernel");
        }
        const size_t bytes_per_sample_total = mean_bytes_per_sample + sampled_bytes_per_sample;
        void *workspace = nullptr;
        GpuDeviceWorkspace workspace_owner;
        double *mean_workspace = nullptr;
        int64_t *sampled_workspace = nullptr;
        auto free_workspace = [&]()
        {
            if (workspace)
            {
                if (workspace_owner.release() != 0)
                    gpu_device_mark_allocation_unknown(device_id);
                workspace = nullptr;
            }
            mean_workspace = nullptr;
            sampled_workspace = nullptr;
        };

        size_t chunk_samples = total_samples;
        auto alloc_workspace = [&](size_t samples) -> bool
        {
            if (samples == 0 || samples > std::numeric_limits<size_t>::max() / bytes_per_sample_total)
            {
                return false;
            }
            const size_t workspace_bytes = samples * bytes_per_sample_total;
            if (workspace_bytes != layouts[1].bytes ||
                workspace_owner.acquire(ctx, device_id, layouts[1].kind,
                    layouts[1].bytes, layouts[1].alignment, stream) != 0) return false;
            workspace = workspace_owner.data;
            auto *workspace_base = reinterpret_cast<uint8_t *>(workspace);
            const size_t mean_bytes = samples * mean_bytes_per_sample;
            mean_workspace = reinterpret_cast<double *>(workspace_base);
            sampled_workspace = reinterpret_cast<int64_t *>(workspace_base + mean_bytes);
            return true;
        };

        // The submitted column range fixes the complete scratch demand. A
        // failed claim does not silently shrink this operation's execution plan.
        if (!alloc_workspace(chunk_samples)) {
            free_workspace();
            free_all();
            return set_error("P1 workspace does not fit its frozen column range");
        }

        for (size_t sample_start = 0; sample_start < total_samples; sample_start += chunk_samples)
        {
            size_t sample_count = std::min(chunk_samples, total_samples - sample_start);
            const int blocks = static_cast<int>((sample_count + threads - 1) / threads);
            matrix_sample_p1_integer_cached_kernel_large<<<blocks, threads, 0, stream>>>(
                tp2_base,
                tp2_stride_bytes,
                tp2_coeff_bytes,
                cache->sqrt_var,
                cache->update_coeff,
                cache->d_rows,
                cols,
                cache->n,
                sample_start,
                sample_count,
                mean_workspace,
                sampled_workspace,
                d_sampled_out,
                cache->modulus,
                c_scale,
                seed);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                free_workspace();
                free_all();
                return set_error(err);
            }
        }
        free_workspace();
    }

    if (sampled_ready_event)
    {
        err = cudaEventRecord(sampled_ready_event, stream);
        if (err != cudaSuccess)
        {
            free_all();
            return set_error(err);
        }
    }

    *sampled_out_device = d_sampled_out;
    d_sampled_out = nullptr;
    free_all();
    return 0;
}

int launch_scatter_p1_integer_to_limb_kernel_device(
    const int64_t *sampled_in_device,
    uint8_t *out_base,
    size_t out_stride_bytes,
    uint8_t out_coeff_bytes,
    size_t entry_count,
    size_t n,
    uint64_t modulus,
    cudaStream_t stream,
    int device_id)
{
    if (entry_count == 0 || n == 0)
    {
        return 0;
    }
    if (!sampled_in_device)
    {
        return set_error("null sampled device buffer in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (!out_base)
    {
        return set_error("null output base pointer in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (out_coeff_bytes == 0 || out_stride_bytes < n * static_cast<size_t>(out_coeff_bytes))
    {
        return set_error("invalid output stride in matrix_scatter_p1_integer_to_limb_kernel");
    }
    if (device_id < 0)
    {
        return set_error("invalid device in matrix_scatter_p1_integer_to_limb_kernel");
    }
    const size_t total = entry_count * n;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_scatter_p1_integer_to_limb_kernel<<<blocks, threads, 0, stream>>>(
        sampled_in_device,
        out_base,
        out_stride_bytes,
        out_coeff_bytes,
        entry_count,
        n,
        modulus);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


extern "C" int gpu_matrix_mul_vertical_pair(
    GpuMatrix *out,
    const GpuMatrix *top,
    const GpuMatrix *bottom,
    const GpuMatrix *rhs)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_mul_vertical_pair");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!out || !top || !bottom || !rhs)
    {
        return set_error("invalid gpu_matrix_mul_vertical_pair arguments");
    }
    if (top->ctx != bottom->ctx || top->ctx != rhs->ctx || top->ctx != out->ctx ||
        top->level != bottom->level || top->level != rhs->level || top->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_mul_vertical_pair");
    }
    if (top->format != GPU_POLY_FORMAT_EVAL || bottom->format != GPU_POLY_FORMAT_EVAL ||
        rhs->format != GPU_POLY_FORMAT_EVAL)
    {
        return set_error("gpu_matrix_mul_vertical_pair requires Eval inputs");
    }
    if (top->cols != bottom->cols || top->cols != rhs->rows ||
        out->rows != top->rows + bottom->rows || out->cols != rhs->cols)
    {
        return set_error("shape mismatch in gpu_matrix_mul_vertical_pair");
    }
    if (!top->ctx || top->level < 0)
    {
        return set_error("invalid context in gpu_matrix_mul_vertical_pair");
    }
    if (out->rows == 0 || out->cols == 0 || top->cols == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const size_t n = static_cast<size_t>(top->ctx->N);
    const size_t crt_depth = static_cast<size_t>(top->level + 1);
    if (top->ctx->limb_gpu_ids.size() < crt_depth || top->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected context size in gpu_matrix_mul_vertical_pair");
    }

    for (size_t limb = 0; limb < crt_depth; ++limb)
    {
        const dim3 limb_id = top->ctx->limb_gpu_ids[limb];
        const uint8_t *top_base = nullptr;
        const uint8_t *bottom_base = nullptr;
        const uint8_t *rhs_base = nullptr;
        uint8_t *out_base = nullptr;
        size_t top_stride = 0;
        size_t bottom_stride = 0;
        size_t rhs_stride = 0;
        size_t out_stride = 0;
        uint8_t top_coeff_bytes = 0;
        uint8_t bottom_coeff_bytes = 0;
        uint8_t rhs_coeff_bytes = 0;
        uint8_t out_coeff_bytes = 0;
        int top_device = -1;
        int bottom_device = -1;
        int rhs_device = -1;
        int out_device = -1;
        cudaStream_t stream = nullptr;
        int status = preimage_const_limb_view(
            top,
            limb_id,
            &top_base,
            &top_stride,
            &top_coeff_bytes,
            &top_device);
        if (status != 0)
        {
            return status;
        }
        status = preimage_const_limb_view(
            bottom,
            limb_id,
            &bottom_base,
            &bottom_stride,
            &bottom_coeff_bytes,
            &bottom_device);
        if (status != 0)
        {
            return status;
        }
        status = preimage_const_limb_view(
            rhs,
            limb_id,
            &rhs_base,
            &rhs_stride,
            &rhs_coeff_bytes,
            &rhs_device);
        if (status != 0)
        {
            return status;
        }
        status = preimage_mut_limb_view(
            out,
            limb_id,
            &out_base,
            &out_stride,
            &out_coeff_bytes,
            &out_device,
            &stream);
        if (status != 0)
        {
            return status;
        }
        if (top_device != out_device || bottom_device != out_device || rhs_device != out_device)
        {
            return set_error("device mismatch in gpu_matrix_mul_vertical_pair");
        }
        if (top_coeff_bytes != bottom_coeff_bytes || top_coeff_bytes != rhs_coeff_bytes ||
            top_coeff_bytes != out_coeff_bytes)
        {
            return set_error("coefficient width mismatch in gpu_matrix_mul_vertical_pair");
        }
        status = matrix_wait_limb_stream(top, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(bottom, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }

        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        const dim3 threads(kPreimageTileN, kPreimageTileM);
        const dim3 blocks(
            static_cast<unsigned int>((out->cols + kPreimageTileN - 1) / kPreimageTileN),
            static_cast<unsigned int>((out->rows + kPreimageTileM - 1) / kPreimageTileM),
            static_cast<unsigned int>(std::min<size_t>(n, 65535)));
        matrix_mul_vertical_pair_kernel<<<blocks, threads, 0, stream>>>(
            top_base,
            bottom_base,
            rhs_base,
            out_base,
            top->rows,
            bottom->rows,
            top->cols,
            out->cols,
            n,
            top_stride,
            bottom_stride,
            rhs_stride,
            out_stride,
            top_coeff_bytes,
            bottom_coeff_bytes,
            rhs_coeff_bytes,
            out_coeff_bytes,
            top->ctx->moduli[limb]);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = matrix_track_limb_consumer(top, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(bottom, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_track_limb_consumer(rhs, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, stream);
        if (status != 0)
        {
            return status;
        }
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_preimage_residual(
    GpuMatrix *out,
    const GpuMatrix *target,
    const GpuMatrix *public_matrix,
    const GpuMatrix *p1,
    const GpuMatrix *p2)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_preimage_residual");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!out || !target || !public_matrix || !p1 || !p2)
    {
        return set_error("invalid gpu_matrix_preimage_residual arguments");
    }
    if (target->ctx != public_matrix->ctx || target->ctx != p1->ctx || target->ctx != p2->ctx ||
        target->ctx != out->ctx || target->level != public_matrix->level ||
        target->level != p1->level || target->level != p2->level || target->level != out->level)
    {
        return set_error("context mismatch in gpu_matrix_preimage_residual");
    }
    if (target->format != GPU_POLY_FORMAT_EVAL || public_matrix->format != GPU_POLY_FORMAT_EVAL ||
        p1->format != GPU_POLY_FORMAT_EVAL || p2->format != GPU_POLY_FORMAT_EVAL)
    {
        return set_error("gpu_matrix_preimage_residual requires Eval inputs");
    }
    if (target->rows != public_matrix->rows || target->rows != out->rows ||
        target->cols > p1->cols || target->cols > p2->cols || target->cols != out->cols ||
        public_matrix->cols != p1->rows + p2->rows)
    {
        return set_error("shape mismatch in gpu_matrix_preimage_residual");
    }
    if (!target->ctx || target->level < 0)
    {
        return set_error("invalid context in gpu_matrix_preimage_residual");
    }
    if (out->rows == 0 || out->cols == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const size_t n = static_cast<size_t>(target->ctx->N);
    const size_t crt_depth = static_cast<size_t>(target->level + 1);
    if (target->ctx->limb_gpu_ids.size() < crt_depth || target->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected context size in gpu_matrix_preimage_residual");
    }

    for (size_t limb = 0; limb < crt_depth; ++limb)
    {
        const dim3 limb_id = target->ctx->limb_gpu_ids[limb];
        const GpuMatrix *inputs[] = {target, public_matrix, p1, p2};
        const uint8_t *bases[4] = {};
        size_t strides[4] = {};
        uint8_t coeff_bytes[4] = {};
        int devices[4] = {};
        for (size_t input_idx = 0; input_idx < 4; ++input_idx)
        {
            int status = preimage_const_limb_view(
                inputs[input_idx],
                limb_id,
                &bases[input_idx],
                &strides[input_idx],
                &coeff_bytes[input_idx],
                &devices[input_idx]);
            if (status != 0)
            {
                return status;
            }
        }
        uint8_t *out_base = nullptr;
        size_t out_stride = 0;
        uint8_t out_coeff_bytes = 0;
        int out_device = -1;
        cudaStream_t stream = nullptr;
        int status = preimage_mut_limb_view(
            out,
            limb_id,
            &out_base,
            &out_stride,
            &out_coeff_bytes,
            &out_device,
            &stream);
        if (status != 0)
        {
            return status;
        }
        for (size_t input_idx = 0; input_idx < 4; ++input_idx)
        {
            if (devices[input_idx] != out_device)
            {
                return set_error("device mismatch in gpu_matrix_preimage_residual");
            }
            if (coeff_bytes[input_idx] != out_coeff_bytes)
            {
                return set_error("coefficient width mismatch in gpu_matrix_preimage_residual");
            }
            status = matrix_wait_limb_stream(inputs[input_idx], limb_id, out_device, stream);
            if (status != 0)
            {
                return status;
            }
        }

        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        const dim3 threads(kPreimageTileN, kPreimageTileM);
        const dim3 blocks(
            static_cast<unsigned int>((out->cols + kPreimageTileN - 1) / kPreimageTileN),
            static_cast<unsigned int>((out->rows + kPreimageTileM - 1) / kPreimageTileM),
            static_cast<unsigned int>(std::min<size_t>(n, 65535)));
        matrix_preimage_residual_kernel<<<blocks, threads, 0, stream>>>(
            bases[0],
            bases[1],
            bases[2],
            bases[3],
            out_base,
            out->rows,
            p1->rows,
            p2->rows,
            p1->cols,
            p2->cols,
            out->cols,
            n,
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            out_stride,
            coeff_bytes[0],
            coeff_bytes[1],
            coeff_bytes[2],
            coeff_bytes[3],
            out_coeff_bytes,
            target->ctx->moduli[limb]);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (const GpuMatrix *input : inputs)
        {
            status = matrix_track_limb_consumer(input, limb_id, out_device, stream);
            if (status != 0)
            {
                return status;
            }
        }
        status = matrix_record_limb_write(out, limb_id, stream);
        if (status != 0)
        {
            return status;
        }
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_preimage_add_correction(
    GpuMatrix *out,
    const GpuMatrix *r,
    const GpuMatrix *e,
    const GpuMatrix *z)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_preimage_add_correction");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (!out || !r || !e || !z)
    {
        return set_error("invalid gpu_matrix_preimage_add_correction arguments");
    }
    if (out->ctx != r->ctx || out->ctx != e->ctx || out->ctx != z->ctx ||
        out->level != r->level || out->level != e->level || out->level != z->level)
    {
        return set_error("context mismatch in gpu_matrix_preimage_add_correction");
    }
    if (out->format != GPU_POLY_FORMAT_EVAL || r->format != GPU_POLY_FORMAT_EVAL ||
        e->format != GPU_POLY_FORMAT_EVAL ||
        z->format != GPU_POLY_FORMAT_EVAL)
    {
        return set_error("gpu_matrix_preimage_add_correction requires Eval inputs");
    }
    if (r->rows != e->rows || r->cols != e->cols || r->cols != z->rows ||
        out->rows != r->rows + e->rows + z->rows || out->cols != z->cols)
    {
        return set_error("shape mismatch in gpu_matrix_preimage_add_correction");
    }
    if (!out->ctx || out->level < 0)
    {
        return set_error("invalid context in gpu_matrix_preimage_add_correction");
    }
    if (out->rows == 0 || out->cols == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const size_t n = static_cast<size_t>(out->ctx->N);
    const size_t crt_depth = static_cast<size_t>(out->level + 1);
    if (out->ctx->limb_gpu_ids.size() < crt_depth || out->ctx->moduli.size() < crt_depth)
    {
        return set_error("unexpected context size in gpu_matrix_preimage_add_correction");
    }

    for (size_t limb = 0; limb < crt_depth; ++limb)
    {
        const dim3 limb_id = out->ctx->limb_gpu_ids[limb];
        const GpuMatrix *inputs[] = {r, e, z};
        const uint8_t *bases[3] = {};
        size_t strides[3] = {};
        uint8_t coeff_bytes[3] = {};
        int devices[3] = {};
        for (size_t input_idx = 0; input_idx < 3; ++input_idx)
        {
            int status = preimage_const_limb_view(
                inputs[input_idx],
                limb_id,
                &bases[input_idx],
                &strides[input_idx],
                &coeff_bytes[input_idx],
                &devices[input_idx]);
            if (status != 0)
            {
                return status;
            }
        }
        uint8_t *out_base = nullptr;
        size_t out_stride = 0;
        uint8_t out_coeff_bytes = 0;
        int out_device = -1;
        cudaStream_t stream = nullptr;
        int status = preimage_mut_limb_view(
            out,
            limb_id,
            &out_base,
            &out_stride,
            &out_coeff_bytes,
            &out_device,
            &stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(out, limb_id, out_device, stream);
        if (status != 0)
        {
            return status;
        }
        for (size_t input_idx = 0; input_idx < 3; ++input_idx)
        {
            if (devices[input_idx] != out_device)
            {
                return set_error("device mismatch in gpu_matrix_preimage_add_correction");
            }
            if (coeff_bytes[input_idx] != out_coeff_bytes)
            {
                return set_error("coefficient width mismatch in gpu_matrix_preimage_add_correction");
            }
            status = matrix_wait_limb_stream(inputs[input_idx], limb_id, out_device, stream);
            if (status != 0)
            {
                return status;
            }
        }

        cudaError_t err = cudaSetDevice(out_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        const dim3 top_threads(kPreimageTileN, kPreimageTileM);
        const dim3 top_blocks(
            static_cast<unsigned int>((out->cols + kPreimageTileN - 1) / kPreimageTileN),
            static_cast<unsigned int>(
                (r->rows + e->rows + kPreimageTileM - 1) / kPreimageTileM),
            static_cast<unsigned int>(std::min<size_t>(n, 65535)));
        matrix_preimage_add_correction_top_kernel<<<top_blocks, top_threads, 0, stream>>>(
            bases[0],
            bases[1],
            bases[2],
            out_base,
            r->rows,
            z->rows,
            z->cols,
            out->cols,
            n,
            strides[0],
            strides[1],
            strides[2],
            out_stride,
            coeff_bytes[0],
            coeff_bytes[1],
            coeff_bytes[2],
            out_coeff_bytes,
            out->ctx->moduli[limb]);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const size_t bottom_entry_count = z->rows * out->cols * n;
        const int bottom_threads = 256;
        const unsigned int bottom_blocks =
            static_cast<unsigned int>((bottom_entry_count + bottom_threads - 1) / bottom_threads);
        matrix_preimage_add_correction_bottom_kernel<<<bottom_blocks, bottom_threads, 0, stream>>>(
            bases[2],
            out_base,
            r->rows + e->rows,
            z->rows,
            z->cols,
            out->cols,
            n,
            strides[2],
            out_stride,
            coeff_bytes[2],
            out_coeff_bytes,
            out->ctx->moduli[limb]);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (const GpuMatrix *input : inputs)
        {
            status = matrix_track_limb_consumer(input, limb_id, out_device, stream);
            if (status != 0)
            {
                return status;
            }
        }
        status = matrix_record_limb_write(out, limb_id, stream);
        if (status != 0)
        {
            return status;
        }
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_query_gaussian_gadget_workspaces(
    const GpuContext *ctx, int level, size_t rows, size_t cols,
    uint32_t base_bits, GpuPreparedWorkspaceLayout *out)
{
    if (!ctx || !out || level < 0 || static_cast<size_t>(level) >= ctx->moduli.size() ||
        base_bits == 0 || base_bits >= 63 || ctx->N <= 0)
        return set_error("invalid Gaussian gadget workspace query");
    uint32_t crt_bits = 0;
    for (const auto modulus : ctx->moduli)
        crt_bits = std::max(crt_bits, bit_width_u64(modulus));
    const size_t digits = (crt_bits + base_bits - 1) / base_bits;
    if (digits == 0 || digits > kGaussMaxDigits)
        return set_error("unsupported Gaussian gadget digit count");
    size_t sampled = rows == 0 || cols == 0 ? 0 : 1;
    for (const size_t factor : {rows, cols, static_cast<size_t>(ctx->N), digits, sizeof(int64_t)}) {
        if (factor != 0 && sampled > SIZE_MAX / factor)
            return set_error("Gaussian gadget workspace size overflow");
        sampled *= factor;
    }
    const size_t limbs = sampled == 0 ? 0 : static_cast<size_t>(level) + 1;
    out[0] = {sampled, alignof(int64_t), GPU_PREPARED_SAMPLER_WORKSPACE};
    out[1] = {limbs * sizeof(uint8_t *), alignof(uint8_t *), GPU_PREPARED_SAMPLER_WORKSPACE};
    out[2] = {limbs * sizeof(size_t), alignof(size_t), GPU_PREPARED_SAMPLER_WORKSPACE};
    out[3] = {limbs * sizeof(uint8_t), alignof(uint8_t), GPU_PREPARED_SAMPLER_WORKSPACE};
    out[4] = {limbs * sizeof(uint64_t), alignof(uint64_t), GPU_PREPARED_SAMPLER_WORKSPACE};
    return 0;
}

extern "C" int gpu_matrix_gauss_samp_gq_arb_base(
    GpuMatrix *src,
    uint32_t base_bits,
    double c,
    double dgg_stddev,
    GpuRngSeed seed,
    GpuMatrix *out)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_gauss_samp_gq_arb_base");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

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
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_gauss_samp_gq_arb_base");
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
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    const GpuMatrix *inputs_matrix = src;
    auto cleanup_tmp_inputs = [&]() {};

    int status = 0;
    if (src->format == GPU_POLY_FORMAT_EVAL)
    {
        status = gpu_matrix_intt_all(src);
        if (status != 0)
        {
            return status;
        }
    }

    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_gauss_samp_gq_arb_base");
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_gauss_samp_gq_arb_base");
    }

    std::vector<dim3> active_limb_ids(crt_depth);
    std::vector<uint8_t *> out_limb_bases(crt_depth, nullptr);
    std::vector<size_t> out_limb_strides(crt_depth, 0);
    std::vector<uint8_t> out_limb_coeff_bytes(crt_depth, 0);
    std::vector<uint64_t> out_limb_moduli(crt_depth, 0);
    std::vector<const uint8_t *> src_limb_bases(crt_depth, nullptr);
    std::vector<size_t> src_limb_strides(crt_depth, 0);
    std::vector<uint8_t> src_limb_coeff_bytes(crt_depth, 0);

    int dispatch_device = -1;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t limb_idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[limb_idx];
        active_limb_ids[limb_idx] = limb_id;
        out_limb_moduli[limb_idx] = src->ctx->moduli[limb_idx];

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
        }
        else if (out_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-GPU path requires all out limbs on one device");
        }
        uint8_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb base pointer in gpu_matrix_gauss_samp_gq_arb_base");
        }
        size_t dst_stride = 0;
        uint8_t dst_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(out, limb_id, &dst_stride, &dst_coeff_bytes))
        {
            cleanup_tmp_inputs();
            return set_error("invalid output limb metadata in gpu_matrix_gauss_samp_gq_arb_base");
        }
        if (dst_stride < static_cast<size_t>(src->ctx->N) * static_cast<size_t>(dst_coeff_bytes))
        {
            cleanup_tmp_inputs();
            return set_error("invalid output stride in gpu_matrix_gauss_samp_gq_arb_base");
        }
        out_limb_bases[limb_idx] = dst_base;
        out_limb_strides[limb_idx] = dst_stride;
        out_limb_coeff_bytes[limb_idx] = dst_coeff_bytes;
    }
    if (dispatch_device < 0)
    {
        cleanup_tmp_inputs();
        return set_error("invalid output device in gpu_matrix_gauss_samp_gq_arb_base");
    }

    cudaStream_t dispatch_stream = nullptr;
    status = matrix_limb_stream(out, active_limb_ids[0], &dispatch_stream);
    if (status != 0)
    {
        cleanup_tmp_inputs();
        return status;
    }
    if (!dispatch_stream)
    {
        cleanup_tmp_inputs();
        return set_error("null dispatch stream in gpu_matrix_gauss_samp_gq_arb_base");
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
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
            return set_error("single-GPU path requires all src limbs on one device");
        }
        const uint8_t *src_base = matrix_limb_ptr_by_id(inputs_matrix, 0, src_limb_id);
        if (!src_base)
        {
            cleanup_tmp_inputs();
            return set_error("null source limb base pointer in gpu_matrix_gauss_samp_gq_arb_base");
        }
        size_t src_stride = 0;
        uint8_t src_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(inputs_matrix, src_limb_id, &src_stride, &src_coeff_bytes))
        {
            cleanup_tmp_inputs();
            return set_error("invalid source limb metadata in gpu_matrix_gauss_samp_gq_arb_base");
        }
        if (src_stride < static_cast<size_t>(src->ctx->N) * static_cast<size_t>(src_coeff_bytes))
        {
            cleanup_tmp_inputs();
            return set_error("invalid source stride in gpu_matrix_gauss_samp_gq_arb_base");
        }
        src_limb_bases[src_idx] = src_base;
        src_limb_strides[src_idx] = src_stride;
        src_limb_coeff_bytes[src_idx] = src_coeff_bytes;
    }

    GpuPreparedWorkspaceLayout sampler_layouts[5]{};
    status = gpu_matrix_query_gaussian_gadget_workspaces(
        src->ctx, level, rows, cols, base_bits, sampler_layouts);
    if (status != 0) { cleanup_tmp_inputs(); return status; }
    const size_t out_ptr_bytes = sampler_layouts[1].bytes;
    const size_t out_stride_bytes = sampler_layouts[2].bytes;
    const size_t out_coeff_bytes = sampler_layouts[3].bytes;
    const size_t out_moduli_bytes = sampler_layouts[4].bytes;

    int64_t *sampled_digits_device = nullptr;
    uint8_t **out_limb_bases_device = nullptr;
    size_t *out_limb_strides_device = nullptr;
    uint8_t *out_limb_coeff_bytes_device = nullptr;
    uint64_t *out_limb_moduli_device = nullptr;
    std::array<GpuDeviceWorkspace, 5> sampler_owners;
    auto cleanup = [&]()
    {
        if (dispatch_device >= 0)
        {
            if (cudaSetDevice(dispatch_device) != cudaSuccess)
            {
                out->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
                out->ctx->execution->unretired_work.store(true, std::memory_order_release);
                return;
            }
        }
        if (sampled_digits_device)
        {
            if (sampler_owners[0].release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            sampled_digits_device = nullptr;
        }
        if (out_limb_bases_device)
        {
            if (sampler_owners[1].release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            out_limb_bases_device = nullptr;
        }
        if (out_limb_strides_device)
        {
            if (sampler_owners[2].release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            out_limb_strides_device = nullptr;
        }
        if (out_limb_coeff_bytes_device)
        {
            if (sampler_owners[3].release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            out_limb_coeff_bytes_device = nullptr;
        }
        if (out_limb_moduli_device)
        {
            if (sampler_owners[4].release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            out_limb_moduli_device = nullptr;
        }
        cleanup_tmp_inputs();
    };

    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    const size_t out_count = out->rows * out->cols;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 out_limb_id = active_limb_ids[static_cast<size_t>(limb)];
        status = matrix_wait_limb_stream(out, out_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        if (out_count > 0)
        {
            const size_t dst_pitch = out_limb_strides[static_cast<size_t>(limb)];
            const size_t zero_width =
                static_cast<size_t>(src->ctx->N) *
                static_cast<size_t>(out_limb_coeff_bytes[static_cast<size_t>(limb)]);
            err = cudaMemset2DAsync(
                out_limb_bases[static_cast<size_t>(limb)],
                dst_pitch,
                0,
                zero_width,
                out_count,
                dispatch_stream);
            if (err != cudaSuccess)
            {
                cleanup();
                return set_error(err);
            }
        }
    }

    status = sampler_owners[0].acquire(out->ctx, dispatch_device,
        sampler_layouts[0].kind, sampler_layouts[0].bytes, sampler_layouts[0].alignment, dispatch_stream);
    if (status != 0) { cleanup(); return status; }
    sampled_digits_device = reinterpret_cast<int64_t *>(sampler_owners[0].data);
    status = sampler_owners[1].acquire(out->ctx, dispatch_device,
        sampler_layouts[1].kind, sampler_layouts[1].bytes, sampler_layouts[1].alignment, dispatch_stream);
    if (status != 0) { cleanup(); return status; }
    out_limb_bases_device = reinterpret_cast<uint8_t * *>(sampler_owners[1].data);
    status = sampler_owners[2].acquire(out->ctx, dispatch_device,
        sampler_layouts[2].kind, sampler_layouts[2].bytes, sampler_layouts[2].alignment, dispatch_stream);
    if (status != 0) { cleanup(); return status; }
    out_limb_strides_device = reinterpret_cast<size_t *>(sampler_owners[2].data);
    status = sampler_owners[3].acquire(out->ctx, dispatch_device,
        sampler_layouts[3].kind, sampler_layouts[3].bytes, sampler_layouts[3].alignment, dispatch_stream);
    if (status != 0) { cleanup(); return status; }
    out_limb_coeff_bytes_device = reinterpret_cast<uint8_t *>(sampler_owners[3].data);
    status = sampler_owners[4].acquire(out->ctx, dispatch_device,
        sampler_layouts[4].kind, sampler_layouts[4].bytes, sampler_layouts[4].alignment, dispatch_stream);
    if (status != 0) { cleanup(); return status; }
    out_limb_moduli_device = reinterpret_cast<uint64_t *>(sampler_owners[4].data);

    err = cudaMemcpyAsync(
        out_limb_bases_device,
        out_limb_bases.data(),
        out_ptr_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_strides_device,
        out_limb_strides.data(),
        out_stride_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_coeff_bytes_device,
        out_limb_coeff_bytes.data(),
        out_coeff_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_moduli_device,
        out_limb_moduli.data(),
        out_moduli_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

    for (int src_limb = 0; src_limb <= level; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
        status = matrix_wait_limb_stream(inputs_matrix, src_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_gauss_samp_gq_arb_base_sample_kernel(
            src_limb_bases[src_idx],
            sampled_digits_device,
            count,
            static_cast<size_t>(src->ctx->N),
            src_limb_strides[src_idx],
            src_limb_coeff_bytes[src_idx],
            src->ctx->moduli[src_idx],
            base_bits,
            digits_per_tower,
            c,
            static_cast<uint32_t>(src_limb),
            seed,
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        const size_t src_digit_offset =
            static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower);
        status = launch_gauss_samp_gq_arb_base_scatter_kernel(
            sampled_digits_device,
            out_limb_bases_device,
            out_limb_strides_device,
            out_limb_coeff_bytes_device,
            out_limb_moduli_device,
            crt_depth,
            count,
            static_cast<size_t>(src->ctx->N),
            cols,
            out->cols,
            log_base_q,
            src_digit_offset,
            digits_per_tower,
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = matrix_track_limb_consumer(
            inputs_matrix,
            src_limb_id,
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

    for (int out_limb = 0; out_limb <= level; ++out_limb)
    {
        status = matrix_record_limb_write(out, active_limb_ids[static_cast<size_t>(out_limb)], dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
    }

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

extern "C" int gpu_matrix_create_p1_covariance_cache(
    const GpuMatrix *a_mat,
    const GpuMatrix *b_mat,
    const GpuMatrix *d_mat,
    double sigma,
    double s,
    double dgg_stddev,
    GpuP1CovarianceCache **out_cache)
{
    if (!a_mat || !a_mat->ctx || !a_mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_create_p1_covariance_cache");
    GpuAllocationActivity activity(a_mat->ctx->execution.get(), -1);

    if (!out_cache)
    {
        return set_error("null output in gpu_matrix_create_p1_covariance_cache");
    }
    *out_cache = nullptr;
    if (!a_mat || !b_mat || !d_mat)
    {
        return set_error("invalid gpu_matrix_create_p1_covariance_cache arguments");
    }
    if (!(sigma > 0.0) || !(s > sigma) || !(dgg_stddev > 0.0))
    {
        return set_error("invalid Gaussian parameters in gpu_matrix_create_p1_covariance_cache");
    }
    if (a_mat->ctx != b_mat->ctx || a_mat->ctx != d_mat->ctx)
    {
        return set_error("context mismatch in gpu_matrix_create_p1_covariance_cache");
    }
    if (a_mat->level != b_mat->level || a_mat->level != d_mat->level)
    {
        return set_error("level mismatch in gpu_matrix_create_p1_covariance_cache");
    }
    if (a_mat->format != GPU_POLY_FORMAT_COEFF ||
        b_mat->format != GPU_POLY_FORMAT_COEFF ||
        d_mat->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("p1 covariance cache inputs must be coefficient matrices");
    }
    const size_t d_rows = a_mat->rows;
    if (a_mat->cols != d_rows || b_mat->rows != d_rows || b_mat->cols != d_rows ||
        d_mat->rows != d_rows || d_mat->cols != d_rows)
    {
        return set_error("A/B/D must be dxd in gpu_matrix_create_p1_covariance_cache");
    }
    if (d_rows == 0)
    {
        auto *empty_cache = new GpuP1CovarianceCache();
        empty_cache->ctx = a_mat->ctx;
        empty_cache->execution = a_mat->ctx->execution;
        empty_cache->level = a_mat->level;
        *out_cache = empty_cache;
        return 0;
    }
    const int level = a_mat->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_create_p1_covariance_cache");
    }
    if (a_mat->ctx->moduli.empty() || a_mat->ctx->limb_gpu_ids.empty())
    {
        return set_error("empty context in gpu_matrix_create_p1_covariance_cache");
    }

    const size_t n = static_cast<size_t>(a_mat->ctx->N);
    const size_t m = 2 * d_rows;
    if (m == 0 || m > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("invalid dimension in gpu_matrix_create_p1_covariance_cache");
    }
    const size_t factor_elems = n * m;
    if (n != 0 && factor_elems / n != m)
    {
        return set_error("factor size overflow in gpu_matrix_create_p1_covariance_cache");
    }
    if (factor_elems > std::numeric_limits<size_t>::max() / m)
    {
        return set_error("factor size overflow in gpu_matrix_create_p1_covariance_cache");
    }
    const size_t update_elems = factor_elems * m;
    if (factor_elems > std::numeric_limits<size_t>::max() / sizeof(double) ||
        update_elems > std::numeric_limits<size_t>::max() / sizeof(double))
    {
        return set_error("factor byte overflow in gpu_matrix_create_p1_covariance_cache");
    }
    const size_t sqrt_bytes = factor_elems * sizeof(double);
    const size_t update_bytes = update_elems * sizeof(double);
    const size_t cov_bytes = update_bytes;

    const dim3 ref_limb_id = a_mat->ctx->limb_gpu_ids[0];
    int ref_device = -1;
    int status = matrix_limb_device(a_mat, ref_limb_id, &ref_device);
    if (status != 0)
    {
        return status;
    }
    int b_device = -1;
    status = matrix_limb_device(b_mat, ref_limb_id, &b_device);
    if (status != 0)
    {
        return status;
    }
    int d_device = -1;
    status = matrix_limb_device(d_mat, ref_limb_id, &d_device);
    if (status != 0)
    {
        return status;
    }
    if (ref_device < 0 || b_device != ref_device || d_device != ref_device)
    {
        return set_error("reference limb device mismatch in gpu_matrix_create_p1_covariance_cache");
    }
    const uint8_t *a_base = matrix_limb_ptr_by_id(a_mat, 0, ref_limb_id);
    const uint8_t *b_base = matrix_limb_ptr_by_id(b_mat, 0, ref_limb_id);
    const uint8_t *d_base = matrix_limb_ptr_by_id(d_mat, 0, ref_limb_id);
    if (!a_base || !b_base || !d_base)
    {
        return set_error("null reference limb base pointer in gpu_matrix_create_p1_covariance_cache");
    }
    size_t a_stride = 0;
    size_t b_stride = 0;
    size_t d_stride = 0;
    uint8_t a_coeff_bytes = 0;
    uint8_t b_coeff_bytes = 0;
    uint8_t d_coeff_bytes = 0;
    if (!matrix_limb_metadata_by_id(a_mat, ref_limb_id, &a_stride, &a_coeff_bytes) ||
        !matrix_limb_metadata_by_id(b_mat, ref_limb_id, &b_stride, &b_coeff_bytes) ||
        !matrix_limb_metadata_by_id(d_mat, ref_limb_id, &d_stride, &d_coeff_bytes))
    {
        return set_error("invalid reference limb metadata in gpu_matrix_create_p1_covariance_cache");
    }

    cudaError_t err = cudaSetDevice(ref_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    auto *cache = new GpuP1CovarianceCache();
    cache->ctx = a_mat->ctx;
    cache->execution = a_mat->ctx->execution;
    cache->level = a_mat->level;
    cache->d_rows = d_rows;
    cache->n = n;
    cache->m = m;
    cache->modulus = a_mat->ctx->moduli[0];
    cache->sigma = sigma;
    cache->s = s;
    cache->device = ref_device;
    {
        // One submission-stream slot provides the private stream and its
        // bridge event, which doubles as the cache's ready event.
        const int acquired = cache->stream_resource.acquire(
            a_mat->ctx, ref_device, GPU_PREPARED_SUBMISSION_STREAM);
        if (acquired != 0)
        {
            delete cache;
            return acquired;
        }
        cache->stream = cache->stream_resource.stream;
        cache->ready_event = cache->stream_resource.event;
    }

    cudaStream_t producer = nullptr;
    status = matrix_limb_stream(a_mat, ref_limb_id, &producer);
    cudaEvent_t started = nullptr;
    GpuCudaResource started_resource;
    if (status == 0) status = matrix_get_thread_local_owner_link_event(a_mat->ctx, ref_device, &started, started_resource);
    if (status == 0)
    {
        // A prepared input's writer event may predate a measurement start.
        // Join the current producer boundary before using this private stream.
        err = cudaEventRecord(started, producer);
        if (err == cudaSuccess) err = cudaStreamWaitEvent(cache->stream, started, 0);
        if (err != cudaSuccess) status = set_error(err);
    }
    if (status != 0)
    {
        gpu_matrix_destroy_p1_covariance_cache(cache);
        return status;
    }

    status = matrix_wait_limb_stream(a_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        gpu_matrix_destroy_p1_covariance_cache(cache);
        return status;
    }
    status = matrix_wait_limb_stream(b_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        gpu_matrix_destroy_p1_covariance_cache(cache);
        return status;
    }
    status = matrix_wait_limb_stream(d_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        gpu_matrix_destroy_p1_covariance_cache(cache);
        return status;
    }

    double *cov_workspace = nullptr;
    GpuDeviceWorkspace cov_owner;
    auto cleanup = [&]()
    {
        if (cudaSetDevice(ref_device) != cudaSuccess)
        {
            cache->execution->memory_release_failed.store(true, std::memory_order_release);
            cache->execution->unretired_work.store(true, std::memory_order_release);
            return;
        }
        if (cov_workspace)
        {
            if (cov_owner.release(cache->stream) != 0)
            {
                cache->execution->memory_release_failed.store(true, std::memory_order_release);
                cache->execution->unretired_work.store(true, std::memory_order_release);
            }
            cov_workspace = nullptr;
        }
        if (cache)
        {
            // A failed construction may have submitted input readers before
            // publishing their ordinary per-matrix completion dependencies.
            gpu_context_retire_stream(cache->ctx, cache->device, cache->stream);
            gpu_matrix_destroy_p1_covariance_cache(cache);
            cache = nullptr;
        }
    };

    status = cache->sqrt_owner.acquire(a_mat->ctx, ref_device, GPU_PREPARED_SAMPLER_WORKSPACE,
        sqrt_bytes, alignof(double), cache->stream);
    if (status != 0) { cleanup(); return status; }
    cache->sqrt_var = reinterpret_cast<double *>(cache->sqrt_owner.data);
    status = cache->update_owner.acquire(a_mat->ctx, ref_device, GPU_PREPARED_SAMPLER_WORKSPACE,
        update_bytes, alignof(double), cache->stream);
    if (status != 0) { cleanup(); return status; }
    cache->update_coeff = reinterpret_cast<double *>(cache->update_owner.data);
    status = cov_owner.acquire(a_mat->ctx, ref_device, GPU_PREPARED_SAMPLER_WORKSPACE,
        cov_bytes, alignof(double), cache->stream);
    if (status != 0) { cleanup(); return status; }
    cov_workspace = reinterpret_cast<double *>(cov_owner.data);
    status = launch_precompute_p1_covariance_kernel(
        a_base,
        b_base,
        d_base,
        a_stride,
        b_stride,
        d_stride,
        a_coeff_bytes,
        b_coeff_bytes,
        d_coeff_bytes,
        d_rows,
        n,
        cache->modulus,
        sigma,
        s,
        dgg_stddev,
        ref_device,
        cache->stream,
        cov_workspace,
        cache->sqrt_var,
        cache->update_coeff);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    err = cudaEventRecord(cache->ready_event, cache->stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    status = cov_owner.release(cache->stream);
    cov_workspace = nullptr;
    if (status != 0)
    {
        cache->execution->memory_release_failed.store(true, std::memory_order_release);
        cache->execution->unretired_work.store(true, std::memory_order_release);
        cleanup();
        return status;
    }

    status = matrix_track_limb_consumer(a_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(b_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer(d_mat, ref_limb_id, ref_device, cache->stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    *out_cache = cache;
    cache = nullptr;
    return 0;
}

extern "C" void gpu_matrix_destroy_p1_covariance_cache(GpuP1CovarianceCache *cache)
{
    if (!cache) return;
    auto owner = cache->execution;
    if (owner->unretired_work.load(std::memory_order_acquire)) return;
    int current = 0;
    const cudaError_t selected = cudaGetDevice(&current);
    {
        GpuAllocationActivity activity(owner.get(), -1);
        const auto fail = [&](cudaError_t error) {
            // Keep the cache's owner reference, private stream and uncertain frees
            // alive. In particular, never retry a free whose completion is unknown.
            owner->memory_release_failed.store(true, std::memory_order_release);
            owner->unretired_work.store(true, std::memory_order_release);
            if (selected == cudaSuccess) cudaSetDevice(current);
            set_error(error);
        };
        if (selected != cudaSuccess) { fail(selected); return; }
        if (!cache->stream && (cache->sqrt_var || cache->update_coeff || cache->ready_event))
        {
            fail(cudaErrorInvalidResourceHandle);
            return;
        }
        if (cache->device >= 0 && cache->stream)
        {
            const auto found = std::find(owner->gpu_ids.begin(), owner->gpu_ids.end(), cache->device);
            if (found == owner->gpu_ids.end()) { fail(cudaErrorInvalidDevice); return; }
            const size_t partition = static_cast<size_t>(found - owner->gpu_ids.begin());
            if (partition >= owner->release_streams_by_partition.size() ||
                !owner->release_streams_by_partition[partition])
            {
                fail(cudaErrorInvalidResourceHandle);
                return;
            }
            cudaError_t error = cudaSetDevice(cache->device);
            // The cache is no longer borrowed. All existing waits capture its
            // earlier ready record; reuse that setup-created handle for frees.
            cudaEvent_t released = cache->ready_event;
            if (error == cudaSuccess && !released)
                error = cudaErrorInvalidResourceHandle;
            if (error == cudaSuccess && cache->sqrt_var)
            {
                if (cache->sqrt_owner.release(cache->stream) != 0) error = cudaErrorUnknown;
                cache->sqrt_var = nullptr;
            }
            if (error == cudaSuccess && cache->update_coeff)
            {
                if (cache->update_owner.release(cache->stream) != 0) error = cudaErrorUnknown;
                cache->update_coeff = nullptr;
            }
            if (error == cudaSuccess) error = cudaEventRecord(released, cache->stream);
            if (error == cudaSuccess)
                error = cudaStreamWaitEvent(owner->release_streams_by_partition[partition], released, 0);
            if (error == cudaSuccess && cache->stream_resource.release() != 0)
                error = cudaErrorUnknown;
            cache->ready_event = nullptr;
            cache->stream = nullptr;
            if (error != cudaSuccess) { fail(error); return; }
        }
        delete cache;
    }
    owner.reset();
    if (cudaSetDevice(current) != cudaSuccess)
        gpu_device_mark_allocation_unknown(current);
}

extern "C" int gpu_matrix_sample_p1_full_cached(
    const GpuP1CovarianceCache *cache,
    const GpuMatrix *tp2,
    GpuRngSeed seed,
    GpuMatrix *out)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_sample_p1_full_cached");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!cache || !tp2 || !out)
    {
        return set_error("invalid gpu_matrix_sample_p1_full_cached arguments");
    }
    if (cache->ctx != tp2->ctx || cache->ctx != out->ctx)
    {
        return set_error("context mismatch in gpu_matrix_sample_p1_full_cached");
    }
    if (cache->level != tp2->level || cache->level != out->level)
    {
        return set_error("level mismatch in gpu_matrix_sample_p1_full_cached");
    }
    if (tp2->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("tp2 must be coefficient format in gpu_matrix_sample_p1_full_cached");
    }
    if (tp2->rows != 2 * cache->d_rows || out->rows != 2 * cache->d_rows || out->cols != tp2->cols)
    {
        return set_error("tp2/out shape mismatch in gpu_matrix_sample_p1_full_cached");
    }
    const size_t cols = tp2->cols;
    if (cols == 0 || cache->d_rows == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }
    const int level = cache->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_sample_p1_full_cached");
    }
    const size_t crt_depth = static_cast<size_t>(level + 1);
    if (tp2->ctx->moduli.size() < crt_depth || tp2->ctx->limb_gpu_ids.size() < crt_depth)
    {
        return set_error("unexpected context size in gpu_matrix_sample_p1_full_cached");
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    const dim3 ref_limb_id = tp2->ctx->limb_gpu_ids[0];
    int ref_device = -1;
    int status = matrix_limb_device(out, ref_limb_id, &ref_device);
    if (status != 0)
    {
        return status;
    }
    cudaStream_t ref_stream = nullptr;
    status = matrix_limb_stream(out, ref_limb_id, &ref_stream);
    if (status != 0)
    {
        return status;
    }
    int tp2_ref_device = -1;
    status = matrix_limb_device(tp2, ref_limb_id, &tp2_ref_device);
    if (status != 0)
    {
        return status;
    }
    if (ref_device != cache->device || tp2_ref_device != ref_device)
    {
        return set_error("reference device mismatch in gpu_matrix_sample_p1_full_cached");
    }
    if (!ref_stream || ref_device < 0)
    {
        return set_error("invalid reference stream/device in gpu_matrix_sample_p1_full_cached");
    }

    const uint8_t *ref_tp2_base = matrix_limb_ptr_by_id(tp2, 0, ref_limb_id);
    if (!ref_tp2_base)
    {
        return set_error("null tp2 reference limb base pointer in gpu_matrix_sample_p1_full_cached");
    }
    size_t ref_tp2_stride = 0;
    uint8_t ref_tp2_coeff_bytes = 0;
    if (!matrix_limb_metadata_by_id(tp2, ref_limb_id, &ref_tp2_stride, &ref_tp2_coeff_bytes))
    {
        return set_error("invalid tp2 reference limb metadata in gpu_matrix_sample_p1_full_cached");
    }
    if (ref_tp2_stride < cache->n * static_cast<size_t>(ref_tp2_coeff_bytes))
    {
        return set_error("invalid tp2 reference limb stride in gpu_matrix_sample_p1_full_cached");
    }
    status = matrix_wait_limb_stream(tp2, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        return status;
    }

    cudaError_t err = cudaSetDevice(ref_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    if (cache->ready_event)
    {
        err = cudaStreamWaitEvent(ref_stream, cache->ready_event, 0);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }

    struct DeviceSampleBuffer
    {
        int device;
        int64_t *ptr;
        cudaStream_t owner_stream;
        cudaEvent_t ready_event;
        std::unique_ptr<GpuDeviceWorkspace> owner;
        std::unique_ptr<GpuCudaResource> ready_resource;
    };
    std::vector<DeviceSampleBuffer> sampled_device_buffers;
    auto reference_owner = std::make_unique<GpuDeviceWorkspace>();
    cudaEvent_t sampled_ready_event = nullptr;
    auto sampled_ready_resource = std::make_unique<GpuCudaResource>();
    int64_t *sampled_ref_device = nullptr;

    auto cleanup = [&]()
    {
        if (sampled_ready_resource && sampled_ready_resource->release() != 0)
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
        sampled_ready_event = nullptr;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device < 0)
            {
                entry.ptr = nullptr;
                if (entry.ready_event)
                {
                    if (entry.ready_resource->release() != 0)
                        gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                    entry.ready_event = nullptr;
                }
                continue;
            }
            if (cudaSetDevice(entry.device) != cudaSuccess)
            {
                out->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
                out->ctx->execution->unretired_work.store(true, std::memory_order_release);
                return;
            }
            if (entry.ptr)
            {
                if (entry.owner->release() != 0)
                    gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                entry.ptr = nullptr;
            }
            if (entry.ready_event)
            {
                if (entry.ready_resource->release() != 0)
                    gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                entry.ready_event = nullptr;
            }
        }
        sampled_device_buffers.clear();
    };

    status = sampled_ready_resource->acquire(out->ctx, ref_device, GPU_PREPARED_COMPLETION_EVENT);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    sampled_ready_event = sampled_ready_resource->event;

    status = launch_sample_p1_integer_cached_kernel(
        ref_tp2_base,
        ref_tp2_stride,
        ref_tp2_coeff_bytes,
        cache,
        cols,
        &sampled_ref_device,
        seed,
        ref_stream,
        ref_device,
        sampled_ready_event, out->ctx, *reference_owner);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer_readonly(tp2, ref_limb_id, ref_device, ref_stream, sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    err = cudaStreamWaitEvent(cache->stream, sampled_ready_event, 0);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    if (sampled_ref_device)
    {
        sampled_device_buffers.push_back(
            DeviceSampleBuffer{
                ref_device,
                sampled_ref_device,
                ref_stream,
                sampled_ready_event,
                std::move(reference_owner),
                std::move(sampled_ready_resource)});
        sampled_ready_event = nullptr;
    }

    const size_t sampled_entry_count = 2 * cache->d_rows * cols;
    if (sampled_entry_count > std::numeric_limits<size_t>::max() / cache->n ||
        sampled_entry_count * cache->n > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        cleanup();
        return set_error("sample byte overflow in gpu_matrix_sample_p1_full_cached");
    }
    const size_t sampled_bytes = sampled_entry_count * cache->n * sizeof(int64_t);

    auto ensure_sample_buffer_on_device = [&](int device, cudaStream_t stream, int64_t **out_ptr) -> int
    {
        if (!out_ptr)
        {
            return set_error("invalid output in cached ensure_sample_buffer_on_device");
        }
        if (device < 0 || !stream)
        {
            return set_error("invalid device/stream in cached ensure_sample_buffer_on_device");
        }
        *out_ptr = nullptr;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device == device && entry.ptr)
            {
                cudaError_t wait_err = cudaSetDevice(device);
                if (wait_err != cudaSuccess)
                {
                    return set_error(wait_err);
                }
                if (entry.ready_event)
                {
                    wait_err = cudaStreamWaitEvent(stream, entry.ready_event, 0);
                    if (wait_err != cudaSuccess)
                    {
                        return set_error(wait_err);
                    }
                }
                *out_ptr = entry.ptr;
                return 0;
            }
        }
        if (!sampled_ref_device)
        {
            return set_error("missing reference sample buffer in gpu_matrix_sample_p1_full_cached");
        }
        if (sampled_bytes == 0)
        {
            return 0;
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int64_t *device_copy = nullptr;
        auto copy_owner = std::make_unique<GpuDeviceWorkspace>();
        const int allocated = copy_owner->acquire(out->ctx, device, GPU_PREPARED_SAMPLER_WORKSPACE,
            sampled_bytes, alignof(int64_t), stream);
        if (allocated != 0) return allocated;
        device_copy = reinterpret_cast<int64_t *>(copy_owner->data);
        auto copy_resource = std::make_unique<GpuCudaResource>();
        const int resource_status = copy_resource->acquire(out->ctx, device, GPU_PREPARED_COMPLETION_EVENT);
        if (resource_status != 0) return resource_status;
        const cudaEvent_t copy_ready = copy_resource->event;
        err = cudaStreamWaitEvent(stream, sampled_device_buffers.front().ready_event, 0);
        if (err != cudaSuccess) return set_error(err);
        err = cudaMemcpyPeerAsync(device_copy, device, sampled_ref_device, ref_device, sampled_bytes, stream);
        if (err != cudaSuccess)
        {
            if (copy_owner->release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            return set_error(err);
        }
        err = cudaEventRecord(copy_ready, stream);
        if (err != cudaSuccess)
        {
            // A submitted peer reader without a completion edge cannot retire.
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            copy_resource->quarantine();
            if (copy_owner->release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            return set_error(err);
        }
        err = cudaSetDevice(ref_device);
        if (err == cudaSuccess) err = cudaStreamWaitEvent(ref_stream, copy_ready, 0);
        if (err == cudaSuccess) err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            copy_resource->quarantine();
            return set_error(err);
        }
        sampled_device_buffers.push_back(DeviceSampleBuffer{
            device, device_copy, stream, copy_ready, std::move(copy_owner), std::move(copy_resource)});
        *out_ptr = device_copy;
        return 0;
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = tp2->ctx->limb_gpu_ids[static_cast<size_t>(limb)];
        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        uint8_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!out_base)
        {
            cleanup();
            return set_error("null output limb base pointer in gpu_matrix_sample_p1_full_cached");
        }
        size_t out_stride = 0;
        uint8_t out_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_stride, &out_coeff_bytes))
        {
            cleanup();
            return set_error("invalid output limb metadata in gpu_matrix_sample_p1_full_cached");
        }

        int64_t *sampled_for_device = nullptr;
        status = ensure_sample_buffer_on_device(out_device, out_stream, &sampled_for_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_scatter_p1_integer_to_limb_kernel_device(
            sampled_for_device,
            out_base,
            out_stride,
            out_coeff_bytes,
            sampled_entry_count,
            cache->n,
            tp2->ctx->moduli[static_cast<size_t>(limb)],
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        // Retire the scratch reader after the scatter, never before it.
        // The output owns this event, and the wait captures this exact record.
        const cudaEvent_t scattered = out->exec_limb_states[limb_id.x][limb_id.y].write_done;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device != out_device || entry.ptr != sampled_for_device ||
                entry.owner_stream == out_stream) continue;
            err = cudaStreamWaitEvent(entry.owner_stream, scattered, 0);
            if (err != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                cleanup();
                return set_error(err);
            }
        }
    }

    status = gpu_matrix_ntt_all(out);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    cleanup();
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
    GpuRngSeed seed,
    GpuMatrix *out)
{
    if (!out || !out->ctx || !out->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_sample_p1_full");
    GpuAllocationActivity activity(out->ctx->execution.get(), -1);

    if (out) out->host_observed_writer_ready.store(false, std::memory_order_release);
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
        out->format = GPU_POLY_FORMAT_EVAL;
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
    auto &limb_map = a_mat->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_sample_p1_full");
    }

    GpuMatrix *tmp_a = nullptr;
    GpuMatrix *tmp_b = nullptr;
    GpuMatrix *tmp_d = nullptr;
    GpuMatrix *tmp_tp2 = nullptr;
    const GpuMatrix *a_input = a_mat;
    const GpuMatrix *b_input = b_mat;
    const GpuMatrix *d_input = d_mat;
    const GpuMatrix *tp2_input = tp2;
    struct DeviceSampleBuffer
    {
        int device;
        int64_t *ptr;
        cudaStream_t owner_stream;
        cudaEvent_t ready_event;
        std::unique_ptr<GpuDeviceWorkspace> owner;
        std::unique_ptr<GpuCudaResource> ready_resource;
    };
    std::vector<DeviceSampleBuffer> sampled_device_buffers;
    auto reference_owner = std::make_unique<GpuDeviceWorkspace>();
    cudaEvent_t sampled_ready_event = nullptr;
    auto sampled_ready_resource = std::make_unique<GpuCudaResource>();

    auto cleanup = [&]()
    {
        if (sampled_ready_resource && sampled_ready_resource->release() != 0)
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
        sampled_ready_event = nullptr;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device < 0)
            {
                entry.ptr = nullptr;
                if (entry.ready_event)
                {
                    if (entry.ready_resource->release() != 0)
                        gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                    entry.ready_event = nullptr;
                }
                continue;
            }
            if (cudaSetDevice(entry.device) != cudaSuccess)
            {
                out->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
                out->ctx->execution->unretired_work.store(true, std::memory_order_release);
                return;
            }
            if (entry.ptr)
            {
                if (entry.owner->release() != 0)
                    gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                entry.ptr = nullptr;
            }
            if (entry.ready_event)
            {
                if (entry.ready_resource->release() != 0)
                    gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                entry.ready_event = nullptr;
            }
        }
        sampled_device_buffers.clear();
        if (tmp_a)
        {
            gpu_matrix_destroy(tmp_a);
            tmp_a = nullptr;
        }
        if (tmp_b)
        {
            gpu_matrix_destroy(tmp_b);
            tmp_b = nullptr;
        }
        if (tmp_d)
        {
            gpu_matrix_destroy(tmp_d);
            tmp_d = nullptr;
        }
        if (tmp_tp2)
        {
            gpu_matrix_destroy(tmp_tp2);
            tmp_tp2 = nullptr;
        }
    };

    auto collect_coeff_input_matrix = [&](
                                          const GpuMatrix *src,
                                          GpuMatrix **owned,
                                          const GpuMatrix **coeff_input) -> int
    {
        *owned = nullptr;
        *coeff_input = src;
        if (src->format == GPU_POLY_FORMAT_EVAL)
        {
            const int matrix_format =
                src->format == GPU_POLY_FORMAT_EVAL ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
            int status = gpu_matrix_create(
                src->ctx,
                src->level,
                src->rows,
                src->cols,
                matrix_format,
                owned);
            if (status != 0)
            {
                return status;
            }
            status = gpu_matrix_copy(*owned, src);
            if (status != 0)
            {
                gpu_matrix_destroy(*owned);
                *owned = nullptr;
                return status;
            }
            status = gpu_matrix_intt_all(*owned);
            if (status != 0)
            {
                gpu_matrix_destroy(*owned);
                *owned = nullptr;
                return status;
            }
            *coeff_input = *owned;
        }
        return 0;
    };

    int status = collect_coeff_input_matrix(
        a_mat,
        &tmp_a,
        &a_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        b_mat,
        &tmp_b,
        &b_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        d_mat,
        &tmp_d,
        &d_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = collect_coeff_input_matrix(
        tp2,
        &tmp_tp2,
        &tp2_input);
    if (status != 0)
    {
        cleanup();
        return status;
    }

    out->format = GPU_POLY_FORMAT_COEFF;

    const dim3 ref_limb_id = limb_map[0];
    int ref_device = -1;
    status = matrix_limb_device(out, ref_limb_id, &ref_device);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    cudaStream_t ref_stream = nullptr;
    status = matrix_limb_stream(out, ref_limb_id, &ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    int tp2_ref_device = -1;
    status = matrix_limb_device(tp2_input, ref_limb_id, &tp2_ref_device);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    if (tp2_ref_device != ref_device)
    {
        cleanup();
        return set_error("input/output limb device mismatch in gpu_matrix_sample_p1_full");
    }
    const uint8_t *ref_a_base = matrix_limb_ptr_by_id(a_input, 0, ref_limb_id);
    const uint8_t *ref_b_base = matrix_limb_ptr_by_id(b_input, 0, ref_limb_id);
    const uint8_t *ref_d_base = matrix_limb_ptr_by_id(d_input, 0, ref_limb_id);
    const uint8_t *ref_tp2_base = matrix_limb_ptr_by_id(tp2_input, 0, ref_limb_id);
    if (!ref_a_base || !ref_b_base || !ref_d_base || !ref_tp2_base)
    {
        cleanup();
        return set_error("null reference limb base pointer in gpu_matrix_sample_p1_full");
    }
    size_t ref_a_stride = 0;
    size_t ref_b_stride = 0;
    size_t ref_d_stride = 0;
    size_t ref_tp2_stride = 0;
    uint8_t ref_a_coeff_bytes = 0;
    uint8_t ref_b_coeff_bytes = 0;
    uint8_t ref_d_coeff_bytes = 0;
    uint8_t ref_tp2_coeff_bytes = 0;
    if (!matrix_limb_metadata_by_id(a_input, ref_limb_id, &ref_a_stride, &ref_a_coeff_bytes) ||
        !matrix_limb_metadata_by_id(b_input, ref_limb_id, &ref_b_stride, &ref_b_coeff_bytes) ||
        !matrix_limb_metadata_by_id(d_input, ref_limb_id, &ref_d_stride, &ref_d_coeff_bytes) ||
        !matrix_limb_metadata_by_id(tp2_input, ref_limb_id, &ref_tp2_stride, &ref_tp2_coeff_bytes))
    {
        cleanup();
        return set_error("invalid reference limb metadata in gpu_matrix_sample_p1_full");
    }
    const size_t n = static_cast<size_t>(a_mat->ctx->N);
    if (ref_a_stride < n * static_cast<size_t>(ref_a_coeff_bytes) ||
        ref_b_stride < n * static_cast<size_t>(ref_b_coeff_bytes) ||
        ref_d_stride < n * static_cast<size_t>(ref_d_coeff_bytes) ||
        ref_tp2_stride < n * static_cast<size_t>(ref_tp2_coeff_bytes))
    {
        cleanup();
        return set_error("invalid reference limb stride in gpu_matrix_sample_p1_full");
    }
    if (!ref_stream || ref_device < 0)
    {
        cleanup();
        return set_error("invalid reference stream/device in gpu_matrix_sample_p1_full");
    }
    status = matrix_wait_limb_stream(a_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(b_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(d_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_wait_limb_stream(tp2_input, ref_limb_id, ref_device, ref_stream);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    cudaError_t err = cudaSetDevice(ref_device);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    status = sampled_ready_resource->acquire(out->ctx, ref_device, GPU_PREPARED_COMPLETION_EVENT);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    sampled_ready_event = sampled_ready_resource->event;

    int64_t *sampled_ref_device = nullptr;
    status = launch_sample_p1_integer_kernel(
        ref_a_base,
        ref_b_base,
        ref_d_base,
        ref_tp2_base,
        ref_a_stride,
        ref_b_stride,
        ref_d_stride,
        ref_tp2_stride,
        ref_a_coeff_bytes,
        ref_b_coeff_bytes,
        ref_d_coeff_bytes,
        ref_tp2_coeff_bytes,
        d_rows,
        cols,
        n,
        a_mat->ctx->moduli[0],
        sigma,
        s,
        dgg_stddev,
        seed,
        ref_stream,
        ref_device,
        &sampled_ref_device,
        sampled_ready_event, out->ctx, *reference_owner);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer_readonly(a_input, ref_limb_id, ref_device, ref_stream, sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer_readonly(b_input, ref_limb_id, ref_device, ref_stream, sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer_readonly(d_input, ref_limb_id, ref_device, ref_stream, sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    status = matrix_track_limb_consumer_readonly(tp2_input, ref_limb_id, ref_device, ref_stream, sampled_ready_event);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    if (sampled_ref_device)
    {
        sampled_device_buffers.push_back(
            DeviceSampleBuffer{ref_device, sampled_ref_device, ref_stream, sampled_ready_event, std::move(reference_owner), std::move(sampled_ready_resource)});
        sampled_ready_event = nullptr;
    }

    size_t sampled_entry_count = 0;
    size_t sampled_value_count = 0;
    size_t sampled_bytes = 0;
    if (d_rows != 0 && cols > std::numeric_limits<size_t>::max() / (2 * d_rows))
    {
        cleanup();
        return set_error("sample count overflow in gpu_matrix_sample_p1_full");
    }
    sampled_entry_count = 2 * d_rows * cols;
    if (static_cast<size_t>(a_mat->ctx->N) != 0 &&
        sampled_entry_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(a_mat->ctx->N))
    {
        cleanup();
        return set_error("sample count overflow in gpu_matrix_sample_p1_full");
    }
    sampled_value_count = sampled_entry_count * static_cast<size_t>(a_mat->ctx->N);
    if (sampled_value_count > std::numeric_limits<size_t>::max() / sizeof(int64_t))
    {
        cleanup();
        return set_error("sample byte overflow in gpu_matrix_sample_p1_full");
    }
    sampled_bytes = sampled_value_count * sizeof(int64_t);

    auto ensure_sample_buffer_on_device = [&](int device, cudaStream_t stream, int64_t **out_ptr) -> int
    {
        if (!out_ptr)
        {
            return set_error("invalid output in ensure_sample_buffer_on_device");
        }
        if (device < 0 || !stream)
        {
            return set_error("invalid device/stream in ensure_sample_buffer_on_device");
        }
        *out_ptr = nullptr;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device == device && entry.ptr)
            {
                cudaError_t wait_err = cudaSetDevice(device);
                if (wait_err != cudaSuccess)
                {
                    return set_error(wait_err);
                }
                if (entry.ready_event)
                {
                    wait_err = cudaStreamWaitEvent(stream, entry.ready_event, 0);
                    if (wait_err != cudaSuccess)
                    {
                        return set_error(wait_err);
                    }
                }
                *out_ptr = entry.ptr;
                return 0;
            }
        }
        if (!sampled_ref_device)
        {
            return set_error("missing reference sample buffer in gpu_matrix_sample_p1_full");
        }
        if (sampled_bytes == 0)
        {
            return 0;
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        int64_t *device_copy = nullptr;
        auto copy_owner = std::make_unique<GpuDeviceWorkspace>();
        const int allocated = copy_owner->acquire(out->ctx, device, GPU_PREPARED_SAMPLER_WORKSPACE,
            sampled_bytes, alignof(int64_t), stream);
        if (allocated != 0) return allocated;
        device_copy = reinterpret_cast<int64_t *>(copy_owner->data);
        auto copy_resource = std::make_unique<GpuCudaResource>();
        const int resource_status = copy_resource->acquire(out->ctx, device, GPU_PREPARED_COMPLETION_EVENT);
        if (resource_status != 0) return resource_status;
        const cudaEvent_t copy_ready = copy_resource->event;
        err = cudaStreamWaitEvent(stream, sampled_device_buffers.front().ready_event, 0);
        if (err != cudaSuccess) return set_error(err);
        err = cudaMemcpyPeerAsync(
            device_copy,
            device,
            sampled_ref_device,
            ref_device,
            sampled_bytes,
            stream);
        if (err != cudaSuccess)
        {
            if (copy_owner->release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            return set_error(err);
        }
        err = cudaEventRecord(copy_ready, stream);
        if (err != cudaSuccess)
        {
            // A submitted peer reader without a completion edge cannot retire.
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            copy_resource->quarantine();
            if (copy_owner->release() != 0)
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            return set_error(err);
        }
        err = cudaSetDevice(ref_device);
        if (err == cudaSuccess) err = cudaStreamWaitEvent(ref_stream, copy_ready, 0);
        if (err == cudaSuccess) err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
            copy_resource->quarantine();
            return set_error(err);
        }
        sampled_device_buffers.push_back(DeviceSampleBuffer{
            device, device_copy, stream, copy_ready, std::move(copy_owner), std::move(copy_resource)});
        *out_ptr = device_copy;
        return 0;
    };

    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        uint8_t *out_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!out_base)
        {
            cleanup();
            return set_error("null output limb base pointer in gpu_matrix_sample_p1_full");
        }
        size_t out_stride = 0;
        uint8_t out_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_stride, &out_coeff_bytes))
        {
            cleanup();
            return set_error("invalid output limb metadata in gpu_matrix_sample_p1_full");
        }

        int64_t *sampled_for_device = nullptr;
        status = ensure_sample_buffer_on_device(out_device, out_stream, &sampled_for_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        status = launch_scatter_p1_integer_to_limb_kernel_device(
            sampled_for_device,
            out_base,
            out_stride,
            out_coeff_bytes,
            sampled_entry_count,
            static_cast<size_t>(a_mat->ctx->N),
            a_mat->ctx->moduli[static_cast<size_t>(limb)],
            out_stream,
            out_device);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, out_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        // Retire the scratch reader after the scatter, never before it.
        // The output owns this event, and the wait captures this exact record.
        const cudaEvent_t scattered = out->exec_limb_states[limb_id.x][limb_id.y].write_done;
        for (auto &entry : sampled_device_buffers)
        {
            if (entry.device != out_device || entry.ptr != sampled_for_device ||
                entry.owner_stream == out_stream) continue;
            err = cudaStreamWaitEvent(entry.owner_stream, scattered, 0);
            if (err != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(out->ctx->execution.get());
                cleanup();
                return set_error(err);
            }
        }
    }

    status = gpu_matrix_ntt_all(out);
    if (status != 0)
    {
        cleanup();
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;

    cleanup();
    return 0;
}
