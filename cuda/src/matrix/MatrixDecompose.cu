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

namespace
{
    constexpr uint32_t kDecomposeThreads = 256;
    constexpr size_t kDecomposeMaxGridY = 65535;
    constexpr size_t kDecomposeMaxGridZ = 65535;
}

__global__ void matrix_decompose_all_slots_kernel(
    const uint8_t *src_base,
    uint8_t *const *dst_bases,
    const size_t *dst_stride_bytes,
    const uint8_t *dst_coeff_bytes,
    const uint64_t *dst_moduli,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    size_t out_limb_count,
    size_t slot_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint32_t src_bits,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    size_t src_digit_offset_base,
    size_t poly_offset,
    size_t slot_offset)
{
    if (!src_base || !dst_bases || !dst_stride_bytes || !dst_coeff_bytes || !dst_moduli)
    {
        return;
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0 || digits_per_tower == 0)
    {
        return;
    }
    const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (coeff_idx >= n)
    {
        return;
    }
    const size_t poly_idx = poly_offset + static_cast<size_t>(blockIdx.y);
    if (poly_idx >= poly_count)
    {
        return;
    }
    const size_t slot_idx = slot_offset + static_cast<size_t>(blockIdx.z);
    if (slot_idx >= slot_count)
    {
        return;
    }
    const size_t out_limb = slot_idx / static_cast<size_t>(digits_per_tower);
    if (out_limb >= out_limb_count)
    {
        return;
    }
    const uint32_t digit_idx = static_cast<uint32_t>(slot_idx % static_cast<size_t>(digits_per_tower));

    uint8_t *const dst_base = dst_bases[out_limb];
    const size_t dst_stride = dst_stride_bytes[out_limb];
    const uint8_t dst_bytes = dst_coeff_bytes[out_limb];
    const uint64_t out_modulus = dst_moduli[out_limb];
    if (!dst_base || dst_bytes == 0 || dst_stride < n * static_cast<size_t>(dst_bytes))
    {
        return;
    }

    const uint64_t residue =
        matrix_load_limb_u64(src_base, poly_idx, coeff_idx, src_stride_bytes, src_coeff_bytes);
    const uint32_t shift = digit_idx * base_bits;
    uint64_t mask = 0;
    if (shift < src_bits)
    {
        const uint32_t remaining = src_bits - shift;
        const uint32_t digit_bits = min(base_bits, remaining);
        mask = digit_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << digit_bits) - 1);
    }
    uint64_t digit = shift >= 64 ? 0 : ((residue >> shift) & mask);
    if (out_modulus != 0 && digit >= out_modulus)
    {
        digit %= out_modulus;
    }

    const size_t row = poly_idx / src_cols;
    const size_t col = poly_idx - row * src_cols;
    const size_t out_row = row * log_base_q + src_digit_offset_base + static_cast<size_t>(digit_idx);
    const size_t out_poly_idx = out_row * out_cols + col;
    matrix_store_limb_u64(dst_base, out_poly_idx, coeff_idx, dst_stride, dst_bytes, digit);
}

int launch_decompose_all_slots_kernel(
    const uint8_t *src_base,
    uint8_t *const *dst_bases,
    const size_t *dst_stride_bytes,
    const uint8_t *dst_coeff_bytes,
    const uint64_t *dst_moduli,
    size_t src_stride_bytes,
    uint8_t src_coeff_bytes,
    size_t out_limb_count,
    size_t poly_count,
    size_t n,
    size_t src_cols,
    size_t out_cols,
    size_t log_base_q,
    uint32_t src_bits,
    uint32_t base_bits,
    uint32_t digits_per_tower,
    size_t src_digit_offset_base,
    cudaStream_t stream)
{
    if (!src_base || !dst_bases || !dst_stride_bytes || !dst_coeff_bytes || !dst_moduli)
    {
        return set_error("null pointer in matrix_decompose_all_slots_kernel");
    }
    if (out_limb_count == 0 || poly_count == 0 || n == 0)
    {
        return 0;
    }
    if (src_coeff_bytes == 0 || src_stride_bytes < n * static_cast<size_t>(src_coeff_bytes))
    {
        return set_error("invalid src stride in matrix_decompose_all_slots_kernel");
    }
    if (src_cols == 0 || out_cols == 0 || log_base_q == 0)
    {
        return set_error("invalid matrix shape in matrix_decompose_all_slots_kernel");
    }
    if (digits_per_tower == 0)
    {
        return set_error("invalid digit count in matrix_decompose_all_slots_kernel");
    }
    if (!stream)
    {
        return set_error("null stream in matrix_decompose_all_slots_kernel");
    }
    if (out_limb_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(digits_per_tower))
    {
        return set_error("slot count overflow in matrix_decompose_all_slots_kernel");
    }
    const size_t slot_count = out_limb_count * static_cast<size_t>(digits_per_tower);
    if (slot_count == 0)
    {
        return 0;
    }

    const uint32_t blocks_x = static_cast<uint32_t>((n + kDecomposeThreads - 1) / kDecomposeThreads);
    for (size_t poly_offset = 0; poly_offset < poly_count; poly_offset += kDecomposeMaxGridY)
    {
        const size_t poly_chunk = std::min(kDecomposeMaxGridY, poly_count - poly_offset);
        for (size_t slot_offset = 0; slot_offset < slot_count; slot_offset += kDecomposeMaxGridZ)
        {
            const size_t slot_chunk = std::min(kDecomposeMaxGridZ, slot_count - slot_offset);
            const dim3 grid{
                blocks_x,
                static_cast<uint32_t>(poly_chunk),
                static_cast<uint32_t>(slot_chunk)};
            matrix_decompose_all_slots_kernel<<<grid, kDecomposeThreads, 0, stream>>>(
                src_base,
                dst_bases,
                dst_stride_bytes,
                dst_coeff_bytes,
                dst_moduli,
                src_stride_bytes,
                src_coeff_bytes,
                out_limb_count,
                slot_count,
                poly_count,
                n,
                src_cols,
                out_cols,
                log_base_q,
                src_bits,
                base_bits,
                digits_per_tower,
                src_digit_offset_base,
                poly_offset,
                slot_offset);
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
    }
    return 0;
}

__global__ void matrix_fill_gadget_multi_limb_kernel(
    uint8_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = poly_count * n;
    if (idx >= total)
    {
        return;
    }
    const size_t poly_idx = idx / n;
    const size_t coeff_idx = idx - poly_idx * n;

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
    matrix_store_limb_u64(dst_base, poly_idx, coeff_idx, dst_stride_bytes, dst_coeff_bytes, value);
}

int launch_fill_gadget_multi_limb_kernel(
    uint8_t *dst_base,
    size_t poly_count,
    size_t n,
    size_t dst_stride_bytes,
    uint8_t dst_coeff_bytes,
    uint64_t modulus,
    uint32_t limb_idx,
    size_t rows,
    size_t cols,
    size_t log_base_q,
    uint32_t digits_per_tower,
    uint32_t base_bits,
    cudaStream_t stream,
    const GpuMatrix *,
    const dim3 *)
{
    if (!dst_base)
    {
        return set_error("null output base pointer in matrix_fill_gadget_multi_limb_kernel");
    }
    if (poly_count == 0 || n == 0)
    {
        return 0;
    }

    const int threads = 256;
    const size_t total = poly_count * n;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    matrix_fill_gadget_multi_limb_kernel<<<blocks, threads, 0, stream>>>(
        dst_base,
        poly_count,
        n,
        dst_stride_bytes,
        dst_coeff_bytes,
        modulus,
        limb_idx,
        rows,
        cols,
        log_base_q,
        digits_per_tower,
        base_bits);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return set_error(err);
    }
    return 0;
}


static int gpu_matrix_fill_gadget_impl(
    GpuMatrix *out,
    uint32_t base_bits,
    bool small)
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
        out->format = GPU_POLY_FORMAT_EVAL;
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
    auto &limb_map = out->ctx->limb_gpu_ids;
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
    const size_t log_base_q =
        small ? static_cast<size_t>(digits_per_tower)
              : static_cast<size_t>(digits_per_tower) * crt_depth;
    if (cols != rows * log_base_q)
    {
        return set_error("output size mismatch in gpu_matrix_fill_gadget");
    }
    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int limb_device = -1;
        cudaStream_t limb_stream = nullptr;
        status = matrix_limb_device(out, limb_id, &limb_device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(out, limb_id, &limb_stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_device < 0 || !limb_stream)
        {
            return set_error("invalid limb metadata in gpu_matrix_fill_gadget");
        }
        uint8_t *dst_base = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst_base)
        {
            return set_error("null output limb base pointer in gpu_matrix_fill_gadget");
        }
        size_t dst_stride_bytes = 0;
        uint8_t dst_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(out, limb_id, &dst_stride_bytes, &dst_coeff_bytes))
        {
            return set_error("invalid output limb metadata in gpu_matrix_fill_gadget");
        }

        cudaError_t err = cudaSetDevice(limb_device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = launch_fill_gadget_multi_limb_kernel(
            dst_base,
            count,
            static_cast<size_t>(out->ctx->N),
            dst_stride_bytes,
            dst_coeff_bytes,
            out->ctx->moduli[static_cast<size_t>(limb)],
            small ? 0u : static_cast<uint32_t>(limb),
            rows,
            cols,
            log_base_q,
            digits_per_tower,
            base_bits,
            limb_stream,
            out,
            &limb_id);
        if (status != 0)
        {
            return status;
        }
        status = matrix_record_limb_write(out, limb_id, limb_stream);
        if (status != 0)
        {
            return status;
        }
    }

    out->format = GPU_POLY_FORMAT_COEFF;
    status = gpu_matrix_ntt_all(out);
    if (status != 0)
    {
        return status;
    }
    out->format = GPU_POLY_FORMAT_EVAL;
    return 0;
}

extern "C" int gpu_matrix_fill_gadget(
    GpuMatrix *out,
    uint32_t base_bits)
{
    return gpu_matrix_fill_gadget_impl(out, base_bits, false);
}

extern "C" int gpu_matrix_fill_small_gadget(
    GpuMatrix *out,
    uint32_t base_bits)
{
    return gpu_matrix_fill_gadget_impl(out, base_bits, true);
}

__global__ void matrix_fill_small_decomposed_identity_chunk_all_limbs_kernel(
    const uint8_t *const *src_bases,
    uint8_t *const *dst_bases,
    const size_t *src_stride_bytes,
    const size_t *dst_stride_bytes,
    const uint8_t *src_coeff_bytes,
    const uint8_t *dst_coeff_bytes,
    size_t limb_count,
    size_t n,
    size_t size,
    size_t chunk_idx,
    size_t chunk_count)
{
    const size_t limb_idx = static_cast<size_t>(blockIdx.z);
    if (limb_idx >= limb_count)
    {
        return;
    }
    const uint8_t *src_base = src_bases ? src_bases[limb_idx] : nullptr;
    uint8_t *dst_base = dst_bases ? dst_bases[limb_idx] : nullptr;
    const size_t src_stride = src_stride_bytes ? src_stride_bytes[limb_idx] : 0;
    const size_t dst_stride = dst_stride_bytes ? dst_stride_bytes[limb_idx] : 0;
    const uint8_t src_bytes = src_coeff_bytes ? src_coeff_bytes[limb_idx] : 0;
    const uint8_t dst_bytes = dst_coeff_bytes ? dst_coeff_bytes[limb_idx] : 0;
    if (!src_base || !dst_base || src_bytes == 0 || dst_bytes == 0 ||
        src_stride < n * static_cast<size_t>(src_bytes) ||
        dst_stride < n * static_cast<size_t>(dst_bytes))
    {
        return;
    }

    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size * n;
    if (idx >= total)
    {
        return;
    }
    const size_t local_row = idx / n;
    const size_t coeff_idx = idx - local_row * n;
    const size_t global_row = chunk_idx * size + local_row;
    const size_t src_row = global_row / chunk_count;
    const size_t digit = global_row - src_row * chunk_count;
    if (src_row >= size || digit >= chunk_count)
    {
        return;
    }

    const size_t src_poly_idx = digit;
    const size_t dst_poly_idx = local_row * size + src_row;
    const uint64_t value = matrix_load_limb_u64(src_base, src_poly_idx, coeff_idx, src_stride, src_bytes);
    matrix_store_limb_u64(dst_base, dst_poly_idx, coeff_idx, dst_stride, dst_bytes, value);
}

extern "C" int gpu_matrix_fill_small_decomposed_identity_chunk(
    GpuMatrix *out,
    const GpuMatrix *scalar_by_digit,
    size_t chunk_idx)
{
    if (!out || !scalar_by_digit)
    {
        return set_error("invalid gpu_matrix_fill_small_decomposed_identity_chunk arguments");
    }
    if (out->ctx != scalar_by_digit->ctx || out->level != scalar_by_digit->level)
    {
        return set_error("context mismatch in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (out->rows != out->cols)
    {
        return set_error("output must be square in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (scalar_by_digit->rows != 1 || scalar_by_digit->cols == 0)
    {
        return set_error("scalar_by_digit must be 1 x chunk_count in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (out->format != scalar_by_digit->format)
    {
        return set_error("format mismatch in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    const size_t size = out->rows;
    const size_t chunk_count = scalar_by_digit->cols;
    if (chunk_idx >= chunk_count)
    {
        return set_error("chunk_idx out of range in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    if (size == 0)
    {
        return 0;
    }

    const int level = out->level;
    if (level < 0)
    {
        return set_error("invalid level in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    const size_t limb_count = static_cast<size_t>(level + 1);
    auto &limb_map = out->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_fill_small_decomposed_identity_chunk");
    }

    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    std::vector<dim3> active_limb_ids(limb_count);
    std::vector<uint8_t *> out_limb_bases(limb_count, nullptr);
    std::vector<const uint8_t *> src_limb_bases(limb_count, nullptr);
    std::vector<size_t> out_limb_stride_bytes(limb_count, 0);
    std::vector<size_t> src_limb_stride_bytes(limb_count, 0);
    std::vector<uint8_t> out_limb_coeff_bytes(limb_count, 0);
    std::vector<uint8_t> src_limb_coeff_bytes(limb_count, 0);

    int status = 0;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[idx];
        active_limb_ids[idx] = limb_id;

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            return status;
        }
        if (out_device < 0 || !out_stream)
        {
            return set_error("invalid output limb metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_stream = out_stream;
        }
        else if (out_device != dispatch_device)
        {
            return set_error(
                "single-device mode requires all limbs on one device in gpu_matrix_fill_small_decomposed_identity_chunk");
        }

        int src_device = -1;
        status = matrix_limb_device(scalar_by_digit, limb_id, &src_device);
        if (status != 0)
        {
            return status;
        }
        if (src_device != dispatch_device)
        {
            return set_error(
                "single-device mode requires scalar limbs on dispatch device in gpu_matrix_fill_small_decomposed_identity_chunk");
        }

        uint8_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            return set_error("null output limb pointer in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        const uint8_t *src = matrix_limb_ptr_by_id(scalar_by_digit, 0, limb_id);
        if (!src)
        {
            return set_error("null source limb pointer in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_limb_stride_bytes[idx], &out_limb_coeff_bytes[idx]) ||
            !matrix_limb_metadata_by_id(
                scalar_by_digit,
                limb_id,
                &src_limb_stride_bytes[idx],
                &src_limb_coeff_bytes[idx]))
        {
            return set_error("invalid limb metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        if (out_limb_coeff_bytes[idx] != src_limb_coeff_bytes[idx])
        {
            return set_error("inconsistent limb byte-width in gpu_matrix_fill_small_decomposed_identity_chunk");
        }
        out_limb_bases[idx] = dst;
        src_limb_bases[idx] = src;
    }

    if (dispatch_device < 0 || !dispatch_stream)
    {
        return set_error("invalid dispatch metadata in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    cudaError_t err = cudaSetDevice(dispatch_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    if (limb_count > kDecomposeMaxGridZ)
    {
        return set_error("too many limbs in gpu_matrix_fill_small_decomposed_identity_chunk");
    }

    const size_t out_count = size * size;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = active_limb_ids[idx];
        status = matrix_wait_limb_stream(out, limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        status = matrix_wait_limb_stream(scalar_by_digit, limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            return status;
        }
        const size_t dst_pitch = out_limb_stride_bytes[idx];
        const size_t row_bytes = static_cast<size_t>(out->ctx->N) * static_cast<size_t>(out_limb_coeff_bytes[idx]);
        err = cudaMemset2DAsync(
            out_limb_bases[idx],
            dst_pitch,
            0,
            row_bytes,
            out_count,
            dispatch_stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }

    if (limb_count > std::numeric_limits<size_t>::max() / sizeof(uint8_t *) ||
        limb_count > std::numeric_limits<size_t>::max() / sizeof(size_t) ||
        limb_count > std::numeric_limits<size_t>::max() / sizeof(uint8_t))
    {
        return set_error("limb metadata size overflow in gpu_matrix_fill_small_decomposed_identity_chunk");
    }
    const size_t limb_ptr_bytes = limb_count * sizeof(uint8_t *);
    const size_t limb_stride_bytes = limb_count * sizeof(size_t);
    const size_t limb_coeff_bytes = limb_count * sizeof(uint8_t);
    uint8_t **src_limb_bases_device = nullptr;
    uint8_t **out_limb_bases_device = nullptr;
    size_t *src_limb_stride_bytes_device = nullptr;
    size_t *out_limb_stride_bytes_device = nullptr;
    uint8_t *src_limb_coeff_bytes_device = nullptr;
    uint8_t *out_limb_coeff_bytes_device = nullptr;
    auto cleanup_dispatch_allocs = [&]()
    {
        if (dispatch_device >= 0)
        {
            cudaSetDevice(dispatch_device);
        }
        if (src_limb_bases_device)
        {
            cudaFreeAsync(src_limb_bases_device, dispatch_stream);
            src_limb_bases_device = nullptr;
        }
        if (out_limb_bases_device)
        {
            cudaFreeAsync(out_limb_bases_device, dispatch_stream);
            out_limb_bases_device = nullptr;
        }
        if (src_limb_coeff_bytes_device)
        {
            cudaFreeAsync(src_limb_coeff_bytes_device, dispatch_stream);
            src_limb_coeff_bytes_device = nullptr;
        }
        if (out_limb_coeff_bytes_device)
        {
            cudaFreeAsync(out_limb_coeff_bytes_device, dispatch_stream);
            out_limb_coeff_bytes_device = nullptr;
        }
        if (src_limb_stride_bytes_device)
        {
            cudaFreeAsync(src_limb_stride_bytes_device, dispatch_stream);
            src_limb_stride_bytes_device = nullptr;
        }
        if (out_limb_stride_bytes_device)
        {
            cudaFreeAsync(out_limb_stride_bytes_device, dispatch_stream);
            out_limb_stride_bytes_device = nullptr;
        }
    };

    err = cudaMallocAsync(reinterpret_cast<void **>(&src_limb_bases_device), limb_ptr_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_bases_device), limb_ptr_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&src_limb_stride_bytes_device), limb_stride_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_stride_bytes_device), limb_stride_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&src_limb_coeff_bytes_device), limb_coeff_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_coeff_bytes_device), limb_coeff_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }

    err = cudaMemcpyAsync(
        src_limb_bases_device,
        src_limb_bases.data(),
        limb_ptr_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_bases_device,
        out_limb_bases.data(),
        limb_ptr_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        src_limb_stride_bytes_device,
        src_limb_stride_bytes.data(),
        limb_stride_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_stride_bytes_device,
        out_limb_stride_bytes.data(),
        limb_stride_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        src_limb_coeff_bytes_device,
        src_limb_coeff_bytes.data(),
        limb_coeff_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        out_limb_coeff_bytes_device,
        out_limb_coeff_bytes.data(),
        limb_coeff_bytes,
        cudaMemcpyHostToDevice,
        dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }

    const int threads = 256;
    const size_t total = size * static_cast<size_t>(out->ctx->N);
    const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / threads);
    const dim3 grid{
        static_cast<unsigned int>(blocks),
        1u,
        static_cast<unsigned int>(limb_count)};
    matrix_fill_small_decomposed_identity_chunk_all_limbs_kernel<<<grid, threads, 0, dispatch_stream>>>(
        src_limb_bases_device,
        out_limb_bases_device,
        src_limb_stride_bytes_device,
        out_limb_stride_bytes_device,
        src_limb_coeff_bytes_device,
        out_limb_coeff_bytes_device,
        limb_count,
        static_cast<size_t>(out->ctx->N),
        size,
        chunk_idx,
        chunk_count);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cleanup_dispatch_allocs();
        return set_error(err);
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        status = matrix_track_limb_consumer(
            scalar_by_digit,
            active_limb_ids[idx],
            dispatch_device,
            dispatch_stream);
        if (status != 0)
        {
            cleanup_dispatch_allocs();
            return status;
        }
        status = matrix_record_limb_write(out, active_limb_ids[idx], dispatch_stream);
        if (status != 0)
        {
            cleanup_dispatch_allocs();
            return status;
        }
    }
    cleanup_dispatch_allocs();
    out->format = scalar_by_digit->format;
    return 0;
}

static int gpu_matrix_decompose_base_impl(
    const GpuMatrix *src,
    uint32_t base_bits,
    GpuMatrix *out,
    bool small)
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
    GpuPolyFormat requested_out_format = GPU_POLY_FORMAT_EVAL;
    if (!parse_format(out->format, requested_out_format))
    {
        return set_error("invalid output format in gpu_matrix_decompose_base");
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
    const size_t out_log_base_q =
        small ? static_cast<size_t>(digits_per_tower)
              : static_cast<size_t>(digits_per_tower) * crt_depth;
    if (out->rows != rows * out_log_base_q || out->cols != cols)
    {
        return set_error("output size mismatch in gpu_matrix_decompose_base");
    }
    if (count == 0)
    {
        out->format = GPU_POLY_FORMAT_EVAL;
        return 0;
    }

    GpuMatrix *tmp_inputs_matrix = nullptr;
    const GpuMatrix *inputs_matrix = src;
    auto cleanup_tmp_inputs = [&]()
    {
        if (tmp_inputs_matrix)
        {
            gpu_matrix_destroy(tmp_inputs_matrix);
            tmp_inputs_matrix = nullptr;
        }
    };

    int status = 0;
    if (src->format == GPU_POLY_FORMAT_EVAL)
    {
        const int matrix_format =
            src->format == GPU_POLY_FORMAT_EVAL ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
        status = gpu_matrix_create(src->ctx, level, rows, cols, matrix_format, &tmp_inputs_matrix);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = gpu_matrix_copy(tmp_inputs_matrix, src);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        status = gpu_matrix_intt_all(tmp_inputs_matrix);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        inputs_matrix = tmp_inputs_matrix;
    }

    auto &limb_map = src->ctx->limb_gpu_ids;
    if (limb_map.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected limb mapping size in gpu_matrix_decompose_base");
    }

    if (src->ctx->moduli.size() < crt_depth)
    {
        cleanup_tmp_inputs();
        return set_error("unexpected modulus count in gpu_matrix_decompose_base");
    }

    const size_t limb_count = static_cast<size_t>(level + 1);
    std::vector<dim3> active_limb_ids(limb_count);
    std::vector<uint8_t *> out_limb_bases(limb_count, nullptr);
    std::vector<size_t> out_limb_stride_bytes(limb_count, 0);
    std::vector<uint8_t> out_limb_coeff_bytes(limb_count, 0);
    std::vector<uint64_t> out_limb_moduli(limb_count, 0);

    int dispatch_device = -1;
    cudaStream_t dispatch_stream = nullptr;
    for (int limb = 0; limb <= level; ++limb)
    {
        const size_t idx = static_cast<size_t>(limb);
        const dim3 limb_id = limb_map[idx];
        active_limb_ids[idx] = limb_id;

        int out_device = -1;
        status = matrix_limb_device(out, limb_id, &out_device);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        cudaStream_t out_stream = nullptr;
        status = matrix_limb_stream(out, limb_id, &out_stream);
        if (status != 0)
        {
            cleanup_tmp_inputs();
            return status;
        }
        if (out_device < 0 || !out_stream)
        {
            cleanup_tmp_inputs();
            return set_error("invalid output limb metadata in gpu_matrix_decompose_base");
        }
        if (limb == 0)
        {
            dispatch_device = out_device;
            dispatch_stream = out_stream;
        }
        else if (out_device != dispatch_device)
        {
            cleanup_tmp_inputs();
            return set_error("single-device mode requires all limbs on one device in gpu_matrix_decompose_base");
        }

        uint8_t *dst = matrix_limb_ptr_by_id(out, 0, limb_id);
        if (!dst)
        {
            cleanup_tmp_inputs();
            return set_error("null output limb pointer in gpu_matrix_decompose_base");
        }
        if (!matrix_limb_metadata_by_id(out, limb_id, &out_limb_stride_bytes[idx], &out_limb_coeff_bytes[idx]))
        {
            cleanup_tmp_inputs();
            return set_error("invalid output limb metadata in gpu_matrix_decompose_base");
        }
        out_limb_bases[idx] = dst;
        out_limb_moduli[idx] = src->ctx->moduli[idx];
    }
    if (dispatch_device < 0 || !dispatch_stream)
    {
        cleanup_tmp_inputs();
        return set_error("invalid dispatch stream in gpu_matrix_decompose_base");
    }

    const int src_limb_begin = 0;
    const int src_limb_end = small ? 1 : (level + 1);
    for (int src_limb = src_limb_begin; src_limb < src_limb_end; ++src_limb)
    {
        const dim3 src_limb_id = active_limb_ids[static_cast<size_t>(src_limb)];
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
            return set_error("single-device mode requires all limbs on one device in gpu_matrix_decompose_base");
        }
    }

    if (limb_count > std::numeric_limits<size_t>::max() / sizeof(uint8_t *) ||
        limb_count > std::numeric_limits<size_t>::max() / sizeof(size_t) ||
        limb_count > std::numeric_limits<size_t>::max() / sizeof(uint8_t) ||
        limb_count > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
    {
        cleanup_tmp_inputs();
        return set_error("limb metadata size overflow in gpu_matrix_decompose_base");
    }
    const size_t out_ptr_bytes = limb_count * sizeof(uint8_t *);
    const size_t out_stride_bytes = limb_count * sizeof(size_t);
    const size_t out_coeff_bytes = limb_count * sizeof(uint8_t);
    const size_t out_moduli_bytes = limb_count * sizeof(uint64_t);

    uint8_t **out_limb_bases_device = nullptr;
    size_t *out_limb_stride_bytes_device = nullptr;
    uint8_t *out_limb_coeff_bytes_device = nullptr;
    uint64_t *out_limb_moduli_device = nullptr;
    auto cleanup = [&]()
    {
        if (dispatch_device >= 0)
        {
            cudaSetDevice(dispatch_device);
        }
        if (out_limb_bases_device)
        {
            cudaFreeAsync(out_limb_bases_device, dispatch_stream);
            out_limb_bases_device = nullptr;
        }
        if (out_limb_stride_bytes_device)
        {
            cudaFreeAsync(out_limb_stride_bytes_device, dispatch_stream);
            out_limb_stride_bytes_device = nullptr;
        }
        if (out_limb_coeff_bytes_device)
        {
            cudaFreeAsync(out_limb_coeff_bytes_device, dispatch_stream);
            out_limb_coeff_bytes_device = nullptr;
        }
        if (out_limb_moduli_device)
        {
            cudaFreeAsync(out_limb_moduli_device, dispatch_stream);
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
    for (int out_limb = 0; out_limb <= level; ++out_limb)
    {
        const size_t out_idx = static_cast<size_t>(out_limb);
        status = matrix_wait_limb_stream(out, active_limb_ids[out_idx], dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        if (out_count > 0)
        {
            const size_t dst_pitch = out_limb_stride_bytes[out_idx];
            const size_t row_bytes =
                static_cast<size_t>(src->ctx->N) * static_cast<size_t>(out_limb_coeff_bytes[out_idx]);
            err = cudaMemset2DAsync(
                out_limb_bases[out_idx],
                dst_pitch,
                0,
                row_bytes,
                out_count,
                dispatch_stream);
            if (err != cudaSuccess)
            {
                cleanup();
                return set_error(err);
            }
        }
    }

    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_bases_device), out_ptr_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_stride_bytes_device), out_stride_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_coeff_bytes_device), out_coeff_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }
    err = cudaMallocAsync(reinterpret_cast<void **>(&out_limb_moduli_device), out_moduli_bytes, dispatch_stream);
    if (err != cudaSuccess)
    {
        cleanup();
        return set_error(err);
    }

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
        out_limb_stride_bytes_device,
        out_limb_stride_bytes.data(),
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

    for (int src_limb = src_limb_begin; src_limb < src_limb_end; ++src_limb)
    {
        const size_t src_idx = static_cast<size_t>(src_limb);
        const dim3 src_limb_id = active_limb_ids[src_idx];
        status = matrix_wait_limb_stream(inputs_matrix, src_limb_id, dispatch_device, dispatch_stream);
        if (status != 0)
        {
            cleanup();
            return status;
        }

        const uint8_t *src_base = matrix_limb_ptr_by_id(inputs_matrix, 0, src_limb_id);
        if (!src_base)
        {
            cleanup();
            return set_error("null source limb base pointer in gpu_matrix_decompose_base");
        }
        size_t src_stride_bytes = 0;
        uint8_t src_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(
                inputs_matrix,
                src_limb_id,
                &src_stride_bytes,
                &src_coeff_bytes))
        {
            cleanup();
            return set_error("invalid source limb metadata in gpu_matrix_decompose_base");
        }
        const uint32_t src_bits = bit_width_u64(src->ctx->moduli[src_idx]);
        const size_t src_digit_offset_base =
            small ? 0 : (src_idx * static_cast<size_t>(digits_per_tower));

        status = launch_decompose_all_slots_kernel(
            src_base,
            out_limb_bases_device,
            out_limb_stride_bytes_device,
            out_limb_coeff_bytes_device,
            out_limb_moduli_device,
            src_stride_bytes,
            src_coeff_bytes,
            limb_count,
            count,
            static_cast<size_t>(src->ctx->N),
            cols,
            out->cols,
            out_log_base_q,
            src_bits,
            base_bits,
            digits_per_tower,
            src_digit_offset_base,
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

extern "C" int gpu_matrix_decompose_base(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out)
{
    return gpu_matrix_decompose_base_impl(src, base_bits, out, false);
}

extern "C" int gpu_matrix_decompose_base_small(const GpuMatrix *src, uint32_t base_bits, GpuMatrix *out)
{
    return gpu_matrix_decompose_base_impl(src, base_bits, out, true);
}
