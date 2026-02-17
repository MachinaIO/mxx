namespace
{
    __global__ void decompose_base_kernel(
        const uint64_t *src,
        uint64_t *dst,
        size_t n,
        uint32_t shift,
        uint64_t mask,
        uint64_t out_modulus)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            uint64_t residue = src[idx];
            uint64_t digit = shift >= 64 ? 0 : ((residue >> shift) & mask);
            if (out_modulus != 0 && digit >= out_modulus)
            {
                digit %= out_modulus;
            }
            dst[idx] = digit;
        }
    }

    __global__ void compare_u64_kernel(
        const uint64_t *lhs,
        const uint64_t *rhs,
        size_t n,
        int *out_equal)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n && lhs[idx] != rhs[idx])
        {
            atomicExch(out_equal, 0);
        }
    }
    int compare_u64_arrays_on_device(
        const uint64_t *lhs,
        const uint64_t *rhs,
        size_t n,
        cudaStream_t lhs_stream,
        cudaStream_t rhs_stream,
        int device,
        bool &is_equal)
    {
        if (n == 0)
        {
            is_equal = true;
            return 0;
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }

        err = cudaStreamSynchronize(lhs_stream);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(rhs_stream);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }

        int *d_equal = nullptr;
        err = cudaMalloc(&d_equal, sizeof(int));
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }

        int h_equal = 1;
        err = cudaMemcpyAsync(d_equal, &h_equal, sizeof(int), cudaMemcpyHostToDevice, lhs_stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_equal);
            return set_error(cudaGetErrorString(err));
        }

        const int threads = 256;
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        compare_u64_kernel<<<blocks, threads, 0, lhs_stream>>>(lhs, rhs, n, d_equal);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFree(d_equal);
            return set_error(cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(&h_equal, d_equal, sizeof(int), cudaMemcpyDeviceToHost, lhs_stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_equal);
            return set_error(cudaGetErrorString(err));
        }

        err = cudaStreamSynchronize(lhs_stream);
        if (err != cudaSuccess)
        {
            cudaFree(d_equal);
            return set_error(cudaGetErrorString(err));
        }
        cudaFree(d_equal);
        is_equal = h_equal != 0;
        return 0;
    }
}

extern "C"
{
    int gpu_poly_add(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_add arguments");
            }

            const int batch = default_batch(a->ctx);
            const PolyFormat desired_format = a->format;
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != desired_format)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, desired_format, batch);
                b_poly = b_tmp.get();
            }

            if (out == a && out->level <= b->level && out->format == desired_format)
            {
                out->poly->add(*b_poly);
            }
            else if (out->level == a->level && out->level <= b->level && out->format == desired_format)
            {
                out->poly->add(*a->poly, *b_poly);
            }
            else
            {
                out->poly->copy(*a->poly);
                out->poly->add(*b_poly);
            }
            out->level = a->level;
            out->format = desired_format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_add");
        }
    }

    int gpu_poly_sub(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_sub arguments");
            }

            const int batch = default_batch(a->ctx);
            const PolyFormat desired_format = a->format;
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != desired_format)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, desired_format, batch);
                b_poly = b_tmp.get();
            }

            if (out == a && out->level <= b->level && out->format == desired_format)
            {
                out->poly->sub(*b_poly);
            }
            else
            {
                out->poly->copy(*a->poly);
                out->poly->sub(*b_poly);
            }
            out->level = a->level;
            out->format = desired_format;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_sub");
        }
    }

    int gpu_poly_mul(GpuPoly *out, const GpuPoly *a, const GpuPoly *b)
    {
        try
        {
            if (!out || !a || !b)
            {
                return set_error("invalid gpu_poly_mul arguments");
            }
            const int batch = default_batch(a->ctx);
            std::unique_ptr<CKKS::RNSPoly> b_tmp;
            const CKKS::RNSPoly *b_poly = b->poly;
            if (b->format != PolyFormat::Eval)
            {
                b_tmp = std::make_unique<CKKS::RNSPoly>(b->poly->clone());
                convert_format(*b_tmp, b->format, PolyFormat::Eval, batch);
                b_poly = b_tmp.get();
            }

            if (out != a)
            {
                int copy_status = gpu_poly_copy(out, a);
                if (copy_status != 0)
                {
                    return copy_status;
                }
            }
            ensure_eval(out, batch);
            out->poly->multElement(*b_poly);
            out->level = a->level;
            out->format = PolyFormat::Eval;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_mul");
        }
    }

    int gpu_poly_equal(const GpuPoly *lhs, const GpuPoly *rhs, int *out_equal)
    {
        try
        {
            if (!lhs || !rhs || !out_equal)
            {
                return set_error("invalid gpu_poly_equal arguments");
            }
            *out_equal = 0;

            if (lhs == rhs)
            {
                *out_equal = 1;
                return 0;
            }
            if (lhs->ctx != rhs->ctx || lhs->level != rhs->level)
            {
                return 0;
            }

            const int batch = default_batch(lhs->ctx);
            std::unique_ptr<CKKS::RNSPoly> rhs_tmp;
            const CKKS::RNSPoly *lhs_poly = lhs->poly;
            const CKKS::RNSPoly *rhs_poly = rhs->poly;
            if (lhs->format != rhs->format)
            {
                rhs_tmp = std::make_unique<CKKS::RNSPoly>(rhs->poly->clone());
                convert_format(*rhs_tmp, rhs->format, lhs->format, batch);
                rhs_poly = rhs_tmp.get();
            }

            auto *ctx = lhs->ctx->ctx;
            const int level = lhs->level;
            const int n = lhs->ctx->N;
            if (n < 0 || level < 0)
            {
                return set_error("invalid poly metadata in gpu_poly_equal");
            }
            if (ctx->limbGPUid.size() < static_cast<size_t>(level + 1))
            {
                return set_error("unexpected limb mapping size in gpu_poly_equal");
            }

            for (int limb = 0; limb <= level; ++limb)
            {
                const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                if (limb_id.x >= lhs_poly->GPU.size() || limb_id.x >= rhs_poly->GPU.size())
                {
                    return set_error("unexpected limb GPU partition in gpu_poly_equal");
                }

                auto &lhs_partition = lhs_poly->GPU[limb_id.x];
                auto &rhs_partition = rhs_poly->GPU[limb_id.x];
                if (limb_id.y >= lhs_partition.limb.size() || limb_id.y >= rhs_partition.limb.size())
                {
                    return set_error("unexpected limb index in gpu_poly_equal");
                }
                if (lhs_partition.device != rhs_partition.device)
                {
                    return set_error("device mismatch in gpu_poly_equal");
                }

                auto &lhs_limb = lhs_partition.limb[limb_id.y];
                auto &rhs_limb = rhs_partition.limb[limb_id.y];
                if (lhs_limb.index() != FIDESlib::U64 || rhs_limb.index() != FIDESlib::U64)
                {
                    return set_error("unsupported limb type in gpu_poly_equal");
                }

                auto &lhs_u64 = std::get<FIDESlib::U64>(lhs_limb);
                auto &rhs_u64 = std::get<FIDESlib::U64>(rhs_limb);
                bool limb_equal = true;
                int status = compare_u64_arrays_on_device(
                    lhs_u64.v.data,
                    rhs_u64.v.data,
                    static_cast<size_t>(n),
                    lhs_u64.stream.ptr,
                    rhs_u64.stream.ptr,
                    lhs_partition.device,
                    limb_equal);
                if (status != 0)
                {
                    return status;
                }
                if (!limb_equal)
                {
                    *out_equal = 0;
                    return 0;
                }
            }

            *out_equal = 1;
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_equal");
        }
    }

    int gpu_poly_decompose_base(
        const GpuPoly *src,
        uint32_t base_bits,
        GpuPoly *const *out_polys,
        size_t out_count)
    {
        try
        {
            if (!src || !out_polys)
            {
                return set_error("invalid gpu_poly_decompose_base arguments");
            }
            if (out_count == 0)
            {
                return 0;
            }
            if (base_bits == 0)
            {
                return set_error("base_bits must be non-zero in gpu_poly_decompose_base");
            }

            const int level = src->level;
            const int N = src->ctx->N;
            const size_t crt_depth = static_cast<size_t>(level + 1);
            uint32_t crt_bits = 0;
            for (const auto &modulus : src->ctx->moduli)
            {
                crt_bits = std::max(crt_bits, bit_width_u64(modulus));
            }
            if (crt_bits == 0)
            {
                return set_error("invalid crt_bits in gpu_poly_decompose_base");
            }
            const uint32_t digits_per_tower =
                static_cast<uint32_t>((crt_bits + base_bits - 1) / base_bits);
            if (digits_per_tower == 0)
            {
                return set_error("invalid digits_per_tower in gpu_poly_decompose_base");
            }
            const size_t expected_out = static_cast<size_t>(digits_per_tower) * crt_depth;
            if (out_count != expected_out)
            {
                return set_error("output size mismatch in gpu_poly_decompose_base");
            }

            const int batch = default_batch(src->ctx);
            const GpuPoly *input = src;
            std::unique_ptr<CKKS::RNSPoly> tmp_poly;
            GpuPoly tmp_wrapper{nullptr, nullptr, 0, PolyFormat::Coeff};
            if (src->format == PolyFormat::Eval)
            {
                tmp_poly = std::make_unique<CKKS::RNSPoly>(src->poly->clone());
                tmp_wrapper = GpuPoly{tmp_poly.get(), src->ctx, src->level, src->format};
                ensure_coeff(&tmp_wrapper, batch);
                input = &tmp_wrapper;
            }
            if (input->format != PolyFormat::Coeff)
            {
                return set_error("input poly must be in coeff format for decomposition");
            }
            input->poly->sync();

            auto *ctx = input->ctx->ctx;
            if (ctx->limbGPUid.size() < crt_depth)
            {
                return set_error("unexpected limb mapping size in gpu_poly_decompose_base");
            }

            for (size_t idx = 0; idx < out_count; ++idx)
            {
                GpuPoly *out = out_polys[idx];
                if (!out || !out->ctx)
                {
                    return set_error("null output poly in gpu_poly_decompose_base");
                }
                if (out->ctx != input->ctx || out->level != level)
                {
                    return set_error("mismatched output poly in gpu_poly_decompose_base");
                }

                for (int limb = 0; limb <= level; ++limb)
                {
                    const dim3 limb_id = ctx->limbGPUid[static_cast<size_t>(limb)];
                    if (limb_id.x >= out->poly->GPU.size())
                    {
                        return set_error("unexpected limb GPU partition in gpu_poly_decompose_base");
                    }
                    auto &partition = out->poly->GPU[limb_id.x];
                    if (limb_id.y >= partition.limb.size())
                    {
                        return set_error("unexpected limb index in gpu_poly_decompose_base");
                    }
                    auto &limb_impl = partition.limb[limb_id.y];
                    if (limb_impl.index() != FIDESlib::U64)
                    {
                        return set_error("unsupported limb type in gpu_poly_decompose_base");
                    }
                    auto &limb_u64 = std::get<FIDESlib::U64>(limb_impl);

                    cudaError_t err = cudaSetDevice(partition.device);
                    if (err != cudaSuccess)
                    {
                        return set_error(cudaGetErrorString(err));
                    }
                    err = cudaMemsetAsync(
                        limb_u64.v.data,
                        0,
                        static_cast<size_t>(N) * sizeof(uint64_t),
                        limb_u64.stream.ptr);
                    if (err != cudaSuccess)
                    {
                        return set_error(cudaGetErrorString(err));
                    }
                }

                out->format = PolyFormat::Coeff;
                out->level = level;
            }

            const int threads = 256;
            const int blocks = static_cast<int>((static_cast<size_t>(N) + threads - 1) / threads);
            if (input->ctx->moduli.size() < crt_depth)
            {
                return set_error("unexpected modulus count in gpu_poly_decompose_base");
            }

            for (int src_limb = 0; src_limb <= level; ++src_limb)
            {
                const dim3 src_limb_id = ctx->limbGPUid[static_cast<size_t>(src_limb)];
                if (src_limb_id.x >= input->poly->GPU.size())
                {
                    return set_error("unexpected source limb GPU partition in gpu_poly_decompose_base");
                }
                auto &in_partition = input->poly->GPU[src_limb_id.x];
                if (src_limb_id.y >= in_partition.limb.size())
                {
                    return set_error("unexpected source limb index in gpu_poly_decompose_base");
                }
                auto &in_limb_impl = in_partition.limb[src_limb_id.y];
                if (in_limb_impl.index() != FIDESlib::U64)
                {
                    return set_error("unsupported source limb type in gpu_poly_decompose_base");
                }
                auto &in_limb_u64 = std::get<FIDESlib::U64>(in_limb_impl);
                const uint32_t src_bits =
                    bit_width_u64(input->ctx->moduli[static_cast<size_t>(src_limb)]);

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

                    const size_t out_idx =
                        static_cast<size_t>(src_limb) * static_cast<size_t>(digits_per_tower) +
                        static_cast<size_t>(digit_idx);
                    GpuPoly *out = out_polys[out_idx];
                    for (int out_limb = 0; out_limb <= level; ++out_limb)
                    {
                        const dim3 out_limb_id = ctx->limbGPUid[static_cast<size_t>(out_limb)];
                        auto &out_partition = out->poly->GPU[out_limb_id.x];
                        if (out_limb_id.y >= out_partition.limb.size())
                        {
                            return set_error("unexpected output limb index in gpu_poly_decompose_base");
                        }
                        auto &out_limb_impl = out_partition.limb[out_limb_id.y];
                        if (out_limb_impl.index() != FIDESlib::U64)
                        {
                            return set_error("unsupported output limb type in gpu_poly_decompose_base");
                        }
                        auto &out_limb_u64 = std::get<FIDESlib::U64>(out_limb_impl);

                        if (out_partition.device != in_partition.device)
                        {
                            return set_error("input/output limb device mismatch in gpu_poly_decompose_base");
                        }

                        cudaError_t err = cudaSetDevice(out_partition.device);
                        if (err != cudaSuccess)
                        {
                            return set_error(cudaGetErrorString(err));
                        }

                        decompose_base_kernel<<<blocks, threads, 0, out_limb_u64.stream.ptr>>>(
                            in_limb_u64.v.data,
                            out_limb_u64.v.data,
                            static_cast<size_t>(N),
                            shift,
                            mask,
                            input->ctx->moduli[static_cast<size_t>(out_limb)]);
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            return set_error(cudaGetErrorString(err));
                        }
                    }
                }
            }

            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_decompose_base");
        }
    }

    int gpu_poly_ntt(GpuPoly *poly, int batch)
    {
        try
        {
            if (!poly)
            {
                return set_error("invalid gpu_poly_ntt arguments");
            }
            ensure_eval(poly, batch);
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_ntt");
        }
    }

    int gpu_poly_intt(GpuPoly *poly, int batch)
    {
        try
        {
            if (!poly)
            {
                return set_error("invalid gpu_poly_intt arguments");
            }
            ensure_coeff(poly, batch);
            return 0;
        }
        catch (const std::exception &e)
        {
            return set_error(e);
        }
        catch (...)
        {
            return set_error("unknown exception in gpu_poly_intt");
        }
    }
} // extern "C"
