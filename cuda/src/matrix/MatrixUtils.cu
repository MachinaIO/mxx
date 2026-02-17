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

    std::vector<GpuPoly *> collect_poly_ptrs(GpuMatrix *mat)
    {
        std::vector<GpuPoly *> ptrs;
        if (!mat)
        {
            return ptrs;
        }
        ptrs.reserve(mat->polys.size());
        for (auto &poly : mat->polys)
        {
            ptrs.push_back(poly.get());
        }
        return ptrs;
    }

    std::vector<const GpuPoly *> collect_poly_const_ptrs(const GpuMatrix *mat)
    {
        std::vector<const GpuPoly *> ptrs;
        if (!mat)
        {
            return ptrs;
        }
        ptrs.reserve(mat->polys.size());
        for (const auto &poly : mat->polys)
        {
            ptrs.push_back(poly.get());
        }
        return ptrs;
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
