namespace
{
    struct TransformScratch
    {
        int device;
        cudaStream_t stream;
        uint64_t *aux;
    };

    int get_or_create_transform_scratch(
        std::vector<TransformScratch> &scratch_list,
        int device,
        cudaStream_t stream,
        size_t n,
        uint64_t **out_aux)
    {
        if (!out_aux || n == 0)
        {
            return set_error("invalid scratch request in get_or_create_transform_scratch");
        }
        for (auto &entry : scratch_list)
        {
            if (entry.device == device && entry.stream == stream)
            {
                *out_aux = entry.aux;
                return 0;
            }
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        uint64_t *aux = nullptr;
        err = cudaMallocAsync(&aux, n * sizeof(uint64_t), stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        scratch_list.push_back(TransformScratch{device, stream, aux});
        *out_aux = aux;
        return 0;
    }

    int free_transform_scratch(std::vector<TransformScratch> &scratch_list)
    {
        for (const auto &entry : scratch_list)
        {
            if (!entry.aux)
            {
                continue;
            }
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaFreeAsync(entry.aux, entry.stream);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        scratch_list.clear();
        return 0;
    }

    template <bool Forward>
    int run_matrix_transform_u64(GpuMatrix *mat)
    {
        if (!mat || !mat->ctx || !mat->ctx->ctx)
        {
            return set_error("invalid matrix in run_matrix_transform_u64");
        }
        if (mat->level < 0)
        {
            return set_error("invalid level in run_matrix_transform_u64");
        }
        if (mat->ctx->N <= 0)
        {
            return set_error("invalid ring dimension in run_matrix_transform_u64");
        }
        const int log_n = mat->ctx->ctx->logN;
        if (log_n < 2)
        {
            return set_error("invalid logN in run_matrix_transform_u64");
        }

        auto &limb_map = mat->ctx->ctx->limbGPUid;
        if (limb_map.size() < static_cast<size_t>(mat->level + 1))
        {
            return set_error("unexpected limb mapping size in run_matrix_transform_u64");
        }

        const size_t poly_count = matrix_poly_count(mat);
        if (poly_count == 0)
        {
            mat->format = Forward ? PolyFormat::Eval : PolyFormat::Coeff;
            return 0;
        }

        constexpr int kM = 4;
        constexpr FIDESlib::ALGO kAlgo = FIDESlib::ALGO_SHOUP;
        const int algo_id = static_cast<int>(kAlgo);
        const int shoup_extra = (algo_id == 2 || algo_id == 3) ? 1 : 0;

        const int first_exp = Forward ? ((log_n + 1) / 2 - 1) : (log_n / 2 - 1);
        const int second_exp = Forward ? (log_n / 2 - 1) : ((log_n + 1) / 2 - 1);
        if (first_exp < 0 || second_exp < 0)
        {
            return set_error("invalid NTT exponent in run_matrix_transform_u64");
        }

        const uint32_t block_first = static_cast<uint32_t>(1U << first_exp);
        const uint32_t block_second = static_cast<uint32_t>(1U << second_exp);
        if (block_first == 0 || block_second == 0)
        {
            return set_error("invalid NTT block size in run_matrix_transform_u64");
        }

        const size_t n = static_cast<size_t>(mat->ctx->N);
        const size_t grid_first_x = n / (static_cast<size_t>(block_first) * kM * 2);
        const size_t grid_second_x = n / (static_cast<size_t>(block_second) * kM * 2);
        if (grid_first_x == 0 || grid_second_x == 0)
        {
            return set_error("invalid NTT grid size in run_matrix_transform_u64");
        }

        const dim3 grid_first{static_cast<uint32_t>(grid_first_x)};
        const dim3 grid_second{static_cast<uint32_t>(grid_second_x)};
        const int bytes_first =
            static_cast<int>(sizeof(uint64_t) * block_first * (2 * kM + 1 + shoup_extra));
        const int bytes_second =
            static_cast<int>(sizeof(uint64_t) * block_second * (2 * kM + 1 + shoup_extra));

        std::vector<TransformScratch> scratch_list;
        auto cleanup = [&]() -> int { return free_transform_scratch(scratch_list); };

        for (int limb = 0; limb <= mat->level; ++limb)
        {
            const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
            int device = -1;
            int status = matrix_limb_device(mat, limb_id, &device);
            if (status != 0)
            {
                int cleanup_status = cleanup();
                return cleanup_status != 0 ? cleanup_status : status;
            }
            cudaStream_t stream = nullptr;
            status = matrix_limb_stream(mat, limb_id, &stream);
            if (status != 0)
            {
                int cleanup_status = cleanup();
                return cleanup_status != 0 ? cleanup_status : status;
            }

            if (limb_id.x >= mat->ctx->ctx->meta.size() ||
                limb_id.y >= mat->ctx->ctx->meta[limb_id.x].size())
            {
                int cleanup_status = cleanup();
                return cleanup_status != 0 ? cleanup_status
                                           : set_error("invalid limb metadata in run_matrix_transform_u64");
            }
            const auto &record = mat->ctx->ctx->meta[limb_id.x][limb_id.y];
            if (record.type != FIDESlib::U64)
            {
                int cleanup_status = cleanup();
                return cleanup_status != 0 ? cleanup_status
                                           : set_error("unsupported limb type in run_matrix_transform_u64");
            }
            const int primeid = record.id;

            uint64_t *aux = nullptr;
            status = get_or_create_transform_scratch(scratch_list, device, stream, n, &aux);
            if (status != 0)
            {
                int cleanup_status = cleanup();
                return cleanup_status != 0 ? cleanup_status : status;
            }

            for (size_t poly_idx = 0; poly_idx < poly_count; ++poly_idx)
            {
                uint64_t *dat = matrix_limb_ptr_by_id(mat, poly_idx, limb_id);
                if (!dat)
                {
                    int cleanup_status = cleanup();
                    return cleanup_status != 0 ? cleanup_status
                                               : set_error("null matrix limb pointer in run_matrix_transform_u64");
                }

                if constexpr (Forward)
                {
                    FIDESlib::NTT_<uint64_t, false, FIDESlib::ALGO_SHOUP, FIDESlib::NTT_NONE>
                        <<<grid_first, block_first, bytes_first, stream>>>(dat, primeid, aux, nullptr);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        int cleanup_status = cleanup();
                        return cleanup_status != 0 ? cleanup_status : set_error(err);
                    }
                    FIDESlib::NTT_<uint64_t, true, FIDESlib::ALGO_SHOUP, FIDESlib::NTT_NONE>
                        <<<grid_second, block_second, bytes_second, stream>>>(aux, primeid, dat, nullptr);
                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        int cleanup_status = cleanup();
                        return cleanup_status != 0 ? cleanup_status : set_error(err);
                    }
                }
                else
                {
                    FIDESlib::INTT_<uint64_t, false, FIDESlib::ALGO_SHOUP, FIDESlib::INTT_NONE>
                        <<<grid_first, block_first, bytes_first, stream>>>(dat, primeid, aux);
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        int cleanup_status = cleanup();
                        return cleanup_status != 0 ? cleanup_status : set_error(err);
                    }
                    FIDESlib::INTT_<uint64_t, true, FIDESlib::ALGO_SHOUP, FIDESlib::INTT_NONE>
                        <<<grid_second, block_second, bytes_second, stream>>>(aux, primeid, dat);
                    err = cudaGetLastError();
                    if (err != cudaSuccess)
                    {
                        int cleanup_status = cleanup();
                        return cleanup_status != 0 ? cleanup_status : set_error(err);
                    }
                }
            }
        }

        int cleanup_status = cleanup();
        if (cleanup_status != 0)
        {
            return cleanup_status;
        }
        mat->format = Forward ? PolyFormat::Eval : PolyFormat::Coeff;
        return 0;
    }
}

int gpu_matrix_ntt_all(GpuMatrix *mat, int batch)
{
    (void)batch;
    if (!mat || !mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid gpu_matrix_ntt_all arguments");
    }
    if (mat->format == PolyFormat::Eval)
    {
        return 0;
    }
    return run_matrix_transform_u64<true>(mat);
}

int gpu_matrix_intt_all(GpuMatrix *mat, int batch)
{
    (void)batch;
    if (!mat || !mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid gpu_matrix_intt_all arguments");
    }
    if (mat->format == PolyFormat::Coeff)
    {
        return 0;
    }
    return run_matrix_transform_u64<false>(mat);
}
