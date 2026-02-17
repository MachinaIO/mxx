extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    *out_events = nullptr;
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
    if (count == 0)
    {
        return 0;
    }

    auto *event_set = new GpuEventSet();
    struct StreamRef
    {
        int device;
        cudaStream_t stream;
    };
    std::vector<StreamRef> streams;
    streams.reserve(mat->ctx ? mat->ctx->gpu_ids.size() : 1);

    for (size_t idx = 0; idx < count; ++idx)
    {
        GpuPoly *poly = mat->polys[idx];
        if (!poly || !poly->poly)
        {
            gpu_event_set_destroy(event_set);
            return set_error("null poly in gpu_matrix_load_rns_batch");
        }
        for (auto &partition : poly->poly->GPU)
        {
            for (auto &limb_impl : partition.limb)
            {
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
                    gpu_event_set_destroy(event_set);
                    return set_error("unsupported limb type in gpu_matrix_load_rns_batch");
                }
                if (!stream)
                {
                    gpu_event_set_destroy(event_set);
                    return set_error("null stream in gpu_matrix_load_rns_batch");
                }
                bool seen = false;
                for (const auto &entry : streams)
                {
                    if (entry.device == partition.device && entry.stream == stream)
                    {
                        seen = true;
                        break;
                    }
                }
                if (!seen)
                {
                    streams.push_back(StreamRef{partition.device, stream});
                }
            }
        }
    }

    event_set->entries.reserve(streams.size());
    for (const auto &entry : streams)
    {
        cudaError_t err = cudaSetDevice(entry.device);
        if (err != cudaSuccess)
        {
            gpu_event_set_destroy(event_set);
            return set_error(cudaGetErrorString(err));
        }

        cudaEvent_t ev = nullptr;
        err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            gpu_event_set_destroy(event_set);
            return set_error(cudaGetErrorString(err));
        }
        err = cudaEventRecord(ev, entry.stream);
        if (err != cudaSuccess)
        {
            cudaEventDestroy(ev);
            gpu_event_set_destroy(event_set);
            return set_error(cudaGetErrorString(err));
        }
        event_set->entries.push_back(GpuEventSet::Entry{ev, entry.device});
    }

    *out_events = event_set;
    return 0;
}

extern "C" int gpu_matrix_store_rns_batch(
    const GpuMatrix *mat,
    uint8_t *bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    *out_events = nullptr;
    const size_t count = mat->rows * mat->cols;
    if (count == 0)
    {
        return 0;
    }

    const int matrix_format =
        mat->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
    GpuMatrix *clones = nullptr;
    int status = gpu_matrix_create(
        mat->ctx,
        mat->level,
        mat->rows,
        mat->cols,
        matrix_format,
        &clones);
    if (status != 0)
    {
        return status;
    }
    status = gpu_matrix_copy(clones, mat);
    if (status != 0)
    {
        gpu_matrix_destroy(clones);
        return status;
    }

    status = gpu_poly_store_rns_batch(
        clones->polys.data(),
        count,
        bytes_out,
        bytes_per_poly,
        format,
        out_events);
    if (status != 0)
    {
        if (*out_events)
        {
            gpu_event_set_destroy(*out_events);
            *out_events = nullptr;
        }
        gpu_matrix_destroy(clones);
        return status;
    }

    if (*out_events)
    {
        status = gpu_event_set_wait(*out_events);
        gpu_event_set_destroy(*out_events);
        *out_events = nullptr;
        if (status != 0)
        {
            gpu_matrix_destroy(clones);
            return status;
        }
    }

    gpu_matrix_destroy(clones);
    return 0;
}
