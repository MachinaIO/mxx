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
    std::vector<GpuPoly *> mat_polys;
    int status = materialize_matrix_poly_views(mat, mat_polys);
    if (status != 0)
    {
        return status;
    }
    status = gpu_poly_load_rns_batch(
        mat_polys.data(),
        count,
        bytes,
        bytes_per_poly,
        format);
    release_poly_views(mat_polys);
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

    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < static_cast<size_t>(mat->level + 1))
    {
        gpu_event_set_destroy(event_set);
        return set_error("unexpected limb mapping size in gpu_matrix_load_rns_batch");
    }
    for (int limb = 0; limb <= mat->level; ++limb)
    {
        const dim3 limb_id = limb_map[static_cast<size_t>(limb)];
        int device = -1;
        cudaStream_t stream = nullptr;
        status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            gpu_event_set_destroy(event_set);
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            gpu_event_set_destroy(event_set);
            return status;
        }
        bool seen = false;
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                seen = true;
                break;
            }
        }
        if (!seen)
        {
            streams.push_back(StreamRef{device, stream});
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

    std::vector<GpuPoly *> clone_polys;
    status = materialize_matrix_poly_views(clones, clone_polys);
    if (status != 0)
    {
        gpu_matrix_destroy(clones);
        return status;
    }
    status = gpu_poly_store_rns_batch(
        clone_polys.data(),
        count,
        bytes_out,
        bytes_per_poly,
        format,
        out_events);
    release_poly_views(clone_polys);
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
