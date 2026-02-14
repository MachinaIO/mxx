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

    std::vector<GpuPoly *> clones;
    clones.reserve(count);

    auto cleanup_clones = [&clones]() {
        for (auto *poly : clones)
        {
            gpu_poly_destroy(poly);
        }
        clones.clear();
    };

    for (size_t i = 0; i < count; ++i)
    {
        GpuPoly *clone = nullptr;
        int status = gpu_poly_clone(mat->polys[i], &clone);
        if (status != 0)
        {
            cleanup_clones();
            return status;
        }
        clones.push_back(clone);
    }

    int status = gpu_poly_store_rns_batch(
        clones.data(),
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
        cleanup_clones();
        return status;
    }

    if (*out_events)
    {
        status = gpu_event_set_wait(*out_events);
        gpu_event_set_destroy(*out_events);
        *out_events = nullptr;
        if (status != 0)
        {
            cleanup_clones();
            return status;
        }
    }

    cleanup_clones();
    return 0;
}
