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
