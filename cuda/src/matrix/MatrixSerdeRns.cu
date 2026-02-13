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
    if (!mat)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    const size_t count = mat->rows * mat->cols;
    return gpu_poly_store_rns_batch(
        const_cast<GpuPoly *const *>(mat->polys.data()),
        count,
        bytes_out,
        bytes_per_poly,
        format,
        out_events);
}
