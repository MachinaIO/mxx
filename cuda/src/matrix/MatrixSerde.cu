#include "matrix/MatrixSerde.cuh"

extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format)
{
    if (!mat || (!bytes && mat->rows * mat->cols > 0))
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }

    PolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }

    const size_t count = mat->rows * mat->cols;
    if (count == 0)
    {
        mat->format = target_format;
        return 0;
    }

    const int level = mat->level;
    const int N = mat->ctx ? mat->ctx->N : 0;
    if (!mat->ctx || level < 0 || N <= 0)
    {
        return set_error("invalid matrix state in gpu_matrix_load_rns_batch");
    }
    const size_t expected_bytes =
        static_cast<size_t>(level + 1) * static_cast<size_t>(N) * sizeof(uint64_t);
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_load_rns_batch");
    }

    for (int limb = 0; limb <= level; ++limb)
    {
        const GpuMatrix::LimbU64Table *table = nullptr;
        bool has_u64 = false;
        int status = try_get_matrix_u64_limb_table(mat, limb, &table, &has_u64);
        if (status != 0)
        {
            return status;
        }
        if (!has_u64 || !table || !table->stream)
        {
            return set_error("non-u64 limb is unsupported in gpu_matrix_load_rns_batch");
        }

        cudaError_t err = cudaSetDevice(table->device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        const size_t limb_offset = static_cast<size_t>(limb) * static_cast<size_t>(N) * sizeof(uint64_t);
        const size_t limb_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
        for (size_t idx = 0; idx < count; ++idx)
        {
            const uint8_t *src = bytes + idx * bytes_per_poly + limb_offset;
            err = cudaMemcpyAsync(
                table->entry_ptrs[idx],
                src,
                limb_bytes,
                cudaMemcpyHostToDevice,
                table->stream);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
        }
        err = cudaStreamSynchronize(table->stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
    }

    mat->format = target_format;
    for (auto *poly : mat->polys)
    {
        if (!poly)
        {
            continue;
        }
        poly->level = mat->level;
        poly->format = target_format;
    }
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
    if (!bytes_out)
    {
        return set_error("invalid bytes_out in gpu_matrix_store_rns_batch");
    }

    PolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_store_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (!mat->ctx || mat->level < 0 || mat->ctx->N <= 0)
    {
        return set_error("invalid matrix state in gpu_matrix_store_rns_batch");
    }
    const int N = mat->ctx->N;
    const size_t expected_bytes =
        static_cast<size_t>(mat->level + 1) * static_cast<size_t>(N) * sizeof(uint64_t);
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_store_rns_batch");
    }

    const int matrix_format =
        mat->format == PolyFormat::Eval ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
    GpuMatrix *clones = nullptr;
    const GpuMatrix *store_mat = mat;
    auto cleanup = [&]()
    {
        if (clones)
        {
            gpu_matrix_destroy(clones);
            clones = nullptr;
        }
    };

    if (mat->format != target_format)
    {
        int status = gpu_matrix_create(
            mat->ctx,
            mat->level,
            mat->rows,
            mat->cols,
            matrix_format,
            &clones);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = gpu_matrix_copy(clones, mat);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        status = transform_matrix_format_sync(
            clones,
            target_format,
            "failed to transform matrix in gpu_matrix_store_rns_batch");
        if (status != 0)
        {
            cleanup();
            return status;
        }
        store_mat = clones;
    }

    for (int limb = 0; limb <= store_mat->level; ++limb)
    {
        const GpuMatrix::LimbU64Table *table = nullptr;
        bool has_u64 = false;
        int status = try_get_matrix_u64_limb_table(store_mat, limb, &table, &has_u64);
        if (status != 0)
        {
            cleanup();
            return status;
        }
        if (!has_u64 || !table || !table->stream)
        {
            cleanup();
            return set_error("non-u64 limb is unsupported in gpu_matrix_store_rns_batch");
        }

        cudaError_t err = cudaSetDevice(table->device);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }

        const size_t limb_offset = static_cast<size_t>(limb) * static_cast<size_t>(N) * sizeof(uint64_t);
        const size_t limb_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
        for (size_t idx = 0; idx < count; ++idx)
        {
            uint8_t *dst = bytes_out + idx * bytes_per_poly + limb_offset;
            err = cudaMemcpyAsync(
                dst,
                table->entry_ptrs[idx],
                limb_bytes,
                cudaMemcpyDeviceToHost,
                table->stream);
            if (err != cudaSuccess)
            {
                cleanup();
                return set_error(err);
            }
        }
        err = cudaStreamSynchronize(table->stream);
        if (err != cudaSuccess)
        {
            cleanup();
            return set_error(err);
        }
    }

    cleanup();
    return 0;
}
