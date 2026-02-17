namespace
{
    struct SerdeStreamRef
    {
        int device;
        cudaStream_t stream;
    };

    bool serde_checked_mul_size(size_t a, size_t b, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (a != 0 && b > static_cast<size_t>(-1) / a)
        {
            return false;
        }
        *out = a * b;
        return true;
    }

    void serde_append_unique_stream(std::vector<SerdeStreamRef> &streams, int device, cudaStream_t stream)
    {
        if (!stream)
        {
            return;
        }
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                return;
            }
        }
        streams.push_back(SerdeStreamRef{device, stream});
    }

    void serde_destroy_event_set(GpuEventSet *events)
    {
        if (!events)
        {
            return;
        }
        for (const auto &entry : events->entries)
        {
            cudaSetDevice(entry.device);
            cudaEventDestroy(entry.event);
        }
        delete events;
    }

    int serde_build_event_set_from_streams(const std::vector<SerdeStreamRef> &streams, GpuEventSet **out_events)
    {
        if (!out_events)
        {
            return set_error("invalid out_events in serde_build_event_set_from_streams");
        }
        *out_events = nullptr;
        if (streams.empty())
        {
            return 0;
        }

        auto *event_set = new GpuEventSet();
        event_set->entries.reserve(streams.size());
        for (const auto &entry : streams)
        {
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }

            cudaEvent_t ev = nullptr;
            err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err != cudaSuccess)
            {
                serde_destroy_event_set(event_set);
                return set_error(err);
            }
            err = cudaEventRecord(ev, entry.stream);
            if (err != cudaSuccess)
            {
                cudaEventDestroy(ev);
                serde_destroy_event_set(event_set);
                return set_error(err);
            }
            event_set->entries.push_back(GpuEventSet::Entry{ev, entry.device});
        }

        *out_events = event_set;
        return 0;
    }
}

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
    if (!mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid context in gpu_matrix_load_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_load_rns_batch");
    }
    if (count > 0 && !bytes)
    {
        return set_error("null bytes in gpu_matrix_load_rns_batch");
    }

    if (mat->level < 0)
    {
        mat->format = target_format;
        return count == 0 ? 0 : set_error("invalid level in gpu_matrix_load_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_load_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_load_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_load_rns_batch");
    }
    if (count == 0)
    {
        mat->format = target_format;
        return 0;
    }

    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_load_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t coeff_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint64_t *dst = matrix_limb_ptr_by_id(mat, 0, limb_id);
        if (!dst)
        {
            return set_error("null matrix limb base pointer in gpu_matrix_load_rns_batch");
        }

        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= mat->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_matrix_load_rns_batch");
        }
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const size_t dst_pitch = buffer.words_per_poly * sizeof(uint64_t);
        const uint8_t *src = bytes + limb * coeff_bytes;

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpy2DAsync(
            dst,
            dst_pitch,
            src,
            bytes_per_poly,
            coeff_bytes,
            count,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        serde_append_unique_stream(streams, device, stream);
    }

    mat->format = target_format;
    return serde_build_event_set_from_streams(streams, out_events);
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
    if (!mat->ctx || !mat->ctx->ctx)
    {
        return set_error("invalid context in gpu_matrix_store_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_store_rns_batch");
    }
    if (target_format != mat->format)
    {
        return set_error("format conversion is not supported in gpu_matrix_store_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_store_rns_batch");
    }
    if (count > 0 && !bytes_out)
    {
        return set_error("null bytes_out in gpu_matrix_store_rns_batch");
    }
    if (count == 0)
    {
        return 0;
    }

    if (mat->level < 0)
    {
        return set_error("invalid level in gpu_matrix_store_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_store_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_store_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_store_rns_batch");
    }

    auto &limb_map = mat->ctx->ctx->limbGPUid;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_store_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t coeff_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        const uint64_t *src = matrix_limb_ptr_by_id(mat, 0, limb_id);
        if (!src)
        {
            return set_error("null matrix limb base pointer in gpu_matrix_store_rns_batch");
        }

        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }
        if (limb_id.x >= mat->shared_limb_buffers.size())
        {
            return set_error("invalid partition index in gpu_matrix_store_rns_batch");
        }
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const size_t src_pitch = buffer.words_per_poly * sizeof(uint64_t);
        uint8_t *dst = bytes_out + limb * coeff_bytes;

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaMemcpy2DAsync(
            dst,
            bytes_per_poly,
            src,
            src_pitch,
            coeff_bytes,
            count,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        serde_append_unique_stream(streams, device, stream);
    }

    return serde_build_event_set_from_streams(streams, out_events);
}
