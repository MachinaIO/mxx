#include "gpu_compact_decompose.cuh"

struct GpuPreparedCompactDecompose {
    GpuMatrix *source;
    GpuSmallMatrix *output;
    GpuMatrixTransformPlan *inverse = nullptr;
    MatrixTransformWorkspace correction;
    CompactRowBlocks blocks{};
    const uint64_t *moduli;
    size_t limbs, retained, dropped, digits, slots, rows, columns, polynomial_count;
    size_t launch_blocks, correction_blocks;
    uint32_t base_bits;
    bool small;
    ~GpuPreparedCompactDecompose() {
        if (inverse) gpu_matrix_destroy_ntt_plan(inverse);
    }
};

extern "C" int gpu_small_matrix_prepare_decompose(
    GpuMatrix *source, GpuSmallMatrix *output, uint32_t base_bits, bool small,
    size_t dropped_moduli, GpuPreparedCompactDecompose **out)
try {
    if (!out || !source || !output || source->ctx != output->ctx ||
        !source->ctx || source->level < 0 || base_bits == 0 || base_bits >= 63 ||
        !output->owns_payload || source->cols != output->cols)
        return set_error("invalid prepared compact decomposition binding");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedCompactDecompose>();
    plan->source = source; plan->output = output;
    plan->limbs = static_cast<size_t>(source->level + 1);
    if (dropped_moduli >= plan->limbs)
        return set_error("invalid prepared compact decomposition drop count");
    plan->dropped = small ? 0 : dropped_moduli; plan->retained = plan->limbs - dropped_moduli;
    plan->base_bits = base_bits; plan->small = small;
    uint32_t crt_bits = 0;
    size_t partition = SIZE_MAX;
    for (size_t limb = 0; limb < plan->limbs; ++limb) {
        const auto id = source->ctx->limb_gpu_ids[limb];
        int device = -1;
        if (matrix_limb_device(source, id, &device) != 0 || device != output->device ||
            id.y != limb || (limb && id.x != partition))
            return set_error("prepared compact decomposition requires ordered local limbs");
        partition = id.x;
        crt_bits = std::max(crt_bits, bit_width_u64(source->ctx->moduli[limb]));
    }
    plan->digits = (crt_bits + base_bits - 1) / base_bits;
    plan->slots = plan->digits * (small ? 1 : plan->retained);
    plan->rows = source->rows; plan->columns = source->cols;
    const uint64_t base = uint64_t{1} << base_bits;
    const uint64_t bound = small ? base - 1 : (base + 1) / 2;
    size_t expected_rows = 0, coefficient_count = 0;
    if (!small_mul_size(plan->rows, plan->slots, &expected_rows) ||
        output->rows != expected_rows || output->bound_words.size() != 1 ||
        output->bound_words[0] != bound ||
        !small_mul_size(plan->rows, plan->columns, &plan->polynomial_count) ||
        !small_mul_size(expected_rows, plan->columns, &coefficient_count) ||
        !small_mul_size(coefficient_count, output->n, &coefficient_count))
        return set_error("prepared compact decomposition shape or bound mismatch");
    plan->launch_blocks = coefficient_count / kSmallThreads + (coefficient_count % kSmallThreads != 0);
    if (plan->launch_blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
        return set_error("prepared compact decomposition exceeds launch capacity");
    plan->blocks.inputs[0] = source->shared_limb_buffers[partition].device_descriptors;
    plan->blocks.input_offsets[0] = 0;
    plan->blocks.input_pitches[0] = source->cols;
    plan->blocks.ends[0] = source->rows;
    plan->moduli = source->ctx->ring_device_constants[partition].moduli;
    if (!plan->blocks.inputs[0] || !plan->moduli)
        return set_error("prepared compact decomposition descriptors missing");
    if (source->format == GPU_POLY_FORMAT_EVAL) {
        const int status = gpu_matrix_prepare_ntt_plan(source, nullptr, false, &plan->inverse);
        if (status != 0) return status;
    }
    if (plan->dropped) {
        GpuMatrixTransformWorkspaceBytes bytes{};
        int status = gpu_matrix_query_gadget_correction_workspace_bytes(source->ctx,
            source->level, source->rows, source->cols, dropped_moduli, &bytes);
        if (status != 0) return status;
        status = plan->correction.acquire_persistent(source, output->device, output->stream, bytes);
        if (status != 0) return status;
        const size_t count = plan->polynomial_count * output->n;
        plan->correction_blocks = count / 256 + (count % 256 != 0);
        const size_t entries = dropped_moduli * (plan->retained + 1);
        // Basis-only weights are setup data, never recomputed on replay.
        gpu_test_record_kernel_launch();
        gadget_low_constants_kernel<<<(entries + 255) / 256, 256, 0, output->stream>>>(
            plan->moduli, reinterpret_cast<uint64_t *>(plan->correction.base),
            plan->retained, dropped_moduli);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        status = plan->correction.complete();
        if (status != 0) return status;
    }
    *out = plan.release();
    return 0;
}
catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_small_matrix_submit_decompose(const GpuPreparedCompactDecompose *opaque)
{
    auto &plan = *const_cast<GpuPreparedCompactDecompose *>(opaque);
    auto *out = plan.output;
    cudaError_t error = cudaSetDevice(out->device);
    if (error != cudaSuccess) return set_error(error);
    if (plan.inverse) {
        const int status = gpu_matrix_submit_ntt_plan(plan.inverse, plan.source);
        if (status != 0) return status;
    }
    int status = matrix_wait_all_limb_streams(plan.source, out->device, out->stream, true, true);
    if (status != 0) return status;
    status = small_wait(out, out->stream);
    if (status != 0) return status;
    if (plan.dropped) {
        status = plan.correction.begin_replay();
        if (status != 0) return status;
        const dim3 grid(static_cast<unsigned int>(plan.correction_blocks),
            static_cast<unsigned int>(plan.retained));
        gpu_test_record_kernel_launch();
        gadget_correct_residues_kernel<<<grid, 256, 0, out->stream>>>(
            plan.blocks.inputs[0], plan.moduli,
            reinterpret_cast<const uint64_t *>(plan.correction.base),
            plan.retained, plan.dropped, out->n, plan.polynomial_count);
        error = cudaGetLastError();
        if (error != cudaSuccess) return set_error(error);
        status = plan.correction.complete();
        if (status != 0) return status;
    }
    gpu_test_record_kernel_launch();
    compact_decompose_kernel<<<static_cast<unsigned int>(plan.launch_blocks), kSmallThreads, 0, out->stream>>>(
        plan.blocks, plan.moduli, out->payload, plan.rows, plan.columns, plan.slots,
        0, out->storage_cols, out->n, plan.digits, out->magnitude_bytes,
        plan.base_bits, !plan.small, plan.small);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    status = small_record(out, out->stream);
    if (status != 0) return status;
    return matrix_track_all_limb_consumers(plan.source, out->device, out->stream,
        out->write_done, true, true);
}

extern "C" void gpu_small_matrix_destroy_decompose(GpuPreparedCompactDecompose *plan) { delete plan; }
