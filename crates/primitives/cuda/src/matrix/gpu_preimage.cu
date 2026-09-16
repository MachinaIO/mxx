// Included after MatrixTrapdoor.cu: these fixed plans reuse its exact sampling
// kernels and covariance calculation, with all workspaces retained by the slot.
#include "gpu_preimage.cuh"

extern "C" int gpu_preimage_phase_layout(GpuContext *ctx, size_t rows, size_t columns,
    GpuPreparedWorkspaceLayout *layouts)
{
    if (!ctx || ctx->N <= 0 || !rows || !columns || !layouts || rows > SIZE_MAX / 2 ||
        static_cast<size_t>(ctx->N) > SIZE_MAX / (2 * rows) / sizeof(double) ||
        ctx->N * (2 * rows) * sizeof(double) > SIZE_MAX / (2 * rows))
        return set_error("invalid prepared preimage phase layout");
    const size_t roots = ctx->N * (2 * rows) * sizeof(double);
    layouts[0] = {roots, alignof(double), GPU_PREPARED_SAMPLER_WORKSPACE};
    layouts[1] = layouts[2] = {roots * (2 * rows), alignof(double), GPU_PREPARED_SAMPLER_WORKSPACE};
    layouts[3] = {0, 1, GPU_PREPARED_COMPLETION_EVENT};
    return gpu_matrix_query_p1_workspaces(ctx, rows, columns, 1, layouts + 4);
}

struct GpuPreparedPreimagePhases {
    const GpuMatrix *gram[3]{};
    const GpuMatrix *product = nullptr;
    const GpuMatrix *residual = nullptr;
    GpuMatrix *p1 = nullptr;
    GpuMatrix *gadget = nullptr;
    GpuContext *ctx = nullptr;
    int device = -1;
    cudaStream_t stream = nullptr;
    dim3 reference{};
    size_t d = 0, n = 0, columns = 0, limbs = 0;
    uint32_t base_bits = 0, digits = 0;
    double c = 0, smoothing = 0, sigma = 0;
    const uint64_t *moduli = nullptr;
    const uint8_t *gram_bases[3]{};
    size_t gram_strides[3]{};
    uint8_t gram_widths[3]{};
    GpuDeviceWorkspace covariance, roots, updates, samples, extra;
    GpuCudaResource completion;
    GpuMatrixTransformPlan *p1_ntt = nullptr, *gadget_ntt = nullptr;
    P1BatchDescriptors p1_job{};
    GadgetBatchDescriptors gadget_job{};
    dim3 covariance_grid{}, p1_grid{}, scatter_grid{}, gadget_grid{};
    cudaEvent_t completed_event = nullptr;

    ~GpuPreparedPreimagePhases() {
        gpu_matrix_destroy_ntt_plan(p1_ntt);
        gpu_matrix_destroy_ntt_plan(gadget_ntt);
        for (auto *owner : {&covariance, &roots, &updates, &samples, &extra})
            (void)owner->release(stream);
    }
};

extern "C" int gpu_preimage_prepare_phases(
    const GpuMatrix *a, const GpuMatrix *b, const GpuMatrix *d,
    const GpuMatrix *product, GpuMatrix *p1, const GpuMatrix *residual, GpuMatrix *gadget,
    uint32_t base_bits, double c, double smoothing, double sigma,
    GpuPreparedPreimagePhases **out)
try {
    if (!a || !b || !d || !product || !p1 || !residual || !gadget || !out ||
        !a->rows || !p1->cols || !(c > 0) || !(smoothing > c) || !(sigma > 0) ||
        !base_bits || base_bits >= 63)
        return set_error("invalid prepared preimage phase arguments");
    *out = nullptr;
    auto plan = std::make_unique<GpuPreparedPreimagePhases>();
    plan->gram[0] = a; plan->gram[1] = b; plan->gram[2] = d;
    plan->product = product; plan->residual = residual;
    plan->p1 = p1; plan->gadget = gadget; plan->ctx = a->ctx;
    plan->d = a->rows; plan->n = a->ctx->N; plan->columns = p1->cols;
    plan->limbs = a->level + 1;
    plan->base_bits = base_bits; plan->c = c; plan->smoothing = smoothing; plan->sigma = sigma;
    for (const auto *owner : {a, b, d, product, static_cast<const GpuMatrix *>(p1), residual,
            static_cast<const GpuMatrix *>(gadget)})
        if (owner->ctx != a->ctx || owner->level != a->level)
            return set_error("prepared preimage phase context/level mismatch");
    if (a->cols != plan->d || b->rows != plan->d || b->cols != plan->d ||
        d->rows != plan->d || d->cols != plan->d || product->rows != 2 * plan->d ||
        product->cols != plan->columns || p1->rows != 2 * plan->d ||
        residual->rows != plan->d || residual->cols != plan->columns ||
        gadget->cols != plan->columns)
        return set_error("prepared preimage phase shape mismatch");
    for (const auto *owner : {a, b, d, product, residual})
        if (owner->format != GPU_POLY_FORMAT_COEFF)
            return set_error("prepared preimage phase input must be coefficient format");
    plan->reference = a->ctx->limb_gpu_ids[0];
    for (size_t limb = 0; limb < plan->limbs; ++limb) {
        if (a->ctx->limb_gpu_ids[limb].x != plan->reference.x)
            return set_error("prepared preimage phase requires one device per matrix");
        plan->p1_job.indices[limb] = a->ctx->limb_gpu_ids[limb].y;
        plan->p1_job.moduli[limb] = a->ctx->moduli[limb];
    }
    int status = matrix_limb_device(p1, plan->reference, &plan->device);
    if (status == 0) status = matrix_limb_stream(p1, plan->reference, &plan->stream);
    if (status != 0) return status;
    const auto selected = cudaSetDevice(plan->device);
    if (selected != cudaSuccess) return set_error(selected);
    for (size_t index = 0; index < 3; ++index) {
        plan->gram_bases[index] = matrix_limb_ptr_by_id(plan->gram[index], 0, plan->reference);
        if (!matrix_limb_metadata_by_id(plan->gram[index], plan->reference,
                &plan->gram_strides[index], &plan->gram_widths[index]))
            return set_error("prepared preimage covariance limb metadata is missing");
    }
    const size_t m = 2 * plan->d;
    if (m / 2 != plan->d || plan->n > SIZE_MAX / m ||
        plan->n * m > SIZE_MAX / m / sizeof(double))
        return set_error("prepared preimage covariance size overflow");
    GpuPreparedWorkspaceLayout layouts[6]{};
    status = gpu_preimage_phase_layout(plan->ctx, plan->d, plan->columns, layouts);
    if (status != 0) return status;
    const size_t roots_bytes = layouts[0].bytes;
    const size_t covariance_bytes = layouts[1].bytes;
    if (plan->roots.acquire(plan->ctx, plan->device, GPU_PREPARED_SAMPLER_WORKSPACE,
            roots_bytes, alignof(double), plan->stream) != 0 ||
        plan->updates.acquire(plan->ctx, plan->device, GPU_PREPARED_SAMPLER_WORKSPACE,
            covariance_bytes, alignof(double), plan->stream) != 0 ||
        plan->covariance.acquire(plan->ctx, plan->device, GPU_PREPARED_SAMPLER_WORKSPACE,
            covariance_bytes, alignof(double), plan->stream) != 0 ||
        plan->completion.acquire(plan->ctx, plan->device, GPU_PREPARED_COMPLETION_EVENT) != 0)
        return 1;
    const auto *workspace = layouts + 4;
    if (plan->samples.acquire(plan->ctx, plan->device, workspace[0].kind,
            workspace[0].bytes, workspace[0].alignment, plan->stream) != 0 ||
        (workspace[1].bytes && plan->extra.acquire(plan->ctx, plan->device, workspace[1].kind,
            workspace[1].bytes, workspace[1].alignment, plan->stream) != 0))
        return 1;
    plan->p1_job.jobs[0] = {
        product->shared_limb_buffers[plan->reference.x].device_descriptors,
        p1->shared_limb_buffers[plan->reference.x].device_descriptors,
        reinterpret_cast<double *>(plan->roots.data), reinterpret_cast<double *>(plan->updates.data),
        reinterpret_cast<int64_t *>(plan->samples.data),
        workspace[1].bytes ? reinterpret_cast<int64_t *>(plan->extra.data +
            2 * plan->d * plan->columns * plan->n * sizeof(double)) : nullptr,
        reinterpret_cast<double *>(plan->extra.data), a->ctx->moduli[0],
        -(c * c) / (smoothing * smoothing - c * c), {}
    };
    plan->gadget_job.jobs[0] = {
        residual->shared_limb_buffers[plan->reference.x].device_descriptors,
        gadget->shared_limb_buffers[plan->reference.x].device_descriptors, {}
    };
    uint32_t bits = 0;
    for (const auto modulus : a->ctx->moduli) bits = std::max(bits, bit_width_u64(modulus));
    plan->digits = (bits + base_bits - 1) / base_bits;
    if (!plan->digits || plan->digits > kGaussMaxDigits ||
        gadget->rows != plan->d * plan->limbs * plan->digits)
        return set_error("prepared preimage gadget dimensions mismatch");
    plan->moduli = a->ctx->ring_device_constants[plan->reference.x].moduli;
    plan->covariance_grid = dim3((plan->n + 255) / 256);
    plan->p1_grid = dim3((plan->columns * plan->n + 255) / 256, 1);
    plan->scatter_grid = dim3((2 * plan->d * plan->columns * plan->n + 255) / 256, 1, plan->limbs);
    plan->gadget_grid = dim3((plan->d * plan->columns * plan->n + 255) / 256, 1, plan->limbs);
    status = gpu_matrix_prepare_ntt_plan(p1, nullptr, true, &plan->p1_ntt);
    if (status == 0) status = gpu_matrix_prepare_ntt_plan(gadget, nullptr, true, &plan->gadget_ntt);
    if (status != 0) return status;
    *out = plan.release();
    return 0;
} catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_preimage_refresh_covariance(GpuPreparedPreimagePhases *plan)
{
    if (!plan) return set_error("missing prepared preimage phases");
    auto error = cudaSetDevice(plan->device);
    if (error != cudaSuccess) return set_error(error);
    for (const auto *input : plan->gram)
        if (matrix_wait_limb_stream(input, plan->reference, plan->device, plan->stream) != 0) return 1;
    gpu_test_record_kernel_launch();
    matrix_precompute_p1_covariance_kernel<<<plan->covariance_grid, 256, 0, plan->stream>>>(
        plan->gram_bases[0], plan->gram_bases[1], plan->gram_bases[2],
        plan->gram_strides[0], plan->gram_strides[1], plan->gram_strides[2],
        plan->gram_widths[0], plan->gram_widths[1], plan->gram_widths[2],
        plan->d, plan->n, plan->ctx->moduli[0], plan->c, plan->smoothing, plan->sigma,
        reinterpret_cast<double *>(plan->covariance.data),
        reinterpret_cast<double *>(plan->roots.data), reinterpret_cast<double *>(plan->updates.data));
    error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaEventRecord(plan->completion.event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    for (const auto *input : plan->gram)
        if (matrix_track_limb_consumer_readonly(input, plan->reference, plan->device,
                plan->stream, plan->completion.event, true) != 0) return 1;
    return 0;
}

extern "C" int gpu_preimage_submit_p1(GpuPreparedPreimagePhases *plan, GpuRngSeed seed)
{
    if (!plan) return set_error("missing prepared preimage phases");
    auto error = cudaSetDevice(plan->device);
    if (error == cudaSuccess && plan->completed_event)
        error = cudaStreamWaitEvent(plan->stream, plan->completed_event, 0);
    if (error != cudaSuccess) return set_error(error);
    if (matrix_wait_all_limb_streams(plan->p1, plan->device, plan->stream, true) != 0 ||
        matrix_wait_limb_stream(plan->product, plan->reference, plan->device, plan->stream) != 0)
        return 1;
    plan->p1_job.jobs[0].seed = seed;
    gpu_test_record_kernel_launch();
    if (2 * plan->d <= kSampleP1LocalMaxM)
        matrix_sample_p1_batch_kernel<false><<<plan->p1_grid, 256, 0, plan->stream>>>(
            plan->p1_job, plan->d, plan->columns, plan->n);
    else
        matrix_sample_p1_batch_kernel<true><<<plan->p1_grid, 256, 0, plan->stream>>>(
            plan->p1_job, plan->d, plan->columns, plan->n);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    gpu_test_record_kernel_launch();
    matrix_scatter_p1_batch_kernel<<<plan->scatter_grid, 256, 0, plan->stream>>>(
        plan->p1_job, 2 * plan->d * plan->columns, plan->n);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    if (matrix_record_all_limb_writes(plan->p1, plan->stream) != 0) return 1;
    error = cudaEventRecord(plan->completion.event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    if (matrix_track_limb_consumer_readonly(plan->product, plan->reference, plan->device,
            plan->stream, plan->completion.event, true) != 0) return 1;
    return gpu_matrix_submit_ntt_plan(plan->p1_ntt, plan->p1);
}

extern "C" int gpu_preimage_submit_gadget(GpuPreparedPreimagePhases *plan, GpuRngSeed seed)
{
    if (!plan) return set_error("missing prepared preimage phases");
    auto error = cudaSetDevice(plan->device);
    if (error == cudaSuccess && plan->completed_event)
        error = cudaStreamWaitEvent(plan->stream, plan->completed_event, 0);
    if (error != cudaSuccess) return set_error(error);
    if (matrix_wait_all_limb_streams(plan->gadget, plan->device, plan->stream, true) != 0 ||
        matrix_wait_all_limb_streams(plan->residual, plan->device, plan->stream, true) != 0) return 1;
    plan->gadget_job.jobs[0].seed = seed;
    gpu_test_record_kernel_launch();
    matrix_sample_gadget_batch_kernel<<<plan->gadget_grid, 256, 0, plan->stream>>>(
        plan->gadget_job, plan->moduli, plan->d * plan->columns, plan->n,
        plan->columns, plan->limbs, plan->base_bits, plan->digits, plan->c);
    error = cudaGetLastError();
    if (error != cudaSuccess) return set_error(error);
    if (matrix_record_all_limb_writes(plan->gadget, plan->stream) != 0) return 1;
    error = cudaEventRecord(plan->completion.event, plan->stream);
    if (error != cudaSuccess) return set_error(error);
    for (size_t limb = 0; limb < plan->limbs; ++limb)
        if (matrix_track_limb_consumer_readonly(plan->residual, plan->ctx->limb_gpu_ids[limb],
                plan->device, plan->stream, plan->completion.event, true) != 0) return 1;
    return gpu_matrix_submit_ntt_plan(plan->gadget_ntt, plan->gadget);
}

extern "C" void gpu_preimage_destroy_phases(GpuPreparedPreimagePhases *plan) { delete plan; }

extern "C" int gpu_preimage_mask_phases(GpuPreparedPreimagePhases *plan,
    const GpuPreparedPreimageCutoff *cutoff, size_t job)
{
    if (!plan || !cutoff || cutoff->ctx != plan->ctx || job >= cutoff->sources.size())
        return set_error("prepared preimage phase acceptance binding mismatch");
    const auto *completed = reinterpret_cast<const int32_t *>(cutoff->success.data) + job;
    plan->p1_job.jobs[0].completed = completed;
    plan->gadget_job.jobs[0].completed = completed;
    plan->completed_event = cutoff->completion.event;
    return 0;
}

extern "C" int gpu_preimage_mask_sampling(GpuPreparedSampling *plan,
    const GpuPreparedPreimageCutoff *cutoff, size_t job)
{
    if (!plan || !cutoff || cutoff->ctx != plan->out->ctx || job >= cutoff->sources.size())
        return set_error("prepared preimage sampling acceptance binding mismatch");
    plan->layout.completed = reinterpret_cast<const int32_t *>(cutoff->success.data) + job;
    plan->completed_event = cutoff->completion.event;
    return 0;
}
