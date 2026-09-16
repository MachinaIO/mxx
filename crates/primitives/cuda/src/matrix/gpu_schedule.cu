// Included after the primitive implementations so stream snapshots cannot
// diverge from private native plan layouts.
#include "gpu_schedule.cuh"

namespace {
struct PreparedScheduleStream {
    const GpuContext *context;
    int device;
    cudaStream_t stream;
    std::unique_ptr<GpuCudaResource> completion;
};
struct PreparedScheduleState {
    std::vector<PreparedScheduleStream> streams;
    std::vector<std::shared_ptr<PreparedScheduleState>> dependencies;
    bool provisioned = false;
    bool bound = false;

    void add(const GpuContext *context, int device, cudaStream_t stream) {
        for (const auto &entry : streams)
            if (entry.device == device && entry.stream == stream) return;
        streams.push_back({context, device, stream, nullptr});
    }
    void ntt(const GpuMatrixTransformPlan *plan) {
        if (!plan) return;
        for (const auto &launch : plan->launches)
            add(plan->matrix->ctx, launch.constants->device, launch.stream);
    }
};
}

struct GpuPreparedSchedule {
    std::shared_ptr<PreparedScheduleState> state;
};

extern "C" int gpu_prepared_schedule_is_ready(const GpuPreparedSchedule *schedule, bool *ready)
{
    if (!schedule || !ready || !schedule->state->provisioned)
        return set_error("invalid prepared schedule readiness query");
    *ready = true;
    int current = 0;
    cudaError_t error = cudaGetDevice(&current);
    if (error != cudaSuccess) return set_error(error);
    for (const auto &stream : schedule->state->streams) {
        error = cudaSetDevice(stream.device);
        if (error == cudaSuccess) error = cudaEventQuery(stream.completion->event);
        if (error == cudaErrorNotReady) {
            // A stream that is still running is a readiness result, never an
            // error: this query is the runtime's nonblocking polling path, and
            // reporting the pending status as a failure would poison a healthy,
            // simply unfinished schedule.
            *ready = false;
            error = cudaSuccess;
            continue;
        }
        if (error != cudaSuccess) break;
    }
    const cudaError_t restored = cudaSetDevice(current);
    if (error == cudaSuccess) error = restored;
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" int gpu_prepared_schedule_create(
    const GpuPreparedPlanRef *plans, size_t count,
    GpuPreparedSchedule *const *dependencies, size_t dependency_count,
    GpuPreparedSchedule **out)
try {
    if (!out || (count && !plans) || (dependency_count && !dependencies))
        return set_error("invalid prepared schedule description");
    *out = nullptr;
    auto schedule = std::make_unique<GpuPreparedSchedule>();
    schedule->state = std::make_shared<PreparedScheduleState>();
    auto &state = *schedule->state;
    for (size_t i = 0; i < dependency_count; ++i) {
        if (!dependencies[i] || !dependencies[i]->state->bound)
            return set_error("prepared schedule predecessor is not finalized");
        state.dependencies.push_back(dependencies[i]->state);
    }
    state.bound = dependency_count != 0;
    for (size_t i = 0; i < count; ++i) {
        if (!plans[i].plan) return set_error("missing prepared schedule plan");
        switch (plans[i].kind) {
        case 0: {
            const auto *p = static_cast<const GpuPreparedArithmeticState *>(plans[i].plan);
            state.add(p->out->ctx, p->device, p->stream); break;
        }
        case 1: state.ntt(static_cast<const GpuMatrixTransformPlan *>(plans[i].plan)); break;
        case 2: {
            const auto *p = static_cast<const GpuPreparedModulusConversion *>(plans[i].plan);
            state.add(p->target_context, p->device, p->stream); break;
        }
        case 3: {
            const auto *p = static_cast<const GpuPreparedInputCopyState *>(plans[i].plan);
            state.add(p->out->ctx, p->device, p->stream); break;
        }
        case 4: {
            const auto *p = static_cast<const GpuPreparedTransposeState *>(plans[i].plan);
            state.add(p->out->ctx, p->device, p->stream); break;
        }
        case 5: {
            const auto *p = static_cast<const GpuPreparedCenteredRebaseState *>(plans[i].plan);
            state.add(p->out->ctx, p->device, p->stream); break;
        }
        case 6: {
            const auto *p = static_cast<const GpuPreparedGadgetDecompose *>(plans[i].plan);
            state.add(p->output->ctx, p->dispatch_device, p->dispatch_stream);
            if (p->coefficient_source) {
                for (const auto &partition : p->coefficient_source->exec_limb_states)
                    for (const auto &limb : partition)
                        state.add(p->coefficient_source->ctx, limb.device, limb.stream);
            }
            state.ntt(p->inverse); state.ntt(p->forward); break;
        }
        case 7: {
            const auto *p = static_cast<const GpuPreparedSampling *>(plans[i].plan);
            state.add(p->out->ctx, p->device, p->stream); state.ntt(p->transform); break;
        }
        case 8: {
            const auto *p = static_cast<const GpuPreparedSmallRhs *>(plans[i].plan);
            state.add(p->ctx, p->device, p->stream); break;
        }
        case 9: {
            const auto *p = static_cast<const GpuPreparedCrtRecompose *>(plans[i].plan);
            state.add(p->output->ctx, p->device, p->stream); break;
        }
        case 10: {
            const auto *p = static_cast<const GpuPreparedConstCoeffReadback *>(plans[i].plan);
            for (const auto &limb : p->limbs) state.add(p->matrix->ctx, limb.device, limb.stream);
            break;
        }
        case 11: {
            const auto *p = static_cast<const GpuPreparedRnsUpload *>(plans[i].plan);
            for (const auto &limb : p->limbs) state.add(p->matrix->ctx, limb.device, limb.stream);
            state.ntt(p->transform); break;
        }
        case 12: {
            const auto *p = static_cast<const GpuPreparedCompactDecompose *>(plans[i].plan);
            state.add(p->output->ctx, p->output->device, p->output->stream);
            state.ntt(p->inverse); break;
        }
        case 13: {
            const auto *p = static_cast<const GpuPreparedThreshold *>(plans[i].plan);
            state.add(p->output->anchor->ctx, p->device, p->stream); break;
        }
        case 14: {
            const auto *p = static_cast<const GpuPreparedScalarPack *>(plans[i].plan);
            state.add(p->output->ctx, p->device, p->stream);
            state.ntt(p->transform); break;
        }
        case 15: {
            const auto *p = static_cast<const GpuPreparedScalarOp *>(plans[i].plan);
            state.add(p->output->anchor->ctx, p->output->device, p->output->stream); break;
        }
        case 16: {
            const auto *p = static_cast<const GpuPreparedScalarBuffer *>(plans[i].plan);
            state.add(p->anchor->ctx, p->device, p->stream); break;
        }
        case 17: {
            const auto *p = static_cast<const GpuPreparedScalarMatrixSelect *>(plans[i].plan);
            state.add(p->output->ctx, p->device, p->stream); break;
        }
        case 18: {
            const auto *p = static_cast<const GpuPreparedPreimageCutoff *>(plans[i].plan);
            state.add(p->ctx, p->device, p->stream); break;
        }
        case 19: {
            const auto *p = static_cast<const GpuPreparedPreimagePhases *>(plans[i].plan);
            state.add(p->ctx, p->device, p->stream);
            state.ntt(p->p1_ntt); state.ntt(p->gadget_ntt); break;
        }
        default: return set_error("unknown prepared schedule plan family");
        }
    }
    *out = schedule.release();
    return 0;
}
catch (const std::exception &error) { return set_error(error.what()); }

extern "C" size_t gpu_prepared_schedule_stream_count(const GpuPreparedSchedule *schedule)
{
    return schedule->state->streams.size();
}

extern "C" const void *gpu_prepared_schedule_stream_context(const GpuPreparedSchedule *schedule, size_t index)
{
    return index < schedule->state->streams.size() ? schedule->state->streams[index].context : nullptr;
}

extern "C" int gpu_prepared_schedule_provision(GpuPreparedSchedule *schedule)
try {
    auto &state = *schedule->state;
    if (state.provisioned) return set_error("prepared schedule is already provisioned");
    for (auto &entry : state.streams) {
        entry.completion = std::make_unique<GpuCudaResource>();
        const int status = entry.completion->acquire(entry.context, entry.device, GPU_PREPARED_COMPLETION_EVENT);
        if (status != 0) return status;
    }
    state.provisioned = true;
    return 0;
}
catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_prepared_schedule_bind_dependencies(GpuPreparedSchedule *schedule,
    const GpuPreparedSchedule *const *dependencies, size_t count)
try {
    auto &state = *schedule->state;
    if (state.bound || !state.provisioned)
        return set_error("schedule dependency binding requires unbound provisioned storage");
    for (size_t i = 0; i < count; ++i) {
        if (!dependencies[i] || !dependencies[i]->state->provisioned || !dependencies[i]->state->bound ||
            dependencies[i]->state.get() == &state)
            return set_error("invalid prepared schedule predecessor");
    }
    for (size_t i = 0; i < count; ++i) state.dependencies.push_back(dependencies[i]->state);
    state.bound = true;
    return 0;
}
catch (const std::exception &error) { return set_error(error.what()); }

extern "C" int gpu_prepared_schedule_begin(const GpuPreparedSchedule *schedule)
{
    // All loops and addresses below are a frozen prepare-time command table.
    for (const auto &entry : schedule->state->streams) {
        cudaError_t error = cudaSetDevice(entry.device);
        if (error != cudaSuccess) return set_error(error);
        for (const auto &dependency : schedule->state->dependencies)
            for (const auto &producer : dependency->streams) {
                error = cudaStreamWaitEvent(entry.stream, producer.completion->event, 0);
                if (error != cudaSuccess) return set_error(error);
            }
    }
    return 0;
}

extern "C" int gpu_prepared_schedule_end(const GpuPreparedSchedule *schedule)
{
    // Keep one completion per stream: a consumer joins ALL of these, including
    // serde limb streams and nested NTT streams. No host or device-wide fence.
    for (const auto &entry : schedule->state->streams) {
        cudaError_t error = cudaSetDevice(entry.device);
        if (error == cudaSuccess) error = cudaEventRecord(entry.completion->event, entry.stream);
        if (error != cudaSuccess) return set_error(error);
    }
    return 0;
}

extern "C" void gpu_prepared_schedule_destroy(GpuPreparedSchedule *schedule) { delete schedule; }
