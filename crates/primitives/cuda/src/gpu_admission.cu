#include "gpu_admission.cuh"
#include "matrix/Matrix.cuh"
#include "matrix/MatrixData.cuh"

#include <atomic>
#include <string>
#include <cstdio>
#include <algorithm>
#include <exception>
#include <memory>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr unsigned int available = 0;
constexpr unsigned int reserved = 1;
constexpr unsigned int leased = 2;
constexpr unsigned int quarantined = 3;
constexpr unsigned int inspecting = 4;
constexpr size_t resetting = std::numeric_limits<size_t>::max();

int fail(const char *message) { return gpu_set_last_error(message); }
int fail(cudaError_t error) { return fail(cudaGetErrorString(error)); }

uint64_t next_identity() {
    static std::atomic<uint64_t> next{1};
    uint64_t value = next.load(std::memory_order_relaxed);
    while (value != std::numeric_limits<uint64_t>::max()) {
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
    throw std::runtime_error("prepared matrix identity space exhausted");
}

bool is_resource(GpuPreparedSlotKind kind) {
    return kind == GPU_PREPARED_COMPLETION_EVENT || kind == GPU_PREPARED_SUBMISSION_STREAM;
}

struct ResourceOccupancy {
    size_t capacity = 0;
    std::atomic<size_t> reserved{0};
    std::atomic<size_t> occupied{0};
    std::atomic<size_t> high_water{0};
};

struct Slot {
    GpuMatrix *matrix = nullptr;
    void *workspace = nullptr;
    cudaEvent_t resource_event = nullptr;
    cudaStream_t resource_stream = nullptr;
    int resource_device = -1;
    GpuPreparedSlotIdentity identity{};
    cudaEvent_t reusable = nullptr;
    size_t auxiliary_capacity_bytes = 0;
    std::atomic<unsigned int> state{available};
    // Exclusive invocation ownership is separate from one wave's live lease.
    // Recycle may make a slot idle without allowing another invocation to steal it.
    std::atomic<uint64_t> reservation_owner{0};
    std::atomic<bool> pinned_deferred{false};
    // Protected by exclusive reserved/leased/inspecting slot ownership. Recycle
    // deliberately keeps this charge until completion or event-ordered reuse.
    size_t occupied_units = 0;

    ~Slot() {
        if (resource_event || resource_stream) {
            GpuAllocationActivity activity(nullptr, resource_device);
            int previous = -1;
            cudaError_t error = cudaGetDevice(&previous);
            if (error == cudaSuccess) error = cudaSetDevice(resource_device);
            if (error == cudaSuccess && resource_event) error = cudaEventDestroy(resource_event);
            if (error == cudaSuccess && resource_stream) error = cudaStreamDestroy(resource_stream);
            if (previous >= 0 && cudaSetDevice(previous) != cudaSuccess)
                gpu_device_mark_allocation_unknown(previous);
            if (error != cudaSuccess) gpu_device_mark_allocation_unknown(resource_device);
        }
        // Host DMA leases can outlive the matrix inventory. Their retirement
        // must never destroy a context on its own pinned-reclaimer thread.
        if (identity.kind == GPU_PREPARED_PINNED_HOST && workspace)
            gpu_pinned_free(workspace);
    }
};

struct PinnedOccupancy {
    size_t capacity_bytes = 0;
    std::atomic<size_t> reserved_bytes{0};
    std::atomic<size_t> occupied_bytes{0};
    std::atomic<size_t> high_water_bytes{0};
};

struct Storage {
    uint64_t identity = 0;
    GpuContext *context = nullptr;
    int device = -1;
    std::vector<std::shared_ptr<Slot>> slots;
    size_t requested_capacity_bytes = 0;
    std::shared_ptr<PinnedOccupancy> pinned = std::make_shared<PinnedOccupancy>();
    std::shared_ptr<ResourceOccupancy> resources = std::make_shared<ResourceOccupancy>();
    std::atomic<size_t> reserved_bytes{0};
    std::atomic<size_t> occupied_bytes{0};
    std::atomic<size_t> occupied_high_water_bytes{0};
    std::atomic<size_t> active_reservations{0};
    bool registered = false;
    void *owner = nullptr;
    void (*release_owner)(void *) = nullptr;

    ~Storage() {
        auto execution = context ? context->execution : nullptr;
        if (registered) {
            execution->prepared_occupied_bytes.fetch_sub(
                occupied_bytes.load(std::memory_order_acquire), std::memory_order_acq_rel);
            execution->prepared_storage_count.fetch_sub(1, std::memory_order_acq_rel);
        }
        GpuAllocationActivity activity(execution.get(), device);
        int previous = -1;
        cudaError_t error = cudaGetDevice(&previous);
        if (error == cudaSuccess && device >= 0) error = cudaSetDevice(device);
        for (auto &slot : slots) {
            if (error == cudaSuccess && slot->workspace &&
                slot->identity.kind != GPU_PREPARED_PINNED_HOST)
                error = cudaFreeAsync(slot->workspace, execution->release_streams_by_partition[0]);
            if (error == cudaSuccess && slot->reusable)
                error = cudaEventDestroy(slot->reusable);
        }
        slots.clear();
        // Every idle slot has already joined its readers on the release stream.
        // The real owners retain their existing event-ordered allocation frees.
        if (error == cudaSuccess) {
            if (release_owner) release_owner(owner);
        } else if (execution) {
            // The callback owns the real backing. Retain it if CUDA cannot
            // safely retire the prepared resources.
            execution->memory_release_failed.store(true, std::memory_order_release);
            execution->unretired_work.store(true, std::memory_order_release);
            gpu_execution_mark_allocation_unknown(execution.get());
        }
        if (previous >= 0 && previous != device && cudaSetDevice(previous) != cudaSuccess)
            gpu_execution_mark_allocation_unknown(execution.get());
    }
};

void quarantine(Storage &storage) {
    storage.context->execution->memory_release_failed.store(true, std::memory_order_release);
    storage.context->execution->unretired_work.store(true, std::memory_order_release);
    gpu_execution_mark_allocation_unknown(storage.context->execution.get());
}

size_t reservation_units(const Slot &slot) {
    return is_resource(slot.identity.kind) ? 1 : slot.identity.requested_backing_bytes;
}

std::atomic<size_t> &reserved_counter(Storage &storage, const Slot &slot) {
    if (is_resource(slot.identity.kind)) return storage.resources->reserved;
    return slot.identity.kind == GPU_PREPARED_PINNED_HOST
        ? storage.pinned->reserved_bytes : storage.reserved_bytes;
}

void occupy(Storage &storage, Slot &slot, size_t requested_bytes) {
    auto &occupied = is_resource(slot.identity.kind) ? storage.resources->occupied :
        (slot.identity.kind == GPU_PREPARED_PINNED_HOST
        ? storage.pinned->occupied_bytes : storage.occupied_bytes);
    auto &high_water = is_resource(slot.identity.kind) ? storage.resources->high_water :
        (slot.identity.kind == GPU_PREPARED_PINNED_HOST
        ? storage.pinned->high_water_bytes : storage.occupied_high_water_bytes);
    // Each layout uses nested prefixes of this group's data/auxiliary spans.
    // Keep their largest unretired use, including preceding readers, once.
    if (requested_bytes > slot.occupied_units) {
        const size_t increase = requested_bytes - slot.occupied_units;
        slot.occupied_units = requested_bytes;
        const size_t bytes = occupied.fetch_add(
            increase, std::memory_order_acq_rel) + increase;
        size_t peak = high_water.load(std::memory_order_relaxed);
        while (peak < bytes && !high_water.compare_exchange_weak(
                   peak, bytes, std::memory_order_relaxed)) {}
        if (!is_resource(slot.identity.kind) && slot.identity.kind != GPU_PREPARED_PINNED_HOST) {
            auto &execution = *storage.context->execution;
            const size_t joint = execution.prepared_occupied_bytes.fetch_add(
                increase, std::memory_order_acq_rel) + increase;
            size_t joint_peak = execution.prepared_high_water_bytes.load(std::memory_order_relaxed);
            while (joint_peak < joint && !execution.prepared_high_water_bytes.compare_exchange_weak(
                       joint_peak, joint, std::memory_order_relaxed)) {}
        }
    }
    reserved_counter(storage, slot).fetch_sub(
        reservation_units(slot), std::memory_order_acq_rel);
}

struct PreparedClaim {
    GpuPreparedRequest request;
    size_t data_bytes = 0;
    size_t auxiliary_slots = 0;
    size_t occupied_units = 0;
};

int prepare_claims(
    const Storage &storage, const GpuPreparedRequest *requests, size_t count,
    std::vector<PreparedClaim> &claims, bool &fits)
{
    fits = false;
    if (count != 0 && !requests) return fail("missing prepared requests");
    std::unordered_set<size_t> unique;
    claims.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const auto &request = requests[index];
        if (request.storage_id != storage.identity || request.slot_index >= storage.slots.size())
            return fail("prepared request belongs to another storage");
        const auto &slot = *storage.slots[request.slot_index];
        const auto &identity = slot.identity;
        if (request.slot_id != identity.slot_id || !unique.insert(request.slot_index).second)
            return fail("stale or duplicate prepared slot request");
        if (request.kind != identity.kind) return 0;
        PreparedClaim claim{request};
        if (request.kind == GPU_PREPARED_MATRIX) {
            if (request.rows == 0 || request.columns == 0 || request.bytes != 0 ||
                request.alignment != 0 ||
                (request.format != GPU_POLY_FORMAT_COEFF && request.format != GPU_POLY_FORMAT_EVAL))
                return fail("invalid prepared matrix request layout");
            if (request.level != identity.level || request.rows > identity.rows ||
                request.columns > identity.columns) return 0;
            GpuMatrixAllocationBytes layout{};
            const int status = gpu_matrix_query_allocation_bytes(
                storage.context, request.level, request.rows, request.columns, request.format,
                &layout);
            if (status != 0) return status;
            if (layout.data_bytes > identity.payload_bytes ||
                layout.aux_workspace_bytes > slot.auxiliary_capacity_bytes ||
                layout.aux_workspace_bytes % sizeof(void *) != 0 ||
                layout.aux_bytes > SIZE_MAX - layout.data_bytes ||
                layout.data_bytes + layout.aux_bytes > identity.requested_backing_bytes) return 0;
            claim.data_bytes = layout.data_bytes;
            claim.auxiliary_slots = layout.aux_workspace_bytes / sizeof(void *);
            claim.occupied_units = layout.data_bytes + layout.aux_bytes;
        } else if (is_resource(request.kind)) {
            if (request.rows != 0 || request.columns != 0 || request.level != -1 ||
                request.format != -1 || request.bytes != 0 || request.alignment != 1)
                return fail("invalid prepared CUDA resource request");
            claim.occupied_units = 1;
        } else {
            if ((request.kind != GPU_PREPARED_BATCH_WORKSPACE &&
                 request.kind != GPU_PREPARED_TRANSFORM_WORKSPACE &&
                 request.kind != GPU_PREPARED_PINNED_HOST &&
                 request.kind != GPU_PREPARED_COMPACT_PAYLOAD &&
                 request.kind != GPU_PREPARED_COMPACT_WORKSPACE &&
                 request.kind != GPU_PREPARED_SAMPLER_WORKSPACE &&
                 request.kind != GPU_PREPARED_TRANSFER_WORKSPACE) ||
                request.rows != 0 || request.columns != 0 || request.level != -1 ||
                request.format != -1 || request.bytes == 0 || request.alignment == 0 ||
                (request.alignment & (request.alignment - 1)) != 0)
                return fail("invalid prepared workspace request layout");
            if (request.bytes > identity.requested_backing_bytes ||
                request.alignment > identity.alignment) return 0;
            claim.occupied_units = request.bytes;
        }
        claims.push_back(claim);
    }
    fits = true;
    return 0;
}
}

struct GpuPreparedStorage { std::shared_ptr<Storage> value; };
struct GpuPreparedMatrixLease {
    std::shared_ptr<Storage> storage;
    size_t slot;
};
struct GpuPreparedWorkspaceLease {
    std::shared_ptr<Storage> storage;
    size_t slot;
};
struct GpuPreparedPinnedLease {
    std::shared_ptr<Slot> slot;
    std::shared_ptr<PinnedOccupancy> occupancy;
    uint64_t execution_identity;
    int device;
};
struct GpuPreparedResourceLease {
    std::shared_ptr<Slot> slot;
    std::shared_ptr<ResourceOccupancy> occupancy;
};
struct GpuMatrixReservation {
    // Reverse member destruction keeps tracking active through the storage's
    // backing callback, then retains execution through the activity destructor.
    std::shared_ptr<GpuExecutionOwner> execution;
    GpuAllocationActivity activity;
    std::shared_ptr<Storage> storage;
    uint64_t identity = next_identity();
    std::vector<size_t> slots;
    std::vector<GpuPreparedRequest> bounds;
    std::vector<PreparedClaim> claims;
    size_t next = 0;
    size_t claimed = 0;

    explicit GpuMatrixReservation(std::shared_ptr<Storage> prepared)
        : execution(prepared->context->execution), activity(execution.get(), prepared->device),
          storage(std::move(prepared)) {
        size_t total = execution->prepared_active_reservations.load(std::memory_order_acquire);
        for (;;) {
            if (total >= resetting - 1)
                throw std::runtime_error("joint prepared occupancy reset or reservation limit reached");
            if (execution->prepared_active_reservations.compare_exchange_weak(
                    total, total + 1, std::memory_order_acq_rel)) break;
        }
        size_t count = storage->active_reservations.load(std::memory_order_acquire);
        for (;;) {
            if (count >= resetting - 1) {
                execution->prepared_active_reservations.fetch_sub(1, std::memory_order_acq_rel);
                throw std::runtime_error("prepared occupancy reset or reservation limit reached");
            }
            if (storage->active_reservations.compare_exchange_weak(
                    count, count + 1, std::memory_order_acq_rel)) break;
        }
    }

    ~GpuMatrixReservation() {
        // Only slots still owned by this reservation are returned. Published
        // output leases retain their independent ownership and reader edges.
        for (size_t index = next; index < claimed; ++index) {
            reserved_counter(*storage, *storage->slots[slots[index]]).fetch_sub(
                reservation_units(*storage->slots[slots[index]]),
                std::memory_order_acq_rel);
            storage->slots[slots[index]]->state.store(available, std::memory_order_release);
        }
        for (size_t index = 0; index < claimed; ++index)
            storage->slots[slots[index]]->reservation_owner.store(0, std::memory_order_release);
        storage->active_reservations.fetch_sub(1, std::memory_order_acq_rel);
        execution->prepared_active_reservations.fetch_sub(1, std::memory_order_acq_rel);
    }
};

struct GpuMatrixDispatchPermit {
    std::vector<std::unique_ptr<GpuMatrixReservation>> reservations;
    size_t current = 0;
    // Reservations the permit was entered with; extensions append beyond it and
    // must be retracted before the permit ends.
    size_t entered = 0;

    GpuMatrixReservation *next_reservation() {
        while (current < reservations.size() &&
               reservations[current]->next == reservations[current]->slots.size()) ++current;
        return current < reservations.size() ? reservations[current].get() : nullptr;
    }
};

namespace {
thread_local GpuMatrixDispatchPermit *active_permit = nullptr;

// Ordered record of the claims an operation would make. Recording happens only
// on open domains (no permit, admission not required), on the calling thread.
thread_local std::vector<GpuClaimTraceEntry> *active_claim_trace = nullptr;
// Recent claims consumed by permits on this thread, for mismatch diagnostics.
// Fixed ring: recording happens after a claim is committed, so it must not
// allocate or throw.
constexpr size_t consumed_capacity = 64;
thread_local GpuClaimTraceEntry consumed_claims[consumed_capacity];
thread_local size_t consumed_next = 0;
thread_local size_t consumed_count = 0;
void note_consumed(GpuPreparedSlotKind kind, size_t rows, size_t cols, int level, int format,
    size_t bytes, size_t alignment) noexcept
{
    consumed_claims[consumed_next] =
        GpuClaimTraceEntry{static_cast<int>(kind), rows, cols, level, format, bytes, alignment};
    consumed_next = (consumed_next + 1) % consumed_capacity;
    if (consumed_count < consumed_capacity) ++consumed_count;
}
std::string consumed_summary()
{
    std::string out;
    const size_t first = (consumed_next + consumed_capacity - consumed_count) % consumed_capacity;
    for (size_t offset = 0; offset < consumed_count; ++offset) {
        const auto &entry = consumed_claims[(first + offset) % consumed_capacity];
        out += "[" + std::to_string(entry.kind) + ":" + std::to_string(entry.rows) + "x" +
            std::to_string(entry.columns) + "/" + std::to_string(entry.bytes) + "]";
    }
    return out;
}
void trace_claim(GpuPreparedSlotKind kind, size_t rows, size_t cols, int level, int format,
    size_t bytes, size_t alignment)
{
    if (!active_claim_trace) return;
    try {
        active_claim_trace->push_back(GpuClaimTraceEntry{
            static_cast<int>(kind), rows, cols, level, format, bytes, alignment});
    } catch (...) {}
}
}

extern "C" int gpu_prepared_pinned_claim(
    GpuContext *ctx, size_t bytes, size_t alignment, void **out, int *handled)
{
    if (!ctx || !ctx->execution || !out || !handled) return fail("invalid prepared pinned claim");
    *out = nullptr;
    *handled = ctx->execution->pinned_admission_required.load(std::memory_order_acquire) ? 1 : 0;
    if (!*handled) {
        trace_claim(GPU_PREPARED_PINNED_HOST, 0, 0, -1, -1, bytes, alignment);
        return 0;
    }
    auto *reservation = active_permit ? active_permit->next_reservation() : nullptr;
    if (!reservation || reservation->storage->context != ctx)
        return fail("pinned allocation requires its next prepared dispatch claim");
    auto &storage = *reservation->storage;
    const auto &claim = reservation->claims[reservation->next];
    auto &slot = *storage.slots[reservation->slots[reservation->next]];
    if (slot.identity.kind != GPU_PREPARED_PINNED_HOST || claim.request.bytes != bytes ||
        claim.request.alignment != alignment || !slot.workspace) {
        static thread_local char message[256];
        std::snprintf(message, sizeof(message),
            "pinned allocation differs from its prepared size or alignment (kind %d, reserved %zu/%zu, requested %zu/%zu)",
            static_cast<int>(slot.identity.kind), claim.request.bytes, claim.request.alignment, bytes, alignment);
        return fail(message);
    }
    try {
        auto lease = std::make_unique<GpuPreparedPinnedLease>(
            GpuPreparedPinnedLease{storage.slots[claim.request.slot_index], storage.pinned,
                storage.context->execution->identity, storage.device});
        const int status = gpu_pinned_bind_prepared(slot.workspace, lease.get());
        if (status != 0) return status;
        occupy(storage, slot, bytes);
        ++reservation->next;
        slot.state.store(leased, std::memory_order_release);
        lease.release();
        note_consumed(GPU_PREPARED_PINNED_HOST, 0, 0, -1, -1, bytes, alignment);
        *out = slot.workspace;
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" int gpu_prepared_pinned_defer(GpuPreparedPinnedLease *lease, GpuContext *ctx)
{
    if (!lease || !ctx || !ctx->execution || lease->execution_identity != ctx->execution->identity)
        return fail("prepared pinned buffer belongs to another execution owner");
    lease->slot->pinned_deferred.store(true, std::memory_order_release);
    return 0;
}

extern "C" void gpu_prepared_pinned_recycle(GpuPreparedPinnedLease *lease)
{
    GpuAllocationActivity activity(nullptr, lease->device);
    auto &slot = *lease->slot;
    // Called only after CPU ownership ends or the reclaimer observes the DMA
    // event. No device-event wait can make an earlier CPU write safe here.
    lease->occupancy->occupied_bytes.fetch_sub(slot.occupied_units, std::memory_order_acq_rel);
    slot.occupied_units = 0;
    slot.pinned_deferred.store(false, std::memory_order_release);
    slot.state.store(available, std::memory_order_release);
    delete lease;
}

GpuCudaResource::GpuCudaResource()
    : event(nullptr), stream(nullptr), execution(nullptr), device(-1), lease(nullptr) {}

int GpuCudaResource::acquire(const GpuContext *ctx, int selected_device, GpuPreparedSlotKind kind)
{
    if (execution || event || stream || !ctx || !ctx->execution || !is_resource(kind) ||
        std::find(ctx->execution->gpu_ids.begin(), ctx->execution->gpu_ids.end(), selected_device) ==
            ctx->execution->gpu_ids.end())
        return fail("invalid CUDA resource request");
    execution = ctx->execution;
    device = selected_device;
    // A refused request leaves the handle reusable.
    const auto refuse = [this](const char *message) {
        execution.reset();
        device = -1;
        return fail(message);
    };
    GpuAllocationActivity activity(execution.get(), device);
    if (execution->unretired_work.load(std::memory_order_acquire))
        return refuse("CUDA resource request has unretired work");
    if (execution->resource_admission_required.load(std::memory_order_acquire)) {
        auto *reservation = active_permit ? active_permit->next_reservation() : nullptr;
        if (!reservation || reservation->storage->context != ctx || reservation->storage->device != device)
            return refuse("CUDA resource requires a matching prepared dispatch permit");
        auto &storage = *reservation->storage;
        auto &slot = storage.slots[reservation->slots[reservation->next]];
        const auto &claim = reservation->claims[reservation->next];
        if (claim.request.kind != kind || !is_resource(slot->identity.kind))
            return refuse("CUDA resource differs from its prepared type");
        auto pending = std::make_unique<GpuPreparedResourceLease>(
            GpuPreparedResourceLease{slot, storage.resources});
        occupy(storage, *slot, claim.occupied_units);
        ++reservation->next;
        slot->state.store(leased, std::memory_order_release);
        event = slot->resource_event;
        stream = slot->resource_stream;
        lease = pending.release();
        note_consumed(kind, 0, 0, -1, -1, 0, 1);
        return 0;
    }
    trace_claim(kind, 0, 0, -1, -1, 0, 1);
    cudaError_t error = cudaSetDevice(device);
    if (error == cudaSuccess) {
        if (kind == GPU_PREPARED_COMPLETION_EVENT)
            error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        else {
            error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (error == cudaSuccess) error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        }
    }
    if (error != cudaSuccess) {
        if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
        event = nullptr;
        stream = nullptr;
        execution.reset();
        device = -1;
        return fail(error);
    }
    return 0;
}

int GpuCudaResource::release()
{
    if (!event && !stream) return 0;
    GpuAllocationActivity activity(execution.get(), device);
    if (execution && execution->unretired_work.load(std::memory_order_acquire)) {
        quarantine();
        return fail("cannot recycle CUDA resource with unretired work");
    }
    if (lease) {
        auto *completed = lease;
        lease = nullptr;
        event = nullptr;
        stream = nullptr;
        // All host users have relinquished this handle. Already enqueued CUDA
        // event waits capture the prior record and are unaffected by rerecording:
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        completed->occupancy->occupied.fetch_sub(1, std::memory_order_acq_rel);
        completed->slot->occupied_units = 0;
        completed->slot->state.store(available, std::memory_order_release);
        delete completed;
        return 0;
    }
    cudaError_t error = cudaSetDevice(device);
    if (error == cudaSuccess && event) error = cudaEventDestroy(event);
    if (error == cudaSuccess && stream) error = cudaStreamDestroy(stream);
    event = nullptr;
    stream = nullptr;
    if (error != cudaSuccess) {
        gpu_execution_mark_allocation_unknown(execution.get());
        gpu_device_mark_allocation_unknown(device);
        if (execution) execution->memory_release_failed.store(true, std::memory_order_release);
        return fail(error);
    }
    return 0;
}

void GpuCudaResource::quarantine()
{
    gpu_execution_mark_allocation_unknown(execution.get());
    gpu_device_mark_allocation_unknown(device);
    if (lease) lease->slot->state.store(quarantined, std::memory_order_release);
    // Preserve uncertain handles and their slot ownership indefinitely.
    lease = nullptr;
    event = nullptr;
    stream = nullptr;
}

void GpuCudaResource::detach_execution() { execution.reset(); }
GpuCudaResource::~GpuCudaResource() { release(); }

GpuDeviceWorkspace::GpuDeviceWorkspace()
    : data(nullptr), execution(nullptr), device(-1), stream(nullptr), lease(nullptr) {}

int GpuDeviceWorkspace::acquire(
    GpuContext *ctx, int selected_device, GpuPreparedSlotKind kind,
    size_t bytes, size_t alignment, cudaStream_t selected_stream)
{
    if (execution || !ctx || !ctx->execution || selected_device < 0 || !selected_stream ||
        alignment == 0 || alignment > 256 || (alignment & (alignment - 1)) != 0 ||
        (kind != GPU_PREPARED_BATCH_WORKSPACE && kind != GPU_PREPARED_TRANSFORM_WORKSPACE &&
         kind != GPU_PREPARED_COMPACT_PAYLOAD && kind != GPU_PREPARED_COMPACT_WORKSPACE &&
         kind != GPU_PREPARED_SAMPLER_WORKSPACE && kind != GPU_PREPARED_TRANSFER_WORKSPACE))
        return fail("invalid device workspace request");
    if (bytes == 0) return 0;
    if (ctx->execution->unretired_work.load(std::memory_order_acquire))
        return fail("device workspace request has unretired work");
    execution = ctx->execution;
    device = selected_device;
    stream = selected_stream;
    // A refused request leaves the owner reusable; only an uncertain reuse
    // edge (quarantine below) retains it.
    const auto refuse = [this](int status) {
        execution.reset();
        device = -1;
        stream = nullptr;
        return status;
    };
    GpuAllocationActivity activity(ctx->execution.get(), device);
    const bool requires_permit = kind == GPU_PREPARED_TRANSFER_WORKSPACE
        ? ctx->execution->transfer_admission_required.load(std::memory_order_acquire)
        : (active_permit || gpu_context_admission_is_required(ctx));
    if (requires_permit) {
        auto *next = active_permit ? active_permit->next_reservation() : nullptr;
        if (!next || next->storage->context != ctx || next->storage->device != device)
            return refuse(fail("device workspace requires a matching prepared dispatch permit"));
        auto &reservation = *next;
        auto &storage = *reservation.storage;
        const size_t index = reservation.slots[reservation.next];
        auto &slot = *storage.slots[index];
        const auto &claim = reservation.claims[reservation.next];
        if (slot.identity.kind != kind || !slot.workspace ||
            bytes != claim.request.bytes || alignment != claim.request.alignment)
            return refuse(fail("device workspace differs from its prepared type, capacity, or alignment"));
        std::unique_ptr<GpuPreparedWorkspaceLease> pending;
        try {
            pending = std::make_unique<GpuPreparedWorkspaceLease>(
                GpuPreparedWorkspaceLease{reservation.storage, index});
        } catch (const std::exception &error) { return refuse(fail(error.what())); }
        cudaError_t error = cudaSetDevice(device);
        if (error != cudaSuccess) return refuse(fail(error));
        occupy(storage, slot, claim.occupied_units);
        ++reservation.next;
        error = cudaStreamWaitEvent(stream, slot.reusable, 0);
        if (error != cudaSuccess) {
            quarantine(storage);
            slot.state.store(quarantined, std::memory_order_release);
            // An uncertain reuse edge must retain the whole backing owner.
            pending.release();
            return fail(error);
        }
        data = static_cast<uint8_t *>(slot.workspace);
        lease = pending.release();
        slot.state.store(leased, std::memory_order_release);
        note_consumed(kind, 0, 0, -1, -1, bytes, alignment);
        return 0;
    }
    trace_claim(kind, 0, 0, -1, -1, bytes, alignment);
    cudaError_t error = cudaSetDevice(device);
    if (error == cudaSuccess)
        error = cudaMallocAsync(reinterpret_cast<void **>(&data), bytes, stream);
    if (error != cudaSuccess) {
        data = nullptr;
        return refuse(fail(error));
    }
    return 0;
}

extern "C" void gpu_claim_trace_record(
    int kind, size_t rows, size_t cols, int level, int format, size_t bytes, size_t alignment)
{
    trace_claim(static_cast<GpuPreparedSlotKind>(kind), rows, cols, level, format, bytes, alignment);
}

// Depth of traced steps on this thread. Inside a traced step, consumer
// tracking claims its event even on the same-stream fast path, so a recorded
// plan does not depend on which streams the step's owners received.
thread_local int deterministic_consumer_events = 0;

extern "C" void gpu_claim_deterministic_events_push() { ++deterministic_consumer_events; }
extern "C" void gpu_claim_deterministic_events_pop()
{
    if (deterministic_consumer_events > 0) --deterministic_consumer_events;
}
extern "C" int gpu_claim_deterministic_events_active()
{
    return deterministic_consumer_events > 0 ? 1 : 0;
}

extern "C" int gpu_claim_trace_begin()
{
    if (active_claim_trace) return fail("a native claim trace is already active on this thread");
    try {
        active_claim_trace = new std::vector<GpuClaimTraceEntry>();
    } catch (const std::exception &error) { return fail(error.what()); }
    return 0;
}

extern "C" int gpu_claim_trace_end(GpuClaimTraceEntry *out, size_t capacity, size_t *count)
{
    if (!count) return fail("missing claim trace count");
    auto *trace = active_claim_trace;
    if (!trace) return fail("no native claim trace is active on this thread");
    *count = trace->size();
    // A null output only reports the length and keeps recording; a buffer
    // receives the entries and ends the trace.
    if (!out) return 0;
    const size_t copied = std::min(capacity, trace->size());
    std::copy(trace->begin(), trace->begin() + copied, out);
    active_claim_trace = nullptr;
    delete trace;
    return 0;
}

int GpuDeviceWorkspace::release(cudaStream_t completed_stream)
{
    if (!data) return 0;
    // A persistent owner joins all of its readers before passing the final
    // release stream. Short-lived scratch defaults to its original stream.
    if (completed_stream) stream = completed_stream;
    auto execution = std::move(this->execution);
    GpuAllocationActivity activity(execution.get(), device);
    void *pointer = data;
    data = nullptr;
    auto *prepared = lease;
    lease = nullptr;
    if (execution->unretired_work.load(std::memory_order_acquire)) {
        if (prepared)
            prepared->storage->slots[prepared->slot]->state.store(
                quarantined, std::memory_order_release);
        return fail("cannot recycle device workspace with unretired work");
    }
    cudaError_t error = cudaSetDevice(device);
    if (prepared) {
        auto &storage = *prepared->storage;
        auto &slot = *storage.slots[prepared->slot];
        if (error == cudaSuccess) error = cudaEventRecord(slot.reusable, stream);
        if (error == cudaSuccess)
            error = cudaStreamWaitEvent(
                execution->release_streams_by_partition[0], slot.reusable, 0);
        if (error != cudaSuccess) {
            quarantine(storage);
            slot.state.store(quarantined, std::memory_order_release);
            return fail(error);
        }
        slot.state.store(available, std::memory_order_release);
        delete prepared;
        return 0;
    }
    if (error == cudaSuccess) error = cudaFreeAsync(pointer, stream);
    if (error != cudaSuccess) {
        execution->memory_release_failed.store(true, std::memory_order_release);
        gpu_execution_mark_allocation_unknown(execution.get());
        return fail(error);
    }
    return 0;
}

cudaEvent_t GpuDeviceWorkspace::completion_event() const
{
    return lease ? lease->storage->slots[lease->slot]->reusable : nullptr;
}

GpuDeviceWorkspace::~GpuDeviceWorkspace() { release(); }

extern "C" int gpu_prepared_storage_create(
    GpuMatrix *const *matrices, size_t count,
    const GpuPreparedWorkspaceLayout *workspaces, size_t workspace_count, void *owner,
    void (*release_owner)(void *), GpuPreparedStorage **out)
{
    if (!out) return fail("null prepared matrix storage output");
    *out = nullptr;
    if (!matrices || count == 0 || !owner || !release_owner || !matrices[0] ||
        !matrices[0]->ctx || !matrices[0]->ctx->execution ||
        matrices[0]->ctx->gpu_ids.size() != 1 ||
        (workspace_count != 0 && !workspaces) || workspace_count > SIZE_MAX - count)
        return fail("invalid prepared matrix backing owners");
    GpuContext *context = matrices[0]->ctx;
    auto execution = context->execution;
    GpuAllocationActivity activity(execution.get(), context->gpu_ids[0]);
    if (gpu_context_admission_is_required(context))
        return fail("prepared matrix resources must be created before checked dispatch");
    if (context->execution->unretired_work.load(std::memory_order_acquire))
        return fail("cannot prepare matrix storage with unretired work");
    std::shared_ptr<Storage> storage;
    int previous = -1;
    cudaError_t error = cudaGetDevice(&previous);
    if (error != cudaSuccess) return fail(error);
    struct RestoreDevice {
        int device;
        GpuExecutionOwner *execution;
        ~RestoreDevice() {
            if (cudaSetDevice(device) != cudaSuccess)
                gpu_execution_mark_allocation_unknown(execution);
        }
    } restore_device{previous, execution.get()};
    try {
        storage = std::make_shared<Storage>();
        storage->identity = next_identity();
        storage->context = context;
        storage->device = context->gpu_ids[0];
        std::unordered_set<GpuMatrix *> unique;
        std::unordered_set<const void *> unique_backings;
        for (size_t index = 0; index < count; ++index) {
            GpuMatrix *matrix = matrices[index];
            if (!matrix || matrix->ctx != context || matrix->rows == 0 || matrix->cols == 0 ||
                matrix->shared_limb_buffers.size() != 1 ||
                matrix->exec_limb_states.size() != 1 ||
                !matrix->shared_limb_buffers[0].ptr ||
                matrix->shared_limb_buffers[0].prepared_lease ||
                !matrix->descriptors_initialized || !unique.insert(matrix).second ||
                !unique_backings.insert(matrix->shared_limb_buffers[0].ptr).second)
                return fail("prepared matrix slots require distinct initialized backing owners");
        }
        error = cudaSetDevice(storage->device);
        if (error != cudaSuccess) return fail(error);
        const cudaStream_t release = context->execution->release_streams_by_partition[0];
        if (!release) return fail("missing prepared matrix release stream");
        for (size_t index = 0; index < workspace_count; ++index) {
            const auto &layout = workspaces[index];
            if (is_resource(layout.kind)) {
                if (layout.bytes != 0 || layout.alignment != 1)
                    return fail("CUDA resource layouts require zero bytes and unit alignment");
                continue;
            }
            if (layout.bytes == 0 || layout.alignment == 0 || layout.alignment > 256 ||
                (layout.alignment & (layout.alignment - 1)) != 0 ||
                (layout.kind != GPU_PREPARED_BATCH_WORKSPACE &&
                 layout.kind != GPU_PREPARED_TRANSFORM_WORKSPACE &&
                 layout.kind != GPU_PREPARED_PINNED_HOST &&
                 layout.kind != GPU_PREPARED_COMPACT_PAYLOAD &&
                 layout.kind != GPU_PREPARED_COMPACT_WORKSPACE &&
                 layout.kind != GPU_PREPARED_SAMPLER_WORKSPACE &&
                 layout.kind != GPU_PREPARED_TRANSFER_WORKSPACE))
                return fail("invalid prepared device workspace layout");
        }
        storage->slots.reserve(count + workspace_count);
        for (size_t index = 0; index < count; ++index) {
            auto slot = std::make_shared<Slot>();
            slot->matrix = matrices[index];
            GpuMatrixAllocationBytes allocation{};
            const int layout_status = gpu_matrix_query_allocation_bytes(
                context, slot->matrix->level, slot->matrix->rows, slot->matrix->cols,
                static_cast<int>(slot->matrix->format), &allocation);
            if (layout_status != 0) return layout_status;
            slot->identity = GpuPreparedSlotIdentity{
                storage->identity, next_identity(), next_identity(), index,
                slot->matrix->rows, slot->matrix->cols,
                allocation.data_bytes, allocation.aux_bytes,
                allocation.data_bytes + allocation.aux_bytes, slot->matrix->level,
                GPU_PREPARED_MATRIX, 256};
            if (slot->identity.requested_backing_bytes >
                std::numeric_limits<size_t>::max() - storage->requested_capacity_bytes)
                return fail("prepared matrix capacity overflow");
            storage->requested_capacity_bytes += slot->identity.requested_backing_bytes;
            slot->auxiliary_capacity_bytes = allocation.aux_workspace_bytes;
            storage->slots.push_back(std::move(slot));
            auto &prepared = *storage->slots.back();
            // Acquire all limb-owned events now; shared completions still retain
            // their original completion_owner until a later writer detaches.
            for (auto &state : prepared.matrix->exec_limb_states[0]) {
                if (!state.write_done) {
                    error = cudaEventCreateWithFlags(&state.write_done, cudaEventDisableTiming);
                    if (error != cudaSuccess) break;
                }
            }
            if (error == cudaSuccess)
                error = cudaEventCreateWithFlags(&prepared.reusable, cudaEventDisableTiming);
            if (error != cudaSuccess) break;
            const int status = matrix_wait_all_limb_streams(
                prepared.matrix, storage->device, release, true, true);
            if (status != 0) {
                cudaSetDevice(previous);
                return status;
            }
            error = cudaEventRecord(prepared.reusable, release);
            if (error != cudaSuccess) break;
        }
        for (size_t index = 0; error == cudaSuccess && index < workspace_count; ++index) {
            const auto &layout = workspaces[index];
            auto &capacity = layout.kind == GPU_PREPARED_PINNED_HOST
                ? storage->pinned->capacity_bytes : storage->requested_capacity_bytes;
            if (layout.bytes > SIZE_MAX - capacity)
                return fail("prepared workspace capacity overflow");
            auto slot = std::make_shared<Slot>();
            slot->identity = GpuPreparedSlotIdentity{
                storage->identity, next_identity(), next_identity(), count + index,
                0, 0, layout.bytes, 0, layout.bytes, -1, layout.kind, layout.alignment};
            capacity += layout.bytes;
            storage->slots.push_back(std::move(slot));
            auto &prepared = *storage->slots.back();
            if (is_resource(layout.kind)) {
                if (storage->resources->capacity == SIZE_MAX)
                    return fail("prepared resource capacity overflow");
                ++storage->resources->capacity;
                prepared.resource_device = storage->device;
                if (layout.kind == GPU_PREPARED_COMPLETION_EVENT)
                    error = cudaEventCreateWithFlags(&prepared.resource_event, cudaEventDisableTiming);
                else {
                    error = cudaStreamCreateWithFlags(&prepared.resource_stream, cudaStreamNonBlocking);
                    if (error == cudaSuccess)
                        error = cudaEventCreateWithFlags(&prepared.resource_event, cudaEventDisableTiming);
                }
            } else if (layout.kind == GPU_PREPARED_PINNED_HOST) {
                prepared.workspace = gpu_pinned_alloc(context, layout.bytes, layout.alignment);
                if (!prepared.workspace) return fail("failed to prepare pinned host storage");
            } else {
                error = cudaMallocAsync(&prepared.workspace, layout.bytes, release);
                if (error == cudaSuccess)
                    error = cudaEventCreateWithFlags(&prepared.reusable, cudaEventDisableTiming);
                if (error == cudaSuccess) error = cudaEventRecord(prepared.reusable, release);
            }
        }
        const cudaError_t restored = cudaSetDevice(previous);
        if (error != cudaSuccess) return fail(error);
        if (restored != cudaSuccess) return fail(restored);
        auto result = std::make_unique<GpuPreparedStorage>();
        result->value = storage;
        storage->owner = owner;
        storage->release_owner = release_owner;
        execution->prepared_storage_count.fetch_add(1, std::memory_order_acq_rel);
        storage->registered = true;
        *out = result.release();
        return 0;
    } catch (const std::exception &error) {
        cudaSetDevice(previous);
        return fail(error.what());
    }
}

extern "C" void gpu_prepared_storage_destroy(GpuPreparedStorage *storage)
{
    delete storage;
}

extern "C" int gpu_prepared_storage_matches_context(
    const GpuPreparedStorage *storage, const GpuContext *context)
{
    return storage && storage->value && context && storage->value->context == context;
}

extern "C" int gpu_prepared_storage_identity(
    const GpuPreparedStorage *storage, uint64_t *out_storage_id,
    uint64_t *out_execution_id, int *out_device)
{
    if (!storage || !storage->value || !out_storage_id || !out_execution_id || !out_device)
        return fail("invalid prepared matrix storage identity arguments");
    const auto &backing = *storage->value;
    *out_storage_id = backing.identity;
    *out_execution_id = backing.context->execution->identity;
    *out_device = backing.device;
    return 0;
}

extern "C" int gpu_prepared_slot_identity(
    const GpuPreparedStorage *storage, size_t slot,
    GpuPreparedSlotIdentity *out)
{
    if (!storage || !storage->value || !out || slot >= storage->value->slots.size())
        return fail("invalid prepared matrix slot identity arguments");
    *out = storage->value->slots[slot]->identity;
    return 0;
}

extern "C" int gpu_prepared_storage_occupancy(
    GpuPreparedStorage *storage, int reset_peak,
    GpuPreparedOccupancy *out)
{
    if (!storage || !storage->value || !out)
        return fail("invalid prepared matrix occupancy arguments");
    auto &backing = *storage->value;
    if (backing.context->execution->unretired_work.load(std::memory_order_acquire))
        return fail("prepared matrix occupancy has unretired work");
    struct ResetBoundary {
        Storage &backing;
        bool owned = false;
        ~ResetBoundary() {
            if (owned) backing.active_reservations.store(0, std::memory_order_release);
        }
    } boundary{backing};
    if (reset_peak) {
        size_t expected = 0;
        if (!backing.active_reservations.compare_exchange_strong(
                expected, resetting, std::memory_order_acq_rel))
            return fail("prepared occupancy reset requires no reservations or dispatches");
        boundary.owned = true;
    }
    int previous = -1;
    cudaError_t error = cudaGetDevice(&previous);
    if (error != cudaSuccess) return fail(error);
    error = cudaSetDevice(backing.device);
    if (error != cudaSuccess) return fail(error);
    bool pending = false;
    size_t available_bytes = 0;
    size_t pinned_available = 0;
    size_t resource_available = 0;
    for (auto &slot : backing.slots) {
        unsigned int expected = available;
        if (!slot->state.compare_exchange_strong(
                expected, inspecting, std::memory_order_acq_rel)) {
            // Leased outputs are allowed as retained baseline. Another poll or
            // a failed slot cannot establish a clean calibration boundary.
            pending |= expected != leased || slot->pinned_deferred.load(std::memory_order_acquire);
            continue;
        }
        if (slot->occupied_units != 0) {
            error = cudaEventQuery(slot->reusable);
            if (error == cudaSuccess) {
                backing.occupied_bytes.fetch_sub(
                    slot->occupied_units, std::memory_order_acq_rel);
                backing.context->execution->prepared_occupied_bytes.fetch_sub(
                    slot->occupied_units, std::memory_order_acq_rel);
                slot->occupied_units = 0;
            } else if (error == cudaErrorNotReady) {
                pending = true;
                error = cudaSuccess;
            } else {
                quarantine(backing);
                slot->state.store(quarantined, std::memory_order_release);
                break;
            }
        }
        if (slot->reservation_owner.load(std::memory_order_acquire) == 0) {
            if (is_resource(slot->identity.kind)) ++resource_available;
            else if (slot->identity.kind == GPU_PREPARED_PINNED_HOST)
                pinned_available += slot->identity.requested_backing_bytes;
            else available_bytes += slot->identity.requested_backing_bytes;
        }
        slot->state.store(available, std::memory_order_release);
    }
    const cudaError_t restored = cudaSetDevice(previous);
    if (restored != cudaSuccess) quarantine(backing);
    if (error != cudaSuccess) return fail(error);
    if (restored != cudaSuccess) return fail(restored);
    if (reset_peak && pending)
        return fail("prepared occupancy reset has pending releases or slot inspection");
    const size_t occupied = backing.occupied_bytes.load(std::memory_order_acquire);
    if (reset_peak)
        backing.occupied_high_water_bytes.store(occupied, std::memory_order_release);
    const size_t pinned_occupied = backing.pinned->occupied_bytes.load(std::memory_order_acquire);
    if (reset_peak) backing.pinned->high_water_bytes.store(pinned_occupied, std::memory_order_release);
    const size_t resource_occupied = backing.resources->occupied.load(std::memory_order_acquire);
    if (reset_peak) backing.resources->high_water.store(resource_occupied, std::memory_order_release);
    const size_t active = backing.active_reservations.load(std::memory_order_acquire);
    *out = GpuPreparedOccupancy{
        backing.requested_capacity_bytes,
        backing.reserved_bytes.load(std::memory_order_acquire), occupied,
        backing.occupied_high_water_bytes.load(std::memory_order_acquire),
        active == resetting ? 0 : active, available_bytes,
        backing.pinned->capacity_bytes, backing.pinned->reserved_bytes.load(std::memory_order_acquire),
        pinned_occupied, backing.pinned->high_water_bytes.load(std::memory_order_acquire), pinned_available,
        backing.resources->capacity, backing.resources->reserved.load(std::memory_order_acquire),
        resource_occupied, backing.resources->high_water.load(std::memory_order_acquire), resource_available};
    return 0;
}

extern "C" int gpu_prepared_storages_occupancy(
    GpuPreparedStorage *const *storages, size_t count, int reset_peak,
    size_t *out_occupied_bytes, size_t *out_high_water_bytes)
{
    if (!storages || count == 0 || !storages[0] || !storages[0]->value ||
        !out_occupied_bytes || !out_high_water_bytes)
        return fail("invalid joint prepared occupancy arguments");
    auto execution = storages[0]->value->context->execution;
    const int device = storages[0]->value->device;
    if (execution->gpu_ids.size() != 1)
        return fail("joint occupancy requires one physical device per execution owner");
    try {
        std::unordered_set<uint64_t> unique;
        for (size_t index = 0; index < count; ++index) {
            if (!storages[index] || !storages[index]->value)
                return fail("missing joint prepared storage");
            const auto &storage = *storages[index]->value;
            if (storage.context->execution != execution || storage.device != device ||
                !unique.insert(storage.identity).second)
                return fail("joint occupancy requires distinct stores on the same execution owner");
        }
        struct ResetBoundary {
            GpuExecutionOwner &execution;
            bool owned = false;
            ~ResetBoundary() {
                if (owned) execution.prepared_active_reservations.store(0, std::memory_order_release);
            }
        } boundary{*execution};
        if (reset_peak) {
            size_t expected = 0;
            if (!execution->prepared_active_reservations.compare_exchange_strong(
                    expected, resetting, std::memory_order_acq_rel))
                return fail("joint occupancy reset requires no reservations or dispatches");
            boundary.owned = true;
        }
        if (execution->prepared_storage_count.load(std::memory_order_acquire) != count)
            return fail("joint occupancy requires the complete live prepared inventory");
        for (size_t index = 0; index < count; ++index) {
            GpuPreparedOccupancy local{};
            const int status = gpu_prepared_storage_occupancy(storages[index], reset_peak, &local);
            if (status != 0) return status;
        }
        const size_t occupied = execution->prepared_occupied_bytes.load(std::memory_order_acquire);
        if (reset_peak) execution->prepared_high_water_bytes.store(occupied, std::memory_order_release);
        *out_occupied_bytes = occupied;
        *out_high_water_bytes = execution->prepared_high_water_bytes.load(std::memory_order_acquire);
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" int gpu_prepared_storages_finish_setup(
    GpuPreparedStorage *const *storages, size_t count)
{
    if (!storages || count == 0 || !storages[0] || !storages[0]->value)
        return fail("managed setup requires a nonempty prepared inventory");
    const auto execution = storages[0]->value->context->execution;
    if (execution->gpu_ids.size() != 1)
        return fail("managed setup requires one device per execution owner");
    if (execution->unretired_work.load(std::memory_order_acquire) ||
        execution->memory_release_failed.load(std::memory_order_acquire) ||
        execution->allocation_activity_unknown.load(std::memory_order_seq_cst))
        return fail("managed setup has uncertain allocation or release activity");
    if (execution->timing_active.load(std::memory_order_acquire) ||
        execution->prepared_active_reservations.load(std::memory_order_acquire) != 0 ||
        execution->active_allocation_calls.load(std::memory_order_seq_cst) != 0)
        return fail("managed setup requires no outstanding host allocations or reservations");
    try {
        std::unordered_set<uint64_t> identities;
        for (size_t index = 0; index < count; ++index) {
            if (!storages[index] || !storages[index]->value)
                return fail("managed setup contains missing storage");
            const auto &storage = *storages[index]->value;
            if (storage.context->execution != execution ||
                storage.device != execution->gpu_ids[0] ||
                !identities.insert(storage.identity).second)
                return fail("managed setup requires distinct stores on the same execution owner");
            for (const auto &slot : storage.slots) {
                if (slot->reservation_owner.load(std::memory_order_acquire) != 0 ||
                    slot->state.load(std::memory_order_acquire) != available)
                    return fail("managed setup has a reserved, leased or quarantined slot");
            }
        }
        if (execution->prepared_storage_count.load(std::memory_order_acquire) != count)
            return fail("managed setup requires the complete live prepared inventory");
        // The audited managed allocation sites all require one of these domains.
        // Close every domain before making coherent observation available. A
        // concurrent older allocation remains visible to the epoch counters;
        // no receipt is issued here and no physical upper bound is asserted.
        execution->admission_required.store(true, std::memory_order_release);
        execution->pinned_admission_required.store(true, std::memory_order_release);
        execution->transfer_admission_required.store(true, std::memory_order_release);
        execution->resource_admission_required.store(true, std::memory_order_release);
        execution->allocation_tracking_complete.store(true, std::memory_order_release);
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" int gpu_matrix_reserve(
    GpuPreparedStorage *storage, const GpuPreparedRequest *requests, size_t count,
    GpuMatrixReservation **out)
{
    if (!out) return fail("null matrix reservation output");
    *out = nullptr;
    if (!storage || !storage->value || (count && !requests))
        return fail("invalid matrix reservation");
    auto &backing = *storage->value;
    if (backing.context->execution->unretired_work.load(std::memory_order_acquire))
        return fail("matrix reservation has unretired work");
    try {
        auto reservation = std::make_unique<GpuMatrixReservation>(storage->value);
        bool fits = false;
        const int status = prepare_claims(backing, requests, count, reservation->claims, fits);
        if (status != 0) return status;
        if (!fits) return fail("prepared request does not fit its backing layout");
        reservation->slots.reserve(count);
        reservation->bounds.reserve(count);
        for (const auto &claim : reservation->claims) {
            reservation->slots.push_back(claim.request.slot_index);
            reservation->bounds.push_back(claim.request);
        }
        for (; reservation->claimed < count; ++reservation->claimed) {
            unsigned int expected = available;
            const size_t slot = reservation->slots[reservation->claimed];
            uint64_t owner = 0;
            if (!backing.slots[slot]->reservation_owner.compare_exchange_strong(
                    owner, reservation->identity, std::memory_order_acq_rel))
                return fail("prepared slot belongs to another invocation");
            if (!backing.slots[slot]->state.compare_exchange_strong(
                    expected, reserved, std::memory_order_acq_rel)) {
                backing.slots[slot]->reservation_owner.store(0, std::memory_order_release);
                return fail("prepared matrix slot is still owned by another reservation or output");
            }
            reserved_counter(backing, *backing.slots[slot]).fetch_add(
                reservation_units(*backing.slots[slot]),
                std::memory_order_acq_rel);
        }
        backing.context->execution->admission_required.store(true, std::memory_order_release);
        for (const auto &claim : reservation->claims) {
            if (claim.request.kind == GPU_PREPARED_PINNED_HOST) {
                backing.context->execution->pinned_admission_required.store(true, std::memory_order_release);
            }
            if (claim.request.kind == GPU_PREPARED_TRANSFER_WORKSPACE)
                backing.context->execution->transfer_admission_required.store(true, std::memory_order_release);
            if (is_resource(claim.request.kind))
                backing.context->execution->resource_admission_required.store(true, std::memory_order_release);
        }
        *out = reservation.release();
        return 0;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

extern "C" int gpu_prepared_storage_demand(
    const GpuPreparedStorage *storage, const GpuPreparedRequest *requests,
    size_t count, GpuPreparedDemand *out)
{
    if (!storage || !storage->value || !out)
        return fail("invalid prepared demand arguments");
    *out = {};
    try {
        std::vector<PreparedClaim> claims;
        bool fits = false;
        const int status = prepare_claims(*storage->value, requests, count, claims, fits);
        if (status != 0) return status;
        if (!fits) return fail("prepared demand exceeds its native layout");
        GpuPreparedDemand demand{};
        for (const auto &claim : claims) {
            auto &total = is_resource(claim.request.kind) ? demand.resource_slots :
                claim.request.kind == GPU_PREPARED_PINNED_HOST ? demand.pinned_bytes : demand.device_bytes;
            if (claim.occupied_units > SIZE_MAX - total)
                return fail("prepared demand overflow");
            total += claim.occupied_units;
        }
        *out = demand;
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" int gpu_matrix_reservation_require_all_resources(GpuMatrixReservation *reservation)
{
    if (!reservation || reservation->next != 0 ||
        reservation->execution->unretired_work.load(std::memory_order_acquire))
        return fail("complete resource coverage requires an unsubmitted reservation");
    auto &owner = *reservation->execution;
    owner.admission_required.store(true, std::memory_order_release);
    owner.pinned_admission_required.store(true, std::memory_order_release);
    owner.transfer_admission_required.store(true, std::memory_order_release);
    owner.resource_admission_required.store(true, std::memory_order_release);
    return 0;
}

extern "C" int gpu_prepared_storage_fits(
    const GpuPreparedStorage *storage, const GpuPreparedRequest *requests, size_t count,
    int *out_fits)
{
    if (!storage || !storage->value || !out_fits)
        return fail("invalid prepared fit arguments");
    *out_fits = 0;
    const auto &backing = *storage->value;
    if (backing.context->execution->unretired_work.load(std::memory_order_acquire))
        return fail("prepared fit has unretired work");
    try {
        std::vector<PreparedClaim> claims;
        bool fits = false;
        const int status = prepare_claims(backing, requests, count, claims, fits);
        if (status != 0 || !fits) return status;
        for (const auto &claim : claims)
            if (backing.slots[claim.request.slot_index]->reservation_owner.load(std::memory_order_acquire) != 0 ||
                backing.slots[claim.request.slot_index]->state.load(std::memory_order_acquire) != available)
                return 0;
        *out_fits = 1;
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" void gpu_matrix_reservation_destroy(GpuMatrixReservation *reservation)
{
    delete reservation;
}

extern "C" int gpu_matrix_reservation_matches_context(
    const GpuMatrixReservation *reservation, const GpuContext *context)
{
    return reservation && context && reservation->storage->context == context;
}

extern "C" int gpu_matrix_reservation_rearm(
    GpuMatrixReservation *reservation, const GpuPreparedRequest *requests, size_t count)
{
    if (!reservation || (reservation->next != 0 && reservation->next != reservation->claimed) ||
        count != reservation->slots.size() || (count && !requests))
        return fail("rearm requires an unsubmitted or complete dispatch and its original slot count");
    if (reservation->execution->unretired_work.load(std::memory_order_acquire))
        return fail("reservation rearm has unretired work");
    auto &storage = *reservation->storage;
    try {
        std::vector<PreparedClaim> claims;
        bool fits = false;
        const int status = prepare_claims(storage, requests, count, claims, fits);
        if (status != 0) return status;
        if (!fits) return fail("rearmed request does not fit its backing layout");
        for (size_t index = 0; index < count; ++index) {
            const auto &request = requests[index];
            const auto &bound = reservation->bounds[index];
            if (request.storage_id != bound.storage_id || request.slot_id != bound.slot_id ||
                request.slot_index != bound.slot_index || request.kind != bound.kind ||
                request.level != bound.level || request.format != bound.format ||
                request.rows > bound.rows || request.columns > bound.columns ||
                request.bytes > bound.bytes || request.alignment > bound.alignment)
                return fail("rearmed request exceeds its original admitted envelope");
        }
        if (reservation->next == 0) {
            // An armed request can only be refined before submission, not
            // rearmed unchanged or widened. Completed waves use the original
            // admitted envelope below and may grow again within that bound.
            bool narrower = false;
            for (size_t index = 0; index < count; ++index) {
                const auto &previous = reservation->claims[index].request;
                const auto &request = requests[index];
                if (request.rows > previous.rows || request.columns > previous.columns ||
                    request.bytes > previous.bytes || request.alignment > previous.alignment)
                    return fail("unsubmitted request specialization must narrow its current layout");
                narrower |= request.rows < previous.rows || request.columns < previous.columns ||
                    request.bytes < previous.bytes || request.alignment < previous.alignment;
            }
            if (!narrower) return fail("reservation is already armed with this request");
            // The initial envelope is already reserved. Specializing it for a
            // narrower first range changes no slot ownership or capacity charge.
            // Bounds remain the original invocation envelope, not this request.
            reservation->claims.swap(claims);
            return 0;
        }
        size_t acquired = 0;
        for (; acquired < count; ++acquired) {
            auto &slot = *storage.slots[reservation->slots[acquired]];
            unsigned int expected = available;
            if (!slot.state.compare_exchange_strong(expected, reserved, std::memory_order_acq_rel))
                break;
            reserved_counter(storage, slot).fetch_add(
                reservation_units(slot), std::memory_order_acq_rel);
        }
        if (acquired != count) {
            for (size_t index = 0; index < acquired; ++index) {
                auto &slot = *storage.slots[reservation->slots[index]];
                reserved_counter(storage, slot).fetch_sub(
                    reservation_units(slot), std::memory_order_acq_rel);
                slot.state.store(available, std::memory_order_release);
            }
            return fail("previous wave still owns a prepared slot or it is being inspected");
        }
        reservation->claims.swap(claims);
        reservation->next = 0;
        return 0;
    } catch (const std::exception &error) { return fail(error.what()); }
}

extern "C" int gpu_matrix_reservation_partition(
    GpuMatrixReservation *reservation, const size_t *counts, size_t child_count,
    GpuMatrixReservation **out_children)
{
    if (!reservation || reservation->next != 0 ||
        reservation->claimed != reservation->slots.size() ||
        (child_count != 0 && (!counts || !out_children)))
        return fail("invalid prepared reservation partition");
    if (reservation->execution->unretired_work.load(std::memory_order_acquire))
        return fail("reservation partition has unretired work");
    size_t total = 0;
    for (size_t index = 0; index < child_count; ++index) {
        out_children[index] = nullptr;
        if (counts[index] > SIZE_MAX - total)
            return fail("prepared reservation partition overflow");
        total += counts[index];
    }
    if (total != reservation->slots.size())
        return fail("prepared child plans must cover the complete reservation");
    try {
        std::vector<std::unique_ptr<GpuMatrixReservation>> children;
        children.reserve(child_count);
        size_t offset = 0;
        for (size_t index = 0; index < child_count; ++index) {
            auto child = std::make_unique<GpuMatrixReservation>(reservation->storage);
            child->slots.assign(reservation->slots.begin() + offset,
                                reservation->slots.begin() + offset + counts[index]);
            child->claims.assign(reservation->claims.begin() + offset,
                                 reservation->claims.begin() + offset + counts[index]);
            child->bounds.assign(reservation->bounds.begin() + offset,
                                 reservation->bounds.begin() + offset + counts[index]);
            offset += counts[index];
            children.push_back(std::move(child));
        }
        // Every fallible host allocation has completed. Until this transfer,
        // only the parent owns the slot claims and the children roll back no
        // slots. Neither reservation counters nor slot states change here.
        for (auto &child : children) {
            for (size_t slot : child->slots)
                child->storage->slots[slot]->reservation_owner.store(
                    child->identity, std::memory_order_release);
            child->claimed = child->slots.size();
        }
        reservation->claimed = 0;
        for (size_t index = 0; index < child_count; ++index)
            out_children[index] = children[index].release();
        delete reservation;
        return 0;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

extern "C" int gpu_matrix_dispatch_enter(
    GpuMatrixReservation *const *reservations, size_t count, GpuMatrixDispatchPermit **out)
{
    if (!out) return fail("null matrix dispatch permit output");
    *out = nullptr;
    if (!reservations || count == 0 || active_permit)
        return fail("invalid or nested matrix dispatch activation");
    try {
        std::unordered_set<GpuMatrixReservation *> unique;
        for (size_t index = 0; index < count; ++index) {
            const auto *reservation = reservations[index];
            if (!reservation || reservation->next != 0 || !unique.insert(reservations[index]).second)
                return fail("invalid, consumed or duplicate matrix dispatch reservation");
            if (reservation->execution->unretired_work.load(std::memory_order_acquire))
                return fail("matrix dispatch has unretired work");
            if (reservation->execution != reservations[0]->execution ||
                reservation->storage->device != reservations[0]->storage->device)
                return fail("composed dispatch requires one device and execution owner");
        }
        auto permit = std::make_unique<GpuMatrixDispatchPermit>();
        permit->reservations.resize(count);
        permit->entered = count;
        // All fallible host preparation precedes ownership transfer. Every
        // reservation keeps its slot identities and invocation holds unchanged.
        for (size_t index = 0; index < count; ++index)
            permit->reservations[index].reset(reservations[index]);
        active_permit = permit.release();
        *out = active_permit;
        return 0;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

extern "C" int gpu_matrix_dispatch_active()
{
    return active_permit ? 1 : 0;
}

// Append ordered reservations to the active permit after all of its earlier
// claims are consumed. Host-driven steps of an admitted job hold their exact
// extra claims this way without nesting a second permit.
extern "C" int gpu_matrix_dispatch_extend(
    GpuMatrixReservation *const *reservations, size_t count, size_t *out_base)
{
    if (!out_base) return fail("null matrix dispatch extension output");
    *out_base = 0;
    if (!reservations || count == 0 || !active_permit)
        return fail("matrix dispatch extension requires an active permit");
    if (active_permit->next_reservation() != nullptr)
        return fail("matrix dispatch extension requires the earlier claims to be consumed");
    try {
        std::unordered_set<GpuMatrixReservation *> unique;
        const auto *anchor = active_permit->reservations.front().get();
        for (size_t index = 0; index < count; ++index) {
            const auto *reservation = reservations[index];
            if (!reservation || reservation->next != 0 || !unique.insert(reservations[index]).second)
                return fail("invalid, consumed or duplicate matrix dispatch reservation");
            if (reservation->execution->unretired_work.load(std::memory_order_acquire))
                return fail("matrix dispatch has unretired work");
            if (reservation->execution != anchor->execution ||
                reservation->storage->device != anchor->storage->device)
                return fail("dispatch extension requires the permit's device and execution owner");
        }
        *out_base = active_permit->reservations.size();
        active_permit->reservations.reserve(*out_base + count);
        for (size_t index = 0; index < count; ++index)
            active_permit->reservations.emplace_back(reservations[index]);
        return 0;
    } catch (const std::exception &error) {
        return fail(error.what());
    }
}

// Remove the most recent extension. On success every extension claim must be
// consumed; retained reservations return to the caller like a completed dispatch.
extern "C" int gpu_matrix_dispatch_retract(
    size_t base, size_t count, int successful, GpuMatrixReservation **out)
{
    if (!active_permit || active_permit->reservations.size() != base + count || count == 0)
        return fail("matrix dispatch retraction does not match the active extension");
    bool complete = true;
    for (size_t index = base; index < base + count; ++index) {
        auto &reservation = active_permit->reservations[index];
        if (reservation->next != reservation->slots.size()) complete = false;
    }
    for (size_t index = 0; index < count; ++index) {
        auto &reservation = active_permit->reservations[base + index];
        if (out) out[index] = successful && complete ? reservation.release() : nullptr;
        reservation.reset();
    }
    active_permit->reservations.resize(base);
    if (active_permit->current > base) active_permit->current = base;
    return successful && !complete
        ? fail("matrix dispatch extension did not consume its prepared slots") : 0;
}

extern "C" int gpu_matrix_dispatch_end(
    GpuMatrixDispatchPermit *permit, int successful, GpuMatrixReservation **out)
{
    if (!permit || permit != active_permit)
        return fail("matrix dispatch permit must end on its submitting thread");
    if (permit->reservations.size() != permit->entered)
        return fail("matrix dispatch permit ended with a live extension");
    const bool complete = permit->next_reservation() == nullptr;
    active_permit = nullptr;
    if (out) {
        for (size_t index = 0; index < permit->reservations.size(); ++index)
            out[index] = successful && complete ? permit->reservations[index].release() : nullptr;
    }
    delete permit;
    return successful && !complete ? fail("matrix dispatch did not consume its prepared slots") : 0;
}

extern "C" int gpu_prepared_matrix_claim(
    GpuContext *ctx, int level, size_t rows, size_t cols, int format,
    GpuMatrix **out, int *handled)
{
    if (!ctx || !out || !handled) return fail("invalid prepared matrix allocation");
    *handled = 0;
    if (!active_permit && !gpu_context_admission_is_required(ctx)) {
        if (rows != 0 && cols != 0) trace_claim(GPU_PREPARED_MATRIX, rows, cols, level, format, 0, 0);
        return 0;
    }
    *handled = 1;
    if (!active_permit)
        return fail("ordinary matrix allocation requires a matching prepared dispatch permit");
    // Empty native matrices own only host metadata and consume no device slot.
    if (rows == 0 || cols == 0) {
        for (const auto &reservation : active_permit->reservations) {
            if (reservation->storage->context == ctx) {
                *handled = 0;
                return 0;
            }
        }
        return fail("empty matrix context is absent from the prepared dispatch");
    }
    auto *next = active_permit->next_reservation();
    if (!next)
        return fail("ordinary matrix allocation exceeds its prepared dispatch slots");
    if (next->storage->context != ctx)
        return fail("ordinary matrix allocation differs from its next reserved context");
    auto &reservation = *next;
    auto &storage = *reservation.storage;
    const size_t slot_index = reservation.slots[reservation.next];
    auto &slot = *storage.slots[slot_index];
    const auto &claim = reservation.claims[reservation.next];
    GpuMatrix *matrix = slot.matrix;
    if (slot.identity.kind != GPU_PREPARED_MATRIX || !matrix ||
        claim.request.rows != rows || claim.request.columns != cols ||
        claim.request.level != level || claim.request.format != format) {
        static thread_local std::string message;
        char head[256];
        std::snprintf(head, sizeof(head),
            "ordinary matrix allocation differs from its reserved shape or format (reserved kind %d %zux%zu level %d format %d; requested %zux%zu level %d format %d); consumed so far: ",
            static_cast<int>(slot.identity.kind), claim.request.rows, claim.request.columns,
            claim.request.level, claim.request.format, rows, cols, level, format);
        try {
            message = std::string(head) + consumed_summary();
        } catch (const std::exception &) {
            return fail(head);
        }
        return fail(message.c_str());
    }
    std::unique_ptr<GpuPreparedMatrixLease> lease;
    try {
        lease = std::make_unique<GpuPreparedMatrixLease>(
            GpuPreparedMatrixLease{reservation.storage, slot_index});
    } catch (const std::exception &error) { return fail(error.what()); }
    cudaError_t error = cudaSetDevice(storage.device);
    if (error != cudaSuccess) return fail(error);
    // Charge before the first device submission. A prior reader may still be
    // using this same physical backing; the following stream waits transfer its
    // ownership without counting that backing twice.
    occupy(storage, slot, claim.occupied_units);
    auto &states = matrix->exec_limb_states[0];
    for (size_t limb = 0; limb < states.size(); ++limb) {
        auto &state = states[limb];
        if (!state.write_done || !state.stream) {
            error = cudaErrorInvalidResourceHandle;
            break;
        }
        error = cudaStreamWaitEvent(state.stream, slot.reusable, 0);
        if (error == cudaSuccess) error = cudaEventRecord(state.write_done, state.stream);
        if (error != cudaSuccess) break;
        state.completion_owner = static_cast<uint32_t>(limb);
        state.last_write_stream = state.stream;
        state.write_done_valid = true;
    }
    if (error != cudaSuccess) {
        quarantine(storage);
        slot.state.store(quarantined, std::memory_order_release);
        // Retain the backing even when CUDA cannot establish reuse ordering.
        matrix->shared_limb_buffers[0].prepared_lease = lease.release();
        ++reservation.next;
        return fail(error);
    }
    // Only the shape-visible prefixes change. Per-polynomial descriptors,
    // physical backing, auxiliary/descriptor pointers and original streams stay
    // fixed, so queued readers keep their original addresses and strides.
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->shared_limb_buffers[0].bytes_total = claim.data_bytes;
    matrix->shared_aux_buffers[0].slots_total = claim.auxiliary_slots;
    matrix->format = static_cast<GpuPolyFormat>(format);
    matrix->host_observed_writer_ready.store(false, std::memory_order_release);
    matrix->shared_limb_buffers[0].prepared_lease = lease.release();
    slot.state.store(leased, std::memory_order_release);
    ++reservation.next;
    note_consumed(GPU_PREPARED_MATRIX, rows, cols, level, format, 0, 0);
    *out = matrix;
    return 0;
}

extern "C" void gpu_prepared_matrix_recycle(GpuMatrix *matrix)
{
    auto *lease = matrix->shared_limb_buffers[0].prepared_lease;
    if (!lease) return;
    auto &storage = *lease->storage;
    auto &slot = *storage.slots[lease->slot];
    // Deleting the last lease can run the Rust backing callback and destroy its
    // contexts. Keep both the outer activity and execution alive across it.
    auto execution = storage.context->execution;
    GpuAllocationActivity activity(execution.get(), storage.device);
    if (storage.context->execution->unretired_work.load(std::memory_order_acquire)) return;
    const cudaStream_t release = storage.context->execution->release_streams_by_partition[0];
    cudaError_t error = cudaSetDevice(storage.device);
    int status = error == cudaSuccess
        ? matrix_wait_all_limb_streams(matrix, storage.device, release, true, true) : fail(error);
    if (status == 0) {
        error = cudaEventRecord(slot.reusable, release);
        if (error != cudaSuccess) status = fail(error);
    }
    if (status != 0) {
        quarantine(storage);
        slot.state.store(quarantined, std::memory_order_release);
        return;
    }
    matrix->shared_limb_buffers[0].prepared_lease = nullptr;
    slot.state.store(available, std::memory_order_release);
    delete lease;
}
