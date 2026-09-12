#pragma once

#include "Runtime.cuh"

struct GpuMatrix;
struct GpuPreparedMatrixLease;
struct GpuPreparedStorage;
struct GpuMatrixReservation;
struct GpuMatrixDispatchPermit;
struct GpuPreparedWorkspaceLease;
struct GpuPreparedPinnedLease;
struct GpuPreparedResourceLease;

enum GpuPreparedSlotKind {
    GPU_PREPARED_MATRIX = 0,
    GPU_PREPARED_BATCH_WORKSPACE = 1,
    GPU_PREPARED_TRANSFORM_WORKSPACE = 2,
    GPU_PREPARED_PINNED_HOST = 3,
    GPU_PREPARED_COMPACT_PAYLOAD = 4,
    GPU_PREPARED_COMPACT_WORKSPACE = 5,
    GPU_PREPARED_SAMPLER_WORKSPACE = 6,
    GPU_PREPARED_TRANSFER_WORKSPACE = 7,
    GPU_PREPARED_COMPLETION_EVENT = 8,
    // Stream and its reusable producer/consumer bridge event form one slot.
    GPU_PREPARED_SUBMISSION_STREAM = 9,
};

/// One claim an operation would make on a closed domain, recorded on an open
/// domain in claim order. Matrix entries carry shape, level and format; other
/// kinds carry bytes and alignment.
struct GpuClaimTraceEntry {
    int kind;
    size_t rows;
    size_t columns;
    int level;
    int format;
    size_t bytes;
    size_t alignment;
};

struct GpuPreparedWorkspaceLayout {
    size_t bytes;
    size_t alignment;
    GpuPreparedSlotKind kind;
};

// Concrete request bound to one native slot identity. Matrix shape/format or
// workspace size/alignment is frozen at reservation and checked at actual use.
struct GpuPreparedRequest {
    uint64_t storage_id;
    uint64_t slot_id;
    size_t slot_index;
    size_t rows;
    size_t columns;
    size_t bytes;
    size_t alignment;
    int level;
    int format;
    GpuPreparedSlotKind kind;
};

// Stable native identities, never pointer-derived. A backing ID identifies the
// entire owned backing group of one slot. Requested bytes exclude opaque CUDA
// resources and allocator rounding; they are not a physical residency receipt.
struct GpuPreparedSlotIdentity {
    uint64_t storage_id;
    uint64_t backing_id;
    uint64_t slot_id;
    size_t slot_index;
    size_t rows;
    size_t columns;
    size_t payload_bytes;
    size_t auxiliary_bytes;
    size_t requested_backing_bytes;
    int level;
    GpuPreparedSlotKind kind;
    size_t alignment;
};

// Logical request bytes of whole backing groups, not physical CUDA residency.
// Reserved bytes cover unconsumed claims. Occupancy begins at actual claim and
// persists through pending readers. Reusing the same backing counts it once.
struct GpuPreparedOccupancy {
    size_t requested_capacity_bytes;
    size_t reserved_bytes;
    size_t occupied_bytes;
    size_t occupied_high_water_bytes;
    size_t active_reservations;
    // Available for event-ordered device reuse, including pending prior readers.
    // A diagnostic candidate hint, not an atomic reservation receipt.
    size_t available_capacity_bytes;
    // Host transfer spans are a separate resource dimension, not VRAM bytes.
    size_t pinned_capacity_bytes;
    size_t pinned_reserved_bytes;
    size_t pinned_occupied_bytes;
    size_t pinned_high_water_bytes;
    size_t pinned_available_bytes;
    // Opaque resources have slot counts, never fictitious VRAM byte sizes.
    size_t resource_capacity_slots;
    size_t resource_reserved_slots;
    size_t resource_occupied_slots;
    size_t resource_high_water_slots;
    size_t resource_available_slots;
};

extern "C" {

// Explicit setup only. The caller transfers ownership of the backing matrices
// through owner/release_owner on success. Optional typed workspaces, missing
// matrix completion events and slot reuse events are allocated at this boundary.
// Their actual residency must be charged before production begins. This is an
// matrix/workspace contract, not a bound on other primitive allocations.
int gpu_prepared_storage_create(
    GpuMatrix *const *matrices, size_t count,
    const GpuPreparedWorkspaceLayout *workspaces, size_t workspace_count, void *owner,
    void (*release_owner)(void *), GpuPreparedStorage **out);
void gpu_prepared_storage_destroy(GpuPreparedStorage *storage);
int gpu_prepared_storage_identity(
    const GpuPreparedStorage *storage, uint64_t *out_storage_id,
    uint64_t *out_execution_id, int *out_device);
int gpu_prepared_storage_matches_context(
    const GpuPreparedStorage *storage, const GpuContext *context);
int gpu_prepared_slot_identity(
    const GpuPreparedStorage *storage, size_t slot,
    GpuPreparedSlotIdentity *out);
// Queries completed reuse events without a host wait. Concurrent submissions
// make the fields diagnostic, not a coherent capacity/admission receipt.
// Reset requires no outstanding reservation/dispatch and rejects pending
// releases. Callers must retain all fixed baseline owners throughout the pilot.
// New reservations racing a reset fail rather than waiting.
int gpu_prepared_storage_occupancy(
    GpuPreparedStorage *storage, int reset_peak,
    GpuPreparedOccupancy *out);

// Joint device-span peak across the complete live inventory on one physical
// execution owner. Duplicate, incomplete and foreign-owner lists are rejected.
// Reset excludes all reservations on that owner; pending releases reject it.
// The caller retains fixed baseline owners and excludes unrelated pilot work.
int gpu_prepared_storages_occupancy(
    GpuPreparedStorage *const *storages, size_t count, int reset_peak,
    size_t *out_occupied_bytes, size_t *out_high_water_bytes);

// Explicit setup transition over the complete native inventory of one execution
// owner. Reject live reservations/leases and close every managed allocation
// domain permanently. A subsequent quiescent allocation epoch, not this call,
// establishes initial residency. Opaque CUDA demand is outside this contract.
int gpu_prepared_storages_finish_setup(GpuPreparedStorage *const *storages, size_t count);

// Atomically claims the complete slot list, without activating thread-local
// dispatch. The token can move to another host thread. Any failure rolls back
// every claim made by this call. The first successful reservation permanently
// requires ordinary-allocation permits; cancellation does not disable that mode.
// The allocation activity scope spans reservation, activation, and teardown.
int gpu_matrix_reserve(
    GpuPreparedStorage *storage, const GpuPreparedRequest *requests, size_t count,
    GpuMatrixReservation **out);
// Nonallocating on the device, nonblocking fit over concrete layouts and free
// slots. This is advisory under concurrency: reserve rechecks and claims all
// slots atomically before any participant is published.
int gpu_prepared_storage_fits(
    const GpuPreparedStorage *storage, const GpuPreparedRequest *requests, size_t count,
    int *out_fits);
// Exact logical demand, independent of the observed high-water mark. Separate
// device bytes, pinned bytes and opaque resource units; this is not VRAM size.
typedef struct GpuPreparedDemand {
    size_t device_bytes;
    size_t pinned_bytes;
    size_t resource_slots;
} GpuPreparedDemand;
int gpu_prepared_storage_demand(
    const GpuPreparedStorage *storage, const GpuPreparedRequest *requests,
    size_t count, GpuPreparedDemand *out);
// Close all managed allocation domains before running a complete invocation.
// Applies to every related context/worker and remains active after cancellation.
// This does not certify unbounded driver/launch resources or physical residency.
int gpu_matrix_reservation_require_all_resources(GpuMatrixReservation *reservation);
// Exact related-ring context match, before any compiled invocation submits work.
int gpu_matrix_reservation_matches_context(
    const GpuMatrixReservation *reservation, const GpuContext *context);
void gpu_matrix_reservation_destroy(GpuMatrixReservation *reservation);
// Strictly narrow an unsubmitted request or reuse a completed reservation without
// exposing its slots to other callers. Previous owners must have recycled;
// pending GPU readers remain ordered
// by the existing reuse events. Requests must fit the original admitted bounds.
// Failure preserves the reservation and its complete exclusive footprint.
int gpu_matrix_reservation_rearm(
    GpuMatrixReservation *reservation, const GpuPreparedRequest *requests, size_t count);
// Split a complete unactivated reservation into ordered, disjoint child plans
// for existing CPU workers. No slot becomes available during the transfer.
// Consumes the parent only on success; errors leave every claim with it.
int gpu_matrix_reservation_partition(
    GpuMatrixReservation *reservation, const size_t *counts, size_t child_count,
    GpuMatrixReservation **out_children);
// Consumes every ordered reservation on success only. All must belong to the
// same device/execution owner; distinct prepared stores and related contexts
// are supported. Failure leaves all tokens with the caller. The resulting
// permit must be ended on this same host thread.
int gpu_matrix_dispatch_enter(
    GpuMatrixReservation *const *reservations, size_t count,
    GpuMatrixDispatchPermit **out);
// On successful complete dispatch, a nonnull out retains the reservation for
// each input in its original order. Its capacity must equal the input count.
// Null out cancels all reservations, including during unwinding.
int gpu_matrix_dispatch_end(
    GpuMatrixDispatchPermit *permit, int successful, GpuMatrixReservation **out);
// Whether this thread holds an active permit; extend/retract add ordered
// reservations to it after its earlier claims are consumed.
int gpu_matrix_dispatch_active();
int gpu_matrix_dispatch_extend(
    GpuMatrixReservation *const *reservations, size_t count, size_t *out_base);
int gpu_matrix_dispatch_retract(
    size_t base, size_t count, int successful, GpuMatrixReservation **out);

// Record the ordered native claims of operations on this thread while the
// context's domains are still open. `end` copies at most `capacity` entries
// and reports the complete count.
// Record a claim that production makes only under sealed admission (an event
// or stream a kernel otherwise creates directly), so traces stay exact.
void gpu_claim_trace_record(
    int kind, size_t rows, size_t cols, int level, int format, size_t bytes, size_t alignment);
void gpu_claim_deterministic_events_push();
void gpu_claim_deterministic_events_pop();
int gpu_claim_deterministic_events_active();
int gpu_claim_trace_begin();
int gpu_claim_trace_end(GpuClaimTraceEntry *out, size_t capacity, size_t *count);

// MatrixData hooks. handled distinguishes the standalone allocation path.
int gpu_prepared_matrix_claim(
    GpuContext *ctx, int level, size_t rows, size_t cols, int format,
    GpuMatrix **out, int *handled);
void gpu_prepared_matrix_recycle(GpuMatrix *matrix);

// Managed pinned-pool hooks. CPU reuse becomes available only after its actual
// owner drops or the existing reclaimer has observed DMA completion.
int gpu_prepared_pinned_claim(
    GpuContext *ctx, size_t bytes, size_t alignment, void **out, int *handled);
int gpu_pinned_bind_prepared(void *pointer, GpuPreparedPinnedLease *lease);
int gpu_prepared_pinned_defer(GpuPreparedPinnedLease *lease, GpuContext *ctx);
void gpu_prepared_pinned_recycle(GpuPreparedPinnedLease *lease);

}

// A single-stream device workspace. Prepared dispatch consumes the next typed
// slot; standalone dispatch keeps the default asynchronous allocator. Both
// paths release after their last consumer without a host completion wait.
// Acquisition borrows ctx; retirement retains execution resources independently.
struct GpuDeviceWorkspace {
    GpuDeviceWorkspace();
    GpuDeviceWorkspace(const GpuDeviceWorkspace &) = delete;
    GpuDeviceWorkspace &operator=(const GpuDeviceWorkspace &) = delete;
    ~GpuDeviceWorkspace();
    int acquire(GpuContext *ctx, int device, GpuPreparedSlotKind kind,
                size_t bytes, size_t alignment, cudaStream_t stream);
    int release(cudaStream_t completed_stream = nullptr);
    cudaEvent_t completion_event() const;
    uint8_t *data;
private:
    std::shared_ptr<GpuExecutionOwner> execution;
    int device;
    cudaStream_t stream;
    GpuPreparedWorkspaceLease *lease;
};

// Exclusive CUDA handle ownership. Event users must finish submitting every
// wait and relinquish every host query before releasing the handle. A stream
// owner joins submitted work to the execution owner before releasing its slot.
// Slot leases retain their resource independently of the matrix inventory.
struct GpuCudaResource {
    GpuCudaResource();
    GpuCudaResource(const GpuCudaResource &) = delete;
    GpuCudaResource &operator=(const GpuCudaResource &) = delete;
    ~GpuCudaResource();
    int acquire(const GpuContext *ctx, int device, GpuPreparedSlotKind kind);
    int release();
    void quarantine();
    // The execution-owned reclaimer is joined by its owner's destructor. Its
    // jobs must not retain that owner and initiate a self-join on the worker.
    void detach_execution();
    cudaEvent_t event;
    cudaStream_t stream;
private:
    std::shared_ptr<GpuExecutionOwner> execution;
    int device;
    GpuPreparedResourceLease *lease;
};
