//! Prepared matrices, typed device workspaces and managed pinned transfer buffers.
//!
//! Storage consumes real backing owners prepared at an explicit setup boundary.
//! The caller must charge their actual residency and the completion resources
//! prepared by `new` before admitting production. Only setup grows device
//! backing; requested bytes never infer physical allocator rounding.
//! Other native scratch, raw native pinned buffers, and private events still require their
//! own production requirement and enforcement before full invocation admission.

use super::GpuDCRTPolyMatrix;
use crate::poly::dcrt::gpu::{
    GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuMatrixOpaque, last_error_string,
};
use std::{cell::Cell, ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

#[repr(C)]
struct PreparedStorageOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct ReservationOpaque {
    _private: [u8; 0],
}

#[repr(C)]
struct DispatchOpaque {
    _private: [u8; 0],
}

/// Owns the automatic runtime graph boundary. Native output leases remain
/// alive independently; dropping this guard only ends the allocation policy.
pub struct GpuGraphAdmissionGuard {
    parameters: Vec<crate::poly::dcrt::gpu::GpuDCRTPolyParams>,
}

impl GpuGraphAdmissionGuard {
    pub fn new(parameters: Vec<crate::poly::dcrt::gpu::GpuDCRTPolyParams>) -> Result<Self, String> {
        let mut guard = Self { parameters: Vec::with_capacity(parameters.len()) };
        for parameters in parameters {
            if unsafe { gpu_graph_admission_begin(parameters.ctx_raw()) } != 0 {
                return Err(last_error_string());
            }
            guard.parameters.push(parameters);
        }
        Ok(guard)
    }
}

impl Drop for GpuGraphAdmissionGuard {
    fn drop(&mut self) {
        for parameters in &self.parameters {
            unsafe { gpu_graph_admission_end(parameters.ctx_raw()) };
        }
    }
}

unsafe extern "C" {
    fn gpu_graph_admission_begin(context: *mut crate::poly::dcrt::gpu::GpuContextOpaque) -> i32;
    fn gpu_graph_admission_end(context: *mut crate::poly::dcrt::gpu::GpuContextOpaque);
}

/// One native claim recorded by [`trace_native_claims`], in claim order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuTracedClaim {
    kind: i32,
    rows: usize,
    columns: usize,
    level: i32,
    format: i32,
    bytes: usize,
    alignment: usize,
}

impl GpuTracedClaim {
    /// A matrix owner claim of the given shape, level and format.
    pub fn matrix(rows: usize, columns: usize, level: usize, evaluation: bool) -> Self {
        Self {
            kind: 0,
            rows,
            columns,
            level: i32::try_from(level).unwrap_or(-1),
            format: if evaluation { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF },
            bytes: 0,
            alignment: 0,
        }
    }

    /// A workspace or CUDA resource claim described by a layout.
    pub fn workspace(layout: GpuPreparedWorkspaceLayout) -> Self {
        Self {
            kind: layout.kind as i32,
            rows: 0,
            columns: 0,
            level: -1,
            format: -1,
            bytes: layout.bytes,
            alignment: layout.alignment,
        }
    }

    /// The same claim for a different column count; matrix owners change their
    /// shape, other kinds are unchanged.
    pub fn with_columns(mut self, columns: usize) -> Self {
        if self.kind == 0 {
            self.columns = columns;
        }
        self
    }

    pub fn kind(&self) -> GpuPreparedSlotKind {
        match self.kind {
            0 => GpuPreparedSlotKind::Matrix,
            1 => GpuPreparedSlotKind::BatchWorkspace,
            2 => GpuPreparedSlotKind::TransformWorkspace,
            3 => GpuPreparedSlotKind::PinnedHost,
            4 => GpuPreparedSlotKind::CompactPayload,
            5 => GpuPreparedSlotKind::CompactWorkspace,
            6 => GpuPreparedSlotKind::SamplerWorkspace,
            7 => GpuPreparedSlotKind::TransferWorkspace,
            8 => GpuPreparedSlotKind::CompletionEvent,
            _ => GpuPreparedSlotKind::SubmissionStream,
        }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn level(&self) -> Option<usize> {
        usize::try_from(self.level).ok()
    }
    pub fn is_evaluation(&self) -> Option<bool> {
        match self.format {
            GPU_POLY_FORMAT_EVAL => Some(true),
            GPU_POLY_FORMAT_COEFF => Some(false),
            _ => None,
        }
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    /// The workspace layout this claim needs when it is not a matrix owner.
    pub fn layout(&self) -> Option<GpuPreparedWorkspaceLayout> {
        (self.kind() != GpuPreparedSlotKind::Matrix).then(|| GpuPreparedWorkspaceLayout {
            kind: self.kind(),
            bytes: self.bytes,
            alignment: self.alignment.max(1),
        })
    }
}

/// Marks this thread as executing a traced step: consumer tracking claims its
/// completion event on every path, matching the recorded plan.
/// Thread-bound: the flag it holds is thread-local, so the guard is `!Send`.
pub struct GpuTracedStepGuard(PhantomData<Rc<()>>);

impl GpuTracedStepGuard {
    pub fn enter() -> Self {
        unsafe { gpu_claim_deterministic_events_push() };
        Self(PhantomData)
    }
}

impl Drop for GpuTracedStepGuard {
    fn drop(&mut self) {
        unsafe { gpu_claim_deterministic_events_pop() };
    }
}

/// Run `operation` on this thread while its execution owner's allocation
/// domains are still open, and return the ordered native claims it would make
/// on a closed domain. The operation itself allocates normally.
pub fn trace_native_claims<T>(
    operation: impl FnOnce() -> T,
) -> Result<(T, Vec<GpuTracedClaim>), String> {
    /// Ends the thread-local trace if `operation` unwinds, so a later trace on
    /// this thread starts clean.
    struct TraceEnd(bool);
    impl Drop for TraceEnd {
        fn drop(&mut self) {
            if self.0 {
                let mut count = 0usize;
                unsafe { gpu_claim_trace_end(std::ptr::null_mut(), 0, &mut count) };
            }
        }
    }
    if unsafe { gpu_claim_trace_begin() } != 0 {
        return Err(last_error_string());
    }
    let mut trace = TraceEnd(true);
    let value = {
        let _deterministic = GpuTracedStepGuard::enter();
        operation()
    };
    trace.0 = false;
    let mut count = 0usize;
    if unsafe { gpu_claim_trace_end(std::ptr::null_mut(), 0, &mut count) } != 0 {
        return Err(last_error_string());
    }
    let mut claims = vec![
        GpuTracedClaim {
            kind: 0,
            rows: 0,
            columns: 0,
            level: -1,
            format: -1,
            bytes: 0,
            alignment: 0
        };
        count
    ];
    let mut copied = 0usize;
    if unsafe { gpu_claim_trace_end(claims.as_mut_ptr(), claims.len(), &mut copied) } != 0 {
        return Err(last_error_string());
    }
    claims.truncate(copied.min(count));
    Ok((value, claims))
}

unsafe extern "C" {
    fn gpu_matrix_dispatch_active() -> i32;
    fn gpu_matrix_dispatch_extend(
        reservations: *const *mut ReservationOpaque,
        count: usize,
        out_base: *mut usize,
    ) -> i32;
    fn gpu_matrix_dispatch_retract(
        base: usize,
        count: usize,
        successful: i32,
        out: *mut *mut ReservationOpaque,
    ) -> i32;
    fn gpu_claim_deterministic_events_push();
    fn gpu_claim_deterministic_events_pop();
    fn gpu_claim_trace_begin() -> i32;
    fn gpu_claim_trace_end(out: *mut GpuTracedClaim, capacity: usize, count: *mut usize) -> i32;
    fn gpu_prepared_storages_finish_setup(
        storages: *const *mut PreparedStorageOpaque,
        count: usize,
    ) -> i32;

    fn gpu_prepared_storage_matches_context(
        storage: *const PreparedStorageOpaque,
        context: *const crate::poly::dcrt::gpu::GpuContextOpaque,
    ) -> i32;
    fn gpu_matrix_reservation_matches_context(
        reservation: *const ReservationOpaque,
        context: *const crate::poly::dcrt::gpu::GpuContextOpaque,
    ) -> i32;
    fn gpu_prepared_storages_occupancy(
        storages: *const *mut PreparedStorageOpaque,
        count: usize,
        reset_peak: i32,
        occupied_bytes: *mut usize,
        high_water_bytes: *mut usize,
    ) -> i32;
    fn gpu_prepared_storage_demand(
        storage: *const PreparedStorageOpaque,
        requests: *const GpuPreparedRequest,
        count: usize,
        out: *mut GpuPreparedDemand,
    ) -> i32;
    fn gpu_matrix_reservation_require_all_resources(reservation: *mut ReservationOpaque) -> i32;
    fn gpu_matrix_query_equality_workspace(out: *mut GpuPreparedWorkspaceLayout) -> i32;
    fn gpu_matrix_query_compact_workspace(
        context: *mut crate::poly::dcrt::gpu::GpuContextOpaque,
        level: i32,
        rows: usize,
        columns: usize,
        matrices: usize,
        kind: i32,
        max_coeff_bits: u16,
        out: *mut GpuPreparedWorkspaceLayout,
    ) -> i32;
    fn gpu_matrix_query_rns_workspace(
        context: *mut crate::poly::dcrt::gpu::GpuContextOpaque,
        level: i32,
        rows: usize,
        columns: usize,
        out: *mut GpuPreparedWorkspaceLayout,
    ) -> i32;
    fn gpu_matrix_query_p1_workspaces(
        context: *mut crate::poly::dcrt::gpu::GpuContextOpaque,
        rows: usize,
        columns: usize,
        cached: i32,
        out: *mut GpuPreparedWorkspaceLayout,
    ) -> i32;
    fn gpu_matrix_query_gaussian_gadget_workspaces(
        context: *mut crate::poly::dcrt::gpu::GpuContextOpaque,
        level: i32,
        rows: usize,
        columns: usize,
        base_bits: u32,
        out: *mut GpuPreparedWorkspaceLayout,
    ) -> i32;
    fn gpu_prepared_storage_create(
        matrices: *const *mut GpuMatrixOpaque,
        count: usize,
        workspaces: *const GpuPreparedWorkspaceLayout,
        workspace_count: usize,
        owner: *mut c_void,
        release_owner: unsafe extern "C" fn(*mut c_void),
        out: *mut *mut PreparedStorageOpaque,
    ) -> i32;
    fn gpu_prepared_storage_destroy(storage: *mut PreparedStorageOpaque);
    fn gpu_prepared_storage_identity(
        storage: *const PreparedStorageOpaque,
        out_storage_id: *mut u64,
        out_execution_id: *mut u64,
        out_device: *mut i32,
    ) -> i32;
    fn gpu_prepared_slot_identity(
        storage: *const PreparedStorageOpaque,
        slot: usize,
        out: *mut GpuPreparedSlotIdentity,
    ) -> i32;
    fn gpu_prepared_storage_occupancy(
        storage: *mut PreparedStorageOpaque,
        reset_peak: i32,
        out: *mut GpuPreparedOccupancy,
    ) -> i32;
    fn gpu_matrix_reserve(
        storage: *mut PreparedStorageOpaque,
        requests: *const GpuPreparedRequest,
        count: usize,
        out: *mut *mut ReservationOpaque,
    ) -> i32;
    fn gpu_prepared_storage_fits(
        storage: *const PreparedStorageOpaque,
        requests: *const GpuPreparedRequest,
        count: usize,
        out_fits: *mut i32,
    ) -> i32;
    fn gpu_matrix_reservation_destroy(reservation: *mut ReservationOpaque);
    fn gpu_matrix_reservation_rearm(
        reservation: *mut ReservationOpaque,
        requests: *const GpuPreparedRequest,
        count: usize,
    ) -> i32;
    fn gpu_matrix_reservation_cpu_ready(
        reservation: *const ReservationOpaque,
        out_ready: *mut i32,
    ) -> i32;
    fn gpu_matrix_reservation_partition(
        reservation: *mut ReservationOpaque,
        counts: *const usize,
        child_count: usize,
        out_children: *mut *mut ReservationOpaque,
    ) -> i32;
    fn gpu_matrix_dispatch_enter(
        reservations: *const *mut ReservationOpaque,
        count: usize,
        out: *mut *mut DispatchOpaque,
    ) -> i32;
    fn gpu_matrix_dispatch_end(
        permit: *mut DispatchOpaque,
        successful: i32,
        out: *mut *mut ReservationOpaque,
    ) -> i32;
}

unsafe extern "C" fn release_backing(owner: *mut c_void) {
    // The native storage invokes this exactly once after its last dispatch and
    // output lease. The backing wrappers retain their real CUDA allocations and
    // parameter contexts throughout that lifetime.
    drop(unsafe { Box::from_raw(owner.cast::<Vec<GpuDCRTPolyMatrix>>()) });
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuPreparedSlotKind {
    Matrix = 0,
    BatchWorkspace = 1,
    TransformWorkspace = 2,
    PinnedHost = 3,
    CompactPayload = 4,
    CompactWorkspace = 5,
    SamplerWorkspace = 6,
    TransferWorkspace = 7,
    CompletionEvent = 8,
    /// One submission stream with its producer/consumer bridge event.
    SubmissionStream = 9,
}

/// One typed device/host buffer or opaque CUDA resource allocated at setup.
/// Events and streams require zero bytes and alignment one; each consumes one slot.
/// Buffer alignment must be a nonzero power of two at most 256; kind must be a workspace
/// kind. Native claims can use up to this capacity with compatible alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuPreparedWorkspaceLayout {
    pub bytes: usize,
    pub alignment: usize,
    pub kind: GpuPreparedSlotKind,
}

/// Codec operation whose complete device scratch is one bounded transfer span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCompactTransferKind {
    Store,
    StoreBatch { matrices: usize },
    Load { max_coefficient_bits: u16 },
}

impl GpuDCRTPolyMatrix {
    /// One comparison result span per active device. CRT limbs reuse it after
    /// their existing host readback; no pointer table is allocated.
    pub fn equality_workspace() -> Result<GpuPreparedWorkspaceLayout, String> {
        let mut layout = std::mem::MaybeUninit::<GpuPreparedWorkspaceLayout>::uninit();
        let status = unsafe { gpu_matrix_query_equality_workspace(layout.as_mut_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layout.assume_init() })
    }
}

impl crate::poly::dcrt::gpu::GpuDCRTPolyParams {
    /// Device descriptors, reconstruction and payload capacity for compact
    /// transfer. Store capacity depends on the modulus, not observed values.
    pub fn compact_transfer_workspace(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
        kind: GpuCompactTransferKind,
    ) -> Result<GpuPreparedWorkspaceLayout, String> {
        let (kind, matrices, bits) = match kind {
            GpuCompactTransferKind::Store => (0, 1, 0),
            GpuCompactTransferKind::StoreBatch { matrices } => (1, matrices, 0),
            GpuCompactTransferKind::Load { max_coefficient_bits } => (2, 1, max_coefficient_bits),
        };
        let mut layout = std::mem::MaybeUninit::<GpuPreparedWorkspaceLayout>::uninit();
        let status = unsafe {
            gpu_matrix_query_compact_workspace(
                self.ctx_raw(),
                i32::try_from(level).map_err(|_| "compact level overflow")?,
                rows,
                columns,
                matrices,
                kind,
                bits,
                layout.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layout.assume_init() })
    }

    /// Exact device pack/unpack span used by a single-device RNS transfer.
    /// The matrix, host buffer and completion resources have separate claims.
    pub fn rns_transfer_workspace(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
    ) -> Result<GpuPreparedWorkspaceLayout, String> {
        let mut layout = std::mem::MaybeUninit::<GpuPreparedWorkspaceLayout>::uninit();
        let status = unsafe {
            gpu_matrix_query_rns_workspace(
                self.ctx_raw(),
                i32::try_from(level).map_err(|_| "RNS level overflow")?,
                rows,
                columns,
                layout.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layout.assume_init() })
    }

    /// Sampled integers and optional large-dimension scratch, frozen for the
    /// submitted P1 column range. Cache and completion resources are separate.
    pub fn p1_sampling_workspaces(
        &self,
        rows: usize,
        columns: usize,
        cached: bool,
    ) -> Result<[GpuPreparedWorkspaceLayout; 2], String> {
        let mut layouts = std::mem::MaybeUninit::<[GpuPreparedWorkspaceLayout; 2]>::uninit();
        let status = unsafe {
            gpu_matrix_query_p1_workspaces(
                self.ctx_raw(),
                rows,
                columns,
                i32::from(cached),
                layouts.as_mut_ptr().cast(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layouts.assume_init() })
    }

    /// Exact logical device spans consumed by Gaussian gadget sampling, in
    /// dispatch order. Completion resources and physical overhead are separate.
    pub fn gaussian_gadget_workspaces(
        &self,
        level: usize,
        rows: usize,
        columns: usize,
    ) -> Result<[GpuPreparedWorkspaceLayout; 5], String> {
        use crate::poly::PolyParams;
        let level = i32::try_from(level).map_err(|_| "Gaussian gadget level overflow")?;
        let mut layouts = std::mem::MaybeUninit::<[GpuPreparedWorkspaceLayout; 5]>::uninit();
        let status = unsafe {
            gpu_matrix_query_gaussian_gadget_workspaces(
                self.ctx_raw(),
                level,
                rows,
                columns,
                self.base_bits(),
                layouts.as_mut_ptr().cast(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { layouts.assume_init() })
    }
}

/// A concrete allocation request tied to a native slot identity. Construct via
/// the identity's matrix_request or workspace_request, then validate/reserve it.
/// Actual native use must match the reserved shape, format, size and alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPreparedRequest {
    storage_id: u64,
    slot_id: u64,
    slot_index: usize,
    rows: usize,
    columns: usize,
    bytes: usize,
    alignment: usize,
    level: i32,
    format: i32,
    kind: GpuPreparedSlotKind,
}

impl GpuPreparedRequest {
    /// Native slot identity for detecting competing claims in a candidate plan.
    pub fn slot_key(&self) -> (u64, u64, usize) {
        (self.storage_id, self.slot_id, self.slot_index)
    }

    pub fn kind(&self) -> GpuPreparedSlotKind {
        self.kind
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    pub fn level(&self) -> Option<usize> {
        usize::try_from(self.level).ok()
    }
    /// Evaluation (NTT) format for matrix requests; `None` for other kinds.
    pub fn is_evaluation(&self) -> Option<bool> {
        match self.format {
            GPU_POLY_FORMAT_EVAL => Some(true),
            GPU_POLY_FORMAT_COEFF => Some(false),
            _ => None,
        }
    }

    /// Check a concrete request against a pre-width layout envelope on the same
    /// native slot. This neither checks current availability nor reserves it;
    /// native fitting and reservation must still validate the resulting list.
    pub fn fits_bound(&self, bound: &Self) -> bool {
        if self.storage_id != bound.storage_id ||
            self.slot_id != bound.slot_id ||
            self.slot_index != bound.slot_index ||
            self.kind != bound.kind ||
            self.level != bound.level ||
            self.format != bound.format
        {
            return false;
        }
        match self.kind {
            GpuPreparedSlotKind::Matrix => {
                self.rows <= bound.rows &&
                    self.columns <= bound.columns &&
                    self.bytes == bound.bytes &&
                    self.alignment == bound.alignment
            }
            _ => {
                self.rows == bound.rows &&
                    self.columns == bound.columns &&
                    self.bytes <= bound.bytes &&
                    self.alignment <= bound.alignment &&
                    self.alignment.is_power_of_two() &&
                    bound.alignment.is_power_of_two()
            }
        }
    }
}

/// Stable native identity and request layout of one prepared backing group.
///
/// Each slot owns a distinct group; slots cannot alias one another. The group
/// includes the matrix payload and its auxiliary/descriptor storage. Its byte
/// counts describe allocator requests, not rounded physical residency or opaque
/// event/driver resources. Copying this metadata does not retain the backing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuPreparedSlotIdentity {
    storage_id: u64,
    backing_id: u64,
    slot_id: u64,
    slot_index: usize,
    rows: usize,
    columns: usize,
    payload_bytes: usize,
    auxiliary_bytes: usize,
    requested_backing_bytes: usize,
    level: i32,
    kind: GpuPreparedSlotKind,
    alignment: usize,
}

impl GpuPreparedSlotIdentity {
    /// Bind a matrix shape within this backing's maximum dimensions. Native fit
    /// checks capacity and retains the original level, descriptors and streams.
    pub fn matrix_request(self, rows: usize, columns: usize, is_ntt: bool) -> GpuPreparedRequest {
        GpuPreparedRequest {
            storage_id: self.storage_id,
            slot_id: self.slot_id,
            slot_index: self.slot_index,
            rows,
            columns,
            bytes: 0,
            alignment: 0,
            level: self.level,
            format: if is_ntt { GPU_POLY_FORMAT_EVAL } else { GPU_POLY_FORMAT_COEFF },
            kind: GpuPreparedSlotKind::Matrix,
        }
    }

    /// Bind a request to this workspace's kind. Native fit validates the size
    /// and alignment; constructing a request alone claims no storage.
    pub fn workspace_request(self, bytes: usize, alignment: usize) -> GpuPreparedRequest {
        GpuPreparedRequest {
            storage_id: self.storage_id,
            slot_id: self.slot_id,
            slot_index: self.slot_index,
            rows: 0,
            columns: 0,
            bytes,
            alignment,
            level: -1,
            format: -1,
            kind: self.kind,
        }
    }

    pub fn storage_id(&self) -> u64 {
        self.storage_id
    }
    pub fn backing_id(&self) -> u64 {
        self.backing_id
    }
    pub fn slot_id(&self) -> u64 {
        self.slot_id
    }
    pub fn slot_index(&self) -> usize {
        self.slot_index
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn level(&self) -> Option<usize> {
        usize::try_from(self.level).ok()
    }
    pub fn kind(&self) -> GpuPreparedSlotKind {
        self.kind
    }
    pub fn alignment(&self) -> usize {
        self.alignment
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
    pub fn auxiliary_bytes(&self) -> usize {
        self.auxiliary_bytes
    }
    pub fn requested_backing_bytes(&self) -> usize {
        self.requested_backing_bytes
    }
}

/// Native logical demand for exact ordered claims. These independent dimensions
/// are not additive: CUDA resource units and pinned bytes are not device bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuPreparedDemand {
    pub device_bytes: usize,
    pub pinned_bytes: usize,
    pub resource_slots: usize,
}

/// Logical requested bytes in prepared matrix and device workspace groups.
///
/// Reservation alone does not occupy storage. Actual claims remain occupied
/// until their outputs and pending readers retire, or transfer to the next user
/// of the same backing. Smaller claims count only their used span; pending reuse
/// retains the largest unretired span in that backing once. Reserved bytes can
/// overlap occupied bytes during that transfer; do not add these two counters.
/// These counters exclude opaque CUDA resources and physical allocator rounding.
/// They neither prove physical admission nor replace native shape/slot fitting.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuPreparedOccupancy {
    requested_capacity_bytes: usize,
    reserved_bytes: usize,
    occupied_bytes: usize,
    occupied_high_water_bytes: usize,
    active_reservations: usize,
    available_capacity_bytes: usize,
    pinned_capacity_bytes: usize,
    pinned_reserved_bytes: usize,
    pinned_occupied_bytes: usize,
    pinned_high_water_bytes: usize,
    pinned_available_bytes: usize,
    resource_capacity_slots: usize,
    resource_reserved_slots: usize,
    resource_occupied_slots: usize,
    resource_high_water_slots: usize,
    resource_available_slots: usize,
}

/// Boundary for the execution owner's joint device-byte high-water counter.
#[derive(Clone, Copy)]
#[repr(i32)]
pub enum GpuPreparedOccupancyMode {
    Observe = 0,
    /// Requires no outstanding reservations and no pending releases.
    ResetCalibration = 1,
    /// Explicit benchmark boundary with parked reservations and retained outputs.
    /// The caller must finish preceding releases and exclude all submissions and
    /// dispatch entry while resetting. This mode never authorizes admission.
    ResetMeasurement = 2,
}

impl GpuPreparedOccupancy {
    pub fn resource_capacity_slots(&self) -> usize {
        self.resource_capacity_slots
    }
    pub fn resource_reserved_slots(&self) -> usize {
        self.resource_reserved_slots
    }
    pub fn resource_occupied_slots(&self) -> usize {
        self.resource_occupied_slots
    }
    pub fn resource_high_water_slots(&self) -> usize {
        self.resource_high_water_slots
    }
    pub fn resource_available_slots(&self) -> usize {
        self.resource_available_slots
    }

    pub fn requested_capacity_bytes(&self) -> usize {
        self.requested_capacity_bytes
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }
    pub fn occupied_bytes(&self) -> usize {
        self.occupied_bytes
    }
    pub fn occupied_high_water_bytes(&self) -> usize {
        self.occupied_high_water_bytes
    }
    pub fn active_reservations(&self) -> usize {
        self.active_reservations
    }
    /// Available for event-ordered device reuse; pending reader completion does
    /// not prevent a subsequent device claim. Concurrent activity makes this a
    /// candidate hint only. Native fit/reserve remain authoritative.
    pub fn available_capacity_bytes(&self) -> usize {
        self.available_capacity_bytes
    }
    pub fn pinned_capacity_bytes(&self) -> usize {
        self.pinned_capacity_bytes
    }
    pub fn pinned_reserved_bytes(&self) -> usize {
        self.pinned_reserved_bytes
    }
    pub fn pinned_occupied_bytes(&self) -> usize {
        self.pinned_occupied_bytes
    }
    pub fn pinned_high_water_bytes(&self) -> usize {
        self.pinned_high_water_bytes
    }
    /// CPU-ready capacity only: pending DMA keeps its host slot unavailable.
    pub fn pinned_available_bytes(&self) -> usize {
        self.pinned_available_bytes
    }
}

/// Owns existing whole matrices followed by typed setup-allocated workspaces.
///
/// Matrix claims may reduce rows/columns within the original capacity, retaining
/// the native level, context, descriptors and producer streams. Shape-visible
/// transfer/auxiliary lengths follow the claim, independent of physical backing.
/// Live outputs keep their slots exclusive even after dispatch returns.
pub struct GpuPreparedStorage {
    raw: NonNull<PreparedStorageOpaque>,
    identity: u64,
    execution_owner_id: u64,
    device: i32,
    slots: Vec<GpuPreparedSlotIdentity>,
}

// Native slots are claimed atomically; each leased matrix has one owning Rust
// wrapper and different slots may be submitted on independent host threads.
unsafe impl Send for GpuPreparedStorage {}
unsafe impl Sync for GpuPreparedStorage {}

impl GpuPreparedStorage {
    /// Prepare matrix completion resources and optional typed buffers/events/streams.
    ///
    /// Call only at checked setup, before the execution owner's first prepared
    /// reservation. This consumes matrices without cloning their device data.
    /// Actual backing and opaque-resource residency must be accounted for by the
    /// caller; successful construction alone is not physical memory admission.
    pub fn new(
        backing: Vec<GpuDCRTPolyMatrix>,
        workspaces: Option<&[GpuPreparedWorkspaceLayout]>,
    ) -> Result<Self, String> {
        let workspaces = workspaces.unwrap_or(&[]);
        let pointers = backing.iter().map(|matrix| matrix.raw).collect::<Vec<_>>();
        let slot_count =
            pointers.len().checked_add(workspaces.len()).ok_or("slot count overflow")?;
        let owner = Box::into_raw(Box::new(backing));
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_prepared_storage_create(
                pointers.as_ptr(),
                pointers.len(),
                workspaces.as_ptr(),
                workspaces.len(),
                owner.cast(),
                release_backing,
                &mut raw,
            )
        };
        if status != 0 {
            let error = last_error_string();
            // Ownership transfers only on successful native preparation.
            drop(unsafe { Box::from_raw(owner) });
            return Err(error);
        }
        let raw = NonNull::new(raw).expect("successful native storage has an owner");
        let mut storage = Self {
            raw,
            identity: 0,
            execution_owner_id: 0,
            device: -1,
            slots: Vec::with_capacity(slot_count),
        };
        let status = unsafe {
            gpu_prepared_storage_identity(
                raw.as_ptr(),
                &mut storage.identity,
                &mut storage.execution_owner_id,
                &mut storage.device,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        for index in 0..slot_count {
            let mut identity = std::mem::MaybeUninit::uninit();
            let status =
                unsafe { gpu_prepared_slot_identity(raw.as_ptr(), index, identity.as_mut_ptr()) };
            if status != 0 {
                return Err(last_error_string());
            }
            storage.slots.push(unsafe { identity.assume_init() });
        }
        Ok(storage)
    }

    /// Finish managed setup over the complete live inventory of one execution
    /// owner, including related parameter rings. No reservations or leased slots
    /// may remain. This permanently requires exact prepared allocation/resource
    /// claims; pending setup transfers and their tracked host reclamation may
    /// continue. It does not wait for GPU work or establish residency acceptance.
    /// Acquire an initial allocation epoch next and apply both setup budget checks
    /// before dispatch. Later opaque CUDA growth is outside managed accounting.
    pub fn finish_setup(storages: &[&Self]) -> Result<(), String> {
        let pointers = storages.iter().map(|storage| storage.raw.as_ptr()).collect::<Vec<_>>();
        let status =
            unsafe { gpu_prepared_storages_finish_setup(pointers.as_ptr(), pointers.len()) };
        if status == 0 { Ok(()) } else { Err(last_error_string()) }
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn device(&self) -> i32 {
        self.device
    }
    pub fn execution_owner_id(&self) -> u64 {
        self.execution_owner_id
    }
    /// Match the exact related-ring context without reservation or submission.
    pub fn matches_parameters(&self, params: &crate::poly::dcrt::gpu::GpuDCRTPolyParams) -> bool {
        unsafe { gpu_prepared_storage_matches_context(self.raw.as_ptr(), params.ctx_raw()) != 0 }
    }
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
    pub fn slot_identity(&self, slot: usize) -> Option<GpuPreparedSlotIdentity> {
        self.slots.get(slot).copied()
    }

    /// Compute exact logical demand with the same native layouts as reserve.
    /// Does not acquire slots, inspect pool counters, or submit GPU commands.
    pub fn demand(&self, requests: &[GpuPreparedRequest]) -> Result<GpuPreparedDemand, String> {
        let mut demand = GpuPreparedDemand::default();
        let status = unsafe {
            gpu_prepared_storage_demand(
                self.raw.as_ptr(),
                requests.as_ptr(),
                requests.len(),
                &mut demand,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(demand)
    }

    /// Check concrete native layouts and available slots without GPU allocation,
    /// submission, or waiting. Concurrent claims can invalidate this observation;
    /// reserve rechecks every layout and acquires the complete list atomically.
    /// Physical sealing and other resource families remain separate obligations.
    pub fn fits(&self, requests: &[GpuPreparedRequest]) -> Result<bool, String> {
        let mut fits = 0;
        let status = unsafe {
            gpu_prepared_storage_fits(
                self.raw.as_ptr(),
                requests.as_ptr(),
                requests.len(),
                &mut fits,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(fits != 0)
    }

    /// Nonblocking observation, retiring logical charges only for available
    /// slots whose reuse events have completed. Concurrent dispatch makes the
    /// fields diagnostic: use reservations, not a snapshot, to claim capacity.
    pub fn occupancy(&self) -> Result<GpuPreparedOccupancy, String> {
        let mut result = std::mem::MaybeUninit::uninit();
        let status =
            unsafe { gpu_prepared_storage_occupancy(self.raw.as_ptr(), 0, result.as_mut_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { result.assume_init() })
    }

    /// Actual joint device-span occupancy and high-water bytes for every live
    /// prepared store on one execution owner, including related parameter rings.
    /// This does not sum independent peaks. Pinned bytes and opaque resource
    /// counts remain separate from this device-byte metric.
    ///
    /// Calibration reset rejects outstanding reservations or pending releases.
    /// Measurement reset permits parked reservations under the exclusive,
    /// completed-release contract of `GpuPreparedOccupancyMode::ResetMeasurement`.
    /// Keep fixed owners alive and exclude unrelated submissions throughout
    /// measurement. Ordinary observations are diagnostic under concurrency;
    /// use exact reservations for admission.
    pub fn joint_occupancy(
        stores: &[&Self],
        mode: GpuPreparedOccupancyMode,
    ) -> Result<(usize, usize), String> {
        let stores = stores.iter().map(|storage| storage.raw.as_ptr()).collect::<Vec<_>>();
        let mut occupied = 0;
        let mut peak = 0;
        let status = unsafe {
            gpu_prepared_storages_occupancy(
                stores.as_ptr(),
                stores.len(),
                mode as i32,
                &mut occupied,
                &mut peak,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok((occupied, peak))
    }

    /// Start an explicit calibration boundary without waiting for GPU work.
    ///
    /// Rejects outstanding reservation/dispatch tokens and pending releases.
    /// Retain every fixed baseline output until the pilot has finished; only
    /// then does peak minus baseline describe incremental logical pressure.
    /// A caller may fence prior releases at its explicit measurement boundary
    /// and retry. Production dispatch never needs this reset or a host wait.
    pub fn reset_occupied_high_water(&mut self) -> Result<GpuPreparedOccupancy, String> {
        let mut result = std::mem::MaybeUninit::uninit();
        let status =
            unsafe { gpu_prepared_storage_occupancy(self.raw.as_ptr(), 1, result.as_mut_ptr()) };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(unsafe { result.assume_init() })
    }

    /// Reserve an exact ordered allocation list without activating a worker.
    ///
    /// The returned token owns its reservation and may move to another thread.
    /// Reserve every participant before publishing any work; dropping a partial
    /// collection rolls back its slots if another device's reservation fails.
    /// The first successful reservation permanently requires ordinary-allocation
    /// permits on this owner. Cancellation does not downgrade that checked mode.
    pub fn reserve(&self, requests: &[GpuPreparedRequest]) -> Result<GpuMatrixReservation, String> {
        let identities = requests
            .iter()
            .map(|request| {
                self.slot_identity(request.slot_index)
                    .ok_or_else(|| "prepared matrix slot index out of bounds".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let requests = requests.to_vec();
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_reserve(self.raw.as_ptr(), requests.as_ptr(), requests.len(), &mut raw)
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(GpuMatrixReservation {
            raw: Some(NonNull::new(raw).expect("successful reservation has an owner")),
            slots: identities,
            requests,
            not_sync: PhantomData,
        })
    }
}

impl Drop for GpuPreparedStorage {
    fn drop(&mut self) {
        unsafe { gpu_prepared_storage_destroy(self.raw.as_ptr()) };
    }
}

/// An owned reservation that can move to its target worker before activation.
/// It retains exclusive slot ownership across completed waves until dropped.
/// Dropping returns idle slots; live matrices retain their separate reader edges.
/// Neither action waits for the GPU or mutates another thread's dispatch.
#[must_use]
pub struct GpuMatrixReservation {
    raw: Option<NonNull<ReservationOpaque>>,
    slots: Vec<GpuPreparedSlotIdentity>,
    requests: Vec<GpuPreparedRequest>,
    not_sync: PhantomData<Cell<()>>,
}

// Native reservation bookkeeping and retained backing owners are thread-safe.
// Activation consumes this unique token; it cannot be entered concurrently.
unsafe impl Send for GpuMatrixReservation {}

impl GpuMatrixReservation {
    /// Match the exact native parameter context, including related rings with
    /// equal tower counts. Equal logical parameters alone do not suffice.
    pub fn matches_parameters(&self, params: &crate::poly::dcrt::gpu::GpuDCRTPolyParams) -> bool {
        unsafe {
            gpu_matrix_reservation_matches_context(
                self.raw.expect("live reservation").as_ptr(),
                params.ctx_raw(),
            ) != 0
        }
    }

    /// Require exact permits for every managed device, pinned and CUDA resource
    /// allocation on this execution owner, including its related contexts and
    /// existing workers. This mode persists after cancellation. It is necessary
    /// for a complete pilot/invocation, but is not a physical-memory seal.
    pub fn require_all_resources(&mut self) -> Result<(), String> {
        let status = unsafe {
            gpu_matrix_reservation_require_all_resources(
                self.raw.expect("live reservation").as_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(())
    }

    pub fn slot_identities(&self) -> &[GpuPreparedSlotIdentity] {
        &self.slots
    }

    /// Exact current requests, distinct from each slot's maximum backing shape.
    /// Preserved through worker transfer, partition, dispatch and completion.
    pub fn requests(&self) -> &[GpuPreparedRequest] {
        &self.requests
    }

    /// Nonblocking readiness of this reservation's deferred pinned CPU leases.
    /// Called by the wave scheduler before reuse, never inside an upload wrapper.
    /// Device-only scratch retains its existing stream-ordered reuse semantics.
    pub fn cpu_staging_ready(&self) -> Result<bool, String> {
        let mut ready = 0;
        let status = unsafe {
            gpu_matrix_reservation_cpu_ready(
                self.raw.expect("live reservation").as_ptr(),
                &mut ready,
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        Ok(ready != 0)
    }

    /// Narrow an unsubmitted request or prepare another wave within the original
    /// bounds. An unchanged or wider request requires completing the armed wave.
    /// Previous matrix owners must have been dropped; GPU readers need not have
    /// completed, because actual claims retain the existing event dependencies.
    /// A rejected request keeps the complete footprint exclusive and retryable.
    pub fn rearm(&mut self, requests: &[GpuPreparedRequest]) -> Result<(), String> {
        let updated = requests.to_vec();
        let status = unsafe {
            gpu_matrix_reservation_rearm(
                self.raw.expect("unconsumed reservation").as_ptr(),
                requests.as_ptr(),
                requests.len(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        self.requests = updated;
        Ok(())
    }

    /// Transfer a fleet-reserved plan into disjoint ordered CPU subjobs.
    ///
    /// The counts must cover every slot, including typed workspaces. Children
    /// can move into existing Rayon jobs and activate on their assigned worker.
    /// There is no release/re-reserve window, new device allocation, or new pool.
    /// Each child cancels only its own unused slots. Invalid partitions cancel
    /// this entire unactivated parent, just like failed activation does.
    pub fn partition(mut self, counts: &[usize]) -> Result<Vec<Self>, String> {
        let total = counts
            .iter()
            .try_fold(0usize, |total, count| total.checked_add(*count))
            .ok_or("prepared reservation partition overflow")?;
        if total != self.slots.len() {
            return Err("prepared child plans must cover the complete reservation".into());
        }
        let mut offset = 0;
        let identities = counts
            .iter()
            .map(|&count| {
                let slots = self.slots[offset..offset + count].to_vec();
                let requests = self.requests[offset..offset + count].to_vec();
                offset += count;
                (slots, requests)
            })
            .collect::<Vec<_>>();
        let mut raw = vec![std::ptr::null_mut(); counts.len()];
        let mut children = Vec::with_capacity(counts.len());
        let status = unsafe {
            gpu_matrix_reservation_partition(
                self.raw.expect("unconsumed reservation").as_ptr(),
                counts.as_ptr(),
                counts.len(),
                raw.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(last_error_string());
        }
        self.raw = None;
        for (raw, (slots, requests)) in raw.into_iter().zip(identities) {
            children.push(Self {
                raw: Some(NonNull::new(raw).expect("successful partition has child owners")),
                slots,
                requests,
                not_sync: PhantomData,
            });
        }
        Ok(children)
    }

    /// Activate this reservation followed by the given ordered reservations.
    /// All stores must share a device/execution owner; related contexts retain
    /// their own exact allocation checks. A failure cancels all supplied tokens
    /// without affecting an already active dispatch. No slot is re-reserved.
    pub fn enter(self, following: Vec<Self>) -> Result<GpuMatrixDispatch, String> {
        Self::activate(self, following, false)
    }

    /// Activate these reservations as a new dispatch, or extend the dispatch
    /// already active on this thread once its earlier claims are consumed. A
    /// host-driven step of an admitted job uses this to hold its exact extra
    /// claims; `finish` retracts an extension without ending the outer dispatch.
    pub fn enter_or_extend(self, following: Vec<Self>) -> Result<GpuMatrixDispatch, String> {
        let extend = unsafe { gpu_matrix_dispatch_active() } != 0;
        Self::activate(self, following, extend)
    }

    fn activate(
        first: Self,
        mut following: Vec<Self>,
        extend: bool,
    ) -> Result<GpuMatrixDispatch, String> {
        let mut reservations = Vec::with_capacity(following.len() + 1);
        reservations.push(first);
        reservations.append(&mut following);
        let pointers = reservations
            .iter()
            .map(|reservation| reservation.raw.expect("unconsumed reservation").as_ptr())
            .collect::<Vec<_>>();
        let mut slots = Vec::with_capacity(reservations.len());
        let mut requests = Vec::with_capacity(reservations.len());
        let mut raw = std::ptr::null_mut();
        let mut base = 0usize;
        let status = if extend {
            unsafe { gpu_matrix_dispatch_extend(pointers.as_ptr(), pointers.len(), &mut base) }
        } else {
            unsafe { gpu_matrix_dispatch_enter(pointers.as_ptr(), pointers.len(), &mut raw) }
        };
        if status != 0 {
            return Err(last_error_string());
        }
        for reservation in &mut reservations {
            reservation.raw = None;
            slots.push(std::mem::take(&mut reservation.slots));
            requests.push(std::mem::take(&mut reservation.requests));
        }
        Ok(GpuMatrixDispatch {
            raw: if extend {
                None
            } else {
                Some(NonNull::new(raw).expect("successful dispatch has a permit"))
            },
            extension: extend.then_some((base, pointers.len())),
            slots,
            requests,
            thread: PhantomData,
        })
    }
}

impl Drop for GpuMatrixReservation {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { gpu_matrix_reservation_destroy(raw.as_ptr()) };
        }
    }
}

/// A thread-bound submission guard. Ending it never waits for GPU completion.
/// Dropping it cancels only unconsumed slots; returned matrices retain storage.
#[must_use]
pub struct GpuMatrixDispatch {
    raw: Option<NonNull<DispatchOpaque>>,
    /// An extension of the thread's active permit: (base index, count).
    extension: Option<(usize, usize)>,
    slots: Vec<Vec<GpuPreparedSlotIdentity>>,
    requests: Vec<Vec<GpuPreparedRequest>>,
    thread: PhantomData<Rc<()>>,
}

impl GpuMatrixDispatch {
    /// Return every exclusive footprint in its original input order. Retain and
    /// rearm scratch tokens for another wave; drop output tokens when their
    /// normal native owners should determine logical release.
    pub fn finish(mut self) -> Result<Vec<GpuMatrixReservation>, String> {
        let mut pointers = vec![std::ptr::null_mut(); self.slots.len()];
        let mut reservations = Vec::with_capacity(self.slots.len());
        let status = if let Some((base, count)) = self.extension.take() {
            unsafe { gpu_matrix_dispatch_retract(base, count, 1, pointers.as_mut_ptr()) }
        } else {
            let raw = self.raw.take().expect("unfinished dispatch");
            unsafe { gpu_matrix_dispatch_end(raw.as_ptr(), 1, pointers.as_mut_ptr()) }
        };
        if status != 0 {
            return Err(last_error_string());
        }
        for ((pointer, slots), requests) in pointers
            .into_iter()
            .zip(std::mem::take(&mut self.slots))
            .zip(std::mem::take(&mut self.requests))
        {
            reservations.push(GpuMatrixReservation {
                raw: Some(NonNull::new(pointer).expect("completed dispatch returns its owners")),
                slots,
                requests,
                not_sync: PhantomData,
            });
        }
        Ok(reservations)
    }
}

impl Drop for GpuMatrixDispatch {
    fn drop(&mut self) {
        if let Some((base, count)) = self.extension.take() {
            unsafe { gpu_matrix_dispatch_retract(base, count, 0, std::ptr::null_mut()) };
        }
        if let Some(raw) = self.raw.take() {
            unsafe { gpu_matrix_dispatch_end(raw.as_ptr(), 0, std::ptr::null_mut()) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            PolyParams,
            dcrt::{
                gpu::{GPU_POLY_FORMAT_EVAL, GpuDCRTPolyParams, gpu_matrix_create},
                params::DCRTPolyParams,
            },
        },
    };

    // Existing fixed-shape fixtures request each slot's complete layout. Keep
    // their mathematical and ownership assertions unchanged across the typed API.
    fn reserve(
        storage: &GpuPreparedStorage,
        slots: &[usize],
    ) -> Result<GpuMatrixReservation, String> {
        let requests = slots
            .iter()
            .map(|&index| {
                let slot = storage.slot_identity(index).ok_or("missing test slot")?;
                Ok(match slot.kind() {
                    GpuPreparedSlotKind::Matrix => {
                        slot.matrix_request(slot.rows(), slot.columns(), true)
                    }
                    _ => slot.workspace_request(slot.requested_backing_bytes(), slot.alignment()),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        storage.reserve(&requests)
    }

    fn parameters() -> (DCRTPolyParams, GpuDCRTPolyParams) {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().expect("ring dimension"))
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let gpu = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        (cpu, gpu)
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_managed_setup_requires_complete_idle_inventory_and_exact_empty_resources() {
        use crate::poly::dcrt::gpu::{
            GpuAllocationEpochBoundary, GpuAllocationEpochObservation,
            GpuAllocationEpochUnverified, gpu_small_matrix_create, gpu_small_matrix_destroy,
        };
        let (_, params) = parameters();
        let first = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            }]),
        )
        .unwrap();
        let second =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], None).unwrap();
        assert!(GpuPreparedStorage::finish_setup(&[]).is_err());
        assert!(GpuPreparedStorage::finish_setup(&[&first]).is_err());
        assert!(GpuPreparedStorage::finish_setup(&[&first, &first]).is_err());
        let claims = [first.slot_identity(1).unwrap().workspace_request(0, 1)];
        let held = first.reserve(&claims).unwrap();
        assert!(GpuPreparedStorage::finish_setup(&[&first, &second]).is_err());
        drop(held);
        assert!(matches!(
            params
                .observe_allocation_epoch(
                    params.device_ids()[0],
                    GpuAllocationEpochBoundary::Refresh,
                    true
                )
                .unwrap(),
            GpuAllocationEpochObservation::Unverified(
                GpuAllocationEpochUnverified::UnsupportedActivity
            )
        ));
        GpuPreparedStorage::finish_setup(&[&first, &second]).unwrap();
        // Native empty descriptors must not bypass typed event admission. The
        // public compact codec rejects empty shapes before reaching this API.
        let mut empty = std::ptr::null_mut();
        let bound = 1u64;
        assert_ne!(
            unsafe {
                gpu_small_matrix_create(params.ctx_raw(), 0, 1, 1, &bound, 1, false, &mut empty)
            },
            0
        );
        assert!(empty.is_null());
        let dispatch = first.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
        assert_eq!(
            unsafe {
                gpu_small_matrix_create(params.ctx_raw(), 0, 1, 1, &bound, 1, false, &mut empty)
            },
            0
        );
        assert!(!empty.is_null());
        assert_eq!(first.occupancy().unwrap().resource_occupied_slots(), 1);
        drop(dispatch.finish().unwrap());
        assert!(GpuPreparedStorage::finish_setup(&[&first, &second]).is_err());
        unsafe { gpu_small_matrix_destroy(empty) };
        GpuPreparedStorage::finish_setup(&[&first, &second]).unwrap();
        assert_eq!(first.occupancy().unwrap().resource_occupied_slots(), 0);
        let mut unexpected = std::ptr::null_mut();
        assert_ne!(
            unsafe {
                gpu_matrix_create(
                    params.ctx_raw(),
                    1,
                    1,
                    1,
                    GPU_POLY_FORMAT_EVAL,
                    &mut unexpected,
                    true,
                )
            },
            0
        );
        assert!(unexpected.is_null());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_requests_survive_first_wave_specialization_and_worker_transfer() {
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let maximum = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(maximum > 0);
        let backing_columns = maximum.checked_add(1).unwrap();
        let storage = GpuPreparedStorage::new(
            (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, 2, backing_columns)).collect(),
            None,
        )
        .unwrap();
        let slots = [storage.slot_identity(0).unwrap(), storage.slot_identity(1).unwrap()];
        let original =
            [slots[0].matrix_request(2, maximum, true), slots[1].matrix_request(1, maximum, true)];
        let narrow = [slots[0].matrix_request(2, 1, true), slots[1].matrix_request(1, 1, true)];
        let mut reservation = storage.reserve(&original).unwrap();
        let held = storage.occupancy().unwrap().reserved_bytes();
        if maximum > 1 {
            reservation.rearm(&narrow).unwrap();
        } else {
            assert!(reservation.rearm(&narrow).is_err());
        }
        assert_eq!(reservation.requests(), narrow);
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), held);
        let outputs = reservation
            .partition(&[1, 1])
            .unwrap()
            .into_par_iter()
            .enumerate()
            .map(|(index, reservation)| {
                let rows = 2 - index;
                assert_eq!(reservation.requests(), &[narrow[index]]);
                let dispatch = reservation.enter(Vec::new()).unwrap();
                let first = GpuDCRTPolyMatrix::zero(&params, rows, 1);
                let mut reservation = dispatch.finish().unwrap().pop().unwrap();
                assert_eq!(reservation.requests(), &[narrow[index]]);
                drop(first);
                // Fits physical backing, but exceeds this invocation's original
                // envelope. Rejection must preserve its current request metadata.
                let oversized = slots[index].matrix_request(rows, backing_columns, true);
                assert!(reservation.rearm(&[oversized]).is_err());
                assert_eq!(reservation.requests(), &[narrow[index]]);
                reservation.rearm(&[original[index]]).unwrap();
                assert_eq!(reservation.requests(), &[original[index]]);
                let dispatch = reservation.enter(Vec::new()).unwrap();
                let output = GpuDCRTPolyMatrix::zero(&params, rows, maximum);
                let reservation = dispatch.finish().unwrap().pop().unwrap();
                assert_eq!(reservation.requests(), &[original[index]]);
                drop(reservation);
                output
            })
            .collect::<Vec<_>>();
        for (index, output) in outputs.iter().enumerate() {
            assert_eq!(output.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 2 - index, maximum));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_execution_reuses_timing_and_rejects_late_contexts() {
        let (cpu, params) = parameters();
        let related = GpuDCRTPolyParams::new_with_gpu(
            cpu.ring_dimension(),
            cpu.to_crt().0,
            4,
            params.device_ids(),
            Some(1),
            Some(&params),
            None,
        );
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            }]),
        )
        .unwrap();
        let claim = storage.slot_identity(1).unwrap().workspace_request(0, 1);
        let reservation = storage.reserve(&[claim]).unwrap();
        // The only dynamic event slot remains reserved and never entered.
        // Both related contexts must use the execution's setup-owned timer.
        for context in [&params, &related] {
            let abandoned = context.begin_device_timing().unwrap();
            assert!(params.begin_device_timing().is_err());
            drop(abandoned);
            let mut timing = context.begin_device_timing().unwrap();
            timing.stop().unwrap();
            timing.stop().unwrap();
            let measured = std::thread::spawn(move || timing.finish().unwrap()).join().unwrap();
            assert_eq!(measured.len(), params.device_ids().len());
            assert!(measured.iter().all(|(_, seconds)| seconds.is_finite() && *seconds >= 0.0));
        }
        let occupancy = storage.occupancy().unwrap();
        assert_eq!(occupancy.resource_reserved_slots(), 1);
        assert_eq!(occupancy.resource_occupied_slots(), 0);
        assert_eq!(occupancy.resource_high_water_slots(), 0);
        let late = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GpuDCRTPolyParams::new_with_gpu(
                cpu.ring_dimension(),
                cpu.to_crt().0,
                4,
                params.device_ids(),
                Some(1),
                Some(&params),
                None,
            )
        }));
        assert!(late.is_err(), "new parameter backing must not grow an admitted execution");
        assert!(last_error_string().contains("must be provisioned before admission"));
        drop(reservation);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_small_store_uses_reserved_completion() {
        use crate::matrix::{PolyMatrixSmallRhs, SmallPolyMatrix};
        let (_, params) = parameters();
        let source = GpuDCRTPolyMatrix::gadget_matrix(&params, 1, None)
            .gadget_decompose(false, None)
            .unwrap();
        let expected = source.to_canonical_coefficients().unwrap();
        let event = GpuPreparedWorkspaceLayout {
            bytes: 0,
            alignment: 1,
            kind: GpuPreparedSlotKind::CompletionEvent,
        };
        let storage =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&[event]))
                .unwrap();
        let claims = [storage.slot_identity(1).unwrap().workspace_request(0, 1)];
        let mut token = storage.reserve(&claims).unwrap();
        for round in 0..2 {
            if round != 0 {
                token.rearm(&claims).unwrap();
            }
            let dispatch = token.enter(Vec::new()).unwrap();
            assert_eq!(source.to_canonical_coefficients().unwrap(), expected);
            token = dispatch.finish().unwrap().pop().unwrap();
        }
        assert_eq!(storage.occupancy().unwrap().resource_occupied_slots(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_completion_slots_follow_host_observers() {
        let (_, params) = parameters();
        let event = GpuPreparedWorkspaceLayout {
            bytes: 0,
            alignment: 1,
            kind: GpuPreparedSlotKind::CompletionEvent,
        };
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
            Some(&[event, event]),
        )
        .unwrap();
        let requests = (1..=2)
            .map(|slot| storage.slot_identity(slot).unwrap().workspace_request(0, 1))
            .collect::<Vec<_>>();
        let baseline = storage.occupancy().unwrap();
        assert_eq!(baseline.resource_capacity_slots(), 2);
        assert_eq!(baseline.resource_available_slots(), 2);
        let token = storage.reserve(&requests).unwrap();
        assert_eq!(storage.occupancy().unwrap().resource_reserved_slots(), 2);
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0);
        let dispatch = token.enter(Vec::new()).unwrap();
        let first = params.record_releases().unwrap();
        let second = params.record_releases().unwrap();
        let mut token = dispatch.finish().unwrap().pop().unwrap();
        let occupied = storage.occupancy().unwrap();
        assert_eq!(occupied.resource_occupied_slots(), 2);
        assert_eq!(occupied.resource_high_water_slots(), 2);
        assert_eq!(occupied.requested_capacity_bytes(), baseline.requested_capacity_bytes());
        assert!(token.rearm(&requests).is_err());
        assert!(params.record_releases().is_err(), "missing event permits must fail");
        drop(first);
        assert_eq!(storage.occupancy().unwrap().resource_occupied_slots(), 1);
        drop(second);
        token.rearm(&requests).unwrap();
        let dispatch = token.enter(Vec::new()).unwrap();
        let first = params.record_releases().unwrap();
        let second = params.record_releases().unwrap();
        drop(dispatch.finish().unwrap());
        drop(storage);
        params.fence_released_memory();
        assert!(first.is_complete().unwrap());
        assert!(second.is_complete().unwrap());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_transfer_events_survive_pending_snapshots_and_reclaimer() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("columns"))
            .unwrap_or(3);
        let bytes = (0..2)
            .map(|_| {
                let value = DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    2,
                    columns,
                    DistType::FinRingDist,
                );
                GpuDCRTPolyMatrix::from_cpu_matrix(&params, &value).into_cpu_staging_bytes()
            })
            .collect::<Vec<_>>();
        let transfer = params.rns_transfer_workspace(params.crt_depth() - 1, 2, columns).unwrap();
        let event = GpuPreparedWorkspaceLayout {
            bytes: 0,
            alignment: 1,
            kind: GpuPreparedSlotKind::CompletionEvent,
        };
        let storage = GpuPreparedStorage::new(
            (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, 2, columns)).collect(),
            Some(&[transfer, event, event, event, event]),
        )
        .unwrap();
        let scratch =
            storage.slot_identity(2).unwrap().workspace_request(transfer.bytes, transfer.alignment);
        let mut outputs = Vec::new();
        for (index, bytes) in bytes.iter().enumerate() {
            let claims = [
                storage.slot_identity(index).unwrap().matrix_request(2, columns, true),
                scratch,
                storage.slot_identity(3 + index).unwrap().workspace_request(0, 1),
            ];
            let dispatch = storage.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
            outputs.push(GpuDCRTPolyMatrix::from_cpu_staging_bytes(&params, bytes));
            drop(dispatch.finish().unwrap());
        }
        let mut downloads = Vec::new();
        for (index, output) in outputs.iter().enumerate() {
            let claims =
                [scratch, storage.slot_identity(5 + index).unwrap().workspace_request(0, 1)];
            let dispatch = storage.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
            downloads.push(output.start_rns_snapshot(None));
            drop(dispatch.finish().unwrap());
        }
        for (download, expected) in downloads.into_iter().zip(bytes) {
            let snapshot = download.finish();
            let actual = bincode::encode_to_vec(
                (
                    1u8,
                    snapshot.nrow,
                    snapshot.ncol,
                    snapshot.level,
                    snapshot.is_ntt,
                    snapshot.bytes_per_poly,
                    snapshot.bytes.as_slice(),
                ),
                bincode::config::standard(),
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
        params.fence_released_memory();
        assert_eq!(storage.occupancy().unwrap().resource_occupied_slots(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_private_stream_is_reusable() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("columns"))
            .unwrap_or(3);
        let value =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &value).into_coeff_domain();
        let expected = input.to_compact_bytes();
        let workspace = params
            .compact_transfer_workspace(
                params.crt_depth() - 1,
                2,
                columns,
                GpuCompactTransferKind::Store,
            )
            .unwrap();
        let stream = GpuPreparedWorkspaceLayout {
            bytes: 0,
            alignment: 1,
            kind: GpuPreparedSlotKind::SubmissionStream,
        };
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 2, columns)],
            Some(&[stream, workspace]),
        )
        .unwrap();
        let claims = [
            storage.slot_identity(0).unwrap().matrix_request(2, columns, false),
            storage.slot_identity(1).unwrap().workspace_request(0, 1),
            storage
                .slot_identity(2)
                .unwrap()
                .workspace_request(workspace.bytes, workspace.alignment),
        ];
        let mut token = storage.reserve(&claims).unwrap();
        for round in 0..2 {
            if round != 0 {
                token.rearm(&claims).unwrap();
            }
            let dispatch = token.enter(Vec::new()).unwrap();
            assert_eq!(input.to_compact_bytes(), expected);
            token = dispatch.finish().unwrap().pop().unwrap();
            assert_eq!(storage.occupancy().unwrap().resource_occupied_slots(), 0);
        }
        assert_eq!(storage.occupancy().unwrap().resource_high_water_slots(), 1);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_equality_reuses_result_across_limbs_and_calls() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("columns"))
            .unwrap_or(3);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let cpu_zero = DCRTPolyMatrix::zero(&cpu, 2, columns);
        let left = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let zero = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &cpu_zero);
        let layout = GpuDCRTPolyMatrix::equality_workspace().unwrap();
        let storage =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&[layout]))
                .unwrap();
        let requests =
            [storage.slot_identity(1).unwrap().workspace_request(layout.bytes, layout.alignment)];
        let mut token = storage.reserve(&requests).unwrap();
        for (index, (other, expected)) in
            [(&right, true), (&zero, original == cpu_zero), (&right, true)].into_iter().enumerate()
        {
            if index != 0 {
                token.rearm(&requests).unwrap();
            }
            let dispatch = token.enter(Vec::new()).unwrap();
            assert_eq!(left.eq(other), expected);
            token = dispatch.finish().unwrap().pop().unwrap();
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_transfer_scalar_batch_and_load_use_bounded_spans() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("columns"))
            .unwrap_or(3);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).into_coeff_domain();
        let zero = GpuDCRTPolyMatrix::zero(&params, 2, columns).into_coeff_domain();
        let expected = vec![source.to_compact_bytes(), zero.to_compact_bytes()];
        let widths = expected
            .iter()
            .map(|bytes| {
                let ((_, _, _, _, _, bits, _, _), used): (
                    (u8, u8, u32, usize, usize, u16, u16, Vec<u8>),
                    usize,
                ) = bincode::decode_from_slice(bytes, bincode::config::standard()).unwrap();
                assert_eq!(used, bytes.len());
                bits
            })
            .collect::<Vec<_>>();
        let level = params.crt_depth() - 1;
        let store = params
            .compact_transfer_workspace(level, 2, columns, GpuCompactTransferKind::Store)
            .unwrap();
        let batch = params
            .compact_transfer_workspace(
                level,
                2,
                columns,
                GpuCompactTransferKind::StoreBatch { matrices: 2 },
            )
            .unwrap();
        let loads = widths
            .iter()
            .map(|&max_coefficient_bits| {
                params
                    .compact_transfer_workspace(
                        level,
                        2,
                        columns,
                        GpuCompactTransferKind::Load { max_coefficient_bits },
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(
            params
                .compact_transfer_workspace(level, usize::MAX, 2, GpuCompactTransferKind::Store)
                .is_err()
        );
        assert!(
            params
                .compact_transfer_workspace(
                    level,
                    2,
                    columns,
                    GpuCompactTransferKind::StoreBatch { matrices: 0 }
                )
                .is_err()
        );
        let storage = GpuPreparedStorage::new(
            (0..3).map(|_| GpuDCRTPolyMatrix::zero(&params, 2, columns)).collect(),
            Some(&[store, batch, loads[0], loads[1]]),
        )
        .unwrap();
        // Scalar serialization clones its source before coefficient conversion.
        // Reserve that real matrix allocation separately from codec scratch.
        let store_request = [
            storage.slot_identity(2).unwrap().matrix_request(2, columns, false),
            storage.slot_identity(3).unwrap().workspace_request(store.bytes, store.alignment),
        ];
        let batch_request =
            [storage.slot_identity(4).unwrap().workspace_request(batch.bytes, batch.alignment)];
        let mut store_token = storage.reserve(&store_request).unwrap();
        let mut batch_token = storage.reserve(&batch_request).unwrap();
        for wave in 0..2 {
            if wave != 0 {
                store_token.rearm(&store_request).unwrap();
                batch_token.rearm(&batch_request).unwrap();
            }
            let dispatch = store_token.enter(Vec::new()).unwrap();
            let serialized = source.to_compact_bytes();
            store_token = dispatch.finish().unwrap().pop().unwrap();
            assert_eq!(serialized, expected[0]);
            let dispatch = batch_token.enter(Vec::new()).unwrap();
            let serialized = GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&[&source, &zero]);
            batch_token = dispatch.finish().unwrap().pop().unwrap();
            assert_eq!(serialized, expected);
        }
        drop(batch_token);
        let mut restored = Vec::new();
        for (index, bytes) in expected.iter().enumerate() {
            let requests = [
                storage.slot_identity(index).unwrap().matrix_request(2, columns, false),
                storage
                    .slot_identity(5 + index)
                    .unwrap()
                    .workspace_request(loads[index].bytes, loads[index].alignment),
            ];
            let dispatch = storage.reserve(&requests).unwrap().enter(Vec::new()).unwrap();
            restored.push(GpuDCRTPolyMatrix::from_compact_bytes(&params, bytes));
            drop(dispatch.finish().unwrap());
        }
        for (value, expected) in restored.iter().zip(&expected) {
            store_token.rearm(&store_request).unwrap();
            let dispatch = store_token.enter(Vec::new()).unwrap();
            assert_eq!(&value.to_compact_bytes(), expected);
            store_token = dispatch.finish().unwrap().pop().unwrap();
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_rns_transfer_reuses_span_across_uploads_and_pending_downloads() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().expect("columns"))
            .unwrap_or(3);
        let expected = (0..2)
            .map(|_| {
                DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    2,
                    columns,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let bytes = expected
            .iter()
            .map(|matrix| {
                GpuDCRTPolyMatrix::from_cpu_matrix(&params, matrix).into_cpu_staging_bytes()
            })
            .collect::<Vec<_>>();
        let layout = params.rns_transfer_workspace(params.crt_depth() - 1, 2, columns).unwrap();
        assert_eq!(layout.kind, GpuPreparedSlotKind::TransferWorkspace);
        assert!(params.rns_transfer_workspace(usize::MAX, 2, columns).is_err());
        assert!(params.rns_transfer_workspace(0, usize::MAX, 2).is_err());
        let storage = GpuPreparedStorage::new(
            (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, 2, columns)).collect(),
            Some(&[layout]),
        )
        .unwrap();
        let outputs = (0..2)
            .map(|index| storage.slot_identity(index).unwrap().matrix_request(2, columns, true))
            .collect::<Vec<_>>();
        let mut output_tokens =
            storage.reserve(&outputs).unwrap().partition(&[1, 1]).unwrap().into_iter();
        let requests =
            [storage.slot_identity(2).unwrap().workspace_request(layout.bytes, layout.alignment)];
        let mut scratch = storage.reserve(&requests).unwrap();
        let mut actual = Vec::new();
        for (wave, payload) in bytes.iter().enumerate() {
            if wave != 0 {
                scratch.rearm(&requests).unwrap();
            }
            let dispatch = output_tokens.next().unwrap().enter(vec![scratch]).unwrap();
            actual.push(GpuDCRTPolyMatrix::from_cpu_staging_bytes(&params, payload));
            let mut tokens = dispatch.finish().unwrap();
            scratch = tokens.pop().unwrap();
            drop(tokens);
        }
        // Enqueue all downloads before waiting for any. Reusing the unpack span
        // must wait for the previous copy's device completion, while the pinned
        // destinations and all outputs remain independently live.
        let mut downloads = Vec::new();
        for value in &actual {
            scratch.rearm(&requests).unwrap();
            let dispatch = scratch.enter(Vec::new()).unwrap();
            downloads.push(value.start_rns_snapshot(None));
            scratch = dispatch.finish().unwrap().pop().unwrap();
        }
        let snapshots = downloads.into_iter().map(|transfer| transfer.finish()).collect::<Vec<_>>();
        for (snapshot, original_bytes) in snapshots.iter().zip(&bytes) {
            let encoded = bincode::encode_to_vec(
                (
                    1u8,
                    snapshot.nrow,
                    snapshot.ncol,
                    snapshot.level,
                    snapshot.is_ntt,
                    snapshot.bytes_per_poly,
                    snapshot.bytes.as_slice(),
                ),
                bincode::config::standard(),
            )
            .unwrap();
            assert_eq!(&encoded, original_bytes);
        }
        drop((scratch, storage, snapshots, actual));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_completion_reuse_orders_pending_readers() {
        use crate::{
            matrix::{PolyMatrixSmallRhs, SmallPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(columns > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let compact = original.clone().gadget_decompose(false, None).unwrap();
        let payload = compact.to_canonical_coefficients().unwrap();
        let fixed = GpuSmallMatrix::from_canonical_coefficients(
            &params,
            compact.rows(),
            compact.columns(),
            compact.max_coefficient_bound().clone(),
            &payload,
        )
        .unwrap();
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, 1, None);
        let report = fixed.allocation_report(&gadget).unwrap();
        let layouts = [
            GpuPreparedWorkspaceLayout {
                bytes: payload.len(),
                alignment: 256,
                kind: GpuPreparedSlotKind::CompactPayload,
            },
            GpuPreparedWorkspaceLayout {
                bytes: report.expanded_rhs_workspace_bytes,
                alignment: 8,
                kind: GpuPreparedSlotKind::CompactWorkspace,
            },
        ];
        let storage = GpuPreparedStorage::new(
            (0..2).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 1, columns)).collect(),
            Some(&layouts),
        )
        .unwrap();
        let payload_request =
            storage.slot_identity(2).unwrap().workspace_request(payload.len(), 256);
        let scratch_request = storage
            .slot_identity(3)
            .unwrap()
            .workspace_request(report.expanded_rhs_workspace_bytes, 8);
        let mut payload_reservation = storage.reserve(&[payload_request]).unwrap();
        let mut scratch = storage.reserve(&[scratch_request]).unwrap();
        let outputs = reserve(&storage, &[0, 1]).unwrap().partition(&[1, 1]).unwrap();
        let mut products = Vec::new();
        for (wave, output) in outputs.into_iter().enumerate() {
            let dispatch = payload_reservation.enter(vec![output, scratch]).unwrap();
            let rhs = fixed.clone();
            products.push(gadget.multiply_small_rhs(&rhs).unwrap());
            drop(rhs);
            let mut reservations = dispatch.finish().unwrap();
            scratch = reservations.pop().unwrap();
            drop(reservations.pop().unwrap());
            payload_reservation = reservations.pop().unwrap();
            if wave == 0 {
                payload_reservation.rearm(&[payload_request]).unwrap();
                scratch.rearm(&[scratch_request]).unwrap();
            }
        }
        drop((payload_reservation, scratch, storage, fixed, gadget));
        products.par_iter().for_each(|product| assert_eq!(product.to_cpu_matrix(), original));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_p1_completion_slot_reuse_preserves_samples() {
        use crate::poly::dcrt::gpu::GpuRngSeed;
        let (_, params) = parameters();
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(rows > 0);
        let columns = 2;
        let covariance = GpuDCRTPolyMatrix::zero(&params, rows, rows).into_coeff_domain();
        let cache = GpuDCRTPolyMatrix::create_p1_covariance_cache(
            &covariance,
            &covariance,
            &covariance,
            1.0,
            8.0,
            4.578,
        );
        let seeds =
            [GpuRngSeed::from_bytes(rand::random()), GpuRngSeed::from_bytes(rand::random())];
        let expected = seeds
            .iter()
            .map(|&seed| {
                GpuDCRTPolyMatrix::sample_p1_full_cached(
                    &cache,
                    GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns).into_coeff_domain(),
                    seed,
                )
                .to_cpu_matrix()
            })
            .collect::<Vec<_>>();
        let inputs = seeds
            .iter()
            .map(|_| GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns).into_coeff_domain())
            .collect::<Vec<_>>();
        let mut layouts = vec![GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompletionEvent,
            bytes: 0,
            alignment: 1,
        }];
        layouts.extend(
            params
                .p1_sampling_workspaces(rows, columns, true)
                .unwrap()
                .into_iter()
                .filter(|layout| layout.bytes != 0),
        );
        let sampling_end = 2 + layouts.len();
        let storage = GpuPreparedStorage::new(
            (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns)).collect(),
            Some(&layouts),
        )
        .unwrap();
        let requests = (2..sampling_end)
            .map(|index| {
                let slot = storage.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        let mut scratch = storage.reserve(&requests).unwrap();
        let outputs = reserve(&storage, &[0, 1]).unwrap().partition(&[1, 1]).unwrap();
        let mut samples = Vec::new();
        for (wave, ((input, output), seed)) in
            inputs.into_iter().zip(outputs).zip(seeds).enumerate()
        {
            let dispatch = output.enter(vec![scratch]).unwrap();
            samples.push(GpuDCRTPolyMatrix::sample_p1_full_cached(&cache, input, seed));
            let mut reservations = dispatch.finish().unwrap();
            scratch = reservations.pop().unwrap();
            drop(reservations.pop().unwrap());
            let occupancy = storage.occupancy().unwrap();
            assert_eq!(occupancy.resource_capacity_slots(), 1);
            assert_eq!(occupancy.resource_occupied_slots(), 0);
            assert_eq!(occupancy.resource_high_water_slots(), 1);
            if wave == 0 {
                scratch.rearm(&requests).unwrap();
            }
        }
        // Destroy the cache after resource admission. Destruction must reuse its
        // setup fence, without demanding a new resource permit or a live ctx.
        drop((scratch, cache, covariance));
        // The sample and its readback reuse one completion-event slot.
        let readback = [storage.slot_identity(2).unwrap().workspace_request(0, 1)];
        for (actual, expected) in samples.iter().zip(expected) {
            let dispatch = storage.reserve(&readback).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(actual.to_cpu_matrix(), expected);
            drop(dispatch.finish().unwrap());
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_p1_cached_and_uncached_spans_reuse() {
        use crate::poly::dcrt::gpu::GpuRngSeed;
        unsafe extern "C" {
            fn gpu_matrix_sample_p1_full(
                a: *const GpuMatrixOpaque,
                b: *const GpuMatrixOpaque,
                d: *const GpuMatrixOpaque,
                tp2: *const GpuMatrixOpaque,
                sigma: f64,
                s: f64,
                dgg_stddev: f64,
                seed: GpuRngSeed,
                output: *mut GpuMatrixOpaque,
            ) -> i32;
        }
        let large = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(5);
        assert!(large > 0);
        for rows in [1, large] {
            for cached in [false, true] {
                let (_, params) = parameters();
                let columns = 2;
                let covariance = GpuDCRTPolyMatrix::zero(&params, rows, rows).into_coeff_domain();
                let cache = cached.then(|| {
                    GpuDCRTPolyMatrix::create_p1_covariance_cache(
                        &covariance,
                        &covariance,
                        &covariance,
                        1.0,
                        8.0,
                        4.578,
                    )
                });
                let sample = |input: GpuDCRTPolyMatrix, seed| {
                    if let Some(cache) = &cache {
                        GpuDCRTPolyMatrix::sample_p1_full_cached(cache, input, seed)
                    } else {
                        let output = GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns);
                        assert_eq!(
                            unsafe {
                                gpu_matrix_sample_p1_full(
                                    covariance.raw,
                                    covariance.raw,
                                    covariance.raw,
                                    input.raw,
                                    1.0,
                                    8.0,
                                    4.578,
                                    seed,
                                    output.raw,
                                )
                            },
                            0,
                            "{}",
                            last_error_string()
                        );
                        output
                    }
                };
                let seeds = [
                    GpuRngSeed::from_bytes(rand::random()),
                    GpuRngSeed::from_bytes(rand::random()),
                ];
                // A fresh random seed per case is replayed through the existing
                // unprepared path, so expected samples do not reimplement RNG.
                let expected = seeds
                    .iter()
                    .map(|&seed| {
                        sample(
                            GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns).into_coeff_domain(),
                            seed,
                        )
                        .to_cpu_matrix()
                    })
                    .collect::<Vec<_>>();
                let inputs = (0..2)
                    .map(|_| {
                        GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns).into_coeff_domain()
                    })
                    .collect::<Vec<_>>();
                let layouts = params.p1_sampling_workspaces(rows, columns, cached).unwrap();
                assert!(params.p1_sampling_workspaces(usize::MAX, columns, cached).is_err());
                assert!(params.p1_sampling_workspaces(rows, usize::MAX, cached).is_err());
                assert_eq!(layouts[1].bytes > 0, rows > 4);
                let layouts =
                    layouts.into_iter().filter(|layout| layout.bytes != 0).collect::<Vec<_>>();
                let storage = GpuPreparedStorage::new(
                    (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, 2 * rows, columns)).collect(),
                    Some(&layouts),
                )
                .unwrap();
                let requests = (2..storage.slot_count())
                    .map(|index| {
                        let slot = storage.slot_identity(index).unwrap();
                        slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                    })
                    .collect::<Vec<_>>();
                let mut scratch = storage.reserve(&requests).unwrap();
                let outputs = reserve(&storage, &[0, 1]).unwrap().partition(&[1, 1]).unwrap();
                let mut samples = Vec::new();
                for (wave, ((input, output), seed)) in
                    inputs.into_iter().zip(outputs).zip(seeds).enumerate()
                {
                    let dispatch = output.enter(vec![scratch]).unwrap();
                    samples.push(sample(input, seed));
                    let mut reservations = dispatch.finish().unwrap();
                    scratch = reservations.pop().unwrap();
                    drop(reservations.pop().unwrap());
                    if wave == 0 {
                        scratch.rearm(&requests).unwrap();
                    }
                }
                drop((scratch, storage, cache, covariance));
                for (actual, expected) in samples.iter().zip(expected) {
                    assert_eq!(actual.to_cpu_matrix(), expected);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_sampler_spans_preserve_gadget_relation_and_reuse() {
        use crate::{
            poly::dcrt::gpu::GpuRngSeed,
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(columns > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let sources = (0..2)
            .into_par_iter()
            .map(|_| GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original))
            .collect::<Vec<_>>();
        let layouts =
            params.gaussian_gadget_workspaces(params.crt_depth() - 1, 1, columns).unwrap();
        assert!(params.gaussian_gadget_workspaces(params.crt_depth(), 1, columns).is_err());
        assert!(params.gaussian_gadget_workspaces(0, usize::MAX, columns).is_err());
        assert!(
            params
                .gaussian_gadget_workspaces(0, 0, columns)
                .unwrap()
                .iter()
                .all(|layout| layout.bytes == 0)
        );
        let rows = params.modulus_digits();
        let storage = GpuPreparedStorage::new(
            (0..2)
                .into_par_iter()
                .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns))
                .collect(),
            Some(&layouts),
        )
        .unwrap();
        let requests = (2..storage.slot_count())
            .map(|index| {
                let slot = storage.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        let mut scratch = storage.reserve(&requests).unwrap();
        let outputs = reserve(&storage, &[0, 1]).unwrap().partition(&[1, 1]).unwrap();
        let mut samples = Vec::new();
        for (wave, (source, output)) in sources.into_iter().zip(outputs).enumerate() {
            let dispatch = output.enter(vec![scratch]).unwrap();
            samples.push(source.gauss_samp_gq_arb_base(
                ((1u64 << params.base_bits()) as f64 + 1.0) * 4.578,
                4.578,
                GpuRngSeed::from_bytes(rand::random()),
            ));
            let mut reservations = dispatch.finish().unwrap();
            scratch = reservations.pop().unwrap();
            drop(reservations.pop().unwrap());
            // Reuse scratch without a host fence while retaining prior outputs.
            if wave == 0 {
                scratch.rearm(&requests).unwrap();
            }
        }
        assert!(
            storage.occupancy().unwrap().occupied_high_water_bytes() >=
                layouts.iter().map(|layout| layout.bytes).sum()
        );
        drop((scratch, storage));
        let gadget = DCRTPolyMatrix::gadget_matrix(&cpu, 1, None);
        samples
            .par_iter()
            .for_each(|sample| assert_eq!(&gadget * &sample.to_cpu_matrix(), original));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_expansion_reuses_typed_workspace() {
        use crate::{
            matrix::{PolyMatrixSmallRhs, SmallPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(columns > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let compact = original.clone().gadget_decompose(false, None).unwrap();
        let payload = compact.to_canonical_coefficients().unwrap();
        let rhs = GpuSmallMatrix::from_canonical_coefficients(
            &params,
            compact.rows(),
            compact.columns(),
            compact.max_coefficient_bound().clone(),
            &payload,
        )
        .unwrap();
        let gadget = GpuDCRTPolyMatrix::gadget_matrix(&params, 1, None);
        let report = rhs.allocation_report(&gadget).unwrap();
        assert_eq!(report.u64_workspace_limb_count, params.crt_depth());
        let storage = GpuPreparedStorage::new(
            (0..2).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 1, columns)).collect(),
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: report.expanded_rhs_workspace_bytes,
                alignment: 8,
                kind: GpuPreparedSlotKind::CompactWorkspace,
            }]),
        )
        .unwrap();
        let slot = storage.slot_identity(2).unwrap();
        let request = slot.workspace_request(report.expanded_rhs_workspace_bytes, 8);
        let mut scratch = storage.reserve(&[request]).unwrap();
        let outputs = reserve(&storage, &[0, 1]).unwrap().partition(&[1, 1]).unwrap();
        let mut products = Vec::new();
        for (wave, output) in outputs.into_iter().enumerate() {
            let dispatch = output.enter(vec![scratch]).unwrap();
            products.push(gadget.multiply_small_rhs(&rhs).unwrap());
            let mut reservations = dispatch.finish().unwrap();
            scratch = reservations.pop().unwrap();
            drop(reservations.pop().unwrap());
            if wave == 0 {
                scratch.rearm(&[request]).unwrap();
            }
        }
        drop((scratch, storage, rhs, gadget));
        products.par_iter().for_each(|product| assert_eq!(product.to_cpu_matrix(), original));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_native_pinned_transform_claims_and_retains_readers() {
        use crate::{
            matrix::gpu_dcrt_poly::GpuMatrixCrtOperation,
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(rows > 0);
        let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let target_cpu = DCRTPolyParams::new(n, 2, 17, 4, Some(cpu.to_crt().0[..2].to_vec()), None);
        let target = GpuDCRTPolyParams::new(n, target_cpu.to_crt().0, 4, None);
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            target.device_ids(),
            Some(1),
            Some(&target),
            None,
        );
        let requirements = target
            .matrix_crt_workspace_bytes(
                target.crt_depth() - 1,
                rows,
                1,
                GpuMatrixCrtOperation::ConvertModulus,
                source_params.crt_depth(),
                1,
                false,
            )
            .unwrap();
        assert!(requirements.pinned_bytes > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, 1, DistType::FinRingDist);
        let source = GpuDCRTPolyMatrix::from_cpu_matrix(&source_params, &original);
        let mut layouts = Vec::new();
        if requirements.additional_bytes > 0 {
            layouts.push(GpuPreparedWorkspaceLayout {
                bytes: requirements.additional_bytes,
                alignment: requirements.alignment,
                kind: GpuPreparedSlotKind::TransformWorkspace,
            });
        }
        layouts.push(GpuPreparedWorkspaceLayout {
            bytes: requirements.pinned_bytes,
            alignment: requirements.alignment,
            kind: GpuPreparedSlotKind::PinnedHost,
        });
        let storage = GpuPreparedStorage::new(
            [(rows, 1), (1, rows), (1, rows)]
                .into_par_iter()
                .map(|(rows, cols)| GpuDCRTPolyMatrix::zero(&target, rows, cols))
                .collect(),
            Some(&layouts),
        )
        .unwrap();
        let scratch_slots = std::iter::once(0).chain(3..storage.slot_count()).collect::<Vec<_>>();
        let requests = scratch_slots
            .iter()
            .map(|&index| {
                let slot = storage.slot_identity(index).unwrap();
                if index == 0 {
                    slot.matrix_request(rows, 1, true)
                } else {
                    slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                }
            })
            .collect::<Vec<_>>();
        let mut scratch = storage.reserve(&requests).unwrap();
        let outputs = reserve(&storage, &[1, 2]).unwrap().partition(&[1, 1]).unwrap();
        let mut readers = Vec::new();
        for (wave, output) in outputs.into_iter().enumerate() {
            let dispatch = scratch.enter(vec![output]).unwrap();
            let converted = source.reduce_modulus(&target);
            readers.push(converted.transpose());
            drop(converted);
            let mut reservations = dispatch.finish().unwrap();
            drop(reservations.pop().unwrap());
            scratch = reservations.pop().unwrap();
            target.fence_released_memory();
            let occupancy = storage.occupancy().unwrap();
            assert_eq!(occupancy.pinned_high_water_bytes(), requirements.pinned_bytes);
            assert_eq!(occupancy.pinned_occupied_bytes(), 0);
            if wave == 0 {
                scratch.rearm(&requests).unwrap();
            }
        }
        drop((scratch, storage, source, source_params));
        let expected = original.reduce_modulus(&target_cpu).transpose();
        readers.par_iter().for_each(|reader| assert_eq!(reader.to_cpu_matrix(), expected));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_native_pinned_compact_upload_and_serialization_claims() {
        use crate::{
            matrix::{PolyMatrixSmallRhs, SmallPolyMatrix, gpu_dcrt_poly::GpuSmallMatrix},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let (cpu, params) = parameters();
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let compact = original.clone().gadget_decompose(false, None).unwrap();
        let payload = compact.to_canonical_coefficients().unwrap();
        let source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).into_coeff_domain();
        let expected = source.to_compact_bytes();
        let (decoded, used): ((u8, u8, u32, usize, usize, u16, u16, Vec<u8>), _) =
            bincode::decode_from_slice(&expected, bincode::config::standard()).unwrap();
        assert_eq!(used, expected.len());
        assert!(!decoded.7.is_empty());
        let layouts = [
            GpuPreparedWorkspaceLayout {
                bytes: payload.len(),
                alignment: 256,
                kind: GpuPreparedSlotKind::CompactPayload,
            },
            GpuPreparedWorkspaceLayout {
                bytes: payload.len(),
                alignment: 1,
                kind: GpuPreparedSlotKind::PinnedHost,
            },
            GpuPreparedWorkspaceLayout {
                bytes: 4,
                alignment: 4,
                kind: GpuPreparedSlotKind::PinnedHost,
            },
            GpuPreparedWorkspaceLayout {
                bytes: decoded.7.len(),
                alignment: 1,
                kind: GpuPreparedSlotKind::PinnedHost,
            },
        ];
        let storage =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap();
        let requests = (1..=4)
            .map(|index| {
                let slot = storage.slot_identity(index).unwrap();
                slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
            })
            .collect::<Vec<_>>();
        let mut reservation = storage.reserve(&requests).unwrap();
        for wave in 0..2 {
            let dispatch = reservation.enter(Vec::new()).unwrap();
            let uploaded = GpuSmallMatrix::from_canonical_coefficients(
                &params,
                compact.rows(),
                compact.columns(),
                compact.max_coefficient_bound().clone(),
                &payload,
            )
            .unwrap();
            let serialized = GpuDCRTPolyMatrix::compact_bytes_batch_borrowed(&[&source]);
            reservation = dispatch.finish().unwrap().pop().unwrap();
            assert_eq!(serialized, vec![expected.clone()]);
            assert_eq!(uploaded.to_canonical_coefficients().unwrap(), payload);
            params.fence_released_memory();
            assert_eq!(storage.occupancy().unwrap().pinned_occupied_bytes(), 0);
            drop(uploaded);
            if wave == 0 {
                reservation.rearm(&requests).unwrap();
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_graph_scope_allows_external_work_but_enforces_dispatch_claims() {
        use crate::poly::dcrt::gpu::{GpuContextOpaque, PinnedHostBuffer};
        unsafe extern "C" {
            fn gpu_pinned_alloc(
                ctx: *mut GpuContextOpaque,
                bytes: usize,
                alignment: usize,
            ) -> *mut u8;
        }
        let (_, params) = parameters();
        let graph = GpuGraphAdmissionGuard::new(vec![params.clone()]).unwrap();
        let storage =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], None).unwrap();
        GpuPreparedStorage::finish_setup(&[&storage]).unwrap();
        drop(graph);
        // Ordinary caller-owned work is legal after execute has returned.
        drop(PinnedHostBuffer::<u64>::zeroed(&params, 2));
        let slot = storage.slot_identity(0).unwrap();
        let mut reservation = storage.reserve(&[slot.matrix_request(1, 1, true)]).unwrap();
        reservation.require_all_resources().unwrap();
        let dispatch = reservation.enter(Vec::new()).unwrap();
        // An active dispatch must not silently allocate unplanned host memory,
        // even though the surrounding caller's context is currently open.
        assert!(unsafe { gpu_pinned_alloc(params.ctx_raw(), 16, 8) }.is_null());
        let output = GpuDCRTPolyMatrix::new_empty_with_state(
            &params,
            1,
            1,
            params.crt_depth() - 1,
            true,
            None,
        );
        drop(dispatch.finish().unwrap());
        drop(output);
        drop(PinnedHostBuffer::<u64>::zeroed(&params, 2));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_pinned_cpu_ownership_and_exact_claims() {
        use crate::poly::dcrt::gpu::{GpuContextOpaque, PinnedHostBuffer};
        unsafe extern "C" {
            fn gpu_pinned_alloc(
                ctx: *mut GpuContextOpaque,
                bytes: usize,
                alignment: usize,
            ) -> *mut u8;
        }
        let (_, params) = parameters();
        let bytes = params.ring_dimension() as usize * std::mem::size_of::<u64>();
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: bytes * 2,
                alignment: 8,
                kind: GpuPreparedSlotKind::PinnedHost,
            }]),
        )
        .unwrap();
        let identity = storage.slot_identity(1).unwrap();
        let request = identity.workspace_request(bytes, 1);
        let before = storage.occupancy().unwrap();
        assert_eq!(before.pinned_capacity_bytes(), bytes * 2);
        assert_eq!(before.pinned_available_bytes(), bytes * 2);
        let mut reservation = storage.reserve(&[request]).unwrap();
        let reserved = storage.occupancy().unwrap();
        assert_eq!(reserved.reserved_bytes(), 0);
        assert_eq!(reserved.pinned_reserved_bytes(), bytes * 2);
        assert_eq!(reserved.pinned_occupied_bytes(), 0);
        assert_eq!(reserved.pinned_available_bytes(), 0);
        assert!(unsafe { gpu_pinned_alloc(params.ctx_raw(), bytes, 1) }.is_null());
        let dispatch = reservation.enter(Vec::new()).unwrap();
        assert!(unsafe { gpu_pinned_alloc(params.ctx_raw(), bytes + 1, 1) }.is_null());
        assert!(unsafe { gpu_pinned_alloc(params.ctx_raw(), bytes, 2) }.is_null());
        let source = (0..bytes).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
        let buffer = PinnedHostBuffer::from_slice(&params, &source);
        assert!(unsafe { gpu_pinned_alloc(params.ctx_raw(), bytes, 1) }.is_null());
        reservation = dispatch.finish().unwrap().pop().unwrap();
        let occupied = storage.occupancy().unwrap();
        assert_eq!(occupied.pinned_reserved_bytes(), 0);
        assert_eq!(occupied.pinned_occupied_bytes(), bytes);
        assert_eq!(occupied.pinned_high_water_bytes(), bytes);
        assert_eq!(occupied.occupied_bytes(), 0);
        assert_eq!(occupied.requested_capacity_bytes(), before.requested_capacity_bytes());
        assert!(reservation.rearm(&[request]).is_err());
        assert_eq!(buffer.as_slice(), source);
        drop(buffer);
        assert_eq!(storage.occupancy().unwrap().pinned_occupied_bytes(), 0);
        // A larger prepared backing does not enlarge the invocation's envelope.
        assert!(reservation.rearm(&[identity.workspace_request(bytes * 2, 1)]).is_err());
        reservation.rearm(&[request]).unwrap();
        let dispatch = reservation.enter(Vec::new()).unwrap();
        let buffer = PinnedHostBuffer::from_slice(&params, &source);
        drop(dispatch.finish().unwrap());
        assert!(!storage.fits(&[request]).unwrap());
        drop(storage);
        assert_eq!(buffer.as_slice(), source);
        drop(buffer);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_pinned_dma_reuse_requires_cpu_ready_slots() {
        use crate::sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler};
        let (cpu, params) = parameters();
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(rows > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, 1, DistType::FinRingDist);
        let reference = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original)
            .to_rns_snapshot()
            .bytes()
            .to_vec();
        let bytes = reference.len();
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, rows, 1)],
            Some(
                &[GpuPreparedWorkspaceLayout {
                    bytes,
                    alignment: 1,
                    kind: GpuPreparedSlotKind::PinnedHost,
                }; 2],
            ),
        )
        .unwrap();
        let matrix_request = storage.slot_identity(0).unwrap().matrix_request(rows, 1, true);
        let upload_request = storage.slot_identity(1).unwrap().workspace_request(bytes, 1);
        let download_request = storage.slot_identity(2).unwrap().workspace_request(bytes, 1);
        let mut matrix_reservation = storage.reserve(&[matrix_request]).unwrap();
        let mut upload_reservation = storage.reserve(&[upload_request]).unwrap();
        let mut download_reservation = storage.reserve(&[download_request]).unwrap();
        for wave in 0..2 {
            let dispatch =
                matrix_reservation.enter(vec![upload_reservation, download_reservation]).unwrap();
            let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
            let transfer = matrix.start_rns_snapshot(None);
            let mut reservations = dispatch.finish().unwrap();
            download_reservation = reservations.pop().unwrap();
            upload_reservation = reservations.pop().unwrap();
            matrix_reservation = reservations.pop().unwrap();
            // Transfer ownership alone blocks host reuse regardless of whether
            // this small DMA happened to finish before this assertion.
            assert!(download_reservation.rearm(&[download_request]).is_err());
            assert!(download_reservation.cpu_staging_ready().is_err());
            assert!(matrix_reservation.cpu_staging_ready().unwrap());
            let snapshot = transfer.finish();
            assert_eq!(snapshot.bytes(), reference);
            assert_eq!(matrix.to_cpu_matrix(), original);
            params.fence_released_memory(); // Explicit test-only completed boundary.
            let occupancy = storage.occupancy().unwrap();
            assert_eq!(occupancy.pinned_occupied_bytes(), bytes);
            assert_eq!(occupancy.pinned_available_bytes(), 0);
            assert!(download_reservation.rearm(&[download_request]).is_err());
            assert!(!storage.fits(&[upload_request]).unwrap());
            assert!(upload_reservation.cpu_staging_ready().unwrap());
            assert!(download_reservation.cpu_staging_ready().is_err());
            drop(snapshot);
            drop(matrix);
            assert_eq!(storage.occupancy().unwrap().pinned_occupied_bytes(), 0);
            assert!(download_reservation.cpu_staging_ready().unwrap());
            if wave == 0 {
                matrix_reservation.rearm(&[matrix_request]).unwrap();
                upload_reservation.rearm(&[upload_request]).unwrap();
                download_reservation.rearm(&[download_request]).unwrap();
            }
        }
        drop((matrix_reservation, upload_reservation, download_reservation));
        assert_eq!(storage.occupancy().unwrap().pinned_available_bytes(), bytes * 2);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_pinned_pending_upload_all_owners_drop() {
        use crate::{
            env::GPU_PREPARED_OCCUPANCY_TEST_CHILD,
            poly::dcrt::gpu::{detected_gpu_device_ids, gpu_device_memory_usage},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        // This assertion observes a process-global context count. Other test
        // groups may create/drop owners concurrently, so keep the original
        // lifetime check in its own process, as in the occupancy regressions.
        if std::env::var_os(GPU_PREPARED_OCCUPANCY_TEST_CHILD).is_none() {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "matrix::gpu_dcrt_poly::gpu_admission::tests::test_gpu_prepared_pinned_pending_upload_all_owners_drop",
                    "--nocapture",
                ])
                .env(GPU_PREPARED_OCCUPANCY_TEST_CHILD, "1")
                .output()
                .expect("run isolated prepared owner-drop unit test");
            assert!(
                result.status.success() &&
                    String::from_utf8_lossy(&result.stdout).contains("1 passed; 0 failed"),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr),
            );
            return;
        }
        let device = detected_gpu_device_ids()[0];
        let contexts = gpu_device_memory_usage(device).unwrap().live_contexts;
        let (cpu, params) = parameters();
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(rows > 0);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, 1, DistType::FinRingDist);
        let bytes = rows * params.ring_dimension() as usize * params.crt_depth() * 8;
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, rows, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                bytes,
                alignment: 1,
                kind: GpuPreparedSlotKind::PinnedHost,
            }]),
        )
        .unwrap();
        let dispatch = reserve(&storage, &[0, 1]).unwrap().enter(Vec::new()).unwrap();
        let matrix = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        drop(dispatch.finish().unwrap());
        // No transfer fence or materialization: the existing reclaimer must
        // retire host backing without retaining/destroying its own context.
        drop(storage);
        drop(params);
        drop(matrix);
        assert_eq!(gpu_device_memory_usage(device).unwrap().live_contexts, contexts);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_joint_prepared_occupancy_tracks_related_store_overlap() {
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let related = GpuDCRTPolyParams::new_with_gpu(
            params.ring_dimension(),
            cpu.to_crt().0,
            params.base_bits(),
            params.device_ids(),
            Some(1),
            Some(&params),
            None,
        );
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(rows > 0);
        let first = GpuPreparedStorage::new(
            (0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, rows, 1)).collect(),
            None,
        )
        .unwrap();
        let second =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&related, rows, 1)], None)
                .unwrap();
        let stores = [&first, &second];
        let fixed_request = first.slot_identity(0).unwrap().matrix_request(rows, 1, true);
        let claims = [
            first.slot_identity(1).unwrap().matrix_request(rows, 1, true),
            second.slot_identity(0).unwrap().matrix_request(rows, 1, true),
        ];
        assert!(
            GpuPreparedStorage::joint_occupancy(&[], GpuPreparedOccupancyMode::ResetCalibration)
                .is_err()
        );
        assert!(
            GpuPreparedStorage::joint_occupancy(
                &[&first],
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .is_err()
        );
        assert!(
            GpuPreparedStorage::joint_occupancy(
                &[&first, &first],
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .is_err()
        );
        let fixed_dispatch = first.reserve(&[fixed_request]).unwrap().enter(Vec::new()).unwrap();
        let fixed = GpuDCRTPolyMatrix::zero(&params, rows, 1);
        drop(fixed_dispatch.finish().unwrap());
        params.fence_released_memory();
        let baseline = first.demand(&[fixed_request]).unwrap().device_bytes;
        let demands = stores
            .iter()
            .zip(claims)
            .map(|(store, claim)| store.demand(&[claim]).unwrap().device_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            GpuPreparedStorage::joint_occupancy(
                &stores,
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .unwrap(),
            (baseline, baseline)
        );
        let reserved = second.reserve(&[claims[1]]).unwrap();
        assert!(
            GpuPreparedStorage::joint_occupancy(
                &stores,
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .is_err()
        );
        drop(reserved);
        // Non-overlapping actual claims must not add independent storage peaks.
        for (index, parameters) in [&params, &related].into_iter().enumerate() {
            let dispatch =
                stores[index].reserve(&[claims[index]]).unwrap().enter(Vec::new()).unwrap();
            let output = GpuDCRTPolyMatrix::zero(parameters, rows, 1);
            drop(dispatch.finish().unwrap());
            drop(output);
            parameters.fence_released_memory();
            stores[index].occupancy().unwrap();
        }
        let (occupied, peak) =
            GpuPreparedStorage::joint_occupancy(&stores, GpuPreparedOccupancyMode::Observe)
                .unwrap();
        assert_eq!(occupied, baseline);
        assert_eq!(peak, baseline + demands.iter().max().unwrap());
        assert!(
            peak < stores
                .iter()
                .map(|store| { store.occupancy().unwrap().occupied_high_water_bytes() })
                .sum()
        );
        assert_eq!(
            GpuPreparedStorage::joint_occupancy(
                &stores,
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .unwrap(),
            (baseline, baseline)
        );
        let reservations = stores
            .iter()
            .zip(claims)
            .map(|(store, claim)| store.reserve(&[claim]).unwrap())
            .collect::<Vec<_>>();
        // Actual claims from existing Rayon workers update one joint maximum.
        let outputs = reservations
            .into_par_iter()
            .zip([&params, &related])
            .map(|(reservation, parameters)| {
                let dispatch = reservation.enter(Vec::new()).unwrap();
                let output = GpuDCRTPolyMatrix::zero(parameters, rows, 1);
                drop(dispatch.finish().unwrap());
                output
            })
            .collect::<Vec<_>>();
        let total = baseline + demands.iter().sum::<usize>();
        assert_eq!(
            GpuPreparedStorage::joint_occupancy(&stores, GpuPreparedOccupancyMode::Observe)
                .unwrap(),
            (total, total)
        );
        drop((outputs, fixed));
        params.fence_released_memory();
        assert_eq!(
            GpuPreparedStorage::joint_occupancy(
                &stores,
                GpuPreparedOccupancyMode::ResetCalibration
            )
            .unwrap(),
            (0, 0)
        );
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_composed_dispatch_reuses_scratch_and_retains_all_related_context_outputs() {
        use crate::{
            matrix::gpu_dcrt_poly::GpuMatrixCrtOperation,
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;

        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let rows = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(2);
        assert!(rows > 0);
        let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let target_cpu = DCRTPolyParams::new(n, 2, 17, 4, Some(cpu.to_crt().0[..2].to_vec()), None);
        let target = GpuDCRTPolyParams::new(n, target_cpu.to_crt().0, 4, None);
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            target.device_ids(),
            Some(1),
            Some(&target),
            None,
        );
        let layout = target
            .matrix_crt_workspace_bytes(
                target.crt_depth() - 1,
                rows,
                1,
                GpuMatrixCrtOperation::ConvertModulus,
                source_params.crt_depth(),
                1,
                false,
            )
            .unwrap();
        assert!(layout.additional_bytes > 0);
        let source_store =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&source_params, rows, 1)], None)
                .unwrap();
        let output_store = GpuPreparedStorage::new(
            [(rows, 1), (1, rows), (1, rows)]
                .into_par_iter()
                .map(|(rows, columns)| GpuDCRTPolyMatrix::zero(&target, rows, columns))
                .collect(),
            None,
        )
        .unwrap();
        let workspace_store = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&target, 1, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: layout.additional_bytes,
                alignment: layout.alignment,
                kind: GpuPreparedSlotKind::TransformWorkspace,
            }]),
        )
        .unwrap();
        let inputs = (0..2)
            .into_par_iter()
            .map(|_| {
                DCRTPolyUniformSampler::new().sample_uniform(&cpu, rows, 1, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let source_request = source_store.slot_identity(0).unwrap().matrix_request(rows, 1, true);
        let temporary_request =
            output_store.slot_identity(0).unwrap().matrix_request(rows, 1, true);
        let workspace_request = workspace_store
            .slot_identity(1)
            .unwrap()
            .workspace_request(layout.additional_bytes, layout.alignment);
        let mut source_reservation = source_store.reserve(&[source_request]).unwrap();
        let mut temporary_reservation = output_store.reserve(&[temporary_request]).unwrap();
        let mut workspace_reservation = workspace_store.reserve(&[workspace_request]).unwrap();
        // Reserve every final destination before submitting the first wave.
        let output_reservations =
            reserve(&output_store, &[1, 2]).unwrap().partition(&[1, 1]).unwrap();
        let mut readers = Vec::new();
        for (wave, (input, output_reservation)) in
            inputs.iter().zip(output_reservations).enumerate()
        {
            let dispatch = source_reservation
                .enter(vec![temporary_reservation, workspace_reservation, output_reservation])
                .unwrap();
            // A context present later in the itinerary cannot skip the next
            // source allocation. Rejection must leave the complete plan usable.
            let mut wrong = std::ptr::null_mut();
            assert_ne!(
                unsafe {
                    gpu_matrix_create(
                        target.ctx_raw(),
                        1,
                        rows,
                        1,
                        GPU_POLY_FORMAT_EVAL,
                        &mut wrong,
                        true,
                    )
                },
                0
            );
            assert!(wrong.is_null());
            let source = GpuDCRTPolyMatrix::from_cpu_matrix(&source_params, input);
            let temporary = source.reduce_modulus(&target);
            readers.push(temporary.transpose());
            drop(temporary);
            drop(source);
            // Empty host-only results remain valid after the last device claim
            // on any participating context, including a non-first context.
            assert_eq!(GpuDCRTPolyMatrix::zero(&target, rows, 0).size(), (rows, 0));
            let mut reservations = dispatch.finish().unwrap();
            assert_eq!(reservations.len(), 4);
            drop(reservations.pop().unwrap());
            workspace_reservation = reservations.pop().unwrap();
            temporary_reservation = reservations.pop().unwrap();
            source_reservation = reservations.pop().unwrap();
            assert!(!source_store.fits(&[source_request]).unwrap());
            assert!(!output_store.fits(&[temporary_request]).unwrap());
            assert!(!workspace_store.fits(&[workspace_request]).unwrap());
            if wave + 1 < inputs.len() {
                source_reservation.rearm(&[source_request]).unwrap();
                temporary_reservation.rearm(&[temporary_request]).unwrap();
                workspace_reservation.rearm(&[workspace_request]).unwrap();
            }
        }
        drop((source_reservation, temporary_reservation, workspace_reservation));
        drop((source_store, output_store, workspace_store, source_params, target));
        readers.par_iter().zip(&inputs).for_each(|(reader, input)| {
            assert_eq!(reader.to_cpu_matrix(), input.reduce_modulus(&target_cpu).transpose());
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_composed_dispatch_rejects_foreign_owners_and_cancels_partial_plans() {
        let (cpu, params) = parameters();
        let foreign = GpuDCRTPolyParams::new(cpu.ring_dimension(), cpu.to_crt().0, 4, None);
        let first =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 2, 1)], None).unwrap();
        let second = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 2, 1), GpuDCRTPolyMatrix::zero(&params, 2, 1)],
            None,
        )
        .unwrap();
        let other =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&foreign, 2, 1)], None).unwrap();
        assert!(
            reserve(&first, &[0]).unwrap().enter(vec![reserve(&other, &[0]).unwrap()]).is_err()
        );
        drop(reserve(&first, &[0]).unwrap());
        drop(reserve(&other, &[0]).unwrap());
        let reservation = reserve(&first, &[0]).unwrap();
        let duplicate = [reservation.raw.unwrap().as_ptr(); 2];
        let mut rejected = std::ptr::null_mut();
        assert_ne!(
            unsafe {
                gpu_matrix_dispatch_enter(duplicate.as_ptr(), duplicate.len(), &mut rejected)
            },
            0
        );
        assert!(rejected.is_null());
        let dispatch = reservation.enter(vec![reserve(&second, &[0, 1]).unwrap()]).unwrap();
        let first_output = GpuDCRTPolyMatrix::zero(&params, 2, 1);
        let second_output = GpuDCRTPolyMatrix::zero(&params, 2, 1);
        assert!(dispatch.finish().is_err());
        // Finishing an incomplete command cancels every unconsumed claim, but
        // neither published output becomes free while its native owner is live.
        assert!(reserve(&first, &[0]).is_err());
        assert!(reserve(&second, &[0]).is_err());
        drop(reserve(&second, &[1]).unwrap());
        assert_eq!(first.occupancy().unwrap().active_reservations(), 0);
        assert_eq!(second.occupancy().unwrap().active_reservations(), 0);
        assert_eq!(second.occupancy().unwrap().reserved_bytes(), 0);
        assert_eq!(first_output.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 2, 1));
        assert_eq!(second_output.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 2, 1));
        drop((first_output, second_output));
        drop(reserve(&first, &[0]).unwrap());
        drop(reserve(&second, &[0, 1]).unwrap());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_wave_reservations_keep_rayon_children_and_pending_readers_exclusive() {
        use crate::poly::dcrt::gpu::gpu_matrix_transpose;
        use rayon::prelude::*;

        let (cpu, params) = parameters();
        let maximum = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(6);
        assert!(maximum >= 3);
        let widths = [maximum - 1, 1, maximum - 2, maximum - 1];
        // Separate complete destinations are prepared before admission. They
        // retain every wave's reader so no host copy gates scratch reuse.
        let readers = (0..2)
            .into_par_iter()
            .map(|_| {
                widths
                    .into_par_iter()
                    .map(|width| GpuDCRTPolyMatrix::zero(&params, width, maximum))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let backing = (0..2)
            .into_par_iter()
            .map(|_| GpuDCRTPolyMatrix::zero(&params, maximum, maximum))
            .collect();
        let storage = GpuPreparedStorage::new(backing, None).unwrap();
        let requests = (0..2)
            .map(|index| {
                storage.slot_identity(index).unwrap().matrix_request(maximum, widths[0], true)
            })
            .collect::<Vec<_>>();
        let children = storage.reserve(&requests).unwrap().partition(&[1, 1]).unwrap();
        let results = children
            .into_par_iter()
            .zip(readers)
            .enumerate()
            .map(|(index, (mut reservation, readers))| {
                let identity = storage.slot_identity(index).unwrap();
                assert!(reservation.rearm(&[requests[index]]).is_err());
                for (wave, (&width, reader)) in widths.iter().zip(&readers).enumerate() {
                    let dispatch = reservation.enter(Vec::new()).unwrap();
                    let source = GpuDCRTPolyMatrix::identity_columns(&params, maximum, 0, width);
                    assert_eq!(
                        unsafe { gpu_matrix_transpose(reader.raw, source.raw, std::ptr::null()) },
                        0
                    );
                    reservation = dispatch.finish().unwrap().pop().unwrap();
                    assert!(
                        reservation
                            .rearm(&[identity.matrix_request(maximum, width, true)])
                            .is_err()
                    );
                    assert!(!storage.fits(&[requests[index]]).unwrap());
                    drop(source);
                    // The host owner is gone but GPU readers may still be
                    // pending. Neither a competing reservation nor an oversized
                    // next wave can claim this retained invocation footprint.
                    assert!(storage.reserve(&[requests[index]]).is_err());
                    assert!(
                        reservation
                            .rearm(&[identity.matrix_request(maximum, maximum, true)])
                            .is_err()
                    );
                    assert!(
                        reservation
                            .rearm(&[identity.matrix_request(maximum, width, false)])
                            .is_err()
                    );
                    if let Some(&next_width) = widths.get(wave + 1) {
                        reservation
                            .rearm(&[identity.matrix_request(maximum, next_width, true)])
                            .unwrap();
                    }
                }
                (reservation, readers)
            })
            .collect::<Vec<_>>();
        let occupancy = storage.occupancy().unwrap();
        assert_eq!(occupancy.active_reservations(), 2);
        assert_eq!(occupancy.reserved_bytes(), 0);
        assert_eq!(occupancy.available_capacity_bytes(), 0);
        let readers = results
            .into_iter()
            .map(|(reservation, readers)| {
                drop(reservation);
                readers
            })
            .collect::<Vec<_>>();
        assert_eq!(storage.occupancy().unwrap().active_reservations(), 0);
        assert!(storage.fits(&requests).unwrap());
        drop(storage);
        readers.par_iter().for_each(|group| {
            group.par_iter().zip(widths).for_each(|(reader, width)| {
                let expected = DCRTPolyMatrix::identity(&cpu, maximum, None)
                    .slice(0, maximum, 0, width)
                    .transpose();
                assert_eq!(reader.to_cpu_matrix(), expected);
            });
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_wave_rearm_rolls_back_partial_claims_and_preserves_live_outputs() {
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        assert!(columns >= 2);
        let storage = GpuPreparedStorage::new(
            (0..2).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 2, columns)).collect(),
            None,
        )
        .unwrap();
        let requests = (0..2)
            .map(|index| storage.slot_identity(index).unwrap().matrix_request(2, columns, true))
            .collect::<Vec<_>>();
        let dispatch = storage.reserve(&requests).unwrap().enter(Vec::new()).unwrap();
        let first = GpuDCRTPolyMatrix::zero(&params, 2, columns);
        let second = GpuDCRTPolyMatrix::zero(&params, 2, columns);
        let mut reservation = dispatch.finish().unwrap().pop().unwrap();
        drop(first);
        assert!(reservation.rearm(&requests).is_err());
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0);
        assert_eq!(storage.occupancy().unwrap().available_capacity_bytes(), 0);
        assert!(storage.reserve(&requests[..1]).is_err());
        drop(second);
        let mut reversed = requests.clone();
        reversed.reverse();
        assert!(reservation.rearm(&reversed).is_err());
        reservation.rearm(&requests).unwrap();
        let mut children = reservation.partition(&[1, 1]).unwrap();
        drop(children.pop().unwrap());
        assert!(storage.fits(&requests[1..]).unwrap());
        let dispatch = children.pop().unwrap().enter(Vec::new()).unwrap();
        let result = GpuDCRTPolyMatrix::zero(&params, 2, columns);
        drop(dispatch.finish().unwrap());
        assert!(!storage.fits(&requests[..1]).unwrap());
        assert_eq!(result.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 2, columns));
        drop(result);
        assert!(storage.fits(&requests).unwrap());
        assert_eq!(storage.occupancy().unwrap().active_reservations(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_wave_workspace_reuses_original_envelope_without_releasing_capacity() {
        use crate::{
            poly::dcrt::gpu::{GpuMatrixBatchOperation, gpu_matrix_negate_batch},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;

        let (cpu, params) = parameters();
        let mut count = 2usize;
        let requirements = loop {
            let requirements = params
                .matrix_batch_workspace_bytes(
                    params.crt_depth() - 1,
                    (1, 1),
                    count,
                    1,
                    GpuMatrixBatchOperation::Negate,
                    false,
                )
                .unwrap();
            if requirements.additional_bytes != 0 {
                break requirements;
            }
            count = count.checked_mul(2).unwrap();
        };
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let outputs = (0..3)
            .into_par_iter()
            .map(|_| {
                (0..count)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: requirements.additional_bytes * 2,
                alignment: requirements.alignment,
                kind: GpuPreparedSlotKind::BatchWorkspace,
            }]),
        )
        .unwrap();
        let slot = storage.slot_identity(1).unwrap();
        let request = slot.workspace_request(requirements.additional_bytes, requirements.alignment);
        let mut reservation = storage.reserve(&[request]).unwrap();
        let inputs = vec![input.raw.cast_const(); count];
        for (wave, outputs) in outputs.iter().enumerate() {
            let dispatch = reservation.enter(Vec::new()).unwrap();
            let pointers = outputs.iter().map(|matrix| matrix.raw).collect::<Vec<_>>();
            assert_eq!(
                unsafe {
                    gpu_matrix_negate_batch(
                        pointers.as_ptr(),
                        inputs.as_ptr(),
                        std::ptr::null(),
                        count,
                    )
                },
                0,
                "{}",
                last_error_string()
            );
            reservation = dispatch.finish().unwrap().pop().unwrap();
            assert!(!storage.fits(&[request]).unwrap());
            assert!(storage.reserve(&[request]).is_err());
            assert!(
                reservation
                    .rearm(&[slot.workspace_request(
                        requirements.additional_bytes * 2,
                        requirements.alignment,
                    )])
                    .is_err()
            );
            if wave + 1 < 3 {
                reservation.rearm(&[request]).unwrap();
            }
        }
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0);
        drop(reservation);
        assert!(storage.fits(&[request]).unwrap());
        drop(storage);
        let expected = original.negate_out_of_place();
        outputs.par_iter().for_each(|group| {
            group.par_iter().for_each(|output| {
                assert_eq!(output.to_cpu_matrix(), expected);
            })
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_shapes_preserve_readers_transfers_and_logical_pressure() {
        use crate::{
            env::GPU_PREPARED_OCCUPANCY_TEST_CHILD,
            poly::dcrt::gpu::{gpu_default_mempool_reset_high_water, gpu_default_mempool_usage},
        };
        use rayon::prelude::*;

        if std::env::var_os(GPU_PREPARED_OCCUPANCY_TEST_CHILD).is_none() {
            let module = module_path!().split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(format!("{module}::test_gpu_prepared_shapes_preserve_readers_transfers_and_logical_pressure"))
                .env(GPU_PREPARED_OCCUPANCY_TEST_CHILD, "1").output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let maximum = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(7);
        assert!(maximum >= 5);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for moduli in [vec![narrow, wide], vec![wide, narrow]] {
            let cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(moduli.clone()), None);
            let params = GpuDCRTPolyParams::new(n, moduli.clone(), 4, None);
            let shapes = [(2, 1), (maximum, maximum), (maximum - 1, maximum - 2), (3, 2)];
            let backing = (0..shapes.len() + 1)
                .into_par_iter()
                .map(|_| GpuDCRTPolyMatrix::zero(&params, maximum, maximum))
                .collect::<Vec<_>>();
            let original_pointer = backing[0].raw;
            let mut storage = GpuPreparedStorage::new(backing, None).unwrap();
            let identity = storage.slot_identity(0).unwrap();
            params.fence_released_memory();
            storage.reset_occupied_high_water().unwrap();
            gpu_default_mempool_reset_high_water(storage.device()).unwrap();
            let pool = gpu_default_mempool_usage(storage.device()).unwrap();
            let mut readers = Vec::new();
            for (index, (rows, columns)) in shapes.into_iter().enumerate() {
                let is_ntt = index % 2 == 0;
                let requests = [
                    identity.matrix_request(rows, columns, true),
                    storage.slot_identity(index + 1).unwrap().matrix_request(columns, rows, is_ntt),
                ];
                assert!(storage.fits(&requests).unwrap());
                let dispatch = storage.reserve(&requests).unwrap().enter(Vec::new()).unwrap();
                let mut source = GpuDCRTPolyMatrix::identity_columns(&params, rows, 0, columns);
                assert_eq!(source.raw, original_pointer);
                if !is_ntt {
                    source.intt_all_in_place();
                }
                let reader = source.transpose();
                assert_eq!(source.size(), (rows, columns));
                assert_eq!(reader.size(), (columns, rows));
                if index == 0 {
                    let layout = params.matrix_allocation_bytes(1, rows, columns, true).unwrap();
                    assert_eq!(
                        storage.occupancy().unwrap().occupied_high_water_bytes(),
                        2 * (layout.data_bytes + layout.aux_bytes)
                    );
                    assert!(
                        layout.data_bytes + layout.aux_bytes < identity.requested_backing_bytes()
                    );
                }
                drop(source);
                drop(dispatch.finish().unwrap());
                // Retain pending readers while the next shape overwrites the
                // same source. No materialization or completion wait intervenes.
                readers.push(reader);
                let after = gpu_default_mempool_usage(storage.device()).unwrap();
                assert_eq!(after.used_current, pool.used_current);
                assert_eq!(after.used_high, pool.used_high);
                assert_eq!(after.reserved_current, pool.reserved_current);
                assert_eq!(storage.slot_identity(0), Some(identity));
            }
            let observation = storage.occupancy().unwrap();
            assert!(
                observation.occupied_high_water_bytes() < observation.requested_capacity_bytes()
            );
            drop(storage);
            // A second execution owner is needed only for the transfer checks,
            // after the single-context pool measurement has ended.
            let peer = GpuDCRTPolyParams::new_with_gpu(
                n,
                moduli,
                4,
                params.device_ids(),
                None,
                None,
                None,
            );
            readers.into_par_iter().zip(shapes).for_each(|(reader, (rows, columns))| {
                let expected = DCRTPolyMatrix::identity(&cpu, rows, None)
                    .slice(0, rows, 0, columns)
                    .transpose();
                let copied =
                    reader.copy_to_params_direct(&peer).expect("direct copy of logical prefix");
                let bytes = reader.into_cpu_staging_bytes();
                let layout = GpuDCRTPolyMatrix::cpu_staging_layout(&params, &bytes).unwrap();
                assert_eq!((layout.rows, layout.columns), (columns, rows));
                let restored = GpuDCRTPolyMatrix::from_cpu_staging_bytes(&peer, &bytes);
                assert_eq!(copied.to_cpu_matrix(), expected);
                assert_eq!(restored.to_cpu_matrix(), expected);
            });
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_fit_rechecks_ownership_and_freezes_the_actual_request() {
        let (cpu, params) = parameters();
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 3, 7), GpuDCRTPolyMatrix::zero(&params, 3, 7)],
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: 64,
                alignment: 8,
                kind: GpuPreparedSlotKind::BatchWorkspace,
            }]),
        )
        .unwrap();
        let first = storage.slot_identity(0).unwrap();
        let second = storage.slot_identity(1).unwrap();
        let workspace = storage.slot_identity(2).unwrap();
        let request = first.matrix_request(3, 2, true);
        assert!(storage.fits(&[request]).unwrap());
        for oversized in [
            first.matrix_request(4, 2, true),
            first.matrix_request(3, 8, true),
            workspace.workspace_request(65, 8),
            workspace.workspace_request(32, 16),
        ] {
            assert!(!storage.fits(&[oversized]).unwrap());
            assert!(storage.reserve(&[request, oversized]).is_err());
            assert_eq!(storage.occupancy().unwrap().reserved_bytes(), 0);
        }
        assert!(storage.fits(&[first.matrix_request(0, 2, true)]).is_err());
        assert!(storage.fits(&[request, request]).is_err());
        let mut stale = request;
        stale.slot_id += 1;
        assert!(storage.fits(&[stale]).is_err());
        assert!(storage.reserve(&[stale]).is_err());

        let both = [request, second.matrix_request(3, 2, true)];
        assert!(storage.fits(&both).unwrap());
        let competing = storage.reserve(&both[1..]).unwrap();
        assert!(!storage.fits(&both).unwrap());
        assert!(storage.reserve(&both).is_err());
        assert!(storage.fits(&[request]).unwrap(), "failed atomic reserve must return its prefix");
        assert!(!storage.fits(&both[1..]).unwrap());
        drop(competing);
        let dispatch = storage.reserve(&[request]).unwrap().enter(Vec::new()).unwrap();
        for (columns, format) in [(1, GPU_POLY_FORMAT_EVAL), (2, GPU_POLY_FORMAT_COEFF)] {
            let mut output = std::ptr::null_mut();
            assert_ne!(
                unsafe {
                    gpu_matrix_create(params.ctx_raw(), 1, 3, columns, format, &mut output, true)
                },
                0
            );
            assert!(output.is_null());
            assert_eq!(storage.occupancy().unwrap().occupied_bytes(), 0);
        }
        let output = GpuDCRTPolyMatrix::zero(&params, 3, 2);
        drop(dispatch.finish().unwrap());
        assert!(!storage.fits(&[request]).unwrap());
        assert!(storage.fits(&[workspace.workspace_request(32, 8)]).unwrap());
        assert_eq!(output.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 3, 2));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_smaller_output_uses_its_queried_auxiliary_layout() {
        use crate::{
            poly::dcrt::gpu::GpuMatrixBatchOperation,
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;
        use std::sync::Arc;
        let (cpu, params) = parameters();
        let mut count = 2;
        let required = loop {
            let required = params
                .matrix_batch_workspace_bytes(
                    1,
                    (1, 1),
                    count,
                    1,
                    GpuMatrixBatchOperation::Negate,
                    false,
                )
                .unwrap();
            if required.additional_bytes != 0 {
                break required;
            }
            count *= 2;
        };
        assert_eq!(
            params
                .matrix_batch_workspace_bytes(
                    1,
                    (1, 2),
                    count,
                    1,
                    GpuMatrixBatchOperation::Negate,
                    false
                )
                .unwrap()
                .additional_bytes,
            0
        );
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let source = Arc::new(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &input));
        let backing = (0..count)
            .into_par_iter()
            .map(|index| GpuDCRTPolyMatrix::zero(&params, 1, if index == 0 { 2 } else { 1 }))
            .collect();
        let storage = GpuPreparedStorage::new(
            backing,
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: required.additional_bytes * 2,
                alignment: required.alignment,
                kind: GpuPreparedSlotKind::BatchWorkspace,
            }]),
        )
        .unwrap();
        let mut requests = (0..count)
            .map(|index| storage.slot_identity(index).unwrap().matrix_request(1, 1, true))
            .collect::<Vec<_>>();
        requests.push(
            storage
                .slot_identity(count)
                .unwrap()
                .workspace_request(required.additional_bytes, required.alignment),
        );
        assert!(storage.fits(&requests).unwrap());
        let dispatch = storage.reserve(&requests).unwrap().enter(Vec::new()).unwrap();
        let outputs = GpuDCRTPolyMatrix::negate_batch_out_of_place(vec![source; count]);
        drop(dispatch.finish().unwrap());
        let layout = params.matrix_allocation_bytes(1, 1, 1, true).unwrap();
        assert_eq!(
            storage.occupancy().unwrap().occupied_high_water_bytes(),
            count * (layout.data_bytes + layout.aux_bytes) + required.additional_bytes
        );
        drop(storage);
        let expected = input.negate_out_of_place();
        outputs.par_iter().for_each(|output| assert_eq!(output.to_cpu_matrix(), expected));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_reservation_children_run_on_rayon_and_cancel_independently() {
        use rayon::prelude::*;
        let (cpu, params) = parameters();
        let backing = [(3, 2), (2, 3), (3, 2), (2, 3), (3, 2), (2, 3)]
            .into_par_iter()
            .map(|(rows, columns)| GpuDCRTPolyMatrix::zero(&params, rows, columns))
            .collect();
        let mut storage = GpuPreparedStorage::new(backing, None).unwrap();
        let parent = reserve(&storage, &[0, 1, 2, 3, 4, 5]).unwrap();
        let capacity = storage.occupancy().unwrap().reserved_bytes();
        let mut rejected = std::ptr::null_mut();
        // The native boundary also rejects incomplete partitions without
        // consuming the parent, independently of the Rust length checks.
        assert_ne!(
            unsafe {
                gpu_matrix_reservation_partition(
                    parent.raw.unwrap().as_ptr(),
                    [1usize].as_ptr(),
                    1,
                    &mut rejected,
                )
            },
            0
        );
        assert!(rejected.is_null());
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), capacity);
        let mut children = parent.partition(&[2, 2, 2]).unwrap();
        let partitioned = storage.occupancy().unwrap();
        assert_eq!(partitioned.reserved_bytes(), capacity);
        assert_eq!(partitioned.occupied_bytes(), 0);
        assert_eq!(partitioned.active_reservations(), 3);
        assert!(storage.reset_occupied_high_water().is_err());
        let cancelled = children.pop().unwrap();
        let cancelled_bytes = cancelled
            .slot_identities()
            .iter()
            .map(GpuPreparedSlotIdentity::requested_backing_bytes)
            .sum::<usize>();
        drop(cancelled);
        assert_eq!(storage.occupancy().unwrap().reserved_bytes(), capacity - cancelled_bytes);
        drop(reserve(&storage, &[4, 5]).unwrap());
        assert!(reserve(&storage, &[0]).is_err());

        let readers = children
            .into_par_iter()
            .enumerate()
            .map(|(index, child)| {
                assert_eq!(child.slot_identities()[0].slot_index(), index * 2);
                let dispatch = child.enter(Vec::new()).unwrap();
                let source = GpuDCRTPolyMatrix::identity_columns(&params, 3, index, 2);
                let reader = source.transpose();
                drop(source);
                drop(dispatch.finish().unwrap());
                reader
            })
            .collect::<Vec<_>>();
        let finished = storage.occupancy().unwrap();
        assert_eq!(finished.active_reservations(), 0);
        assert_eq!(finished.reserved_bytes(), 0);
        assert_eq!(finished.occupied_high_water_bytes(), capacity - cancelled_bytes);
        assert!(reserve(&storage, &[4, 5]).unwrap().partition(&[1]).is_err());
        assert!(reserve(&storage, &[4, 5]).unwrap().partition(&[1, usize::MAX]).is_err());
        drop(reserve(&storage, &[4, 5]).unwrap());
        assert!(reserve(&storage, &[]).unwrap().partition(&[]).unwrap().is_empty());
        assert_eq!(storage.occupancy().unwrap().active_reservations(), 0);
        drop(storage);
        readers.par_iter().enumerate().for_each(|(index, reader)| {
            let expected =
                DCRTPolyMatrix::identity(&cpu, 3, None).slice(0, 3, index, index + 2).transpose();
            assert_eq!(reader.to_cpu_matrix(), expected);
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_batch_workspace_checks_layout_and_reuses_after_consumers() {
        use crate::{
            poly::dcrt::gpu::{GpuMatrixBatchOperation, gpu_matrix_negate_batch},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;

        let (cpu, params) = parameters();
        let level = params.crt_depth() - 1;
        let mut count = 2usize;
        let requirements = loop {
            let requirements = params
                .matrix_batch_workspace_bytes(
                    level,
                    (1, 1),
                    count,
                    1,
                    GpuMatrixBatchOperation::Negate,
                    false,
                )
                .unwrap();
            if requirements.additional_bytes != 0 {
                break requirements;
            }
            count = count.checked_mul(2).unwrap();
        };
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let input = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        let expected = original.negate_out_of_place();
        let layout = GpuPreparedWorkspaceLayout {
            bytes: requirements.additional_bytes,
            alignment: requirements.alignment,
            kind: GpuPreparedSlotKind::BatchWorkspace,
        };
        let workspaces = [
            GpuPreparedWorkspaceLayout { kind: GpuPreparedSlotKind::TransformWorkspace, ..layout },
            GpuPreparedWorkspaceLayout { bytes: layout.bytes - 1, ..layout },
            GpuPreparedWorkspaceLayout { alignment: 1, ..layout },
            layout,
        ];
        let backing = (0..count * 2)
            .into_par_iter()
            .map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1))
            .collect();
        let mut storage = GpuPreparedStorage::new(backing, Some(&workspaces)).unwrap();
        let workspace_start = count * 2;
        let identity = storage.slot_identity(workspace_start + 3).unwrap();
        assert_eq!(identity.kind(), GpuPreparedSlotKind::BatchWorkspace);
        assert_eq!(identity.alignment(), layout.alignment);
        assert_eq!(identity.level(), None);
        assert_eq!(identity.requested_backing_bytes(), layout.bytes);
        let dispatch =
            reserve(&storage, &(0..count).collect::<Vec<_>>()).unwrap().enter(Vec::new()).unwrap();
        // Allocation order is part of the thread-bound native permit. Rayon is
        // used for independent setup and observations, outside that dispatch.
        let first = (0..count).map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1)).collect::<Vec<_>>();
        drop(dispatch.finish().unwrap());
        let outputs = first.iter().map(|matrix| matrix.raw).collect::<Vec<_>>();
        let inputs = vec![input.raw.cast_const(); count];
        let fixed = storage.reset_occupied_high_water().unwrap().occupied_bytes();
        for index in workspace_start..workspace_start + 3 {
            let dispatch = reserve(&storage, &[index]).unwrap().enter(Vec::new()).unwrap();
            let status = unsafe {
                gpu_matrix_negate_batch(outputs.as_ptr(), inputs.as_ptr(), std::ptr::null(), count)
            };
            assert_ne!(status, 0, "wrong workspace layout reached the batch kernel");
            assert_eq!(storage.occupancy().unwrap().occupied_high_water_bytes(), fixed);
            assert!(dispatch.finish().is_err(), "rejected workspace must remain unconsumed");
        }
        let dispatch =
            reserve(&storage, &[workspace_start + 3]).unwrap().enter(Vec::new()).unwrap();
        assert_eq!(
            unsafe {
                gpu_matrix_negate_batch(outputs.as_ptr(), inputs.as_ptr(), std::ptr::null(), count)
            },
            0,
            "{}",
            last_error_string()
        );
        drop(dispatch.finish().unwrap());
        // Retain every first-wave output; reuse the same workspace immediately
        // on the second wave's stream without polling its completion.
        let slots =
            (count..count * 2).chain(std::iter::once(workspace_start + 3)).collect::<Vec<_>>();
        let dispatch = reserve(&storage, &slots).unwrap().enter(Vec::new()).unwrap();
        let second = (0..count).map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1)).collect::<Vec<_>>();
        let outputs = second.iter().map(|matrix| matrix.raw).collect::<Vec<_>>();
        assert_eq!(
            unsafe {
                gpu_matrix_negate_batch(outputs.as_ptr(), inputs.as_ptr(), std::ptr::null(), count)
            },
            0,
            "{}",
            last_error_string()
        );
        drop(dispatch.finish().unwrap());
        assert_eq!(
            storage.occupancy().unwrap().occupied_high_water_bytes(),
            fixed * 2 + layout.bytes
        );
        drop(storage);
        first.par_iter().chain(second.par_iter()).for_each(|output| {
            assert_eq!(output.to_cpu_matrix(), expected);
        });
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_transform_workspace_preserves_related_context_readers() {
        use crate::{
            matrix::gpu_dcrt_poly::GpuMatrixCrtOperation,
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        use rayon::prelude::*;

        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 3, 17, 4, None, None);
        let target_cpu = DCRTPolyParams::new(n, 2, 17, 4, Some(cpu.to_crt().0[..2].to_vec()), None);
        let target = GpuDCRTPolyParams::new(n, target_cpu.to_crt().0, 4, None);
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            target.device_ids(),
            Some(1),
            Some(&target),
            None,
        );
        let requirements = target
            .matrix_crt_workspace_bytes(
                target.crt_depth() - 1,
                2,
                1,
                GpuMatrixCrtOperation::ConvertModulus,
                source_params.crt_depth(),
                1,
                false,
            )
            .unwrap();
        assert!(requirements.additional_bytes > 0);
        let input = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 1, DistType::FinRingDist);
        let expected = input.reduce_modulus(&target_cpu).transpose();
        let source = GpuDCRTPolyMatrix::from_cpu_matrix(&source_params, &input);
        let backing = [(2, 1), (1, 2), (2, 1), (1, 2)]
            .into_par_iter()
            .map(|(rows, columns)| GpuDCRTPolyMatrix::zero(&target, rows, columns))
            .collect();
        let storage = GpuPreparedStorage::new(
            backing,
            Some(&[GpuPreparedWorkspaceLayout {
                bytes: requirements.additional_bytes,
                alignment: requirements.alignment,
                kind: GpuPreparedSlotKind::TransformWorkspace,
            }]),
        )
        .unwrap();
        let mut readers = Vec::new();
        for slots in [[0, 4, 1], [2, 4, 3]] {
            let dispatch = reserve(&storage, &slots).unwrap().enter(Vec::new()).unwrap();
            let output = source.reduce_modulus(&target);
            readers.push(output.transpose());
            drop(output);
            drop(dispatch.finish().unwrap());
        }
        let observation = storage.occupancy().unwrap();
        assert!(observation.occupied_high_water_bytes() >= requirements.additional_bytes);
        assert_eq!(observation.reserved_bytes(), 0);
        drop(storage);
        drop(source);
        drop(source_params);
        readers.par_iter().for_each(|reader| assert_eq!(reader.to_cpu_matrix(), expected));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_occupancy_measures_claims_without_pool_growth() {
        use crate::{
            env::GPU_PREPARED_OCCUPANCY_TEST_CHILD,
            poly::dcrt::gpu::{gpu_default_mempool_reset_high_water, gpu_default_mempool_usage},
        };

        if std::env::var_os(GPU_PREPARED_OCCUPANCY_TEST_CHILD).is_none() {
            let module = module_path!().split_once("::").unwrap().1;
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg(format!(
                    "{module}::test_gpu_prepared_occupancy_measures_claims_without_pool_growth"
                ))
                .env(GPU_PREPARED_OCCUPANCY_TEST_CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }

        let (cpu, params) = parameters();
        let mut storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 3, 2), GpuDCRTPolyMatrix::zero(&params, 3, 2)],
            None,
        )
        .unwrap();
        // Only this explicit setup/measurement boundary waits. Subsequent
        // claim, query, cancellation and reuse paths remain host-nonblocking.
        params.fence_released_memory();
        let bytes = storage.slot_identity(0).unwrap().requested_backing_bytes();
        let baseline = storage.reset_occupied_high_water().unwrap();
        assert_eq!(baseline.requested_capacity_bytes(), bytes * 2);
        assert_eq!(baseline.reserved_bytes(), 0);
        assert_eq!(baseline.occupied_bytes(), 0);
        assert_eq!(baseline.occupied_high_water_bytes(), 0);
        let device = storage.device();
        gpu_default_mempool_reset_high_water(device).unwrap();
        let pool = gpu_default_mempool_usage(device).unwrap();

        let reservation = reserve(&storage, &[0, 1]).unwrap();
        let reserved = storage.occupancy().unwrap();
        assert_eq!(reserved.reserved_bytes(), bytes * 2);
        assert_eq!(reserved.active_reservations(), 1);
        assert_eq!(reserved.occupied_bytes(), 0);
        assert_eq!(reserved.occupied_high_water_bytes(), 0);
        assert!(storage.reset_occupied_high_water().is_err());
        let dispatch = reservation.enter(Vec::new()).unwrap();
        assert!(storage.reset_occupied_high_water().is_err());
        let output = GpuDCRTPolyMatrix::zero(&params, 3, 2);
        let claimed = storage.occupancy().unwrap();
        assert_eq!(claimed.reserved_bytes(), bytes);
        assert_eq!(claimed.occupied_bytes(), bytes);
        assert_eq!(claimed.occupied_high_water_bytes(), bytes);
        // Cancels only the unconsumed second slot. The returned output remains
        // occupied and cannot be reused merely because its dispatch ended.
        drop(dispatch);
        assert!(reserve(&storage, &[0]).is_err());
        drop(reserve(&storage, &[1]).unwrap());
        let retained = storage.reset_occupied_high_water().unwrap();
        assert_eq!(retained.reserved_bytes(), 0);
        assert_eq!(retained.active_reservations(), 0);
        assert_eq!(retained.occupied_high_water_bytes(), bytes);

        let dispatch = reserve(&storage, &[1]).unwrap().enter(Vec::new()).unwrap();
        let second = GpuDCRTPolyMatrix::zero(&params, 3, 2);
        drop(dispatch.finish().unwrap());
        let observation = storage.occupancy().unwrap();
        assert_eq!(observation.occupied_high_water_bytes() - retained.occupied_bytes(), bytes);
        assert_eq!(observation.occupied_bytes(), bytes * 2);
        let after = gpu_default_mempool_usage(device).unwrap();
        assert_eq!(after.used_current, pool.used_current);
        assert_eq!(after.used_high, pool.used_high);
        assert_eq!(after.reserved_current, pool.reserved_current);
        assert_eq!(output.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 3, 2));
        assert_eq!(second.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 3, 2));
        drop((output, second));
        params.fence_released_memory();
        let released = storage.occupancy().unwrap();
        assert_eq!(released.occupied_bytes(), 0);
        assert_eq!(released.occupied_high_water_bytes(), bytes * 2);
        assert_eq!(storage.reset_occupied_high_water().unwrap().occupied_high_water_bytes(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_occupancy_counts_reader_reuse_once_across_workers() {
        let (cpu, params) = parameters();
        let mut storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 5, 2), GpuDCRTPolyMatrix::zero(&params, 2, 5)],
            None,
        )
        .unwrap();
        let bytes = storage.slot_identity(0).unwrap().requested_backing_bytes();
        let capacity = storage.slot_identity(1).unwrap().requested_backing_bytes() + bytes;
        let reservation = reserve(&storage, &[0, 1]).unwrap();
        let producer_params = params.clone();
        let (source, reader) = std::thread::spawn(move || {
            let dispatch = reservation.enter(Vec::new()).unwrap();
            let source = GpuDCRTPolyMatrix::identity_columns(&producer_params, 5, 0, 2);
            let reader = source.transpose();
            drop(dispatch.finish().unwrap());
            (source, reader)
        })
        .join()
        .unwrap();
        assert_eq!(storage.occupancy().unwrap().occupied_bytes(), capacity);
        // Do not poll or wait between recycle and reuse: the pending reader
        // edges, not a host observation, must make overwriting the source safe.
        let original = source.raw;
        drop(source);
        let reservation = reserve(&storage, &[0]).unwrap();
        let before = storage.occupancy().unwrap();
        assert_eq!(before.reserved_bytes(), bytes);
        assert_eq!(before.occupied_bytes(), capacity);
        assert!(storage.reset_occupied_high_water().is_err());
        let producer_params = params.clone();
        let replacement = std::thread::spawn(move || {
            let dispatch = reservation.enter(Vec::new()).unwrap();
            let replacement = GpuDCRTPolyMatrix::zero(&producer_params, 5, 2);
            drop(dispatch.finish().unwrap());
            replacement
        })
        .join()
        .unwrap();
        assert_eq!(replacement.raw, original);
        let after = storage.occupancy().unwrap();
        assert_eq!(after.occupied_bytes(), capacity);
        assert_eq!(after.occupied_high_water_bytes(), capacity);
        assert_eq!(after.reserved_bytes(), 0);
        assert_eq!(after.active_reservations(), 0);
        assert!(reserve(&storage, &[0, 1]).is_err());
        let expected = DCRTPolyMatrix::identity(&cpu, 5, None).slice(0, 5, 0, 2);
        assert_eq!(reader.to_cpu_matrix(), expected.transpose());
        assert_eq!(replacement.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 5, 2));
        drop((reader, replacement));
        params.fence_released_memory();
        let empty = storage.reset_occupied_high_water().unwrap();
        assert_eq!(empty.occupied_bytes(), 0);
        assert_eq!(empty.occupied_high_water_bytes(), 0);
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_matrix_dispatch_retains_owner_through_native_teardown() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        struct Backing {
            matrices: Vec<GpuDCRTPolyMatrix>,
            released: Arc<AtomicBool>,
        }

        unsafe extern "C" fn release(owner: *mut c_void) {
            let Backing { matrices, released } = *unsafe { Box::from_raw(owner.cast::<Backing>()) };
            // Dropping these can destroy the last parameter contexts inside
            // dispatch_end, before its allocation activity guard is destroyed.
            drop(matrices);
            released.store(true, Ordering::Release);
        }

        let (_, params) = parameters();
        let matrix = GpuDCRTPolyMatrix::zero(&params, 2, 2);
        let pointer = matrix.raw;
        let released = Arc::new(AtomicBool::new(false));
        let owner = Box::into_raw(Box::new(Backing {
            matrices: vec![matrix],
            released: Arc::clone(&released),
        }));
        let mut storage = std::ptr::null_mut();
        let status = unsafe {
            gpu_prepared_storage_create(
                &pointer,
                1,
                std::ptr::null(),
                0,
                owner.cast(),
                release,
                &mut storage,
            )
        };
        if status != 0 {
            unsafe { release(owner.cast()) };
            panic!("native storage preparation failed: {}", last_error_string());
        }
        let mut reservation = std::ptr::null_mut();
        let status = unsafe { gpu_matrix_reserve(storage, std::ptr::null(), 0, &mut reservation) };
        if status != 0 {
            unsafe { gpu_prepared_storage_destroy(storage) };
            panic!("native dispatch preparation failed: {}", last_error_string());
        }
        // The unactivated reservation owns its backing even after every setup
        // handle and the original parameter reference have been released.
        drop(params);
        unsafe { gpu_prepared_storage_destroy(storage) };
        assert!(!released.load(Ordering::Acquire));
        let mut permit = std::ptr::null_mut();
        let status = unsafe { gpu_matrix_dispatch_enter(&reservation, 1, &mut permit) };
        if status != 0 {
            unsafe {
                gpu_matrix_reservation_destroy(reservation);
            }
            panic!("native dispatch activation failed: {}", last_error_string());
        }
        assert!(!released.load(Ordering::Acquire));
        assert_eq!(
            unsafe { gpu_matrix_dispatch_end(permit, 1, std::ptr::null_mut()) },
            0,
            "{}",
            last_error_string()
        );
        assert!(released.load(Ordering::Acquire));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_matrix_slots_reuse_after_readers_and_retain_backing() {
        for rows in [3, 5] {
            let (cpu, params) = parameters();
            let backing = vec![
                GpuDCRTPolyMatrix::zero(&params, rows, 2),
                GpuDCRTPolyMatrix::zero(&params, 2, rows),
            ];
            let original = backing[0].raw;
            let storage = GpuPreparedStorage::new(backing, None).unwrap();
            assert_eq!(storage.slot_count(), 2);
            let dispatch = reserve(&storage, &[0, 1]).unwrap().enter(Vec::new()).unwrap();
            let source = GpuDCRTPolyMatrix::identity_columns(&params, rows, 0, 2);
            assert_eq!(source.raw, original);
            let reader = source.transpose();
            drop(source);
            drop(dispatch.finish().unwrap());

            // No host completion wait separates the old reader from overwriting
            // its source. Slot reuse must establish the required device edges.
            let dispatch = reserve(&storage, &[0]).unwrap().enter(Vec::new()).unwrap();
            let replacement = GpuDCRTPolyMatrix::zero(&params, rows, 2);
            assert_eq!(replacement.raw, original);
            drop(dispatch.finish().unwrap());
            assert!(reserve(&storage, &[0]).is_err());
            drop(storage);

            let expected = DCRTPolyMatrix::identity(&cpu, rows, None).slice(0, rows, 0, 2);
            assert_eq!(reader.to_cpu_matrix(), expected.transpose());
            assert_eq!(replacement.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, rows, 2));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_matrix_dispatch_rejects_unplanned_allocations() {
        let (_, params) = parameters();
        let storage =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 2, 3)], None).unwrap();
        assert!(reserve(&storage, &[0, 0]).is_err());
        let dispatch = reserve(&storage, &[0]).unwrap().enter(Vec::new()).unwrap();
        let create = |rows, columns| {
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                gpu_matrix_create(
                    params.ctx_raw(),
                    1,
                    rows,
                    columns,
                    GPU_POLY_FORMAT_EVAL,
                    &mut output,
                    true,
                )
            };
            (status, output)
        };
        let (status, output) = create(3, 2);
        assert_ne!(status, 0);
        assert!(output.is_null());
        let matrix = GpuDCRTPolyMatrix::zero(&params, 2, 3);
        let (status, output) = create(2, 3);
        assert_ne!(status, 0);
        assert!(output.is_null());
        drop(dispatch.finish().unwrap());
        assert!(reserve(&storage, &[0]).is_err());
        drop(matrix);
        let dispatch = reserve(&storage, &[0]).unwrap().enter(Vec::new()).unwrap();
        let other_thread_params = params.clone();
        let (status, is_null) = std::thread::spawn(move || {
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                gpu_matrix_create(
                    other_thread_params.ctx_raw(),
                    1,
                    2,
                    3,
                    GPU_POLY_FORMAT_EVAL,
                    &mut output,
                    true,
                )
            };
            (status, output.is_null())
        })
        .join()
        .unwrap();
        assert_ne!(status, 0);
        assert!(is_null);
        drop(dispatch);
        let cancelled = reserve(&storage, &[0]).unwrap().enter(Vec::new()).unwrap();
        assert!(cancelled.finish().is_err());
        drop(reserve(&storage, &[0]).unwrap().enter(Vec::new()).unwrap());
        let empty_dispatch = reserve(&storage, &[]).unwrap().enter(Vec::new()).unwrap();
        let empty = GpuDCRTPolyMatrix::zero(&params, 2, 0);
        drop(empty_dispatch.finish().unwrap());
        assert_eq!(empty.size(), (2, 0));
        let (status, output) = create(2, 3);
        assert_ne!(status, 0);
        assert!(output.is_null());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_matrix_slots_cross_thread_drop_and_reuse() {
        let (cpu, params) = parameters();
        let storage = std::sync::Arc::new(
            GpuPreparedStorage::new(
                vec![
                    GpuDCRTPolyMatrix::zero(&params, 5, 2),
                    GpuDCRTPolyMatrix::zero(&params, 2, 5),
                ],
                None,
            )
            .unwrap(),
        );
        let reservation = reserve(&storage, &[0, 1]).unwrap();
        let identities = reservation.slot_identities().to_vec();
        let producer_params = params.clone();
        let (source, reader) = std::thread::spawn(move || {
            assert_eq!(reservation.slot_identities(), identities.as_slice());
            let dispatch = reservation.enter(Vec::new()).unwrap();
            let source = GpuDCRTPolyMatrix::identity_columns(&producer_params, 5, 0, 2);
            let reader = source.transpose();
            drop(dispatch.finish().unwrap());
            (source, reader)
        })
        .join()
        .unwrap();

        let recycler_storage = storage.clone();
        let recycler_params = params.clone();
        let replacement = std::thread::spawn(move || {
            assert!(reserve(&recycler_storage, &[0]).is_err());
            let original = source.raw as usize;
            // Only host submission completed on the producer thread. Dropping
            // and reusing this source must still preserve its pending reader.
            drop(source);
            let dispatch = reserve(&recycler_storage, &[0]).unwrap().enter(Vec::new()).unwrap();
            let replacement = GpuDCRTPolyMatrix::zero(&recycler_params, 5, 2);
            assert_eq!(replacement.raw as usize, original);
            drop(dispatch.finish().unwrap());
            replacement
        })
        .join()
        .unwrap();
        drop(storage);

        let expected = DCRTPolyMatrix::identity(&cpu, 5, None).slice(0, 5, 0, 2);
        assert_eq!(reader.to_cpu_matrix(), expected.transpose());
        assert_eq!(replacement.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 5, 2));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_reservations_rollback_without_activation() {
        let (_, params) = parameters();
        let first =
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 2, 3)], None).unwrap();
        let second = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 2, 3), GpuDCRTPolyMatrix::zero(&params, 2, 3)],
            None,
        )
        .unwrap();
        assert_ne!(first.identity(), second.identity());
        assert_eq!(Some(first.execution_owner_id()), params.execution_owner_id());
        assert_eq!(first.device(), params.device_ids()[0]);
        let identity = first.slot_identity(0).unwrap();
        assert_eq!(identity.storage_id(), first.identity());
        assert_eq!(identity.slot_index(), 0);
        assert_eq!((identity.rows(), identity.columns(), identity.level()), (2, 3, Some(1)));
        let allocation = params.matrix_allocation_bytes(1, 2, 3, true).unwrap();
        assert_eq!(identity.payload_bytes(), allocation.data_bytes);
        assert_eq!(identity.auxiliary_bytes(), allocation.aux_bytes);
        assert_eq!(
            identity.requested_backing_bytes(),
            allocation.data_bytes + allocation.aux_bytes
        );
        let other = second.slot_identity(0).unwrap();
        assert_ne!(identity.backing_id(), other.backing_id());
        assert_ne!(identity.slot_id(), other.slot_id());
        assert_ne!(other.backing_id(), second.slot_identity(1).unwrap().backing_id());

        let occupied = reserve(&second, &[1]).unwrap();
        // The first storage succeeds; the second reserves slot 0 then fails on
        // slot 1. Collect drops all prior tokens before any worker is published.
        let requests = [(&first, &[0][..]), (&second, &[0, 1][..])];
        let result = requests
            .iter()
            .map(|(storage, slots)| reserve(&storage, slots))
            .collect::<Result<Vec<_>, _>>();
        assert!(result.is_err());
        drop(reserve(&first, &[0]).unwrap());
        drop(reserve(&second, &[0]).unwrap());
        assert!(reserve(&second, &[1]).is_err(), "rollback must preserve another token's slot");

        // Reservation is not TLS activation, and cancellation never downgrades
        // checked allocation mode on this execution owner.
        let mut output = std::ptr::null_mut();
        let status = unsafe {
            gpu_matrix_create(params.ctx_raw(), 1, 2, 3, GPU_POLY_FORMAT_EVAL, &mut output, true)
        };
        assert_ne!(status, 0);
        assert!(output.is_null());
        std::thread::spawn(move || drop(occupied)).join().unwrap();
        let retry = reserve(&second, &[0, 1]).unwrap();
        assert_eq!(retry.slot_identities()[0], other);
        drop(retry);
        assert_eq!(first.slot_identity(0), Some(identity));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_nested_activation_rolls_back_only_incoming_reservation() {
        let (cpu, params) = parameters();
        let storage = GpuPreparedStorage::new(
            vec![GpuDCRTPolyMatrix::zero(&params, 2, 3), GpuDCRTPolyMatrix::zero(&params, 2, 3)],
            None,
        )
        .unwrap();
        let first = reserve(&storage, &[0]).unwrap();
        let second = reserve(&storage, &[1]).unwrap();
        let active = first.enter(Vec::new()).unwrap();
        assert!(second.enter(Vec::new()).is_err());
        assert!(reserve(&storage, &[0]).is_err());
        // Cancellation is safe even while another dispatch is active here.
        drop(reserve(&storage, &[1]).unwrap());
        let result = GpuDCRTPolyMatrix::zero(&params, 2, 3);
        drop(active.finish().unwrap());
        drop(storage);
        assert_eq!(result.to_cpu_matrix(), DCRTPolyMatrix::zero(&cpu, 2, 3));
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_matrix_failed_mixed_context_setup_preserves_contexts() {
        let (first_cpu, first_params) = parameters();
        let (second_cpu, second_params) = parameters();
        assert_ne!(first_params.ctx_raw(), second_params.ctx_raw());
        let result = GpuPreparedStorage::new(
            vec![
                GpuDCRTPolyMatrix::zero(&first_params, 3, 2),
                GpuDCRTPolyMatrix::zero(&second_params, 3, 2),
            ],
            None,
        );
        assert!(result.is_err());

        // Failed preparation consumed and cleaned up its backing wrappers, but
        // neither original context may be quarantined or require a permit.
        let first = GpuDCRTPolyMatrix::identity_columns(&first_params, 3, 0, 2);
        let second = GpuDCRTPolyMatrix::identity_columns(&second_params, 3, 1, 2);
        assert_eq!(
            first.to_cpu_matrix(),
            DCRTPolyMatrix::identity(&first_cpu, 3, None).slice(0, 3, 0, 2)
        );
        assert_eq!(
            second.to_cpu_matrix(),
            DCRTPolyMatrix::identity(&second_cpu, 3, None).slice(0, 3, 1, 3)
        );
    }
}
