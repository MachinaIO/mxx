//! Admission accounting for an asynchronous GPU fleet.
//!
//! The dispatcher owns this ledger exclusively. CUDA counters establish a baseline
//! only at a quiescent setup boundary; rolling counter samples never retire charges.
//! Native allocation/release owners report lifecycle transitions, including the
//! completion of queued frees. Submission by itself does not return capacity.
//! Accepted setup residency and admitted managed allocations determine planning
//! capacity. Later physical/allocator observations are diagnostics: exceeding the
//! configured budget during production does not stop an otherwise valid plan.

use crate::gpu_calibration::{GpuCalibrationError, GpuDeviceMemory};
use mxx_primitives::{
    matrix::{
        PolyMatrix, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixReleaseObserver, GpuMatrixReservation, GpuPreparedRequest,
            GpuPreparedReservationError, GpuPreparedSlotKind, GpuPreparedStorage,
            GpuPreparedWorkspaceRequest, GpuPreparedWorkspaceReservation, GpuSmallMatrix,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{
            GpuAllocationEpoch, GpuAllocationEpochBoundary, GpuDCRTPolyParams, GpuReleaseCompletion,
        },
    },
};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

static NEXT_LEDGER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct GpuAllocationId {
    ledger: u64,
    allocation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct GpuChargeId {
    ledger: u64,
    charge: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuAllocationRequirement {
    pub device: usize,
    pub bytes: u64,
    pub pinned_bytes: u64,
}

pub struct GpuPreparedAllocationRequirement<'a> {
    pub device: usize,
    pub storage: &'a GpuPreparedStorage,
    pub requests: &'a [GpuPreparedRequest],
    pub workspace_requests: &'a [GpuPreparedWorkspaceRequest],
}

pub struct GpuMemoryReservation {
    pub allocations: Vec<GpuAllocationId>,
    pub prepared: Vec<(usize, GpuMatrixReservation)>,
}

/// A rollback marker for a speculative prepared-program generation.  The
/// marker is deliberately value-only so the fleet can keep the ledger usable
/// while native graph construction runs.
#[derive(Clone, Debug)]
pub(crate) struct GpuWarmupCheckpoint {
    storage_ids: BTreeSet<u64>,
    charge_ids: BTreeSet<GpuChargeId>,
}

/// Capacity reserved by one warmup transaction before a native owner exists.
/// A reservation is consumed exactly once by either a known storage owner or
/// an [`UnknownCharge`].  It is the only part of a provisioning charge that a
/// failed transaction may refund directly.
#[derive(Clone)]
struct ConstructionReservation {
    device: Vec<u64>,
    pinned: Vec<u64>,
}

impl ConstructionReservation {
    fn is_empty(&self) -> bool {
        self.device.iter().all(|bytes| *bytes == 0) && self.pinned.iter().all(|bytes| *bytes == 0)
    }

    fn consume(
        &mut self,
        device: usize,
        device_bytes: u64,
        pinned_bytes: u64,
    ) -> Result<(), GpuAdmissionError> {
        let Some(available_device) = self.device.get_mut(device) else {
            return Err(GpuAdmissionError::InvalidDevice(device));
        };
        let Some(available_pinned) = self.pinned.get_mut(device) else {
            return Err(GpuAdmissionError::InvalidDevice(device));
        };
        if device_bytes > *available_device || pinned_bytes > *available_pinned {
            return Err(GpuAdmissionError::AllocationExceedsReservation {
                allocated_bytes: device_bytes.saturating_add(pinned_bytes),
                reserved_bytes: available_device.saturating_add(*available_pinned),
            });
        }
        *available_device -= device_bytes;
        *available_pinned -= pinned_bytes;
        Ok(())
    }

    fn take_unknown(&mut self, reason: &str) -> Vec<UnknownCharge> {
        let mut unknown = Vec::new();
        for (device, (&device_bytes, &pinned_bytes)) in
            self.device.iter().zip(&self.pinned).enumerate()
        {
            if device_bytes == 0 && pinned_bytes == 0 {
                continue;
            }
            unknown.push(UnknownCharge {
                device,
                device_bytes,
                pinned_bytes,
                reason: reason.to_owned(),
                owner: None,
            });
        }
        self.device.fill(0);
        self.pinned.fill(0);
        unknown
    }
}

struct StorageCharge {
    identity: u64,
    storage: Arc<GpuPreparedStorage>,
    device: usize,
    device_bytes: u64,
    pinned_bytes: u64,
    state: StorageChargeState,
}

enum StorageChargeState {
    Owned,
    ReleasePending(GpuReleaseCompletion),
}

/// Capacity whose native owner or exact release proof is no longer known.
/// Unknown charges are deliberately sticky: polling never refunds them.
struct UnknownCharge {
    device: usize,
    device_bytes: u64,
    pinned_bytes: u64,
    reason: String,
    // Keep an owner alive if native teardown itself failed.  The bytes remain
    // sticky even if this owner later drops; no unproven release can refund it.
    owner: Option<Arc<GpuPreparedStorage>>,
}

impl UnknownCharge {
    fn is_well_formed(&self, device_count: usize) -> bool {
        if self.device >= device_count ||
            self.reason.is_empty() ||
            self.device_bytes.checked_add(self.pinned_bytes).is_none()
        {
            return false;
        }
        self.owner.as_ref().map_or(true, |owner| {
            owner.is_workspace_only() ||
                (owner.device() >= 0 && owner.device() as usize == self.device)
        })
    }
}

struct ProvisioningCharge {
    reservation: ConstructionReservation,
    storages: Vec<StorageCharge>,
    unknown: Vec<UnknownCharge>,
}

/// RAII transaction for warmup backing.  Charges are installed before native
/// construction and are rolled back automatically unless the resulting
/// storages are appended to the accepted inventory.  This keeps each device's
/// exact charge live across every fallible construction step, including panic
/// unwinding and asynchronous native destruction.
pub(crate) struct PendingProvisioning<'a> {
    ledger: &'a mut GpuMemoryLedger,
    charge_id: GpuChargeId,
    committed: bool,
}

impl PendingProvisioning<'_> {
    /// Keep this charge permanently accounted when native construction failed
    /// before it could return an owner/probe.  A conservative uncertain charge
    /// is preferable to refunding bytes whose asynchronous native lifetime is
    /// no longer observable by the transaction.
    pub(crate) fn mark_uncertain(&mut self) {
        if let Some(charge) = self.ledger.provisioning_charges.get_mut(&self.charge_id) {
            charge.unknown.extend(
                charge
                    .reservation
                    .take_unknown("native construction failed before an owner was returned"),
            );
        }
    }

    pub(crate) fn append(
        &mut self,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<(), GpuAdmissionError> {
        self.ledger.append_prepared_storages(self.charge_id, prepared)
    }

    pub(crate) fn commit(mut self) {
        // A complete production transaction consumes its reservation through
        // `append`.  Keep any unconsumed reservation explicit until the
        // generation checkpoint rolls it back; it is still proven unallocated
        // capacity, not an unknown native owner.
        self.committed = true;
    }
}

impl Drop for PendingProvisioning<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.ledger.rollback_prepared_provisioning(self.charge_id);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AllocationState {
    Reserved,
    Leased,
    Submitted,
    Resident,
    ReleasePending,
    Abandoned,
}

enum AllocationNotice {
    Cancelled,
    /// A committed prepared generation whose native publication failed.
    /// Unlike quarantine, this is recoverable because no native owner was
    /// published and the reservation can be removed immediately.
    Submitted,
    Resident,
    Released(Result<GpuReleaseCompletion, String>),
    Abandoned,
}

fn validate_prepared_inventory(
    device: usize,
    storage: &GpuPreparedStorage,
    inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
) -> Result<(), GpuAdmissionError> {
    let Some((owner, accepted)) = inventory.get(&storage.identity()) else {
        return Err(GpuAdmissionError::UnknownPreparedStorage);
    };
    let same_execution = if storage.is_workspace_only() {
        accepted.execution_owner_id() == storage.execution_owner_id()
    } else {
        accepted.device() == storage.device() &&
            accepted.execution_owner_id() == storage.execution_owner_id()
    };
    if *owner != device || !same_execution {
        return Err(GpuAdmissionError::ExecutionMismatch);
    }
    Ok(())
}

pub struct GpuAllocationLease {
    id: GpuAllocationId,
    requirement: GpuAllocationRequirement,
    execution_identity: Option<(i32, u64)>,
    sender: mpsc::Sender<(GpuAllocationId, AllocationNotice)>,
    submitted: bool,
    bound: bool,
}

impl GpuAllocationLease {
    pub fn requirement(&self) -> GpuAllocationRequirement {
        self.requirement
    }

    fn begin(&mut self) -> Result<(), GpuAdmissionError> {
        self.sender
            .send((self.id, AllocationNotice::Submitted))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.submitted = true;
        Ok(())
    }

    fn validate_owner(
        &self,
        parameters: &GpuDCRTPolyParams,
        minimum_bytes: usize,
    ) -> Result<(), GpuAdmissionError> {
        let allocated_bytes =
            u64::try_from(minimum_bytes).map_err(|_| GpuAdmissionError::Overflow)?;
        if allocated_bytes > self.requirement.bytes {
            return Err(GpuAdmissionError::AllocationExceedsReservation {
                allocated_bytes,
                reserved_bytes: self.requirement.bytes,
            });
        }
        let (device, owner) =
            self.execution_identity.ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if parameters.device_ids() != [device] || parameters.execution_owner_id() != Some(owner) {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        Ok(())
    }

    fn release_observer(
        &self,
        parameters: &GpuDCRTPolyParams,
        minimum_bytes: usize,
    ) -> Result<GpuMatrixReleaseObserver, GpuAdmissionError> {
        self.validate_owner(parameters, minimum_bytes)?;
        let sender = self.sender.clone();
        let id = self.id;
        Ok(Box::new(move |completion| {
            // The dispatcher may already have been dropped. Normal native
            // destruction has run; this notification owns no allocation.
            drop(sender.send((id, AllocationNotice::Released(completion))));
        }))
    }

    /// The closure may enqueue construction and initialization. Its complete
    /// additional live set must be covered by this owner and separately reserved
    /// temporary leases. On failure its reservation is never cancelled as though
    /// the closure had not run. No GPU completion wait is introduced here.
    pub fn allocate_matrix<E: From<GpuAdmissionError>>(
        mut self,
        parameters: &GpuDCRTPolyParams,
        create: impl FnOnce(&GpuDCRTPolyParams) -> Result<GpuDCRTPolyMatrix, E>,
    ) -> Result<GpuDCRTPolyMatrix, E> {
        self.validate_owner(parameters, 0)?;
        self.begin()?;
        let mut matrix = create(parameters)?;
        let bytes =
            matrix.resident_allocation_bytes().map_err(GpuAdmissionError::ReleaseFailure)?;
        let observer = self.release_observer(matrix.params(), bytes)?;
        matrix.observe_release(observer).map_err(GpuAdmissionError::ReleaseFailure)?;
        self.sender
            .send((self.id, AllocationNotice::Resident))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(matrix)
    }

    /// Compact owners use the same protocol without expanding their payload.
    pub fn allocate_small_matrix<E: From<GpuAdmissionError>>(
        mut self,
        parameters: &GpuDCRTPolyParams,
        create: impl FnOnce(&GpuDCRTPolyParams) -> Result<GpuSmallMatrix, E>,
    ) -> Result<GpuSmallMatrix, E> {
        self.validate_owner(parameters, 0)?;
        self.begin()?;
        let mut matrix = create(parameters)?;
        let observer = self.release_observer(matrix.params(), matrix.resident_payload_bytes())?;
        matrix.observe_release(observer).map_err(GpuAdmissionError::ReleaseFailure)?;
        self.sender
            .send((self.id, AllocationNotice::Resident))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(matrix)
    }
}

impl Drop for GpuAllocationLease {
    fn drop(&mut self) {
        if !self.bound {
            let notice = if self.submitted {
                AllocationNotice::Abandoned
            } else {
                AllocationNotice::Cancelled
            };
            drop(self.sender.send((self.id, notice)));
        }
    }
}

struct Allocation {
    requirement: GpuAllocationRequirement,
    state: AllocationState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuDeviceAdmission {
    pub budget_bytes: u64,
    /// Accepted initial adjusted residency; retained for this inventory's life.
    pub baseline_bytes: u64,
    /// Managed reservations, including submitted allocations pending release.
    pub allocation_bytes: u64,
    /// Managed pinned-host reservations, tracked independently from VRAM.
    pub pinned_allocation_bytes: u64,
    /// Independent pinned-host budget. This is intentionally not folded into
    /// `budget_bytes`, which is the device-byte budget.
    pub pinned_budget_bytes: u64,
    /// Physical residency accepted at setup, including idle pool pages.
    pub physical_baseline_bytes: u64,
    /// Latest accepted observations, never a prospective bound or stop signal.
    pub observed_resident_bytes: u64,
    pub observed_physical_bytes: u64,
}

impl GpuDeviceAdmission {
    pub fn charged_bytes(self) -> u64 {
        // Admission checks this sum before publishing any reservation.
        self.baseline_bytes + self.allocation_bytes
    }

    pub fn pinned_charged_bytes(self) -> u64 {
        self.pinned_allocation_bytes
    }

    pub fn observed_budget_excess_bytes(self) -> u64 {
        self.observed_physical_bytes.saturating_sub(self.budget_bytes)
    }

    fn available_bytes(self) -> u64 {
        self.budget_bytes - self.charged_bytes()
    }

    fn pinned_available_bytes(self) -> u64 {
        self.pinned_budget_bytes.saturating_sub(self.pinned_charged_bytes())
    }
}

/// Transient containing the exact prepared storage selected for one wave.
/// Dropping this owner restores the enclosing wave's inventory.
pub struct GpuMemoryRegion {
    _inventory: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    workspace_requests: std::sync::Mutex<BTreeMap<u64, Vec<GpuPreparedWorkspaceRequest>>>,
}

impl GpuMemoryRegion {
    pub(crate) fn empty() -> Self {
        Self {
            _inventory: BTreeMap::new(),
            workspace_requests: std::sync::Mutex::new(BTreeMap::new()),
        }
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test() -> Self {
        Self::empty()
    }

    pub(crate) fn take_workspace_reservation_matching(
        &self,
        storage: u64,
        request: GpuPreparedWorkspaceRequest,
    ) -> Option<GpuPreparedWorkspaceReservation> {
        // Workspace claims are immutable codec descriptors, not one-shot
        // reservations.  Re-arm the exact slot from its owning storage for
        // every serialization so retaining an output permits repeated reads.
        let expected = {
            let requests = self.workspace_requests.lock().ok()?;
            requests.get(&storage)?.iter().find(|expected| **expected == request).copied()?
        };
        let (_, workspace) = self._inventory.get(&storage)?;
        workspace.reserve_workspace(std::slice::from_ref(&expected)).ok()
    }

    pub(crate) fn reserve_matrix(
        &self,
        storage: u64,
        requests: &[GpuPreparedRequest],
    ) -> Result<GpuMatrixReservation, String> {
        let (_, storage) = self
            ._inventory
            .get(&storage)
            .ok_or_else(|| "prepared matrix storage is missing from the region".to_owned())?;
        storage.reserve(requests)
    }
}

pub struct GpuMemoryLedger {
    identity: u64,
    observation_epochs: Vec<GpuAllocationEpoch>,
    // Retain the actual backing accepted at setup. A later allocation on the
    // same execution owner is not implicitly part of the charged inventory.
    prepared_storages: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    device_totals: Vec<u64>,
    next_allocation: u64,
    devices: Vec<GpuDeviceAdmission>,
    allocations: BTreeMap<GpuAllocationId, Allocation>,
    release_completions: BTreeMap<GpuAllocationId, GpuReleaseCompletion>,
    execution_identities: Option<Vec<(i32, u64)>>,
    releases: mpsc::Receiver<(GpuAllocationId, AllocationNotice)>,
    release_sender: mpsc::Sender<(GpuAllocationId, AllocationNotice)>,
    release_error: Option<String>,
    abandoned_submission: bool,
    provisioning_charges: BTreeMap<GpuChargeId, ProvisioningCharge>,
    next_charge: u64,
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum GpuAdmissionError {
    #[error(transparent)]
    Calibration(#[from] GpuCalibrationError),
    #[error("GPU admission arithmetic overflow")]
    Overflow,
    #[error("GPU placement {0} is outside the configured fleet")]
    InvalidDevice(usize),
    #[error(
        "GPU {device} cannot reserve {requested_bytes} bytes: charged={charged_bytes}, budget={budget_bytes}"
    )]
    Capacity { device: usize, requested_bytes: u64, charged_bytes: u64, budget_bytes: u64 },
    #[error(
        "GPU {device} cannot reserve physical growth {requested_bytes}: physical charge={charged_bytes}, budget={budget_bytes}"
    )]
    PhysicalCapacity { device: usize, requested_bytes: u64, charged_bytes: u64, budget_bytes: u64 },
    #[error(
        "GPU {device} cannot reserve pinned host memory {requested_bytes}: charged={charged_bytes}, budget={budget_bytes}"
    )]
    PinnedCapacity { device: usize, requested_bytes: u64, charged_bytes: u64, budget_bytes: u64 },
    #[error("GPU physical and allocator-adjusted residency counters are inconsistent")]
    InvalidPhysicalResidency,
    #[error("unknown or foreign GPU allocation")]
    UnknownAllocation,
    #[error("invalid GPU allocation lifecycle transition")]
    InvalidTransition,
    #[error("a GPU allocation appears more than once in a transaction")]
    DuplicateAllocation,
    #[error("GPU observation refresh requires the same fleet with no pending allocations or frees")]
    InvalidRefresh,
    #[error("GPU allocation observation evidence is invalid or no longer current: {0}")]
    InvalidEpoch(String),
    #[error("GPU release completion could not be established: {0}")]
    ReleaseFailure(String),
    #[error(
        "a submitted GPU allocation lost its native release owner; its capacity remains charged"
    )]
    AbandonedSubmission,
    #[error("the GPU allocation dispatcher has closed")]
    Closed,
    #[error("GPU allocation does not belong to the configured execution owner")]
    ExecutionMismatch,
    #[error("GPU prepared backing is absent from the accepted setup inventory")]
    UnknownPreparedStorage,
    #[error("GPU prepared acquisition conflict: {0}")]
    AcquisitionConflict(String),
    #[error("GPU prepared backing appears more than once in the setup inventory")]
    DuplicatePreparedStorage,
    #[error("native GPU reservation failed: {0}")]
    NativeReservation(String),
    #[error(
        "GPU allocation exceeds its reservation: allocated={allocated_bytes}, reserved={reserved_bytes}"
    )]
    AllocationExceedsReservation { allocated_bytes: u64, reserved_bytes: u64 },
    #[error("GPU {device} prepared region is stale: native availability changed before commit")]
    StaleRegion { device: usize },
}

impl GpuMemoryLedger {
    /// Charge the complete backing demand before any native prepared storage
    /// is constructed.  The charge is part of the provisioning transaction;
    /// callers must either append the resulting stores or roll it back.
    pub(crate) fn begin_prepared_provisioning(
        &mut self,
        additional_device: &[u64],
        additional_pinned: &[u64],
    ) -> Result<PendingProvisioning<'_>, GpuAdmissionError> {
        if additional_device.len() != self.devices.len() ||
            additional_pinned.len() != self.devices.len()
        {
            return Err(GpuAdmissionError::InvalidDevice(additional_device.len()));
        }
        self.poll_releases()?;
        for (index, ((device, requested_bytes), pinned)) in
            self.devices.iter().zip(additional_device).zip(additional_pinned).enumerate()
        {
            if *requested_bytes > device.available_bytes() {
                return Err(GpuAdmissionError::Capacity {
                    device: index,
                    requested_bytes: *requested_bytes,
                    charged_bytes: device.charged_bytes(),
                    budget_bytes: device.budget_bytes,
                });
            }
            if *pinned > device.pinned_available_bytes() {
                return Err(GpuAdmissionError::PinnedCapacity {
                    device: index,
                    requested_bytes: *pinned,
                    charged_bytes: device.pinned_charged_bytes(),
                    budget_bytes: device.pinned_budget_bytes,
                });
            }
        }
        // Validate every counter update before mutating the ledger.  The
        // returned guard must own either the complete per-device charge or
        // none of it, including the overflow path.
        for ((device, requested_bytes), pinned) in
            self.devices.iter().zip(additional_device).zip(additional_pinned)
        {
            device
                .allocation_bytes
                .checked_add(*requested_bytes)
                .ok_or(GpuAdmissionError::Overflow)?;
            device
                .pinned_allocation_bytes
                .checked_add(*pinned)
                .ok_or(GpuAdmissionError::Overflow)?;
        }
        let charge = self.next_charge;
        let next_charge = charge.checked_add(1).ok_or(GpuAdmissionError::Overflow)?;
        crate::backend::poly_gpu::record_provisioning_begin();
        for ((device, requested_bytes), pinned) in
            self.devices.iter_mut().zip(additional_device).zip(additional_pinned)
        {
            device.allocation_bytes += *requested_bytes;
            device.pinned_allocation_bytes += *pinned;
        }
        let charge_id = GpuChargeId { ledger: self.identity, charge };
        self.provisioning_charges.insert(
            charge_id,
            ProvisioningCharge {
                reservation: ConstructionReservation {
                    device: additional_device.to_vec(),
                    pinned: additional_pinned.to_vec(),
                },
                storages: Vec::new(),
                unknown: Vec::new(),
            },
        );
        self.next_charge = next_charge;
        Ok(PendingProvisioning { ledger: self, charge_id, committed: false })
    }

    /// Undo a not-yet-published provisioning charge.  This is intentionally
    /// only used by the owning warmup transaction after native construction
    /// or detached-region acquisition fails.
    fn rollback_prepared_provisioning(&mut self, charge_id: GpuChargeId) {
        let Some(mut charge) = self.provisioning_charges.remove(&charge_id) else {
            self.release_error = Some("prepared charge was already removed".into());
            return;
        };
        for storage in &charge.storages {
            self.prepared_storages.remove(&storage.identity);
        }
        if !charge.reservation.is_empty() {
            let reservation = charge.reservation.clone();
            match self.refund_construction_reservation(&reservation) {
                Ok(()) => {
                    charge.reservation.device.fill(0);
                    charge.reservation.pinned.fill(0);
                }
                Err(error) => {
                    self.release_error = Some(error.to_string());
                }
            }
        }

        let storages = std::mem::take(&mut charge.storages);
        for mut storage in storages {
            match storage.state {
                StorageChargeState::Owned => match storage.storage.start_release() {
                    Ok(probe) => {
                        storage.state = StorageChargeState::ReleasePending(probe.into_completion());
                        charge.storages.push(storage);
                    }
                    Err(error) => {
                        self.release_error = Some(error.clone());
                        charge.unknown.push(UnknownCharge {
                            device: storage.device,
                            device_bytes: storage.device_bytes,
                            pinned_bytes: storage.pinned_bytes,
                            reason: error,
                            owner: Some(storage.storage),
                        });
                    }
                },
                StorageChargeState::ReleasePending(probe) => {
                    storage.state = StorageChargeState::ReleasePending(probe);
                    charge.storages.push(storage);
                }
            }
        }
        if !charge.storages.is_empty() ||
            !charge.unknown.is_empty() ||
            !charge.reservation.is_empty()
        {
            self.provisioning_charges.insert(charge_id, charge);
        }
    }

    fn refund_construction_reservation(
        &mut self,
        reservation: &ConstructionReservation,
    ) -> Result<(), GpuAdmissionError> {
        if reservation.device.len() != self.devices.len() ||
            reservation.pinned.len() != self.devices.len()
        {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        if self.devices.iter().zip(&reservation.device).zip(&reservation.pinned).any(
            |((device, requested_bytes), pinned)| {
                *requested_bytes > device.allocation_bytes ||
                    *pinned > device.pinned_allocation_bytes
            },
        ) {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        for ((device, requested_bytes), pinned) in
            self.devices.iter_mut().zip(&reservation.device).zip(&reservation.pinned)
        {
            device.allocation_bytes -= *requested_bytes;
            device.pinned_allocation_bytes -= *pinned;
        }
        Ok(())
    }

    fn snapshot_storage_bytes(
        storage: &GpuPreparedStorage,
    ) -> Result<(u64, u64), GpuAdmissionError> {
        if storage.is_workspace_only() {
            let accounting =
                storage.workspace_accounting().map_err(GpuAdmissionError::ReleaseFailure)?;
            return Ok((
                u64::try_from(accounting.device_bytes).map_err(|_| GpuAdmissionError::Overflow)?,
                u64::try_from(accounting.pinned_bytes).map_err(|_| GpuAdmissionError::Overflow)?,
            ));
        }
        let mut device_bytes = 0u64;
        let mut pinned_bytes = 0u64;
        for slot in storage.snapshot().map_err(GpuAdmissionError::ReleaseFailure)? {
            let bytes = u64::try_from(slot.identity().requested_backing_bytes())
                .map_err(|_| GpuAdmissionError::Overflow)?;
            match slot.identity().kind() {
                GpuPreparedSlotKind::PinnedHost => {
                    pinned_bytes =
                        pinned_bytes.checked_add(bytes).ok_or(GpuAdmissionError::Overflow)?;
                }
                GpuPreparedSlotKind::CompletionEvent | GpuPreparedSlotKind::SubmissionStream => {}
                _ => {
                    device_bytes =
                        device_bytes.checked_add(bytes).ok_or(GpuAdmissionError::Overflow)?;
                }
            }
        }
        Ok((device_bytes, pinned_bytes))
    }

    /// Consume the remaining reservation conservatively after a fallible
    /// append.  Every fresh owner is retained by an unknown record, while any
    /// reservation that cannot be attributed to an exact owner remains a
    /// per-device sticky unknown charge.
    fn quarantine_append_failure(
        &mut self,
        charge_id: GpuChargeId,
        prepared: &[(usize, Arc<GpuPreparedStorage>)],
        reason: &str,
    ) {
        let Some(mut charge) = self.provisioning_charges.remove(&charge_id) else {
            self.release_error = Some("prepared charge was already removed".into());
            return;
        };
        let mut seen = BTreeSet::new();
        for (device, storage) in prepared {
            if *device >= self.devices.len() ||
                self.prepared_storages.contains_key(&storage.identity()) ||
                !seen.insert(storage.identity())
            {
                continue;
            }
            match Self::snapshot_storage_bytes(storage) {
                Ok((device_bytes, pinned_bytes))
                    if charge.reservation.consume(*device, device_bytes, pinned_bytes).is_ok() =>
                {
                    charge.unknown.push(UnknownCharge {
                        device: *device,
                        device_bytes,
                        pinned_bytes,
                        reason: reason.to_owned(),
                        owner: Some(Arc::clone(storage)),
                    });
                }
                _ => {
                    // The exact owner demand could not be observed.  Keep the
                    // owner alive and account the reservation remainder below.
                    charge.unknown.push(UnknownCharge {
                        device: *device,
                        device_bytes: 0,
                        pinned_bytes: 0,
                        reason: reason.to_owned(),
                        owner: Some(Arc::clone(storage)),
                    });
                }
            }
        }
        charge.unknown.extend(charge.reservation.take_unknown(reason));
        self.provisioning_charges.insert(charge_id, charge);
    }

    /// Append newly provisioned prepared storage to this execution owner.
    /// Each owner consumes its exact reservation record; no aggregate charge
    /// or fresh-charge polling exception is needed.

    fn append_prepared_storages(
        &mut self,
        charge_id: GpuChargeId,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<(), GpuAdmissionError> {
        if prepared.is_empty() {
            return Ok(());
        }
        if let Err(error) = self.poll_releases() {
            self.quarantine_append_failure(
                charge_id,
                &prepared,
                "release polling failed while appending prepared storage",
            );
            return Err(error);
        }
        let mut identities = BTreeSet::new();
        for (device, storage) in &prepared {
            let Some(expected) =
                self.execution_identities.as_ref().and_then(|ids| ids.get(*device))
            else {
                self.quarantine_append_failure(charge_id, &prepared, "invalid prepared device");
                return Err(GpuAdmissionError::InvalidDevice(*device));
            };
            let execution_matches = if storage.is_workspace_only() {
                storage.execution_owner_id() == expected.1 &&
                    storage.workspace_claims().is_some_and(|claims| {
                        claims.iter().all(|claim| claim.device == expected.0)
                    })
            } else {
                *expected == (storage.device(), storage.execution_owner_id())
            };
            if !execution_matches ||
                !identities.insert(storage.identity()) ||
                self.prepared_storages.contains_key(&storage.identity())
            {
                self.quarantine_append_failure(
                    charge_id,
                    &prepared,
                    "prepared storage identity was duplicated or mismatched",
                );
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
        }
        let mut storage_charges = Vec::with_capacity(prepared.len());
        for (device, storage) in &prepared {
            let (device_bytes, pinned_bytes) = match Self::snapshot_storage_bytes(storage) {
                Ok(bytes) => bytes,
                Err(error) => {
                    self.quarantine_append_failure(
                        charge_id,
                        &prepared,
                        "prepared storage snapshot failed during append",
                    );
                    return Err(error);
                }
            };
            storage_charges.push(StorageCharge {
                identity: storage.identity(),
                storage: Arc::clone(storage),
                device: *device,
                device_bytes,
                pinned_bytes,
                state: StorageChargeState::Owned,
            });
        }
        let reservation_error = {
            let charge = self
                .provisioning_charges
                .get_mut(&charge_id)
                .ok_or(GpuAdmissionError::UnknownAllocation)?;
            let mut remaining_reservation = charge.reservation.clone();
            let result = storage_charges.iter().try_for_each(|storage| {
                remaining_reservation.consume(
                    storage.device,
                    storage.device_bytes,
                    storage.pinned_bytes,
                )
            });
            if result.is_ok() {
                charge.reservation = remaining_reservation;
            }
            result
        };
        if let Err(error) = reservation_error {
            self.quarantine_append_failure(
                charge_id,
                &prepared,
                "prepared storage demand exceeded its construction reservation",
            );
            return Err(error);
        }
        let charge = self
            .provisioning_charges
            .get_mut(&charge_id)
            .ok_or(GpuAdmissionError::UnknownAllocation)?;
        charge.storages.extend(storage_charges);
        for (device, storage) in prepared {
            crate::backend::poly_gpu::record_provisioning_permit();
            self.prepared_storages.insert(storage.identity(), (device, storage));
            crate::backend::poly_gpu::record_provisioning_append();
        }
        Ok(())
    }

    pub(crate) fn warmup_checkpoint(&self) -> GpuWarmupCheckpoint {
        GpuWarmupCheckpoint {
            storage_ids: self.prepared_storages.keys().copied().collect(),
            charge_ids: self.provisioning_charges.keys().copied().collect(),
        }
    }

    /// Remove only backing introduced by a failed speculative generation.
    /// Existing setup/program owners are left untouched; native storage owners
    /// are dropped after all failed reservations have unwound.
    pub(crate) fn rollback_warmup(&mut self, checkpoint: GpuWarmupCheckpoint) {
        let candidate_charges = self
            .provisioning_charges
            .keys()
            .copied()
            .filter(|id| !checkpoint.charge_ids.contains(id))
            .collect::<Vec<_>>();
        for charge_id in candidate_charges {
            if let Some(charge) = self.provisioning_charges.get(&charge_id) {
                for storage in &charge.storages {
                    self.prepared_storages.remove(&storage.identity);
                }
            }
            self.rollback_prepared_provisioning(charge_id);
        }
        self.prepared_storages.retain(|identity, _| checkpoint.storage_ids.contains(identity));
    }

    /// Roll back every speculative charge when the first generation never had
    /// an accepted ledger checkpoint.  This must run before dropping the
    /// ledger, so native owners remain reachable by their release probes.
    pub(crate) fn rollback_all_warmup(&mut self) {
        self.rollback_warmup(GpuWarmupCheckpoint {
            storage_ids: BTreeSet::new(),
            charge_ids: BTreeSet::new(),
        });
    }

    /// The complete accepted base-owner inventory. This is the only inventory
    /// suitable for resolving a replacement generation; regions are never
    /// consulted as implicit parents.
    pub(crate) fn accepted_prepared_inventory(
        &self,
    ) -> impl Iterator<Item = (usize, Arc<GpuPreparedStorage>)> {
        self.prepared_storages.values().cloned().collect::<Vec<_>>().into_iter()
    }

    /// Commit complete native region capacity, without taking operation leases.
    /// Matching/liveness chooses these exact slots first; this transaction only
    /// publishes an inventory after every storage owner has been acquired.
    /// Dropping the returned owner releases only this independent native region.
    pub fn reserve_region(
        &mut self,
        requirements: &[GpuPreparedAllocationRequirement<'_>],
    ) -> Result<Arc<GpuMemoryRegion>, GpuAdmissionError> {
        crate::backend::poly_gpu::record_prepared_forbidden(5);
        let inventory = self
            .accepted_prepared_inventory()
            .map(|(device, storage)| (storage.identity(), (device, storage)))
            .collect::<BTreeMap<_, _>>();
        let mut selected = BTreeMap::<u64, (usize, Vec<usize>)>::new();
        let mut selected_workspace = BTreeMap::<
            u64,
            (usize, Arc<GpuPreparedStorage>, Vec<GpuPreparedWorkspaceRequest>),
        >::new();
        for requirement in requirements {
            self.validate_prepared_storage(requirement.device, requirement.storage, &inventory)?;
            if !requirement.requests.is_empty() {
                if requirement.storage.is_workspace_only() {
                    return Err(GpuAdmissionError::NativeReservation(
                        "workspace-only storage received a matrix reservation".into(),
                    ));
                }
                let slots = &mut selected
                    .entry(requirement.storage.identity())
                    .or_insert_with(|| (requirement.device, Vec::new()))
                    .1;
                slots.extend(requirement.requests.iter().map(|request| request.slot_key().2));
            }
            if !requirement.workspace_requests.is_empty() {
                if !requirement.storage.is_workspace_only() {
                    return Err(GpuAdmissionError::NativeReservation(
                        "matrix storage received a workspace reservation".into(),
                    ));
                }
                let identity = requirement.storage.identity();
                let entry = selected_workspace.entry(identity).or_insert_with(|| {
                    (requirement.device, Arc::clone(&inventory[&identity].1), Vec::new())
                });
                entry.2.extend_from_slice(requirement.workspace_requests);
            }
        }
        let mut regions = selected
            .into_par_iter()
            .map(|(identity, (device, slots))| {
                let storage = &inventory[&identity].1;
                let region = Arc::new(
                    storage
                        .reserve_region(&slots)
                        .map_err(GpuAdmissionError::NativeReservation)?
                        .ok_or(GpuAdmissionError::StaleRegion { device })?,
                );
                Ok((identity, (device, region.storage())))
            })
            .collect::<Result<BTreeMap<_, _>, GpuAdmissionError>>()?;
        for (identity, (device, storage, _)) in &selected_workspace {
            regions.insert(*identity, (*device, Arc::clone(storage)));
        }
        let workspace_requests = selected_workspace
            .into_par_iter()
            .map(|(identity, (device, storage, requests))| {
                let _ = device;
                let requests = requests
                    .iter()
                    .map(|request| {
                        storage
                            .reserve_workspace(std::slice::from_ref(request))
                            .map(|reservation| {
                                drop(reservation);
                                *request
                            })
                            .map_err(GpuAdmissionError::NativeReservation)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((identity, requests))
            })
            .collect::<Result<BTreeMap<_, _>, GpuAdmissionError>>()?;
        Ok(Arc::new(GpuMemoryRegion {
            _inventory: regions,
            workspace_requests: std::sync::Mutex::new(workspace_requests),
        }))
    }

    pub(crate) fn execution_identities(&self) -> Option<&[(i32, u64)]> {
        self.execution_identities.as_deref()
    }

    fn validate_prepared_storage(
        &self,
        device: usize,
        storage: &GpuPreparedStorage,
        inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<(), GpuAdmissionError> {
        if storage.is_region_authorized() {
            return Err(GpuAdmissionError::AcquisitionConflict(
                "prepared region construction cannot consume a region-authorized storage view"
                    .into(),
            ));
        }
        self.devices.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
        let expected = self
            .execution_identities
            .as_ref()
            .and_then(|identities| identities.get(device))
            .ok_or(GpuAdmissionError::ExecutionMismatch)?;
        let execution_matches = if storage.is_workspace_only() {
            storage.execution_owner_id() == expected.1 &&
                storage
                    .workspace_claims()
                    .is_some_and(|claims| claims.iter().all(|claim| claim.device == expected.0))
        } else {
            *expected == (storage.device(), storage.execution_owner_id())
        };
        if !execution_matches {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        validate_prepared_inventory(device, storage, inventory)
    }

    /// Start accounting from native evidence acquired at the explicit setup
    /// boundary. Numeric CUDA observations cannot authorize an allocation ledger.
    /// Consuming the non-clonable receipts retains the execution owners and keeps
    /// their setup evidence distinct from diagnostics and subsequent refreshes.
    /// The fleet dispatcher exclusively owns this ledger and all its submissions.
    pub fn new(
        epochs: Vec<GpuAllocationEpoch>,
        percent: u32,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<Self, GpuAdmissionError> {
        let snapshots = Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::InitialSetup)?;
        let memory = snapshots.iter().map(|(memory, _, _)| *memory).collect::<Vec<_>>();
        let physical = snapshots.iter().map(|(_, physical, _)| *physical).collect::<Vec<_>>();
        let identities = snapshots.iter().map(|(_, _, identity)| *identity).collect::<Vec<_>>();
        let mut ledger =
            Self::from_accounting_snapshot(&memory, &physical, percent, Some(&identities))?;
        let mut pinned_setup = vec![0u64; ledger.devices.len()];
        for (device, storage) in &prepared {
            if storage.is_workspace_only() {
                let bytes = u64::try_from(
                    storage
                        .workspace_accounting()
                        .map_err(GpuAdmissionError::ReleaseFailure)?
                        .pinned_bytes,
                )
                .map_err(|_| GpuAdmissionError::Overflow)?;
                pinned_setup[*device] =
                    pinned_setup[*device].checked_add(bytes).ok_or(GpuAdmissionError::Overflow)?;
                continue;
            }
            let bytes = storage
                .snapshot()
                .map_err(GpuAdmissionError::ReleaseFailure)?
                .into_iter()
                .filter(|slot| {
                    slot.identity().kind() ==
                        mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotKind::PinnedHost
                })
                .try_fold(0u64, |total, slot| {
                    total
                        .checked_add(
                            u64::try_from(slot.identity().requested_backing_bytes())
                                .map_err(|_| GpuAdmissionError::Overflow)?,
                        )
                        .ok_or(GpuAdmissionError::Overflow)
                })?;
            pinned_setup[*device] =
                pinned_setup[*device].checked_add(bytes).ok_or(GpuAdmissionError::Overflow)?;
        }
        for (index, (device, bytes)) in ledger.devices.iter_mut().zip(pinned_setup).enumerate() {
            if bytes > device.pinned_budget_bytes {
                return Err(GpuAdmissionError::PinnedCapacity {
                    device: index,
                    requested_bytes: bytes,
                    charged_bytes: 0,
                    budget_bytes: device.pinned_budget_bytes,
                });
            }
            device.pinned_allocation_bytes = bytes;
        }
        for (device, storage) in prepared {
            crate::backend::poly_gpu::record_provisioning_permit();
            let identity =
                identities.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
            let execution_matches = if storage.is_workspace_only() {
                storage.execution_owner_id() == identity.1 &&
                    storage.workspace_claims().is_some_and(|claims| {
                        claims.iter().all(|claim| claim.device == identity.0)
                    })
            } else {
                *identity == (storage.device(), storage.execution_owner_id())
            };
            if !execution_matches {
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
            if ledger.prepared_storages.insert(storage.identity(), (device, storage)).is_some() {
                return Err(GpuAdmissionError::DuplicatePreparedStorage);
            }
            crate::backend::poly_gpu::record_provisioning_append();
        }
        // Registering the inventory cannot promote stale setup observations.
        drop(Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::InitialSetup)?);
        ledger.observation_epochs = epochs;
        Ok(ledger)
    }

    fn validate_epochs(
        epochs: &[GpuAllocationEpoch],
        boundary: GpuAllocationEpochBoundary,
    ) -> Result<Vec<(GpuDeviceMemory, u64, (i32, u64))>, GpuAdmissionError> {
        epochs
            .par_iter()
            .map(|epoch| {
                if epoch.boundary() != boundary ||
                    !epoch.is_current().map_err(GpuAdmissionError::InvalidEpoch)?
                {
                    return Err(GpuAdmissionError::InvalidEpoch(
                        "the boundary or native activity revision changed".into(),
                    ));
                }
                let memory = GpuDeviceMemory {
                    total_bytes: u64::try_from(epoch.total_bytes())
                        .map_err(|_| GpuAdmissionError::Overflow)?,
                    resident_bytes: u64::try_from(epoch.resident_bytes())
                        .map_err(|_| GpuAdmissionError::Overflow)?,
                };
                let physical = u64::try_from(
                    epoch.physical_resident_bytes().map_err(GpuAdmissionError::InvalidEpoch)?,
                )
                .map_err(|_| GpuAdmissionError::Overflow)?;
                if memory.resident_bytes > physical || physical > memory.total_bytes {
                    return Err(GpuAdmissionError::InvalidPhysicalResidency);
                }
                Ok((memory, physical, (epoch.device(), epoch.execution_identity())))
            })
            .collect()
    }

    // Shared arithmetic for verified initialization and synthetic lifecycle tests.
    // Keep this private: raw snapshots are not production admission evidence.
    fn from_accounting_snapshot(
        memory: &[GpuDeviceMemory],
        physical_resident_bytes: &[u64],
        percent: u32,
        execution_identities: Option<&[(i32, u64)]>,
    ) -> Result<Self, GpuAdmissionError> {
        if memory.is_empty() {
            return Err(GpuCalibrationError::ZeroGpuCount.into());
        }
        if physical_resident_bytes.len() != memory.len() {
            return Err(GpuAdmissionError::InvalidPhysicalResidency);
        }
        if let Some(identities) = execution_identities {
            let mut devices = BTreeSet::new();
            if identities.len() != memory.len() ||
                identities.iter().any(|(device, owner)| {
                    *device < 0 || *owner == 0 || !devices.insert(*device)
                })
            {
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
        }
        let devices = memory
            .iter()
            .zip(physical_resident_bytes)
            .enumerate()
            .map(|(device, (memory, physical))| {
                if memory.resident_bytes > *physical || *physical > memory.total_bytes {
                    return Err(GpuAdmissionError::InvalidPhysicalResidency);
                }
                let budget_bytes = memory.budget_bytes(percent)?;
                if memory.resident_bytes > budget_bytes {
                    return Err(GpuAdmissionError::Capacity {
                        device,
                        requested_bytes: 0,
                        charged_bytes: memory.resident_bytes,
                        budget_bytes,
                    });
                }
                if *physical > budget_bytes {
                    return Err(GpuAdmissionError::PhysicalCapacity {
                        device,
                        requested_bytes: 0,
                        charged_bytes: *physical,
                        budget_bytes,
                    });
                }
                Ok(GpuDeviceAdmission {
                    budget_bytes,
                    baseline_bytes: memory.resident_bytes,
                    allocation_bytes: 0,
                    pinned_allocation_bytes: 0,
                    pinned_budget_bytes: budget_bytes,
                    physical_baseline_bytes: *physical,
                    observed_resident_bytes: memory.resident_bytes,
                    observed_physical_bytes: *physical,
                })
            })
            .collect::<Result<Vec<_>, GpuAdmissionError>>()?;
        let identity = NEXT_LEDGER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| GpuAdmissionError::Overflow)?;
        let (release_sender, releases) = mpsc::channel();
        Ok(Self {
            identity,
            observation_epochs: Vec::new(),
            prepared_storages: BTreeMap::new(),
            device_totals: memory.iter().map(|device| device.total_bytes).collect(),
            next_allocation: 0,
            devices,
            allocations: BTreeMap::new(),
            release_completions: BTreeMap::new(),
            execution_identities: execution_identities.map(<[_]>::to_vec),
            release_sender,
            releases,
            release_error: None,
            abandoned_submission: false,
            provisioning_charges: BTreeMap::new(),
            next_charge: 0,
        })
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_test(
        memory: &[GpuDeviceMemory],
        physical_resident_bytes: &[u64],
        percent: u32,
        execution_identities: Option<&[(i32, u64)]>,
    ) -> Result<Self, GpuAdmissionError> {
        Self::from_accounting_snapshot(
            memory,
            physical_resident_bytes,
            percent,
            execution_identities,
        )
    }

    pub fn devices(&self) -> &[GpuDeviceAdmission] {
        &self.devices
    }

    /// Acquire a complete fleet reservation before publishing any participant.
    /// Managed allocation bounds and native logical groups are separate dimensions:
    /// prepared backing is already charged at setup and adds no physical bytes.
    /// All native groups must succeed before managed IDs/charges are committed.
    /// Failure drops every acquired native token; no command or GPU allocation
    /// runs here. Cancellation never disables native allocation checks.
    pub fn reserve(
        &mut self,
        requirements: &[GpuAllocationRequirement],
        prepared: &[GpuPreparedAllocationRequirement<'_>],
    ) -> Result<GpuMemoryReservation, GpuAdmissionError> {
        crate::backend::poly_gpu::record_prepared_forbidden(5);
        self.reserve_inner(requirements, prepared)
    }

    fn reserve_inner(
        &mut self,
        requirements: &[GpuAllocationRequirement],
        prepared: &[GpuPreparedAllocationRequirement<'_>],
    ) -> Result<GpuMemoryReservation, GpuAdmissionError> {
        self.poll_releases()?;
        let count = u64::try_from(requirements.len()).map_err(|_| GpuAdmissionError::Overflow)?;
        let next = self.next_allocation.checked_add(count).ok_or(GpuAdmissionError::Overflow)?;
        let mut additional = vec![0u64; self.devices.len()];
        let mut additional_pinned = vec![0u64; self.devices.len()];
        for requirement in requirements {
            let device = self
                .devices
                .get(requirement.device)
                .ok_or(GpuAdmissionError::InvalidDevice(requirement.device))?;
            let bytes = &mut additional[requirement.device];
            *bytes = bytes.checked_add(requirement.bytes).ok_or(GpuAdmissionError::Overflow)?;
            if *bytes > device.budget_bytes - device.charged_bytes() {
                return Err(GpuAdmissionError::Capacity {
                    device: requirement.device,
                    requested_bytes: *bytes,
                    charged_bytes: device.charged_bytes(),
                    budget_bytes: device.budget_bytes,
                });
            }
            let pinned = &mut additional_pinned[requirement.device];
            *pinned =
                pinned.checked_add(requirement.pinned_bytes).ok_or(GpuAdmissionError::Overflow)?;
            if *pinned > device.pinned_available_bytes() {
                return Err(GpuAdmissionError::PinnedCapacity {
                    device: requirement.device,
                    requested_bytes: *pinned,
                    charged_bytes: device.pinned_charged_bytes(),
                    budget_bytes: device.pinned_budget_bytes,
                });
            }
        }
        for (device, pinned) in self.devices.iter().zip(&additional_pinned) {
            device
                .pinned_allocation_bytes
                .checked_add(*pinned)
                .ok_or(GpuAdmissionError::Overflow)?;
        }
        // Validate the whole requested inventory before acquiring any logical
        // slots. Storage IDs are native and cannot be reused for later backing.
        let inventory = self
            .accepted_prepared_inventory()
            .map(|(device, storage)| (storage.identity(), (device, storage)))
            .collect::<BTreeMap<_, _>>();
        for request in prepared {
            self.validate_prepared_storage(request.device, request.storage, &inventory)?;
        }
        // Rayon joins all in-flight attempts before returning. Result collection
        // drops successes on any error, including independent groups on the same
        // storage. Each native failure also rolls back its own partial claims.
        let native = prepared
            .par_iter()
            .map(|request| {
                inventory[&request.storage.identity()]
                    .1
                    .reserve_with_outcome(request.requests)
                    .map(|reservation| (request.device, reservation))
                    .map_err(|error| match error {
                        GpuPreparedReservationError::AcquisitionConflict(message) => {
                            GpuAdmissionError::AcquisitionConflict(message)
                        }
                        GpuPreparedReservationError::Native(message) => {
                            GpuAdmissionError::NativeReservation(message)
                        }
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ids = requirements
            .iter()
            .enumerate()
            .map(|(index, requirement)| {
                let id = GpuAllocationId {
                    ledger: self.identity,
                    allocation: self.next_allocation + index as u64,
                };
                self.allocations.insert(
                    id,
                    Allocation { requirement: *requirement, state: AllocationState::Reserved },
                );
                id
            })
            .collect();
        for (device, additional) in self.devices.iter_mut().zip(additional) {
            device.allocation_bytes += additional;
        }
        for (device, pinned) in self.devices.iter_mut().zip(additional_pinned) {
            device.pinned_allocation_bytes += pinned;
        }
        self.next_allocation = next;
        Ok(GpuMemoryReservation { allocations: ids, prepared: native })
    }

    /// Allocation becoming resident never deducts its reservation. Its pool
    /// visibility can change at a different time from a physical-memory query.
    fn mark_resident(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        let allocation =
            self.allocations.get_mut(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::Submitted {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        allocation.state = AllocationState::Resident;
        Ok(())
    }

    /// Transfer a checked reservation to enqueue commands, atomically across the
    /// fleet. Leases are not clonable and cancellation through their allocation
    /// IDs is no longer possible. Dropping a not-yet-started lease cancels it.
    pub fn submit(
        &mut self,
        ids: &[GpuAllocationId],
    ) -> Result<Vec<GpuAllocationLease>, GpuAdmissionError> {
        self.poll_releases()?;
        let mut unique = BTreeSet::new();
        for id in ids {
            if !unique.insert(*id) {
                return Err(GpuAdmissionError::DuplicateAllocation);
            }
            let allocation =
                self.allocations.get(id).ok_or(GpuAdmissionError::UnknownAllocation)?;
            if allocation.state != AllocationState::Reserved {
                return Err(GpuAdmissionError::InvalidTransition);
            }
        }
        Ok(ids
            .iter()
            .map(|id| {
                let allocation = self.allocations.get_mut(id).expect("validated allocation");
                allocation.state = AllocationState::Leased;
                GpuAllocationLease {
                    id: *id,
                    requirement: allocation.requirement,
                    execution_identity: self
                        .execution_identities
                        .as_ref()
                        .map(|identities| identities[allocation.requirement.device]),
                    sender: self.release_sender.clone(),
                    submitted: false,
                    bound: false,
                }
            })
            .collect())
    }

    /// The final allocation owner has enqueued its event-ordered release.
    fn queue_release(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        let allocation =
            self.allocations.get_mut(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::Resident {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        allocation.state = AllocationState::ReleasePending;
        Ok(())
    }

    /// Attach the native completion returned when the allocation owner is
    /// consumed. No capacity is reclaimed until a later successful query.
    fn track_release(
        &mut self,
        id: GpuAllocationId,
        completion: GpuReleaseCompletion,
    ) -> Result<(), GpuAdmissionError> {
        self.queue_release(id)?;
        self.release_completions.insert(id, completion);
        Ok(())
    }

    /// Called at admission boundaries, never in a per-column kernel loop.
    /// Incomplete or failed CUDA events keep their allocation charged.
    pub fn poll_releases(&mut self) -> Result<usize, GpuAdmissionError> {
        let mut poll_error = self.release_error.clone();
        let charge_ids = self.provisioning_charges.keys().copied().collect::<Vec<_>>();
        for id in charge_ids {
            let Some(charge) = self.provisioning_charges.get_mut(&id) else { continue };
            if charge.unknown.iter().any(|unknown| !unknown.is_well_formed(self.devices.len())) {
                let error = GpuAdmissionError::InvalidTransition;
                self.release_error = Some(error.to_string());
                return Err(error);
            }
            let mut completed = Vec::new();
            for (index, storage) in charge.storages.iter().enumerate() {
                let StorageChargeState::ReleasePending(release) = &storage.state else {
                    continue;
                };
                match release.is_complete() {
                    Ok(true) => completed.push(index),
                    Ok(false) => {}
                    Err(error) => {
                        poll_error = Some(error.clone());
                        self.release_error = Some(error);
                    }
                }
            }
            for index in completed.into_iter().rev() {
                let storage = &charge.storages[index];
                if storage.device >= self.devices.len() ||
                    storage.device_bytes > self.devices[storage.device].allocation_bytes ||
                    storage.pinned_bytes > self.devices[storage.device].pinned_allocation_bytes
                {
                    let error = GpuAdmissionError::InvalidTransition;
                    poll_error = Some(error.to_string());
                    self.release_error = Some(error.to_string());
                    continue;
                }
                let storage = charge.storages.remove(index);
                self.devices[storage.device].allocation_bytes -= storage.device_bytes;
                self.devices[storage.device].pinned_allocation_bytes -= storage.pinned_bytes;
            }
        }

        let completed = self
            .provisioning_charges
            .iter()
            .filter_map(|(id, charge)| {
                (charge.storages.is_empty() &&
                    charge.reservation.is_empty() &&
                    charge.unknown.is_empty())
                .then_some(*id)
            })
            .collect::<Vec<_>>();
        for id in completed {
            self.provisioning_charges.remove(&id);
        }
        if let Some(error) = poll_error {
            return Err(GpuAdmissionError::ReleaseFailure(error));
        }
        while let Ok((id, notice)) = self.releases.try_recv() {
            match notice {
                AllocationNotice::Cancelled => {
                    let allocation =
                        self.allocations.get(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Leased {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    let allocation = self.allocations.remove(&id).expect("validated allocation");
                    self.devices[allocation.requirement.device].allocation_bytes -=
                        allocation.requirement.bytes;
                    self.devices[allocation.requirement.device].pinned_allocation_bytes -=
                        allocation.requirement.pinned_bytes;
                }
                AllocationNotice::Submitted => {
                    let allocation = self
                        .allocations
                        .get_mut(&id)
                        .ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Leased {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    allocation.state = AllocationState::Submitted;
                }
                AllocationNotice::Resident => self.mark_resident(id)?,
                AllocationNotice::Abandoned => {
                    let allocation = self
                        .allocations
                        .get_mut(&id)
                        .ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if allocation.state != AllocationState::Submitted {
                        return Err(GpuAdmissionError::InvalidTransition);
                    }
                    allocation.state = AllocationState::Abandoned;
                    self.abandoned_submission = true;
                }
                AllocationNotice::Released(Ok(completion)) => self.track_release(id, completion)?,
                AllocationNotice::Released(Err(error)) => {
                    self.queue_release(id)?;
                    self.release_error = Some(error);
                }
            }
        }
        if let Some(error) = &self.release_error {
            return Err(GpuAdmissionError::ReleaseFailure(error.clone()));
        }
        if self.abandoned_submission {
            return Err(GpuAdmissionError::AbandonedSubmission);
        }
        let completed = self
            .release_completions
            .iter()
            .filter_map(|(id, completion)| match completion.is_complete() {
                Ok(true) => Some(Ok(*id)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<Result<Vec<_>, _>>();
        let completed = match completed {
            Ok(completed) => completed,
            Err(error) => {
                // A later successful CUDA query cannot rehabilitate an epoch
                // whose native completion proof has already failed. Preserve
                // every pending charge, even for other events ready in this poll.
                self.release_error = Some(error.clone());
                return Err(GpuAdmissionError::ReleaseFailure(error));
            }
        };
        for id in &completed {
            self.release_completions.remove(id);
            self.complete_release(*id)?;
        }
        Ok(completed.len())
    }

    /// Called only after the native release completion proves storage reusable.
    /// Reader completion or host enqueue acknowledgement is not sufficient.
    fn complete_release(&mut self, id: GpuAllocationId) -> Result<(), GpuAdmissionError> {
        if self.release_completions.contains_key(&id) {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        let allocation = self.allocations.get(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
        if allocation.state != AllocationState::ReleasePending {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        let allocation = self.allocations.remove(&id).expect("validated allocation");
        self.devices[allocation.requirement.device].allocation_bytes -=
            allocation.requirement.bytes;
        self.devices[allocation.requirement.device].pinned_allocation_bytes -=
            allocation.requirement.pinned_bytes;
        // Completion returns managed capacity even if the CUDA pool retains
        // physical pages. Their continued residency remains visible separately.
        Ok(())
    }

    /// Cancel only unsubmitted reservations; validate the complete cancellation
    /// before changing accounting so a launch failure cannot retire live outputs.
    pub fn cancel(&mut self, ids: &[GpuAllocationId]) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        let mut unique = BTreeSet::new();
        for id in ids {
            if !unique.insert(*id) {
                return Err(GpuAdmissionError::DuplicateAllocation);
            }
            let allocation =
                self.allocations.get(id).ok_or(GpuAdmissionError::UnknownAllocation)?;
            if allocation.state != AllocationState::Reserved {
                return Err(GpuAdmissionError::InvalidTransition);
            }
        }
        for id in ids {
            let allocation = self.allocations.remove(id).expect("validated allocation");
            self.devices[allocation.requirement.device].allocation_bytes -=
                allocation.requirement.bytes;
            self.devices[allocation.requirement.device].pinned_allocation_bytes -=
                allocation.requirement.pinned_bytes;
        }
        Ok(())
    }

    /// Refresh observations at a native nonblocking boundary. Physical budget
    /// excess is reported without rejecting further valid managed reservations.
    /// Observations never absorb or retire allocation IDs or their charges.
    pub fn refresh(&mut self, epochs: Vec<GpuAllocationEpoch>) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        // Issued but unexecuted leases may allocate on enqueue workers while
        // this dispatcher validates evidence. Never accept an epoch until all
        // such commands are already resident or fully retired.
        self.validate_refresh_state()?;
        let snapshots = Self::validate_epochs(&epochs, GpuAllocationEpochBoundary::Refresh)?;
        let identities =
            self.execution_identities.as_ref().ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if snapshots.len() != self.devices.len() ||
            snapshots.iter().zip(identities).zip(&self.device_totals).any(
                |((snapshot, identity), total)| {
                    snapshot.2 != *identity || snapshot.0.total_bytes != *total
                },
            )
        {
            return Err(GpuAdmissionError::ExecutionMismatch);
        }
        let resident_bytes =
            snapshots.iter().map(|(memory, _, _)| memory.resident_bytes).collect::<Vec<_>>();
        let physical = snapshots.iter().map(|(_, physical, _)| *physical).collect::<Vec<_>>();
        self.refresh_accounting_snapshot(&resident_bytes, &physical, Some(&epochs))?;
        self.observation_epochs = epochs;
        Ok(())
    }

    fn validate_refresh_state(&self) -> Result<(), GpuAdmissionError> {
        if self.allocations.values().any(|allocation| allocation.state != AllocationState::Resident)
        {
            return Err(GpuAdmissionError::InvalidRefresh);
        }
        Ok(())
    }

    // None is used only by synthetic accounting tests. Production supplies the
    // native receipts and revalidates them after the final lifecycle drain.
    fn refresh_accounting_snapshot(
        &mut self,
        resident_bytes: &[u64],
        physical_resident_bytes: &[u64],
        epochs: Option<&[GpuAllocationEpoch]>,
    ) -> Result<(), GpuAdmissionError> {
        self.poll_releases()?;
        self.validate_refresh_state()?;
        if resident_bytes.len() != self.devices.len() ||
            physical_resident_bytes.len() != self.devices.len()
        {
            return Err(GpuAdmissionError::InvalidRefresh);
        }
        for (index, (resident, physical)) in
            resident_bytes.iter().zip(physical_resident_bytes).enumerate()
        {
            if *resident > *physical || *physical > self.device_totals[index] {
                return Err(GpuAdmissionError::InvalidPhysicalResidency);
            }
        }
        if let Some(epochs) = epochs {
            drop(Self::validate_epochs(epochs, GpuAllocationEpochBoundary::Refresh)?);
        }
        // No lifecycle polling or callback occurs between the last validation
        // and publication. With no outstanding leases, workers cannot add an
        // allocation that the accepted receipts did not observe.
        for ((device, resident), physical) in
            self.devices.iter_mut().zip(resident_bytes).zip(physical_resident_bytes)
        {
            device.observed_resident_bytes = *resident;
            device.observed_physical_bytes = *physical;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> GpuMemoryLedger {
        GpuMemoryLedger::from_accounting_snapshot(
            &[
                GpuDeviceMemory { total_bytes: 1_000, resident_bytes: 100 },
                GpuDeviceMemory { total_bytes: 2_000, resident_bytes: 200 },
            ],
            &[100, 200],
            100,
            None,
        )
        .unwrap()
    }

    #[test]
    fn construction_reservation_consumes_device_and_pinned_bytes_once() {
        let mut reservation =
            ConstructionReservation { device: vec![100, 200], pinned: vec![7, 11] };
        reservation.consume(0, 40, 3).unwrap();
        reservation.consume(1, 80, 5).unwrap();
        assert_eq!(reservation.device, vec![60, 120]);
        assert_eq!(reservation.pinned, vec![4, 6]);
        assert!(matches!(
            reservation.consume(0, 61, 0),
            Err(GpuAdmissionError::AllocationExceedsReservation { .. })
        ));
        assert_eq!(reservation.device, vec![60, 120]);
        assert_eq!(reservation.pinned, vec![4, 6]);
    }

    #[test]
    fn construction_reservation_unknown_conversion_preserves_each_device_delta() {
        let mut reservation =
            ConstructionReservation { device: vec![60, 0, 9], pinned: vec![4, 6, 0] };
        let unknown = reservation.take_unknown("native probe unavailable");
        assert_eq!(unknown.len(), 3);
        assert_eq!(
            (unknown[0].device, unknown[0].device_bytes, unknown[0].pinned_bytes),
            (0, 60, 4)
        );
        assert_eq!(
            (unknown[1].device, unknown[1].device_bytes, unknown[1].pinned_bytes),
            (1, 0, 6)
        );
        assert_eq!(
            (unknown[2].device, unknown[2].device_bytes, unknown[2].pinned_bytes),
            (2, 9, 0)
        );
        assert!(reservation.is_empty());
        assert!(unknown.iter().all(|charge| charge.reason == "native probe unavailable"));
    }

    #[test]
    fn pending_provisioning_rolls_back_each_device_charge() {
        let mut ledger = ledger();
        let before =
            ledger.devices().iter().map(|device| device.allocation_bytes).collect::<Vec<_>>();
        {
            let _pending = ledger.begin_prepared_provisioning(&[123, 456], &[11, 22]).unwrap();
        }
        assert_eq!(
            ledger.devices().iter().map(|device| device.allocation_bytes).collect::<Vec<_>>(),
            before,
        );
        assert_eq!(
            ledger
                .devices()
                .iter()
                .map(|device| device.pinned_allocation_bytes)
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
    }

    #[test]
    fn committed_provisioning_keeps_its_exact_charge() {
        let mut ledger = ledger();
        let pending = ledger.begin_prepared_provisioning(&[123, 456], &[11, 22]).unwrap();
        pending.commit();
        assert_eq!(ledger.devices()[0].allocation_bytes, 123);
        assert_eq!(ledger.devices()[1].allocation_bytes, 456);
        assert_eq!(ledger.devices()[0].pinned_allocation_bytes, 11);
        assert_eq!(ledger.devices()[1].pinned_allocation_bytes, 22);
    }

    #[test]
    fn in_progress_and_unassigned_provisioning_survives_poll() {
        let mut ledger = ledger();
        let pending = ledger.begin_prepared_provisioning(&[123, 0], &[11, 0]).unwrap();
        assert_eq!(pending.ledger.poll_releases().unwrap(), 0);
        assert_eq!(pending.ledger.provisioning_charges.len(), 1);
        pending.commit();
        assert_eq!(ledger.poll_releases().unwrap(), 0);
        assert_eq!(ledger.provisioning_charges.len(), 1);
    }

    #[test]
    fn zero_byte_provisioning_can_complete_without_storage() {
        let mut ledger = ledger();
        ledger.begin_prepared_provisioning(&[0, 0], &[0, 0]).unwrap().commit();
        assert_eq!(ledger.poll_releases().unwrap(), 0);
        assert!(ledger.provisioning_charges.is_empty());
    }

    #[test]
    fn warmup_checkpoint_restores_device_and_pinned_charges() {
        let mut ledger = ledger();
        let checkpoint = ledger.warmup_checkpoint();
        let pending = ledger.begin_prepared_provisioning(&[123, 456], &[11, 22]).unwrap();
        pending.commit();
        ledger.rollback_warmup(checkpoint);
        assert_eq!(ledger.devices()[0].allocation_bytes, 0);
        assert_eq!(ledger.devices()[1].allocation_bytes, 0);
        assert_eq!(ledger.devices()[0].pinned_allocation_bytes, 0);
        assert_eq!(ledger.devices()[1].pinned_allocation_bytes, 0);
    }

    #[test]
    fn warmup_checkpoint_rolls_back_only_pinned_transaction_delta_after_poll_release() {
        let mut ledger = ledger();
        let reservation = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 100, pinned_bytes: 17 }], &[])
            .unwrap();
        let checkpoint = ledger.warmup_checkpoint();
        ledger.cancel(&reservation.allocations).unwrap();
        let pending = ledger.begin_prepared_provisioning(&[123, 0], &[11, 0]).unwrap();
        pending.commit();
        ledger.rollback_warmup(checkpoint);
        assert_eq!(ledger.devices()[0].allocation_bytes, 0);
        assert_eq!(ledger.devices()[0].pinned_allocation_bytes, 0);
    }

    #[test]
    fn reserve_accumulates_pinned_requirements_per_device() {
        let mut ledger = ledger();
        let error = match ledger.reserve(
            &[
                GpuAllocationRequirement { device: 0, bytes: 1, pinned_bytes: 600 },
                GpuAllocationRequirement { device: 0, bytes: 1, pinned_bytes: 500 },
            ],
            &[],
        ) {
            Ok(_) => panic!("pinned requirements must be aggregated before admission"),
            Err(error) => error,
        };
        assert!(matches!(error, GpuAdmissionError::PinnedCapacity { requested_bytes: 1_100, .. }));
    }
}
