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
            GpuPreparedReservationError, GpuPreparedStorage, GpuSmallMatrix,
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
        Arc, Weak,
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
    region_generations: BTreeSet<u64>,
    provisioning_count: usize,
}

struct ProvisioningCharge {
    device: Vec<u64>,
    pinned: Vec<u64>,
}

/// RAII transaction for warmup backing.  Charges are installed before native
/// construction and are rolled back automatically unless the resulting
/// storages are appended to the accepted inventory.  This keeps each device's
/// exact charge live across every fallible construction step, including panic
/// unwinding and asynchronous native destruction.
pub(crate) struct PendingProvisioning<'a> {
    ledger: &'a mut GpuMemoryLedger,
    additional_device: Vec<u64>,
    additional_pinned: Vec<u64>,
    provisioning_index: usize,
    committed: bool,
}

impl PendingProvisioning<'_> {
    pub(crate) fn append(
        &mut self,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<(), GpuAdmissionError> {
        self.ledger.append_prepared_storages(prepared)
    }

    pub(crate) fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for PendingProvisioning<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.ledger
                .rollback_prepared_provisioning(&self.additional_device, &self.additional_pinned);
            self.ledger.provisioning_charges.remove(self.provisioning_index);
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
    Discarded,
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
    if *owner != device ||
        accepted.device() != storage.device() ||
        accepted.execution_owner_id() != storage.execution_owner_id()
    {
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

    /// Commit a prepared scalar backing whose native owner is managed by the
    /// scalar buffer rather than by a matrix allocation observer.
    pub(crate) fn commit_prepared_scalar(&mut self) -> Result<(), GpuAdmissionError> {
        self.begin()?;
        self.sender
            .send((self.id, AllocationNotice::Resident))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(())
    }

    /// Attach the scalar buffer's event-ordered release to this lease. Keep
    /// the lease bound until its owner is dropped so `Drop` cannot report a
    /// second transition for the same allocation.
    pub(crate) fn retire_prepared_scalar(
        &mut self,
        completion: GpuReleaseCompletion,
    ) -> Result<(), GpuAdmissionError> {
        if !self.bound {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        self.sender
            .send((self.id, AllocationNotice::Released(Ok(completion))))
            .map_err(|_| GpuAdmissionError::Closed)
    }

    pub(crate) fn cancel_prepared_scalar(&mut self) -> Result<(), GpuAdmissionError> {
        if !self.submitted {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        self.sender
            .send((self.id, AllocationNotice::Discarded))
            .map_err(|_| GpuAdmissionError::Closed)?;
        self.bound = true;
        Ok(())
    }

    /// Mark a prepared scalar generation as permanently unknown. The charge
    /// remains live and the dispatcher becomes sticky-failed; it must never be
    /// reclaimed without native completion evidence.
    pub(crate) fn quarantine_prepared_scalar(&mut self) -> Result<(), GpuAdmissionError> {
        if !self.bound {
            return Err(GpuAdmissionError::InvalidTransition);
        }
        self.sender
            .send((
                self.id,
                AllocationNotice::Released(Err(
                    "prepared scalar generation lost its native release owner".into(),
                )),
            ))
            .map_err(|_| GpuAdmissionError::Closed)
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
    inventory: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
}

impl GpuMemoryRegion {
    pub(crate) fn empty() -> Self {
        Self { inventory: BTreeMap::new() }
    }

    #[cfg(test)]
    pub(crate) fn empty_for_test() -> Self {
        Self::empty()
    }

    pub(crate) fn prepared_inventory(
        &self,
    ) -> impl Iterator<Item = (usize, Arc<GpuPreparedStorage>)> {
        self.inventory.values().cloned().collect::<Vec<_>>().into_iter()
    }
}

pub struct GpuMemoryLedger {
    identity: u64,
    observation_epochs: Vec<GpuAllocationEpoch>,
    // Retain the actual backing accepted at setup. A later allocation on the
    // same execution owner is not implicitly part of the charged inventory.
    prepared_storages: BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    // The wave owns each inventory; the ledger only observes the innermost
    // surviving region. Error unwinding restores its parent without a callback
    // borrowing this mutable ledger, and without retaining completed waves.
    region_inventories: Vec<(u64, Weak<GpuMemoryRegion>)>,
    next_region_generation: u64,
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
    provisioning_charges: Vec<ProvisioningCharge>,
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
        crate::backend::poly_gpu::record_provisioning_begin();
        for ((device, requested_bytes), pinned) in
            self.devices.iter_mut().zip(additional_device).zip(additional_pinned)
        {
            device.allocation_bytes += *requested_bytes;
            device.pinned_allocation_bytes += *pinned;
        }
        let provisioning_index = self.provisioning_charges.len();
        self.provisioning_charges.push(ProvisioningCharge {
            device: additional_device.to_vec(),
            pinned: additional_pinned.to_vec(),
        });
        Ok(PendingProvisioning {
            ledger: self,
            additional_device: additional_device.to_vec(),
            additional_pinned: additional_pinned.to_vec(),
            provisioning_index,
            committed: false,
        })
    }

    /// Undo a not-yet-published provisioning charge.  This is intentionally
    /// only used by the owning warmup transaction after native construction
    /// or detached-region acquisition fails.
    pub(crate) fn rollback_prepared_provisioning(
        &mut self,
        additional_device: &[u64],
        additional_pinned: &[u64],
    ) {
        debug_assert_eq!(additional_device.len(), self.devices.len());
        debug_assert_eq!(additional_pinned.len(), self.devices.len());
        for ((device, requested_bytes), pinned) in
            self.devices.iter_mut().zip(additional_device).zip(additional_pinned)
        {
            debug_assert!(device.allocation_bytes >= *requested_bytes);
            device.allocation_bytes -= *requested_bytes;
            debug_assert!(device.pinned_allocation_bytes >= *pinned);
            device.pinned_allocation_bytes -= *pinned;
        }
    }

    /// Append newly provisioned prepared storage to this execution owner.
    /// The ledger is never replaced after setup; all identities and the full
    /// fleet budget are validated before any store becomes visible.
    pub(crate) fn append_prepared_storages(
        &mut self,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
    ) -> Result<(), GpuAdmissionError> {
        if prepared.is_empty() {
            return Ok(());
        }
        self.poll_releases()?;
        let mut identities = BTreeSet::new();
        for (device, storage) in &prepared {
            let expected = self
                .execution_identities
                .as_ref()
                .and_then(|identities| identities.get(*device))
                .ok_or(GpuAdmissionError::InvalidDevice(*device))?;
            if *expected != (storage.device(), storage.execution_owner_id()) ||
                !identities.insert(storage.identity()) ||
                self.prepared_storages.contains_key(&storage.identity())
            {
                return Err(GpuAdmissionError::ExecutionMismatch);
            }
        }
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
            region_generations: self
                .region_inventories
                .iter()
                .map(|(generation, _)| *generation)
                .collect(),
            provisioning_count: self.provisioning_charges.len(),
        }
    }

    /// Remove only backing introduced by a failed speculative generation.
    /// Existing setup/program owners are left untouched; native storage owners
    /// are dropped after all failed reservations have unwound.
    pub(crate) fn rollback_warmup(&mut self, checkpoint: GpuWarmupCheckpoint) {
        self.prepared_storages.retain(|identity, _| checkpoint.storage_ids.contains(identity));
        self.region_inventories
            .retain(|(generation, _)| checkpoint.region_generations.contains(generation));
        if checkpoint.provisioning_count <= self.provisioning_charges.len() {
            let charges = self.provisioning_charges.split_off(checkpoint.provisioning_count);
            for charge in charges.into_iter().rev() {
                for (device, bytes) in self.devices.iter_mut().zip(charge.device) {
                    if let Some(value) = device.allocation_bytes.checked_sub(bytes) {
                        device.allocation_bytes = value;
                    } else {
                        self.release_error = Some("warmup rollback device charge underflow".into());
                    }
                }
                for (device, bytes) in self.devices.iter_mut().zip(charge.pinned) {
                    if let Some(value) = device.pinned_allocation_bytes.checked_sub(bytes) {
                        device.pinned_allocation_bytes = value;
                    } else {
                        self.release_error = Some("warmup rollback pinned charge underflow".into());
                    }
                }
            }
        } else {
            self.release_error = Some("warmup rollback provisioning journal changed".into());
        }
    }

    /// The complete accepted setup inventory. Unlike `prepared_inventory`,
    /// this is not narrowed by a currently-live execution region and is the
    /// only inventory suitable for resolving a replacement generation.
    pub(crate) fn accepted_prepared_inventory(
        &self,
    ) -> impl Iterator<Item = (usize, Arc<GpuPreparedStorage>)> {
        self.prepared_storages.values().cloned().collect::<Vec<_>>().into_iter()
    }

    pub(crate) fn prepared_inventory(
        &self,
    ) -> impl Iterator<Item = (usize, Arc<GpuPreparedStorage>)> {
        let region = self.region_inventories.iter().rev().find_map(|(_, region)| region.upgrade());
        region
            .as_ref()
            .map(|region| &region.inventory)
            .unwrap_or(&self.prepared_storages)
            .values()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Commit complete native region capacity, without taking operation leases.
    /// Matching/liveness chooses these exact slots first; this transaction only
    /// publishes an inventory after every storage owner has been acquired.
    /// Dropping the returned owner restores the nearest surviving parent.
    pub fn reserve_region(
        &mut self,
        requirements: &[GpuPreparedAllocationRequirement<'_>],
    ) -> Result<Arc<GpuMemoryRegion>, GpuAdmissionError> {
        crate::backend::poly_gpu::record_prepared_forbidden(5);
        let inventory = self
            .prepared_inventory()
            .map(|(device, storage)| (storage.identity(), (device, storage)))
            .collect::<BTreeMap<_, _>>();
        let mut selected = BTreeMap::<u64, (usize, Vec<usize>)>::new();
        for requirement in requirements {
            self.validate_prepared_storage(requirement.device, requirement.storage, &inventory)?;
            let slots = &mut selected
                .entry(requirement.storage.identity())
                .or_insert_with(|| (requirement.device, Vec::new()))
                .1;
            slots.extend(requirement.requests.iter().map(|request| request.slot_key().2));
        }
        let regions = selected
            .into_par_iter()
            .map(|(identity, (device, slots))| {
                let storage = &inventory[&identity].1;
                let region = Arc::new(
                    storage
                        .reserve_region(&slots, None)
                        .map_err(GpuAdmissionError::NativeReservation)?
                        .ok_or(GpuAdmissionError::StaleRegion { device })?,
                );
                Ok((identity, (device, region.storage())))
            })
            .collect::<Result<BTreeMap<_, _>, GpuAdmissionError>>()?;
        let regions = Arc::new(GpuMemoryRegion { inventory: regions });
        self.region_inventories.retain(|(_, entry)| entry.strong_count() != 0);
        let generation = self.next_region_generation;
        self.next_region_generation =
            self.next_region_generation.checked_add(1).ok_or(GpuAdmissionError::Overflow)?;
        self.region_inventories.push((generation, Arc::downgrade(&regions)));
        Ok(regions)
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
        self.devices.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
        let expected = self
            .execution_identities
            .as_ref()
            .and_then(|identities| identities.get(device))
            .ok_or(GpuAdmissionError::ExecutionMismatch)?;
        if *expected != (storage.device(), storage.execution_owner_id()) {
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
            if *identity != (storage.device(), storage.execution_owner_id()) {
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
            region_inventories: Vec::new(),
            next_region_generation: 0,
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
            provisioning_charges: Vec::new(),
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
            .prepared_inventory()
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
                AllocationNotice::Discarded => {
                    let allocation =
                        self.allocations.get(&id).ok_or(GpuAdmissionError::UnknownAllocation)?;
                    if !matches!(
                        allocation.state,
                        AllocationState::Leased |
                            AllocationState::Submitted |
                            AllocationState::Resident
                    ) {
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
    fn warmup_checkpoint_uses_region_identity_after_weak_compaction() {
        let mut ledger = ledger();
        let old = Arc::new(GpuMemoryRegion::empty_for_test());
        ledger.region_inventories.push((41, Arc::downgrade(&old)));
        let checkpoint = ledger.warmup_checkpoint();
        drop(old);

        let newer = Arc::new(GpuMemoryRegion::empty_for_test());
        ledger.region_inventories.push((42, Arc::downgrade(&newer)));
        ledger.region_inventories.retain(|(_, entry)| entry.strong_count() != 0);
        assert_eq!(ledger.region_inventories.len(), 1);
        ledger.rollback_warmup(checkpoint);
        assert!(ledger.region_inventories.is_empty());
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

    #[test]
    fn committed_scalar_resize_failure_discards_without_poisoning_ledger() {
        let mut ledger = ledger();
        let reservation = ledger
            .reserve(&[GpuAllocationRequirement { device: 0, bytes: 32, pinned_bytes: 16 }], &[])
            .unwrap();
        let mut leases = ledger.submit(&reservation.allocations).unwrap();
        let mut lease = leases.pop().unwrap();
        lease.commit_prepared_scalar().unwrap();
        lease.cancel_prepared_scalar().unwrap();
        assert_eq!(ledger.poll_releases().unwrap(), 0);
        assert_eq!(ledger.devices()[0].allocation_bytes, 0);
        assert_eq!(ledger.devices()[0].pinned_allocation_bytes, 0);
        assert!(ledger.release_error.is_none());
    }
}
