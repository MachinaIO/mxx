//! Pure typed-slot matching and transactional primitive-step claims.

use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuMatrixReservation, GpuPreparedRequest, GpuPreparedSlotKind, GpuPreparedSlotSnapshot,
        GpuPreparedStorage, GpuTracedClaim,
    },
    poly::dcrt::gpu::GpuDCRTPolyParams,
};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

/// Warmup-owned exact claim lease. The native reservation is acquired once
/// during preparation and rearmed after each submitted phase; production
/// replay therefore never performs claim matching or reservation creation.
pub(super) struct PreparedClaimLease {
    reservations: Mutex<Option<Vec<GpuMatrixReservation>>>,
}

impl PreparedClaimLease {
    pub(super) fn new(reservations: Vec<GpuMatrixReservation>) -> Self {
        Self { reservations: Mutex::new(Some(reservations)) }
    }

    pub(super) fn run<T>(
        &self,
        operation: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let reservations = self
            .reservations
            .lock()
            .map_err(|_| "prepared claim lease poisoned".to_owned())?
            .take()
            .ok_or_else(|| "prepared claim lease is unavailable".to_owned())?;
        let mut reservations = reservations.into_iter();
        let Some(first) = reservations.next() else {
            return operation();
        };
        let dispatch = first.enter_or_extend(reservations.collect())?;
        let result = operation();
        match result {
            Ok(value) => {
                let reservations = dispatch.finish()?;
                self.reservations
                    .lock()
                    .map_err(|_| "prepared claim lease poisoned".to_owned())?
                    .replace(reservations);
                Ok(value)
            }
            Err(error) => {
                drop(dispatch);
                Err(error)
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct PreparedClaimBroker {
    parameters: GpuDCRTPolyParams,
    storages: Vec<Arc<GpuPreparedStorage>>,
}

impl PreparedClaimBroker {
    pub(super) fn new(
        parameters: &GpuDCRTPolyParams,
        mut storages: Vec<Arc<GpuPreparedStorage>>,
    ) -> Self {
        storages.retain(|storage| storage.matches_parameters(parameters));
        Self { parameters: parameters.clone(), storages }
    }

    /// Hold `claims` around `run`. Use for steps whose claim lists were derived
    /// from the live owners (exports, readbacks).
    pub(super) fn hold<T>(
        &self,
        claims: &[GpuTracedClaim],
        reclaim_uploads: bool,
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, false, reclaim_uploads, run)
    }

    /// Hold recorded (traced) claims around `run`: the step executes with
    /// deterministic consumer-event claims, exactly as it was traced.
    pub(super) fn hold_traced<T>(
        &self,
        claims: &[GpuTracedClaim],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, true, false, run)
    }

    /// Hold traced claims for a host boundary that may release pending upload
    /// backing. The claim order remains native-trace order; reclamation only
    /// polls the existing event/pinned-resource lifetime records.
    pub(super) fn hold_traced_reclaim<T>(
        &self,
        claims: &[GpuTracedClaim],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, true, true, run)
    }

    pub(super) fn reserve_traced(
        &self,
        claims: &[GpuTracedClaim],
    ) -> Result<Arc<PreparedClaimLease>, String> {
        let assignment = self
            .assignment(claims, &HashSet::new(), &HashSet::new())?
            .ok_or_else(|| "prepared inventory cannot fit warmup claim lease".to_owned())?;
        let mut groups = Vec::<(&Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
        for (storage, request) in assignment {
            if let Some((previous, requests)) = groups.last_mut() {
                if previous.identity() == storage.identity() {
                    requests.push(request);
                    continue;
                }
            }
            groups.push((storage, vec![request]));
        }
        let mut reservations = groups
            .into_iter()
            .map(|(storage, requests)| storage.reserve(&requests))
            .collect::<Result<Vec<_>, _>>()?;
        for reservation in &mut reservations {
            reservation.require_all_resources()?;
        }
        Ok(Arc::new(PreparedClaimLease::new(reservations)))
    }

    fn match_claims(
        &self,
        claims: &[GpuTracedClaim],
        excluded: &HashSet<u64>,
        reclaiming: &HashSet<u64>,
    ) -> Result<Vec<Option<GpuPreparedRequest>>, String> {
        let snapshot = self
            .storages
            .par_iter()
            .map(|storage| {
                storage.snapshot().map(|slots| {
                    slots
                        .into_iter()
                        .map(|slot| {
                            let identity = slot.identity();
                            let deferred_transfer = reclaiming.contains(&identity.slot_id()) &&
                                matches!(
                                    identity.kind(),
                                    GpuPreparedSlotKind::PinnedHost |
                                        GpuPreparedSlotKind::CompletionEvent
                                );
                            let eligible = !excluded.contains(&identity.slot_id()) &&
                                (slot.is_available() || deferred_transfer);
                            (slot, eligible)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        GpuPreparedSlotSnapshot::assign(&self.parameters, &snapshot, claims)
    }

    /// Fit the largest subset before provisioning more backing. A flexible
    /// claim must not consume the only slot compatible with a later claim.
    pub(super) fn missing(
        &self,
        claims: &[GpuTracedClaim],
        used: &mut HashSet<u64>,
    ) -> Result<Vec<GpuTracedClaim>, String> {
        let selected = self.match_claims(claims, used, &HashSet::new())?;
        let mut missing = Vec::new();
        for (claim, selected) in claims.iter().zip(selected) {
            if let Some(request) = selected {
                used.insert(request.slot_key().1);
            } else {
                missing.push(*claim);
            }
        }
        Ok(missing)
    }

    pub(super) fn assignment(
        &self,
        claims: &[GpuTracedClaim],
        excluded: &HashSet<u64>,
        reclaiming: &HashSet<u64>,
    ) -> Result<Option<Vec<(&Arc<GpuPreparedStorage>, GpuPreparedRequest)>>, String> {
        crate::backend::poly_gpu::record_prepared_forbidden(2);
        let storage_by_id = self
            .storages
            .iter()
            .map(|storage| (storage.identity(), storage))
            .collect::<HashMap<_, _>>();
        Ok(self
            .match_claims(claims, excluded, reclaiming)?
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|requests| {
                requests
                    .into_iter()
                    .map(|request| (storage_by_id[&request.slot_key().0], request))
                    .collect()
            }))
    }

    /// Apply the boundary's pending-upload policy once: poll existing release
    /// events, admit deferred pinned/event slots when necessary, and poll only
    /// the selected deferred slots before reservation. Preflight and execution
    /// consume this same selection policy.
    pub(super) fn assignment_with_reclaim_policy(
        &self,
        claims: &[GpuTracedClaim],
    ) -> Result<Option<Vec<(&Arc<GpuPreparedStorage>, GpuPreparedRequest)>>, String> {
        let mut pending = HashSet::new();
        for storage in &self.storages {
            for index in storage.poll_releases(&[])? {
                let slot = storage.slot_identity(index).unwrap();
                if matches!(
                    slot.kind(),
                    GpuPreparedSlotKind::PinnedHost | GpuPreparedSlotKind::CompletionEvent
                ) {
                    pending.insert(slot.slot_id());
                }
            }
        }
        let mut assignment = self.assignment(claims, &HashSet::new(), &HashSet::new())?;
        if assignment.is_none() && !pending.is_empty() {
            assignment = self.assignment(claims, &HashSet::new(), &pending)?;
        }
        if let Some(assignment) = &assignment {
            for storage in &self.storages {
                let selected = assignment
                    .iter()
                    .filter(|(owner, request)| {
                        owner.identity() == storage.identity() &&
                            pending.contains(&request.slot_key().1)
                    })
                    .map(|(_, request)| request.slot_key().2)
                    .collect::<Vec<_>>();
                if !selected.is_empty() {
                    storage.poll_releases(&selected)?;
                }
            }
        }
        Ok(assignment)
    }

    fn hold_inner<T>(
        &self,
        claims: &[GpuTracedClaim],
        traced: bool,
        reclaim_uploads: bool,
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        // Readback already completes on the host. Apply the same deferred
        // upload policy as boundary preflight when requested.
        let assignment = if reclaim_uploads {
            self.assignment_with_reclaim_policy(claims)?
        } else {
            self.assignment(claims, &HashSet::new(), &HashSet::new())?
        };
        let assignment = assignment.ok_or_else(|| {
            let missing = self.match_claims(claims, &HashSet::new(), &HashSet::new())
                    .map(|selected| claims.iter().zip(selected).filter_map(|(claim, request)| request.is_none().then_some(*claim)).collect::<Vec<_>>());
                format!("prepared inventory cannot fit simultaneous admitted claims; missing: {missing:?}; requested: {claims:?}")
            })?;
        // Do not activate a dispatch until every chosen slot has been reserved.
        // If availability changed, dropping the prefix rolls back the transaction.
        // Batch only consecutive claims from the same backing store. Native
        // dispatch consumes claims in order; regrouping A, B, A into A, A, B
        // would change the resource type seen by the second allocation.
        let mut groups = Vec::<(&Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
        for (storage, request) in assignment {
            if let Some((previous, requests)) = groups.last_mut() {
                if previous.identity() == storage.identity() {
                    requests.push(request);
                    continue;
                }
            }
            groups.push((storage, vec![request]));
        }
        let mut reservations = groups
            .into_iter()
            .map(|(storage, requests)| storage.reserve(&requests))
            .collect::<Result<Vec<_>, _>>()?;
        for reservation in &mut reservations {
            reservation.require_all_resources()?;
        }
        let mut reservations = reservations.into_iter();
        let dispatch = match reservations.next() {
            Some(first) => Some(first.enter_or_extend(reservations.collect())?),
            None => None,
        };
        let guard = traced.then(mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedStepGuard::enter);
        let result = run();
        drop(guard);
        match (result, dispatch) {
            // A failed step retracts its permit (dropping cancels); its own
            // error is reported rather than the unconsumed-claims diagnostic.
            (Err(error), dispatch) => {
                drop(dispatch);
                Err(error)
            }
            (Ok(value), Some(dispatch)) => {
                drop(dispatch.finish()?);
                Ok(value)
            }
            (Ok(value), None) => Ok(value),
        }
    }
}
