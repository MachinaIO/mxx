//! Pure typed-slot matching and transactional primitive-step claims.

use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuPreparedRequest, GpuPreparedSlotKind, GpuPreparedSlotSnapshot, GpuPreparedStorage,
        GpuTracedClaim,
    },
    poly::dcrt::gpu::GpuDCRTPolyParams,
};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

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
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, false, run)
    }

    /// Hold recorded (traced) claims around `run`: the step executes with
    /// deterministic consumer-event claims, exactly as it was traced.
    pub(super) fn hold_traced<T>(
        &self,
        claims: &[GpuTracedClaim],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, true, run)
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

    fn hold_inner<T>(
        &self,
        claims: &[GpuTracedClaim],
        traced: bool,
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let assignment =
            self.assignment(claims, &HashSet::new(), &HashSet::new())?.ok_or_else(|| {
                format!("prepared inventory cannot fit simultaneous admitted claims: {claims:?}")
            })?;
        // Do not activate a dispatch until every chosen slot has been reserved.
        // If availability changed, dropping the prefix rolls back the transaction.
        // One native reservation per backing store, rather than per claim.
        // Preserve first-use storage order and claim order inside each store;
        // a failed store reservation drops and cancels every preceding group.
        let mut groups = Vec::<(&Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
        let mut group_indices = HashMap::new();
        for (storage, request) in assignment {
            let index = *group_indices.entry(storage.identity()).or_insert_with(|| {
                let index = groups.len();
                groups.push((storage, Vec::new()));
                index
            });
            groups[index].1.push(request);
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
