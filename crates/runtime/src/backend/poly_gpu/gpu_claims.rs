//! Pure typed-slot matching and transactional primitive-step claims.

use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuPreparedRequest, GpuPreparedSlotKind, GpuPreparedSlotSnapshot, GpuPreparedStorage,
        GpuTracedClaim,
    },
    poly::dcrt::gpu::GpuDCRTPolyParams,
};
use rayon::prelude::*;
use std::{collections::HashSet, sync::Arc};

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
}
