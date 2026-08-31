//! Private support schedules and key-layout generation.
//!
//! A schedule assigns every support coordinate to a distinct candidate bucket
//! and chooses one slot per bucket; buckets without support use their dummy
//! cell. It is deliberately non-serializable and has a redacted `Debug`
//! implementation because selected slots are private data.

use super::{
    PbcCell, PbcError, PbcLayoutId, PbcParameters, PbcPublicLayout,
    matching::deterministic_matching,
};

/// A sorted, range-checked sparse support used internally during one key
/// generation or diagnostic run. Keeping the validated representation
/// private prevents retry code from accidentally revalidating or changing the
/// support between layout attempts.
pub(crate) struct ValidatedSupport {
    coordinates: Vec<usize>,
}

impl ValidatedSupport {
    /// Sorts and validates the support against the parameter set exactly once.
    pub(crate) fn new(parameters: &PbcParameters, support: &[usize]) -> Result<Self, PbcError> {
        if support.len() != parameters.support_weight {
            return Err(PbcError::SupportSize);
        }
        let mut coordinates = support.to_vec();
        coordinates.sort_unstable();
        if coordinates.iter().any(|&coordinate| coordinate >= parameters.universe_size) ||
            coordinates.windows(2).any(|window| window[0] == window[1])
        {
            return Err(PbcError::InvalidSupport);
        }
        Ok(Self { coordinates })
    }

    pub(crate) fn as_slice(&self) -> &[usize] {
        &self.coordinates
    }

    pub(crate) fn len(&self) -> usize {
        self.coordinates.len()
    }
}

/// Private bucket selections for one validated public layout.
pub struct PbcPrivateSchedule {
    pub(crate) layout_id: PbcLayoutId,
    pub(crate) selected_slots: Vec<usize>,
    pub(crate) assigned_coordinates: Vec<Option<usize>>,
    pub(crate) support_assignments: Vec<(usize, usize)>,
}

impl std::fmt::Debug for PbcPrivateSchedule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcPrivateSchedule")
            .field("layout_id", &self.layout_id)
            .field("bucket_count", &self.selected_slots.len())
            .field("support_weight", &self.support_assignments.len())
            .finish()
    }
}

impl PbcPrivateSchedule {
    /// Returns the public layout identity this private schedule belongs to.
    pub fn layout_id(&self) -> PbcLayoutId {
        self.layout_id
    }

    /// Validates dimensions, identity, selected cells, and one-to-one support assignment.
    pub fn validate(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        let bucket_count = layout.parameters.bucket_count;
        if self.selected_slots.len() != bucket_count ||
            self.assigned_coordinates.len() != bucket_count
        {
            return Err(PbcError::InvalidSchedule("bucket count mismatch".into()));
        }
        if self.support_assignments.len() != layout.parameters.support_weight {
            return Err(PbcError::InvalidSchedule("support assignment count mismatch".into()));
        }
        let mut assignments = Vec::with_capacity(self.support_assignments.len());
        for (bucket, (&slot, assigned)) in
            self.selected_slots.iter().zip(&self.assigned_coordinates).enumerate()
        {
            if slot >= layout.bucket_width {
                return Err(PbcError::InvalidSchedule("selected slot is out of range".into()));
            }
            let cell = &layout.cells[bucket][slot];
            match (cell, assigned) {
                (PbcCell::Real { coordinate, .. }, Some(assigned)) if coordinate == assigned => {
                    assignments.push((*assigned, bucket));
                }
                (PbcCell::Dummy, None) => {}
                (PbcCell::Padding, _) => {
                    return Err(PbcError::InvalidSchedule("padding cell was selected".into()));
                }
                _ => {
                    return Err(PbcError::InvalidSchedule(
                        "assigned coordinate does not match selected cell".into(),
                    ));
                }
            }
        }
        assignments.sort_unstable();
        if assignments != self.support_assignments {
            return Err(PbcError::InvalidSchedule("support assignments are not canonical".into()));
        }
        if self.support_assignments.windows(2).any(|window| window[0].0 == window[1].0) {
            return Err(PbcError::InvalidSchedule("support coordinate is selected twice".into()));
        }
        Ok(())
    }

    /// Returns the private slot selected for `bucket`.
    ///
    /// Callers must have validated the schedule first. This accessor remains
    /// crate-private because exposing selected slots would reveal the sparse
    /// support assignment.
    pub(crate) fn selected_slot(&self, bucket: usize) -> usize {
        self.selected_slots[bucket]
    }
}

/// Creates a private schedule for `support` against an already-built layout.
///
/// This exact scheduler is crate-private by design.  A long-lived public
/// layout must not be paired by an external caller with an arbitrary support:
/// version 1 only supports the key-generation order in which the support is
/// fixed before the layout seed is sampled and retried.  The public
/// [`generate_key_layout`] entry point enforces that order and keeps the
/// resulting schedule inside the trusted key-generation path.  Crate tests
/// may call this helper to compare the deterministic matcher with exhaustive
/// matching on toy instances.  Its validated-support argument is intentionally
/// not constructible outside this crate, so it cannot be used to schedule an
/// arbitrary raw support against a long-lived layout.
pub(crate) fn schedule(
    layout: &PbcPublicLayout,
    support: &ValidatedSupport,
) -> Result<PbcPrivateSchedule, PbcError> {
    layout.validate()?;
    let owners = deterministic_matching(layout, support)?;
    schedule_from_owners(layout, support, owners)
}

/// Materializes dummy selections and validates a matching after the matching
/// algorithm has completed. This split lets diagnostics report the exact
/// augmenting-path matcher time without charging schedule construction and
/// invariant checks to that metric.
pub(crate) fn schedule_from_owners(
    layout: &PbcPublicLayout,
    support: &ValidatedSupport,
    owners: Vec<Option<(usize, usize)>>,
) -> Result<PbcPrivateSchedule, PbcError> {
    if owners.len() != layout.parameters.bucket_count {
        return Err(PbcError::InvalidSchedule("owner count mismatch".into()));
    }
    let mut selected_slots = Vec::with_capacity(layout.parameters.bucket_count);
    let mut assigned_coordinates = Vec::with_capacity(layout.parameters.bucket_count);
    for (bucket, owner) in owners.into_iter().enumerate() {
        match owner {
            Some((coordinate, replica)) => {
                if support.as_slice().binary_search(&coordinate).is_err() {
                    return Err(PbcError::InvalidSchedule(
                        "matching selected a coordinate outside the support".into(),
                    ));
                }
                selected_slots.push(layout.locations[coordinate][replica].slot);
                assigned_coordinates.push(Some(coordinate));
            }
            None => {
                let dummy_slot = layout.cells[bucket]
                    .iter()
                    .position(|cell| matches!(cell, PbcCell::Dummy))
                    .ok_or_else(|| PbcError::InvalidLayout("missing dummy".into()))?;
                selected_slots.push(dummy_slot);
                assigned_coordinates.push(None);
            }
        }
    }
    let mut support_assignments = assigned_coordinates
        .iter()
        .enumerate()
        .filter_map(|(bucket, coordinate)| coordinate.map(|coordinate| (coordinate, bucket)))
        .collect::<Vec<_>>();
    support_assignments.sort_unstable();
    let schedule = PbcPrivateSchedule {
        layout_id: layout.layout_id,
        selected_slots,
        assigned_coordinates,
        support_assignments,
    };
    schedule.validate(layout)?;
    Ok(schedule)
}

/// Public layout paired with its private schedule during key generation.
pub struct PbcGeneratedKeyLayout {
    /// Public metadata safe to serialize and share.
    pub public_layout: PbcPublicLayout,
    private_schedule: PbcPrivateSchedule,
}

impl std::fmt::Debug for PbcGeneratedKeyLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PbcGeneratedKeyLayout")
            .field("public_layout", &self.public_layout)
            .field("private_schedule", &self.private_schedule)
            .finish()
    }
}

impl PbcGeneratedKeyLayout {
    /// Borrows the private schedule without exposing its fields for serialization.
    pub fn private_schedule(&self) -> &PbcPrivateSchedule {
        &self.private_schedule
    }
}

/// Retries deterministic seeds until layout width and support matching succeed.
pub fn generate_key_layout(
    parameters: &super::PbcParameters,
    root_seed: super::PbcRootSeed,
    sparse_support: &[usize],
) -> Result<PbcGeneratedKeyLayout, PbcError> {
    parameters.validate()?;
    let support = ValidatedSupport::new(parameters, sparse_support)?;
    let mut bucket_width_failures = 0;
    let mut no_perfect_schedule_failures = 0;
    let mut last_public_cause = None;
    for attempt in 0..parameters.max_seed_attempts {
        let seed = super::derive_attempt_seed(root_seed, attempt);
        let layout = match PbcPublicLayout::build(parameters, seed, attempt) {
            Ok(layout) => layout,
            Err(PbcError::BucketWidthExceeded) => {
                bucket_width_failures += 1;
                last_public_cause = Some(super::PbcRetryCause::BucketWidthExceeded);
                continue;
            }
            Err(error) => return Err(error),
        };
        match schedule(&layout, &support) {
            Ok(private_schedule) => {
                return Ok(PbcGeneratedKeyLayout { public_layout: layout, private_schedule });
            }
            Err(PbcError::NoPerfectSchedule) => {
                no_perfect_schedule_failures += 1;
                last_public_cause = Some(super::PbcRetryCause::NoPerfectSchedule);
                continue;
            }
            Err(error) => return Err(error),
        }
    }
    Err(PbcError::SeedAttemptsExhausted(super::PbcRetryDiagnostics {
        attempts: parameters.max_seed_attempts,
        bucket_width_failures,
        no_perfect_schedule_failures,
        last_public_cause,
    }))
}
