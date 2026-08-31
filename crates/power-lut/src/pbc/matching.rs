//! Deterministic support-to-bucket matching for PBC schedules.
//!
//! The augmenting-path matcher sees the public candidate graph and a private
//! support list. Its output is kept in the private schedule; no matching
//! choice is included in the public layout identity or compiler artifact names.

use super::{PbcError, PbcPublicLayout, schedule::ValidatedSupport};

pub(crate) fn deterministic_matching(
    layout: &PbcPublicLayout,
    support: &ValidatedSupport,
) -> Result<Vec<Option<(usize, usize)>>, PbcError> {
    let mut owner = vec![None; layout.parameters.bucket_count];
    for &coordinate in support.as_slice() {
        let mut seen = vec![false; layout.parameters.bucket_count];
        if !augment(layout, coordinate, &mut owner, &mut seen, 0, support.len()) {
            return Err(PbcError::NoPerfectSchedule);
        }
    }
    Ok(owner)
}

/// Tries to place `coordinate` by recursively displacing an existing owner.
///
/// Candidate buckets are visited in public hash order. The owner map is
/// private schedule material, so this helper never contributes to a public
/// layout identity or artifact name.
fn augment(
    layout: &PbcPublicLayout,
    coordinate: usize,
    owner: &mut [Option<(usize, usize)>],
    seen_bucket: &mut [bool],
    depth: usize,
    support_weight: usize,
) -> bool {
    if depth >= support_weight {
        return false;
    }
    for (replica, &bucket) in layout.candidates[coordinate].iter().enumerate() {
        if seen_bucket[bucket] {
            continue;
        }
        seen_bucket[bucket] = true;
        let displaced = owner[bucket].map(|(coordinate, _)| coordinate);
        match displaced {
            None => {
                owner[bucket] = Some((coordinate, replica));
                return true;
            }
            Some(displaced)
                if augment(layout, displaced, owner, seen_bucket, depth + 1, support_weight) =>
            {
                owner[bucket] = Some((coordinate, replica));
                return true;
            }
            Some(_) => {}
        }
    }
    false
}
