//! Independent clear-text PBC oracles.
//!
//! These functions provide a small, auditable reference for checking that a
//! private schedule selects exactly the support coordinates represented by the
//! public layout. They do not inspect encrypted artifacts or build DSL nodes.

use super::{PbcCell, PbcError, PbcPrivateSchedule, PbcPublicLayout};

/// Computes the scheduled public inner product modulo `modulus`.
///
/// A real selected cell contributes its original coordinate; a selected dummy
/// contributes zero. Selecting padding is malformed because padding exists
/// only to rectangularize storage and is never a valid schedule choice.
pub fn clear_pbc_inner_product(
    layout: &PbcPublicLayout,
    schedule: &PbcPrivateSchedule,
    public_vector: &[u64],
    modulus: u64,
) -> Result<u64, PbcError> {
    layout.validate()?;
    schedule.validate(layout)?;
    if public_vector.len() != layout.parameters.universe_size {
        return Err(PbcError::InvalidLayout("public vector dimension mismatch".into()));
    }
    if modulus == 0 {
        return Err(PbcError::InvalidParameters("modulus must be positive".into()));
    }
    let mut accumulator = 0u64;
    for bucket in 0..layout.parameters.bucket_count {
        let slot = schedule.selected_slot(bucket);
        let contribution = match layout.cells[bucket][slot] {
            PbcCell::Real { coordinate, .. } => public_vector[coordinate] % modulus,
            PbcCell::Dummy => 0,
            PbcCell::Padding => {
                return Err(PbcError::InvalidSchedule("padding cell was selected".into()));
            }
        };
        accumulator =
            ((u128::from(accumulator) + u128::from(contribution)) % u128::from(modulus)) as u64;
    }
    Ok(accumulator)
}

/// Returns support coordinates in canonical assignment order.
pub fn canonical_decode(
    layout: &PbcPublicLayout,
    schedule: &PbcPrivateSchedule,
) -> Result<Vec<usize>, PbcError> {
    layout.validate()?;
    schedule.validate(layout)?;
    Ok(schedule.support_assignments.iter().map(|(coordinate, _)| *coordinate).collect())
}

/// Converts a dense binary support vector into sorted coordinate indices.
pub fn dense_binary_support(universe_size: usize, dense: &[u8]) -> Result<Vec<usize>, PbcError> {
    if dense.len() != universe_size {
        return Err(PbcError::InvalidSupport);
    }
    if dense.iter().any(|&value| value > 1) {
        return Err(PbcError::InvalidSupport);
    }
    Ok(dense
        .iter()
        .enumerate()
        .filter_map(|(coordinate, &value)| (value == 1).then_some(coordinate))
        .collect())
}

/// Validates and converts a dense support representation for a PBC layout.
pub fn support_from_dense(
    parameters: &super::PbcParameters,
    dense: &[u8],
) -> Result<Vec<usize>, PbcError> {
    parameters.validate()?;
    let support = dense_binary_support(parameters.universe_size, dense)?;
    if support.len() != parameters.support_weight {
        return Err(PbcError::SupportSize);
    }
    Ok(support)
}
