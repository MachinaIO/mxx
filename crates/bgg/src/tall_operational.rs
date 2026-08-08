//! Deterministic parameter-search estimates for the packed Tall BGG+ nested-RNS circuit family.
//!
//! This module intentionally models only the integration circuit family.  It is not a general
//! noise analyzer and does not replace the Lean operational checker.  The descriptor makes the
//! Rust graph family explicit so that the checker can reject a candidate built for another
//! circuit/layout before considering its residual estimate.

use crate::BggSamplerLayout;
use mxx_gadgets::circuit::{PolyCircuit, PolyGateKind};
use mxx_primitives::poly::Poly;
use num_bigint::BigUint;
use num_traits::One;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Canonical structural data for the packed nested-RNS multiplication circuit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallNestedRnsDescriptor {
    pub multiplication_count: usize,
    pub q_moduli_depth: usize,
    pub coefficient_slots: usize,
    pub gate_kinds: Vec<PolyGateKind>,
    pub digest: [u8; 32],
}

impl TallNestedRnsDescriptor {
    /// Captures the exact ordered circuit gate sequence together with packed-lane dimensions.
    pub fn from_circuit<P: Poly>(
        circuit: &PolyCircuit<P>,
        multiplication_count: usize,
        q_moduli_depth: usize,
        coefficient_slots: usize,
    ) -> Self {
        let gate_kinds =
            circuit.gates_in_id_order().map(|(_, gate)| gate.gate_type.kind()).collect::<Vec<_>>();
        let mut hasher = Sha256::new();
        hasher.update(b"mxx:tall-nested-rns:v1");
        hasher.update((multiplication_count as u64).to_le_bytes());
        hasher.update((q_moduli_depth as u64).to_le_bytes());
        hasher.update((coefficient_slots as u64).to_le_bytes());
        hasher.update((gate_kinds.len() as u64).to_le_bytes());
        for kind in &gate_kinds {
            hasher.update(format!("{kind:?}").as_bytes());
            hasher.update([0]);
        }
        let digest = hasher.finalize().into();
        Self { multiplication_count, q_moduli_depth, coefficient_slots, gate_kinds, digest }
    }

    pub fn validate(&self) -> Result<(), TallOperationalError> {
        if self.q_moduli_depth == 0 || self.coefficient_slots == 0 || self.gate_kinds.is_empty() {
            return Err(TallOperationalError::InvalidDescriptor);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"mxx:tall-nested-rns:v1");
        hasher.update((self.multiplication_count as u64).to_le_bytes());
        hasher.update((self.q_moduli_depth as u64).to_le_bytes());
        hasher.update((self.coefficient_slots as u64).to_le_bytes());
        hasher.update((self.gate_kinds.len() as u64).to_le_bytes());
        for kind in &self.gate_kinds {
            hasher.update(format!("{kind:?}").as_bytes());
            hasher.update([0]);
        }
        if <[u8; 32]>::from(hasher.finalize()) != self.digest {
            return Err(TallOperationalError::DescriptorDigestMismatch);
        }
        Ok(())
    }
}

/// Exact integer sampler cutoffs and structural values consumed by the Tall operational estimate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallOperationalInputs {
    pub descriptor: TallNestedRnsDescriptor,
    pub layout: BggSamplerLayout,
    pub ring_dimension: usize,
    pub q_moduli: Vec<u64>,
    pub error_cutoff: BigUint,
    pub preimage_cutoff: BigUint,
}

/// A deterministic estimate used only to select a candidate for the runtime integration test.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TallOperationalEstimate {
    pub residual_bound: BigUint,
    pub threshold: BigUint,
}

impl TallOperationalEstimate {
    pub fn accepted(&self) -> bool {
        self.residual_bound < self.threshold
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum TallOperationalError {
    #[error("Tall nested-RNS descriptor is malformed")]
    InvalidDescriptor,
    #[error("Tall nested-RNS descriptor digest does not match its contents")]
    DescriptorDigestMismatch,
    #[error("Tall operational input has no CRT modulus")]
    EmptyCrtBasis,
    #[error("Tall operational input dimensions do not agree")]
    DimensionMismatch,
}

/// Computes the conservative recurrence used by the Tall Lean operational checker.
///
/// One multiplication may combine an existing residual with a freshly sampled/preimage-derived
/// Tall row.  The packed q lanes preserve the same logical matrix width, so their contribution is
/// a linear `ring_dimension * digit_count` factor rather than a factor of physical SIMD slots.
pub fn estimate_tall_nested_rns(
    input: &TallOperationalInputs,
) -> Result<TallOperationalEstimate, TallOperationalError> {
    input.descriptor.validate()?;
    if input.ring_dimension == 0 ||
        input.layout.digit_count == 0 ||
        input.descriptor.coefficient_slots != input.ring_dimension ||
        input.q_moduli.len() != input.descriptor.q_moduli_depth
    {
        return Err(TallOperationalError::DimensionMismatch);
    }
    let q_max = input.q_moduli.iter().copied().max().ok_or(TallOperationalError::EmptyCrtBasis)?;
    let modulus = input
        .q_moduli
        .iter()
        .fold(BigUint::one(), |product, modulus| product * BigUint::from(*modulus));
    let threshold = &modulus / (BigUint::from(2u8) * BigUint::from(q_max));
    let width = BigUint::from(input.ring_dimension) * BigUint::from(input.layout.digit_count);
    let fresh = &input.error_cutoff + &input.preimage_cutoff;
    let mut residual = fresh.clone();
    for _ in 0..input.descriptor.multiplication_count {
        // (e_left * signal_right) + (signal_left * e_right) + (e_left * e_right), with
        // coefficient convolution bounded by the logical ring/digit width.
        residual = (&residual * &width * 2u8) + (&fresh * &width) + (&residual * &fresh * &width);
    }
    Ok(TallOperationalEstimate { residual_bound: residual, threshold })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_detects_mutation() {
        let descriptor = TallNestedRnsDescriptor {
            multiplication_count: 1,
            q_moduli_depth: 2,
            coefficient_slots: 4,
            gate_kinds: vec![PolyGateKind::Input],
            digest: [0; 32],
        };
        assert_eq!(descriptor.validate(), Err(TallOperationalError::DescriptorDigestMismatch));
    }
}
