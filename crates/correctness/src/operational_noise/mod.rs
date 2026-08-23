//! Deterministic operational-noise simulation for closed protocol graphs.
//!
//! This module validates and analyzes frozen Graph IR through the production arenas.

// The production adapter consumes real plans directly. Legacy identity, family, scalar, and
// egg-era normal-form authorities are intentionally not compiled.
pub(crate) mod arena;
pub(crate) mod facts;
pub(crate) mod job;
pub(crate) mod lower;
pub(crate) mod monomial;
pub(crate) mod normal_form;
pub(crate) mod program;
pub(crate) mod protocol;
pub(crate) mod relation;
pub(crate) mod replay;
pub(crate) mod report;

pub mod bound;
pub mod error;
pub mod simulation;

use std::collections::BTreeSet;

use num_bigint::{BigInt, BigUint};

pub use error::{
    OperationalSimulationError, ProductionArenaContext, ProductionError, ProductionMatrixType,
    ProductionPhase, ProductionRootRole, ProductionValueType, RequestError,
};
pub use simulation::{
    BASE_FEASIBILITY_SCHEMA_ID, BASE_FEASIBILITY_SCHEMA_VERSION, BaseFeasibilityCounters,
    BaseFeasibilitySummary, BaseNBreakdown, OrdinaryBaselineCounters, ProgressEvent,
    ProgressEventKind, ResidualTraceCounters, check_operational_noise_candidate,
    check_operational_noise_candidate_with_progress, prepare_base_feasibility_summary,
    serialize_base_feasibility_summary,
};

/// A concrete value supplied for one named protocol parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationalParameterValue {
    Integer(BigInt),
    Rational { numerator: BigInt, denominator: BigInt },
}

/// The concrete gadget layout used by one closed parameter environment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalGadgetLayout {
    pub params_id: String,
    pub ring_dimension: usize,
    pub crt_moduli: Vec<u64>,
    pub crt_bits: usize,
    pub base_bits: usize,
    pub base: BigInt,
    pub regular_digit_count: usize,
    pub small_digit_count: usize,
    pub smallest_crt_modulus: u64,
}

/// Selects one closed decoder target under a fully specified parameter environment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalCheckRequest {
    pub environment: Vec<(String, OperationalParameterValue)>,
    pub layouts: Vec<OperationalGadgetLayout>,
    pub target_id: String,
}

impl OperationalCheckRequest {
    /// Validates request-owned values before a graph stage can observe them.
    /// The checker has no rational transfer domain, so rationals are rejected
    /// rather than omitted while constructing integer environments.
    pub(crate) fn validate(
        &self,
        parameter_names: impl IntoIterator<Item = String>,
    ) -> Result<(), RequestError> {
        let expected = parameter_names.into_iter().collect::<BTreeSet<_>>();
        let mut supplied = BTreeSet::new();
        for (name, value) in &self.environment {
            if name.is_empty() {
                return Err(RequestError::EmptyParameterName);
            }
            if !supplied.insert(name.clone()) {
                return Err(RequestError::DuplicateParameter { name: name.clone() });
            }
            if matches!(value, OperationalParameterValue::Rational { .. }) {
                return Err(RequestError::RationalParameter { name: name.clone() });
            }
            if !expected.contains(name) {
                return Err(RequestError::UnexpectedParameter { name: name.clone() });
            }
        }
        if let Some(name) = expected.into_iter().find(|name| !supplied.contains(name)) {
            return Err(RequestError::MissingParameter { name });
        }

        let mut layouts = BTreeSet::new();
        let mut layout_rings = BTreeSet::new();
        for layout in &self.layouts {
            if layout.params_id.is_empty() {
                return Err(RequestError::EmptyLayoutId);
            }
            if !layouts.insert(layout.params_id.clone()) {
                return Err(RequestError::DuplicateLayout { params_id: layout.params_id.clone() });
            }
            let expected_base = BigInt::from(1_u8) << layout.base_bits;
            let modulus = layout
                .crt_moduli
                .iter()
                .fold(BigUint::from(1_u8), |product, modulus| product * modulus);
            let expected_small_digits =
                (layout.base_bits != 0).then(|| layout.crt_bits.div_ceil(layout.base_bits));
            let valid = layout.ring_dimension != 0 &&
                !layout.crt_moduli.is_empty() &&
                layout.crt_moduli.iter().all(|modulus| *modulus != 0) &&
                layout.crt_moduli.iter().copied().collect::<BTreeSet<_>>().len() ==
                    layout.crt_moduli.len() &&
                layout.crt_moduli.iter().copied().min() == Some(layout.smallest_crt_modulus) &&
                layout.base_bits != 0 &&
                layout.crt_bits != 0 &&
                layout.base == expected_base &&
                expected_small_digits == Some(layout.small_digit_count) &&
                layout.regular_digit_count == layout.small_digit_count * layout.crt_moduli.len();
            if !valid {
                return Err(RequestError::InvalidLayout { params_id: layout.params_id.clone() });
            }
            if !layout_rings.insert((layout.ring_dimension, modulus.clone())) {
                return Err(RequestError::DuplicateLayoutRing {
                    ring_dimension: layout.ring_dimension,
                    modulus,
                });
            }
        }
        Ok(())
    }
}

/// Exact result of the decoder-specific acceptance rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationalAcceptanceReport {
    Threshold {
        plaintext_modulus: BigUint,
        threshold_left: BigUint,
        margin: BigInt,
    },
    BooleanInterval {
        quarter: BigInt,
        false_lower_margin: BigInt,
        false_upper_margin: BigInt,
        true_lower_margin: BigInt,
        true_upper_margin: BigInt,
    },
}

/// Quantitative work and timing observations for one simulation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OperationalSimulationDiagnostics {
    pub lowered_term_count: u64,
    /// Number of expression-DAG nodes actually visited by normalization.
    pub normalization_node_count: u64,
    /// Number of expression-DAG nodes reachable from the normalization root(s).
    pub normalization_node_total: u64,
    /// Number of exact polynomial terms retained at the normalization root(s).
    pub normalization_exact_term_count: u64,
    /// Number of relation-boundary candidates inspected by normalization.
    pub normalization_relation_count: u64,
    /// Number of relation-boundary candidates rewritten by normalization.
    pub normalization_relation_applied: u64,
    /// Number of relation-boundary candidates still present at the root.
    pub normalization_relation_remaining: u64,
    /// Number of bounded-only folds performed during normalization.
    pub normalization_bounded_fold_count: u64,
    pub normalization_milliseconds: u64,
    pub final_term_count: u64,
    pub lowering_milliseconds: u64,
    pub bound_milliseconds: u64,
    pub total_milliseconds: u64,
}

/// Deterministic simulation result.  An unsupported analysis is an error;
/// `accepted = false` only means that a successfully evaluated bound misses
/// the selected decoder's acceptance condition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalSimulationReport {
    pub target_id: String,
    pub noise_bound: BigUint,
    pub ciphertext_modulus: BigUint,
    pub accepted: bool,
    pub acceptance: OperationalAcceptanceReport,
    pub diagnostics: OperationalSimulationDiagnostics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_validation_rejects_rational_and_duplicate_layout_inputs() {
        let rational = OperationalCheckRequest {
            environment: vec![(
                "cutoff".to_owned(),
                OperationalParameterValue::Rational { numerator: 1.into(), denominator: 2.into() },
            )],
            layouts: Vec::new(),
            target_id: "target".to_owned(),
        };
        assert_eq!(
            rational.validate(["cutoff".to_owned()]),
            Err(RequestError::RationalParameter { name: "cutoff".to_owned() })
        );

        let layout = OperationalGadgetLayout {
            params_id: "layout".to_owned(),
            ring_dimension: 8,
            crt_moduli: vec![17],
            crt_bits: 5,
            base_bits: 2,
            base: 4.into(),
            regular_digit_count: 3,
            small_digit_count: 3,
            smallest_crt_modulus: 17,
        };
        let duplicate = OperationalCheckRequest {
            environment: Vec::new(),
            layouts: vec![layout.clone(), layout],
            target_id: "target".to_owned(),
        };
        assert_eq!(
            duplicate.validate(std::iter::empty()),
            Err(RequestError::DuplicateLayout { params_id: "layout".to_owned() })
        );
    }
}
