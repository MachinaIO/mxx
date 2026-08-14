//! Deterministic operational-noise simulation for closed protocol graphs.
//!
//! This module lowers a frozen Graph IR program into a compact egg expression,
//! applies only checked relations, and evaluates the extracted expression in
//! Rust.  It is intentionally separate from the legacy Lean runner while the
//! migration is in progress.

pub mod analysis;
pub mod bound;
pub mod error;
pub mod extract;
pub mod family;
pub mod identity;
pub mod language;
pub mod lower;
pub mod relation;
pub mod simulation;

use num_bigint::{BigInt, BigUint};

pub use error::OperationalSimulationError;
pub use simulation::check_operational_noise_candidate;

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
    pub egraph_node_count: u64,
    pub egraph_class_count: u64,
    pub rewrite_iteration_count: u64,
    pub relation_candidate_count: u64,
    pub relation_rewrite_count: u64,
    pub final_term_count: u64,
    pub lowering_milliseconds: u64,
    pub rewrite_milliseconds: u64,
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
