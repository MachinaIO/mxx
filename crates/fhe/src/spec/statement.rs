//! The statements a scheme claims and the reductions a security proof supplies.

use mxx_ir_core::{Graph, security::AssumptionParams};

/// The correctness a scheme claims for the circuits of its depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Correctness {
    /// Every execution decrypts correctly.
    WorstCase,
    /// Decryption fails with probability at most `2^-failure_probability_log2`.
    AverageCase { failure_probability_log2: u32 },
}

/// The statements claimed for a scheme: the correctness of a circuit, and IND-CPA security
/// relative to allowed assumptions with reductions of bounded cost.
#[derive(Clone, Debug)]
pub struct Statement {
    pub correctness: Correctness,
    /// The security statement, when one is claimed.
    pub security: Option<SecurityStatement>,
}

/// IND-CPA security relative to allowed assumptions with reductions of bounded cost.
#[derive(Clone, Debug)]
pub struct SecurityStatement {
    /// The assumptions the statement allows, in the order of its game hops.
    pub assumptions: Vec<AssumptionParams>,
    /// A bound on the cost of each reduction's simulator beyond running the adversary, in the
    /// units of `MxxSecurity.IR.Program.cost`.
    pub cost_bound: u128,
}

/// The simulator of the reduction to one assumption of a [`Statement`], as the evidence a proof
/// of its security uses; no statement names it.
#[derive(Clone, Debug)]
pub struct Reduction {
    pub setup: Graph,
    pub challenge: Graph,
}
