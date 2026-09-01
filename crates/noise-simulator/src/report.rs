//! Stable simulation result and diagnostic summaries.

use crate::{DiagnosticSite, SimulationRoot, SourceId};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootNoiseReport {
    pub root: SimulationRoot,
    pub maximum_absolute_coefficient_error: BigUint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DroppedCarrierDiagnostic {
    /// The exact graph occurrence where the source witness stopped being
    /// representable by the output state.
    pub site: DiagnosticSite,
    /// Stable, operation-level explanation independent of application names.
    pub reason: String,
    /// Source identity that was present on an input but absent from the
    /// output.  `actual_source` is the output source, if one was retained.
    pub expected_source: Option<SourceId>,
    pub actual_source: Option<SourceId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimulationDiagnostics {
    pub planned_wires: usize,
    pub transfer_steps: u64,
    pub dropped_carriers: Vec<DroppedCarrierDiagnostic>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimulationReport {
    pub roots: Vec<RootNoiseReport>,
    pub diagnostics: SimulationDiagnostics,
}

impl SimulationReport {
    pub fn new(roots: Vec<RootNoiseReport>, diagnostics: SimulationDiagnostics) -> Self {
        Self { roots, diagnostics }
    }
}
