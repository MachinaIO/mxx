use super::*;

/// One topological layer for the extracted `ErrorNorm` simulator.
///
/// The simulator evaluates regular gates, direct sub-circuit calls, and summed sub-circuit calls
/// in separate phases within each layer. The structural layer planner now lives in
/// `src/circuit/poly_circuit/analysis.rs`; this alias keeps the simulator-side naming stable.
pub(super) type ErrorNormExecutionLayer = GroupedCallExecutionLayer;

impl PolyCircuit<DCRTPoly> {
    /// Build the grouped execution plan consumed by the `ErrorNorm` simulator.
    ///
    /// The underlying layering logic is generic to `PolyCircuit` structure, so `ErrorNorm`
    /// delegates to the shared planner in `poly_circuit` rather than maintaining its own copy.
    pub(super) fn error_norm_execution_layers(&self) -> Vec<ErrorNormExecutionLayer> {
        self.grouped_execution_layers()
    }
}
