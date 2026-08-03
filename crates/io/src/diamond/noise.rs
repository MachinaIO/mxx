//! Noise simulation for the manifest-linked Diamond iO graphs.

use super::{DiamondIoCompileError, DiamondIoCompiler, DiamondIoFunction, DiamondIoPoly};
use crate::linked_noise::simulate_linked_graphs;
use mxx_ir_core::artifact::ProductionId;
use mxx_noise_simulator::{DecodeNoiseReport, NoiseReport};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondIoNoiseSimulation {
    pub production: ProductionId,
    pub report: NoiseReport,
    pub final_decodes: Vec<DecodeNoiseReport>,
    pub within_threshold: bool,
}

#[derive(Debug, Error)]
pub enum DiamondIoNoiseError {
    #[error(transparent)]
    Compile(#[from] DiamondIoCompileError),
    #[error("Diamond iO linked noise simulation failed: {0}")]
    Linked(String),
    #[error("Diamond iO evaluation graph has no final decode target")]
    MissingFinalDecode,
}

/// Simulates the exact preprocessing/evaluation graph pair used by runtime.
///
/// Runtime input digits are intentionally left unspecified. The symbolic
/// simulator therefore takes the maximum over every reachable selection
/// branch instead of estimating a hand-written representative path.
pub fn simulate_diamond_io_noise<P: DiamondIoPoly + 'static>(
    compiler: &DiamondIoCompiler<P>,
    function: &DiamondIoFunction,
) -> Result<DiamondIoNoiseSimulation, DiamondIoNoiseError> {
    let preprocessing = compiler.build_preprocessing(function)?.graph;
    let linked = simulate_linked_graphs(
        &preprocessing,
        |production| {
            compiler
                .build_evaluation(function, production)
                .map(|evaluation| evaluation.graph)
                .map_err(|error| error.to_string())
        },
        &BTreeMap::new(),
    )
    .map_err(|error| DiamondIoNoiseError::Linked(error.to_string()))?;
    let final_decodes = linked.report.decode_targets.clone();
    if final_decodes.is_empty() {
        return Err(DiamondIoNoiseError::MissingFinalDecode);
    }
    let within_threshold = final_decodes.iter().all(|decode| decode.within_threshold);
    Ok(DiamondIoNoiseSimulation {
        production: linked.production,
        report: linked.report,
        final_decodes,
        within_threshold,
    })
}
