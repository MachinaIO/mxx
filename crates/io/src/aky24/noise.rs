//! Noise simulation for the manifest-linked AKY24 iO cascade.

use super::cascade::{Aky24CascadeCompiler, Aky24CascadeGraphError};
use crate::linked_noise::simulate_linked_graphs;
use mxx_ir_core::artifact::ProductionId;
use mxx_noise_simulator::{DecodeNoiseReport, NoiseReport};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24IoNoiseSimulation {
    pub production: ProductionId,
    pub report: NoiseReport,
    /// Every intermediate cascade decode and the final output decode.
    pub decode_targets: Vec<DecodeNoiseReport>,
    pub within_threshold: bool,
}

#[derive(Debug, Error)]
pub enum Aky24IoNoiseError {
    #[error(transparent)]
    Compile(#[from] Aky24CascadeGraphError),
    #[error("AKY24 iO linked noise simulation failed: {0}")]
    Linked(String),
    #[error("AKY24 iO evaluation graph has no decode target")]
    MissingDecodeTarget,
}

/// Simulates the exact preprocessing/evaluation graph pair used for `input`.
///
/// The two ciphertexts for each input position have the same graph shape and
/// sampling metadata, so parameter search may use one canonical bit vector.
/// Callers can still pass any concrete vector here to audit its exact imported
/// atom identities and decode margins.
pub fn simulate_aky24_io_noise(
    compiler: &Aky24CascadeCompiler,
    input: &[bool],
) -> Result<Aky24IoNoiseSimulation, Aky24IoNoiseError> {
    let preprocessing = compiler.build_preprocessing()?.graph;
    let linked = simulate_linked_graphs(
        &preprocessing,
        |production| {
            compiler
                .build_evaluation(input, production)
                .map(|evaluation| evaluation.graph)
                .map_err(|error| error.to_string())
        },
        &BTreeMap::new(),
    )
    .map_err(|error| Aky24IoNoiseError::Linked(error.to_string()))?;
    let decode_targets = linked.report.decode_targets.clone();
    if decode_targets.is_empty() {
        return Err(Aky24IoNoiseError::MissingDecodeTarget);
    }
    let within_threshold = decode_targets.iter().all(|decode| decode.within_threshold);
    Ok(Aky24IoNoiseSimulation {
        production: linked.production,
        report: linked.report,
        decode_targets,
        within_threshold,
    })
}
