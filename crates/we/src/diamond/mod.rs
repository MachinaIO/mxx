//! Diamond witness encryption expressed with the typed DSL.

mod artifacts;
mod config;
mod estimate;
mod graph;
mod noise;
mod parameter_search;
mod runtime;

pub use artifacts::DiamondArtifactNames;
pub use config::{DiamondConfigError, DiamondWeConfig};
pub use estimate::{DiamondCostEstimate, DiamondEstimateError, estimate_diamond_cost};
pub use graph::{
    DiamondCompileError, DiamondDecryptionGraph, DiamondEncryptionGraph, DiamondWeCompiler,
};
pub use noise::{
    DiamondDecodeNoiseReport, DiamondNoiseError, DiamondNoiseSimulation, simulate_diamond_noise,
};
pub use parameter_search::{
    DiamondParameterSearch, DiamondParameterSearchError, DiamondSelectedParameters,
};
pub use runtime::{DiamondRuntimeError, DiamondWeCiphertext, DiamondWeRuntime};
