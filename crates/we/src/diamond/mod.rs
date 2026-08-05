//! Diamond witness encryption expressed with the typed DSL.

mod artifacts;
mod config;
mod estimate;
mod graph;
mod parameter_search;
mod runtime;

pub use artifacts::DiamondArtifactNames;
pub use config::{
    DiamondConfigError, DiamondSamplerBoundError, DiamondWeConfig,
    default_error_max_coefficient_bound, default_preimage_max_coefficient_bound,
};
pub use estimate::{DiamondCostEstimate, DiamondEstimateError, estimate_diamond_cost};
pub use graph::{
    DIAMOND_PROTOCOL_SOURCE_PATHS, DiamondCompileError, DiamondDecryptionGraph,
    DiamondEncryptionGraph, DiamondWeCompiler, DiamondWeProtocolFamily,
};
pub use parameter_search::{
    DiamondParameterSearch, DiamondParameterSearchError, DiamondSelectedParameters,
};
pub use runtime::{DiamondRuntimeError, DiamondWeCiphertext, DiamondWeRuntime};
