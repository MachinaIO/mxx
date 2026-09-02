//! Diamond witness encryption expressed with the typed DSL.

mod artifacts;
mod config;
pub mod correctness;
mod estimate;
#[cfg(feature = "gpu")]
mod estimate_gpu;
mod graph;
mod parameter_search;
mod representation;
mod runtime;

pub use artifacts::DiamondArtifactNames;
pub use config::{
    DiamondConfigError, DiamondSamplerBoundError, DiamondWeConfig,
    default_error_max_coefficient_bound, default_preimage_max_coefficient_bound,
};
pub use estimate::{DiamondCostEstimate, DiamondEstimateError, estimate_diamond_cost};
#[cfg(feature = "gpu")]
pub use estimate_gpu::{DiamondGpuMeasurementBackend, DiamondGpuMeasurementError};
pub use graph::{
    DiamondCircuitSemanticRefs, DiamondCompileError, DiamondDecryptionGraph,
    DiamondDecryptionSemanticRefs, DiamondDecryptionSiteRefs, DiamondEncryptionGraph,
    DiamondEncryptionSemanticRefs, DiamondStructuralSiteRefs, DiamondWeCompiler,
    DiamondWeProtocolFamily,
};
pub use parameter_search::{
    DiamondParameterSearch, DiamondParameterSearchError, DiamondSelectedParameters,
};
pub use representation::{DcrtRuntimeRepresentation, DcrtRuntimeRepresentationError};
pub use runtime::{
    DiamondDecryptionResult, DiamondRuntimeError, DiamondWeCiphertext, DiamondWeRuntime,
};
