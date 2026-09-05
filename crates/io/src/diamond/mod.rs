//! Diamond iO expressed with the typed declarative DSL.

mod artifacts;
mod circuits;
mod config;
mod final_circuit;
mod graph;
#[cfg(feature = "gpu")]
mod graph_gpu;
mod parameter_search;
mod runtime;

pub use artifacts::{DiamondIoArtifactNameError, DiamondIoArtifactNames};
pub use circuits::{build_goldreich_suffix_circuit, goldreich_round_seed};
pub use config::{DiamondIoConfig, DiamondIoConfigError, DiamondIoFunction};
pub use graph::{
    DiamondIoCompileError, DiamondIoCompiler, DiamondIoEvaluationGraph, DiamondIoPoly,
    DiamondIoPreprocessingGraph, HASH_KEY_INPUT, NATIVE_SEED_INPUT_PREFIX, OUTPUT_PREFIX,
    PRIVATE_K_INPUT, PUBLIC_INPUT_DIGIT_PREFIX, output_name,
};
pub use parameter_search::{
    DiamondIoParameterSearch, DiamondIoParameterSearchError, DiamondIoSelectedParameters,
};
pub use runtime::{
    DiamondIoNativeSeedError, DiamondIoNativeSeedSetup, DiamondIoObfuscation, DiamondIoRuntime,
    DiamondIoRuntimeError, declare_native_seed_inputs, native_seed_bindings, sample_native_seed,
};
