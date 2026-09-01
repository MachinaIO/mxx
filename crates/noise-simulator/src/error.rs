//! Typed simulator failures and occurrence-aware diagnostic locations.

use crate::StageId;
use mxx_ir_core::{NodeId, Port};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticSite {
    pub stage: Option<StageId>,
    pub occurrence: Vec<String>,
    pub node: Option<NodeId>,
    pub port: Option<Port>,
    pub operation: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error, Serialize, Deserialize)]
pub enum SimulationError {
    #[error("invalid or incomplete parameter environment: {message}")]
    InvalidParameterEnvironment { message: String },
    #[error("invalid graph, type, or shape: {message}")]
    InvalidGraph { message: String, site: Option<DiagnosticSite> },
    #[error("missing external input fact for {stage:?}:{input}")]
    MissingExternalInputFact { stage: StageId, input: String },
    #[error("conflicting external input fact for {stage:?}:{input}")]
    ConflictingExternalInputFact { stage: StageId, input: String },
    #[error("unknown stage {stage:?}")]
    UnknownStage { stage: StageId },
    #[error("unknown graph output {output:?} in stage {stage:?}")]
    UnknownOutput { stage: StageId, output: String },
    #[error("duplicate stage id {stage:?}")]
    DuplicateStage { stage: StageId },
    #[error("duplicate production id")]
    DuplicateProduction,
    #[error("duplicate simulation root")]
    DuplicateRoot,
    #[error("duplicate external input fact")]
    DuplicateExternalInput,
    #[error("artifact resolution failed: {message}")]
    ArtifactResolution { message: String, site: Option<DiagnosticSite> },
    #[error("unsupported IR operation {operation}")]
    Unsupported { operation: String, site: Option<DiagnosticSite> },
    #[error("selector range is outside its domain: {message}")]
    SelectorOutOfRange { message: String, site: Option<DiagnosticSite> },
    #[error("invalid deterministic index map: {message}")]
    InvalidIndexMap { message: String, site: Option<DiagnosticSite> },
    #[error("preimage relation error: {message}")]
    Relation { message: String, site: Option<DiagnosticSite> },
    #[error("shared-source relation depends on its branch axis")]
    BranchDependentSource { site: Option<DiagnosticSite> },
    #[error("resource limit exceeded: {message}")]
    ResourceLimitExceeded { message: String, site: Option<DiagnosticSite> },
}

impl From<crate::state::StateError> for SimulationError {
    fn from(error: crate::state::StateError) -> Self {
        Self::InvalidGraph { message: error.to_string(), site: None }
    }
}

impl From<crate::bound::BoundError> for SimulationError {
    fn from(error: crate::bound::BoundError) -> Self {
        Self::InvalidGraph { message: error.to_string(), site: None }
    }
}
