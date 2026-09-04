//! Exponent-LUT operations built on the generic BGG+ encoding layer.
//!
//! The crate has two deliberately separate views of the same computation. The
//! [`ExponentLutEncodingCompiler`] consumes plain BGG encoding wires and private
//! RHS packages, while [`ExponentLutPublicKeyCompiler`] emits the public matrix
//! projection of each operation. Artifact importers validate setup identity
//! before producing runtime wires; that validation does not add provenance to
//! the wires. The role-free RHS material stays in the generic core and
//! lower-level algebra remains in `mxx-bgg`.
//!
//! The [`noise_refresh`] module contains the sparse-LWR application, its
//! private-bucket layout, value readout, setup artifacts, CRT refresh path,
//! and application-specific noise simulation. Native PRF evaluation is
//! represented by typed accumulation and readout artifacts; no legacy
//! reduction stage is exposed.

pub mod encoding;
pub mod layout;
pub mod noise;
pub mod noise_refresh;
pub mod program;
pub mod public_key;
pub mod rhs;
use rhs::ExponentRhsPackageError;

pub use encoding::{ExponentArtifactImportError, ExponentLutEncodingCompiler};
pub use layout::{ExpLayout, LayoutError};
pub use mxx_bgg::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout,
    EncodingCompileError,
};
pub use mxx_primitives::sampler::bounds::TrapdoorPreimageCutoffPolicy;
pub use noise::{AverageVariance, ProgramNoiseReport, ProgramNoiseState};
pub use noise_refresh::{
    artifacts::ExponentLutStorageReport,
    prf::{
        NativePrfEncodingBatchOutputs, NativePrfOnlineBatchInputs, NativePrfPublicInputs,
        NativePrfPublicReadoutPlan, SparseLwrPrfProfile, SparseLwrPrfProgram,
    },
    readout::{
        NormalizationPreimageRequirement, NormalizationPreimageRole, NormalizationTargetDescriptor,
        ReadoutPreimageSchema, ValueReadout, ValueReadoutError, ValueReadoutHelper,
        ValueReadoutIdentity, ValueReadoutPublicHelper, ValueReadoutTracePlan,
    },
    simulation::{
        AVERAGE_CASE_REPORT_SCHEMA_VERSION, AverageCaseConfig, ExponentLutAverageNoiseReport,
        ExponentLutNoiseReport, ExponentLutNoiseSnapshot,
    },
};
pub use program::{
    AttributeKnowledge, FamilyAggregation, FamilyProgram, FamilyRhsSource, FamilyWireSource, Gate,
    ProgramFamilyId, ProgramStep, ProgramWireType, RhsId, WireFamilyId, WireId,
};
pub use public_key::{
    ExponentLutPublicKeyCompiler, ProgramCacheBindings, ProgramCacheDeclaration,
    ProgramCacheLocation, ProgramCacheRole,
};
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
/// Errors raised while validating or lowering an Exponent-LUT operation.
pub enum ExponentLutError {
    #[error(transparent)]
    /// A generic BGG encoding operation rejected its inputs.
    Bgg(#[from] EncodingCompileError),
    #[error(transparent)]
    /// An RHS package failed material or artifact validation.
    Rhs(#[from] ExponentRhsPackageError),
    #[error(transparent)]
    /// An imported artifact manifest is not compatible with the requested value.
    Artifact(#[from] ExponentArtifactImportError),
    #[error(transparent)]
    /// A shared Exponent-LUT program failed validation or lowering.
    Program(#[from] program::ProgramValidationError),
    #[error("LUT dimensions, output exponents, or RHS width are invalid")]
    /// A table, exponent, or matrix shape is inconsistent with the operation.
    InvalidLut,
    #[error("sparse-LWR block layout or package count is invalid")]
    /// A sparse-LWR block has no valid public/package pairing.
    InvalidSparseLwrBlock,
    #[error("program operation {gate} failed: {detail}")]
    /// A backend rejected one identified primitive operation while lowering.
    ProgramOperation { gate: String, detail: String },
    #[error("online PRF stage `{stage}` failed: {detail}")]
    /// Structural online batch construction failed before a graph was emitted.
    OnlinePrfStage { stage: &'static str, detail: String },
}

impl From<program::ProgramLowerError<ExponentLutError>> for ExponentLutError {
    fn from(error: program::ProgramLowerError<ExponentLutError>) -> Self {
        match error {
            program::ProgramLowerError::Validation(error) => Self::Program(error),
            program::ProgramLowerError::Operation { gate, source } => {
                Self::ProgramOperation { gate: format!("{gate:?}"), detail: source.to_string() }
            }
        }
    }
}
