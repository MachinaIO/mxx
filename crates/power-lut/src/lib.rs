//! Power-LUT operations built on the generic BGG+ encoding layer.
//!
//! The crate has two deliberately separate views of the same computation. The
//! [`PowerLutEncodingCompiler`] consumes plain BGG encoding wires and private
//! RHS packages, while [`PowerLutPublicKeyCompiler`] emits the public matrix
//! projection of each operation. Artifact importers validate setup identity
//! before producing runtime wires; that validation does not add provenance to
//! the wires. The role-free RHS material stays in the generic core and
//! lower-level algebra remains in `mxx-bgg`.
//!
//! The [`prf`] module is the sparse-LWR application and uses the sibling
//! [`pbc`] module for private-bucket layout. [`refresh`] is the public
//! Section 7 CRT refresh path. The small sparse-LWR helpers reimplement only a
//! clear reference calculation and its rounding table for tests and host-side
//! checks; they do not construct an encrypted graph.

pub mod encoding;
pub mod noise;
pub mod pbc;
pub mod prf;
pub mod program;
pub mod public_key;
pub mod refresh;
pub mod refresh_setup;
pub mod rhs;
use rhs::PowerRhsPackageError;

pub use encoding::{PowerLutEncodingCompiler, flattened_lut_index};

pub use encoding::PowerArtifactImportError;
pub use mxx_bgg::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, BggSamplerLayout,
    EncodingCompileError, PreimageCoefficientBound,
};
pub use noise::{
    AVERAGE_CASE_REPORT_SCHEMA_VERSION, AcceptedUnder, AffineNoiseTransfer, AverageCaseConfig,
    AverageEventBudget, AverageGateNoiseStep, AverageNoiseTransfer, AverageProgramNoiseReport,
    AverageRefreshNoiseReport, AverageRefreshSlotNoiseReport, AverageSparsePrfNoiseReport,
    AverageVariance, GateNoiseStep, HeuristicId, NoiseMagnitude, NoiseModelKind,
    NoiseSimulationError, PowerLutAverageNoiseReport, PowerLutNoiseParameters, PowerLutNoiseReport,
    PowerLutNoiseSnapshot, ProgramNoiseInputs, ProgramNoiseReport, RefreshHardAuthority,
    RefreshNoiseParameters, RefreshNoiseReport, RefreshSlotNoiseParameters, RefreshSlotNoiseReport,
    SparsePrfNoiseReport, average_fixed_fuse_transfer, average_fixed_lut_transfer,
    average_monomial_one_hot_transfer, average_refresh_accepts, average_two_input_lut_transfer,
    average_variance_transfer, fixed_fuse_transfer, fixed_lut_transfer, monomial_one_hot_transfer,
    simulate_average_program, simulate_average_refresh, simulate_average_sparse_prf,
    simulate_program, simulate_refresh, simulate_sparse_prf, two_input_lut_transfer,
};
pub use prf::{
    SparseLwrPrfHelperBundle, SparseLwrPrfProfile, SparseLwrPrfProgram,
    SparseLwrPrfPublicHelperBundle, SparseLwrPublicReductionHelpers,
    SparseLwrPublicTerminalHelpers, SparseLwrReductionHelpers, SparseLwrReductionPlan,
    SparseLwrTerminalHelpers,
};
pub use public_key::PowerLutPublicKeyCompiler;
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
/// Errors raised while validating or lowering a Power-LUT operation.
pub enum PowerLutError {
    #[error("Power-LUT encoding carries forbidden plaintext metadata")]
    /// Power-LUT boundaries accept ciphertext-only encoding wires.
    PlaintextMetadataForbidden,
    #[error(transparent)]
    /// A generic BGG encoding operation rejected its inputs.
    Bgg(#[from] EncodingCompileError),
    #[error(transparent)]
    /// An RHS package failed material or artifact validation.
    Rhs(#[from] PowerRhsPackageError),
    #[error(transparent)]
    /// An imported artifact manifest is not compatible with the requested value.
    Artifact(#[from] PowerArtifactImportError),
    #[error(transparent)]
    /// A shared Power-LUT program failed validation or lowering.
    Program(#[from] program::ProgramValidationError),
    #[error("LUT dimensions, output exponents, or RHS width are invalid")]
    /// A table, exponent, or matrix shape is inconsistent with the operation.
    InvalidLut,
    #[error("sparse-LWR block layout or package count is invalid")]
    /// A sparse-LWR block has no valid public/package pairing.
    InvalidSparseLwrBlock,
    #[error("the final scalar LUT requires an explicitly supplied helper set")]
    /// Rounding helpers are setup artifacts and cannot be inferred from bucket helpers.
    MissingRoundingHelpers,
}

/// Rejects any encoding wire that exposes plaintext metadata at a Power-LUT
/// boundary.  The matrix relation itself remains plain `BggEncodingWire`; the
/// check is deliberately centralized so callers cannot forge metadata after
/// setup sampling or PRF evaluation.
pub(crate) fn ensure_ciphertext_only(encoding: &BggEncodingWire) -> Result<(), PowerLutError> {
    if encoding.plaintext.is_some() || encoding.pubkey.reveal_plaintext {
        return Err(PowerLutError::PlaintextMetadataForbidden);
    }
    Ok(())
}

// Keep test-only modules at the end of the module so the production surface
// above is easy to scan. Their contents remain in dedicated source files.
#[cfg(all(test, feature = "gpu"))]
mod encoding_gpu_tests;

#[cfg(test)]
mod refresh_setup_gpu_tests;
