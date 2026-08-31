//! Private-bucket cuckoo (PBC) support for sparse-LWR evaluation.
//!
//! PBC separates data that may be public from the secret sparse support. A
//! [`crate::pbc::PbcPublicLayout`] deterministically hashes every universe
//! coordinate into candidate buckets and rectangularizes those buckets with dummy and padding
//! cells. A [`crate::pbc::PbcPrivateSchedule`] then records which slot is
//! selected for each bucket to realize the secret support assignment. The
//! compiler consumes the public layout and
//! selector artifacts, while the private schedule is never serialized
//! or used to name public families.
//!
//! The module also exposes clear oracles and diagnostics. They are reference
//! and setup tools, not a replacement for the encrypted DSL graph.

mod artifacts;
pub mod diagnostics;
mod evaluation;
mod layout;
mod schedule;

use thiserror::Error;

/// Semantic version of the serialized public layout schema.
pub const PBC_LAYOUT_SEMANTIC_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Root seed from which deterministic layout-attempt seeds are derived.
pub struct PbcRootSeed(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Seed used for candidate-bucket hashing in one layout attempt.
pub struct PbcLayoutSeed(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Public location of a coordinate replica in the rectangular layout.
pub struct PbcLocation {
    /// Bucket row containing the cell.
    pub bucket: usize,
    /// Slot within that bucket row.
    pub slot: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// A public cell used by the bucket compiler.
pub enum PbcCell {
    /// A universe coordinate and its hash replica.
    Real {
        /// Coordinate in the original public universe.
        coordinate: usize,
        /// Candidate replica number for that coordinate.
        replica: usize,
    },
    /// A harmless public zero cell used when a bucket has no selected support.
    Dummy,
    /// Storage-only cell added to make every bucket the same width.
    Padding,
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
/// Digest of the complete public PBC layout and its parameters.
pub struct PbcLayoutId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Public reason why a layout seed attempt was rejected.
pub enum PbcRetryCause {
    /// A bucket exceeded the configured rectangular-width limit.
    BucketWidthExceeded,
    /// The secret support could not be assigned to distinct candidate buckets.
    NoPerfectSchedule,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Aggregate diagnostics for failed or accepted public layout attempts.
pub struct PbcRetryDiagnostics {
    /// Number of attempts performed.
    pub attempts: u32,
    /// Attempts rejected for excessive bucket width.
    pub bucket_width_failures: u32,
    /// Attempts rejected because matching found no schedule.
    pub no_perfect_schedule_failures: u32,
    /// Last failure cause, if no attempt was accepted.
    pub last_public_cause: Option<PbcRetryCause>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
/// Errors from PBC parameter, layout, schedule, or artifact validation.
pub enum PbcError {
    #[error("invalid PBC parameters: {0}")]
    /// Parameter validation failed.
    InvalidParameters(String),
    #[error("sparse support has the wrong size")]
    /// Support cardinality differs from the configured weight.
    SupportSize,
    #[error("sparse support contains a duplicate or out-of-range coordinate")]
    /// Support contains an invalid coordinate or duplicate.
    InvalidSupport,
    #[error("candidate derivation nonce overflow")]
    /// Hash nonce space was exhausted.
    HashNonceOverflow,
    #[error("derived bucket width exceeds the configured limit")]
    /// Rectangularization would exceed the configured width limit.
    BucketWidthExceeded,
    #[error("the cuckoo graph has no schedule covering the support")]
    /// No distinct bucket assignment covers the support.
    NoPerfectSchedule,
    #[error("all PBC seed attempts failed: {0:?}")]
    /// Retry budget was exhausted.
    SeedAttemptsExhausted(PbcRetryDiagnostics),
    #[error("public PBC layout is malformed: {0}")]
    /// Public layout structure or identity is invalid.
    InvalidLayout(String),
    #[error("private PBC schedule is malformed: {0}")]
    /// Private schedule does not match its public layout.
    InvalidSchedule(String),
    #[error("PBC layout identity mismatch")]
    /// Values were paired with a different layout digest.
    LayoutIdentityMismatch,
    #[error("artifact layout or key identity mismatch")]
    /// Artifact names or metadata do not match the requested setup.
    ArtifactIdentityMismatch,
    #[error("integer conversion or size overflow")]
    /// A host-size conversion or arithmetic bound overflowed.
    SizeOverflow,
}

pub use artifacts::{
    PbcSelectorArtifactNames, PbcSelectorArtifacts, PbcSelectorPackageArtifactNames,
    PbcStructuralSelectorFamilies, PbcTrustedSelectorBits, build_structural_selector_families,
    canonical_component_name, public_family_artifact_name, selector_bit_family_name,
    selector_family_artifact_name, selector_family_artifact_name_from_names,
};
pub use diagnostics::{
    PERFORMANCE_CATEGORIES, PbcDiagnosticAggregator, PbcDiagnosticReport, PbcDiagnosticSample,
    measure_key_layout,
};
pub use evaluation::{
    PbcEncodedPublicVector, PbcLayoutFamilies, PbcPublicVectorFamilyBinding, derive_lwr_vector,
};
pub use layout::{
    PbcActiveCellIndex, PbcParameters, PbcProfile, PbcPublicLayout, derive_attempt_seed,
    derive_candidate_buckets,
};
pub use schedule::{
    PbcGeneratedKeyLayout, PbcPrivateSchedule, canonical_decode, clear_pbc_inner_product,
    dense_binary_support, generate_key_layout, support_from_dense,
};

#[cfg(test)]
mod tests;
