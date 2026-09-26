//! Executable DSL graphs for TFHE and leveled BGV.
//!
//! Methods construct graphs; randomness and cryptographic arithmetic run in
//! `mxx-backends`. Parameters do not imply a security level. TFHE provides
//! integer LWE encryption, blind rotation, sample extraction, key switching,
//! and NAND; BGV provides leveled arithmetic. Ciphertext/key compatibility
//! beyond ring and shape is the caller's responsibility: graph handles are not
//! authenticated cryptographic objects. Runtime inputs are passed directly;
//! staged FHE artifacts use `mxx_backends::MemoryArtifactStore`, with no file
//! persistence.

mod bgv;
mod params;
#[cfg(all(test, feature = "gpu"))]
mod tests_gpu;
mod tfhe;
pub mod utils;

pub use bgv::{BgvCiphertext, BgvCiphertextSchema, BgvHybridParams, BgvParams};
use mxx_dsl::{DslError, GraphValue, Mat};
pub use params::FheCommonParams;
pub use tfhe::{
    BootstrappingKey, BootstrappingKeySchema, KeySwitchKey, KeySwitchKeySchema, LweCiphertext,
    LweCiphertextSchema, RingCiphertext, RingCiphertextSchema, TfheKeys, TfheKeysSchema,
    TfheParams,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FheError {
    #[error("invalid FHE parameters: {0}")]
    InvalidParameters(&'static str),
    #[error("matrix or family shape does not match the expected FHE value")]
    ShapeMismatch,
    #[error("ciphertext modulus does not match the required chain level")]
    LevelMismatch,
    #[error("ciphertexts have different BGV correction factors")]
    CorrectionFactorMismatch,
    #[error("BGV correction factor must be a canonical unit modulo the plaintext modulus")]
    InvalidCorrectionFactor,
    #[error("this evaluation requires a key")]
    MissingEvaluationKey,
    #[error(transparent)]
    Dsl(#[from] DslError),
}

/// Shared matrix-plaintext graph operations. TFHE uses its dedicated integer
/// LWE API; BGV multiplication uses a relinearization key.
pub trait FheScheme {
    type Plaintext: GraphValue;
    type Ciphertext: GraphValue;
    type MulRhs: GraphValue;
    type EvaluationKey;

    fn common_params(&self) -> &FheCommonParams;
    /// Returns `(secret_key, encryption_key)` graph handles.
    fn keygen(&self) -> Result<(Mat, Mat), FheError>;
    fn encrypt(&self, key: &Mat, plaintext: &Self::Plaintext)
    -> Result<Self::Ciphertext, FheError>;
    fn decrypt(
        &self,
        secret: &Mat,
        ciphertext: &Self::Ciphertext,
    ) -> Result<Self::Plaintext, FheError>;
    fn add(
        &self,
        lhs: &Self::Ciphertext,
        rhs: &Self::Ciphertext,
    ) -> Result<Self::Ciphertext, FheError>;
    fn mul(
        &self,
        lhs: &Self::Ciphertext,
        rhs: &Self::MulRhs,
        eval_key: &Self::EvaluationKey,
    ) -> Result<Self::Ciphertext, FheError>;
}
