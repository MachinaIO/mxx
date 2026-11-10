//! Executable DSL graphs for TFHE and leveled BGV.
//!
//! Methods construct graphs; key generation, sampling, polynomial arithmetic, and decryption run
//! when `mxx-backends` executes the validated graph, on the CPU or the GPU. TFHE implements NAND
//! bootstrapping over integer LWE; BGV is leveled, with SIMD slots, rotations, and hybrid RNS key
//! switching, and has no bootstrapping. Both implement the one implementation-agnostic syntax of
//! public-key homomorphic encryption, [`spec::Fhe`], which binds them to the Lean statements.
//!
//! Parameters do not imply a security level, and the test parameters are correctness fixtures.
//! Ciphertext and key compatibility beyond ring and shape is the caller's responsibility: graph
//! handles are not authenticated cryptographic objects.
//!
//! To execute a graph:
//!
//! 1. Construct a `DslContext`, declare inputs, and use the FHE methods to build encryption,
//!    evaluation, and decryption nodes. Mark secrets and decoded values as private outputs.
//! 2. Validate the graph with its parameter bindings and register the exact ordered ciphertext CRT
//!    bases with the backend: [`BgvParams::runtime_parameters`] gives every ciphertext, hybrid,
//!    single-prime, and plaintext ring in tower order, and [`TfheParams::runtime_parameters`] gives
//!    the CRT prefixes and single-prime rings TFHE uses.
//! 3. Execute with inputs, a backend, a `MemoryArtifactStore`, and a sampling mode, and materialize
//!    lazy family outputs before reading them.
//!
//! Noise bounds propagate with each ciphertext through evaluation. `can_decrypt` checks a
//! conservative sufficient correctness condition and does not inspect secret values; declared
//! input bounds and compatible keys remain caller obligations. A matrix artifact alone does not
//! carry correction factors or bounds, so carry them alongside the components when connecting
//! separate protocol stages.

pub mod bgv;
mod params;
pub mod spec;
#[cfg(all(test, feature = "gpu"))]
mod tests_gpu;
pub mod tfhe;
pub mod utils;

pub use bgv::{BgvCiphertext, BgvCiphertextSchema, BgvHybridParams, BgvParams};
use mxx_dsl::DslError;
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
    #[error(transparent)]
    Protocol(#[from] mxx_ir_core::protocol::ProtocolError),
    #[error(transparent)]
    Specification(#[from] mxx_ir_core::protocol::SpecificationError),
}
