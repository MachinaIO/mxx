//! Predicate encryption for branching programs over key-homomorphic
//! encodings, without FHE.
//!
//! This crate implements the scheme of *Implementing Predicate Encryption for
//! Branching Programs: Private Computation over Key-Homomorphic Encodings
//! Without FHE* (Sections 4 and 5) on the Dynamic RNS-BGG evaluator of
//! `mxx-exponent-lut`:
//!
//! - [`setup::setup`] samples the anchor `B = [B^(1); B^(2)]` with the existing one-row TrapGen and
//!   fixes the class's Dynamic setup: the initial-state key `A_0`, the attribute and automorphism
//!   packages' public matrices, and the helper matrices `H_k`.
//! - [`encryption::encrypt`] produces, under a fresh secret `s = (s_bar, 1)`, the anchor encoding
//!   `b = s B + e_b`, the attribute gadget ciphertexts `C_t = P Z_t + X^(Bz x_t) G` with `P = B -
//!   [0; b]`, the message component `c_m = s d^T + e_m + floor(q/2) m`, and the Dynamic packages
//!   (digit companions, helpers, and the initial state `c_0`). It needs no predicate.
//! - [`keygen::keygen`] evaluates the predicate's public key `A_F` and samples a short key `k` with
//!   `M_F k = d_1 delta_1` for the implicit family matrix `M_F` by SampleLeft: one shared extension
//!   sample, then one existing preimage call per anchor block.
//! - [`decryption::decrypt`] evaluates `c_F` on the ciphertext's packages and contracts the family
//!   row with the key: `c_m - d_2 - (y_F - kappa_F) k = floor(q/2) m + e` when `F(x) = 0`.
//!
//! [`noise`] obtains the evaluation bound `E_F` from the exponent-lut
//! simulator and applies the decryption margin; [`params`] fixes the class,
//! the equal-basis chain, the sampling contracts, and the key width the
//! existing sampler realizes.
//!
//! # Conventions
//!
//! - `F(x) = 0` authorizes decryption. An unauthorized decryption returns the rounded phase; it is
//!   not a reliable rejection.
//! - The anchor row `B^(1)` is the public matrix of the existing TrapGen, with its identity block
//!   and its column count `m_B = D + 2`, as directed for this implementation; the manuscript's
//!   uniform-anchor and anchor-width requirements are not enforced.
//! - The setup's public matrices other than `B^(1)` are expanded from a public hash key; they are
//!   assumed indistinguishable from the manuscript's uniform matrices.
//! - Gaussian widths are the sampler's standard-deviation parameter `sigma`; the manuscript's
//!   `exp(-pi |x|^2 / s^2)` width is `s = sqrt(2 pi) sigma` ([`params::paper_width`]).

pub mod branching_program;
pub mod decryption;
pub mod encryption;
pub mod keygen;
pub mod noise;
pub mod params;
pub mod setup;

pub use branching_program::{BranchingProgram, CompiledPredicate, PeClass};
pub use decryption::decrypt;
pub use encryption::{PeCiphertext, encrypt};
pub use keygen::{PePredicateKey, keygen};
pub use noise::PeNoiseReport;
pub use params::{PeParameterRequest, PeParameters};
pub use setup::{PeMasterSecretKey, PePublicKey, setup};

use thiserror::Error;

/// Errors of the predicate encryption scheme.
#[derive(Debug, Error)]
pub enum PeError {
    #[error("invalid function class: {0}")]
    Class(&'static str),
    #[error("invalid branching program: {0}")]
    Predicate(&'static str),
    #[error("branching program compilation failed: {0}")]
    Compilation(String),
    #[error("invalid scheme parameters: {0}")]
    Parameters(String),
    #[error("the attribute does not belong to the class")]
    Attribute,
    #[error("the message must have binary coefficients and at most n of them")]
    Message,
    #[error("the artifacts belong to another setup, predicate, or ciphertext")]
    Mismatch,
    #[error(transparent)]
    Rns(#[from] mxx_exponent_lut::rns_bgg::RnsBggError),
    #[error(transparent)]
    Stage(#[from] mxx_exponent_lut::api::ApiError),
    #[error("graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for PeError {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(all(test, feature = "gpu"))]
mod gpu;
