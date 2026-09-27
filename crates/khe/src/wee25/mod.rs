//! WEE25 commitments: commitment trees, public-parameter preprocessing, and openings with
//! their verification.
//!
//! - `commitment`: the commitment compiler and its tree wires.
//! - `public_parameters`: the public parameters and their trapdoor preprocessing.
//! - `opening`: opening and verification graphs.
//!
//! The commitment-backed lookup evaluator is intentionally absent.

pub mod commitment;
pub mod opening;
pub mod public_parameters;

pub use commitment::{Wee25CommitmentCompiler, Wee25CommitmentError, Wee25CommitmentTreeWire};
pub use opening::{
    WEE25_COMMITMENT, WEE25_COMMITMENT_NODES, WEE25_PUBLIC_B, WEE25_T_BOTTOM, WEE25_T_TOP,
    Wee25CommitmentArtifacts, Wee25PublicParameterArtifacts, Wee25PublicParameterWires,
    Wee25VerificationWire,
};
pub use public_parameters::{
    WEE25_PUBLIC_B_TRAPDOOR, Wee25PublicParameterCompiler, Wee25PublicParameterPreprocessingWires,
};
