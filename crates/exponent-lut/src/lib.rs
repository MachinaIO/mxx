//! Scheme-independent Exponent-LUT programs and their RNS-BGG adapter.
//!
//! Public preprocessing and online evaluation are exposed through [`api`].
//! Concrete cryptographic types and errors belong to the corresponding
//! backend evaluator.

pub mod api;
pub mod nand_circuit;
pub mod noise_refresh;
pub mod noise_utils;
pub mod program;
pub mod rns_bgg;
mod utils;

pub use nand_circuit::{CompiledNandCircuit, NandCircuit, NandCircuitError};
pub use noise_refresh::prf::SparseLwrPrfProfile;
pub use program::{ExpLayout, ExponentLutProgram, ExponentLutProgramId, LayoutError};
use thiserror::Error;

/// Shared declaration and graph-construction errors, independent of a backend.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExponentLutError {
    #[error(transparent)]
    Program(#[from] program::ProgramValidationError),
    #[error("LUT dimensions or output exponents are invalid")]
    InvalidLut,
    #[error("program operation {gate} failed: {detail}")]
    ProgramOperation { gate: String, detail: String },
}

#[cfg(test)]
extern crate self as mxx_exponent_lut;
