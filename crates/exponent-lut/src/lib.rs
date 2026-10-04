//! Scheme-independent Exponent-LUT programs and their RNS-BGG adapter.
//!
//! Public preprocessing and online evaluation are exposed through [`api`].
//! Concrete cryptographic types and errors belong to the corresponding
//! backend evaluator.

pub mod api;
#[cfg(feature = "gpu")]
mod gpu_execution;
pub mod nand_circuit;
pub mod noise_refresh;
pub mod noise_utils;
pub mod program;
pub mod program_gadgets;
pub mod rns_bgg;
pub mod utils;

#[cfg(test)]
mod test_support;

pub use nand_circuit::{CompiledNandCircuit, NandCircuit, NandCircuitError};
pub use program::{ExpLayout, ExponentLutProgram, ExponentLutProgramId, LayoutError};
pub use program_gadgets::prf::SparseLwrPrfProfile;
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
