//! Scheme-independent Exponent-LUT programs and their BGG+ and RNS-BGG adapters.
//!
//! Public preprocessing and online evaluation interpret the same frozen Program
//! through [`scheme::ExponentLUTEncodingScheme`]. Concrete cryptographic types
//! and errors belong to the corresponding backend module.

pub mod bgg;
pub mod layout;
pub mod lean;
pub mod noise_refresh;
pub mod noise_utils;
pub mod program;
pub mod readout;
pub mod rns_bgg;
pub mod scheme;
mod utils;

pub use layout::{ExpLayout, LayoutError};
pub use program::{
    AttributeKnowledge, FamilyAggregation, FamilyProgram, FamilyRhsSource, FamilyWireSource, Gate,
    ProgramFamilyId, ProgramStep, ProgramWireType, RhsId, WireFamilyId, WireId,
};
use thiserror::Error;

/// Shared declaration and graph-construction errors, independent of a backend.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExponentLutError {
    #[error(transparent)]
    Program(#[from] program::ProgramValidationError),
    #[error("LUT dimensions, output exponents, or RHS width are invalid")]
    InvalidLut,
    #[error("program operation {gate} failed: {detail}")]
    ProgramOperation { gate: String, detail: String },
}

#[cfg(test)]
extern crate self as mxx_exponent_lut;
