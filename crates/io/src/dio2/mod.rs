//! Diamond iO 2: obfuscation of streaming functions over RNS-BGG encodings with a noise refresh
//! at every step. The specification is `crates/io/DIO2-SPEC.md`; section numbers below refer to
//! it.
//!
//! An obfuscation of a [`StreamingFunction`] fixes the obfuscator's chunks `y_i` and the initial
//! state. Its evaluation on the evaluator's chunks `x_i` runs every step on the encodings of the
//! step input `S_i` under the step's mask `s_i`: the step Program ([`step::ChainProgram`]) on
//! the chain of the root modulus (every level of `Q_0`), then the refresh of every target
//! ([`refresh`]), decoded for every prime, to the next step's input under the mask `s_(i+1)` the
//! σ sampler derived, and finally the output decoder. [`noise`] simulates every decoder.

pub mod claim;
pub mod function;
pub mod noise;
pub mod params;
pub mod refresh;
pub mod scheme;
pub mod step;
#[cfg(test)]
mod test_support;

pub use function::{StepProgram, StreamingFunction};
pub use params::Dio2Parameters;
pub use scheme::{Dio2, Dio2Function, Dio2Obfuscation};

use mxx_backends::sampler::bounds::default_preimage_cutoff;
use mxx_exponent_lut::{
    api::ApiError,
    decoder::unmask::UNMASK_ROWS,
    program::ProgramValidationError,
    program_gadgets::{prf::PrfError, prf_sampler::PrfSamplerError},
    rns_bgg::RnsBggError,
};
use mxx_ir_core::types::TrapdoorInstantiation;
use num_bigint::BigUint;
use thiserror::Error;

/// Errors of a Diamond iO 2 obfuscation or evaluation.
#[derive(Debug, Error)]
pub enum Dio2Error {
    #[error("invalid Diamond iO 2 parameters: {0}")]
    Parameters(&'static str),
    #[error("invalid input: {0}")]
    Input(&'static str),
    #[error(transparent)]
    Program(#[from] ProgramValidationError),
    #[error(transparent)]
    Prf(#[from] PrfError),
    #[error(transparent)]
    Sampler(#[from] PrfSamplerError),
    #[error(transparent)]
    RnsBgg(#[from] RnsBggError),
    #[error(transparent)]
    Api(#[from] ApiError),
    #[error("graph construction failed: {0}")]
    Graph(String),
}

impl From<mxx_dsl::DslError> for Dio2Error {
    fn from(error: mxx_dsl::DslError) -> Self {
        Self::Graph(error.to_string())
    }
}

/// The columns `2 (w_0 + 2)` of the anchor `B_i` (an unmask trapdoor over `Q_0`, whose gadget
/// has the `w_0` digits of `Q_0`).
pub fn anchor_columns(columns: usize) -> usize {
    UNMASK_ROWS * (columns + 2)
}

/// The coefficient bound of the anchors' preimages: the default cutoff of the trapdoor's
/// minimum preimage width at `trapdoor_sigma`.
pub fn preimage_bound(parameters: &Dio2Parameters) -> Result<BigUint, Dio2Error> {
    default_preimage_cutoff(
        parameters.ring_dimension,
        UNMASK_ROWS,
        parameters.columns()?,
        1u32 << parameters.base_bits,
        f64::from(parameters.trapdoor_sigma),
        TrapdoorInstantiation::Computational,
    )
    .ok_or(Dio2Error::Parameters("the anchor trapdoor has no preimage width"))
}
