//! Diamond iO 2: obfuscation of streaming functions over RNS-BGG encodings with a noise refresh
//! at every step. The specification is `crates/io/DIO2-SPEC.md`; section numbers below refer to
//! it.
//!
//! An obfuscation of a [`StreamingFunction`] fixes the obfuscator's chunks `y_i` and the initial
//! state. It is one graph ([`obfuscate`]): the step input `S_0`, the anchors of every step, and,
//! in parallel loops over the steps and chunks, the Static preprocessing of the uniform step
//! Program ([`step::ChainProgram`]) with the key-only refresh composition ([`compose`]) whose
//! decoded keys the anchors' preimages are sampled for. Its evaluation on the evaluator's chunks
//! `x_i` is one graph too ([`evaluate`]): a sequential loop over the refreshing steps whose state
//! is the step input `S_i` under the step's mask `s_i` runs the step Program on the chain of the
//! root modulus (every level of `Q_0`) and the refresh of every target, decoded for every prime,
//! to the next step's input under the mask `s_(i+1)` the σ sampler derived; the last step's
//! output decoder follows the loop. Both compositions build their keys with the same helpers
//! ([`carrier`]). [`claim`] connects the two graphs into the correctness protocol, and [`noise`]
//! simulates every decoder.

mod carrier;
pub mod claim;
mod compose;
mod evaluate;
pub mod function;
mod labels;
pub mod noise;
mod obfuscate;
pub mod params;
pub mod scheme;
mod search;
pub mod step;
#[cfg(test)]
mod test_support;

pub use function::{StepData, StepProgram, StepWires, StreamingFunction};
pub use labels::{LabelProducts, label_transition, next_labels};
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
use std::cell::RefCell;
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

/// The first scheme error of a graph construction that runs inside DSL loop bodies, whose
/// closures return [`mxx_dsl::DslError`]: a body keeps its error here and fails the loop, and
/// the caller reports the kept error instead of the loop's.
#[derive(Default)]
pub(crate) struct Failure(RefCell<Option<Dio2Error>>);

impl Failure {
    /// Keep the error of `result`, if any, and fail the enclosing loop body.
    pub(crate) fn keep<T>(&self, result: Result<T, Dio2Error>) -> Result<T, mxx_dsl::DslError> {
        result.map_err(|error| {
            self.0.borrow_mut().get_or_insert(error);
            mxx_dsl::DslError::Schema
        })
    }

    /// The loop's `result`, or the kept error that failed it.
    pub(crate) fn finish<T>(&self, result: Result<T, mxx_dsl::DslError>) -> Result<T, Dio2Error> {
        match self.0.borrow_mut().take() {
            Some(error) => Err(error),
            None => Ok(result?),
        }
    }
}
