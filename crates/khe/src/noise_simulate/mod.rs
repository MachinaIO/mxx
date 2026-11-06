//! Error-norm simulation of BGG+ evaluations and the parameter selection built on it.
//!
//! A BGG+ encoding of `x` under the public matrix `A` is `c = s (A - x G) + e`; the simulator
//! bounds `||e||_inf`, the largest absolute coefficient of the error, through an evaluation. Every
//! rule follows one evaluator operation term by term, and its documentation names the operation.
//!
//! - [`rules`]: the error of one operation: addition, the right action `e G^(-1)(T)` of a
//!   decomposition, and the product `c_L G^(-1)(A_R) + x_L c_R`.
//! - [`boolean`]: the layered Boolean evaluation of
//!   [`crate::bgg::evaluate_boolean_encoding_layers`] followed by the decoding action of the
//!   output, and the condition under which rounding `(c' - s A') 2 / q` recovers the output bit.
//! - [`search`]: the smallest ring dimension and CRT depth whose simulated error decodes and whose
//!   Ring-LWE instance reaches a security target.
//!
//! The bounds are deterministic. Every sampled error is drawn by a sampler that enforces its
//! coefficient cutoff, so a bound that holds for the cutoffs holds on every sampling tape: an
//! evaluation within the bounds fails with probability zero, which also bounds the failure
//! probability of an average-case claim.

pub mod boolean;
pub mod rules;
pub mod search;

pub use boolean::{BooleanBggNoise, BooleanBggNoiseReport};
pub use search::{BooleanBggParameterRequest, BooleanBggSelection, select_boolean_parameters};
