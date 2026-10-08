//! Reusable Exponent-LUT Program gadgets.
//!
//! Each gadget only assembles an [`crate::program::ExponentLutProgram`] with
//! [`crate::program::ExponentLutProgramBuilder`] and prepares the public data
//! its inputs need; public keys and encodings of the Program are computed by
//! the generic backend evaluators.
//!
//! - [`prf`]: the sparse-LWR weak PRF, whose sparse keys are routed by the probabilistic batch
//!   codes of [`pbc`].
//! - [`prf_sampler`]: binary and centered-binomial samples read from the [`prf`] output.
//! - [`public_ip`]: the inner product of a public vector with a hash-sampled public binary vector.

pub mod pbc;
pub mod prf;
pub mod prf_sampler;
pub mod public_ip;
