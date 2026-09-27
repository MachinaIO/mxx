//! Key-homomorphic encodings and the circuits they evaluate.
//!
//! [`circuit`] defines circuit models and the lowering framework that an encoding scheme
//! implements, independently of any scheme. [`bgg`] is the BGG+ key-homomorphic encoding built on
//! it: public keys, encodings, circuit evaluation, lookups, and slot operations. [`wee25`] holds
//! WEE25 commitments. [`circuit_gadgets`] provides reusable gadgets written as circuits or DSL
//! graphs.
//! [`ring_from_params`] converts backend parameters into a DSL ring with the same ordered basis,
//! and the `test-support` feature exposes `test_utils` to dependent crates' tests.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bgg;
pub mod circuit;
pub mod circuit_gadgets;
pub mod decoder;
pub mod input_injector;
pub mod noise_refresh;
pub mod utils;
pub mod wee25;

pub fn ring_from_params(
    parameters: &mxx_backends::poly::dcrt::params::DCRTPolyParams,
) -> mxx_dsl::Ring {
    use mxx_backends::poly::PolyParams;
    mxx_dsl::Ring::from_crt_moduli(
        parameters.to_crt().0.into_iter().map(Into::into).collect(),
        parameters.ring_dimension(),
    )
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_utils;
#[cfg(all(test, feature = "gpu"))]
mod test_utils_gpu;

// BGG-specific lookup evaluation lives in `bgg`.

pub use mxx_backends::{element::PolyElem, impl_binop_with_refs, parallel_iter, poly::Poly};
pub(crate) use mxx_backends::{matrix, poly, sampler};
