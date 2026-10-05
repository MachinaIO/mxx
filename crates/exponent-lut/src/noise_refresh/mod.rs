//! Native exponent-encoding noise refresh.
//!
//! [`decoder`] removes the `s A` term of an encoding with a trapdoor
//! preimage, and nothing else. [`refresh`] builds the noise refresh on top of
//! it: it adds PRF-derived mask encodings to a source encoding whose payload
//! is `(q/p) mu`, decodes the sum, and rounds to `mu mod p`. Both are
//! scheme-agnostic; the RNS-BGG adapter binds them to concrete carriers.
//! [`simulation`] holds the noise-model settings shared with Program audits.

pub mod decoder;
pub mod refresh;
pub mod simulation;

pub use refresh::{NoiseRefresh, RefreshDeclaration};
