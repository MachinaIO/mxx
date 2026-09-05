//! Protocol declarations, input contracts, and workflow connection validation.
//!
//! This crate does not infer noise bounds. Applications own their mathematical bounds and
//! proofs; `mxx-ir-core::lean` mechanically exports execution relations and linked claims.

pub mod bundle;
pub mod protocol;
#[cfg(test)]
mod test_protocol;

pub use bundle::*;
pub use protocol::*;
