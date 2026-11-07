//! Reusable Exponent-LUT Program gadgets.
//!
//! Each gadget only assembles an [`crate::program::ExponentLutProgram`] with
//! [`crate::program::ExponentLutProgramBuilder`] and prepares the public data
//! its inputs need; public keys and encodings of the Program are computed by
//! the generic backend evaluators.
//!
//! - [`air`]: one streaming step of an AIR constraint check, over one public attribute per field.
//! - [`public_ip`]: the inner product of a public vector with a hash-sampled public binary vector.
//! - [`turboshake128`]: counter-mode TurboSHAKE128 over public sign encodings, packed into label
//!   encodings.

pub mod air;
pub mod public_ip;
pub mod turboshake128;
