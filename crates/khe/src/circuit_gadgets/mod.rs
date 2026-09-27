//! Gadgets written as circuits or DSL graphs.
//!
//! - `arith`: modular arithmetic contexts, lane-packed nested RNS arithmetic (`NestedRnsPoly`), and
//!   carry and Montgomery arithmetic.
//! - `conv_mul`: negacyclic convolution without an NTT.
//! - `ntt`: radix-2 NTT and inverse NTT over nested RNS polynomials.
//! - `mod_switch`: nested-RNS modulus switching and its error bounds.
//! - `fhe`: Ring-GSW gadgets.
//! - `fhe_prg`: a Goldreich PRG evaluated over Ring-GSW bits.
//! - `secret_ip`: secret inner products.
//!
//! Nested-RNS level switching is one of these gadgets, not a graph node: the ring-conversion nodes
//! of `mxx-ir-core` are fused CRT operations with explicit destination rings.

pub mod arith;
pub mod conv_mul;
pub mod fhe;
pub mod fhe_prg;
pub mod mod_switch;
// The NTT gadget backs CKKS; its slow runtime tests are ignored by default.
pub mod ntt;
pub mod secret_ip;
