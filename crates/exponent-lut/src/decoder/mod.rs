//! Native smudged decoder of exponent encodings.
//!
//! [`unmask`] removes the `s A` term of an encoding with a trapdoor preimage,
//! and nothing else. [`decoding`] builds the decoder on top of it: it adds
//! PRF-derived smudge encodings to a source encoding whose payload is
//! `(q/p) mu`, unmasks the sum, and rounds to the plaintext `mu mod p`. Both
//! are scheme-agnostic; the RNS-BGG adapter binds them to concrete carriers.
//! [`simulation`] holds the noise-model settings shared with Program audits.
//!
//! The decoder outputs a plaintext, not an encoding. A decoder targets either
//! a Program output, rounded to the next level's modulus, or one column of a
//! noise refresh, rounded to one CRT prime ([`decoding::DecoderTarget`]);
//! [`crate::refresh`] combines the latter into an encoding with fresh noise.

pub mod decoding;
pub mod simulation;
pub mod unmask;

pub use decoding::{Decoder, DecoderDeclaration};
