//! The trapdoor decoder: it removes `s A` from an encoding and nothing else.
//!
//! An encoding under the mask secret `s = (s_bar, 1)` with public key
//! `A in R_q^(2 x w)` is a row `c = s A - P + e_c`. The decoder removes the
//! `s A` term without rounding, so the payload `P` keeps its error; removing
//! that error is the job of the noise refresh built on top of it.
//!
//! - Trusted setup samples a public `B in R_q^(2 x m_B)` together with its trapdoor
//!   ([`sample_trapdoor`]). The trapdoor never leaves the trusted preprocessor.
//! - Preprocessing samples a short preimage `K in R_q^(m_B x w)` with `B K = A` ([`preimage`]).
//! - Encryption publishes `b = s B + e_b` ([`decoder_row`]).
//! - Online evaluation computes `d = b K = s A + e_b K` and returns `d - c = P + e_b K - e_c`
//!   ([`decode`]).
//!
//! `K` is a preimage of `A` only, so an evaluator can decode exactly the
//! encodings whose key is `A`. The operations are plain DSL values; a scheme
//! adapter chooses the ring, the key and the encoding row it decodes.

use mxx_dsl::{Mat, Preimage, Ring, Trapdoor};
use mxx_ir_core::{IntExpr, RealExpr};

/// Rows of `B`: the length of the mask secret row `(s_bar, 1)`.
pub const DECODER_ROWS: usize = 2;

/// Sample the public `B` (`2 x 2 (digit_count + 2)`) and its trapdoor in
/// `ring`, whose gadget has `gadget_base` and `digit_count` digits.
pub fn sample_trapdoor(
    ring: &Ring,
    sigma: impl Into<RealExpr>,
    gadget_base: impl Into<IntExpr>,
    digit_count: impl Into<IntExpr>,
    preimage_bound: impl Into<IntExpr>,
) -> Trapdoor {
    ring.sample_trapdoor(DECODER_ROWS, sigma, gadget_base, digit_count, preimage_bound)
}

/// A short `K` with `B K = key` for a `2 x w` key, kept as a compact
/// bounded matrix.
pub fn preimage(trapdoor: &Trapdoor, key: Mat) -> Preimage {
    let columns = key.matrix_type().columns.clone();
    let rows = trapdoor.public_matrix().matrix_type().columns.clone();
    trapdoor.sample_preimage(key, (rows, columns))
}

/// The encryption's decoder row `b = s B + error` for the `1 x 2` secret
/// row `s`.
pub fn decoder_row(secret: Mat, public: Mat, error: Mat) -> Mat {
    secret * public + error
}

/// The decoded payload `b K - c` of the encoding row `c` whose key `K`
/// is a preimage of. The compact `K` is consumed directly by the bounded
/// product and never expanded.
pub fn decode(row: Mat, preimage: Preimage, encoding: Mat) -> Mat {
    row.mul_small_rhs(preimage) - encoding
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{execute_cpu, full, ring_of, validate};
    use mxx_backends::{
        poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
        sampler::bounds::default_preimage_cutoff,
    };
    use mxx_dsl::DslContext;
    use num_bigint::{BigInt, BigUint};
    use std::collections::BTreeMap;

    /// Ring dimension, CRT depth and key width are overridable by
    /// `MXX_EXPONENT_LUT_DECODER_TEST_{RING_DIMENSION,CRT_DEPTH,WIDTH}`.
    fn read(name: &str, default: usize) -> usize {
        std::env::var(format!("MXX_EXPONENT_LUT_DECODER_TEST_{name}"))
            .map_or(default, |value| value.parse().unwrap())
    }

    /// Setup, preprocessing, encryption and decoding of a random key: `B K`
    /// equals the key exactly, and `d - c` is the payload plus an error
    /// far below the modulus.
    #[test]
    fn decoder_removes_the_key_term_and_keeps_the_payload() {
        let parameters = DCRTPolyParams::new(
            read("RING_DIMENSION", 16) as u32,
            read("CRT_DEPTH", 2),
            17,
            4,
            None,
            None,
        );
        let ring = ring_of(&parameters);
        let width = read("WIDTH", 3);
        let sigma = 4.578;
        let trapdoor_sigma = crate::utils::decimal_to_real_expr(&"4.578".parse().unwrap()).unwrap();
        let base = 1u32 << parameters.base_bits();
        let digits = parameters.modulus_digits();
        let bound =
            default_preimage_cutoff(parameters.ring_dimension(), DECODER_ROWS, digits, base, sigma)
                .unwrap();
        let trapdoor =
            sample_trapdoor(&ring, trapdoor_sigma, base, digits, BigInt::from(bound.clone()));
        let key = ring.uniform_residue((DECODER_ROWS, width));
        let k = preimage(&trapdoor, key.clone());
        let secret = Mat::concat(
            mxx_ir_core::node::ConcatAxis::Columns,
            vec![ring.uniform_interval((1, 1), -1, 1), ring.identity(1)],
        );
        let columns = trapdoor.public_matrix().matrix_type().columns.clone();
        let error_bound = 8;
        let b = decoder_row(
            secret.clone(),
            trapdoor.public_matrix(),
            ring.gaussian((1, columns), 1, error_bound),
        );
        let payload = ring.uniform_residue((1, width));
        let encoding =
            secret * key.clone() - payload.clone() + ring.gaussian((1, width), 1, error_bound);
        let graph = DslContext::new("decoder-round-trip")
            .output("key-residual", trapdoor.public_matrix().mul_small_rhs(k.clone()) - key)
            .unwrap()
            .output("payload-residual", decode(b, k, encoding) - payload)
            .unwrap()
            .build()
            .unwrap();
        let result = execute_cpu(&validate(&graph), [parameters.clone()], BTreeMap::new());
        let modulus = BigInt::from(parameters.modulus().as_ref().clone());
        let centered = |value: BigUint| {
            let value = BigInt::from(value);
            if &value * 2u8 > modulus { value - &modulus } else { value }
        };
        let key_residual = full(&result.outputs["key-residual"]);
        for row in 0..DECODER_ROWS {
            for column in 0..width {
                assert!(
                    key_residual
                        .entry(row, column)
                        .coeffs_biguints()
                        .into_iter()
                        .all(|c| c == 0u8.into()),
                    "B K must equal the key"
                );
            }
        }
        // |e_b K - e_c| <= N m_B B_e B_K + B_e.
        let m_b = DECODER_ROWS * (digits + 2);
        let limit = BigInt::from(parameters.ring_dimension()) *
            BigInt::from(m_b) *
            BigInt::from(error_bound) *
            BigInt::from(bound) +
            BigInt::from(error_bound);
        assert!(limit.clone() * 4u8 < modulus, "test parameters leave no decoding margin");
        let payload_residual = full(&result.outputs["payload-residual"]);
        for column in 0..width {
            for coefficient in payload_residual.entry(0, column).coeffs_biguints() {
                assert!(centered(coefficient).magnitude() <= limit.magnitude());
            }
        }
    }
}
