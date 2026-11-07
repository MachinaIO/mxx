//! The layered Boolean evaluation of BGG+ encodings and the decoding of its output.
//!
//! [`crate::bgg::evaluate_boolean_encoding_layers`] computes every slot of a layer from its left
//! and right operands in the preceding layer, whose errors are at most `E`, and from the encoding
//! of one, whose error is at most the fresh bound `B <= E`. It evaluates six candidates and
//! selects one, so a slot's error is at most the largest candidate error, with
//! `alpha = m n base / 2` the [`super::rules::decomposition_gain`] of the `1 x m` encodings:
//!
//! | candidate | evaluator expression | error bound |
//! | --- | --- | --- |
//! | false | `one - one` | `0` |
//! | true | `one` | `B` |
//! | copy | `left` | `E` |
//! | not | `one - left` | `B + E` |
//! | and | `c_L G^(-1)(A_R) + x_L c_R` | `alpha E + E` (the plaintext `x_L` is a bit) |
//! | xor | `left + right - 2 and` | `2 E + 2 (alpha E + E)` |
//!
//! Inactive slots select false. The xor bound is the largest, so the errors after layer `l` are
//! at most `E_l = (2 alpha + 4)^l B`. The selected output encoding `(c, A)` of plaintext `x` is
//! then acted on by the public decomposition `D = G^(-1)(floor(q / 2) e_last)` of the constant
//! column with `floor(q / 2)` in the row of the secret's constant component, giving
//! `c' = c D` and `A' = A D`. For `s = (s_bar, 1)`, `s G D = floor(q / 2)`, so
//! `c' - s A' = -x floor(q / 2) + e D` with `||e D||_inf <= m (base / 2) E_depth`. Rounding
//! `2 (c' - s A') / q` modulo two returns `x` when `4 ||e D||_inf + 4 < q` for odd `q`, the
//! condition of the Lean proofs in `crates/khe/lean/ring_gsw_bgg` (`decode_output`).

use super::rules::{constant_decomposition_gain, decomposition_gain, product_error};
use num_bigint::BigUint;

/// The public parameters that determine the errors of a layered Boolean BGG+ evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanBggNoise {
    pub ring_dimension: u32,
    pub modulus: BigUint,
    /// The number of secret components, including the constant one.
    pub secret_dimension: usize,
    /// The gadget digits of one secret component.
    pub digit_count: usize,
    pub gadget_base: BigUint,
    /// The coefficient cutoff of every sampled encoding error.
    pub fresh_error_bound: BigUint,
}

/// The simulated errors of one evaluation and its decoding condition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanBggNoiseReport {
    /// The error bound of every slot after the last layer.
    pub evaluation_error: BigUint,
    /// The error bound of `c' - s A'` after the decoding action.
    pub output_error: BigUint,
    pub modulus: BigUint,
}

impl BooleanBggNoiseReport {
    /// Whether rounding `2 (c' - s A') / q` recovers the output bit: `4 E + 4 < q`.
    pub fn decrypts(&self) -> bool {
        self.output_error.clone() * 4u32 + 4u32 < self.modulus
    }
}

impl BooleanBggNoise {
    /// The columns `m` of an encoding.
    pub fn columns(&self) -> usize {
        self.secret_dimension * self.digit_count
    }

    /// The error bound of every slot of a layer whose preceding layer's errors are at most
    /// `preceding`, as the largest candidate error.
    pub fn layer_error(&self, preceding: &BigUint) -> BigUint {
        let fresh = &self.fresh_error_bound;
        let gain = decomposition_gain(self.ring_dimension, self.columns(), &self.gadget_base);
        let and = product_error(&gain, preceding, preceding, &BigUint::from(1u8));
        let xor = preceding * 2u32 + &and * 2u32;
        [BigUint::ZERO, fresh.clone(), preceding.clone(), fresh + preceding, and, xor]
            .into_iter()
            .max()
            .expect("six candidates")
    }

    /// The errors of `depth` layers from fresh encodings, then the decoding action.
    pub fn simulate(&self, depth: usize) -> BooleanBggNoiseReport {
        let evaluation_error = (0..depth)
            .fold(self.fresh_error_bound.clone(), |preceding, _| self.layer_error(&preceding));
        let output_error =
            constant_decomposition_gain(self.columns(), &self.gadget_base) * &evaluation_error;
        BooleanBggNoiseReport { evaluation_error, output_error, modulus: self.modulus.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_errors_grow_by_the_xor_candidate() {
        let noise = BooleanBggNoise {
            ring_dimension: 8,
            modulus: BigUint::from(1u64 << 40),
            secret_dimension: 2,
            digit_count: 4,
            gadget_base: BigUint::from(1u32 << 10),
            fresh_error_bound: BigUint::from(3u8),
        };
        // alpha = 8 columns * 8 * 512, and xor gives (2 alpha + 4) E.
        let alpha = BigUint::from(8u32 * 8 * 512);
        assert_eq!(noise.layer_error(&BigUint::from(5u8)), (alpha * 2u32 + 4u32) * 5u32);
        let report = noise.simulate(0);
        assert_eq!(report.evaluation_error, BigUint::from(3u8));
        assert_eq!(report.output_error, BigUint::from(8u32 * 512 * 3));
        assert!(report.decrypts());
    }
}
