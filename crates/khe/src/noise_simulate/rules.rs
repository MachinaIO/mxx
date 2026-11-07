//! Error bounds of single BGG+ operations.
//!
//! Errors are bounded in the infinity norm of their coefficients. The gadget decomposition
//! `G^(-1)` of the regular layout writes every coefficient of every CRT residue with balanced
//! digits of absolute value at most `base / 2`, and is exact when the digit count covers every
//! tower, which is the layout used here.

use mxx_backends::poly::PolyParams;
use num_bigint::BigUint;

/// The bound on `||e G^(-1)(T)||_inf / ||e||_inf` for a `1 x columns` error row `e` and a
/// ring-valued target `T`: each output coefficient sums `columns * n` products of an error
/// coefficient and a digit of absolute value at most `base / 2`.
pub fn decomposition_gain(ring_dimension: u32, columns: usize, base: &BigUint) -> BigUint {
    BigUint::from(columns) * ring_dimension * (base / 2u32)
}

/// [`decomposition_gain`] for a target with constant polynomial entries, whose digits are
/// constant: each output coefficient sums `columns` products.
pub fn constant_decomposition_gain(columns: usize, base: &BigUint) -> BigUint {
    BigUint::from(columns) * (base / 2u32)
}

/// The error of `c_L G^(-1)(A_R) + x_L c_R`, the BGG+ product of encodings with errors at most
/// `left_error` and `right_error` and a left plaintext of coefficient l1 norm at most
/// `plaintext_l1_bound`: `gain * left_error + plaintext_l1_bound * right_error`.
pub fn product_error(
    gain: &BigUint,
    left_error: &BigUint,
    right_error: &BigUint,
    plaintext_l1_bound: &BigUint,
) -> BigUint {
    gain * left_error + plaintext_l1_bound * right_error
}

/// Worst-case BGG product error for a scalar plaintext with the given coefficient l1 norm, when
/// the decomposition may drop CRT towers.
/// The final term is `x_L * s * (A_R - G D(A_R))`, sharing the actual decomposition residual.
/// For binary plaintexts pass one; no Gaussian independence assumption is made.
pub fn multiplication_error_bound<P: PolyParams>(
    params: &P,
    secret_dimension: usize,
    left_error: &BigUint,
    right_error: &BigUint,
    secret_bound: &BigUint,
    plaintext_l1_bound: &BigUint,
) -> BigUint {
    let n = BigUint::from(params.ring_dimension());
    let columns = BigUint::from(secret_dimension) * params.modulus_digits();
    let digit_bound = BigUint::from(1u8) << (params.base_bits() - 1);
    columns * &n * left_error * digit_bound +
        plaintext_l1_bound *
            (right_error +
                n * secret_dimension * secret_bound * params.gadget_error_bound(None))
}
