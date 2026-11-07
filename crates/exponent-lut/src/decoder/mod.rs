//! Native decoding support for exponent encodings.
//!
//! [`unmask`] removes the `s A` term of an encoding with a trapdoor preimage,
//! and nothing else; it is scheme-agnostic. [`simulation`] holds the
//! noise-model settings shared with Program audits.

pub mod simulation;
pub mod unmask;

/// Interpret an already-rounded signed monomial in an odd plaintext modulus.
/// Returns its exponent divided by `stride`, or `-1` when the plaintext is not
/// exactly one signed monomial or its exponent is not aligned to the layout.
/// This is a plaintext validity check after noise removal, not a bound on the
/// noisy encoding. Coefficient extraction returns canonical residues, so `q - 1`
/// represents `-1`. Callers must reject the negative sentinel.
pub fn monomial_exponent(
    plaintext: mxx_dsl::Mat,
    modulus: u64,
    dimension: usize,
    stride: usize,
) -> Result<mxx_dsl::Int, mxx_dsl::DslError> {
    use mxx_dsl::Int;
    if modulus < 3 || modulus % 2 == 0 || dimension == 0 || stride == 0 {
        return Err(mxx_dsl::DslError::Schema);
    }
    let (exponent, nonzero, invalid) = mxx_dsl::iterate(
        dimension,
        (Int::constant(0), Int::constant(0), Int::constant(0)),
        |k, (exponent, nonzero, invalid)| {
            let coefficient = plaintext.clone().extract_coefficient(k.expression()?);
            let positive = coefficient.clone().equal(1).to_int();
            let negative = coefficient.clone().equal(modulus - 1).to_int();
            let valid = positive.clone().add(negative.clone()).add(coefficient.equal(0).to_int());
            Ok((
                exponent
                    .add(k.clone().mul(positive.clone()))
                    .add(k.add(dimension).mul(negative.clone())),
                nonzero.add(positive).add(negative),
                invalid.add(Int::constant(1).sub(valid)),
            ))
        },
    )?;
    let valid = nonzero
        .equal(1)
        .to_int()
        .mul(invalid.equal(0).to_int())
        .mul(exponent.clone().rem(stride).equal(0).to_int());
    mxx_dsl::select(valid, vec![Int::constant(-1), exponent.div(stride)])
}
