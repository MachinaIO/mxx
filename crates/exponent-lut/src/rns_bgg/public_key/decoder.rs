//! The preprocessing side of the smudged trapdoor decoder (see
//! [`crate::decoder`]).
//!
//! The decoded column is component 1 of the source output (its payload
//! secret component is 1) under the selection `G_(L+1)^-1(1)`, whose payload
//! is `Delta_L mu`, plus the smudge sum, whose members were
//! finalized and weighted in their root bodies. Its key is the `2 x 1`
//! matrix `A = (A_01; A_11) G_(L+1)^-1(1) + sum (A_01; A_11)_smudge`.
use super::*;
use crate::decoder::decoding::DecoderDeclaration;

/// Sample the public `B` and its trapdoor for `declaration`'s decoder of
/// `source_program` at its decoding level, with trapdoor sigma
/// `trapdoor_sigma`, in the preprocessing graph: the trapdoor never leaves
/// the stage, which publishes only `B` and the preimage of the decoder key.
/// Returns the trapdoor with the preimage coefficient bound.
pub(super) fn sample_unmask_trapdoor(
    parameters: &RnsBggParameters,
    declaration: &DecoderDeclaration,
    source_program: &ExponentLutProgram,
    trapdoor_sigma: &bigdecimal::BigDecimal,
) -> Result<(Trapdoor, BigUint), ApiError> {
    let level = RnsBggParameters::decoder_source_level(declaration, source_program)
        .map_err(ApiError::scheme)?;
    let sigma = trapdoor_sigma
        .to_f64()
        .ok_or_else(|| ApiError::input("the unmask trapdoor sigma is not finite"))?;
    let (base, digits, bound) = parameters.unmask_gadget(level, sigma).map_err(ApiError::scheme)?;
    let ring = parameters.ring(level).map_err(ApiError::scheme)?;
    let sigma = crate::utils::decimal_to_real_expr(trapdoor_sigma)
        .map_err(|_| ApiError::input("the unmask trapdoor sigma is not a decimal"))?;
    let trapdoor = crate::decoder::unmask::sample_trapdoor(
        &ring,
        sigma,
        base,
        digits,
        BigInt::from(bound.clone()),
    );
    Ok((trapdoor, bound))
}

/// The component-1 key column `(A_01; A_11)` of `matrix`.
fn decoded_key_column(matrix: &RnsBggMatrix) -> (Mat, Mat) {
    (matrix.0.1.clone(), matrix.1.1.clone())
}

/// The preprocessing stage's decoder: the summed unmask key `A`.
impl Decoder for RnsBggDecoder<'_, PublicKeyLowering<'_>> {
    /// The lowered smudge sum.
    type Smudge = Vec<RnsBggPublicKey>;
    /// The `2 x 1` summed key `A`.
    type Output = Mat;
    type Error = RnsBggError;

    fn finish_decoding(
        &mut self,
        declaration: &DecoderDeclaration,
        smudge: Self::Smudge,
    ) -> Result<Self::Output, Self::Error> {
        let parameters = self.lowerer.parameters;
        let level = self.lowerer.workload.level(self.source.wire)?;
        if self.source.wire != declaration.source_output() || smudge.len() != 1 {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        let ring = parameters.ring(level)?;
        let plaintext = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let selection = parameters.gadget_inverse(
            plaintext,
            level,
            parameters.ring(plaintext)?.polynomial([IntExpr::constant(1)]),
        )?;
        let (top, bottom) = decoded_key_column(&self.source.matrix);
        let (mut top, mut bottom) =
            (top.mul_small_rhs(selection.clone()), bottom.mul_small_rhs(selection));
        for key in smudge {
            let (smudge_top, smudge_bottom) = decoded_key_column(&key.matrix);
            top = top + checked_modulus_convert(smudge_top, &ring)?;
            bottom = bottom + checked_modulus_convert(smudge_bottom, &ring)?;
        }
        Ok(Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![top, bottom]))
    }
}
