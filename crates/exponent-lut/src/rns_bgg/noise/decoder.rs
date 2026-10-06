//! Noise of the smudged trapdoor decoder.
//!
//! The evaluator sees `x = d - c - m_enc - 2^-1 = Delta_L mu + m + E` (the
//! public `2^-1` removes the offset of the halved smudge weights) with
//!
//! `E = e_b K - e_c G_(L+1)^-1(1) - sum_members X^j eps_member G_L^-1(w)`,
//!
//! term by term with the encoding stage:
//!
//! - `e_c`: the source output's level-`L` error, under the column selection `G_(L+1)^-1(1)`, whose
//!   `l1` is the tower count of `q_(L+1)`;
//! - the smudge members: the child evaluator lowers the smudge Program with the same member
//!   transform as the encoding (one finalization, then
//!   [`super::RnsBggNoiseValue::constant_digit_action`] with the `l1` of the halved weights `w`),
//!   and its smudge sum aggregates them with their provenance;
//! - `e_b K`: the unmask hint's fresh error against the unknown preimage, bounded by `N m_B B_e
//!   B_K` (coefficient-wise: `m_B` products of `N`-term negacyclic convolutions), with second
//!   moment `N m_B sigma_b^2 B_K^2`, which assumes no zero mean of `K`.
//!
//! The parts are combined coherently: the smudge carriers' errors are not
//! independent of each other or of the smudge.
use super::*;

/// The noise of one decoder, for the decoding margin `spacing = q_L / q_(L+1)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggDecoderNoise {
    /// Commitment of the decoder declaration.
    pub declaration: [u8; 32],
    /// Worst-case coefficient bound of `E`.
    pub error: BigUint,
    /// Moments of `E`, when every part has a moment model.
    pub moments: Option<NoiseMoments>,
    /// The smudge magnitude bound `P^d / 2`: the smudge is uniform on
    /// `[-P^d/2, P^d/2)`.
    pub smudge: BigUint,
    /// The decoding spacing `q_L / q_(L+1)`.
    pub spacing: BigUint,
}

impl RnsBggDecoderNoise {
    /// Whether `|m + E| < spacing / 2` holds for every smudge and error.
    pub fn fits(&self) -> bool {
        (&self.smudge + &self.error) * 2u8 < self.spacing
    }

    /// Whether the smudge range `P^d` hides the error by the smudging-lemma
    /// ratio of [`crate::decoder::decoding`]: `P^d > E 2^bits`.
    pub fn hides(&self, statistical_bits: usize) -> bool {
        &self.smudge * 2u8 > &self.error << statistical_bits
    }
}

/// The noise stage's decoder: `source` is the lowered source output.
impl Decoder for super::super::RnsBggDecoder<'_, RnsBggNoiseEvaluator<'_>> {
    /// The lowered smudge sum of the smudge Program child.
    type Smudge = Vec<RnsBggNoiseValue>;
    type Output = RnsBggDecoderNoise;
    type Error = RnsBggError;

    fn finish_decoding(
        &mut self,
        declaration: &DecoderDeclaration,
        smudge: Self::Smudge,
    ) -> Result<Self::Output, Self::Error> {
        let evaluator = &*self.lowerer;
        let parameters = evaluator.parameters;
        let (wire, source) = &self.source;
        if *wire != declaration.source_output() || smudge.len() != 1 {
            return Err(RnsBggError::Decoder(DecoderError::TargetMismatch));
        }
        let level = evaluator.workload.level(*wire)?;
        let plaintext = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let selection = parameters.gadget_inverse_l1(plaintext, &[BigUint::one()])?;
        let data = source.clone().constant_digit_action(&selection, &BigUint::one());
        let (unmask_error, unmask_variance) = self.unmask_noise(level)?;
        let mut error = &data.bound.error + &unmask_error;
        let mut moments = data.moments.clone().map(|moments| {
            moments
                .add_with_covariance(&zero_mean_moments(unmask_variance), NoiseCorrelation::Unknown)
        });
        for value in &smudge {
            error += &value.bound.error;
            moments = match (moments, &value.moments) {
                (Some(left), Some(right)) => {
                    Some(left.add_with_covariance(right, NoiseCorrelation::Unknown))
                }
                _ => None,
            };
        }
        Ok(RnsBggDecoderNoise {
            declaration: *declaration.commitment(),
            error,
            moments,
            smudge: declaration.smudge_bound(),
            spacing: parameters.scale(level)?.clone(),
        })
    }
}

impl super::super::RnsBggDecoder<'_, RnsBggNoiseEvaluator<'_>> {
    /// `(N m_B B_e B_K, N m_B sigma_b^2 B_K^2)` of the unmask product
    /// `e_b K` at `level`, for the trapdoor sigma bound to the evaluator.
    fn unmask_noise(&self, level: usize) -> Result<(BigUint, NonnegativeRational), RnsBggError> {
        let evaluator = &*self.lowerer;
        let parameters = evaluator.parameters;
        let sigma = evaluator
            .unmask_trapdoor_sigma
            .ok_or(RnsBggError::Binding("the unmask trapdoor sigma is not bound"))?;
        let (_, digits, bound) = parameters.unmask_gadget(level, sigma)?;
        let terms = BigUint::from(parameters.level(level)?.ring_dimension()) *
            BigUint::from(crate::decoder::unmask::UNMASK_ROWS * (digits + 2));
        let error = &terms * parameters.encoding_error_bound() * &bound;
        let variance = parameters.encoding_error_variance().scaled(&(terms * bound.pow(2u32)));
        Ok((error, variance))
    }
}

/// The simulated noise of `declaration`'s decoder of output `output_index` of
/// `source` under `bindings`, with the smudge child on `smudge_parameters` and
/// the unmask trapdoor sigma `unmask_sigma`. The smudge inputs are bound by
/// their contract: the accumulator start and the public vector `a` are
/// monomials and the selectors are bits.
pub(crate) fn simulate_decoder_noise(
    parameters: &RnsBggParameters,
    smudge_parameters: &RnsBggParameters,
    source: &crate::program::ExponentLutProgram,
    bindings: &RnsBggNoiseBindings,
    declaration: &DecoderDeclaration,
    output_index: usize,
    unmask_sigma: f64,
) -> Result<RnsBggDecoderNoise, crate::api::ApiError> {
    use crate::api::ApiError;
    let source_workload = RnsBggDerivedWorkload::from_program(source).map_err(ApiError::scheme)?;
    let smudge_program = declaration.smudge().program();
    let smudge_workload =
        RnsBggDerivedWorkload::from_program(smudge_program).map_err(ApiError::scheme)?;
    let one = [num_bigint::BigInt::from(1u8)];
    let smudge_bindings = RnsBggNoiseBindings::from_inputs(
        smudge_program,
        &[
            RnsBggNoiseValue::sampled_encoding(smudge_parameters, b"smudge-start", &one),
            RnsBggNoiseValue::sampled_encoding(smudge_parameters, b"smudge-a", &one)
                .independent_family_members(),
        ],
        &vec![
            RnsBggNoiseRhs { payload_l1: BigUint::one(), payload_l2_squared: BigUint::one() };
            smudge_program.rhs_inputs().len()
        ],
    )
    .map_err(ApiError::scheme)?;
    let mut smudge =
        RnsNoiseBuild::new(smudge_parameters, smudge_program, &smudge_workload, &smudge_bindings)
            .map_err(ApiError::scheme)?;
    smudge.bind_decoder_smudge(declaration).map_err(ApiError::scheme)?;
    let trace = crate::api::noise_simulate(
        source,
        RnsNoiseBuild::new(parameters, source, &source_workload, bindings)
            .map_err(ApiError::scheme)?,
        |build, outputs| {
            let output = outputs
                .get(output_index)
                .cloned()
                .ok_or_else(|| ApiError::input("decoder output index is out of range"))?;
            build
                .apply_decoder(output_index, declaration, output, smudge, unmask_sigma)
                .map_err(ApiError::scheme)?;
            Ok(outputs)
        },
    )?;
    trace.decoder.ok_or_else(|| ApiError::input("the decoder noise was not composed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_noise_fits_strictly_inside_half_the_spacing() {
        let noise = |error: u32, smudge: u32, spacing: u32| RnsBggDecoderNoise {
            declaration: [0; 32],
            error: BigUint::from(error),
            moments: None,
            smudge: BigUint::from(smudge),
            spacing: BigUint::from(spacing),
        };
        assert!(noise(3, 16, 40).fits());
        assert!(!noise(4, 16, 40).fits(), "2 (16 + 4) = 40 is not below 40");
        // The range is 32: it hides 3 * 2^3 = 24 but not 4 * 2^3 = 32.
        assert!(noise(3, 16, 40).hides(3));
        assert!(!noise(4, 16, 40).hides(3));
    }
}
