//! Noise of the masked trapdoor-decoder refresh.
//!
//! The evaluator sees `x = d - c - m_enc = Delta_L mu + m + E` with
//!
//! `E = e_b K - e_c G_(L+1)^-1(1) - sum_members X^j eps_member G_L^-1(w)`,
//!
//! term by term with the encoding stage:
//!
//! - `e_c`: the source output's level-`L` error, under the column selection `G_(L+1)^-1(1)`, whose
//!   `l1` is the tower count of `q_(L+1)`;
//! - the mask members: the child evaluator lowers the mask Program with the same member transform
//!   as the encoding (one finalization, then [`super::RnsBggNoiseValue::constant_digit_action`]),
//!   and its family sums aggregate them with their provenance;
//! - `e_b K`: the decoder row's fresh error against the unknown preimage, bounded by `N m_B B_e
//!   B_K` (coefficient-wise: `m_B` products of `N`-term negacyclic convolutions), with second
//!   moment `N m_B sigma_b^2 B_K^2`, which assumes no zero mean of `K`.
//!
//! The parts are combined coherently: the mask carriers' errors are not
//! independent of each other or of the mask.
use super::*;
use crate::noise_refresh::refresh::MaskFamily;

/// The noise of one refresh, for the decoding margin `spacing = q_L / q_(L+1)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnsBggRefreshNoise {
    /// Commitment of the refresh declaration.
    pub declaration: [u8; 32],
    /// Worst-case coefficient bound of `E`.
    pub error: BigUint,
    /// Moments of `E`, when every part has a moment model.
    pub moments: Option<NoiseMoments>,
    /// The mask magnitude bound `P^d`.
    pub mask: BigUint,
    /// The decoding spacing `q_L / q_(L+1)`.
    pub spacing: BigUint,
}

impl RnsBggRefreshNoise {
    /// Whether `|m + E| < spacing / 2` holds for every mask and error.
    pub fn fits(&self) -> bool {
        (&self.mask + &self.error) * 2u8 < self.spacing
    }

    /// Whether the mask range hides the error by the concrete criterion of
    /// [`crate::noise_refresh::refresh`]: `P^d >= 2 observations E 2^bits`.
    pub fn hides(&self, observations: &BigUint, statistical_bits: usize) -> bool {
        self.mask >= (observations * &self.error * 2u8) << statistical_bits
    }
}

/// The noise stage's refresh: `source` is the lowered source output.
impl NoiseRefresh for super::super::RnsBggRefresh<'_, RnsBggNoiseEvaluator<'_>> {
    /// The two lowered family sums of the mask Program child.
    type Mask = Vec<RnsBggNoiseValue>;
    type Output = RnsBggRefreshNoise;
    type Error = RnsBggError;

    fn finish_noise_refresh(
        &mut self,
        declaration: &RefreshDeclaration,
        mask: Self::Mask,
    ) -> Result<Self::Output, Self::Error> {
        let evaluator = &*self.lowerer;
        let parameters = evaluator.parameters;
        let (wire, source) = &self.source;
        if *wire != declaration.source_output() || mask.len() != MaskFamily::ALL.len() {
            return Err(RnsBggError::Refresh(RefreshError::TargetMismatch));
        }
        let level = evaluator.workload.level(*wire)?;
        let plaintext = level.checked_add(1).ok_or(RnsBggError::Parameters("level overflow"))?;
        let selection = parameters.gadget_inverse_l1(plaintext, &[BigUint::one()])?;
        let data = source.clone().constant_digit_action(&selection, &BigUint::one());
        let (decoder_error, decoder_variance) = self.decoder_noise(level)?;
        let mut error = &data.bound.error + &decoder_error;
        let mut moments = data.moments.clone().map(|moments| {
            moments.add_with_covariance(
                &zero_mean_moments(decoder_variance),
                NoiseCorrelation::Unknown,
            )
        });
        for value in &mask {
            error += &value.bound.error;
            moments = match (moments, &value.moments) {
                (Some(left), Some(right)) => {
                    Some(left.add_with_covariance(right, NoiseCorrelation::Unknown))
                }
                _ => None,
            };
        }
        Ok(RnsBggRefreshNoise {
            declaration: *declaration.commitment(),
            error,
            moments,
            mask: declaration.mask_bound(),
            spacing: parameters.scale(level)?.clone(),
        })
    }
}

impl super::super::RnsBggRefresh<'_, RnsBggNoiseEvaluator<'_>> {
    /// `(N m_B B_e B_K, N m_B sigma_b^2 B_K^2)` of the decoder product
    /// `e_b K` at `level`, for the trapdoor sigma bound to the evaluator.
    fn decoder_noise(&self, level: usize) -> Result<(BigUint, NonnegativeRational), RnsBggError> {
        let evaluator = &*self.lowerer;
        let parameters = evaluator.parameters;
        let sigma = evaluator
            .refresh_trapdoor_sigma
            .ok_or(RnsBggError::Binding("the refresh trapdoor sigma is not bound"))?;
        let (_, digits, bound) = parameters.refresh_decoder_gadget(level, sigma)?;
        let terms = BigUint::from(parameters.level(level)?.ring_dimension()) *
            BigUint::from(crate::noise_refresh::decoder::DECODER_ROWS * (digits + 2));
        let error = &terms * parameters.encoding_error_bound() * &bound;
        let variance = parameters.encoding_error_variance().scaled(&(terms * bound.pow(2u32)));
        Ok((error, variance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_noise_fits_strictly_inside_half_the_spacing() {
        let noise = |error: u32, mask: u32, spacing: u32| RnsBggRefreshNoise {
            declaration: [0; 32],
            error: BigUint::from(error),
            moments: None,
            mask: BigUint::from(mask),
            spacing: BigUint::from(spacing),
        };
        assert!(noise(3, 16, 40).fits());
        assert!(!noise(4, 16, 40).fits(), "2 (16 + 4) = 40 is not below 40");
        assert!(noise(1, 16, 40).hides(&BigUint::one(), 3));
        assert!(!noise(2, 16, 40).hides(&BigUint::one(), 3));
    }
}
