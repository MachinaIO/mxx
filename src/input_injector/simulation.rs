use super::{DIAMOND_SECRET_SIZE, DiamondInjector};
use crate::{
    matrix::PolyMatrix,
    poly::PolyParams,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    simulator::{
        SimulatorContext, error_norm::compute_preimage_norm, poly_matrix_norm::PolyMatrixNorm,
    },
};
use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use num_traits::FromPrimitive;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error-growth summary for Diamond input insertion.
///
/// The simulator exposes the final one, k-input, bit-input, and decoder output
/// bounds. The intermediate selector, preimage, and state values are emitted
/// with `tracing::debug!` while the recurrence runs.
pub struct DiamondInputErrorSimulation {
    /// Final propagated error for the one-output encoding.
    pub one_error: PolyMatrixNorm,
    /// Final propagated error for the additional k-output encoding.
    pub k_error: PolyMatrixNorm,
    /// Final propagated error for each input-digit output.
    pub input_errors: Vec<PolyMatrixNorm>,
    /// Final propagated error for each decoder output.
    pub decoder_errors: Vec<PolyMatrixNorm>,
}

impl<M, US, HS, TS> DiamondInjector<M, US, HS, TS>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    /// Simulate how the original Gaussian error in `p_{epsilon,0}` and the
    /// injected Gaussian target errors in `K_{i,b,j}` contribute to the final
    /// one, k-input, bit-input, and decoder outputs.
    pub fn simulate_output_error_bounds(&self) -> DiamondInputErrorSimulation {
        let ring_dim_sqrt = BigDecimal::from(self.params.ring_dimension() as u64)
            .sqrt()
            .expect("sqrt(ring_dimension) failed");
        let base = BigDecimal::from(BigInt::from(BigUint::from(1u64) << self.params.base_bits()));
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt,
            base,
            DIAMOND_SECRET_SIZE,
            self.params.modulus_digits(),
            self.params.modulus_digits(),
        ));
        let state_cols = self.state_col_size(&self.params);
        let gadget_cols = self.gadget_col_size(&self.params);
        let initial_sigma = BigDecimal::from_f64(self.error_sigma)
            .expect("DiamondInjector error_sigma must be finite");
        let initial_state_error =
            PolyMatrixNorm::sample_gauss(ctx.clone(), 1, state_cols, initial_sigma);
        let preimage_norm = compute_preimage_norm(
            &ctx.ring_dim_sqrt,
            ctx.m_g as u64,
            &ctx.base,
            Some(self.state_row_size() / DIAMOND_SECRET_SIZE),
            None,
        );
        let transition_preimage =
            PolyMatrixNorm::new(ctx.clone(), state_cols, state_cols, preimage_norm.clone(), None);
        let output_preimage =
            PolyMatrixNorm::new(ctx.clone(), state_cols, gadget_cols, preimage_norm, None);
        let transition_target_error = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            self.state_row_size(),
            state_cols,
            BigDecimal::from_f64(self.error_sigma)
                .expect("DiamondInjector error_sigma must be finite"),
        );
        let regular_selector = PolyMatrixNorm::new(
            ctx.clone(),
            self.state_row_size(),
            self.state_row_size(),
            BigDecimal::from(1u64),
            None,
        );
        let base_selector = PolyMatrixNorm::new(
            ctx.clone(),
            self.state_row_size(),
            self.state_row_size(),
            BigDecimal::from(1u64),
            None,
        );
        let special_selector = PolyMatrixNorm::new(
            ctx.clone(),
            self.state_row_size(),
            self.state_row_size(),
            BigDecimal::from(1u64),
            Some(DIAMOND_SECRET_SIZE),
        );
        debug!(
            ?initial_state_error,
            ?transition_preimage,
            ?output_preimage,
            ?transition_target_error,
            ?regular_selector,
            ?base_selector,
            ?special_selector,
            "diamond input-insertion simulator parameters",
        );

        let mut secret_state_factors =
            vec![PolyMatrixNorm::new(ctx, 1, self.state_row_size(), BigDecimal::from(1u64), None)];
        let mut state_errors = vec![initial_state_error];
        debug!(
            ?secret_state_factors,
            ?state_errors,
            "diamond input-insertion simulator initial state",
        );

        for level in 1..=self.input_count {
            let mut next_secret_factors = secret_state_factors
                .iter()
                .enumerate()
                .map(|(state_idx, secret_factor)| {
                    let selector = if state_idx == 0 { &base_selector } else { &regular_selector };
                    secret_factor.clone() * selector
                })
                .collect::<Vec<_>>();
            let mut next_state_errors = secret_state_factors
                .iter()
                .zip(state_errors.iter())
                .map(|(secret_factor, state_error)| {
                    state_error.clone() * &transition_preimage +
                        secret_factor.clone() * &transition_target_error
                })
                .collect::<Vec<_>>();
            for _ in 0..self.batch_bits() {
                let born_secret_factor = secret_state_factors[0].clone() * &special_selector;
                let born_state_error = state_errors[0].clone() * &transition_preimage +
                    secret_state_factors[0].clone() * &transition_target_error;
                next_secret_factors.push(born_secret_factor);
                next_state_errors.push(born_state_error);
            }
            secret_state_factors = next_secret_factors;
            state_errors = next_state_errors;
            debug!(
                level,
                ?secret_state_factors,
                ?state_errors,
                "diamond input-insertion simulator advanced state",
            );
        }

        let one_error = state_errors[0].clone() * &output_preimage;
        let k_error = state_errors[0].clone() * &output_preimage;
        let input_errors = (0..self.input_bit_count())
            .map(|bit_idx| state_errors[bit_idx + 1].clone() * &output_preimage)
            .collect::<Vec<_>>();
        let decoder_errors = vec![one_error.clone(); self.decoder_count];
        debug!(
            ?one_error,
            ?k_error,
            ?input_errors,
            ?decoder_errors,
            "diamond input-insertion simulator final output bounds",
        );

        DiamondInputErrorSimulation { one_error, k_error, input_errors, decoder_errors }
    }
}
