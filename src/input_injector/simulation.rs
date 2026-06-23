use super::{DIAMOND_SECRET_SIZE, DiamondInjector};
use crate::{
    matrix::PolyMatrix,
    poly::PolyParams,
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    simulator::{
        SimulatorContext, error_norm::compute_preimage_sigma, poly_matrix_norm::PolyMatrixNorm,
    },
};
use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use num_traits::FromPrimitive;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Error-growth summary for Diamond state insertion.
///
/// The simulator exposes raw sigma-mode error norms for the final Diamond
/// states and the generic one-step final projection preimage sigma. Callers
/// that own output projections combine these values internally, then convert
/// to a maximum coefficient bound at public correctness/reporting boundaries.
pub struct DiamondInputErrorSimulation {
    /// Final propagated error for each Diamond state branch.
    pub state_errors: Vec<PolyMatrixNorm>,
    /// Final secret-selector norm for each Diamond state branch.
    ///
    /// These factors model the product of the secret transition matrices that
    /// multiply the sampled Gaussian target errors. Callers that need a norm
    /// for the final online secret, such as noise-refresh rounding analysis,
    /// should use the factor for the branch they decode from.
    pub secret_state_factors: Vec<PolyMatrixNorm>,
    /// Generic final projection preimage sigma from the final state basis to a
    /// single BGG output public key.
    pub output_preimage: PolyMatrixNorm,
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
    /// Diamond states.
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
        let preimage_sigma = compute_preimage_sigma(
            &ctx.ring_dim_sqrt,
            ctx.m_g as u64,
            &ctx.base,
            Some(self.state_row_size() / DIAMOND_SECRET_SIZE),
            None,
        );
        let transition_preimage =
            PolyMatrixNorm::new(ctx.clone(), state_cols, state_cols, preimage_sigma.clone(), None);
        let output_preimage =
            PolyMatrixNorm::new(ctx.clone(), state_cols, gadget_cols, preimage_sigma, None);
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

        debug!(
            ?secret_state_factors,
            ?state_errors,
            ?output_preimage,
            "diamond input-insertion simulator final state sigmas",
        );

        DiamondInputErrorSimulation { state_errors, secret_state_factors, output_preimage }
    }
}
