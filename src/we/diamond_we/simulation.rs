use crate::{
    circuit::{Evaluable, PolyCircuit},
    input_injector::DiamondInputErrorSimulation,
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams, dcrt::poly::DCRTPoly},
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    simulator::{error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
    slot_transfer::SlotTransferEvaluator,
};
use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use tracing::{debug, info};

use super::DiamondWE;

/// Error-growth summary for the DiamondWE online decryption path.
///
/// DiamondWE does not use the DiamondIO PRF/noise-refresh path, so this
/// simulation records the Diamond input-injection bounds, the per-input BGG
/// encoding bounds used to evaluate the witness circuit, the circuit output
/// bound, and the final decoder cancellation bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondWEErrorSimulation {
    pub input_injection: DiamondInputErrorSimulation,
    pub one_encoding_error: ErrorNorm,
    pub k_encoding_error: ErrorNorm,
    pub witness_encoding_errors: Vec<ErrorNorm>,
    pub instance_encoding_errors: Vec<ErrorNorm>,
    pub circuit_output_error: ErrorNorm,
    pub dec_encoding_error: ErrorNorm,
    pub decoder_projection_error: PolyMatrixNorm,
    pub noisy_plaintext_error: PolyMatrixNorm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondWECrtDepthSearchResult {
    pub crt_depth: usize,
    pub simulation: DiamondWEErrorSimulation,
}

fn diamond_we_decode_margin(params: &<DCRTPoly as Poly>::Params) -> BigDecimal {
    let q: std::sync::Arc<BigUint> = params.modulus().into();
    BigDecimal::from(BigInt::from(q.as_ref() / 4u32))
}

fn diamond_we_correctness_margin_holds(
    simulation: &DiamondWEErrorSimulation,
    margin: &BigDecimal,
) -> bool {
    simulation.noisy_plaintext_error.poly_norm.norm < *margin
}

pub fn diamond_we_find_crt_depth<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST, PE, ST, BuildCandidate>(
    min_crt_depth: usize,
    max_crt_depth: usize,
    circuit: &PolyCircuit<DCRTPoly>,
    mut build_candidate: BuildCandidate,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&ST>,
) -> Option<DiamondWECrtDepthSearchResult>
where
    M: PolyMatrix<P = DCRTPoly>,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
    BuildCandidate: FnMut(usize) -> DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
{
    assert!(min_crt_depth > 0, "minimum CRT depth must be positive");
    assert!(min_crt_depth <= max_crt_depth, "CRT-depth search range must be non-empty");
    info!(
        min_crt_depth,
        max_crt_depth, "starting DiamondWE CRT-depth search with q/4 correctness margin"
    );
    let mut high = max_crt_depth;
    let upper_valid = loop {
        info!(crt_depth = high, "evaluating DiamondWE CRT-depth upper-bound candidate");
        let candidate = build_candidate(high);
        let slot_transfer_evaluator = slot_transfer_evaluator
            .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<ErrorNorm>);
        let simulation =
            candidate.simulate_error_growth(circuit, plt_evaluator, slot_transfer_evaluator);
        let margin = diamond_we_decode_margin(&candidate.injector.params);
        let valid = diamond_we_correctness_margin_holds(&simulation, &margin);
        debug!(
            crt_depth = high,
            valid,
            noisy_plaintext_error = %simulation.noisy_plaintext_error.poly_norm.norm,
            decode_margin = %margin,
            "DiamondWE CRT-depth upper-bound candidate evaluated"
        );
        if valid {
            break DiamondWECrtDepthSearchResult { crt_depth: high, simulation };
        }
        let next_high =
            high.checked_mul(2).expect("DiamondWE CRT-depth search upper bound overflowed usize");
        assert!(next_high > high, "DiamondWE CRT-depth search upper bound must grow");
        info!(
            old_max_crt_depth = high,
            new_max_crt_depth = next_high,
            "expanding DiamondWE CRT-depth search range after failed q/4 correctness margin"
        );
        high = next_high;
    };

    let mut low = min_crt_depth;
    let mut result = Some(upper_valid);
    while low <= high {
        let crt_depth = low + (high - low) / 2;
        info!(crt_depth, low, high, "evaluating DiamondWE CRT-depth candidate");
        let candidate = build_candidate(crt_depth);
        let slot_transfer_evaluator = slot_transfer_evaluator
            .map(|evaluator| evaluator as &dyn SlotTransferEvaluator<ErrorNorm>);
        let simulation =
            candidate.simulate_error_growth(circuit, plt_evaluator, slot_transfer_evaluator);
        let margin = diamond_we_decode_margin(&candidate.injector.params);
        let valid = diamond_we_correctness_margin_holds(&simulation, &margin);
        debug!(
            crt_depth,
            valid,
            noisy_plaintext_error = %simulation.noisy_plaintext_error.poly_norm.norm,
            decode_margin = %margin,
            "DiamondWE CRT-depth candidate evaluated"
        );
        if valid {
            result = Some(DiamondWECrtDepthSearchResult { crt_depth, simulation });
            if crt_depth == min_crt_depth {
                break;
            }
            high = crt_depth - 1;
        } else {
            low = crt_depth + 1;
        }
    }
    info!(found = result.is_some(), "finished DiamondWE CRT-depth search");
    result
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondWE<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix<P = DCRTPoly>,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    /// Simulate DiamondWE decryption error growth for one circuit.
    ///
    /// Public instance inputs are simulated as all-true bits. The number of
    /// instance inputs is derived as `circuit.num_input() - witness_size`.
    ///
    /// `plt_evaluator` and `slot_transfer_evaluator` should match the norm
    /// evaluators for any public-lookup or slot-transfer gates used by
    /// `circuit`. They may be `None` for circuits without those gate kinds.
    pub fn simulate_error_growth<PE>(
        &self,
        circuit: &PolyCircuit<DCRTPoly>,
        plt_evaluator: Option<&PE>,
        slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
    ) -> DiamondWEErrorSimulation
    where
        PE: PltEvaluator<ErrorNorm>,
    {
        assert_eq!(
            circuit.num_output(),
            1,
            "DiamondWE error simulation currently requires one circuit output"
        );
        assert!(
            circuit.num_input() >= self.witness_size,
            "DiamondWE error-simulation input count must be at least witness_size"
        );
        assert_eq!(
            self.witness_size,
            self.injector.input_count * self.injector.batch_bits(),
            "DiamondWE witness_size must match the DiamondInjector bit input count"
        );

        let input_injection = self.injector.simulate_output_error_bounds();
        let ctx = input_injection.output_preimage.clone_ctx();
        let one_plaintext = PolyNorm::one(ctx.clone());
        let q = self.injector.params.modulus();
        let q: std::sync::Arc<num_bigint::BigUint> = q.into();
        let hidden_plaintext =
            PolyNorm::constant(ctx.clone(), BigDecimal::from(BigInt::from(q.as_ref() / 2u32)));
        let k_preimage = PolyMatrixNorm::new(
            ctx.clone(),
            input_injection.output_preimage.nrow,
            1,
            input_injection.output_preimage.poly_norm.norm.clone(),
            None,
        );

        let projected_state_error = |state_idx: usize| {
            input_injection.state_errors[state_idx].clone() * &input_injection.output_preimage
        };
        let projected_k_error = input_injection.state_errors[0].clone() * &k_preimage;
        let one_encoding_error = ErrorNorm::new(one_plaintext.clone(), projected_state_error(0));
        let k_encoding_error = ErrorNorm::new(hidden_plaintext.clone(), projected_k_error);

        let witness_encoding_errors = (0..self.witness_size)
            .map(|bit_idx| {
                let digit_idx = bit_idx / self.injector.batch_bits();
                let bit_in_digit = bit_idx % self.injector.batch_bits();
                let state_idx = self.injector.bit_state_idx(digit_idx, bit_in_digit);
                ErrorNorm::new(one_plaintext.clone(), projected_state_error(state_idx))
            })
            .collect::<Vec<_>>();
        let instance_size = circuit.num_input() - self.witness_size;
        let instance_encoding_errors = (0..instance_size)
            .map(|_| one_encoding_error.small_scalar_mul(&(), &[1]))
            .collect::<Vec<_>>();

        let mut circuit_inputs = witness_encoding_errors.clone();
        circuit_inputs.extend(instance_encoding_errors.clone());
        let mut outputs = circuit.eval(
            &(),
            one_encoding_error.clone(),
            circuit_inputs,
            plt_evaluator,
            slot_transfer_evaluator,
            None,
        );
        assert_eq!(
            outputs.len(),
            1,
            "DiamondWE error simulation expects exactly one circuit output"
        );
        let circuit_output_error = outputs.remove(0);
        let difference = one_encoding_error.clone() - &circuit_output_error;
        let difference_ctx = difference.clone_ctx();
        let r_scaled_difference =
            difference.matrix_norm * &PolyMatrixNorm::gadget_decomposed(difference_ctx, 1);
        let dec_encoding_matrix = k_encoding_error.matrix_norm.clone() + &r_scaled_difference;
        let decoder_projection_error = input_injection.state_errors[0].clone() * &k_preimage;
        let noisy_plaintext_error = decoder_projection_error.clone() + &dec_encoding_matrix;

        DiamondWEErrorSimulation {
            input_injection,
            one_encoding_error,
            k_encoding_error,
            witness_encoding_errors,
            instance_encoding_errors,
            circuit_output_error,
            dec_encoding_error: ErrorNorm::new(hidden_plaintext, dec_encoding_matrix),
            decoder_projection_error,
            noisy_plaintext_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use keccak_asm::Keccak256;
    use tempfile::tempdir;

    use super::*;
    use crate::{
        circuit::PolyCircuit,
        func_enc::NoCircuitEvaluator,
        input_injector::DiamondInjector,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::dcrt::params::DCRTPolyParams,
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
    };

    #[test]
    fn test_diamond_we_error_simulation_records_injection_and_noisy_plaintext() {
        let params = DCRTPolyParams::default();
        let witness_size = 2;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(witness_size + 1).to_vec();
        let output = circuit.or_gate(inputs[1], inputs[2]);
        circuit.output(vec![output]);

        let injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params, 1, 4, 2, 4.578, 0.0);
        let dir = tempdir().expect("temporary DiamondWE artifact directory should be created");
        let we =
            DiamondWE::new(injector, witness_size, dir.path(), b"diamond_we_sim_test".to_vec());

        let simulation = we.simulate_error_growth(&circuit, None::<&NoCircuitEvaluator>, None);
        assert_eq!(simulation.input_injection.state_errors.len(), 1 + witness_size);
        assert_eq!(simulation.witness_encoding_errors.len(), witness_size);
        assert_eq!(simulation.instance_encoding_errors.len(), 1);
        assert_eq!(simulation.noisy_plaintext_error.nrow, 1);
        assert_eq!(
            simulation.noisy_plaintext_error.ncol,
            simulation.dec_encoding_error.matrix_norm.ncol
        );

        let difference = simulation.one_encoding_error.clone() - &simulation.circuit_output_error;
        let difference_ctx = difference.clone_ctx();
        let expected_r_scaled_difference =
            difference.matrix_norm * &PolyMatrixNorm::gadget_decomposed(difference_ctx, 1);
        let expected_dec_encoding_matrix =
            simulation.k_encoding_error.matrix_norm.clone() + &expected_r_scaled_difference;
        assert_eq!(simulation.dec_encoding_error.matrix_norm, expected_dec_encoding_matrix);
        assert_eq!(
            simulation.noisy_plaintext_error,
            simulation.decoder_projection_error.clone() +
                &simulation.dec_encoding_error.matrix_norm
        );
    }
}
