use std::sync::Arc;

use bigdecimal::BigDecimal;
use digest::Digest;
use keccak_asm::Keccak256;
use num_bigint::{BigInt, BigUint};
use tracing::{debug, info};

use crate::{
    circuit::{Evaluable, PolyCircuit},
    gadgets::{
        arith::NestedRnsPolyContext,
        fhe::{ring_gsw::RingGswCiphertext, ring_gsw_nested_rns::NestedRnsRingGswContext},
        fhe_prg::goldreich::evaluate_goldreich_uniform_range,
    },
    input_injector::DiamondInputErrorSimulation,
    lookup::PltEvaluator,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    noise_refresh::{
        NoiseRefreshErrorSimulation,
        circuit_decrypt::{
            decrypt_bit_decomposed_polynomial, mask_plaintext_moduli_from_full_modulus,
        },
        simulate_noise_refresh_error_growth,
        simulation::validate_error_bound_security_margin,
    },
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    simulator::{
        SimulatorContext, error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};

use super::{DIAMOND_SECRET_SIZE, DiamondIO, DiamondIOFuncType};

/// Error-growth summary for the DiamondIO online path.
///
/// The simulation exposes the requested durable checkpoints: the Diamond input-injection state
/// errors, one representative noise-refresh simulation per PRF round, the final projection
/// residual before decoder addition, and the final noisy-plaintext error after decoder
/// cancellation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOErrorSimulation {
    pub input_injection: DiamondInputErrorSimulation,
    pub prf_refreshes: Vec<DiamondIOPrfRoundErrorSimulation>,
    pub projection_residual_error: PolyMatrixNorm,
    pub noisy_plaintext_error: PolyMatrixNorm,
}

/// Error-growth summary for one DiamondIO PRF seed-refresh round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOPrfRoundErrorSimulation {
    pub round_idx: usize,
    pub representative_selected_prg_output: ErrorNorm,
    pub noise_refresh: NoiseRefreshErrorSimulation,
}

/// Largest safe PRF mask bit-width together with the simulation that certified it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOPrfMaskOutputCoeffBitsSearchResult {
    /// Largest PRF mask coefficient bit-width satisfying the final decode margin.
    pub prf_mask_output_coeff_bits: usize,
    /// Error simulation evaluated with `prf_mask_output_coeff_bits`.
    pub simulation: DiamondIOErrorSimulation,
}

/// Successful DiamondIO CRT-depth search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOCrtDepthSearchResult {
    /// Smallest CRT depth found by the search.
    pub crt_depth: usize,
    /// Largest PRF mask coefficient bit-width that was safe at `crt_depth`.
    pub prf_mask_output_coeff_bits: usize,
    /// Final noisy-plaintext error bound certified for the selected candidate.
    pub total_noisy_plaintext_error: PolyMatrixNorm,
    /// Representative input-injection contribution used to seed the DiamondIO simulation.
    ///
    /// This is `input_injection.state_errors[0] * input_injection.output_preimage`, matching the
    /// projected state error used as the `one`, decryption-key, and decoder input bound.
    pub input_injection_projection_error: PolyMatrixNorm,
}

/// Finds the smallest CRT depth whose correctness and security checks pass.
///
/// The search is binary over the inclusive range `[min_crt_depth, max_crt_depth]`. For each
/// candidate depth, `build_candidate` must construct a DiamondIO instance with matching polynomial
/// parameters and Ring-GSW context. The candidate first runs
/// `DiamondIO::max_safe_prf_mask_output_coeff_bits`; if no mask width is valid, the search moves to
/// larger CRT depths. If mask search succeeds, the certified simulation is checked with
/// `validate_error_bound_security_margin` for both the final noisy-plaintext bound and every
/// noise-refresh pre-rounding bound. Passing candidates are recorded and the search continues
/// toward smaller CRT depths.
pub fn diamond_io_find_crt_depth<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST, PE, ST, BuildCandidate>(
    min_crt_depth: usize,
    max_crt_depth: usize,
    max_prf_mask_output_coeff_bits: usize,
    security_bit: usize,
    func_type: DiamondIOFuncType,
    mut build_candidate: BuildCandidate,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<DiamondIOCrtDepthSearchResult>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
    BuildCandidate: FnMut(usize) -> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
{
    info!(
        min_crt_depth,
        max_crt_depth,
        max_prf_mask_output_coeff_bits,
        security_bit,
        "starting DiamondIO CRT-depth search"
    );
    assert!(min_crt_depth > 0, "minimum CRT depth must be positive");
    assert!(min_crt_depth <= max_crt_depth, "CRT-depth search range must be non-empty");
    let mut low = min_crt_depth;
    let mut high = max_crt_depth;
    let mut found = Vec::new();
    while low <= high {
        let crt_depth = low + (high - low) / 2;
        info!(crt_depth, low, high, "evaluating DiamondIO CRT-depth candidate");
        let candidate = build_candidate(crt_depth);
        let Some(mask_search) = candidate.max_safe_prf_mask_output_coeff_bits(
            func_type,
            plt_evaluator,
            slot_transfer_evaluator,
            max_prf_mask_output_coeff_bits,
        ) else {
            info!(crt_depth, "DiamondIO CRT-depth candidate failed correctness mask-bit search");
            low = crt_depth + 1;
            continue;
        };
        let mask_bits = mask_search.prf_mask_output_coeff_bits;
        let cpu_params = cpu_params_from_poly_params(&candidate.injector.params);
        if diamond_io_security_margins_hold(
            &cpu_params,
            &mask_search.simulation,
            mask_bits,
            security_bit,
        ) {
            info!(crt_depth, mask_bits, "DiamondIO CRT-depth candidate passed");
            found.push(DiamondIOCrtDepthSearchResult {
                crt_depth,
                prf_mask_output_coeff_bits: mask_bits,
                total_noisy_plaintext_error: mask_search.simulation.noisy_plaintext_error.clone(),
                input_injection_projection_error: diamond_io_input_injection_projection_error(
                    &mask_search.simulation.input_injection,
                ),
            });
            high = crt_depth - 1;
        } else {
            info!(crt_depth, mask_bits, "DiamondIO CRT-depth candidate failed security margins");
            low = crt_depth + 1;
        }
    }
    let result = found.into_iter().min_by_key(|candidate| candidate.crt_depth);
    info!(found = result.is_some(), "finished DiamondIO CRT-depth search");
    result
}

impl<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
    US: PolyUniformSampler<M = M> + Send + Sync,
    HS: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    TS: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    /// Simulate DiamondIO error growth for the selected function family.
    ///
    /// Goldreich PRG calls are evaluated only for one representative output ciphertext. Those
    /// representative bounds are simulated with one active q-level and extrapolated to the
    /// configured full active-level count.
    pub fn simulate_error_growth<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOErrorSimulation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        self.simulate_error_growth_with_prf_mask_output_coeff_bits(
            func_type,
            self.prf_mask_output_coeff_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )
    }

    /// Search for the largest PRF mask coefficient bit-width that satisfies the final decode
    /// margin.
    ///
    /// Each candidate bit-width is evaluated with a fresh DiamondIO error simulation, so the
    /// returned simulation accounts for that candidate's final PRF mask PRG and mask
    /// decrypt/recomposition error. A candidate is valid exactly when
    /// `noisy_plaintext_error + 2^candidate < q / 4`.
    pub fn max_safe_prf_mask_output_coeff_bits<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
        max_bits: usize,
    ) -> Option<DiamondIOPrfMaskOutputCoeffBitsSearchResult>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let cpu_params = cpu_params_from_poly_params(&self.injector.params);
        let max_candidate = max_bits.min(cpu_params.modulus_bits());
        let mut low = 1usize;
        let mut high = max_candidate;
        let mut best = None;
        info!(max_bits, max_candidate, "starting DiamondIO PRF mask bit search");
        while low <= high {
            let candidate = low + (high - low) / 2;
            info!(candidate, low, high, "evaluating DiamondIO PRF mask bit candidate");
            let simulation = self.simulate_error_growth_with_prf_mask_output_coeff_bits(
                func_type,
                candidate,
                plt_evaluator,
                slot_transfer_evaluator,
            )?;
            let q: Arc<BigUint> = cpu_params.modulus().into();
            let quarter_q = biguint_to_decimal(&(q.as_ref() / 4u32));
            let mask_value = biguint_to_decimal(&(BigUint::from(1u32) << candidate));
            let valid = &simulation.noisy_plaintext_error.poly_norm.norm + &mask_value < quarter_q;
            debug!(
                candidate,
                valid,
                noisy_plaintext_error = %simulation.noisy_plaintext_error.poly_norm.norm,
                mask_value = %mask_value,
                quarter_q = %quarter_q,
                "DiamondIO PRF mask bit candidate evaluated"
            );
            if valid {
                best = Some(DiamondIOPrfMaskOutputCoeffBitsSearchResult {
                    prf_mask_output_coeff_bits: candidate,
                    simulation,
                });
                low = candidate + 1;
            } else if candidate == 1 {
                break;
            } else {
                high = candidate - 1;
            }
        }
        info!(found = best.is_some(), "finished DiamondIO PRF mask bit search");
        best
    }

    fn simulate_error_growth_with_prf_mask_output_coeff_bits<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        prf_mask_output_coeff_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOErrorSimulation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        assert_eq!(
            self.input_size,
            self.injector.input_count * self.injector.batch_bits(),
            "DiamondIO input_size must match the injector bit input count"
        );
        assert!(self.seed_bits > 0, "DiamondIO error simulation requires seed_bits > 0");
        assert!(
            prf_mask_output_coeff_bits > 0,
            "DiamondIO error simulation requires prf_mask_output_coeff_bits > 0"
        );

        let cpu_params = cpu_params_from_poly_params(&self.injector.params);
        let (_, _, crt_depth) = cpu_params.to_crt();
        let full_active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        assert!(
            full_active_levels > 0 && self.ring_gsw_level_offset + full_active_levels <= crt_depth,
            "DiamondIO Ring-GSW active levels must fit in the CRT modulus window"
        );
        let ctx = simulator_context(&cpu_params);
        let input_injection = self.injector.simulate_output_error_bounds();
        let projected_state_error = diamond_io_input_injection_projection_error(&input_injection);
        info!(
            state_count = input_injection.state_errors.len(),
            projected_state_error_bits =
                bigdecimal_bits_ceil(&projected_state_error.poly_norm.norm),
            "DiamondIO input-injection error simulation finished"
        );
        debug!(state_errors = ?input_injection.state_errors, "DiamondIO simulated state errors");

        let one = ErrorNorm::new(PolyNorm::one(ctx.clone()), projected_state_error.clone());
        let decryption_key = ErrorNorm::new(PolyNorm::one(ctx.clone()), projected_state_error);
        let decoder_error = ErrorNorm::new(PolyNorm::one(ctx.clone()), one.matrix_norm.clone());
        // Noise-refresh online decoding material is indexed by `(slot_idx, crt_idx)`.
        // DiamondIO persists one refresh decoder for every slot and every CRT
        // level, so the simulator supplies the representative decoder bound in
        // the same `slot_idx * crt_depth + crt_idx` layout expected by
        // `simulate_noise_refresh_error_growth`.
        let decoders =
            vec![decoder_error.clone(); cpu_params.ring_dimension() as usize * crt_depth];
        let seed_wire_error = self.initial_seed_wire_error(&cpu_params, &one);
        let mut seed_errors = vec![seed_wire_error.clone(); self.seed_bits];
        let mut prf_refreshes = Vec::with_capacity(self.input_size);

        for round_idx in 0..self.input_size {
            let low = self.simulate_representative_prg_output(
                &cpu_params,
                round_idx,
                2 * self.seed_bits,
                0,
                one.clone(),
                seed_errors.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
            );
            let high = self.simulate_representative_prg_output(
                &cpu_params,
                round_idx,
                2 * self.seed_bits,
                self.seed_bits,
                one.clone(),
                seed_errors.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
            );
            let selected = simulate_selected_half_error_norm(one.clone(), low, high, one.clone());
            let mut refresh_context_circuit = PolyCircuit::<DCRTPoly>::new();
            let refresh = simulate_noise_refresh_error_growth(
                self.build_cpu_ring_gsw_context(
                    &cpu_params,
                    &mut refresh_context_circuit,
                    full_active_levels,
                ),
                self.seed_bits,
                self.goldreich_graph_seed,
                self.noise_refresh_cbd_n,
                self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
                cpu_params.ring_dimension() as usize,
                one.clone(),
                selected.clone(),
                &seed_errors,
                decryption_key.clone(),
                &decoders,
                plt_evaluator,
                slot_transfer_evaluator,
            )?;
            let worst_rounded_error = refresh
                .rounded_errors
                .iter()
                .max_by(|lhs, rhs| {
                    lhs.poly_norm
                        .norm
                        .partial_cmp(&rhs.poly_norm.norm)
                        .expect("DiamondIO rounded errors must be comparable")
                })
                .expect("noise refresh must produce rounded errors")
                .clone();
            let refreshed_seed =
                ErrorNorm::new(worst_rounded_error.poly_norm.clone(), worst_rounded_error);
            seed_errors = vec![refreshed_seed.clone(); self.seed_bits];
            info!(
                round_idx,
                v_bits = refresh.v_bits,
                refreshed_seed_error_bits =
                    bigdecimal_bits_ceil(&refreshed_seed.matrix_norm.poly_norm.norm),
                "DiamondIO PRF refresh error simulation finished"
            );
            prf_refreshes.push(DiamondIOPrfRoundErrorSimulation {
                round_idx,
                representative_selected_prg_output: selected,
                noise_refresh: refresh,
            });
        }

        let final_mask_prg_output = self.simulate_representative_prg_output(
            &cpu_params,
            self.input_size,
            self.output_size
                .checked_mul(cpu_params.ring_dimension() as usize)
                .and_then(|count| count.checked_mul(prf_mask_output_coeff_bits))
                .expect("DiamondIO final mask PRG conceptual output count overflow"),
            0,
            one.clone(),
            seed_errors,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let final_mask_error = self.simulate_final_mask_error(
            &cpu_params,
            prf_mask_output_coeff_bits,
            final_mask_prg_output,
            one.clone(),
            decryption_key.clone(),
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let (function_secret_error, function_public_bottom_error) = self
            .simulate_representative_function_output(
                func_type,
                &cpu_params,
                one.clone(),
                decryption_key,
                seed_wire_error,
                plt_evaluator,
                slot_transfer_evaluator,
            );
        let evaluated_secret_dependent = function_secret_error + &final_mask_error;
        let projected_evaluated_error = evaluated_secret_dependent.matrix_norm.clone() *
            PolyMatrixNorm::gadget_decomposed(ctx.clone(), 1);
        let public_bottom_decryption_error =
            self.simulate_public_bottom_decryption_error(&cpu_params, ctx.clone());
        let projection_residual_error = projected_evaluated_error.clone() +
            &function_public_bottom_error.matrix_norm +
            &public_bottom_decryption_error;
        let noisy_plaintext_error = projection_residual_error.clone() + &decoder_error.matrix_norm;
        info!(
            projected_evaluated_bits =
                bigdecimal_bits_ceil(&projected_evaluated_error.poly_norm.norm),
            public_bottom_bits =
                bigdecimal_bits_ceil(&function_public_bottom_error.matrix_norm.poly_norm.norm),
            public_bottom_decryption_bits =
                bigdecimal_bits_ceil(&public_bottom_decryption_error.poly_norm.norm),
            projection_residual_bits =
                bigdecimal_bits_ceil(&projection_residual_error.poly_norm.norm),
            noisy_plaintext_bits = bigdecimal_bits_ceil(&noisy_plaintext_error.poly_norm.norm),
            "DiamondIO final output error simulation finished"
        );

        Some(DiamondIOErrorSimulation {
            input_injection,
            prf_refreshes,
            projection_residual_error,
            noisy_plaintext_error,
        })
    }

    fn initial_seed_wire_error(&self, _params: &DCRTPolyParams, one: &ErrorNorm) -> ErrorNorm {
        let max_p_modulus = self
            .ring_gsw_context
            .p_moduli
            .iter()
            .copied()
            .max()
            .expect("nested-RNS Ring-GSW context must have at least one p modulus");
        let scalar_bound = max_p_modulus - 1;
        // Ring-GSW native ciphertext inputs are first encoded as nested-RNS
        // p-residue wires via `ciphertext_inputs_from_native`. Each BGG lift
        // therefore multiplies by one residue modulo some p_i, not by a full
        // native q coefficient, so max(p_i) - 1 is the tight scalar bound.
        let scalar_bound = BigUint::from(scalar_bound);
        one.large_scalar_mul(&(), &[scalar_bound])
    }

    fn build_cpu_ring_gsw_context(
        &self,
        params: &DCRTPolyParams,
        circuit: &mut PolyCircuit<DCRTPoly>,
        active_levels: usize,
    ) -> Arc<NestedRnsRingGswContext<DCRTPoly>> {
        let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            params,
            self.ring_gsw_context.p_moduli_bits,
            self.ring_gsw_context.max_unreduced_muls,
            self.ring_gsw_context.scale,
            false,
            Some(active_levels),
        ));
        Arc::new(NestedRnsRingGswContext::<DCRTPoly>::from_arith_context(
            circuit,
            params,
            params.ring_dimension() as usize,
            nested_rns_context,
            Some(active_levels),
            Some(self.ring_gsw_level_offset),
        ))
    }

    fn simulate_representative_prg_output<PE, ST>(
        &self,
        params: &DCRTPolyParams,
        round_idx: usize,
        conceptual_output_bits: usize,
        range_start: usize,
        one: ErrorNorm,
        seed_errors: Vec<ErrorNorm>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> ErrorNorm
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let circuit = self.build_cpu_goldreich_prg_range_circuit(
            params,
            round_idx,
            conceptual_output_bits,
            range_start,
            1,
            1,
        );
        let wire_count = ring_gsw_wire_count(circuit.num_input(), self.seed_bits);
        let seed_wire_errors = expand_logical_errors(&seed_errors, wire_count);
        let mut output = eval_first_error_output(
            circuit,
            one,
            seed_wire_errors,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator),
        );
        let (_, _, crt_depth) = params.to_crt();
        let full_active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        output.matrix_norm = output.matrix_norm * BigDecimal::from(full_active_levels as u64);
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            simulated_active_levels = 1usize,
            extrapolated_active_levels = full_active_levels,
            output_error_bits = bigdecimal_bits_ceil(&output.matrix_norm.poly_norm.norm),
            "DiamondIO representative Goldreich PRG error simulated"
        );
        output
    }

    fn build_cpu_goldreich_prg_range_circuit(
        &self,
        params: &DCRTPolyParams,
        round_idx: usize,
        conceptual_output_bits: usize,
        range_start: usize,
        range_len: usize,
        active_levels: usize,
    ) -> PolyCircuit<DCRTPoly> {
        assert!(self.seed_bits >= 5, "DiamondIO Goldreich PRF seed bit length must be at least 5");
        let mut circuit = PolyCircuit::new();
        let ring_gsw_context = self.build_cpu_ring_gsw_context(params, &mut circuit, active_levels);
        let seed_ciphertexts = (0..self.seed_bits)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let outputs = evaluate_goldreich_uniform_range(
            &mut circuit,
            ring_gsw_context,
            &seed_ciphertexts,
            conceptual_output_bits,
            range_start,
            range_len,
            self.derive_diamond_goldreich_graph_seed(round_idx),
        );
        circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
        circuit
    }

    fn simulate_final_mask_error<PE, ST>(
        &self,
        params: &DCRTPolyParams,
        prf_mask_output_coeff_bits: usize,
        final_mask_prg_output: ErrorNorm,
        one: ErrorNorm,
        decryption_key: ErrorNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> ErrorNorm
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let circuit = self.build_cpu_prf_mask_circuit(params, prf_mask_output_coeff_bits);
        let encrypted_bit_count = params.ring_dimension() as usize * prf_mask_output_coeff_bits;
        let wire_count = ring_gsw_wire_count(circuit.num_input() - 1, encrypted_bit_count);
        let mut inputs = Vec::with_capacity(circuit.num_input());
        inputs.push(decryption_key);
        inputs.extend(expand_logical_errors(
            &vec![final_mask_prg_output; encrypted_bit_count],
            wire_count,
        ));
        assert_eq!(
            inputs.len(),
            circuit.num_input(),
            "DiamondIO final mask error-simulation input count mismatch"
        );
        let mut outputs = circuit.eval(
            &(),
            one,
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
            None,
        );
        assert_eq!(outputs.len(), 1, "DiamondIO final mask circuit must produce one output");
        outputs.remove(0)
    }

    fn build_cpu_prf_mask_circuit(
        &self,
        params: &DCRTPolyParams,
        prf_mask_output_coeff_bits: usize,
    ) -> PolyCircuit<DCRTPoly> {
        let mut circuit = PolyCircuit::new();
        let (_, _, crt_depth) = params.to_crt();
        let active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        let ring_gsw_context = self.build_cpu_ring_gsw_context(params, &mut circuit, active_levels);
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let q: Arc<BigUint> = params.modulus().into();
        let plaintext_moduli =
            mask_plaintext_moduli_from_full_modulus(q.as_ref(), prf_mask_output_coeff_bits);
        let encrypted_bit_count = params.ring_dimension() as usize * prf_mask_output_coeff_bits;
        let encrypted_bits = (0..encrypted_bit_count)
            .map(|_| {
                RingGswCiphertext::input(
                    ring_gsw_context.clone(),
                    Some(BigUint::from(1u64)),
                    &mut circuit,
                )
            })
            .collect::<Vec<_>>();
        let decrypted = decrypt_bit_decomposed_polynomial::<
            DCRTPoly,
            crate::gadgets::arith::NestedRnsPoly<DCRTPoly>,
            DCRTPolyMatrix,
        >(&mut circuit, &encrypted_bits, decryption_key, &plaintext_moduli);
        circuit.output(vec![decrypted]);
        circuit
    }

    fn simulate_representative_function_output<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        params: &DCRTPolyParams,
        one: ErrorNorm,
        decryption_key: ErrorNorm,
        seed_wire_error: ErrorNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> (ErrorNorm, ErrorNorm)
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let circuit = match func_type {
            DiamondIOFuncType::DebugDecryption => {
                self.build_cpu_debug_decryption_circuit(params, 1)
            }
        };
        let seed_wire_count = circuit.num_input() - 1;
        let mut inputs = Vec::with_capacity(circuit.num_input());
        inputs.push(decryption_key);
        inputs.extend(std::iter::repeat_n(seed_wire_error, seed_wire_count));
        let outputs = circuit.eval(
            &(),
            one,
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
            None,
        );
        assert_eq!(outputs.len(), 2, "DiamondIO representative function output must be a pair");
        (outputs[0].clone(), outputs[1].clone())
    }

    fn build_cpu_debug_decryption_circuit(
        &self,
        params: &DCRTPolyParams,
        representative_seed_bits: usize,
    ) -> PolyCircuit<DCRTPoly> {
        let mut circuit = PolyCircuit::new();
        let (_, _, crt_depth) = params.to_crt();
        let active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        let ring_gsw_context = self.build_cpu_ring_gsw_context(params, &mut circuit, active_levels);
        let decryption_key = circuit.input(1).at(0).as_single_wire();
        let mut outputs = Vec::with_capacity(2 * representative_seed_bits);
        for _ in 0..representative_seed_bits {
            let ciphertext = RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            );
            let decrypted = ciphertext.decrypt::<DCRTPolyMatrix>(
                decryption_key,
                BigUint::from(2u64),
                &mut circuit,
            );
            outputs.push(decrypted.secret_dependent);
            outputs.push(decrypted.public_bottom);
        }
        circuit.output(outputs);
        circuit
    }

    fn simulate_public_bottom_decryption_error(
        &self,
        params: &DCRTPolyParams,
        output_ctx: Arc<SimulatorContext>,
    ) -> PolyMatrixNorm {
        let mut circuit = PolyCircuit::new();
        let (_, _, crt_depth) = params.to_crt();
        let active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        let ring_gsw_context = self.build_cpu_ring_gsw_context(params, &mut circuit, active_levels);
        let ciphertext =
            RingGswCiphertext::input(ring_gsw_context, Some(BigUint::from(1u64)), &mut circuit);
        let raw = ciphertext
            .estimate_decryption_error_norm(self.ring_gsw_public_key_error_sigma.unwrap_or(0.0));
        PolyMatrixNorm::new(output_ctx, 1, 1, raw.poly_norm.norm, None)
    }

    fn derive_diamond_goldreich_graph_seed(&self, round_idx: usize) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(b"DiamondIOGoldreichPrfGraph/v1");
        hasher.update(self.goldreich_graph_seed);
        hasher.update(round_idx.to_le_bytes());
        hasher.finalize().into()
    }
}

fn cpu_params_from_poly_params<P: PolyParams>(params: &P) -> DCRTPolyParams {
    let (q_moduli, crt_bits, crt_depth) = params.to_crt();
    let cpu_params =
        DCRTPolyParams::new(params.ring_dimension(), crt_depth, crt_bits, params.base_bits());
    let (cpu_q_moduli, cpu_crt_bits, cpu_crt_depth) = cpu_params.to_crt();
    assert_eq!(cpu_crt_bits, crt_bits, "CPU simulation CRT bit width mismatch");
    assert_eq!(cpu_crt_depth, crt_depth, "CPU simulation CRT depth mismatch");
    assert_eq!(
        cpu_q_moduli, q_moduli,
        "CPU simulation parameters must reproduce the DiamondIO CRT moduli"
    );
    cpu_params
}

fn simulator_context(params: &DCRTPolyParams) -> Arc<SimulatorContext> {
    let ring_dim_sqrt = BigDecimal::from(params.ring_dimension() as u64)
        .sqrt()
        .expect("sqrt(ring_dimension) failed");
    let base = BigDecimal::from(BigInt::from(BigUint::from(1u64) << params.base_bits()));
    Arc::new(SimulatorContext::new(
        ring_dim_sqrt,
        base,
        DIAMOND_SECRET_SIZE,
        params.modulus_digits(),
        params.modulus_digits(),
    ))
}

fn biguint_to_decimal(value: &BigUint) -> BigDecimal {
    BigDecimal::from(BigInt::from(value.clone()))
}

fn diamond_io_input_injection_projection_error(
    input_injection: &DiamondInputErrorSimulation,
) -> PolyMatrixNorm {
    input_injection
        .state_errors
        .first()
        .expect("DiamondIO input injection must produce a base final state error")
        .clone() *
        &input_injection.output_preimage
}

fn diamond_io_security_margins_hold(
    params: &DCRTPolyParams,
    simulation: &DiamondIOErrorSimulation,
    prf_mask_output_coeff_bits: usize,
    security_bit: usize,
) -> bool {
    let Some(final_threshold) = diamond_io_final_decode_margin(params, prf_mask_output_coeff_bits)
    else {
        return false;
    };
    if !validate_error_bound_security_margin(
        &final_threshold,
        &simulation.noisy_plaintext_error.poly_norm.norm,
        security_bit,
    ) {
        return false;
    }
    simulation.prf_refreshes.iter().all(|refresh| {
        let Some(threshold) =
            diamond_io_noise_refresh_pre_round_margin(params, refresh.noise_refresh.v_bits)
        else {
            return false;
        };
        let worst_error = refresh
            .noise_refresh
            .pre_round_outputs
            .iter()
            .map(|error| error.poly_norm.norm.clone())
            .max_by(|lhs, rhs| {
                lhs.partial_cmp(rhs).expect("noise-refresh pre-rounding norms must be comparable")
            })
            .expect("noise-refresh simulation must produce at least one pre-round output");
        validate_error_bound_security_margin(&threshold, &worst_error, security_bit)
    })
}

fn diamond_io_final_decode_margin(
    params: &DCRTPolyParams,
    prf_mask_output_coeff_bits: usize,
) -> Option<BigDecimal> {
    let q: Arc<BigUint> = params.modulus().into();
    let threshold = biguint_to_decimal(&(q.as_ref() / 4u32));
    let mask = biguint_to_decimal(&(BigUint::from(1u32) << prf_mask_output_coeff_bits));
    (threshold > mask).then_some(threshold - mask)
}

fn diamond_io_noise_refresh_pre_round_margin(
    params: &DCRTPolyParams,
    v_bits: usize,
) -> Option<BigDecimal> {
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    let threshold =
        biguint_to_decimal(full_q.as_ref()) / (BigDecimal::from(2u32) * BigDecimal::from(q_max));
    let mask = biguint_to_decimal(&(BigUint::from(1u32) << v_bits));
    (threshold > mask).then_some(threshold - mask)
}

fn eval_first_error_output<PE>(
    mut circuit: PolyCircuit<DCRTPoly>,
    one: ErrorNorm,
    inputs: Vec<ErrorNorm>,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "DiamondIO first-output error-simulation input count mismatch"
    );
    assert!(circuit.num_output() > 0, "DiamondIO representative circuit has no output");
    circuit.restrict_outputs_to_indices(&[0]);
    let mut outputs = circuit.eval(&(), one, inputs, plt_evaluator, slot_transfer_evaluator, None);
    assert_eq!(outputs.len(), 1, "DiamondIO representative evaluation must return one output");
    outputs.remove(0)
}

fn simulate_selected_half_error_norm(
    one: ErrorNorm,
    low: ErrorNorm,
    high: ErrorNorm,
    selector: ErrorNorm,
) -> ErrorNorm {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let low_wire = circuit.input(1).at(0).as_single_wire();
    let high_wire = circuit.input(1).at(0).as_single_wire();
    let selector_wire = circuit.input(1).at(0).as_single_wire();
    let high_minus_low = circuit.sub_gate(high_wire, low_wire);
    let selected_delta = circuit.mul_gate(selector_wire, high_minus_low);
    let selected = circuit.add_gate(low_wire, selected_delta);
    circuit.output([selected]);
    let mut outputs = circuit.eval(
        &(),
        one,
        vec![low, high, selector],
        None::<&crate::simulator::error_norm::NormPltLWEEvaluator>,
        None,
        None,
    );
    assert_eq!(outputs.len(), 1, "DiamondIO selector circuit must produce one output");
    outputs.remove(0)
}

fn expand_logical_errors(logical_errors: &[ErrorNorm], wire_count: usize) -> Vec<ErrorNorm> {
    assert!(wire_count > 0, "Ring-GSW wire count must be positive");
    logical_errors.iter().flat_map(|error| std::iter::repeat_n(error.clone(), wire_count)).collect()
}

fn ring_gsw_wire_count(flat_input_count: usize, logical_bit_count: usize) -> usize {
    assert!(logical_bit_count > 0, "logical Ring-GSW bit count must be positive");
    assert_eq!(
        flat_input_count % logical_bit_count,
        0,
        "flattened Ring-GSW input count must be divisible by logical bit count"
    );
    flat_input_count / logical_bit_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        func_enc::aky24::NoCircuitEvaluator,
        gadgets::arith::ModularArithmeticContext,
        input_injector::DiamondInjector,
        matrix::dcrt_poly::DCRTPolyMatrix,
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        simulator::error_norm::{NormNaiveBggEncodingVecSTEvaluator, NormPltLWEEvaluator},
    };
    use num_traits::FromPrimitive;

    type TestDiamondIO = DiamondIO<
        DCRTPolyMatrix,
        DCRTPolyUniformSampler,
        DCRTPolyHashSampler<Keccak256>,
        DCRTPolyTrapdoorSampler,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    >;

    fn test_scheme(active_levels: usize) -> TestDiamondIO {
        test_scheme_with_input_count(active_levels, 1)
    }

    fn test_scheme_with_input_count(active_levels: usize, input_count: usize) -> TestDiamondIO {
        let params = DCRTPolyParams::new(2, active_levels, 10, 5);
        let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &params,
            5,
            2,
            1 << 8,
            false,
            Some(active_levels),
        ));
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<DCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                Some(active_levels),
                Some(0),
            );
        let injector = DiamondInjector::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new(params.clone(), input_count, 2, 4.578, 0.0);
        TestDiamondIO::new(
            injector,
            input_count,
            5,
            params.clone(),
            ring_gsw_context,
            ring_gsw_width,
            0,
            Some(active_levels),
            Some(0.0),
            b"diamond_io_error_simulation_test".to_vec(),
            5,
            1,
            1,
            1,
            [0x24; 32],
            [0x42; 32],
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    #[ignore = "full DiamondIO error simulation is heavier than default unit tests"]
    fn test_simulate_error_growth_reuses_input_injector_state_errors() {
        let scheme = test_scheme_with_input_count(2, 0);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let expected = scheme.injector.simulate_output_error_bounds().state_errors;
        let simulation = scheme
            .simulate_error_growth(
                DiamondIOFuncType::DebugDecryption,
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("DiamondIO error simulation should find a noise-refresh mask size");

        assert_eq!(simulation.input_injection.state_errors, expected);
        assert!(simulation.prf_refreshes.is_empty());
    }

    #[test]
    #[ignore = "full noise-refresh path is intentionally heavier than default unit tests"]
    fn test_simulate_error_growth_records_one_refresh_per_input_bit() {
        let scheme = test_scheme(2);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let simulation = scheme
            .simulate_error_growth(
                DiamondIOFuncType::DebugDecryption,
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("DiamondIO error simulation should find a noise-refresh mask size");

        assert_eq!(simulation.prf_refreshes.len(), scheme.input_size);
    }

    #[test]
    fn test_goldreich_prg_helper_builds_one_logical_output() {
        let scheme = test_scheme(1);
        let params = cpu_params_from_poly_params(&scheme.injector.params);
        let one_output =
            scheme.build_cpu_goldreich_prg_range_circuit(&params, 0, 2 * scheme.seed_bits, 0, 1, 1);
        let two_outputs =
            scheme.build_cpu_goldreich_prg_range_circuit(&params, 0, 2 * scheme.seed_bits, 0, 2, 1);
        assert!(one_output.num_output() > 0);
        assert_eq!(two_outputs.num_output(), 2 * one_output.num_output());
    }

    #[test]
    #[ignore = "representative PRG ErrorNorm evaluation is heavier than default unit tests"]
    fn test_representative_prg_extrapolates_active_levels() {
        let scheme = test_scheme(2);
        let params = cpu_params_from_poly_params(&scheme.injector.params);
        let ctx = simulator_context(&params);
        let projected = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(1u64), None);
        let one = ErrorNorm::new(PolyNorm::one(ctx.clone()), projected);
        let seed_errors = vec![one.clone(); scheme.seed_bits];
        let plt_evaluator = NormPltLWEEvaluator::new(ctx, &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let mut unscaled = eval_first_error_output(
            scheme.build_cpu_goldreich_prg_range_circuit(&params, 0, 2 * scheme.seed_bits, 0, 1, 1),
            one.clone(),
            expand_logical_errors(
                &seed_errors,
                ring_gsw_wire_count(
                    scheme
                        .build_cpu_goldreich_prg_range_circuit(
                            &params,
                            0,
                            2 * scheme.seed_bits,
                            0,
                            1,
                            1,
                        )
                        .num_input(),
                    scheme.seed_bits,
                ),
            ),
            Some(&plt_evaluator),
            Some(&slot_transfer_evaluator),
        );
        unscaled.matrix_norm = unscaled.matrix_norm * BigDecimal::from(2u64);
        let scaled = scheme.simulate_representative_prg_output(
            &params,
            0,
            2 * scheme.seed_bits,
            0,
            one,
            seed_errors,
            &plt_evaluator,
            &slot_transfer_evaluator,
        );
        assert_eq!(scaled.matrix_norm, unscaled.matrix_norm);
    }
}
