use crate::{
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
    },
    lookup::PltEvaluator,
    matrix::dcrt_poly::DCRTPolyMatrix,
    noise_refresh::{
        circuit_prg::{
            evaluate_goldreich_noise_refresh_material, goldreich_noise_refresh_output_sizes,
        },
        naive_vec::build_noise_refresh_material_circuit,
    },
    poly::{PolyParams, dcrt::poly::DCRTPoly},
    simulator::{SimulatorContext, error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm},
    slot_transfer::SlotTransferEvaluator,
};
use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use rayon::prelude::*;
use std::sync::Arc;

/// Returns the largest mask bit-size that is guaranteed to round away.
///
/// The noise-refresh mask is decoded as a non-scaled binary integer.  To make sure this mask and
/// the accumulated pre-rounding perturbation disappear when the `q/q_i`-scaled term is rounded back
/// modulo every CRT factor, the conservative sufficient condition is:
///
/// ```text
/// 2^v_bits + rounding_error + pre_rounding_error <= full_q / (2 * q_max)
/// ```
///
/// where `full_q` is the full DCRT modulus and `q_max` is the largest CRT factor.  The
/// `rounding_error` term is modeled as `secret_norm * 1`, so this helper uses
/// `secret_norm.poly_norm.norm` directly.
pub fn max_safe_noise_refresh_v_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    secret_norm: &PolyMatrixNorm,
    pre_rounding_error: &PolyMatrixNorm,
) -> Option<usize> {
    debug_assert_eq!(secret_norm.ctx(), pre_rounding_error.ctx());
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    let bound =
        biguint_to_decimal(full_q.as_ref()) / (BigDecimal::from(2u32) * BigDecimal::from(q_max));
    let available = bound - &secret_norm.poly_norm.norm - &pre_rounding_error.poly_norm.norm;
    if available < BigDecimal::from(1u32) {
        return None;
    }

    let mut best = 0usize;
    for candidate in 1..=params.modulus_bits() {
        if biguint_to_decimal(&(BigUint::from(1u32) << candidate)) <= available {
            best = candidate;
        } else {
            break;
        }
    }
    Some(best)
}

/// Checks whether a proposed mask bit-size satisfies the conservative rounding-away bound.
pub fn validate_noise_refresh_v_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    v_bits: usize,
    secret_norm: &PolyMatrixNorm,
    pre_rounding_error: &PolyMatrixNorm,
) -> bool {
    max_safe_noise_refresh_v_bits(params, secret_norm, pre_rounding_error)
        .is_some_and(|max_v_bits| v_bits <= max_v_bits)
}

/// Error-growth summary for one noise-refresh material evaluation.
///
/// `pre_round_outputs` preserves the online CRT-level layout:
/// `slot_idx * crt_depth + crt_idx`.
///
/// Each entry is the simulated norm of the full `log_base_q`-column vector immediately before the
/// online path rounds that CRT level by `q_i / q`. This includes the PRG/decrypt/merge material,
/// the slot-collapse and column-concatenation used to form `crt_level_vectors`, the
/// `G^{-1}((q/q_i) * A')` one-term subtraction, the `G^{-1}((q/q_i) * G)` refreshed-input term, and
/// the decoder subtraction. It also includes the Ring-GSW FHE decryption error `e'`: after
/// decrypting PRG material, the residual `e' * G` becomes an ordinary BGG+ encoding error once the
/// decoded material is multiplied by `G^{-1}(identity_1)`.
///
/// `rounded_errors` is the norm of the PRG-sampled CBD error plaintext that survives rounding.
/// It intentionally does not use circuit-evaluation error: masks and pre-existing input/decoder
/// perturbations are chosen to round away, while the scaled CBD plaintext rounds back to itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseRefreshErrorSimulation {
    pub v_bits: usize,
    pub pre_round_outputs: Vec<PolyMatrixNorm>,
    pub rounded_errors: Vec<PolyMatrixNorm>,
}

/// Simulates noise growth and chooses the largest mask bit-size that still rounds away.
///
/// `v_bits` affects the mask decrypt circuit, which in turn affects the pre-rounding error.  This
/// function breaks that dependency with a monotone binary search: it evaluates a candidate,
/// checks the measured pre-rounding norm with `max_safe_noise_refresh_v_bits`, and searches for the
/// largest valid candidate.
///
/// `ring_gsw_error_sigma` is the public-key Gaussian error used by the Ring-GSW ciphertexts.  The
/// simulator uses each material ciphertext's tracked `randomizer_norm` to add the FHE decryption
/// error `e'` that remains in the decoded refresh material before rounding.
pub fn simulate_noise_refresh_error_growth<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    secret_norm: &PolyMatrixNorm,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoders: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let zero_pre_rounding =
        PolyMatrixNorm::new(secret_norm.clone_ctx(), 1, 1, BigDecimal::from(0u32), None);
    let max_candidate = max_safe_noise_refresh_v_bits(
        &ring_gsw.params,
        secret_norm,
        &zero_pre_rounding,
    )
    .expect(
        "no positive noise-refresh v_bits satisfies the rounding bound even before circuit error",
    );
    let mut low = 1usize;
    let mut high = max_candidate;
    let mut best = None;
    while low <= high {
        let candidate = low + (high - low) / 2;
        let simulation = simulate_noise_refresh_error_growth_for_v_bits(
            ring_gsw.clone(),
            seed_bits,
            candidate,
            graph_seed,
            cbd_n,
            ring_gsw_error_sigma,
            num_slots,
            one.clone(),
            refreshed_input.clone(),
            enc_seeds,
            decryption_key.clone(),
            decoders,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let mut worst_pre_rounding = simulation
            .pre_round_outputs
            .first()
            .expect("simulation must produce at least one pre-round output")
            .clone();
        for pre_rounding in simulation.pre_round_outputs.iter().skip(1) {
            if pre_rounding.poly_norm.norm > worst_pre_rounding.poly_norm.norm {
                worst_pre_rounding = pre_rounding.clone();
            }
        }
        if validate_noise_refresh_v_bits(
            &ring_gsw.params,
            candidate,
            secret_norm,
            &worst_pre_rounding,
        ) {
            best = Some(simulation);
            low = candidate + 1;
        } else if candidate == 1 {
            break;
        } else {
            high = candidate - 1;
        }
    }
    best.expect("no positive noise-refresh v_bits satisfies the rounding bound after simulation")
}

/// Simulates noise growth for a fixed mask bit-size.
///
/// Prefer `simulate_noise_refresh_error_growth` for production estimates.  This fixed-size helper
/// is kept available for experiments that need to inspect a specific candidate.
pub fn simulate_noise_refresh_error_growth_for_v_bits<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoders: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    assert_eq!(enc_seeds.len(), seed_bits, "encrypted seed norm count mismatch");
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let log_base_q = ring_gsw.params.modulus_digits();
    assert_eq!(
        decoders.len(),
        num_slots * crt_depth,
        "decoder norm count must equal num_slots * crt_depth"
    );

    let circuit = build_noise_refresh_material_circuit::<DCRTPoly, A, DCRTPolyMatrix>(
        ring_gsw.clone(),
        seed_bits,
        v_bits,
        graph_seed,
        cbd_n,
        num_slots,
        false,
    );
    let mut inputs = enc_seeds.to_vec();
    inputs.push(decryption_key.clone());
    let material_outputs = circuit.eval(
        &(),
        one.clone(),
        inputs,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
        None,
    );
    assert_eq!(
        material_outputs.len(),
        num_slots * crt_depth * log_base_q,
        "noise-refresh error simulator output layout mismatch"
    );
    let material_decryption_errors = material_decryption_error_norms(
        ring_gsw,
        seed_bits,
        v_bits,
        graph_seed,
        cbd_n,
        ring_gsw_error_sigma,
        num_slots,
        one.clone_ctx(),
    );
    assert_eq!(
        material_decryption_errors.len(),
        material_outputs.len(),
        "Ring-GSW decryption-error layout must match material output layout"
    );

    let ctx = one.clone_ctx();
    let one_term =
        one.matrix_norm.clone() * PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);
    let input_term = refreshed_input.matrix_norm.clone() *
        PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);

    let pre_round_outputs = (0..num_slots * crt_depth)
        .into_par_iter()
        .map(|flat_idx| {
            let slot_idx = flat_idx / crt_depth;
            let crt_idx = flat_idx % crt_depth;
            let mut refresh_entry_norm = BigDecimal::from(0u32);
            let mut refresh_nrow = None;
            let mut refresh_ncol = 0usize;
            for digit_idx in 0..log_base_q {
                let material_idx =
                    slot_idx * crt_depth * log_base_q + crt_idx * log_base_q + digit_idx;
                let decoded_column = material_outputs[material_idx].matrix_norm.clone() *
                    PolyMatrixNorm::gadget_decomposed(ctx.clone(), 1);
                let collapsed_column =
                    (1..num_slots).fold(decoded_column.clone(), |acc, _| acc + &decoded_column);
                let collapsed_column = collapsed_column + &material_decryption_errors[material_idx];
                refresh_nrow = Some(collapsed_column.nrow);
                refresh_ncol += collapsed_column.ncol;
                refresh_entry_norm += collapsed_column.poly_norm.norm;
            }
            let refresh_term = PolyMatrixNorm::new(
                ctx.clone(),
                refresh_nrow.expect("log_base_q must be positive"),
                refresh_ncol,
                refresh_entry_norm,
                None,
            );
            input_term.clone() + &refresh_term + &one_term + &decoders[flat_idx].matrix_norm
        })
        .collect::<Vec<_>>();

    let rounded_errors = (0..num_slots)
        .into_par_iter()
        .map(|_| {
            PolyMatrixNorm::new(ctx.clone(), 1, log_base_q, BigDecimal::from(cbd_n as u64), None)
        })
        .collect::<Vec<_>>();
    NoiseRefreshErrorSimulation { v_bits, pre_round_outputs, rounded_errors }
}

fn material_decryption_error_norms<A>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    v_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    output_ctx: Arc<SimulatorContext>,
) -> Vec<PolyMatrixNorm>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    // This mirrors `build_noise_refresh_material_circuit` only at the Ring-GSW ciphertext metadata
    // level.  We need the concrete `RingGswCiphertext::randomizer_norm` values produced by the
    // Goldreich PRG operations, but we do not need to evaluate any native ciphertext plaintexts.
    //
    // For each output `(slot_idx, crt_idx, digit_idx)`, the material circuit decrypts `ring_dim`
    // error ciphertexts and `ring_dim * v_bits` mask ciphertexts into a slotwise polynomial, then
    // the online path collapses all slots into one polynomial column.  The returned norm is already
    // collapsed across those coefficient slots so the caller can add it once to the collapsed
    // decoded-column norm.
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim, "num_slots must match ring_dim");
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let log_base_q = ring_gsw.params.modulus_digits();
    let output_sizes =
        goldreich_noise_refresh_output_sizes(ring_dim, log_base_q, crt_depth, v_bits);
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");

    let zero = || PolyMatrixNorm::new(output_ctx.clone(), 1, 1, BigDecimal::from(0u32), None);
    (0..num_slots)
        .into_par_iter()
        .flat_map_iter(|slot_idx| {
            let mut circuit = ring_gsw.fresh_circuit();
            let encrypted_seeds = (0..seed_bits)
                .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
                .collect::<Vec<_>>();
            let material = evaluate_goldreich_noise_refresh_material(
                &mut circuit,
                ring_gsw.clone(),
                &encrypted_seeds,
                v_bits,
                graph_seed,
                cbd_n,
                slot_idx as u64,
            );
            assert_eq!(material.errors.len(), output_sizes.cbd_values);
            assert_eq!(material.masks.len(), output_sizes.mask_bits);

            (0..crt_depth * log_base_q)
                .into_par_iter()
                .map(|flat_idx| {
                    let crt_idx = flat_idx / log_base_q;
                    let digit_idx = flat_idx % log_base_q;
                    let mut collapsed_norm = zero();
                    for coeff_idx in 0..ring_dim {
                        let error_idx = digit_idx * ring_dim + coeff_idx;
                        let error_norm = material.errors[error_idx]
                            .estimate_decryption_error_norm(ring_gsw_error_sigma);
                        let mut coeff_norm = PolyMatrixNorm::new(
                            output_ctx.clone(),
                            error_norm.nrow,
                            error_norm.ncol,
                            error_norm.poly_norm.norm,
                            error_norm.zero_rows,
                        );

                        let mask_start = crt_idx * log_base_q * mask_q_chunk_len +
                            digit_idx * mask_q_chunk_len +
                            coeff_idx * v_bits;
                        for mask_bit in &material.masks[mask_start..mask_start + v_bits] {
                            let mask_norm =
                                mask_bit.estimate_decryption_error_norm(ring_gsw_error_sigma);
                            coeff_norm += PolyMatrixNorm::new(
                                output_ctx.clone(),
                                mask_norm.nrow,
                                mask_norm.ncol,
                                mask_norm.poly_norm.norm,
                                mask_norm.zero_rows,
                            );
                        }
                        collapsed_norm += coeff_norm;
                    }
                    collapsed_norm
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn biguint_to_decimal(value: &BigUint) -> BigDecimal {
    BigDecimal::from(BigInt::from(value.clone()))
}
