use crate::{
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::RingGswContext,
    },
    lookup::PltEvaluator,
    matrix::dcrt_poly::DCRTPolyMatrix,
    noise_refresh::naive_vec::build_noise_refresh_material_circuit,
    poly::{PolyParams, dcrt::poly::DCRTPoly},
    simulator::{error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm},
    slot_transfer::SlotTransferEvaluator,
};
use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
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
/// the decoder subtraction.
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
pub fn simulate_noise_refresh_error_growth<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    graph_seed: [u8; 32],
    cbd_n: usize,
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

    let ctx = one.clone_ctx();
    let one_term =
        one.matrix_norm.clone() * PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);
    let input_term = refreshed_input.matrix_norm.clone() *
        PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);

    let pre_round_outputs = (0..num_slots * crt_depth)
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
        .map(|_| {
            PolyMatrixNorm::new(ctx.clone(), 1, log_base_q, BigDecimal::from(cbd_n as u64), None)
        })
        .collect::<Vec<_>>();
    NoiseRefreshErrorSimulation { v_bits, pre_round_outputs, rounded_errors }
}

fn biguint_to_decimal(value: &BigUint) -> BigDecimal {
    BigDecimal::from(BigInt::from(value.clone()))
}
