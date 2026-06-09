use crate::{
    circuit::PolyCircuit,
    decoder::simulation::{
        DecodeThreshold, decode_threshold, max_centered_mask_bits_for_available_range,
        validate_error_bound_security_margin as validate_decoder_error_bound_security_margin,
    },
    gadgets::{
        arith::{DecomposeArithmeticGadget, ModularArithmeticPlanner},
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::{
            GoldreichEdge, GoldreichFhePrg, GoldreichGraph, minimum_goldreich_input_size,
        },
    },
    lookup::PltEvaluator,
    matrix::dcrt_poly::DCRTPolyMatrix,
    noise_refresh::{
        circuit_prg::goldreich_noise_refresh_uniform_output_bits,
        naive_vec::build_noise_refresh_material_circuit,
    },
    poly::{PolyParams, dcrt::poly::DCRTPoly},
    simulator::{SimulatorContext, error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm},
    slot_transfer::SlotTransferEvaluator,
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::{debug, info};

/// Returns the largest mask bit-size that is guaranteed to round away.
///
/// The noise-refresh mask is decoded as a centered non-scaled binary integer.  To make sure this
/// mask and the accumulated pre-rounding perturbation disappear when the `q/q_i`-scaled term is
/// rounded back modulo every CRT factor, the conservative sufficient condition is:
///
/// ```text
/// 2^(v_bits - 1) + pre_rounding_error <= full_q / (2 * q_max)
/// ```
///
/// where `full_q` is the full DCRT modulus and `q_max` is the largest CRT factor. The ideal
/// `q/q_i`-scaled terms round back exactly, so no separate rounding-error budget is needed.
pub fn max_safe_noise_refresh_v_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    pre_rounding_error: &PolyMatrixNorm,
) -> Option<usize> {
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    let available = decode_threshold(full_q.as_ref(), &DecodeThreshold::new(q_max)) -
        &pre_rounding_error.poly_norm.norm;
    max_centered_mask_bits_for_available_range(&available, params.modulus_bits())
}

/// Checks whether a proposed mask bit-size satisfies the conservative rounding-away bound.
pub fn validate_noise_refresh_v_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    v_bits: usize,
    pre_rounding_error: &PolyMatrixNorm,
) -> bool {
    max_safe_noise_refresh_v_bits(params, pre_rounding_error)
        .is_some_and(|max_v_bits| v_bits <= max_v_bits)
}

/// Counts every uniform Goldreich bit drawn from one noise-refresh encrypted seed.
pub fn noise_refresh_uniform_output_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    v_bits: usize,
    cbd_n: usize,
) -> usize {
    let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
    goldreich_noise_refresh_uniform_output_bits(
        params.ring_dimension() as usize,
        params.modulus_digits(),
        crt_depth,
        v_bits,
        cbd_n,
    )
}

/// Returns the minimum seed length required by the Goldreich output bound for noise refresh.
pub fn minimum_noise_refresh_seed_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    v_bits: usize,
    cbd_n: usize,
) -> usize {
    minimum_goldreich_input_size(noise_refresh_uniform_output_bits(params, v_bits, cbd_n))
}

/// Returns the smallest seed length at least `min_seed_bits` that is safe for noise refresh.
pub fn search_noise_refresh_seed_bits(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    v_bits: usize,
    cbd_n: usize,
    min_seed_bits: usize,
) -> usize {
    min_seed_bits.max(minimum_noise_refresh_seed_bits(params, v_bits, cbd_n))
}

/// Checks whether an error bound has the requested security margin below a threshold.
///
/// Returns `true` exactly when `threshold >= error_bound * 2^security_bit`.
pub fn validate_error_bound_security_margin(
    threshold: &BigDecimal,
    error_bound: &BigDecimal,
    security_bit: usize,
) -> bool {
    validate_decoder_error_bound_security_margin(threshold, error_bound, security_bit)
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
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoders: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<NoiseRefreshErrorSimulation>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    info!(seed_bits, cbd_n, num_slots, "starting noise-refresh error simulation mask-bit search");
    search_noise_refresh_v_bits(
        &ring_gsw.params,
        one.clone_ctx(),
        seed_bits,
        cbd_n,
        "noise-refresh",
        |candidate| {
            simulate_noise_refresh_error_growth_for_v_bits(
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
            )
        },
    )
}

/// Simulates noise-refresh error growth using the protocol's slot symmetry.
///
/// The production refresh material contains one decoded output for every `(slot, crt, digit)`.
/// DiamondIO only needs the worst representative bound: every refreshed seed wire is symmetric, and
/// each CRT/digit branch has the same circuit shape up to the plaintext CRT modulus.  This helper
/// therefore evaluates one Goldreich material ciphertext and one representative decrypt/merge
/// circuit, then analytically scales the coefficient and slot collapses that are explicit sums in
/// the production circuit.  It returns representative vectors: `pre_round_outputs` has one entry
/// per CRT level, and `rounded_errors` has one entry for the symmetric refreshed wire.
pub fn simulate_symmetric_noise_refresh_error_growth<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoder: ErrorNorm,
    active_level_material_wire_error_override: Option<ErrorNorm>,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<NoiseRefreshErrorSimulation>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    info!(
        seed_bits,
        cbd_n, num_slots, "starting symmetric noise-refresh error simulation mask-bit search"
    );
    info!("building symmetric noise-refresh representative material prefix");
    let prg_prefix = build_symmetric_noise_refresh_prg_error_prefix(
        &ring_gsw,
        seed_bits,
        cbd_n,
        ring_gsw_error_sigma,
        one.clone(),
        enc_seeds,
        active_level_material_wire_error_override.clone(),
        plt_evaluator,
        slot_transfer_evaluator,
    );
    info!("finished symmetric noise-refresh representative material prefix");
    info!("building symmetric noise-refresh representative decrypt prefix");
    let decrypt_prefix = build_symmetric_noise_refresh_decrypt_prefix(
        &ring_gsw,
        &prg_prefix,
        one.clone(),
        decryption_key.clone(),
        plt_evaluator,
        slot_transfer_evaluator,
    );
    info!("finished symmetric noise-refresh representative decrypt prefix");
    search_noise_refresh_v_bits(
        &ring_gsw.params,
        one.clone_ctx(),
        seed_bits,
        cbd_n,
        "symmetric noise-refresh",
        |candidate| {
            simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_prefix(
                ring_gsw.clone(),
                &prg_prefix,
                &decrypt_prefix,
                seed_bits,
                candidate,
                cbd_n,
                ring_gsw_error_sigma,
                num_slots,
                one.clone(),
                refreshed_input.clone(),
                enc_seeds,
                decryption_key.clone(),
                decoder.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
            )
        },
    )
}

fn search_noise_refresh_v_bits<F>(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    ctx: Arc<SimulatorContext>,
    seed_bits: usize,
    cbd_n: usize,
    simulation_name: &'static str,
    mut simulate_candidate: F,
) -> Option<NoiseRefreshErrorSimulation>
where
    F: FnMut(usize) -> NoiseRefreshErrorSimulation,
{
    let zero_pre_rounding = PolyMatrixNorm::new(ctx, 1, 1, BigDecimal::from(0u32), None);
    let Some(max_candidate) = max_safe_noise_refresh_v_bits(params, &zero_pre_rounding) else {
        info!(
            simulation = simulation_name,
            "noise-refresh error simulation found no candidate before circuit error"
        );
        return None;
    };
    debug!(simulation = simulation_name, max_candidate, "noise-refresh initial v_bits candidate");
    let mut low = 1usize;
    let mut high = max_candidate;
    let mut best = None;
    while low <= high {
        let candidate = low + (high - low) / 2;
        info!(simulation = simulation_name, candidate, low, high, "evaluating v_bits candidate");
        let required_seed_bits = minimum_noise_refresh_seed_bits(params, candidate, cbd_n);
        if seed_bits < required_seed_bits {
            debug!(
                simulation = simulation_name,
                candidate,
                seed_bits,
                required_seed_bits,
                "noise-refresh candidate rejected by Goldreich PRG output bound"
            );
            if candidate == 1 {
                break;
            }
            high = candidate - 1;
            continue;
        }
        let simulation = simulate_candidate(candidate);
        let worst_pre_rounding = worst_pre_rounding_error(&simulation);
        let max_valid_bits = max_safe_noise_refresh_v_bits(params, worst_pre_rounding);
        let valid = max_valid_bits.is_some_and(|max_valid_bits| candidate <= max_valid_bits);
        debug!(
            simulation = simulation_name,
            candidate,
            valid,
            ?max_valid_bits,
            worst_pre_rounding_norm = %worst_pre_rounding.poly_norm.norm,
            "noise-refresh candidate evaluated"
        );
        if valid {
            best = Some(simulation);
            low = candidate + 1;
        } else if candidate == 1 {
            break;
        } else {
            high = max_valid_bits.unwrap_or(0).min(candidate - 1);
        }
    }
    info!(
        simulation = simulation_name,
        found = best.is_some(),
        "finished noise-refresh error simulation mask-bit search"
    );
    best
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
    info!(
        seed_bits,
        v_bits, cbd_n, num_slots, "starting fixed-v_bits noise-refresh error simulation"
    );
    assert_eq!(enc_seeds.len(), seed_bits, "encrypted seed norm count mismatch");
    let required_seed_bits = minimum_noise_refresh_seed_bits(&ring_gsw.params, v_bits, cbd_n);
    assert!(
        seed_bits >= required_seed_bits,
        "noise-refresh seed_bits {seed_bits} is below Goldreich PRG safety minimum {required_seed_bits} for v_bits={v_bits}, cbd_n={cbd_n}"
    );
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let log_base_q = ring_gsw.params.modulus_digits();
    assert_eq!(
        decoders.len(),
        num_slots * crt_depth,
        "decoder norm count must equal num_slots * crt_depth"
    );

    let active_level_scale = ring_gsw.active_levels;
    let material_ring_gsw = one_active_level_ring_gsw_context(&ring_gsw);
    debug!(
        original_active_levels = active_level_scale,
        material_active_levels = material_ring_gsw.active_levels,
        "noise-refresh material error simulation using one-active-level representative PRG"
    );

    let circuit = build_noise_refresh_material_circuit::<DCRTPoly, A, DCRTPolyMatrix>(
        material_ring_gsw.clone(),
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
    debug!(
        outputs = material_outputs.len(),
        "noise-refresh material circuit error evaluation finished"
    );
    assert_eq!(
        material_outputs.len(),
        num_slots * crt_depth * log_base_q,
        "noise-refresh error simulator output layout mismatch"
    );
    let mut material_output_matrix_norms =
        material_outputs.into_iter().map(|output| output.matrix_norm).collect::<Vec<_>>();
    extrapolate_matrix_norms_to_active_levels(
        &mut material_output_matrix_norms,
        active_level_scale,
    );
    let mut material_decryption_error = representative_material_decryption_error_norm(
        material_ring_gsw,
        seed_bits,
        v_bits,
        cbd_n,
        ring_gsw_error_sigma,
        num_slots,
        one.clone_ctx(),
    );
    extrapolate_matrix_norms_to_active_levels(
        std::slice::from_mut(&mut material_decryption_error),
        active_level_scale,
    );
    debug!(
        active_level_scale,
        "noise-refresh representative material and decryption error norms extrapolated"
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
                let decoded_column = material_output_matrix_norms[material_idx].clone() *
                    PolyMatrixNorm::gadget_decomposed(ctx.clone(), 1);
                let collapsed_column =
                    (1..num_slots).fold(decoded_column.clone(), |acc, _| acc + &decoded_column);
                let collapsed_column = collapsed_column + &material_decryption_error;
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
    info!(v_bits, "finished fixed-v_bits noise-refresh error simulation");
    NoiseRefreshErrorSimulation { v_bits, pre_round_outputs, rounded_errors }
}

pub fn simulate_symmetric_noise_refresh_error_growth_for_v_bits<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    v_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoder: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let prg_prefix = build_symmetric_noise_refresh_prg_error_prefix(
        &ring_gsw,
        seed_bits,
        cbd_n,
        ring_gsw_error_sigma,
        one.clone(),
        enc_seeds,
        None,
        plt_evaluator,
        slot_transfer_evaluator,
    );
    let decrypt_prefix = build_symmetric_noise_refresh_decrypt_prefix(
        &ring_gsw,
        &prg_prefix,
        one.clone(),
        decryption_key.clone(),
        plt_evaluator,
        slot_transfer_evaluator,
    );
    simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_prefix(
        ring_gsw,
        &prg_prefix,
        &decrypt_prefix,
        seed_bits,
        v_bits,
        cbd_n,
        ring_gsw_error_sigma,
        num_slots,
        one,
        refreshed_input,
        enc_seeds,
        decryption_key,
        decoder,
        plt_evaluator,
        slot_transfer_evaluator,
    )
}

/// Simulates fixed-`v_bits` symmetric noise-refresh using a caller-provided representative
/// material-wire bound.
///
/// DiamondIO already evaluates the representative Goldreich material PRG while simulating the
/// selected-half PRF branch. Passing that bound here keeps fixed-width validation from rebuilding
/// the equivalent material PRG prefix after the global `v_bits` has already been selected.
pub fn simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_material_override<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    v_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    decryption_key: ErrorNorm,
    decoder: ErrorNorm,
    active_level_material_wire_error_override: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let prg_prefix = build_symmetric_noise_refresh_prg_error_prefix(
        &ring_gsw,
        seed_bits,
        cbd_n,
        ring_gsw_error_sigma,
        one.clone(),
        enc_seeds,
        Some(active_level_material_wire_error_override),
        plt_evaluator,
        slot_transfer_evaluator,
    );
    let decrypt_prefix = build_symmetric_noise_refresh_decrypt_prefix(
        &ring_gsw,
        &prg_prefix,
        one.clone(),
        decryption_key.clone(),
        plt_evaluator,
        slot_transfer_evaluator,
    );
    simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_prefix(
        ring_gsw,
        &prg_prefix,
        &decrypt_prefix,
        seed_bits,
        v_bits,
        cbd_n,
        ring_gsw_error_sigma,
        num_slots,
        one,
        refreshed_input,
        enc_seeds,
        decryption_key,
        decoder,
        plt_evaluator,
        slot_transfer_evaluator,
    )
}

fn simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_prefix<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    prg_prefix: &SymmetricNoiseRefreshPrgErrorPrefix<A>,
    decrypt_prefix: &SymmetricNoiseRefreshDecryptPrefix,
    seed_bits: usize,
    v_bits: usize,
    cbd_n: usize,
    _ring_gsw_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    refreshed_input: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    _decryption_key: ErrorNorm,
    decoder: ErrorNorm,
    _plt_evaluator: &PE,
    _slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    info!(
        seed_bits,
        v_bits, cbd_n, num_slots, "starting fixed-v_bits symmetric noise-refresh error simulation"
    );
    assert!(
        seed_bits >= 5,
        "symmetric noise-refresh seed_bits must be at least five for representative simulation"
    );
    assert!(
        enc_seeds.len() >= 5,
        "symmetric noise-refresh representative simulation requires at least five seed norms"
    );
    let required_seed_bits = minimum_noise_refresh_seed_bits(&ring_gsw.params, v_bits, cbd_n);
    assert!(
        seed_bits >= required_seed_bits,
        "symmetric noise-refresh seed_bits {seed_bits} is below Goldreich PRG safety minimum {required_seed_bits} for v_bits={v_bits}, cbd_n={cbd_n}"
    );
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim, "num_slots must match ring_dim");
    let (_q_moduli, _crt_bits, crt_depth) = ring_gsw.params.to_crt();
    let log_base_q = ring_gsw.params.modulus_digits();
    let ctx = one.clone_ctx();
    assert_eq!(
        prg_prefix.material_ring_gsw.active_levels, 1,
        "symmetric noise-refresh PRG prefix must use one active level"
    );

    let mask_bit_scale = BigDecimal::from(v_bits as u64);

    let material_decryption_error = representative_symmetric_material_decryption_error_norm(
        prg_prefix,
        ring_dim,
        v_bits,
        ctx.clone(),
    );

    let one_term =
        one.matrix_norm.clone() * PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);
    let input_term = refreshed_input.matrix_norm.clone() *
        PolyMatrixNorm::gadget_decomposed(ctx.clone(), log_base_q);
    let slot_collapse_scale = BigDecimal::from(num_slots as u64);

    let per_digit_decoded = (decrypt_prefix.error_decoded.matrix_norm.clone() +
        &(decrypt_prefix.mask_decoded.matrix_norm.clone() * &mask_bit_scale)) *
        PolyMatrixNorm::gadget_decomposed(ctx.clone(), 1);
    let collapsed_column = per_digit_decoded * &slot_collapse_scale + &material_decryption_error;
    let mut refresh_nrow = None;
    let mut refresh_ncol = 0usize;
    let mut refresh_norm = BigDecimal::from(0u32);
    for _ in 0..log_base_q {
        refresh_nrow = Some(collapsed_column.nrow);
        refresh_ncol += collapsed_column.ncol;
        refresh_norm += collapsed_column.poly_norm.norm.clone();
    }
    let refresh_term = PolyMatrixNorm::new(
        ctx.clone(),
        refresh_nrow.expect("log_base_q must be positive"),
        refresh_ncol,
        refresh_norm,
        None,
    );
    let representative_pre_round_output =
        input_term + &refresh_term + &one_term + &decoder.matrix_norm;
    let pre_round_outputs = vec![representative_pre_round_output; crt_depth];
    assert_eq!(
        pre_round_outputs.len(),
        crt_depth,
        "symmetric noise-refresh output must contain one representative entry per CRT level"
    );

    let rounded_errors =
        vec![PolyMatrixNorm::new(ctx, 1, log_base_q, BigDecimal::from(cbd_n as u64), None)];
    info!(v_bits, "finished fixed-v_bits symmetric noise-refresh error simulation");
    NoiseRefreshErrorSimulation { v_bits, pre_round_outputs, rounded_errors }
}

fn worst_pre_rounding_error(simulation: &NoiseRefreshErrorSimulation) -> &PolyMatrixNorm {
    simulation
        .pre_round_outputs
        .iter()
        .max_by(|lhs, rhs| {
            lhs.poly_norm
                .norm
                .partial_cmp(&rhs.poly_norm.norm)
                .expect("noise-refresh pre-rounding norms must be comparable")
        })
        .expect("simulation must produce at least one pre-round output")
}

fn one_active_level_ring_gsw_context<A>(
    ring_gsw: &Arc<RingGswContext<DCRTPoly, A>>,
) -> Arc<RingGswContext<DCRTPoly, A>>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    if ring_gsw.active_levels == 1 {
        return ring_gsw.clone();
    }
    let mut circuit = ring_gsw.fresh_circuit();
    Arc::new(RingGswContext::from_arith_context(
        &mut circuit,
        &ring_gsw.params,
        ring_gsw.num_slots,
        ring_gsw.arith_ctx.clone(),
        Some(1),
        Some(ring_gsw.level_offset),
    ))
}

fn extrapolate_matrix_norms_to_active_levels(norms: &mut [PolyMatrixNorm], active_levels: usize) {
    assert!(active_levels > 0, "active-level extrapolation scale must be positive");
    if active_levels == 1 {
        return;
    }
    let scale = BigDecimal::from(active_levels as u64);
    for norm in norms {
        *norm = norm.clone() * &scale;
    }
}

fn scale_error_norm(input: ErrorNorm, scale: &BigDecimal) -> ErrorNorm {
    ErrorNorm {
        plaintext_norm: input.plaintext_norm * scale,
        matrix_norm: input.matrix_norm * scale,
    }
}

struct SymmetricNoiseRefreshPrgErrorPrefix<A>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    material_ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    material_wire_error: ErrorNorm,
    material_error_decryption_error: PolyMatrixNorm,
    material_mask_decryption_error: PolyMatrixNorm,
}

struct SymmetricNoiseRefreshDecryptPrefix {
    error_decoded: ErrorNorm,
    mask_decoded: ErrorNorm,
}

fn build_symmetric_noise_refresh_decrypt_prefix<A, PE, ST>(
    ring_gsw: &Arc<RingGswContext<DCRTPoly, A>>,
    prg_prefix: &SymmetricNoiseRefreshPrgErrorPrefix<A>,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> SymmetricNoiseRefreshDecryptPrefix
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    // One decrypt_batch in the real circuit packs `ring_dim` ciphertexts into one polynomial.
    // Scaling the representative ciphertext before decrypting models that linear slot reduction
    // without constructing every coefficient ciphertext.
    let coefficient_scale = BigDecimal::from(ring_dim as u64);
    let material_wire_error =
        scale_error_norm(prg_prefix.material_wire_error.clone(), &coefficient_scale);
    let representative_plaintext_modulus = BigUint::from(2u64);
    // The production decrypt path uses `active_q / plaintext_modulus` constants. The exact
    // plaintext modulus changes the literal residues, but the symmetric ErrorNorm model only needs
    // one non-zero representative decrypt circuit: CRT levels and mask bits share the same
    // nested-RNS tower-width shape, while coefficient, slot, bit-count, and explicit Ring-GSW
    // decryption-error growth are scaled outside this circuit evaluation. Using the one-active
    // material context keeps this representative decrypt consistent with the one-active PRG
    // simulation rule. The resulting ErrorNorm is independent of v_bits, so it is cached across
    // the v_bits binary search.
    let (error_decoded, mask_decoded) = representative_material_decrypt_outputs(
        prg_prefix.material_ring_gsw.clone(),
        representative_plaintext_modulus.clone(),
        representative_plaintext_modulus,
        one,
        material_wire_error,
        decryption_key,
        plt_evaluator,
        slot_transfer_evaluator,
    );
    SymmetricNoiseRefreshDecryptPrefix { error_decoded, mask_decoded }
}

fn build_symmetric_noise_refresh_prg_error_prefix<A, PE, ST>(
    ring_gsw: &Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    one: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    active_level_material_wire_error_override: Option<ErrorNorm>,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> SymmetricNoiseRefreshPrgErrorPrefix<A>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let active_level_scale = ring_gsw.active_levels;
    let material_ring_gsw = one_active_level_ring_gsw_context(ring_gsw);
    let material_wire_error =
        if let Some(material_wire_error) = active_level_material_wire_error_override {
            // DiamondIO has already evaluated a representative uniform Goldreich PRG output for the
            // same seed-error profile and active-level window. Passing its CBD-scaled bound here
            // avoids a second equivalent one-output PRG ErrorNorm evaluation inside the v_bits
            // search. The override is expected to have already been extrapolated to
            // `ring_gsw.active_levels`.
            info!("using caller-provided symmetric noise-refresh material wire error");
            material_wire_error
        } else {
            info!("evaluating symmetric noise-refresh representative material wire error");
            let mut material_wire_error = representative_material_ciphertext_wire_error(
                material_ring_gsw.clone(),
                seed_bits,
                cbd_n,
                one,
                enc_seeds,
                plt_evaluator,
                slot_transfer_evaluator,
            );
            material_wire_error.matrix_norm =
                material_wire_error.matrix_norm * BigDecimal::from(active_level_scale as u64);
            info!("finished symmetric noise-refresh representative material wire error");
            material_wire_error
        };

    info!("estimating symmetric noise-refresh material decryption error");
    let (mut material_error_decryption_error, mut material_mask_decryption_error) =
        representative_symmetric_material_decryption_error_base_norms(
            material_ring_gsw.clone(),
            seed_bits,
            cbd_n,
            ring_gsw_error_sigma,
        );
    extrapolate_matrix_norms_to_active_levels(
        std::slice::from_mut(&mut material_error_decryption_error),
        active_level_scale,
    );
    extrapolate_matrix_norms_to_active_levels(
        std::slice::from_mut(&mut material_mask_decryption_error),
        active_level_scale,
    );
    info!("finished symmetric noise-refresh material decryption-error estimate");

    SymmetricNoiseRefreshPrgErrorPrefix {
        material_ring_gsw,
        material_wire_error,
        material_error_decryption_error,
        material_mask_decryption_error,
    }
}

fn representative_material_ciphertext_wire_error<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    cbd_n: usize,
    one: ErrorNorm,
    enc_seeds: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> ErrorNorm
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    assert!(
        seed_bits >= 5,
        "noise-refresh representative material simulation requires at least five seed bits"
    );
    assert!(
        enc_seeds.len() >= 5,
        "noise-refresh representative material simulation requires at least five seed errors"
    );
    let mut circuit = ring_gsw.fresh_circuit();
    let encrypted_seeds = (0..5)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let error = representative_goldreich_cbd_output(
        ring_gsw.clone(),
        &encrypted_seeds,
        cbd_n,
        &mut circuit,
    );
    // The symmetric simulator uses the CBD material output as the representative bound for both
    // CBD error ciphertexts and uniform mask ciphertexts. A CBD output combines `2 * cbd_n`
    // Goldreich uniform branches, so it is a conservative representative for a single mask bit
    // while avoiding a second equivalent one-output PRG evaluation.
    circuit.output(error.sub_circuit_wires());
    let wire_count = representative_ring_gsw_wire_count(ring_gsw);
    let inputs = enc_seeds[..5]
        .iter()
        .flat_map(|seed| std::iter::repeat_n(seed.clone(), wire_count))
        .collect::<Vec<_>>();
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "noise-refresh representative material input count mismatch"
    );
    circuit
        .eval(
            &(),
            one,
            inputs,
            Some(plt_evaluator),
            Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
            None,
        )
        .into_iter()
        .max_by(|lhs, rhs| {
            lhs.matrix_norm
                .poly_norm
                .norm
                .partial_cmp(&rhs.matrix_norm.poly_norm.norm)
                .expect("representative material norms must be comparable")
        })
        .expect("representative material circuit must produce output wires")
}

fn representative_ring_gsw_wire_count<A>(ring_gsw: Arc<RingGswContext<DCRTPoly, A>>) -> usize
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    RingGswCiphertext::input(ring_gsw, None, &mut circuit)
        .sub_circuit_wires()
        .iter()
        .map(|wire| wire.len())
        .sum()
}

fn representative_material_decrypt_outputs<A, PE, ST>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    error_plaintext_modulus: BigUint,
    mask_plaintext_modulus: BigUint,
    one: ErrorNorm,
    material_wire_error: ErrorNorm,
    decryption_key: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> (ErrorNorm, ErrorNorm)
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let mut circuit = ring_gsw.fresh_circuit();
    let decryption_key_wire = circuit.input(1).at(0).as_single_wire();
    let error_ciphertext = RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit);
    let mask_ciphertext = RingGswCiphertext::input(ring_gsw, None, &mut circuit);
    let error_parts = error_ciphertext.decrypt::<DCRTPolyMatrix>(
        decryption_key_wire,
        error_plaintext_modulus,
        &mut circuit,
    );
    let mask_parts = mask_ciphertext.decrypt::<DCRTPolyMatrix>(
        decryption_key_wire,
        mask_plaintext_modulus,
        &mut circuit,
    );
    let error_output =
        circuit.add_gate(error_parts.secret_dependent, error_parts.public_bottom).as_single_wire();
    let mask_output =
        circuit.add_gate(mask_parts.secret_dependent, mask_parts.public_bottom).as_single_wire();
    circuit.output(vec![error_output, mask_output]);

    let wire_count = (circuit.num_input() - 1) / 2;
    let mut inputs = Vec::with_capacity(circuit.num_input());
    inputs.push(decryption_key);
    inputs.extend(std::iter::repeat_n(material_wire_error.clone(), wire_count));
    inputs.extend(std::iter::repeat_n(material_wire_error, wire_count));
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "noise-refresh representative decrypt input count mismatch"
    );
    let outputs = circuit.eval(
        &(),
        one,
        inputs,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
        None,
    );
    assert_eq!(outputs.len(), 2, "representative decrypt circuit must produce two outputs");
    (outputs[0].clone(), outputs[1].clone())
}

fn representative_symmetric_material_decryption_error_base_norms<A>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
) -> (PolyMatrixNorm, PolyMatrixNorm)
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    assert!(
        seed_bits >= 5,
        "symmetric noise-refresh representative decryption simulation requires at least five seed bits"
    );
    let mut circuit = ring_gsw.fresh_circuit();
    let output_ctx = ring_gsw.randomizer_norm_ctx.clone();
    let encrypted_seeds = (0..5)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let error = representative_goldreich_cbd_output(
        ring_gsw.clone(),
        &encrypted_seeds,
        cbd_n,
        &mut circuit,
    );
    let error_norm = error.estimate_decryption_error_norm(ring_gsw_error_sigma);
    // Reuse the CBD randomizer bound for mask ciphertexts as well. This is intentionally
    // conservative and keeps the representative noise-refresh PRG simulation to one CBD output.
    let mask_norm = error_norm.clone();
    (
        PolyMatrixNorm::new(
            output_ctx.clone(),
            error_norm.nrow,
            error_norm.ncol,
            error_norm.poly_norm.norm,
            error_norm.zero_rows,
        ),
        PolyMatrixNorm::new(
            output_ctx,
            mask_norm.nrow,
            mask_norm.ncol,
            mask_norm.poly_norm.norm,
            mask_norm.zero_rows,
        ),
    )
}

fn representative_goldreich_cbd_output<A>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    encrypted_seeds: &[RingGswCiphertext<DCRTPoly, A>],
    cbd_n: usize,
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> RingGswCiphertext<DCRTPoly, A>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    assert_eq!(
        encrypted_seeds.len(),
        5,
        "representative Goldreich CBD simulation uses exactly five seed inputs"
    );
    assert!(cbd_n > 0, "representative Goldreich CBD simulation requires cbd_n > 0");
    let positive_terms = (0..cbd_n)
        .map(|idx| representative_goldreich_output(ring_gsw.clone(), encrypted_seeds, idx, circuit))
        .collect::<Vec<_>>();
    let negative_terms = (0..cbd_n)
        .map(|idx| {
            representative_goldreich_output(ring_gsw.clone(), encrypted_seeds, cbd_n + idx, circuit)
        })
        .collect::<Vec<_>>();
    let positive = reduce_ring_gsw_add_pairwise(positive_terms, circuit);
    let negative = reduce_ring_gsw_add_pairwise(negative_terms, circuit);
    positive.sub(&negative, circuit)
}

fn representative_goldreich_output<A>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    encrypted_seeds: &[RingGswCiphertext<DCRTPoly, A>],
    edge_idx: usize,
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> RingGswCiphertext<DCRTPoly, A>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    let graph = GoldreichGraph::from_edges(
        5,
        vec![representative_goldreich_edge(edge_idx)],
        Default::default(),
    );
    let goldreich = GoldreichFhePrg::from_public_graph(circuit, ring_gsw, graph);
    let mut outputs = goldreich.evaluate_uniform(encrypted_seeds, circuit);
    assert_eq!(outputs.len(), 1, "representative Goldreich graph must produce one output");
    outputs.remove(0)
}

fn representative_goldreich_edge(edge_idx: usize) -> GoldreichEdge {
    const ROLE_SPLITS: [([usize; 3], [usize; 2]); 10] = [
        ([0, 1, 2], [3, 4]),
        ([0, 1, 3], [2, 4]),
        ([0, 1, 4], [2, 3]),
        ([0, 2, 3], [1, 4]),
        ([0, 2, 4], [1, 3]),
        ([0, 3, 4], [1, 2]),
        ([1, 2, 3], [0, 4]),
        ([1, 2, 4], [0, 3]),
        ([1, 3, 4], [0, 2]),
        ([2, 3, 4], [0, 1]),
    ];
    let (xor_inputs, and_inputs) = ROLE_SPLITS[edge_idx % ROLE_SPLITS.len()];
    GoldreichEdge::new(xor_inputs, and_inputs)
}

fn reduce_ring_gsw_add_pairwise<A>(
    mut terms: Vec<RingGswCiphertext<DCRTPoly, A>>,
    circuit: &mut PolyCircuit<DCRTPoly>,
) -> RingGswCiphertext<DCRTPoly, A>
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    assert!(!terms.is_empty(), "representative Ring-GSW reduction requires at least one term");
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        for pair in terms.chunks(2) {
            if pair.len() == 2 {
                next.push(pair[0].add(&pair[1], circuit));
            } else {
                next.push(pair[0].clone());
            }
        }
        terms = next;
    }
    terms.pop().expect("pairwise Ring-GSW reduction must leave one term")
}

fn representative_symmetric_material_decryption_error_norm<A>(
    prg_prefix: &SymmetricNoiseRefreshPrgErrorPrefix<A>,
    ring_dim: usize,
    v_bits: usize,
    output_ctx: Arc<SimulatorContext>,
) -> PolyMatrixNorm
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    let coefficient_scale = BigDecimal::from(ring_dim as u64);
    let mask_scale = BigDecimal::from(
        ring_dim
            .checked_mul(v_bits)
            .expect("noise-refresh representative mask decryption-error scale overflow")
            as u64,
    );
    PolyMatrixNorm::new(
        output_ctx,
        prg_prefix
            .material_error_decryption_error
            .nrow
            .max(prg_prefix.material_mask_decryption_error.nrow),
        prg_prefix
            .material_error_decryption_error
            .ncol
            .max(prg_prefix.material_mask_decryption_error.ncol),
        prg_prefix.material_error_decryption_error.poly_norm.norm.clone() * coefficient_scale +
            prg_prefix.material_mask_decryption_error.poly_norm.norm.clone() * mask_scale,
        None,
    )
}

fn representative_material_decryption_error_norm<A>(
    ring_gsw: Arc<RingGswContext<DCRTPoly, A>>,
    seed_bits: usize,
    v_bits: usize,
    cbd_n: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    output_ctx: Arc<SimulatorContext>,
) -> PolyMatrixNorm
where
    A: DecomposeArithmeticGadget<DCRTPoly> + ModularArithmeticPlanner<DCRTPoly>,
{
    let ring_dim = ring_gsw.params.ring_dimension() as usize;
    assert_eq!(num_slots, ring_dim, "num_slots must match ring_dim");
    let mask_q_chunk_len = ring_dim.checked_mul(v_bits).expect("mask q chunk length overflow");
    assert!(
        seed_bits >= 5,
        "noise-refresh representative material decryption simulation requires at least five seed bits"
    );

    let mut circuit = ring_gsw.fresh_circuit();
    let encrypted_seeds = (0..5)
        .map(|_| RingGswCiphertext::input(ring_gsw.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let error = representative_goldreich_cbd_output(
        ring_gsw.clone(),
        &encrypted_seeds,
        cbd_n,
        &mut circuit,
    );
    let error_norm = error.estimate_decryption_error_norm(ring_gsw_error_sigma);
    let base = PolyMatrixNorm::new(
        output_ctx,
        error_norm.nrow,
        error_norm.ncol,
        error_norm.poly_norm.norm,
        error_norm.zero_rows,
    );
    // One representative CBD ciphertext bounds both CBD error ciphertexts and uniform mask
    // ciphertexts. Noise refresh decrypts one error ciphertext per coefficient and `v_bits` mask
    // ciphertexts per coefficient, so the collapsed decrypt-error contribution scales by
    // `ring_dim + ring_dim * v_bits`.
    base * BigDecimal::from(
        ring_dim
            .checked_add(mask_q_chunk_len)
            .expect("noise-refresh material decryption-error scale overflow") as u64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gadgets::fhe_prg::goldreich::goldreich_output_bound_holds,
        poly::dcrt::params::DCRTPolyParams,
    };

    #[test]
    fn test_max_safe_noise_refresh_v_bits_accepts_exact_power_of_two_available_range() {
        let params = DCRTPolyParams::new(2, 2, 20, 5);
        let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
        let q_max = q_moduli.iter().copied().max().expect("CRT moduli must be nonempty");
        let full_q: Arc<BigUint> = params.modulus().into();
        let exact_available = BigDecimal::from(128u32);
        let pre_rounding_error =
            decode_threshold(full_q.as_ref(), &DecodeThreshold::new(q_max)) - exact_available;
        let ctx = Arc::new(SimulatorContext::new(
            BigDecimal::from(1u32),
            BigDecimal::from(2u32),
            1,
            1,
            1,
        ));
        let pre_rounding = PolyMatrixNorm::new(ctx, 1, 1, pre_rounding_error, None);

        assert_eq!(max_safe_noise_refresh_v_bits(&params, &pre_rounding), Some(8));
    }

    #[test]
    fn test_noise_refresh_seed_bit_search_covers_uniform_output_bits() {
        let params = DCRTPolyParams::new(2, 1, 10, 5);
        let v_bits = 2usize;
        let cbd_n = 3usize;
        let output_bits = noise_refresh_uniform_output_bits(&params, v_bits, cbd_n);
        let seed_bits = minimum_noise_refresh_seed_bits(&params, v_bits, cbd_n);

        assert!(goldreich_output_bound_holds(seed_bits, output_bits));
        assert!(!goldreich_output_bound_holds(seed_bits - 1, output_bits));
        assert_eq!(
            search_noise_refresh_seed_bits(&params, v_bits, cbd_n, seed_bits + 7),
            seed_bits + 7,
            "search helper must respect caller-provided lower bounds"
        );
    }
}
