use std::sync::Arc;

use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use num_traits::FromPrimitive;

use crate::{
    circuit::PolyCircuit,
    func_enc::aky24::{
        Aky24Func, Aky24Params, build_func_circuit, build_goldreich_prg_range_circuit,
        build_prf_mask_circuit, build_ring_gsw_context,
    },
    lookup::PltEvaluator,
    matrix::dcrt_poly::DCRTPolyMatrix,
    noise_refresh::simulation::{
        NoiseRefreshErrorSimulation, simulate_noise_refresh_error_growth,
        validate_error_bound_security_margin,
    },
    poly::{PolyParams, dcrt::poly::DCRTPoly},
    simulator::{
        SimulatorContext,
        error_norm::{ErrorNorm, compute_preimage_norm},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::SlotTransferEvaluator,
};
use tracing::{debug, info};

/// Candidate-specific inputs for AKY24 CRT-depth search.
///
/// `aky24_find_crt_depth` asks a caller-provided builder to create this value for each candidate
/// `crt_depth`, because changing the CRT depth changes polynomial parameters, Ring-GSW context, and
/// simulator norm contexts together.
#[derive(Debug, Clone)]
pub struct Aky24CrtDepthSearchCandidate<TD> {
    /// AKY24 parameters constructed for one candidate CRT depth.
    pub params: Aky24Params<DCRTPolyMatrix, TD>,
    /// Decryption error-simulation inputs that use the same simulator context as `params`.
    pub inputs: Aky24DecErrorSimulationInputs,
    /// Bound for the secret term used by the noise-refresh rounding check.
    pub secret_norm: PolyMatrixNorm,
    /// Error-norm representation of the circuit constant one.
    pub one: ErrorNorm,
    /// Error bound for the Ring-GSW decryption-key input.
    pub decryption_key: ErrorNorm,
    /// Decoder error bounds in the layout expected by the noise-refresh simulator.
    pub decoders: Vec<ErrorNorm>,
}

/// Successful AKY24 CRT-depth search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24CrtDepthSearchResult {
    /// Smallest CRT depth found by the search.
    pub crt_depth: usize,
    /// Largest PRF mask coefficient bit-width that was safe at `crt_depth`.
    pub prf_mask_output_coeff_bits: usize,
}

/// Largest safe PRF mask bit-width together with the simulation that certified it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24PrfMaskOutputCoeffBitsSearchResult {
    /// Largest PRF mask coefficient bit-width satisfying the correctness margin.
    pub prf_mask_output_coeff_bits: usize,
    /// Error simulation evaluated with `prf_mask_output_coeff_bits`.
    pub simulation: Aky24DecErrorSimulation,
}

/// Top-level error-simulation result for AKY24 decryption.
///
/// The result intentionally exposes only the quantities needed for parameter selection: the
/// per-seed-bit noise-refresh simulations and the final decode error bound for the configured PRF
/// mask coefficient bit-width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24DecErrorSimulation {
    /// Representative noise-refresh simulations for the first and steady-state PRF-seed rounds.
    ///
    /// The vector contains at most two entries. Entry `0` is the first refresh from the caller's
    /// initial PRF-seed error norms. Entry `1`, when present, is the second refresh, which also
    /// represents later rounds because the refreshed PRF-seed error input is then the same CBD
    /// rounded-error bound in every subsequent round.
    pub prf_seed_bit_refreshes: Vec<NoiseRefreshErrorSimulation>,
    /// Final conservative error bound before AKY24 bit decoding.
    ///
    /// This bound adds the function-evaluation output error, the final PRG mask decode/recompose
    /// error for the selected `prf_mask_output_coeff_bits`, and the `c_b0 * output_preimage`
    /// functional-key decoder term. The PRF mask value range itself is not included here; callers
    /// that search for a valid mask width compare this bound and the candidate mask range against
    /// the `q / 4` margin.
    pub error_bound: PolyMatrixNorm,
}

/// Inputs needed by the top-level AKY24 decryption error simulation.
///
/// The caller supplies the norms that depend on ciphertexts, input encodings, and sampled
/// functional-key material. The simulator then composes them with the AKY24 function circuit, PRF
/// seed refresh path, final mask PRG/decrypt path, and final decoder arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24DecErrorSimulationInputs {
    /// Error bound of the ciphertext `c_b0` term used by AKY24 decoder preimages.
    pub c_b0_error_norm: PolyMatrixNorm,
    /// Error bounds for the inputs to `build_func_circuit`, in circuit input order.
    ///
    /// For the current AKY24 decryption path this means the functional decryption-key encoding
    /// followed by the message input encodings.
    pub func_input_error_norms: Vec<ErrorNorm>,
    /// Error bounds for the current encrypted private PRF-seed bits before public PRF traversal.
    ///
    /// The top-level simulator assumes this vector is uniform across all PRF seed bits and asserts
    /// that invariant before relying on PRG output symmetry.
    pub prf_seed_error_norms: Vec<ErrorNorm>,
    /// AKY24 function whose circuit should be evaluated in the error simulator.
    pub func: Aky24Func,
    /// Norm bound for the functional-key output preimage used in the final `c_b0 * preimage`
    /// decoder term.
    pub output_preimage_norm: PolyMatrixNorm,
}

impl Aky24DecErrorSimulationInputs {
    /// Builds standard AKY24 decryption error-simulation inputs from AKY24 parameters.
    ///
    /// This constructor derives the ciphertext `c_b0` error, function input encoding errors,
    /// initial PRF-seed encoding errors, and final functional-key preimage norm from the parameter
    /// dimensions and Gaussian widths. The public PRF seed is intentionally not included because
    /// the simulator uses the all-zero public-seed branch without changing the error size.
    pub fn new_from_params<TD>(params: &Aky24Params<DCRTPolyMatrix, TD>, func: Aky24Func) -> Self {
        let ring_dim_sqrt = BigDecimal::from(params.poly_params.ring_dimension() as u64)
            .sqrt()
            .expect("sqrt(ring_dimension) failed");
        let base =
            BigDecimal::from(BigInt::from(BigUint::from(1u64) << params.poly_params.base_bits()));
        let ctx = Arc::new(SimulatorContext::new(
            ring_dim_sqrt,
            base,
            params.secret_size(),
            params.poly_params.modulus_digits(),
            params.poly_params.modulus_digits(),
        ));
        let c_b0_error_norm = match params.b_error_sigma {
            Some(sigma) => PolyMatrixNorm::sample_gauss(
                ctx.clone(),
                1,
                ctx.m_b,
                BigDecimal::from_f64(sigma).expect("AKY24 b_error_sigma must be finite"),
            ),
            None => PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, BigDecimal::from(0u32), None),
        };
        let encoding_error_norm = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            match params.encoding_error_sigma {
                Some(sigma) => PolyMatrixNorm::sample_gauss(
                    ctx.clone(),
                    1,
                    ctx.m_g,
                    BigDecimal::from_f64(sigma).expect("AKY24 encoding_error_sigma must be finite"),
                ),
                None => PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(0u32), None),
            },
        );
        let func_input_count = build_func_circuit(params, &func).num_input();
        let output_preimage_norm = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            func.output_size(),
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None),
            None,
        );
        Self {
            c_b0_error_norm,
            func_input_error_norms: vec![encoding_error_norm.clone(); func_input_count],
            prf_seed_error_norms: vec![encoding_error_norm; params.prf_seed_bits()],
            func,
            output_preimage_norm,
        }
    }
}

/// Simulates the first output for a representative encrypted PRF-seed update branch.
///
/// Error size is independent of the public PRF-seed value, so the top-level simulator always uses
/// the first half of the conceptual `2 * prf_seed_bits` PRG output. The top-level simulator feeds
/// this helper uniform seed input norms, so output symmetry makes this representative `ErrorNorm`
/// a bound for every generated seed bit in that half.
pub fn simulate_representative_prf_enc_seed_error<TD, PE>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    round_idx: usize,
    one: ErrorNorm,
    seed_wires: Vec<ErrorNorm>,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    let circuit =
        build_goldreich_prg_range_circuit(params, round_idx, 2 * params.prf_seed_bits(), 0, 1);
    let wire_count = ring_gsw_wire_count(circuit.num_input(), params.prf_seed_bits());
    let seed_wire_errors = expand_logical_bit_errors(&seed_wires, wire_count);
    eval_first_error_output(circuit, one, seed_wire_errors, plt_evaluator, slot_transfer_evaluator)
}

/// Simulates the first output of the final AKY24 PRF-mask expansion PRG.
///
/// This is the PRG call that expands the final refreshed private seed into encrypted mask bits.
/// Output symmetry lets the first mask-bit error stand for all generated mask bits.
pub fn simulate_final_mask_prg_error<TD, PE>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    round_idx: usize,
    conceptual_output_bits: usize,
    one: ErrorNorm,
    seed_wires: Vec<ErrorNorm>,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    let circuit =
        build_goldreich_prg_range_circuit(params, round_idx, conceptual_output_bits, 0, 1);
    let wire_count = ring_gsw_wire_count(circuit.num_input(), params.prf_seed_bits());
    let seed_wire_errors = expand_logical_bit_errors(&seed_wires, wire_count);
    eval_first_error_output(circuit, one, seed_wire_errors, plt_evaluator, slot_transfer_evaluator)
}

/// Runs the top-level AKY24 decryption error simulation.
///
/// The simulation first evaluates `func` over the supplied input `ErrorNorm`s. It then simulates
/// the public PRF-seed traversal by repeatedly evaluating one representative selected-half
/// Goldreich PRG output and running the existing noise-refresh simulator for every private seed
/// bit. Finally, it evaluates the final mask PRG and scalar mask decrypt/recomposition error, adds
/// the `c_b0 * output_preimage` decoder error, and returns the resulting bound for the configured
/// `params.prf_mask_output_coeff_bits()`.
pub fn simulate_aky24_dec_error<TD, PE, ST>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    inputs: Aky24DecErrorSimulationInputs,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    secret_norm: &PolyMatrixNorm,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    decoders: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<Aky24DecErrorSimulation>
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    info!(
        prf_seed_bits = params.prf_seed_bits(),
        public_prf_seed_bits = params.public_prf_seed_bits(),
        prf_mask_output_coeff_bits = params.prf_mask_output_coeff_bits(),
        "starting AKY24 decryption error simulation"
    );
    assert_eq!(
        inputs.prf_seed_error_norms.len(),
        params.prf_seed_bits(),
        "AKY24 PRF seed error norm count must match params.prf_seed_bits"
    );

    let function_error = simulate_function_output_error(
        params,
        inputs.func,
        one.clone(),
        inputs.func_input_error_norms,
        plt_evaluator,
        slot_transfer_evaluator,
    );
    debug!(
        function_error = %function_error.poly_norm.norm,
        "AKY24 function error simulation finished"
    );
    let first_seed_error = inputs
        .prf_seed_error_norms
        .first()
        .expect("AKY24 PRF seed error norm list must be non-empty")
        .clone();
    for seed in inputs.prf_seed_error_norms.iter().skip(1) {
        assert_eq!(
            seed, &first_seed_error,
            "AKY24 PRF seed error simulation assumes uniform PRF seed error norms"
        );
    }
    let mut current_seed_errors = inputs.prf_seed_error_norms;
    let public_prf_seed_bits = params.public_prf_seed_bits();
    let stored_refresh_rounds = public_prf_seed_bits.min(2);
    let mut prf_seed_bit_refreshes = Vec::with_capacity(stored_refresh_rounds);
    for round_idx in 0..stored_refresh_rounds {
        info!(round_idx, "starting representative AKY24 PRF seed refresh simulation");
        let prg_output_error = simulate_representative_prf_enc_seed_error(
            params,
            round_idx,
            one.clone(),
            current_seed_errors.clone(),
            Some(plt_evaluator),
            Some(slot_transfer_evaluator),
        );
        let mut refresh_context_circuit = PolyCircuit::new();
        let refresh = simulate_noise_refresh_error_growth(
            build_ring_gsw_context(params, &mut refresh_context_circuit),
            params.prf_seed_bits(),
            params.goldreich_graph_seed,
            params.noise_refresh_cbd_n,
            ring_gsw_error_sigma,
            num_slots,
            secret_norm,
            one.clone(),
            prg_output_error,
            &current_seed_errors,
            decryption_key.clone(),
            decoders,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        debug!(
            round_idx,
            v_bits = refresh.v_bits,
            "AKY24 representative PRF seed refresh simulation finished"
        );
        let worst_rounded_error = refresh
            .rounded_errors
            .iter()
            .max_by(|lhs, rhs| {
                lhs.poly_norm
                    .norm
                    .partial_cmp(&rhs.poly_norm.norm)
                    .expect("noise-refresh rounded-error norms must be comparable")
            })
            .expect("noise-refresh simulation must produce at least one rounded error")
            .clone();
        let refreshed_seed_error =
            ErrorNorm::new(worst_rounded_error.poly_norm.clone(), worst_rounded_error);
        current_seed_errors = vec![refreshed_seed_error; params.prf_seed_bits()];
        prf_seed_bit_refreshes.push(refresh);
    }

    info!("starting AKY24 final mask PRG error simulation");
    let final_mask_prg_output_error = simulate_final_mask_prg_error(
        params,
        public_prf_seed_bits,
        params.prf_mask_output_coeff_bits(),
        one.clone(),
        current_seed_errors,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
    );
    info!("starting AKY24 final mask decrypt error simulation");
    let mask_circuit = build_prf_mask_circuit(params);
    let mut mask_inputs = Vec::with_capacity(mask_circuit.num_input());
    mask_inputs.push(decryption_key);
    let mask_wire_count =
        ring_gsw_wire_count(mask_circuit.num_input() - 1, params.prf_mask_output_coeff_bits());
    let logical_mask_errors =
        vec![final_mask_prg_output_error; params.prf_mask_output_coeff_bits()];
    mask_inputs.extend(expand_logical_bit_errors(&logical_mask_errors, mask_wire_count));
    assert_eq!(
        mask_inputs.len(),
        mask_circuit.num_input(),
        "AKY24 PRF mask error-simulation input count must match the mask decrypt circuit"
    );
    let mut mask_outputs = mask_circuit.eval(
        &(),
        one,
        mask_inputs,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        None,
    );
    assert_eq!(
        mask_outputs.len(),
        1,
        "AKY24 PRF mask error simulation must produce exactly one output"
    );
    let final_mask_encoding_error = mask_outputs.remove(0).matrix_norm;
    let final_mask_error = final_mask_encoding_error.clone() *
        PolyMatrixNorm::gadget_decomposed(final_mask_encoding_error.clone_ctx(), 1);
    let decoder_error = inputs.c_b0_error_norm * &inputs.output_preimage_norm;
    let error_bound = function_error + final_mask_error + decoder_error;
    info!(
        error_bound = %error_bound.poly_norm.norm,
        "finished AKY24 decryption error simulation"
    );
    Some(Aky24DecErrorSimulation { prf_seed_bit_refreshes, error_bound })
}

/// Searches for the largest PRF mask coefficient bit-width that satisfies the AKY24 decode margin.
///
/// Each candidate bit-width is evaluated with a fresh call to `simulate_aky24_dec_error`, so the
/// returned value accounts for that candidate's PRG and mask decrypt/recomposition error instead of
/// reusing an error bound from another mask width.
pub fn max_safe_aky24_prf_mask_output_coeff_bits<TD, PE, ST>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    inputs: Aky24DecErrorSimulationInputs,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    secret_norm: &PolyMatrixNorm,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    decoders: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    max_bits: usize,
) -> Option<Aky24PrfMaskOutputCoeffBitsSearchResult>
where
    TD: Clone,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    info!(max_bits, "starting AKY24 PRF mask output coefficient bit search");
    let max_candidate = max_bits.min(params.poly_params.modulus_bits());
    let mut low = 1usize;
    let mut high = max_candidate;
    let mut best = None;
    while low <= high {
        let candidate = low + (high - low) / 2;
        info!(candidate, low, high, "evaluating AKY24 PRF mask bit candidate");
        let mut candidate_params = params.clone();
        candidate_params.prf_mask_output_coeff_bits = candidate;
        let simulation = simulate_aky24_dec_error(
            &candidate_params,
            inputs.clone(),
            ring_gsw_error_sigma,
            num_slots,
            secret_norm,
            one.clone(),
            decryption_key.clone(),
            decoders,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        let q: Arc<BigUint> = candidate_params.poly_params.modulus().into();
        let quarter_q = biguint_to_decimal(&(q.as_ref() / 4u32));
        let mask_value = biguint_to_decimal(&(BigUint::from(1u32) << candidate));
        let valid = &simulation.error_bound.poly_norm.norm + &mask_value < quarter_q;
        debug!(
            candidate,
            valid,
            error_bound = %simulation.error_bound.poly_norm.norm,
            "AKY24 PRF mask bit candidate evaluated"
        );
        if valid {
            best = Some(Aky24PrfMaskOutputCoeffBitsSearchResult {
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
    info!(found = best.is_some(), "finished AKY24 PRF mask output coefficient bit search");
    best
}

/// Finds the smallest CRT depth whose correctness and security checks pass.
///
/// The search is binary over the inclusive range `[min_crt_depth, max_crt_depth]`. For each
/// candidate depth it first calls `max_safe_aky24_prf_mask_output_coeff_bits`; `None` is treated as
/// a correctness failure and moves the search to larger CRT depths. If mask-bit search succeeds,
/// this function re-runs the full error simulation at the discovered mask width and checks every
/// per-round noise-refresh error/mask pair plus the final decode error/mask pair with
/// `validate_error_bound_security_margin`. Any failed margin check is treated as a security failure
/// and also moves the search to larger CRT depths. Passing candidates are recorded and the search
/// continues toward smaller CRT depths.
pub fn aky24_find_crt_depth<TD, PE, ST, BuildCandidate>(
    min_crt_depth: usize,
    max_crt_depth: usize,
    max_prf_mask_output_coeff_bits: usize,
    security_bit: usize,
    ring_gsw_error_sigma: f64,
    num_slots: usize,
    mut build_candidate: BuildCandidate,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<Aky24CrtDepthSearchResult>
where
    TD: Clone,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
    BuildCandidate: FnMut(usize) -> Aky24CrtDepthSearchCandidate<TD>,
{
    info!(
        min_crt_depth,
        max_crt_depth,
        max_prf_mask_output_coeff_bits,
        security_bit,
        "starting AKY24 CRT-depth search"
    );
    assert!(min_crt_depth > 0, "minimum CRT depth must be positive");
    assert!(min_crt_depth <= max_crt_depth, "CRT-depth search range must be non-empty");
    let mut low = min_crt_depth;
    let mut high = max_crt_depth;
    let mut found = Vec::new();
    while low <= high {
        let crt_depth = low + (high - low) / 2;
        info!(crt_depth, low, high, "evaluating AKY24 CRT-depth candidate");
        let candidate = build_candidate(crt_depth);
        let Some(mask_search) = max_safe_aky24_prf_mask_output_coeff_bits(
            &candidate.params,
            candidate.inputs.clone(),
            ring_gsw_error_sigma,
            num_slots,
            &candidate.secret_norm,
            candidate.one.clone(),
            candidate.decryption_key.clone(),
            &candidate.decoders,
            plt_evaluator,
            slot_transfer_evaluator,
            max_prf_mask_output_coeff_bits,
        ) else {
            info!(crt_depth, "AKY24 CRT-depth candidate failed correctness mask-bit search");
            low = crt_depth + 1;
            continue;
        };
        let mask_bits = mask_search.prf_mask_output_coeff_bits;
        let mut checked_params = candidate.params.clone();
        checked_params.prf_mask_output_coeff_bits = mask_bits;

        if aky24_security_margins_hold(
            &checked_params,
            &mask_search.simulation,
            &candidate.secret_norm,
            mask_bits,
            security_bit,
        ) {
            info!(crt_depth, mask_bits, "AKY24 CRT-depth candidate passed");
            found.push(Aky24CrtDepthSearchResult {
                crt_depth,
                prf_mask_output_coeff_bits: mask_bits,
            });
            if crt_depth == 0 {
                break;
            }
            high = crt_depth - 1;
        } else {
            info!(crt_depth, mask_bits, "AKY24 CRT-depth candidate failed security margins");
            low = crt_depth + 1;
        }
    }
    let result = found.into_iter().min_by_key(|candidate| candidate.crt_depth);
    info!(found = result.is_some(), "finished AKY24 CRT-depth search");
    result
}

fn aky24_security_margins_hold<TD>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    simulation: &Aky24DecErrorSimulation,
    secret_norm: &PolyMatrixNorm,
    prf_mask_output_coeff_bits: usize,
    security_bit: usize,
) -> bool {
    let Some(final_threshold) =
        aky24_final_decode_margin(&params.poly_params, prf_mask_output_coeff_bits)
    else {
        return false;
    };
    if !validate_error_bound_security_margin(
        &final_threshold,
        &simulation.error_bound.poly_norm.norm,
        security_bit,
    ) {
        return false;
    }
    simulation.prf_seed_bit_refreshes.iter().all(|refresh| {
        let Some(threshold) =
            noise_refresh_pre_round_margin(&params.poly_params, secret_norm, refresh.v_bits)
        else {
            return false;
        };
        let worst_error = refresh
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

fn aky24_final_decode_margin(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    prf_mask_output_coeff_bits: usize,
) -> Option<BigDecimal> {
    let q: Arc<BigUint> = params.modulus().into();
    let threshold = biguint_to_decimal(&(q.as_ref() / 4u32));
    let mask = biguint_to_decimal(&(BigUint::from(1u32) << prf_mask_output_coeff_bits));
    (threshold > mask).then_some(threshold - mask)
}

fn noise_refresh_pre_round_margin(
    params: &<DCRTPoly as crate::poly::Poly>::Params,
    secret_norm: &PolyMatrixNorm,
    v_bits: usize,
) -> Option<BigDecimal> {
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    let threshold =
        biguint_to_decimal(full_q.as_ref()) / (BigDecimal::from(2u32) * BigDecimal::from(q_max));
    let mask = biguint_to_decimal(&(BigUint::from(1u32) << v_bits));
    let used_margin = &secret_norm.poly_norm.norm + mask;
    (threshold > used_margin).then_some(threshold - used_margin)
}

/// Evaluates the AKY24 function circuit and returns the first decoded output's matrix error.
///
/// The function circuit output is a BGG encoding norm. AKY24's current final decode extracts the
/// first function output, so this helper returns that output's matrix error contribution.
fn simulate_function_output_error<TD, PE>(
    params: &Aky24Params<DCRTPolyMatrix, TD>,
    func: Aky24Func,
    one: ErrorNorm,
    inputs: Vec<ErrorNorm>,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &dyn SlotTransferEvaluator<ErrorNorm>,
) -> PolyMatrixNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    let circuit = build_func_circuit(params, &func);
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "AKY24 function error-simulation input count must match the function circuit"
    );
    let ctx = inputs
        .first()
        .expect("AKY24 function error simulation must receive at least one input")
        .matrix_norm
        .clone_ctx();
    let outputs =
        circuit.eval(&(), one, inputs, Some(plt_evaluator), Some(slot_transfer_evaluator), None);
    assert_eq!(
        outputs.len(),
        func.output_size(),
        "AKY24 function error simulation must produce one output per function output"
    );
    outputs
        .first()
        .expect("AKY24 function error simulation must produce at least one output")
        .matrix_norm
        .clone() *
        PolyMatrixNorm::gadget_decomposed(ctx, 1)
}

/// Evaluates an AKY24 helper circuit using `ErrorNorm` and returns the first output.
///
/// This is used for PRG error simulation, where output symmetry makes the first output a sufficient
/// representative bound. The helper restricts the circuit output list before evaluation, so
/// symmetric outputs that do not affect this representative bound are not evaluated.
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
        "AKY24 first-output error-simulation input count must match the helper circuit"
    );
    assert!(
        circuit.num_output() > 0,
        "AKY24 first-output error-simulation helper must expose at least one output"
    );
    circuit.restrict_outputs_to_indices(&[0]);
    let mut outputs = circuit.eval(&(), one, inputs, plt_evaluator, slot_transfer_evaluator, None);
    assert_eq!(
        outputs.len(),
        1,
        "AKY24 first-output error simulation must produce exactly one output"
    );
    outputs.remove(0)
}

/// Repeats one logical Ring-GSW ciphertext error bound for every flattened circuit input wire.
fn expand_logical_bit_errors(logical_errors: &[ErrorNorm], wire_count: usize) -> Vec<ErrorNorm> {
    assert!(wire_count > 0, "Ring-GSW wire count must be positive");
    logical_errors.iter().flat_map(|error| std::iter::repeat_n(error.clone(), wire_count)).collect()
}

/// Computes the flattened Ring-GSW wire count from a circuit input count and logical bit count.
fn ring_gsw_wire_count(flat_input_count: usize, logical_bit_count: usize) -> usize {
    assert!(logical_bit_count > 0, "logical Ring-GSW bit count must be positive");
    assert_eq!(
        flat_input_count % logical_bit_count,
        0,
        "flattened Ring-GSW input count must be divisible by logical bit count"
    );
    flat_input_count / logical_bit_count
}

/// Converts an unsigned integer modulus quantity into `BigDecimal` for margin arithmetic.
fn biguint_to_decimal(value: &BigUint) -> BigDecimal {
    BigDecimal::from(BigInt::from(value.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::gate::GateId,
        gadgets::arith::{ModularArithmeticContext, NestedRnsPolyContext},
        poly::dcrt::params::DCRTPolyParams,
    };

    fn test_params() -> Aky24Params<DCRTPolyMatrix, ()> {
        test_params_with_crt_depth(1)
    }

    fn test_params_with_crt_depth(crt_depth: usize) -> Aky24Params<DCRTPolyMatrix, ()> {
        let ring_dim = 2;
        let crt_bits = 10;
        let base_bits = (crt_bits / 2) as u32;
        let poly_params = DCRTPolyParams::new(ring_dim, crt_depth, crt_bits, base_bits);
        let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
        let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
            &mut setup_circuit,
            &poly_params,
            5,
            2,
            1 << 8,
            false,
            Some(crt_depth),
        ));
        let ring_gsw_level_offset = 0;
        let ring_gsw_enable_levels = Some(crt_depth);
        let ring_gsw_width = 2 *
            <NestedRnsPolyContext as ModularArithmeticContext<DCRTPoly>>::gadget_len(
                ring_gsw_context.as_ref(),
                ring_gsw_enable_levels,
                Some(ring_gsw_level_offset),
            );
        Aky24Params::<DCRTPolyMatrix, ()>::new(
            poly_params.clone(),
            poly_params.clone(),
            ring_gsw_context,
            ring_gsw_width,
            ring_gsw_level_offset,
            ring_gsw_enable_levels,
            Some(0.0),
            b"aky24_error_sim_test".to_vec(),
            4.578,
            None,
            DCRTPolyMatrix::zero(&poly_params, 1, 1),
            (),
            5,
            2,
            Some(2),
            [0x42; 32],
            1,
            1,
            [0x24; 32],
        )
    }

    struct TestSlotTransferEvaluator {
        transfer: crate::simulator::error_norm::NormBggPolyEncodingSTEvaluator,
    }

    impl SlotTransferEvaluator<ErrorNorm> for TestSlotTransferEvaluator {
        fn slot_transfer(
            &self,
            params: &(),
            input: &ErrorNorm,
            src_slots: &[(u32, Option<u32>)],
            gate_id: GateId,
        ) -> ErrorNorm {
            self.transfer.slot_transfer(params, input, src_slots, gate_id)
        }

        fn slot_reduce(
            &self,
            _params: &(),
            inputs: &[ErrorNorm],
            _num_slots: usize,
            _gate_id: GateId,
        ) -> ErrorNorm {
            inputs
                .iter()
                .cloned()
                .reduce(|acc, input| &acc + &input)
                .expect("slot_reduce test input must be non-empty")
        }
    }

    #[test]
    fn test_error_simulation_expands_logical_bits_to_flat_ring_gsw_inputs() {
        let params = test_params();
        let selected_prg =
            build_goldreich_prg_range_circuit(&params, 0, 2 * params.prf_seed_bits(), 0, 1);
        let prg_wire_count = ring_gsw_wire_count(selected_prg.num_input(), params.prf_seed_bits());
        assert!(prg_wire_count > 1, "test must exercise flattened Ring-GSW inputs");

        let inputs =
            Aky24DecErrorSimulationInputs::new_from_params(&params, Aky24Func::DebugIdentity);
        let expanded_seed_errors =
            expand_logical_bit_errors(&inputs.prf_seed_error_norms, prg_wire_count);
        assert_eq!(expanded_seed_errors.len(), selected_prg.num_input());

        let mask_circuit = build_prf_mask_circuit(&params);
        let mask_wire_count =
            ring_gsw_wire_count(mask_circuit.num_input() - 1, params.prf_mask_output_coeff_bits());
        assert_eq!(
            1 + params.prf_mask_output_coeff_bits() * mask_wire_count,
            mask_circuit.num_input(),
            "mask decrypt error simulation must feed one key plus flattened mask ciphertext wires"
        );
    }

    #[test]
    #[ignore = "expensive CRT-depth smoke test; run explicitly when changing search orchestration"]
    fn test_aky24_find_crt_depth_returns_candidate() {
        let plt_evaluator = {
            let params = test_params();
            let inputs =
                Aky24DecErrorSimulationInputs::new_from_params(&params, Aky24Func::DebugIdentity);
            crate::simulator::error_norm::NormPltLWEEvaluator::new(
                inputs.c_b0_error_norm.clone_ctx(),
                &BigDecimal::from(0u32),
            )
        };
        let slot_transfer_evaluator = {
            let params = test_params();
            let inputs =
                Aky24DecErrorSimulationInputs::new_from_params(&params, Aky24Func::DebugIdentity);
            TestSlotTransferEvaluator {
                transfer: crate::simulator::error_norm::NormBggPolyEncodingSTEvaluator::new(
                    inputs.c_b0_error_norm.clone_ctx(),
                    0.0,
                    &BigDecimal::from(0u32),
                    None,
                ),
            }
        };

        let result = aky24_find_crt_depth(
            1,
            1,
            1,
            0,
            0.0,
            2,
            |crt_depth| {
                let mut params = test_params_with_crt_depth(crt_depth);
                params.public_prf_seed_bits = Some(0);
                let inputs = Aky24DecErrorSimulationInputs::new_from_params(
                    &params,
                    Aky24Func::DebugIdentity,
                );
                let ctx = inputs.c_b0_error_norm.clone_ctx();
                let one = ErrorNorm::new(
                    PolyNorm::one(ctx.clone()),
                    PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(0u32), None),
                );
                let decryption_key = ErrorNorm::new(
                    PolyNorm::one(ctx.clone()),
                    PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(0u32), None),
                );
                let secret_norm = PolyMatrixNorm::new(
                    ctx.clone(),
                    1,
                    ctx.secret_size,
                    BigDecimal::from(0u32),
                    None,
                );
                let decoder = ErrorNorm::new(
                    PolyNorm::one(ctx.clone()),
                    PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(0u32), None),
                );
                Aky24CrtDepthSearchCandidate {
                    params,
                    inputs,
                    secret_norm,
                    one,
                    decryption_key,
                    decoders: vec![decoder],
                }
            },
            &plt_evaluator,
            &slot_transfer_evaluator,
        );
        assert!(result.is_some(), "AKY24 CRT-depth search should find a test candidate");
    }
}
