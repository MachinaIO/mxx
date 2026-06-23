use std::{sync::Arc, thread, time::Duration};

use bigdecimal::BigDecimal;
use num_bigint::{BigInt, BigUint};
use num_traits::FromPrimitive;

use crate::{
    circuit::{Evaluable, PolyCircuit},
    decoder::{
        mask_circuit::build_one_ciphertext_bit_decrypt_circuit,
        simulation::{
            DecodeThreshold, decode_margin, decode_threshold,
            max_centered_mask_bits_for_available_range,
        },
    },
    gadgets::{
        arith::{NestedRnsPoly, NestedRnsPolyContext},
        fhe::{ring_gsw::RingGswCiphertext, ring_gsw_nested_rns::NestedRnsRingGswContext},
        fhe_prg::goldreich::{
            GoldreichEdge, GoldreichFhePrg, GoldreichGraph, goldreich_output_bound_holds,
        },
    },
    lookup::PltEvaluator,
    matrix::dcrt_poly::DCRTPolyMatrix,
    noise_refresh::{
        NoiseRefreshErrorSimulation,
        simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_material_override,
    },
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    simulator::{
        SimulatorContext,
        error_norm::ErrorNorm,
        lattice_estimator::{Distribution, run_lattice_estimator_cli_with_timeout},
        poly_matrix_norm::PolyMatrixNorm,
    },
    slot_transfer::SlotTransferEvaluator,
};
use tracing::info;

const LATTICE_ESTIMATOR_TIMEOUT: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SecureRingDimSearchResult {
    pub log_ring_dim: usize,
    pub ring_dim: u32,
    pub achieved_secpar_for_gauss: Option<u64>,
    pub achieved_secpar_for_cbd: Option<u64>,
}

#[derive(Debug, Default)]
pub(crate) struct SecureRingDimLatticeCache {
    entries: Vec<(usize, SecureRingDimSearchResult)>,
}

impl SecureRingDimLatticeCache {
    pub(crate) fn secure_for(
        &self,
        crt_depth: usize,
        log_ring_dim: usize,
    ) -> Option<SecureRingDimSearchResult> {
        self.entries
            .iter()
            .filter(|(passed_crt_depth, result)| {
                *passed_crt_depth >= crt_depth && result.log_ring_dim == log_ring_dim
            })
            .max_by_key(|(passed_crt_depth, _result)| *passed_crt_depth)
            .map(|(_passed_crt_depth, result)| *result)
    }

    pub(crate) fn record(&mut self, crt_depth: usize, result: SecureRingDimSearchResult) {
        self.entries.push((crt_depth, result));
    }
}

pub(crate) fn assert_lattice_estimator_available(protocol_name: &str) {
    let ring_dim = BigUint::from(1024u32);
    let q = BigUint::from(3329u32);
    let s_dist = Distribution::Ternary;
    let e_dist = Distribution::DiscreteGaussian { stddev: "4".to_string(), mean: None, n: None };
    info!(
        protocol_name,
        ring_dim = %ring_dim,
        q = %q,
        "probing lattice-estimator-cli availability"
    );
    let achieved_secpar = run_lattice_estimator_cli_with_timeout(
        &ring_dim,
        &q,
        &s_dist,
        &e_dist,
        None,
        false,
        LATTICE_ESTIMATOR_TIMEOUT,
    )
    .unwrap_or_else(|err| {
        panic!(
            "{protocol_name} CRT-depth search requires a working lattice-estimator-cli on PATH. \
             If SageMath is installed in the `sage` conda environment, run this command inside \
             that environment, e.g. `conda run -n sage ...` or activate it before running tests. \
             Lattice estimator probe failed: {err}"
        )
    });
    info!(protocol_name, achieved_secpar, "lattice-estimator-cli availability probe succeeded");
}

pub(crate) fn select_min_secure_ring_dim<P, BuildParams>(
    protocol_name: &str,
    crt_depth: usize,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
    security_bits: usize,
    error_sigma: f64,
    noise_refresh_cbd_n: usize,
    skip_lattice_check: bool,
    lattice_cache: &mut SecureRingDimLatticeCache,
    mut build_params: BuildParams,
) -> Option<SecureRingDimSearchResult>
where
    P: PolyParams,
    BuildParams: FnMut(u32) -> P,
{
    assert!(
        min_log_ring_dim <= max_log_ring_dim,
        "{protocol_name} log-ring-dimension search range must be non-empty"
    );
    assert!(
        max_log_ring_dim < u32::BITS as usize,
        "{protocol_name} max_log_ring_dim must be less than 32"
    );
    assert!(
        error_sigma >= 0.0,
        "{protocol_name} lattice-estimator Gaussian stddev must be nonnegative"
    );
    assert!(noise_refresh_cbd_n > 0, "{protocol_name} lattice-estimator CBD eta must be positive");
    if skip_lattice_check {
        assert_eq!(
            min_log_ring_dim, max_log_ring_dim,
            "{protocol_name} explicit lattice-check skip requires a single log_ring_dim"
        );
        let ring_dim = 1u32
            .checked_shl(min_log_ring_dim.try_into().expect("log_ring_dim must fit in u32"))
            .expect("ring_dim shift overflow");
        info!(
            protocol_name,
            crt_depth,
            log_ring_dim = min_log_ring_dim,
            ring_dim,
            "skipping lattice-estimator security checks because an explicit single-log-ring-dim skip was requested"
        );
        return Some(SecureRingDimSearchResult {
            log_ring_dim: min_log_ring_dim,
            ring_dim,
            achieved_secpar_for_gauss: None,
            achieved_secpar_for_cbd: None,
        });
    }
    let s_dist = Distribution::Ternary;
    let e_dist_gauss =
        Distribution::DiscreteGaussian { stddev: error_sigma.to_string(), mean: None, n: None };
    let e_dist_cbd = Distribution::CenteredBinomial {
        eta: noise_refresh_cbd_n.try_into().expect("noise_refresh_cbd_n must fit in u64"),
        n: None,
    };
    let required_security: u64 = security_bits.try_into().expect("security_bits must fit in u64");
    let mut low = min_log_ring_dim;
    let mut high = max_log_ring_dim;
    let mut found = None;
    while low <= high {
        let log_ring_dim = low + (high - low) / 2;
        let ring_dim = 1u32
            .checked_shl(log_ring_dim.try_into().expect("log_ring_dim must fit in u32"))
            .expect("ring_dim shift overflow");
        if let Some(cached) = lattice_cache.secure_for(crt_depth, log_ring_dim) {
            info!(
                protocol_name,
                crt_depth,
                log_ring_dim,
                ring_dim,
                achieved_secpar_for_gauss = cached.achieved_secpar_for_gauss,
                achieved_secpar_for_cbd = cached.achieved_secpar_for_cbd,
                "skipping lattice-estimator security checks using larger CRT-depth cache"
            );
            found = Some(cached);
            if log_ring_dim == 0 {
                break;
            }
            high = log_ring_dim - 1;
            continue;
        }
        let params = build_params(ring_dim);
        let q: Arc<BigUint> = params.modulus().into();
        let ring_dim_big = BigUint::from(ring_dim);
        info!(
            protocol_name,
            crt_depth,
            log_ring_dim,
            ring_dim,
            modulus_bits = q.bits(),
            required_security,
            error_sigma,
            noise_refresh_cbd_n,
            "running lattice-estimator security checks for CRT-depth ring-dimension candidate"
        );
        let (achieved_secpar_for_gauss, achieved_secpar_for_cbd) = thread::scope(|scope| {
            let gauss_handle = scope.spawn(|| {
                run_lattice_estimator_cli_with_timeout(
                    &ring_dim_big,
                    q.as_ref(),
                    &s_dist,
                    &e_dist_gauss,
                    None,
                    false,
                    LATTICE_ESTIMATOR_TIMEOUT,
                )
            });
            let cbd_handle = scope.spawn(|| {
                run_lattice_estimator_cli_with_timeout(
                    &ring_dim_big,
                    q.as_ref(),
                    &s_dist,
                    &e_dist_cbd,
                    None,
                    false,
                    LATTICE_ESTIMATOR_TIMEOUT,
                )
            });
            (
                gauss_handle.join().expect("Gaussian lattice-estimator thread panicked"),
                cbd_handle.join().expect("CBD lattice-estimator thread panicked"),
            )
        });
        match (achieved_secpar_for_gauss, achieved_secpar_for_cbd) {
            (Ok(achieved_secpar_for_gauss), Ok(achieved_secpar_for_cbd)) => {
                info!(
                    protocol_name,
                    crt_depth,
                    log_ring_dim,
                    ring_dim,
                    achieved_secpar_for_gauss,
                    achieved_secpar_for_cbd,
                    required_security,
                    "evaluated CRT-depth ring-dimension security candidate"
                );
                if achieved_secpar_for_gauss >= required_security &&
                    achieved_secpar_for_cbd >= required_security
                {
                    let result = SecureRingDimSearchResult {
                        log_ring_dim,
                        ring_dim,
                        achieved_secpar_for_gauss: Some(achieved_secpar_for_gauss),
                        achieved_secpar_for_cbd: Some(achieved_secpar_for_cbd),
                    };
                    lattice_cache.record(crt_depth, result);
                    found = Some(result);
                    if log_ring_dim == 0 {
                        break;
                    }
                    high = log_ring_dim - 1;
                } else {
                    low = log_ring_dim + 1;
                }
            }
            (gauss_result, cbd_result) => {
                info!(
                    protocol_name,
                    crt_depth,
                    log_ring_dim,
                    ring_dim,
                    gauss_error = ?gauss_result.err(),
                    cbd_error = ?cbd_result.err(),
                    "lattice-estimator failed for CRT-depth ring-dimension candidate"
                );
                low = log_ring_dim + 1;
            }
        }
    }
    if found.is_none() {
        info!(
            protocol_name,
            crt_depth,
            min_log_ring_dim,
            max_log_ring_dim,
            required_security,
            "no secure ring dimension found for CRT-depth candidate"
        );
    }
    found
}

pub(crate) fn select_min_secure_ring_dim_gaussian_only<P, BuildParams>(
    protocol_name: &str,
    crt_depth: usize,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
    security_bits: usize,
    error_sigma: f64,
    skip_lattice_check: bool,
    lattice_cache: &mut SecureRingDimLatticeCache,
    mut build_params: BuildParams,
) -> Option<SecureRingDimSearchResult>
where
    P: PolyParams,
    BuildParams: FnMut(u32) -> P,
{
    assert!(
        min_log_ring_dim <= max_log_ring_dim,
        "{protocol_name} log-ring-dimension search range must be non-empty"
    );
    assert!(
        max_log_ring_dim < u32::BITS as usize,
        "{protocol_name} max_log_ring_dim must be less than 32"
    );
    assert!(
        error_sigma >= 0.0,
        "{protocol_name} lattice-estimator Gaussian stddev must be nonnegative"
    );
    if skip_lattice_check {
        assert_eq!(
            min_log_ring_dim, max_log_ring_dim,
            "{protocol_name} explicit lattice-check skip requires a single log_ring_dim"
        );
        let ring_dim = 1u32
            .checked_shl(min_log_ring_dim.try_into().expect("log_ring_dim must fit in u32"))
            .expect("ring_dim shift overflow");
        info!(
            protocol_name,
            crt_depth,
            log_ring_dim = min_log_ring_dim,
            ring_dim,
            "skipping Gaussian lattice-estimator security check because an explicit single-log-ring-dim skip was requested"
        );
        return Some(SecureRingDimSearchResult {
            log_ring_dim: min_log_ring_dim,
            ring_dim,
            achieved_secpar_for_gauss: None,
            achieved_secpar_for_cbd: None,
        });
    }
    let s_dist = Distribution::Ternary;
    let e_dist_gauss =
        Distribution::DiscreteGaussian { stddev: error_sigma.to_string(), mean: None, n: None };
    let required_security: u64 = security_bits.try_into().expect("security_bits must fit in u64");
    let mut low = min_log_ring_dim;
    let mut high = max_log_ring_dim;
    let mut found = None;
    while low <= high {
        let log_ring_dim = low + (high - low) / 2;
        let ring_dim = 1u32
            .checked_shl(log_ring_dim.try_into().expect("log_ring_dim must fit in u32"))
            .expect("ring_dim shift overflow");
        if let Some(cached) = lattice_cache.secure_for(crt_depth, log_ring_dim) {
            info!(
                protocol_name,
                crt_depth,
                log_ring_dim,
                ring_dim,
                achieved_secpar_for_gauss = cached.achieved_secpar_for_gauss,
                "skipping Gaussian lattice-estimator security check using larger CRT-depth cache"
            );
            found = Some(cached);
            if log_ring_dim == 0 {
                break;
            }
            high = log_ring_dim - 1;
            continue;
        }
        let params = build_params(ring_dim);
        let q: Arc<BigUint> = params.modulus().into();
        let ring_dim_big = BigUint::from(ring_dim);
        info!(
            protocol_name,
            crt_depth,
            log_ring_dim,
            ring_dim,
            modulus_bits = q.bits(),
            required_security,
            error_sigma,
            "running Gaussian lattice-estimator security check for CRT-depth ring-dimension candidate"
        );
        match run_lattice_estimator_cli_with_timeout(
            &ring_dim_big,
            q.as_ref(),
            &s_dist,
            &e_dist_gauss,
            None,
            false,
            LATTICE_ESTIMATOR_TIMEOUT,
        ) {
            Ok(achieved_secpar_for_gauss) => {
                info!(
                    protocol_name,
                    crt_depth,
                    log_ring_dim,
                    ring_dim,
                    achieved_secpar_for_gauss,
                    required_security,
                    "evaluated CRT-depth ring-dimension Gaussian security candidate"
                );
                if achieved_secpar_for_gauss >= required_security {
                    let result = SecureRingDimSearchResult {
                        log_ring_dim,
                        ring_dim,
                        achieved_secpar_for_gauss: Some(achieved_secpar_for_gauss),
                        achieved_secpar_for_cbd: None,
                    };
                    lattice_cache.record(crt_depth, result);
                    found = Some(result);
                    if log_ring_dim == 0 {
                        break;
                    }
                    high = log_ring_dim - 1;
                } else {
                    low = log_ring_dim + 1;
                }
            }
            Err(err) => {
                info!(
                    protocol_name,
                    crt_depth,
                    log_ring_dim,
                    ring_dim,
                    gauss_error = ?err,
                    "Gaussian lattice-estimator failed for CRT-depth ring-dimension candidate"
                );
                low = log_ring_dim + 1;
            }
        }
    }
    if found.is_none() {
        info!(
            protocol_name,
            crt_depth,
            min_log_ring_dim,
            max_log_ring_dim,
            required_security,
            "no Gaussian-secure ring dimension found for CRT-depth candidate"
        );
    }
    found
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CpuRingGswContextConfig {
    pub p_moduli_bits: usize,
    pub max_unreduced_muls: usize,
    pub scale: u64,
    pub level_offset: usize,
}

pub(crate) fn cpu_params_from_poly_params<P: PolyParams>(
    params: &P,
    protocol_name: &str,
) -> DCRTPolyParams {
    let (q_moduli, crt_bits, crt_depth) = params.to_crt();
    let cpu_params =
        DCRTPolyParams::new(params.ring_dimension(), crt_depth, crt_bits, params.base_bits());
    let (cpu_q_moduli, cpu_crt_bits, cpu_crt_depth) = cpu_params.to_crt();
    assert_eq!(cpu_crt_bits, crt_bits, "CPU simulation CRT bit width mismatch");
    assert_eq!(cpu_crt_depth, crt_depth, "CPU simulation CRT depth mismatch");
    assert_eq!(
        cpu_q_moduli, q_moduli,
        "CPU simulation parameters must reproduce the {protocol_name} CRT moduli"
    );
    cpu_params
}

pub(crate) fn simulator_context(
    params: &DCRTPolyParams,
    secret_size: usize,
) -> Arc<SimulatorContext> {
    let ring_dim_sqrt = BigDecimal::from(params.ring_dimension() as u64)
        .sqrt()
        .expect("sqrt(ring_dimension) failed");
    let base = BigDecimal::from(BigInt::from(BigUint::from(1u64) << params.base_bits()));
    Arc::new(SimulatorContext::new(
        ring_dim_sqrt,
        base,
        secret_size,
        params.modulus_digits(),
        params.modulus_digits(),
    ))
}

pub(crate) fn final_decode_margin(
    params: &DCRTPolyParams,
    prf_mask_output_coeff_bits: usize,
) -> Option<BigDecimal> {
    let q: Arc<BigUint> = params.modulus().into();
    decode_margin(q.as_ref(), prf_mask_output_coeff_bits, &DecodeThreshold::boolean())
}

pub(crate) fn final_mask_coeff_bits_for_error_margin(
    params: &DCRTPolyParams,
    error_bound: &BigDecimal,
    security_bit: Option<usize>,
    max_bits: usize,
) -> Option<usize> {
    let q: Arc<BigUint> = params.modulus().into();
    let reserved_error = scaled_error_bound(error_bound, security_bit);
    let available = decode_threshold(q.as_ref(), &DecodeThreshold::boolean()) - reserved_error;
    max_centered_mask_bits_for_available_range(&available, max_bits)
}

#[cfg(test)]
pub(crate) fn noise_refresh_pre_round_margin(
    params: &DCRTPolyParams,
    v_bits: usize,
) -> Option<BigDecimal> {
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    decode_margin(full_q.as_ref(), v_bits, &DecodeThreshold::new(q_max))
}

pub(crate) fn noise_refresh_v_bits_for_error_margin(
    params: &DCRTPolyParams,
    pre_rounding_error: &BigDecimal,
    security_bit: Option<usize>,
    max_bits: usize,
) -> Option<usize> {
    let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
    let q_max = q_moduli.iter().copied().max().expect("CRT modulus list must be nonempty");
    let full_q: Arc<BigUint> = params.modulus().into();
    let reserved_error = scaled_error_bound(pre_rounding_error, security_bit);
    let available =
        decode_threshold(full_q.as_ref(), &DecodeThreshold::new(q_max)) - reserved_error;
    max_centered_mask_bits_for_available_range(&available, max_bits)
}

pub(crate) fn noise_refresh_security_margin_holds(
    params: &DCRTPolyParams,
    refresh: &NoiseRefreshErrorSimulation,
    security_bit: usize,
) -> bool {
    noise_refresh_v_bits_for_error_margin(
        params,
        &noise_refresh_worst_pre_rounding_error_bound(refresh),
        Some(security_bit),
        params.modulus_bits(),
    )
    .is_some_and(|max_bits| refresh.v_bits <= max_bits)
}

pub(crate) fn noise_refresh_worst_pre_rounding_error_bound(
    refresh: &NoiseRefreshErrorSimulation,
) -> BigDecimal {
    refresh
        .pre_round_outputs
        .iter()
        .map(|error| error.maximum_coefficient_bound())
        .max_by(|lhs, rhs| {
            lhs.partial_cmp(rhs).expect("noise-refresh pre-rounding bounds must be comparable")
        })
        .expect("noise-refresh simulation must produce at least one pre-round output")
}

fn scaled_error_bound(error_bound: &BigDecimal, security_bit: Option<usize>) -> BigDecimal {
    match security_bit {
        Some(security_bit) => {
            let security_factor =
                BigDecimal::from(BigInt::from(BigUint::from(1u32) << security_bit));
            error_bound * security_factor
        }
        None => error_bound.clone(),
    }
}

pub(crate) fn minimum_seed_refresh_prf_seed_bits(
    branch_count: usize,
    protocol_name: &str,
    initial_seed_bits: usize,
) -> usize {
    assert!(branch_count > 0, "{protocol_name} seed-refresh branch count must be positive");
    let mut seed_bits = initial_seed_bits;
    while !goldreich_output_bound_holds(
        seed_bits,
        seed_bits
            .checked_mul(branch_count)
            .unwrap_or_else(|| panic!("{protocol_name} seed-refresh PRF output size overflow")),
    ) {
        seed_bits = seed_bits
            .checked_add(1)
            .unwrap_or_else(|| panic!("{protocol_name} seed-bit search overflow"));
    }
    seed_bits
}

pub(crate) fn max_final_mask_coeff_bits_for_seed<F>(
    params: &DCRTPolyParams,
    output_size: usize,
    function_output_bits: usize,
    seed_bits: usize,
    max_bits: usize,
    final_prg_uniform_output_bits: F,
) -> usize
where
    F: Fn(usize, usize, usize, usize) -> usize,
{
    if max_bits == 0 {
        return 0;
    }
    let ring_dim = params.ring_dimension() as usize;
    let mut low = 1usize;
    let mut high = max_bits;
    let mut best = 0usize;
    while low <= high {
        let candidate = low + (high - low) / 2;
        let output_bits =
            final_prg_uniform_output_bits(output_size, ring_dim, candidate, function_output_bits);
        if goldreich_output_bound_holds(seed_bits, output_bits) {
            best = candidate;
            low = candidate + 1;
        } else if candidate == 1 {
            break;
        } else {
            high = candidate - 1;
        }
    }
    best
}

pub(crate) fn eval_first_error_output<PE>(
    mut circuit: PolyCircuit<DCRTPoly>,
    one: ErrorNorm,
    inputs: Vec<ErrorNorm>,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
    protocol_name: &str,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "{protocol_name} first-output error-simulation input count mismatch"
    );
    assert!(circuit.num_output() > 0, "{protocol_name} representative circuit has no output");
    circuit.restrict_outputs_to_indices(&[0]);
    let mut outputs = circuit.eval(&(), one, inputs, plt_evaluator, slot_transfer_evaluator, None);
    assert_eq!(
        outputs.len(),
        1,
        "{protocol_name} representative evaluation must return one output"
    );
    outputs.remove(0)
}

pub(crate) fn simulate_selected_branch_rebase_error_norm(
    one: ErrorNorm,
    selected_prg: ErrorNorm,
    rebase_decoder: ErrorNorm,
) -> ErrorNorm {
    let zero_branch_sub = one.clone() - &one;
    let target_col_size = selected_prg.matrix_norm.ncol;
    let branch_mask_error = ErrorNorm::new(
        zero_branch_sub.plaintext_norm.clone(),
        zero_branch_sub.matrix_norm.clone() *
            PolyMatrixNorm::gadget_decomposed(zero_branch_sub.clone_ctx(), target_col_size),
    );
    assert_same_matrix_shape(
        &selected_prg.matrix_norm,
        &branch_mask_error.matrix_norm,
        "selected PRG and branch mask error must share a shape",
    );
    assert_same_matrix_shape(
        &selected_prg.matrix_norm,
        &rebase_decoder.matrix_norm,
        "selected PRG and rebase decoder error must share a shape",
    );
    selected_prg + &branch_mask_error + &rebase_decoder
}

pub(crate) fn branch_rebase_decoder_error(error: ErrorNorm, target_col_size: usize) -> ErrorNorm {
    assert!(target_col_size > 0, "target_col_size must be positive");
    ErrorNorm::new(
        error.plaintext_norm,
        PolyMatrixNorm::new(
            error.matrix_norm.clone_ctx(),
            error.matrix_norm.nrow,
            target_col_size,
            error.matrix_norm.poly_norm.sigma,
            error.matrix_norm.zero_rows,
        ),
    )
}

pub(crate) fn ring_gsw_and_randomizer_norm(
    lhs: &PolyMatrixNorm,
    decomposed: &PolyMatrixNorm,
) -> PolyMatrixNorm {
    (lhs * decomposed) + lhs
}

pub(crate) fn ring_gsw_xor_randomizer_norm(
    lhs: &PolyMatrixNorm,
    rhs: &PolyMatrixNorm,
    decomposed: &PolyMatrixNorm,
) -> PolyMatrixNorm {
    let sum = lhs + rhs;
    let product = (lhs * decomposed) + rhs;
    sum + &(&product + &product)
}

pub(crate) fn representative_goldreich_prg_ciphertext_randomizer_norm(
    seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
    decomposed: &PolyMatrixNorm,
) -> PolyMatrixNorm {
    let and_term = ring_gsw_and_randomizer_norm(seed_ciphertext_randomizer_norm, decomposed);
    let left = ring_gsw_xor_randomizer_norm(
        seed_ciphertext_randomizer_norm,
        seed_ciphertext_randomizer_norm,
        decomposed,
    );
    let right =
        ring_gsw_xor_randomizer_norm(seed_ciphertext_randomizer_norm, &and_term, decomposed);
    ring_gsw_xor_randomizer_norm(&left, &right, decomposed)
}

pub(crate) fn ciphertext_decryption_error_from_randomizer(
    randomizer_norm: &PolyMatrixNorm,
    decomposed_randomizer_norm: &PolyMatrixNorm,
    output_ctx: Arc<SimulatorContext>,
    full_active_levels: usize,
    ring_gsw_public_key_error_sigma: f64,
) -> PolyMatrixNorm {
    let sigma = BigDecimal::from_f64(ring_gsw_public_key_error_sigma)
        .expect("finite Ring-GSW error sigma must convert to BigDecimal");
    let (_top, bottom_half_randomizer) = randomizer_norm.split_rows(randomizer_norm.nrow / 2);
    let p_max_matrix = PolyMatrixNorm::new(
        randomizer_norm.clone_ctx(),
        bottom_half_randomizer.ncol,
        1,
        decomposed_randomizer_norm.poly_norm.sigma.clone(),
        None,
    );
    let public_key_error = PolyMatrixNorm::sample_gauss(
        randomizer_norm.clone_ctx(),
        1,
        bottom_half_randomizer.nrow,
        sigma,
    );
    let raw = public_key_error * (bottom_half_randomizer * p_max_matrix);
    PolyMatrixNorm::new(
        output_ctx,
        1,
        1,
        raw.poly_norm.sigma * BigDecimal::from(full_active_levels as u64),
        None,
    )
}

pub(crate) fn prf_refresh_ciphertext_state(
    seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
    decomposed_randomizer_norm: &PolyMatrixNorm,
    output_ctx: Arc<SimulatorContext>,
    full_active_levels: usize,
    ring_gsw_public_key_error_sigma: f64,
) -> (PolyMatrixNorm, PolyMatrixNorm) {
    let selected_ciphertext_randomizer_norm =
        representative_goldreich_prg_ciphertext_randomizer_norm(
            seed_ciphertext_randomizer_norm,
            decomposed_randomizer_norm,
        );
    let selected_ciphertext_decryption_error = ciphertext_decryption_error_from_randomizer(
        &selected_ciphertext_randomizer_norm,
        decomposed_randomizer_norm,
        output_ctx,
        full_active_levels,
        ring_gsw_public_key_error_sigma,
    );
    // Noise refresh resets the BGG encoding error of the seed wires, but the Ring-GSW ciphertext
    // that those refreshed wires encode is still the selected PRF ciphertext. Carry its randomizer
    // norm into the next PRF round so the FHE ciphertext-side decryption noise accumulates across
    // the Goldreich PRF chain.
    (selected_ciphertext_decryption_error, selected_ciphertext_randomizer_norm)
}

pub(crate) fn initial_seed_wire_error(
    one: &ErrorNorm,
    p_moduli: &[u64],
    protocol_name: &str,
) -> ErrorNorm {
    let max_p_modulus = p_moduli.iter().copied().max().unwrap_or_else(|| {
        panic!("{protocol_name} nested-RNS Ring-GSW context must have at least one p modulus")
    });
    // Ring-GSW native ciphertext inputs are first encoded as nested-RNS p-residue wires via
    // `ciphertext_inputs_from_native`. Each BGG lift therefore multiplies by one residue modulo
    // some p_i, not by a full native q coefficient, so max(p_i) - 1 is the tight scalar bound.
    let scalar_bound = BigUint::from(max_p_modulus - 1);
    one.large_scalar_mul(&(), &[scalar_bound])
}

pub(crate) fn full_active_levels(
    params: &DCRTPolyParams,
    ring_gsw_enable_levels: Option<usize>,
    ring_gsw_level_offset: usize,
) -> usize {
    let (_, _, crt_depth) = params.to_crt();
    ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - ring_gsw_level_offset)
}

pub(crate) fn build_cpu_ring_gsw_context(
    params: &DCRTPolyParams,
    circuit: &mut PolyCircuit<DCRTPoly>,
    active_levels: usize,
    config: CpuRingGswContextConfig,
) -> Arc<NestedRnsRingGswContext<DCRTPoly>> {
    let nested_rns_context = Arc::new(NestedRnsPolyContext::setup(
        circuit,
        params,
        config.p_moduli_bits,
        config.max_unreduced_muls,
        config.scale,
        false,
        Some(active_levels),
    ));
    Arc::new(NestedRnsRingGswContext::<DCRTPoly>::from_arith_context(
        circuit,
        params,
        // Error simulation evaluates one representative Ring-GSW ciphertext at a time. The
        // polynomial ring dimension is still carried by `params`; `num_slots = 1` only avoids
        // constructing the production batch of per-coefficient input wires. Callers that need the
        // full coefficient or slot collapse scale it explicitly in the surrounding norm
        // calculation.
        1,
        nested_rns_context,
        Some(active_levels),
        Some(config.level_offset),
    ))
}

pub(crate) fn seed_ciphertext_randomizer_norms(
    params: &DCRTPolyParams,
    config: CpuRingGswContextConfig,
) -> (PolyMatrixNorm, PolyMatrixNorm) {
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_cpu_ring_gsw_context(params, &mut circuit, 1, config);
    (ring_gsw_context.fresh_randomizer_norm(), ring_gsw_context.decomposed_randomizer_norm())
}

pub(crate) fn representative_goldreich_prg_one_output_circuit(
    params: &DCRTPolyParams,
    active_levels: usize,
    seed_bits: usize,
    config: CpuRingGswContextConfig,
) -> PolyCircuit<DCRTPoly> {
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_cpu_ring_gsw_context(params, &mut circuit, active_levels, config);
    let seed_ciphertexts = (0..seed_bits)
        .map(|_| {
            RingGswCiphertext::input(
                ring_gsw_context.clone(),
                Some(BigUint::from(1u64)),
                &mut circuit,
            )
        })
        .collect::<Vec<_>>();
    let graph = GoldreichGraph::from_edges(
        seed_bits,
        vec![GoldreichEdge::new([0, 1, 2], [3, 4])],
        Default::default(),
    );
    let goldreich = GoldreichFhePrg::from_public_graph(&mut circuit, ring_gsw_context, graph);
    let outputs = goldreich.evaluate_uniform(&seed_ciphertexts, &mut circuit);
    circuit.output(outputs.iter().flat_map(|output| output.sub_circuit_wires()));
    circuit
}

pub(crate) fn simulate_final_mask_base_error<PE, ST>(
    params: &DCRTPolyParams,
    active_levels: usize,
    config: CpuRingGswContextConfig,
    final_mask_prg_output: ErrorNorm,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    protocol_name: &str,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let mut circuit = PolyCircuit::new();
    let ring_gsw_context = build_cpu_ring_gsw_context(params, &mut circuit, 1, config);
    let decryption_key_wire = circuit.input(1).at(0).as_single_wire();
    let encrypted_bit =
        RingGswCiphertext::input(ring_gsw_context.clone(), Some(BigUint::from(1u64)), &mut circuit);
    // The exact final-mask plaintext modulus changes public constants, but the representative
    // ErrorNorm growth is the same one-bit decrypt/recompose circuit. The caller applies the PRF
    // mask coefficient bit-width as an external linear scale.
    let representative_plaintext_modulus = BigUint::from(2u64);
    let decrypted = encrypted_bit.decrypt::<DCRTPolyMatrix>(
        decryption_key_wire,
        representative_plaintext_modulus,
        &mut circuit,
    );
    let output =
        circuit.add_gate(decrypted.secret_dependent, decrypted.public_bottom).as_single_wire();
    circuit.output(vec![output]);
    let wire_count = circuit.num_input() - 1;
    let coefficient_scale = BigDecimal::from(params.ring_dimension() as u64);
    let final_mask_prg_output = scale_error_norm(final_mask_prg_output, &coefficient_scale);
    let mut inputs = Vec::with_capacity(circuit.num_input());
    inputs.push(decryption_key);
    inputs.extend(std::iter::repeat_n(final_mask_prg_output, wire_count));
    assert_eq!(
        inputs.len(),
        circuit.num_input(),
        "{protocol_name} final mask error-simulation input count mismatch"
    );
    let mut outputs = circuit.eval(
        &(),
        one,
        inputs,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
        None,
    );
    assert_eq!(outputs.len(), 1, "{protocol_name} final mask circuit must produce one output");
    extrapolate_error_norm_matrix_to_active_levels(outputs.remove(0), active_levels)
}

pub(crate) fn simulate_representative_function_output<PE, ST>(
    params: &DCRTPolyParams,
    active_levels: usize,
    config: CpuRingGswContextConfig,
    function_prg_output: ErrorNorm,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    protocol_name: &str,
) -> (ErrorNorm, ErrorNorm)
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let mut context_circuit = PolyCircuit::new();
    let circuit = build_one_ciphertext_bit_decrypt_circuit::<
        DCRTPoly,
        NestedRnsPoly<DCRTPoly>,
        DCRTPolyMatrix,
    >(
        build_cpu_ring_gsw_context(params, &mut context_circuit, 1, config),
        BigUint::from(2u64),
    );
    let seed_wire_count = circuit.num_input() - 1;
    let mut inputs = Vec::with_capacity(circuit.num_input());
    inputs.push(decryption_key);
    inputs.extend(std::iter::repeat_n(function_prg_output, seed_wire_count));
    let mut outputs = circuit.eval(
        &(),
        one,
        inputs,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator as &dyn SlotTransferEvaluator<ErrorNorm>),
        None,
    );
    assert_eq!(outputs.len(), 2, "{protocol_name} representative function output must be a pair");
    for output in &mut outputs {
        *output = extrapolate_error_norm_matrix_to_active_levels(output.clone(), active_levels);
    }
    (outputs[0].clone(), outputs[1].clone())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_representative_prg_output<PE, ST>(
    params: &DCRTPolyParams,
    active_levels: usize,
    representative_seed_bits: usize,
    config: CpuRingGswContextConfig,
    one: ErrorNorm,
    seed_errors: &[ErrorNorm],
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    protocol_name: &str,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let circuit = representative_goldreich_prg_one_output_circuit(
        params,
        1,
        representative_seed_bits,
        config,
    );
    let wire_count = ring_gsw_wire_count(circuit.num_input(), representative_seed_bits);
    let seed_wire_errors =
        expand_logical_errors(&seed_errors[..representative_seed_bits], wire_count);
    let mut output = eval_first_error_output(
        circuit,
        one,
        seed_wire_errors,
        Some(plt_evaluator),
        Some(slot_transfer_evaluator),
        protocol_name,
    );
    output.matrix_norm = output.matrix_norm * BigDecimal::from(active_levels as u64);
    output
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_noise_refresh_from_material_state<PE, ST>(
    ring_gsw_context: Arc<NestedRnsRingGswContext<DCRTPoly>>,
    seed_bits: usize,
    noise_refresh_v_bits: usize,
    noise_refresh_cbd_n: usize,
    ring_gsw_public_key_error_sigma: f64,
    num_slots: usize,
    one: ErrorNorm,
    selected: ErrorNorm,
    seed_errors: &[ErrorNorm],
    decryption_key: ErrorNorm,
    refresh_decoder_error: ErrorNorm,
    material_wire_error: ErrorNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> NoiseRefreshErrorSimulation
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    simulate_symmetric_noise_refresh_error_growth_for_v_bits_with_material_override(
        ring_gsw_context,
        seed_bits,
        noise_refresh_v_bits,
        noise_refresh_cbd_n,
        ring_gsw_public_key_error_sigma,
        num_slots,
        one,
        selected,
        seed_errors,
        decryption_key,
        refresh_decoder_error,
        material_wire_error,
        plt_evaluator,
        slot_transfer_evaluator,
    )
}

pub(crate) fn prf_refresh_material_inputs(
    one: ErrorNorm,
    representative_branch: ErrorNorm,
    final_decoder_error: ErrorNorm,
    noise_refresh_cbd_n: usize,
) -> (ErrorNorm, ErrorNorm) {
    let selected = simulate_selected_branch_rebase_error_norm(
        one,
        representative_branch.clone(),
        branch_rebase_decoder_error(final_decoder_error, representative_branch.matrix_norm.ncol),
    );
    let material_wire_error = scale_error_norm(
        representative_branch,
        &BigDecimal::from((2 * noise_refresh_cbd_n) as u64),
    );
    (selected, material_wire_error)
}

pub(crate) fn refreshed_seed_from_noise_refresh(
    refresh: &NoiseRefreshErrorSimulation,
    protocol_name: &str,
) -> ErrorNorm {
    let worst_rounded_error = refresh
        .rounded_errors
        .iter()
        .max_by(|lhs, rhs| {
            lhs.poly_norm
                .sigma
                .partial_cmp(&rhs.poly_norm.sigma)
                .unwrap_or_else(|| panic!("{protocol_name} rounded errors must be comparable"))
        })
        .expect("noise refresh must produce rounded errors")
        .clone();
    ErrorNorm::new(worst_rounded_error.poly_norm.clone(), worst_rounded_error)
}

pub(crate) fn add_ciphertext_decryption_residual_to_noise_refresh_pre_rounds(
    refresh: &mut NoiseRefreshErrorSimulation,
    residual: &PolyMatrixNorm,
) {
    for pre_round in &mut refresh.pre_round_outputs {
        pre_round.poly_norm = &pre_round.poly_norm + &residual.poly_norm;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrfRefreshRoundCore {
    pub refresh: NoiseRefreshErrorSimulation,
    pub refreshed_seed: ErrorNorm,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_prf_refresh_round_fixed_core<PE, ST>(
    params: &DCRTPolyParams,
    ring_gsw_context: Arc<NestedRnsRingGswContext<DCRTPoly>>,
    seed_bits: usize,
    noise_refresh_v_bits: usize,
    noise_refresh_cbd_n: usize,
    ring_gsw_public_key_error_sigma: f64,
    one: ErrorNorm,
    selected: ErrorNorm,
    seed_errors: &[ErrorNorm],
    decryption_key: ErrorNorm,
    refresh_decoder_error: ErrorNorm,
    material_wire_error: ErrorNorm,
    selected_ciphertext_decryption_error: &PolyMatrixNorm,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
    protocol_name: &str,
) -> Option<PrfRefreshRoundCore>
where
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
{
    let required_seed_bits = crate::noise_refresh::simulation::minimum_noise_refresh_seed_bits(
        params,
        noise_refresh_v_bits,
        noise_refresh_cbd_n,
    );
    if seed_bits < required_seed_bits {
        return None;
    }
    let mut refresh = simulate_noise_refresh_from_material_state(
        ring_gsw_context,
        seed_bits,
        noise_refresh_v_bits,
        noise_refresh_cbd_n,
        ring_gsw_public_key_error_sigma,
        params.ring_dimension() as usize,
        one,
        selected,
        seed_errors,
        decryption_key,
        refresh_decoder_error,
        material_wire_error,
        plt_evaluator,
        slot_transfer_evaluator,
    );
    add_ciphertext_decryption_residual_to_noise_refresh_pre_rounds(
        &mut refresh,
        selected_ciphertext_decryption_error,
    );
    let refreshed_seed = refreshed_seed_from_noise_refresh(&refresh, protocol_name);
    Some(PrfRefreshRoundCore { refresh, refreshed_seed })
}

pub(crate) fn expand_logical_errors(
    logical_errors: &[ErrorNorm],
    wire_count: usize,
) -> Vec<ErrorNorm> {
    assert!(wire_count > 0, "Ring-GSW wire count must be positive");
    logical_errors.iter().flat_map(|error| std::iter::repeat_n(error.clone(), wire_count)).collect()
}

pub(crate) fn scale_error_norm(input: ErrorNorm, scale: &BigDecimal) -> ErrorNorm {
    ErrorNorm {
        plaintext_norm: input.plaintext_norm * scale,
        matrix_norm: input.matrix_norm * scale,
    }
}

pub(crate) fn assert_same_matrix_shape(lhs: &PolyMatrixNorm, rhs: &PolyMatrixNorm, message: &str) {
    assert!(lhs.ctx() == rhs.ctx(), "{message}: simulator contexts must match");
    assert_eq!(
        (lhs.nrow, lhs.ncol),
        (rhs.nrow, rhs.ncol),
        "{message}: left shape {:?} must match right shape {:?}",
        (lhs.nrow, lhs.ncol),
        (rhs.nrow, rhs.ncol)
    );
}

pub(crate) fn extrapolate_error_norm_matrix_to_active_levels(
    mut input: ErrorNorm,
    active_levels: usize,
) -> ErrorNorm {
    assert!(active_levels > 0, "active-level extrapolation scale must be positive");
    if active_levels > 1 {
        // Active-level extrapolation models the ciphertext-side Ring-GSW contribution. The
        // plaintext norm is a bound on the represented value flowing through the circuit, so it is
        // not scaled by the number of CRT levels.
        input.matrix_norm = input.matrix_norm * BigDecimal::from(active_levels as u64);
    }
    input
}

pub(crate) fn ring_gsw_wire_count(flat_input_count: usize, logical_bit_count: usize) -> usize {
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

    #[test]
    fn final_mask_bits_are_chosen_after_reserving_security_error_margin() {
        let params = DCRTPolyParams::new(2, 1, 10, 5);
        let max_bits = params.modulus_bits();
        let unreserved = final_mask_coeff_bits_for_error_margin(
            &params,
            &BigDecimal::from(0u32),
            None,
            max_bits,
        )
        .expect("zero-error mask width should fit");
        let reserved = final_mask_coeff_bits_for_error_margin(
            &params,
            &BigDecimal::from(1u32),
            Some(4),
            max_bits,
        )
        .expect("security-reserved mask width should fit");

        assert!(reserved <= unreserved);
        assert!(
            final_decode_margin(&params, reserved).expect("reserved margin should exist") >=
                BigDecimal::from(16u32)
        );

        let q: Arc<BigUint> = params.modulus().into();
        let exact_available = BigDecimal::from(128u32);
        let exact_error =
            decode_threshold(q.as_ref(), &DecodeThreshold::boolean()) - exact_available;
        assert_eq!(
            final_mask_coeff_bits_for_error_margin(&params, &exact_error, None, max_bits),
            Some(8),
            "a final-mask range exactly equal to 2^(k - 1) must select k"
        );
        if reserved < max_bits {
            assert!(
                final_decode_margin(&params, reserved + 1)
                    .is_none_or(|margin| margin < BigDecimal::from(16u32))
            );
        }
    }

    #[test]
    fn noise_refresh_bits_are_chosen_after_reserving_security_error_margin() {
        let params = DCRTPolyParams::new(2, 2, 20, 5);
        let max_bits = params.modulus_bits();
        let unreserved =
            noise_refresh_v_bits_for_error_margin(&params, &BigDecimal::from(0u32), None, max_bits)
                .expect("zero-error noise-refresh mask width should fit");
        let reserved = noise_refresh_v_bits_for_error_margin(
            &params,
            &BigDecimal::from(1u32),
            Some(4),
            max_bits,
        )
        .expect("security-reserved noise-refresh mask width should fit");

        assert!(reserved <= unreserved);
        assert!(
            noise_refresh_pre_round_margin(&params, reserved)
                .expect("reserved margin should exist") >=
                BigDecimal::from(16u32)
        );

        let (q_moduli, _crt_bits, _crt_depth) = params.to_crt();
        let q_max = q_moduli.iter().copied().max().expect("CRT moduli must be nonempty");
        let full_q: Arc<BigUint> = params.modulus().into();
        let exact_available = BigDecimal::from(128u32);
        let exact_error =
            decode_threshold(full_q.as_ref(), &DecodeThreshold::new(q_max)) - exact_available;
        assert_eq!(
            noise_refresh_v_bits_for_error_margin(&params, &exact_error, None, max_bits),
            Some(8),
            "a noise-refresh range exactly equal to 2^(k - 1) must select k"
        );
        if reserved < max_bits {
            assert!(
                noise_refresh_pre_round_margin(&params, reserved + 1)
                    .is_none_or(|margin| margin < BigDecimal::from(16u32))
            );
        }
    }
}
