use std::{sync::Arc, time::Instant};

use bigdecimal::BigDecimal;
use num_traits::FromPrimitive;
use tracing::{debug, info};

use crate::{
    circuit::PolyCircuit,
    decoder::simulation::validate_error_bound_security_margin,
    gadgets::{
        fhe::ring_gsw_nested_rns::NestedRnsRingGswContext,
        fhe_prg::goldreich::{goldreich_output_bound_holds, minimum_goldreich_input_size},
    },
    lookup::PltEvaluator,
    matrix::PolyMatrix,
    noise_refresh::{
        NoiseRefreshErrorSimulation,
        simulation::{max_safe_noise_refresh_v_bits, minimum_noise_refresh_seed_bits},
    },
    poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    },
    simulator::{
        SimulatorContext,
        error_norm::{ErrorNorm, compute_preimage_norm},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};

use super::{Aky24IO, Aky24IOFuncType};
use crate::io::utils::simulation::{self as sim_utils, assert_same_matrix_shape, scale_error_norm};

const AKY24_IO_SECRET_SIZE: usize = 1;
const REPRESENTATIVE_GOLDREICH_SEED_BITS: usize = 5;

/// Error-growth summary for the conventional AKY24 FE-to-iO online path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24IOErrorSimulation {
    /// Fresh encoding error that replaces DiamondIO input-injection state error.
    pub initial_fresh_error: ErrorNorm,
    /// Final decoder term derived from fresh `c_b0` error and output preimage norm.
    pub final_decoder_error: PolyMatrixNorm,
    /// Representative noise-refresh simulation per PRF seed-refresh round.
    pub prf_refreshes: Vec<Aky24IOPrfRoundErrorSimulation>,
    /// Final projection residual before decoder cancellation.
    pub projection_residual_error: PolyMatrixNorm,
    /// Final noisy-plaintext error after decoder cancellation.
    pub noisy_plaintext_error: PolyMatrixNorm,
}

/// Error-growth summary for one AKY24 iO PRF seed-refresh round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24IOPrfRoundErrorSimulation {
    pub round_idx: usize,
    pub representative_selected_prg_output: ErrorNorm,
    pub representative_selected_prg_ciphertext_decryption_error: PolyMatrixNorm,
    pub noise_refresh: NoiseRefreshErrorSimulation,
}

/// Largest safe PRF mask bit-width together with the simulation that certified it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24IOPrfMaskOutputCoeffBitsSearchResult {
    pub prf_mask_output_coeff_bits: usize,
    pub noise_refresh_v_bits: usize,
    pub simulation: Aky24IOErrorSimulation,
}

/// Successful AKY24 iO CRT-depth search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aky24IOCrtDepthSearchResult {
    pub crt_depth: usize,
    pub log_ring_dim: usize,
    pub ring_dim: u32,
    pub achieved_secpar_for_gauss: Option<u64>,
    pub achieved_secpar_for_cbd: Option<u64>,
    pub prf_mask_output_coeff_bits: usize,
    pub noise_refresh_v_bits: usize,
    pub seed_bits: usize,
    pub total_noisy_plaintext_error: PolyMatrixNorm,
    pub initial_fresh_error: PolyMatrixNorm,
}

#[derive(Debug, Clone)]
struct Aky24IOMaskIndependentErrorPrefix {
    cpu_params: DCRTPolyParams,
    ctx: Arc<SimulatorContext>,
    initial_fresh_error: ErrorNorm,
    final_decoder_error: ErrorNorm,
    prf_refreshes: Vec<Aky24IOPrfRoundErrorSimulation>,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    final_seed_errors: Vec<ErrorNorm>,
    function_secret_error: ErrorNorm,
    public_bottom_decryption_error: PolyMatrixNorm,
}

#[derive(Debug)]
struct Aky24IOMaskIndependentErrorBase {
    cpu_params: DCRTPolyParams,
    ctx: Arc<SimulatorContext>,
    initial_fresh_error: ErrorNorm,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    refresh_decoder_error: ErrorNorm,
    final_decoder_error: ErrorNorm,
    initial_seed_errors: Vec<ErrorNorm>,
    initial_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
    seed_ciphertext_decomposed_randomizer_norm: PolyMatrixNorm,
    noise_refresh_ring_gsw_context: Arc<NestedRnsRingGswContext<DCRTPoly>>,
}

impl Clone for Aky24IOMaskIndependentErrorBase {
    fn clone(&self) -> Self {
        Self {
            cpu_params: self.cpu_params.clone(),
            ctx: self.ctx.clone(),
            initial_fresh_error: self.initial_fresh_error.clone(),
            one: self.one.clone(),
            decryption_key: self.decryption_key.clone(),
            refresh_decoder_error: self.refresh_decoder_error.clone(),
            final_decoder_error: self.final_decoder_error.clone(),
            initial_seed_errors: self.initial_seed_errors.clone(),
            initial_seed_ciphertext_randomizer_norm: self
                .initial_seed_ciphertext_randomizer_norm
                .clone(),
            seed_ciphertext_decomposed_randomizer_norm: self
                .seed_ciphertext_decomposed_randomizer_norm
                .clone(),
            noise_refresh_ring_gsw_context: self.noise_refresh_ring_gsw_context.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct Aky24IOPrfRefreshRoundEvaluation {
    round: Aky24IOPrfRoundErrorSimulation,
    refreshed_seed: ErrorNorm,
    refreshed_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
    material_wire_error: ErrorNorm,
}

#[derive(Debug, Clone)]
struct Aky24IOPrfRefreshMaterialState {
    selected: ErrorNorm,
    material_wire_error: ErrorNorm,
    representative_selected_prg_ciphertext_decryption_error: PolyMatrixNorm,
    refreshed_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
}

/// Counts the uniform Goldreich output bits needed for the final PRF mask.
pub fn aky24_io_final_mask_uniform_output_bits(
    output_size: usize,
    ring_dim: usize,
    prf_mask_output_coeff_bits: usize,
) -> usize {
    assert!(output_size > 0, "AKY24IO output_size must be positive");
    assert!(ring_dim > 0, "AKY24IO ring_dim must be positive");
    assert!(
        prf_mask_output_coeff_bits > 0,
        "AKY24IO PRF mask output coefficient bits must be positive"
    );
    output_size
        .checked_mul(ring_dim)
        .and_then(|count| count.checked_mul(prf_mask_output_coeff_bits))
        .expect("AKY24IO final mask uniform output bit count overflow")
}

/// Counts the final Goldreich PRG stream: mask bits first, then function bits.
pub fn aky24_io_final_prg_uniform_output_bits(
    output_size: usize,
    ring_dim: usize,
    prf_mask_output_coeff_bits: usize,
    function_output_bits: usize,
) -> usize {
    aky24_io_final_mask_uniform_output_bits(output_size, ring_dim, prf_mask_output_coeff_bits)
        .checked_add(function_output_bits)
        .expect("AKY24IO final PRG uniform output bit count overflow")
}

/// Returns the minimum seed length satisfying every Goldreich PRG output bound.
pub fn minimum_aky24_io_prf_seed_bits(
    params: &DCRTPolyParams,
    output_size: usize,
    function_output_bits: usize,
    prf_batch_bits: usize,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
    cbd_n: usize,
) -> usize {
    let ring_dim = params.ring_dimension() as usize;
    let seed_refresh_seed_bits =
        minimum_seed_refresh_prf_seed_bits(aky24_io_prf_branch_count(prf_batch_bits));
    let final_mask_seed_bits =
        minimum_goldreich_input_size(aky24_io_final_prg_uniform_output_bits(
            output_size,
            ring_dim,
            prf_mask_output_coeff_bits,
            function_output_bits,
        ));
    let noise_refresh_seed_bits =
        minimum_noise_refresh_seed_bits(params, noise_refresh_v_bits, cbd_n);
    seed_refresh_seed_bits.max(final_mask_seed_bits).max(noise_refresh_seed_bits)
}

fn aky24_io_prf_branch_count(prf_batch_bits: usize) -> usize {
    assert!(prf_batch_bits > 0, "AKY24IO prf_batch_bits must be positive");
    assert!(
        prf_batch_bits < usize::BITS as usize,
        "AKY24IO prf_batch_bits must fit in a usize branch count"
    );
    1usize.checked_shl(prf_batch_bits as u32).expect("AKY24IO PRF branch count overflow")
}

/// Returns the largest noise-refresh `v_bits` allowed before pre-rounding error is added.
pub fn aky24_io_max_noise_refresh_v_bits_without_pre_rounding_error(
    params: &DCRTPolyParams,
) -> Option<usize> {
    let zero_pre_rounding =
        PolyMatrixNorm::new(simulator_context(params), 1, 1, BigDecimal::from(0u32), None);
    max_safe_noise_refresh_v_bits(params, &zero_pre_rounding)
}

/// Finds the smallest CRT depth whose correctness and security checks pass.
pub fn aky24_io_find_crt_depth<M, PKPE, PKST, ENCPE, ENCST, PE, ST, BuildCandidate>(
    min_crt_depth: usize,
    max_crt_depth: usize,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
    max_prf_mask_output_coeff_bits: usize,
    security_bit: usize,
    error_sigma: f64,
    func_type: Aky24IOFuncType,
    mut build_candidate: BuildCandidate,
    plt_evaluator: &PE,
    slot_transfer_evaluator: &ST,
) -> Option<Aky24IOCrtDepthSearchResult>
where
    M: PolyMatrix,
    PE: PltEvaluator<ErrorNorm>,
    ST: SlotTransferEvaluator<ErrorNorm>,
    BuildCandidate: FnMut(u32, usize, usize, Option<usize>) -> Aky24IO<M, PKPE, PKST, ENCPE, ENCST>,
{
    info!(
        min_crt_depth,
        max_crt_depth,
        min_log_ring_dim,
        max_log_ring_dim,
        max_prf_mask_output_coeff_bits,
        security_bit,
        "starting AKY24IO CRT-depth search"
    );
    let force_lattice_check = std::env::var_os("MXX_IO_FORCE_LATTICE_CHECK").is_some();
    let explicit_log_ring_dim = min_log_ring_dim == max_log_ring_dim && !force_lattice_check;
    if !explicit_log_ring_dim {
        sim_utils::assert_lattice_estimator_available("AKY24IO");
    }
    assert!(min_crt_depth > 0, "minimum CRT depth must be positive");
    assert!(min_crt_depth <= max_crt_depth, "CRT-depth search range must be non-empty");
    let mut low = min_crt_depth;
    let mut high = max_crt_depth;
    let mut found = Vec::new();
    let mut lattice_cache = sim_utils::SecureRingDimLatticeCache::default();
    while low <= high {
        let crt_depth = low + (high - low) / 2;
        info!(crt_depth, low, high, "evaluating AKY24IO CRT-depth candidate");
        let min_ring_dim = 1u32
            .checked_shl(min_log_ring_dim.try_into().expect("min_log_ring_dim must fit in u32"))
            .expect("minimum ring_dim shift overflow");
        let probe_candidate =
            build_candidate(min_ring_dim, crt_depth, max_prf_mask_output_coeff_bits, None);
        let noise_refresh_cbd_n = probe_candidate.noise_refresh_cbd_n;
        let Some(ring_dim_search) = sim_utils::select_min_secure_ring_dim(
            "AKY24IO",
            crt_depth,
            min_log_ring_dim,
            max_log_ring_dim,
            security_bit,
            error_sigma,
            noise_refresh_cbd_n,
            explicit_log_ring_dim,
            &mut lattice_cache,
            |ring_dim| {
                let candidate =
                    build_candidate(ring_dim, crt_depth, max_prf_mask_output_coeff_bits, None);
                candidate.params.clone()
            },
        ) else {
            return None;
        };
        let ring_dim = ring_dim_search.ring_dim;
        let provisional_candidate =
            build_candidate(ring_dim, crt_depth, max_prf_mask_output_coeff_bits, None);
        let cpu_params = cpu_params_from_poly_params(&provisional_candidate.params);
        let provisional_base = provisional_candidate.simulate_mask_independent_error_base(
            func_type,
            error_sigma,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let Some(global_noise_refresh_v_bits) = provisional_candidate
            .select_global_noise_refresh_v_bits(
                &provisional_base,
                security_bit,
                plt_evaluator,
                slot_transfer_evaluator,
            )
        else {
            low = crt_depth + 1;
            continue;
        };

        let mask_search_candidate = build_candidate(
            ring_dim,
            crt_depth,
            max_prf_mask_output_coeff_bits,
            Some(global_noise_refresh_v_bits),
        );
        let Some(mask_search_prefix) = mask_search_candidate
            .build_fixed_noise_refresh_prefix_from_base(
                provisional_base.clone(),
                func_type,
                global_noise_refresh_v_bits,
                plt_evaluator,
                slot_transfer_evaluator,
            )
        else {
            low = crt_depth + 1;
            continue;
        };
        let Some(mask_search) = mask_search_candidate
            .search_final_prf_mask_output_coeff_bits_from_prefix(
                func_type,
                &mask_search_prefix,
                max_prf_mask_output_coeff_bits,
                global_noise_refresh_v_bits,
                Some(security_bit),
                plt_evaluator,
                slot_transfer_evaluator,
            )
        else {
            low = crt_depth + 1;
            continue;
        };
        let mask_bits = mask_search.prf_mask_output_coeff_bits;
        let final_candidate =
            build_candidate(ring_dim, crt_depth, mask_bits, Some(global_noise_refresh_v_bits));
        let expected_seed_bits = minimum_aky24_io_prf_seed_bits(
            &cpu_params,
            func_type.output_bits(),
            func_type.output_bits(),
            final_candidate.prf_batch_bits,
            mask_bits,
            global_noise_refresh_v_bits,
            final_candidate.noise_refresh_cbd_n,
        );
        if final_candidate.seed_bits != expected_seed_bits {
            low = crt_depth + 1;
            continue;
        }
        let final_simulation = final_candidate
            .build_fixed_noise_refresh_prefix_from_base(
                provisional_base.clone(),
                func_type,
                global_noise_refresh_v_bits,
                plt_evaluator,
                slot_transfer_evaluator,
            )
            .map(|prefix| {
                final_candidate.finish_error_growth_from_mask_independent_prefix(
                    &prefix,
                    func_type,
                    mask_bits,
                    None,
                    None,
                    plt_evaluator,
                    slot_transfer_evaluator,
                )
            });
        let Some(final_simulation) = final_simulation else {
            low = crt_depth + 1;
            continue;
        };
        if !fixed_noise_refresh_v_bits_match(&final_simulation, global_noise_refresh_v_bits) ||
            !aky24_io_security_margins_hold(
                &cpu_params,
                &final_simulation,
                mask_bits,
                security_bit,
            )
        {
            low = crt_depth + 1;
            continue;
        }

        found.push(Aky24IOCrtDepthSearchResult {
            crt_depth,
            log_ring_dim: ring_dim_search.log_ring_dim,
            ring_dim,
            achieved_secpar_for_gauss: ring_dim_search.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd: ring_dim_search.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits: mask_bits,
            noise_refresh_v_bits: global_noise_refresh_v_bits,
            seed_bits: final_candidate.seed_bits,
            total_noisy_plaintext_error: final_simulation.noisy_plaintext_error.clone(),
            initial_fresh_error: final_simulation.initial_fresh_error.matrix_norm.clone(),
        });
        high = crt_depth - 1;
    }
    let result = found.into_iter().min_by_key(|candidate| candidate.crt_depth);
    info!(found = result.is_some(), "finished AKY24IO CRT-depth search");
    result
}

impl<M, PKPE, PKST, ENCPE, ENCST> Aky24IO<M, PKPE, PKST, ENCPE, ENCST>
where
    M: PolyMatrix,
{
    /// Simulate AKY24 iO error growth for the selected function family.
    pub fn simulate_error_growth<PE, ST>(
        &self,
        func_type: Aky24IOFuncType,
        error_sigma: f64,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOErrorSimulation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        self.simulate_error_growth_with_prf_mask_output_coeff_bits(
            func_type,
            error_sigma,
            self.prf_mask_output_coeff_bits,
            self.noise_refresh_v_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )
    }

    /// Search for the largest PRF mask coefficient bit-width satisfying final decode margin.
    pub fn max_safe_prf_mask_output_coeff_bits<PE, ST>(
        &self,
        func_type: Aky24IOFuncType,
        error_sigma: f64,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
        max_bits: usize,
    ) -> Option<Aky24IOPrfMaskOutputCoeffBitsSearchResult>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let prefix = self.simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
            func_type,
            error_sigma,
            self.noise_refresh_v_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        self.search_final_prf_mask_output_coeff_bits_from_prefix(
            func_type,
            &prefix,
            max_bits,
            self.noise_refresh_v_bits,
            None,
            plt_evaluator,
            slot_transfer_evaluator,
        )
    }

    fn simulate_error_growth_with_prf_mask_output_coeff_bits<PE, ST>(
        &self,
        func_type: Aky24IOFuncType,
        error_sigma: f64,
        prf_mask_output_coeff_bits: usize,
        noise_refresh_v_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOErrorSimulation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        assert!(error_sigma.is_finite(), "AKY24IO error_sigma must be finite");
        assert!(error_sigma >= 0.0, "AKY24IO error_sigma must be nonnegative");
        assert!(self.seed_bits > 0, "AKY24IO error simulation requires seed_bits > 0");
        assert!(
            prf_mask_output_coeff_bits > 0,
            "AKY24IO error simulation requires prf_mask_output_coeff_bits > 0"
        );
        let cpu_params = cpu_params_from_poly_params(&self.params);
        let final_mask_prg_output_bits = aky24_io_final_prg_uniform_output_bits(
            func_type.output_bits(),
            cpu_params.ring_dimension() as usize,
            prf_mask_output_coeff_bits,
            func_type.output_bits(),
        );
        if !goldreich_output_bound_holds(self.seed_bits, final_mask_prg_output_bits) {
            return None;
        }

        let prefix = self.simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
            func_type,
            error_sigma,
            noise_refresh_v_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        Some(self.finish_error_growth_from_mask_independent_prefix(
            &prefix,
            func_type,
            prf_mask_output_coeff_bits,
            None,
            None,
            plt_evaluator,
            slot_transfer_evaluator,
        ))
    }

    fn simulate_mask_independent_error_base<PE, ST>(
        &self,
        _func_type: Aky24IOFuncType,
        error_sigma: f64,
        _plt_evaluator: &PE,
        _slot_transfer_evaluator: &ST,
    ) -> Aky24IOMaskIndependentErrorBase
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        assert!(self.seed_bits >= REPRESENTATIVE_GOLDREICH_SEED_BITS);
        let cpu_params = cpu_params_from_poly_params(&self.params);
        let prefix_required_seed_bits = minimum_seed_refresh_prf_seed_bits(self.prf_branch_count())
            .max(minimum_noise_refresh_seed_bits(
                &cpu_params,
                self.noise_refresh_v_bits,
                self.noise_refresh_cbd_n,
            ));
        assert!(
            self.seed_bits >= prefix_required_seed_bits,
            "AKY24IO seed_bits {} is below Goldreich PRG safety minimum {} for seed refresh and noise refresh",
            self.seed_bits,
            prefix_required_seed_bits
        );
        let (_, _, crt_depth) = cpu_params.to_crt();
        let full_active_levels = self.full_active_levels(&cpu_params);
        assert!(
            full_active_levels > 0 && self.ring_gsw_level_offset + full_active_levels <= crt_depth,
            "AKY24IO Ring-GSW active levels must fit in the CRT modulus window"
        );
        let ctx = simulator_context(&cpu_params);
        let sigma = BigDecimal::from_f64(error_sigma).expect("finite error_sigma must convert");
        let fresh_matrix_error =
            PolyMatrixNorm::sample_gauss(ctx.clone(), 1, ctx.m_g, sigma.clone());
        let initial_fresh_error =
            ErrorNorm::new(PolyNorm::one(ctx.clone()), fresh_matrix_error.clone());
        let one = initial_fresh_error.clone();
        let decryption_key = initial_fresh_error.clone();
        let refresh_decoder_error = initial_fresh_error.clone();
        let c_b0_error = PolyMatrixNorm::sample_gauss(ctx.clone(), 1, ctx.m_b, sigma);
        let output_preimage_norm = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            1,
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None, None),
            None,
        );
        let final_decoder_projection_error = c_b0_error * &output_preimage_norm;
        let final_decoder_error =
            ErrorNorm::new(PolyNorm::one(ctx.clone()), final_decoder_projection_error);
        let seed_wire_error = sim_utils::initial_seed_wire_error(
            &one,
            self.ring_gsw_context.p_moduli.as_slice(),
            "AKY24IO",
        );
        let initial_seed_errors = vec![seed_wire_error; REPRESENTATIVE_GOLDREICH_SEED_BITS];
        let (initial_seed_ciphertext_randomizer_norm, seed_ciphertext_decomposed_randomizer_norm) =
            sim_utils::seed_ciphertext_randomizer_norms(&cpu_params, self.cpu_ring_gsw_config());
        let mut noise_refresh_context_circuit = PolyCircuit::<DCRTPoly>::new();
        let noise_refresh_ring_gsw_context = sim_utils::build_cpu_ring_gsw_context(
            &cpu_params,
            &mut noise_refresh_context_circuit,
            self.full_active_levels(&cpu_params),
            self.cpu_ring_gsw_config(),
        );
        info!(
            initial_fresh_error_bits =
                bigdecimal_bits_ceil(&initial_fresh_error.matrix_norm.poly_norm.norm),
            final_decoder_error_bits =
                bigdecimal_bits_ceil(&final_decoder_error.matrix_norm.poly_norm.norm),
            "AKY24IO fresh initial error simulation base finished"
        );

        Aky24IOMaskIndependentErrorBase {
            cpu_params,
            ctx,
            initial_fresh_error,
            one,
            decryption_key,
            refresh_decoder_error,
            final_decoder_error,
            initial_seed_errors,
            initial_seed_ciphertext_randomizer_norm,
            seed_ciphertext_decomposed_randomizer_norm,
            noise_refresh_ring_gsw_context,
        }
    }

    fn select_global_noise_refresh_v_bits<PE, ST>(
        &self,
        base: &Aky24IOMaskIndependentErrorBase,
        security_bit: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<usize>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let prf_round_count = self.prf_round_count();
        if prf_round_count == 0 {
            return Some(self.noise_refresh_v_bits);
        }
        let max_candidate =
            aky24_io_max_noise_refresh_v_bits_without_pre_rounding_error(&base.cpu_params)?;
        let first = self.search_prf_refresh_round_fixed_v_bits(
            base,
            0,
            max_candidate,
            security_bit,
            &base.initial_seed_errors,
            &base.initial_seed_ciphertext_randomizer_norm,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        if prf_round_count == 1 {
            return Some(first.round.noise_refresh.v_bits);
        }
        let steady_seed_errors = vec![first.refreshed_seed; REPRESENTATIVE_GOLDREICH_SEED_BITS];
        let steady = self.search_prf_refresh_round_fixed_v_bits(
            base,
            1,
            first.round.noise_refresh.v_bits,
            security_bit,
            &steady_seed_errors,
            &first.refreshed_seed_ciphertext_randomizer_norm,
            plt_evaluator,
            slot_transfer_evaluator,
        )?;
        Some(first.round.noise_refresh.v_bits.min(steady.round.noise_refresh.v_bits))
    }

    fn simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits<PE, ST>(
        &self,
        func_type: Aky24IOFuncType,
        error_sigma: f64,
        noise_refresh_v_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOMaskIndependentErrorPrefix>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let base = self.simulate_mask_independent_error_base(
            func_type,
            error_sigma,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        self.build_fixed_noise_refresh_prefix_from_base(
            base,
            func_type,
            noise_refresh_v_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )
    }

    fn build_fixed_noise_refresh_prefix_from_base<PE, ST>(
        &self,
        base: Aky24IOMaskIndependentErrorBase,
        func_type: Aky24IOFuncType,
        noise_refresh_v_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOMaskIndependentErrorPrefix>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let prf_round_count = self.prf_round_count();
        let mut prf_refreshes = Vec::with_capacity(prf_round_count);
        let mut final_seed_errors = base.initial_seed_errors.clone();
        let mut final_seed_ciphertext_randomizer_norm =
            base.initial_seed_ciphertext_randomizer_norm.clone();
        if prf_round_count > 0 {
            let first = self.simulate_prf_refresh_round_fixed(
                &base,
                0,
                noise_refresh_v_bits,
                &base.initial_seed_errors,
                &base.initial_seed_ciphertext_randomizer_norm,
                plt_evaluator,
                slot_transfer_evaluator,
            )?;
            final_seed_errors =
                vec![first.refreshed_seed.clone(); REPRESENTATIVE_GOLDREICH_SEED_BITS];
            final_seed_ciphertext_randomizer_norm =
                first.refreshed_seed_ciphertext_randomizer_norm.clone();
            prf_refreshes.push(first.round);

            if prf_round_count > 1 {
                let steady = self.simulate_prf_refresh_round_fixed(
                    &base,
                    1,
                    noise_refresh_v_bits,
                    &final_seed_errors,
                    &final_seed_ciphertext_randomizer_norm,
                    plt_evaluator,
                    slot_transfer_evaluator,
                )?;
                final_seed_errors =
                    vec![steady.refreshed_seed.clone(); REPRESENTATIVE_GOLDREICH_SEED_BITS];
                final_seed_ciphertext_randomizer_norm =
                    steady.refreshed_seed_ciphertext_randomizer_norm.clone();
                prf_refreshes.push(steady.round.clone());
                for round_idx in 2..prf_round_count {
                    let (
                        representative_selected_prg_ciphertext_decryption_error,
                        refreshed_seed_ciphertext_randomizer_norm,
                    ) = sim_utils::prf_refresh_ciphertext_state(
                        &final_seed_ciphertext_randomizer_norm,
                        &base.seed_ciphertext_decomposed_randomizer_norm,
                        base.ctx.clone(),
                        self.full_active_levels(&base.cpu_params),
                        self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
                    );
                    let core = sim_utils::simulate_prf_refresh_round_fixed_core(
                        &base.cpu_params,
                        base.noise_refresh_ring_gsw_context.clone(),
                        self.seed_bits,
                        noise_refresh_v_bits,
                        self.noise_refresh_cbd_n,
                        self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
                        base.one.clone(),
                        steady.round.representative_selected_prg_output.clone(),
                        &final_seed_errors,
                        base.decryption_key.clone(),
                        base.refresh_decoder_error.clone(),
                        steady.material_wire_error.clone(),
                        &representative_selected_prg_ciphertext_decryption_error,
                        plt_evaluator,
                        slot_transfer_evaluator,
                        "AKY24IO",
                    )
                    .expect("steady-state AKY24IO PRF refresh seed bound was prevalidated");
                    let evaluation = Aky24IOPrfRefreshRoundEvaluation {
                        round: Aky24IOPrfRoundErrorSimulation {
                            round_idx,
                            representative_selected_prg_output: steady
                                .round
                                .representative_selected_prg_output
                                .clone(),
                            representative_selected_prg_ciphertext_decryption_error,
                            noise_refresh: core.refresh,
                        },
                        refreshed_seed: core.refreshed_seed,
                        refreshed_seed_ciphertext_randomizer_norm:
                            refreshed_seed_ciphertext_randomizer_norm.clone(),
                        material_wire_error: steady.material_wire_error.clone(),
                    };
                    final_seed_errors =
                        vec![evaluation.refreshed_seed.clone(); REPRESENTATIVE_GOLDREICH_SEED_BITS];
                    final_seed_ciphertext_randomizer_norm =
                        refreshed_seed_ciphertext_randomizer_norm;
                    prf_refreshes.push(evaluation.round);
                }
            }
        }
        let function_prg_output = self.simulate_representative_prg_output(
            &base.cpu_params,
            self.prf_final_round_idx(),
            func_type.output_bits(),
            0,
            base.one.clone(),
            final_seed_errors.clone(),
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let (function_secret_error, _function_public_bottom_error) =
            sim_utils::simulate_representative_function_output(
                &base.cpu_params,
                self.full_active_levels(&base.cpu_params),
                self.cpu_ring_gsw_config(),
                function_prg_output,
                base.one.clone(),
                base.decryption_key.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
                "AKY24IO",
            );
        let (public_bottom_decryption_error, _public_bottom_randomizer_norm) =
            sim_utils::prf_refresh_ciphertext_state(
                &final_seed_ciphertext_randomizer_norm,
                &base.seed_ciphertext_decomposed_randomizer_norm,
                base.ctx.clone(),
                self.full_active_levels(&base.cpu_params),
                self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
            );
        Some(Aky24IOMaskIndependentErrorPrefix {
            cpu_params: base.cpu_params,
            ctx: base.ctx,
            initial_fresh_error: base.initial_fresh_error,
            final_decoder_error: base.final_decoder_error,
            prf_refreshes,
            one: base.one,
            decryption_key: base.decryption_key,
            final_seed_errors,
            function_secret_error,
            public_bottom_decryption_error,
        })
    }

    fn search_prf_refresh_round_fixed_v_bits<PE, ST>(
        &self,
        base: &Aky24IOMaskIndependentErrorBase,
        round_idx: usize,
        max_candidate: usize,
        security_bit: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOPrfRefreshRoundEvaluation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let mut low = 1usize;
        let mut high = max_candidate;
        let mut best = None;
        while low <= high {
            let candidate = low + (high - low) / 2;
            let Some(evaluation) = self.simulate_prf_refresh_round_fixed(
                base,
                round_idx,
                candidate,
                seed_errors,
                seed_ciphertext_randomizer_norm,
                plt_evaluator,
                slot_transfer_evaluator,
            ) else {
                if candidate == 1 {
                    break;
                }
                high = candidate - 1;
                continue;
            };
            let valid = aky24_io_noise_refresh_security_margin_holds(
                &base.cpu_params,
                &evaluation.round.noise_refresh,
                security_bit,
            );
            if valid {
                best = Some(evaluation);
                low = candidate + 1;
            } else if candidate == 1 {
                break;
            } else {
                high = candidate - 1;
            }
        }
        best
    }

    fn simulate_prf_refresh_round_fixed<PE, ST>(
        &self,
        base: &Aky24IOMaskIndependentErrorBase,
        round_idx: usize,
        noise_refresh_v_bits: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOPrfRefreshRoundEvaluation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let material_state = self.prf_refresh_material_state(
            base,
            round_idx,
            seed_errors,
            seed_ciphertext_randomizer_norm,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let core = sim_utils::simulate_prf_refresh_round_fixed_core(
            &base.cpu_params,
            base.noise_refresh_ring_gsw_context.clone(),
            self.seed_bits,
            noise_refresh_v_bits,
            self.noise_refresh_cbd_n,
            self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
            base.one.clone(),
            material_state.selected.clone(),
            seed_errors,
            base.decryption_key.clone(),
            base.refresh_decoder_error.clone(),
            material_state.material_wire_error.clone(),
            &material_state.representative_selected_prg_ciphertext_decryption_error,
            plt_evaluator,
            slot_transfer_evaluator,
            "AKY24IO",
        )?;
        Some(Aky24IOPrfRefreshRoundEvaluation {
            round: Aky24IOPrfRoundErrorSimulation {
                round_idx,
                representative_selected_prg_output: material_state.selected,
                representative_selected_prg_ciphertext_decryption_error: material_state
                    .representative_selected_prg_ciphertext_decryption_error,
                noise_refresh: core.refresh,
            },
            refreshed_seed: core.refreshed_seed,
            refreshed_seed_ciphertext_randomizer_norm: material_state
                .refreshed_seed_ciphertext_randomizer_norm,
            material_wire_error: material_state.material_wire_error,
        })
    }

    fn prf_refresh_material_state<PE, ST>(
        &self,
        base: &Aky24IOMaskIndependentErrorBase,
        round_idx: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Aky24IOPrfRefreshMaterialState
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let (
            representative_selected_prg_ciphertext_decryption_error,
            refreshed_seed_ciphertext_randomizer_norm,
        ) = sim_utils::prf_refresh_ciphertext_state(
            seed_ciphertext_randomizer_norm,
            &base.seed_ciphertext_decomposed_randomizer_norm,
            base.ctx.clone(),
            self.full_active_levels(&base.cpu_params),
            self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
        );
        let (selected, material_wire_error) = self.simulate_prf_refresh_material_inputs(
            base,
            round_idx,
            seed_errors,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        Aky24IOPrfRefreshMaterialState {
            selected,
            material_wire_error,
            representative_selected_prg_ciphertext_decryption_error,
            refreshed_seed_ciphertext_randomizer_norm,
        }
    }

    fn simulate_prf_refresh_material_inputs<PE, ST>(
        &self,
        base: &Aky24IOMaskIndependentErrorBase,
        round_idx: usize,
        seed_errors: &[ErrorNorm],
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> (ErrorNorm, ErrorNorm)
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let representative_branch = self.simulate_representative_prg_output(
            &base.cpu_params,
            round_idx,
            self.prf_branch_count()
                .checked_mul(self.seed_bits)
                .expect("AKY24IO seed-refresh PRF output size overflow"),
            0,
            base.one.clone(),
            seed_errors.to_vec(),
            plt_evaluator,
            slot_transfer_evaluator,
        );
        sim_utils::prf_refresh_material_inputs(
            base.one.clone(),
            representative_branch,
            base.final_decoder_error.clone(),
            self.noise_refresh_cbd_n,
        )
    }

    fn search_final_prf_mask_output_coeff_bits_from_prefix<PE, ST>(
        &self,
        func_type: Aky24IOFuncType,
        prefix: &Aky24IOMaskIndependentErrorPrefix,
        max_bits: usize,
        noise_refresh_v_bits: usize,
        security_bit: Option<usize>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<Aky24IOPrfMaskOutputCoeffBitsSearchResult>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let max_candidate =
            max_bits.min(prefix.cpu_params.modulus_bits()).min(max_final_mask_coeff_bits_for_seed(
                &prefix.cpu_params,
                func_type.output_bits(),
                func_type.output_bits(),
                self.seed_bits,
                max_bits,
            ));
        if max_candidate == 0 {
            return None;
        }
        let cached_final_mask_prg_output = self.simulate_representative_prg_output(
            &prefix.cpu_params,
            self.prf_final_round_idx(),
            func_type
                .output_bits()
                .checked_mul(prefix.cpu_params.ring_dimension() as usize)
                .and_then(|count| count.checked_mul(max_candidate))
                .and_then(|count| count.checked_add(func_type.output_bits()))
                .expect("AKY24IO final mask PRG conceptual output count overflow"),
            0,
            prefix.one.clone(),
            prefix.final_seed_errors.clone(),
            plt_evaluator,
            slot_transfer_evaluator,
        );
        let cached_final_mask_base_error = sim_utils::simulate_final_mask_base_error(
            &prefix.cpu_params,
            self.full_active_levels(&prefix.cpu_params),
            self.cpu_ring_gsw_config(),
            cached_final_mask_prg_output.clone(),
            prefix.one.clone(),
            prefix.decryption_key.clone(),
            plt_evaluator,
            slot_transfer_evaluator,
            "AKY24IO",
        );
        let mut low = 1usize;
        let mut high = max_candidate;
        let mut best = None;
        while low <= high {
            let candidate = low + (high - low) / 2;
            let simulation = self.finish_error_growth_from_mask_independent_prefix(
                prefix,
                func_type,
                candidate,
                Some(cached_final_mask_prg_output.clone()),
                Some(cached_final_mask_base_error.clone()),
                plt_evaluator,
                slot_transfer_evaluator,
            );
            let Some(final_margin) = aky24_io_final_decode_margin(&prefix.cpu_params, candidate)
            else {
                if candidate == 1 {
                    break;
                }
                high = candidate - 1;
                continue;
            };
            let error = &simulation.noisy_plaintext_error.poly_norm.norm;
            let valid = if let Some(security_bit) = security_bit {
                validate_error_bound_security_margin(&final_margin, error, security_bit)
            } else {
                final_margin > *error
            };
            if valid {
                best = Some(Aky24IOPrfMaskOutputCoeffBitsSearchResult {
                    prf_mask_output_coeff_bits: candidate,
                    noise_refresh_v_bits,
                    simulation,
                });
                low = candidate + 1;
            } else if candidate == 1 {
                break;
            } else {
                high = candidate - 1;
            }
        }
        best
    }

    fn finish_error_growth_from_mask_independent_prefix<PE, ST>(
        &self,
        prefix: &Aky24IOMaskIndependentErrorPrefix,
        func_type: Aky24IOFuncType,
        prf_mask_output_coeff_bits: usize,
        cached_final_mask_prg_output: Option<ErrorNorm>,
        cached_final_mask_base_error: Option<ErrorNorm>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Aky24IOErrorSimulation
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let final_mask_prg_output = cached_final_mask_prg_output.unwrap_or_else(|| {
            self.simulate_representative_prg_output(
                &prefix.cpu_params,
                self.prf_final_round_idx(),
                func_type
                    .output_bits()
                    .checked_mul(prefix.cpu_params.ring_dimension() as usize)
                    .and_then(|count| count.checked_mul(prf_mask_output_coeff_bits))
                    .and_then(|count| count.checked_add(func_type.output_bits()))
                    .expect("AKY24IO final mask PRG conceptual output count overflow"),
                0,
                prefix.one.clone(),
                prefix.final_seed_errors.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
            )
        });
        let final_mask_base_error = cached_final_mask_base_error.unwrap_or_else(|| {
            sim_utils::simulate_final_mask_base_error(
                &prefix.cpu_params,
                self.full_active_levels(&prefix.cpu_params),
                self.cpu_ring_gsw_config(),
                final_mask_prg_output,
                prefix.one.clone(),
                prefix.decryption_key.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
                "AKY24IO",
            )
        });
        let final_mask_error = scale_error_norm(
            final_mask_base_error,
            &BigDecimal::from(prf_mask_output_coeff_bits as u64),
        );
        assert_same_matrix_shape(
            &final_mask_error.matrix_norm,
            &prefix.function_secret_error.matrix_norm,
            "final mask and function secret-dependent errors must share pre-projection shape",
        );
        let evaluated_secret_dependent = prefix.function_secret_error.clone() + &final_mask_error;
        let projected_evaluated_error = evaluated_secret_dependent.matrix_norm.clone() *
            PolyMatrixNorm::gadget_decomposed(prefix.ctx.clone(), 1);
        assert_same_matrix_shape(
            &prefix.public_bottom_decryption_error,
            &projected_evaluated_error,
            "public-bottom decryption error must be a final scalar residual",
        );
        let projection_residual_error =
            projected_evaluated_error.clone() + &prefix.public_bottom_decryption_error;
        assert_same_matrix_shape(
            &prefix.final_decoder_error.matrix_norm,
            &projection_residual_error,
            "decoder error must be a final scalar residual",
        );
        let noisy_plaintext_error =
            projection_residual_error.clone() + &prefix.final_decoder_error.matrix_norm;
        info!(
            projection_residual_bits =
                bigdecimal_bits_ceil(&projection_residual_error.poly_norm.norm),
            noisy_plaintext_bits = bigdecimal_bits_ceil(&noisy_plaintext_error.poly_norm.norm),
            "AKY24IO final output error simulation finished"
        );

        Aky24IOErrorSimulation {
            initial_fresh_error: prefix.initial_fresh_error.clone(),
            final_decoder_error: prefix.final_decoder_error.matrix_norm.clone(),
            prf_refreshes: prefix.prf_refreshes.clone(),
            projection_residual_error,
            noisy_plaintext_error,
        }
    }

    fn full_active_levels(&self, params: &DCRTPolyParams) -> usize {
        sim_utils::full_active_levels(
            params,
            self.ring_gsw_enable_levels,
            self.ring_gsw_level_offset,
        )
    }

    fn cpu_ring_gsw_config(&self) -> sim_utils::CpuRingGswContextConfig {
        sim_utils::CpuRingGswContextConfig {
            p_moduli_bits: self.ring_gsw_context.p_moduli_bits,
            max_unreduced_muls: self.ring_gsw_context.max_unreduced_muls,
            scale: self.ring_gsw_context.scale,
            level_offset: self.ring_gsw_level_offset,
        }
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
        assert!(
            seed_errors.len() >= REPRESENTATIVE_GOLDREICH_SEED_BITS,
            "representative Goldreich PRG error simulation requires at least five seed errors"
        );
        let (_, _, crt_depth) = params.to_crt();
        let full_active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        let build_start = Instant::now();
        debug!(
            round_idx,
            conceptual_output_bits,
            range_start,
            elapsed_ms = build_start.elapsed().as_millis(),
            "AKY24IO representative Goldreich PRG ErrorNorm circuit built"
        );
        sim_utils::simulate_representative_prg_output(
            params,
            full_active_levels,
            REPRESENTATIVE_GOLDREICH_SEED_BITS,
            self.cpu_ring_gsw_config(),
            one,
            &seed_errors,
            plt_evaluator,
            slot_transfer_evaluator,
            "AKY24IO",
        )
    }
}

fn cpu_params_from_poly_params<P: crate::poly::PolyParams>(params: &P) -> DCRTPolyParams {
    sim_utils::cpu_params_from_poly_params(params, "AKY24IO")
}

fn simulator_context(params: &DCRTPolyParams) -> Arc<SimulatorContext> {
    sim_utils::simulator_context(params, AKY24_IO_SECRET_SIZE)
}

fn fixed_noise_refresh_v_bits_match(
    simulation: &Aky24IOErrorSimulation,
    noise_refresh_v_bits: usize,
) -> bool {
    simulation
        .prf_refreshes
        .iter()
        .all(|refresh| refresh.noise_refresh.v_bits == noise_refresh_v_bits)
}

fn aky24_io_security_margins_hold(
    params: &DCRTPolyParams,
    simulation: &Aky24IOErrorSimulation,
    prf_mask_output_coeff_bits: usize,
    security_bit: usize,
) -> bool {
    let Some(final_threshold) = aky24_io_final_decode_margin(params, prf_mask_output_coeff_bits)
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
        aky24_io_noise_refresh_security_margin_holds(params, &refresh.noise_refresh, security_bit)
    })
}

fn aky24_io_noise_refresh_security_margin_holds(
    params: &DCRTPolyParams,
    refresh: &NoiseRefreshErrorSimulation,
    security_bit: usize,
) -> bool {
    sim_utils::noise_refresh_security_margin_holds(params, refresh, security_bit)
}

fn aky24_io_final_decode_margin(
    params: &DCRTPolyParams,
    prf_mask_output_coeff_bits: usize,
) -> Option<BigDecimal> {
    sim_utils::final_decode_margin(params, prf_mask_output_coeff_bits)
}

fn minimum_seed_refresh_prf_seed_bits(branch_count: usize) -> usize {
    sim_utils::minimum_seed_refresh_prf_seed_bits(
        branch_count,
        "AKY24IO",
        REPRESENTATIVE_GOLDREICH_SEED_BITS,
    )
}

fn max_final_mask_coeff_bits_for_seed(
    params: &DCRTPolyParams,
    output_size: usize,
    function_output_bits: usize,
    seed_bits: usize,
    max_bits: usize,
) -> usize {
    sim_utils::max_final_mask_coeff_bits_for_seed(
        params,
        output_size,
        function_output_bits,
        seed_bits,
        max_bits,
        aky24_io_final_prg_uniform_output_bits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        func_enc::NoCircuitEvaluator,
        gadgets::arith::{ModularArithmeticContext, NestedRnsPolyContext},
        matrix::dcrt_poly::DCRTPolyMatrix,
    };

    type TestAky24IO = Aky24IO<
        DCRTPolyMatrix,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
        NoCircuitEvaluator,
    >;

    fn test_scheme(active_levels: usize, input_size: usize, prf_batch_bits: usize) -> TestAky24IO {
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
        TestAky24IO::new(
            params.clone(),
            params,
            ring_gsw_context,
            ring_gsw_width,
            0,
            Some(active_levels),
            Some(0.0),
            b"aky24_io_error_simulation_test".to_vec(),
            input_size,
            1,
            6,
            prf_batch_bits,
            1,
            1,
            1,
            [0x24; 32],
            [0x42; 32],
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn test_minimum_aky24_io_prf_seed_bits_covers_seed_refresh_outputs() {
        let params = DCRTPolyParams::new(2, 1, 10, 5);
        let output_size = 3usize;
        let prf_batch_bits = 3usize;
        let prf_mask_output_coeff_bits = 2usize;
        let noise_refresh_v_bits = 1usize;
        let cbd_n = 1usize;
        let seed_bits = minimum_aky24_io_prf_seed_bits(
            &params,
            output_size,
            output_size,
            prf_batch_bits,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            cbd_n,
        );
        let ring_dim = params.ring_dimension() as usize;
        let prf_branch_count = aky24_io_prf_branch_count(prf_batch_bits);

        assert!(goldreich_output_bound_holds(seed_bits, prf_branch_count * seed_bits));
        assert!(goldreich_output_bound_holds(
            seed_bits,
            aky24_io_final_prg_uniform_output_bits(
                output_size,
                ring_dim,
                prf_mask_output_coeff_bits,
                output_size,
            ),
        ));
        assert!(seed_bits >= minimum_noise_refresh_seed_bits(&params, noise_refresh_v_bits, cbd_n));
    }

    #[test]
    fn test_prf_final_round_separator_uses_batched_public_prf_rounds() {
        let scheme = test_scheme(1, 12, 4);
        assert_eq!(scheme.prf_round_count(), 3);
        assert_eq!(scheme.prf_branch_count(), 16);
        assert_eq!(scheme.prf_final_round_idx(), 3);
    }

    #[test]
    fn test_fresh_error_base_does_not_require_input_injection() {
        let scheme = test_scheme(1, 1, 1);
        let base = scheme.simulate_mask_independent_error_base(
            Aky24IOFuncType::GoldreichPRF { output_bits: 1 },
            2.0,
            &NoCircuitEvaluator,
            &NoCircuitEvaluator,
        );

        assert_eq!(base.initial_fresh_error.matrix_norm.nrow, 1);
        assert_eq!(base.initial_fresh_error.matrix_norm.ncol, base.ctx.m_g);
        assert_eq!(base.final_decoder_error.matrix_norm.nrow, 1);
        assert_eq!(base.final_decoder_error.matrix_norm.ncol, 1);
        assert_eq!(
            base.initial_fresh_error.matrix_norm.poly_norm.norm,
            BigDecimal::from_f64(13.0).unwrap(),
            "fresh initial ErrorNorm must come directly from 6.5 * error_sigma"
        );
    }

    #[test]
    fn test_finish_error_growth_composes_projection_and_decoder_error() {
        let scheme = test_scheme(1, 1, 1);
        let params = DCRTPolyParams::new(2, 1, 10, 5);
        let ctx = simulator_context(&params);
        let matrix = |norm: u32, ncol: usize| {
            PolyMatrixNorm::new(ctx.clone(), 1, ncol, BigDecimal::from(norm), None)
        };
        let error =
            |norm: u32, ncol: usize| ErrorNorm::new(PolyNorm::one(ctx.clone()), matrix(norm, ncol));
        let prefix = Aky24IOMaskIndependentErrorPrefix {
            cpu_params: params,
            ctx: ctx.clone(),
            initial_fresh_error: error(1, ctx.m_g),
            final_decoder_error: error(5, 1),
            prf_refreshes: Vec::new(),
            one: error(1, ctx.m_g),
            decryption_key: error(1, ctx.m_g),
            final_seed_errors: vec![error(1, ctx.m_g); REPRESENTATIVE_GOLDREICH_SEED_BITS],
            function_secret_error: error(2, ctx.m_g),
            public_bottom_decryption_error: matrix(7, 1),
        };
        let final_mask_base_error = error(3, ctx.m_g);

        let simulation = scheme.finish_error_growth_from_mask_independent_prefix(
            &prefix,
            Aky24IOFuncType::GoldreichPRF { output_bits: 1 },
            2,
            Some(error(1, ctx.m_g)),
            Some(final_mask_base_error),
            &NoCircuitEvaluator,
            &NoCircuitEvaluator,
        );

        let expected_secret =
            error(2, ctx.m_g) + &scale_error_norm(error(3, ctx.m_g), &BigDecimal::from(2u32));
        let expected_projection =
            expected_secret.matrix_norm * PolyMatrixNorm::gadget_decomposed(ctx.clone(), 1);
        let expected_residual = expected_projection + &matrix(7, 1);
        let expected_noisy = expected_residual.clone() + &matrix(5, 1);
        assert_eq!(simulation.projection_residual_error, expected_residual);
        assert_eq!(simulation.noisy_plaintext_error, expected_noisy);
    }
}
