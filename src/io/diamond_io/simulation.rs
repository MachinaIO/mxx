use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use bigdecimal::BigDecimal;
#[cfg(test)]
use digest::Digest;
#[cfg(test)]
use keccak_asm::Keccak256;
use num_bigint::BigUint;
use tracing::{debug, info};

#[cfg(test)]
use crate::gadgets::fhe_prg::goldreich::evaluate_goldreich_uniform_range;
use crate::{
    circuit::PolyCircuit,
    decoder::simulation::{centered_mask_magnitude, validate_error_bound_security_margin},
    gadgets::{
        arith::NestedRnsPolyContext,
        fhe::{ring_gsw::RingGswCiphertext, ring_gsw_nested_rns::NestedRnsRingGswContext},
        fhe_prg::goldreich::{goldreich_output_bound_holds, minimum_goldreich_input_size},
    },
    input_injector::DiamondInputErrorSimulation,
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
    sampler::{PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler},
    simulator::{
        SimulatorContext, error_norm::ErrorNorm, poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};

use super::{DIAMOND_SECRET_SIZE, DiamondIO, DiamondIOFuncType};
use crate::io::utils::simulation::{
    self as sim_utils, assert_same_matrix_shape, branch_rebase_decoder_error,
    expand_logical_errors, ring_gsw_wire_count, scale_error_norm,
    simulate_selected_branch_rebase_error_norm,
};

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
    pub representative_selected_prg_ciphertext_decryption_error: PolyMatrixNorm,
    pub noise_refresh: NoiseRefreshErrorSimulation,
}

/// Largest safe PRF mask bit-width together with the simulation that certified it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOPrfMaskOutputCoeffBitsSearchResult {
    /// Largest PRF mask coefficient bit-width satisfying the final decode margin.
    pub prf_mask_output_coeff_bits: usize,
    /// Single noise-refresh mask bit-width that is safe for every simulated PRF refresh round.
    pub noise_refresh_v_bits: usize,
    /// Error simulation evaluated with `prf_mask_output_coeff_bits`.
    pub simulation: DiamondIOErrorSimulation,
}

/// Successful DiamondIO CRT-depth search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondIOCrtDepthSearchResult {
    /// Smallest CRT depth found by the search.
    pub crt_depth: usize,
    /// Smallest log2 ring dimension that satisfied the lattice-estimator security checks.
    pub log_ring_dim: usize,
    /// Ring dimension selected as `1 << log_ring_dim`.
    pub ring_dim: u32,
    /// Lattice-estimator security estimate for ternary secret and Gaussian error.
    pub achieved_secpar_for_gauss: Option<u64>,
    /// Lattice-estimator security estimate for ternary secret and CBD error.
    pub achieved_secpar_for_cbd: Option<u64>,
    /// Largest PRF mask coefficient bit-width that was safe at `crt_depth`.
    pub prf_mask_output_coeff_bits: usize,
    /// Single noise-refresh mask bit-width that is safe for every simulated PRF refresh round.
    pub noise_refresh_v_bits: usize,
    /// Minimum PRF seed length that was revalidated for the selected mask widths.
    pub seed_bits: usize,
    /// Final noisy-plaintext error bound certified for the selected candidate.
    pub total_noisy_plaintext_error: PolyMatrixNorm,
    /// Representative input-injection contribution used to seed the DiamondIO simulation.
    ///
    /// This is `input_injection.state_errors[0] * input_injection.output_preimage`, matching the
    /// projected state error used as the `one`, decryption-key, and decoder input bound.
    pub input_injection_projection_error: PolyMatrixNorm,
}

/// Counts the uniform Goldreich output bits needed for DiamondIO's final PRF mask.
pub fn diamond_io_final_mask_uniform_output_bits(
    output_size: usize,
    ring_dim: usize,
    prf_mask_output_coeff_bits: usize,
) -> usize {
    assert!(output_size > 0, "DiamondIO output_size must be positive");
    assert!(ring_dim > 0, "DiamondIO ring_dim must be positive");
    assert!(
        prf_mask_output_coeff_bits > 0,
        "DiamondIO PRF mask output coefficient bits must be positive"
    );
    output_size
        .checked_mul(ring_dim)
        .and_then(|count| count.checked_mul(prf_mask_output_coeff_bits))
        .expect("DiamondIO final mask uniform output bit count overflow")
}

/// Counts the final Goldreich PRG stream: mask bits first, then GoldreichPRF function bits.
pub fn diamond_io_final_prg_uniform_output_bits(
    output_size: usize,
    ring_dim: usize,
    prf_mask_output_coeff_bits: usize,
    function_output_bits: usize,
) -> usize {
    diamond_io_final_mask_uniform_output_bits(output_size, ring_dim, prf_mask_output_coeff_bits)
        .checked_add(function_output_bits)
        .expect("DiamondIO final PRG uniform output bit count overflow")
}

/// Returns the minimum seed length satisfying every Goldreich PRG output bound in DiamondIO.
pub fn minimum_diamond_io_prf_seed_bits(
    params: &DCRTPolyParams,
    input_batch_bits: usize,
    output_size: usize,
    function_output_bits: usize,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
    cbd_n: usize,
) -> usize {
    let ring_dim = params.ring_dimension() as usize;
    let seed_refresh_seed_bits = minimum_seed_refresh_prf_seed_bits(
        1usize
            .checked_shl(input_batch_bits as u32)
            .expect("DiamondIO seed-refresh branch count overflow"),
    );
    let final_mask_seed_bits =
        minimum_goldreich_input_size(diamond_io_final_prg_uniform_output_bits(
            output_size,
            ring_dim,
            prf_mask_output_coeff_bits,
            function_output_bits,
        ));
    let noise_refresh_seed_bits =
        minimum_noise_refresh_seed_bits(params, noise_refresh_v_bits, cbd_n);
    seed_refresh_seed_bits.max(final_mask_seed_bits).max(noise_refresh_seed_bits)
}

/// Returns the largest noise-refresh `v_bits` allowed before pre-rounding error is added.
pub fn diamond_io_max_noise_refresh_v_bits_without_pre_rounding_error(
    params: &DCRTPolyParams,
) -> Option<usize> {
    let zero_pre_rounding =
        PolyMatrixNorm::new(simulator_context(params), 1, 1, BigDecimal::from(0u32), None);
    max_safe_noise_refresh_v_bits(params, &zero_pre_rounding)
}

#[derive(Debug, Clone)]
struct DiamondIOMaskIndependentErrorPrefix {
    cpu_params: DCRTPolyParams,
    ctx: Arc<SimulatorContext>,
    input_injection: DiamondInputErrorSimulation,
    prf_refreshes: Vec<DiamondIOPrfRoundErrorSimulation>,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    final_decoder_error: ErrorNorm,
    final_seed_errors: Vec<ErrorNorm>,
    function_secret_error: ErrorNorm,
    public_bottom_decryption_error: PolyMatrixNorm,
}

#[derive(Debug)]
struct DiamondIOMaskIndependentErrorBase {
    cpu_params: DCRTPolyParams,
    ctx: Arc<SimulatorContext>,
    input_injection: DiamondInputErrorSimulation,
    one: ErrorNorm,
    decryption_key: ErrorNorm,
    refresh_decoder_error: ErrorNorm,
    final_decoder_error: ErrorNorm,
    initial_seed_errors: Vec<ErrorNorm>,
    initial_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
    seed_ciphertext_decomposed_randomizer_norm: PolyMatrixNorm,
    noise_refresh_ring_gsw_context: Arc<NestedRnsRingGswContext<DCRTPoly>>,
    prf_refresh_material_cache:
        Arc<Mutex<HashMap<DiamondIOPrfRefreshMaterialCacheKey, DiamondIOPrfRefreshMaterialState>>>,
}

impl Clone for DiamondIOMaskIndependentErrorBase {
    fn clone(&self) -> Self {
        Self {
            cpu_params: self.cpu_params.clone(),
            ctx: self.ctx.clone(),
            input_injection: self.input_injection.clone(),
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
            prf_refresh_material_cache: self.prf_refresh_material_cache.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct DiamondIOPrfRefreshRoundEvaluation {
    round: DiamondIOPrfRoundErrorSimulation,
    refreshed_seed: ErrorNorm,
    refreshed_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
    material_wire_error: ErrorNorm,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiamondIOPrfRefreshMaterialCacheKey {
    round_idx: usize,
    seed_bits: usize,
    noise_refresh_cbd_n: usize,
    ring_gsw_public_key_error_sigma: String,
    seed_error_matrix_norms: Vec<String>,
    seed_ciphertext_randomizer_rows: usize,
    seed_ciphertext_randomizer_cols: usize,
    seed_ciphertext_randomizer_norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiamondIORepresentativePrgOutputCacheKey {
    params: String,
    seed_bits: usize,
    ring_gsw_level_offset: usize,
    full_active_levels: usize,
    one_plaintext_norm: String,
    one_matrix_norm: String,
    seed_error_matrix_norms: Vec<String>,
}

static REPRESENTATIVE_PRG_OUTPUT_CACHE: OnceLock<
    Mutex<HashMap<DiamondIORepresentativePrgOutputCacheKey, ErrorNorm>>,
> = OnceLock::new();

#[derive(Debug, Clone)]
struct DiamondIOPrfRefreshMaterialState {
    selected: ErrorNorm,
    material_wire_error: ErrorNorm,
    representative_selected_prg_ciphertext_decryption_error: PolyMatrixNorm,
    refreshed_seed_ciphertext_randomizer_norm: PolyMatrixNorm,
}

/// Finds the smallest CRT depth whose correctness and security checks pass.
///
/// The search is binary over the inclusive range `[min_crt_depth, max_crt_depth]`. For each
/// candidate depth, `build_candidate` must construct a DiamondIO instance with matching polynomial
/// parameters and Ring-GSW context. Each candidate first selects one global noise-refresh `v_bits`
/// from the first PRF refresh round and the steady-state refresh round. With that value fixed, it
/// searches only the final PRF mask bit width, then revalidates the exact selected protocol with
/// the minimum seed length derived from those final widths. Passing candidates are recorded and the
/// search continues toward smaller CRT depths.
pub fn diamond_io_find_crt_depth<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST, PE, ST, BuildCandidate>(
    min_crt_depth: usize,
    max_crt_depth: usize,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
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
    BuildCandidate: FnMut(
        u32,
        usize,
        usize,
        Option<usize>,
    ) -> DiamondIO<M, US, HS, TS, PKPE, PKST, ENCPE, ENCST>,
{
    info!(
        min_crt_depth,
        max_crt_depth,
        min_log_ring_dim,
        max_log_ring_dim,
        max_prf_mask_output_coeff_bits,
        security_bit,
        "starting DiamondIO CRT-depth search"
    );
    let force_lattice_check = std::env::var_os("MXX_IO_FORCE_LATTICE_CHECK").is_some();
    let explicit_log_ring_dim = min_log_ring_dim == max_log_ring_dim && !force_lattice_check;
    if !explicit_log_ring_dim {
        sim_utils::assert_lattice_estimator_available("DiamondIO");
    }
    assert!(min_crt_depth > 0, "minimum CRT depth must be positive");
    assert!(min_crt_depth <= max_crt_depth, "CRT-depth search range must be non-empty");
    let mut low = min_crt_depth;
    let mut high = max_crt_depth;
    let mut found = Vec::new();
    let mut lattice_cache = sim_utils::SecureRingDimLatticeCache::default();
    while low <= high {
        let crt_depth = low + (high - low) / 2;
        info!(crt_depth, low, high, "evaluating DiamondIO CRT-depth candidate");
        let min_ring_dim = 1u32
            .checked_shl(min_log_ring_dim.try_into().expect("min_log_ring_dim must fit in u32"))
            .expect("minimum ring_dim shift overflow");
        let probe_candidate =
            build_candidate(min_ring_dim, crt_depth, max_prf_mask_output_coeff_bits, None);
        let error_sigma = probe_candidate.injector.error_sigma;
        let noise_refresh_cbd_n = probe_candidate.noise_refresh_cbd_n;
        let Some(ring_dim_search) = sim_utils::select_min_secure_ring_dim(
            "DiamondIO",
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
                candidate.injector.params.clone()
            },
        ) else {
            return None;
        };
        let ring_dim = ring_dim_search.ring_dim;
        let provisional_candidate =
            build_candidate(ring_dim, crt_depth, max_prf_mask_output_coeff_bits, None);
        let cpu_params = cpu_params_from_poly_params(&provisional_candidate.injector.params);
        let provisional_base = provisional_candidate.simulate_mask_independent_error_base(
            func_type,
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
            info!(crt_depth, "DiamondIO CRT-depth candidate failed noise-refresh v_bits search");
            low = crt_depth + 1;
            continue;
        };
        info!(
            crt_depth,
            log_ring_dim = ring_dim_search.log_ring_dim,
            ring_dim,
            global_noise_refresh_v_bits,
            "DiamondIO CRT-depth candidate selected fixed noise-refresh v_bits"
        );

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
            info!(
                crt_depth,
                global_noise_refresh_v_bits,
                "DiamondIO CRT-depth candidate failed fixed-v prefix construction"
            );
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
            info!(
                crt_depth,
                global_noise_refresh_v_bits,
                "DiamondIO CRT-depth candidate failed final PRF mask-bit search"
            );
            low = crt_depth + 1;
            continue;
        };
        let mask_bits = mask_search.prf_mask_output_coeff_bits;
        let final_candidate =
            build_candidate(ring_dim, crt_depth, mask_bits, Some(global_noise_refresh_v_bits));
        let expected_seed_bits = minimum_diamond_io_prf_seed_bits(
            &cpu_params,
            final_candidate.injector.batch_bits(),
            func_type.output_bits(),
            func_type.output_bits(),
            mask_bits,
            global_noise_refresh_v_bits,
            final_candidate.noise_refresh_cbd_n,
        );
        if final_candidate.seed_bits != expected_seed_bits {
            info!(
                crt_depth,
                mask_bits,
                global_noise_refresh_v_bits,
                final_seed_bits = final_candidate.seed_bits,
                expected_seed_bits,
                "DiamondIO final candidate seed_bits did not match the selected minimum"
            );
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
            info!(
                crt_depth,
                mask_bits,
                global_noise_refresh_v_bits,
                final_seed_bits = final_candidate.seed_bits,
                "DiamondIO final fixed-v validation failed to produce a simulation"
            );
            low = crt_depth + 1;
            continue;
        };
        if !fixed_noise_refresh_v_bits_match(&final_simulation, global_noise_refresh_v_bits) ||
            !diamond_io_security_margins_hold(
                &cpu_params,
                &final_simulation,
                mask_bits,
                security_bit,
            )
        {
            info!(
                crt_depth,
                mask_bits,
                global_noise_refresh_v_bits,
                final_seed_bits = final_candidate.seed_bits,
                "DiamondIO final fixed-v validation failed security margins"
            );
            low = crt_depth + 1;
            continue;
        }

        info!(
            crt_depth,
            mask_bits,
            noise_refresh_v_bits = global_noise_refresh_v_bits,
            final_seed_bits = final_candidate.seed_bits,
            "DiamondIO CRT-depth candidate passed final minimum-seed validation"
        );
        found.push(DiamondIOCrtDepthSearchResult {
            crt_depth,
            log_ring_dim: ring_dim_search.log_ring_dim,
            ring_dim,
            achieved_secpar_for_gauss: ring_dim_search.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd: ring_dim_search.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits: mask_bits,
            noise_refresh_v_bits: global_noise_refresh_v_bits,
            seed_bits: final_candidate.seed_bits,
            total_noisy_plaintext_error: final_simulation.noisy_plaintext_error.clone(),
            input_injection_projection_error: diamond_io_input_injection_projection_error(
                &final_simulation.input_injection,
            ),
        });
        high = crt_depth - 1;
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
    fn prf_round_count(&self) -> usize {
        self.injector.input_count
    }

    fn prf_branch_count(&self) -> usize {
        1usize
            .checked_shl(self.injector.batch_bits() as u32)
            .expect("DiamondIO PRF digit branch count overflow")
    }

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
            self.noise_refresh_v_bits,
            plt_evaluator,
            slot_transfer_evaluator,
        )
    }

    /// Search for the largest PRF mask coefficient bit-width that satisfies the final decode
    /// margin.
    ///
    /// The mask-independent prefix and final-mask representative PRG are cached across candidates.
    /// The remaining per-candidate work accounts for that bit width's mask decrypt/recomposition
    /// error. The final mask is centered, so a candidate is valid exactly when
    /// `noisy_plaintext_error + 2^(candidate - 1) < q / 4`.
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
        let prefix = self.simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
            func_type,
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

    fn search_final_prf_mask_output_coeff_bits_from_prefix<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        prefix: &DiamondIOMaskIndependentErrorPrefix,
        max_bits: usize,
        noise_refresh_v_bits: usize,
        security_bit: Option<usize>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOPrfMaskOutputCoeffBitsSearchResult>
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
            info!(
                max_bits,
                seed_bits = self.seed_bits,
                "DiamondIO PRF mask bit search found no candidate satisfying Goldreich output bound"
            );
            return None;
        }
        info!(
            max_bits,
            max_candidate,
            noise_refresh_v_bits,
            ?security_bit,
            "starting DiamondIO fixed-v final PRF mask bit search"
        );
        info!(
            prf_rounds = prefix.prf_refreshes.len(),
            "DiamondIO fixed-v final PRF mask bit search using cached prefix"
        );
        let cached_final_mask_prg_output = self.simulate_representative_prg_output(
            &prefix.cpu_params,
            self.prf_final_round_idx(),
            func_type
                .output_bits()
                .checked_mul(prefix.cpu_params.ring_dimension() as usize)
                .and_then(|count| count.checked_mul(max_candidate))
                .and_then(|count| count.checked_add(func_type.output_bits()))
                .expect("DiamondIO final mask PRG conceptual output count overflow"),
            0,
            prefix.one.clone(),
            prefix.final_seed_errors.clone(),
            plt_evaluator,
            slot_transfer_evaluator,
        );
        info!(
            max_candidate,
            final_mask_prg_bits =
                bigdecimal_bits_ceil(&cached_final_mask_prg_output.matrix_norm.poly_norm.norm),
            "DiamondIO PRF mask bit search cached final-mask representative PRG"
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
            "DiamondIO",
        );
        info!(
            final_mask_base_bits =
                bigdecimal_bits_ceil(&cached_final_mask_base_error.matrix_norm.poly_norm.norm),
            "DiamondIO PRF mask bit search cached final-mask representative decrypt"
        );
        let mut low = 1usize;
        let mut high = max_candidate;
        let mut best = None;
        while low <= high {
            let candidate = low + (high - low) / 2;
            info!(candidate, "evaluating DiamondIO fixed-v final PRF mask bit candidate");
            let simulation = self.finish_error_growth_from_mask_independent_prefix(
                prefix,
                func_type,
                candidate,
                Some(cached_final_mask_prg_output.clone()),
                Some(cached_final_mask_base_error.clone()),
                plt_evaluator,
                slot_transfer_evaluator,
            );
            let Some(final_margin) = diamond_io_final_decode_margin(&prefix.cpu_params, candidate)
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
            debug!(
                candidate,
                valid,
                noisy_plaintext_error = %error,
                mask_value = %centered_mask_magnitude(candidate),
                "DiamondIO fixed-v final PRF mask bit candidate evaluated"
            );
            if valid {
                best = Some(DiamondIOPrfMaskOutputCoeffBitsSearchResult {
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
        info!(found = best.is_some(), "finished DiamondIO fixed-v final PRF mask bit search");
        best
    }

    fn simulate_error_growth_with_prf_mask_output_coeff_bits<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        prf_mask_output_coeff_bits: usize,
        noise_refresh_v_bits: usize,
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
        let final_mask_prg_output_bits = diamond_io_final_prg_uniform_output_bits(
            func_type.output_bits(),
            self.injector.params.ring_dimension() as usize,
            prf_mask_output_coeff_bits,
            func_type.output_bits(),
        );
        if !goldreich_output_bound_holds(self.seed_bits, final_mask_prg_output_bits) {
            info!(
                seed_bits = self.seed_bits,
                final_mask_prg_output_bits,
                prf_mask_output_coeff_bits,
                "DiamondIO error simulation rejected unsafe final-mask PRG output bound"
            );
            return None;
        }

        let prefix = self.simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
            func_type,
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
        _func_type: DiamondIOFuncType,
        _plt_evaluator: &PE,
        _slot_transfer_evaluator: &ST,
    ) -> DiamondIOMaskIndependentErrorBase
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

        let cpu_params = cpu_params_from_poly_params(&self.injector.params);
        let prefix_required_seed_bits = minimum_seed_refresh_prf_seed_bits(self.prf_branch_count())
            .max(minimum_noise_refresh_seed_bits(
                &cpu_params,
                self.noise_refresh_v_bits,
                self.noise_refresh_cbd_n,
            ));
        assert!(
            self.seed_bits >= prefix_required_seed_bits,
            "DiamondIO seed_bits {} is below Goldreich PRG safety minimum {} for seed-refresh PRF and noise refresh",
            self.seed_bits,
            prefix_required_seed_bits
        );
        let (_, _, crt_depth) = cpu_params.to_crt();
        let full_active_levels = self.full_active_levels(&cpu_params);
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
        let refresh_decoder_error =
            ErrorNorm::new(PolyNorm::one(ctx.clone()), one.matrix_norm.clone());
        let final_decoder_projection_error =
            diamond_io_input_injection_scalar_projection_error(&input_injection);
        let final_decoder_error =
            ErrorNorm::new(PolyNorm::one(ctx.clone()), final_decoder_projection_error);
        let seed_wire_error = sim_utils::initial_seed_wire_error(
            &one,
            self.ring_gsw_context.p_moduli.as_slice(),
            "DiamondIO",
        );
        let initial_seed_errors = vec![seed_wire_error.clone(); 5];
        let (initial_seed_ciphertext_randomizer_norm, seed_ciphertext_decomposed_randomizer_norm) =
            sim_utils::seed_ciphertext_randomizer_norms(&cpu_params, self.cpu_ring_gsw_config());
        let mut noise_refresh_context_circuit = PolyCircuit::<DCRTPoly>::new();
        let noise_refresh_ring_gsw_context = sim_utils::build_cpu_ring_gsw_context(
            &cpu_params,
            &mut noise_refresh_context_circuit,
            self.full_active_levels(&cpu_params),
            self.cpu_ring_gsw_config(),
        );
        info!("DiamondIO mask-independent error simulation base finished");

        DiamondIOMaskIndependentErrorBase {
            cpu_params,
            ctx,
            input_injection,
            one,
            decryption_key,
            refresh_decoder_error,
            final_decoder_error,
            initial_seed_errors,
            initial_seed_ciphertext_randomizer_norm,
            seed_ciphertext_decomposed_randomizer_norm,
            noise_refresh_ring_gsw_context,
            prf_refresh_material_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn select_global_noise_refresh_v_bits<PE, ST>(
        &self,
        base: &DiamondIOMaskIndependentErrorBase,
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
            diamond_io_max_noise_refresh_v_bits_without_pre_rounding_error(&base.cpu_params)?;
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
        let steady_seed_errors = vec![first.refreshed_seed; 5];
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
        // Each searched round returns its largest security-margin-safe mask width. A single global
        // fixed width must be no larger than both maxima, so the largest common safe choice is
        // their minimum. Reducing `v_bits` reduces the centered mask magnitude and the mask-bit
        // contribution to the simulated pre-rounding error, so it cannot make the rounding margin
        // worse; final validation below still rechecks the chosen global value end-to-end.
        Some(first.round.noise_refresh.v_bits.min(steady.round.noise_refresh.v_bits))
    }

    fn simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits<PE, ST>(
        &self,
        func_type: DiamondIOFuncType,
        noise_refresh_v_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOMaskIndependentErrorPrefix>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let base = self.simulate_mask_independent_error_base(
            func_type,
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
        base: DiamondIOMaskIndependentErrorBase,
        func_type: DiamondIOFuncType,
        noise_refresh_v_bits: usize,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOMaskIndependentErrorPrefix>
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
            final_seed_errors = vec![first.refreshed_seed.clone(); 5];
            final_seed_ciphertext_randomizer_norm =
                first.refreshed_seed_ciphertext_randomizer_norm.clone();
            prf_refreshes.push(first.round);

            if prf_round_count > 1 {
                // After the first refresh, every seed wire has the same rounded CBD bound. The
                // second round therefore gives the steady-state representative refresh used for
                // all later rounds; later graph seeds change, but the error shape does not.
                let steady = self.simulate_prf_refresh_round_fixed(
                    &base,
                    1,
                    noise_refresh_v_bits,
                    &final_seed_errors,
                    &final_seed_ciphertext_randomizer_norm,
                    plt_evaluator,
                    slot_transfer_evaluator,
                )?;
                final_seed_errors = vec![steady.refreshed_seed.clone(); 5];
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
                        "DiamondIO",
                    )
                    .expect("steady-state DiamondIO PRF refresh seed bound was prevalidated");
                    let evaluation = DiamondIOPrfRefreshRoundEvaluation {
                        round: DiamondIOPrfRoundErrorSimulation {
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
                    final_seed_errors = vec![evaluation.refreshed_seed.clone(); 5];
                    final_seed_ciphertext_randomizer_norm =
                        refreshed_seed_ciphertext_randomizer_norm;
                    info!(
                        round_idx,
                        v_bits = evaluation.round.noise_refresh.v_bits,
                        refreshed_seed_error_bits = bigdecimal_bits_ceil(
                            &evaluation.refreshed_seed.matrix_norm.poly_norm.norm
                        ),
                        selected_ciphertext_decryption_bits = bigdecimal_bits_ceil(
                            &evaluation
                                .round
                                .representative_selected_prg_ciphertext_decryption_error
                                .poly_norm
                                .norm
                        ),
                        "DiamondIO reusing fixed-v steady-state PRF refresh error simulation"
                    );
                    prf_refreshes.push(evaluation.round);
                }
            }
        }
        info!(
            prf_rounds = prf_refreshes.len(),
            noise_refresh_v_bits,
            "DiamondIO fixed-v mask-independent error simulation prefix finished"
        );
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
        let (function_secret_error, function_public_bottom_error) =
            sim_utils::simulate_representative_function_output(
                &base.cpu_params,
                self.full_active_levels(&base.cpu_params),
                self.cpu_ring_gsw_config(),
                function_prg_output,
                base.one.clone(),
                base.decryption_key.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
                "DiamondIO",
            );
        let (public_bottom_decryption_error, _public_bottom_randomizer_norm) =
            sim_utils::prf_refresh_ciphertext_state(
                &final_seed_ciphertext_randomizer_norm,
                &base.seed_ciphertext_decomposed_randomizer_norm,
                base.ctx.clone(),
                self.full_active_levels(&base.cpu_params),
                self.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
            );
        info!(
            function_secret_bits =
                bigdecimal_bits_ceil(&function_secret_error.matrix_norm.poly_norm.norm),
            function_public_bottom_plaintext_bits =
                bigdecimal_bits_ceil(&function_public_bottom_error.plaintext_norm.norm),
            function_public_bottom_encoding_bits =
                bigdecimal_bits_ceil(&function_public_bottom_error.matrix_norm.poly_norm.norm),
            accumulated_public_bottom_decryption_bits =
                bigdecimal_bits_ceil(&public_bottom_decryption_error.poly_norm.norm),
            "DiamondIO representative GoldreichPRF output error simulation finished"
        );
        Some(DiamondIOMaskIndependentErrorPrefix {
            cpu_params: base.cpu_params,
            ctx: base.ctx,
            input_injection: base.input_injection,
            prf_refreshes,
            one: base.one,
            decryption_key: base.decryption_key,
            final_decoder_error: base.final_decoder_error,
            final_seed_errors,
            function_secret_error,
            public_bottom_decryption_error,
        })
    }

    fn search_prf_refresh_round_fixed_v_bits<PE, ST>(
        &self,
        base: &DiamondIOMaskIndependentErrorBase,
        round_idx: usize,
        max_candidate: usize,
        security_bit: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOPrfRefreshRoundEvaluation>
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let mut low = 1usize;
        let mut high = max_candidate;
        let mut best = None;
        while low <= high {
            let candidate = low + (high - low) / 2;
            info!(
                round_idx,
                candidate,
                low,
                high,
                "DiamondIO fixed-v noise-refresh candidate simulation started"
            );
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
            let valid = diamond_io_noise_refresh_security_margin_holds(
                &base.cpu_params,
                &evaluation.round.noise_refresh,
                security_bit,
            );
            info!(
                round_idx,
                candidate,
                valid,
                selected_prg_bits = bigdecimal_bits_ceil(
                    &evaluation.round.representative_selected_prg_output.matrix_norm.poly_norm.norm
                ),
                selected_ciphertext_decryption_bits = bigdecimal_bits_ceil(
                    &evaluation
                        .round
                        .representative_selected_prg_ciphertext_decryption_error
                        .poly_norm
                        .norm
                ),
                "DiamondIO fixed-v noise-refresh security candidate evaluated"
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
        base: &DiamondIOMaskIndependentErrorBase,
        round_idx: usize,
        noise_refresh_v_bits: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> Option<DiamondIOPrfRefreshRoundEvaluation>
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
        info!(
            round_idx,
            noise_refresh_v_bits, "DiamondIO fixed-v PRF refresh noise-refresh simulation started"
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
            "DiamondIO",
        )?;
        info!(
            round_idx,
            noise_refresh_v_bits, "DiamondIO fixed-v PRF refresh noise-refresh simulation finished"
        );
        info!(
            round_idx,
            v_bits = core.refresh.v_bits,
            refreshed_seed_error_bits =
                bigdecimal_bits_ceil(&core.refreshed_seed.matrix_norm.poly_norm.norm),
            selected_ciphertext_decryption_bits = bigdecimal_bits_ceil(
                &material_state
                    .representative_selected_prg_ciphertext_decryption_error
                    .poly_norm
                    .norm
            ),
            noise_refresh_material_branch_count_online = 1usize,
            "DiamondIO PRF refresh error simulation finished"
        );
        Some(DiamondIOPrfRefreshRoundEvaluation {
            round: DiamondIOPrfRoundErrorSimulation {
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
        base: &DiamondIOMaskIndependentErrorBase,
        round_idx: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> DiamondIOPrfRefreshMaterialState
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        let cache_key = self.prf_refresh_material_cache_key(
            round_idx,
            seed_errors,
            seed_ciphertext_randomizer_norm,
        );
        if let Some(cached) = base
            .prf_refresh_material_cache
            .lock()
            .expect("DiamondIO PRF material cache mutex must not be poisoned")
            .get(&cache_key)
            .cloned()
        {
            info!(round_idx, "DiamondIO fixed-v PRF refresh material state cache hit");
            return cached;
        }

        info!(round_idx, "DiamondIO fixed-v PRF refresh material state cache miss");
        info!(
            round_idx,
            "DiamondIO fixed-v PRF refresh representative ciphertext-state simulation started"
        );
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
        info!(
            round_idx,
            selected_ciphertext_decryption_bits = bigdecimal_bits_ceil(
                &representative_selected_prg_ciphertext_decryption_error.poly_norm.norm
            ),
            "DiamondIO fixed-v PRF refresh representative ciphertext-state simulation finished"
        );
        info!(round_idx, "DiamondIO fixed-v PRF refresh material-input simulation started");
        let (selected, material_wire_error) = self.simulate_prf_refresh_material_inputs(
            base,
            round_idx,
            seed_errors,
            plt_evaluator,
            slot_transfer_evaluator,
        );
        info!(
            round_idx,
            selected_prg_bits = bigdecimal_bits_ceil(&selected.matrix_norm.poly_norm.norm),
            material_wire_bits =
                bigdecimal_bits_ceil(&material_wire_error.matrix_norm.poly_norm.norm),
            "DiamondIO fixed-v PRF refresh material-input simulation finished"
        );
        let state = DiamondIOPrfRefreshMaterialState {
            selected,
            material_wire_error,
            representative_selected_prg_ciphertext_decryption_error,
            refreshed_seed_ciphertext_randomizer_norm,
        };
        base.prf_refresh_material_cache
            .lock()
            .expect("DiamondIO PRF material cache mutex must not be poisoned")
            .insert(cache_key, state.clone());
        state
    }

    fn prf_refresh_material_cache_key(
        &self,
        round_idx: usize,
        seed_errors: &[ErrorNorm],
        seed_ciphertext_randomizer_norm: &PolyMatrixNorm,
    ) -> DiamondIOPrfRefreshMaterialCacheKey {
        DiamondIOPrfRefreshMaterialCacheKey {
            round_idx,
            seed_bits: self.seed_bits,
            noise_refresh_cbd_n: self.noise_refresh_cbd_n,
            ring_gsw_public_key_error_sigma: self
                .ring_gsw_public_key_error_sigma
                .map(|sigma| sigma.to_string())
                .unwrap_or_else(|| "None".to_string()),
            seed_error_matrix_norms: seed_errors
                .iter()
                .map(|error| error.matrix_norm.poly_norm.norm.to_string())
                .collect(),
            seed_ciphertext_randomizer_rows: seed_ciphertext_randomizer_norm.nrow,
            seed_ciphertext_randomizer_cols: seed_ciphertext_randomizer_norm.ncol,
            seed_ciphertext_randomizer_norm: seed_ciphertext_randomizer_norm
                .poly_norm
                .norm
                .to_string(),
        }
    }

    fn simulate_prf_refresh_material_inputs<PE, ST>(
        &self,
        base: &DiamondIOMaskIndependentErrorBase,
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
                .expect("DiamondIO seed-refresh PRF output size overflow"),
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

    fn finish_error_growth_from_mask_independent_prefix<PE, ST>(
        &self,
        prefix: &DiamondIOMaskIndependentErrorPrefix,
        func_type: DiamondIOFuncType,
        prf_mask_output_coeff_bits: usize,
        cached_final_mask_prg_output: Option<ErrorNorm>,
        cached_final_mask_base_error: Option<ErrorNorm>,
        plt_evaluator: &PE,
        slot_transfer_evaluator: &ST,
    ) -> DiamondIOErrorSimulation
    where
        PE: PltEvaluator<ErrorNorm>,
        ST: SlotTransferEvaluator<ErrorNorm>,
    {
        assert!(
            prf_mask_output_coeff_bits > 0,
            "DiamondIO error simulation requires prf_mask_output_coeff_bits > 0"
        );
        let final_mask_prg_output = cached_final_mask_prg_output.unwrap_or_else(|| {
            self.simulate_representative_prg_output(
                &prefix.cpu_params,
                self.prf_final_round_idx(),
                func_type
                    .output_bits()
                    .checked_mul(prefix.cpu_params.ring_dimension() as usize)
                    .and_then(|count| count.checked_mul(prf_mask_output_coeff_bits))
                    .and_then(|count| count.checked_add(func_type.output_bits()))
                    .expect("DiamondIO final mask PRG conceptual output count overflow"),
                0,
                prefix.one.clone(),
                prefix.final_seed_errors.clone(),
                plt_evaluator,
                slot_transfer_evaluator,
            )
        });
        info!(
            prf_mask_output_coeff_bits,
            final_mask_prg_bits =
                bigdecimal_bits_ceil(&final_mask_prg_output.matrix_norm.poly_norm.norm),
            "DiamondIO final mask representative PRG error simulation finished"
        );
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
                "DiamondIO",
            )
        });
        let final_mask_error = scale_error_norm(
            final_mask_base_error,
            &BigDecimal::from(prf_mask_output_coeff_bits as u64),
        );
        assert_same_matrix_shape(
            &final_mask_error.matrix_norm,
            &prefix.function_secret_error.matrix_norm,
            "final mask and function secret-dependent representative errors must share the pre-projection BGG encoding shape",
        );
        let evaluated_secret_dependent = prefix.function_secret_error.clone() + &final_mask_error;
        let projected_evaluated_error = evaluated_secret_dependent.matrix_norm.clone() *
            PolyMatrixNorm::gadget_decomposed(prefix.ctx.clone(), 1);
        assert_same_matrix_shape(
            &prefix.public_bottom_decryption_error,
            &projected_evaluated_error,
            "public-bottom decryption error must be a representative final scalar residual error",
        );
        let projection_residual_error =
            projected_evaluated_error.clone() + &prefix.public_bottom_decryption_error;
        assert_same_matrix_shape(
            &prefix.final_decoder_error.matrix_norm,
            &projection_residual_error,
            "decoder projection error must be a representative final scalar residual error",
        );
        let noisy_plaintext_error =
            projection_residual_error.clone() + &prefix.final_decoder_error.matrix_norm;
        let input_injection_projection_error =
            diamond_io_input_injection_projection_error(&prefix.input_injection);
        let input_injection_error_bits =
            bigdecimal_bits_ceil(&input_injection_projection_error.poly_norm.norm);
        let noisy_plaintext_bits = bigdecimal_bits_ceil(&noisy_plaintext_error.poly_norm.norm);
        info!(
            projected_evaluated_bits =
                bigdecimal_bits_ceil(&projected_evaluated_error.poly_norm.norm),
            public_bottom_decryption_bits =
                bigdecimal_bits_ceil(&prefix.public_bottom_decryption_error.poly_norm.norm),
            projection_residual_bits =
                bigdecimal_bits_ceil(&projection_residual_error.poly_norm.norm),
            noisy_plaintext_bits,
            input_injection_error_bits,
            input_injection_error_percent_upper_bound =
                percent_upper_bound_from_bits(input_injection_error_bits, noisy_plaintext_bits),
            "DiamondIO final output error simulation finished"
        );

        DiamondIOErrorSimulation {
            input_injection: prefix.input_injection.clone(),
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
            seed_errors.len() >= 5,
            "representative Goldreich PRG error simulation requires at least five seed errors"
        );
        let (_, _, crt_depth) = params.to_crt();
        let full_active_levels =
            self.ring_gsw_enable_levels.unwrap_or_else(|| crt_depth - self.ring_gsw_level_offset);
        let cache_key = self.representative_prg_output_cache_key(
            params,
            full_active_levels,
            &one,
            &seed_errors,
        );
        let output_cache =
            REPRESENTATIVE_PRG_OUTPUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(cached) = output_cache
            .lock()
            .expect("DiamondIO representative PRG ErrorNorm cache mutex must not be poisoned")
            .get(&cache_key)
            .cloned()
        {
            info!(
                round_idx,
                conceptual_output_bits,
                range_start,
                output_error_bits = bigdecimal_bits_ceil(&cached.matrix_norm.poly_norm.norm),
                "DiamondIO representative Goldreich PRG ErrorNorm cache hit"
            );
            return cached;
        }

        let build_start = Instant::now();
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            "DiamondIO representative Goldreich PRG ErrorNorm circuit build started"
        );
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            elapsed_ms = build_start.elapsed().as_millis(),
            "DiamondIO representative Goldreich PRG ErrorNorm circuit build finished"
        );
        let representative_seed_bits = 5usize;
        let eval_start = Instant::now();
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            "DiamondIO representative Goldreich PRG ErrorNorm eval started"
        );
        let output = sim_utils::simulate_representative_prg_output(
            params,
            full_active_levels,
            representative_seed_bits,
            self.cpu_ring_gsw_config(),
            one,
            &seed_errors,
            plt_evaluator,
            slot_transfer_evaluator,
            "DiamondIO",
        );
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            elapsed_ms = eval_start.elapsed().as_millis(),
            "DiamondIO representative Goldreich PRG ErrorNorm eval finished"
        );
        info!(
            round_idx,
            conceptual_output_bits,
            range_start,
            simulated_active_levels = 1usize,
            extrapolated_active_levels = full_active_levels,
            output_error_bits = bigdecimal_bits_ceil(&output.matrix_norm.poly_norm.norm),
            "DiamondIO representative Goldreich PRG error simulated"
        );
        output_cache
            .lock()
            .expect("DiamondIO representative PRG ErrorNorm cache mutex must not be poisoned")
            .insert(cache_key, output.clone());
        output
    }

    fn representative_prg_output_cache_key(
        &self,
        params: &DCRTPolyParams,
        full_active_levels: usize,
        one: &ErrorNorm,
        seed_errors: &[ErrorNorm],
    ) -> DiamondIORepresentativePrgOutputCacheKey {
        DiamondIORepresentativePrgOutputCacheKey {
            params: format!("{params:?}"),
            seed_bits: self.seed_bits,
            ring_gsw_level_offset: self.ring_gsw_level_offset,
            full_active_levels,
            one_plaintext_norm: one.plaintext_norm.norm.to_string(),
            one_matrix_norm: one.matrix_norm.poly_norm.norm.to_string(),
            seed_error_matrix_norms: seed_errors
                .iter()
                .map(|error| error.matrix_norm.poly_norm.norm.to_string())
                .collect(),
        }
    }

    #[cfg(test)]
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
        let ring_gsw_context = sim_utils::build_cpu_ring_gsw_context(
            params,
            &mut circuit,
            active_levels,
            self.cpu_ring_gsw_config(),
        );
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

    #[cfg(test)]
    fn derive_diamond_goldreich_graph_seed(&self, round_idx: usize) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(b"DiamondIOGoldreichPrfGraph/v1");
        hasher.update(self.goldreich_graph_seed);
        hasher.update(round_idx.to_le_bytes());
        hasher.finalize().into()
    }
}

fn cpu_params_from_poly_params<P: crate::poly::PolyParams>(params: &P) -> DCRTPolyParams {
    sim_utils::cpu_params_from_poly_params(params, "DiamondIO")
}

fn simulator_context(params: &DCRTPolyParams) -> Arc<SimulatorContext> {
    sim_utils::simulator_context(params, DIAMOND_SECRET_SIZE)
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

fn diamond_io_input_injection_scalar_projection_error(
    input_injection: &DiamondInputErrorSimulation,
) -> PolyMatrixNorm {
    let state_error = input_injection
        .state_errors
        .first()
        .expect("DiamondIO input injection must produce a base final state error");
    let scalar_output_preimage = PolyMatrixNorm::new(
        input_injection.output_preimage.clone_ctx(),
        input_injection.output_preimage.nrow,
        1,
        input_injection.output_preimage.poly_norm.norm.clone(),
        None,
    );
    state_error.clone() * &scalar_output_preimage
}

fn percent_upper_bound_from_bits(part_bits: u64, total_bits: u64) -> f64 {
    let exponent = (part_bits as i64) - (total_bits as i64);
    if exponent < i32::MIN as i64 {
        0.0
    } else if exponent > i32::MAX as i64 {
        f64::INFINITY
    } else {
        100.0 * 2.0_f64.powi(exponent as i32)
    }
}

fn fixed_noise_refresh_v_bits_match(
    simulation: &DiamondIOErrorSimulation,
    noise_refresh_v_bits: usize,
) -> bool {
    simulation
        .prf_refreshes
        .iter()
        .all(|refresh| refresh.noise_refresh.v_bits == noise_refresh_v_bits)
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
        diamond_io_noise_refresh_security_margin_holds(params, &refresh.noise_refresh, security_bit)
    })
}

fn diamond_io_noise_refresh_security_margin_holds(
    params: &DCRTPolyParams,
    refresh: &NoiseRefreshErrorSimulation,
    security_bit: usize,
) -> bool {
    sim_utils::noise_refresh_security_margin_holds(params, refresh, security_bit)
}

fn diamond_io_final_decode_margin(
    params: &DCRTPolyParams,
    prf_mask_output_coeff_bits: usize,
) -> Option<BigDecimal> {
    sim_utils::final_decode_margin(params, prf_mask_output_coeff_bits)
}

fn minimum_seed_refresh_prf_seed_bits(branch_count: usize) -> usize {
    sim_utils::minimum_seed_refresh_prf_seed_bits(branch_count, "DiamondIO", 5usize)
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
        diamond_io_final_prg_uniform_output_bits,
    )
}

fn eval_first_error_output<PE>(
    circuit: PolyCircuit<DCRTPoly>,
    one: ErrorNorm,
    inputs: Vec<ErrorNorm>,
    plt_evaluator: Option<&PE>,
    slot_transfer_evaluator: Option<&dyn SlotTransferEvaluator<ErrorNorm>>,
) -> ErrorNorm
where
    PE: PltEvaluator<ErrorNorm>,
{
    sim_utils::eval_first_error_output(
        circuit,
        one,
        inputs,
        plt_evaluator,
        slot_transfer_evaluator,
        "DiamondIO",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        func_enc::NoCircuitEvaluator,
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
        >::new(params.clone(), input_count, 2, 1, 4.578, 0.0);
        TestDiamondIO::new(
            injector,
            input_count,
            6,
            params.clone(),
            ring_gsw_context,
            ring_gsw_width,
            0,
            Some(active_levels),
            Some(0.0),
            b"diamond_io_error_simulation_test".to_vec(),
            6,
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

    fn make_scheme_seed_bound_safe(scheme: &mut TestDiamondIO) {
        let params = cpu_params_from_poly_params(&scheme.injector.params);
        scheme.seed_bits = scheme.seed_bits.max(minimum_diamond_io_prf_seed_bits(
            &params,
            scheme.injector.batch_bits(),
            scheme.output_size,
            scheme.output_size,
            scheme.prf_mask_output_coeff_bits,
            scheme.noise_refresh_v_bits,
            scheme.noise_refresh_cbd_n,
        ));
    }

    #[test]
    fn test_minimum_diamond_io_prf_seed_bits_covers_digit_branch_outputs() {
        let params = DCRTPolyParams::new(2, 1, 10, 5);
        let input_batch_bits = 3usize;
        let branch_count = 1usize << input_batch_bits;
        let output_size = 3usize;
        let prf_mask_output_coeff_bits = 2usize;
        let noise_refresh_v_bits = 1usize;
        let cbd_n = 1usize;
        let seed_bits = minimum_diamond_io_prf_seed_bits(
            &params,
            input_batch_bits,
            output_size,
            output_size,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            cbd_n,
        );
        let ring_dim = params.ring_dimension() as usize;

        assert!(goldreich_output_bound_holds(seed_bits, branch_count * seed_bits));
        assert!(goldreich_output_bound_holds(
            seed_bits,
            diamond_io_final_prg_uniform_output_bits(
                output_size,
                ring_dim,
                prf_mask_output_coeff_bits,
                output_size,
            ),
        ));
        assert!(seed_bits >= minimum_noise_refresh_seed_bits(&params, noise_refresh_v_bits, cbd_n));

        let previous = seed_bits - 1;
        assert!(
            !goldreich_output_bound_holds(previous, branch_count * previous) ||
                !goldreich_output_bound_holds(
                    previous,
                    diamond_io_final_prg_uniform_output_bits(
                        output_size,
                        ring_dim,
                        prf_mask_output_coeff_bits,
                        output_size,
                    ),
                ) ||
                previous < minimum_noise_refresh_seed_bits(&params, noise_refresh_v_bits, cbd_n),
            "one smaller seed length must fail at least one DiamondIO PRG output bound"
        );
    }

    #[test]
    fn test_prf_final_round_separator_uses_digit_round_count() {
        let scheme = test_scheme_with_input_count(1, 2);
        assert_eq!(scheme.injector.batch_bits(), 1);

        let digit_scheme = {
            let params = DCRTPolyParams::new(2, 1, 10, 5);
            let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
            let ring_gsw_context = Arc::new(NestedRnsPolyContext::setup(
                &mut setup_circuit,
                &params,
                5,
                2,
                1 << 8,
                false,
                Some(1),
            ));
            let ring_gsw_width = 2 *
                <NestedRnsPolyContext as ModularArithmeticContext<DCRTPoly>>::gadget_len(
                    ring_gsw_context.as_ref(),
                    Some(1),
                    Some(0),
                );
            let injector = DiamondInjector::<
                DCRTPolyMatrix,
                DCRTPolyUniformSampler,
                DCRTPolyHashSampler<Keccak256>,
                DCRTPolyTrapdoorSampler,
            >::new(params.clone(), 2, 4, 2, 4.578, 0.0);
            TestDiamondIO::new(
                injector,
                4,
                6,
                params,
                ring_gsw_context,
                ring_gsw_width,
                0,
                Some(1),
                Some(0.0),
                b"diamond_io_final_round_separator_test".to_vec(),
                6,
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
        };
        assert_eq!(digit_scheme.input_size, 4);
        assert_eq!(digit_scheme.injector.input_count, 2);
        assert_ne!(digit_scheme.input_size, digit_scheme.injector.input_count);
        assert_eq!(digit_scheme.prf_final_round_idx(), digit_scheme.injector.input_count);
    }

    #[test]
    fn test_prf_ciphertext_decryption_noise_accumulates_across_seed_rounds() {
        let mut scheme = test_scheme_with_input_count(2, 2);
        scheme.ring_gsw_public_key_error_sigma = Some(1.0);
        let params = cpu_params_from_poly_params(&scheme.injector.params);
        let ctx = simulator_context(&params);
        let (initial_seed_randomizer, decomposed_randomizer) =
            sim_utils::seed_ciphertext_randomizer_norms(&params, scheme.cpu_ring_gsw_config());
        let (first_decryption_error, first_seed_randomizer) =
            sim_utils::prf_refresh_ciphertext_state(
                &initial_seed_randomizer,
                &decomposed_randomizer,
                ctx.clone(),
                scheme.full_active_levels(&params),
                scheme.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
            );
        let (second_decryption_error, second_seed_randomizer) =
            sim_utils::prf_refresh_ciphertext_state(
                &first_seed_randomizer,
                &decomposed_randomizer,
                ctx,
                scheme.full_active_levels(&params),
                scheme.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
            );

        assert!(
            second_seed_randomizer.poly_norm.norm > first_seed_randomizer.poly_norm.norm,
            "the refreshed next_seed Ring-GSW ciphertext must carry accumulated randomizer noise"
        );
        assert!(
            second_decryption_error.poly_norm.norm > first_decryption_error.poly_norm.norm,
            "PRF ciphertext-side decryption error must grow from one seed round to the next"
        );
    }

    #[test]
    fn test_selected_branch_rebase_matches_selected_prg_column_shape() {
        let ctx = simulator_context(&DCRTPolyParams::new(2, 2, 10, 5));
        let one = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(2u32), None),
        );
        let selected_prg = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(3u32), None),
        );
        let rebase_decoder = branch_rebase_decoder_error(
            ErrorNorm::new(
                PolyNorm::one(ctx.clone()),
                PolyMatrixNorm::new(ctx.clone(), 1, 1, BigDecimal::from(5u32), None),
            ),
            ctx.m_g,
        );
        let selected =
            simulate_selected_branch_rebase_error_norm(one, selected_prg, rebase_decoder);
        assert_eq!(
            selected.matrix_norm.ncol, ctx.m_g,
            "branch rebase decoder and mask error must use the selected PRG column shape"
        );
    }

    #[test]
    fn test_ciphertext_decryption_residual_stays_scalar_in_noise_refresh_pre_rounds() {
        let ctx = simulator_context(&DCRTPolyParams::new(2, 2, 10, 5));
        let residual = PolyMatrixNorm::new(ctx.clone(), 1, 1, BigDecimal::from(5u32), None);
        let mut refresh = NoiseRefreshErrorSimulation {
            v_bits: 1,
            pre_round_outputs: vec![PolyMatrixNorm::new(
                ctx.clone(),
                1,
                ctx.m_g,
                BigDecimal::from(7u32),
                None,
            )],
            rounded_errors: vec![PolyMatrixNorm::new(
                ctx.clone(),
                1,
                ctx.m_g,
                BigDecimal::from(11u32),
                None,
            )],
        };
        sim_utils::add_ciphertext_decryption_residual_to_noise_refresh_pre_rounds(
            &mut refresh,
            &residual,
        );
        assert_eq!(refresh.pre_round_outputs[0].ncol, ctx.m_g);
        assert_eq!(refresh.pre_round_outputs[0].poly_norm.norm, BigDecimal::from(12u32));
        assert_eq!(
            refresh.rounded_errors[0].poly_norm.norm,
            BigDecimal::from(11u32),
            "ciphertext decryption residual is a scalar pre-rounding perturbation, not refreshed seed BGG error"
        );
    }

    #[test]
    #[should_panic(expected = "selected PRG and rebase decoder error must share a shape")]
    fn test_selected_branch_rebase_rejects_unshaped_scalar_decoder() {
        let ctx = simulator_context(&DCRTPolyParams::new(2, 2, 10, 5));
        let one = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(2u32), None),
        );
        let selected_prg = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(3u32), None),
        );
        let scalar_rebase_decoder = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, 1, BigDecimal::from(5u32), None),
        );
        let _ =
            simulate_selected_branch_rebase_error_norm(one, selected_prg, scalar_rebase_decoder);
    }

    #[test]
    #[ignore = "full DiamondIO error simulation is heavier than default unit tests"]
    fn test_simulate_error_growth_reuses_input_injector_state_errors() {
        let mut scheme = test_scheme_with_input_count(2, 0);
        make_scheme_seed_bound_safe(&mut scheme);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let expected = scheme.injector.simulate_output_error_bounds().state_errors;
        let simulation = scheme
            .simulate_error_growth(
                DiamondIOFuncType::GoldreichPRF { output_bits: scheme.output_size },
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("DiamondIO error simulation should find a noise-refresh mask size");

        assert_eq!(simulation.input_injection.state_errors, expected);
        assert!(simulation.prf_refreshes.is_empty());
    }

    #[test]
    #[ignore = "full noise-refresh path is intentionally heavier than default unit tests"]
    fn test_simulate_error_growth_records_one_refresh_per_input_digit() {
        let mut scheme = test_scheme(2);
        make_scheme_seed_bound_safe(&mut scheme);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let simulation = scheme
            .simulate_error_growth(
                DiamondIOFuncType::GoldreichPRF { output_bits: scheme.output_size },
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("DiamondIO error simulation should find a noise-refresh mask size");

        assert_eq!(simulation.prf_refreshes.len(), scheme.injector.input_count);
    }

    #[test]
    #[ignore = "fixed-v DiamondIO prefix simulation is heavier than default unit tests"]
    fn test_fixed_v_prefix_uses_selected_v_bits_for_single_round() {
        let mut scheme = test_scheme_with_input_count(2, 1);
        make_scheme_seed_bound_safe(&mut scheme);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let prefix = scheme
            .simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
                DiamondIOFuncType::GoldreichPRF { output_bits: scheme.output_size },
                scheme.noise_refresh_v_bits,
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("fixed-v prefix should build with the fixture seed size");

        assert_eq!(prefix.prf_refreshes.len(), 1);
        assert_eq!(prefix.prf_refreshes[0].round_idx, 0);
        assert_eq!(prefix.prf_refreshes[0].noise_refresh.v_bits, scheme.noise_refresh_v_bits);
    }

    #[test]
    #[ignore = "fixed-v DiamondIO prefix simulation is heavier than default unit tests"]
    fn test_fixed_v_prefix_reuses_steady_state_without_researching_v_bits() {
        let mut scheme = test_scheme_with_input_count(2, 3);
        make_scheme_seed_bound_safe(&mut scheme);
        scheme.ring_gsw_public_key_error_sigma = Some(1.0);
        let ctx = simulator_context(&scheme.injector.params);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(0.0).unwrap());
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();

        let prefix = scheme
            .simulate_mask_independent_error_prefix_fixed_noise_refresh_v_bits(
                DiamondIOFuncType::GoldreichPRF { output_bits: scheme.output_size },
                scheme.noise_refresh_v_bits,
                &plt_evaluator,
                &slot_transfer_evaluator,
            )
            .expect("fixed-v prefix should build with the fixture seed size");

        assert_eq!(prefix.prf_refreshes.len(), scheme.injector.input_count);
        assert_eq!(
            prefix.prf_refreshes.iter().map(|refresh| refresh.round_idx).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(
            prefix
                .prf_refreshes
                .iter()
                .all(|refresh| refresh.noise_refresh.v_bits == scheme.noise_refresh_v_bits)
        );
        assert!(
            prefix.prf_refreshes[2]
                .representative_selected_prg_ciphertext_decryption_error
                .poly_norm
                .norm >
                prefix.prf_refreshes[1]
                    .representative_selected_prg_ciphertext_decryption_error
                    .poly_norm
                    .norm,
            "ciphertext-side FHE noise must keep accumulating across PRF seed rounds"
        );
        assert!(
            prefix.prf_refreshes[2].noise_refresh.pre_round_outputs[0].poly_norm.norm >
                prefix.prf_refreshes[1].noise_refresh.pre_round_outputs[0].poly_norm.norm,
            "noise-refresh pre-round error must grow when the underlying seed ciphertext decryption residual grows"
        );
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
