#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BenchEstimator, BggEncodingBenchEstimator, BggPublicKeyBenchEstimator,
        BggPublicKeyBenchSamples, NaiveBGGVecBenchEstimator, PublicLutSampleAuxBenchEstimator,
        SlotTransferSampleAuxBenchEstimator,
    },
    bgg::{
        naive_vec::{
            NaiveBGGEncodingVec, NaiveBGGEncodingVecSampler, NaiveBGGPublicKeyVec,
            NaiveBGGPublicKeyVecSampler,
        },
        public_key::BggPublicKey,
        sampler::BGGPublicKeySampler,
    },
    circuit::{PolyCircuit, PolyGateKind, evaluable::PolyVec, gate::GateId},
    element::PolyElem,
    gadgets::{
        arith::{MontgomeryPoly, MontgomeryPolyContext},
        fhe::{
            ring_gsw::{
                RingGswCiphertext as GenericRingGswCiphertext,
                RingGswContext as GenericRingGswContext,
            },
            ring_gsw_montgomery_gpu::{
                encrypt_montgomery_plaintext_bit, montgomery_ciphertext_inputs_from_native,
                sample_montgomery_public_key, sample_montgomery_secret_key,
            },
        },
    },
    lookup::{
        PublicLut,
        lwe::{
            LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
            NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
        },
        poly::PolyPltEvaluator,
        poly_vec::PolyVecPltEvaluator,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{
                GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_count, detected_gpu_device_ids,
                gpu_device_sync,
            },
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{
        SimulatorContext,
        error_norm::{NormNaiveBggEncodingVecSTEvaluator, NormPltLWEEvaluator},
    },
    slot_transfer::{
        NaiveBGGVecSlotTransferEvaluator, PolyVecSlotTransferEvaluator,
        bgg_pubkey::BggPublicKeySTEvaluator,
    },
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::{debug, info};

const DEFAULT_RING_DIM: u32 = 1 << 16;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_LIMB_BITS: usize = 7;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_BENCH_SEED: [u8; 32] = [0u8; 32];
const TRAPDOOR_SIGMA: f64 = 4.578;
const ERROR_SIM_ACTIVE_LEVELS: usize = 1;

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type GpuPubKeyPltEvaluator =
    NaiveLWEBGGPublicKeyVecPltEvaluator<GpuMatrix, GpuHashSampler, GpuDCRTPolyTrapdoorSampler>;
type GpuScalarPubKeyPltEvaluator =
    LWEBGGPubKeyPltEvaluator<GpuMatrix, GpuHashSampler, GpuDCRTPolyTrapdoorSampler>;
type GpuPubKeySlotEvaluator = BggPublicKeySTEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuEncodingPltEvaluator = NaiveLWEBGGEncodingVecPltEvaluator<GpuMatrix, GpuHashSampler>;
type RingGswCiphertext<P> = GenericRingGswCiphertext<P, MontgomeryPoly<P>>;
type RingGswContext<P> = GenericRingGswContext<P, MontgomeryPoly<P>>;

#[derive(Debug, Clone)]
struct RingGswMulBenchConfig {
    ring_dim: u32,
    crt_bits: usize,
    limb_bits: usize,
    base_bits: u32,
    max_crt_depth: usize,
    error_sigma: f64,
    d_secret: usize,
    bench_iterations: usize,
    active_levels_override: Option<usize>,
    dir_name_override: Option<String>,
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be a valid u32: {e}")),
        Err(_) => default,
    }
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    match env::var(key) {
        Ok(raw) => {
            raw.parse::<usize>().unwrap_or_else(|e| panic!("{key} must be a valid usize: {e}"))
        }
        Err(_) => default,
    }
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    match env::var(key) {
        Ok(raw) => raw.parse::<f64>().unwrap_or_else(|e| panic!("{key} must be a valid f64: {e}")),
        Err(_) => default,
    }
}

impl RingGswMulBenchConfig {
    fn from_env() -> Self {
        let ring_dim =
            env_or_parse_u32("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_RING_DIM", DEFAULT_RING_DIM);
        let crt_bits =
            env_or_parse_usize("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_CRT_BITS", DEFAULT_CRT_BITS);
        let limb_bits =
            env_or_parse_usize("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_LIMB_BITS", DEFAULT_LIMB_BITS);
        let base_bits =
            env_or_parse_u32("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth = env_or_parse_usize(
            "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MAX_CRT_DEPTH",
            DEFAULT_MAX_CRT_DEPTH,
        );
        let error_sigma =
            env_or_parse_f64("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret =
            env_or_parse_usize("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_D_SECRET", DEFAULT_D_SECRET);
        let bench_iterations = env_or_parse_usize(
            "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_BENCH_ITERATIONS",
            DEFAULT_BENCH_ITERATIONS,
        );
        let active_levels_override = env::var("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_ACTIVE_LEVELS")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(|value| {
                value.parse::<usize>().unwrap_or_else(|e| {
                    panic!(
                        "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_ACTIVE_LEVELS must be a valid usize: {e}"
                    )
                })
            });
        let dir_name_override = env::var("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_DIR_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        assert!(ring_dim > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_RING_DIM must be > 0");
        assert!(crt_bits > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_CRT_BITS must be > 0");
        assert!(limb_bits > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_LIMB_BITS must be > 0");
        assert!(base_bits > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma >= 0.0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_ERROR_SIGMA must be >= 0");
        assert!(d_secret > 0, "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_D_SECRET must be > 0");
        assert!(
            bench_iterations > 0,
            "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_BENCH_ITERATIONS must be > 0"
        );

        Self {
            ring_dim,
            crt_bits,
            limb_bits,
            base_bits,
            max_crt_depth,
            error_sigma,
            d_secret,
            bench_iterations,
            active_levels_override,
            dir_name_override,
        }
    }

    fn num_slots(&self) -> usize {
        self.ring_dim as usize
    }

    fn bench_dir_base(&self, crt_depth: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!(
                "test_data/test_gpu_lwe_montgomery_ring_gsw_mul_bench_ring{}_active{}_crt{}_depth{}",
                self.ring_dim, crt_depth, self.crt_bits, crt_depth
            )
        })
    }

    fn active_levels_for_crt_depth(&self, crt_depth: usize) -> usize {
        let active_levels = self.active_levels_override.unwrap_or(crt_depth);
        assert!(
            active_levels <= crt_depth,
            "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_ACTIVE_LEVELS exceeds crt_depth: active_levels={}, crt_depth={}",
            active_levels,
            crt_depth
        );
        active_levels
    }
}

fn ensure_dir_exists(dir: &Path) {
    if !dir.exists() {
        fs::create_dir_all(dir).expect("failed to create benchmark directory");
    }
}

fn active_q_moduli_and_modulus<T: PolyParams>(
    params: &T,
    active_levels: usize,
) -> (Vec<u64>, BigUint, usize) {
    let (q_moduli, _, crt_depth) = params.to_crt();
    assert!(
        active_levels <= crt_depth,
        "active_levels exceeds CRT depth: active_levels={}, crt_depth={}",
        active_levels,
        crt_depth
    );
    let active_q_moduli = q_moduli.into_iter().take(active_levels).collect::<Vec<_>>();
    let active_q =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    (active_q_moduli, active_q, crt_depth)
}

fn ring_gsw_q_modulus<P: Poly + 'static>(ctx: &RingGswContext<P>) -> BigUint {
    let (q_moduli, _, _) = ctx.params.to_crt();
    q_moduli
        .iter()
        .skip(ctx.level_offset)
        .take(ctx.active_levels)
        .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
}

fn corrected_max_eval_error_for_active_levels(
    simulated_error: &BigDecimal,
    requested_active_levels: usize,
) -> BigDecimal {
    simulated_error * BigDecimal::from_biguint(BigUint::from(requested_active_levels), 0)
}

fn max_bigdecimal<I>(values: I) -> BigDecimal
where
    I: IntoIterator<Item = BigDecimal>,
{
    let mut iter = values.into_iter();
    let mut max = iter.next().expect("max_bigdecimal requires at least one value");
    for value in iter {
        if value > max {
            max = value;
        }
    }
    max
}

struct CrtDepthProbe {
    params: DCRTPolyParams,
    eval_ok: bool,
}

impl CrtDepthProbe {
    fn search_ok(&self) -> bool {
        self.eval_ok
    }
}

fn build_ring_gsw_mul_circuit<P: Poly + 'static>(
    params: &P::Params,
    cfg: &RingGswMulBenchConfig,
    active_levels: usize,
) -> (PolyCircuit<P>, Arc<RingGswContext<P>>, Vec<RingGswCiphertext<P>>) {
    let build_start = Instant::now();
    let mut circuit = PolyCircuit::<P>::new();
    let montgomery =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bits, false));
    let ctx = Arc::new(RingGswContext::from_arith_context(
        &mut circuit,
        params,
        cfg.num_slots(),
        montgomery,
        Some(active_levels),
        None,
    ));
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let product = lhs.mul(&rhs, &mut circuit);
    let reconstructed_outputs = product.reconstruct(&mut circuit);
    circuit.output(reconstructed_outputs);
    info!(
        "ring_gsw mul circuit build elapsed_ms={:.3} encrypted_outputs=1",
        build_start.elapsed().as_secs_f64() * 1000.0
    );
    (circuit, ctx, vec![product])
}

fn build_ring_gsw_mul_probe_circuit<P: Poly + 'static>(
    params: &P::Params,
    cfg: &RingGswMulBenchConfig,
    active_levels: usize,
) -> (PolyCircuit<P>, Arc<RingGswContext<P>>) {
    let build_start = Instant::now();
    let mut circuit = PolyCircuit::<P>::new();
    let montgomery =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bits, false));
    let ctx = Arc::new(RingGswContext::from_arith_context(
        &mut circuit,
        params,
        cfg.num_slots(),
        montgomery,
        Some(active_levels),
        None,
    ));
    let lhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let rhs = RingGswCiphertext::input(ctx.clone(), None, &mut circuit);
    let product = lhs.mul(&rhs, &mut circuit);
    let reconstructed_outputs = product.reconstruct(&mut circuit);
    circuit.output(reconstructed_outputs);
    info!(
        "ring_gsw mul circuit build elapsed_ms={:.3} encrypted_outputs=1",
        build_start.elapsed().as_secs_f64() * 1000.0
    );
    (circuit, ctx)
}

fn probe_crt_depth_for_ring_gsw_mul_bench(
    cfg: &RingGswMulBenchConfig,
    crt_depth: usize,
    ring_dim_sqrt: &BigDecimal,
    base: &BigDecimal,
    error_sigma: &BigDecimal,
    e_init_norm: &BigDecimal,
) -> CrtDepthProbe {
    let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (_, _, actual_crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(actual_crt_depth);
    assert!(
        ERROR_SIM_ACTIVE_LEVELS <= actual_crt_depth,
        "error simulation requires crt_depth >= ERROR_SIM_ACTIVE_LEVELS: crt_depth={}, ERROR_SIM_ACTIVE_LEVELS={}",
        actual_crt_depth,
        ERROR_SIM_ACTIVE_LEVELS
    );
    let (active_q_moduli, active_q, _) = active_q_moduli_and_modulus(&params, active_levels);
    let full_q = params.modulus();
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let (circuit, ctx) =
        build_ring_gsw_mul_probe_circuit::<DCRTPoly>(&params, cfg, ERROR_SIM_ACTIVE_LEVELS);
    let log_base_q = params.modulus_digits();
    let log_base_q_small = log_base_q / actual_crt_depth;
    let sim_ctx = Arc::new(SimulatorContext::new(
        ring_dim_sqrt.clone(),
        base.clone(),
        cfg.d_secret,
        log_base_q,
        log_base_q_small,
    ));
    let plt_evaluator = NormPltLWEEvaluator::new(sim_ctx.clone(), error_sigma);
    let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();
    let out_errors = circuit.simulate_max_error_norm(
        sim_ctx,
        BigDecimal::from(1u64),
        circuit.num_input(),
        e_init_norm,
        Some(&plt_evaluator),
        Some(&slot_transfer_evaluator),
    );
    debug!("ring_gsw_mul simulate_max_error_norm finished output_errors={}", out_errors.len());
    let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
    let threshold_bd = BigDecimal::from_biguint(threshold.clone(), 0);
    let max_eval_error = max_bigdecimal(
        out_errors
            .par_iter()
            .map(|error| error.matrix_norm.poly_norm.norm.clone())
            .collect::<Vec<_>>(),
    );
    let corrected_max_eval_error =
        corrected_max_eval_error_for_active_levels(&max_eval_error, active_levels);
    let ring_gsw_q = ring_gsw_q_modulus(ctx.as_ref());
    let eval_ok = corrected_max_eval_error < threshold_bd;
    let non_free_depth = circuit.non_free_depth();
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "ring_gsw_mul crt_depth={} active_levels={} error_sim_active_levels={} active_q_bits={} full_q_bits={} ring_gsw_q_bits={} non_free_depth={} non_free_depth_contributions={:?} num_inputs={} output_polys={} sim_max_eval_error_bits={} max_eval_error_bits={} eval_threshold_bits={} eval_ok={}",
        crt_depth,
        active_levels,
        ERROR_SIM_ACTIVE_LEVELS,
        active_q.bits(),
        full_q.bits(),
        ring_gsw_q.bits(),
        non_free_depth,
        non_free_depth_contributions,
        circuit.num_input(),
        out_errors.len(),
        bigdecimal_bits_ceil(&max_eval_error),
        bigdecimal_bits_ceil(&corrected_max_eval_error),
        bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold, 0)),
        eval_ok
    );

    CrtDepthProbe { params, eval_ok }
}

fn find_crt_depth_for_ring_gsw_mul_bench(cfg: &RingGswMulBenchConfig) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let e_init_norm = &error_sigma * BigDecimal::from_f32(6.5).unwrap();

    if let Some(raw_crt_depth) = env::var("CRT_DEPTH").ok().filter(|value| !value.trim().is_empty())
    {
        let requested_crt_depth = raw_crt_depth
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("CRT_DEPTH must be a valid usize: {e}"));
        assert!(requested_crt_depth > 0, "CRT_DEPTH must be > 0");
        let params =
            DCRTPolyParams::new(cfg.ring_dim, requested_crt_depth, cfg.crt_bits, cfg.base_bits);
        return (requested_crt_depth, params);
    }

    let min_crt_depth =
        env_or_parse_usize("LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MIN_CRT_DEPTH", 1usize);
    assert!(
        min_crt_depth <= cfg.max_crt_depth,
        "LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MIN_CRT_DEPTH must be <= LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MAX_CRT_DEPTH"
    );
    let mut low = min_crt_depth;
    let mut high = cfg.max_crt_depth;
    while low < high {
        let mid = low + (high - low) / 2;
        let probe = probe_crt_depth_for_ring_gsw_mul_bench(
            cfg,
            mid,
            &ring_dim_sqrt,
            &base,
            &error_sigma,
            &e_init_norm,
        );
        if probe.search_ok() {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    let probe = probe_crt_depth_for_ring_gsw_mul_bench(
        cfg,
        low,
        &ring_dim_sqrt,
        &base,
        &error_sigma,
        &e_init_norm,
    );
    if probe.search_ok() {
        info!("selected crt_depth={} for Montgomery Ring-GSW multiplication benchmark search", low);
        return (low, probe.params);
    }

    panic!(
        "crt_depth satisfying the LWE error threshold was not found up to LWE_MONTGOMERY_RING_GSW_MUL_BENCH_MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
}

fn build_lwe_pubkey_vec_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    d_secret: usize,
    dir_path: PathBuf,
) -> GpuPubKeyPltEvaluator {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
    GpuPubKeyPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
        GpuMatrix,
        GpuHashSampler,
        GpuDCRTPolyTrapdoorSampler,
    >::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    ))
}

fn build_lwe_scalar_pubkey_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    d_secret: usize,
    dir_path: PathBuf,
) -> GpuScalarPubKeyPltEvaluator {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
    GpuScalarPubKeyPltEvaluator::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    )
}

fn build_benchmark_public_lut(params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
    PublicLut::<GpuDCRTPoly>::new(
        params,
        16,
        move |params, x| {
            if x >= 16 {
                return None;
            }
            let y_elem = <<GpuDCRTPoly as Poly>::Elem as PolyElem>::constant(
                &params.modulus(),
                (x + 1) % 16,
            );
            Some((x, y_elem))
        },
        None,
    )
}

fn build_benchmark_public_lookup_gate(
    params: &GpuDCRTPolyParams,
) -> (PublicLut<GpuDCRTPoly>, usize, GateId) {
    let lut = build_benchmark_public_lut(params);
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let input = circuit.input(1);
    let lut_id = circuit.register_public_lookup(lut.clone());
    let gate_id = circuit.public_lookup_gate(input, lut_id).as_single_wire();
    (lut, lut_id, gate_id)
}

fn build_benchmark_slot_transfer_gate(src_slots: &[(u32, Option<u32>)]) -> GateId {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let input = circuit.input(1);
    circuit.slot_transfer_gate(input, src_slots).as_single_wire()
}

fn identity_slot_transfer_plan(num_slots: usize) -> Vec<(u32, Option<u32>)> {
    (0..num_slots).map(|slot| (slot as u32, None)).collect()
}

fn sample_benchmark_pubkeys(
    params: &GpuDCRTPolyParams,
    seed: [u8; 32],
    d_secret: usize,
    non_constant_key_count: usize,
    tag: &[u8],
) -> Vec<BggPublicKey<GpuMatrix>> {
    let sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; non_constant_key_count];
    sampler.sample(params, tag, &reveal_plaintexts)
}

fn constant_and_shared_benchmark_pubkeys(
    pubkeys: &[BggPublicKey<GpuMatrix>],
) -> (&BggPublicKey<GpuMatrix>, &BggPublicKey<GpuMatrix>) {
    assert!(
        pubkeys.len() >= 2,
        "benchmark pubkeys must contain the constant-one key and at least one reusable input key"
    );
    (&pubkeys[0], &pubkeys[1])
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor / BigUint::from(2u64);
    (value + half) / divisor
}

fn verify_naive_encoding_outputs(
    params: &GpuDCRTPolyParams,
    active_q_moduli: &[u64],
    full_q: &BigUint,
    secret_vec: &GpuDCRTPolyMatrix,
    pubkey_outputs: &[NaiveBGGPublicKeyVec<GpuMatrix>],
    encoding_outputs: &[NaiveBGGEncodingVec<GpuMatrix>],
    expected_outputs: &[PolyVec<GpuDCRTPoly>],
) {
    assert_eq!(encoding_outputs.len(), pubkey_outputs.len());
    assert_eq!(encoding_outputs.len(), expected_outputs.len());
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let q_over_qmax = full_q / BigUint::from(q_max);
    let gadget = GpuDCRTPolyMatrix::gadget_matrix(params, secret_vec.col_size());

    for ((encoding_vec, pubkey_vec), expected_vec) in
        encoding_outputs.iter().zip(pubkey_outputs.iter()).zip(expected_outputs.iter())
    {
        assert_eq!(encoding_vec.num_slots(), pubkey_vec.num_slots());
        assert_eq!(encoding_vec.num_slots(), expected_vec.len());
        for slot in 0..encoding_vec.num_slots() {
            let encoding = encoding_vec.encoding(slot);
            assert_eq!(encoding.pubkey, pubkey_vec.key(slot));
            let expected = &expected_vec.as_slice()[slot];
            assert_eq!(
                encoding.plaintext.as_ref().expect("naive BGG output plaintext should be revealed"),
                expected
            );

            let expected_times_gadget = secret_vec.clone() * (gadget.clone() * expected.clone());
            let s_times_pk = secret_vec.clone() * &encoding.pubkey.matrix;
            let diff = encoding.vector.clone() - s_times_pk + expected_times_gadget;
            let coeff = diff
                .entry(0, 0)
                .coeffs_biguints()
                .into_iter()
                .next()
                .expect("diff poly must have at least one coefficient");
            let random_int: u64 = rand::random::<u64>() % q_max;
            let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
            let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
            let decoded_random: u64 = (&rounded % BigUint::from(q_max))
                .try_into()
                .expect("decoded random coefficient must fit u64");
            assert_eq!(decoded_random, random_int);
        }
    }
}

#[tokio::test]
async fn test_gpu_lwe_montgomery_ring_gsw_mul_bench() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    gpu_device_sync();

    let cfg = RingGswMulBenchConfig::from_env();
    info!("montgomery ring_gsw mul bench config: {:?}", cfg);

    let (crt_depth, cpu_params) = find_crt_depth_for_ring_gsw_mul_bench(&cfg);
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_ids = detected_gpu_device_ids();
    let detected_gpu_count = detected_gpu_device_count();
    assert_eq!(
        detected_gpu_count,
        detected_gpu_ids.len(),
        "detected GPU count and ids length must match"
    );
    let single_gpu_id = *detected_gpu_ids.first().expect(
        "at least one GPU device is required for test_gpu_lwe_montgomery_ring_gsw_mul_bench",
    );
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    assert_eq!(params.modulus(), cpu_params.modulus());

    let (all_q_moduli, _, actual_crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(actual_crt_depth);
    let (active_q_moduli, active_q, _) = active_q_moduli_and_modulus(&params, active_levels);
    let (circuit, ctx, encrypted_outputs) =
        build_ring_gsw_mul_circuit::<GpuDCRTPoly>(&params, &cfg, active_levels);
    let max_selected_decryption_error = max_bigdecimal(
        encrypted_outputs
            .iter()
            .map(|ciphertext| {
                ciphertext.estimate_decryption_error_norm(cfg.error_sigma).poly_norm.norm
            })
            .collect::<Vec<_>>(),
    );
    let ring_gsw_q = ring_gsw_q_modulus(ctx.as_ref());
    let ring_gsw_threshold = &ring_gsw_q / BigUint::from(2u64);
    let ring_gsw_threshold_bd = BigDecimal::from_biguint(ring_gsw_threshold.clone(), 0);
    let selected_decryption_ok = max_selected_decryption_error < ring_gsw_threshold_bd;
    let gate_counts = circuit.count_gates_by_type_vec();
    let total_lut_entries = circuit.total_registered_public_lut_entries();
    let total_public_lut_gates = gate_counts.get(&PolyGateKind::PubLut).copied().unwrap_or(0);
    let total_slot_transfer_gates =
        gate_counts.get(&PolyGateKind::SlotTransfer).copied().unwrap_or(0);
    info!(
        "forcing single GPU for ring_gsw mul benchmark path: eval_gpu_id={} detected_gpu_count={} detected_gpu_ids={:?}",
        single_gpu_id, detected_gpu_count, detected_gpu_ids
    );
    info!(
        "ring_gsw_mul selected crt_depth={} actual_crt_depth={} ring_dim={} num_slots={} active_levels={} crt_bits={} limb_bits={} base_bits={} q_moduli={:?}",
        crt_depth,
        actual_crt_depth,
        params.ring_dimension(),
        cfg.num_slots(),
        active_levels,
        cfg.crt_bits,
        cfg.limb_bits,
        cfg.base_bits,
        all_q_moduli
    );
    let non_free_depth = circuit.non_free_depth();
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "ring_gsw_mul circuit non_free_depth={} non_free_depth_contributions={:?} gate_counts={:?} num_inputs={} encrypted_outputs={} reconstructed_output_polys={}",
        non_free_depth,
        non_free_depth_contributions,
        gate_counts,
        circuit.num_input(),
        encrypted_outputs.len(),
        encrypted_outputs.len() * 2 * ctx.width()
    );
    info!(
        "ring_gsw_mul active_q_bits={} active_q_moduli_len={} ring_gsw_q_bits={} selected_max_decryption_error_bits={} selected_decryption_threshold_bits={} selected_decryption_ok={}",
        active_q.bits(),
        active_q_moduli.len(),
        ring_gsw_q.bits(),
        bigdecimal_bits_ceil(&max_selected_decryption_error),
        bigdecimal_bits_ceil(&BigDecimal::from_biguint(ring_gsw_threshold, 0)),
        selected_decryption_ok
    );

    let mul_bench_dir = Path::new(&cfg.bench_dir_base(crt_depth)).join("ring_gsw_mul");
    let pubkey_bench_dir = mul_bench_dir.join("pubkey");
    ensure_dir_exists(&pubkey_bench_dir);
    init_storage_system(pubkey_bench_dir.clone());
    let pubkey_bench_plt_evaluator = build_lwe_pubkey_vec_plt_evaluator(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        pubkey_bench_dir.clone(),
    );
    let pubkey_bench_slot_evaluator = NaiveBGGVecSlotTransferEvaluator::new();
    let pubkey_public_lut_aux_estimate = pubkey_bench_plt_evaluator
        .estimate_public_lut_sample_aux_matrices(
            &params,
            total_lut_entries,
            total_public_lut_gates,
        );
    gpu_device_sync();
    let pubkey_slot_transfer_aux_slot_time =
        <NaiveBGGVecSlotTransferEvaluator as SlotTransferSampleAuxBenchEstimator<
            GpuMatrix,
        >>::sample_aux_matrices_slot_time(&pubkey_bench_slot_evaluator, &params);
    let pubkey_slot_transfer_aux_gate_time =
        <NaiveBGGVecSlotTransferEvaluator as SlotTransferSampleAuxBenchEstimator<
            GpuMatrix,
        >>::sample_aux_matrices_gate_time(&pubkey_bench_slot_evaluator, &params);
    info!(
        "ring_gsw_mul naive pubkey vec sample_aux estimates: public_lut={:?} slot_transfer_slot_time={:?} slot_transfer_gate_time={:?} slot_transfer_gates={}",
        pubkey_public_lut_aux_estimate,
        pubkey_slot_transfer_aux_slot_time,
        pubkey_slot_transfer_aux_gate_time,
        total_slot_transfer_gates
    );
    gpu_device_sync();

    let (pubkey_bench_lut, pubkey_bench_public_lut_id, pubkey_bench_public_lut_gate_id) =
        build_benchmark_public_lookup_gate(&params);
    let pubkey_bench_slot_transfer_src_slots = identity_slot_transfer_plan(cfg.num_slots());
    let pubkey_bench_slot_transfer_gate_id =
        build_benchmark_slot_transfer_gate(&pubkey_bench_slot_transfer_src_slots);
    let pubkey_bench_public_keys = sample_benchmark_pubkeys(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        1,
        b"LWE_MONTGOMERY_RING_GSW_MUL_PUBKEY_BENCH",
    );
    let (pubkey_bench_public_lut_one, pubkey_bench_shared_input) =
        constant_and_shared_benchmark_pubkeys(&pubkey_bench_public_keys);
    let pubkey_scalar_bench_plt_evaluator = build_lwe_scalar_pubkey_plt_evaluator(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        pubkey_bench_dir.clone(),
    );
    let pubkey_scalar_bench_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        cfg.num_slots(),
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        pubkey_bench_dir.clone(),
    );
    let pubkey_scalar_bench = BggPublicKeyBenchEstimator::benchmark(
        &BggPublicKeyBenchSamples {
            params: &params,
            add_lhs: pubkey_bench_shared_input,
            add_rhs: pubkey_bench_shared_input,
            sub_lhs: pubkey_bench_shared_input,
            sub_rhs: pubkey_bench_shared_input,
            mul_lhs: pubkey_bench_shared_input,
            mul_rhs: pubkey_bench_shared_input,
            small_scalar_input: pubkey_bench_shared_input,
            small_scalar: &[3u32, 5u32],
            large_scalar_input: pubkey_bench_shared_input,
            large_scalar: &[BigUint::from(7u32)],
            public_lut_one: pubkey_bench_public_lut_one,
            public_lut_input: pubkey_bench_shared_input,
            public_lut: &pubkey_bench_lut,
            public_lut_id: pubkey_bench_public_lut_id,
            public_lut_gate_id: pubkey_bench_public_lut_gate_id,
            slot_transfer_input: pubkey_bench_shared_input,
            slot_transfer_src_slots: &pubkey_bench_slot_transfer_src_slots,
            slot_transfer_gate_id: pubkey_bench_slot_transfer_gate_id,
        },
        &pubkey_scalar_bench_plt_evaluator,
        &pubkey_scalar_bench_slot_evaluator,
        build_lwe_pubkey_vec_plt_evaluator(
            &params,
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            pubkey_bench_dir.clone(),
        ),
        GpuPubKeySlotEvaluator::new(
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            cfg.num_slots(),
            TRAPDOOR_SIGMA,
            cfg.error_sigma,
            pubkey_bench_dir.clone(),
        ),
        cfg.bench_iterations,
    );
    let pubkey_vec_bench_estimator =
        NaiveBGGVecBenchEstimator::new(pubkey_scalar_bench, cfg.num_slots());
    let pubkey_circuit_bench = pubkey_vec_bench_estimator.estimate_circuit_bench(&circuit);
    info!(
        "ring_gsw_mul naive bgg pubkey vec circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        pubkey_circuit_bench.total_time,
        pubkey_circuit_bench.latency,
        pubkey_circuit_bench.max_parallelism,
        pubkey_circuit_bench.peak_vram
    );

    let encoding_scalar_bench =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(&params, cfg.bench_iterations);
    let encoding_vec_bench_estimator =
        NaiveBGGVecBenchEstimator::new(encoding_scalar_bench, cfg.num_slots());
    let poly_circuit_bench = encoding_vec_bench_estimator.estimate_circuit_bench(&circuit);
    info!(
        "ring_gsw_mul naive bgg encoding vec circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        poly_circuit_bench.total_time,
        poly_circuit_bench.latency,
        poly_circuit_bench.max_parallelism,
        poly_circuit_bench.peak_vram
    );

    drop(pubkey_bench_plt_evaluator);
    drop(pubkey_bench_slot_evaluator);
    drop(encrypted_outputs);
    gpu_device_sync();

    if cfg.ring_dim <= (1 << 10) {
        let eval_dir = mul_bench_dir.join("actual_naive_vec");
        if eval_dir.exists() {
            fs::remove_dir_all(&eval_dir).expect("failed to clear actual naive vector eval dir");
        }
        fs::create_dir_all(&eval_dir).expect("failed to create actual naive vector eval dir");
        init_storage_system(eval_dir.clone());

        let lhs_bit = rand::random::<bool>();
        let rhs_bit = rand::random::<bool>();
        let ring_gsw_secret_key = sample_montgomery_secret_key(&params);
        let ring_gsw_public_key = sample_montgomery_public_key(
            &params,
            ctx.width(),
            &ring_gsw_secret_key,
            DEFAULT_BENCH_SEED,
            b"LWE_MONTGOMERY_RING_GSW_MUL_ACTUAL_RING_GSW_PK",
            Some(cfg.error_sigma),
        );
        let plaintext_inputs = [
            montgomery_ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &encrypt_montgomery_plaintext_bit(
                    &params,
                    ctx.as_ref(),
                    &ring_gsw_public_key,
                    lhs_bit,
                ),
            ),
            montgomery_ciphertext_inputs_from_native(
                &params,
                ctx.as_ref(),
                &encrypt_montgomery_plaintext_bit(
                    &params,
                    ctx.as_ref(),
                    &ring_gsw_public_key,
                    rhs_bit,
                ),
            ),
        ]
        .concat();
        assert_eq!(
            plaintext_inputs.len(),
            circuit.num_input(),
            "small-ring plaintext input count must match the Ring-GSW multiplication circuit input count"
        );

        let poly_vec_plt_evaluator = PolyVecPltEvaluator { plt_evaluator: PolyPltEvaluator::new() };
        let poly_vec_slot_evaluator = PolyVecSlotTransferEvaluator::new();
        let expected_outputs = circuit.eval::<PolyVec<GpuDCRTPoly>, PolyVecPltEvaluator>(
            &params,
            PolyVec::new(vec![GpuDCRTPoly::const_one(&params); cfg.num_slots()]),
            plaintext_inputs.clone(),
            Some(&poly_vec_plt_evaluator),
            Some(&poly_vec_slot_evaluator),
            None,
        );

        let pubkey_sampler = NaiveBGGPublicKeyVecSampler::<_, GpuHashSampler>::new(
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            cfg.num_slots(),
        );
        let reveal_plaintexts = vec![true; circuit.num_input()];
        let pubkeys = pubkey_sampler.sample(
            &params,
            b"LWE_MONTGOMERY_RING_GSW_MUL_ACTUAL_PUBKEY",
            &reveal_plaintexts,
        );
        let uniform_sampler = GpuDCRTPolyUniformSampler::new();
        let secrets = uniform_sampler
            .sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist)
            .get_row(0);
        let secret_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
        let encoding_sampler = NaiveBGGEncodingVecSampler::<GpuDCRTPolyUniformSampler>::new(
            &params,
            &secrets,
            Some(cfg.error_sigma),
            cfg.num_slots(),
        );
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);

        let pubkey_evaluator = build_lwe_pubkey_vec_plt_evaluator(
            &params,
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            eval_dir.clone(),
        );
        let pubkey_eval_matrix = Arc::clone(&pubkey_evaluator.inner.pub_matrix);
        let naive_slot_evaluator = NaiveBGGVecSlotTransferEvaluator::new();
        let pubkey_outputs = circuit.eval(
            &params,
            pubkeys[0].clone(),
            pubkeys[1..].to_vec(),
            Some(&pubkey_evaluator),
            Some(&naive_slot_evaluator),
            None,
        );
        pubkey_evaluator.sample_aux_matrices(&params);
        wait_for_all_writes(eval_dir.clone())
            .await
            .expect("actual naive vector pubkey auxiliary writes should complete");

        let c_b = secret_vec.clone() * pubkey_eval_matrix.as_ref();
        let encoding_evaluator =
            GpuEncodingPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
                GpuDCRTPolyMatrix,
                GpuHashSampler,
            >::new(
                DEFAULT_BENCH_SEED, eval_dir.clone(), c_b
            ));
        let encoding_outputs = circuit.eval(
            &params,
            encodings[0].clone(),
            encodings[1..].to_vec(),
            Some(&encoding_evaluator),
            Some(&naive_slot_evaluator),
            None,
        );
        let full_q = params.modulus();
        verify_naive_encoding_outputs(
            &params,
            &active_q_moduli,
            full_q.as_ref(),
            &secret_vec,
            &pubkey_outputs,
            &encoding_outputs,
            &expected_outputs,
        );
        gpu_device_sync();
    }

    drop(ctx);
    gpu_device_sync();
}
