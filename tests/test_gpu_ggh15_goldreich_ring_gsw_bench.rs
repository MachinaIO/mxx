#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BenchEstimator, BggPolyEncodingBenchEstimator, BggPolyEncodingBenchSamples,
        BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
    },
    bgg::{
        public_key::BggPublicKey,
        sampler::{BGGPolyEncodingSampler, BGGPublicKeySampler},
    },
    circuit::{PolyCircuit, PolyGateKind, gate::GateId},
    element::PolyElem,
    gadgets::{
        fhe::ring_gsw::{RingGswCiphertext, RingGswContext},
        fhe_prg::goldreich::GoldreichFhePrg,
    },
    lookup::{
        PublicLut,
        ggh15_eval::{GGH15BGGPolyEncodingPltEvaluator, GGH15BGGPubKeyPltEvaluator},
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
        DistType, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{
        SimulatorContext,
        error_norm::{NormBggPolyEncodingSTEvaluator, NormPltGGH15Evaluator},
    },
    slot_transfer::{BggPolyEncodingSTEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
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
const DEFAULT_INPUT_SIZE: usize = 5;
const DEFAULT_OUTPUT_SIZE: usize = 1;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_P_MODULI_BITS: usize = 7;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 4;
const DEFAULT_SCALE: u64 = 1 << 8;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_GRAPH_SEED: [u8; 32] = [0u8; 32];
const DEFAULT_BENCH_SEED: [u8; 32] = [0u8; 32];
const TRAPDOOR_SIGMA: f64 = 4.578;

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type GpuPubKeyPltEvaluator = GGH15BGGPubKeyPltEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuPubKeySlotEvaluator = BggPublicKeySTEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuPolyPltEvaluator = GGH15BGGPolyEncodingPltEvaluator<GpuMatrix, GpuHashSampler>;
type GpuPolySlotEvaluator = BggPolyEncodingSTEvaluator<GpuMatrix, GpuHashSampler>;

#[derive(Debug, Clone)]
struct GoldreichRingGswBenchConfig {
    ring_dim: u32,
    input_size: usize,
    output_size: usize,
    crt_bits: usize,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
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

fn env_or_parse_u64(key: &str, default: u64) -> u64 {
    match env::var(key) {
        Ok(raw) => raw.parse::<u64>().unwrap_or_else(|e| panic!("{key} must be a valid u64: {e}")),
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

impl GoldreichRingGswBenchConfig {
    fn from_env() -> Self {
        let ring_dim =
            env_or_parse_u32("GGH15_GOLDREICH_RING_GSW_BENCH_RING_DIM", DEFAULT_RING_DIM);
        let input_size =
            env_or_parse_usize("GGH15_GOLDREICH_RING_GSW_BENCH_INPUT_SIZE", DEFAULT_INPUT_SIZE);
        let output_size =
            env_or_parse_usize("GGH15_GOLDREICH_RING_GSW_BENCH_OUTPUT_SIZE", DEFAULT_OUTPUT_SIZE);
        let crt_bits =
            env_or_parse_usize("GGH15_GOLDREICH_RING_GSW_BENCH_CRT_BITS", DEFAULT_CRT_BITS);
        let p_moduli_bits = env_or_parse_usize(
            "GGH15_GOLDREICH_RING_GSW_BENCH_P_MODULI_BITS",
            DEFAULT_P_MODULI_BITS,
        );
        let max_unreduced_muls = env_or_parse_usize(
            "GGH15_GOLDREICH_RING_GSW_BENCH_MAX_UNREDUCED_MULS",
            DEFAULT_MAX_UNREDUCED_MULS,
        );
        let scale = env_or_parse_u64("GGH15_GOLDREICH_RING_GSW_BENCH_SCALE", DEFAULT_SCALE);
        let base_bits =
            env_or_parse_u32("GGH15_GOLDREICH_RING_GSW_BENCH_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth = env_or_parse_usize(
            "GGH15_GOLDREICH_RING_GSW_BENCH_MAX_CRT_DEPTH",
            DEFAULT_MAX_CRT_DEPTH,
        );
        let error_sigma =
            env_or_parse_f64("GGH15_GOLDREICH_RING_GSW_BENCH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret =
            env_or_parse_usize("GGH15_GOLDREICH_RING_GSW_BENCH_D_SECRET", DEFAULT_D_SECRET);
        let bench_iterations = env_or_parse_usize(
            "GGH15_GOLDREICH_RING_GSW_BENCH_BENCH_ITERATIONS",
            DEFAULT_BENCH_ITERATIONS,
        );
        let active_levels_override = env::var("GGH15_GOLDREICH_RING_GSW_BENCH_ACTIVE_LEVELS")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(|value| {
                value.parse::<usize>().unwrap_or_else(|e| {
                    panic!(
                        "GGH15_GOLDREICH_RING_GSW_BENCH_ACTIVE_LEVELS must be a valid usize: {e}"
                    )
                })
            });
        let dir_name_override = env::var("GGH15_GOLDREICH_RING_GSW_BENCH_DIR_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        assert!(ring_dim > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_RING_DIM must be > 0");
        assert!(input_size >= 5, "GGH15_GOLDREICH_RING_GSW_BENCH_INPUT_SIZE must be at least 5");
        assert!(output_size > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_OUTPUT_SIZE must be > 0");
        assert!(crt_bits > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_CRT_BITS must be > 0");
        assert!(p_moduli_bits > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_P_MODULI_BITS must be > 0");
        assert!(
            max_unreduced_muls > 0,
            "GGH15_GOLDREICH_RING_GSW_BENCH_MAX_UNREDUCED_MULS must be > 0"
        );
        assert!(scale > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_SCALE must be > 0");
        assert!(base_bits > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma >= 0.0, "GGH15_GOLDREICH_RING_GSW_BENCH_ERROR_SIGMA must be >= 0");
        assert!(d_secret > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_D_SECRET must be > 0");
        assert!(
            bench_iterations > 0,
            "GGH15_GOLDREICH_RING_GSW_BENCH_BENCH_ITERATIONS must be > 0"
        );
        if let Some(active_levels) = active_levels_override {
            assert!(active_levels > 0, "GGH15_GOLDREICH_RING_GSW_BENCH_ACTIVE_LEVELS must be > 0");
        }

        Self {
            ring_dim,
            input_size,
            output_size,
            crt_bits,
            p_moduli_bits,
            max_unreduced_muls,
            scale,
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
                "test_data/test_gpu_ggh15_goldreich_ring_gsw_bench_ring{}_in{}_out{}_active{}_crt{}_depth{}",
                self.ring_dim,
                self.input_size,
                self.output_size,
                crt_depth,
                self.crt_bits,
                crt_depth
            )
        })
    }

    fn pubkey_bench_dir(&self, crt_depth: usize) -> PathBuf {
        Path::new(&self.bench_dir_base(crt_depth)).join("pubkey")
    }

    fn poly_bench_dir(&self, crt_depth: usize) -> PathBuf {
        Path::new(&self.bench_dir_base(crt_depth)).join("poly")
    }
}

impl GoldreichRingGswBenchConfig {
    fn active_levels_for_crt_depth(&self, crt_depth: usize) -> usize {
        let active_levels = self.active_levels_override.unwrap_or(crt_depth);
        assert!(
            active_levels <= crt_depth,
            "GGH15_GOLDREICH_RING_GSW_BENCH_ACTIVE_LEVELS exceeds crt_depth: active_levels={}, crt_depth={}",
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

fn ring_gsw_q_modulus<P: Poly>(ctx: &RingGswContext<P>) -> BigUint {
    let (q_moduli, _, _) = ctx.params.to_crt();
    q_moduli
        .iter()
        .skip(ctx.level_offset)
        .take(ctx.active_levels)
        .fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i))
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
    decryption_ok: bool,
}

impl CrtDepthProbe {
    fn search_ok(&self) -> bool {
        self.eval_ok
    }
}

fn probe_crt_depth_for_goldreich_ring_gsw_bench(
    cfg: &GoldreichRingGswBenchConfig,
    crt_depth: usize,
    ring_dim_sqrt: &BigDecimal,
    base: &BigDecimal,
    error_sigma: &BigDecimal,
    e_init_norm: &BigDecimal,
) -> CrtDepthProbe {
    let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (_, _, actual_crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(actual_crt_depth);
    let (active_q_moduli, active_q, _) = active_q_moduli_and_modulus(&params, active_levels);
    let full_q = params.modulus();
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let (circuit, ctx, encrypted_outputs) =
        build_goldreich_ring_gsw_circuit::<DCRTPoly>(&params, cfg);
    let log_base_q = params.modulus_digits();
    let log_base_q_small = log_base_q / actual_crt_depth;
    let sim_ctx = Arc::new(SimulatorContext::new(
        ring_dim_sqrt.clone(),
        base.clone(),
        cfg.d_secret,
        log_base_q,
        log_base_q_small,
    ));
    let plt_evaluator = NormPltGGH15Evaluator::new(sim_ctx.clone(), error_sigma, error_sigma, None);
    let slot_transfer_evaluator =
        NormBggPolyEncodingSTEvaluator::new(sim_ctx.clone(), cfg.error_sigma, error_sigma, None);
    let out_errors = circuit.simulate_max_error_norm(
        sim_ctx,
        BigDecimal::from(1u64),
        circuit.num_input(),
        e_init_norm,
        Some(&plt_evaluator),
        Some(&slot_transfer_evaluator),
    );
    debug!("simulate_max_error_norm finished output_errors={}", out_errors.len());
    let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
    let threshold_bd = BigDecimal::from_biguint(threshold.clone(), 0);
    let max_eval_error = max_bigdecimal(
        out_errors
            .par_iter()
            .map(|error| error.matrix_norm.poly_norm.norm.clone())
            .collect::<Vec<_>>(),
    );
    debug!("max_eval_error_bits={}", bigdecimal_bits_ceil(&max_eval_error));
    let ring_gsw_q = ring_gsw_q_modulus(ctx.as_ref());
    let ring_gsw_threshold = &ring_gsw_q / BigUint::from(2u64);
    let ring_gsw_threshold_bd = BigDecimal::from_biguint(ring_gsw_threshold.clone(), 0);
    let max_decryption_error = max_bigdecimal(
        encrypted_outputs
            .par_iter()
            .map(|ciphertext| ciphertext.estimate_decryption_error_norm(cfg.error_sigma))
            .collect::<Vec<_>>(),
    );
    debug!("max_decryption_error_bits={}", bigdecimal_bits_ceil(&max_decryption_error));
    let eval_ok = max_eval_error < threshold_bd;
    let decryption_ok = max_decryption_error < ring_gsw_threshold_bd;

    info!(
        "crt_depth={} active_levels={} active_q_bits={} full_q_bits={} ring_gsw_q_bits={} non_free_depth={} num_inputs={} output_polys={} max_eval_error_bits={} eval_threshold_bits={} max_decryption_error_bits={} decryption_threshold_bits={} eval_ok={} decryption_ok={}",
        crt_depth,
        active_levels,
        active_q.bits(),
        full_q.bits(),
        ring_gsw_q.bits(),
        circuit.non_free_depth(),
        circuit.num_input(),
        out_errors.len(),
        bigdecimal_bits_ceil(&max_eval_error),
        bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold, 0)),
        bigdecimal_bits_ceil(&max_decryption_error),
        bigdecimal_bits_ceil(&BigDecimal::from_biguint(ring_gsw_threshold, 0)),
        eval_ok,
        decryption_ok
    );

    CrtDepthProbe { params, eval_ok, decryption_ok }
}

fn build_goldreich_ring_gsw_circuit<P: Poly + 'static>(
    params: &P::Params,
    cfg: &GoldreichRingGswBenchConfig,
) -> (PolyCircuit<P>, Arc<RingGswContext<P>>, Vec<RingGswCiphertext<P>>) {
    let build_start = Instant::now();
    let mut circuit = PolyCircuit::<P>::new();
    let (_, _, crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(crt_depth);
    let ctx = Arc::new(RingGswContext::setup(
        &mut circuit,
        params,
        cfg.num_slots(),
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        Some(active_levels),
        None,
    ));
    let goldreich =
        GoldreichFhePrg::setup(ctx.clone(), cfg.input_size, cfg.output_size, DEFAULT_GRAPH_SEED);
    let encrypted_inputs = (0..goldreich.input_size)
        .map(|_| RingGswCiphertext::input(ctx.clone(), None, &mut circuit))
        .collect::<Vec<_>>();
    let encrypted_outputs = goldreich.evaluate(&encrypted_inputs, &mut circuit);
    let reconstructed_outputs = encrypted_outputs
        .iter()
        .flat_map(|ciphertext| ciphertext.reconstruct(&mut circuit))
        .collect::<Vec<_>>();
    circuit.output(reconstructed_outputs);
    info!(
        "goldreich ring_gsw circuit build elapsed_ms={:.3} input_size={} output_size={} encrypted_outputs={}",
        build_start.elapsed().as_secs_f64() * 1000.0,
        cfg.input_size,
        cfg.output_size,
        encrypted_outputs.len()
    );
    (circuit, ctx, encrypted_outputs)
}

fn find_crt_depth_for_goldreich_ring_gsw_bench(
    cfg: &GoldreichRingGswBenchConfig,
) -> (usize, DCRTPolyParams) {
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
        // let probe = probe_crt_depth_for_goldreich_ring_gsw_bench(
        //     cfg,
        //     requested_crt_depth,
        //     &ring_dim_sqrt,
        //     &base,
        //     &error_sigma,
        //     &e_init_norm,
        // );
        // let (_, _, actual_crt_depth) = probe.params.to_crt();
        // info!(
        //     "using CRT_DEPTH override: requested_crt_depth={} actual_crt_depth={} eval_ok={}
        // decryption_ok={}",     requested_crt_depth, actual_crt_depth, probe.eval_ok,
        // probe.decryption_ok );
        let params =
            DCRTPolyParams::new(cfg.ring_dim, requested_crt_depth, cfg.crt_bits, cfg.base_bits);
        // assert!(
        //     probe.eval_ok,
        //     "CRT_DEPTH={} failed the eval threshold check for the Goldreich Ring-GSW benchmark",
        //     requested_crt_depth
        // );
        return (requested_crt_depth, params);
    }

    let min_crt_depth = 1usize;
    let mut low = min_crt_depth;
    let mut high = cfg.max_crt_depth;
    while low < high {
        let mid = low + (high - low) / 2;
        let probe = probe_crt_depth_for_goldreich_ring_gsw_bench(
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

    let probe = probe_crt_depth_for_goldreich_ring_gsw_bench(
        cfg,
        low,
        &ring_dim_sqrt,
        &base,
        &error_sigma,
        &e_init_norm,
    );
    if probe.search_ok() {
        info!("selected crt_depth={} for Goldreich Ring-GSW benchmark search", low);
        return (low, probe.params);
    }

    panic!(
        "crt_depth satisfying the GGH15 error threshold was not found up to GGH15_GOLDREICH_RING_GSW_BENCH_MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
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

fn single_slot_plaintext(value: usize, params: &GpuDCRTPolyParams) -> Arc<[u8]> {
    Arc::<[u8]>::from(GpuDCRTPoly::from_usize_to_constant(params, value).to_compact_bytes())
}

fn single_slot_secret_mats(slot_secret_mats: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![slot_secret_mats.first().expect("slot secret mats must not be empty").clone()]
}

fn sample_benchmark_pubkeys(
    params: &GpuDCRTPolyParams,
    seed: [u8; 32],
    d_secret: usize,
    count: usize,
    tag: &[u8],
) -> Vec<BggPublicKey<GpuMatrix>> {
    let sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new(seed, d_secret);
    let reveal_plaintexts = vec![true; count];
    sampler.sample(params, tag, &reveal_plaintexts)
}

fn constant_and_shared_benchmark_pubkeys(
    pubkeys: &[BggPublicKey<GpuMatrix>],
) -> (&BggPublicKey<GpuMatrix>, &BggPublicKey<GpuMatrix>) {
    assert_eq!(
        pubkeys.len(),
        2,
        "benchmark pubkeys must contain exactly the constant-one key and one reusable input key"
    );
    (&pubkeys[0], &pubkeys[1])
}

fn repeated_shared_benchmark_pubkeys<'a>(
    pubkeys: &'a [BggPublicKey<GpuMatrix>],
    non_constant_count: usize,
) -> Vec<&'a BggPublicKey<GpuMatrix>> {
    let (constant_one, shared_input) = constant_and_shared_benchmark_pubkeys(pubkeys);
    std::iter::once(constant_one)
        .chain((0..non_constant_count).map(move |_| shared_input))
        .collect()
}

#[tokio::test]
async fn test_gpu_ggh15_goldreich_ring_gsw_bench() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    gpu_device_sync();

    let cfg = GoldreichRingGswBenchConfig::from_env();
    info!("goldreich ring_gsw bench config: {:?}", cfg);

    let (crt_depth, cpu_params) = find_crt_depth_for_goldreich_ring_gsw_bench(&cfg);
    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_ids = detected_gpu_device_ids();
    let detected_gpu_count = detected_gpu_device_count();
    assert_eq!(
        detected_gpu_count,
        detected_gpu_ids.len(),
        "detected GPU count and ids length must match"
    );
    let single_gpu_id = *detected_gpu_ids
        .first()
        .expect("at least one GPU device is required for test_gpu_ggh15_goldreich_ring_gsw_bench");
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
        build_goldreich_ring_gsw_circuit::<GpuDCRTPoly>(&params, &cfg);
    let max_selected_decryption_error = max_bigdecimal(
        encrypted_outputs
            .iter()
            .map(|ciphertext| ciphertext.estimate_decryption_error_norm(cfg.error_sigma))
            .collect::<Vec<_>>(),
    );
    let ring_gsw_q = ring_gsw_q_modulus(ctx.as_ref());
    let gate_counts = circuit.count_gates_by_type_vec();
    let total_lut_entries = circuit.total_registered_public_lut_entries();
    let total_public_lut_gates = gate_counts.get(&PolyGateKind::PubLut).copied().unwrap_or(0);
    let total_slot_transfer_gates =
        gate_counts.get(&PolyGateKind::SlotTransfer).copied().unwrap_or(0);
    info!(
        "forcing single GPU for benchmark path: eval_gpu_id={} detected_gpu_count={} detected_gpu_ids={:?}",
        single_gpu_id, detected_gpu_count, detected_gpu_ids
    );
    info!(
        "selected crt_depth={} actual_crt_depth={} ring_dim={} num_slots={} input_size={} output_size={} active_levels={} crt_bits={} p_moduli_bits={} base_bits={} q_moduli={:?}",
        crt_depth,
        actual_crt_depth,
        params.ring_dimension(),
        cfg.num_slots(),
        cfg.input_size,
        cfg.output_size,
        active_levels,
        cfg.crt_bits,
        cfg.p_moduli_bits,
        cfg.base_bits,
        all_q_moduli
    );
    info!(
        "circuit non_free_depth={} gate_counts={:?} num_inputs={} encrypted_outputs={} reconstructed_output_polys={}",
        circuit.non_free_depth(),
        gate_counts,
        circuit.num_input(),
        encrypted_outputs.len(),
        encrypted_outputs.len() * 2 * ctx.width()
    );
    info!(
        "active_q_bits={} active_q_moduli_len={} ring_gsw_q_bits={} selected_max_decryption_error_bits={}",
        active_q.bits(),
        active_q_moduli.len(),
        ring_gsw_q.bits(),
        bigdecimal_bits_ceil(&max_selected_decryption_error)
    );
    drop(encrypted_outputs);
    drop(ctx);
    gpu_device_sync();

    let pubkey_bench_dir = cfg.pubkey_bench_dir(crt_depth);
    ensure_dir_exists(&pubkey_bench_dir);
    init_storage_system(pubkey_bench_dir.clone());
    let (bench_lut, pubkey_bench_public_lut_id, pubkey_bench_public_lut_gate_id) =
        build_benchmark_public_lookup_gate(&params);
    let pubkey_bench_slot_transfer_src_slots = identity_slot_transfer_plan(cfg.num_slots());
    let pubkey_bench_slot_transfer_gate_id =
        build_benchmark_slot_transfer_gate(&pubkey_bench_slot_transfer_src_slots);
    let pubkey_bench_public_keys = sample_benchmark_pubkeys(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        1,
        b"GGH15_GOLDREICH_RING_GSW_PKBENCH",
    );
    let (pubkey_bench_public_lut_one, pubkey_bench_shared_input) =
        constant_and_shared_benchmark_pubkeys(&pubkey_bench_public_keys);
    let pubkey_bench_plt_evaluator = GpuPubKeyPltEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        pubkey_bench_dir.clone(),
    );
    let pubkey_bench_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        cfg.num_slots(),
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        pubkey_bench_dir.clone(),
    );
    let pubkey_bench_estimator_owner = BggPublicKeyBenchEstimator::benchmark(
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
            public_lut: &bench_lut,
            public_lut_gate_id: pubkey_bench_public_lut_gate_id,
            public_lut_id: pubkey_bench_public_lut_id,
            slot_transfer_input: pubkey_bench_shared_input,
            slot_transfer_src_slots: &pubkey_bench_slot_transfer_src_slots,
            slot_transfer_gate_id: pubkey_bench_slot_transfer_gate_id,
        },
        &pubkey_bench_plt_evaluator,
        &pubkey_bench_slot_evaluator,
        GpuPubKeyPltEvaluator::new(
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            TRAPDOOR_SIGMA,
            cfg.error_sigma,
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
    let pubkey_public_lut_aux_estimate = pubkey_bench_estimator_owner
        .estimate_public_lut_sample_aux_matrices(
            &params,
            total_lut_entries,
            total_public_lut_gates,
        );
    let pubkey_slot_transfer_aux_estimate = pubkey_bench_estimator_owner
        .estimate_slot_transfer_sample_aux_matrices(
            &params,
            cfg.num_slots(),
            total_slot_transfer_gates,
        );
    info!(
        "pubkey sample_aux estimates: public_lut={:?} slot_transfer={:?}",
        pubkey_public_lut_aux_estimate, pubkey_slot_transfer_aux_estimate
    );
    let pubkey_circuit_bench = pubkey_bench_estimator_owner.estimate_circuit_bench(&circuit);
    info!(
        "pubkey circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        pubkey_circuit_bench.total_time,
        pubkey_circuit_bench.latency,
        pubkey_circuit_bench.max_parallelism,
        pubkey_circuit_bench.peak_vram
    );
    drop(pubkey_bench_public_keys);
    gpu_device_sync();

    let poly_bench_dir = cfg.poly_bench_dir(crt_depth);
    ensure_dir_exists(&poly_bench_dir);
    init_storage_system(poly_bench_dir.clone());
    let poly_bench_pubkeys = sample_benchmark_pubkeys(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        1,
        b"GGH15_GOLDREICH_RING_GSW_POLYBENCH",
    );
    let (_poly_bench_public_lut_one, poly_bench_shared_input) =
        constant_and_shared_benchmark_pubkeys(&poly_bench_pubkeys);
    let poly_bench_pubkey_plt_evaluator = GpuPubKeyPltEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        poly_bench_dir.clone(),
    );
    let poly_bench_pubkey_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        1,
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        poly_bench_dir.clone(),
    );
    let poly_bench_slot_transfer_src_slots = vec![(0u32, None)];
    let (_poly_bench_lut, poly_bench_public_lut_id, poly_bench_public_lut_gate_id) =
        build_benchmark_public_lookup_gate(&params);
    let poly_bench_slot_transfer_gate_id =
        build_benchmark_slot_transfer_gate(&poly_bench_slot_transfer_src_slots);
    poly_bench_pubkey_plt_evaluator.record_public_lookup_state(
        &bench_lut,
        poly_bench_shared_input,
        poly_bench_public_lut_gate_id,
        poly_bench_public_lut_id,
    );
    poly_bench_pubkey_slot_evaluator.record_slot_transfer_state(
        &params,
        poly_bench_shared_input,
        &poly_bench_slot_transfer_src_slots,
        poly_bench_slot_transfer_gate_id,
    );
    let poly_bench_aux_start = Instant::now();
    poly_bench_pubkey_plt_evaluator.sample_aux_matrices(&params);
    poly_bench_pubkey_slot_evaluator.sample_aux_matrices(&params);
    info!(
        "poly bench sample_aux_matrices elapsed_ms={:.3}",
        poly_bench_aux_start.elapsed().as_secs_f64() * 1000.0
    );
    wait_for_all_writes(poly_bench_dir.clone())
        .await
        .expect("benchmark storage writes should complete");

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let bench_secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let bench_s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, bench_secrets.clone());
    let bench_slot_secret_mats = single_slot_secret_mats(
        &poly_bench_pubkey_slot_evaluator
            .load_slot_secret_mats_checkpoint(&params)
            .expect("one-slot slot secret matrix checkpoints must exist for poly benchmark"),
    );
    let bench_plaintext_rows = vec![
        vec![single_slot_plaintext(2, &params)],
        vec![single_slot_plaintext(3, &params)],
        vec![single_slot_plaintext(4, &params)],
        vec![single_slot_plaintext(1, &params)],
        vec![single_slot_plaintext(5, &params)],
        vec![single_slot_plaintext(7, &params)],
        vec![single_slot_plaintext(3, &params)],
        vec![single_slot_plaintext(6, &params)],
        vec![single_slot_plaintext(2, &params)],
        vec![single_slot_plaintext(4, &params)],
    ];
    let bench_encoding_sampler = BGGPolyEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &bench_secrets,
        Some(cfg.error_sigma),
    );
    let repeated_poly_bench_pubkeys = repeated_shared_benchmark_pubkeys(&poly_bench_pubkeys, 10);
    let bench_poly_encodings = bench_encoding_sampler.sample(
        &params,
        &repeated_poly_bench_pubkeys,
        &bench_plaintext_rows,
        Some(&bench_slot_secret_mats),
    );
    assert_eq!(
        bench_poly_encodings.len(),
        11,
        "one-slot benchmark encoding set must contain const one plus ten sample encodings"
    );

    let bench_plt_b0_matrix = poly_bench_pubkey_plt_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("one-slot public-lookup b0 checkpoint must exist for poly benchmark");
    let bench_slot_b0_matrix = poly_bench_pubkey_slot_evaluator
        .load_b0_matrix_checkpoint(&params)
        .expect("one-slot slot-transfer b0 checkpoint must exist for poly benchmark");
    let bench_plt_c_b0_compact_bytes_by_slot =
        GpuPolyPltEvaluator::build_c_b0_compact_bytes_by_slot::<GpuDCRTPolyUniformSampler>(
            &params,
            &bench_s_vec,
            &bench_plt_b0_matrix,
            &bench_slot_secret_mats,
            Some(cfg.error_sigma),
        );
    let mut bench_slot_c_b0 = bench_s_vec.clone() * &bench_slot_b0_matrix;
    if cfg.error_sigma != 0.0 {
        let slot_c_b0_error = uniform_sampler.sample_uniform(
            &params,
            bench_slot_c_b0.row_size(),
            bench_slot_c_b0.col_size(),
            DistType::GaussDist { sigma: cfg.error_sigma },
        );
        bench_slot_c_b0 = bench_slot_c_b0 + slot_c_b0_error;
    }
    let bench_poly_plt_evaluator = GpuPolyPltEvaluator::new(
        DEFAULT_BENCH_SEED,
        poly_bench_dir.clone(),
        poly_bench_pubkey_plt_evaluator.checkpoint_prefix(&params),
        bench_plt_c_b0_compact_bytes_by_slot,
    );
    let bench_poly_slot_evaluator = GpuPolySlotEvaluator::new(
        DEFAULT_BENCH_SEED,
        poly_bench_dir.clone(),
        poly_bench_pubkey_slot_evaluator.checkpoint_prefix(&params),
        bench_slot_c_b0.to_compact_bytes(),
    );
    let poly_bench_estimator = BggPolyEncodingBenchEstimator::<GpuMatrix>::benchmark_with_samples(
        &BggPolyEncodingBenchSamples {
            num_slots: cfg.num_slots(),
            params: &params,
            add_lhs: &bench_poly_encodings[1],
            add_rhs: &bench_poly_encodings[2],
            sub_lhs: &bench_poly_encodings[3],
            sub_rhs: &bench_poly_encodings[4],
            mul_lhs: &bench_poly_encodings[5],
            mul_rhs: &bench_poly_encodings[6],
            small_scalar_input: &bench_poly_encodings[7],
            small_scalar: &[3u32, 5u32],
            large_scalar_input: &bench_poly_encodings[8],
            large_scalar: &[BigUint::from(7u32)],
            public_lut_one: &bench_poly_encodings[0],
            public_lut_input: &bench_poly_encodings[9],
            public_lut: &bench_lut,
            public_lut_gate_id: poly_bench_public_lut_gate_id,
            public_lut_id: poly_bench_public_lut_id,
            slot_transfer_input: &bench_poly_encodings[10],
            slot_transfer_src_slots: &poly_bench_slot_transfer_src_slots,
            slot_transfer_gate_id: poly_bench_slot_transfer_gate_id,
        },
        &bench_poly_plt_evaluator,
        &bench_poly_slot_evaluator,
        cfg.bench_iterations,
    );
    let poly_circuit_bench = poly_bench_estimator.estimate_circuit_bench(&circuit);
    info!(
        "bgg poly encoding circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        poly_circuit_bench.total_time,
        poly_circuit_bench.latency,
        poly_circuit_bench.max_parallelism,
        poly_circuit_bench.peak_vram
    );

    wait_for_all_writes(poly_bench_dir.clone())
        .await
        .expect("final storage writes should complete");
    gpu_device_sync();
}
