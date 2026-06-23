#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BenchEstimator, BggEncodingBenchEstimator, BggPublicKeyBenchEstimator,
        BggPublicKeyBenchSamples, PublicLutSampleAuxBenchEstimator,
    },
    bgg::{
        public_key::BggPublicKey,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    circuit::{PolyCircuit, PolyGateKind, gate::GateId},
    element::PolyElem,
    gadgets::arith::{
        ModularArithmeticGadget, MontgomeryPoly, MontgomeryPolyContext,
        encode_montgomery_poly_with_window,
    },
    lookup::{
        PublicLut,
        lwe::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator},
        poly::PolyPltEvaluator,
    },
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyTrapdoorSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{SimulatorContext, error_norm::NormPltLWEEvaluator},
    slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::{debug, info};

const DEFAULT_RING_DIM: u32 = 1 << 16;
const DEFAULT_CRT_BITS: usize = 32;
const DEFAULT_BASE_BITS: u32 = 16;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HEIGHT: usize = 1;
const DEFAULT_LIMB_BIT_SIZE: usize = 6;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_BENCH_SEED: [u8; 32] = [0u8; 32];
const TRAPDOOR_SIGMA: f64 = 4.578;
const ERROR_SIM_ACTIVE_LEVELS: usize = 1;
const ACTUAL_HASH_KEY_CHECKPOINT_FILE: &str = "actual_lwe_hash_key.bin";
const ACTUAL_TRAPDOOR_CHECKPOINT_FILE: &str = "actual_lwe_trapdoor.bin";
const ACTUAL_PUB_MATRIX_CHECKPOINT_FILE: &str = "actual_lwe_pub_matrix.bin";

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type GpuScalarPubKeyPltEvaluator =
    LWEBGGPubKeyPltEvaluator<GpuMatrix, GpuHashSampler, GpuDCRTPolyTrapdoorSampler>;
type GpuScalarEncodingPltEvaluator = LWEBGGEncodingPltEvaluator<GpuMatrix, GpuHashSampler>;
type GpuPubKeySlotEvaluator = BggPublicKeySTEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type GpuTrapdoor = <GpuDCRTPolyTrapdoorSampler as PolyTrapdoorSampler>::Trapdoor;

#[derive(Debug, Clone)]
struct MontgomeryModqArithConfig {
    ring_dim: u32,
    crt_bits: usize,
    base_bits: u32,
    max_crt_depth: usize,
    active_levels: Option<usize>,
    error_sigma: f64,
    d_secret: usize,
    height: usize,
    limb_bit_size: usize,
    bench_iterations: usize,
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

fn load_or_create_actual_lwe_checkpoint(
    params: &GpuDCRTPolyParams,
    d_secret: usize,
    dir: &Path,
) -> ([u8; 32], GpuDCRTPolyTrapdoorSampler, GpuTrapdoor, GpuMatrix) {
    fs::create_dir_all(dir).expect("failed to create actual LWE checkpoint dir");

    let hash_key_path = dir.join(ACTUAL_HASH_KEY_CHECKPOINT_FILE);
    let hash_key = match fs::read(&hash_key_path) {
        Ok(bytes) => {
            let hash_key: [u8; 32] = bytes
                .try_into()
                .expect("actual LWE hash key checkpoint must contain exactly 32 bytes");
            info!("loaded actual LWE hash key checkpoint from {}", hash_key_path.display());
            hash_key
        }
        Err(_) => {
            fs::write(&hash_key_path, DEFAULT_BENCH_SEED)
                .expect("failed to write actual LWE hash key checkpoint");
            info!("wrote actual LWE hash key checkpoint to {}", hash_key_path.display());
            DEFAULT_BENCH_SEED
        }
    };

    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, TRAPDOOR_SIGMA);
    let trapdoor_path = dir.join(ACTUAL_TRAPDOOR_CHECKPOINT_FILE);
    let pub_matrix_path = dir.join(ACTUAL_PUB_MATRIX_CHECKPOINT_FILE);
    match (fs::read(&trapdoor_path), fs::read(&pub_matrix_path)) {
        (Ok(trapdoor_bytes), Ok(pub_matrix_bytes)) => {
            let trapdoor = GpuDCRTPolyTrapdoorSampler::trapdoor_from_bytes(params, &trapdoor_bytes)
                .expect("failed to decode actual LWE trapdoor checkpoint");
            let pub_matrix = GpuMatrix::from_compact_bytes(params, &pub_matrix_bytes);
            info!("loaded actual LWE trapdoor/pub_matrix checkpoints from {}", dir.display());
            (hash_key, trapdoor_sampler, trapdoor, pub_matrix)
        }
        _ => {
            let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
            fs::write(&trapdoor_path, GpuDCRTPolyTrapdoorSampler::trapdoor_to_bytes(&trapdoor))
                .expect("failed to write actual LWE trapdoor checkpoint");
            fs::write(&pub_matrix_path, pub_matrix.to_compact_bytes())
                .expect("failed to write actual LWE pub_matrix checkpoint");
            info!("wrote actual LWE trapdoor/pub_matrix checkpoints to {}", dir.display());
            (hash_key, trapdoor_sampler, trapdoor, pub_matrix)
        }
    }
}

impl MontgomeryModqArithConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("LWE_MONTGOMERY_MODQ_ARITH_RING_DIM", DEFAULT_RING_DIM);
        let crt_bits = env_or_parse_usize("LWE_MONTGOMERY_MODQ_ARITH_CRT_BITS", DEFAULT_CRT_BITS);
        let base_bits = env_or_parse_u32("LWE_MONTGOMERY_MODQ_ARITH_BASE_BITS", DEFAULT_BASE_BITS);
        let max_crt_depth =
            env_or_parse_usize("LWE_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH", DEFAULT_MAX_CRT_DEPTH);
        let active_levels = env::var("LWE_MONTGOMERY_MODQ_ARITH_Q_LEVEL").ok().map(|raw| {
            let level = raw
                .parse::<usize>()
                .expect("LWE_MONTGOMERY_MODQ_ARITH_Q_LEVEL must be a positive integer");
            assert!(
                level > 0,
                "LWE_MONTGOMERY_MODQ_ARITH_Q_LEVEL must be greater than or equal to 1"
            );
            level
        });
        let error_sigma =
            env_or_parse_f64("LWE_MONTGOMERY_MODQ_ARITH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA);
        let d_secret = env_or_parse_usize("LWE_MONTGOMERY_MODQ_ARITH_D_SECRET", DEFAULT_D_SECRET);
        let height = env_or_parse_usize("LWE_MONTGOMERY_MODQ_ARITH_HEIGHT", DEFAULT_HEIGHT);
        let limb_bit_size =
            env_or_parse_usize("LWE_MONTGOMERY_MODQ_ARITH_LIMB_BIT_SIZE", DEFAULT_LIMB_BIT_SIZE);
        let bench_iterations = env_or_parse_usize(
            "LWE_MONTGOMERY_MODQ_ARITH_BENCH_ITERATIONS",
            DEFAULT_BENCH_ITERATIONS,
        );
        let dir_name_override = env::var("LWE_MONTGOMERY_MODQ_ARITH_DIR_NAME")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());

        assert!(ring_dim > 0, "LWE_MONTGOMERY_MODQ_ARITH_RING_DIM must be > 0");
        assert!(crt_bits > 0, "LWE_MONTGOMERY_MODQ_ARITH_CRT_BITS must be > 0");
        assert!(base_bits > 0, "LWE_MONTGOMERY_MODQ_ARITH_BASE_BITS must be > 0");
        assert!(max_crt_depth > 0, "LWE_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH must be > 0");
        assert!(error_sigma > 0.0, "LWE_MONTGOMERY_MODQ_ARITH_ERROR_SIGMA must be > 0");
        assert!(d_secret > 0, "LWE_MONTGOMERY_MODQ_ARITH_D_SECRET must be > 0");
        assert!(height > 0, "LWE_MONTGOMERY_MODQ_ARITH_HEIGHT must be > 0");
        assert!(
            limb_bit_size > 0 && limb_bit_size < 32,
            "LWE_MONTGOMERY_MODQ_ARITH_LIMB_BIT_SIZE must be in 1..32"
        );
        assert!(bench_iterations > 0, "LWE_MONTGOMERY_MODQ_ARITH_BENCH_ITERATIONS must be > 0");

        Self {
            ring_dim,
            crt_bits,
            base_bits,
            max_crt_depth,
            active_levels,
            error_sigma,
            d_secret,
            height,
            limb_bit_size,
            bench_iterations,
            dir_name_override,
        }
    }

    fn num_slots(&self) -> usize {
        self.ring_dim as usize
    }

    fn active_levels_for_crt_depth(&self, crt_depth: usize, q_level: Option<usize>) -> usize {
        let active_levels = q_level.unwrap_or(crt_depth);
        assert!(
            active_levels <= crt_depth,
            "LWE_MONTGOMERY_MODQ_ARITH_Q_LEVEL exceeds crt_depth: q_level={}, crt_depth={}",
            active_levels,
            crt_depth
        );
        active_levels
    }

    fn dir_name(&self, crt_depth: usize) -> String {
        self.dir_name_override.clone().unwrap_or_else(|| {
            format!(
                "test_data/test_gpu_lwe_montgomery_modq_arith_ring{}_active{}_crt{}_depth{}",
                self.ring_dim, crt_depth, self.crt_bits, crt_depth
            )
        })
    }
}

fn active_q_moduli_and_modulus<T: PolyParams>(
    params: &T,
    q_level: Option<usize>,
) -> (Vec<u64>, BigUint, usize) {
    let (q_moduli, _, max_q_level) = params.to_crt();
    let active_q_level = q_level.unwrap_or(max_q_level);
    assert!(
        active_q_level <= max_q_level,
        "q_level exceeds CRT depth: q_level={}, crt_depth={}",
        active_q_level,
        max_q_level
    );
    let active_q_moduli = q_moduli.into_iter().take(active_q_level).collect::<Vec<_>>();
    let active_q =
        active_q_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
    (active_q_moduli, active_q, active_q_level)
}

fn round_div_biguint(value: &BigUint, divisor: &BigUint) -> BigUint {
    let half = divisor >> 1;
    (value + &half) / divisor
}

fn build_montgomery_multiplication_tree<P: Poly + 'static>(
    ctx: Arc<MontgomeryPolyContext<P>>,
    circuit: &mut PolyCircuit<P>,
    height: usize,
    q_level: Option<usize>,
) {
    let num_inputs =
        1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
    let mut current_layer: Vec<MontgomeryPoly<P>> = (0..num_inputs)
        .map(|_| {
            <MontgomeryPoly<P> as ModularArithmeticGadget<P>>::input(
                ctx.clone(),
                q_level,
                None,
                circuit,
            )
        })
        .collect();

    while current_layer.len() > 1 {
        debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
        let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
        for pair in current_layer.chunks(2) {
            let parent = pair[0].mul(&pair[1], circuit);
            next_layer.push(parent);
        }
        current_layer = next_layer;
    }

    let root = current_layer.pop().expect("multiplication tree must contain at least one node");
    let out = root.finalize(circuit);
    circuit.output(vec![out]);
}

fn build_modq_arith_circuit_cpu(
    params: &DCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
    q_level: Option<usize>,
) -> (PolyCircuit<DCRTPoly>, Arc<MontgomeryPolyContext<DCRTPoly>>) {
    let mut circuit = PolyCircuit::<DCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx.clone(), &mut circuit, cfg.height, q_level);
    (circuit, ctx)
}

fn build_modq_arith_circuit_gpu(
    params: &GpuDCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
    q_level: Option<usize>,
) -> (PolyCircuit<GpuDCRTPoly>, Arc<MontgomeryPolyContext<GpuDCRTPoly>>) {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx.clone(), &mut circuit, cfg.height, q_level);
    (circuit, ctx)
}

fn build_modq_arith_value_circuit_gpu(
    params: &GpuDCRTPolyParams,
    cfg: &MontgomeryModqArithConfig,
    q_level: Option<usize>,
) -> PolyCircuit<GpuDCRTPoly> {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let ctx =
        Arc::new(MontgomeryPolyContext::setup(&mut circuit, params, cfg.limb_bit_size, false));
    build_montgomery_multiplication_tree(ctx, &mut circuit, cfg.height, q_level);
    circuit
}

#[derive(Debug)]
struct CrtDepthProbe {
    params: DCRTPolyParams,
    eval_ok: bool,
}

impl CrtDepthProbe {
    fn search_ok(&self) -> bool {
        self.eval_ok
    }
}

fn corrected_max_eval_error_for_active_levels(
    simulated_error: &BigDecimal,
    requested_active_levels: usize,
) -> BigDecimal {
    simulated_error * BigDecimal::from_biguint(BigUint::from(requested_active_levels), 0)
}

fn probe_crt_depth_for_modq_arith(
    cfg: &MontgomeryModqArithConfig,
    q_level: Option<usize>,
    crt_depth: usize,
    ring_dim_sqrt: &BigDecimal,
    base: &BigDecimal,
    error_sigma: &BigDecimal,
    input_bound: &BigDecimal,
    e_init_norm: &BigDecimal,
) -> CrtDepthProbe {
    let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (_, _, actual_crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(actual_crt_depth, q_level);
    assert!(
        ERROR_SIM_ACTIVE_LEVELS <= actual_crt_depth,
        "error simulation requires crt_depth >= ERROR_SIM_ACTIVE_LEVELS: crt_depth={}, ERROR_SIM_ACTIVE_LEVELS={}",
        actual_crt_depth,
        ERROR_SIM_ACTIVE_LEVELS
    );
    let (active_q_moduli, active_q, _) = active_q_moduli_and_modulus(&params, Some(active_levels));
    let full_q = params.modulus();
    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let (circuit, _ctx) = build_modq_arith_circuit_cpu(&params, cfg, Some(ERROR_SIM_ACTIVE_LEVELS));

    let log_base_q = params.modulus_digits();
    let log_base_q_small = log_base_q / actual_crt_depth;
    let ctx = Arc::new(SimulatorContext::new(
        ring_dim_sqrt.clone(),
        base.clone(),
        cfg.d_secret,
        log_base_q,
        log_base_q_small,
    ));
    let plt_evaluator = NormPltLWEEvaluator::new(ctx.clone(), error_sigma);

    let out_errors = circuit.simulate_max_error_norm(
        ctx,
        input_bound.clone(),
        circuit.num_input(),
        e_init_norm,
        Some(&plt_evaluator),
        None,
    );

    debug!(
        "montgomery modq_arith simulate_max_error_norm finished output_errors={}",
        out_errors.len()
    );
    assert_eq!(out_errors.len(), 1);

    let threshold = full_q.as_ref() / BigUint::from(2u64 * q_max);
    let threshold_bd = BigDecimal::from_biguint(threshold.clone(), 0);
    let max_eval_error = out_errors[0].matrix_norm.maximum_coefficient_bound();
    let corrected_max_eval_error =
        corrected_max_eval_error_for_active_levels(&max_eval_error, active_levels);
    let eval_ok = corrected_max_eval_error < threshold_bd;

    info!(
        "montgomery modq_arith crt_depth={} active_levels={} error_sim_active_levels={} active_q_bits={} full_q_bits={} num_inputs={} output_polys={} sim_max_eval_error_bits={} max_eval_error_bits={} eval_threshold_bits={} eval_ok={}",
        crt_depth,
        active_levels,
        ERROR_SIM_ACTIVE_LEVELS,
        active_q.bits(),
        full_q.bits(),
        circuit.num_input(),
        out_errors.len(),
        bigdecimal_bits_ceil(&max_eval_error),
        bigdecimal_bits_ceil(&corrected_max_eval_error),
        bigdecimal_bits_ceil(&BigDecimal::from_biguint(threshold, 0)),
        eval_ok
    );

    CrtDepthProbe { params, eval_ok }
}

fn find_crt_depth_for_modq_arith(
    cfg: &MontgomeryModqArithConfig,
    q_level: Option<usize>,
) -> (usize, DCRTPolyParams) {
    let ring_dim_sqrt = BigDecimal::from_u32(cfg.ring_dim).unwrap().sqrt().unwrap();
    let base = BigDecimal::from_biguint(BigUint::from(1u32) << cfg.base_bits, 0);
    let error_sigma = BigDecimal::from_f64(cfg.error_sigma).expect("valid error sigma");
    let input_bound = BigDecimal::from((1u64 << cfg.limb_bit_size) - 1);
    let e_init_norm = error_sigma.clone();

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

    let min_crt_depth = q_level.unwrap_or(1);
    assert!(
        min_crt_depth <= cfg.max_crt_depth,
        "minimum crt_depth must be <= LWE_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH"
    );

    let mut low = min_crt_depth;
    let mut high = cfg.max_crt_depth;
    while low < high {
        let mid = low + (high - low) / 2;
        let probe = probe_crt_depth_for_modq_arith(
            cfg,
            q_level,
            mid,
            &ring_dim_sqrt,
            &base,
            &error_sigma,
            &input_bound,
            &e_init_norm,
        );
        if probe.search_ok() {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    let probe = probe_crt_depth_for_modq_arith(
        cfg,
        q_level,
        low,
        &ring_dim_sqrt,
        &base,
        &error_sigma,
        &input_bound,
        &e_init_norm,
    );
    if probe.search_ok() {
        info!("selected crt_depth={} for Montgomery modq arithmetic search", low);
        return (low, probe.params);
    }

    panic!(
        "crt_depth satisfying error < q/(2*q_i) for all CRT moduli not found up to LWE_MONTGOMERY_MODQ_ARITH_MAX_CRT_DEPTH ({})",
        cfg.max_crt_depth
    );
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

#[tokio::test]
async fn test_gpu_lwe_montgomery_modq_arith() {
    let _ = tracing_subscriber::fmt::try_init();
    gpu_device_sync();
    let cfg = MontgomeryModqArithConfig::from_env();
    info!("montgomery modq arith test config: {:?}", cfg);

    let q_level = cfg.active_levels;
    let num_inputs = 1usize
        .checked_shl(cfg.height as u32)
        .expect("LWE_MONTGOMERY_MODQ_ARITH_HEIGHT is too large");
    let depth_search_start = Instant::now();
    let (crt_depth, cpu_params) = find_crt_depth_for_modq_arith(&cfg, q_level);
    info!("crt depth search elapsed_ms={:.3}", depth_search_start.elapsed().as_secs_f64() * 1000.0);

    let (moduli, _, _) = cpu_params.to_crt();
    let detected_gpu_params =
        GpuDCRTPolyParams::new(cpu_params.ring_dimension(), moduli.clone(), cpu_params.base_bits());
    let single_gpu_id = *detected_gpu_params
        .gpu_ids()
        .first()
        .expect("at least one GPU device is required for test_gpu_lwe_montgomery_modq_arith");
    let params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![single_gpu_id],
        Some(1),
    );
    let (all_q_moduli, _, actual_crt_depth) = params.to_crt();
    let active_levels = cfg.active_levels_for_crt_depth(actual_crt_depth, q_level);
    let (active_q_moduli, active_q, active_q_level) =
        active_q_moduli_and_modulus(&params, Some(active_levels));
    let (circuit, ctx) = build_modq_arith_circuit_gpu(&params, &cfg, Some(active_levels));
    let gate_counts = circuit.count_gates_by_type_vec();
    let total_gates = circuit.num_gates();
    let total_lut_entries = circuit.total_registered_public_lut_entries();
    let total_public_lut_gates = gate_counts.get(&PolyGateKind::PubLut).copied().unwrap_or(0);
    let total_slot_transfer_gates =
        gate_counts.get(&PolyGateKind::SlotTransfer).copied().unwrap_or(0);

    info!("forcing single GPU for this test: gpu_id={}", single_gpu_id);
    info!("found crt_depth={}", crt_depth);
    info!(
        "selected crt_depth={} actual_crt_depth={} cfg_active_levels={:?} ring_dim={} crt_bits={} base_bits={} q_level={:?} active_levels={} limb_bit_size={} q_moduli={:?}",
        crt_depth,
        actual_crt_depth,
        cfg.active_levels,
        params.ring_dimension(),
        cfg.crt_bits,
        cfg.base_bits,
        q_level,
        active_levels,
        cfg.limb_bit_size,
        all_q_moduli
    );
    info!(
        "active_q_level={} active_q_bits={} active_q_moduli_len={}",
        active_q_level,
        active_q.bits(),
        active_q_moduli.len()
    );
    info!("multiplication tree config: height={} num_inputs={}", cfg.height, num_inputs);
    let non_free_depth_contributions = circuit.non_free_depth_contributions();
    info!(
        "circuit total_gates={} non_free_depth_contributions={:?} gate_counts={:?} total_lut_entries={} total_public_lut_gates={} total_slot_transfer_gates={}",
        total_gates,
        non_free_depth_contributions,
        gate_counts,
        total_lut_entries,
        total_public_lut_gates,
        total_slot_transfer_gates
    );

    assert_eq!(params.modulus(), cpu_params.modulus());
    assert_eq!(circuit.num_output(), 1, "multiplication tree should emit one output gate");
    assert_eq!(
        circuit.num_input(),
        num_inputs * ctx.num_limbs * active_q_level,
        "Montgomery inputs should contribute one limb polynomial per input limb at each active q-level"
    );
    assert!(total_gates > 0, "circuit must contain gates");

    let dir_name = cfg.dir_name(crt_depth);
    let dir = Path::new(&dir_name);
    fs::create_dir_all(dir).expect("failed to create modq arithmetic test directory");
    init_storage_system(dir.to_path_buf());

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
        b"LWE_MONTGOMERY_MODQ_ARITH_PUBKEY_BENCH",
    );
    let (pubkey_bench_public_lut_one, pubkey_bench_shared_input) =
        constant_and_shared_benchmark_pubkeys(&pubkey_bench_public_keys);
    let pubkey_scalar_bench_plt_evaluator = build_lwe_scalar_pubkey_plt_evaluator(
        &params,
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        dir.to_path_buf(),
    );
    let pubkey_public_lut_aux_estimate = pubkey_scalar_bench_plt_evaluator
        .estimate_public_lut_sample_aux_matrices(
            &params,
            total_lut_entries,
            total_public_lut_gates,
        );
    gpu_device_sync();
    info!(
        "montgomery modq_arith scalar pubkey sample_aux estimates: public_lut={:?} total_lut_entries={} public_lut_gates={}",
        pubkey_public_lut_aux_estimate, total_lut_entries, total_public_lut_gates
    );
    let pubkey_scalar_bench_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_BENCH_SEED,
        cfg.d_secret,
        cfg.num_slots(),
        TRAPDOOR_SIGMA,
        cfg.error_sigma,
        dir.to_path_buf(),
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
        build_lwe_scalar_pubkey_plt_evaluator(
            &params,
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            dir.to_path_buf(),
        ),
        GpuPubKeySlotEvaluator::new(
            DEFAULT_BENCH_SEED,
            cfg.d_secret,
            cfg.num_slots(),
            TRAPDOOR_SIGMA,
            cfg.error_sigma,
            dir.to_path_buf(),
        ),
        cfg.bench_iterations,
    );
    let pubkey_circuit_bench = pubkey_scalar_bench.estimate_circuit_bench(&circuit);
    info!(
        "montgomery modq_arith scalar bgg pubkey circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        pubkey_circuit_bench.total_time,
        pubkey_circuit_bench.latency,
        pubkey_circuit_bench.max_parallelism,
        pubkey_circuit_bench.peak_vram
    );

    let encoding_scalar_bench =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(&params, cfg.bench_iterations, || ());
    let encoding_circuit_bench = encoding_scalar_bench.estimate_circuit_bench(&circuit);
    info!(
        "montgomery modq_arith scalar bgg encoding circuit bench estimate: total_time={:.6} latency={:.6} max_parallelism={} peak_vram={}",
        encoding_circuit_bench.total_time,
        encoding_circuit_bench.latency,
        encoding_circuit_bench.max_parallelism,
        encoding_circuit_bench.peak_vram
    );

    let input_value = BigUint::from(1u64);
    let expected = BigUint::from(1u64);
    let expected_poly = GpuDCRTPoly::from_biguint_to_constant(&params, expected.clone());
    let encoded_input = encode_montgomery_poly_with_window(
        cfg.limb_bit_size,
        &params,
        &input_value,
        Some(active_q_level),
        0,
    );
    let plaintext_inputs: Vec<GpuDCRTPoly> =
        (0..num_inputs).flat_map(|_| encoded_input.clone()).collect();
    assert_eq!(
        plaintext_inputs.len(),
        circuit.num_input(),
        "encoded Montgomery input count must match the multiplication tree circuit input count"
    );

    let dry_circuit = build_modq_arith_value_circuit_gpu(&params, &cfg, Some(active_q_level));
    let dry_plt_evaluator = PolyPltEvaluator::new();
    let dry_eval_start = Instant::now();
    let dry_one = GpuDCRTPoly::const_one(&params);
    let dry_inputs = plaintext_inputs.clone();
    info!("starting dry-run evaluation to verify correctness before GPU evaluation");
    let dry_out =
        dry_circuit.eval(&params, dry_one, dry_inputs, Some(&dry_plt_evaluator), None, None);
    info!("dry eval elapsed_ms={:.3}", dry_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(dry_out.len(), 1, "dry-run should output one value polynomial");
    let dry_const_term = dry_out[0].coeffs_biguints()[0].clone();
    assert_eq!(dry_const_term, expected, "dry-run output must match the expected product mod q");
    info!("dry-run succeeded with expected constant term");
    drop(dry_out);
    drop(dry_circuit);
    drop(dry_plt_evaluator);

    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(&params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let s_vec = GpuDCRTPolyMatrix::from_poly_vec_row(&params, secrets.clone());
    info!("sampled secret vector with {} polynomials", secrets.len());

    let actual_eval_dir = dir.join("actual_scalar_bgg");
    let (actual_hash_key, trapdoor_sampler, trapdoor, pub_matrix) =
        load_or_create_actual_lwe_checkpoint(&params, cfg.d_secret, &actual_eval_dir);
    let trapdoor = Arc::new(trapdoor);
    let pub_matrix = Arc::new(pub_matrix);

    let pk_sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new(actual_hash_key, cfg.d_secret);
    let reveal_plaintexts = vec![true; circuit.num_input()];
    info!("sampling public keys");
    let mut pubkeys = pk_sampler.sample(&params, b"BGG_PUBKEY", &reveal_plaintexts);
    info!("sampled {} public keys", pubkeys.len());

    let enc_setup_start = Instant::now();
    let encoding_sampler = BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(
        &params,
        &secrets,
        Some(cfg.error_sigma),
    );
    drop(secrets);
    let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintext_inputs);
    drop(plaintext_inputs);
    info!("encoding sampling elapsed_ms={:.3}", enc_setup_start.elapsed().as_secs_f64() * 1000.0);
    let encodings_compact_start = Instant::now();
    let encodings_compact: Vec<_> = encodings
        .into_iter()
        .map(|encoding| {
            (
                encoding.vector.into_compact_bytes(),
                encoding.pubkey.matrix.into_compact_bytes(),
                encoding.pubkey.reveal_plaintext,
                encoding.plaintext,
            )
        })
        .collect();
    info!(
        "encoding compact serialization elapsed_ms={:.3}",
        encodings_compact_start.elapsed().as_secs_f64() * 1000.0
    );

    let pk_evaluator_setup_start = Instant::now();
    init_storage_system(actual_eval_dir.clone());

    let pk_evaluator = GpuScalarPubKeyPltEvaluator::new(
        actual_hash_key,
        trapdoor_sampler,
        pub_matrix.clone(),
        trapdoor,
        actual_eval_dir.clone(),
    );
    info!(
        "pk evaluator setup elapsed_ms={:.3}",
        pk_evaluator_setup_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_pubkeys = pubkeys.split_off(1);
    let one_pubkey = pubkeys.pop().expect("pubkeys must contain one entry for const one");
    let pubkey_eval_start = Instant::now();
    let pubkey_out =
        circuit.eval(&params, one_pubkey, input_pubkeys, Some(&pk_evaluator), None, None);
    info!("pubkey eval elapsed_ms={:.3}", pubkey_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(pubkey_out.len(), 1);

    let sample_aux_start = Instant::now();
    pk_evaluator.sample_aux_matrices(&params);
    info!(
        "sample_aux_matrices elapsed_ms={:.3}",
        sample_aux_start.elapsed().as_secs_f64() * 1000.0
    );

    let wait_writes_start = Instant::now();
    wait_for_all_writes(actual_eval_dir.clone())
        .await
        .expect("actual scalar BGG pubkey auxiliary writes should complete");
    info!(
        "wait_for_all_writes elapsed_ms={:.3}",
        wait_writes_start.elapsed().as_secs_f64() * 1000.0
    );

    let mut c_b = s_vec.clone() * pub_matrix.as_ref();
    if cfg.error_sigma != 0.0 {
        let c_b_error = uniform_sampler.sample_uniform(
            &params,
            c_b.row_size(),
            c_b.col_size(),
            DistType::GaussDist { sigma: cfg.error_sigma },
        );
        c_b = c_b + c_b_error;
    }
    drop(pk_evaluator);

    let enc_evaluator = GpuScalarEncodingPltEvaluator::new(actual_hash_key, actual_eval_dir, c_b);

    let encodings_restore_start = Instant::now();
    let mut encodings: Vec<_> = encodings_compact
        .into_iter()
        .map(|(vector_bytes, pubkey_bytes, reveal_plaintext, plaintext)| {
            mxx::bgg::encoding::BggEncoding::new(
                GpuDCRTPolyMatrix::from_compact_bytes(&params, &vector_bytes),
                mxx::bgg::public_key::BggPublicKey::new(
                    GpuDCRTPolyMatrix::from_compact_bytes(&params, &pubkey_bytes),
                    reveal_plaintext,
                ),
                plaintext,
            )
        })
        .collect();
    info!(
        "encoding restore elapsed_ms={:.3}",
        encodings_restore_start.elapsed().as_secs_f64() * 1000.0
    );

    let input_encodings = encodings.split_off(1);
    let one_encoding = encodings.pop().expect("encodings must contain one entry for const one");
    let encoding_eval_start = Instant::now();
    let encoding_out =
        circuit.eval(&params, one_encoding, input_encodings, Some(&enc_evaluator), None, None);
    info!("encoding eval elapsed_ms={:.3}", encoding_eval_start.elapsed().as_secs_f64() * 1000.0);
    assert_eq!(encoding_out.len(), 1);

    assert_eq!(encoding_out[0].pubkey, pubkey_out[0]);
    assert_eq!(
        encoding_out[0].plaintext.as_ref().expect("output encoding plaintext should be revealed"),
        &expected_poly
    );
    let expected_poly = GpuDCRTPoly::from_biguint_to_constant(&params, expected.clone());
    let expected_times_gadget = s_vec.clone() *
        (GpuDCRTPolyMatrix::gadget_matrix(&params, cfg.d_secret) * expected_poly.clone());
    let s_times_pk = s_vec.clone() * &pubkey_out[0].matrix;
    let diff = encoding_out[0].vector.clone() - s_times_pk + expected_times_gadget;
    let coeff = diff
        .entry(0, 0)
        .coeffs()
        .into_iter()
        .next()
        .expect("diff poly must have at least one coefficient")
        .value()
        .clone();

    let q_max = *active_q_moduli.iter().max().expect("active_q_moduli must not be empty");
    let q_over_qmax = params.modulus().as_ref() / BigUint::from(q_max);
    let random_int: u64 = rand::random::<u64>() % q_max;
    let randomized_coeff = coeff + q_over_qmax.clone() * BigUint::from(random_int);
    let rounded = round_div_biguint(&randomized_coeff, &q_over_qmax);
    let decoded_random: u64 = (&rounded % BigUint::from(q_max))
        .try_into()
        .expect("decoded random coefficient must fit u64");
    assert_eq!(decoded_random, random_int);

    gpu_device_sync();
}
