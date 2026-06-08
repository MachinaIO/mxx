#![cfg(feature = "gpu")]

use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BggEncodingBenchEstimator, BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
    },
    bgg::{public_key::BggPublicKey, sampler::BGGPublicKeySampler},
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    func_enc::NoCircuitEvaluator,
    input_injector::DiamondInjector,
    io::diamond_io::GpuDCRTPolyMatrixNativeBenchEstimator,
    lookup::{PublicLut, lwe::LWEBGGPubKeyPltEvaluator},
    matrix::{dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        PolyTrapdoorSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        hash::DCRTPolyHashSampler,
        trapdoor::{DCRTPolyTrapdoorSampler, GpuDCRTPolyTrapdoorSampler},
        uniform::DCRTPolyUniformSampler,
    },
    slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
    storage::write::init_storage_system,
    utils::bigdecimal_bits_ceil,
    we::{
        DiamondWE, WitnessEnc,
        diamond_we::{DiamondWEBenchEstimator, diamond_we_find_crt_depth},
    },
};
use num_bigint::BigUint;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tempfile::tempdir;
use tracing::info;
use tracing_subscriber::prelude::*;

const DEFAULT_RING_DIM: u32 = 1 << 16;
const DEFAULT_CIRCUIT_HEIGHT: usize = 10;
const DEFAULT_WITNESS_SIZE: usize = 10;
const DEFAULT_INJECTOR_BATCH_BITS: usize = 1;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_MIN_CRT_DEPTH: usize = 1;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_SECURITY_BITS: usize = 100;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_TRAPDOOR_SIGMA: f64 = 4.578;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HASH_KEY: [u8; 32] = [0x42; 32];

type CpuMatrix = DCRTPolyMatrix;
type GpuMatrix = GpuDCRTPolyMatrix;
type CpuHashSampler = DCRTPolyHashSampler<Keccak256>;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type CpuInjector =
    DiamondInjector<CpuMatrix, DCRTPolyUniformSampler, CpuHashSampler, DCRTPolyTrapdoorSampler>;
type GpuInjector = DiamondInjector<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;
type CpuDiamondWE = DiamondWE<
    CpuMatrix,
    DCRTPolyUniformSampler,
    CpuHashSampler,
    DCRTPolyTrapdoorSampler,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
>;
type GpuDiamondWE = DiamondWE<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
>;
type GpuPubKeyPltEvaluator =
    LWEBGGPubKeyPltEvaluator<GpuMatrix, GpuHashSampler, GpuDCRTPolyTrapdoorSampler>;
type GpuPubKeySlotEvaluator = BggPublicKeySTEvaluator<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
>;

#[derive(Debug, Clone)]
struct DiamondWEGpuBenchConfig {
    ring_dim: u32,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
    circuit_height: usize,
    witness_size: usize,
    injector_batch_bits: usize,
    crt_bits: usize,
    base_bits: u32,
    min_crt_depth: usize,
    max_crt_depth: usize,
    security_bits: usize,
    bench_iterations: usize,
    error_sigma: f64,
    trapdoor_sigma: f64,
    d_secret: usize,
}

#[derive(Debug, Clone, Copy)]
struct DiamondWEGpuBenchSelectedSimulation {
    crt_depth: usize,
    ring_dim: u32,
    log_ring_dim: usize,
    achieved_secpar_for_gauss: Option<u64>,
    achieved_secpar_for_cbd: Option<u64>,
    noisy_plaintext_error_bits: usize,
    input_injection_error_bits: usize,
}

impl DiamondWEGpuBenchConfig {
    fn from_env() -> Self {
        let ring_dim = env_or_parse_u32("DIAMOND_WE_GPU_BENCH_RING_DIM", DEFAULT_RING_DIM);
        assert!(ring_dim > 0, "DIAMOND_WE_GPU_BENCH_RING_DIM must be positive");
        assert!(ring_dim.is_power_of_two(), "DIAMOND_WE_GPU_BENCH_RING_DIM must be a power of two");
        let default_log_ring_dim = ring_dim.trailing_zeros() as usize;
        let cfg = Self {
            ring_dim,
            min_log_ring_dim: env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_MIN_LOG_RING_DIM")
                .unwrap_or(default_log_ring_dim),
            max_log_ring_dim: env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_MAX_LOG_RING_DIM")
                .unwrap_or(default_log_ring_dim),
            circuit_height: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT",
                DEFAULT_CIRCUIT_HEIGHT,
            ),
            witness_size: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_WITNESS_SIZE",
                DEFAULT_WITNESS_SIZE,
            ),
            injector_batch_bits: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS",
                DEFAULT_INJECTOR_BATCH_BITS,
            ),
            crt_bits: env_or_parse_usize("DIAMOND_WE_GPU_BENCH_CRT_BITS", DEFAULT_CRT_BITS),
            base_bits: env_or_parse_u32("DIAMOND_WE_GPU_BENCH_BASE_BITS", DEFAULT_BASE_BITS),
            min_crt_depth: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_MIN_CRT_DEPTH",
                DEFAULT_MIN_CRT_DEPTH,
            ),
            max_crt_depth: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_MAX_CRT_DEPTH",
                DEFAULT_MAX_CRT_DEPTH,
            ),
            security_bits: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_SECURITY_BITS",
                DEFAULT_SECURITY_BITS,
            ),
            bench_iterations: env_or_parse_usize(
                "DIAMOND_WE_GPU_BENCH_ITERATIONS",
                DEFAULT_BENCH_ITERATIONS,
            ),
            error_sigma: env_or_parse_f64("DIAMOND_WE_GPU_BENCH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA),
            trapdoor_sigma: env_or_parse_f64(
                "DIAMOND_WE_GPU_BENCH_TRAPDOOR_SIGMA",
                DEFAULT_TRAPDOOR_SIGMA,
            ),
            d_secret: env_or_parse_usize("DIAMOND_WE_GPU_BENCH_D_SECRET", DEFAULT_D_SECRET),
        };
        assert!(
            cfg.min_log_ring_dim <= cfg.max_log_ring_dim,
            "DIAMOND_WE_GPU_BENCH_MIN_LOG_RING_DIM must be <= DIAMOND_WE_GPU_BENCH_MAX_LOG_RING_DIM"
        );
        assert!(
            cfg.max_log_ring_dim < u32::BITS as usize,
            "DIAMOND_WE_GPU_BENCH_MAX_LOG_RING_DIM must be < 32"
        );
        assert!(cfg.circuit_height > 0, "DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT must be positive");
        assert!(cfg.witness_size > 0, "DIAMOND_WE_GPU_BENCH_WITNESS_SIZE must be positive");
        assert!(
            cfg.injector_batch_bits > 0,
            "DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS must be positive"
        );
        assert!(
            cfg.injector_batch_bits <= u32::BITS as usize,
            "DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS must be <= 32"
        );
        assert_eq!(
            cfg.witness_size % cfg.injector_batch_bits,
            0,
            "DIAMOND_WE_GPU_BENCH_WITNESS_SIZE must be divisible by DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS"
        );
        assert!(
            cfg.witness_size <= cfg.total_input_size(),
            "DIAMOND_WE_GPU_BENCH_WITNESS_SIZE must be <= 2^DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT"
        );
        assert!(cfg.crt_bits > 0, "DIAMOND_WE_GPU_BENCH_CRT_BITS must be positive");
        assert!(cfg.base_bits > 0, "DIAMOND_WE_GPU_BENCH_BASE_BITS must be positive");
        assert!(
            cfg.min_crt_depth <= cfg.max_crt_depth,
            "DIAMOND_WE_GPU_BENCH_MIN_CRT_DEPTH must be <= DIAMOND_WE_GPU_BENCH_MAX_CRT_DEPTH"
        );
        assert!(cfg.security_bits > 0, "DIAMOND_WE_GPU_BENCH_SECURITY_BITS must be positive");
        assert!(cfg.bench_iterations > 0, "DIAMOND_WE_GPU_BENCH_ITERATIONS must be positive");
        assert!(cfg.error_sigma >= 0.0, "DIAMOND_WE_GPU_BENCH_ERROR_SIGMA must be nonnegative");
        assert!(cfg.trapdoor_sigma > 0.0, "DIAMOND_WE_GPU_BENCH_TRAPDOOR_SIGMA must be positive");
        assert!(cfg.d_secret > 0, "DIAMOND_WE_GPU_BENCH_D_SECRET must be positive");
        cfg
    }

    fn input_base(&self) -> usize {
        1usize
            .checked_shl(
                self.injector_batch_bits
                    .try_into()
                    .expect("DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS must fit into u32"),
            )
            .expect("DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS overflowed input base")
    }

    fn injector_input_count(&self) -> usize {
        self.witness_size / self.injector_batch_bits
    }

    fn total_input_size(&self) -> usize {
        1usize
            .checked_shl(
                self.circuit_height
                    .try_into()
                    .expect("DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT must fit in u32"),
            )
            .expect("2^DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT overflowed usize")
    }

    fn instance_size(&self) -> usize {
        self.total_input_size() - self.witness_size
    }

    fn selected_simulation_from_env(&self) -> Option<DiamondWEGpuBenchSelectedSimulation> {
        let crt_depth = env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_SELECTED_CRT_DEPTH")?;
        let ring_dim = env_or_parse_optional_u32("DIAMOND_WE_GPU_BENCH_SELECTED_RING_DIM")
            .unwrap_or(self.ring_dim);
        let log_ring_dim =
            env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_SELECTED_LOG_RING_DIM")
                .unwrap_or_else(|| {
                    assert!(
                        ring_dim.is_power_of_two(),
                        "DIAMOND_WE_GPU_BENCH_SELECTED_RING_DIM must be a power of two"
                    );
                    ring_dim.trailing_zeros() as usize
                });
        let achieved_secpar_for_gauss =
            env_or_parse_optional_u64("DIAMOND_WE_GPU_BENCH_SELECTED_ACHIEVED_SECPAR_FOR_GAUSS");
        let achieved_secpar_for_cbd =
            env_or_parse_optional_u64("DIAMOND_WE_GPU_BENCH_SELECTED_ACHIEVED_SECPAR_FOR_CBD");
        let noisy_plaintext_error_bits =
            env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_SELECTED_NOISY_PLAINTEXT_ERROR_BITS")
                .unwrap_or(0);
        let input_injection_error_bits =
            env_or_parse_optional_usize("DIAMOND_WE_GPU_BENCH_SELECTED_INPUT_INJECTION_ERROR_BITS")
                .unwrap_or(0);
        assert!(crt_depth > 0, "DIAMOND_WE_GPU_BENCH_SELECTED_CRT_DEPTH must be positive");
        Some(DiamondWEGpuBenchSelectedSimulation {
            crt_depth,
            ring_dim,
            log_ring_dim,
            achieved_secpar_for_gauss,
            achieved_secpar_for_cbd,
            noisy_plaintext_error_bits,
            input_injection_error_bits,
        })
    }
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<u32>().unwrap_or_else(|err| panic!("{key} must be u32: {err}")))
        .unwrap_or(default)
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<usize>().unwrap_or_else(|err| panic!("{key} must be usize: {err}")))
        .unwrap_or(default)
}

fn env_or_parse_optional_usize(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<usize>().unwrap_or_else(|err| panic!("{key} must be usize: {err}")))
}

fn env_or_parse_optional_u32(key: &str) -> Option<u32> {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<u32>().unwrap_or_else(|err| panic!("{key} must be u32: {err}")))
}

fn env_or_parse_optional_u64(key: &str) -> Option<u64> {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<u64>().unwrap_or_else(|err| panic!("{key} must be u64: {err}")))
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<f64>().unwrap_or_else(|err| panic!("{key} must be f64: {err}")))
        .unwrap_or(default)
}

fn ensure_dir(dir: &Path) {
    fs::create_dir_all(dir).expect("failed to create DiamondWE GPU bench directory");
}

fn artifact_dir_from_env(default: PathBuf) -> PathBuf {
    env::var("DIAMOND_WE_GPU_BENCH_ARTIFACT_DIR").ok().map(PathBuf::from).unwrap_or(default)
}

fn gpu_params_for_crt_depth(
    cfg: &DiamondWEGpuBenchConfig,
    crt_depth: usize,
    gpu_id: i32,
) -> (DCRTPolyParams, GpuDCRTPolyParams) {
    let cpu_params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (moduli, _, actual_depth) = cpu_params.to_crt();
    assert_eq!(actual_depth, crt_depth, "DCRTPolyParams returned an unexpected CRT depth");
    let gpu_params = GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![gpu_id],
        Some(1),
    );
    assert_eq!(gpu_params.modulus(), cpu_params.modulus());
    (cpu_params, gpu_params)
}

fn build_circuit<P: Poly>(height: usize) -> PolyCircuit<P> {
    let mut circuit = PolyCircuit::<P>::new();
    let input_count = 1usize
        .checked_shl(height.try_into().expect("circuit height must fit in u32"))
        .expect("binary-tree circuit input count overflow");
    let mut level = circuit.input(input_count).to_vec();
    for _ in 0..height {
        level = level
            .chunks_exact(2)
            .map(|pair| circuit.and_gate(pair[0], pair[1]).as_single_wire())
            .collect();
    }
    assert_eq!(level.len(), 1, "binary-tree circuit must produce one root");
    circuit.output(vec![level[0]]);
    circuit
}

fn build_cpu_diamond_we_for_search(
    cfg: &DiamondWEGpuBenchConfig,
    ring_dim: u32,
    crt_depth: usize,
    dir_path: PathBuf,
) -> CpuDiamondWE {
    let params = DCRTPolyParams::new(ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let injector = CpuInjector::new(
        params,
        cfg.injector_input_count(),
        cfg.input_base(),
        cfg.injector_batch_bits,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
    );
    DiamondWE::new(injector, cfg.witness_size, dir_path, b"test_gpu_diamond_we_cpu_search".to_vec())
}

fn build_gpu_diamond_we(
    cfg: &DiamondWEGpuBenchConfig,
    gpu_params: GpuDCRTPolyParams,
    gpu_id: i32,
    dir_path: PathBuf,
) -> GpuDiamondWE {
    let injector = GpuInjector::new(
        gpu_params,
        cfg.injector_input_count(),
        cfg.input_base(),
        cfg.injector_batch_bits,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
    )
    .with_gpu_device_ids(vec![gpu_id]);
    DiamondWE::new(injector, cfg.witness_size, dir_path, b"test_gpu_diamond_we".to_vec())
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

fn identity_slot_transfer_plan(num_slots: usize) -> Vec<(u32, Option<u32>)> {
    (0..num_slots).map(|slot| (slot as u32, None)).collect()
}

fn build_benchmark_slot_transfer_gate(src_slots: &[(u32, Option<u32>)]) -> GateId {
    let mut circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let input = circuit.input(1);
    circuit.slot_transfer_gate(input, src_slots).as_single_wire()
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

fn build_lwe_scalar_pubkey_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    d_secret: usize,
    dir_path: PathBuf,
) -> GpuPubKeyPltEvaluator {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, DEFAULT_TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
    GpuPubKeyPltEvaluator::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    )
}

fn build_public_key_bench_estimator(
    params: &GpuDCRTPolyParams,
    cfg: &DiamondWEGpuBenchConfig,
    bench_dir: PathBuf,
) -> BggPublicKeyBenchEstimator<GpuMatrix, GpuPubKeyPltEvaluator, GpuPubKeySlotEvaluator> {
    let (public_lut, public_lut_id, public_lut_gate_id) =
        build_benchmark_public_lookup_gate(params);
    let slot_transfer_src_slots = identity_slot_transfer_plan(params.ring_dimension() as usize);
    let slot_transfer_gate_id = build_benchmark_slot_transfer_gate(&slot_transfer_src_slots);
    let public_keys = sample_benchmark_pubkeys(
        params,
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        1,
        b"test_gpu_diamond_we_pubkey_bench",
    );
    let public_lut_one = &public_keys[0];
    let shared_input = &public_keys[1];
    let scalar_plt_evaluator = build_lwe_scalar_pubkey_plt_evaluator(
        params,
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        bench_dir.clone(),
    );
    let public_lut_estimator = build_lwe_scalar_pubkey_plt_evaluator(
        params,
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        bench_dir.clone(),
    );
    let scalar_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        params.ring_dimension() as usize,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
        bench_dir,
    );
    BggPublicKeyBenchEstimator::benchmark(
        &BggPublicKeyBenchSamples {
            params,
            add_lhs: shared_input,
            add_rhs: shared_input,
            sub_lhs: shared_input,
            sub_rhs: shared_input,
            mul_lhs: shared_input,
            mul_rhs: shared_input,
            small_scalar_input: shared_input,
            small_scalar: &[3u32, 5u32],
            large_scalar_input: shared_input,
            large_scalar: &[BigUint::from(7u32)],
            public_lut_one,
            public_lut_input: shared_input,
            public_lut: &public_lut,
            public_lut_id,
            public_lut_gate_id,
            slot_transfer_input: shared_input,
            slot_transfer_src_slots: &slot_transfer_src_slots,
            slot_transfer_gate_id,
        },
        &scalar_plt_evaluator,
        &scalar_slot_evaluator,
        public_lut_estimator,
        scalar_slot_evaluator.clone(),
        cfg.bench_iterations,
    )
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_diamond_we_error_search_bench_estimate_and_round_trip() {
    let log_filter = tracing_subscriber::filter::Targets::new()
        .with_target("test_gpu_diamond_we", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::we::diamond_we", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::io::utils::simulation", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target(
            "mxx::we::diamond_we::bench_estimator",
            tracing_subscriber::filter::LevelFilter::DEBUG,
        )
        .with_target("mxx::bench_estimator", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::storage::write", tracing_subscriber::filter::LevelFilter::INFO)
        .with_default(tracing_subscriber::filter::LevelFilter::WARN);
    let _ = tracing_subscriber::registry()
        .with(log_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
    gpu_device_sync();

    let cfg = DiamondWEGpuBenchConfig::from_env();
    info!("DiamondWE GPU bench config: {:?}", cfg);
    let gpu_id =
        *detected_gpu_device_ids().first().expect("test_gpu_diamond_we requires at least one GPU");
    let temp_dir = tempdir().expect("DiamondWE GPU bench test must create a tempdir");
    init_storage_system(temp_dir.path().to_path_buf());

    let cpu_circuit = build_circuit::<DCRTPoly>(cfg.circuit_height);
    let selected = if let Some(selected) = cfg.selected_simulation_from_env() {
        info!(
            crt_depth = selected.crt_depth,
            log_ring_dim = selected.log_ring_dim,
            ring_dim = selected.ring_dim,
            achieved_secpar_for_gauss = selected.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd = selected.achieved_secpar_for_cbd,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            input_injection_error_bits = selected.input_injection_error_bits,
            "DiamondWE selected simulation parameters provided; skipping error simulation"
        );
        selected
    } else {
        let search_dir = temp_dir.path().join("search");
        let search = diamond_we_find_crt_depth(
            cfg.min_crt_depth,
            cfg.max_crt_depth,
            cfg.min_log_ring_dim,
            cfg.max_log_ring_dim,
            cfg.security_bits,
            &cpu_circuit,
            |ring_dim, crt_depth| {
                build_cpu_diamond_we_for_search(
                    &cfg,
                    ring_dim,
                    crt_depth,
                    search_dir.join("candidate"),
                )
            },
            None::<&NoCircuitEvaluator>,
            None::<&NoCircuitEvaluator>,
        )
        .expect("DiamondWE CRT-depth search must find a valid benchmark candidate");
        let selected = DiamondWEGpuBenchSelectedSimulation {
            crt_depth: search.crt_depth,
            ring_dim: search.ring_dim,
            log_ring_dim: search.log_ring_dim,
            achieved_secpar_for_gauss: search.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd: search.achieved_secpar_for_cbd,
            noisy_plaintext_error_bits: bigdecimal_bits_ceil(
                &search.simulation.noisy_plaintext_error.poly_norm.norm,
            ) as usize,
            input_injection_error_bits: bigdecimal_bits_ceil(
                &search.simulation.input_injection.state_errors[0].poly_norm.norm,
            ) as usize,
        };
        info!(
            crt_depth = selected.crt_depth,
            log_ring_dim = selected.log_ring_dim,
            ring_dim = selected.ring_dim,
            achieved_secpar_for_gauss = selected.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd = selected.achieved_secpar_for_cbd,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            input_injection_error_bits = selected.input_injection_error_bits,
            "DiamondWE CRT-depth search selected parameters"
        );
        selected
    };

    let selected_cfg = DiamondWEGpuBenchConfig { ring_dim: selected.ring_dim, ..cfg.clone() };
    let (_cpu_params, gpu_params) =
        gpu_params_for_crt_depth(&selected_cfg, selected.crt_depth, gpu_id);
    let final_dir = artifact_dir_from_env(temp_dir.path().join("final_estimate"));
    ensure_dir(&final_dir);
    info!(
        artifact_dir = %final_dir.display(),
        "DiamondWE GPU artifact directory selected"
    );
    init_storage_system(final_dir.clone());
    let diamond = build_gpu_diamond_we(&cfg, gpu_params.clone(), gpu_id, final_dir.clone());
    let gpu_circuit = build_circuit::<GpuDCRTPoly>(cfg.circuit_height);

    info!("starting DiamondWE GPU public-key bench estimator construction");
    let public_key_estimator =
        build_public_key_bench_estimator(&gpu_params, &cfg, final_dir.join("pubkey_bench"));
    info!("starting DiamondWE GPU encoding bench estimator construction");
    let encoding_estimator =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(&gpu_params, cfg.bench_iterations);
    info!("starting DiamondWE GPU native bench estimator construction");
    let native_estimator =
        GpuDCRTPolyMatrixNativeBenchEstimator::new(gpu_params.clone(), cfg.bench_iterations);
    let diamond_bench_estimator = DiamondWEBenchEstimator::new(
        &public_key_estimator,
        &encoding_estimator,
        &native_estimator,
        &diamond,
        cfg.bench_iterations,
    );
    let estimate = diamond_bench_estimator.estimate(&diamond, &gpu_circuit);
    info!(
        enc_latency = estimate.enc.latency,
        enc_total_time_nanos = %estimate.enc.total_time,
        enc_max_parallelism = %estimate.enc.max_parallelism,
        dec_latency = estimate.dec.latency,
        dec_total_time_nanos = %estimate.dec.total_time,
        dec_max_parallelism = %estimate.dec.max_parallelism,
        ciphertext_bytes = %estimate.ciphertext_bytes,
        input_injection_bytes = %estimate.input_injection_bytes,
        we_preimage_bytes = %estimate.we_preimage_bytes,
        "DiamondWE GPU benchmark estimate"
    );
    assert!(estimate.enc.total_time > BigUint::from(0u32));
    assert!(estimate.dec.total_time > BigUint::from(0u32));
    assert!(estimate.ciphertext_bytes > BigUint::from(0u32));

    let witness = { vec![true; cfg.witness_size] };
    let instance = vec![true; cfg.instance_size()];
    let msg = rand::random::<bool>();
    info!(msg, "starting DiamondWE GPU enc");
    let ct = diamond.enc(&msg, gpu_circuit, &instance);
    gpu_device_sync();
    info!("starting DiamondWE GPU dec");
    let decoded = diamond.dec(&ct, &witness);
    gpu_device_sync();
    info!(msg, decoded, "DiamondWE GPU round trip finished");
    assert_eq!(decoded, msg);
}
