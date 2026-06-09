#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BggEncodingBenchEstimator, BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
        CircuitBenchEstimate, CircuitBenchSummary, NaiveBGGVecBenchEstimator,
        PublicLutSampleAuxBenchEstimator,
    },
    bgg::{
        naive_vec::NaiveBGGEncodingVec,
        public_key::BggPublicKey,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    func_enc::NoCircuitEvaluator,
    gadgets::{
        arith::{ModularArithmeticContext, NestedRnsPolyContext},
        fhe::ring_gsw_nested_rns::sample_public_key_columns_with_samplers,
    },
    io::{
        aky24_io::{
            Aky24IO, Aky24IOBenchEstimator, Aky24IOFuncType, aky24_io_find_crt_depth,
            aky24_io_max_noise_refresh_v_bits_without_pre_rounding_error,
            minimum_aky24_io_prf_seed_bits,
        },
        diamond_io::GpuDCRTPolyMatrixNativeBenchEstimator,
    },
    lookup::{
        PltEvaluator, PublicLut,
        lwe::{
            LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
            NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
        },
    },
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
            params::DCRTPolyParams,
            poly::DCRTPoly,
        },
    },
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::error_norm::{ErrorNorm, NormNaiveBggEncodingVecSTEvaluator, NormPltLWEEvaluator},
    slot_transfer::{NaiveBGGVecSlotTransferEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
    utils::bigdecimal_bits_ceil,
};
use num_bigint::BigUint;
use std::{
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tempfile::tempdir;
use tracing::info;
use tracing_subscriber::prelude::*;

const DEFAULT_RING_DIM: u32 = 1 << 16;
const DEFAULT_MIN_LOG_RING_DIM: usize = 16;
const DEFAULT_MAX_LOG_RING_DIM: usize = 16;
const DEFAULT_INPUT_SIZE: usize = 5;
const DEFAULT_OUTPUT_SIZE: usize = 6;
const DEFAULT_PRF_BATCH_BITS: usize = 1;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_P_MODULI_BITS: usize = 7;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 4;
const DEFAULT_SCALE: u64 = 1 << 8;
const DEFAULT_MIN_CRT_DEPTH: usize = 1;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_SECURITY_BITS: usize = 100;
const DEFAULT_NOISE_REFRESH_CBD_N: usize = 2;
const DEFAULT_BENCH_ITERATIONS: usize = 5;
const DEFAULT_ERROR_SIGMA: f64 = 8.0;
const DEFAULT_TRAPDOOR_SIGMA: f64 = 4.578;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HASH_KEY: [u8; 32] = [0x42; 32];
const AKY24_IO_SECRET_SIZE: usize = 1;

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type CpuMatrix = DCRTPolyMatrix;
type CpuAky24IO = Aky24IO<
    CpuMatrix,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
>;
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
type GpuAky24IO = Aky24IO<
    GpuMatrix,
    GpuPubKeyPltEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
    GpuEncodingPltEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
>;

#[derive(Debug, Clone)]
struct Aky24IOGpuBenchConfig {
    ring_dim: u32,
    min_log_ring_dim: usize,
    max_log_ring_dim: usize,
    input_size: usize,
    output_size: usize,
    prf_batch_bits: usize,
    crt_bits: usize,
    base_bits: u32,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    min_crt_depth: usize,
    max_crt_depth: usize,
    security_bits: usize,
    noise_refresh_cbd_n: usize,
    bench_iterations: usize,
    search_only: bool,
    error_sigma: f64,
    trapdoor_sigma: f64,
    d_secret: usize,
}

#[derive(Debug, Clone, Copy)]
struct Aky24IOGpuBenchSelectedSimulation {
    crt_depth: usize,
    ring_dim: u32,
    log_ring_dim: usize,
    achieved_secpar_for_gauss: Option<u64>,
    achieved_secpar_for_cbd: Option<u64>,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
    seed_bits: usize,
    noisy_plaintext_error_bits: usize,
    initial_fresh_error_bits: usize,
}

impl Aky24IOGpuBenchConfig {
    fn from_env() -> Self {
        let input_size = env_or_parse_usize("AKY24_IO_GPU_BENCH_INPUT_SIZE", DEFAULT_INPUT_SIZE);
        let cfg = Self {
            ring_dim: env_or_parse_u32("AKY24_IO_GPU_BENCH_RING_DIM", DEFAULT_RING_DIM),
            min_log_ring_dim: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_MIN_LOG_RING_DIM",
                DEFAULT_MIN_LOG_RING_DIM,
            ),
            max_log_ring_dim: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_MAX_LOG_RING_DIM",
                DEFAULT_MAX_LOG_RING_DIM,
            ),
            input_size,
            output_size: env_or_parse_usize("AKY24_IO_GPU_BENCH_OUTPUT_SIZE", DEFAULT_OUTPUT_SIZE),
            prf_batch_bits: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_PRF_BATCH_BITS",
                DEFAULT_PRF_BATCH_BITS,
            ),
            crt_bits: env_or_parse_usize("AKY24_IO_GPU_BENCH_CRT_BITS", DEFAULT_CRT_BITS),
            base_bits: env_or_parse_u32("AKY24_IO_GPU_BENCH_BASE_BITS", DEFAULT_BASE_BITS),
            p_moduli_bits: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_P_MODULI_BITS",
                DEFAULT_P_MODULI_BITS,
            ),
            max_unreduced_muls: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_MAX_UNREDUCED_MULS",
                DEFAULT_MAX_UNREDUCED_MULS,
            ),
            scale: env_or_parse_u64("AKY24_IO_GPU_BENCH_SCALE", DEFAULT_SCALE),
            min_crt_depth: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_MIN_CRT_DEPTH",
                DEFAULT_MIN_CRT_DEPTH,
            ),
            max_crt_depth: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_MAX_CRT_DEPTH",
                DEFAULT_MAX_CRT_DEPTH,
            ),
            security_bits: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_SECURITY_BITS",
                DEFAULT_SECURITY_BITS,
            ),
            noise_refresh_cbd_n: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_NOISE_REFRESH_CBD_N",
                DEFAULT_NOISE_REFRESH_CBD_N,
            ),
            bench_iterations: env_or_parse_usize(
                "AKY24_IO_GPU_BENCH_ITERATIONS",
                DEFAULT_BENCH_ITERATIONS,
            ),
            search_only: env_or_parse_bool("AKY24_IO_GPU_BENCH_SEARCH_ONLY", false),
            error_sigma: env_or_parse_f64("AKY24_IO_GPU_BENCH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA),
            trapdoor_sigma: env_or_parse_f64(
                "AKY24_IO_GPU_BENCH_TRAPDOOR_SIGMA",
                DEFAULT_TRAPDOOR_SIGMA,
            ),
            d_secret: env_or_parse_usize("AKY24_IO_GPU_BENCH_D_SECRET", DEFAULT_D_SECRET),
        };
        assert!(cfg.ring_dim > 0, "AKY24_IO_GPU_BENCH_RING_DIM must be positive");
        assert!(
            cfg.ring_dim.is_power_of_two(),
            "AKY24_IO_GPU_BENCH_RING_DIM must be a power of two"
        );
        assert!(
            cfg.min_log_ring_dim <= cfg.max_log_ring_dim,
            "AKY24_IO_GPU_BENCH_MIN_LOG_RING_DIM must be <= AKY24_IO_GPU_BENCH_MAX_LOG_RING_DIM"
        );
        assert!(
            cfg.max_log_ring_dim < u32::BITS as usize,
            "AKY24_IO_GPU_BENCH_MAX_LOG_RING_DIM must be < 32"
        );
        assert!(cfg.input_size > 0, "AKY24_IO_GPU_BENCH_INPUT_SIZE must be positive");
        assert!(cfg.output_size > 0, "AKY24_IO_GPU_BENCH_OUTPUT_SIZE must be positive");
        assert!(cfg.prf_batch_bits > 0, "AKY24_IO_GPU_BENCH_PRF_BATCH_BITS must be positive");
        assert!(
            cfg.prf_batch_bits < usize::BITS as usize,
            "AKY24_IO_GPU_BENCH_PRF_BATCH_BITS must fit in a usize branch count"
        );
        assert_eq!(
            cfg.input_size % cfg.prf_batch_bits,
            0,
            "AKY24_IO_GPU_BENCH_INPUT_SIZE must be divisible by AKY24_IO_GPU_BENCH_PRF_BATCH_BITS"
        );
        assert!(cfg.crt_bits > 0, "AKY24_IO_GPU_BENCH_CRT_BITS must be positive");
        assert!(cfg.base_bits > 0, "AKY24_IO_GPU_BENCH_BASE_BITS must be positive");
        assert!(
            cfg.min_crt_depth <= cfg.max_crt_depth,
            "AKY24_IO_GPU_BENCH_MIN_CRT_DEPTH must be <= AKY24_IO_GPU_BENCH_MAX_CRT_DEPTH"
        );
        assert!(
            cfg.noise_refresh_cbd_n > 0,
            "AKY24_IO_GPU_BENCH_NOISE_REFRESH_CBD_N must be positive"
        );
        assert!(cfg.bench_iterations > 0, "AKY24_IO_GPU_BENCH_ITERATIONS must be positive");
        assert!(cfg.error_sigma >= 0.0, "AKY24_IO_GPU_BENCH_ERROR_SIGMA must be nonnegative");
        assert!(cfg.trapdoor_sigma > 0.0, "AKY24_IO_GPU_BENCH_TRAPDOOR_SIGMA must be positive");
        assert!(cfg.d_secret > 0, "AKY24_IO_GPU_BENCH_D_SECRET must be positive");
        cfg
    }

    fn output_size(&self) -> usize {
        self.output_size
    }

    fn seed_bits_for_params(
        &self,
        params: &DCRTPolyParams,
        prf_mask_output_coeff_bits: usize,
        noise_refresh_v_bits: usize,
    ) -> usize {
        minimum_aky24_io_prf_seed_bits(
            params,
            self.output_size,
            self.output_size,
            self.prf_batch_bits,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            self.noise_refresh_cbd_n,
        )
    }

    fn prf_mask_output_coeff_bits_search_bound(&self) -> usize {
        self.max_crt_depth
            .checked_mul(self.crt_bits)
            .expect("AKY24IO PRF-mask search bound overflow")
            .max(1)
    }

    fn selected_simulation_from_env(&self) -> Option<Aky24IOGpuBenchSelectedSimulation> {
        let crt_depth = env_or_parse_optional_usize("AKY24_IO_GPU_BENCH_SELECTED_CRT_DEPTH")?;
        let prf_mask_output_coeff_bits =
            env_or_parse_optional_usize("AKY24_IO_GPU_BENCH_SELECTED_PRF_MASK_OUTPUT_COEFF_BITS")?;
        let noise_refresh_v_bits =
            env_or_parse_optional_usize("AKY24_IO_GPU_BENCH_SELECTED_NOISE_REFRESH_V_BITS")?;
        let noisy_plaintext_error_bits =
            env_or_parse_optional_usize("AKY24_IO_GPU_BENCH_SELECTED_NOISY_PLAINTEXT_ERROR_BITS")?;
        let initial_fresh_error_bits =
            env_or_parse_optional_usize("AKY24_IO_GPU_BENCH_SELECTED_INITIAL_FRESH_ERROR_BITS")?;
        assert!(crt_depth > 0, "AKY24_IO_GPU_BENCH_SELECTED_CRT_DEPTH must be positive");
        assert!(
            prf_mask_output_coeff_bits > 0,
            "AKY24_IO_GPU_BENCH_SELECTED_PRF_MASK_OUTPUT_COEFF_BITS must be positive"
        );
        assert!(
            noise_refresh_v_bits > 0,
            "AKY24_IO_GPU_BENCH_SELECTED_NOISE_REFRESH_V_BITS must be positive"
        );
        let params = DCRTPolyParams::new(self.ring_dim, crt_depth, self.crt_bits, self.base_bits);
        let seed_bits =
            self.seed_bits_for_params(&params, prf_mask_output_coeff_bits, noise_refresh_v_bits);
        Some(Aky24IOGpuBenchSelectedSimulation {
            crt_depth,
            ring_dim: self.ring_dim,
            log_ring_dim: self.ring_dim.ilog2() as usize,
            achieved_secpar_for_gauss: None,
            achieved_secpar_for_cbd: None,
            prf_mask_output_coeff_bits,
            noise_refresh_v_bits,
            seed_bits,
            noisy_plaintext_error_bits,
            initial_fresh_error_bits,
        })
    }
}

struct DynamicNormPltLWEEvaluator {
    error_sigma: BigDecimal,
}

impl DynamicNormPltLWEEvaluator {
    fn new(error_sigma: f64) -> Self {
        Self { error_sigma: BigDecimal::from_f64(error_sigma).expect("finite error sigma") }
    }
}

impl PltEvaluator<ErrorNorm> for DynamicNormPltLWEEvaluator {
    fn public_lookup(
        &self,
        params: &(),
        plt: &PublicLut<DCRTPoly>,
        one: &ErrorNorm,
        input: &ErrorNorm,
        gate_id: GateId,
        lut_id: usize,
    ) -> ErrorNorm {
        let evaluator = NormPltLWEEvaluator::new(input.clone_ctx(), &self.error_sigma);
        evaluator.public_lookup(params, plt, one, input, gate_id, lut_id)
    }
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<u32>().unwrap_or_else(|err| panic!("{key} must be u32: {err}")))
        .unwrap_or(default)
}

fn env_or_parse_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<u64>().unwrap_or_else(|err| panic!("{key} must be u64: {err}")))
        .unwrap_or(default)
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<usize>().unwrap_or_else(|err| panic!("{key} must be usize: {err}")))
        .unwrap_or(default)
}

fn env_or_parse_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|raw| match raw.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => true,
            "0" | "false" | "FALSE" | "no" | "NO" => false,
            _ => panic!("{key} must be a bool-like value: 1/0, true/false, or yes/no"),
        })
        .unwrap_or(default)
}

fn env_or_parse_optional_usize(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<usize>().unwrap_or_else(|err| panic!("{key} must be usize: {err}")))
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .map(|raw| raw.parse::<f64>().unwrap_or_else(|err| panic!("{key} must be f64: {err}")))
        .unwrap_or(default)
}

fn ensure_clean_dir(dir: &Path) {
    if dir.exists() {
        fs::remove_dir_all(dir).expect("failed to remove existing AKY24IO GPU bench directory");
    }
    fs::create_dir_all(dir).expect("failed to create AKY24IO GPU bench directory");
}

fn gpu_params_for_crt_depth(
    cfg: &Aky24IOGpuBenchConfig,
    ring_dim: u32,
    crt_depth: usize,
    gpu_id: i32,
) -> (DCRTPolyParams, GpuDCRTPolyParams) {
    let cpu_params = DCRTPolyParams::new(ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
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

fn build_ring_gsw_context(
    params: &GpuDCRTPolyParams,
    cfg: &Aky24IOGpuBenchConfig,
    active_levels: usize,
) -> Arc<NestedRnsPolyContext> {
    let mut setup_circuit = PolyCircuit::<GpuDCRTPoly>::new();
    Arc::new(NestedRnsPolyContext::setup(
        &mut setup_circuit,
        params,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        false,
        Some(active_levels),
    ))
}

fn build_cpu_ring_gsw_context(
    params: &DCRTPolyParams,
    cfg: &Aky24IOGpuBenchConfig,
    active_levels: usize,
) -> Arc<NestedRnsPolyContext> {
    let mut setup_circuit = PolyCircuit::<DCRTPoly>::new();
    Arc::new(NestedRnsPolyContext::setup(
        &mut setup_circuit,
        params,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.scale,
        false,
        Some(active_levels),
    ))
}

fn build_lwe_pubkey_vec_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    cfg: &Aky24IOGpuBenchConfig,
    dir_path: PathBuf,
) -> (GpuPubKeyPltEvaluator, GpuMatrix) {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, cfg.trapdoor_sigma);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, cfg.d_secret);
    let evaluator = GpuPubKeyPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
        GpuMatrix,
        GpuHashSampler,
        GpuDCRTPolyTrapdoorSampler,
    >::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix.clone()),
        Arc::new(trapdoor),
        dir_path,
    ));
    (evaluator, pub_matrix)
}

fn build_lwe_scalar_pubkey_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    cfg: &Aky24IOGpuBenchConfig,
    dir_path: PathBuf,
) -> GpuScalarPubKeyPltEvaluator {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, cfg.trapdoor_sigma);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, cfg.d_secret);
    GpuScalarPubKeyPltEvaluator::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    )
}

fn build_aky24_io(
    cfg: &Aky24IOGpuBenchConfig,
    cpu_params: DCRTPolyParams,
    gpu_params: GpuDCRTPolyParams,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
    dir_path: PathBuf,
) -> GpuAky24IO {
    let crt_depth = gpu_params.crt_depth();
    let ring_gsw_context = build_ring_gsw_context(&gpu_params, cfg, crt_depth);
    let ring_gsw_level_offset = 0usize;
    let ring_gsw_enable_levels = Some(crt_depth);
    let ring_gsw_width = 2 *
        <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
            ring_gsw_context.as_ref(),
            ring_gsw_enable_levels,
            Some(ring_gsw_level_offset),
        );
    let lookup_dir = dir_path.join(format!("lookup_crt{crt_depth}"));
    fs::create_dir_all(&lookup_dir).expect("failed to create AKY24IO lookup directory");
    let (pk_lookup_evaluator, _) =
        build_lwe_pubkey_vec_plt_evaluator(&gpu_params, DEFAULT_HASH_KEY, cfg, lookup_dir.clone());
    let enc_lookup_dir = lookup_dir.clone();
    let c_b0 = GpuMatrix::zero(&gpu_params, 1, 1);
    let seed_bits =
        cfg.seed_bits_for_params(&cpu_params, prf_mask_output_coeff_bits, noise_refresh_v_bits);
    info!(
        output_size = cfg.output_size(),
        seed_bits,
        prf_mask_output_coeff_bits,
        noise_refresh_v_bits,
        "derived AKY24IO GPU benchmark seed_bits from output_size"
    );
    Aky24IO::new(
        gpu_params,
        cpu_params,
        ring_gsw_context,
        ring_gsw_width,
        ring_gsw_level_offset,
        ring_gsw_enable_levels,
        Some(cfg.error_sigma),
        b"test_gpu_aky24_io".to_vec(),
        cfg.input_size,
        cfg.output_size(),
        seed_bits,
        cfg.prf_batch_bits,
        prf_mask_output_coeff_bits,
        noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
        [0x25; 32],
        Some(pk_lookup_evaluator),
        Some(NaiveBGGVecSlotTransferEvaluator::new()),
        Some(GpuEncodingPltEvaluator::new(
            LWEBGGEncodingPltEvaluator::<GpuMatrix, GpuHashSampler>::new(
                DEFAULT_HASH_KEY,
                enc_lookup_dir,
                c_b0,
            ),
        )),
        Some(NaiveBGGVecSlotTransferEvaluator::new()),
    )
}

fn build_cpu_aky24_io_for_search(
    cfg: &Aky24IOGpuBenchConfig,
    ring_dim: u32,
    crt_depth: usize,
    prf_mask_output_coeff_bits: usize,
    noise_refresh_v_bits: usize,
) -> CpuAky24IO {
    let params = DCRTPolyParams::new(ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (_, _, actual_crt_depth) = params.to_crt();
    assert_eq!(actual_crt_depth, crt_depth, "DCRTPolyParams returned an unexpected CRT depth");
    let ring_gsw_context = build_cpu_ring_gsw_context(&params, cfg, crt_depth);
    let ring_gsw_level_offset = 0usize;
    let ring_gsw_enable_levels = Some(crt_depth);
    let ring_gsw_width = 2 *
        <NestedRnsPolyContext as ModularArithmeticContext<DCRTPoly>>::gadget_len(
            ring_gsw_context.as_ref(),
            ring_gsw_enable_levels,
            Some(ring_gsw_level_offset),
        );
    let seed_bits =
        cfg.seed_bits_for_params(&params, prf_mask_output_coeff_bits, noise_refresh_v_bits);
    info!(
        crt_depth,
        output_size = cfg.output_size(),
        seed_bits,
        prf_mask_output_coeff_bits,
        noise_refresh_v_bits,
        "derived AKY24IO CPU search seed_bits from output_size"
    );
    Aky24IO::new(
        params.clone(),
        params,
        ring_gsw_context,
        ring_gsw_width,
        ring_gsw_level_offset,
        ring_gsw_enable_levels,
        Some(cfg.error_sigma),
        b"test_gpu_aky24_io_cpu_search".to_vec(),
        cfg.input_size,
        cfg.output_size(),
        seed_bits,
        cfg.prf_batch_bits,
        prf_mask_output_coeff_bits,
        noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
        [0x25; 32],
        None,
        None,
        None,
        None,
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
        "benchmark pubkeys must contain the constant-one key and one reusable input key"
    );
    (&pubkeys[0], &pubkeys[1])
}

fn build_public_key_bench_estimator(
    params: &GpuDCRTPolyParams,
    cfg: &Aky24IOGpuBenchConfig,
    bench_dir: PathBuf,
) -> NaiveBGGVecBenchEstimator<
    BggPublicKeyBenchEstimator<GpuMatrix, GpuPubKeyPltEvaluator, GpuPubKeySlotEvaluator>,
> {
    let (public_lut, public_lut_id, public_lut_gate_id) =
        build_benchmark_public_lookup_gate(params);
    let slot_transfer_src_slots = identity_slot_transfer_plan(params.ring_dimension() as usize);
    let slot_transfer_gate_id = build_benchmark_slot_transfer_gate(&slot_transfer_src_slots);
    let public_keys = sample_benchmark_pubkeys(
        params,
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        1,
        b"test_gpu_aky24_io_pubkey_bench",
    );
    let (public_lut_one, shared_input) = constant_and_shared_benchmark_pubkeys(&public_keys);
    let scalar_plt_evaluator =
        build_lwe_scalar_pubkey_plt_evaluator(params, DEFAULT_HASH_KEY, cfg, bench_dir.clone());
    let scalar_slot_evaluator = GpuPubKeySlotEvaluator::new(
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        params.ring_dimension() as usize,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
        bench_dir.clone(),
    );
    let (public_lut_estimator, _) =
        build_lwe_pubkey_vec_plt_evaluator(params, DEFAULT_HASH_KEY, cfg, bench_dir);
    let slot_transfer_estimator = GpuPubKeySlotEvaluator::new(
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        params.ring_dimension() as usize,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
        PathBuf::from("test_data/test_gpu_aky24_io_slot_transfer_estimator"),
    );
    let scalar_estimator = BggPublicKeyBenchEstimator::benchmark(
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
        slot_transfer_estimator,
        cfg.bench_iterations,
    );
    NaiveBGGVecBenchEstimator::new(scalar_estimator, params.ring_dimension() as usize)
}

#[derive(Debug, Clone)]
struct Aky24IOBenchUnitEstimates {
    bgg_public_key_sample: CircuitBenchEstimate,
    ring_gsw_public_key_sample: CircuitBenchEstimate,
    ring_gsw_encrypt_bit: CircuitBenchEstimate,
    full_w_block_hash_sample: CircuitBenchEstimate,
    a_prime_hash_sample: CircuitBenchSummary,
    final_output_preimage_extend: CircuitBenchEstimate,
    final_decoder_preimage_extend: CircuitBenchEstimate,
}

fn bench_estimate_named<R, F>(name: &'static str, iterations: usize, op: F) -> CircuitBenchEstimate
where
    F: FnMut() -> R,
{
    info!(unit = name, iterations, "starting AKY24IO benchmark unit cost");
    let start = Instant::now();
    let time = measure_bench_operation(iterations.max(1), op);
    let estimate = CircuitBenchEstimate {
        total_time: seconds_to_nanos(time),
        latency: time,
        max_parallelism: BigUint::from(1u32),
        peak_vram: 0,
    };
    info!(
        unit = name,
        elapsed = ?start.elapsed(),
        ?estimate,
        "finished AKY24IO benchmark unit cost"
    );
    estimate
}

fn seconds_to_nanos(seconds: f64) -> BigUint {
    BigUint::from((seconds.max(0.0) * 1_000_000_000.0).ceil().max(1.0) as u64)
}

fn measure_bench_operation<R, F>(iterations: usize, mut op: F) -> f64
where
    F: FnMut() -> R,
{
    let iterations = iterations.max(1);
    gpu_device_sync();
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(op());
    }
    gpu_device_sync();
    start.elapsed().as_secs_f64() / iterations as f64
}

fn scale_independent_estimate(
    estimate: CircuitBenchEstimate,
    count: usize,
) -> CircuitBenchEstimate {
    let count = BigUint::from(count);
    CircuitBenchEstimate {
        total_time: estimate.total_time * &count,
        latency: estimate.latency,
        max_parallelism: estimate.max_parallelism * count,
        peak_vram: estimate.peak_vram,
    }
}

fn estimate_to_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    CircuitBenchSummary {
        total_time: estimate.total_time,
        latency: estimate.latency,
        max_parallelism: estimate.max_parallelism,
        peak_vram: estimate.peak_vram,
    }
}

fn sample_native_ternary_secret(
    params: &GpuDCRTPolyParams,
    native_params: &DCRTPolyParams,
) -> DCRTPoly {
    let secret = GpuDCRTPolyUniformSampler::new().sample_poly(params, &DistType::TernaryDist);
    DCRTPoly::from_biguints(native_params, &secret.coeffs_biguints())
}

fn gpu_native_params_from_cpu(
    native_params: &DCRTPolyParams,
    params: &GpuDCRTPolyParams,
) -> GpuDCRTPolyParams {
    let (moduli, _, _) = native_params.to_crt();
    GpuDCRTPolyParams::new_with_gpu(
        native_params.ring_dimension(),
        moduli,
        native_params.base_bits(),
        params.device_ids(),
        None,
    )
}

fn benchmark_aky24_io_unit_costs(
    aky24: &GpuAky24IO,
    iterations: usize,
) -> Aky24IOBenchUnitEstimates {
    let iterations = iterations.max(1);
    let params = &aky24.params;
    let state_row_size = AKY24_IO_SECRET_SIZE;
    let gadget_col_size = AKY24_IO_SECRET_SIZE
        .checked_mul(params.modulus_digits())
        .expect("AKY24IO benchmark gadget column count overflow");
    info!(
        iterations,
        state_row_size,
        gadget_col_size,
        modulus_digits = params.modulus_digits(),
        ring_dimension = params.ring_dimension(),
        "starting AKY24IO benchmark unit-cost measurement"
    );

    let trap_sampler = GpuDCRTPolyTrapdoorSampler::new(params, DEFAULT_TRAPDOOR_SIGMA);
    let (trapdoor, public_matrix) = trap_sampler.trapdoor(params, state_row_size);
    let ext_matrix = GpuHashSampler::new().sample_hash(
        params,
        [0x51u8; 32],
        b"aky24_io_bench_preimage_extend_ext",
        state_row_size,
        gadget_col_size,
        DistType::FinRingDist,
    );
    let one_column_target = GpuMatrix::zero(params, state_row_size, 1);
    let final_output_preimage_extend_one_col =
        bench_estimate_named("aky24_final_output_preimage_extend_one_col", iterations, || {
            let preimage = trap_sampler.preimage_extend(
                params,
                &trapdoor,
                &public_matrix,
                &ext_matrix,
                &one_column_target,
            );
            preimage.to_compact_bytes()
        });
    let final_output_preimage_extend =
        scale_independent_estimate(final_output_preimage_extend_one_col.clone(), gadget_col_size);
    let final_decoder_preimage_extend = final_output_preimage_extend_one_col;
    let full_w_block_hash_sample =
        bench_estimate_named("aky24_full_w_block_hash_sample", iterations, || {
            let matrix = GpuHashSampler::new().sample_hash(
                params,
                [0x52u8; 32],
                b"aky24_io_bench_full_w_block_hash",
                state_row_size,
                gadget_col_size,
                DistType::FinRingDist,
            );
            matrix.to_compact_bytes()
        });
    let a_prime_hash_sample = estimate_to_summary(bench_estimate_named(
        "aky24_noise_refresh_a_prime_hash_sample",
        iterations,
        || {
            let matrix = GpuHashSampler::new().sample_hash(
                params,
                aky24.noise_refresh_hash_key,
                b"aky24-io-noise-refresh-bench-a-prime",
                state_row_size,
                gadget_col_size,
                DistType::FinRingDist,
            );
            matrix.to_compact_bytes()
        },
    ));
    let mut bgg_public_key_tag = aky24.bgg_tag.clone();
    bgg_public_key_tag.extend_from_slice(b":public_keys");
    let bgg_public_key_sample =
        bench_estimate_named("aky24_bgg_public_key_sample", iterations, || {
            BGGPublicKeySampler::<[u8; 32], GpuHashSampler>::new([0x5au8; 32], 1).sample(
                params,
                &bgg_public_key_tag,
                &[],
            )
        });
    let native_secret = sample_native_ternary_secret(params, &aky24.native_poly_params);
    let gpu_native_params = gpu_native_params_from_cpu(&aky24.native_poly_params, params);
    let gpu_secret =
        GpuDCRTPoly::from_biguints(&gpu_native_params, &native_secret.coeffs_biguints());
    let ring_gsw_public_key_sample_one_col =
        bench_estimate_named("aky24_ring_gsw_public_key_sample_one_col", iterations, || {
            let public_key_col = sample_public_key_columns_with_samplers::<
                GpuDCRTPoly,
                GpuDCRTPolyMatrix,
                GpuHashSampler,
                GpuDCRTPolyUniformSampler,
                _,
            >(
                &gpu_native_params,
                aky24.ring_gsw_width,
                &gpu_secret,
                [0x6du8; 32],
                b"aky24_io_bench_ring_gsw_public_key",
                0,
                1,
                aky24.ring_gsw_public_key_error_sigma,
            );
            black_box(public_key_col)
        });
    let ring_gsw_public_key_sample =
        scale_independent_estimate(ring_gsw_public_key_sample_one_col, aky24.ring_gsw_width);
    let ring_gsw_public_key_col = sample_public_key_columns_with_samplers::<
        GpuDCRTPoly,
        GpuDCRTPolyMatrix,
        GpuHashSampler,
        GpuDCRTPolyUniformSampler,
        _,
    >(
        &gpu_native_params,
        aky24.ring_gsw_width,
        &gpu_secret,
        [0x6du8; 32],
        b"aky24_io_bench_ring_gsw_public_key",
        0,
        1,
        aky24.ring_gsw_public_key_error_sigma,
    );
    let ring_gsw_encrypt_bit_key_col_contribution =
        bench_estimate_named("aky24_ring_gsw_encrypt_bit_key_col_contribution", iterations, || {
            let sampler = GpuDCRTPolyUniformSampler::new();
            let randomizer = sampler.sample_poly(&gpu_native_params, &DistType::BitDist);
            let top = ring_gsw_public_key_col[0][0].clone() * &randomizer;
            let bottom = ring_gsw_public_key_col[1][0].clone() * &randomizer;
            black_box((top, bottom))
        });
    let ring_gsw_encrypt_bit_one_ciphertext_col =
        scale_independent_estimate(ring_gsw_encrypt_bit_key_col_contribution, aky24.ring_gsw_width);
    let ring_gsw_encrypt_bit =
        scale_independent_estimate(ring_gsw_encrypt_bit_one_ciphertext_col, aky24.ring_gsw_width);

    Aky24IOBenchUnitEstimates {
        bgg_public_key_sample,
        ring_gsw_public_key_sample,
        ring_gsw_encrypt_bit,
        full_w_block_hash_sample,
        a_prime_hash_sample,
        final_output_preimage_extend,
        final_decoder_preimage_extend,
    }
}

async fn build_naive_vec_encoding_bench_estimator(
    params: &GpuDCRTPolyParams,
    cfg: &Aky24IOGpuBenchConfig,
    bench_dir: PathBuf,
) -> NaiveBGGVecBenchEstimator<BggEncodingBenchEstimator<GpuMatrix>> {
    ensure_clean_dir(&bench_dir);

    let (public_lut, public_lut_id, public_lut_gate_id) =
        build_benchmark_public_lookup_gate(params);
    let public_key_sampler =
        BGGPublicKeySampler::<_, GpuHashSampler>::new(DEFAULT_HASH_KEY, cfg.d_secret);
    let uniform_sampler = GpuDCRTPolyUniformSampler::new();
    let secrets =
        uniform_sampler.sample_uniform(params, 1, cfg.d_secret, DistType::TernaryDist).get_row(0);
    let plaintexts = vec![GpuDCRTPoly::from_usize_to_constant(params, 2)];
    let public_keys =
        public_key_sampler.sample(params, b"test_gpu_aky24_io_encoding_bench", &[true]);
    let encoding_sampler =
        BGGEncodingSampler::<GpuDCRTPolyUniformSampler>::new(params, &secrets, None);
    let encodings = encoding_sampler.sample(params, &public_keys, &plaintexts);

    let (public_lut_aux_writer, lookup_base_matrix) =
        build_lwe_pubkey_vec_plt_evaluator(params, DEFAULT_HASH_KEY, cfg, bench_dir.clone());
    public_lut_aux_writer.write_dummy_aux_for_poly_encode_bench(
        params,
        &public_lut,
        &[plaintexts[0].const_coeff_u64()],
        public_lut_id,
        public_lut_gate_id,
        cfg.error_sigma,
    );
    wait_for_all_writes(bench_dir.clone())
        .await
        .expect("AKY24IO encoding public lookup benchmark aux writes must flush");

    let secret_vec = GpuMatrix::from_poly_vec_row(params, secrets);
    let c_b = secret_vec * &lookup_base_matrix;
    let encoding_evaluator = GpuEncodingPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
        GpuMatrix,
        GpuHashSampler,
    >::new(
        DEFAULT_HASH_KEY, bench_dir, c_b
    ));
    let one = NaiveBGGEncodingVec::new(params, vec![encodings[0].clone()]);
    let input = NaiveBGGEncodingVec::new(params, vec![encodings[1].clone()]);
    let scalar_estimator =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(params, cfg.bench_iterations, || {
            encoding_evaluator.public_lookup(
                params,
                &public_lut,
                &one,
                &input,
                public_lut_gate_id,
                public_lut_id,
            )
        });
    NaiveBGGVecBenchEstimator::new(scalar_estimator, params.ring_dimension() as usize)
}

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_aky24_io_error_search_and_bench_estimate() {
    let log_filter = tracing_subscriber::filter::Targets::new()
        .with_target("test_gpu_aky24_io", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::io::aky24_io", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::io::utils::simulation", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target(
            "mxx::io::diamond_io::bench_estimator_native",
            tracing_subscriber::filter::LevelFilter::INFO,
        )
        .with_target(
            "mxx::io::aky24_io::bench_estimator",
            tracing_subscriber::filter::LevelFilter::INFO,
        )
        .with_target("mxx::bench_estimator", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::noise_refresh", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::storage::write", tracing_subscriber::filter::LevelFilter::INFO)
        .with_default(tracing_subscriber::filter::LevelFilter::WARN);
    let _ = tracing_subscriber::registry()
        .with(log_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
    gpu_device_sync();

    let cfg = Aky24IOGpuBenchConfig::from_env();
    info!("AKY24IO GPU bench config: {:?}", cfg);
    let gpu_id =
        *detected_gpu_device_ids().first().expect("test_gpu_aky24_io requires at least one GPU");
    let temp_dir = tempdir().expect("AKY24IO GPU bench test must create a tempdir");
    init_storage_system(temp_dir.path().to_path_buf());

    let selected = if let Some(selected) = cfg.selected_simulation_from_env() {
        info!(
            crt_depth = selected.crt_depth,
            log_ring_dim = selected.log_ring_dim,
            ring_dim = selected.ring_dim,
            achieved_secpar_for_gauss = selected.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd = selected.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits = selected.prf_mask_output_coeff_bits,
            noise_refresh_v_bits = selected.noise_refresh_v_bits,
            final_seed_bits = selected.seed_bits,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            initial_fresh_error_bits = selected.initial_fresh_error_bits,
            "AKY24IO selected simulation parameters provided; skipping error simulation"
        );
        selected
    } else {
        let plt_evaluator = DynamicNormPltLWEEvaluator::new(cfg.error_sigma);
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();
        let search = aky24_io_find_crt_depth(
            cfg.min_crt_depth,
            cfg.max_crt_depth,
            cfg.min_log_ring_dim,
            cfg.max_log_ring_dim,
            cfg.prf_mask_output_coeff_bits_search_bound(),
            cfg.security_bits,
            cfg.error_sigma,
            Aky24IOFuncType::GoldreichPRF { output_bits: cfg.output_size() },
            |ring_dim, crt_depth, prf_mask_output_coeff_bits, noise_refresh_v_bits| {
                let search_params =
                    DCRTPolyParams::new(ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
                let selected_noise_refresh_v_bits = noise_refresh_v_bits.unwrap_or_else(|| {
                    aky24_io_max_noise_refresh_v_bits_without_pre_rounding_error(&search_params)
                        .expect(
                            "AKY24IO CRT-depth search requires a noise-refresh v_bits candidate",
                        )
                });
                build_cpu_aky24_io_for_search(
                    &cfg,
                    ring_dim,
                    crt_depth,
                    prf_mask_output_coeff_bits,
                    selected_noise_refresh_v_bits,
                )
            },
            &plt_evaluator,
            &slot_transfer_evaluator,
        )
        .expect("AKY24IO CRT-depth search must find a valid benchmark candidate");
        let selected = Aky24IOGpuBenchSelectedSimulation {
            crt_depth: search.crt_depth,
            ring_dim: search.ring_dim,
            log_ring_dim: search.log_ring_dim,
            achieved_secpar_for_gauss: search.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd: search.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits: search.prf_mask_output_coeff_bits,
            noise_refresh_v_bits: search.noise_refresh_v_bits,
            seed_bits: search.seed_bits,
            noisy_plaintext_error_bits: bigdecimal_bits_ceil(
                &search.total_noisy_plaintext_error.poly_norm.norm,
            ) as usize,
            initial_fresh_error_bits: bigdecimal_bits_ceil(
                &search.initial_fresh_error.poly_norm.norm,
            ) as usize,
        };
        info!(
            crt_depth = selected.crt_depth,
            log_ring_dim = selected.log_ring_dim,
            ring_dim = selected.ring_dim,
            achieved_secpar_for_gauss = selected.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd = selected.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits = selected.prf_mask_output_coeff_bits,
            noise_refresh_v_bits = selected.noise_refresh_v_bits,
            final_seed_bits = selected.seed_bits,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            initial_fresh_error_bits = selected.initial_fresh_error_bits,
            "AKY24IO CRT-depth search selected parameters"
        );
        selected
    };

    if cfg.search_only {
        info!(
            crt_depth = selected.crt_depth,
            log_ring_dim = selected.log_ring_dim,
            ring_dim = selected.ring_dim,
            achieved_secpar_for_gauss = selected.achieved_secpar_for_gauss,
            achieved_secpar_for_cbd = selected.achieved_secpar_for_cbd,
            prf_mask_output_coeff_bits = selected.prf_mask_output_coeff_bits,
            noise_refresh_v_bits = selected.noise_refresh_v_bits,
            final_seed_bits = selected.seed_bits,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            initial_fresh_error_bits = selected.initial_fresh_error_bits,
            "AKY24IO GPU bench search-only mode selected parameters; skipping GPU benchmark estimation"
        );
        return;
    }

    let (cpu_params, gpu_params) =
        gpu_params_for_crt_depth(&cfg, selected.ring_dim, selected.crt_depth, gpu_id);
    assert_eq!(
        cfg.seed_bits_for_params(
            &cpu_params,
            selected.prf_mask_output_coeff_bits,
            selected.noise_refresh_v_bits,
        ),
        selected.seed_bits,
        "selected AKY24IO final seed_bits must match the minimum derived from selected mask widths"
    );
    let final_dir = temp_dir.path().join("final_estimate");
    ensure_clean_dir(&final_dir);
    init_storage_system(final_dir.clone());
    let aky24 = build_aky24_io(
        &cfg,
        cpu_params,
        gpu_params.clone(),
        selected.prf_mask_output_coeff_bits,
        selected.noise_refresh_v_bits,
        final_dir.clone(),
    );
    let public_key_estimator =
        build_public_key_bench_estimator(&gpu_params, &cfg, final_dir.join("pubkey_bench"));
    let encoding_estimator = build_naive_vec_encoding_bench_estimator(
        &gpu_params,
        &cfg,
        final_dir.join("encoding_bench"),
    )
    .await;
    let native_estimator =
        GpuDCRTPolyMatrixNativeBenchEstimator::new(gpu_params.clone(), cfg.bench_iterations);
    let units = benchmark_aky24_io_unit_costs(&aky24, cfg.bench_iterations);
    let aky24_bench_estimator = Aky24IOBenchEstimator::new(
        &public_key_estimator,
        &encoding_estimator,
        &native_estimator,
        units.bgg_public_key_sample,
        units.ring_gsw_public_key_sample,
        units.ring_gsw_encrypt_bit,
        units.full_w_block_hash_sample,
        units.a_prime_hash_sample,
        units.final_output_preimage_extend,
        units.final_decoder_preimage_extend,
    );
    let estimate = aky24_bench_estimator
        .estimate(&aky24, Aky24IOFuncType::GoldreichPRF { output_bits: cfg.output_size() });

    info!(
        obfuscate_latency = estimate.obfuscate.latency,
        obfuscate_total_time_nanos = %estimate.obfuscate.total_time,
        obfuscate_max_parallelism = %estimate.obfuscate.max_parallelism,
        fe_to_io_obfuscate_latency = estimate.fe_to_io_obfuscate.latency,
        fe_to_io_obfuscate_total_time_nanos = %estimate.fe_to_io_obfuscate_total_time,
        fe_to_io_obfuscate_max_parallelism = %estimate.fe_to_io_obfuscate.max_parallelism,
        final_fe_obfuscate_latency = estimate.final_fe_obfuscate.latency,
        final_fe_obfuscate_total_time_nanos = %estimate.final_fe_obfuscate_total_time,
        final_fe_obfuscate_max_parallelism = %estimate.final_fe_obfuscate.max_parallelism,
        eval_latency = estimate.eval.latency,
        eval_total_time_nanos = %estimate.eval.total_time,
        eval_max_parallelism = %estimate.eval.max_parallelism,
        fe_to_io_eval_latency = estimate.fe_to_io_eval.latency,
        fe_to_io_eval_total_time_nanos = %estimate.fe_to_io_eval_total_time,
        fe_to_io_eval_max_parallelism = %estimate.fe_to_io_eval.max_parallelism,
        final_fe_eval_latency = estimate.final_fe_eval.latency,
        final_fe_eval_total_time_nanos = %estimate.final_fe_eval_total_time,
        final_fe_eval_max_parallelism = %estimate.final_fe_eval.max_parallelism,
        obfuscated_circuit_bytes = %estimate.obfuscated_circuit_bytes,
        fe_to_io_obfuscated_circuit_bytes = %estimate.fe_to_io_obfuscated_circuit_bytes,
        final_fe_obfuscated_circuit_bytes = %estimate.final_fe_obfuscated_circuit_bytes,
        "AKY24IO GPU benchmark estimate"
    );
    assert!(estimate.obfuscate.total_time >= BigUint::from(0u32));
    assert!(estimate.eval.total_time >= BigUint::from(0u32));
    assert!(estimate.obfuscated_circuit_bytes > BigUint::from(0u32));
    assert!(estimate.final_fe_obfuscate_total_time > BigUint::from(0u32));
    assert!(estimate.final_fe_eval_total_time > BigUint::from(0u32));
    assert!(estimate.final_fe_obfuscated_circuit_bytes > BigUint::from(0u32));
    if cfg.input_size / cfg.prf_batch_bits > 1 {
        assert!(estimate.fe_to_io_obfuscate_total_time > BigUint::from(0u32));
        assert!(estimate.fe_to_io_eval_total_time > BigUint::from(0u32));
        assert!(estimate.fe_to_io_obfuscated_circuit_bytes > BigUint::from(0u32));
    } else {
        assert_eq!(estimate.fe_to_io_obfuscate_total_time, BigUint::from(0u32));
        assert_eq!(estimate.fe_to_io_eval_total_time, BigUint::from(0u32));
        assert_eq!(estimate.fe_to_io_obfuscated_circuit_bytes, BigUint::from(0u32));
    }
}
