#![cfg(feature = "gpu")]

use bigdecimal::{BigDecimal, FromPrimitive};
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BggEncodingBenchEstimator, BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
        NaiveBGGVecBenchEstimator,
    },
    bgg::{public_key::BggPublicKey, sampler::BGGPublicKeySampler},
    circuit::{PolyCircuit, gate::GateId},
    element::PolyElem,
    func_enc::aky24::NoCircuitEvaluator,
    gadgets::arith::{ModularArithmeticContext, NestedRnsPolyContext},
    input_injector::DiamondInjector,
    io::diamond_io::{
        DiamondIO, DiamondIOBenchEstimator, DiamondIOFuncType,
        GpuDCRTPolyMatrixNativeBenchEstimator, diamond_io_find_crt_depth,
    },
    lookup::{
        PltEvaluator, PublicLut,
        lwe::{
            LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
            NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
        },
    },
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
    simulator::error_norm::{ErrorNorm, NormNaiveBggEncodingVecSTEvaluator, NormPltLWEEvaluator},
    slot_transfer::{NaiveBGGVecSlotTransferEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::init_storage_system,
    utils::bigdecimal_bits_ceil,
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
const DEFAULT_INPUT_SIZE: usize = 5;
const DEFAULT_SEED_BITS: usize = 5;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_P_MODULI_BITS: usize = 7;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 4;
const DEFAULT_SCALE: u64 = 1 << 8;
const DEFAULT_MIN_CRT_DEPTH: usize = 1;
const DEFAULT_MAX_CRT_DEPTH: usize = 64;
const DEFAULT_SECURITY_BITS: usize = 0;
const DEFAULT_NOISE_REFRESH_V_BITS: usize = 200;
const DEFAULT_NOISE_REFRESH_CBD_N: usize = 1;
const DEFAULT_BENCH_ITERATIONS: usize = 1;
const DEFAULT_ERROR_SIGMA: f64 = 4.0;
const DEFAULT_TRAPDOOR_SIGMA: f64 = 4.578;
const DEFAULT_D_SECRET: usize = 1;
const DEFAULT_HASH_KEY: [u8; 32] = [0x42; 32];

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type CpuMatrix = DCRTPolyMatrix;
type CpuHashSampler = DCRTPolyHashSampler<Keccak256>;
type CpuInjector =
    DiamondInjector<CpuMatrix, DCRTPolyUniformSampler, CpuHashSampler, DCRTPolyTrapdoorSampler>;
type CpuDiamondIO = DiamondIO<
    CpuMatrix,
    DCRTPolyUniformSampler,
    CpuHashSampler,
    DCRTPolyTrapdoorSampler,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
    NoCircuitEvaluator,
>;
type GpuInjector = DiamondInjector<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
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
type GpuDiamondIO = DiamondIO<
    GpuMatrix,
    GpuDCRTPolyUniformSampler,
    GpuHashSampler,
    GpuDCRTPolyTrapdoorSampler,
    GpuPubKeyPltEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
    GpuEncodingPltEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
>;

#[derive(Debug, Clone)]
struct DiamondIOGpuBenchConfig {
    ring_dim: u32,
    input_size: usize,
    seed_bits: usize,
    crt_bits: usize,
    base_bits: u32,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    scale: u64,
    min_crt_depth: usize,
    max_crt_depth: usize,
    security_bits: usize,
    noise_refresh_v_bits: usize,
    noise_refresh_cbd_n: usize,
    bench_iterations: usize,
    error_sigma: f64,
    trapdoor_sigma: f64,
    d_secret: usize,
}

#[derive(Debug, Clone, Copy)]
struct DiamondIOGpuBenchSelectedSimulation {
    crt_depth: usize,
    prf_mask_output_coeff_bits: usize,
    noisy_plaintext_error_bits: usize,
    input_injection_error_bits: usize,
}

impl DiamondIOGpuBenchConfig {
    fn from_env() -> Self {
        let cfg = Self {
            ring_dim: env_or_parse_u32("DIAMOND_IO_GPU_BENCH_RING_DIM", DEFAULT_RING_DIM),
            input_size: env_or_parse_usize("DIAMOND_IO_GPU_BENCH_INPUT_SIZE", DEFAULT_INPUT_SIZE),
            seed_bits: env_or_parse_usize("DIAMOND_IO_GPU_BENCH_SEED_BITS", DEFAULT_SEED_BITS),
            crt_bits: env_or_parse_usize("DIAMOND_IO_GPU_BENCH_CRT_BITS", DEFAULT_CRT_BITS),
            base_bits: env_or_parse_u32("DIAMOND_IO_GPU_BENCH_BASE_BITS", DEFAULT_BASE_BITS),
            p_moduli_bits: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_P_MODULI_BITS",
                DEFAULT_P_MODULI_BITS,
            ),
            max_unreduced_muls: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_MAX_UNREDUCED_MULS",
                DEFAULT_MAX_UNREDUCED_MULS,
            ),
            scale: env_or_parse_u64("DIAMOND_IO_GPU_BENCH_SCALE", DEFAULT_SCALE),
            min_crt_depth: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_MIN_CRT_DEPTH",
                DEFAULT_MIN_CRT_DEPTH,
            ),
            max_crt_depth: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_MAX_CRT_DEPTH",
                DEFAULT_MAX_CRT_DEPTH,
            ),
            security_bits: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_SECURITY_BITS",
                DEFAULT_SECURITY_BITS,
            ),
            noise_refresh_v_bits: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_NOISE_REFRESH_V_BITS",
                DEFAULT_NOISE_REFRESH_V_BITS,
            ),
            noise_refresh_cbd_n: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_NOISE_REFRESH_CBD_N",
                DEFAULT_NOISE_REFRESH_CBD_N,
            ),
            bench_iterations: env_or_parse_usize(
                "DIAMOND_IO_GPU_BENCH_ITERATIONS",
                DEFAULT_BENCH_ITERATIONS,
            ),
            error_sigma: env_or_parse_f64("DIAMOND_IO_GPU_BENCH_ERROR_SIGMA", DEFAULT_ERROR_SIGMA),
            trapdoor_sigma: env_or_parse_f64(
                "DIAMOND_IO_GPU_BENCH_TRAPDOOR_SIGMA",
                DEFAULT_TRAPDOOR_SIGMA,
            ),
            d_secret: env_or_parse_usize("DIAMOND_IO_GPU_BENCH_D_SECRET", DEFAULT_D_SECRET),
        };
        assert!(cfg.ring_dim > 0, "DIAMOND_IO_GPU_BENCH_RING_DIM must be positive");
        assert!(cfg.input_size > 0, "DIAMOND_IO_GPU_BENCH_INPUT_SIZE must be positive");
        assert!(cfg.seed_bits > 0, "DIAMOND_IO_GPU_BENCH_SEED_BITS must be positive");
        assert!(cfg.crt_bits > 0, "DIAMOND_IO_GPU_BENCH_CRT_BITS must be positive");
        assert!(cfg.base_bits > 0, "DIAMOND_IO_GPU_BENCH_BASE_BITS must be positive");
        assert!(
            cfg.min_crt_depth <= cfg.max_crt_depth,
            "DIAMOND_IO_GPU_BENCH_MIN_CRT_DEPTH must be <= DIAMOND_IO_GPU_BENCH_MAX_CRT_DEPTH"
        );
        assert!(
            cfg.noise_refresh_v_bits > 0,
            "DIAMOND_IO_GPU_BENCH_NOISE_REFRESH_V_BITS must be positive"
        );
        assert!(
            cfg.noise_refresh_cbd_n > 0,
            "DIAMOND_IO_GPU_BENCH_NOISE_REFRESH_CBD_N must be positive"
        );
        assert!(cfg.bench_iterations > 0, "DIAMOND_IO_GPU_BENCH_ITERATIONS must be positive");
        assert!(cfg.error_sigma >= 0.0, "DIAMOND_IO_GPU_BENCH_ERROR_SIGMA must be nonnegative");
        assert!(cfg.trapdoor_sigma > 0.0, "DIAMOND_IO_GPU_BENCH_TRAPDOOR_SIGMA must be positive");
        assert!(cfg.d_secret > 0, "DIAMOND_IO_GPU_BENCH_D_SECRET must be positive");
        cfg
    }

    fn output_size(&self) -> usize {
        self.seed_bits
    }

    fn input_base(&self) -> usize {
        2
    }

    fn prf_mask_output_coeff_bits_search_bound(&self) -> usize {
        self.max_crt_depth
            .checked_mul(self.crt_bits)
            .expect("DiamondIO PRF-mask search bound overflow")
            .max(1)
    }

    fn selected_simulation_from_env() -> Option<DiamondIOGpuBenchSelectedSimulation> {
        let crt_depth = env_or_parse_optional_usize("DIAMOND_IO_GPU_BENCH_SELECTED_CRT_DEPTH")?;
        let prf_mask_output_coeff_bits = env_or_parse_optional_usize(
            "DIAMOND_IO_GPU_BENCH_SELECTED_PRF_MASK_OUTPUT_COEFF_BITS",
        )?;
        let noisy_plaintext_error_bits = env_or_parse_optional_usize(
            "DIAMOND_IO_GPU_BENCH_SELECTED_NOISY_PLAINTEXT_ERROR_BITS",
        )?;
        let input_injection_error_bits = env_or_parse_optional_usize(
            "DIAMOND_IO_GPU_BENCH_SELECTED_INPUT_INJECTION_ERROR_BITS",
        )?;
        assert!(crt_depth > 0, "DIAMOND_IO_GPU_BENCH_SELECTED_CRT_DEPTH must be positive");
        assert!(
            prf_mask_output_coeff_bits > 0,
            "DIAMOND_IO_GPU_BENCH_SELECTED_PRF_MASK_OUTPUT_COEFF_BITS must be positive"
        );
        Some(DiamondIOGpuBenchSelectedSimulation {
            crt_depth,
            prf_mask_output_coeff_bits,
            noisy_plaintext_error_bits,
            input_injection_error_bits,
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
        fs::remove_dir_all(dir).expect("failed to remove existing DiamondIO GPU bench directory");
    }
    fs::create_dir_all(dir).expect("failed to create DiamondIO GPU bench directory");
}

fn gpu_params_for_crt_depth(
    cfg: &DiamondIOGpuBenchConfig,
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

fn build_ring_gsw_context(
    params: &GpuDCRTPolyParams,
    cfg: &DiamondIOGpuBenchConfig,
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
    cfg: &DiamondIOGpuBenchConfig,
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
    d_secret: usize,
    dir_path: PathBuf,
) -> (GpuPubKeyPltEvaluator, GpuMatrix) {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, DEFAULT_TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
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
    d_secret: usize,
    dir_path: PathBuf,
) -> GpuScalarPubKeyPltEvaluator {
    let trapdoor_sampler = GpuDCRTPolyTrapdoorSampler::new(params, DEFAULT_TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, d_secret);
    GpuScalarPubKeyPltEvaluator::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    )
}

fn build_diamond_io(
    cfg: &DiamondIOGpuBenchConfig,
    crt_depth: usize,
    prf_mask_output_coeff_bits: usize,
    gpu_id: i32,
    dir_path: PathBuf,
) -> GpuDiamondIO {
    let (cpu_params, gpu_params) = gpu_params_for_crt_depth(cfg, crt_depth, gpu_id);
    let ring_gsw_context = build_ring_gsw_context(&gpu_params, cfg, crt_depth);
    let ring_gsw_level_offset = 0usize;
    let ring_gsw_enable_levels = Some(crt_depth);
    let ring_gsw_width = 2 *
        <NestedRnsPolyContext as ModularArithmeticContext<GpuDCRTPoly>>::gadget_len(
            ring_gsw_context.as_ref(),
            ring_gsw_enable_levels,
            Some(ring_gsw_level_offset),
        );
    let injector = GpuInjector::new(
        gpu_params.clone(),
        cfg.input_size,
        cfg.input_base(),
        cfg.trapdoor_sigma,
        cfg.error_sigma,
    )
    .with_gpu_device_ids(vec![gpu_id]);
    let lookup_dir = dir_path.join(format!("lookup_crt{crt_depth}"));
    fs::create_dir_all(&lookup_dir).expect("failed to create DiamondIO lookup directory");
    let (pk_lookup_evaluator, lookup_base_matrix) = build_lwe_pubkey_vec_plt_evaluator(
        &gpu_params,
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        lookup_dir.clone(),
    );
    let enc_lookup_dir = lookup_dir.clone();
    DiamondIO::new(
        injector,
        cfg.input_size,
        cfg.output_size(),
        cpu_params,
        ring_gsw_context,
        ring_gsw_width,
        ring_gsw_level_offset,
        ring_gsw_enable_levels,
        Some(cfg.error_sigma),
        b"test_gpu_diamond_io".to_vec(),
        cfg.seed_bits,
        prf_mask_output_coeff_bits,
        cfg.noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
        [0x25; 32],
        Some(pk_lookup_evaluator),
        Some(NaiveBGGVecSlotTransferEvaluator::new()),
        Some(lookup_base_matrix),
        Some(Arc::new(move |c_b0| {
            GpuEncodingPltEvaluator::new(
                LWEBGGEncodingPltEvaluator::<GpuMatrix, GpuHashSampler>::new(
                    DEFAULT_HASH_KEY,
                    enc_lookup_dir.clone(),
                    c_b0,
                ),
            )
        })),
        Some(NaiveBGGVecSlotTransferEvaluator::new()),
    )
}

fn build_cpu_diamond_io_for_search(
    cfg: &DiamondIOGpuBenchConfig,
    crt_depth: usize,
    prf_mask_output_coeff_bits: usize,
) -> CpuDiamondIO {
    let params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
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
    let injector = CpuInjector::new(
        params.clone(),
        cfg.input_size,
        cfg.input_base(),
        cfg.trapdoor_sigma,
        cfg.error_sigma,
    );
    DiamondIO::new(
        injector,
        cfg.input_size,
        cfg.output_size(),
        params,
        ring_gsw_context,
        ring_gsw_width,
        ring_gsw_level_offset,
        ring_gsw_enable_levels,
        Some(cfg.error_sigma),
        b"test_gpu_diamond_io_cpu_search".to_vec(),
        cfg.seed_bits,
        prf_mask_output_coeff_bits,
        cfg.noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
        [0x25; 32],
        None,
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
    cfg: &DiamondIOGpuBenchConfig,
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
        b"test_gpu_diamond_io_pubkey_bench",
    );
    let (public_lut_one, shared_input) = constant_and_shared_benchmark_pubkeys(&public_keys);
    let scalar_plt_evaluator = build_lwe_scalar_pubkey_plt_evaluator(
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
        bench_dir.clone(),
    );
    let (public_lut_estimator, _) =
        build_lwe_pubkey_vec_plt_evaluator(params, DEFAULT_HASH_KEY, cfg.d_secret, bench_dir);
    let slot_transfer_estimator = GpuPubKeySlotEvaluator::new(
        DEFAULT_HASH_KEY,
        cfg.d_secret,
        params.ring_dimension() as usize,
        cfg.trapdoor_sigma,
        cfg.error_sigma,
        PathBuf::from("test_data/test_gpu_diamond_io_slot_transfer_estimator"),
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

#[tokio::test]
#[sequential_test::sequential]
async fn test_gpu_diamond_io_error_search_and_bench_estimate() {
    let log_filter = tracing_subscriber::filter::Targets::new()
        .with_target("test_gpu_diamond_io", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target("mxx::io::diamond_io", tracing_subscriber::filter::LevelFilter::INFO)
        .with_target(
            "mxx::io::diamond_io::bench_estimator_native",
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

    let cfg = DiamondIOGpuBenchConfig::from_env();
    info!("DiamondIO GPU bench config: {:?}", cfg);
    let gpu_id =
        *detected_gpu_device_ids().first().expect("test_gpu_diamond_io requires at least one GPU");
    let temp_dir = tempdir().expect("DiamondIO GPU bench test must create a tempdir");
    init_storage_system(temp_dir.path().to_path_buf());

    let selected = if let Some(selected) = DiamondIOGpuBenchConfig::selected_simulation_from_env() {
        info!(
            crt_depth = selected.crt_depth,
            prf_mask_output_coeff_bits = selected.prf_mask_output_coeff_bits,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            input_injection_error_bits = selected.input_injection_error_bits,
            "DiamondIO selected simulation parameters provided; skipping error simulation"
        );
        selected
    } else {
        let plt_evaluator = DynamicNormPltLWEEvaluator::new(cfg.error_sigma);
        let slot_transfer_evaluator = NormNaiveBggEncodingVecSTEvaluator::new();
        let search = diamond_io_find_crt_depth(
            cfg.min_crt_depth,
            cfg.max_crt_depth,
            cfg.prf_mask_output_coeff_bits_search_bound(),
            cfg.security_bits,
            DiamondIOFuncType::DebugDecryption,
            |crt_depth| {
                build_cpu_diamond_io_for_search(
                    &cfg,
                    crt_depth,
                    cfg.prf_mask_output_coeff_bits_search_bound(),
                )
            },
            &plt_evaluator,
            &slot_transfer_evaluator,
        )
        .expect("DiamondIO CRT-depth search must find a valid benchmark candidate");
        let selected = DiamondIOGpuBenchSelectedSimulation {
            crt_depth: search.crt_depth,
            prf_mask_output_coeff_bits: search.prf_mask_output_coeff_bits,
            noisy_plaintext_error_bits: bigdecimal_bits_ceil(
                &search.total_noisy_plaintext_error.poly_norm.norm,
            ) as usize,
            input_injection_error_bits: bigdecimal_bits_ceil(
                &search.input_injection_projection_error.poly_norm.norm,
            ) as usize,
        };
        info!(
            crt_depth = selected.crt_depth,
            prf_mask_output_coeff_bits = selected.prf_mask_output_coeff_bits,
            noisy_plaintext_error_bits = selected.noisy_plaintext_error_bits,
            input_injection_error_bits = selected.input_injection_error_bits,
            "DiamondIO CRT-depth search selected parameters"
        );
        selected
    };

    let (_, gpu_params) = gpu_params_for_crt_depth(&cfg, selected.crt_depth, gpu_id);
    let final_dir = temp_dir.path().join("final_estimate");
    ensure_clean_dir(&final_dir);
    init_storage_system(final_dir.clone());
    let diamond = build_diamond_io(
        &cfg,
        selected.crt_depth,
        selected.prf_mask_output_coeff_bits,
        gpu_id,
        final_dir.clone(),
    );
    let public_key_estimator =
        build_public_key_bench_estimator(&gpu_params, &cfg, final_dir.join("pubkey_bench"));
    let encoding_scalar_estimator =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(&gpu_params, cfg.bench_iterations);
    let encoding_estimator = NaiveBGGVecBenchEstimator::new(
        encoding_scalar_estimator,
        gpu_params.ring_dimension() as usize,
    );
    let native_estimator =
        GpuDCRTPolyMatrixNativeBenchEstimator::new(gpu_params.clone(), cfg.bench_iterations);
    let diamond_bench_estimator = DiamondIOBenchEstimator::new(
        &public_key_estimator,
        &encoding_estimator,
        &native_estimator,
        &diamond,
        cfg.bench_iterations,
    );
    let estimate = diamond_bench_estimator.estimate(&diamond, DiamondIOFuncType::DebugDecryption);

    info!(
        obfuscate_latency = estimate.obfuscate.latency,
        obfuscate_total_time = estimate.obfuscate.total_time,
        obfuscate_max_parallelism = estimate.obfuscate.max_parallelism,
        eval_latency = estimate.eval.latency,
        eval_total_time = estimate.eval.total_time,
        eval_max_parallelism = estimate.eval.max_parallelism,
        obfuscated_circuit_bytes = %estimate.obfuscated_circuit_bytes,
        input_injection_bytes = %estimate.input_injection_bytes,
        "DiamondIO GPU benchmark estimate"
    );
    assert!(estimate.obfuscate.total_time >= 0.0);
    assert!(estimate.eval.total_time >= 0.0);
    assert!(estimate.obfuscated_circuit_bytes > BigUint::from(0u32));
    assert!(estimate.input_injection_bytes > BigUint::from(0u32));
}
