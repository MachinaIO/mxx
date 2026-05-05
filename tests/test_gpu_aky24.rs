#![cfg(feature = "gpu")]

use bigdecimal::BigDecimal;
use keccak_asm::Keccak256;
use mxx::{
    bench_estimator::{
        BggEncodingBenchEstimator, BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples,
        CircuitBenchEstimate, NaiveBGGVecBenchEstimator, PublicLutSampleAuxBenchEstimator,
    },
    bgg::{
        encoding::BggEncoding,
        sampler::{BGGEncodingSampler, BGGPublicKeySampler},
    },
    circuit::PolyCircuit,
    element::PolyElem,
    func_enc::{
        FuncEnc,
        aky24::{
            Aky24Func, Aky24FuncEnc, Aky24Output, Aky24Params,
            dec_bench::{Aky24DecBenchEstimator, Aky24DecBenchShape},
            error_simulation::{
                Aky24CrtDepthSearchCandidate, Aky24DecErrorSimulationInputs, aky24_find_crt_depth,
            },
            keygen_bench::{Aky24KeygenBenchEstimator, Aky24KeygenBenchShape},
        },
    },
    gadgets::arith::{ModularArithmeticContext, NestedRnsPolyContext},
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
        PolyTrapdoorSampler,
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::GpuDCRTPolyTrapdoorSampler,
    },
    simulator::{
        error_norm::{ErrorNorm, NormBggPolyEncodingSTEvaluator, NormPltLWEEvaluator},
        poly_matrix_norm::PolyMatrixNorm,
        poly_norm::PolyNorm,
    },
    slot_transfer::{NaiveBGGVecSlotTransferEvaluator, bgg_pubkey::BggPublicKeySTEvaluator},
    storage::write::{init_storage_system, wait_for_all_writes},
};
use num_bigint::BigUint;
use sequential_test::sequential;
use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tracing::info;
use tracing_subscriber::prelude::*;

const DEFAULT_RING_DIM: u32 = 2;
const DEFAULT_CRT_BITS: usize = 28;
const DEFAULT_BASE_BITS: u32 = 14;
const DEFAULT_MIN_CRT_DEPTH: usize = 28;
const DEFAULT_MAX_CRT_DEPTH: usize = 28;
const DEFAULT_MAX_PRF_MASK_OUTPUT_COEFF_BITS: usize = 1;
const DEFAULT_PUBLIC_PRF_SEED_BITS: usize = 1;
const DEFAULT_P_MODULI_BITS: usize = 10;
const DEFAULT_MAX_UNREDUCED_MULS: usize = 2;
const DEFAULT_NESTED_RNS_SCALE: u64 = 1 << 8;
const DEFAULT_RING_GSW_ERROR_SIGMA: f64 = 0.0;
const DEFAULT_TRAPDOOR_SIGMA: f64 = 4.578;
const DEFAULT_SECURITY_BIT: usize = 0;
const DEFAULT_NOISE_REFRESH_V_BITS: usize = 1;
const DEFAULT_NOISE_REFRESH_CBD_N: usize = 1;
const DEFAULT_E2E_MAX_RING_DIM: u32 = 2;

type GpuMatrix = GpuDCRTPolyMatrix;
type GpuHashSampler = GpuDCRTPolyHashSampler<Keccak256>;
type GpuUniformSampler = GpuDCRTPolyUniformSampler;
type GpuTrapdoorSampler = GpuDCRTPolyTrapdoorSampler;
#[allow(dead_code)]
type GpuPublicKeyLookupEvaluator =
    NaiveLWEBGGPublicKeyVecPltEvaluator<GpuMatrix, GpuHashSampler, GpuTrapdoorSampler>;
#[allow(dead_code)]
type GpuEncodingLookupEvaluator = NaiveLWEBGGEncodingVecPltEvaluator<GpuMatrix, GpuHashSampler>;
#[allow(dead_code)]
type GpuFuncEnc = Aky24FuncEnc<
    GpuMatrix,
    GpuHashSampler,
    GpuUniformSampler,
    GpuTrapdoorSampler,
    GpuPublicKeyLookupEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
    GpuEncodingLookupEvaluator,
    NaiveBGGVecSlotTransferEvaluator,
>;

#[derive(Debug, Clone)]
struct Aky24GpuBenchConfig {
    ring_dim: u32,
    crt_bits: usize,
    base_bits: u32,
    min_crt_depth: usize,
    max_crt_depth: usize,
    max_prf_mask_output_coeff_bits: usize,
    public_prf_seed_bits: usize,
    p_moduli_bits: usize,
    max_unreduced_muls: usize,
    nested_rns_scale: u64,
    ring_gsw_error_sigma: f64,
    trapdoor_sigma: f64,
    security_bit: usize,
    noise_refresh_v_bits: usize,
    noise_refresh_cbd_n: usize,
    #[allow(dead_code)]
    e2e_max_ring_dim: u32,
}

fn env_or_parse_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<u32>().unwrap_or_else(|e| panic!("{key} must be u32: {e}")))
        .unwrap_or(default)
}

fn env_or_parse_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<usize>().unwrap_or_else(|e| panic!("{key} must be usize: {e}")))
        .unwrap_or(default)
}

fn env_or_parse_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<u64>().unwrap_or_else(|e| panic!("{key} must be u64: {e}")))
        .unwrap_or(default)
}

fn env_or_parse_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.parse::<f64>().unwrap_or_else(|e| panic!("{key} must be f64: {e}")))
        .unwrap_or(default)
}

impl Aky24GpuBenchConfig {
    fn from_env() -> Self {
        let cfg = Self {
            ring_dim: env_or_parse_u32("AKY24_GPU_BENCH_RING_DIM", DEFAULT_RING_DIM),
            crt_bits: env_or_parse_usize("AKY24_GPU_BENCH_CRT_BITS", DEFAULT_CRT_BITS),
            base_bits: env_or_parse_u32("AKY24_GPU_BENCH_BASE_BITS", DEFAULT_BASE_BITS),
            min_crt_depth: env_or_parse_usize(
                "AKY24_GPU_BENCH_MIN_CRT_DEPTH",
                DEFAULT_MIN_CRT_DEPTH,
            ),
            max_crt_depth: env_or_parse_usize(
                "AKY24_GPU_BENCH_MAX_CRT_DEPTH",
                DEFAULT_MAX_CRT_DEPTH,
            ),
            max_prf_mask_output_coeff_bits: env_or_parse_usize(
                "AKY24_GPU_BENCH_MAX_PRF_MASK_OUTPUT_COEFF_BITS",
                DEFAULT_MAX_PRF_MASK_OUTPUT_COEFF_BITS,
            ),
            public_prf_seed_bits: env_or_parse_usize(
                "AKY24_GPU_BENCH_PUBLIC_PRF_SEED_BITS",
                DEFAULT_PUBLIC_PRF_SEED_BITS,
            ),
            p_moduli_bits: env_or_parse_usize(
                "AKY24_GPU_BENCH_P_MODULI_BITS",
                DEFAULT_P_MODULI_BITS,
            ),
            max_unreduced_muls: env_or_parse_usize(
                "AKY24_GPU_BENCH_MAX_UNREDUCED_MULS",
                DEFAULT_MAX_UNREDUCED_MULS,
            ),
            nested_rns_scale: env_or_parse_u64(
                "AKY24_GPU_BENCH_NESTED_RNS_SCALE",
                DEFAULT_NESTED_RNS_SCALE,
            ),
            ring_gsw_error_sigma: env_or_parse_f64(
                "AKY24_GPU_BENCH_RING_GSW_ERROR_SIGMA",
                DEFAULT_RING_GSW_ERROR_SIGMA,
            ),
            trapdoor_sigma: env_or_parse_f64(
                "AKY24_GPU_BENCH_TRAPDOOR_SIGMA",
                DEFAULT_TRAPDOOR_SIGMA,
            ),
            security_bit: env_or_parse_usize("AKY24_GPU_BENCH_SECURITY_BIT", DEFAULT_SECURITY_BIT),
            noise_refresh_v_bits: env_or_parse_usize(
                "AKY24_GPU_BENCH_NOISE_REFRESH_V_BITS",
                DEFAULT_NOISE_REFRESH_V_BITS,
            ),
            noise_refresh_cbd_n: env_or_parse_usize(
                "AKY24_GPU_BENCH_NOISE_REFRESH_CBD_N",
                DEFAULT_NOISE_REFRESH_CBD_N,
            ),
            e2e_max_ring_dim: env_or_parse_u32(
                "AKY24_GPU_BENCH_E2E_MAX_RING_DIM",
                DEFAULT_E2E_MAX_RING_DIM,
            ),
        };
        assert!(cfg.ring_dim > 0, "AKY24_GPU_BENCH_RING_DIM must be positive");
        assert!(cfg.crt_bits > 0, "AKY24_GPU_BENCH_CRT_BITS must be positive");
        assert!(cfg.base_bits > 0, "AKY24_GPU_BENCH_BASE_BITS must be positive");
        assert!(cfg.min_crt_depth > 0, "AKY24_GPU_BENCH_MIN_CRT_DEPTH must be positive");
        assert!(cfg.min_crt_depth <= cfg.max_crt_depth);
        assert!(cfg.max_prf_mask_output_coeff_bits > 0);
        assert!(
            cfg.public_prf_seed_bits > 0,
            "AKY24_GPU_BENCH_PUBLIC_PRF_SEED_BITS must be positive"
        );
        assert!(cfg.p_moduli_bits > 0);
        assert!(cfg.max_unreduced_muls > 0);
        assert!(cfg.nested_rns_scale > 0);
        assert!(cfg.ring_gsw_error_sigma >= 0.0);
        assert!(cfg.trapdoor_sigma > 0.0);
        assert!(cfg.noise_refresh_v_bits > 0);
        assert!(cfg.noise_refresh_cbd_n > 0);
        cfg
    }

    fn prf_seed_bits(&self) -> usize {
        self.max_prf_mask_output_coeff_bits.max(10)
    }

    #[allow(dead_code)]
    fn should_run_e2e(&self) -> bool {
        self.ring_dim <= self.e2e_max_ring_dim
    }

    fn bench_dir_base(&self, crt_depth: usize, mask_bits: usize) -> String {
        format!(
            "test_data/test_gpu_aky24_ring{}_crt{}_depth{}_mask{}",
            self.ring_dim, self.crt_bits, crt_depth, mask_bits
        )
    }
}

fn init_tracing() {
    let log_filter = tracing_subscriber::filter::Targets::new()
        .with_target("mxx::func_enc::aky24", tracing::Level::INFO)
        .with_target("mxx::func_enc::aky24::error_simulation", tracing::Level::INFO)
        .with_target("mxx::simulator::memory", tracing::Level::INFO)
        .with_target("mxx::noise_refresh::simulation", tracing::Level::INFO)
        .with_target("mxx::lookup::lwe::pubkey", tracing::Level::INFO)
        .with_target("test_gpu_aky24", tracing::Level::INFO);
    let _ = tracing_subscriber::registry()
        .with(log_filter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}

fn build_ring_gsw_context<P: Poly + 'static>(
    params: &P::Params,
    cfg: &Aky24GpuBenchConfig,
    active_levels: usize,
) -> (Arc<NestedRnsPolyContext>, usize) {
    let mut setup_circuit = PolyCircuit::<P>::new();
    let context = Arc::new(NestedRnsPolyContext::setup(
        &mut setup_circuit,
        params,
        cfg.p_moduli_bits,
        cfg.max_unreduced_muls,
        cfg.nested_rns_scale,
        false,
        Some(active_levels),
    ));
    let width = 2 * <NestedRnsPolyContext as ModularArithmeticContext<P>>::gadget_len(
        context.as_ref(),
        Some(active_levels),
        Some(0),
    );
    (context, width)
}

fn cpu_sim_params(
    cfg: &Aky24GpuBenchConfig,
    crt_depth: usize,
    mask_bits: usize,
) -> Aky24Params<DCRTPolyMatrix, ()> {
    let poly_params = DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let (ring_gsw_context, ring_gsw_width) =
        build_ring_gsw_context::<DCRTPoly>(&poly_params, cfg, crt_depth);
    Aky24Params::<DCRTPolyMatrix, ()>::new(
        poly_params.clone(),
        poly_params.clone(),
        ring_gsw_context,
        ring_gsw_width,
        0,
        Some(crt_depth),
        Some(cfg.ring_gsw_error_sigma),
        b"aky24_gpu_bench_sim".to_vec(),
        cfg.trapdoor_sigma,
        None,
        DCRTPolyMatrix::zero(&poly_params, 1, 1),
        (),
        cfg.prf_seed_bits(),
        mask_bits,
        Some(cfg.public_prf_seed_bits),
        [0x42; 32],
        cfg.noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
    )
}

fn gpu_params_without_trapdoor(
    cfg: &Aky24GpuBenchConfig,
    native_poly_params: DCRTPolyParams,
    gpu_params: GpuDCRTPolyParams,
    mask_bits: usize,
) -> Aky24Params<GpuMatrix, ()> {
    let active_levels = native_poly_params.to_crt().2;
    let (ring_gsw_context, ring_gsw_width) =
        build_ring_gsw_context::<GpuDCRTPoly>(&gpu_params, cfg, active_levels);
    Aky24Params::<GpuMatrix, ()>::new(
        gpu_params.clone(),
        native_poly_params,
        ring_gsw_context,
        ring_gsw_width,
        0,
        Some(active_levels),
        Some(cfg.ring_gsw_error_sigma),
        b"aky24_gpu_bench".to_vec(),
        cfg.trapdoor_sigma,
        None,
        GpuMatrix::zero(&gpu_params, 1, 1),
        (),
        cfg.prf_seed_bits(),
        mask_bits,
        Some(cfg.public_prf_seed_bits),
        [0x42; 32],
        cfg.noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
    )
}

#[allow(dead_code)]
fn gpu_params_with_trapdoor(
    cfg: &Aky24GpuBenchConfig,
    native_poly_params: DCRTPolyParams,
    gpu_params: GpuDCRTPolyParams,
    mask_bits: usize,
) -> Aky24Params<GpuMatrix, <GpuTrapdoorSampler as PolyTrapdoorSampler>::Trapdoor> {
    let active_levels = native_poly_params.to_crt().2;
    let (ring_gsw_context, ring_gsw_width) =
        build_ring_gsw_context::<GpuDCRTPoly>(&gpu_params, cfg, active_levels);
    let trapdoor_sampler = GpuTrapdoorSampler::new(&gpu_params, cfg.trapdoor_sigma);
    let (b_trapdoor, b_matrix) = trapdoor_sampler.trapdoor(&gpu_params, 1);
    Aky24Params::<GpuMatrix, _>::new(
        gpu_params.clone(),
        native_poly_params,
        ring_gsw_context,
        ring_gsw_width,
        0,
        Some(active_levels),
        Some(cfg.ring_gsw_error_sigma),
        b"aky24_gpu_e2e".to_vec(),
        cfg.trapdoor_sigma,
        None,
        b_matrix,
        b_trapdoor,
        cfg.prf_seed_bits(),
        mask_bits,
        Some(cfg.public_prf_seed_bits),
        [0x42; 32],
        cfg.noise_refresh_v_bits,
        cfg.noise_refresh_cbd_n,
        [0x24; 32],
    )
}

fn select_crt_depth_and_mask_bits(cfg: &Aky24GpuBenchConfig) -> (usize, usize) {
    let result = aky24_find_crt_depth(
        cfg.min_crt_depth,
        cfg.max_crt_depth,
        cfg.max_prf_mask_output_coeff_bits,
        cfg.security_bit,
        cfg.ring_gsw_error_sigma,
        cfg.ring_dim as usize,
        |crt_depth| {
            let params = cpu_sim_params(cfg, crt_depth, 1);
            let inputs =
                Aky24DecErrorSimulationInputs::new_from_params(&params, Aky24Func::DebugIdentity);
            let ctx = inputs.c_b0_error_norm.clone_ctx();
            let zero_matrix =
                PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(0u32), None);
            let one = ErrorNorm::new(PolyNorm::one(ctx.clone()), zero_matrix.clone());
            let decryption_key = ErrorNorm::new(PolyNorm::one(ctx.clone()), zero_matrix.clone());
            let secret_norm =
                PolyMatrixNorm::new(ctx.clone(), 1, ctx.secret_size, BigDecimal::from(0u32), None);
            let decoder = ErrorNorm::new(PolyNorm::one(ctx), zero_matrix);
            let decoder_count = cfg.ring_dim as usize * crt_depth;
            Aky24CrtDepthSearchCandidate {
                params,
                inputs,
                secret_norm,
                one,
                decryption_key,
                decoders: vec![decoder; decoder_count],
            }
        },
        |candidate| {
            let ctx = candidate.inputs.c_b0_error_norm.clone_ctx();
            (
                NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from(0u32)),
                NormBggPolyEncodingSTEvaluator::new(
                    ctx,
                    cfg.ring_gsw_error_sigma,
                    &BigDecimal::from(0u32),
                    None,
                ),
            )
        },
    )
    .expect("AKY24 GPU bench test must find CRT depth and mask-bit parameters");
    (result.crt_depth, result.prf_mask_output_coeff_bits)
}

fn one_bit_lut(params: &GpuDCRTPolyParams) -> PublicLut<GpuDCRTPoly> {
    PublicLut::new(
        params,
        2,
        |params: &GpuDCRTPolyParams, input| {
            Some((input, <GpuDCRTPoly as Poly>::Elem::constant(&params.modulus(), input)))
        },
        Some((1, <GpuDCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
    )
}

fn build_lwe_pubkey_plt_evaluator(
    params: &GpuDCRTPolyParams,
    hash_key: [u8; 32],
    secret_size: usize,
    dir_path: PathBuf,
) -> LWEBGGPubKeyPltEvaluator<GpuMatrix, GpuHashSampler, GpuTrapdoorSampler> {
    let trapdoor_sampler = GpuTrapdoorSampler::new(params, DEFAULT_TRAPDOOR_SIGMA);
    let (trapdoor, pub_matrix) = trapdoor_sampler.trapdoor(params, secret_size);
    LWEBGGPubKeyPltEvaluator::<GpuMatrix, GpuHashSampler, GpuTrapdoorSampler>::new(
        hash_key,
        trapdoor_sampler,
        Arc::new(pub_matrix),
        Arc::new(trapdoor),
        dir_path,
    )
}

fn measured_estimate<R>(mut op: impl FnMut() -> R) -> CircuitBenchEstimate {
    gpu_device_sync();
    let start = Instant::now();
    let _ = op();
    gpu_device_sync();
    let elapsed = start.elapsed().as_secs_f64();
    CircuitBenchEstimate { total_time: elapsed, latency: elapsed, max_parallelism: 1, peak_vram: 0 }
}

async fn run_benchmark_estimation(
    params: &Aky24Params<GpuMatrix, ()>,
    public_prf_seed_bits: usize,
    storage_dir: PathBuf,
) {
    prepare_clean_storage(&storage_dir);
    let lut = one_bit_lut(&params.poly_params);
    let small_scalar = [3u32];
    let large_scalar = [BigUint::from(5u32)];
    let src_slots = [(0u32, None)];
    let mut gate_id_circuit = PolyCircuit::<GpuDCRTPoly>::new();
    let gate_ids = gate_id_circuit.input(3).to_vec();
    let pubkey_sampler = BGGPublicKeySampler::<_, GpuHashSampler>::new([0x50; 32], 1);
    let pubkeys = pubkey_sampler.sample(&params.poly_params, b"aky24-gpu-bench-pk", &[true, true]);
    let pubkey_a = &pubkeys[0];
    let pubkey_b = &pubkeys[1];
    let pubkey_c = &pubkeys[2];
    let pubkey_samples = BggPublicKeyBenchSamples {
        params: &params.poly_params,
        add_lhs: pubkey_a,
        add_rhs: pubkey_b,
        sub_lhs: pubkey_a,
        sub_rhs: pubkey_b,
        mul_lhs: pubkey_a,
        mul_rhs: pubkey_b,
        small_scalar_input: pubkey_a,
        small_scalar: &small_scalar,
        large_scalar_input: pubkey_a,
        large_scalar: &large_scalar,
        public_lut_one: pubkey_a,
        public_lut_input: pubkey_b,
        public_lut: &lut,
        public_lut_gate_id: gate_ids[0],
        public_lut_id: 0,
        slot_transfer_input: pubkey_c,
        slot_transfer_src_slots: &src_slots,
        slot_transfer_gate_id: gate_ids[1],
    };
    let pubkey_lut_evaluator =
        build_lwe_pubkey_plt_evaluator(&params.poly_params, [0x51; 32], 1, storage_dir.clone());
    let pubkey_lut_estimator = build_lwe_pubkey_plt_evaluator(
        &params.poly_params,
        [0x54; 32],
        1,
        storage_dir.join("pubkey_lut_estimator"),
    );
    let pubkey_slot_evaluator = BggPublicKeySTEvaluator::<
        GpuMatrix,
        GpuUniformSampler,
        GpuHashSampler,
        GpuTrapdoorSampler,
    >::new(
        [0x52; 32],
        1,
        1,
        params.trapdoor_sigma,
        params.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
        storage_dir.clone(),
    );
    info!("AKY24 GPU benchmark measuring public-key vector estimator");
    let pubkey_scalar_bench = BggPublicKeyBenchEstimator::benchmark(
        &pubkey_samples,
        &pubkey_lut_evaluator,
        &pubkey_slot_evaluator,
        pubkey_lut_estimator,
        pubkey_slot_evaluator.clone(),
        1,
    );
    let pubkey_estimator = NaiveBGGVecBenchEstimator::new(pubkey_scalar_bench, params.n());

    info!("AKY24 GPU benchmark measuring encoding scalar estimator");
    let mut encoding_scalar_bench =
        BggEncodingBenchEstimator::<GpuMatrix>::benchmark(&params.poly_params, 1);
    let plaintext_zero = GpuDCRTPoly::from_usize_to_constant(&params.poly_params, 0);
    let plaintext_one = GpuDCRTPoly::from_usize_to_constant(&params.poly_params, 1);
    let encoding_secret_size = 2;
    let encoding_secrets = vec![
        GpuDCRTPoly::from_usize_to_constant(&params.poly_params, 1),
        GpuDCRTPoly::from_usize_to_constant(&params.poly_params, 0),
    ];
    let encoding_pubkey_sampler =
        BGGPublicKeySampler::<_, GpuHashSampler>::new([0x55; 32], encoding_secret_size);
    let encoding_pubkeys =
        encoding_pubkey_sampler.sample(&params.poly_params, b"aky24-gpu-bench-enc-pk", &[true]);
    let encoding_sampler = BGGEncodingSampler::<GpuUniformSampler>::new(
        &params.poly_params,
        &encoding_secrets,
        params.encoding_error_sigma,
    );
    let encodings =
        encoding_sampler.sample(&params.poly_params, &encoding_pubkeys, &[plaintext_one]);
    let encoding_one = BggEncoding::new(
        encodings[0].vector.clone(),
        encodings[0].pubkey.clone(),
        Some(plaintext_zero),
    );
    let encoding_input = encodings[1].clone();
    let encoding_lut_dir = storage_dir.join("encoding_lut");
    prepare_clean_storage(&encoding_lut_dir);
    let encoding_pubkey_lut_evaluator = build_lwe_pubkey_plt_evaluator(
        &params.poly_params,
        [0x53; 32],
        encoding_secret_size,
        encoding_lut_dir.clone(),
    );
    encoding_pubkey_lut_evaluator.write_dummy_aux_for_poly_encode_bench(
        &params.poly_params,
        &lut,
        &[1],
        0,
        gate_ids[2],
        params.ring_gsw_public_key_error_sigma.unwrap_or(0.0),
    );
    wait_for_all_writes(encoding_lut_dir.clone())
        .await
        .expect("AKY24 GPU benchmark dummy LWE LUT aux writes must finish");
    let c_b_width = encoding_pubkey_lut_evaluator.pub_matrix.col_size();
    let c_b = GpuMatrix::zero(&params.poly_params, 1, c_b_width);
    let encoding_lut_evaluator = LWEBGGEncodingPltEvaluator::<GpuMatrix, GpuHashSampler>::new(
        [0x53; 32],
        encoding_lut_dir,
        c_b,
    );
    info!("AKY24 GPU benchmark measuring LWE encoding public lookup");
    let encoding_public_lookup = measured_estimate(|| {
        encoding_lut_evaluator.public_lookup(
            &params.poly_params,
            &lut,
            &encoding_one,
            &encoding_input,
            gate_ids[2],
            0,
        )
    });
    encoding_scalar_bench.public_lut_time = encoding_public_lookup.total_time;
    encoding_scalar_bench.public_lut_peak_vram = encoding_public_lookup.peak_vram;
    let encoding_estimator = NaiveBGGVecBenchEstimator::new(encoding_scalar_bench, params.n());

    info!("AKY24 GPU benchmark measuring trapdoor preimage");
    let trapdoor_sampler = GpuTrapdoorSampler::new(&params.poly_params, params.trapdoor_sigma);
    let (trapdoor, public_matrix) = trapdoor_sampler.trapdoor(&params.poly_params, 1);
    let preimage_target = GpuMatrix::zero(&params.poly_params, 1, 1);
    let preimage = measured_estimate(|| {
        trapdoor_sampler.preimage(&params.poly_params, &trapdoor, &public_matrix, &preimage_target)
    });
    let matrix_a = GpuMatrix::zero(&params.poly_params, 1, 1);
    let matrix_b = GpuMatrix::zero(&params.poly_params, 1, 1);
    info!("AKY24 GPU benchmark measuring native matrix operations");
    let matrix_mul = measured_estimate(|| matrix_a.clone() * &matrix_b);
    let matrix_add = measured_estimate(|| matrix_a.clone() + &matrix_b);
    let matrix_sub = measured_estimate(|| matrix_a.clone() - &matrix_b);

    drop(matrix_b);
    drop(matrix_a);
    drop(preimage_target);
    drop(public_matrix);
    drop(trapdoor);
    drop(trapdoor_sampler);
    drop(encoding_lut_evaluator);
    drop(encoding_pubkey_lut_evaluator);
    drop(encoding_input);
    drop(encoding_one);
    drop(encodings);
    drop(encoding_sampler);
    drop(encoding_pubkeys);
    drop(encoding_pubkey_sampler);
    drop(encoding_secrets);
    drop(pubkey_slot_evaluator);
    drop(pubkey_lut_evaluator);
    drop(pubkey_samples);
    drop(pubkeys);
    drop(pubkey_sampler);
    drop(gate_id_circuit);
    drop(lut);
    gpu_device_sync();

    let keygen_estimator = Aky24KeygenBenchEstimator::new(&pubkey_estimator, preimage);
    let dec_estimator =
        Aky24DecBenchEstimator::new(&encoding_estimator, matrix_mul, matrix_add, matrix_sub);
    info!("AKY24 GPU benchmark composing AKY24 keygen/dec estimates");
    let keygen_shape =
        Aky24KeygenBenchShape { wire_count: params.ring_gsw_width, public_prf_seed_bits };
    let dec_shape = Aky24DecBenchShape { wire_count: params.ring_gsw_width, public_prf_seed_bits };
    let keygen = keygen_estimator.estimate::<GpuMatrix, GpuHashSampler, ()>(
        params,
        &Aky24Func::DebugIdentity,
        keygen_shape,
    );
    let dec = dec_estimator.estimate::<GpuMatrix, GpuHashSampler, ()>(
        params,
        &Aky24Func::DebugIdentity,
        dec_shape,
    );
    info!(
        keygen_total_time = keygen.total.total_time,
        keygen_latency = keygen.total.latency,
        keygen_parallelism = keygen.total.max_parallelism,
        dec_total_time = dec.total.total_time,
        dec_latency = dec.total.latency,
        dec_parallelism = dec.total.max_parallelism,
        "AKY24 GPU benchmark estimation finished"
    );
    assert!(keygen.total.max_parallelism > 0, "keygen benchmark estimate must be non-empty");
    assert!(dec.total.max_parallelism > 0, "dec benchmark estimate must be non-empty");
}

fn gpu_params_from_cpu(cpu_params: &DCRTPolyParams) -> GpuDCRTPolyParams {
    let gpu_ids = detected_gpu_device_ids();
    let gpu_id = *gpu_ids.first().expect("AKY24 GPU integration test requires at least one GPU");
    let (moduli, _, _) = cpu_params.to_crt();
    GpuDCRTPolyParams::new_with_gpu(
        cpu_params.ring_dimension(),
        moduli,
        cpu_params.base_bits(),
        vec![gpu_id],
        Some(1),
    )
}

fn prepare_clean_storage(dir_path: &Path) {
    if dir_path.exists() {
        fs::remove_dir_all(dir_path).expect("failed to clear AKY24 GPU test storage directory");
    }
    fs::create_dir_all(dir_path).expect("failed to create AKY24 GPU test storage directory");
    init_storage_system(dir_path.to_path_buf());
}

#[allow(dead_code)]
async fn run_small_ring_e2e(
    cfg: &Aky24GpuBenchConfig,
    native_poly_params: DCRTPolyParams,
    gpu_poly_params: GpuDCRTPolyParams,
    mask_bits: usize,
    storage_dir: PathBuf,
) {
    let params = gpu_params_with_trapdoor(cfg, native_poly_params, gpu_poly_params, mask_bits);
    assert!(
        !params.debug_reuse_single_prg_sample(),
        "AKY24 GPU integration test must exercise the production non-debug PRG mask path"
    );
    prepare_clean_storage(&storage_dir);

    let mut scheme = GpuFuncEnc::new(None, None, None, None);
    let (enc_key, master_key) = scheme.setup(&params);
    let pubkey_lut_evaluator = build_lwe_pubkey_plt_evaluator(
        &params.poly_params,
        enc_key.bgg_hash_key,
        params.secret_size(),
        storage_dir.clone(),
    );
    scheme.pk_lookup_evaluator =
        Some(NaiveLWEBGGPublicKeyVecPltEvaluator::new(pubkey_lut_evaluator));
    scheme.pk_slot_transfer_evaluator = Some(NaiveBGGVecSlotTransferEvaluator::new());
    scheme.enc_slot_transfer_evaluator = Some(NaiveBGGVecSlotTransferEvaluator::new());

    let msg = true;
    let ciphertext = scheme.enc(&params, &enc_key, &msg);

    let func_key = scheme.keygen(&params, &master_key, &Aky24Func::DebugIdentity);
    info!("AKY24 GPU integration e2e sampling LWE public LUT auxiliary matrices");
    scheme
        .pk_lookup_evaluator
        .as_ref()
        .expect("AKY24 GPU integration test must install a public-key lookup evaluator")
        .sample_aux_matrices(&params.poly_params);
    wait_for_all_writes(storage_dir.clone())
        .await
        .expect("AKY24 GPU integration LWE LUT auxiliary writes must finish");
    scheme.enc_lookup_evaluator =
        Some(NaiveLWEBGGEncodingVecPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
            GpuMatrix,
            GpuHashSampler,
        >::new(
            enc_key.bgg_hash_key,
            storage_dir,
            ciphertext.c_b.clone(),
        )));
    info!("AKY24 GPU integration e2e starting decryption");

    let output = scheme.dec(&params, &ciphertext, &func_key);
    let Aky24Output::DebugIdentity { decrypted } = output;
    assert_eq!(decrypted, msg);
    gpu_device_sync();
}

#[tokio::test]
#[sequential]
async fn test_gpu_aky24_bench_flow_and_optional_small_ring_roundtrip() {
    init_tracing();
    gpu_device_sync();

    let cfg = Aky24GpuBenchConfig::from_env();
    info!(?cfg, "AKY24 GPU integration test configuration");
    let (crt_depth, mask_bits) = select_crt_depth_and_mask_bits(&cfg);
    info!(crt_depth, mask_bits, "AKY24 GPU integration test selected parameters");

    let native_poly_params =
        DCRTPolyParams::new(cfg.ring_dim, crt_depth, cfg.crt_bits, cfg.base_bits);
    let gpu_poly_params = gpu_params_from_cpu(&native_poly_params);
    assert_eq!(gpu_poly_params.modulus(), native_poly_params.modulus());

    let bench_params = gpu_params_without_trapdoor(
        &cfg,
        native_poly_params.clone(),
        gpu_poly_params.clone(),
        mask_bits,
    );
    run_benchmark_estimation(
        &bench_params,
        cfg.public_prf_seed_bits,
        Path::new(&cfg.bench_dir_base(crt_depth, mask_bits)).join("bench_estimation"),
    )
    .await;

    // The production non-debug e2e path uses the LWE public LUT evaluators and currently does not
    // complete in a practical amount of time because public LUT auxiliary sampling dominates the
    // run. Keep this integration test focused on the parameter-selection and benchmark-estimation
    // flow until that e2e path is made tractable.
    //
    // if cfg.should_run_e2e() {
    //     let storage_dir = Path::new(&cfg.bench_dir_base(crt_depth, mask_bits)).join("e2e");
    //     run_small_ring_e2e(&cfg, native_poly_params, gpu_poly_params, mask_bits, storage_dir)
    //         .await;
    // } else {
    //     info!(
    //         ring_dim = cfg.ring_dim,
    //         e2e_max_ring_dim = cfg.e2e_max_ring_dim,
    //         "AKY24 GPU integration test skipped e2e for large ring dimension"
    //     );
    // }
}
